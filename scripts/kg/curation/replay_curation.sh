#!/usr/bin/env bash
# Curate a rebuilt graph after stage 3, always in the same order: strict
# re-judgement of the stage-4 merges, stage 4-6, then alias collapse, cleanup,
# edge rules, anaphoric nodes, two rounds of bilingual unions, isolated nodes,
# indexes.
#
#   bash scripts/kg/curation/replay_curation.sh <run_dir>
#   START_STEP=9 bash scripts/kg/curation/replay_curation.sh <run_dir>   # resume
#
# <run_dir> must hold config.yaml and rebuild.env pointing at the STAGING graph
# (never the hosted one), the stage 0-3 artifacts, and the stage-4 approvals
# (stage4_merge_approved.json, or its _unfiltered copy).
#
# Every decision is a file in <run_dir>, and a step whose file is there replays
# it instead of asking the model:
#   merge_verdicts.json                 step 1, the strict judge's verdicts
#   bilingual_proposals_round{1,2}.json steps 9-10, the unions the judge proposed
#   bilingual_excluded.json             steps 9-10, unions rejected on reading
# When a bilingual round has no proposals yet, the step writes them and stops:
# read them, list the wrong ones in bilingual_excluded.json, and resume.
#
# Steps 1 and 9-10 need a Qwen3-32B server on the endpoints below unless their
# decision files are complete; step 12 needs the encoder on :8002.
set -euo pipefail

RUN="$(cd "${1:?usage: replay_curation.sh <run_dir>}" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-conda run --no-capture-output -n graphllm python}"
START_STEP="${START_STEP:-1}"
ENDPOINTS="${CURATION_ENDPOINTS:-http://localhost:8000/v1,http://localhost:8003/v1}"
STAGING=(--uri bolt://localhost:7689 --password staging-kg-v2 --database neo4j)
export PYTHONNOUSERSITE=1
# Stage 4 picks among spellings that differ only in case ("Class", "class") in
# set order, which follows the string hash: without a fixed seed two replays
# name those nodes differently, and the name-matched steps below diverge.
export PYTHONHASHSEED=42
cd "$ROOT"

[[ -f "$RUN/config.yaml" && -f "$RUN/rebuild.env" ]] || { echo "config.yaml or rebuild.env missing in $RUN" >&2; exit 2; }
# The repository defaults point at the hosted graph the demo serves.
grep -q "bolt://localhost:7689" "$RUN/rebuild.env" || { echo "$RUN/rebuild.env does not point at the staging graph" >&2; exit 2; }

step() { [[ "$1" -ge "$START_STEP" ]]; }
say() { echo; echo "=== step $1: $2 ($(date -u +%H:%M:%S)) ==="; }

if step 1; then
  say 1 "strict re-judgement of the stage-4 merges"
  $PY scripts/kg/curation/rejudge_merges.py --run-dir "$RUN" --endpoints "$ENDPOINTS"
fi

if step 2; then
  say 2 "stage 4-5 from the filtered approvals"
  rm -f "$RUN/stage4_triples_resolved.json" "$RUN/stage4_registry.json" "$RUN/stage5_triples_linked.json"
  $PY -c 'import json, sys
p = sys.argv[1]
d = json.load(open(p))
for k in ("triples_resolved", "triples_linked"):
    d.pop(k, None)
json.dump(d, open(p, "w"), indent=2)' "$RUN/stage_fingerprints.json"
  $PY -m kg_pipeline.main --config "$RUN/config.yaml" --env-file "$RUN/rebuild.env" --run-dir "$RUN" --stage linking
fi

if step 3; then
  say 3 "wipe the staging graph"
  $PY scripts/kg/kg_wipe.py --config "$RUN/config.yaml" --env-file "$RUN/rebuild.env" --yes
fi

if step 4; then
  say 4 "stage 6, write the graph"
  $PY -m kg_pipeline.main --config "$RUN/config.yaml" --env-file "$RUN/rebuild.env" --run-dir "$RUN"
fi

if step 5; then
  say 5 "collapse alias nodes"
  $PY scripts/kg/kg_collapse_aliases.py --config "$RUN/config.yaml" --env-file "$RUN/rebuild.env" --yes
fi

if step 6; then
  say 6 "self-loops and generic nodes"
  $PY scripts/kg/quality/pass1_cleanup.py "${STAGING[@]}" --report-dir "$RUN/quality" --apply
fi

if step 7; then
  say 7 "edge rules"
  $PY scripts/kg/curation/edge_rules.py --chunks-dir "$RUN" --log "$RUN/edge_rules_log.jsonl" --apply
fi

if step 8; then
  say 8 "anaphoric nodes"
  $PY scripts/kg/curation/drop_nodes.py --anaphoric --log "$RUN/anaphoric_deleted.jsonl" --apply
fi

bilingual_round() {  # round, min-degree, top-k, min-cos
  local round="$1" proposals="$RUN/bilingual_proposals_round$1.json"
  if [[ ! -f "$proposals" ]]; then
    $PY scripts/kg/curation/bilingual_merges.py propose --min-degree "$2" --top-k "$3" --min-cos "$4" \
      --endpoints "$ENDPOINTS" --out "$proposals"
    echo
    echo "Read $proposals, list the wrong unions in $RUN/bilingual_excluded.json"
    echo "({\"escluse\": [{\"giro\": $round, \"unito\": ..., \"centro\": ...}]}), then resume with START_STEP=$((8 + round))."
    exit 0
  fi
  local exclude=()
  [[ -f "$RUN/bilingual_excluded.json" ]] && exclude=(--exclude "$RUN/bilingual_excluded.json" --round "$round")
  $PY scripts/kg/curation/bilingual_merges.py apply --proposals "$proposals" "${exclude[@]}" \
    --log "$RUN/bilingual_applied_round$round.jsonl"
}

if step 9; then
  say 9 "bilingual unions, round 1 (degree >= 4, 3 neighbours)"
  bilingual_round 1 4 3 0.87
fi

if step 10; then
  say 10 "bilingual unions, round 2 (central concepts: degree >= 20, 10 neighbours)"
  bilingual_round 2 20 10 0.85
fi

if step 11; then
  say 11 "isolated nodes"
  $PY scripts/kg/curation/drop_nodes.py --isolated --apply
fi

if step 12; then
  say 12 "indexes"
  $PY scripts/kg/kg_search_index.py --config "$RUN/config.yaml" --env-file "$RUN/rebuild.env"
  ( set -a; source "$RUN/rebuild.env"; set +a
    $PY scripts/kg/kg_vector_index.py --drop
    $PY scripts/kg/kg_vector_index.py
    $PY scripts/kg/check_vector_index.py --min-resolving 1000 )
fi

echo
echo "Curation finished for $RUN."
