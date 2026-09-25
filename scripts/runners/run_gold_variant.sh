#!/usr/bin/env bash
# One gold campaign against the local staging graph, for one KG variant.
#
# Same 30 questions, same 8 strategies, same generator and same flags as the
# reference campaigns — the only thing that changes between calls is the state of
# the graph. That is what makes the variants comparable to each other.
#
# They are NOT comparable to the numbers in exp_results/: those ran against Aura
# with a different server session. Every claim about "before and after" must be
# read inside this family of runs.
#
#   VARIANT=v2_baseline bash scripts/runners/run_gold_variant.sh
#
# Preconditions: staging Neo4j on 7689, generator on 8000, encoder on 8002, and
# a vector index that resolves (checked below).
set -euo pipefail

cd "$(dirname "$0")/../.."

VARIANT="${VARIANT:?set VARIANT, e.g. v2_baseline}"
OUT_ROOT="${OUT_ROOT:-/srv/projects/graphllm/experiments/exp_results_kg_v2}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
GOLD="${GOLD:-evaluation/gold/gold_v3.json}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
STRATEGIES="${STRATEGIES:-default,hybrid,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path}"
# Extra CLI flags for variants that change the retrieval budget rather than the
# graph. Everything else stays fixed, so a run with EXTRA_ARGS is comparable to
# one without it on the same graph state.
EXTRA_ARGS="${EXTRA_ARGS:-}"

export NEO4J_URL="${STAGING_URL:-bolt://localhost:7689}"
export NEO4J_USERNAME="${STAGING_USER:-neo4j}"
export NEO4J_PASSWORD="${STAGING_PASSWORD:-staging-kg-v2}"
export NEO4J_DATABASE="${STAGING_DB:-neo4j}"

mkdir -p "$OUT_ROOT"

curl -sf --max-time 10 "$BASE_URL/models" > /dev/null \
  || { echo "generator not answering at $BASE_URL"; exit 1; }
curl -sf --max-time 10 http://localhost:8002/v1/models > /dev/null \
  || { echo "embedding encoder not answering on 8002"; exit 1; }
# A carrier count cannot tell a live index from one whose identifiers went stale;
# this checks that the carriers still resolve to a node.
conda run -n graphllm python scripts/kg/check_vector_index.py --min-resolving 1000 \
  || { echo "vector index unusable; rebuild with scripts/kg/kg_vector_index.py"; exit 1; }
echo "preflight ok — variant ${VARIANT} against ${NEO4J_URL}"

echo "=== ${VARIANT} start : $(date -Is) ==="
conda run --no-capture-output -n graphllm python -m graphrag.cli --experiment \
  --questions-file "$GOLD" \
  --strategies "$STRATEGIES" \
  --llm --vllm --vllm-base-url "$BASE_URL" \
  --model-id "$MODEL" \
  --profile thesis_campaign \
  --max-new-tokens 1024 \
  --text-docs-dir artifacts/corpus_circular22 \
  --evidence-max-triple-items "${EVIDENCE_CAP:-30}" \
  ${EXTRA_ARGS} \
  --output-dir "${OUT_ROOT}/${VARIANT}" \
  --experiment-tag "kgv2_${VARIANT}"
echo "=== ${VARIANT} done : $(date -Is) ==="
