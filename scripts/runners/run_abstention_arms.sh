#!/usr/bin/env bash
# Three arms measuring the abstention path, all in one server session.
#
# The abstention path has two parts: the closing line of the answer prompt, in
# its pre-repair wording (insufficiency may be declared only for an empty
# context) or its repaired one, and the domain gate, a terminal state that
# produces a refusal.
#
# The arms differ in the abstention mechanism and in nothing else:
#   A0  pre-repair wording, no gate      the reference-campaign configuration
#   A1  repaired wording, no gate        isolates the prompt line
#   A2  repaired wording, domain gate    adds the terminal refusal state
#
# One server session throughout, so the comparison is within-session and the
# +/-0.03 cross-session band does not apply.
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT_ROOT="${OUT_ROOT:-/srv/projects/graphllm/experiments/exp_results_abstention}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
GOLD="${GOLD:-evaluation/gold/gold_v3.json}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
STRATEGIES="default,hybrid,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path"

mkdir -p "$OUT_ROOT"

preflight() {
  curl -sf --max-time 10 "$BASE_URL/models" > /dev/null \
    || { echo "generator not answering at $BASE_URL"; exit 1; }
  curl -sf --max-time 10 http://localhost:8002/v1/models > /dev/null \
    || { echo "embedding encoder not answering on 8002"; exit 1; }
  # Counting carriers is not enough. A reload of the graph store reassigns every
  # internal identifier, which leaves the carriers in place and pointing at
  # nothing: the count still passes, the vector channel silently degrades to
  # lexical matching, and the campaign looks complete. Check that the
  # identifiers still resolve.
  conda run -n graphllm python scripts/kg/check_vector_index.py --min-resolving 1000 \
    || { echo "vector index unusable; rebuild with scripts/kg/kg_vector_index.py"; exit 1; }
  echo "preflight ok: generator, encoder and a resolving vector index"
}

run_arm() {
  local tag="$1"; shift
  echo "=== arm ${tag} : $(date -Is) ==="
  conda run --no-capture-output -n graphllm python -m graphrag.cli --experiment \
    --questions-file "$GOLD" \
    --strategies "$STRATEGIES" \
    --llm --vllm --vllm-base-url "$BASE_URL" \
    --model-id "$MODEL" \
    --profile thesis_campaign \
    --max-new-tokens 1024 \
    --text-docs-dir artifacts/corpus_circular22 \
    --evidence-max-triple-items "${EVIDENCE_CAP:-30}" \
    --output-dir "${OUT_ROOT}/${tag}" \
    --experiment-tag "abst_${tag}" \
    "$@"
  echo "=== arm ${tag} done : $(date -Is) ==="
}

preflight
run_arm a0_legacy_prompt --legacy-insufficiency-wording
run_arm a1_repaired_prompt
run_arm a2_repaired_plus_gate --enable-domain-gate
echo "ALL ARMS COMPLETE $(date -Is)"
