#!/usr/bin/env bash
# The 30 reference questions asked in Italian instead of English.
#
# Far more expected concept slots exist in the graph only under an Italian name
# than are reachable under an English one. The reference set is English, so
# every reference number is measured on the harder side of that asymmetry. This
# run measures the other side: identical
# annotation, identical configuration, identical server session, and only the
# language of the question changed.
#
# Its control is the a1_repaired_prompt arm of run_abstention_arms.sh, which is
# the same configuration asked in English. Run this after that script so the two
# share a server session and the cross-session noise band does not apply.
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT_ROOT="${OUT_ROOT:-/srv/projects/graphllm/experiments/exp_results_crosslingual}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
GOLD_IT="${GOLD_IT:-evaluation/gold/gold_v3_it.json}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
STRATEGIES="default,hybrid,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path"

mkdir -p "$OUT_ROOT"

curl -sf --max-time 10 "$BASE_URL/models" > /dev/null \
  || { echo "generator not answering at $BASE_URL"; exit 1; }
curl -sf --max-time 10 http://localhost:8002/v1/models > /dev/null \
  || { echo "embedding encoder not answering on 8002"; exit 1; }
# See the note in run_abstention_arms.sh: a carrier count cannot tell a live
# index from one whose identifiers went stale under a store reload.
conda run -n graphllm python scripts/kg/check_vector_index.py --min-resolving 1000 \
  || { echo "vector index unusable; rebuild with scripts/kg/kg_vector_index.py"; exit 1; }
[ -f "$GOLD_IT" ] || { echo "missing $GOLD_IT: run scripts/gold/build_gold_it.py"; exit 1; }
echo "preflight ok: generator, encoder, resolving vector index, Italian reference set"

echo "=== italian arm : $(date -Is) ==="
conda run --no-capture-output -n graphllm python -m graphrag.cli --experiment \
  --questions-file "$GOLD_IT" \
  --strategies "$STRATEGIES" \
  --llm --vllm --vllm-base-url "$BASE_URL" \
  --model-id "$MODEL" \
  --profile thesis_campaign \
  --max-new-tokens 1024 \
  --text-docs-dir artifacts/corpus_circular22 \
  --evidence-max-triple-items "${EVIDENCE_CAP:-30}" \
  --output-dir "${OUT_ROOT}/it_questions" \
  --experiment-tag "crosslingual_it"
echo "=== italian arm done : $(date -Is) ==="
