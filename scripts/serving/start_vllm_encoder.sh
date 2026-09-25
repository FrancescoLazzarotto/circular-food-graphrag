#!/usr/bin/env bash
# Multilingual sentence encoder for the cross-lingual vector channel — GPU 1.
#
# This is the half of retrieval that crosses the language gap: the graph is
# largely Italian, the gold questions are English, and lexical lookup cannot
# bridge that. Without this server the vector channel is unavailable and
# retrieval silently falls back to lexical-only.
#
# `--runner pooling` serves the model as an embedder rather than a generator.
# 0.12 memory utilisation leaves GPU 1 free for a generation server alongside it;
# max-model-len 512 is the e5 family's own limit.
#
# The index and the query encoder must use the SAME model and prefixes — see
# src/graphrag/embeddings.py. Changing MODEL here means rebuilding the index with
# scripts/kg/kg_vector_index.py.
set -euo pipefail

MODEL="${GRAPHRAG_EMBED_MODEL:-intfloat/multilingual-e5-base}"
PORT="${EMBED_PORT:-8002}"
GPU="${EMBED_GPU:-1}"
UTIL="${EMBED_GPU_UTIL:-0.12}"

VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"
export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" | grep -q '"id"'; then
  echo "encoder already serving on port ${PORT}"
  exit 0
fi

# Loopback by default: these servers have no authentication and two A40s
# behind them. Export VLLM_HOST=0.0.0.0 to open them deliberately.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

exec env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
  --runner pooling \
  --port "$PORT" \
  --host "$VLLM_HOST" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len 512
