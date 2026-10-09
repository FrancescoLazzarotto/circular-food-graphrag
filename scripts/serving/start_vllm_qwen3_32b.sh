#!/usr/bin/env bash
# vLLM: Qwen3-32B-AWQ on port 8000 — the graph's extractor.
#
# Defaults to GPU 0; set VLLM_QWEN3_32B_GPU=1 where GPU 0 is taken.
# qwen3_nothink.jinja turns thinking off unless the client asks for it, so no
# --reasoning-parser.

MODEL="${VLLM_QWEN3_32B_MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${VLLM_QWEN3_32B_PORT:-8000}"
GPU="${VLLM_QWEN3_32B_GPU:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${VLLM_QWEN3_32B_CHAT_TEMPLATE:-$SCRIPT_DIR/chat_templates/qwen3_nothink.jinja}"

# Serving virtualenv: `import vllm` is broken in the graphllm conda env.
VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"

export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

# Loopback by default: these servers have no authentication and two A40s
# behind them. Export VLLM_HOST=0.0.0.0 to open them deliberately.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

exec env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.87 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --chat-template "$CHAT_TEMPLATE" \
  --port "$PORT" \
  --host "$VLLM_HOST"
