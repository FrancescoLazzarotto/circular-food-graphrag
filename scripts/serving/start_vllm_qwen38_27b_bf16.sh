#!/usr/bin/env bash
# vLLM: Qwen3.8-27B BF16 across both A40s (tensor parallel), port 8000.
#
# 55.6 GB of weights need both GPUs: stop every other generator first. The
# encoder on :8002 stays up; UTIL 0.88 leaves it room on GPU 1. Over PCIe
# without NVLink this is slower than the INT4 on one card: it buys weight
# fidelity, not speed. dtype stays auto (float16 breaks this family); same
# no-think template as the INT4.

MODEL="${VLLM_QWEN38_BF16_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${VLLM_QWEN38_BF16_PORT:-8000}"
GPUS="${VLLM_QWEN38_BF16_GPUS:-0,1}"
UTIL="${VLLM_QWEN38_BF16_UTIL:-0.88}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${VLLM_QWEN38_BF16_CHAT_TEMPLATE:-$SCRIPT_DIR/chat_templates/qwen38_nothink.jinja}"

# Serving virtualenv: `import vllm` is broken in the graphllm conda env.
VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"

export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

# Loopback by default: these servers have no authentication and two A40s
# behind them. Export VLLM_HOST=0.0.0.0 to open them deliberately.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

exec env CUDA_VISIBLE_DEVICES="$GPUS" "$VLLM_BIN" serve "$MODEL" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":0}' \
  --chat-template "$CHAT_TEMPLATE" \
  --port "$PORT" \
  --host "$VLLM_HOST"
