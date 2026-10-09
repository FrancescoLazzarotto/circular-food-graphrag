#!/usr/bin/env bash
# vLLM: Qwen2.5-72B-Instruct-AWQ across both A40s (tensor parallel), port 8000.
#
# The largest dense Qwen available. It takes both GPUs: stop every other
# server first.

MODEL="${VLLM_QWEN25_72B_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
PORT="${VLLM_QWEN25_72B_PORT:-8000}"
GPUS="${VLLM_QWEN25_72B_GPUS:-0,1}"
# Lower it when the encoder is already holding a slice of GPU 1, or the
# allocation fails outright instead of starting smaller.
UTIL="${VLLM_QWEN25_72B_UTIL:-0.90}"

# Serving virtualenv: `import vllm` is broken in the graphllm conda env.
VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"

export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

# Loopback by default: these servers have no authentication and two A40s
# behind them. Export VLLM_HOST=0.0.0.0 to open them deliberately.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

exec env CUDA_VISIBLE_DEVICES="$GPUS" "$VLLM_BIN" serve "$MODEL" \
  --tensor-parallel-size 2 \
  --quantization awq_marlin \
  --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --port "$PORT" \
  --host "$VLLM_HOST"
