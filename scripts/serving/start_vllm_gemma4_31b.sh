#!/usr/bin/env bash
# vLLM: Gemma-4-31B-it QAT w4a16 on GPU 1, port 8001 — a generator to compare.
#
# Google's own QAT checkpoint, loaded natively with Marlin kernels on sm_86.
# Thinking is off by default in its template. Image inputs are disabled so
# vLLM reserves no memory for the vision tower.
#
# UTIL 0.85: the encoder on :8002 already holds 0.12 of GPU 1; a higher value
# fails to allocate instead of starting smaller.

MODEL="${VLLM_GEMMA4_MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"
PORT="${VLLM_GEMMA4_PORT:-8001}"
GPU="${VLLM_GEMMA4_GPU:-1}"
UTIL="${VLLM_GEMMA4_UTIL:-0.85}"

# Serving virtualenv: `import vllm` is broken in the graphllm conda env.
VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"

export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

# Loopback by default: these servers have no authentication and two A40s
# behind them. Export VLLM_HOST=0.0.0.0 to open them deliberately.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

exec env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":0}' \
  --port "$PORT" \
  --host "$VLLM_HOST"
