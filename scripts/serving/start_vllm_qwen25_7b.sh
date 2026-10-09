#!/usr/bin/env bash
# vLLM: Qwen2.5-7B-Instruct (BF16) on GPU 1, port 8001 — a small generator to compare.

MODEL="${VLLM_QWEN25_7B_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_QWEN25_7B_PORT:-8001}"
GPU="${VLLM_QWEN25_7B_GPU:-1}"

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
  --port "$PORT" \
  --host "$VLLM_HOST"
