#!/usr/bin/env bash
# vLLM: Qwen2.5-32B-Instruct-AWQ on GPU 0, port 8000.
#
# --max-model-len 32768: vLLM rejects a request whose prompt plus max_tokens
# exceeds it, and extraction asks for up to 4096 new tokens.
# Speculative decoding: uncomment the two lines at the end and lower
# --gpu-memory-utilization to 0.83.

MODEL="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8000}"
GPU="${VLLM_GPU:-0}"

# Bare `vllm` resolves to the conda env, where `import vllm` is broken, so this
# script uses the serving virtualenv like every other start_vllm*.sh.
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
  --dtype float16 \
  --port "$PORT" \
  --host "$VLLM_HOST"
  # --speculative-model Qwen/Qwen2.5-1.5B-Instruct \
  # --num-speculative-tokens 5
