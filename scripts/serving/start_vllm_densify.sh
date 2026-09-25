#!/usr/bin/env bash
# Second Qwen2.5-32B-AWQ, on GPU 1, port 8001 — for the KG v2 densification pass.
#
# GPU 0 already serves the same model on port 8000 for the demo and the
# experiments. Densification is thousands of LLM calls that would otherwise queue
# behind them; GPU 1 holds only the e5 encoder (0.12 utilisation on port 8002),
# so there is room for a second AWQ copy alongside it.
#
# Same venv reason as start_vllm_qwen25_7b.sh: the `graphllm` conda env has a
# torch that breaks the vLLM import.
#
# Utilisation is 0.72, not 0.87: the encoder on 8002 already holds a slice of
# this GPU and the two must not fight over it.

MODEL="${VLLM_DENSIFY_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
PORT="${VLLM_DENSIFY_PORT:-8001}"
GPU="${VLLM_DENSIFY_GPU:-1}"

VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"

export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

# Loopback by default: these servers have no authentication and two A40s
# behind them. Export VLLM_HOST=0.0.0.0 to open them deliberately.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

exec env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.72 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --port "$PORT" \
  --host "$VLLM_HOST"
