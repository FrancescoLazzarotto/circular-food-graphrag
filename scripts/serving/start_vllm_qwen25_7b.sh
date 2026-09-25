#!/usr/bin/env bash
# Avvio vLLM per Qwen2.5-7B-Instruct su 1x NVIDIA A40 46GB — GPU 1.
#
# Convive con il server Qwen2.5-32B-AWQ della demo (GPU 0, porta 8000):
# questo serve su porta 8001 e vede SOLO la GPU 1 via CUDA_VISIBLE_DEVICES.
#
# Modello: checkpoint BF16 ufficiale (~15GB pesi), gira largo su A40 46GB.
#
# Flag chiave (speculari a start_vllm_qwen3.sh):
#   --enable-prefix-caching   riusa KV cache per prefissi condivisi
#   --enable-chunked-prefill  riduce latency spikes su prompt lunghi
#   --max-num-seqs 16         continuous batching
#   --max-model-len 32768     allineato agli altri server

MODEL="${VLLM_QWEN25_7B_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_QWEN25_7B_PORT:-8001}"
GPU="${VLLM_QWEN25_7B_GPU:-1}"

# Venv di serving dedicato: l'env conda `graphllm` ha torch che rompe
# l'import di vLLM. Questo venv tiene vLLM col suo torch pinnato.
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
