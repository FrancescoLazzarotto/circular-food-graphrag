#!/usr/bin/env bash
# Avvio vLLM per Qwen3-30B-A3B (MoE, 3B attivi) su 1x NVIDIA A40 46GB — GPU 1.
#
# Convive con il server Qwen2.5-32B-AWQ della demo (GPU 0, porta 8000):
# questo serve su porta 8001 e vede SOLO la GPU 1 via CUDA_VISIBLE_DEVICES.
#
# Modello: checkpoint FP8 ufficiale Qwen (~31GB pesi). Su Ampere (A40, sm86)
# vLLM usa i kernel Marlin per FP8 weight-only: pesi FP8, attivazioni BF16.
# NON passare --dtype float16: il checkpoint FP8 richiede dtype auto/bfloat16.
#
# Variante Instruct-2507 (non-thinking): niente blocchi <think>, output diretto
# — necessario per estrazione triple JSON e risposte a temperature 0.
#
# Flag chiave (speculari a start_vllm.sh):
#   --enable-prefix-caching   riusa KV cache per prefissi condivisi
#   --enable-chunked-prefill  riduce latency spikes su prompt lunghi
#   --max-num-seqs 16         continuous batching
#   --max-model-len 32768     allineato al server Qwen2.5 in produzione

MODEL="${VLLM_QWEN3_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
PORT="${VLLM_QWEN3_PORT:-8001}"
GPU="${VLLM_QWEN3_GPU:-1}"

# Venv di serving dedicato: l'env conda `graphllm` ha torch 2.10 che rompe
# l'import di vLLM 0.19 (ModuleNotFoundError: torch._inductor.custom_graph_pass).
# Questo venv tiene vLLM col suo torch pinnato, isolato dalla pipeline.
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
