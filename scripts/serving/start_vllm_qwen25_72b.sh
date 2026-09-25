#!/usr/bin/env bash
# Avvio vLLM per Qwen2.5-72B-Instruct-AWQ splittato su 2x NVIDIA A40 46GB
# (tensor-parallel-size 2, GPU 0+1). Unico modello ~70B dense reale
# disponibile in famiglia Qwen: 3.5/3.6 hanno droppato il tier dense
# 70B a favore di MoE (35B-A3B) o 397B-A17B.
#
# Occupa ENTRAMBE le GPU: nessun altro server (demo 32B, 7B, embedding
# encoder :8002) puo' girare in contemporanea. Ferma tutto prima.
#
# Checkpoint AWQ int4 ~40GB pesi, margine KV cache su 92GB totali.

MODEL="${VLLM_QWEN25_72B_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
PORT="${VLLM_QWEN25_72B_PORT:-8000}"
GPUS="${VLLM_QWEN25_72B_GPUS:-0,1}"
# Lower it when the encoder is already holding a slice of GPU 1, or the
# allocation fails outright instead of starting smaller.
UTIL="${VLLM_QWEN25_72B_UTIL:-0.90}"

# Venv di serving dedicato: l'env conda `graphllm` ha torch che rompe
# l'import di vLLM. Questo venv tiene vLLM col suo torch pinnato.
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
