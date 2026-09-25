#!/usr/bin/env bash
# Avvio vLLM per Qwen3.8-27B in BF16 (non quantizzato) su 2x NVIDIA A40 46GB.
#
# 55.6GB di pesi non stanno su una A40: servono entrambe le GPU con
# --tensor-parallel-size 2. Occupa quindi TUTTA la macchina tranne la fetta
# dell'encoder, e va fermato ogni altro generatore prima di partire
# (start_vllm.sh su :8000, start_vllm_gemma4_31b.sh su :8001).
#
# L'encoder su :8002 resta acceso di proposito: tiene ~1.1GB reali su GPU 1 e
# senza di lui il canale vettoriale muore, che su questo corpus bilingue e' meta'
# del retrieval (src/graphrag/kg/retriever.py solleva invece di degradare).
# UTIL 0.88 e' calcolato lasciandogli quel margine: 0.88 * 46068 = 40.5GB per
# scheda contro i 44.9GB liberi su GPU 1.
#
# Costo da mettere in conto: queste due A40 sono su PCIe, senza NVLink. Il
# tensor-parallel scambia attivazioni sul bus a ogni layer, quindi il BF16 su
# due schede e' piu' LENTO dell'INT4 su una sola (start_vllm_qwen38_27b.sh).
# Il non quantizzato compra fedelta' dei pesi, non velocita'.
#
# --dtype non si passa: il checkpoint e' BF16 e 'auto' lo rispetta. Forzare
# float16 su questa famiglia rompe, come gia' annotato in start_vllm_qwen3.sh.
#
# Thinking: il template di serie ha enable_thinking a default TRUE. Si riusa lo
# stesso qwen38_nothink.jinja dell'INT4 — i due repo pubblicano un template
# byte-identico.

MODEL="${VLLM_QWEN38_BF16_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${VLLM_QWEN38_BF16_PORT:-8000}"
GPUS="${VLLM_QWEN38_BF16_GPUS:-0,1}"
UTIL="${VLLM_QWEN38_BF16_UTIL:-0.88}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${VLLM_QWEN38_BF16_CHAT_TEMPLATE:-$SCRIPT_DIR/chat_templates/qwen38_nothink.jinja}"

# Venv di serving dedicato: l'env conda `graphllm` ha torch che rompe
# l'import di vLLM. Questo venv tiene vLLM col suo torch pinnato.
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
