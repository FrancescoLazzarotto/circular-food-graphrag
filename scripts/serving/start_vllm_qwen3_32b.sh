#!/usr/bin/env bash
# Avvio vLLM per Qwen3-32B-AWQ (dense, non-MoE) su 1x NVIDIA A40 46GB — GPU 0.
#
# Modello di riferimento per la demo: prende il posto di Qwen2.5-32B-AWQ sulla
# porta 8000, quindi product/app.py lo trova senza cambiare DEMO_VLLM_ENDPOINTS.
# Checkpoint AWQ 4-bit (~19GB pesi), kernel Marlin su Ampere (sm86).
#
# Thinking disattivato via chat template (scripts/serving/chat_templates/qwen3_nothink.jinja):
# Qwen3-32B e' un modello ibrido e di default emette blocchi <think>...</think>,
# che finirebbero nella risposta mostrata all'esperto e mangerebbero il budget
# di DEMO_MAX_NEW_TOKENS. Il template e' quello ufficiale del repo con la sola
# condizione finale invertita: enable_thinking assente => niente thinking.
# Un client puo' comunque riattivarlo con chat_template_kwargs {"enable_thinking": true}.
# Per questo NON si passa --reasoning-parser: senza <think> in output servirebbe
# solo a rischiare content vuoto.
#
# Flag chiave (speculari a start_vllm_qwen3.sh):
#   --enable-prefix-caching   riusa KV cache per prefissi condivisi
#   --enable-chunked-prefill  riduce latency spikes su prompt lunghi
#   --max-num-seqs 16         continuous batching
#   --max-model-len 32768     allineato agli altri server

MODEL="${VLLM_QWEN3_32B_MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${VLLM_QWEN3_32B_PORT:-8000}"
GPU="${VLLM_QWEN3_32B_GPU:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${VLLM_QWEN3_32B_CHAT_TEMPLATE:-$SCRIPT_DIR/chat_templates/qwen3_nothink.jinja}"

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
  --chat-template "$CHAT_TEMPLATE" \
  --port "$PORT" \
  --host "$VLLM_HOST"
