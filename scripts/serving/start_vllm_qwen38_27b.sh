#!/usr/bin/env bash
# Avvio vLLM per Qwen3.8-27B INT4 (dense, 27B) su 1x NVIDIA A40 46GB — GPU 1.
#
# Candidato dell'A/B sul generatore, come Gemma-4-31B. Serve sulla stessa
# porta 8001: i due non convivono su GPU 1, si provano in sequenza.
#
# Checkpoint: RedHatAI/Qwen3.8-27B-INT4 (~19.5GB pesi). Qwen ha pubblicato solo
# FP8 (~31GB) come quantizzazione ufficiale; su Ampere l'FP8 e' weight-only via
# Marlin, quindi non compra niente in velocita' e costa 11GB in piu' di VRAM.
# L'INT4 e' la scelta giusta per questa macchina.
#
# Architettura: Qwen3_5ForConditionalGeneration, ibrida
# 16 x (3 x GatedDeltaNet -> 1 x GatedAttention). Solo 16 layer su 64 hanno
# attenzione piena, quindi la KV cache e' molto piu' piccola di un dense
# classico di pari taglia: i contesti RAG lunghi costano poco. In cambio i
# kernel DeltaNet su sm_86 sono meno battuti di quelli di attenzione standard:
# se il caricamento fallisce, e' il primo posto dove guardare.
#
# Thinking: il template di serie ha `enable_thinking` a default TRUE, al
# contrario di Gemma 4. Senza override i blocchi <think> finirebbero nella
# risposta mostrata all'esperto e mangerebbero DEMO_MAX_NEW_TOKENS. Il
# template qui sotto e' quello del repo con le due condizioni invertite:
# thinking solo se il client lo chiede con chat_template_kwargs
# {"enable_thinking": true}. Stesso trattamento di qwen3_nothink.jinja.
# Per questo NON si passa --reasoning-parser.
#
# --gpu-memory-utilization 0.85: l'encoder su :8002 tiene gia' 0.12 di GPU 1.

# Revisione pinnata, non `main`: la revisione bf08f3db di RedHatAI aggiunge la
# quantizzazione FP8 della KV cache
# (`kv_cache_scheme`, num_bits 8). L'A40 e' sm_86 e non ha FP8 hardware: il
# backend Triton rifiuta di compilare ("type fp8e4nv not supported in this
# architecture") e FlashInfer, che resta l'unico candidato, prova a emularlo e
# genera testo spazzatura fin dal primo token. Con 2fb0debc la KV cache resta
# bf16, FLASH_ATTN torna eleggibile e il modello risponde correttamente. Da
# rivedere solo su hardware Hopper o piu' recente.
MODEL="${VLLM_QWEN38_MODEL:-RedHatAI/Qwen3.8-27B-INT4}"
REVISION="${VLLM_QWEN38_REVISION:-2fb0debc365fb6c1683d7d3ad7722470919627a8}"
PORT="${VLLM_QWEN38_PORT:-8001}"
GPU="${VLLM_QWEN38_GPU:-1}"
UTIL="${VLLM_QWEN38_UTIL:-0.85}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${VLLM_QWEN38_CHAT_TEMPLATE:-$SCRIPT_DIR/chat_templates/qwen38_nothink.jinja}"

# Venv di serving dedicato: l'env conda `graphllm` ha torch che rompe
# l'import di vLLM. Questo venv tiene vLLM col suo torch pinnato.
VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"

export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

# Loopback by default: these servers have no authentication and two A40s
# behind them. Export VLLM_HOST=0.0.0.0 to open them deliberately.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

exec env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
  --revision "$REVISION" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$UTIL" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":0}' \
  --chat-template "$CHAT_TEMPLATE" \
  --port "$PORT" \
  --host "$VLLM_HOST"
