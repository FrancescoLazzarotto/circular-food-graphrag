#!/usr/bin/env bash
# Avvio vLLM per Gemma-4-31B-it QAT w4a16 (dense, 30.7B) su 1x NVIDIA A40 46GB — GPU 1.
#
# Convive con il generatore della demo su GPU 0 (porta 8000) e con l'encoder
# e5 su GPU 1 (porta 8002, util 0.12): questo serve su porta 8001 e vede SOLO
# la GPU 1 via CUDA_VISIBLE_DEVICES. Serve per l'A/B contro il generatore in
# produzione senza fermare la demo.
#
# Checkpoint: quantizzazione QAT ufficiale Google in formato compressed-tensors
# (pack-quantized, w4a16, group 32, ~23GB pesi). Non e' un PTQ community: la
# quantizzazione e' stata addestrata, quindi la perdita rispetto a bf16 e'
# molto inferiore a un AWQ post-hoc. vLLM la carica nativa con kernel Marlin
# su Ampere (sm_86) — nessun flag --quantization esplicito serve.
#
# Thinking: il chat template canonico di Gemma 4 ha `enable_thinking` a
# default false, quindi NON serve il trattamento con template modificato che
# Qwen3-32B richiede (scripts/serving/chat_templates/qwen3_nothink.jinja).
# Un client puo' riattivarlo con chat_template_kwargs {"enable_thinking": true}.
#
# --limit-mm-per-prompt: Gemma 4 e' multimodale (Gemma4ForConditionalGeneration).
# La pipeline passa solo testo, quindi azzerare le immagini evita che vLLM
# riservi memoria di profiling per il vision tower.
#
# --gpu-memory-utilization 0.85: l'encoder su :8002 tiene gia' 0.12 della
# stessa GPU. Alzarlo fa fallire l'allocazione invece di partire piu' piccolo.

MODEL="${VLLM_GEMMA4_MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"
PORT="${VLLM_GEMMA4_PORT:-8001}"
GPU="${VLLM_GEMMA4_GPU:-1}"
UTIL="${VLLM_GEMMA4_UTIL:-0.85}"

# Venv di serving dedicato: l'env conda `graphllm` ha torch che rompe
# l'import di vLLM. Questo venv tiene vLLM col suo torch pinnato.
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
