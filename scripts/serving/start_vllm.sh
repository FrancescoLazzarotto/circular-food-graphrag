#!/usr/bin/env bash
# Avvio vLLM ottimizzato per 1x NVIDIA A40 46GB
#
# Flag chiave:
#   --enable-prefix-caching   RadixAttention: riusa KV cache per prefissi condivisi
#                             (system prompt + relation vocab ripetuti ogni chunk = speedup ~20-40%)
#   --enable-chunked-prefill  Riduce latency spikes su prompt lunghi
#   --max-num-seqs 16         Continuous batching fino a 16 request parallele
#   --gpu-memory-utilization  Lascia ~6GB per GLiNER + SentenceTransformer + CUDA overhead
#   --max-model-len 32768     Allineato agli altri wrapper. Non e' un lusso: vLLM
#                             rifiuta la richiesta se prompt + max_tokens supera
#                             il tetto, quindi con gli 8192 di prima e
#                             KG_EXTRACTION_MAX_TOKENS=4096 il prompt massimo
#                             accettato era 4096 token — e 222 chunk del corpus
#                             (10,2%, fino a 7801 token) lo superano. Finivano in
#                             errore 400, e il retry che raddoppia max_tokens
#                             peggiorava il rifiuto invece di ripararlo.
#
# Speculative decoding (opzionale, ~2-3x speedup su generation, costa ~3-4GB VRAM):
#   Decommentare le righe --speculative-model / --num-speculative-tokens
#   e abbassare --gpu-memory-utilization a 0.83

MODEL="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8000}"
GPU="${VLLM_GPU:-0}"

# Bare `vllm` resolves to the conda env, where `import vllm` is broken: this
# script died on "vllm: not found" while every other start_vllm*.sh had already
# been pointed at the serving virtualenv. Same treatment here.
VLLM_BIN="${VLLM_BIN:-/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm}"
export HF_HOME="${HF_HOME:-/mnt/storage/hf-cache}"

# Loopback by default: these servers have no authentication and two A40s
# behind them, and they were bound to 0.0.0.0. Export VLLM_HOST=0.0.0.0 to
# open them deliberately.
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
