#!/usr/bin/env bash
# vLLM: Qwen3.8-27B-INT4 on GPU 1, port 8001 — the demo's generator.
#
# Hybrid attention (16 of 64 layers keep a KV cache), so long RAG contexts are
# cheap. The stock template turns thinking on; qwen38_nothink.jinja turns it
# off unless the client asks (chat_template_kwargs {"enable_thinking": true}),
# hence no --reasoning-parser.
#
# UTIL (default 0.70) follows the KV cache the task needs; the encoder on
# :8002 already holds 0.12 of GPU 1.
#
# Revision pinned: later ones quantise the KV cache to FP8, which the A40
# (sm_86) cannot run and FlashInfer emulates into garbage output.

MODEL="${VLLM_QWEN38_MODEL:-RedHatAI/Qwen3.8-27B-INT4}"
REVISION="${VLLM_QWEN38_REVISION:-2fb0debc365fb6c1683d7d3ad7722470919627a8}"
PORT="${VLLM_QWEN38_PORT:-8001}"
GPU="${VLLM_QWEN38_GPU:-1}"
UTIL="${VLLM_QWEN38_UTIL:-0.70}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${VLLM_QWEN38_CHAT_TEMPLATE:-$SCRIPT_DIR/chat_templates/qwen38_nothink.jinja}"

# Serving virtualenv: `import vllm` is broken in the graphllm conda env.
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
