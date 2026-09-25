#!/usr/bin/env bash
# The servers the demo can run, and where each one listens.
#
# Sourced by start_demo.sh and stop_demo.sh, so the two cannot keep diverging
# copies: a stop script that misses a generator or a port reports success and
# leaves the process holding the port, and the next start then says "already
# up" and serves the old build.
#
# Ports come from the same environment variables start_demo.sh honours, so a
# demo started with DEMO_UI_PORT=8600 is a demo stop_demo.sh can find.

# key -> "start script|port|gpus|description"
declare -A MODELS=(
  [qwen25-32b]="start_vllm.sh|8000|0|Qwen2.5-32B-AWQ — the generator the thesis numbers were measured on"
  [qwen3-32b]="start_vllm_qwen3_32b.sh|8000|0|Qwen3-32B-AWQ — newer, reasoning model (verbose unless thinking is off)"
  [qwen25-7b]="start_vllm_qwen25_7b.sh|8001|1|Qwen2.5-7B — small, fast, for side-by-side comparison"
  [qwen3-30b-a3b]="start_vllm_qwen3.sh|8001|1|Qwen3-30B-A3B-FP8 — MoE, cheap to run"
  [qwen38-27b-bf16]="start_vllm_qwen38_27b_bf16.sh|8000|0,1|Qwen3.8-27B BF16 — unquantised, needs BOTH GPUs, slower over PCIe"
  [qwen38-27b]="start_vllm_qwen38_27b.sh|8001|1|Qwen3.8-27B-INT4 — dense, hybrid attention, thinking off via template"
  [gemma4-31b]="start_vllm_gemma4_31b.sh|8001|1|Gemma-4-31B QAT w4a16 — dense, Apache 2.0, thinking off by default"
  [qwen25-72b]="start_vllm_qwen25_72b.sh|8000|0,1|Qwen2.5-72B-AWQ — largest available, needs BOTH GPUs"
)

# label -> port, for every process start_demo.sh can launch. Generator ports are
# derived from MODELS rather than restated, so the two cannot disagree.
declare -A LABEL_PORTS=(
  [streamlit]="${DEMO_UI_PORT:-8501}"
  [encoder]="${EMBED_PORT:-8002}"
)
for _key in "${!MODELS[@]}"; do
  IFS='|' read -r _ _port _ _ <<<"${MODELS[$_key]}"
  LABEL_PORTS[$_key]="$_port"
done
unset _key _port
