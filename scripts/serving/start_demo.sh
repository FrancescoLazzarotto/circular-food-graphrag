#!/usr/bin/env bash
# Bring the whole demo up with one command: encoder, generator(s), UI.
#
# By hand it takes four terminals and four commands in the right order (the
# encoder has to claim its slice of GPU 1 before a generator takes the rest),
# and anything forgotten degrades the demo silently rather than failing: no
# encoder means no cross-lingual retrieval, and the answer just comes out
# thinner.
#
# The best generator for this domain has not been settled by measurement yet, so
# this script does not pin one. Name one or more model keys and every server
# that comes up appears in the demo's model selector, ready to be compared side
# by side on the same question.
#
#   scripts/serving/start_demo.sh                     # default generator + encoder + UI
#   scripts/serving/start_demo.sh --list              # what can be served
#   scripts/serving/start_demo.sh qwen25-32b qwen25-7b
#   scripts/serving/start_demo.sh qwen25-72b --no-ui  # both GPUs, no browser UI
#
# Logs and pids land in artifacts/demo_logs/. Stop everything with
# scripts/serving/stop_demo.sh.
set -euo pipefail

# ~/.local/lib/python3.12/site-packages shadows the conda env with an
# incompatible pair (torch 2.5.1+cu124 / sentence-transformers 5.3.0 against the
# env's 2.10.0+cu128 / 3.4.1). The two resolve module by module, and the mixture
# raises `ModuleNotFoundError: Could not import module 'PreTrainedModel'` the
# first time the dense text backend builds its embeddings — i.e. at demo
# startup, before the UI is reachable. The env's own packages are fine; only the
# shadowing is not. Exported here so both the preflight and streamlit inherit it.
export PYTHONNOUSERSITE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The per-model start scripts sit beside this one. Resolved once, absolute:
# the launcher cd's to ROOT before spawning, so a relative path would not
# survive being invoked from another directory.
SERVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${DEMO_LOG_DIR_RUNTIME:-$ROOT/artifacts/demo_logs}"
UI_PORT="${DEMO_UI_PORT:-8501}"
UI_ADDRESS="${DEMO_UI_ADDRESS:-0.0.0.0}"
CONDA_ENV="${DEMO_CONDA_ENV:-graphllm}"
ENCODER_PORT="${EMBED_PORT:-8002}"
# vLLM loading a 32B checkpoint from a cold page cache is minutes, not seconds.
BOOT_TIMEOUT_SEC="${DEMO_BOOT_TIMEOUT_SEC:-900}"

# MODELS and LABEL_PORTS: shared with stop_demo.sh, which has to find the same
# ports this script binds.
# shellcheck source=scripts/serving/_models.sh
source "$SERVING_DIR/_models.sh"

usage() {
  sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

list_models() {
  echo "Modelli disponibili:"
  for key in "${!MODELS[@]}"; do
    IFS='|' read -r _ port gpus desc <<<"${MODELS[$key]}"
    printf '  %-14s :%s  GPU %-3s  %s\n' "$key" "$port" "$gpus" "$desc"
  done | sort
}

WANTED=()
WITH_ENCODER=1
WITH_UI=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) list_models; exit 0 ;;
    -h|--help) usage; echo; list_models; exit 0 ;;
    --no-encoder) WITH_ENCODER=0 ;;
    --no-ui) WITH_UI=0 ;;
    --port) UI_PORT="$2"; shift ;;
    -*) echo "Opzione sconosciuta: $1" >&2; exit 2 ;;
    *)
      if [[ -z "${MODELS[$1]:-}" ]]; then
        echo "Modello sconosciuto: $1" >&2
        list_models >&2
        exit 2
      fi
      WANTED+=("$1")
      ;;
  esac
  shift
done
[[ ${#WANTED[@]} -eq 0 ]] && WANTED=("${DEMO_DEFAULT_MODEL:-qwen25-32b}")

mkdir -p "$LOG_DIR"

port_answers() {  # port
  curl -s --max-time 3 "http://localhost:$1/v1/models" | grep -q '"id"'
}

wait_for_port() {  # port, label, log file
  local port="$1" label="$2" log="$3" waited=0
  while ! port_answers "$port"; do
    if ! kill -0 "$(cat "$LOG_DIR/$label.pid" 2>/dev/null || echo 0)" 2>/dev/null; then
      echo "  $label è morto durante l'avvio. Ultime righe di $log:" >&2
      tail -n 20 "$log" >&2
      return 1
    fi
    if (( waited >= BOOT_TIMEOUT_SEC )); then
      echo "  $label non risponde dopo ${BOOT_TIMEOUT_SEC}s (log: $log)" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    printf '\r  attendo %s su :%s (%ds)...' "$label" "$port" "$waited"
  done
  printf '\r  %s pronto su :%s (%ds)          \n' "$label" "$port" "$waited"
}

start_server() {  # label, port, script, env assignments...
  local label="$1" port="$2" script="$3"; shift 3
  if port_answers "$port"; then
    echo "  $label già attivo su :$port"
    return 0
  fi
  local log="$LOG_DIR/$label.log"
  echo "  avvio $label -> :$port (log: $log)"
  # setsid --fork: its own session, so stop_demo.sh can kill the server and its
  # children as a group without signalling the shell that launched all of this.
  # The pid is written from inside that session and survives the exec, because
  # setsid's own pid is not the server's — setsid exits as soon as it has forked.
  ( cd "$ROOT" && setsid --fork bash -c \
      'echo $$ > "$1"; shift; exec "$@"' _ "$LOG_DIR/$label.pid" \
      env "$@" "$SERVING_DIR/$script" >"$log" 2>&1 < /dev/null & )
  # The pid file is written by the child; wait for it before polling it.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    [[ -s "$LOG_DIR/$label.pid" ]] && break
    sleep 0.5
  done
  wait_for_port "$port" "$label" "$log"
}

echo "== encoder =="
if [[ $WITH_ENCODER -eq 1 ]]; then
  start_server encoder "$ENCODER_PORT" start_vllm_encoder.sh "EMBED_PORT=$ENCODER_PORT"
else
  echo "  saltato (--no-encoder): il canale cross-lingua resterà spento"
fi

echo "== generatori =="
for key in "${WANTED[@]}"; do
  IFS='|' read -r script port _ _ <<<"${MODELS[$key]}"
  extra=()
  # The encoder already holds ~12 % of GPU 1; a generator asking for 0.90 of
  # both cards then fails to allocate instead of starting smaller. Keyed on the
  # model, not on the "0,1" layout: the variable name is the wrapper's own, and
  # a layout test would set a VLLM_QWEN25_72B_UTIL that qwen38-27b-bf16 never
  # reads. That one needs no override anyway — its 0.88 is already computed
  # against the 44.9 GB the encoder leaves free on GPU 1.
  if [[ "$key" == "qwen25-72b" && $WITH_ENCODER -eq 1 ]]; then
    extra+=("VLLM_QWEN25_72B_UTIL=${VLLM_QWEN25_72B_UTIL:-0.82}")
  fi
  start_server "$key" "$port" "$script" "${extra[@]}"
done

echo "== preflight =="
# Probe the generators this invocation actually started, not the single one
# named in kg_pipeline/.env: that variable pins one port and one model id, so
# probing it would fail the preflight for any other model — and the preflight
# is fatal.
LLM_PROBES=()
for key in "${WANTED[@]}"; do
  IFS='|' read -r _ port _ _ <<<"${MODELS[$key]}"
  LLM_PROBES+=(--llm-base-url "http://localhost:$port/v1")
done

conda run -n "$CONDA_ENV" python "$ROOT/scripts/smoke/smoke_check.py" \
  "${LLM_PROBES[@]}" \
  $([[ $WITH_ENCODER -eq 1 ]] || echo --skip-encoder) || {
    echo "Preflight fallito: la demo partirebbe degradata. Correggi e riprova." >&2
    exit 1
  }

if [[ $WITH_UI -eq 1 ]]; then
  echo "== interfaccia =="
  log="$LOG_DIR/streamlit.log"
  if curl -s --max-time 3 "http://localhost:$UI_PORT" >/dev/null 2>&1; then
    echo "  già attiva su :$UI_PORT"
  else
    ( cd "$ROOT" && setsid --fork bash -c \
        'echo $$ > "$1"; shift; exec "$@"' _ "$LOG_DIR/streamlit.pid" \
        # --no-capture-output: plain `conda run` buffers the child's stdio,
        # so the log file would stay empty and the UI could not be debugged.
        conda run --no-capture-output -n "$CONDA_ENV" streamlit run product/app.py \
        --server.address "$UI_ADDRESS" --server.port "$UI_PORT" >"$log" 2>&1 < /dev/null & )
    echo "  avviata su :$UI_PORT (log: $log)"
  fi
  echo
  echo "Dal tuo portatile:  ssh -L $UI_PORT:localhost:$UI_PORT <utente>@<server>"
  echo "poi apri:           http://localhost:$UI_PORT"
fi

echo
echo "Tutto su. Per fermare: scripts/serving/stop_demo.sh"
