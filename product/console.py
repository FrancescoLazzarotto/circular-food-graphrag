#!/usr/bin/env python3
"""Interactive demo console for domain-expert evaluation sessions.

A minimal REPL over the GraphRAG agent, designed so a non-technical expert can
query the knowledge graph directly: type a question, read the answer with its
graph evidence. Every exchange is appended to a JSONL session log (questions
collected this way seed the gold set for the new domain).

Answer quality is configured in ``product.config``, shared with the
Streamlit demo: citations, language pin, verbatim definitions, MMR, domain gate,
the cross-lingual vector channel. Neither demo sets its own, so the two cannot
answer the same question differently.

Usage:
    conda run -n graphllm python product/console.py
    # options: --strategy hybrid --max-context-tokens 6000 --model-id ...

Exit with 'esci', 'exit', 'quit' or Ctrl-D. 'nuova' starts a fresh thread
(clears the follow-up memory). Errors never kill the session: they are logged
with the stack trace and the expert sees a polite retry message.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import readline  # noqa: F401 - enables line editing/history in input()
import uuid
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Streamlit and `python product/console.py` both put this file's own directory
# on the path, not the repository root, so `product.config` needs it added.
sys.path.insert(0, str(ROOT))

from graphrag.strategies import STRATEGY_PRESETS  # noqa: E402

logger = logging.getLogger("expert_demo")

BANNER = """
============================================================
  Demo GraphRAG — Economia Circolare del Cibo
  Scrivi una domanda e premi Invio.
  Comandi: 'nuova' per ripartire da zero, 'esci' per uscire.
  Le risposte citano le fonti (documento | pagine) quando disponibili.
============================================================
"""

# CLI flag -> environment variable read by product.config. Passing a flag
# sets the variable before that module is imported, so the console and the
# Streamlit demo are configured through exactly one code path.
ENV_OVERRIDES = {
    "strategy": "DEMO_STRATEGY",
    "env_file": "DEMO_ENV_FILE",
    "max_context_tokens": "DEMO_MAX_CONTEXT_TOKENS",
    "max_new_tokens": "DEMO_MAX_NEW_TOKENS",
    "text_retriever_backend": "DEMO_TEXT_RETRIEVER_BACKEND",
    "log_dir": "DEMO_LOG_DIR",
    "complexity": "DEMO_COMPLEXITY",
    "vllm_endpoints": "DEMO_VLLM_ENDPOINTS",
}


def _parse_args() -> argparse.Namespace:
    """Parse the command line; the module docstring is the ``--help`` text."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Every default is None on purpose: an unset flag must leave the shared
    # default in place rather than overwrite it with a copy that can drift.
    parser.add_argument("--strategy", default=None, choices=sorted(STRATEGY_PRESETS))
    parser.add_argument("--model-id", default=None, help="Skips the model chooser")
    parser.add_argument("--vllm-base-url", default=None, help="Required with --model-id")
    parser.add_argument("--vllm-endpoints", default=None, help="Comma-separated, probed at startup")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--max-context-tokens", default=None)
    parser.add_argument("--max-new-tokens", default=None)
    parser.add_argument("--complexity", default=None, choices=("low", "medium", "high"))
    parser.add_argument("--text-retriever-backend", default=None, choices=("tfidf", "dense"))
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable the intra-session follow-up memory",
    )
    return parser.parse_args()


def _choose_model(options: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Ask which served model to use, when more than one endpoint answers.

    The best generator for this domain is not settled, so the demo offers
    whatever is currently served instead of pinning one.

    Args:
        options: Selector label -> ``(base_url, model_id)``.

    Returns:
        The chosen ``(base_url, model_id)``; the first one on an empty answer.
    """
    labels = sorted(options)
    if len(labels) == 1:
        return options[labels[0]]
    print("\nModelli disponibili:")
    for index, label in enumerate(labels, start=1):
        print(f"  {index}. {label}")
    while True:
        try:
            raw = input(f"Scegli [1-{len(labels)}, invio = 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not raw:
            return options[labels[0]]
        if raw.isdigit() and 1 <= int(raw) <= len(labels):
            return options[labels[int(raw) - 1]]
        print("Scelta non valida.")


def main() -> None:
    """Run the question loop, logging every turn to a JSONL session file."""
    args = _parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Compression/retrieval warnings are routine noise between questions and
    # would interleave with the expert's prompt; failures still surface.
    logging.getLogger("graphrag").setLevel(logging.ERROR)

    for attr, env_name in ENV_OVERRIDES.items():
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env_name] = str(value)
    if args.no_memory:
        os.environ["DEMO_MEMORY"] = "0"

    # Imported here, not at module scope: the overrides above must land in the
    # environment before product.config reads it.
    from graphrag.agent.memory import ConversationMemory
    from product import config as settings

    # The CLI's stage0 auto-discovery resolves kg_pipeline/artifacts relative
    # to the working directory.
    os.chdir(ROOT)

    if args.model_id or args.vllm_base_url:
        if not (args.model_id and args.vllm_base_url):
            sys.exit("--model-id e --vllm-base-url vanno passati insieme.")
        base_url, model_id = args.vllm_base_url, args.model_id
    else:
        options = settings.probe_vllm_endpoints()
        if not options:
            sys.exit(
                "Nessun modello raggiungibile su "
                f"{settings.VLLM_ENDPOINTS}. Avvia un server "
                "(scripts/serving/start_demo.sh) oppure passa --model-id/--vllm-base-url."
            )
        base_url, model_id = _choose_model(options)

    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    session_log = log_dir / f"session_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"

    print("\nAvvio in corso (connessione al grafo e indice testi)...")
    try:
        agent, graph_label = settings.build_demo_agent(base_url, model_id)
    except RuntimeError as exc:
        sys.exit(f"\n{exc}")

    memory = ConversationMemory() if settings.MEMORY else None

    print(BANNER)
    print(f"[strategia: {settings.STRATEGY} | modello: {model_id}]")
    print(f"[grafo: {graph_label}]")
    print(f"[log sessione: {session_log}]\n")

    n_questions = 0
    # Same three ids the Streamlit surface writes, for the same reason: the
    # feedback record points at a `turn_id`, several conversations share one
    # log file (`chat_id`), and `turn_index` says whether an answer was the
    # opening question or a follow-up. They keep the two surfaces' logs
    # comparable.
    chat_id = uuid.uuid4().hex[:8]
    turn_index = 0
    while True:
        try:
            question = input("Domanda> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"esci", "exit", "quit"}:
            break
        if question.lower() in {"nuova", "new", "reset"}:
            if memory is not None:
                memory.reset()
            chat_id = uuid.uuid4().hex[:8]
            turn_index = 0
            print("[nuova conversazione: il contesto precedente è stato azzerato]\n")
            continue

        started = time.perf_counter()
        turn_index += 1
        record: dict[str, object] = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            # The console and the Streamlit demo write different shapes into
            # files that look alike. These two fields are what lets a reader
            # tell them apart instead of inferring it from which keys exist.
            "surface": "console",
            "kind": "turn",
            "question": question,
            "strategy": settings.STRATEGY,
            "model_id": model_id,
            # Which graph answered. Without it a session served by the local
            # mirror during an Aura outage reads exactly like a healthy one.
            "graph_label": graph_label,
            "chat_id": chat_id,
            "turn_id": uuid.uuid4().hex[:12],
            "turn_index": turn_index,
        }
        try:
            print("... sto consultando il grafo e i documenti (10-30 secondi) ...")
            result = agent.invoke(question, memory=memory)
            answer = str(result.get("answer", "")).strip()
            elapsed = time.perf_counter() - started
            record["answer"] = answer
            record["latency_s"] = round(elapsed, 2)
            # Same split the Streamlit surface records: which stage spent the
            # seconds, not just how many there were.
            stage_timings = result.get("stage_timings_ms")
            if stage_timings:
                record["stage_timings_ms"] = stage_timings
            if result.get("rewritten_question"):
                record["rewritten_question"] = result["rewritten_question"]
            # The trailing "Verifica nel grafo:" block carries internal node
            # ids for debugging: keep it in the JSONL log, hide it on screen.
            shown = answer.split("\nVerifica nel grafo:")[0].strip()
            print(f"\n{shown}\n")
            print(f"[{elapsed:.0f}s]\n")
            n_questions += 1
        except Exception as exc:  # noqa: BLE001 - REPL must survive any failure
            elapsed = time.perf_counter() - started
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_s"] = round(elapsed, 2)
            logger.error("Question failed: %s\n%s", exc, traceback.format_exc())
            print(
                "\nSi è verificato un problema tecnico con questa domanda. "
                "Riprova, magari riformulandola.\n"
            )
        with session_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Sessione terminata: {n_questions} domande. Log: {session_log}")


if __name__ == "__main__":
    main()
