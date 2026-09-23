#!/usr/bin/env python3
"""Streamlit front-end for the circular-food assistant.

Browser UI over the same GraphRAG agent used by product/console.py. The page
renders one `result` per turn: the answer's prose, the citation check, the
documents it cited and the evidence it retrieved. Everything shown is read from
that dict — the answer's own text is only split on its section headings, never
mined for data. Every exchange is logged to the JSONL format under
artifacts/demo_sessions/.

Usage (on the server):
    conda run -n graphllm streamlit run product/app.py --server.address 0.0.0.0 --server.port 8501

Then from your machine, open an SSH tunnel and browse to the forwarded port:
    ssh -L 8501:localhost:8501 <user>@<server>
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import queue
import threading
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Streamlit and `python product/console.py` both put this file's own directory
# on the path, not the repository root, so `product.config` needs it added.
sys.path.insert(0, str(ROOT))

from graphrag.agent.core import KGRAGAgent  # noqa: E402
from graphrag.agent.memory import ConversationMemory  # noqa: E402

from product import ui  # noqa: E402

# Answer-quality settings and the agent itself come from the module shared with
# the console demo: the two surfaces are the same product and must not drift
# apart.
from product.config import (  # noqa: E402
    CITATION_DOC_CHARS,
    CITATION_STYLE,
    DEBUG,
    EXAMPLE_QUESTIONS,
    LOG_DIR,
    MEMORY,
    PRODUCT_ICON,
    PRODUCT_NAME,
    PRODUCT_TAGLINE,
    PRODUCT_TAGLINE_EN,
    SHOW_FULL_ANSWER,
    STRATEGY,
    UI_LANGUAGE,
    build_demo_agent,
    corpus_manifest,
    probe_vllm_endpoints,
)

logger = logging.getLogger("expert_demo")

# The legacy graph-verification block, appended by the engine only when
# `cite_evidence` is off. It carries internal element ids, so it is cut from
# what the page shows; the evidence panel is built from `result` instead.
LEGACY_VERIFICATION_MARKER = "\nVerifica nel grafo:"

# Sentinel closing the token queue. A plain None already means "discard what you
# have", which a retry sends, so the end of the stream needs its own value.
_STREAM_DONE = object()

try:  # pragma: no cover - depends on the installed driver
    from neo4j.exceptions import ServiceUnavailable, SessionExpired

    _GRAPH_OUTAGE_EXCEPTIONS: tuple[type[BaseException], ...] = (
        ServiceUnavailable,
        SessionExpired,
    )
except Exception:  # pragma: no cover - driver without these names
    _GRAPH_OUTAGE_EXCEPTIONS = ()

# Rebuilding the agent re-indexes the text corpus, so a burst of questions
# arriving during an outage must not each pay for it. One rebuild per minute is
# enough: after the first, the others find the rebuilt agent in the cache.
_FAILOVER_COOLDOWN_SEC = 60.0
_last_failover_at = 0.0

# Shown on an answer built without the cross-lingual channel, because the
# encoder was unreachable. The product degrades instead of failing (see
# product/config.py); this is the half that keeps the degradation honest.
DEGRADED_NOTICE = (
    "\n\n---\n*Nota: il canale di ricerca cross-lingua non era disponibile per "
    "questa domanda. La risposta usa solo la ricerca testuale e per parole "
    "chiave, quindi può essere meno completa — soprattutto se la domanda è in "
    "una lingua diversa da quella dei documenti.*"
)


def _rewrite_notice(question: str, retrieval_question: str) -> str:
    """Line telling the reader how memory rewrote the question for retrieval.

    Without it, an answer that went somewhere the reader did not ask about
    gives no way to see why. Shown only when the rewrite actually changed the
    question: on most turns it is the question as typed, and saying so every
    time would train the reader to skip the line that matters.

    Args:
        question: The question as typed.
        retrieval_question: The question sent to retrieval.

    Returns:
        The Markdown notice, or ``""`` when nothing was rewritten.
    """
    typed = " ".join(str(question or "").split())
    used = " ".join(str(retrieval_question or "").split())
    if not used or used.casefold() == typed.casefold():
        return ""
    return f"\n\n*Cercato nei documenti come: «{used}»*"


@st.cache_resource(show_spinner=False)
def _available_models() -> dict[str, tuple[str, str]]:
    """Probe the configured vLLM endpoints once per process."""
    return probe_vllm_endpoints()


@st.cache_data(show_spinner=False)
def _corpus() -> dict[str, Any]:
    """How much the collection holds, read once per process from the manifest."""
    return corpus_manifest()


def _configure_logging() -> None:
    """Send the engine's own warnings to a file that survives the session.

    WARNING, not ERROR: the warnings are what report silent degradation —
    "embedding endpoint unavailable, vector channel skipped", "fulltext index
    disabled, falling back", "answer language mismatch", "discarding an
    implausible rewrite" — and without them a session that answered badly is
    indistinguishable from a healthy one after the fact.

    The file is separate from streamlit.log because that one is mostly Neo4j
    deprecation notices, and a warning worth reading gets lost in it.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Both loggers, not just the engine: this module logs as "expert_demo",
    # including the traceback of a failed question, which is the line someone
    # reads when a session is reported to have answered badly.
    targets = [logging.getLogger("graphrag"), logging.getLogger("expert_demo")]
    for target in targets:
        target.setLevel(logging.WARNING)

    # Streamlit re-runs this module on every interaction, so attaching the
    # handler unguarded would multiply it by the number of clicks.
    if any(getattr(h, "_demo_handler", False) for t in targets for h in t.handlers):
        return
    # Beside streamlit.log, not beside the session transcripts: this is
    # operational output, while LOG_DIR holds what people asked and were told
    # and is meant to become access-restricted.
    run_log_dir = Path(os.environ.get("DEMO_LOG_DIR_RUNTIME", ROOT / "artifacts" / "demo_logs"))
    run_log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_log_dir / "graphrag.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    handler._demo_handler = True  # type: ignore[attr-defined]
    for target in targets:
        target.addHandler(handler)


@st.cache_resource(show_spinner="Avvio in corso (connessione al grafo e indice testi)...")
def _load_agent(base_url: str, model_id: str) -> tuple[KGRAGAgent, str, str]:
    """Build the agent once per process and model.

    Returns:
        The agent, the model id and the label of the graph it connected to.
    """
    _configure_logging()
    os.chdir(ROOT)
    agent, graph_label = build_demo_agent(base_url, model_id)
    return agent, model_id, graph_label


def _session_log_path() -> Path:
    """The JSONL log of this browser session, created on first use."""
    if "session_log" not in st.session_state:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.session_log = LOG_DIR / f"session_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    return st.session_state.session_log


def _record_feedback(
    turn_id: str, chat_id: str, verdict: str = "", note: str = "", reason: str = ""
) -> None:
    """Append one rating, or one note, to the log the turn was written to.

    The demo is how readers judge the answers, so it records their ratings.
    Ratings live in the same file as the turns they rate, so the two are read
    together and nothing new has to be kept in sync.

    A vote and a note are separate lines and a note carries no `feedback` key,
    because the log is append-only and a note that repeated the vote would be
    counted as a second vote. Reading it: lines with `feedback` are votes, last
    one per `turn_id` wins; lines with `note` are comments on that turn.

    `reason` is one of a fixed set offered next to a negative vote. Free text
    says what went wrong for one reader; a fixed reason is the only part that
    can be counted across readers.

    Args:
        turn_id: The rated turn.
        chat_id: The conversation it belongs to.
        verdict: ``"up"`` or ``"down"`` for a vote.
        note: Free-text comment.
        reason: Fixed reason for a negative vote.
    """
    row: dict[str, object] = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        # Which surface wrote the line and what kind of line it is, so readers
        # of the log do not have to guess from the keys present. See
        # docs/configuration.md.
        "surface": "streamlit",
        "kind": "feedback",
        "chat_id": chat_id,
        "turn_id": turn_id,
    }
    if note or reason:
        if note:
            row["note"] = note
        if reason:
            row["reason"] = reason
    else:
        row["feedback"] = verdict
    with _session_log_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _new_chat() -> str:
    """Start an empty conversation and make it current.

    Each chat owns its transcript and its own memory: two threads of questions
    must not resolve each other's follow-ups. Streamlit keeps this per browser
    session, so a second tab is a second independent set of chats — and the
    only way to have two questions running at the same time, since one session
    processes one question at a time.

    Returns:
        The new chat id.
    """
    chat_id = uuid.uuid4().hex[:8]
    st.session_state.chats[chat_id] = {
        "title": "",
        "messages": [],
        "memory": ConversationMemory() if MEMORY else None,
    }
    st.session_state.chat_order.append(chat_id)
    st.session_state.current_chat = chat_id
    return chat_id


def _init_state() -> None:
    """Create the session-state keys the page relies on, once per session."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.chat_order = []
        _new_chat()
    # Ratings already given in this browser session, so the buttons can show
    # what was recorded. The record itself is the JSONL line, not this: a
    # reload loses the highlight, never the feedback.
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = UI_LANGUAGE if UI_LANGUAGE in ui.LANGUAGES else "it"
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = ""


def _current_chat() -> dict:
    """The conversation currently shown."""
    return st.session_state.chats[st.session_state.current_chat]


def _delete_chat(chat_id: str) -> None:
    """Delete a conversation, keeping at least one and a valid current one."""
    st.session_state.chats.pop(chat_id, None)
    st.session_state.chat_order.remove(chat_id)
    if not st.session_state.chat_order:
        _new_chat()
    elif st.session_state.current_chat == chat_id:
        st.session_state.current_chat = st.session_state.chat_order[-1]


def _chat_label(text: str, max_chars: int = 30) -> str:
    """Name a conversation after its first question."""
    label = " ".join(str(text or "").split())
    if len(label) <= max_chars:
        return label
    return label[:max_chars].rsplit(" ", 1)[0] + "…"


def _is_graph_outage(exc: BaseException) -> bool:
    """True when this failure is the graph becoming unreachable, not a bug.

    The whole chain is inspected because the driver's exception reaches here
    wrapped by whatever re-raised it, and a failover decision taken on the
    outermost type alone would miss the case it exists for.
    """
    if not _GRAPH_OUTAGE_EXCEPTIONS:
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _GRAPH_OUTAGE_EXCEPTIONS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _rebuild_agent(base_url: str, model_id: str) -> tuple[KGRAGAgent, str, str] | None:
    """Drop the cached agent and build a new one, or None if that failed too.

    `build_kg_manager` already falls back from the suspended Aura instance to
    the local mirror — but it only runs while the agent is being built, and the
    agent is a cached resource built once per process. Without a rebuild, an
    instance that suspends *after* the demo started would fail every later
    question, with a working mirror one rebuild away. Rebuilds are rate-limited
    by ``_FAILOVER_COOLDOWN_SEC``.

    Args:
        base_url: vLLM endpoint of the current model.
        model_id: Served model name.

    Returns:
        ``(agent, model_id, graph_label)`` from :func:`_load_agent`, or
        ``None`` when the rebuild failed.
    """
    global _last_failover_at
    now = time.monotonic()
    if now - _last_failover_at < _FAILOVER_COOLDOWN_SEC:
        # Another question already rebuilt; use what it left in the cache.
        try:
            return _load_agent(base_url, model_id)
        except Exception:  # noqa: BLE001 - the caller reports the original failure
            return None
    _last_failover_at = now
    logger.warning("Graph unreachable mid-session: rebuilding the agent once.")
    try:
        _load_agent.clear()
        return _load_agent(base_url, model_id)
    except Exception as exc:  # noqa: BLE001 - no graph at all is the caller's problem
        logger.error("Rebuild after the graph outage failed too: %s", exc)
        return None


def _vector_skips(agent: KGRAGAgent) -> int:
    """How many times the cross-lingual channel has been skipped so far.

    Two independent causes, and the reader of an answer cannot tell them
    apart nor should have to: the encoder is unreachable (counted by the
    retriever) or the vector index cannot be queried (counted by the graph
    manager). Either way the answer in front of them was built without that
    channel, so both feed the same notice.
    """
    retriever = getattr(agent, "kg_retriever", None)
    store = getattr(retriever, "kg_store", None)
    return int(getattr(retriever, "vector_skips", 0) or 0) + int(
        getattr(store, "vector_skips", 0) or 0
    )


def _stream(run: Any, placeholder: Any) -> dict[str, Any]:
    """Run one question in a thread and paint the answer as it is written.

    Generation is nearly the whole wait, so streaming lets the reader watch the
    answer fill instead of a spinner. The agent runs off the script thread
    because Streamlit only repaints from this one; the worker touches nothing
    but the queue.

    What arrives is the model's own text, minus the reference tags it writes as
    it goes: the citation gate turns those into document labels afterwards, and
    the caller replaces this draft with the verified answer.

    Args:
        run: Callable taking the token callback and returning the agent result.
        placeholder: Streamlit element the draft is painted into.

    Returns:
        The agent result, or ``{}`` when it returned nothing.

    Raises:
        BaseException: Whatever ``run`` raised, re-raised on this thread.
    """
    lang = _lang()
    sink: queue.Queue = queue.Queue()
    box: dict[str, Any] = {}

    def worker() -> None:
        """Run the question, then close the queue with the end sentinel."""
        try:
            box["result"] = run(sink.put)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the script thread
            box["error"] = exc
        finally:
            sink.put(_STREAM_DONE)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    scrubber = ui.StreamScrubber()
    shown = ""
    painted = 0.0
    placeholder.caption(ui.t(lang, "thinking"))
    while True:
        piece = sink.get()
        if piece is _STREAM_DONE:
            break
        if piece is None:
            # A retry threw away what it had already written.
            scrubber.reset()
            shown = ""
        else:
            shown += scrubber.feed(str(piece))
        now = time.monotonic()
        # Ten repaints a second is past what a reader can follow, and every one
        # of them is a message over the socket.
        if shown and now - painted > 0.1:
            painted = now
            placeholder.markdown(shown + " ▌")
    thread.join()

    if "error" in box:
        raise box["error"]
    return box.get("result") or {}


def _ask(
    agent: KGRAGAgent,
    model_id: str,
    question: str,
    turn_id: str = "",
    turn_index: int = 0,
    base_url: str = "",
    memory: ConversationMemory | None = None,
    chat_id: str = "",
    graph_label: str = "",
    placeholder: Any = None,
) -> dict[str, Any]:
    """Answer one question and return everything the page needs to render it.

    The returned dict is assembled from the agent's `result`, not from its
    prose: the counts, the citation check and the evidence all come from state
    keys. The only thing read out of the answer text is where its own sections
    start, so the engine's closing source list is not printed twice. The turn
    is also appended to the session log, and a failed question is reported in
    ``error`` rather than raised.

    Args:
        agent: The cached agent.
        model_id: Served model name, for the log.
        question: The question.
        turn_id: Id feedback lines point at.
        turn_index: Position of the turn in its conversation.
        base_url: vLLM endpoint, needed to rebuild the agent on a graph
            outage; empty disables the failover.
        memory: The conversation's memory, if any.
        chat_id: The conversation id, for the log.
        graph_label: Which graph answered, for the log.
        placeholder: Streamlit element to stream the draft into, if any.

    Returns:
        The render payload: ``body``, ``limits``, evidence, citation report,
        counts, latency, gate flags and ``error`` (``"service"`` or
        ``"question"`` on failure).
    """
    started = time.perf_counter()
    record: dict[str, object] = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        # See the feedback row above: a line has to say where it came from.
        "surface": "streamlit",
        "kind": "turn",
        "question": question,
        "strategy": STRATEGY,
        "model_id": model_id,
        # Which graph answered. Without it a session served by the local mirror
        # during an Aura outage reads exactly like a healthy one.
        "graph_label": graph_label,
        # One log per browser session, several conversations inside it: the id
        # is what separates two parallel threads of questions after the fact.
        "chat_id": chat_id,
        # What a feedback line points at. Without it a rating could only be
        # matched to a turn by timestamp and question text, which stops working
        # the moment the same question is asked twice.
        "turn_id": turn_id,
        # Where this turn sits in its conversation. `turn_id` says which turn,
        # `chat_id` says which conversation, and neither says whether an answer
        # was the opening question or the fifth follow-up — which is the first
        # thing to look at when reading back a session that went wrong.
        "turn_index": turn_index,
    }
    payload: dict[str, Any] = {
        "question": question,
        "turn_id": turn_id,
        "body": "",
        "limits": "",
        "evidence_index": [],
        "cited_refs": [],
        "citation_report": {},
        "counts": {"passages": 0, "facts": 0, "documents": 0},
        "latency_s": 0.0,
        "out_of_scope": False,
        "meta_question": False,
        "vector_degraded": False,
        "error": "",
    }
    skips_before = _vector_skips(agent)

    def _run(on_token: Any = None) -> dict[str, Any]:
        """One question against the graph, with the mid-session failover."""
        nonlocal agent, skips_before
        try:
            return agent.invoke(question, memory=memory, on_token=on_token)
        except Exception as exc:  # noqa: BLE001 - only a graph outage is handled here
            if not (base_url and _is_graph_outage(exc)):
                raise
            rebuilt = _rebuild_agent(base_url, model_id)
            if rebuilt is None:
                raise
            # The caption above still names the old graph; the st.rerun() at the
            # end of the question redraws it from the rebuilt agent.
            agent, _, record["graph_label"] = rebuilt
            record["graph_failover"] = True
            # Counters belong to the agent, and this one is new.
            skips_before = _vector_skips(agent)
            # memory.observe() runs after a successful graph.invoke, so the
            # failed attempt left no turn behind and this is not a double count.
            return agent.invoke(question, memory=memory, on_token=on_token)

    try:
        result = _stream(_run, placeholder) if placeholder is not None else _run()
        answer = str(result.get("answer", "")).strip()
        elapsed = time.perf_counter() - started
        record["answer"] = answer
        record["latency_s"] = round(elapsed, 2)
        # Where the seconds went: `latency_s` alone says how long a turn took,
        # not which stage spent it.
        stage_timings = result.get("stage_timings_ms")
        if stage_timings:
            record["stage_timings_ms"] = stage_timings
        # What the answer was actually built from. A thin answer has two very
        # different causes — the gate refused, or retrieval came back empty —
        # and without these counts the log cannot tell them apart.
        record["out_of_scope"] = bool(result.get("out_of_scope"))
        record["meta_question"] = bool(result.get("meta_question"))
        record["insufficient"] = bool(result.get("insufficient_answer"))
        record["n_triples"] = len(result.get("kg_triples") or [])
        record["n_nodes"] = len(result.get("retrieved_nodes") or [])
        record["n_text_sources"] = len(result.get("retrieved_text_sources") or [])
        # The question actually sent to retrieval, and what resolved it.
        # Logged separately from `question` so a rewrite that hurt the answer
        # can be recognised as such after the session.
        if memory is not None:
            record["follow_up"] = bool(result.get("follow_up"))
            record["retrieval_question"] = result.get("retrieval_question", question)
            record["memory_entities"] = result.get("memory_entities", [])
        # Phantom-reference rate per model: how often the model cites evidence
        # it was never shown, comparable across models.
        citation_report = result.get("citation_report")
        if isinstance(citation_report, dict):
            record["citation_report"] = citation_report
            payload["citation_report"] = citation_report
            payload["cited_refs"] = list(citation_report.get("cited_refs") or [])

        # The legacy triple dump never reaches the page: it carries element ids,
        # and the evidence panel shows the same facts as sentences instead.
        body = answer.partition(LEGACY_VERIFICATION_MARKER)[0]
        parts = ui.split_answer(body)
        shown = parts.body
        # Both notices qualify the answer above them, so they are appended to
        # the prose after the engine's own source list has been split off.
        shown += _rewrite_notice(question, str(result.get("retrieval_question") or ""))
        # A degraded answer has to say so. Per-question, not per-session: the
        # encoder can come back, and an answer given while it was down is worth
        # less than the one before it and the one after it.
        record["vector_degraded"] = _vector_skips(agent) > skips_before
        if record["vector_degraded"]:
            shown += DEGRADED_NOTICE

        payload.update(
            {
                "body": shown,
                "limits": parts.limits,
                "evidence_index": list(result.get("evidence_index") or []),
                "counts": ui.retrieval_counts(result),
                "latency_s": round(elapsed, 1),
                "out_of_scope": bool(result.get("out_of_scope")),
                "meta_question": bool(result.get("meta_question")),
                "vector_degraded": bool(record["vector_degraded"]),
            }
        )
    except Exception as exc:  # noqa: BLE001 - UI must survive any failure
        elapsed = time.perf_counter() - started
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_s"] = round(elapsed, 2)
        logger.error("Question failed: %s\n%s", exc, traceback.format_exc())
        # Two failures that look identical to a reader and are not. An
        # unreachable graph is nothing the person typing can fix, and telling
        # them to rephrase sends them chasing their own question.
        payload["error"] = "service" if _is_graph_outage(exc) else "question"
        payload["latency_s"] = round(elapsed, 1)
    with _session_log_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return payload


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _lang() -> str:
    """The interface language of this session."""
    return st.session_state.get("ui_lang", "it")


def _render_metadata(turn: dict[str, Any]) -> None:
    """What the answer was built from, and how long it took.

    The elapsed time sits beside the three counts, where it reads as what the
    wait bought rather than as an apology for it.
    """
    lang = _lang()
    counts = turn.get("counts") or {}
    bits = [
        ui.count_label(lang, "meta_passages", int(counts.get("passages", 0))),
        ui.count_label(lang, "meta_facts", int(counts.get("facts", 0))),
        ui.count_label(lang, "meta_documents", int(counts.get("documents", 0))),
        ui.t(lang, "meta_seconds", n=f"{float(turn.get('latency_s', 0.0)):.0f}"),
    ]
    state, text = ui.citation_summary(turn.get("citation_report"), lang)
    colour = {"clean": "green", "phantom": "orange", "none": "gray"}[state]
    st.caption(f"{' · '.join(bits)}  ·  :{colour}[{text}]", help=ui.t(lang, "cit_help"))


def _render_sources(turn: dict[str, Any]) -> None:
    """Where this answer came from, in one line.

    What belongs under an answer is the short statement of its provenance; the
    evidence itself, every passage in full, has its own panel.
    """
    line = ui.compact_sources_line(
        turn.get("evidence_index") or [],
        turn.get("cited_refs") or [],
        _lang(),
    )
    if line:
        st.caption(line)


def _render_evidence(turn: dict[str, Any], container: Any) -> None:
    """One answer's evidence, in a box per kind.

    Two kinds of thing, two boxes: a passage is something to read, a graph fact
    is something to check, and one fold for both would make the panel a single
    undifferentiated block. Closed, this is two lines; each opens on its own.

    Args:
        turn: The rendered turn payload.
        container: Streamlit container to draw into.
    """
    lang = _lang()
    panel = ui.panel_evidence(turn.get("evidence_index") or [], turn.get("cited_refs") or [])
    if not (panel.passages or panel.facts):
        container.caption(ui.t(lang, "evidence_none"))
        return

    if panel.passages:
        label = f"{ui.t(lang, 'passages')} · {len(panel.passages)}"
        with container.expander(label, expanded=False):
            for index, passage in enumerate(panel.passages):
                if index:
                    st.divider()
                st.markdown(f"**{ui.passage_label(passage)}**")
                if passage["cited"]:
                    st.write(passage["text"])
                else:
                    # Its identity without its body: the answer did not lean on
                    # it, and eight unused passages in full is a wall.
                    st.caption(ui.t(lang, "also_retrieved"))

    if panel.facts:
        label = f"{ui.t(lang, 'graph_facts')} · {len(panel.facts)}"
        with container.expander(label, expanded=False):
            spare_started = False
            for fact in panel.facts:
                if not fact["cited"] and not spare_started:
                    # One rule between what the answer used and what it did not,
                    # instead of a marker repeated on every line below it.
                    spare_started = True
                    st.divider()
                    st.caption(ui.t(lang, "also_retrieved"))
                line = ui.fact_line(fact)
                st.markdown(line if fact["cited"] else f":gray[{line}]")


def _render_out_of_scope(turn_id: str, meta: bool = False) -> None:
    """Turn a refusal into a direction.

    The gate answers with a fixed sentence and nothing else, which ends the
    session for anyone who does not already know what the collection holds. The
    count comes from the corpus manifest, so it stays true as the corpus grows.

    Args:
        turn_id: The turn these buttons belong to; it keys their widget ids.
        meta: The question was about the assistant, not out of scope. Same
            panel, different title: the answer above it is an introduction.
    """
    lang = _lang()
    count = int(_corpus().get("count", 0) or 0)
    with st.container(border=True):
        st.markdown(f"**{ui.t(lang, 'meta_title' if meta else 'oos_title')}**")
        if count:
            st.write(ui.t(lang, "oos_covers", n=count))
        if EXAMPLE_QUESTIONS:
            st.caption(ui.t(lang, "oos_try"))
            for index, example in enumerate(EXAMPLE_QUESTIONS):
                if st.button(example, key=f"oos_{turn_id}_{index}"):
                    st.session_state.pending_question = example
                    st.rerun()


def _feedback_row(turn: dict[str, Any], chat_id: str) -> None:
    """Two buttons, and the reason asked for straight away on a negative one.

    Rendered from the history loop rather than beside the fresh answer, so a
    reader who changes their mind three questions later can still say so. The
    reason and note appear right after a negative vote, since they are the
    only part of the feedback that says what went wrong.

    Args:
        turn: The rendered turn payload.
        chat_id: The conversation it belongs to.
    """
    lang = _lang()
    turn_id = str(turn.get("turn_id") or "")
    if not turn_id:
        return
    given = st.session_state.feedback.get(turn_id, "")

    up, down, said = st.columns([1, 1, 10])
    if up.button("👍", key=f"up_{turn_id}", help=ui.t(lang, "fb_useful")):
        st.session_state.feedback[turn_id] = "up"
        _record_feedback(turn_id, chat_id, "up")
        st.rerun()
    if down.button("👎", key=f"down_{turn_id}", help=ui.t(lang, "fb_wrong")):
        st.session_state.feedback[turn_id] = "down"
        _record_feedback(turn_id, chat_id, "down")
        st.rerun()
    if given.startswith("up"):
        said.caption(ui.t(lang, "fb_thanks"))
    if not given.startswith("down"):
        return
    if given == "down+detail":
        said.caption(ui.t(lang, "fb_sent"))
        return

    # A fixed reason is the part that can be counted across readers; the free
    # note is the part that says what a count cannot.
    reasons = {
        "incomplete": ui.t(lang, "fb_reason_incomplete"),
        "offtarget": ui.t(lang, "fb_reason_offtarget"),
        "sources": ui.t(lang, "fb_reason_sources"),
    }
    with st.container(border=True):
        st.caption(ui.t(lang, "fb_why"))
        chosen = st.pills(
            ui.t(lang, "fb_why"),
            list(reasons),
            format_func=lambda key: reasons[key],
            key=f"reason_{turn_id}",
            label_visibility="collapsed",
        )
        note = st.text_input(
            ui.t(lang, "fb_note_placeholder"),
            key=f"note_{turn_id}",
            placeholder=ui.t(lang, "fb_note_placeholder"),
            label_visibility="collapsed",
        )
        if st.button(ui.t(lang, "fb_send"), key=f"send_{turn_id}"):
            _record_feedback(turn_id, chat_id, note=note.strip(), reason=chosen or "")
            st.session_state.feedback[turn_id] = "down+detail"
            st.rerun()


def _render_turn(turn: dict[str, Any], chat_id: str, *, with_evidence: bool) -> None:
    """One question and its answer, with everything the answer stands on.

    Args:
        turn: The rendered turn payload.
        chat_id: The conversation it belongs to.
        with_evidence: Draw the evidence under the answer; off for the latest
            answer, whose evidence is in the side panel.
    """
    lang = _lang()
    with st.chat_message("user"):
        st.markdown(turn.get("question", ""))
    with st.chat_message("assistant"):
        error = turn.get("error")
        if error:
            st.warning(ui.t(lang, "err_service" if error == "service" else "err_question"))
            return
        st.markdown(
            ui.style_citations(
                turn.get("body", ""),
                doc_chars=CITATION_DOC_CHARS,
                dim=CITATION_STYLE == "dim",
            )
            if CITATION_STYLE != "plain"
            else turn.get("body", "")
        )
        if turn.get("out_of_scope"):
            _render_out_of_scope(
                str(turn.get("turn_id") or ""), meta=bool(turn.get("meta_question"))
            )
            return
        if turn.get("limits"):
            # Folded, like the evidence: it qualifies the answer, it is not part
            # of reading it, and open on every turn it would double the block a
            # reader has to scroll past to reach the next question.
            with st.expander(ui.t(lang, "limits_title"), expanded=False):
                st.write(turn["limits"])
        _render_metadata(turn)
        _render_sources(turn)

        actions, _ = st.columns([3, 5])
        with actions.popover(ui.t(lang, "copy_with_sources")):
            st.caption(ui.t(lang, "copy_hint"))
            st.code(ui.answer_markdown(turn, lang), language="markdown", wrap_lines=True)

        # Only for the answers the reserved panel is not already showing, so
        # the same evidence is never on screen twice.
        if with_evidence and SHOW_FULL_ANSWER:
            st.caption(ui.t(lang, "evidence_expander"))
            _render_evidence(turn, st.container())
        _feedback_row(turn, chat_id)


def _render_status(graph_label: str, turns: list[dict[str, Any]]) -> None:
    """One line saying whether the system is whole, without naming the servers.

    The strategy, the model id and the graph label mean nothing to a reader,
    and the last names the hosted database, so they are shown only in debug
    mode.

    Args:
        graph_label: Which graph answered; a fallback means reduced mode.
        turns: The conversation; a degraded last answer means reduced mode.
    """
    lang = _lang()
    reduced = str(graph_label or "").startswith("fallback")
    degraded = bool(turns and turns[-1].get("vector_degraded"))
    if reduced or degraded:
        why = ui.t(lang, "status_reduced_why" if reduced else "status_degraded_why")
        st.caption(f":orange[● {ui.t(lang, 'status_reduced')}] — {why}")
    else:
        st.caption(f":green[● {ui.t(lang, 'status_ok')}]")


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title=PRODUCT_NAME,
    page_icon=PRODUCT_ICON,
    layout="wide",
)

_init_state()
LANG = _lang()

models = _available_models()
if not models:
    st.error("Nessun server vLLM raggiungibile (DEMO_VLLM_ENDPOINTS) e VLLM_MODEL_NAME/VLLM_BASE_URL mancanti.")
    st.stop()

env_base_url = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
labels = list(models)
default_index = next(
    (i for i, lbl in enumerate(labels) if models[lbl][0] == env_base_url), 0
)

with st.sidebar:
    st.markdown(f"### {PRODUCT_NAME}")

    st.markdown(f"**{ui.t(LANG, 'conversations')}**")
    if st.button(ui.t(LANG, "new_chat"), width="stretch"):
        _new_chat()
        st.rerun()
    for chat_id in list(st.session_state.chat_order):
        entry = st.session_state.chats[chat_id]
        is_current = chat_id == st.session_state.current_chat
        if st.button(
            entry["title"] or ui.t(LANG, "empty_chat"),
            key=f"select_{chat_id}",
            width="stretch",
            type="primary" if is_current else "secondary",
        ):
            st.session_state.current_chat = chat_id
            st.rerun()

    chat = _current_chat()

    # Destructive and irreversible, so it stays out of the way until there is
    # something to destroy, and it asks before doing it.
    if chat["messages"]:
        st.divider()
        if st.session_state.confirm_delete == st.session_state.current_chat:
            st.caption(ui.t(LANG, "delete_confirm"))
            yes, no = st.columns(2)
            if yes.button(ui.t(LANG, "delete_yes"), width="stretch", type="primary"):
                _delete_chat(st.session_state.current_chat)
                st.session_state.confirm_delete = ""
                st.rerun()
            if no.button(ui.t(LANG, "delete_no"), width="stretch"):
                st.session_state.confirm_delete = ""
                st.rerun()
        elif st.button(ui.t(LANG, "delete"), width="stretch"):
            st.session_state.confirm_delete = st.session_state.current_chat
            st.rerun()

        st.download_button(
            ui.t(LANG, "download_conversation"),
            data=ui.conversation_markdown(
                chat["title"] or ui.t(LANG, "empty_chat"),
                [m for m in chat["messages"] if not m.get("error")],
                LANG,
            ),
            file_name=f"{dt.datetime.now():%Y%m%d_%H%M}_conversazione.md",
            mime="text/markdown",
            width="stretch",
        )

    # Which entities the follow-up rewrite is carrying. Named as the thread it
    # is, not as the seed list it is inside the engine, and only when it holds
    # something: the seeds are empty more often than not, and "no active topic"
    # asks the reader to worry about a mechanism they were never shown.
    if MEMORY and chat["memory"] is not None:
        active = chat["memory"].seed_entities()
        if active:
            st.divider()
            st.caption(ui.t(LANG, "thread_following", topics=", ".join(active)))
            if st.button(ui.t(LANG, "thread_reset"), width="stretch"):
                chat["memory"].reset()
                st.rerun()

    st.divider()
    # A select, bound to the state key it sets, rather than a segmented control:
    # clicking the language already in use would deselect it and return None,
    # and a focused button group answers the Enter key that belongs to the
    # question box. A select does neither.
    st.selectbox(
        ui.t(LANG, "interface_language"),
        list(ui.LANGUAGES),
        format_func=lambda code: ui.LANGUAGES[code],
        key="ui_lang",
    )

    # One reachable model is not a choice, and a list of served model ids with
    # their ports is not a question a student or a public-sector reader can
    # answer. It stays for the people who run comparisons.
    choice = labels[default_index]
    if len(labels) > 1:
        with st.expander(ui.t(LANG, "advanced"), expanded=False):
            choice = st.selectbox(
                ui.t(LANG, "model"),
                labels,
                index=default_index,
                format_func=lambda label: ui.model_display_name(models[label][1]),
            )

base_url, model_id = models[choice]

try:
    agent, model_id, graph_label = _load_agent(base_url, model_id)
except RuntimeError as exc:
    # build_kg_manager raises this one, already phrased for a reader.
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001 - the browser must not receive a traceback
    # The same call builds the text pipeline — FAISS, a local e5 — which can
    # raise anything at all. Uncaught, a missing index directory or a
    # torch/sentence-transformers mismatch would reach the browser as a full
    # traceback: file paths, model names and the shape of the deployment,
    # shown to whoever opened the page.
    _configure_logging()
    logger.error("Startup failed: %s\n%s", exc, traceback.format_exc())
    st.error(
        "Avvio non riuscito: il servizio non è disponibile in questo momento. "
        "Il dettaglio tecnico è nel log del server."
    )
    st.stop()

# The question being answered on this run. It is put here by the box at the end
# of the script, or by one of the example questions offered on a refusal.
question = st.session_state.pop("pending_question", None)

reading, evidence_panel = st.columns([2, 1], gap="large")

with reading:
    with st.container(width=760):
        st.title(PRODUCT_NAME)
        st.caption(PRODUCT_TAGLINE_EN if LANG == "en" else PRODUCT_TAGLINE)
        _render_status(graph_label, chat["messages"])
        if DEBUG:
            st.caption(f"strategia: {STRATEGY} | modello: {model_id} | grafo: {graph_label}")

        last_index = len(chat["messages"]) - 1
        for index, turn in enumerate(chat["messages"]):
            _render_turn(turn, st.session_state.current_chat, with_evidence=index != last_index)

        if question:
            if not chat["messages"]:
                chat["title"] = _chat_label(question)
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                # The answer is painted here as it is written, then thrown away:
                # what stays is the verified text, with its citations turned
                # into document labels and its source list attached.
                streaming = st.empty()
                turn = _ask(
                    agent,
                    model_id,
                    question,
                    turn_id=uuid.uuid4().hex[:12],
                    turn_index=len(chat["messages"]),
                    base_url=base_url,
                    memory=chat["memory"],
                    chat_id=st.session_state.current_chat,
                    graph_label=graph_label,
                    placeholder=streaming,
                )
                streaming.empty()
            # Appended before rendering: a rerun raised inside the renderer (a
            # sidebar click during the spinner, a browser reconnect) would drop
            # the answer from the transcript while the JSONL row is already
            # written and memory.observe() has already run, leaving the three
            # disagreeing.
            chat["messages"].append(turn)
            # The sidebar and the evidence panel both rendered before the answer
            # existed: rerun so the title, the thread and the panel catch up.
            st.rerun()

with evidence_panel:
    answered = [m for m in chat["messages"] if not m.get("error") and not m.get("out_of_scope")]
    if answered and SHOW_FULL_ANSWER:
        st.caption(ui.t(LANG, "evidence_of_last"))
        _render_evidence(answered[-1], st.container())

# Last statement in the script, which is what keeps Streamlit pinning it to the
# bottom of the page and submitting it on Enter. The question is handed to the
# next run rather than answered here: the transcript above has already been
# drawn, and the example questions on a refusal reach the agent by the same
# route, so there is one way in and not two.
typed = st.chat_input(ui.t(LANG, "ask_placeholder"))
if typed:
    st.session_state.pending_question = typed
    st.rerun()
