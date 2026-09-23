"""The demo's own logic, underneath the widgets.

`product/app.py` is the interface readers actually use. Most of the file
paints; what is pinned here is the part that decides — which conversation a
question belongs to, what gets written to the session log, when a failure is
the graph rather than the question, and what the streaming worker does with
an exception raised off the script thread.

Streamlit is imported but never rendered: `st.session_state` accepts writes
outside a script run, so the state functions are exercised directly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pytest

# CI installs `.[dev]`, and streamlit lives in `.[demo]` — importing
# `product.app` without it fails at collection and takes the whole test job
# with it. The module under test is the browser demo, so skipping is the
# honest outcome where the browser demo is not installed.
pytest.importorskip("streamlit", reason="product/app.py needs the [demo] extra")

# `product/app.py` is a Streamlit script, and importing it runs its body —
# including `_configure_logging`, which calls `logging.basicConfig`, pins the
# "graphrag" and "expert_demo" loggers to WARNING and attaches a file handler,
# for the whole process. Right for a demo, wrong for a test session: every
# test after this one that reads a log would see nothing. The state is taken
# before the import and put back after it.
_ROOT = logging.getLogger()
_SAVED_HANDLERS = list(_ROOT.handlers)
_SAVED_ROOT_LEVEL = _ROOT.level
_SAVED_LEVELS = {
    name: logging.getLogger(name).level for name in ("graphrag", "expert_demo")
}

from product import app  # noqa: E402

_ROOT.handlers[:] = _SAVED_HANDLERS
_ROOT.setLevel(_SAVED_ROOT_LEVEL)
for _name, _level in _SAVED_LEVELS.items():
    logging.getLogger(_name).setLevel(_level)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    """Empty session state and a temporary log directory for each test."""
    app.st.session_state.clear()
    monkeypatch.setattr(app, "LOG_DIR", tmp_path)
    yield
    app.st.session_state.clear()


def _rows(path) -> list[dict[str, Any]]:
    """The JSON rows of a JSONL log."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- conversations ---------------------------------------------------------


def test_a_new_chat_becomes_the_current_one():
    app._init_state()

    chat_id = app._new_chat()

    assert app.st.session_state.current_chat == chat_id
    assert app._current_chat()["messages"] == []


def test_each_chat_keeps_its_own_transcript():
    # Two threads of questions must not resolve each other's follow-ups.
    app._init_state()
    first = app.st.session_state.current_chat
    app.st.session_state.chats[first]["messages"].append({"question": "prima"})

    second = app._new_chat()

    assert app.st.session_state.chats[second]["messages"] == []
    assert len(app.st.session_state.chats[first]["messages"]) == 1


def test_chats_keep_the_order_they_were_opened_in():
    app._init_state()
    first = app.st.session_state.current_chat
    second = app._new_chat()

    assert app.st.session_state.chat_order == [first, second]


def test_deleting_the_current_chat_falls_back_to_the_most_recent():
    app._init_state()
    first = app.st.session_state.current_chat
    second = app._new_chat()

    app._delete_chat(second)

    assert app.st.session_state.current_chat == first
    assert second not in app.st.session_state.chats


def test_deleting_another_chat_leaves_the_current_one_alone():
    app._init_state()
    first = app.st.session_state.current_chat
    second = app._new_chat()

    app._delete_chat(first)

    assert app.st.session_state.current_chat == second


def test_deleting_the_last_chat_opens_a_fresh_one():
    # The page has nothing to render without a current chat.
    app._init_state()
    only = app.st.session_state.current_chat

    app._delete_chat(only)

    assert app.st.session_state.chat_order
    assert app.st.session_state.current_chat != only


def test_initialising_twice_does_not_discard_the_conversation():
    app._init_state()
    chat_id = app.st.session_state.current_chat

    app._init_state()

    assert app.st.session_state.current_chat == chat_id


def test_initialising_sets_the_defaults_the_page_reads():
    app._init_state()

    assert app.st.session_state.feedback == {}
    assert app.st.session_state.ui_lang in app.ui.LANGUAGES
    assert app.st.session_state.confirm_delete == ""


# --- naming a conversation -------------------------------------------------


def test_a_short_question_names_the_chat_as_it_is():
    assert app._chat_label("Cos'e' la scotta?") == "Cos'e' la scotta?"


def test_a_long_question_is_cut_at_a_word_boundary():
    label = app._chat_label(
        "Quali sono le tre componenti principali del capitale secondo il framework"
    )

    assert label.endswith("…")
    assert len(label) <= 31
    assert not label.rstrip("…").endswith(" ")


def test_whitespace_is_collapsed_before_naming():
    assert app._chat_label("  due   parole  ") == "due parole"


@pytest.mark.parametrize("value", ["", None, "   "])
def test_a_chat_with_no_question_yet_has_an_empty_name(value):
    assert app._chat_label(value) == ""


# --- the session log -------------------------------------------------------


def test_one_log_file_per_browser_session():
    first = app._session_log_path()
    second = app._session_log_path()

    assert first == second
    assert first.name.startswith("session_")


def test_a_vote_is_one_line_naming_the_turn_it_rates():
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="up")

    row = _rows(app._session_log_path())[0]
    assert row["kind"] == "feedback"
    assert row["surface"] == "streamlit"
    assert (row["turn_id"], row["chat_id"], row["feedback"]) == ("t1", "c1", "up")


def test_a_note_is_a_separate_line_and_is_not_counted_as_a_vote():
    # The log is append-only: a note repeating the vote would be read as a
    # second vote.
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="down")
    app._record_feedback(turn_id="t1", chat_id="c1", note="la risposta è generica")

    rows = _rows(app._session_log_path())
    assert len(rows) == 2
    assert "feedback" not in rows[1]
    assert rows[1]["note"] == "la risposta è generica"


def test_a_fixed_reason_is_recorded_alongside_the_free_text():
    # Free text says what went wrong for one reader; the fixed reason is the
    # only part that can be counted across readers.
    app._record_feedback(turn_id="t1", chat_id="c1", reason="generic", note="troppo vaga")

    row = _rows(app._session_log_path())[0]
    assert row["reason"] == "generic"
    assert row["note"] == "troppo vaga"
    assert "feedback" not in row


def test_a_reason_without_a_note_is_still_not_a_vote():
    app._record_feedback(turn_id="t1", chat_id="c1", reason="wrong")

    row = _rows(app._session_log_path())[0]
    assert row["reason"] == "wrong"
    assert "feedback" not in row


def test_the_last_vote_on_a_turn_is_the_one_that_counts():
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="up")
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="down")

    votes = [r for r in _rows(app._session_log_path()) if "feedback" in r]
    assert [v["feedback"] for v in votes] == ["up", "down"]


def test_every_line_says_which_surface_wrote_it():
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="up")

    assert _rows(app._session_log_path())[0]["surface"] == "streamlit"


# --- telling a broken graph from a broken question -------------------------


def test_an_ordinary_failure_is_not_a_graph_outage():
    # An unreachable graph is nothing the person typing can fix; telling them
    # to rephrase sends them chasing their own question.
    assert app._is_graph_outage(ValueError("bad question")) is False


def test_a_driver_outage_wrapped_by_another_exception_is_still_found():
    if not app._GRAPH_OUTAGE_EXCEPTIONS:
        pytest.skip("neo4j exception classes unavailable")
    inner = app._GRAPH_OUTAGE_EXCEPTIONS[0]("unreachable")

    try:
        try:
            raise inner
        except Exception as exc:
            raise RuntimeError("answering failed") from exc
    except RuntimeError as wrapped:
        assert app._is_graph_outage(wrapped) is True


def test_a_cycle_in_the_exception_chain_does_not_hang():
    first = RuntimeError("a")
    second = RuntimeError("b")
    first.__cause__ = second
    second.__cause__ = first

    assert app._is_graph_outage(first) is False


# --- the degraded-channel counter ------------------------------------------


def test_skips_from_both_causes_reach_the_same_notice():
    # The encoder being unreachable and the vector index being unqueryable are
    # different causes; the reader cannot tell them apart and should not have to.
    class _Store:
        vector_skips = 2

    class _Retriever:
        vector_skips = 3
        kg_store = _Store()

    class _Agent:
        kg_retriever = _Retriever()

    assert app._vector_skips(_Agent()) == 5


def test_an_agent_that_counts_nothing_reports_zero():
    class _Agent:
        pass

    assert app._vector_skips(_Agent()) == 0


# --- streaming off the script thread ---------------------------------------


def _placeholder():
    """A stand-in for `st.empty()` that records what is painted."""
    class _P:
        def __init__(self) -> None:
            self.painted: list[str] = []
            self.captions: list[str] = []

        def caption(self, text: str) -> None:
            self.captions.append(text)

        def markdown(self, text: str) -> None:
            self.painted.append(text)

    return _P()


def test_the_answer_is_returned_once_the_worker_finishes():
    app._init_state()
    placeholder = _placeholder()

    def _run(sink):
        sink("La scotta ")
        sink("e' un residuo.")
        return {"answer": "La scotta e' un residuo."}

    assert app._stream(_run, placeholder) == {"answer": "La scotta e' un residuo."}


def test_a_failure_on_the_worker_thread_is_raised_on_the_script_thread():
    # Streamlit only repaints from the script thread, so an exception swallowed
    # in the worker would leave the page spinning with no error.
    app._init_state()

    def _run(sink):
        raise RuntimeError("vLLM down")

    with pytest.raises(RuntimeError, match="vLLM down"):
        app._stream(_run, _placeholder())


def test_a_retry_that_throws_away_its_draft_repaints_from_empty():
    app._init_state()
    placeholder = _placeholder()

    def _run(sink):
        sink("prima meta")
        time.sleep(0.15)  # past the repaint throttle, so the draft is painted
        sink(None)  # the retry discards what it had written
        sink("la risposta buona")
        time.sleep(0.15)
        return {"answer": "la risposta buona"}

    app._stream(_run, placeholder)

    assert any("prima meta" in frame for frame in placeholder.painted)
    assert "prima meta" not in placeholder.painted[-1]
    assert "la risposta buona" in placeholder.painted[-1]


def test_the_reader_is_told_something_is_happening_before_any_text():
    app._init_state()
    placeholder = _placeholder()

    app._stream(lambda sink: {"answer": ""}, placeholder)

    assert placeholder.captions


def test_a_run_that_returns_nothing_yields_an_empty_result():
    app._init_state()

    assert app._stream(lambda sink: None, _placeholder()) == {}


# --- answering one question ------------------------------------------------


class _Agent:
    """Answers, or fails, and counts the skips the notice is built from."""

    def __init__(self, results: list[Any] | None = None, skips: list[int] | None = None):
        self.results = list(results or [])
        self.skips = list(skips or [])
        self.asked: list[str] = []

        class _Store:
            vector_skips = 0

        class _Retriever:
            vector_skips = 0
            kg_store = _Store()

        self.kg_retriever = _Retriever()

    def invoke(self, question: str, memory=None, on_token=None) -> dict[str, Any]:
        self.asked.append(question)
        if self.skips:
            self.kg_retriever.vector_skips = self.skips.pop(0)
        outcome = self.results.pop(0) if self.results else {"answer": "una risposta"}
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _ask(agent, **kwargs: Any) -> dict[str, Any]:
    """`app._ask` on `agent`, with defaults for every other argument."""
    defaults: dict[str, Any] = {
        "model_id": "qwen",
        "question": "cos'e' la scotta?",
        "turn_id": "t1",
        "turn_index": 1,
        "chat_id": "c1",
        "graph_label": "aura",
    }
    defaults.update(kwargs)
    return app._ask(agent, **defaults)


def _turn_rows() -> list[dict[str, Any]]:
    """The turn rows of this session's log."""
    return [r for r in _rows(app._session_log_path()) if r.get("kind") == "turn"]


def test_a_turn_is_logged_with_the_identity_a_rating_points_at():
    _ask(_Agent([{"answer": "La scotta e' un residuo."}]))

    row = _turn_rows()[0]
    assert (row["turn_id"], row["chat_id"], row["turn_index"]) == ("t1", "c1", 1)
    assert row["graph_label"] == "aura"
    assert row["answer"] == "La scotta e' un residuo."


def test_the_counts_say_what_the_answer_was_built_from():
    # A thin answer has two very different causes — the gate refused, or
    # retrieval came back empty — and without these the log cannot tell them
    # apart.
    _ask(
        _Agent([{
            "answer": "x",
            "kg_triples": [{"subject": "a"}, {"subject": "b"}],
            "retrieved_nodes": [{"text": "n"}],
            "retrieved_text_sources": [{"source": "s"}],
        }])
    )

    row = _turn_rows()[0]
    assert (row["n_triples"], row["n_nodes"], row["n_text_sources"]) == (2, 1, 1)


def test_a_refused_question_is_marked_as_such():
    payload = _ask(_Agent([{"answer": "fuori dominio", "out_of_scope": True}]))

    assert payload["out_of_scope"] is True
    assert _turn_rows()[0]["out_of_scope"] is True


def test_the_question_sent_to_retrieval_is_logged_next_to_the_one_typed():
    # Logged separately so a rewrite that hurt the answer can be recognised as
    # such after the session.
    memory = object()
    _ask(
        _Agent([{
            "answer": "x",
            "retrieval_question": "cos'e' la scotta di caseificio?",
            "follow_up": True,
            "memory_entities": ["Scotta"],
        }]),
        memory=memory,
    )

    row = _turn_rows()[0]
    assert row["question"] == "cos'e' la scotta?"
    assert row["retrieval_question"] == "cos'e' la scotta di caseificio?"
    assert row["follow_up"] is True


def test_without_memory_no_follow_up_fields_are_written():
    _ask(_Agent([{"answer": "x", "follow_up": True}]), memory=None)

    assert "follow_up" not in _turn_rows()[0]


def test_the_citation_report_reaches_both_the_log_and_the_page():
    report = {"cited_refs": ["S1"], "phantom": 0}

    payload = _ask(_Agent([{"answer": "x [S1]", "citation_report": report}]))

    assert payload["citation_report"] == report
    assert payload["cited_refs"] == ["S1"]
    assert _turn_rows()[0]["citation_report"] == report


def test_the_stage_split_is_logged_when_the_agent_reports_one():
    _ask(_Agent([{"answer": "x", "stage_timings_ms": {"generate": 30000.0}}]))

    assert _turn_rows()[0]["stage_timings_ms"] == {"generate": 30000.0}


def test_the_debugging_dump_never_reaches_the_page():
    # It carries element ids; the evidence panel shows the same facts as
    # sentences instead.
    answer = f"La risposta.{app.LEGACY_VERIFICATION_MARKER}4:abc:1"

    payload = _ask(_Agent([{"answer": answer}]))

    assert "4:abc:1" not in payload["body"]
    assert "4:abc:1" in _turn_rows()[0]["answer"]


# --- an answer given while a channel was down ------------------------------


def test_an_answer_built_without_the_vector_channel_says_so():
    # Per question, not per session: the encoder can come back, and an answer
    # given while it was down is worth less than the one before it.
    payload = _ask(_Agent([{"answer": "x"}], skips=[1]))

    assert payload["vector_degraded"] is True
    assert app.DEGRADED_NOTICE.strip() in payload["body"]


def test_an_answer_given_while_everything_worked_says_nothing():
    payload = _ask(_Agent([{"answer": "x"}]))

    assert payload["vector_degraded"] is False
    assert app.DEGRADED_NOTICE.strip() not in payload["body"]


def test_a_skip_from_an_earlier_turn_does_not_mark_this_one():
    agent = _Agent([{"answer": "prima"}, {"answer": "seconda"}], skips=[1, 1])

    _ask(agent)
    second = _ask(agent, turn_id="t2", turn_index=2)

    assert second["vector_degraded"] is False


# --- the mid-session failover ----------------------------------------------


def test_a_graph_outage_is_retried_on_the_rebuilt_agent(monkeypatch):
    if not app._GRAPH_OUTAGE_EXCEPTIONS:
        pytest.skip("neo4j exception classes unavailable")
    outage = app._GRAPH_OUTAGE_EXCEPTIONS[0]("unreachable")
    fallback = _Agent([{"answer": "dalla copia locale"}])
    monkeypatch.setattr(
        app, "_rebuild_agent", lambda base_url, model_id: (fallback, "", "local mirror")
    )

    payload = _ask(_Agent([outage]), base_url="http://localhost:8000/v1")

    assert payload["body"].startswith("dalla copia locale")
    row = _turn_rows()[0]
    assert row["graph_failover"] is True
    assert row["graph_label"] == "local mirror"


def test_without_a_model_url_there_is_nothing_to_fail_over_to(monkeypatch):
    if not app._GRAPH_OUTAGE_EXCEPTIONS:
        pytest.skip("neo4j exception classes unavailable")
    monkeypatch.setattr(
        app, "_rebuild_agent", lambda base_url, model_id: pytest.fail("must not rebuild")
    )

    payload = _ask(_Agent([app._GRAPH_OUTAGE_EXCEPTIONS[0]("unreachable")]), base_url="")

    assert payload["error"] == "service"


def test_a_rebuild_that_fails_leaves_the_original_failure(monkeypatch):
    if not app._GRAPH_OUTAGE_EXCEPTIONS:
        pytest.skip("neo4j exception classes unavailable")
    monkeypatch.setattr(app, "_rebuild_agent", lambda base_url, model_id: None)

    payload = _ask(
        _Agent([app._GRAPH_OUTAGE_EXCEPTIONS[0]("unreachable")]),
        base_url="http://localhost:8000/v1",
    )

    assert payload["error"] == "service"


def test_an_ordinary_failure_is_not_retried_anywhere(monkeypatch):
    monkeypatch.setattr(
        app, "_rebuild_agent", lambda base_url, model_id: pytest.fail("must not rebuild")
    )

    payload = _ask(_Agent([ValueError("bad prompt")]), base_url="http://x/v1")

    assert payload["error"] == "question"


# --- a failed turn -------------------------------------------------------


def test_a_failed_turn_is_logged_with_its_error_and_its_identity():
    _ask(_Agent([RuntimeError("vLLM down")]))

    row = _turn_rows()[0]
    assert row["error"] == "RuntimeError: vLLM down"
    assert row["turn_id"] == "t1"


def test_a_failure_tells_the_reader_which_kind_it_was():
    payload = _ask(_Agent([RuntimeError("vLLM down")]))

    assert payload["error"] == "question"
    assert payload["body"] == ""


def test_a_turn_is_logged_whatever_happened():
    agent = _Agent([RuntimeError("boom"), {"answer": "ok"}])

    _ask(agent)
    _ask(agent, turn_id="t2", turn_index=2)

    assert len(_turn_rows()) == 2
