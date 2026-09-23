"""What the console writes about a turn, so the turn can be read back.

Both frontends append to files that look alike and carry `surface`/`kind` to
say which wrote which. The promise those fields make is that the two are
comparable, so the console writes the same `chat_id`, `turn_id` and
`turn_index`: without them a turn answered there could not be rated — the
feedback record points at a `turn_id` — and a session could not be put back
in order.

No model, no graph: `build_demo_agent` is replaced and the REPL is driven by a
scripted stdin.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from product import config as settings
from product import console


class _Agent:
    """Agent answering from a queue of results or exceptions; records each question."""

    def __init__(self, answers: list[Any] | None = None) -> None:
        self.answers = list(answers or [])
        self.asked: list[str] = []

    def invoke(self, question: str, memory: Any = None) -> dict[str, Any]:
        self.asked.append(question)
        outcome = self.answers.pop(0) if self.answers else {"answer": "una risposta"}
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def run_console(monkeypatch, tmp_path):
    """Drives one console session over a scripted stdin, returns its log rows."""

    def _run(inputs: list[str], answers: list[Any] | None = None, label: str = "staging"):
        agent = _Agent(answers)
        typed = iter(inputs)

        monkeypatch.setattr(
            "sys.argv",
            ["console", "--model-id", "m", "--vllm-base-url", "http://localhost:8000/v1"],
        )
        monkeypatch.setattr("builtins.input", lambda *_a: next(typed))
        monkeypatch.setattr(settings, "LOG_DIR", tmp_path)
        monkeypatch.setattr(settings, "MEMORY", False)
        monkeypatch.setattr(
            settings, "build_demo_agent", lambda base_url, model_id: (agent, label)
        )
        monkeypatch.chdir(tmp_path)

        # `console.main` calls logging.basicConfig and silences the "graphrag"
        # logger for the whole process, which is right for a REPL and wrong for
        # every test that runs after this one.
        graphrag_logger = logging.getLogger("graphrag")
        previous_level = graphrag_logger.level
        root = logging.getLogger()
        previous_root_level = root.level
        previous_handlers = list(root.handlers)
        try:
            console.main()
        finally:
            graphrag_logger.setLevel(previous_level)
            root.setLevel(previous_root_level)
            root.handlers[:] = previous_handlers

        # The file is opened on the first write, so a session that asked
        # nothing leaves none behind.
        logs = sorted(tmp_path.glob("session_*.jsonl"))
        assert len(logs) <= 1, logs
        rows = [
            json.loads(line)
            for log in logs
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return agent, rows

    return _run


# --- the identity a turn needs ---------------------------------------------


def test_every_turn_carries_the_three_ids(run_console):
    _, rows = run_console(["cos'e' la scotta?", "esci"])

    assert len(rows) == 1
    assert set(rows[0]) >= {"chat_id", "turn_id", "turn_index", "surface", "kind"}
    assert rows[0]["surface"] == "console"
    assert rows[0]["kind"] == "turn"


def test_turns_of_one_conversation_are_numbered_in_order(run_console):
    _, rows = run_console(["prima", "seconda", "terza", "esci"])

    assert [r["turn_index"] for r in rows] == [1, 2, 3]
    assert len({r["chat_id"] for r in rows}) == 1


def test_every_turn_has_its_own_id(run_console):
    _, rows = run_console(["prima", "seconda", "esci"])

    assert len({r["turn_id"] for r in rows}) == 2


@pytest.mark.parametrize("word", ["nuova", "new", "reset"])
def test_starting_a_new_conversation_starts_a_new_chat(run_console, word):
    _, rows = run_console(["prima", word, "seconda", "esci"])

    assert len(rows) == 2
    assert rows[0]["chat_id"] != rows[1]["chat_id"]
    assert [r["turn_index"] for r in rows] == [1, 1]


def test_which_graph_answered_is_recorded(run_console):
    # Without it a session served by the local mirror during an Aura outage
    # reads exactly like a healthy one.
    _, rows = run_console(["prima", "esci"], label="local mirror (7689)")

    assert rows[0]["graph_label"] == "local mirror (7689)"


# --- the rest of the record ------------------------------------------------


def test_a_successful_turn_records_its_answer_and_latency(run_console):
    _, rows = run_console(
        ["cos'e' la scotta?", "esci"], answers=[{"answer": "E' un residuo."}]
    )

    assert rows[0]["question"] == "cos'e' la scotta?"
    assert rows[0]["answer"] == "E' un residuo."
    assert isinstance(rows[0]["latency_s"], float)
    assert "error" not in rows[0]


def test_a_rewritten_question_is_recorded_when_there_is_one(run_console):
    _, rows = run_console(
        ["e quindi?", "esci"],
        answers=[{"answer": "x", "rewritten_question": "e quindi la scotta?"}],
    )

    assert rows[0]["rewritten_question"] == "e quindi la scotta?"


def test_no_rewrite_means_no_such_field(run_console):
    _, rows = run_console(["prima", "esci"], answers=[{"answer": "x"}])

    assert "rewritten_question" not in rows[0]


def test_a_failed_turn_is_still_logged_with_its_identity(run_console):
    # The REPL survives any failure, and the turn that failed is exactly the
    # one someone will want to look up afterwards.
    _, rows = run_console(["prima", "esci"], answers=[RuntimeError("vLLM down")])

    assert rows[0]["error"] == "RuntimeError: vLLM down"
    assert rows[0]["turn_index"] == 1
    assert rows[0]["chat_id"] and rows[0]["turn_id"]


def test_a_failure_does_not_end_the_session(run_console):
    agent, rows = run_console(
        ["prima", "seconda", "esci"], answers=[RuntimeError("boom"), {"answer": "ok"}]
    )

    assert agent.asked == ["prima", "seconda"]
    assert [r["turn_index"] for r in rows] == [1, 2]


def test_a_failed_turn_still_advances_the_numbering(run_console):
    _, rows = run_console(
        ["prima", "seconda", "esci"], answers=[RuntimeError("boom"), {"answer": "ok"}]
    )

    assert rows[1]["turn_index"] == 2


def test_the_debugging_block_stays_in_the_log_and_off_the_screen(run_console, capsys):
    _, rows = run_console(
        ["prima", "esci"],
        answers=[{"answer": "La risposta.\nVerifica nel grafo: 4:abc:1"}],
    )

    assert "Verifica nel grafo" in rows[0]["answer"]
    assert "4:abc:1" not in capsys.readouterr().out


# --- what is not a question ------------------------------------------------


@pytest.mark.parametrize("word", ["esci", "exit", "quit"])
def test_leaving_writes_nothing(run_console, word):
    agent, rows = run_console([word])

    assert rows == []
    assert agent.asked == []


def test_an_empty_line_is_not_a_turn(run_console):
    agent, rows = run_console(["", "   ", "esci"])

    assert rows == []
    assert agent.asked == []


def test_resetting_before_asking_anything_logs_nothing(run_console):
    _, rows = run_console(["nuova", "esci"])

    assert rows == []


def test_the_stage_split_is_recorded_when_the_agent_reports_one(run_console):
    # `latency_s` says how long a turn took; this says which stage spent the
    # time, on every turn.
    _, rows = run_console(
        ["prima", "esci"],
        answers=[{"answer": "x", "stage_timings_ms": {"retrieve": 900.0, "generate": 30000.0}}],
    )

    assert rows[0]["stage_timings_ms"] == {"retrieve": 900.0, "generate": 30000.0}


def test_an_agent_that_reports_no_split_adds_no_field(run_console):
    _, rows = run_console(["prima", "esci"], answers=[{"answer": "x"}])

    assert "stage_timings_ms" not in rows[0]
