"""The operational log must not be a copy of what the corpus says.

`generate` can log the rendered prompt and the raw answer, ~1.3 KB per call,
in a log that is world-readable on the host. The two lines are different
problems: the prompt line slices 500 characters and the context starts around
character 830, so it only ever repeats the identical system prompt; the answer
line is the one carrying corpus material, verbatim whenever
`prefer_verbatim_definitions` is on. Both are worth having when a bad answer
needs explaining, so they live at DEBUG, and GRAPHRAG_LOG_PROMPT_TEXT=1
restores them at INFO for one session.
"""

from __future__ import annotations

import logging
from typing import Any

from graphrag.config import AgentConfig
from graphrag.llm.manager import LLMManager

# A sentence that only ever reaches the log by way of the retrieved context.
_CORPUS = "La paglia di riso viene impiegata come substrato nei processi."
_ANSWER = "Il substrato descritto deriva dalla paglia di riso."


class _FakeOutput:
    """Chat-model output carrying only `content`."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    """Model that records its prompts and returns a fixed answer."""

    def __init__(self) -> None:
        self.prompts: list[Any] = []

    def invoke(self, payload: Any) -> _FakeOutput:
        self.prompts.append(payload)
        return _FakeOutput(_ANSWER)


def _manager() -> LLMManager:
    """An `LLMManager` over `_FakeModel`, retries bypassed."""
    manager = LLMManager.__new__(LLMManager)
    model = _FakeModel()
    manager.load_llm = lambda: model  # type: ignore[method-assign]
    manager._invoke_with_retry = lambda m, payload: m.invoke(payload)  # type: ignore[method-assign]
    return manager


def _generate(manager: LLMManager) -> None:
    """One generate call whose context is corpus text."""
    config = AgentConfig()
    config.enforce_language = False
    manager.generate(query="Che substrato si usa?", context=_CORPUS, config=config)


def test_corpus_text_does_not_reach_an_info_log(monkeypatch, caplog):
    monkeypatch.delenv("GRAPHRAG_LOG_PROMPT_TEXT", raising=False)

    with caplog.at_level(logging.INFO, logger="graphrag"):
        _generate(_manager())

    assert _CORPUS not in caplog.text
    assert _ANSWER not in caplog.text


def test_the_shape_of_the_turn_is_still_visible_at_info(monkeypatch, caplog):
    monkeypatch.delenv("GRAPHRAG_LOG_PROMPT_TEXT", raising=False)

    with caplog.at_level(logging.INFO, logger="graphrag"):
        _generate(_manager())

    # An operator must still be able to tell a truncated answer from an empty
    # one without reading a single word of the corpus.
    assert f"Context length (chars): {len(_CORPUS)}" in caplog.text
    assert f"Answer length (chars): {len(_ANSWER)}" in caplog.text


def test_debug_still_carries_the_answer_for_a_real_investigation(monkeypatch, caplog):
    monkeypatch.delenv("GRAPHRAG_LOG_PROMPT_TEXT", raising=False)

    with caplog.at_level(logging.DEBUG, logger="graphrag"):
        _generate(_manager())

    assert _ANSWER in caplog.text
    assert "Rendered prompt" in caplog.text


def test_the_prompt_window_never_reached_the_context_anyway(caplog):
    """Guards the assumption that keeps the prompt line harmless.

    The logged slice is 500 characters and the context begins around 830, so
    widening the slice would start leaking the corpus for real. If someone
    raises it, this test says so.
    """
    with caplog.at_level(logging.DEBUG, logger="graphrag"):
        _generate(_manager())

    prompt_lines = [r for r in caplog.records if "Rendered prompt" in r.getMessage()]
    assert prompt_lines, "the prompt line disappeared"
    assert _CORPUS not in prompt_lines[0].getMessage()


def test_the_flag_puts_the_text_back_at_info(monkeypatch, caplog):
    monkeypatch.setenv("GRAPHRAG_LOG_PROMPT_TEXT", "1")

    with caplog.at_level(logging.INFO, logger="graphrag"):
        _generate(_manager())

    assert _ANSWER in caplog.text
    assert "Rendered prompt" in caplog.text


def test_the_flag_is_read_per_call_not_pinned_at_import(monkeypatch, caplog):
    manager = _manager()

    monkeypatch.setenv("GRAPHRAG_LOG_PROMPT_TEXT", "1")
    with caplog.at_level(logging.INFO, logger="graphrag"):
        _generate(manager)
    assert _ANSWER in caplog.text

    caplog.clear()
    monkeypatch.setenv("GRAPHRAG_LOG_PROMPT_TEXT", "0")
    with caplog.at_level(logging.INFO, logger="graphrag"):
        _generate(manager)
    assert _ANSWER not in caplog.text
