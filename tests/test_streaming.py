"""The answer has to reach the reader while it is being written.

Generation is nearly the whole wait: an answer runs to hundreds of tokens,
written a few dozen a second. The tests here pin the two properties that make
streaming safe to switch on — nothing changes for a caller that does not ask
for it, and a caller that does gets the text in the order the model produced
it.
"""

from __future__ import annotations

from typing import Any

from graphrag.config import AgentConfig
from graphrag.llm.manager import LLMManager


class _Chunk:
    """What a LangChain chat backend yields while streaming."""

    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.content = content
        self.response_metadata: dict[str, Any] = (
            {"finish_reason": finish_reason} if finish_reason else {}
        )

    def __add__(self, other: "_Chunk") -> "_Chunk":
        merged = _Chunk(self.content + other.content)
        merged.response_metadata = {**self.response_metadata, **other.response_metadata}
        return merged


class _StreamingModel:
    """Model streaming fixed pieces, the last with a finish reason."""

    def __init__(self, pieces: list[str], finish_reason: str = "stop") -> None:
        self.pieces = pieces
        self.finish_reason = finish_reason
        self.invoked = 0

    def stream(self, payload: Any):
        for index, piece in enumerate(self.pieces):
            last = index == len(self.pieces) - 1
            yield _Chunk(piece, self.finish_reason if last else None)

    def invoke(self, payload: Any) -> _Chunk:
        self.invoked += 1
        return _Chunk("".join(self.pieces), self.finish_reason)


def _manager() -> LLMManager:
    """An `LLMManager` without warmup, for tests that swap in a fake model."""
    return LLMManager(model_id="test", warmup=False)


def test_a_caller_that_listens_gets_the_text_as_it_is_written():
    model = _StreamingModel(["Le tre C ", "sono Capitale", " e Ciclicità."])
    seen: list[str] = []
    out = _manager()._invoke_with_retry(model, "prompt", on_token=seen.append)
    assert seen == ["Le tre C ", "sono Capitale", " e Ciclicità."]
    assert out.content == "Le tre C sono Capitale e Ciclicità."
    assert model.invoked == 0


def test_the_summed_chunks_keep_the_finish_reason():
    """The token-limit check reads it, and it only arrives on the last chunk."""
    model = _StreamingModel(["mezza ", "frase"], finish_reason="length")
    out = _manager()._invoke_with_retry(model, "prompt", on_token=lambda _p: None)
    assert LLMManager._hit_token_limit(out) is True


def test_nothing_changes_for_a_caller_that_does_not_listen():
    """Every campaign, the CLI and the console go through this path."""
    model = _StreamingModel(["una ", "risposta"])
    out = _manager()._invoke_with_retry(model, "prompt")
    assert out.content == "una risposta"
    assert model.invoked == 1


def test_a_backend_without_streaming_still_answers():
    class _Blocking:
        def invoke(self, payload: Any) -> _Chunk:
            return _Chunk("risposta")

    out = _manager()._invoke_with_retry(_Blocking(), "prompt", on_token=lambda _p: None)
    assert out.content == "risposta"


def test_an_empty_stream_falls_back_to_the_blocking_call():
    """A stream that yields nothing is not an answer."""
    model = _StreamingModel([])
    out = _manager()._invoke_with_retry(model, "prompt", on_token=lambda _p: None)
    assert out.content == ""
    assert model.invoked == 1


def test_generate_streams_only_the_first_attempt(monkeypatch):
    """The rescue retry rewrites the answer; streaming it would show two."""
    manager = _manager()
    calls: list[bool] = []

    def fake_invoke(model: Any, payload: Any, on_token: Any = None) -> _Chunk:
        calls.append(on_token is not None)
        return _Chunk("Non posso rispondere con il contesto fornito.")

    monkeypatch.setattr(manager, "load_llm", lambda: object())
    monkeypatch.setattr(manager, "_invoke_with_retry", fake_invoke)
    manager.generate(
        query="che cos'e il biochar?",
        context="del contesto",
        config=AgentConfig(enforce_language=False),
        on_token=lambda _p: None,
    )
    assert calls[0] is True
    assert all(streamed is False for streamed in calls[1:])
