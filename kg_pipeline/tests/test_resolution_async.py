"""Stage 4's merge confirmation calls the LLM concurrently.

What is pinned here is that concurrency changes only the wall clock: the same
pairs are approved, a failed batch still costs only itself, and verdicts
naming a group that does not exist are still dropped.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from kg_pipeline.stages import resolution


class _FakeCompletions:
    """`chat.completions` of the fake client; tracks overlapping calls."""

    def __init__(self, owner: "_FakeAsyncClient") -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> Any:
        prompt = kwargs["messages"][0]["content"]
        self._owner.in_flight += 1
        self._owner.peak_in_flight = max(
            self._owner.peak_in_flight, self._owner.in_flight
        )
        try:
            # Yield control so overlapping calls actually overlap.
            await asyncio.sleep(0.01)
            if self._owner.fail_on and self._owner.fail_on in prompt:
                raise RuntimeError("vLLM said no")
            return self._owner.reply_for(prompt)
        finally:
            self._owner.in_flight -= 1


class _FakeAsyncClient:
    """Stands in for AsyncOpenAI: approves every pair it is shown."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.fail_on: str | None = _FakeAsyncClient.fail_on_next
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(self)})()

    fail_on_next: str | None = None

    async def __aenter__(self) -> "_FakeAsyncClient":
        _FakeAsyncClient.last = self
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def reply_for(self, prompt: str) -> Any:
        pairs = json.loads(prompt.split("Pairs:\n", 1)[1])
        content = json.dumps(
            [
                {
                    "left_group": pair["left_group"],
                    "right_group": pair["right_group"],
                    "merge": True,
                }
                for pair in pairs
            ]
        )
        message = type("_Msg", (), {"content": content})()
        choice = type("_Choice", (), {"message": message})()
        return type("_Response", (), {"choices": [choice]})()


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    """Use the fake client in place of AsyncOpenAI for each test."""
    _FakeAsyncClient.fail_on_next = None
    monkeypatch.setattr(resolution, "AsyncOpenAI", _FakeAsyncClient)
    yield


def _batch(doc: str, *pairs: tuple[int, int]) -> tuple[str, list[dict[str, Any]]]:
    """A document batch of (left, right) group pairs."""
    return doc, [
        {"left_group": left, "right_group": right} for left, right in pairs
    ]


def _run(doc_batches, concurrent_requests=8, group_count=100):
    """Confirm `doc_batches` through the fake client."""
    return asyncio.run(
        resolution._confirm_batches_async(
            doc_batches=doc_batches,
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            http_timeout=30.0,
            concurrent_requests=concurrent_requests,
            model_name="fake",
            group_count=group_count,
        )
    )


def test_every_batch_is_confirmed_and_nothing_is_lost():
    batches = [_batch(f"doc{i}.pdf", (i, i + 1)) for i in range(12)]
    approved: set[tuple[int, int]] = set()
    for result in _run(batches):
        approved.update(result)
    assert approved == {(i, i + 1) for i in range(12)}


def test_batches_actually_overlap():
    batches = [_batch(f"doc{i}.pdf", (i, i + 1)) for i in range(12)]
    _run(batches, concurrent_requests=4)
    assert _FakeAsyncClient.last.peak_in_flight > 1, "calls still serialised"


def test_concurrency_never_exceeds_the_configured_limit():
    batches = [_batch(f"doc{i}.pdf", (i, i + 1)) for i in range(20)]
    _run(batches, concurrent_requests=3)
    assert _FakeAsyncClient.last.peak_in_flight <= 3


def test_one_failed_batch_does_not_take_the_others_with_it():
    _FakeAsyncClient.fail_on_next = "doc3.pdf"
    batches = [_batch(f"doc{i}.pdf", (i, i + 1)) for i in range(6)]
    approved: set[tuple[int, int]] = set()
    for result in _run(batches):
        approved.update(result)
    assert (3, 4) not in approved
    assert approved == {(i, i + 1) for i in range(6)} - {(3, 4)}


def test_a_verdict_naming_a_group_that_does_not_exist_is_dropped():
    content = json.dumps(
        [
            {"left_group": 0, "right_group": 1, "merge": True},
            {"left_group": 0, "right_group": 99, "merge": True},
            {"left_group": 2, "right_group": 3, "merge": False},
        ]
    )
    assert resolution._approved_from_response(content, group_count=5) == {(0, 1)}


def test_pair_order_does_not_change_the_key():
    content = json.dumps([{"left_group": 4, "right_group": 1, "merge": True}])
    assert resolution._approved_from_response(content, group_count=5) == {(1, 4)}
