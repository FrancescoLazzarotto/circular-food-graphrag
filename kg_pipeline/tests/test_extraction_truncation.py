"""Stage 3 must not lose a chunk because the model was cut off at the cap.

A chunk whose triples do not fit in the token budget comes back as a JSON
array cut mid-value, and the parser raises. Re-sending the byte-identical
request cannot help — vLLM decodes greedily at temperature 0 and never reads
the seed — so without reading ``finish_reason`` the chunk would be dropped
after paying for four calls, and nothing in the run would say so.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from kg_pipeline.models.types import ChunkRecord
from kg_pipeline.stages import llm_extraction


def _chunk(chunk_id: str = "c1") -> ChunkRecord:
    """A chunk of the demo document."""
    return ChunkRecord(
        doc_id="d1",
        filename="demo.pdf",
        chunk_id=chunk_id,
        page_range="1-2",
        section_title="Circular economy",
        chunk_index=1,
        text="Rice husk is used as a substrate.",
    )


_TRIPLE = {
    "subject": "Rice husk",
    "predicate": "USED_AS",
    "object": "substrate",
    "subject_labels": ["Material"],
    "object_labels": ["Material"],
    "subject_properties": {"name": "Rice husk"},
    "object_properties": {"name": "substrate"},
    "relationship_properties": {"source_doc": "demo.pdf", "extraction_method": "llm"},
}

_COMPLETE = json.dumps([_TRIPLE], ensure_ascii=False)
# What the cap actually produces: valid JSON up to the point it was cut.
_TRUNCATED = _COMPLETE[: len(_COMPLETE) // 2]


class _FakeClient:
    """Records every call and replays a scripted list of (text, finish_reason)."""

    def __init__(self, script: list[tuple[str, str]]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []
        self.chat = self  # the SDK shape is client.chat.completions.create
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        text, finish_reason = self._script[min(len(self.calls) - 1, len(self._script) - 1)]

        class _Message:
            content = text

        class _Choice:
            message = _Message()

        choice = _Choice()
        choice.finish_reason = finish_reason

        class _Response:
            choices = [choice]

        return _Response()


def _run(client: _FakeClient, tmp_path: Path, *, max_retries: int = 3, temperature: float = 0.0):
    """Run one chunk's extraction against the fake client."""
    return asyncio.run(
        llm_extraction._extract_chunk_async(
            client=client,
            semaphore=asyncio.Semaphore(1),
            chunk_idx=7,
            chunk=_chunk(),
            prompt="extract",
            model_name="test-model",
            temperature=temperature,
            seed=42,
            use_structured_output=True,
            max_retries=max_retries,
            allowed_label_set={"Material"},
            failed_chunks_path=tmp_path / "failed_chunks.jsonl",
            new_label_log_path=tmp_path / "new_labels.log",
            allowed_predicates=["USED_AS"],
        )
    )


def test_truncated_chunk_is_recovered_with_a_bigger_budget(tmp_path):
    client = _FakeClient([(_TRUNCATED, "length"), (_COMPLETE, "stop")])

    chunk_idx, triples, success = _run(client, tmp_path)

    assert success is True
    assert chunk_idx == 7
    assert [t.object for t in triples] == ["substrate"]
    # The retry is the same request with room to finish, not a coin flip.
    assert client.calls[1]["max_tokens"] == 2 * client.calls[0]["max_tokens"]


def test_retry_stops_resending_an_identical_request(tmp_path):
    client = _FakeClient([("not json at all", "stop"), (_COMPLETE, "stop")])

    _, triples, success = _run(client, tmp_path)

    assert success is True and len(triples) == 1
    first, second = client.calls[0], client.calls[1]
    # At temperature 0 vLLM ignores the seed, so a seed-only change leaves the
    # two requests identical and the second failure guaranteed.
    assert first["temperature"] == 0.0
    assert second["temperature"] > 0.0
    assert second["seed"] != first["seed"]


def test_a_run_that_succeeds_first_time_stays_deterministic(tmp_path):
    client = _FakeClient([(_COMPLETE, "stop")])

    _, triples, success = _run(client, tmp_path)

    assert success is True and len(triples) == 1
    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["seed"] == 42
    assert client.calls[0]["max_tokens"] == llm_extraction._MAX_OUTPUT_TOKENS


def test_budget_stops_doubling_at_the_ceiling(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(llm_extraction, "_MAX_OUTPUT_TOKENS", 100)
    monkeypatch.setattr(llm_extraction, "_MAX_OUTPUT_TOKENS_CEILING", 200)
    client = _FakeClient([(_TRUNCATED, "length")])

    with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
        _, triples, success = _run(client, tmp_path, max_retries=5)

    assert success is False and triples == []
    assert [c["max_tokens"] for c in client.calls] == [100, 200]
    assert "ceiling" in caplog.text


def test_a_lost_chunk_says_it_was_truncated(tmp_path):
    client = _FakeClient([(_TRUNCATED, "length")])
    failed = tmp_path / "failed_chunks.jsonl"

    _, _, success = _run(client, tmp_path, max_retries=1)

    assert success is False
    rows = [json.loads(line) for line in failed.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    # The operator needs to read "cut off at the cap", not "unterminated string".
    assert rows[0]["error"].startswith("truncated:")
    assert rows[0]["chunk_metadata"]["chunk_id"] == "c1"


def test_an_empty_array_is_still_an_answer_not_a_failure(tmp_path):
    client = _FakeClient([("[]", "stop")])

    _, triples, success = _run(client, tmp_path, max_retries=3)

    assert success is True and triples == []
    # Accepted after the second look, not after burning every retry.
    assert len(client.calls) == 2
    assert not (tmp_path / "failed_chunks.jsonl").exists()


@pytest.mark.parametrize("finish_reason", ["stop", "", None])
def test_a_complete_answer_is_never_treated_as_truncated(tmp_path, finish_reason):
    client = _FakeClient([(_COMPLETE, finish_reason or "")])

    _, triples, success = _run(client, tmp_path)

    assert success is True and len(triples) == 1
    assert len(client.calls) == 1


class _FakeAsyncOpenAI(_FakeClient):
    """`_extract_all_batches_async` opens the client as an async context."""

    def __init__(self, script, **_kwargs):
        super().__init__(script)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def test_stage_three_reports_the_chunks_it_lost(tmp_path, monkeypatch, caplog):
    script = [(_TRUNCATED, "length")]
    monkeypatch.setattr(
        llm_extraction,
        "AsyncOpenAI",
        lambda **kwargs: _FakeAsyncOpenAI(script, **kwargs),
    )
    monkeypatch.setattr(llm_extraction, "_MAX_OUTPUT_TOKENS", 100)
    monkeypatch.setattr(llm_extraction, "_MAX_OUTPUT_TOKENS_CEILING", 100)

    with caplog.at_level(logging.INFO, logger="kg_pipeline"):
        triples, acronyms = llm_extraction.extract_triples(
            chunks=[_chunk("c1"), _chunk("c2")],
            ner_map={},
            allowed_labels=["Material"],
            base_url="http://localhost:9/v1",
            model_name="test-model",
            api_key="EMPTY",
            max_retries_per_chunk=1,
            temperature=0.0,
            seed=42,
            use_structured_output=True,
            failed_chunks_path=tmp_path / "failed_chunks.jsonl",
            new_label_log_path=tmp_path / "new_labels.log",
            relation_vocab=["USED_AS"],
            checkpoint_every=0,
        )

    assert triples == []
    assert isinstance(acronyms, dict)
    # The whole point: the run does not end quietly on a corpus it dropped.
    assert "2 of 2 chunks produced no triples" in caplog.text
    assert "failed_chunks.jsonl" in caplog.text


def test_stage_three_says_so_when_nothing_was_lost(tmp_path, monkeypatch, caplog):
    script = [(_COMPLETE, "stop")]
    monkeypatch.setattr(
        llm_extraction,
        "AsyncOpenAI",
        lambda **kwargs: _FakeAsyncOpenAI(script, **kwargs),
    )

    with caplog.at_level(logging.INFO, logger="kg_pipeline"):
        triples, _ = llm_extraction.extract_triples(
            chunks=[_chunk("c1")],
            ner_map={},
            allowed_labels=["Material"],
            base_url="http://localhost:9/v1",
            model_name="test-model",
            api_key="EMPTY",
            max_retries_per_chunk=1,
            temperature=0.0,
            seed=42,
            use_structured_output=True,
            failed_chunks_path=tmp_path / "failed_chunks.jsonl",
            new_label_log_path=tmp_path / "new_labels.log",
            relation_vocab=["USED_AS"],
            checkpoint_every=0,
        )

    assert len(triples) == 1
    assert "no chunk lost" in caplog.text


# --- one bad chunk must not cost the batch ---------------------------------
#
# `_run_batch_async` gathers a batch of coroutines. Every `write_failed_chunk`
# call sits inside an `except` handler, so an I/O error there (full disk,
# read-only run directory) raises *out* of the handler; reaching the gather,
# it would be handed to the awaiter and the other results dropped. Nothing
# above catches it, so stage 3 — seven hours at production size — would end
# on a failed log write.


def _batch_of(n: int) -> list[ChunkRecord]:
    """`n` chunks, c0 to c(n-1)."""
    return [_chunk(f"c{i}") for i in range(n)]


def _run_batch(client: _FakeClient, tmp_path: Path, chunks: list[ChunkRecord]):
    """Run one extraction batch over `chunks`."""
    tasks = [(i, ch, "extract") for i, ch in enumerate(chunks)]
    return asyncio.run(
        llm_extraction._run_batch_async(
            batch_tasks=tasks,
            client=client,
            concurrent_requests=4,
            model_name="test-model",
            temperature=0.0,
            seed=42,
            use_structured_output=True,
            max_retries=1,
            allowed_label_set={"Material"},
            failed_chunks_path=tmp_path / "failed_chunks.jsonl",
            new_label_log_path=tmp_path / "new_labels.log",
            allowed_predicates=["USED_AS"],
        )
    )


def test_a_failing_failure_log_does_not_lose_the_batch(tmp_path, monkeypatch, caplog):
    # The trigger: the append that records a lost chunk fails.
    def _no_space(**_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(llm_extraction, "write_failed_chunk", _no_space)
    # c0 fails and tries to log it; c1..c3 are ordinary successes.
    client = _FakeClient([("not json at all", "stop"), (_COMPLETE, "stop")])

    with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
        results = _run_batch(client, tmp_path, _batch_of(4))

    # Every chunk still comes back, in order, with its own verdict.
    assert [idx for idx, _, _ in results] == [0, 1, 2, 3]
    assert sum(1 for _, _, success in results if success) == 3
    # The row could not be written, and the run says so instead of dying.
    assert "could not record the failure of chunk" in caplog.text


def test_an_unexpected_exception_costs_one_chunk_not_the_others(tmp_path, caplog):
    boom = OSError("connection reset by peer")
    calls = {"n": 0}

    async def _flaky(**kwargs):
        calls["n"] += 1
        if kwargs["chunk"].chunk_id == "c1":
            raise boom
        return kwargs["chunk_idx"], [], True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llm_extraction, "_extract_chunk_async", _flaky)
        with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
            results = _run_batch(_FakeClient([(_COMPLETE, "stop")]), tmp_path, _batch_of(4))

    assert calls["n"] == 4, "the siblings must still be run, not cancelled"
    assert [(idx, success) for idx, _, success in results] == [
        (0, True),
        (1, False),
        (2, True),
        (3, True),
    ]
    assert "lost to an unhandled OSError" in caplog.text

    # A chunk lost this way is filed like any other lost chunk, not vanished.
    rows = [
        json.loads(line)
        for line in (tmp_path / "failed_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["chunk_metadata"]["chunk_id"] for r in rows] == ["c1"]
    assert rows[0]["error"] == "unhandled OSError: connection reset by peer"


def test_a_cancellation_is_not_filed_as_a_failed_chunk(tmp_path):
    async def _cancelled(**kwargs):
        if kwargs["chunk"].chunk_id == "c1":
            raise asyncio.CancelledError()
        return kwargs["chunk_idx"], [], True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llm_extraction, "_extract_chunk_async", _cancelled)
        with pytest.raises(asyncio.CancelledError):
            _run_batch(_FakeClient([(_COMPLETE, "stop")]), tmp_path, _batch_of(3))

    # Shutting the run down is not a corpus defect: nothing is written.
    assert not (tmp_path / "failed_chunks.jsonl").exists()


# --- stage 3 writes down what only stage 3 knows --------------------------


def test_stage_three_writes_a_summary_of_what_it_lost(tmp_path, monkeypatch):
    script = [(_TRUNCATED, "length")]
    monkeypatch.setattr(
        llm_extraction,
        "AsyncOpenAI",
        lambda **kwargs: _FakeAsyncOpenAI(script, **kwargs),
    )
    monkeypatch.setattr(llm_extraction, "_MAX_OUTPUT_TOKENS", 100)
    monkeypatch.setattr(llm_extraction, "_MAX_OUTPUT_TOKENS_CEILING", 100)

    # One ordinary chunk and one that is back matter and never attempted.
    back_matter = _chunk("c2")
    back_matter.section_title = "References"

    llm_extraction.extract_triples(
        chunks=[_chunk("c1"), back_matter],
        ner_map={},
        allowed_labels=["Material"],
        base_url="http://localhost:9/v1",
        model_name="test-model",
        api_key="EMPTY",
        max_retries_per_chunk=1,
        temperature=0.0,
        seed=42,
        use_structured_output=True,
        failed_chunks_path=tmp_path / "failed_chunks.jsonl",
        new_label_log_path=tmp_path / "new_labels.log",
        relation_vocab=["USED_AS"],
        checkpoint_every=0,
    )

    summary = json.loads((tmp_path / "stage3_summary.json").read_text(encoding="utf-8"))
    assert summary["chunks_in"] == 2
    assert summary["chunks_skipped_front_back_matter"] == 1
    assert summary["chunks_attempted"] == 1
    assert summary["chunks_failed"] == 1
    assert summary["failed_chunk_ids"] == ["c1"]
    assert summary["max_retries_per_chunk"] == 1
    # The log holds one row per attempt; the summary holds the verdict.
    rows = (tmp_path / "failed_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) >= summary["chunks_failed"]


def test_a_clean_stage_three_says_zero_rather_than_saying_nothing(tmp_path, monkeypatch):
    script = [(_COMPLETE, "stop")]
    monkeypatch.setattr(
        llm_extraction,
        "AsyncOpenAI",
        lambda **kwargs: _FakeAsyncOpenAI(script, **kwargs),
    )

    llm_extraction.extract_triples(
        chunks=[_chunk("c1")],
        ner_map={},
        allowed_labels=["Material"],
        base_url="http://localhost:9/v1",
        model_name="test-model",
        api_key="EMPTY",
        max_retries_per_chunk=1,
        temperature=0.0,
        seed=42,
        use_structured_output=True,
        failed_chunks_path=tmp_path / "failed_chunks.jsonl",
        new_label_log_path=tmp_path / "new_labels.log",
        relation_vocab=["USED_AS"],
        checkpoint_every=0,
    )

    summary = json.loads((tmp_path / "stage3_summary.json").read_text(encoding="utf-8"))
    assert summary["chunks_failed"] == 0
    assert summary["failed_chunk_ids"] == []
    assert summary["triples_extracted"] == 1
