"""Stage 3's dispatch window and its checkpoint cadence are independent.

A batch is the dispatch window: every chunk in it is in flight behind a
semaphore, and the next batch cannot start until the slowest one returns. A
checkpoint is how much work a crash costs. Tied to one number, asking for
safer recovery would narrow the window, and widening the window would put
more work at risk.

What is pinned here is that the two are independent, and that the checkpoint
still keeps its guarantees: one lands at least every `checkpoint_every`
chunks, and the last one always points at the last chunk actually done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kg_pipeline.models.types import ChunkRecord, KGTriple
from kg_pipeline.stages import llm_extraction


def _chunk(idx: int) -> ChunkRecord:
    """Chunk `idx` of one document."""
    return ChunkRecord.model_validate(
        {
            "doc_id": "doc.pdf",
            "filename": "doc.pdf",
            "chunk_id": f"doc.pdf::c{idx}",
            "page_range": "1-1",
            "section_title": "Introduction",
            "chunk_index": idx + 1,
            "text": f"chunk {idx} text",
        }
    )


def _triple(chunk_id: str) -> KGTriple:
    """A triple extracted from `chunk_id`."""
    return KGTriple.model_validate(
        {
            "subject": "Rice husk",
            "predicate": "USES",
            "object": "Substrate",
            "subject_labels": ["Material"],
            "object_labels": ["Material"],
            "subject_properties": {"name": "Rice husk"},
            "object_properties": {"name": "Substrate"},
            "relationship_properties": {"source_doc": "doc.pdf", "chunk_id": chunk_id},
        }
    )


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Runs stage 3 with the network replaced by a per-chunk stub.

    Records the size of every dispatched batch and the chunk index of every
    checkpoint, which is all these tests are about.
    """

    state: dict[str, Any] = {"batches": [], "checkpoints": [], "fail": set()}

    class _NullClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(llm_extraction, "AsyncOpenAI", lambda **kwargs: _NullClient())

    async def _fake_batch(*, batch_tasks, **kwargs):
        state["batches"].append(len(batch_tasks))
        out = []
        for chunk_idx, chunk, _prompt in batch_tasks:
            if chunk.chunk_id in state["fail"]:
                out.append((chunk_idx, [], False))
            else:
                out.append((chunk_idx, [_triple(chunk.chunk_id)], True))
        return out

    monkeypatch.setattr(llm_extraction, "_run_batch_async", _fake_batch)

    real_save_json = llm_extraction._save_json

    def _spy_save_json(path: Path, payload: Any) -> None:
        if path.name == "stage3_checkpoint_info.json":
            state["checkpoints"].append(payload["last_completed_chunk_idx"])
        real_save_json(path, payload)

    monkeypatch.setattr(llm_extraction, "_save_json", _spy_save_json)

    def _go(n_chunks: int, **kwargs):
        state["batches"].clear()
        state["checkpoints"].clear()
        triples, acronyms = llm_extraction.extract_triples(
            chunks=[_chunk(i) for i in range(n_chunks)],
            ner_map={},
            allowed_labels=["Material", "Concept"],
            base_url="http://localhost:8000/v1",
            model_name="m",
            api_key="EMPTY",
            max_retries_per_chunk=2,
            temperature=0.0,
            seed=1,
            use_structured_output=False,
            failed_chunks_path=tmp_path / "failed_chunks.jsonl",
            new_label_log_path=tmp_path / "new_labels.log",
            **kwargs,
        )
        return triples, state

    _go.state = state
    _go.dir = tmp_path
    return _go


# --- the two knobs are independent -----------------------------------------


def test_the_dispatch_window_no_longer_follows_the_checkpoint_setting(run, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", "8")

    _, state = run(64, checkpoint_every=4)

    # Tied to the checkpoint cadence this would be sixteen batches of 4.
    assert state["batches"] == [32, 32]


def test_an_explicit_batch_size_is_used_as_given(run):
    _, state = run(25, checkpoint_every=50, batch_size=10)

    assert state["batches"] == [10, 10, 5]


@pytest.mark.parametrize("bad", [0, -7])
def test_a_meaningless_batch_size_falls_back_to_the_default(run, monkeypatch, bad):
    monkeypatch.setenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", "2")

    _, state = run(16, checkpoint_every=50, batch_size=bad)

    assert state["batches"] == [8, 8]


def test_the_default_window_is_several_concurrency_windows_deep(run, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", "3")

    _, state = run(12, checkpoint_every=50)

    assert state["batches"] == [12]  # 3 * _BATCH_WINDOWS_IN_FLIGHT


def test_turning_checkpointing_off_no_longer_dispatches_the_whole_corpus_at_once(
    run, monkeypatch
):
    # `checkpoint_every=0` must not mean batch_size=len(chunks): one batch
    # building a coroutine per chunk in the corpus.
    monkeypatch.setenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", "4")

    _, state = run(50, checkpoint_every=0)

    assert state["batches"] == [16, 16, 16, 2]
    assert state["checkpoints"] == []


# --- the cadence the checkpoint still guarantees ---------------------------


def test_a_checkpoint_lands_at_least_every_n_chunks(run, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", "2")

    _, state = run(40, checkpoint_every=10, batch_size=4)

    # Every gap between checkpoints is within the interval asked for.
    gaps = [b - a for a, b in zip([-1] + state["checkpoints"], state["checkpoints"])]
    assert max(gaps) <= 12  # one interval plus at most one batch
    assert state["checkpoints"][-1] == 39


def test_a_batch_larger_than_the_interval_checkpoints_once_per_batch(run):
    _, state = run(60, checkpoint_every=5, batch_size=20)

    assert state["checkpoints"] == [19, 39, 59]


def test_the_last_checkpoint_always_points_at_the_last_chunk(run):
    # The loop only writes once enough chunks are behind it, so without the
    # final write a short tail would leave the checkpoint pointing backwards
    # and a rerun would redo work already in hand.
    _, state = run(23, checkpoint_every=10, batch_size=4)

    assert state["checkpoints"][-1] == 22


def test_a_run_shorter_than_one_interval_still_checkpoints_once(run):
    _, state = run(3, checkpoint_every=50, batch_size=10)

    assert state["checkpoints"] == [2]


def test_the_checkpoint_holds_everything_extracted_so_far(run):
    triples, state = run(20, checkpoint_every=10, batch_size=10)

    payload = json.loads((run.dir / "stage3_checkpoint.json").read_text(encoding="utf-8"))
    assert len(payload) == len(triples) == 20


def test_a_failed_chunk_still_advances_the_checkpoint(run):
    # The chunk is lost, not retried forever: the checkpoint has to move past
    # it or a resume would sit on it.
    run.state["fail"] = {"doc.pdf::c3"}
    triples, state = run(10, checkpoint_every=5, batch_size=5)

    assert state["checkpoints"] == [4, 9]
    assert len(triples) == 9
    run.state["fail"] = set()


def test_a_checkpoint_that_cannot_be_written_does_not_kill_the_run(
    run, monkeypatch, caplog
):
    real_save_json = llm_extraction._save_json

    def _explode(path: Path, payload: Any) -> None:
        # Only the checkpoint files: the end-of-run summary write is not
        # guarded, so breaking that one would take the whole stage down with
        # it — a different question from this one.
        if path.name.startswith("stage3_checkpoint"):
            raise OSError("disk full")
        real_save_json(path, payload)

    monkeypatch.setattr(llm_extraction, "_save_json", _explode)
    monkeypatch.setattr(
        llm_extraction,
        "save_triples",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    triples, _ = run(10, checkpoint_every=5, batch_size=5)

    assert len(triples) == 10
    assert "Failed to save checkpoint" in caplog.text


# --- resume works across the checkpoint cadence ----------------------------


def test_a_resumed_run_only_asks_for_what_is_missing(run, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", "2")
    run(20, checkpoint_every=5, batch_size=5)

    # Rewind the checkpoint to chunk 9 and run again over the same corpus.
    info_path = run.dir / "stage3_checkpoint_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["last_completed_chunk_idx"] = 9
    info_path.write_text(json.dumps(info), encoding="utf-8")
    triples = json.loads((run.dir / "stage3_checkpoint.json").read_text(encoding="utf-8"))
    (run.dir / "stage3_checkpoint.json").write_text(
        json.dumps(triples[:10]), encoding="utf-8"
    )

    _, state = run(20, checkpoint_every=5, batch_size=5)

    assert sum(state["batches"]) == 10
    assert state["checkpoints"][-1] == 19


def test_the_summary_counts_only_the_chunks_this_run_attempted(run):
    run(20, checkpoint_every=5, batch_size=5)

    summary = json.loads((run.dir / "stage3_summary.json").read_text(encoding="utf-8"))
    assert summary["chunks_attempted"] == 20
    assert summary["chunks_resumed_from_checkpoint"] == 0
    assert summary["chunks_failed"] == 0
