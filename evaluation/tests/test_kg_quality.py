"""Tests for the KG quality measures computed from pipeline artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
for p in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evalkit.kg.kg_quality import compute_from_artifacts


def _make_artifacts(
    tmp_path: Path,
    n_triples: int = 10,
    n_entities: int = 5,
    n_chunks: int = 20,
    n_failed: int = 2,
    stage3_summary: dict | None = None,
) -> Path:
    """A fake KG pipeline run directory with the stage files the measures read."""
    # stage6
    stage6 = {
        "relationships_written": n_triples,
        "summary": {
            "nodes_by_label": [
                {"label": "Concept", "count": n_entities // 2},
                {"label": "Organization", "count": n_entities - n_entities // 2},
            ]
        },
    }
    (tmp_path / "stage6_neo4j_summary.json").write_text(json.dumps(stage6))

    # stage5
    triples = [
        {
            "subject": f"entity_{i % n_entities}",
            "predicate": "IS_RELATED_TO" if i % 2 == 0 else "CAUSES",
            "object": f"entity_{(i + 1) % n_entities}",
        }
        for i in range(n_triples)
    ]
    (tmp_path / "stage5_triples_linked.json").write_text(json.dumps(triples))

    # stage4 registry
    registry = {f"entity_{i}_raw": {"canonical_name": f"entity_{i}"} for i in range(n_entities + 3)}
    (tmp_path / "stage4_registry.json").write_text(json.dumps(registry))

    # stage1 chunks
    chunks = [{"chunk_id": f"chunk_{i}"} for i in range(n_chunks)]
    (tmp_path / "stage1_chunks.json").write_text(json.dumps(chunks))

    # stage0 documents
    docs = [{"doc_id": f"doc_{i}"} for i in range(3)]
    (tmp_path / "stage0_documents.json").write_text(json.dumps(docs))

    # failed_chunks.jsonl, in the shape `write_failed_chunk` actually produces:
    # one row per attempt, chunk nested under `chunk_metadata`.
    with (tmp_path / "failed_chunks.jsonl").open("w") as fh:
        for i in range(n_failed):
            for attempt in (1, 2, 3):
                fh.write(
                    json.dumps(
                        {
                            "chunk_metadata": {"chunk_id": f"chunk_{i}"},
                            "attempt": attempt,
                            "error": "LLM error",
                            "raw_response": "",
                        }
                    )
                    + "\n"
                )

    if stage3_summary is not None:
        (tmp_path / "stage3_summary.json").write_text(json.dumps(stage3_summary))

    return tmp_path


def test_basic_metrics(tmp_path: Path) -> None:
    artifacts = _make_artifacts(tmp_path, n_triples=10, n_entities=5, n_chunks=20, n_failed=2)
    result = compute_from_artifacts(artifacts)

    assert result.n_triples == 10
    assert result.n_entities == 5
    assert result.density == pytest.approx(10 / 5)
    assert result.n_predicates == 2  # IS_RELATED_TO, CAUSES
    assert result.predicate_entropy > 0
    # Two chunks failed, across six rows. The count is chunks, not rows.
    assert result.failed_chunks == 2
    assert result.failed_chunks_ratio == pytest.approx(2 / 20)
    assert result.n_documents == 3


def test_resolution_collapse(tmp_path: Path) -> None:
    artifacts = _make_artifacts(tmp_path, n_entities=5)
    result = compute_from_artifacts(artifacts)
    # pre-resolution: 8 entries (5 + 3), post: 5 nodes → collapse > 0
    assert result.resolution_collapse_ratio > 0.0


def test_gold_entity_coverage(tmp_path: Path) -> None:
    artifacts = _make_artifacts(tmp_path, n_entities=5)
    gold = ["entity_0", "entity_1", "entity_99"]  # entity_99 not in registry
    result = compute_from_artifacts(artifacts, gold_entities=gold)
    # entity_0_raw and entity_1_raw are keys in registry (lowercased match)
    assert result.entity_gold_coverage is not None
    assert 0.0 <= result.entity_gold_coverage <= 1.0


def test_missing_artifacts_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_from_artifacts(tmp_path / "nonexistent")


def test_partial_artifacts(tmp_path: Path) -> None:
    # Only stage5 present — should still produce partial results
    triples = [{"subject": "a", "predicate": "IS", "object": "b"}]
    (tmp_path / "stage5_triples_linked.json").write_text(json.dumps(triples))
    result = compute_from_artifacts(tmp_path)
    assert result.n_triples == 1
    assert result.n_predicates == 1


# --- the failure rate must be the one stage 3 measured ----------------------


def test_a_retried_chunk_is_counted_once_not_once_per_attempt(tmp_path: Path) -> None:
    # Three rows per chunk is what the retry loop writes. Counting lines would
    # turn 2 lost chunks out of 20 into 6, i.e. 30 % against a true 10 %.
    artifacts = _make_artifacts(tmp_path, n_chunks=20, n_failed=2)
    result = compute_from_artifacts(artifacts)

    assert (tmp_path / "failed_chunks.jsonl").read_text().count("\n") == 6
    assert result.failed_chunks == 2
    assert result.extra["failed_chunks_is_upper_bound"] is True


def test_a_chunk_that_recovered_is_not_a_lost_chunk(tmp_path: Path) -> None:
    # chunk_0 failed an attempt and then produced a triple. The log cannot say
    # so; the triples file can.
    artifacts = _make_artifacts(tmp_path, n_chunks=20, n_failed=2)
    (artifacts / "stage3_triples_raw.json").write_text(
        json.dumps(
            [{"subject": "a", "predicate": "IS", "object": "b",
              "relationship_properties": {"chunk_id": "chunk_0"}}]
        )
    )
    result = compute_from_artifacts(artifacts)

    assert result.failed_chunks == 1


def test_the_stage_three_summary_wins_over_the_log(tmp_path: Path) -> None:
    artifacts = _make_artifacts(
        tmp_path,
        n_chunks=20,
        n_failed=2,
        stage3_summary={
            "chunks_in": 20,
            "chunks_skipped_front_back_matter": 4,
            "chunks_eligible": 16,
            "chunks_attempted": 16,
            "chunks_failed": 1,
            "failed_chunk_ids": ["chunk_1"],
        },
    )
    result = compute_from_artifacts(artifacts)

    # Stage 3 is the only party that knows which chunks it gave up on.
    assert result.failed_chunks == 1
    assert "failed_chunks_is_upper_bound" not in result.extra
    # Chunks never attempted are not in the denominator.
    assert result.failed_chunks_ratio == pytest.approx(1 / 16)
    assert result.extra["stage3"]["failed_chunk_ids"] == ["chunk_1"]


def test_an_absent_summary_is_not_a_warning(tmp_path: Path, caplog) -> None:
    artifacts = _make_artifacts(tmp_path, n_chunks=20, n_failed=0)
    with caplog.at_level("WARNING", logger="graphrag"):
        compute_from_artifacts(artifacts)
    # Some run layouts have no stage3_summary.json. That is expected, not a defect.
    assert "stage3_summary.json" not in caplog.text


def test_a_log_it_cannot_parse_is_reported_not_swallowed(tmp_path: Path, caplog) -> None:
    artifacts = _make_artifacts(tmp_path, n_chunks=20, n_failed=1)
    with (artifacts / "failed_chunks.jsonl").open("a") as fh:
        fh.write('{"chunk_metadata": {"chunk_id": "chunk_9"}, "att')  # killed mid-append

    with caplog.at_level("WARNING", logger="graphrag"):
        result = compute_from_artifacts(artifacts)

    assert result.failed_chunks == 1
    assert "could not be attributed to a chunk" in caplog.text
