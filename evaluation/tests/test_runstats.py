"""Tests for per-strategy run statistics: latency, insufficiency, throughput."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
for p in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evalkit.metrics.runstats import _percentile, compute_run_stats
from evalkit.models import EvalRow


def _make_row(strategy: str, latency: float, insufficient: bool = False) -> EvalRow:
    """An evaluation row for `strategy`, with a refusal answer when insufficient."""
    return EvalRow(
        run_dir="test_run",
        strategy=strategy,
        framework="graphrag",
        model_id="test-model",
        run_index="0",
        question_id="",
        question_type="factoid",
        difficulty="easy",
        notes="",
        question="What is X?",
        answer="X is Y." if not insufficient else "I don't have enough information",
        ground_truth="X is Y.",
        answer_variants=[],
        contexts=["context one", "context two"],
        retrieved_triples=[],
        retrieved_entities=[],
        expected_entities=[],
        gold_triples=[],
        latency_ms=latency,
        kg_triples_used=3,
        kg_neighbors_used=2,
        kg_subgraph_triples_used=0,
        kg_shortest_path_triples_used=0,
        sub_questions=1,
        insufficient=insufficient,
        skip_reason="",
    )


def test_percentile_sorted() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(vals, 0.0) == pytest.approx(1.0)
    assert _percentile(vals, 1.0) == pytest.approx(5.0)
    assert _percentile(vals, 0.5) == pytest.approx(3.0)


def test_compute_run_stats_basic() -> None:
    rows = [
        _make_row("default", 100.0),
        _make_row("default", 200.0),
        _make_row("text_only", 50.0),
    ]
    stats = compute_run_stats(rows)

    assert "default" in stats
    assert "text_only" in stats
    assert "global" in stats

    default = stats["default"]
    assert default["n_rows"] == 2
    assert default["latency_mean"] == pytest.approx(150.0)
    assert default["latency_p50"] == pytest.approx(150.0)
    assert default["insufficient_rate"] == pytest.approx(0.0)


def test_insufficiency_rate() -> None:
    rows = [
        _make_row("default", 100.0, insufficient=False),
        _make_row("default", 100.0, insufficient=True),
        _make_row("default", 100.0, insufficient=True),
    ]
    stats = compute_run_stats(rows)
    assert stats["default"]["insufficient_rate"] == pytest.approx(2 / 3)
    assert stats["default"]["insufficient_count"] == 2


def test_throughput_positive() -> None:
    rows = [_make_row("default", 1000.0) for _ in range(10)]
    stats = compute_run_stats(rows)
    # 10 rows, total 10000 ms → 1 q/s
    assert stats["default"]["throughput_qps"] == pytest.approx(1.0)
