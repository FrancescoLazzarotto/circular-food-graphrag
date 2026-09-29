"""Tests for dataset helpers, retrieval metrics, bootstrap CIs and aggregation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
for p in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evalkit.io.dataset import _determine_skip_reason, _validate_schema
from evalkit.io.gold_loader import normalize_question
from evalkit.judge.ragas_backend import _coerce_score, _to_ragas_sample
from evalkit.metrics.retrieval import (
    RETRIEVAL_METRICS,
    compute_retrieval_row,
    hit_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evalkit.metrics.stats import aggregate, bootstrap_ci
from evalkit.models import EvalRow


def _make_row(
    question: str = "q",
    answer: str = "a",
    ground_truth: str = "gt",
    question_type: str = "factoid",
    strategy: str = "default",
    model_id: str = "m1",
    expected_entities: list | None = None,
    retrieved_entities: list | None = None,
    gold_triples: list | None = None,
    retrieved_triples: list | None = None,
    contexts: list[str] | None = None,
    latency_ms: float = 10.0,
    skip_reason: str = "",
) -> EvalRow:
    """An evaluation row; every field has a default."""
    return EvalRow(
        run_dir="r1",
        strategy=strategy,
        framework="graphrag",
        model_id=model_id,
        run_index="0",
        question_id="",
        question_type=question_type,
        difficulty="",
        notes="",
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        answer_variants=[],
        contexts=contexts or ["ctx"],
        retrieved_triples=retrieved_triples or [],
        retrieved_entities=retrieved_entities or [],
        expected_entities=expected_entities or [],
        gold_triples=gold_triples or [],
        latency_ms=latency_ms,
        kg_triples_used=0,
        kg_neighbors_used=0,
        kg_subgraph_triples_used=0,
        kg_shortest_path_triples_used=0,
        sub_questions=0,
        insufficient=False,
        skip_reason=skip_reason,
    )


# ─── normalize_question ───────────────────────────────────────────────────────

def test_normalize_question_applies_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_NORMALIZE_REMOVE_ACCENTS", "1")
    assert normalize_question("  Caffè   latte???  ") == "caffe latte"

    monkeypatch.setenv("EVAL_NORMALIZE_REMOVE_ACCENTS", "0")
    assert normalize_question("  Caffè   latte???  ") == "caffè latte"


# ─── _determine_skip_reason ───────────────────────────────────────────────────

def test_determine_skip_reason_priority() -> None:
    assert _determine_skip_reason("", has_gold=True, malformed_json=False, contexts=["ctx"]) == "question_not_found"
    assert _determine_skip_reason("q", has_gold=True, malformed_json=True, contexts=["ctx"]) == "malformed_json"
    assert _determine_skip_reason("q", has_gold=True, malformed_json=False, contexts=[]) == "empty_context"
    assert _determine_skip_reason("q", has_gold=False, malformed_json=False, contexts=["ctx"]) == "no_gold"
    assert _determine_skip_reason("q", has_gold=True, malformed_json=False, contexts=["ctx"]) == ""


# ─── _validate_schema ────────────────────────────────────────────────────────

def test_validate_schema_warns_on_low_coverage(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        _make_row(ground_truth=""),  # missing
        _make_row(ground_truth=""),  # missing
        _make_row(ground_truth=""),  # missing
        _make_row(ground_truth="gt"),
    ]
    caplog.set_level(logging.WARNING, logger="graphrag")
    _validate_schema(rows, min_coverage=0.95)
    assert "coverage below" in caplog.text


# ─── retrieval metrics ────────────────────────────────────────────────────────

def test_precision_and_recall_with_k() -> None:
    relevant = ["a", "b"]
    retrieved = ["x", "a", "b"]

    assert precision_at_k(relevant, retrieved, k=2) == pytest.approx(0.5)
    assert recall_at_k(relevant, retrieved, k=2) == pytest.approx(0.5)
    assert precision_at_k(relevant, retrieved, k=None) == pytest.approx(2.0 / 3.0)
    assert recall_at_k(relevant, retrieved, k=None) == pytest.approx(1.0)


def test_rank_aware_metrics() -> None:
    relevant = ["a", "b"]
    retrieved = ["x", "a", "b"]

    assert hit_at_k(relevant, retrieved, k=1) == 0.0
    assert hit_at_k(relevant, retrieved, k=2) == 1.0
    assert ndcg_at_k(relevant, retrieved, k=3) == pytest.approx(0.6934, rel=1e-3)
    assert mrr(relevant, retrieved) == pytest.approx(0.5)
    assert mean_average_precision(relevant, retrieved) == pytest.approx((0.5 + (2.0 / 3.0)) / 2.0)


# ─── bootstrap_ci ─────────────────────────────────────────────────────────────

def test_bootstrap_ci_is_reproducible() -> None:
    scores = [0.1, 0.2, 0.4, 0.8]

    first = bootstrap_ci(scores, n_bootstrap=250, ci=0.95, seed=7)
    second = bootstrap_ci(scores, n_bootstrap=250, ci=0.95, seed=7)

    assert first == second
    assert first[0] <= first[1]
    assert 0.1 <= first[0] <= 0.8
    assert 0.1 <= first[1] <= 0.8


# ─── aggregate ───────────────────────────────────────────────────────────────

def test_aggregate_includes_segment_rows() -> None:
    rows = [
        _make_row(
            question_type="factoid",
            expected_entities=["a", "b"],
            retrieved_entities=["a", "b"],
            gold_triples=[{"subject": "a", "predicate": "R", "object": "b"}],
            retrieved_triples=[{"subject": "a", "predicate": "R", "object": "b"}],
            latency_ms=10.0,
        ),
        _make_row(
            question_type="relation",
            expected_entities=["x", "y"],
            retrieved_entities=["x"],
            gold_triples=[{"subject": "x", "predicate": "R", "object": "y"}],
            retrieved_triples=[{"subject": "x", "predicate": "R", "object": "y"}],
            latency_ms=20.0,
        ),
    ]

    row_metrics = []
    for row in rows:
        entry = compute_retrieval_row(row, k=None)
        entry["model_id"] = row.model_id
        entry["framework"] = row.framework
        entry["strategy"] = row.strategy
        entry["question_type"] = row.question_type
        row_metrics.append(entry)

    summaries = aggregate(
        row_metrics=row_metrics,
        metric_names=RETRIEVAL_METRICS,
        group_keys=("model_id", "framework", "strategy"),
        segment_key="question_type",
        n_bootstrap=50,
        ci=0.95,
        seed=42,
    )

    segments = {g.keys["segment"] for g in summaries}
    assert "global" in segments
    assert "factoid" in segments
    assert "relation" in segments

    global_group = next(g for g in summaries if g.keys["segment"] == "global")
    assert "precision_at_k" in global_group.metrics
    assert "mean" in global_group.metrics["precision_at_k"]
    assert "ci_lower" in global_group.metrics["precision_at_k"]
    assert "n" in global_group.metrics["precision_at_k"]


# ─── ragas helpers ────────────────────────────────────────────────────────────

def test_to_ragas_sample_filters_empty_contexts() -> None:
    row = _make_row(
        question="Q",
        answer="A",
        ground_truth="G",
        contexts=["  context one  ", "   ", "context two"],
    )
    sample = _to_ragas_sample(row)
    assert sample["contexts"] == ["context one", "context two"]
    assert sample["question"] == "Q"
    assert sample["answer"] == "A"
    assert sample["ground_truth"] == "G"


def test_coerce_score_filters_non_finite() -> None:
    assert _coerce_score("0.2") == pytest.approx(0.2)
    assert _coerce_score("nan") is None
    assert _coerce_score("inf") is None
