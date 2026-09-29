"""Tests for the JSON and Markdown reports, regression checks and the judge join."""

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

from evalkit.models import (
    GroupSummary,
    KGQualityResult,
    RegressionResult,
    ReportModel,
)
from evalkit.report.json_report import write_json_report
from evalkit.report.markdown import render_experiment, render_project
from evalkit.report.regression import (
    compare_to_baseline,
    load_baseline,
    update_baseline,
)


def _make_group(strategy: str, metrics: dict) -> GroupSummary:
    """A global group for `strategy` with the given metric stats."""
    return GroupSummary(
        keys={"model_id": "m1", "framework": "graphrag", "strategy": strategy, "segment": "global"},
        n_rows=10,
        metrics=metrics,
    )


def _simple_stats(mean: float, n: int = 10) -> dict:
    """Metric stats around `mean`, with a fixed spread."""
    return {"mean": mean, "std": 0.05, "ci_lower": mean - 0.05, "ci_upper": mean + 0.05, "n": n}


def _make_report(scope: str = "experiment") -> ReportModel:
    """A two-strategy report of the given scope."""
    groups = [
        _make_group("default", {
            "entity_coverage": _simple_stats(0.7),
            "precision_at_k": _simple_stats(0.6),
            "recall_at_k": _simple_stats(0.5),
            "hit_at_k": _simple_stats(0.8),
            "ndcg_at_k": _simple_stats(0.65),
            "mrr": _simple_stats(0.55),
            "map": _simple_stats(0.5),
            "latency_ms": _simple_stats(1500.0),
            "token_f1": _simple_stats(0.6),
            "rouge_l": _simple_stats(0.5),
            "bleu": _simple_stats(0.3),
        }),
        _make_group("text_only", {
            "entity_coverage": _simple_stats(0.4),
            "precision_at_k": _simple_stats(0.4),
            "recall_at_k": _simple_stats(0.35),
            "hit_at_k": _simple_stats(0.6),
            "ndcg_at_k": _simple_stats(0.45),
            "mrr": _simple_stats(0.4),
            "map": _simple_stats(0.38),
            "latency_ms": _simple_stats(800.0),
            "token_f1": _simple_stats(0.45),
            "rouge_l": _simple_stats(0.4),
            "bleu": _simple_stats(0.2),
        }),
    ]
    kg = KGQualityResult(
        n_entities=1000, n_triples=5000, n_predicates=30,
        density=5.0, predicate_entropy=2.5, failed_chunks=10,
        failed_chunks_ratio=0.05,
    )
    regression = [
        RegressionResult("precision_at_k", 0.5, 0.6, 0.1, "improved"),
        RegressionResult("latency_ms", 1200.0, 1500.0, 300.0, "regressed"),
    ]
    return ReportModel(
        scope=scope,
        runs=["20260604_163532_test_run"],
        groups=groups,
        kg=kg,
        regression=regression,
        meta={"n_rows": 20, "run_stats": {
            "default": {"n_rows": 10, "latency_p50": 1400.0, "latency_p95": 1900.0,
                        "insufficient_rate": 0.05, "throughput_qps": 0.7}
        }},
    )


# ─── JSON report ─────────────────────────────────────────────────────────────

def test_write_json_report(tmp_path: Path) -> None:
    report = _make_report()
    out = tmp_path / "report.json"
    write_json_report(report, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["scope"] == "experiment"
    assert len(data["groups"]) == 2
    assert data["kg"]["n_entities"] == 1000


# ─── Markdown render ─────────────────────────────────────────────────────────

def test_render_experiment_has_sections() -> None:
    report = _make_report()
    md = render_experiment(report)
    assert "# Evaluation Report" in md
    assert "Retrieval Metrics" in md
    assert "Run Statistics" in md
    assert "Knowledge Graph Quality" in md
    assert "Regression vs Baseline" in md
    assert "improved" in md
    assert "regressed" in md


def test_render_experiment_contains_strategies() -> None:
    report = _make_report()
    md = render_experiment(report)
    assert "default" in md
    assert "text_only" in md


def test_render_project_has_sections() -> None:
    report = _make_report(scope="project")
    md = render_project(report)
    assert "# Project Evaluation Report" in md
    assert "Runs analysed" in md


# ─── Regression ──────────────────────────────────────────────────────────────

def test_compare_to_baseline_statuses() -> None:
    current = {
        "precision_at_k": 0.65,  # improved from 0.50
        "recall_at_k": 0.50,     # stable (same)
        "mrr": 0.30,             # regressed from 0.55
        "latency_ms": 900.0,     # improved (lower)
    }
    baseline = {
        "precision_at_k": 0.50,
        "recall_at_k": 0.50,
        "mrr": 0.55,
        "latency_ms": 1200.0,
    }
    results = {r.metric: r for r in compare_to_baseline(current, baseline, threshold=0.05)}

    assert results["precision_at_k"].status == "improved"
    assert results["recall_at_k"].status == "stable"
    assert results["mrr"].status == "regressed"
    assert results["latency_ms"].status == "improved"  # lower latency = improved


def test_load_baseline_flat_dict(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"precision_at_k": 0.5, "mrr": 0.4}))
    baseline = load_baseline(path)
    assert baseline["precision_at_k"] == pytest.approx(0.5)
    assert baseline["mrr"] == pytest.approx(0.4)


def test_load_baseline_missing_file(tmp_path: Path) -> None:
    baseline = load_baseline(tmp_path / "nonexistent.json")
    assert baseline == {}


def test_update_baseline_creates_and_merges(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    update_baseline(path, {"precision_at_k": 0.7})
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["precision_at_k"] == pytest.approx(0.7)

    # Merge: add new key + overwrite existing
    update_baseline(path, {"precision_at_k": 0.75, "mrr": 0.5})
    data = json.loads(path.read_text())
    assert data["precision_at_k"] == pytest.approx(0.75)
    assert data["mrr"] == pytest.approx(0.5)


def test_judge_scores_join_by_identity_with_skipped_rows(tmp_path: Path) -> None:
    """Judge skips rows with skip_reason: scores must land on the right rows.

    A positional-index merge would shift every judge score onto the wrong
    question whenever at least one row is skipped.
    """
    from evalkit.report.aggregate import build_experiment_report

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {  # skipped by the judge (empty contexts -> skip_reason=empty_context);
           # different strategy so a positional merge would leak the judge
           # score into the wrong aggregation group.
            "strategy": "text_only", "question": "Q-skipped", "answer": "a",
            "latency_ms": 1.0, "contexts": [],
            "metadata": {"framework": "graph_rag", "model_id": "m1", "run_index": 1},
        },
        {  # judged
            "strategy": "default", "question": "Q-judged", "answer": "b",
            "latency_ms": 1.0, "contexts": ["ctx"],
            "metadata": {"framework": "graph_rag", "model_id": "m1", "run_index": 1},
        },
    ]
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "question,ground_truth,question_type\n"
        "Q-skipped,x,fact_based\nQ-judged,y,fact_based\n",
        encoding="utf-8",
    )

    class FakeJudge:
        def score_dataset(self, rows, n_bootstrap=1000, ci=0.95, seed=42):
            scored = [r for r in rows if not r.skip_reason]
            assert [r.question for r in scored] == ["Q-judged"]
            return {
                "rows_evaluated": 1,
                "rows_skipped": 1,
                "rubrics": {},
                "row_scores": [{
                    "run_dir": scored[0].run_dir, "model_id": scored[0].model_id,
                    "framework": scored[0].framework, "strategy": scored[0].strategy,
                    "question": scored[0].question, "question_type": scored[0].question_type,
                    "skip_reason": "", "answer_correctness": 0.9,
                }],
            }

    report = build_experiment_report(run_dir, gold_path=gold, judge=FakeJudge())
    assert report.meta["n_rows"] == 2

    by_strategy = {
        g.keys["strategy"]: g
        for g in report.groups
        if g.keys.get("segment") == "global"
    }
    judged = by_strategy["default"].metrics["answer_correctness"]
    assert judged["n"] == 1 and judged["mean"] == pytest.approx(0.9)
    # The skipped row's group must not have inherited the score.
    assert by_strategy["text_only"].metrics["answer_correctness"]["n"] == 0
