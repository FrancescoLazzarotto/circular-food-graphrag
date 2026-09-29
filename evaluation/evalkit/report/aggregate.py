"""Experiment and project reports: metrics, judge, run stats, KG, regression."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evalkit.config import EvalConfig
from evalkit.io.dataset import build_dataset, rows_from_csv
from evalkit.kg.kg_quality import compute_from_artifacts
from evalkit.metrics.retrieval import RETRIEVAL_METRICS, compute_retrieval_row
from evalkit.metrics.runstats import compute_run_stats, compute_run_stats_from_summary
from evalkit.metrics.stats import aggregate
from evalkit.metrics.text import TEXT_METRICS, compute_text_row
from evalkit.models import EvalRow, GroupSummary, KGQualityResult, ReportModel
from evalkit.report.regression import compare_to_baseline, load_baseline, load_cross_run_trend

logger = logging.getLogger("graphrag")


def build_experiment_report(
    run_dir: Path,
    gold_path: Path | None = None,
    eval_dataset_csv: Path | None = None,
    kg_artifacts_dir: Path | None = None,
    config: EvalConfig | None = None,
    judge: Any = None,
) -> ReportModel:
    """Build a ReportModel for a single experiment run.

    Args:
        run_dir: Path to the experiment run directory.
        gold_path: Optional gold CSV for join.
        eval_dataset_csv: Optional pre-built eval dataset CSV (skips join step).
        kg_artifacts_dir: Optional KG pipeline run directory for KG quality metrics.
        config: EvalConfig with metric parameters. Defaults used if None.
        judge: Optional LLMJudge instance; skipped if None.

    Returns:
        ReportModel with groups, optional KG quality, optional regression.
    """
    cfg = config or EvalConfig()

    # ── Load / build eval rows ─────────────────────────────────────────────
    if eval_dataset_csv and eval_dataset_csv.exists():
        rows = rows_from_csv(eval_dataset_csv)
        logger.info("Loaded eval dataset from CSV: %s (%d rows)", eval_dataset_csv, len(rows))
    else:
        rows = build_dataset(
            run_dirs=[run_dir],
            gold_path=gold_path,
        )
        logger.info("Built eval dataset: %d rows", len(rows))

    # ── Deterministic metrics ──────────────────────────────────────────────
    retrieval_rows = [compute_retrieval_row(row, k=cfg.k) for row in rows]
    text_rows = [compute_text_row(row, bertscore=cfg.bertscore) for row in rows]

    # Merge per-row metrics
    merged: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        entry: dict[str, Any] = {**retrieval_rows[i]}
        for metric in TEXT_METRICS:
            entry[metric] = text_rows[i].get(metric)
        merged.append(entry)

    # ── LLM judge (optional) ──────────────────────────────────────────────
    judge_result: dict[str, Any] = {}
    if judge is not None:
        judge_result = judge.score_dataset(rows, n_bootstrap=cfg.n_bootstrap, ci=cfg.ci, seed=cfg.seed)
        # Join judge scores by row identity, never by position: the judge skips
        # rows with a skip_reason, so row_scores can be shorter than merged and
        # positional indexing would attach scores to the wrong questions.
        def _judge_join_key(d: dict[str, Any]) -> tuple[str, str, str, str]:
            """Row identity: run, model, strategy, question."""
            return (
                str(d.get("run_dir", "")),
                str(d.get("model_id", "")),
                str(d.get("strategy", "")),
                str(d.get("question", "")),
            )

        scores_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for score_entry in judge_result.get("row_scores", []):
            scores_by_key.setdefault(_judge_join_key(score_entry), []).append(score_entry)

        for entry in merged:
            bucket = scores_by_key.get(_judge_join_key(entry))
            if not bucket:
                continue
            score_entry = bucket.pop(0)
            for k, v in score_entry.items():
                if k not in entry:
                    entry[k] = v

    all_metric_names = RETRIEVAL_METRICS + TEXT_METRICS
    if judge is not None:
        from evalkit.judge.rubrics import RUBRICS
        all_metric_names += list(RUBRICS.keys())

    # ── Aggregate by (model_id, framework, strategy) + question_type ───────
    groups = aggregate(
        row_metrics=merged,
        metric_names=all_metric_names,
        group_keys=("model_id", "framework", "strategy"),
        segment_key="question_type",
        n_bootstrap=cfg.n_bootstrap,
        ci=cfg.ci,
        seed=cfg.seed,
    )

    # ── Run stats (no gold needed) ─────────────────────────────────────────
    run_stats = compute_run_stats(rows, run_dir=run_dir)

    # ── KG quality (optional) ─────────────────────────────────────────────
    kg_result: KGQualityResult | None = None
    if kg_artifacts_dir and kg_artifacts_dir.is_dir():
        try:
            kg_result = compute_from_artifacts(kg_artifacts_dir)
        except Exception as exc:
            logger.warning("KG quality computation failed: %s", exc)

    # ── Regression vs baseline ─────────────────────────────────────────────
    regression = None
    if cfg.baselines_path.exists():
        baseline = load_baseline(cfg.baselines_path)
        if baseline:
            # Use global segment of first group for current values
            global_groups = [g for g in groups if g.keys.get("segment") == "global"]
            if global_groups:
                current: dict[str, float] = {}
                for name in all_metric_names:
                    stats = global_groups[0].metrics.get(name)
                    if stats and stats.get("n", 0) > 0:
                        current[name] = float(stats["mean"])
                current["latency_ms"] = run_stats.get("global", {}).get("latency_mean", 0.0)
                regression = compare_to_baseline(current, baseline, threshold=cfg.regression_threshold)

    run_summary = compute_run_stats_from_summary(run_dir)
    meta: dict[str, Any] = {
        "run_dir": run_dir.name,
        "run_summary": run_summary,
        "run_stats": run_stats,
        "judge_summary": judge_result.get("rubrics", {}),
        "n_rows": len(rows),
    }

    return ReportModel(
        scope="experiment",
        runs=[run_dir.name],
        groups=groups,
        kg=kg_result,
        regression=regression,
        meta=meta,
    )


def build_project_report(
    experiments_root: Path,
    gold_path: Path | None = None,
    kg_artifacts_dir: Path | None = None,
    config: EvalConfig | None = None,
    tag_contains: str = "",
) -> ReportModel:
    """Build a project-level ReportModel across all runs under experiments_root.

    Does not run LLM judge (too expensive for project-level); uses summary.json
    for run stats and trend analysis.

    Args:
        experiments_root: Parent of all experiment run directories.
        gold_path: Optional gold CSV for individual-run joins.
        kg_artifacts_dir: Optional KG pipeline run dir for KG metrics.
        config: EvalConfig.
        tag_contains: Filter string for run directory names.

    Returns:
        ReportModel with scope="project", multi-run groups, trends.
    """
    cfg = config or EvalConfig()

    run_dirs = [
        child
        for child in sorted(experiments_root.iterdir())
        if child.is_dir()
        and (not tag_contains or tag_contains in child.name)
        and ((child / "results.jsonl").exists() or (child / "results.csv").exists())
    ]

    if not run_dirs:
        logger.warning("No run directories found under: %s", experiments_root)

    all_rows: list[EvalRow] = []
    all_run_summaries: list[dict[str, Any]] = []
    run_names: list[str] = []

    for run_dir in run_dirs:
        run_names.append(run_dir.name)
        try:
            rows = build_dataset([run_dir], gold_path=gold_path)
            all_rows.extend(rows)
        except Exception as exc:
            logger.warning("Failed to load rows from %s: %s", run_dir.name, exc)

        all_run_summaries.append(compute_run_stats_from_summary(run_dir))

    # Aggregate all rows
    merged: list[dict[str, Any]] = []
    for row in all_rows:
        entry = compute_retrieval_row(row, k=cfg.k)
        text = compute_text_row(row)
        for metric in TEXT_METRICS:
            entry[metric] = text.get(metric)
        # Tag with run_dir for cross-run breakdown
        entry["run_dir"] = row.run_dir
        merged.append(entry)

    all_metrics = RETRIEVAL_METRICS + TEXT_METRICS
    groups = aggregate(
        row_metrics=merged,
        metric_names=all_metrics,
        group_keys=("model_id", "framework", "strategy"),
        segment_key="question_type",
        n_bootstrap=cfg.n_bootstrap,
        ci=cfg.ci,
        seed=cfg.seed,
    )

    # Latency trend — scoped to the same filtered run_dirs (not whole experiments_root)
    latency_trend = load_cross_run_trend(
        experiments_root, metric_name="avg_latency_ms", strategy="global",
        run_dirs=run_dirs,
    )

    kg_result: KGQualityResult | None = None
    if kg_artifacts_dir and kg_artifacts_dir.is_dir():
        try:
            kg_result = compute_from_artifacts(kg_artifacts_dir)
        except Exception as exc:
            logger.warning("KG quality computation failed: %s", exc)

    regression = None
    if cfg.baselines_path.exists():
        baseline = load_baseline(cfg.baselines_path)
        if baseline and groups:
            global_groups = [g for g in groups if g.keys.get("segment") == "global"]
            if global_groups:
                current: dict[str, float] = {}
                for name in all_metrics:
                    stats = global_groups[0].metrics.get(name)
                    if stats and stats.get("n", 0) > 0:
                        current[name] = float(stats["mean"])
                regression = compare_to_baseline(current, baseline, threshold=cfg.regression_threshold)

    meta: dict[str, Any] = {
        "n_runs": len(run_dirs),
        "n_rows_total": len(all_rows),
        "run_summaries": all_run_summaries,
        "latency_trend": latency_trend,
    }

    return ReportModel(
        scope="project",
        runs=run_names,
        groups=groups,
        kg=kg_result,
        regression=regression,
        meta=meta,
    )
