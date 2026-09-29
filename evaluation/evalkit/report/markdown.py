"""Markdown rendering of experiment and project reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalkit.models import GroupSummary, KGQualityResult, RegressionResult, ReportModel


def _fmt(value: Any, decimals: int = 3) -> str:
    """`value` to `decimals` places; a dash when None, as-is when not numeric."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _metric_table(
    groups: list[GroupSummary],
    metric_names: list[str],
    segment: str = "global",
) -> str:
    """Render a Markdown table for one segment (default: global)."""
    filtered = [g for g in groups if g.keys.get("segment", "global") == segment]
    if not filtered:
        return "_No data._\n"

    # Header
    short_names = [m.replace("_at_k", "@k").replace("_", "-") for m in metric_names]
    header_cols = ["model", "strategy"] + short_names
    header = "| " + " | ".join(header_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(header_cols)) + " |"
    lines = [header, sep]

    for g in filtered:
        model = g.keys.get("model_id", "")
        strategy = g.keys.get("strategy", "")
        cells = [model, strategy]
        for m in metric_names:
            stats = g.metrics.get(m)
            if stats and stats.get("n", 0) > 0:
                mean_val = _fmt(stats["mean"])
                ci_lower = _fmt(stats.get("ci_lower"), 3)
                ci_upper = _fmt(stats.get("ci_upper"), 3)
                cells.append(f"{mean_val} [{ci_lower}–{ci_upper}]")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _run_stats_section(run_stats: dict[str, Any]) -> str:
    """Markdown table of the run statistics per strategy."""
    lines = ["| strategy | n | latency p50 (ms) | latency p95 (ms) | insufficiency | throughput (q/s) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for strat, stats in sorted(run_stats.items()):
        if strat.startswith("_") or not isinstance(stats, dict):
            continue
        lines.append(
            f"| {strat} | {stats.get('n_rows', '—')} "
            f"| {_fmt(stats.get('latency_p50'))} "
            f"| {_fmt(stats.get('latency_p95'))} "
            f"| {_fmt(stats.get('insufficient_rate'), 2)} "
            f"| {_fmt(stats.get('throughput_qps'))} |"
        )
    return "\n".join(lines) + "\n"


def _kg_section(kg: KGQualityResult) -> str:
    """Markdown list of the KG quality measures."""
    lines = [
        f"- **Entities**: {kg.n_entities:,}",
        f"- **Triples**: {kg.n_triples:,}",
        f"- **Predicates**: {kg.n_predicates}",
        f"- **Documents**: {kg.n_documents}",
        f"- **Density** (triples/entity): {_fmt(kg.density)}",
        f"- **Avg degree**: {_fmt(kg.avg_degree)}",
        f"- **Median degree**: {_fmt(kg.median_degree)}",
        f"- **Isolated node ratio**: {_fmt(kg.isolated_ratio)}",
        f"- **Predicate entropy**: {_fmt(kg.predicate_entropy)}",
        f"- **Failed chunks**: {kg.failed_chunks} ({_fmt(kg.failed_chunks_ratio * 100, 1)}%)",
        f"- **Entity resolution collapse**: {_fmt(kg.resolution_collapse_ratio * 100, 1)}%",
    ]
    if kg.entity_gold_coverage is not None:
        lines.append(f"- **Gold entity coverage**: {_fmt(kg.entity_gold_coverage * 100, 1)}%")
    return "\n".join(lines) + "\n"


def _regression_section(regression: list[RegressionResult]) -> str:
    """Markdown table of the regression checks."""
    lines = [
        "| metric | baseline | current | delta | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    status_emoji = {"improved": "✅", "stable": "➖", "regressed": "⚠️"}
    for r in sorted(regression, key=lambda x: x.status):
        emoji = status_emoji.get(r.status, "")
        lines.append(
            f"| {r.metric} | {_fmt(r.baseline)} | {_fmt(r.current)} "
            f"| {_fmt(r.delta, 4)} | {emoji} {r.status} |"
        )
    return "\n".join(lines) + "\n"


RETRIEVAL_REPORT_METRICS = [
    "entity_coverage", "precision_at_k", "recall_at_k",
    "hit_at_k", "ndcg_at_k", "mrr", "map",
]
TEXT_REPORT_METRICS = ["exact_match", "token_f1", "rouge_l", "bleu"]
# Canonical rubrics first, then `answer_correctness` so artifacts scored under
# that legacy rubric, which folded completeness into accuracy, still render.
# `abstention` only carries a value on distractor rows; it stays blank elsewhere.
JUDGE_REPORT_METRICS = [
    "factual_correctness",
    "completeness",
    "groundedness",
    "relevance",
    "abstention",
    "answer_correctness",
]


def render_experiment(report: ReportModel, plots_dir: Path | None = None) -> str:
    """Render a Markdown report for a single experiment run."""
    run_name = report.runs[0] if report.runs else "unknown"
    lines: list[str] = [
        f"# Evaluation Report — `{run_name}`",
        "",
        f"**Scope**: {report.scope}  ",
        f"**Rows**: {report.meta.get('n_rows', '?')}  ",
        "",
    ]

    # ── Retrieval Metrics ──
    lines += [
        "## Retrieval Metrics (global)",
        "",
        _metric_table(report.groups, RETRIEVAL_REPORT_METRICS, segment="global"),
    ]

    # ── Per question-type breakdown ──
    segments = {
        g.keys["segment"]
        for g in report.groups
        if g.keys.get("segment", "global") != "global"
    }
    if segments:
        lines += ["### By Question Type", ""]
        for seg in sorted(segments):
            lines += [f"#### {seg}", "", _metric_table(report.groups, RETRIEVAL_REPORT_METRICS, segment=seg)]

    # ── Text Metrics ──
    if any(
        g.metrics.get("token_f1", {}).get("n", 0) > 0
        for g in report.groups
        if g.keys.get("segment") == "global"
    ):
        lines += [
            "## Text Similarity Metrics",
            "",
            _metric_table(report.groups, TEXT_REPORT_METRICS, segment="global"),
        ]

    # ── Judge Metrics ──
    judge_summary = report.meta.get("judge_summary", {})
    if judge_summary:
        lines += ["## LLM-as-a-Judge Metrics", ""]
        lines += [
            "| rubric | mean | std | CI 95% |",
            "| --- | --- | --- | --- |",
        ]
        for rubric, stats in sorted(judge_summary.items()):
            lines.append(
                f"| {rubric} | {_fmt(stats.get('mean'))} "
                f"| {_fmt(stats.get('std'))} "
                f"| [{_fmt(stats.get('ci_lower'))}–{_fmt(stats.get('ci_upper'))}] |"
            )
        lines.append("")

    # ── Run Stats ──
    run_stats = report.meta.get("run_stats", {})
    if run_stats:
        lines += ["## Run Statistics", "", _run_stats_section(run_stats)]

    # ── KG Quality ──
    if report.kg:
        lines += ["## Knowledge Graph Quality", "", _kg_section(report.kg)]

    # ── Regression ──
    if report.regression:
        lines += ["## Regression vs Baseline", "", _regression_section(report.regression)]

    # ── Plots ──
    if plots_dir:
        plot_files = sorted(plots_dir.glob("*.png"))
        if plot_files:
            lines += ["## Plots", ""]
            for pf in plot_files:
                rel = pf.name
                lines.append(f"![{pf.stem}](plots/{rel})")
            lines.append("")

    return "\n".join(lines)


def render_project(report: ReportModel, plots_dir: Path | None = None) -> str:
    """Render a Markdown report aggregating all runs in a project."""
    lines: list[str] = [
        "# Project Evaluation Report",
        "",
        f"**Runs analysed**: {len(report.runs)}  ",
        f"**Total rows**: {report.meta.get('n_rows_total', '?')}  ",
        "",
        "## Runs",
        "",
    ]
    for run in report.runs:
        lines.append(f"- `{run}`")
    lines.append("")

    # Aggregate retrieval table
    lines += [
        "## Retrieval Metrics (global, all runs)",
        "",
        _metric_table(report.groups, RETRIEVAL_REPORT_METRICS, segment="global"),
    ]

    # Text metrics
    if any(
        g.metrics.get("token_f1", {}).get("n", 0) > 0
        for g in report.groups
        if g.keys.get("segment") == "global"
    ):
        lines += [
            "## Text Similarity Metrics",
            "",
            _metric_table(report.groups, TEXT_REPORT_METRICS, segment="global"),
        ]

    # KG
    if report.kg:
        lines += ["## Knowledge Graph Quality", "", _kg_section(report.kg)]

    # Regression
    if report.regression:
        lines += ["## Regression vs Baseline", "", _regression_section(report.regression)]

    # Trend
    latency_trend = report.meta.get("latency_trend", [])
    if latency_trend:
        lines += ["## Latency Trend (avg across strategies)", ""]
        lines += ["| run | timestamp | avg_latency_ms |", "| --- | --- | --- |"]
        for entry in latency_trend:
            lines.append(
                f"| `{entry['run_dir']}` | {entry['timestamp']} | {_fmt(entry['value'])} |"
            )
        lines.append("")

    # Plots
    if plots_dir:
        plot_files = sorted(plots_dir.glob("*.png"))
        if plot_files:
            lines += ["## Plots", ""]
            for pf in plot_files:
                lines.append(f"![{pf.stem}](plots/{pf.name})")
            lines.append("")

    return "\n".join(lines)


def write_markdown(report: ReportModel, output_path: Path, plots_dir: Path | None = None) -> None:
    """Write Markdown report to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report.scope == "project":
        content = render_project(report, plots_dir=plots_dir)
    else:
        content = render_experiment(report, plots_dir=plots_dir)
    output_path.write_text(content, encoding="utf-8")
