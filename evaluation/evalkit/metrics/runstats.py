"""Per-strategy run statistics: latency percentiles, insufficiency, resources."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evalkit.io.run_loader import load_resource_summary, load_run_summary
from evalkit.models import EvalRow


def _percentile(values: list[float], q: float) -> float:
    """q-th quantile (linear interpolation)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def compute_run_stats(
    rows: list[EvalRow],
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute per-strategy run statistics from a list of EvalRow.

    Args:
        rows: All rows from one run (may span multiple strategies).
        run_dir: Optional run directory to also load resource_summary.json.

    Returns:
        Dict[strategy, stats_dict] with per-strategy breakdown plus a
        "global" entry aggregating all rows.
    """
    by_strategy: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_strategy[row.strategy].append(row)

    result: dict[str, Any] = {}

    for strategy, strat_rows in sorted(by_strategy.items()):
        result[strategy] = _stats_for_group(strat_rows)

    if rows:
        result["global"] = _stats_for_group(rows)

    if run_dir:
        resource = load_resource_summary(run_dir)
        if resource:
            result["_resource_summary"] = resource

    return result


def _stats_for_group(rows: list[EvalRow]) -> dict[str, Any]:
    """Latency, insufficiency and context statistics of one group of rows."""
    if not rows:
        return {}

    latencies = [r.latency_ms for r in rows]
    n = len(rows)
    insufficient = sum(1 for r in rows if r.insufficient)
    total_ms = sum(latencies)

    stats: dict[str, Any] = {
        "n_rows": n,
        "latency_mean": statistics.mean(latencies),
        "latency_std": statistics.stdev(latencies) if n > 1 else 0.0,
        "latency_p50": _percentile(latencies, 0.50),
        "latency_p95": _percentile(latencies, 0.95),
        "latency_p99": _percentile(latencies, 0.99),
        "latency_min": min(latencies),
        "latency_max": max(latencies),
        "throughput_qps": (n / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        "insufficient_count": insufficient,
        "insufficient_rate": insufficient / n,
        "avg_kg_triples_used": statistics.mean(r.kg_triples_used for r in rows),
        "avg_kg_neighbors_used": statistics.mean(r.kg_neighbors_used for r in rows),
        "avg_kg_subgraph_triples_used": statistics.mean(r.kg_subgraph_triples_used for r in rows),
        "avg_kg_shortest_path_triples_used": statistics.mean(r.kg_shortest_path_triples_used for r in rows),
        "avg_sub_questions": statistics.mean(r.sub_questions for r in rows),
        "avg_contexts_count": statistics.mean(len(r.contexts) for r in rows),
        "avg_context_chars": statistics.mean(
            sum(len(c) for c in r.contexts) for r in rows
        ),
    }
    return stats


def compute_run_stats_from_summary(run_dir: Path) -> dict[str, Any]:
    """Load and enrich stats directly from summary.json (no EvalRow needed).

    Useful for project-level trend analysis where gold labels are not available.
    """
    summary = load_run_summary(run_dir)
    resource = load_resource_summary(run_dir)
    result: dict[str, Any] = {
        "run_dir": run_dir.name,
        "timestamp": summary.get("timestamp", ""),
        "tag": summary.get("tag", ""),
        "model_id": summary.get("llm", {}).get("model_id", ""),
        "questions_count": summary.get("questions_count", 0),
        "strategies": summary.get("strategies", []),
        "agent_pipeline": summary.get("agent_pipeline", {}),
        "stats": summary.get("stats", {}),
    }
    if resource:
        result["resource_summary"] = resource
    return result
