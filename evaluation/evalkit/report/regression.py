"""Metric baselines, regression checks against them, and cross-run trends."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evalkit.models import RegressionResult

logger = logging.getLogger("graphrag")

DEFAULT_THRESHOLD = 0.05  # 5% relative degradation triggers "regressed"


def load_baseline(baseline_path: Path) -> dict[str, float]:
    """Load baseline metrics from a JSON file.

    Expected format: flat dict of metric_name → float value, or
    list of dicts with {"metric": "...", "value": ...} entries.

    Returns:
        Dict[metric_name, float]. Empty dict if file not found or invalid.
    """
    if not baseline_path.exists():
        logger.warning("Baseline file not found: %s", baseline_path)
        return {}

    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load baseline %s: %s", baseline_path, exc)
        return {}

    if isinstance(data, dict):
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}

    if isinstance(data, list):
        result: dict[str, float] = {}
        for entry in data:
            if isinstance(entry, dict):
                name = entry.get("metric") or entry.get("name", "")
                value = entry.get("value") or entry.get("mean")
                if name and isinstance(value, (int, float)):
                    result[str(name)] = float(value)
        return result

    return {}


def compare_to_baseline(
    current: dict[str, float],
    baseline: dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
    higher_is_better: set[str] | None = None,
) -> list[RegressionResult]:
    """Compare current metrics to baseline.

    Args:
        current: Dict[metric_name, current_value].
        baseline: Dict[metric_name, baseline_value] loaded from file.
        threshold: Relative change threshold for "regressed" vs "stable".
        higher_is_better: Set of metric names where higher = better.
            Defaults to all IR and text metrics (everything except latency).

    Returns:
        List of RegressionResult for metrics present in both.
    """
    if higher_is_better is None:
        # Latency is lower-is-better; everything else is higher-is-better
        higher_is_better = {
            m for m in set(current) | set(baseline)
            if "latency" not in m
        }

    results: list[RegressionResult] = []
    for metric, baseline_val in sorted(baseline.items()):
        current_val = current.get(metric)
        if current_val is None:
            continue

        delta = current_val - baseline_val
        if baseline_val != 0:
            relative_change = delta / abs(baseline_val)
        else:
            relative_change = 0.0

        is_higher_better = metric in higher_is_better
        if is_higher_better:
            if relative_change < -threshold:
                status = "regressed"
            elif relative_change > threshold:
                status = "improved"
            else:
                status = "stable"
        else:
            # Lower is better (latency)
            if relative_change > threshold:
                status = "regressed"
            elif relative_change < -threshold:
                status = "improved"
            else:
                status = "stable"

        results.append(
            RegressionResult(
                metric=metric,
                baseline=baseline_val,
                current=current_val,
                delta=delta,
                status=status,
            )
        )

    return results


def update_baseline(
    baseline_path: Path,
    new_values: dict[str, float],
) -> None:
    """Write new_values to baseline_path as a flat JSON dict.

    Merges with existing baseline if present; new values overwrite old ones.
    """
    existing = load_baseline(baseline_path)
    existing.update(new_values)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Baseline updated at %s (%d metrics)", baseline_path, len(existing))


def load_cross_run_trend(
    experiments_root: Path,
    metric_name: str,
    strategy: str = "global",
    run_dirs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Load a metric's value across all runs for trend plotting.

    Reads summary.json files from experiment run directories.
    Falls back to summary stats if available.

    Args:
        experiments_root: Parent directory of all run dirs.
        metric_name: Metric key to extract (e.g. "avg_latency_ms").
        strategy: Strategy key in stats dict (or "global" for overall).
        run_dirs: Explicit list of run dirs to use. If None, iterates experiments_root.

    Returns:
        List of {"run_dir", "timestamp", "value"} sorted by timestamp.
    """
    if run_dirs is not None:
        dirs = sorted(run_dirs)
    elif experiments_root.is_dir():
        dirs = sorted(d for d in experiments_root.iterdir() if d.is_dir())
    else:
        return []

    trend: list[dict[str, Any]] = []
    for run_dir in dirs:
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        timestamp = summary.get("timestamp", run_dir.name)
        stats = summary.get("stats", {})
        value: Any = None

        if strategy == "global":
            # Average across all strategies
            values = [
                s.get(metric_name)
                for s in stats.values()
                if isinstance(s, dict) and metric_name in s
            ]
            numeric = [float(v) for v in values if v is not None]
            if numeric:
                value = sum(numeric) / len(numeric)
        else:
            strat_stats = stats.get(strategy, {})
            value = strat_stats.get(metric_name)

        if value is not None:
            trend.append({
                "run_dir": run_dir.name,
                "timestamp": timestamp,
                "value": float(value),
            })

    return trend
