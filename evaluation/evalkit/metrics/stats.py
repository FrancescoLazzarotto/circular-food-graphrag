"""Bootstrap confidence intervals and grouped metric summaries."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, stdev
from typing import Any

from evalkit.models import GroupSummary


def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Linear-interpolation quantile of already-sorted values; 0 when empty."""
    if not sorted_values:
        return 0.0
    if quantile <= 0:
        return sorted_values[0]
    if quantile >= 1:
        return sorted_values[-1]
    position = quantile * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] + fraction * (sorted_values[high] - sorted_values[low])


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval over the mean.

    Args:
        scores: Observed values.
        n_bootstrap: Number of resamples.
        ci: Confidence level in (0, 1).
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds.
    """
    if not scores:
        return (0.0, 0.0)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    if len(scores) == 1:
        return (scores[0], scores[0])

    rng = random.Random(seed)
    n = len(scores)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()

    alpha = (1.0 - ci) / 2.0
    return (_percentile(means, alpha), _percentile(means, 1.0 - alpha))


def metric_summary(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Compute summary stats for a list of metric values.

    With no observations the statistics are ``None``, not ``0.0``: a printed
    "0.0" reads as "the system scored zero" when it means "never measured",
    and a JSON gold leaves the triple-level metrics unpopulated on every row.

    Args:
        values: Observed values.
        n_bootstrap: Bootstrap resamples.
        ci: Confidence level.
        seed: Bootstrap seed.

    Returns:
        Dict with mean, std, ci_lower, ci_upper, n; the four statistics are
        ``None`` when ``values`` is empty.
    """
    if not values:
        return {
            "mean": None,
            "std": None,
            "ci_lower": None,
            "ci_upper": None,
            "n": 0,
        }
    ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": len(values),
    }


def aggregate(
    row_metrics: list[dict[str, Any]],
    metric_names: list[str],
    group_keys: tuple[str, ...] = ("model_id", "framework", "strategy"),
    segment_key: str = "question_type",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> list[GroupSummary]:
    """Aggregate per-row metrics into grouped summaries.

    Args:
        row_metrics: List of dicts, one per row, containing group_keys + segment_key + metric values.
        metric_names: Names of metric fields to aggregate.
        group_keys: Fields used to group rows.
        segment_key: Optional second grouping dimension (e.g. question_type).
        n_bootstrap: Bootstrap resamples per metric.
        ci: Confidence level.
        seed: Bootstrap seed.

    Returns:
        List of GroupSummary (one global + one per segment value per group).
    """
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in row_metrics:
        key = tuple(str(row.get(k, "")) for k in group_keys)
        grouped[key].append(row)

    summaries: list[GroupSummary] = []
    for group_key_vals, items in sorted(grouped.items()):
        keys_dict = dict(zip(group_keys, group_key_vals))

        summaries.append(
            _summarize_group(
                items=items,
                keys={**keys_dict, "segment": "global"},
                metric_names=metric_names,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
        )

        # Per-segment breakdowns
        by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            segment = str(item.get(segment_key, "") or "").strip()
            if segment:
                by_segment[segment].append(item)

        for segment, seg_items in sorted(by_segment.items()):
            summaries.append(
                _summarize_group(
                    items=seg_items,
                    keys={**keys_dict, "segment": segment},
                    metric_names=metric_names,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    seed=seed,
                )
            )

    return summaries


def _summarize_group(
    items: list[dict[str, Any]],
    keys: dict[str, str],
    metric_names: list[str],
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> GroupSummary:
    """One :class:`GroupSummary` over `items`, a summary per metric."""
    metrics_dict: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        values = [float(item[name]) for item in items if item.get(name) is not None]
        metrics_dict[name] = metric_summary(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    return GroupSummary(keys=keys, n_rows=len(items), metrics=metrics_dict)
