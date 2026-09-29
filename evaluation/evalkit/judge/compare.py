"""Agreement between two judge runs over the rows both scored."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SUMMARY_FILE = "judge_summary.json"


def _row_key(entry: dict[str, Any]) -> str:
    """Identity of a judge row, from ``_key`` or its run, strategy and question."""
    return entry.get("_key") or f"{entry.get('run_dir','')}|{entry.get('strategy','')}|{entry.get('question','')}"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; ``None`` with fewer than two points or no variance."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _ranks(vals: list[float]) -> list[float]:
    """Average ranks of `vals`, ties sharing their mean rank."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman correlation, as Pearson over ranks."""
    if len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _load_summary(path: Path) -> dict[str, Any]:
    """A judge summary, from its file or from its run directory."""
    p = path / SUMMARY_FILE if path.is_dir() else path
    return json.loads(p.read_text(encoding="utf-8"))


def compare_judges(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, Any]:
    """Compare two judge runs (e.g. Haiku vs Sonnet) for inter-judge agreement.

    Matches rows by identity key, then per rubric reports each judge's mean and
    the agreement (Pearson, Spearman, mean absolute difference) over rows both
    judges scored.

    Args:
        summary_a, summary_b: judge_summary.json payloads.
        label_a, label_b: human-readable judge labels.

    Returns:
        Dict with per-rubric means + agreement metrics and the matched-row count.
    """
    rows_a = {_row_key(e): e for e in summary_a.get("row_scores", [])}
    rows_b = {_row_key(e): e for e in summary_b.get("row_scores", [])}
    shared = sorted(set(rows_a) & set(rows_b))

    rubric_names = sorted(set(summary_a.get("rubrics", {})) | set(summary_b.get("rubrics", {})))
    out: dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "n_rows_a": len(rows_a),
        "n_rows_b": len(rows_b),
        "n_matched": len(shared),
        "rubrics": {},
    }
    for rubric in rubric_names:
        xs: list[float] = []
        ys: list[float] = []
        for key in shared:
            va = rows_a[key].get(rubric)
            vb = rows_b[key].get(rubric)
            if va is None or vb is None:
                continue
            xs.append(float(va))
            ys.append(float(vb))
        mad = sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs) if xs else None
        out["rubrics"][rubric] = {
            f"mean_{label_a}": summary_a.get("rubrics", {}).get(rubric, {}).get("mean"),
            f"mean_{label_b}": summary_b.get("rubrics", {}).get(rubric, {}).get("mean"),
            "n_paired": len(xs),
            "pearson": _pearson(xs, ys),
            "spearman": _spearman(xs, ys),
            "mean_abs_diff": mad,
        }
    return out


def render_markdown(cmp: dict[str, Any]) -> str:
    """Render a compact agreement table from a compare_judges result."""
    a, b = cmp["label_a"], cmp["label_b"]
    lines = [
        f"# Judge agreement: {a} vs {b}",
        "",
        f"Matched rows: {cmp['n_matched']} ({a}={cmp['n_rows_a']}, {b}={cmp['n_rows_b']})",
        "",
        f"| rubric | mean {a} | mean {b} | n | Pearson | Spearman | mean|Δ| |",
        "|---|---|---|---|---|---|---|",
    ]

    def _fmt(v: Any) -> str:
        """Three decimals, or a dash when missing."""
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    for rubric, m in cmp["rubrics"].items():
        lines.append(
            f"| {rubric} | {_fmt(m.get(f'mean_{a}'))} | {_fmt(m.get(f'mean_{b}'))} | "
            f"{m['n_paired']} | {_fmt(m['pearson'])} | {_fmt(m['spearman'])} | {_fmt(m['mean_abs_diff'])} |"
        )
    return "\n".join(lines) + "\n"


def compare_from_paths(
    path_a: Path, path_b: Path, label_a: str = "a", label_b: str = "b"
) -> dict[str, Any]:
    """Load two judge runs from disk and compare them."""
    return compare_judges(_load_summary(path_a), _load_summary(path_b), label_a, label_b)
