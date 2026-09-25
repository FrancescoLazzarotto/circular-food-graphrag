"""Put the scored KG variants side by side, one row per retrieval strategy.

``score_gold_run.py`` scores one run. This reads several of its JSON outputs and
prints the difference between them, which is what a change to the graph has
to answer: does it change the answers, and in which direction, for which
strategy.

Both channels are reported because they answer different questions. The
retrieval channel says what the graph handed the generator; the answer channel
says what came out. A naming fix that raises retrieval and not the answer means
the generator was never the bottleneck for that question.

Usage::

    conda run -n graphllm python scripts/analysis/compare_kg_variants.py \\
        artifacts/evaluation/kgv2_v0_baseline.json \\
        artifacts/evaluation/kgv2_v2_names.json \\
        artifacts/evaluation/kgv2_v3_densified.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHANNELS = ("retrieval", "answer")


def pipelines(report: dict, channel: str, metric: str) -> dict[str, float | None]:
    """Read one metric per pipeline from a scored report.

    Args:
        report: JSON output of ``score_gold_run.py``.
        channel: ``"retrieval"`` or ``"answer"``.
        metric: ``block.field`` inside a ``by_pipeline`` entry, e.g.
            ``concept_micro.f1``.

    Returns:
        Pipeline name -> metric value, ``None`` where the report has none.
    """
    out: dict[str, float | None] = {}
    for entry in report[channel]["by_pipeline"]:
        name = entry["keys"]["pipeline"]
        block, field = metric.split(".")
        out[name] = entry[block][field]
    return out


def fmt(value: float | None) -> str:
    """Format a metric to three decimals, or a dash when it is missing."""
    return "  --  " if value is None else f"{value:6.3f}"


def delta(new: float | None, old: float | None) -> str:
    """Signed ``new - old`` to three decimals, blank when either is missing."""
    if new is None or old is None:
        return "     "
    diff = new - old
    sign = "+" if diff >= 0 else "-"
    return f"{sign}{abs(diff):5.3f}"


def main(argv: list[str] | None = None) -> int:
    """Print the comparison for each channel, optionally also as Markdown.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--metric", default="concept_micro.f1",
                        help="block.field inside a by_pipeline entry")
    parser.add_argument("--out", type=Path, default=None, help="also write this Markdown file")
    args = parser.parse_args(argv)

    reports = [(path.stem, json.loads(path.read_text(encoding="utf-8")))
               for path in args.reports]

    lines: list[str] = []
    for channel in CHANNELS:
        tables = [(name, pipelines(report, channel, args.metric)) for name, report in reports]
        names = sorted({p for _, table in tables for p in table})
        header = f"{'strategy':<20}" + "".join(f"{name[:16]:>18}" for name, _ in tables)
        if len(tables) > 1:
            header += f"{'Δ vs first':>12}"
        lines.append(f"\n### {channel} channel — {args.metric}\n")
        lines.append("```")
        lines.append(header)
        lines.append("-" * len(header))
        for pipeline in names:
            row = f"{pipeline:<20}" + "".join(f"{fmt(table.get(pipeline)):>18}" for _, table in tables)
            if len(tables) > 1:
                row += f"{delta(tables[-1][1].get(pipeline), tables[0][1].get(pipeline)):>12}"
            lines.append(row)
        lines.append("```")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("# KG variant comparison\n" + text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
