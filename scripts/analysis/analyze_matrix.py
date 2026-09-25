"""Aggregate the ``results.csv`` of several runs per model and strategy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _safe_float(value: str, default: float = 0.0) -> float:
    """`value` as a float, or `default` when it does not parse."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    """`value` as an int (via float), or `default` when unparseable."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_metadata(raw_value: str) -> dict[str, Any]:
    """The ``metadata_json`` cell as a dict, empty when absent or invalid."""
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _iter_result_files(root: Path, tag_contains: str) -> list[Path]:
    """Find the ``results.csv`` files to aggregate.

    Args:
        root: A ``results.csv``, a run directory, or a directory of runs.
        tag_contains: Keep only run directories whose name contains this.

    Returns:
        The CSV paths, sorted.

    Raises:
        FileNotFoundError: If `root` does not exist or holds no results.
    """
    if root.is_file() and root.name == "results.csv":
        return [root]

    if root.is_dir() and (root / "results.csv").exists():
        return [root / "results.csv"]

    if not root.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    files: list[Path] = []
    for candidate in sorted(root.glob("*/results.csv")):
        run_dir = candidate.parent.name
        if tag_contains and tag_contains not in run_dir:
            continue
        files.append(candidate)

    if not files:
        raise FileNotFoundError(f"No results.csv files found under: {root}")

    return files


def _load_rows(csv_files: list[Path]) -> list[dict[str, Any]]:
    """Read every row of `csv_files`, with the model id from the run metadata.

    Args:
        csv_files: The ``results.csv`` files.

    Returns:
        One typed dict per row.
    """
    rows: list[dict[str, Any]] = []

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                metadata = _load_metadata(row.get("metadata_json", ""))
                rows.append(
                    {
                        "run_dir": csv_path.parent.name,
                        "model_id": str(metadata.get("model_id", "unknown")),
                        "llm_enabled": bool(metadata.get("llm_enabled", False)),
                        "run_index": _safe_int(
                            str(metadata.get("run_index", "0")), default=0
                        ),
                        "strategy": row.get("strategy", ""),
                        "question": row.get("question", ""),
                        "latency_ms": _safe_float(row.get("latency_ms", "0")),
                        "kg_triples_used": _safe_int(
                            row.get("kg_triples_used", "0"), default=0
                        ),
                        "kg_neighbors_used": _safe_int(
                            row.get("kg_neighbors_used", "0"), default=0
                        ),
                        "kg_subgraph_triples_used": _safe_int(
                            row.get("kg_subgraph_triples_used", "0"), default=0
                        ),
                        "kg_shortest_path_triples_used": _safe_int(
                            row.get("kg_shortest_path_triples_used", "0"), default=0
                        ),
                        "sub_questions": _safe_int(
                            row.get("sub_questions", "0"), default=0
                        ),
                    }
                )

    return rows


def _p95(values: list[float]) -> float:
    """The 95th-percentile value of `values`, 0 when empty."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, int(0.95 * len(sorted_values)) - 1)
    return float(sorted_values[index])


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | str]]:
    """Latency and mean retrieval counts per (model, strategy).

    Args:
        rows: As returned by `_load_rows`.

    Returns:
        ``"model::strategy"`` -> summary statistics.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["strategy"]))].append(row)

    summary: dict[str, dict[str, float | int | str]] = {}
    for (model_id, strategy), items in grouped.items():
        latencies = [float(item["latency_ms"]) for item in items]
        avg_triples = mean(int(item["kg_triples_used"]) for item in items)
        avg_neighbors = mean(int(item["kg_neighbors_used"]) for item in items)
        avg_subgraph = mean(int(item["kg_subgraph_triples_used"]) for item in items)
        avg_shortest_path = mean(
            int(item["kg_shortest_path_triples_used"]) for item in items
        )
        avg_sub_questions = mean(int(item["sub_questions"]) for item in items)

        key = f"{model_id}::{strategy}"
        summary[key] = {
            "model_id": model_id,
            "strategy": strategy,
            "runs": len(items),
            "questions": len({str(item["question"]) for item in items}),
            "avg_latency_ms": mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "avg_kg_triples_used": avg_triples,
            "avg_kg_neighbors_used": avg_neighbors,
            "avg_kg_subgraph_triples_used": avg_subgraph,
            "avg_kg_shortest_path_triples_used": avg_shortest_path,
            "avg_sub_questions": avg_sub_questions,
        }

    return summary


def _print_table(summary: dict[str, dict[str, float | int | str]]) -> None:
    """Print the summary grouped by model, fastest strategy first."""
    if not summary:
        print("No rows found.")
        return

    ordered = sorted(
        summary.values(),
        key=lambda item: (str(item["model_id"]), float(item["avg_latency_ms"])),
    )

    header = (
        f"{'model_id':<38} {'strategy':<18} {'runs':>6} {'q':>4} {'avg_ms':>10} {'p95_ms':>10} "
        f"{'avg_triples':>12} {'avg_neigh':>10} {'avg_subg':>10} {'avg_sp':>10} {'avg_subq':>10}"
    )
    print(header)
    print("-" * len(header))

    for item in ordered:
        print(
            f"{str(item['model_id'])[:38]:<38}"
            f" {str(item['strategy'])[:18]:<18}"
            f" {int(item['runs']):>6}"
            f" {int(item['questions']):>4}"
            f" {float(item['avg_latency_ms']):>10.2f}"
            f" {float(item['p95_latency_ms']):>10.2f}"
            f" {float(item['avg_kg_triples_used']):>12.2f}"
            f" {float(item['avg_kg_neighbors_used']):>10.2f}"
            f" {float(item['avg_kg_subgraph_triples_used']):>10.2f}"
            f" {float(item['avg_kg_shortest_path_triples_used']):>10.2f}"
            f" {float(item['avg_sub_questions']):>10.2f}"
        )


def main() -> None:
    """Aggregate the runs under the input path; optionally save JSON and CSV."""
    parser = argparse.ArgumentParser(
        description="Aggregate GraphRAG experiment runs across multiple output folders"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="artifacts/experiments",
        help="Path to experiments root, a run folder, or a results.csv file",
    )
    parser.add_argument(
        "--tag-contains",
        default="",
        help="Optional substring filter on run directory name",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path for aggregated JSON output",
    )
    parser.add_argument(
        "--save-csv",
        default="",
        help="Optional path for aggregated CSV output",
    )
    args = parser.parse_args()

    try:
        csv_files = _iter_result_files(
            Path(args.input), tag_contains=args.tag_contains.strip()
        )
    except FileNotFoundError as exc:
        print(str(exc))
        print(
            "No completed run found yet. If an experiment is still running, retry after the first model finishes."
        )
        return

    rows = _load_rows(csv_files)
    summary = _aggregate(rows)

    print(f"Loaded {len(rows)} rows from {len(csv_files)} run file(s).")
    _print_table(summary)

    if args.save_json:
        output_json = Path(args.save_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nSaved JSON summary: {output_json}")

    if args.save_csv:
        output_csv = Path(args.save_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                [
                    "model_id",
                    "strategy",
                    "runs",
                    "questions",
                    "avg_latency_ms",
                    "p95_latency_ms",
                    "avg_kg_triples_used",
                    "avg_kg_neighbors_used",
                    "avg_kg_subgraph_triples_used",
                    "avg_kg_shortest_path_triples_used",
                    "avg_sub_questions",
                ]
            )
            for item in sorted(
                summary.values(), key=lambda x: (str(x["model_id"]), str(x["strategy"]))
            ):
                writer.writerow(
                    [
                        item["model_id"],
                        item["strategy"],
                        int(item["runs"]),
                        int(item["questions"]),
                        f"{float(item['avg_latency_ms']):.6f}",
                        f"{float(item['p95_latency_ms']):.6f}",
                        f"{float(item['avg_kg_triples_used']):.6f}",
                        f"{float(item['avg_kg_neighbors_used']):.6f}",
                        f"{float(item['avg_kg_subgraph_triples_used']):.6f}",
                        f"{float(item['avg_kg_shortest_path_triples_used']):.6f}",
                        f"{float(item['avg_sub_questions']):.6f}",
                    ]
                )
        print(f"Saved CSV summary: {output_csv}")


if __name__ == "__main__":
    main()
