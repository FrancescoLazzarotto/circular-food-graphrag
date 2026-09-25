"""Summarise a run's ``results.csv``: latency and retrieval counts per strategy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _load_rows(csv_path: Path) -> list[dict[str, object]]:
    """Read the columns this summary uses from ``results.csv``, typed.

    Args:
        csv_path: The run's ``results.csv``.

    Returns:
        One dict per row; missing numbers read as 0.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "strategy": row.get("strategy", ""),
                    "question": row.get("question", ""),
                    "latency_ms": float(row.get("latency_ms", "0") or 0.0),
                    "kg_triples_used": int(
                        float(row.get("kg_triples_used", "0") or 0.0)
                    ),
                    "kg_neighbors_used": int(
                        float(row.get("kg_neighbors_used", "0") or 0.0)
                    ),
                    "kg_subgraph_triples_used": int(
                        float(row.get("kg_subgraph_triples_used", "0") or 0.0)
                    ),
                    "kg_shortest_path_triples_used": int(
                        float(row.get("kg_shortest_path_triples_used", "0") or 0.0)
                    ),
                    "sub_questions": int(float(row.get("sub_questions", "0") or 0.0)),
                }
            )
    return rows


def _aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    """Mean and p95 latency, and mean retrieval counts, per strategy.

    Args:
        rows: As returned by `_load_rows`.

    Returns:
        Strategy -> summary statistics.
    """
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for strategy, items in grouped.items():
        latencies = [float(item["latency_ms"]) for item in items]
        avg_triples = mean(int(item["kg_triples_used"]) for item in items)
        avg_neighbors = mean(int(item["kg_neighbors_used"]) for item in items)
        avg_subgraph = mean(int(item["kg_subgraph_triples_used"]) for item in items)
        avg_shortest_path = mean(
            int(item["kg_shortest_path_triples_used"]) for item in items
        )
        avg_sub_questions = mean(int(item["sub_questions"]) for item in items)

        summary[strategy] = {
            "runs": len(items),
            "avg_latency_ms": mean(latencies),
            "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
            "avg_kg_triples_used": avg_triples,
            "avg_kg_neighbors_used": avg_neighbors,
            "avg_kg_subgraph_triples_used": avg_subgraph,
            "avg_kg_shortest_path_triples_used": avg_shortest_path,
            "avg_sub_questions": avg_sub_questions,
        }

    return summary


def _resolve_csv(input_path: Path) -> Path:
    """Find ``results.csv``, given directly or inside a run directory.

    Args:
        input_path: A CSV file or a run directory.

    Returns:
        The CSV path.

    Raises:
        FileNotFoundError: If neither form is found.
    """
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return input_path

    if input_path.is_dir():
        candidate = input_path / "results.csv"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Cannot find results.csv from input path: {input_path}")


def _print_table(summary: dict[str, dict[str, float | int]]) -> None:
    """Print the summary as a table, fastest strategy first."""
    if not summary:
        print("No rows found.")
        return

    ordered = sorted(summary.items(), key=lambda item: float(item[1]["avg_latency_ms"]))

    header = (
        f"{'strategy':<22} {'runs':>6} {'avg_ms':>10} {'p95_ms':>10} "
        f"{'avg_triples':>12} {'avg_neigh':>10} {'avg_subg':>10} {'avg_sp':>10} {'avg_subq':>10}"
    )
    print(header)
    print("-" * len(header))

    for strategy, stats in ordered:
        print(
            f"{strategy:<22}"
            f" {int(stats['runs']):>6}"
            f" {float(stats['avg_latency_ms']):>10.2f}"
            f" {float(stats['p95_latency_ms']):>10.2f}"
            f" {float(stats['avg_kg_triples_used']):>12.2f}"
            f" {float(stats['avg_kg_neighbors_used']):>10.2f}"
            f" {float(stats['avg_kg_subgraph_triples_used']):>10.2f}"
            f" {float(stats['avg_kg_shortest_path_triples_used']):>10.2f}"
            f" {float(stats['avg_sub_questions']):>10.2f}"
        )


def main() -> None:
    """Summarise one run on stdout, optionally saving the summary as JSON."""
    parser = argparse.ArgumentParser(
        description="Analyze GraphRAG experiment artifacts"
    )
    parser.add_argument("input", help="Path to experiment directory or results.csv")
    parser.add_argument(
        "--save-json", default="", help="Optional output path for aggregated JSON"
    )
    args = parser.parse_args()

    csv_path = _resolve_csv(Path(args.input))
    rows = _load_rows(csv_path)
    summary = _aggregate(rows)

    print(f"Loaded {len(rows)} rows from: {csv_path}")
    _print_table(summary)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nSaved JSON summary: {output_path}")


if __name__ == "__main__":
    main()
