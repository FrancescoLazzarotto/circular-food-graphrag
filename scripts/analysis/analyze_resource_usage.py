"""Aggregate the memory, CPU and GPU peaks recorded by several runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _safe_float(value: Any) -> float | None:
    """`value` as a float, or ``None`` when missing or unparseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _load_resource_summary(path: Path) -> dict[str, Any] | None:
    """The parsed summary file, or ``None`` when it cannot be read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_resource_summaries(root: Path, tag_contains: str = "") -> list[Path]:
    """Find the ``resource_summary.json`` of every run directory under `root`.

    Args:
        root: Directory of run folders.
        tag_contains: Keep only run folders whose name contains this.

    Returns:
        The summary paths, sorted by run folder.

    Raises:
        FileNotFoundError: If `root` is not a directory.
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    summaries: list[Path] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if tag_contains and tag_contains not in run_dir.name:
            continue
        summary_path = run_dir / "resource_summary.json"
        if summary_path.exists():
            summaries.append(summary_path)

    return summaries


def _extract_gpu_peaks(summary: dict[str, Any]) -> tuple[float | None, float | None]:
    """Highest memory (MB) and utilisation (%) across the run's GPUs."""
    gpus = summary.get("gpus", [])
    if not isinstance(gpus, list):
        return None, None

    mem_peaks: list[float] = []
    util_peaks: list[float] = []

    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        mem_peak = _safe_float(gpu.get("peak_memory_used_mb"))
        util_peak = _safe_float(gpu.get("peak_utilization_gpu_percent"))
        if mem_peak is not None:
            mem_peaks.append(mem_peak)
        if util_peak is not None:
            util_peaks.append(util_peak)

    max_mem = max(mem_peaks) if mem_peaks else None
    max_util = max(util_peaks) if util_peaks else None
    return max_mem, max_util


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and maximum of each peak per (model, tag).

    Args:
        rows: One dict of peaks per run.

    Returns:
        One summary dict per (model, tag), sorted; ``None`` where no run
        reported the value.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["model_id"], row["tag"])
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (model_id, tag), items in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):

        def values(field: str) -> list[float]:
            """The parseable values of `field` across the group's runs."""
            collected: list[float] = []
            for item in items:
                value = _safe_float(item.get(field))
                if value is not None:
                    collected.append(value)
            return collected

        duration_values = values("monitoring_duration_sec")
        process_rss_values = values("peak_process_rss_mb")
        system_ram_pct_values = values("peak_system_ram_percent")
        process_cpu_values = values("peak_process_cpu_percent")
        gpu_mem_values = values("peak_gpu_memory_used_mb")
        gpu_util_values = values("peak_gpu_util_percent")

        output.append(
            {
                "model_id": model_id,
                "tag": tag,
                "runs": len(items),
                "avg_duration_sec": mean(duration_values) if duration_values else None,
                "max_duration_sec": max(duration_values) if duration_values else None,
                "avg_peak_process_rss_mb": mean(process_rss_values)
                if process_rss_values
                else None,
                "max_peak_process_rss_mb": max(process_rss_values)
                if process_rss_values
                else None,
                "avg_peak_system_ram_percent": mean(system_ram_pct_values)
                if system_ram_pct_values
                else None,
                "max_peak_system_ram_percent": max(system_ram_pct_values)
                if system_ram_pct_values
                else None,
                "avg_peak_process_cpu_percent": mean(process_cpu_values)
                if process_cpu_values
                else None,
                "max_peak_process_cpu_percent": max(process_cpu_values)
                if process_cpu_values
                else None,
                "avg_peak_gpu_memory_used_mb": mean(gpu_mem_values)
                if gpu_mem_values
                else None,
                "max_peak_gpu_memory_used_mb": max(gpu_mem_values)
                if gpu_mem_values
                else None,
                "avg_peak_gpu_util_percent": mean(gpu_util_values)
                if gpu_util_values
                else None,
                "max_peak_gpu_util_percent": max(gpu_util_values)
                if gpu_util_values
                else None,
            }
        )

    return output


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print the maximum peaks per (model, tag)."""
    if not rows:
        print("No resource summaries found.")
        return

    header = (
        f"{'model_id':<38} {'tag':<34} {'runs':>4} {'max_rss_mb':>12} {'max_sys_ram%':>12} "
        f"{'max_gpu_mem':>12} {'max_gpu_util%':>13}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        max_rss = _safe_float(row.get("max_peak_process_rss_mb"))
        max_sys_ram = _safe_float(row.get("max_peak_system_ram_percent"))
        max_gpu_mem = _safe_float(row.get("max_peak_gpu_memory_used_mb"))
        max_gpu_util = _safe_float(row.get("max_peak_gpu_util_percent"))

        print(
            f"{str(row['model_id'])[:38]:<38} "
            f"{str(row['tag'])[:34]:<34} "
            f"{int(row['runs']):>4d} "
            f"{(f'{max_rss:.2f}' if max_rss is not None else 'n/a'):>12} "
            f"{(f'{max_sys_ram:.2f}' if max_sys_ram is not None else 'n/a'):>12} "
            f"{(f'{max_gpu_mem:.2f}' if max_gpu_mem is not None else 'n/a'):>12} "
            f"{(f'{max_gpu_util:.2f}' if max_gpu_util is not None else 'n/a'):>13}"
        )


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Aggregate resource telemetry from experiment runs"
    )
    parser.add_argument("root", help="Root directory containing experiment run folders")
    parser.add_argument(
        "--tag-contains", default="", help="Filter run folder names by substring"
    )
    parser.add_argument(
        "--save-json", default="", help="Path to save aggregated JSON output"
    )
    parser.add_argument(
        "--save-csv", default="", help="Path to save aggregated CSV output"
    )
    return parser


def main() -> int:
    """Aggregate the runs under the given root, optionally saving JSON and CSV.

    Returns:
        The exit status: 1 when no summary is found.
    """
    args = _build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()

    summary_paths = _discover_resource_summaries(root, tag_contains=args.tag_contains)
    if not summary_paths:
        print("No resource_summary.json files found.")
        return 1

    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = _load_resource_summary(summary_path)
        if not summary:
            continue

        model_id = str(summary.get("model_id", "unknown"))
        tag = str(summary.get("tag", summary_path.parent.name))

        peak_gpu_mem, peak_gpu_util = _extract_gpu_peaks(summary)

        rows.append(
            {
                "run_dir": str(summary_path.parent),
                "model_id": model_id,
                "tag": tag,
                "status": str(summary.get("status", "unknown")),
                "monitoring_duration_sec": _safe_float(
                    summary.get("monitoring_duration_sec")
                ),
                "sample_count": summary.get("sample_count", 0),
                "peak_process_rss_mb": _safe_float(summary.get("peak_process_rss_mb")),
                "peak_system_ram_percent": _safe_float(
                    summary.get("peak_system_ram_percent")
                ),
                "peak_process_cpu_percent": _safe_float(
                    summary.get("peak_process_cpu_percent")
                ),
                "peak_gpu_memory_used_mb": peak_gpu_mem,
                "peak_gpu_util_percent": peak_gpu_util,
            }
        )

    aggregated = _aggregate(rows)
    _print_table(aggregated)

    if args.save_json:
        destination = Path(args.save_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(aggregated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved JSON: {destination}")

    if args.save_csv:
        destination = Path(args.save_csv).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "model_id",
            "tag",
            "runs",
            "avg_duration_sec",
            "max_duration_sec",
            "avg_peak_process_rss_mb",
            "max_peak_process_rss_mb",
            "avg_peak_system_ram_percent",
            "max_peak_system_ram_percent",
            "avg_peak_process_cpu_percent",
            "max_peak_process_cpu_percent",
            "avg_peak_gpu_memory_used_mb",
            "max_peak_gpu_memory_used_mb",
            "avg_peak_gpu_util_percent",
            "max_peak_gpu_util_percent",
        ]
        with destination.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            for row in aggregated:
                writer.writerow(row)
        print(f"Saved CSV: {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
