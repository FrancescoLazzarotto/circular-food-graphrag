"""A/B of the ``production_fast`` performance profile against the default one.

Runs ``run_retrieval_matrix.py`` twice on the same question subset -- baseline
with decomposition and adaptive routing, ``production_fast`` without -- and
reports the latency difference, the agreement between the two sets of answers
and, when a gold CSV is available, token F1 and exact match against it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import select
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _normalize_question(question: str) -> str:
    """Lowercased text with whitespace collapsed, the join key."""
    text = question.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _token_f1(a: str, b: str) -> float:
    """Token-overlap F1 between two texts; 1 when both are empty."""
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0

    a_counts: dict[str, int] = {}
    for token in a_tokens:
        a_counts[token] = a_counts.get(token, 0) + 1

    b_counts: dict[str, int] = {}
    for token in b_tokens:
        b_counts[token] = b_counts.get(token, 0) + 1

    overlap = 0
    for token, count in a_counts.items():
        overlap += min(count, b_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _load_questions(path: Path, limit: int) -> list[str]:
    """The first `limit` non-empty lines of a question file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it holds no question.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")

    questions = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise ValueError(f"Questions file is empty: {path}")

    return questions[: max(1, int(limit))]


def _write_questions(path: Path, questions: list[str]) -> None:
    """Write one question per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(questions) + "\n", encoding="utf-8")


def _extract_output_dir(stdout_text: str) -> str:
    """The last ``Output directory:`` the matrix runner printed, or ``""``."""
    matches = re.findall(r"Output directory:\s*(.+)", stdout_text)
    return matches[-1].strip() if matches else ""


def _resolve_output_dir_fallback(output_root: Path, tag: str) -> Path:
    """The newest run directory ending in `tag`, when the runner printed none.

    Raises:
        FileNotFoundError: If there is none.
    """
    candidates = sorted(
        output_root.glob(f"*_{tag}"), key=lambda item: item.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"Cannot resolve run output directory for tag: {tag}")
    return candidates[-1]


def _run_matrix(
    project_root: Path,
    output_root: Path,
    questions_file: Path,
    graph_strategies: str,
    model_id: str,
    runs_per_strategy: int,
    gpu_memory_fraction: float,
    allow_large_model_fp16_fallback: bool,
    keep_monitor_resources: bool,
    tag: str,
    performance_profile: str,
    max_new_tokens: int | None,
    use_vllm: bool,
    vllm_base_url: str,
    enable_decomposition_step: bool,
    enable_adaptive_routing_step: bool,
    matrix_timeout_sec: int,
    run_log_path: Path,
) -> dict[str, Any]:
    """Run ``run_retrieval_matrix.py`` as a subprocess and find its output.

    Output is streamed to the console and to `run_log_path`, with a heartbeat
    line every 30 s.

    Args:
        project_root: Working directory of the subprocess.
        output_root: ``--output-dir`` of the matrix run.
        questions_file: The question subset.
        graph_strategies: Comma-separated graph strategies.
        model_id: Generator model.
        runs_per_strategy: Repetitions of each strategy.
        gpu_memory_fraction: GPU memory share for a local model.
        allow_large_model_fp16_fallback: Passed through to the runner.
        keep_monitor_resources: Keep resource monitoring on.
        tag: Experiment tag of the run.
        performance_profile: Profile to request.
        max_new_tokens: Token budget; ``None`` leaves the profile's.
        use_vllm: Use the vLLM endpoint.
        vllm_base_url: The vLLM endpoint.
        enable_decomposition_step: Turn on question decomposition.
        enable_adaptive_routing_step: Turn on adaptive routing.
        matrix_timeout_sec: Kill the run after this long; 0 never.
        run_log_path: Where the full output is logged.

    Returns:
        The command, the run directory, the output tail, the heartbeat count
        and the log path.

    Raises:
        TimeoutError: If the run exceeds `matrix_timeout_sec`.
        RuntimeError: If the run exits non-zero.
        FileNotFoundError: If the run directory cannot be found.
    """
    cmd = [
        sys.executable,
        "scripts/runners/run_retrieval_matrix.py",
        "--llm",
        "--model-id",
        model_id,
        "--questions-file",
        str(questions_file),
        "--skip-standard",
        "--graph-strategies",
        graph_strategies,
        "--runs-per-strategy",
        str(max(1, int(runs_per_strategy))),
        "--output-dir",
        str(output_root),
        "--experiment-tag",
        tag,
        "--performance-profile",
        performance_profile,
        "--gpu-memory-fraction",
        f"{float(gpu_memory_fraction):.2f}",
    ]

    if use_vllm:
        cmd.append("--vllm")
        cmd.extend(["--vllm-base-url", str(vllm_base_url)])

    if max_new_tokens is not None and int(max_new_tokens) > 0:
        cmd.extend(["--max-new-tokens", str(int(max_new_tokens))])

    if allow_large_model_fp16_fallback:
        cmd.append("--allow-large-model-fp16-fallback")

    if not keep_monitor_resources:
        cmd.append("--no-monitor-resources")

    if enable_decomposition_step:
        cmd.append("--enable-decomposition-step")

    if enable_adaptive_routing_step:
        cmd.append("--enable-adaptive-routing-step")

    combined_lines: list[str] = []
    heartbeats = 0
    started_at = time.monotonic()
    last_heartbeat = started_at

    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with run_log_path.open("w", encoding="utf-8") as run_log:
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError("Failed to capture matrix process output stream")

        while True:
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if ready:
                line = process.stdout.readline()
                if line:
                    normalized = line.rstrip("\n")
                    combined_lines.append(normalized)
                    if len(combined_lines) > 2000:
                        combined_lines.pop(0)
                    run_log.write(normalized + "\n")
                    run_log.flush()
                    if normalized.strip():
                        print(f"[{tag}] {normalized}", flush=True)

            if (
                matrix_timeout_sec > 0
                and (time.monotonic() - started_at) > matrix_timeout_sec
            ):
                process.terminate()
                raise TimeoutError(
                    f"Matrix execution timed out after {matrix_timeout_sec}s for tag '{tag}'. "
                    f"See log: {run_log_path}"
                )

            if process.poll() is not None:
                for residual in process.stdout.readlines():
                    normalized = residual.rstrip("\n")
                    combined_lines.append(normalized)
                    if len(combined_lines) > 2000:
                        combined_lines.pop(0)
                    run_log.write(normalized + "\n")
                run_log.flush()
                break

            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                elapsed = int(now - started_at)
                heartbeats += 1
                message = f"[{tag}] still running... elapsed_sec={elapsed}"
                run_log.write(message + "\n")
                run_log.flush()
                print(message, flush=True)
                last_heartbeat = now

    if process.returncode != 0:
        output_tail = "\n".join(combined_lines[-40:])
        raise RuntimeError(
            "Matrix execution failed for tag '"
            + tag
            + "'.\n"
            + f"Command: {' '.join(cmd)}\n"
            + f"Combined output (tail):\n{output_tail}\n"
            + f"Log file: {run_log_path}"
        )

    combined_text = "\n".join(combined_lines)
    output_dir_raw = _extract_output_dir(combined_text)
    if output_dir_raw:
        output_dir = Path(output_dir_raw)
    else:
        output_dir = _resolve_output_dir_fallback(output_root=output_root, tag=tag)

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Reported output directory does not exist: {output_dir}"
        )

    return {
        "command": cmd,
        "output_dir": str(output_dir),
        "stdout_tail": "\n".join(combined_lines[-20:]),
        "heartbeat_count": heartbeats,
        "run_log_path": str(run_log_path),
    }


def _load_summary_stats(path: Path) -> dict[str, Any]:
    """The ``stats`` block of a run's ``summary.json``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("stats", {})
    return stats if isinstance(stats, dict) else {}


def _aggregate_stats(
    stats_by_strategy: dict[str, Any], target_strategies: list[str]
) -> dict[str, float]:
    """Run-weighted mean latency and sub-question count over `target_strategies`."""
    weighted_latency = 0.0
    weighted_sub_questions = 0.0
    total_runs = 0.0

    for strategy in target_strategies:
        payload = stats_by_strategy.get(strategy, {})
        if not isinstance(payload, dict):
            continue

        runs = float(payload.get("runs", 0) or 0)
        if runs <= 0:
            continue

        total_runs += runs
        weighted_latency += float(payload.get("avg_latency_ms", 0.0) or 0.0) * runs
        weighted_sub_questions += (
            float(payload.get("avg_sub_questions", 0.0) or 0.0) * runs
        )

    if total_runs <= 0:
        return {
            "runs": 0.0,
            "avg_latency_ms": 0.0,
            "avg_sub_questions": 0.0,
        }

    return {
        "runs": total_runs,
        "avg_latency_ms": weighted_latency / total_runs,
        "avg_sub_questions": weighted_sub_questions / total_runs,
    }


def _parse_metadata(raw: str) -> dict[str, Any]:
    """The ``metadata_json`` cell as a dict, empty when absent or invalid."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _load_answer_rows(results_csv_path: Path) -> list[dict[str, str]]:
    """Strategy, question, answer and run index of every row of a ``results.csv``."""
    rows: list[dict[str, str]] = []

    with results_csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            metadata = _parse_metadata(row.get("metadata_json", ""))
            rows.append(
                {
                    "strategy": str(row.get("strategy", "") or ""),
                    "question": str(row.get("question", "") or ""),
                    "answer": str(row.get("answer", "") or ""),
                    "run_index": str(metadata.get("run_index", "0")),
                }
            )

    return rows


def _load_gold(gold_file: Path) -> dict[str, str]:
    """Normalised question -> reference answer from a gold CSV; empty when absent."""
    if not gold_file.exists() or not gold_file.is_file():
        return {}

    gold_map: dict[str, str] = {}
    with gold_file.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            question = str(row.get("question", "") or "").strip()
            if not question:
                continue

            answer = ""
            for key in ("ground_truth", "gold_answer", "answer"):
                candidate = str(row.get(key, "") or "").strip()
                if candidate:
                    answer = candidate
                    break

            if not answer:
                continue

            gold_map[_normalize_question(question)] = answer

    return gold_map


def _evaluate_against_gold(
    rows: list[dict[str, str]], gold_map: dict[str, str]
) -> dict[str, float]:
    """Mean token F1 and exact-match rate of the answers that have a gold."""
    if not gold_map:
        return {
            "rows_with_gold": 0.0,
            "avg_token_f1": 0.0,
            "exact_match_rate": 0.0,
        }

    f1_scores: list[float] = []
    exact_matches = 0

    for row in rows:
        gold = gold_map.get(_normalize_question(row["question"]))
        if not gold:
            continue
        answer = row["answer"]
        f1 = _token_f1(answer, gold)
        f1_scores.append(f1)
        if _normalize_question(answer) == _normalize_question(gold):
            exact_matches += 1

    if not f1_scores:
        return {
            "rows_with_gold": 0.0,
            "avg_token_f1": 0.0,
            "exact_match_rate": 0.0,
        }

    return {
        "rows_with_gold": float(len(f1_scores)),
        "avg_token_f1": float(mean(f1_scores)),
        "exact_match_rate": float(exact_matches / len(f1_scores)),
    }


def _answer_agreement(
    baseline_rows: list[dict[str, str]], fast_rows: list[dict[str, str]]
) -> dict[str, float]:
    """Token F1 and exact-match rate of the fast answers against the baseline ones.

    Rows are paired on (strategy, question, run index).
    """
    baseline_map = {
        (row["strategy"], _normalize_question(row["question"]), row["run_index"]): row[
            "answer"
        ]
        for row in baseline_rows
    }
    fast_map = {
        (row["strategy"], _normalize_question(row["question"]), row["run_index"]): row[
            "answer"
        ]
        for row in fast_rows
    }

    common_keys = sorted(set(baseline_map.keys()) & set(fast_map.keys()))
    if not common_keys:
        return {
            "rows_compared": 0.0,
            "avg_token_f1": 0.0,
            "exact_match_rate": 0.0,
        }

    f1_scores: list[float] = []
    exact_matches = 0
    for key in common_keys:
        baseline_answer = baseline_map[key]
        fast_answer = fast_map[key]
        f1_scores.append(_token_f1(fast_answer, baseline_answer))
        if _normalize_question(fast_answer) == _normalize_question(baseline_answer):
            exact_matches += 1

    return {
        "rows_compared": float(len(common_keys)),
        "avg_token_f1": float(mean(f1_scores)),
        "exact_match_rate": float(exact_matches / len(common_keys)),
    }


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run baseline vs production_fast A/B for 32B GraphRAG and report latency/quality deltas",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--questions-file", default="evaluation/fixtures/questions_matrix_long.txt")
    parser.add_argument("--questions-count", type=int, default=10)
    parser.add_argument("--graph-strategies", default="default")
    parser.add_argument("--runs-per-strategy", type=int, default=1)

    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--report-dir", default="artifacts/evaluation")
    parser.add_argument("--experiment-tag-prefix", default="ab_32b_fast")

    parser.add_argument("--gold-file", default="evaluation/gold_questions_template.csv")
    parser.add_argument("--skip-gold-eval", action="store_true")

    parser.add_argument("--baseline-max-new-tokens", type=int, default=256)
    parser.add_argument("--fast-max-new-tokens", type=int, default=0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.92)
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible API",
    )

    parser.add_argument("--allow-large-model-fp16-fallback", action="store_true")
    parser.add_argument("--keep-monitor-resources", action="store_true")
    parser.add_argument(
        "--matrix-timeout-sec",
        type=int,
        default=0,
        help="Optional timeout for each matrix run (0 disables timeout)",
    )
    return parser


def main() -> int:
    """Run both arms and write ``ab_report.json`` and ``ab_report.txt``."""
    args = _build_parser().parse_args()

    if args.questions_count < 1:
        raise ValueError("--questions-count must be >= 1")
    if args.runs_per_strategy < 1:
        raise ValueError("--runs-per-strategy must be >= 1")
    if args.gpu_memory_fraction <= 0 or args.gpu_memory_fraction > 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1]")
    if args.matrix_timeout_sec < 0:
        raise ValueError("--matrix-timeout-sec must be >= 0")

    project_root = Path(__file__).resolve().parents[2]

    questions_path = Path(args.questions_file)
    if not questions_path.is_absolute():
        questions_path = project_root / questions_path

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    report_root = Path(args.report_dir)
    if not report_root.is_absolute():
        report_root = project_root / report_root
    report_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = report_root / f"{timestamp}_{args.experiment_tag_prefix}"
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_questions = _load_questions(
        path=questions_path, limit=args.questions_count
    )
    subset_questions_path = report_dir / "questions_subset.txt"
    _write_questions(path=subset_questions_path, questions=selected_questions)

    baseline_tag = f"{args.experiment_tag_prefix}_baseline"
    fast_tag = f"{args.experiment_tag_prefix}_fast"
    baseline_log_path = report_dir / "baseline_matrix.log"
    fast_log_path = report_dir / "production_fast_matrix.log"

    print("Running baseline profile...", flush=True)
    baseline_run = _run_matrix(
        project_root=project_root,
        output_root=output_root,
        questions_file=subset_questions_path,
        graph_strategies=args.graph_strategies,
        model_id=args.model_id,
        runs_per_strategy=args.runs_per_strategy,
        gpu_memory_fraction=args.gpu_memory_fraction,
        allow_large_model_fp16_fallback=args.allow_large_model_fp16_fallback,
        keep_monitor_resources=args.keep_monitor_resources,
        tag=baseline_tag,
        performance_profile="default",
        max_new_tokens=args.baseline_max_new_tokens,
        use_vllm=args.vllm,
        vllm_base_url=args.vllm_base_url,
        enable_decomposition_step=True,
        enable_adaptive_routing_step=True,
        matrix_timeout_sec=args.matrix_timeout_sec,
        run_log_path=baseline_log_path,
    )

    print("Running production_fast profile...", flush=True)
    fast_run = _run_matrix(
        project_root=project_root,
        output_root=output_root,
        questions_file=subset_questions_path,
        graph_strategies=args.graph_strategies,
        model_id=args.model_id,
        runs_per_strategy=args.runs_per_strategy,
        gpu_memory_fraction=args.gpu_memory_fraction,
        allow_large_model_fp16_fallback=args.allow_large_model_fp16_fallback,
        keep_monitor_resources=args.keep_monitor_resources,
        tag=fast_tag,
        performance_profile="production_fast",
        max_new_tokens=(
            None if args.fast_max_new_tokens <= 0 else int(args.fast_max_new_tokens)
        ),
        use_vllm=args.vllm,
        vllm_base_url=args.vllm_base_url,
        enable_decomposition_step=False,
        enable_adaptive_routing_step=False,
        matrix_timeout_sec=args.matrix_timeout_sec,
        run_log_path=fast_log_path,
    )

    baseline_output_dir = Path(baseline_run["output_dir"])
    fast_output_dir = Path(fast_run["output_dir"])

    baseline_stats = _load_summary_stats(baseline_output_dir / "summary.json")
    fast_stats = _load_summary_stats(fast_output_dir / "summary.json")

    target_strategies = [
        item.strip() for item in args.graph_strategies.split(",") if item.strip()
    ]
    if not target_strategies:
        target_strategies = ["default"]

    baseline_agg = _aggregate_stats(baseline_stats, target_strategies=target_strategies)
    fast_agg = _aggregate_stats(fast_stats, target_strategies=target_strategies)

    baseline_rows = _load_answer_rows(baseline_output_dir / "results.csv")
    fast_rows = _load_answer_rows(fast_output_dir / "results.csv")
    agreement = _answer_agreement(baseline_rows=baseline_rows, fast_rows=fast_rows)

    gold_map: dict[str, str] = {}
    if not args.skip_gold_eval:
        gold_path = Path(args.gold_file)
        if not gold_path.is_absolute():
            gold_path = project_root / gold_path
        gold_map = _load_gold(gold_path)

    baseline_gold = _evaluate_against_gold(rows=baseline_rows, gold_map=gold_map)
    fast_gold = _evaluate_against_gold(rows=fast_rows, gold_map=gold_map)

    baseline_avg_latency = float(baseline_agg.get("avg_latency_ms", 0.0) or 0.0)
    fast_avg_latency = float(fast_agg.get("avg_latency_ms", 0.0) or 0.0)
    latency_delta_ms = fast_avg_latency - baseline_avg_latency
    speedup_pct = (
        ((baseline_avg_latency - fast_avg_latency) / baseline_avg_latency * 100.0)
        if baseline_avg_latency > 0
        else 0.0
    )

    report = {
        "timestamp": timestamp,
        "model_id": args.model_id,
        "questions_count": len(selected_questions),
        "graph_strategies": target_strategies,
        "runs_per_strategy": args.runs_per_strategy,
        "vllm": {
            "enabled": bool(args.vllm),
            "base_url": args.vllm_base_url if args.vllm else "",
        },
        "baseline": {
            "tag": baseline_tag,
            "output_dir": str(baseline_output_dir),
            "aggregate": baseline_agg,
            "gold_eval": baseline_gold,
            "command": baseline_run["command"],
            "matrix_log_path": baseline_run.get("run_log_path", ""),
            "heartbeat_count": baseline_run.get("heartbeat_count", 0),
        },
        "production_fast": {
            "tag": fast_tag,
            "output_dir": str(fast_output_dir),
            "aggregate": fast_agg,
            "gold_eval": fast_gold,
            "command": fast_run["command"],
            "matrix_log_path": fast_run.get("run_log_path", ""),
            "heartbeat_count": fast_run.get("heartbeat_count", 0),
        },
        "delta": {
            "latency_ms": latency_delta_ms,
            "latency_speedup_pct": speedup_pct,
            "avg_sub_questions": float(fast_agg.get("avg_sub_questions", 0.0) or 0.0)
            - float(baseline_agg.get("avg_sub_questions", 0.0) or 0.0),
            "gold_avg_token_f1": float(fast_gold.get("avg_token_f1", 0.0) or 0.0)
            - float(baseline_gold.get("avg_token_f1", 0.0) or 0.0),
            "gold_exact_match_rate": float(
                fast_gold.get("exact_match_rate", 0.0) or 0.0
            )
            - float(baseline_gold.get("exact_match_rate", 0.0) or 0.0),
        },
        "answer_agreement_vs_baseline": agreement,
        "notes": {
            "gold_rows_evaluated": int(fast_gold.get("rows_with_gold", 0.0) or 0.0),
            "gold_eval_enabled": bool(not args.skip_gold_eval and bool(gold_map)),
            "quality_fallback_used": bool(not gold_map),
        },
    }

    report_json_path = report_dir / "ab_report.json"
    report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_txt_path = report_dir / "ab_report.txt"
    report_txt_path.write_text(
        "\n".join(
            [
                f"model_id={args.model_id}",
                f"questions_count={len(selected_questions)}",
                f"vllm_enabled={int(bool(args.vllm))}",
                f"vllm_base_url={args.vllm_base_url if args.vllm else ''}",
                f"baseline_output_dir={baseline_output_dir}",
                f"production_fast_output_dir={fast_output_dir}",
                f"baseline_avg_latency_ms={baseline_avg_latency:.2f}",
                f"production_fast_avg_latency_ms={fast_avg_latency:.2f}",
                f"latency_delta_ms={latency_delta_ms:.2f}",
                f"latency_speedup_pct={speedup_pct:.2f}",
                f"delta_gold_avg_token_f1={report['delta']['gold_avg_token_f1']:.6f}",
                f"delta_gold_exact_match_rate={report['delta']['gold_exact_match_rate']:.6f}",
                f"answer_agreement_avg_token_f1={agreement['avg_token_f1']:.6f}",
                f"answer_agreement_exact_match_rate={agreement['exact_match_rate']:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nA/B completed", flush=True)
    print(f"- baseline_output_dir: {baseline_output_dir}")
    print(f"- production_fast_output_dir: {fast_output_dir}")
    print(f"- baseline_avg_latency_ms: {baseline_avg_latency:.2f}")
    print(f"- production_fast_avg_latency_ms: {fast_avg_latency:.2f}")
    print(f"- latency_delta_ms: {latency_delta_ms:.2f}")
    print(f"- latency_speedup_pct: {speedup_pct:.2f}")
    print(f"- delta_gold_avg_token_f1: {report['delta']['gold_avg_token_f1']:.6f}")
    print(
        f"- delta_gold_exact_match_rate: {report['delta']['gold_exact_match_rate']:.6f}"
    )
    print(f"- answer_agreement_avg_token_f1: {agreement['avg_token_f1']:.6f}")
    print(f"- answer_agreement_exact_match_rate: {agreement['exact_match_rate']:.6f}")
    print(f"- report_json: {report_json_path}", flush=True)
    print(f"- report_txt: {report_txt_path}", flush=True)
    print(f"- baseline_matrix_log: {baseline_log_path}", flush=True)
    print(f"- production_fast_matrix_log: {fast_log_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
