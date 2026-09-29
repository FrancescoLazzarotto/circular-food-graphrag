"""Gold standard generation pipeline.

Subcommands:
    generate   Poll vLLM until ready, generate question suite, convert to gold CSV.
    convert    Convert an existing JSON suite to gold CSV (no vLLM needed).

Usage:
    conda run -n graphllm python evaluation/gold/build_gold.py generate [options]
    conda run -n graphllm python evaluation/gold/build_gold.py convert --suite-input PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("build_gold")

DEFAULT_SUITE_OUTPUT = REPO_ROOT / "artifacts" / "tmp" / "graphrag_test_suite.json"
DEFAULT_GOLD_OUTPUT = REPO_ROOT / "evaluation" / "gold" / "gold_generated.csv"
DEFAULT_POLL_INTERVAL = 30
DEFAULT_POLL_TIMEOUT = 7200  # 2 hours

GOLD_FIELDNAMES = [
    "question_id",
    "question",
    "canonical_answer",
    "answer_variants",
    "expected_entities",
    "gold_triples",
    "question_type",
    "difficulty",
    "notes",
]

TYPE_MAP = {
    "fact_based": "factoid",
    "multi_hop": "multi_hop",
    "comparative": "comparative",
    "aggregation": "aggregation",
    "cross_doc": "cross_doc",
}


# ─── vLLM polling ─────────────────────────────────────────────────────────────

def _vllm_health_ok(base_url: str) -> bool:
    """Whether vLLM's /health endpoint answers 200."""
    # vLLM serves /health at the server root, not under the OpenAI /v1 prefix.
    health_base = base_url.rstrip("/")
    if health_base.endswith("/v1"):
        health_base = health_base[: -len("/v1")]
    url = health_base + "/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def _vllm_responsive(base_url: str, model_name: str, api_key: str) -> bool:
    """Try a 1-token completion to verify vLLM is not overloaded."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status == 200
    except Exception:
        return False


def _poll_until_ready(
    base_url: str,
    model_name: str,
    api_key: str,
    interval: int,
    timeout: int,
) -> bool:
    """Poll vLLM until it is healthy and answers a completion.

    Returns:
        True when vLLM is ready, False when `timeout` runs out first.
    """
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if _vllm_health_ok(base_url):
            LOGGER.info("Health OK after %d poll(s), checking responsiveness...", attempt)
            if _vllm_responsive(base_url, model_name, api_key):
                LOGGER.info("vLLM responsive — starting generation.")
                return True
            LOGGER.info("vLLM up but busy, retry in %ds...", interval)
        else:
            LOGGER.info("Poll %d: vLLM not available, retry in %ds...", attempt, interval)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    return False


# ─── JSON → gold CSV conversion ───────────────────────────────────────────────

def _convert_suite_to_gold(suite_path: Path, gold_output: Path) -> int:
    """Write a generated question suite as a gold CSV.

    Returns:
        Process exit code; 1 when the suite is missing or yields no rows.
    """
    if not suite_path.exists():
        LOGGER.error("Suite file not found: %s", suite_path)
        return 1

    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    if not isinstance(questions, list) or not questions:
        LOGGER.error("No questions in suite: %s", suite_path)
        return 1

    rows = []
    skipped = 0
    for item in questions:
        if not isinstance(item, dict):
            skipped += 1
            continue
        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        if not question:
            skipped += 1
            continue

        expected_entities = item.get("expected_entities", [])
        if not isinstance(expected_entities, list):
            expected_entities = []

        source_doc = item.get("source_doc")
        source_docs = item.get("source_docs")
        if source_doc:
            notes = str(source_doc)
        elif isinstance(source_docs, list):
            notes = ", ".join(str(d) for d in source_docs if d)
        else:
            notes = ""

        qtype_raw = str(item.get("type", ""))
        rows.append({
            "question_id": str(item.get("id", f"q_{len(rows) + 1:03d}")),
            "question": question,
            "canonical_answer": ground_truth,
            "answer_variants": "[]",
            "expected_entities": json.dumps(expected_entities, ensure_ascii=False),
            "gold_triples": "[]",
            "question_type": TYPE_MAP.get(qtype_raw, qtype_raw),
            "difficulty": "",
            "notes": notes,
        })

    if not rows:
        LOGGER.error("No valid rows to write after filtering")
        return 1

    gold_output.parent.mkdir(parents=True, exist_ok=True)
    with gold_output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=GOLD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Gold CSV saved: %s (%d rows, %d skipped)", gold_output, len(rows), skipped)
    _print_next_steps(gold_output)
    return 0


def _print_next_steps(gold_path: Path) -> None:
    """Print how to use the new gold CSV."""
    print("\n─── Next steps ────────────────────────────────────────────────")
    print(f"  Gold CSV : {gold_path}")
    print("  Review   : open the CSV, check canonical_answer quality")
    print("  gold_triples: campo vuoto — aggiungi manualmente o via Neo4j query")
    print()
    print("  Build eval dataset (dopo aver fatto girare esperimenti):")
    print(f"    python -m evaluation.evalkit.cli build-dataset \\")
    print(f"      --input artifacts/experiments/<run_dir> \\")
    print(f"      --gold-file {gold_path} \\")
    print(f"      --output artifacts/evaluation/eval_dataset.csv")
    print()
    print("  Report completo:")
    print(f"    python -m evaluation.evalkit.cli report-experiment \\")
    print(f"      --run-dir artifacts/experiments/<run_dir> \\")
    print(f"      --gold {gold_path}")
    print("────────────────────────────────────────────────────────────────")


# ─── Subcommand: generate ─────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> int:
    """``generate``: wait for vLLM, generate the suite, convert it to CSV."""
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip().rstrip("/")
    model_name = os.getenv("VLLM_MODEL_NAME", "").strip()
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")).strip()

    if not model_name:
        LOGGER.error("VLLM_MODEL_NAME env var not set")
        return 1

    LOGGER.info(
        "Polling vLLM at %s every %ds (timeout=%ds)...",
        base_url, args.poll_interval, args.poll_timeout,
    )
    if not _poll_until_ready(base_url, model_name, api_key, args.poll_interval, args.poll_timeout):
        LOGGER.error("vLLM did not become ready within %d seconds. Aborting.", args.poll_timeout)
        return 1

    suite_output = Path(args.suite_output) if args.suite_output else DEFAULT_SUITE_OUTPUT
    suite_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_questions.py"),
        "generate",
        "--output", str(suite_output),
        "--question-language", args.question_language,
    ]
    if args.run_dir:
        cmd += ["--run-dir", args.run_dir]
    if args.verbose:
        cmd += ["--verbose"]

    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        LOGGER.error("generate_questions.py failed (exit %d)", result.returncode)
        return result.returncode

    print(f"Suite saved: {suite_output}")

    if args.no_convert:
        return 0

    gold_output = Path(args.gold_output) if args.gold_output else DEFAULT_GOLD_OUTPUT
    return _convert_suite_to_gold(suite_output, gold_output)


# ─── Subcommand: convert ──────────────────────────────────────────────────────

def cmd_convert(args: argparse.Namespace) -> int:
    """``convert``: convert an existing question suite to gold CSV."""
    suite_path = Path(args.suite_input)
    gold_output = Path(args.gold_output) if args.gold_output else DEFAULT_GOLD_OUTPUT
    return _convert_suite_to_gold(suite_path, gold_output)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """The argument parser, one subparser per subcommand."""
    parser = argparse.ArgumentParser(
        prog="python evaluation/gold/build_gold.py",
        description="Gold standard generation pipeline for graphRAGPipelineExp1",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p = sub.add_parser("generate", help="Poll vLLM, generate suite, convert to gold CSV")
    p.add_argument("--run-dir", default="", help="KG pipeline run dir (auto-discovers latest if omitted)")
    p.add_argument("--suite-output", default="", help=f"JSON suite output path (default: {DEFAULT_SUITE_OUTPUT})")
    p.add_argument("--gold-output", default="", help=f"Gold CSV output path (default: {DEFAULT_GOLD_OUTPUT})")
    p.add_argument("--question-language", choices=["en", "it"], default="en", help="Language for questions and ground truth")
    p.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Seconds between vLLM health polls")
    p.add_argument("--poll-timeout", type=int, default=DEFAULT_POLL_TIMEOUT, help="Max seconds to wait for vLLM (default: 7200)")
    p.add_argument("--no-convert", action="store_true", help="Skip JSON→CSV conversion after generation")
    p.add_argument("--verbose", action="store_true", help="Print each question as generated")

    # convert
    p = sub.add_parser("convert", help="Convert existing JSON suite to gold CSV")
    p.add_argument("--suite-input", required=True, help="Path to JSON suite from generate_questions.py")
    p.add_argument("--gold-output", default="", help=f"Gold CSV output path (default: {DEFAULT_GOLD_OUTPUT})")

    return parser


def main() -> int:
    """Parse the command line and run the chosen subcommand."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "convert":
        return cmd_convert(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
