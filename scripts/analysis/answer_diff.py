#!/usr/bin/env python3
"""Compare the *textual answers* of each strategy against a baseline strategy.

On the answers themselves, do the strategies differ -- and in what? This reads
``results.jsonl`` (from ``graphrag.cli --experiment`` /
``run_retrieval_matrix.py``) and, per query, measures how far each strategy's
answer drifts from a baseline strategy's answer (default: ``text_only``).

Two cheap, dependency-free similarity signals per (query, strategy) vs baseline:

* ``ratio``   -- difflib character-level similarity (1.0 = identical text).
* ``jaccard`` -- token-set overlap (1.0 = same word set, order ignored).

Low similarity = the strategy materially changed the answer. Pair this with
``provenance_precision.py`` (why it changed: which retrieved units differ / where
they come from) and evalkit (whether the change helped vs the gold).

Usage:
    # Table: per strategy, mean similarity to the text_only answer + n changed.
    python scripts/analysis/answer_diff.py \
        --results artifacts/experiments/<run>/results.jsonl \
        --baseline text_only --output-csv answer_diff.csv

    # Side-by-side markdown of the most-divergent answers (for eyeballing).
    python scripts/analysis/answer_diff.py \
        --results artifacts/experiments/<run>/results.jsonl \
        --baseline text_only --side-by-side answers.md --top 15
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger("graphrag")

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# A strategy's answer counts as "materially changed" below this char-ratio.
_CHANGED_RATIO = 0.85


def _norm(text: str) -> str:
    """Lowercase and collapse whitespace for stable comparison."""
    return " ".join((text or "").lower().split())


def _tokens(text: str) -> set[str]:
    """The lowercased word set of `text`."""
    return set(_WORD_RE.findall((text or "").lower()))


def char_ratio(a: str, b: str) -> float:
    """difflib character-level similarity of two answers (0..1)."""
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two answers (0..1)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def resolve_results_paths(results: list[str], results_dir: str | None) -> list[Path]:
    """Collect the ``results.jsonl`` files to analyse (see provenance_precision.py)."""
    paths: set[Path] = set()
    for item in results:
        candidate = Path(item).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Results file not found: {candidate}")
        paths.add(candidate)
    if results_dir:
        root = Path(results_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Results dir not found: {root}")
        paths.update(p.resolve() for p in root.rglob("results.jsonl"))
    if not paths:
        raise FileNotFoundError("No results.jsonl files to analyse.")
    return sorted(paths)


def iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield JSON records from the given result files, skipping malformed lines."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line %d in %s", line_no, path)
                    continue
                if isinstance(record, dict):
                    yield record


def _answer_key(record: dict[str, Any]) -> str:
    """Join key for a query: prefer query_id, fall back to question text."""
    qid = str(record.get("query_id", "") or "").strip()
    return qid or _norm(str(record.get("question", "")))[:120]


def build_pairs(
    records: Iterable[dict[str, Any]], baseline: str
) -> list[dict[str, Any]]:
    """Pair every strategy's answer with the baseline answer for the same query.

    Args:
        records: results.jsonl records.
        baseline: Strategy whose answer is the reference (e.g. ``text_only``).

    Returns:
        One row per (query, non-baseline strategy) with similarity signals.
    """
    # query -> strategy -> answer
    by_query: dict[str, dict[str, str]] = defaultdict(dict)
    questions: dict[str, str] = {}
    for record in records:
        key = _answer_key(record)
        strategy = str(record.get("strategy", "") or "?")
        by_query[key][strategy] = str(record.get("answer", "") or "")
        questions.setdefault(key, str(record.get("question", "") or ""))

    rows: list[dict[str, Any]] = []
    for key, answers in by_query.items():
        if baseline not in answers:
            logger.warning("Query %s has no baseline (%s) answer; skipped.", key, baseline)
            continue
        base_ans = answers[baseline]
        for strategy, ans in answers.items():
            if strategy == baseline:
                continue
            ratio = char_ratio(base_ans, ans)
            rows.append(
                {
                    "query": key,
                    "question": questions.get(key, ""),
                    "strategy": strategy,
                    "baseline": baseline,
                    "ratio": ratio,
                    "jaccard": jaccard(base_ans, ans),
                    "changed": ratio < _CHANGED_RATIO,
                    "baseline_answer": base_ans,
                    "answer": ans,
                }
            )
    return rows


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per strategy: mean similarity to baseline and #changed."""
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy[row["strategy"]].append(row)
    out: list[dict[str, Any]] = []
    for strategy in sorted(by_strategy):
        group = by_strategy[strategy]
        n = len(group)
        mean_ratio = sum(r["ratio"] for r in group) / n
        mean_jacc = sum(r["jaccard"] for r in group) / n
        changed = sum(1 for r in group if r["changed"])
        out.append(
            {
                "strategy": strategy,
                "n": n,
                "mean_ratio": mean_ratio,
                "mean_jaccard": mean_jacc,
                "n_changed": changed,
                "pct_changed": changed / n if n else 0.0,
            }
        )
    return out


def print_summary(summary: list[dict[str, Any]], baseline: str) -> None:
    """Print the per-strategy divergence summary."""
    header = (
        f"vs baseline '{baseline}'\n"
        f"{'strategy':<20} {'n':>3} {'mean_ratio':>10} {'mean_jacc':>10} "
        f"{'changed':>8} {'%changed':>9}"
    )
    print(header)
    print("-" * 62)
    for row in summary:
        print(
            f"{row['strategy']:<20} {row['n']:>3} {row['mean_ratio']:>10.3f} "
            f"{row['mean_jaccard']:>10.3f} {row['n_changed']:>8} "
            f"{row['pct_changed']:>8.0%}"
        )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write per (query, strategy) similarity rows (answers truncated)."""
    fields = ["query", "strategy", "baseline", "ratio", "jaccard", "changed", "question"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["strategy"], r["ratio"])):
            out = dict(row)
            out["ratio"] = f"{out['ratio']:.4f}"
            out["jaccard"] = f"{out['jaccard']:.4f}"
            writer.writerow(out)


def write_side_by_side(rows: list[dict[str, Any]], path: Path, top: int) -> None:
    """Dump the most-divergent (query, strategy) pairs as readable markdown."""
    ordered = sorted(rows, key=lambda r: r["ratio"])[:top]
    lines: list[str] = [f"# Most-divergent answers vs baseline (top {top})\n"]
    for row in ordered:
        lines.append(
            f"## {row['query']} — {row['strategy']} "
            f"(ratio={row['ratio']:.3f}, jaccard={row['jaccard']:.3f})"
        )
        lines.append(f"**Q:** {row['question']}\n")
        lines.append(f"**{row['baseline']} (baseline):** {row['baseline_answer']}\n")
        lines.append(f"**{row['strategy']}:** {row['answer']}\n")
        lines.append("---\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare strategy answers against a baseline strategy's answers.",
    )
    parser.add_argument("--results", nargs="+", default=[], help="results.jsonl file(s).")
    parser.add_argument("--results-dir", default=None, help="Dir searched for results.jsonl.")
    parser.add_argument(
        "--baseline",
        default="text_only",
        help="Baseline strategy every other strategy is compared against.",
    )
    parser.add_argument("--output-csv", default=None, help="Write per-pair CSV here.")
    parser.add_argument(
        "--side-by-side",
        default=None,
        help="Write a markdown side-by-side of the most-divergent answers here.",
    )
    parser.add_argument(
        "--top", type=int, default=15, help="How many divergent pairs in --side-by-side."
    )
    return parser


def main() -> None:
    """Summarise answer drift per strategy, optionally writing CSV and markdown."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()

    paths = resolve_results_paths(args.results, args.results_dir)
    logger.info("Analysing %d result file(s).", len(paths))

    rows = build_pairs(iter_records(paths), args.baseline)
    if not rows:
        raise SystemExit(f"No comparable answers found for baseline '{args.baseline}'.")

    summary = summarise(rows)
    print_summary(summary, args.baseline)

    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
        logger.info("Wrote CSV: %s", args.output_csv)
    if args.side_by_side:
        write_side_by_side(rows, Path(args.side_by_side), args.top)
        logger.info("Wrote side-by-side: %s", args.side_by_side)


if __name__ == "__main__":
    main()
