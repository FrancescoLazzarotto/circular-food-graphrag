"""Answer-channel precision, answer length, and text overlap with the reference.

Three measurements the main tables leave out, in one pass over the campaign.

*Precision.* The main table reports concept F1 alone. Its precision counts a
concept the reference set does not expect for that question as a false positive,
including one the gold expects for a different question, so it falls as an answer
grows. Reporting it next to answer length is the honest way to show what F1
carries.

*Length.* Generators differ by a factor of two in how much they write. A metric
that penalises length is in part a measure of verbosity, and the correlation
between length and F1 says how much.

*Text overlap.* The gold carries an ``expected_answer`` per question. ROUGE-L
recall asks how much of that reference wording the answer reproduces, and token
F1 is the SQuAD-style overlap. Both are computed here rather than imported: an
LCS and a bag of tokens have no thresholds to pre-register and no model to name.

Precision is not reported for the text metrics, and for the same reason it is
reported with a warning for the concept channel: the reference answers run one to
three sentences against generated answers of hundreds of words, so precision
against them measures length.

The closing source list is stripped before scoring. It repeats document names and
page numbers that belong to the citation measurement, not to the answer.

Usage:
    python evaluation/scripts/answer_text_metrics.py \\
        --runs <RUN_1> ... <RUN_6> --gold evaluation/gold/gold_v3.json \\
        --out-prefix artifacts/evaluation/answer_text
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from evalkit.io.dataset import build_dataset  # noqa: E402
from evalkit.io.gold_loader import load_gold_json  # noqa: E402
from evalkit.metrics.entities import score_row  # noqa: E402
from evalkit.metrics.mentions import Gazetteer, answer_channel_row  # noqa: E402
from evalkit.metrics.resolver import Resolver  # noqa: E402

logger = logging.getLogger("evalkit.answer_text")

SOURCE_BLOCK = re.compile(r"\n\s*(?:Sources|Fonti)\s*:.*\Z", re.DOTALL)
ARTICLES = {"a", "an", "the", "il", "lo", "la", "i", "gli", "le", "un", "uno", "una"}


def body(answer: str) -> str:
    """The answer without its closing source list."""
    return SOURCE_BLOCK.sub("", str(answer or "")).strip()


def tokens(text: str) -> list[str]:
    """Lowercase, drop punctuation and articles, split on whitespace."""
    lowered = str(text or "").lower()
    stripped = lowered.translate(str.maketrans("", "", string.punctuation))
    return [t for t in stripped.split() if t and t not in ARTICLES]


def token_f1(prediction: list[str], reference: list[str]) -> tuple[float, float, float]:
    """SQuAD-style overlap over token multisets."""
    if not prediction or not reference:
        return (0.0, 0.0, 0.0)
    shared = Counter(prediction) & Counter(reference)
    overlap = sum(shared.values())
    if overlap == 0:
        return (0.0, 0.0, 0.0)
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return (precision, recall, 2 * precision * recall / (precision + recall))


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, two rows of the usual table."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[-1], previous[index + 1]))
        previous = current
    return previous[-1]


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; NaN with fewer than three points or no variance."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = (sum(d * d for d in dx) ** 0.5) * (sum(d * d for d in dy) ** 0.5)
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else float("nan")


def collect(run_dirs: list[Path], gold_path: Path) -> list[dict[str, Any]]:
    """One record per scored answer: concept P/R/F1, length, text overlap."""
    gold = load_gold_json(gold_path)
    gazetteer = Gazetteer.from_gold(list(gold.values()))
    resolver = Resolver.from_gold(gold_path)
    references = {
        query.query_id: tokens(getattr(query, "expected_answer", "") or "")
        for query in gold.values()
    }

    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for row in build_dataset([run_dir], gold_path=gold_path):
            if row.gold_query is None:
                continue
            scored = score_row(answer_channel_row(row, gazetteer), resolver)
            if scored.concept is None:
                continue
            text = body(row.answer)
            predicted = tokens(text)
            reference = references.get(row.question_id, [])
            _, tok_recall, tok_f1 = token_f1(predicted, reference)
            rouge_recall = lcs_length(predicted, reference) / len(reference) if reference else 0.0
            concept = scored.concept
            records.append(
                {
                    "generator": row.model_id,
                    "strategy": row.strategy,
                    "question": row.question_id,
                    "chars": len(text),
                    # precision is None when the answer names no gold form at
                    # all, which is 0/0 rather than zero precision.
                    "concept_p": None if concept.precision is None else float(concept.precision),
                    "concept_r": None if concept.recall is None else float(concept.recall),
                    "concept_f1": 0.0 if concept.f1 is None else float(concept.f1),
                    "rouge_l_recall": float(rouge_recall),
                    "token_f1": float(tok_f1),
                    "token_recall": float(tok_recall),
                }
            )
    return records


def _mean(values: list[Any]) -> float:
    """Mean over the defined values; 0/0 cells carry None and drop out."""
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else float("nan")


def aggregate(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    """Means per value of `key`, with the length correlations."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    out: dict[str, dict[str, float]] = {}
    for name, rows in sorted(groups.items()):
        n = len(rows)
        out[name] = {
            "n": n,
            "n_precision": sum(1 for r in rows if r["concept_p"] is not None),
            "chars": _mean([r["chars"] for r in rows]),
            "concept_p": _mean([r["concept_p"] for r in rows]),
            "concept_r": _mean([r["concept_r"] for r in rows]),
            "concept_f1": _mean([r["concept_f1"] for r in rows]),
            "rouge_l_recall": _mean([r["rouge_l_recall"] for r in rows]),
            "token_f1": _mean([r["token_f1"] for r in rows]),
            "token_recall": _mean([r["token_recall"] for r in rows]),
            "r_len_f1": pearson([r["chars"] for r in rows], [r["concept_f1"] for r in rows]),
            "r_len_precision": pearson(
                [r["chars"] for r in rows if r["concept_p"] is not None],
                [r["concept_p"] for r in rows if r["concept_p"] is not None],
            ),
        }
    return out


def table(rows: dict[str, dict[str, float]], label: str) -> str:
    """Fixed-width text table of the aggregates."""
    head = (
        f"{label:34s} {'n':>5s} {'chars':>7s} {'C-P':>6s} {'C-R':>6s} {'C-F1':>6s} "
        f"{'ROUGE-L R':>10s} {'tok-F1':>7s} {'r(len,F1)':>10s} {'r(len,P)':>9s}"
    )
    lines = [head, "-" * len(head)]
    for name, stats in rows.items():
        lines.append(
            f"{name[-34:]:34s} {stats['n']:5.0f} {stats['chars']:7.0f} {stats['concept_p']:6.3f} "
            f"{stats['concept_r']:6.3f} {stats['concept_f1']:6.3f} {stats['rouge_l_recall']:10.3f} "
            f"{stats['token_f1']:7.3f} {stats['r_len_f1']:10.3f} {stats['r_len_precision']:9.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, score the runs, print the tables and save them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    records = collect(args.runs, args.gold)

    by_strategy = aggregate(records, "strategy")
    by_generator = aggregate(records, "generator")
    overall_r = pearson(
        [r["chars"] for r in records], [r["concept_f1"] for r in records]
    )
    defined = [r for r in records if r["concept_p"] is not None]
    overall_rp = pearson(
        [r["chars"] for r in defined], [r["concept_p"] for r in defined]
    )

    print(table(by_strategy, "strategy"))
    print()
    print(table(by_generator, "generator"))
    print(
        f"\npooled over {len(records)} cells: r(length, concept F1) = {overall_r:+.3f}, "
        f"r(length, concept precision) = {overall_rp:+.3f}"
    )

    if args.out_prefix:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.out_prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "cells": len(records),
                    "by_strategy": by_strategy,
                    "by_generator": by_generator,
                    "pooled_r_len_f1": overall_r,
                    "pooled_r_len_precision": overall_rp,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.warning("wrote %s.json", args.out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
