"""Paired bootstrap on the per-question difference between two KG variants.

``compare_kg_variants.py`` prints micro aggregates, which say which variant is
ahead but not whether the gap survives resampling. With a few dozen scored
questions, a 0.03 difference can easily be one question changing its mind.

The bootstrap is paired: the same question index is drawn for both variants, so
the resample cancels question difficulty and measures only the variant effect —
the two runs answer the identical questions with the identical generator, and
the only thing that differs is the state of the graph.

Reported per strategy: the mean per-question difference, its 95 % interval, and
the share of resamples on the positive side.

Usage::

    conda run -n graphllm python scripts/analysis/kg_variant_significance.py \\
        --baseline exp_results_kg_v2/v0_baseline/<run> \\
        --variant  exp_results_kg_v2/v3_densified/<run> \\
        --channel answer
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "evaluation"))

from evalkit.io.dataset import build_dataset  # noqa: E402
from evalkit.metrics.entities import score_row  # noqa: E402
from evalkit.metrics.mentions import Gazetteer, answer_channel_row  # noqa: E402
from evalkit.metrics.resolver import Resolver  # noqa: E402


def per_question_f1(run_dir: Path, gold_path: Path, channel: str,
                    metric: str = "f1") -> dict[tuple[str, str], float]:
    """(strategy, query_id) -> one concept-level figure for one run.

    F1 by default. Recall is worth asking for separately: precision here counts a
    concept expected by another question as a false positive, so an F1 difference
    mixes what a pipeline found with how much benchmark vocabulary it happened to
    use elsewhere in the same answer.

    Args:
        run_dir: Run directory holding ``results.jsonl``.
        gold_path: Gold set the run is scored against.
        channel: ``"answer"`` or ``"retrieval"``.
        metric: ``"f1"``, ``"recall"`` or ``"precision"``.

    Returns:
        The figure per scored (strategy, query_id); distractors and questions
        with no concept score are left out.
    """
    resolver = Resolver.from_gold(gold_path)
    rows = [r for r in build_dataset([run_dir], gold_path=gold_path) if r.gold_query is not None]
    if channel == "answer":
        gold = {}
        for row in rows:
            gold[row.question_id] = row.gold_query
        gazetteer = Gazetteer.from_gold(list(gold.values()))

    out: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.is_distractor:
            continue  # a distractor has no expected entity, so no F1
        # The answer channel is the same scorer over a row whose entity list has
        # been swapped for the gazetteer's mentions of the answer text.
        scored = score_row(answer_channel_row(row, gazetteer) if channel == "answer" else row,
                           resolver)
        concept = scored.concept
        if concept is None:
            continue
        out[(row.strategy, row.question_id)] = getattr(concept, metric) or 0.0
    return out


def paired_bootstrap(diffs: list[float], resamples: int, seed: int) -> tuple[float, float, float, float]:
    """Bootstrap the mean of paired per-question differences.

    Args:
        diffs: Per-question differences, variant minus baseline.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling generator.

    Returns:
        ``(mean, lower, upper, positive)``: the observed mean difference, the
        95 % percentile interval of the resampled means, and the share of
        resampled means above zero.
    """
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(statistics.fmean(sample))
    means.sort()
    lower = means[int(0.025 * len(means))]
    upper = means[int(0.975 * len(means)) - 1]
    positive = sum(1 for m in means if m > 0) / len(means)
    return statistics.fmean(diffs), lower, upper, positive


def main(argv: list[str] | None = None) -> int:
    """Print the paired bootstrap for every strategy.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="ignored with --within")
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--within", metavar="STRATEGY",
                        help="compare every strategy of --variant against this one, inside the "
                             "same run; use --within text_only to ask whether the graph beats "
                             "the text-only pipeline")
    parser.add_argument("--gold", type=Path, default=REPO / "evaluation" / "gold" / "gold_v3.json")
    parser.add_argument("--channel", choices=("answer", "retrieval"), default="answer")
    parser.add_argument("--metric", choices=("f1", "recall", "precision"), default="f1")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.within and args.baseline is None:
        parser.error("--baseline is required unless --within is given")

    var = per_question_f1(args.variant, args.gold, args.channel, args.metric)
    if args.within:
        # Same run, same generator, same questions: the pairing key is the
        # question, and the only thing that differs is which strategy answered
        # it. Reuse the cross-run dictionary shape by relabelling the reference
        # strategy onto every other one.
        reference = {qid: f1 for (strategy, qid), f1 in var.items() if strategy == args.within}
        if not reference:
            parser.error(f"strategy {args.within!r} absent from {args.variant}")
        base = {key: reference[key[1]] for key in var if key[1] in reference}
    else:
        base = per_question_f1(args.baseline, args.gold, args.channel, args.metric)

    by_strategy: dict[str, list[float]] = defaultdict(list)
    for key, value in base.items():
        if key in var:
            by_strategy[key[0]].append(var[key] - value)

    print(f"\npaired bootstrap, {args.channel} channel, concept {args.metric}")
    print(f"baseline: {args.within if args.within else args.baseline.name}")
    print(f"variant : {args.variant.name}\n")
    print(f"{'strategy':<20}{'n':>4}{'mean Δ':>10}{'95% CI':>20}{'P(Δ>0)':>9}")
    print("-" * 63)
    for strategy in sorted(by_strategy):
        diffs = by_strategy[strategy]
        mean, lower, upper, positive = paired_bootstrap(diffs, args.resamples, args.seed)
        print(f"{strategy:<20}{len(diffs):>4}{mean:>+10.3f}"
              f"{f'[{lower:+.3f}, {upper:+.3f}]':>20}{positive:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
