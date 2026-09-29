"""Fit the campaign as one model instead of six aggregates.

The tables in the results chapter compare strategies through a mean over six
generators, then test that mean with a sign test on n=6 or a paired bootstrap on
26 questions. Both throw away the structure of the design: every strategy is
observed on the same 26 scorable questions under the same six generators, and
1,248 cells carry the information that a mean over six numbers hides.

This script fits the whole grid at once:

    concept_f1 ~ strategy + question + generator

with question and generator as fixed effects, which is the within-question
estimator: the coefficient on a strategy is its average difference from the
reference strategy on the same question under the same generator, and question
difficulty cancels rather than adding noise. Inference is a cluster bootstrap
that resamples whole questions, since the six generators see the same question
and their errors are not independent.

The model is deliberately not a random-effects one. Random intercepts would buy
partial pooling over 26 questions and would need a distributional assumption on
question difficulty that nothing here justifies; the fixed-effect version answers
the same comparison without it. ``--lmm`` fits the random-intercept model as a
cross-check when statsmodels is installed.

Usage:
    python evaluation/scripts/mixed_effects.py \\
        --runs <RUN_1> ... <RUN_6> --gold evaluation/gold/gold_v3.json \\
        --reference text_only --out-prefix artifacts/evaluation/mixed_effects
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from evalkit.io.dataset import build_dataset  # noqa: E402
from evalkit.io.gold_loader import load_gold_json  # noqa: E402
from evalkit.metrics.entities import score_row  # noqa: E402
from evalkit.metrics.mentions import Gazetteer, answer_channel_row  # noqa: E402
from evalkit.metrics.resolver import Resolver  # noqa: E402

logger = logging.getLogger("evalkit.mixed_effects")


def long_table(run_dirs: list[Path], gold_path: Path) -> list[dict[str, Any]]:
    """One record per (generator, strategy, question), concept F1 on the answer.

    Scoring goes through the same path as ``score_gold_run.py``: the gazetteer
    over the answer text, then the shared resolver. Distractors carry no expected
    concepts and score None, so they drop out here.
    """
    gold = load_gold_json(gold_path)
    gazetteer = Gazetteer.from_gold(list(gold.values()))
    resolver = Resolver.from_gold(gold_path)

    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows = [r for r in build_dataset([run_dir], gold_path=gold_path) if r.gold_query]
        for row in rows:
            scores = score_row(answer_channel_row(row, gazetteer), resolver)
            if scores.concept is None:
                continue
            records.append(
                {
                    "generator": row.model_id,
                    "strategy": row.strategy,
                    "question": row.question_id,
                    "question_type": scores.query_type,
                    "f1": float(scores.concept.f1),
                }
            )
    return records


def design(
    records: list[dict[str, Any]], reference: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build the design matrix: strategy contrasts, question and generator dummies."""
    strategies = sorted({r["strategy"] for r in records})
    if reference not in strategies:
        raise SystemExit(f"reference {reference!r} not among {strategies}")
    others = [s for s in strategies if s != reference]
    questions = sorted({r["question"] for r in records})[1:]
    generators = sorted({r["generator"] for r in records})[1:]

    names = ["intercept"] + others + questions + generators
    columns = []
    for record in records:
        row = [1.0]
        row += [1.0 if record["strategy"] == s else 0.0 for s in others]
        row += [1.0 if record["question"] == q else 0.0 for q in questions]
        row += [1.0 if record["generator"] == g else 0.0 for g in generators]
        columns.append(row)

    X = np.array(columns)
    y = np.array([r["f1"] for r in records])
    return X, y, names, others


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of `y` on `X`."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def cluster_bootstrap(
    records: list[dict[str, Any]],
    reference: str,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Resample whole questions, refit, and report the strategy contrasts."""
    rng = np.random.default_rng(seed)
    X, y, names, strategies = design(records, reference)
    point = fit(X, y)

    by_question: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_question.setdefault(record["question"], []).append(record)
    question_ids = sorted(by_question)

    draws_by_name: dict[str, list[float]] = {name: [] for name in strategies}
    for _ in range(draws):
        picked = rng.choice(len(question_ids), size=len(question_ids), replace=True)
        resampled: list[dict[str, Any]] = []
        for new_id, index in enumerate(picked):
            for record in by_question[question_ids[index]]:
                clone = dict(record)
                # A question drawn twice must enter as two distinct fixed effects,
                # or its dummy column is duplicated and the design goes singular.
                clone["question"] = f"{record['question']}#{new_id}"
                resampled.append(clone)
        try:
            Xb, yb, names_b, _ = design(resampled, reference)
            beta = fit(Xb, yb)
        except np.linalg.LinAlgError:
            continue
        index_b = {name: i for i, name in enumerate(names_b)}
        for name in strategies:
            if name in index_b:
                draws_by_name[name].append(float(beta[index_b[name]]))

    index = {name: i for i, name in enumerate(names)}
    out: dict[str, dict[str, float]] = {}
    for name in strategies:
        samples = np.array(draws_by_name[name])
        estimate = float(point[index[name]])
        low, high = (np.percentile(samples, [2.5, 97.5]) if samples.size else (float("nan"),) * 2)
        # Two-sided bootstrap p: how often the resampled effect crosses zero.
        share = float((samples <= 0).mean()) if samples.size else float("nan")
        out[name] = {
            "estimate": estimate,
            "ci_low": float(low),
            "ci_high": float(high),
            "p_boot": float(min(1.0, 2 * min(share, 1 - share))),
            "draws": int(samples.size),
        }
    return out


def cell_bootstrap(
    records: list[dict[str, Any]],
    reference: str,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Resample cells instead of questions, which is the wrong unit.

    Reported for contrast, never as a result. Treating the 1,248 cells as
    independent draws ignores that six of them answer the same question under
    six generators, and the interval it produces is the interval a design with
    1,248 independent observations would have earned.
    """
    rng = np.random.default_rng(seed)
    X, y, names, strategies = design(records, reference)
    point = fit(X, y)
    index = {name: i for i, name in enumerate(names)}

    draws_by_name: dict[str, list[float]] = {name: [] for name in strategies}
    for _ in range(draws):
        picked = rng.choice(len(records), size=len(records), replace=True)
        resampled = [records[i] for i in picked]
        try:
            Xb, yb, names_b, _ = design(resampled, reference)
            beta = fit(Xb, yb)
        except (np.linalg.LinAlgError, SystemExit):
            continue
        index_b = {name: i for i, name in enumerate(names_b)}
        for name in strategies:
            if name in index_b:
                draws_by_name[name].append(float(beta[index_b[name]]))

    out: dict[str, dict[str, float]] = {}
    for name in strategies:
        samples = np.array(draws_by_name[name])
        low, high = (np.percentile(samples, [2.5, 97.5]) if samples.size else (float("nan"),) * 2)
        share = float((samples <= 0).mean()) if samples.size else float("nan")
        out[name] = {
            "estimate": float(point[index[name]]),
            "ci_low": float(low),
            "ci_high": float(high),
            "p_boot": float(min(1.0, 2 * min(share, 1 - share))),
            "draws": int(samples.size),
        }
    return out


def fit_lmm(records: list[dict[str, Any]], reference: str) -> dict[str, Any] | None:
    """Random-intercept cross-check, when statsmodels is available."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    frame = pd.DataFrame(records)
    frame["strategy"] = pd.Categorical(
        frame["strategy"],
        categories=[reference] + sorted(set(frame["strategy"]) - {reference}),
    )
    model = smf.mixedlm(
        "f1 ~ strategy + C(generator)", frame, groups=frame["question"]
    ).fit(reml=True)
    return {
        "params": {k: float(v) for k, v in model.params.items() if k.startswith("strategy")},
        "pvalues": {k: float(v) for k, v in model.pvalues.items() if k.startswith("strategy")},
        "group_var": float(model.cov_re.iloc[0, 0]),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, fit the campaign grid, report the strategy contrasts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--reference", default="text_only")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=Path)
    parser.add_argument("--lmm", action="store_true", help="also fit the random-intercept model")
    parser.add_argument(
        "--cell-bootstrap",
        action="store_true",
        help="also resample cells rather than questions, to show what treating "
        "the grid as independent observations would have claimed",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    records = long_table(args.runs, args.gold)
    logger.info(
        "%d cells: %d questions x %d strategies x %d generators",
        len(records),
        len({r["question"] for r in records}),
        len({r["strategy"] for r in records}),
        len({r["generator"] for r in records}),
    )

    result = cluster_bootstrap(records, args.reference, args.draws, args.seed)
    print(f"\nconcept F1 against {args.reference}, within question and generator")
    print(f"{'strategy':22s} {'estimate':>9s} {'95% CI':>20s} {'p':>7s}")
    for name, stats in sorted(result.items(), key=lambda kv: -kv[1]["estimate"]):
        ci = f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]"
        print(f"{name:22s} {stats['estimate']:+9.4f} {ci:>20s} {stats['p_boot']:7.3f}")

    report: dict[str, Any] = {
        "reference": args.reference,
        "cells": len(records),
        "questions": sorted({r["question"] for r in records}),
        "generators": sorted({r["generator"] for r in records}),
        "draws": args.draws,
        "seed": args.seed,
        "fixed_effects": result,
    }
    if args.cell_bootstrap:
        naive = cell_bootstrap(records, args.reference, args.draws, args.seed)
        report["cell_bootstrap"] = naive
        print("\nsame model, cells resampled instead of questions (wrong unit, shown for contrast)")
        print(f"{'strategy':22s} {'estimate':>9s} {'95% CI':>20s} {'p':>7s}")
        for name, stats in sorted(naive.items(), key=lambda kv: -kv[1]["estimate"]):
            ci = f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]"
            print(f"{name:22s} {stats['estimate']:+9.4f} {ci:>20s} {stats['p_boot']:7.3f}")

    if args.lmm:
        lmm = fit_lmm(records, args.reference)
        if lmm is None:
            logger.warning("statsmodels not installed; skipping the random-intercept fit")
        else:
            report["lmm"] = lmm
            print("\nrandom-intercept cross-check")
            for name, value in lmm["params"].items():
                print(f"{name:34s} {value:+.4f}  p={lmm['pvalues'][name]:.3f}")

    if args.out_prefix:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.out_prefix.with_suffix(".json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("wrote %s.json", args.out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
