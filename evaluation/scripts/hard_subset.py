"""Answer recall on the slots a generator cannot produce without the corpus.

Roughly half of the expected concepts are ones the generator names correctly
with no retrieval at all, because several corpus documents are open-access and
plausibly appeared in pretraining. Those slots are noise in a retrieval
comparison: every pipeline gets them, so they compress every difference in the
table toward zero.

This script splits the expected concepts per generator, using that generator's
own ``no_retrieval`` answers as the divider, and reports answer-channel recall
over the half that retrieval actually has to supply. The split is per generator
on purpose: what a model already knows is a property of the model, so a subset
computed on one and applied to another would measure the wrong thing.

Matching follows the concept level of ``evalkit.metrics.entities`` exactly, down
to the private key helper, so a slot counted here is a slot the reported scorer
counts.

Usage::

    conda run -n graphllm python evaluation/scripts/hard_subset.py \\
        --run-dir exp_results_v3graph/qwen25_32b_awq/<run> \\
        --gold evaluation/gold/gold_v3.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "evaluation"))

from evalkit.io.dataset import build_dataset  # noqa: E402
from evalkit.metrics.entities import _distinct_keys, entity_label, retrieved_labels  # noqa: E402
from evalkit.metrics.mentions import Gazetteer, answer_channel_row  # noqa: E402

BASELINE = "no_retrieval"


def matched_slots(row, gazetteer) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(matched, expected) slots for one row, as (query_id, entity label) pairs.

    A slot is one expected concept of one question. The answer channel replaces
    the row's entity list with the gold surface forms the gazetteer finds in the
    generated text, which is the only channel where a text pipeline and a graph
    pipeline can be compared at all.
    """
    scored_row = answer_channel_row(row, gazetteer)
    keys = set(_distinct_keys(retrieved_labels(scored_row)))
    expected, matched = set(), set()
    for entity in row.gold_query.expected_entities:
        if not entity.surface_forms:
            continue
        slot = (row.question_id, entity.label)
        expected.add(slot)
        if keys & set(entity.surface_forms):
            matched.add(slot)
    return matched, expected


def main(argv: list[str] | None = None) -> int:
    """Split one run's slots on its no_retrieval answers; print recall per strategy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=REPO / "evaluation" / "gold" / "gold_v3.json")
    args = parser.parse_args(argv)

    rows = [r for r in build_dataset([args.run_dir], gold_path=args.gold)
            if r.gold_query is not None and not r.is_distractor]
    if not rows:
        parser.error(f"no scorable rows under {args.run_dir}")
    gazetteer = Gazetteer.from_gold([r.gold_query for r in rows])

    by_strategy: dict[str, tuple[set, set]] = {}
    for row in rows:
        matched, expected = matched_slots(row, gazetteer)
        got, want = by_strategy.setdefault(row.strategy, (set(), set()))
        by_strategy[row.strategy] = (got | matched, want | expected)

    if BASELINE not in by_strategy:
        parser.error(f"{BASELINE!r} absent from {args.run_dir}: no divider to split on")
    known, all_slots = by_strategy[BASELINE]
    hard = all_slots - known

    print(f"run: {args.run_dir.name}")
    print(f"expected slots: {len(all_slots)}")
    print(f"answered without retrieval: {len(known)} ({len(known) / len(all_slots):.1%})")
    print(f"hard subset: {len(hard)} ({len(hard) / len(all_slots):.1%})\n")
    print(f"{'strategy':<22}{'recall, all':>14}{'recall, hard':>15}{'hard hits':>12}")
    print("-" * 63)
    for strategy in sorted(by_strategy, key=lambda s: -len(by_strategy[s][0] & hard)):
        got, _ = by_strategy[strategy]
        print(f"{strategy:<22}{len(got) / len(all_slots):>14.3f}"
              f"{len(got & hard) / len(hard):>15.3f}{len(got & hard):>8}/{len(hard)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
