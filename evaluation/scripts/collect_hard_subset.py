"""Build the hard-subset matrix for the whole campaign, as JSON.

``hard_subset.py`` answers the question for one run and prints a table. The thesis
needs the same split over all six generators at once, pooled, so the figure and the
table read from one file instead of six terminal captures. Scoring goes through
``hard_subset.matched_slots``, so a slot counted here is a slot the reported scorer
counts.

Usage::

    conda run -n graphllm python evaluation/scripts/collect_hard_subset.py \\
        --campaign-root /srv/projects/graphllm/experiments/exp_results_fixed \\
        --out artifacts/evaluation/hard_subset_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evalkit.io.dataset import build_dataset  # noqa: E402
from evalkit.metrics.mentions import Gazetteer  # noqa: E402
from hard_subset import BASELINE, matched_slots  # noqa: E402
from build_results_tables import MODELS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Build the hard-subset matrix for every generator and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=REPO / "evaluation" / "gold" / "gold_v3.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    out: dict[str, dict] = {}
    for slug, _ in MODELS:
        score = args.campaign_root / slug / "gold_score.json"
        if not score.exists():
            print(f"WARNING: no gold_score.json for {slug}; skipped")
            continue
        # The scored run is the one the scorer recorded, which for the reasoning
        # models is the <think>-stripped copy rather than the directory beside it.
        run = Path(json.loads(score.read_text())["run_dir"])
        rows = [r for r in build_dataset([run], gold_path=args.gold)
                if r.gold_query is not None and not r.is_distractor]
        if not rows:
            print(f"WARNING: no scorable rows under {run}; skipped")
            continue
        gazetteer = Gazetteer.from_gold([r.gold_query for r in rows])

        by_strategy: dict[str, tuple[set, set]] = {}
        for row in rows:
            matched, expected = matched_slots(row, gazetteer)
            got, want = by_strategy.setdefault(row.strategy, (set(), set()))
            by_strategy[row.strategy] = (got | matched, want | expected)

        known, all_slots = by_strategy[BASELINE]
        hard = all_slots - known
        out[slug] = {
            "run_dir": str(run),
            "expected": len(all_slots),
            "hard": len(hard),
            "recall_all": {s: len(g) / len(all_slots) for s, (g, _) in by_strategy.items()},
            "recall_hard": {s: len(g & hard) / len(hard) for s, (g, _) in by_strategy.items()},
            "hits_hard": {s: len(g & hard) for s, (g, _) in by_strategy.items()},
        }
        print(f"{slug}: {len(hard)} hard of {len(all_slots)} expected")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
