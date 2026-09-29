"""Citation coverage per generator, as JSON.

``score_citations.py`` pools the campaign, which is the number the results chapter
reports. The claim underneath it is per generator: the graph channel raises coverage
on every one of the six, which is what carries the sign test. This runs the same
scorer once per run and keeps the per-generator table.

Usage::

    conda run -n graphllm python evaluation/scripts/collect_citation_coverage.py \\
        --campaign-root /srv/projects/graphllm/experiments/exp_results_fixed \\
        --out artifacts/evaluation/citation_coverage.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_results_tables import MODELS  # noqa: E402
from score_citations import score as score_citations  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Score citation coverage for every generator's run and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    out: dict[str, dict[str, float]] = {}
    for slug, name in MODELS:
        score_path = args.campaign_root / slug / "gold_score.json"
        if not score_path.exists():
            print(f"WARNING: no gold_score.json for {slug}; skipped")
            continue
        run = Path(json.loads(score_path.read_text())["run_dir"])
        report = score_citations([run / "results.jsonl"],
                                 skip_strategies={"no_retrieval"})
        out[slug] = {strategy: row["coverage"] for strategy, row in report["per_strategy"].items()}
        print(f"{name}: text_only {out[slug]['text_only']:.3f} -> "
              f"hybrid {out[slug]['hybrid']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
