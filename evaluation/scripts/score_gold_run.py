"""Score one experiment run against the gold, both channels, both levels.

Entry point for the gold entity evaluation:

* **retrieval channel** — ``retrieved_entities`` as reported by the run: what
  the retriever surfaced from the KG. text-RAG reports none by design.
* **answer channel** — gold surface forms found in the generated answer text
  by the deterministic gazetteer (``evalkit.metrics.mentions``): what the
  answer actually says. Symmetric across pipelines; the only channel where
  text-RAG and no-retrieval can score at all.

Both channels go through the identical path: shared normalisation, shared
resolver, two levels never merged. Results are written as JSON (full counts)
and a compact Markdown table.

Usage:
    python evaluation/scripts/score_gold_run.py \\
        --run-dir artifacts/experiments/<run>/ \\
        --gold evaluation/gold/gold.json \\
        --out-prefix artifacts/evaluation/<name>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from evalkit.io.dataset import build_dataset  # noqa: E402
from evalkit.io.gold_loader import load_gold_json  # noqa: E402
from evalkit.metrics.entities import (  # noqa: E402
    LevelSummary,
    aggregate,
    level_gaps,
    score_row,
)
from evalkit.metrics.mentions import Gazetteer, answer_channel_row  # noqa: E402
from evalkit.metrics.resolver import Resolver  # noqa: E402

logger = logging.getLogger("evalkit.score_gold_run")


def _fmt(value: float | None) -> str:
    """Three decimals, or a dash for None."""
    return f"{value:.3f}" if value is not None else "—"


def _summary_dict(summary: LevelSummary) -> dict[str, Any]:
    """A LevelSummary as a JSON-ready dict, micro P/R/F1 included."""
    out: dict[str, Any] = {"keys": summary.keys, "n_rows": summary.n_rows}
    for name in ("concept_micro", "grounding_micro"):
        prf = getattr(summary, name)
        out[name] = (
            None
            if prf is None
            else {
                **dataclasses.asdict(prf),
                "precision": prf.precision,
                "recall": prf.recall,
                "f1": prf.f1,
            }
        )
    for name in ("concept_macro", "grounding_macro"):
        macro = getattr(summary, name)
        out[name] = None if macro is None else dataclasses.asdict(macro)
    out["abstention_rate"] = summary.abstention_rate
    out["n_distractor_rows"] = summary.n_distractor_rows
    return out


def _channel_block(rows: list, resolver: Resolver) -> dict[str, Any]:
    """One channel scored by pipeline, by pipeline and type, plus gaps."""
    scores = [score_row(r, resolver) for r in rows if r.gold_query is not None]
    return {
        "by_pipeline": [
            _summary_dict(s) for s in aggregate(scores, by=("pipeline",))
        ],
        "by_pipeline_and_type": [
            _summary_dict(s)
            for s in aggregate(scores, by=("pipeline", "query_type"))
        ],
        "level_gaps": [dataclasses.asdict(g) for g in level_gaps(scores)],
    }


def _markdown(report: dict[str, Any]) -> str:
    """Compact Markdown report of both channels."""
    lines = [
        "# Gold entity scoring — dual channel",
        "",
        f"Run: `{report['run_dir']}`  ",
        f"Gold: `{report['gold']}` (sha256 `{report.get('gold_sha256', '?')[:12]}…`)",
        "",
        "Channels: **retrieval** = entities the retriever surfaced; "
        "**answer** = gold surface forms found in the generated answer "
        "(gazetteer, symmetric across pipelines). Levels per protocol §2, "
        "never merged.",
        "",
    ]
    for channel in ("retrieval", "answer"):
        lines += [f"## {channel} channel", ""]
        lines += [
            "| pipeline | rows | concept P | concept R | concept F1 "
            "| grounding P | grounding R | grounding F1 | abstention |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in report[channel]["by_pipeline"]:
            c = s["concept_micro"] or {}
            g = s["grounding_micro"] or {}
            lines.append(
                f"| {s['keys'].get('pipeline', '?')} | {s['n_rows']} "
                f"| {_fmt(c.get('precision'))} | {_fmt(c.get('recall'))} "
                f"| {_fmt(c.get('f1'))} | {_fmt(g.get('precision'))} "
                f"| {_fmt(g.get('recall'))} | {_fmt(g.get('f1'))} "
                f"| {_fmt(s.get('abstention_rate'))} |"
            )
        lines += ["", "### Level gaps (concept − grounding)", ""]
        lines += [
            "| pipeline | concept F1 | grounding F1 | gap (literal §6) "
            "| gap like-for-like | unresolved | ambiguous |",
            "|---|---|---|---|---|---|---|",
        ]
        for gap in report[channel]["level_gaps"]:
            lines.append(
                f"| {gap['pipeline']} | {_fmt(gap['concept_f1'])} "
                f"| {_fmt(gap['grounding_f1'])} | {_fmt(gap['f1_gap'])} "
                f"| {_fmt(gap['f1_gap_like_for_like'])} "
                f"| {gap['n_unresolved']} | {gap['n_ambiguous']} |"
            )
        lines.append("")
    lines += [
        "> Answer-channel precision is measured against the gold vocabulary "
        "only (a gazetteer cannot see out-of-gold mentions); recall is the "
        "meaningful direction there. Abstention uses the deterministic "
        "conjunct plus the lexical fabrication fallback unless a judge-backed "
        "check was injected — see `evalkit.metrics.entities.abstention`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Parse arguments, score the run on both channels, write JSON and Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=REPO_ROOT / "evaluation" / "gold" / "gold.json")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="writes <prefix>.json and <prefix>.md",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import hashlib

    gold_sha = hashlib.sha256(args.gold.read_bytes()).hexdigest()

    gold = load_gold_json(args.gold)
    gazetteer = Gazetteer.from_gold(list(gold.values()))
    resolver = Resolver.from_gold(args.gold)

    rows = build_dataset([args.run_dir], gold_path=args.gold)
    joined = [r for r in rows if r.gold_query is not None]
    if len(joined) < len(rows):
        logger.warning(
            "%d/%d rows did not join the gold and are excluded",
            len(rows) - len(joined),
            len(rows),
        )
    if not joined:
        logger.error("no rows joined the gold — nothing to score")
        return 1

    answer_rows = [answer_channel_row(r, gazetteer) for r in joined]

    report: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "gold": str(args.gold),
        "gold_sha256": gold_sha,
        "n_rows": len(joined),
        "retrieval": _channel_block(joined, resolver),
        "answer": _channel_block(answer_rows, resolver),
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    md_path = args.out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md_path.write_text(_markdown(report))
    logger.info("wrote %s and %s", json_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
