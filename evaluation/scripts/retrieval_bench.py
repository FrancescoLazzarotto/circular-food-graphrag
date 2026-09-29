"""Retrieval-only benchmark: does the assembled context contain the gold terms?

No generator is involved, so one configuration over the gold set takes seconds
instead of the ~25 minutes a full generation matrix needs per model. That makes
it the loop to tune retrieval against; generation is only worth running once a
configuration has won here.

The metric is deliberately channel-agnostic. ``score_gold_run.py`` scores the
retrieval channel over *entities*, so a text-only pipeline scores 0 there by
construction and graph and text cannot be compared. Here the check is whether an
accepted surface form of a gold entity occurs anywhere in the context the
generator would have received, which both channels can satisfy.

Reported per strategy:

* ``ctx_recall`` — share of gold slots whose surface form is in the context.
  This is the ceiling on what any generator can ground.
* ``terms`` / ``entities`` / ``chars`` — search terms issued, entities returned,
  context size, to see *why* a number moved.

Usage::

    python evaluation/scripts/retrieval_bench.py --label baseline
    python evaluation/scripts/retrieval_bench.py --label lexical --lexical-specificity
    python evaluation/scripts/retrieval_bench.py --compare baseline lexical
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

from graphrag.config import AgentConfig, build_kg_config_from_env  # noqa: E402
from graphrag.kg.manager import KnowledgeGraphManager  # noqa: E402
from graphrag.kg.retriever import KGRetriever  # noqa: E402
from graphrag.strategies import STRATEGY_PRESETS, apply_strategy  # noqa: E402
from graphrag.text_rag.factory import make_text_pipeline  # noqa: E402

DEFAULT_GOLD = REPO / "evaluation" / "gold" / "gold_v3.json"
DEFAULT_DOCS = REPO / "artifacts" / "corpus_circular22"
DEFAULT_OUT = REPO / "artifacts" / "retrieval_bench"

logger = logging.getLogger("retrieval_bench")


# --------------------------------------------------------------------------- #
# gold handling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GoldSlot:
    """One (question, expected entity) pair, with every accepted surface form."""

    query_id: str
    query: str
    label: str
    forms: tuple[str, ...]


def load_gold(path: Path, include_distractors: bool = False) -> list[GoldSlot]:
    """Gold slots with their accepted surface forms, lowercased."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    slots: list[GoldSlot] = []
    for query in payload["queries"]:
        if not include_distractors and query["scoring"].get("distractor_expected"):
            continue
        for entity in query["expected_entities"]:
            forms = {entity["normalised_label"], entity["label"]}
            forms.update(entity.get("alt_labels") or [])
            cleaned = tuple(
                sorted({form.lower().strip() for form in forms if form and form.strip()})
            )
            if not cleaned:
                continue
            slots.append(
                GoldSlot(
                    query_id=query["query_id"],
                    query=query["query"],
                    label=entity["label"],
                    forms=cleaned,
                )
            )
    return slots


def questions(slots: Sequence[GoldSlot]) -> list[tuple[str, str]]:
    """Unique (query_id, query) pairs, in gold order."""
    seen: dict[str, str] = {}
    for slot in slots:
        seen.setdefault(slot.query_id, slot.query)
    return list(seen.items())


def form_in_text(text: str, forms: Iterable[str]) -> bool:
    """Word-boundary match of any accepted surface form.

    Substring matching would count "capital" inside "capitale" and inflate every
    cross-lingual number this script exists to measure.
    """
    lowered = text.lower()
    return any(
        re.search(rf"(?<!\w){re.escape(form)}(?!\w)", lowered) for form in forms
    )


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #


def build_base_config(args: argparse.Namespace) -> AgentConfig:
    """Baseline mirrors the thesis campaign, minus everything generator-side."""
    return AgentConfig(
        include_nodes=True,
        include_triples=True,
        include_neighbors=True,
        include_subgraph=True,
        include_shortest_path=True,
        max_content_tokens=args.max_context_tokens,
        cite_evidence=True,
        citation_display="label",
        prefer_verbatim_definitions=True,
        text_retriever_mmr=True,
        text_retriever_mmr_lambda=0.7,
        text_retriever_max_per_doc=2,
        text_retriever_backend=args.text_retriever_backend,
        lexical_specificity=args.lexical_specificity,
        lexical_df_max_ratio=args.lexical_df_max_ratio,
        lexical_phrase_boost=args.lexical_phrase_boost,
        seed_from_retrieved=args.seed_from_retrieved,
        subgraph_seed_count=args.subgraph_seed_count,
        vector_retrieval=args.vector_retrieval,
        vector_nodes_limit=args.vector_nodes_limit,
        vector_triples_limit=args.vector_triples_limit,
        vector_seed_limit=args.vector_seed_limit,
    )


def run_strategy(
    kg_store: KnowledgeGraphManager,
    base: AgentConfig,
    strategy: str,
    pairs: Sequence[tuple[str, str]],
    slots_by_query: dict[str, list[GoldSlot]],
    text_pipeline: Any | None,
) -> dict[str, Any]:
    """Retrieve every question under one strategy and score the contexts."""
    config = apply_strategy(base, strategy)
    retriever = KGRetriever(
        kg_store=kg_store,
        config=config,
        text_pipeline=text_pipeline if config.use_text_retriever else None,
    )

    hits = 0
    total = 0
    per_question: list[dict[str, Any]] = []
    term_counts: list[int] = []
    entity_counts: list[int] = []
    char_counts: list[int] = []
    started = time.perf_counter()

    for query_id, query in pairs:
        result = retriever.retrieve(query)
        context = result.get("context_text") or ""
        # The context echoes the question; a gold form appearing only there was
        # never retrieved and must not be credited.
        context = re.sub(r"^Query:.*$", "", context, count=1, flags=re.MULTILINE)

        found: list[str] = []
        missed: list[str] = []
        for slot in slots_by_query.get(query_id, []):
            total += 1
            if form_in_text(context, slot.forms):
                hits += 1
                found.append(slot.label)
            else:
                missed.append(slot.label)

        term_counts.append(len(result.get("search_terms") or []))
        entity_counts.append(len(result.get("nodes") or []))
        char_counts.append(len(context))
        per_question.append(
            {
                "query_id": query_id,
                "search_terms": list(result.get("search_terms") or []),
                "resolved_entity": result.get("entity"),
                "found": found,
                "missed": missed,
                "context_chars": len(context),
            }
        )

    return {
        "strategy": strategy,
        "ctx_recall": hits / total if total else 0.0,
        "hits": hits,
        "slots": total,
        "avg_search_terms": statistics.mean(term_counts) if term_counts else 0.0,
        "avg_entities": statistics.mean(entity_counts) if entity_counts else 0.0,
        "avg_context_chars": statistics.mean(char_counts) if char_counts else 0.0,
        "elapsed_s": time.perf_counter() - started,
        "per_question": per_question,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def print_table(report: dict[str, Any]) -> None:
    """Print one report as a table, one row per strategy."""
    header = (
        f"{'strategy':18} {'ctx recall':>10} {'hit/slot':>10} "
        f"{'terms':>6} {'entita':>7} {'chars':>7} {'s':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in report["strategies"]:
        print(
            f"{row['strategy']:18} {row['ctx_recall']:10.3f} "
            f"{row['hits']:5}/{row['slots']:<4} {row['avg_search_terms']:6.1f} "
            f"{row['avg_entities']:7.1f} {row['avg_context_chars']:7.0f} "
            f"{row['elapsed_s']:6.1f}"
        )


def compare(out_dir: Path, labels: Sequence[str]) -> None:
    """Print ctx_recall per strategy across saved reports, last minus first."""
    reports = []
    for label in labels:
        path = out_dir / f"{label}.json"
        if not path.exists():
            raise SystemExit(f"missing report: {path}")
        reports.append((label, json.loads(path.read_text(encoding="utf-8"))))

    by_strategy: dict[str, dict[str, float]] = {}
    for label, report in reports:
        for row in report["strategies"]:
            by_strategy.setdefault(row["strategy"], {})[label] = row["ctx_recall"]

    head = f"{'strategy':18} " + " ".join(f"{label:>10}" for label, _ in reports)
    if len(reports) > 1:
        head += f" {'delta':>8}"
    print(head)
    print("-" * len(head))
    for strategy, values in by_strategy.items():
        cells = " ".join(f"{values.get(label, float('nan')):10.3f}" for label, _ in reports)
        line = f"{strategy:18} {cells}"
        if len(reports) > 1:
            first, last = reports[0][0], reports[-1][0]
            if first in values and last in values:
                line += f" {values[last] - values[first]:+8.3f}"
        print(line)


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="baseline", help="report name under --out-dir")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--strategies",
        default="default,hybrid,text_plus_triples,neighbors_focus,"
        "subgraph_2hop,shortest_path,text_only,no_retrieval",
    )
    parser.add_argument("--text-docs-dir", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--text-retriever-backend", default="tfidf", choices=("tfidf", "dense"))
    parser.add_argument("--max-context-tokens", type=int, default=6000)
    parser.add_argument("--include-distractors", action="store_true")
    parser.add_argument(
        "--lexical-specificity",
        action="store_true",
        help="P1: drop over-frequent query tokens and boost terms by rarity",
    )
    parser.add_argument(
        "--seed-from-retrieved",
        action="store_true",
        help="P1: anchor neighbours/subgraph/path on retrieved node names, not query words",
    )
    parser.add_argument(
        "--vector-retrieval",
        action="store_true",
        help="P0: add the multilingual vector channel beside the lexical one",
    )
    parser.add_argument("--vector-nodes-limit", type=int, default=10)
    parser.add_argument("--vector-triples-limit", type=int, default=10)
    parser.add_argument("--vector-seed-limit", type=int, default=5)
    parser.add_argument("--subgraph-seed-count", type=int, default=1)
    parser.add_argument("--lexical-df-max-ratio", type=float, default=0.01)
    parser.add_argument("--lexical-phrase-boost", type=float, default=4.0)
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="LABEL",
        help="print a comparison of existing reports and exit",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark, or compare saved reports with ``--compare``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    load_dotenv(REPO / ".env", override=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        compare(args.out_dir, args.compare)
        return 0

    slots = load_gold(args.gold, include_distractors=args.include_distractors)
    pairs = questions(slots)
    slots_by_query: dict[str, list[GoldSlot]] = {}
    for slot in slots:
        slots_by_query.setdefault(slot.query_id, []).append(slot)
    print(f"gold: {len(pairs)} domande, {len(slots)} slot\n")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGY_PRESETS]
    if unknown:
        raise SystemExit(f"unknown strategies: {unknown}")

    base = build_base_config(args)
    needs_text = any(
        apply_strategy(base, strategy).use_text_retriever for strategy in strategies
    )
    text_pipeline = None
    if needs_text:
        text_pipeline = make_text_pipeline(
            backend=args.text_retriever_backend,
            embedding_model=base.dense_embedding_model,
            vector_index_dir=base.vector_index_dir,
        )
        indexed = text_pipeline.index_directory(args.text_docs_dir)
        print(f"text pipeline: {indexed} chunk da {args.text_docs_dir}\n")

    kg_config = build_kg_config_from_env()
    kg_store = KnowledgeGraphManager(kg_config)
    rows = [
        run_strategy(kg_store, base, strategy, pairs, slots_by_query, text_pipeline)
        for strategy in strategies
    ]

    report = {
        "label": args.label,
        "gold": str(args.gold),
        "questions": len(pairs),
        "slots": len(slots),
        "base_config": {
            k: (v.value if hasattr(v, "value") else v)
            for k, v in asdict(base).items()
            if not k.endswith("_prompt")
        },
        "strategies": rows,
    }
    out_path = args.out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_table(report)
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
