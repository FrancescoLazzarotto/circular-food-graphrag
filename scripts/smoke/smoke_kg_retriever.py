"""Smoke test of the KG retriever against the configured graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from graphrag.config import AgentConfig, build_kg_config_from_env
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.kg.retriever import KGRetriever


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(description="Smoke test for KG retriever behavior")
    parser.add_argument(
        "--question", default="Quali sono le relazioni tra Entita A e Entita B?"
    )
    parser.add_argument("--entity", default="")
    parser.add_argument("--entity-a", default="")
    parser.add_argument("--entity-b", default="")
    parser.add_argument("--labels", default="", help="Comma-separated labels filter")
    parser.add_argument(
        "--relationship-types",
        default="",
        help="Comma-separated relationship type filter",
    )

    parser.add_argument("--min-nodes", type=int, default=0)
    parser.add_argument("--min-triples", type=int, default=1)
    parser.add_argument("--min-neighbors", type=int, default=1)
    parser.add_argument("--min-subgraph", type=int, default=1)
    parser.add_argument("--min-shortest-path", type=int, default=0)
    parser.add_argument("--min-total-hits", type=int, default=3)

    parser.add_argument("--nodes-limit", type=int, default=10)
    parser.add_argument("--triples-limit", type=int, default=20)
    parser.add_argument("--neighbors-limit", type=int, default=25)
    parser.add_argument("--subgraph-limit", type=int, default=200)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=6)

    return parser


def _parse_csv(raw_value: str) -> tuple[str, ...]:
    """The non-empty items of a comma-separated value."""
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def main() -> int:
    """Retrieve for one question and check every channel's hit count.

    Returns:
        The exit status: 0 on pass, 1 on a failed check or runtime error, 2 on
        a configuration error.
    """
    args = _build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)

    try:
        kg_manager = KnowledgeGraphManager(build_kg_config_from_env())
    except ValueError as exc:
        print("KG RETRIEVER SMOKE FAILED")
        print(f"- config/env error: {exc}")
        return 2

    include_shortest_path = (
        bool(args.entity_a and args.entity_b) or args.min_shortest_path > 0
    )

    config = AgentConfig(
        query=args.question,
        entity=args.entity,
        entity_a=args.entity_a or None,
        entity_b=args.entity_b or None,
        hops=max(1, int(args.hops)),
        max_depth=max(1, int(args.max_depth)),
        nodes_limit=max(1, int(args.nodes_limit)),
        triples_limit=max(1, int(args.triples_limit)),
        neighbors_limit=max(1, int(args.neighbors_limit)),
        subgraph_limit=max(1, int(args.subgraph_limit)),
        labels=_parse_csv(args.labels),
        relationship_types=_parse_csv(args.relationship_types),
        include_nodes=True,
        include_triples=True,
        include_neighbors=True,
        include_subgraph=True,
        include_shortest_path=include_shortest_path,
    )

    retriever = KGRetriever(kg_store=kg_manager, config=config)

    try:
        result = retriever.retrieve(config.query)
    except Exception as exc:
        print("KG RETRIEVER SMOKE FAILED")
        print(f"- runtime error: {type(exc).__name__}: {exc}")
        return 1

    counts = {
        "nodes": len(result.get("nodes", [])),
        "triples": len(result.get("triples", [])),
        "neighbors": len(result.get("neighbors", [])),
        "subgraph": len(result.get("subgraph", [])),
        "shortest_path": len(result.get("shortest_path", [])),
    }
    total_hits = counts["triples"] + counts["neighbors"] + counts["subgraph"]

    print("KG RETRIEVER SMOKE RESULT")
    print(f"- query: {args.question}")
    print(f"- resolved_entity: {result.get('entity')}")
    print(f"- search_terms: {result.get('search_terms', [])}")
    print(f"- counts: {counts}")
    print(f"- total_hits(triples+neighbors+subgraph): {total_hits}")

    failures: list[str] = []
    thresholds = {
        "nodes": max(0, int(args.min_nodes)),
        "triples": max(0, int(args.min_triples)),
        "neighbors": max(0, int(args.min_neighbors)),
        "subgraph": max(0, int(args.min_subgraph)),
        "shortest_path": max(0, int(args.min_shortest_path)),
    }

    for key, minimum in thresholds.items():
        if counts[key] < minimum:
            failures.append(f"{key}: expected >= {minimum}, got {counts[key]}")

    min_total_hits = max(0, int(args.min_total_hits))
    if total_hits < min_total_hits:
        failures.append(f"total_hits: expected >= {min_total_hits}, got {total_hits}")

    if not str(result.get("context_text", "")).strip():
        failures.append("context_text is empty")

    if failures:
        print("KG RETRIEVER SMOKE FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    print("KG RETRIEVER SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
