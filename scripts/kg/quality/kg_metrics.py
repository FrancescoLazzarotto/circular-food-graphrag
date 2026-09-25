"""Structural quality metrics for the knowledge graph.

Connects to a Neo4j instance, pulls the full graph and computes the
structural indicators used as regression gates before and after a repair
pass: counts, connectivity, fragmentation, name quality and provenance
coverage. Writes a JSON report and prints a summary.

Usage:
    python scripts/kg/quality/kg_metrics.py \
        --uri bolt://localhost:7689 --password staging-password \
        --output artifacts/kg_quality/metrics_staging_baseline.json

The staging instance opens bolt on **7689**
(scripts/serving/start_neo4j_staging.sh); 7688 is a different, live Neo4j.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
from kg_pipeline.utils import neo4j_env

CITATION_RE = re.compile(
    r"(et al\.?$|et al\.,|\b(19|20)\d{2}[a-z]?\)?$|^fig(ure)?\.? ?\d|^table ?\d)",
    re.IGNORECASE,
)


def compute_metrics(nodes: list[dict], edges: list[dict]) -> dict:
    """Compute the structural metric set from raw node/edge dumps."""
    label_of = {n["id"]: (n["labels"][0] if n["labels"] else "?") for n in nodes}

    graph = nx.Graph()
    graph.add_nodes_from(label_of)
    graph.add_edges_from((e["src"], e["dst"]) for e in edges)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    sizes = [len(c) for c in components]
    degrees = dict(graph.degree())

    rel_types = Counter(e["type"] for e in edges)
    total_edges = len(edges)
    names = [n["props"].get("name", "") for n in nodes]
    norm_names = Counter(re.sub(r"\s+", " ", nm.strip().lower()) for nm in names)

    per_label: dict[str, dict] = {}
    for label in sorted(set(label_of.values())):
        ids = [i for i, l in label_of.items() if l == label]
        degs = [degrees[i] for i in ids]
        per_label[label] = {
            "nodes": len(ids),
            "deg1": sum(1 for d in degs if d <= 1),
            "mean_degree": round(statistics.mean(degs), 2) if degs else 0,
        }

    return {
        "nodes": len(nodes),
        "edges": total_edges,
        "components": len(components),
        "giant_component_nodes": sizes[0] if sizes else 0,
        "giant_component_share": round(sizes[0] / len(nodes), 4) if nodes else 0,
        "pair_components": sum(1 for s in sizes if s == 2),
        "degree_le1_share": round(
            sum(1 for d in degrees.values() if d <= 1) / len(nodes), 4),
        "self_loops": sum(1 for e in edges if e["src"] == e["dst"]),
        "related_to_share": round(rel_types.get("RELATED_TO", 0) / total_edges, 4)
        if total_edges else 0,
        "duplicate_name_groups": sum(1 for c in norm_names.values() if c > 1),
        "citation_like_names": sum(1 for nm in names if CITATION_RE.search(nm)),
        "long_names_gt6_words": sum(1 for nm in names if len(nm.split()) > 6),
        "distinct_source_docs": len({e["props"].get("source_doc") for e in edges}),
        "nodes_by_label": dict(Counter(label_of.values()).most_common()),
        "edges_by_type": dict(rel_types.most_common()),
        "per_label": per_label,
    }


def main() -> int:
    """Compute the metrics of the given graph, print them, optionally write JSON.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )
    with driver.session(database=args.database) as session:
        # :NodeVec carriers hold the embeddings on separate nodes (see
        # scripts/kg/kg_vector_index.py). They have no edges, so counting them
        # would halve the giant-component share and inflate the degree-≤1
        # share.
        nodes = session.run(
            "MATCH (n) WHERE NOT n:NodeVec RETURN elementId(n) AS id, "
            "labels(n) AS labels, properties(n) AS props"
        ).data()
        edges = session.run(
            "MATCH (a)-[r]->(b) RETURN elementId(a) AS src, elementId(b) AS dst, "
            "type(r) AS type, properties(r) AS props"
        ).data()
    driver.close()

    metrics = compute_metrics(nodes, edges)
    metrics["_meta"] = {
        "uri": args.uri,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    summary_keys = [
        "nodes", "edges", "components", "giant_component_share",
        "pair_components", "degree_le1_share", "self_loops",
        "related_to_share", "duplicate_name_groups", "citation_like_names",
        "long_names_gt6_words", "distinct_source_docs",
    ]
    for key in summary_keys:
        print(f"{key}: {metrics[key]}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
