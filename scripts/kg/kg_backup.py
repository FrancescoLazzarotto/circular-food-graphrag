"""Dump a Neo4j knowledge graph to the JSON layout ``kg_restore.py`` expects.

The companion of ``scripts/kg/kg_restore.py``. Writes ``nodes.json``,
``edges.json``, ``schema.json`` and ``manifest.json`` into a timestamped folder.

``NodeVec`` carriers are skipped by default: they hold 768-float embeddings that
dominate the dump size and are rebuilt from the graph by
``scripts/kg/kg_vector_index.py`` anyway. Pass ``--include-vectors`` to keep them.

Usage::

    conda run -n graphllm python scripts/kg/kg_backup.py
    conda run -n graphllm python scripts/kg/kg_backup.py --output-dir artifacts/kg_backups/pre_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

from graphrag.config import build_kg_config_from_env  # noqa: E402
from graphrag.kg.manager import KnowledgeGraphManager  # noqa: E402

logger = logging.getLogger("kg_backup")

PAGE = 5000


def _paged(store: KnowledgeGraphManager, query: str, params: dict | None = None) -> list[dict]:
    """Run a query that carries ``SKIP $skip LIMIT $limit`` until it runs dry."""
    out: list[dict] = []
    skip = 0
    while True:
        rows = store.run_query(query, {**(params or {}), "skip": skip, "limit": PAGE})
        if not rows:
            break
        out.extend(rows)
        skip += PAGE
        logger.info("  %d rows", len(out))
    return out


def dump_nodes(store: KnowledgeGraphManager, include_vectors: bool) -> list[dict]:
    """Every node with labels and properties; carriers only with `include_vectors`."""
    where = "" if include_vectors else "WHERE NOT n:NodeVec "
    query = (
        f"MATCH (n) {where}"
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
        "ORDER BY id SKIP $skip LIMIT $limit"
    )
    return _paged(store, query)


def dump_edges(store: KnowledgeGraphManager) -> list[dict]:
    """Every relationship with its endpoints' ids, type and properties."""
    query = (
        "MATCH (a)-[r]->(b) "
        "RETURN elementId(a) AS src, elementId(b) AS dst, type(r) AS type, "
        "properties(r) AS props ORDER BY src, dst SKIP $skip LIMIT $limit"
    )
    return _paged(store, query)


def dump_schema(store: KnowledgeGraphManager) -> dict:
    """The graph's indexes and constraints."""
    indexes = store.run_query(
        "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options "
        "RETURN name, type, entityType, labelsOrTypes, properties, options"
    )
    constraints = store.run_query(
        "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties "
        "RETURN name, type, entityType, labelsOrTypes, properties"
    )
    return {"indexes": indexes, "constraints": constraints}


def main(argv: list[str] | None = None) -> int:
    """Dump the graph into a backup folder.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="target folder (default: timestamped)")
    parser.add_argument("--include-vectors", action="store_true", help="keep :NodeVec carriers")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(REPO / ".env", override=False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir or REPO / "artifacts" / "kg_backups" / stamp)
    out.mkdir(parents=True, exist_ok=True)

    cfg = build_kg_config_from_env()
    store = KnowledgeGraphManager(cfg)

    logger.info("dumping nodes")
    nodes = dump_nodes(store, args.include_vectors)
    logger.info("dumping edges")
    edges = dump_edges(store)
    logger.info("dumping schema")
    schema = dump_schema(store)

    (out / "nodes.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
    (out / "edges.json").write_text(json.dumps(edges, ensure_ascii=False), encoding="utf-8")
    (out / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, default=str), encoding="utf-8")

    label_counts = Counter(lbl for n in nodes for lbl in n["labels"])
    type_counts = Counter(e["type"] for e in edges)
    manifest = {
        "timestamp": stamp,
        "uri": getattr(cfg, "url", None) or getattr(cfg, "uri", None),
        "database": getattr(cfg, "database", None),
        "nodes": len(nodes),
        "relationships": len(edges),
        "indexes": len(schema["indexes"]),
        "constraints": len(schema["constraints"]),
        "include_vectors": args.include_vectors,
        "labels": dict(label_counts),
        "relationship_types": dict(type_counts),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    logger.info("backup written to %s: %d nodes, %d edges", out, len(nodes), len(edges))
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
