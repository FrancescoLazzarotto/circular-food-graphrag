#!/usr/bin/env python3
"""Maintain the full-text search index over node names and aliases.

Neo4j full-text indexes do not index list properties, so the ``aliases`` array
written by ``kg_collapse_aliases.py`` is invisible to Lucene. This script
materialises a scalar ``search_text`` property on every node (``name`` plus all
aliases, newline-joined) and (re)creates the ``node_search`` full-text index on
``[name, search_text]`` across all labels present in the database. Indexing
``name`` alongside ``search_text`` intentionally double-weights exact name
matches over alias matches.

Idempotent: ``search_text`` is recomputed for every node on each run and the
index is created with ``IF NOT EXISTS``. Re-run after any change to node names
or aliases (e.g. a new ingestion or a new collapse pass). If new labels appear
after index creation, pass ``--recreate`` so the index covers them.

Connection env resolution matches the KG pipeline (``NEO4J_URL``,
``NEO4J_USERNAME``, ``NEO4J_PASSWORD``, ``NEO4J_DATABASE``; database falls back
to ``neo4j.database`` in ``config.yaml``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from kg_pipeline.utils import neo4j_env

ROOT = Path(__file__).resolve().parents[2]
# The kg_pipeline package lives at the repo root and is not pip-installed, so
# make it importable when this script is run directly.
sys.path.insert(0, str(ROOT))

from kg_pipeline.stages import neo4j_ingestion  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("kg_search_index")

INDEX_NAME = "node_search"


def _refresh_search_text(session, batch_size: int) -> int:
    """Recompute ``search_text`` on every node, in batches; return node count."""
    total = 0
    while True:
        result = session.run(
            """
            MATCH (n)
            WHERE n.name IS NOT NULL AND (
                n.search_text IS NULL
                OR n.search_text <> reduce(
                    acc = toString(n.name), a IN coalesce(n.aliases, []) | acc + '\n' + a
                )
            )
            WITH n LIMIT $batch
            SET n.search_text = reduce(
                acc = toString(n.name), a IN coalesce(n.aliases, []) | acc + '\n' + a
            )
            RETURN count(n) AS updated
            """,
            batch=batch_size,
        )
        updated = result.single()["updated"]
        if updated == 0:
            return total
        total += updated


def _index_labels(session) -> list[str]:
    """Labels that carry at least one node.

    ``db.labels()`` also returns tokens that older schemas left with no node.
    """
    rows = session.run(
        """
        CALL db.labels() YIELD label
        CALL (label) {
            MATCH (n) WHERE label IN labels(n) RETURN count(n) AS c
        }
        RETURN label, c
        """
    )
    return [r["label"] for r in rows if r["c"] > 0]


def main() -> None:
    """Refresh ``search_text`` and index it, with ``name``, over the labels in use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "kg_pipeline" / "config.yaml"))
    parser.add_argument("--env-file", default=str(ROOT / "kg_pipeline" / ".env"))
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the index (needed when new labels appear).",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file, override=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    uri, user, password, env_db = neo4j_ingestion._resolve_neo4j_env()
    db = config.get("neo4j", {}).get("database") or env_db

    with neo4j_env.connect(
        neo4j_env.Neo4jTarget(uri, user, password, None)
    ) as driver:
        with driver.session(database=db) as session:
            updated = _refresh_search_text(session, args.batch_size)
            LOGGER.info("search_text refreshed on %d nodes.", updated)

            labels = _index_labels(session)
            if not labels:
                LOGGER.warning("No labelled nodes found — index not created.")
                return

            if args.recreate:
                session.run(f"DROP INDEX {INDEX_NAME} IF EXISTS").consume()
                LOGGER.info("Dropped index %s.", INDEX_NAME)

            spec = "|".join(f"`{label}`" for label in labels)
            session.run(
                f"CREATE FULLTEXT INDEX {INDEX_NAME} IF NOT EXISTS "
                f"FOR (n:{spec}) ON EACH [n.name, n.search_text]"
            ).consume()
            session.run(
                "CALL db.awaitIndex($name, 300)", name=INDEX_NAME
            ).consume()
            LOGGER.info(
                "Index %s online over %d labels: %s",
                INDEX_NAME,
                len(labels),
                ", ".join(labels),
            )


if __name__ == "__main__":
    main()
