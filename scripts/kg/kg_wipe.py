#!/usr/bin/env python3
"""Wipe all nodes and relationships from the configured Neo4j database.

Destructive and irreversible. Requires the ``--yes`` flag to run; without it
the script only reports the current node/relationship counts and exits.

Connection is resolved from the same env vars and config the KG pipeline uses
(``NEO4J_URL``/``NEO4J_URI``, ``NEO4J_USERNAME``/``NEO4J_USER``,
``NEO4J_PASSWORD``, ``NEO4J_DATABASE``); the database name falls back to
``neo4j.database`` in ``config.yaml``. Deletion runs in batches so a large
graph does not exhaust server memory.
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
# make it importable when this script is run directly (python scripts/kg/kg_wipe.py).
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_guard import require_hosted_target_named  # noqa: E402

from kg_pipeline.stages import neo4j_ingestion  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("kg_wipe")


def _counts(session) -> tuple[int, int]:
    """Node and relationship counts of the database."""
    nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return nodes, rels


def _delete_batch(tx, batch_size: int) -> int:
    """Detach-delete up to `batch_size` nodes; returns how many went."""
    result = tx.run(
        "MATCH (n) WITH n LIMIT $batch DETACH DELETE n RETURN count(n) AS deleted",
        batch=batch_size,
    )
    return result.single()["deleted"]


def main() -> None:
    """Report the counts and, with ``--yes``, delete everything in batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "kg_pipeline" / "config.yaml")
    )
    parser.add_argument("--env-file", default=str(ROOT / "kg_pipeline" / ".env"))
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the wipe; without it the script only reports counts.",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file, override=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    uri, user, password, env_db = neo4j_ingestion._resolve_neo4j_env()
    db = config.get("neo4j", {}).get("database") or env_db

    # `--env-file` defaults to kg_pipeline/.env, which points at the hosted
    # graph the demo serves. `--yes` says "do it"; on a remote instance it does
    # not say "to that one".
    if args.yes:
        require_hosted_target_named("KG Wipe", uri, db)

    with neo4j_env.connect(
        neo4j_env.Neo4jTarget(uri, user, password, None)
    ) as driver:
        with driver.session(database=db) as session:
            nodes, rels = _counts(session)
            LOGGER.info(
                "Target database=%s nodes=%d relationships=%d", db, nodes, rels
            )

            if not args.yes:
                LOGGER.warning(
                    "Dry run: pass --yes to delete all %d nodes and %d relationships.",
                    nodes,
                    rels,
                )
                return

            if nodes == 0:
                LOGGER.info("Database already empty, nothing to delete.")
                return

            deleted_total = 0
            while True:
                deleted = session.execute_write(_delete_batch, args.batch_size)
                if deleted == 0:
                    break
                deleted_total += deleted
                LOGGER.info("Deleted %d nodes (total %d)", deleted, deleted_total)

            nodes, rels = _counts(session)
            LOGGER.info(
                "Wipe complete. Remaining nodes=%d relationships=%d", nodes, rels
            )


if __name__ == "__main__":
    main()
