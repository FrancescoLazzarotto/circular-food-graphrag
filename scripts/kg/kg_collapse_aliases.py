#!/usr/bin/env python3
"""Collapse alias nodes produced by the linking stage into their canonical nodes.

The linking stage materialises every registry alias as its own node with a
``(alias)-[:SAME_AS]->(canonical)`` relationship. Those alias nodes are leaves
that shadow the canonical entity at query time: a lexical match lands on the
alias (whose only relationship, SAME_AS, is ranked down as a system link) and
never reaches the canonical node that holds the real relationships.

This pass rewrites that structure in place:

1. every alias name is appended to the ``aliases`` list property of its
   canonical node (deduplicated, canonical's own name excluded);
2. all SAME_AS relationships are deleted;
3. former alias nodes left with no relationships are deleted. Alias nodes that
   also carry content relationships of their own, or that are themselves the
   canonical target of another alias, are kept and reported.

Idempotent: a second run finds no SAME_AS relationships and changes nothing.
Destructive only under ``--yes``; without it the script reports what it would
do and exits. The graph can always be rebuilt from ``stage5_triples_linked.json``.

Connection env resolution matches the KG pipeline (``NEO4J_URL``,
``NEO4J_USERNAME``, ``NEO4J_PASSWORD``, ``NEO4J_DATABASE``; database falls back
to ``neo4j.database`` in ``config.yaml``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from kg_pipeline.utils import neo4j_env

ROOT = Path(__file__).resolve().parents[2]
# The kg_pipeline package lives at the repo root and is not pip-installed, so
# make it importable when this script is run directly.
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_guard import require_hosted_target_named  # noqa: E402

from kg_pipeline.stages import neo4j_ingestion  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("kg_collapse_aliases")


def _counts(session) -> tuple[int, int]:
    """Node and relationship counts of the database."""
    nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return nodes, rels


def _collect_same_as(session) -> list[dict[str, Any]]:
    """Return one row per SAME_AS relationship with alias/canonical identity.

    ``alias_content_degree`` counts the alias node's non-SAME_AS relationships;
    ``alias_is_target`` flags alias nodes that other aliases point to.
    """
    result = session.run(
        """
        MATCH (a)-[s:SAME_AS]->(c)
        RETURN elementId(a) AS alias_id,
               a.name AS alias_name,
               elementId(c) AS canonical_id,
               c.name AS canonical_name,
               COUNT { (a)-[r]-() WHERE type(r) <> 'SAME_AS' } AS alias_content_degree,
               EXISTS { (a)<-[:SAME_AS]-() } AS alias_is_target
        """
    )
    return [r.data() for r in result]


def _merge_alias_names(tx, rows: list[dict[str, Any]]) -> None:
    """Fold the alias names into each canonical node's ``aliases``."""
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (c) WHERE elementId(c) = row.canonical_id
        SET c.aliases = [x IN coalesce(c.aliases, []) WHERE NOT x IN row.names]
                        + [x IN row.names WHERE x <> c.name]
        """,
        rows=rows,
    )


def _delete_same_as_batch(tx, batch_size: int) -> int:
    """Delete up to `batch_size` SAME_AS relationships; returns how many."""
    result = tx.run(
        """
        MATCH ()-[s:SAME_AS]->()
        WITH s LIMIT $batch
        DELETE s
        RETURN count(s) AS deleted
        """,
        batch=batch_size,
    )
    return result.single()["deleted"]


def _delete_orphan_aliases(tx, ids: list[str]) -> int:
    """Delete the listed nodes that have no relationship left."""
    result = tx.run(
        """
        UNWIND $ids AS id
        MATCH (a) WHERE elementId(a) = id AND COUNT { (a)--() } = 0
        DELETE a
        RETURN count(a) AS deleted
        """,
        ids=ids,
    )
    return result.single()["deleted"]


def main() -> None:
    """Report the SAME_AS structure and, with ``--yes``, collapse it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "kg_pipeline" / "config.yaml"))
    parser.add_argument("--env-file", default=str(ROOT / "kg_pipeline" / ".env"))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply the collapse; without it the script only reports counts.",
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
        require_hosted_target_named("KG Collapse Aliases", uri, db)

    with neo4j_env.connect(
        neo4j_env.Neo4jTarget(uri, user, password, None)
    ) as driver:
        with driver.session(database=db) as session:
            nodes, rels = _counts(session)
            LOGGER.info(
                "Target database=%s nodes=%d relationships=%d", db, nodes, rels
            )

            pairs = _collect_same_as(session)
            if not pairs:
                LOGGER.info("No SAME_AS relationships found — nothing to collapse.")
                return

            # Alias nodes safe to delete: pure leaves that are not themselves
            # the canonical target of another alias.
            deletable = {
                p["alias_id"]
                for p in pairs
                if p["alias_content_degree"] == 0 and not p["alias_is_target"]
            }
            kept = [
                p
                for p in pairs
                if p["alias_content_degree"] > 0 or p["alias_is_target"]
            ]
            canonical_ids = {p["canonical_id"] for p in pairs}

            LOGGER.info(
                "Found %d SAME_AS relationships: %d canonical nodes, "
                "%d alias nodes to delete, %d alias nodes kept (own content "
                "or canonical target).",
                len(pairs),
                len(canonical_ids),
                len(deletable),
                len(kept),
            )
            for p in kept[:20]:
                LOGGER.info(
                    "  kept alias %r (content_degree=%d, is_target=%s) -> %r",
                    p["alias_name"],
                    p["alias_content_degree"],
                    p["alias_is_target"],
                    p["canonical_name"],
                )

            if not args.yes:
                LOGGER.warning("Dry run: pass --yes to apply the collapse.")
                return

            # Phase 1: fold alias names into the canonical nodes' aliases list,
            # grouped per canonical so one UNWIND row carries all its names.
            by_canonical: dict[str, list[str]] = {}
            for p in pairs:
                if p["alias_id"] == p["canonical_id"]:
                    continue
                names = by_canonical.setdefault(p["canonical_id"], [])
                if p["alias_name"] and p["alias_name"] not in names:
                    names.append(p["alias_name"])
            rows = [
                {"canonical_id": cid, "names": names}
                for cid, names in by_canonical.items()
            ]
            for i in range(0, len(rows), args.batch_size):
                session.execute_write(_merge_alias_names, rows[i : i + args.batch_size])
            LOGGER.info("Updated aliases property on %d canonical nodes.", len(rows))

            # Phase 2: drop all SAME_AS relationships.
            deleted_rels = 0
            while True:
                deleted = session.execute_write(_delete_same_as_batch, args.batch_size)
                if deleted == 0:
                    break
                deleted_rels += deleted
            LOGGER.info("Deleted %d SAME_AS relationships.", deleted_rels)

            # Phase 3: delete former alias nodes now left without relationships.
            ids = sorted(deletable)
            deleted_nodes = 0
            for i in range(0, len(ids), args.batch_size):
                deleted_nodes += session.execute_write(
                    _delete_orphan_aliases, ids[i : i + args.batch_size]
                )
            LOGGER.info("Deleted %d orphaned alias nodes.", deleted_nodes)

            nodes, rels = _counts(session)
            LOGGER.info(
                "Collapse complete. nodes=%d relationships=%d", nodes, rels
            )


if __name__ == "__main__":
    main()
