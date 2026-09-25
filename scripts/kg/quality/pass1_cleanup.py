"""Cleanup pass: self-loops and low-degree generic/anaphoric nodes.

Dry-run by default: writes CSV reports of what *would* change and touches
nothing. With ``--apply`` it deletes self-loop relationships and the generic
nodes marked ``delete`` in the report.

Usage (dry run):
    python scripts/kg/quality/pass1_cleanup.py \
        --uri bolt://localhost:7689 --password staging-password \
        --report-dir artifacts/kg_quality

The staging instance opens bolt on **7689**
(scripts/serving/start_neo4j_staging.sh); 7688 is a different, live Neo4j.
This pass deletes with ``--apply``, so a wrong port is a cleanup of the
wrong graph.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from kg_pipeline.utils import neo4j_env

# Generic/anaphoric entity names that carry no referent outside their chunk.
GENERIC_NAME_RE = re.compile(
    r"^(il |lo |la |i |gli |le |l')?(progetto|studio|study|ricerca|research|"
    r"report|autori|authors?|author|participants?|partecipanti|children|"
    r"bambini|students?|studenti|consumers?|consumatori|respondents?|paper|"
    r"articolo|article|document[oi]?|questionnaire|questionario|survey|"
    r"analisi|analysis|results?|risultati|data|dati|this study|the study|"
    r"lo studio|il presente lavoro)$",
    re.IGNORECASE,
)

# Nodes with degree above this are kept: hub anaphora are renamed by
# pass3_rename_merge.py instead, because deleting them would drop real
# information.
DELETE_MAX_DEGREE = 3


def main() -> int:
    """Report, and with ``--apply`` delete, self-loops and generic nodes.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default=None)
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/kg_quality"))
    parser.add_argument("--apply", action="store_true",
                        help="Apply deletions instead of only reporting them.")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )
    with driver.session(database=args.database) as session:
        self_loops = session.run(
            "MATCH (n)-[r]->(n) "
            "RETURN elementId(r) AS rel_id, type(r) AS rel_type, n.name AS name, "
            "labels(n)[0] AS label, r.source_doc AS source_doc"
        ).data()

        candidates = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL "
            "RETURN elementId(n) AS node_id, n.name AS name, labels(n) AS labels, "
            "COUNT { (n)--() } AS degree"
        ).data()
        generic = [
            {**c, "labels": "|".join(c["labels"]),
             "action": "delete" if c["degree"] <= DELETE_MAX_DEGREE else "keep_for_fase3"}
            for c in candidates
            if GENERIC_NAME_RE.match(c["name"].strip())
        ]

        loops_csv = args.report_dir / "pass1_selfloops.csv"
        with loops_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["rel_id", "rel_type", "name", "label", "source_doc"])
            writer.writeheader()
            writer.writerows(self_loops)

        generic_csv = args.report_dir / "pass1_generic_nodes.csv"
        with generic_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["node_id", "name", "labels", "degree", "action"])
            writer.writeheader()
            writer.writerows(sorted(generic, key=lambda g: -g["degree"]))

        to_delete = [g for g in generic if g["action"] == "delete"]
        print(f"self-loops: {len(self_loops)} -> {loops_csv}")
        print(f"generic nodes: {len(generic)} "
              f"(delete: {len(to_delete)}, kept for fase 3: "
              f"{len(generic) - len(to_delete)}) -> {generic_csv}")

        if not args.apply:
            print("dry-run only; re-run with --apply to delete")
            return 0

        deleted_rels = session.run(
            "MATCH (n)-[r]->(n) DELETE r RETURN count(*) AS c").single()["c"]
        deleted_nodes = session.run(
            "UNWIND $ids AS nid MATCH (n) WHERE elementId(n) = nid "
            "DETACH DELETE n RETURN count(*) AS c",
            ids=[g["node_id"] for g in to_delete],
        ).single()["c"]
        print(f"APPLIED: deleted {deleted_rels} self-loops, "
              f"{deleted_nodes} generic nodes")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
