"""Delete anaphoric hub nodes, or nodes left without any relationship.

``--anaphoric``: in the case-study pages the model writes "the project" or
"the initiative" instead of the name, so every initiative of the corpus lands
on one node that joins hundreds of unrelated things. Their edges say nothing
about which initiative, and re-anchoring them to the project of the same chunk
gets most of them wrong; deleting the nodes leaves the gold-slot recall
unchanged and shortens the context. Their edges are written to ``--log``
before deletion.

``--isolated``: nodes with no relationship at all (alias leftovers, nodes
emptied by merges and edge rules). Vector carriers (``:NodeVec``) are kept.

Without ``--apply`` it only counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j import GraphDatabase

ANAPHORIC = [
    "progetto", "il progetto", "questo progetto",
    "iniziativa", "l'iniziativa", "l’iniziativa",
    "azienda", "l'azienda", "l’azienda", "la cooperativa",
    "project", "the project", "company", "startup",
]


def main() -> None:
    """Count, and with ``--apply`` delete, the anaphoric or the isolated nodes."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--anaphoric", action="store_true")
    what.add_argument("--isolated", action="store_true")
    p.add_argument("--uri", default="bolt://localhost:7689")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="staging-kg-v2")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--log", type=Path, help="JSONL of the edges removed with --anaphoric")
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()

    with GraphDatabase.driver(a.uri, auth=(a.user, a.password)) as d, d.session(database=a.database) as s:
        if a.anaphoric:
            rows = s.run(
                "MATCH (g)-[r]-(m) WHERE toLower(g.name) IN $g RETURN g.name AS g, labels(g) AS gl, "
                "startNode(r)=g AS out, type(r) AS t, m.name AS m, properties(r) AS props", g=ANAPHORIC).data()
            nodes = s.run("MATCH (g) WHERE toLower(g.name) IN $g RETURN count(g) AS c", g=ANAPHORIC).single()["c"]
            print(f"anaphoric nodes {nodes}, their edges {len(rows)}")
            if not a.apply:
                print("dry run: nothing written")
                return
            if a.log:
                with a.log.open("w", encoding="utf-8") as f:
                    for x in rows:
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
            c = s.run("MATCH (g) WHERE toLower(g.name) IN $g WITH g, g.name AS n DETACH DELETE g "
                      "RETURN count(*) AS c", g=ANAPHORIC).single()["c"]
            print(f"applied: {c} anaphoric nodes deleted with their edges")
        else:
            n = s.run("MATCH (n) WHERE NOT n:NodeVec AND NOT (n)--() RETURN count(n) AS c").single()["c"]
            print(f"isolated nodes {n}")
            if not a.apply:
                print("dry run: nothing written")
                return
            c = s.run("MATCH (n) WHERE NOT n:NodeVec AND NOT (n)--() DELETE n RETURN count(*) AS c").single()["c"]
            print(f"applied: {c} isolated nodes deleted")


if __name__ == "__main__":
    main()
