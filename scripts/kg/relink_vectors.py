#!/usr/bin/env python3
"""Point the vector carriers at their nodes again after a neo4j-admin load.

A carrier (``:NodeVec``) names its node by element id, which is
``<version>:<database id>:<record id>``. A dump loaded into another instance
keeps every record id, since the store files are copied as they are, but the
database gets its own id, so every carrier points at nothing and the dense
channel goes silent. This rewrites the database part of each pointer to the
current database's.

It refuses when the foreign pointers name more than one database: a graph
whose carriers came from several stores is not a store copied as it is, and
the same record id would name different nodes. After rewriting, a carrier that
knows its node's name (``of_name``, written by kg_vector_index.py) must land on
a node with that name, and every carrier must land on a named node; otherwise
the script exits 1, and the vector index must be rebuilt with
kg_vector_index.py instead.

Only for a graph that arrived through neo4j-admin dump/load: after a JSON
restore (kg_restore.py) record ids change too.

    python scripts/kg/relink_vectors.py        # target from NEO4J_* in the environment
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The kg_pipeline package lives at the repo root and is not pip-installed.
sys.path.insert(0, str(ROOT))

from kg_pipeline.utils import neo4j_env  # noqa: E402


def main() -> int:
    """Rewrite the pointers; return 1 when a carrier points at nothing or at another node."""
    target = neo4j_env.resolve_target()
    with neo4j_env.connect(target) as driver, driver.session(**target.session_kwargs()) as session:
        prefix = session.run(
            "MATCH (n) WHERE NOT n:NodeVec WITH split(elementId(n), ':') AS p LIMIT 1 "
            "RETURN p[0] + ':' + p[1] + ':' AS prefix"
        ).single()
        if prefix is None:
            print("grafo vuoto: niente da ricollegare")
            return 0
        foreign = session.run(
            "MATCH (v:NodeVec) WHERE NOT v.of STARTS WITH $prefix "
            "RETURN collect(DISTINCT split(v.of, ':')[1]) AS dbs, count(v) AS c",
            prefix=prefix["prefix"],
        ).single()
        if len(foreign["dbs"]) > 1:
            print(f"i vettori puntano a {len(foreign['dbs'])} database diversi: non li ricollego, "
                  "ricostruire l'indice con kg_vector_index.py")
            return 1
        session.run(
            "MATCH (v:NodeVec) WHERE NOT v.of STARTS WITH $prefix "
            "SET v.of = $prefix + split(v.of, ':')[2]",
            prefix=prefix["prefix"],
        )
        checked = session.run(
            "MATCH (v:NodeVec) OPTIONAL MATCH (n) WHERE elementId(n) = v.of "
            "RETURN count(v) AS carriers, count(n.name) AS named, "
            "count(v.of_name) AS with_name, "
            "count(CASE WHEN v.of_name IS NOT NULL AND v.of_name <> toString(n.name) THEN 1 END) AS wrong"
        ).single()
    dangling = checked["carriers"] - checked["named"]
    print(f"vettori ricollegati: {foreign['c']}; senza nodo con nome: {dangling}; "
          f"con nome verificabile: {checked['with_name']}, su un nodo con un altro nome: {checked['wrong']}")
    if dangling or checked["wrong"]:
        print("ricostruire l'indice con kg_vector_index.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
