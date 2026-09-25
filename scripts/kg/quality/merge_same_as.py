"""Merge SAME_AS-linked node pairs and remove self-loops.

Post-ingest cleanup: stage-6 ingestion writes SAME_AS relationships between
entities the resolution stage considered equivalent; this pass merges each
pair into the higher-degree node (keeping its name, unioning aliases,
rebuilding search_text) and then deletes any self-loop left behind. Iterates
one pair at a time so alias chains (a SAME_AS b SAME_AS c) collapse
correctly.

Usage:
    python scripts/kg/quality/merge_same_as.py --uri "$NEO4J_URL" \
        --user "$NEO4J_USERNAME" --password "$NEO4J_PASSWORD" \
        --database "$NEO4J_DATABASE"
"""

from __future__ import annotations

import argparse
import sys

from kg_pipeline.utils import neo4j_env

MERGE_QUERY = """
MATCH (keep) WHERE elementId(keep) = $keep
MATCH (drop) WHERE elementId(drop) = $drop
WITH keep, drop,
     [x IN coalesce(drop.aliases, []) + drop.name
      WHERE NOT x IN coalesce(keep.aliases, []) + keep.name] AS extra
SET keep.aliases = coalesce(keep.aliases, []) + extra
WITH keep, drop
CALL apoc.refactor.mergeNodes([keep, drop],
     {properties: 'discard', mergeRels: false}) YIELD node
SET node.search_text = node.name +
    CASE WHEN size(coalesce(node.aliases, [])) > 0
         THEN '\n' + apoc.text.join(node.aliases, '\n') ELSE '' END
RETURN elementId(node) AS id
"""


def main() -> int:
    """Merge every SAME_AS pair, then delete the self-loops left behind.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default=None)
    args = parser.parse_args()

    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )
    merged = 0
    with driver.session(database=args.database) as session:
        while True:
            pair = session.run(
                "MATCH (a)-[:SAME_AS]->(b) WHERE a <> b "
                "RETURN elementId(a) AS a, elementId(b) AS b, "
                "COUNT { (a)--() } AS da, COUNT { (b)--() } AS db LIMIT 1"
            ).single()
            if pair is None:
                break
            keep, drop = ((pair["a"], pair["b"]) if pair["da"] >= pair["db"]
                          else (pair["b"], pair["a"]))
            session.run(MERGE_QUERY, keep=keep, drop=drop)
            merged += 1
        same_as_loops = session.run(
            "MATCH (a)-[r:SAME_AS]->(a) DELETE r RETURN count(*) AS c"
        ).single()["c"]
        self_loops = session.run(
            "MATCH (n)-[r]->(n) DELETE r RETURN count(*) AS c"
        ).single()["c"]
    driver.close()
    print(f"merged {merged} SAME_AS pairs, removed {same_as_loops} SAME_AS "
          f"self-loops and {self_loops} other self-loops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
