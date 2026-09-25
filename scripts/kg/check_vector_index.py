"""Fail when the vector index exists but no longer points at the graph.

Counting the nodes that carry an embedding is not enough: the count passes on a
graph whose store has been reloaded, because the carrier nodes survive the
reload and only the internal identifiers they hold go stale.
The vector channel then fails open: retrieval degrades to lexical matching,
returns results rather than an error, and the run completes looking healthy.

This script checks what the count cannot: how many carriers still resolve to a
node. Exit code 1 when too few do.

    conda run -n graphllm python scripts/kg/check_vector_index.py --min-resolving 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from graphrag.config import build_kg_config_from_env  # noqa: E402
from graphrag.kg.manager import KnowledgeGraphManager  # noqa: E402

RESOLVE_QUERY = """
MATCH (v:NodeVec)
WITH v LIMIT $sample
OPTIONAL MATCH (n) WHERE elementId(n) = v.of
RETURN count(v) AS carriers,
       sum(CASE WHEN n IS NULL THEN 0 ELSE 1 END) AS resolving
"""


def main(argv: list[str] | None = None) -> int:
    """Count the carriers that still resolve to a node.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status: 1 when fewer than ``--min-resolving`` resolve.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-resolving",
        type=int,
        default=1000,
        help="fail when fewer carriers than this resolve to a node",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20000,
        help="how many carriers to check (0 disables the limit)",
    )
    args = parser.parse_args(argv)

    load_dotenv(REPO / ".env", override=False)
    store = KnowledgeGraphManager(build_kg_config_from_env())
    rows = store.run_query(RESOLVE_QUERY, {"sample": args.sample or 10**9})
    row = rows[0] if rows else {"carriers": 0, "resolving": 0}
    carriers = int(row.get("carriers") or 0)
    resolving = int(row.get("resolving") or 0)

    print(f"vector carriers checked: {carriers}, resolving to a node: {resolving}")
    if resolving < args.min_resolving:
        print(
            f"FAIL: only {resolving} carriers resolve, below the floor of "
            f"{args.min_resolving}. The store was probably reloaded; rebuild "
            f"with scripts/kg/kg_vector_index.py --context-chars 300",
            file=sys.stderr,
        )
        return 1
    print("OK: the vector index still points at the graph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
