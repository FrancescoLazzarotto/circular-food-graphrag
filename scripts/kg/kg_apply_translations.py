"""Write the translated names from ``kg_translate_names.py`` onto the graph.

Separate from the pass that produced them so the expensive part runs once and
the cheap part can be redone with different filters.

Only the ``en`` field is applied. The ``head`` field is deliberately **not**:
heads collapse to generic single words — a phrase about training outcomes
yields ``results``, one about a ministerial body yields ``council`` — and
attaching those as aliases builds a magnet: a query anchored on a generic head
such as ``framework`` retrieves dozens of nodes and none of the answers. Pass
``--apply-heads`` to include them anyway, as a separate variant.

Reversible the same way as the AGROVOC pass: ``aliases_v1`` holds the
pre-intervention list, and ``kg_ontology_align.py --revert`` restores it.

Usage::

    conda run -n graphllm python scripts/kg/kg_apply_translations.py            # dry run
    conda run -n graphllm python scripts/kg/kg_apply_translations.py --apply
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
for extra in (REPO / "src", REPO / "evaluation"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evalkit.normalisation import match_key  # noqa: E402
from kg_pipeline.utils import neo4j_env  # noqa: E402

logger = logging.getLogger("kg_apply_translations")

FETCH = (
    "MATCH (n) WHERE elementId(n) IN $ids "
    "RETURN elementId(n) AS id, toString(n.name) AS name, "
    "coalesce(n.aliases, []) AS aliases, coalesce(n.aliases_v1, []) AS aliases_v1, "
    "n.aliases_v1 IS NOT NULL AS snapshotted"
)

WRITE = """
UNWIND $rows AS row
MATCH (n) WHERE elementId(n) = row.id
SET n.aliases_v1 = CASE WHEN n.aliases_v1 IS NULL
                        THEN coalesce(n.aliases, []) ELSE n.aliases_v1 END,
    n.aliases = row.aliases,
    n.search_text = row.search_text,
    n.aliases_llm = row.added
"""


def main(argv: list[str] | None = None) -> int:
    """Add the translated names to the aliases; a dry run unless ``--apply``.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(REPO / "artifacts/kg_v2/translations.jsonl"))
    parser.add_argument("--uri", default="bolt://localhost:7689")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="staging-kg-v2")
    parser.add_argument("--database", default=None)
    parser.add_argument("--max-words", type=int, default=8,
                        help="drop translations longer than this; they are sentences")
    parser.add_argument("--apply-heads", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-dir", default=str(REPO / "artifacts/kg_v2"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = [json.loads(line) for line in
               Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("%d translation records", len(records))

    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )
    ids = [r["id"] for r in records]
    current: dict[str, dict] = {}
    with driver.session(database=args.database) as session:
        for start in range(0, len(ids), 5000):
            for row in session.run(FETCH, ids=ids[start:start + 5000]):
                current[row["id"]] = dict(row)

    reasons: Counter = Counter()
    rows: list[dict] = []
    for record in records:
        node = current.get(record["id"])
        if node is None:
            reasons["node_gone"] += 1
            continue
        base = node["aliases_v1"] if node["snapshotted"] else node["aliases"]
        have = {match_key(node["name"])} | {match_key(str(a)) for a in base}

        candidates = [record.get("en") or ""]
        if args.apply_heads:
            candidates.append(record.get("head") or "")

        added: list[str] = []
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            if len(candidate.split()) > args.max_words:
                reasons["too_long"] += 1
                continue
            key = match_key(candidate)
            if not key:
                reasons["empty_key"] += 1
                continue
            if key in have:
                reasons["already_present"] += 1
                continue
            have.add(key)
            added.append(candidate)

        if not added:
            continue
        aliases = [*[str(a) for a in base], *added]
        rows.append({
            "id": record["id"],
            "aliases": aliases,
            "added": added,
            "search_text": "\n".join([node["name"], *aliases]),
        })
        reasons["applied"] += len(added)

    summary = {
        "measured": datetime.now().isoformat(timespec="seconds"),
        "applied": bool(args.apply),
        "records": len(records),
        "nodes_changed": len(rows),
        "aliases_added": reasons["applied"],
        "skipped": {k: v for k, v in reasons.items() if k != "applied"},
        "heads_included": bool(args.apply_heads),
    }

    if args.apply:
        with driver.session(database=args.database) as session:
            for start in range(0, len(rows), 1000):
                session.run(WRITE, rows=rows[start:start + 1000]).consume()
                logger.info("  wrote %d/%d", min(start + 1000, len(rows)), len(rows))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / f"apply_translations_{stamp}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
