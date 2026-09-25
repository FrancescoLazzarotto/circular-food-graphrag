"""Populate expected_entities and gold_triples in a gold CSV from Neo4j.

For each gold row:
  - Finds entity names (from Neo4j) that appear verbatim in canonical_answer
  - Queries Neo4j for triples where subject AND object are in that entity set
  - Writes updated CSV with expected_entities and gold_triples populated

Usage:
    conda run -n graphllm python scripts/gold/populate_gold_annotations.py \
        --gold evaluation/gold/gold_generated.csv \
        --env-file .env
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

LOGGER = logging.getLogger("populate_gold")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env(env_file: str) -> None:
    """Set the variables of a ``KEY=value`` file, without overriding the environment."""
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)


def _neo4j_driver():
    """A driver to the graph the environment names."""
    from kg_pipeline.utils import neo4j_env  # type: ignore

    return neo4j_env.connect()


def _fetch_all_entity_names(session) -> list[str]:
    """Every distinct node name in the graph."""
    result = session.run("MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name")
    names = [r["name"] for r in result if r["name"]]
    LOGGER.info("Fetched %d entity names from Neo4j", len(names))
    return names


def _match_entities(text: str, entity_names: list[str], min_len: int = 4) -> list[str]:
    """Find entity names that appear as whole-word substrings in text."""
    text_lower = text.lower()
    matched = []
    for name in entity_names:
        if len(name) < min_len:
            continue
        pattern = re.escape(name.lower())
        if re.search(r"\b" + pattern + r"\b", text_lower):
            matched.append(name)
    return matched


def _fetch_triples(session, entities: list[str], database: str | None) -> list[dict]:
    """The triples whose subject and object are both in `entities`.

    Args:
        session: Open Neo4j session.
        entities: Node names.
        database: Unused.

    Returns:
        ``subject``/``predicate``/``object`` dicts.
    """
    if not entities:
        return []
    result = session.run(
        "MATCH (a)-[r]->(b) "
        "WHERE a.name IN $names AND b.name IN $names "
        "RETURN a.name AS subject, type(r) AS predicate, b.name AS object",
        names=entities,
    )
    triples = [
        {"subject": r["subject"], "predicate": r["predicate"], "object": r["object"]}
        for r in result
    ]
    return triples


def populate(gold_path: Path, env_file: str, min_entity_len: int, dry_run: bool) -> int:
    """Fill ``expected_entities`` and ``gold_triples`` for every row.

    Args:
        gold_path: Gold CSV, rewritten in place.
        env_file: ``KEY=value`` file loaded first.
        min_entity_len: Shortest node name that may match.
        dry_run: Print a sample and write nothing.

    Returns:
        The exit status: 1 when the CSV has no rows.
    """
    _load_env(env_file)

    rows = list(csv.DictReader(gold_path.open(encoding="utf-8")))
    if not rows:
        LOGGER.error("No rows in %s", gold_path)
        return 1

    database = os.environ.get("NEO4J_DATABASE") or None
    driver = _neo4j_driver()

    try:
        with driver.session(database=database) as session:
            entity_names = _fetch_all_entity_names(session)

            for i, row in enumerate(rows):
                canon = row.get("canonical_answer", "").strip()
                if not canon:
                    LOGGER.warning("Row %d (%s): empty canonical_answer, skipping", i, row.get("question_id"))
                    continue

                matched = _match_entities(canon, entity_names, min_len=min_entity_len)
                triples = _fetch_triples(session, matched, database)

                row["expected_entities"] = json.dumps(matched, ensure_ascii=False)
                row["gold_triples"] = json.dumps(triples, ensure_ascii=False)

                LOGGER.info(
                    "Row %d [%s]: %d entities, %d triples",
                    i, row.get("question_id", f"q{i}"), len(matched), len(triples),
                )
    finally:
        driver.close()

    if dry_run:
        LOGGER.info("Dry run — not writing output")
        for row in rows[:3]:
            print(json.dumps({
                "question_id": row.get("question_id"),
                "expected_entities": json.loads(row.get("expected_entities", "[]")),
                "gold_triples_count": len(json.loads(row.get("gold_triples", "[]"))),
            }, ensure_ascii=False, indent=2))
        return 0

    fieldnames = list(rows[0].keys())
    with gold_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Saved updated gold CSV: %s (%d rows)", gold_path, len(rows))
    return 0


def main() -> int:
    """Parse the command line and run `populate`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Populate expected_entities and gold_triples in gold CSV from Neo4j")
    parser.add_argument("--gold", default=str(REPO_ROOT / "evaluation" / "gold" / "gold_generated.csv"))
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--min-entity-len", type=int, default=4, help="Min chars for entity name match")
    parser.add_argument("--dry-run", action="store_true", help="Print sample output, do not write")
    args = parser.parse_args()
    return populate(Path(args.gold), args.env_file, args.min_entity_len, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
