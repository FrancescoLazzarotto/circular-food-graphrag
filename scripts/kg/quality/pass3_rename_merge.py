"""Rename junk canonical names and merge duplicate entities.

Dry-run by default: writes CSVs with the proposed renames and merges.
``--apply`` executes them (renames first, then merges) and rebuilds
``search_text`` (= name + aliases, newline-separated) on touched nodes.

Rename candidates: well-connected nodes whose name reads like a clause or an
anaphoric phrase but that carry a clean short alias (collected during entity
resolution). The old name is preserved as an alias, so fulltext retrieval
still matches it.

Merge candidates: same-label nodes whose normalized names are identical, plus
an explicit curated list. DataValue nodes are never merged (near-identical
names there are *different* measurements).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from kg_pipeline.utils import neo4j_env

MIN_DEGREE_FOR_RENAME = 8
CLAUSE_WORDS = 7  # names with at least this many words read as propositions
ANAPHORIC_START_RE = re.compile(
    r"^(una?|il|lo|la|gli|le|questa?|queste|questi|attuali|the|a|an|this|these)\s",
    re.IGNORECASE,
)
# A usable replacement alias: short, not itself a clause, no citation year.
GOOD_ALIAS_RE = re.compile(r"^[^,;]{3,60}$")

# (keep_name, keep_label, drop_name, drop_label) — curated, order matters:
# the first node survives and receives the other's relationships/aliases.
CURATED_MERGES = [
    ("Economia Circolare", "Concept", "una progettualità di economia circolare", "Concept"),
]

# Labels with a uniqueness constraint on name (renames must not collide).
UNIQUE_LABELS = {"Event", "Indicator", "Method", "Organization", "Policy"}


def _propose_alias(name: str, aliases: list[str]) -> str | None:
    """Pick the best replacement name from a node's aliases."""
    ranked = []
    for alias in aliases:
        alias = alias.strip()
        if alias.lower() == name.lower():
            continue
        words = len(alias.split())
        if words > 5 or not GOOD_ALIAS_RE.match(alias):
            continue
        if ANAPHORIC_START_RE.match(alias):
            continue
        # Prefer fewer words, then Title Case / acronyms over lowercase.
        cased = 0 if (alias[:1].isupper()) else 1
        ranked.append((words, cased, len(alias), alias))
    if not ranked:
        return None
    return sorted(ranked)[0][3]


def collect(session):
    """Read every named node and propose the renames and merges.

    Args:
        session: Open Neo4j session.

    Returns:
        The node rows, the proposed renames, and the proposed merges
        (same-label duplicates by normalised name, plus the curated list).
    """
    rows = session.run(
        "MATCH (n) WHERE n.name IS NOT NULL "
        "RETURN elementId(n) AS id, n.name AS name, labels(n) AS labels, "
        "n.aliases AS aliases, COUNT { (n)--() } AS degree"
    ).data()

    renames = []
    for r in rows:
        name = r["name"].strip()
        looks_junk = (len(name.split()) >= CLAUSE_WORDS
                      or (ANAPHORIC_START_RE.match(name) and len(name.split()) >= 3))
        if r["degree"] < MIN_DEGREE_FOR_RENAME or not looks_junk:
            continue
        proposal = _propose_alias(name, r["aliases"] or [])
        if proposal:
            renames.append({
                "node_id": r["id"], "label": r["labels"][0], "degree": r["degree"],
                "old_name": name, "new_name": proposal,
            })

    by_key = defaultdict(list)
    for r in rows:
        primary = next((l for l in r["labels"] if not l.startswith("_")), "?")
        if primary == "DataValue":
            continue
        key = (primary, re.sub(r"\s+", " ", r["name"].strip().lower()))
        by_key[key].append(r)
    merges = []
    for (label, _), group in by_key.items():
        if len(group) > 1:
            group.sort(key=lambda g: -g["degree"])
            for dup in group[1:]:
                merges.append({
                    "keep_id": group[0]["id"], "keep_name": group[0]["name"],
                    "drop_id": dup["id"], "drop_name": dup["name"],
                    "label": label,
                    "keep_degree": group[0]["degree"], "drop_degree": dup["degree"],
                })

    for keep_name, keep_label, drop_name, drop_label in CURATED_MERGES:
        found = {}
        for r in rows:
            if r["name"] == keep_name and keep_label in r["labels"]:
                found["keep"] = r
            if r["name"] == drop_name and drop_label in r["labels"]:
                found["drop"] = r
        if "keep" in found and "drop" in found:
            merges.append({
                "keep_id": found["keep"]["id"], "keep_name": keep_name,
                "drop_id": found["drop"]["id"], "drop_name": drop_name,
                "label": keep_label,
                "keep_degree": found["keep"]["degree"],
                "drop_degree": found["drop"]["degree"],
            })
    return rows, renames, merges


def main() -> int:
    """Report, and with ``--apply`` run, the renames and merges.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default=None)
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/kg_quality"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overrides", type=Path, default=None,
                        help="JSON {node_id: new_name} replacing heuristic "
                             "proposals after human review ('skip' drops one).")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, str] = {}
    if args.overrides:
        import json
        overrides = json.loads(args.overrides.read_text())

    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )
    with driver.session(database=args.database) as session:
        rows, renames, merges = collect(session)
        # Overrides may be keyed by elementId or by exact old name, so the
        # same overrides file works across instances (staging vs Aura).
        for r in renames:
            if r["node_id"] in overrides:
                r["new_name"] = overrides[r["node_id"]]
            elif r["old_name"] in overrides:
                r["new_name"] = overrides[r["old_name"]]
        renames = [r for r in renames if r["new_name"] != "skip"]
        names_by_label = defaultdict(set)
        for r in rows:
            for l in r["labels"]:
                names_by_label[l].add(r["name"])

        # A rename that collides with an existing name on a unique-constrained
        # label would fail; on any label it would create ambiguity — skip both.
        safe_renames, skipped = [], []
        merge_drop_ids = {m["drop_id"] for m in merges}
        for r in renames:
            if r["node_id"] in merge_drop_ids:
                skipped.append({**r, "reason": "node merged away"})
            elif r["new_name"] in names_by_label.get(r["label"], set()):
                skipped.append({**r, "reason": "name collision"})
            else:
                safe_renames.append(r)

        for fname, data, fields in (
            ("pass3_renames.csv", safe_renames,
             ["node_id", "label", "degree", "old_name", "new_name"]),
            ("pass3_renames_skipped.csv", skipped,
             ["node_id", "label", "degree", "old_name", "new_name", "reason"]),
            ("pass3_merges.csv", merges,
             ["keep_id", "keep_name", "keep_degree", "drop_id", "drop_name",
              "drop_degree", "label"]),
        ):
            path = args.report_dir / fname
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(data)
        print(f"renames: {len(safe_renames)} (skipped {len(skipped)}), "
              f"merges: {len(merges)} -> {args.report_dir}/pass3_*.csv")

        if not args.apply:
            print("dry-run only; re-run with --apply")
            return 0

        for r in safe_renames:
            session.run(
                "MATCH (n) WHERE elementId(n) = $id "
                "SET n.aliases = coalesce(n.aliases, []) + n.name, "
                "    n.name = $new, "
                "    n.search_text = $new + '\\n' + "
                "        reduce(s = '', a IN coalesce(n.aliases, []) | "
                "               s + CASE WHEN s = '' THEN a ELSE '\\n' + a END)",
                id=r["node_id"], new=r["new_name"],
            )
        merged = 0
        for m in merges:
            result = session.run(
                "MATCH (keep) WHERE elementId(keep) = $keep "
                "MATCH (drop) WHERE elementId(drop) = $drop "
                "WITH keep, drop, "
                "     [a IN coalesce(drop.aliases, []) + drop.name "
                "      WHERE NOT a IN coalesce(keep.aliases, []) + keep.name] AS extra "
                "SET keep.aliases = coalesce(keep.aliases, []) + extra "
                "WITH keep, drop "
                "CALL apoc.refactor.mergeNodes([keep, drop], "
                "     {properties: 'discard', mergeRels: false}) YIELD node "
                "SET node.search_text = node.name + '\\n' + "
                "    reduce(s = '', a IN coalesce(node.aliases, []) | "
                "           s + CASE WHEN s = '' THEN a ELSE '\\n' + a END) "
                "RETURN elementId(node) AS id",
                keep=m["keep_id"], drop=m["drop_id"],
            ).single()
            if result:
                merged += 1
        # merging can create new self-loops (edges between merged pair)
        loops = session.run("MATCH (n)-[r]->(n) DELETE r RETURN count(*) AS c").single()["c"]
        print(f"APPLIED: {len(safe_renames)} renames, {merged} merges, "
              f"{loops} post-merge self-loops removed")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
