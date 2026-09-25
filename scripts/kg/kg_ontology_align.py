"""Attach ontology IRIs and English labels to nodes that only have Italian ones.

The graph was extracted from a bilingual corpus and most concepts landed under
their Italian surface form, while the reference questions are English: many
gold concepts exist in the graph *only* under an Italian name, so no English
query can reach them lexically. Nothing about that is a topology problem: the
node is there, under the wrong name.

This script matches node names and existing aliases against AGROVOC's Italian
labels and writes back, on a hit:

``ontology_uri``     the AGROVOC concept IRI (a list; a node can carry several)
``ontology_source``  ``agrovoc``
``ontology_form``    the surface form that matched, for auditing
``aliases``          extended with the concept's English labels
``search_text``      rebuilt, because it is what the full-text index reads

**Not a benchmark leak.** The English labels come from a public vocabulary, not
from the gold file, whose ``alt_labels`` are off limits (protocol §5.4.2). Gold
and graph end up agreeing because both are aligned to the same standard, which
is the thing ontology grounding is supposed to buy — but it is worth saying out
loud that the concept-level metric benefits from that shared alignment, while
the grounding-level metric is the one this genuinely earns.

Reversible: the pre-existing alias list is snapshotted to ``aliases_v1`` before
the first write, so ``--revert`` restores the graph exactly.

Usage::

    # look, change nothing
    conda run -n graphllm python scripts/kg/kg_ontology_align.py
    # write
    conda run -n graphllm python scripts/kg/kg_ontology_align.py --apply
    # undo
    conda run -n graphllm python scripts/kg/kg_ontology_align.py --revert
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO / "src", REPO / "evaluation"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evalkit.normalisation import match_key  # noqa: E402
from kg_pipeline.utils import neo4j_env  # noqa: E402

logger = logging.getLogger("kg_ontology_align")

# Labels whose names are not vocabulary terms. A person's surname or a document
# title that happens to collide with an AGROVOC label would attach a wrong IRI
# and a wrong English alias, and neither would ever help retrieval.
SKIP_LABELS = {"Person", "Document", "DataValue", "Dataset", "NodeVec"}

FETCH = (
    "MATCH (n) WHERE n.name IS NOT NULL AND NOT n:NodeVec "
    "RETURN elementId(n) AS id, toString(n.name) AS name, labels(n) AS labels, "
    "coalesce(n.aliases, []) AS aliases, coalesce(n.aliases_v1, []) AS aliases_v1, "
    "n.aliases_v1 IS NOT NULL AS snapshotted "
    "SKIP $skip LIMIT $limit"
)

WRITE = """
UNWIND $rows AS row
MATCH (n) WHERE elementId(n) = row.id
SET n.aliases_v1 = CASE WHEN n.aliases_v1 IS NULL
                        THEN coalesce(n.aliases, []) ELSE n.aliases_v1 END,
    n.aliases = row.aliases,
    n.search_text = row.search_text,
    n.ontology_uri = row.uris,
    n.ontology_source = row.sources,
    n.ontology_form = row.form,
    n.ontology_conflict = row.conflict
"""

REVERT = """
MATCH (n) WHERE n.aliases_v1 IS NOT NULL
CALL (n) {
  SET n.aliases = n.aliases_v1,
      n.search_text = CASE WHEN size(n.aliases_v1) = 0 THEN toString(n.name)
                      ELSE toString(n.name) + '\n' + reduce(s = '', a IN n.aliases_v1 |
                           CASE WHEN s = '' THEN a ELSE s + '\n' + a END) END
  REMOVE n.aliases_v1, n.ontology_uri, n.ontology_source, n.ontology_form,
         n.ontology_conflict, n.aliases_llm
} IN TRANSACTIONS OF 1000 ROWS
"""


def load_lexicon(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, dict]]:
    """(italian key -> uris, english key -> uris, uri -> concept)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    it_index: dict[str, set[str]] = defaultdict(set)
    en_index: dict[str, set[str]] = defaultdict(set)
    concepts: dict[str, dict] = {}
    for concept in payload["concepts"]:
        concepts[concept["uri"]] = concept
        for label in concept["it"]:
            key = match_key(label)
            if key:
                it_index[key].add(concept["uri"])
        for label in concept["en"]:
            key = match_key(label)
            if key:
                en_index[key].add(concept["uri"])
    return it_index, en_index, concepts


def fetch_nodes(driver, database: str | None) -> list[dict]:
    """Every named node except the carriers, with its aliases, paged."""
    rows: list[dict] = []
    skip = 0
    with driver.session(database=database) as session:
        while True:
            page = [dict(r) for r in session.run(FETCH, skip=skip, limit=5000)]
            if not page:
                break
            rows.extend(page)
            skip += 5000
    return rows


def plan_node(node: dict, it_index, en_index, concepts, max_alt: int) -> dict | None:
    """What to write for one node, or None when nothing matches.

    Both the name and the aliases already on the node are tried, because
    resolution folds surface variants into ``aliases`` and any of them can be
    the one AGROVOC knows.

    Args:
        node: A row from `fetch_nodes`.
        it_index: Italian key -> URIs.
        en_index: English key -> URIs.
        concepts: URI -> concept.
        max_alt: English altLabels added beyond the prefLabel.

    Returns:
        The new aliases, search text, URIs and matched form, with any conflict
        between name and aliases; ``None`` when nothing matches.
    """
    if SKIP_LABELS & set(node["labels"]):
        return None

    # aliases_v1 is the pre-intervention truth once a run has happened; using it
    # keeps a second run from matching against labels the first run added.
    base_aliases = node["aliases_v1"] if node["snapshotted"] else node["aliases"]

    def lookup(forms: list[str]) -> tuple[set[str], str | None]:
        """URIs uniquely claimed by these forms, and the first form that hit."""
        found: set[str] = set()
        first: str | None = None
        for form in forms:
            key = match_key(form)
            if not key:
                continue
            for index in (it_index, en_index):
                hits = index.get(key) or set()
                if len(hits) == 1:  # an ambiguous label carries two senses
                    found |= hits
                    first = first or form
                    break
        return found, first

    # The name decides. Aliases are consulted only when the name matches
    # nothing, because resolution can fold genuinely different substances onto
    # one node: `paglia` (straw) carrying `fecce` (wine lees) and `pula` (bran)
    # as aliases. Unioning across forms would attach three AGROVOC concepts and
    # three sets of English labels to that one node, making the merge error
    # worse instead of visible. Disagreement is recorded on
    # ``ontology_conflict`` instead — the vocabulary is what detects it.
    name_uris, name_form = lookup([node["name"]])
    alias_uris, alias_form = lookup([str(a) for a in base_aliases])
    conflict = bool(name_uris and alias_uris and alias_uris - name_uris)

    uris = name_uris or alias_uris
    matched_form = name_form or alias_form
    if not uris:
        return None

    have = {match_key(node["name"])} | {match_key(str(a)) for a in base_aliases}
    additions: list[str] = []
    for uri in sorted(uris):
        concept = concepts[uri]
        for label in concept["en"][: max_alt + 1]:
            key = match_key(label)
            if key and key not in have:
                have.add(key)
                additions.append(label)

    aliases = [*[str(a) for a in base_aliases], *additions]
    search_text = "\n".join([node["name"], *aliases])
    return {
        "id": node["id"],
        "name": node["name"],
        "aliases": aliases,
        "added": additions,
        "search_text": search_text,
        "uris": sorted(uris),
        "sources": ["agrovoc"],
        "form": matched_form,
        "conflict": conflict,
        "conflict_uris": sorted(alias_uris - name_uris) if conflict else [],
    }


def main(argv: list[str] | None = None) -> int:
    """Match the graph against AGROVOC and report; write with ``--apply``.

    ``--revert`` restores the snapshotted aliases and exits.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", default=str(REPO / "artifacts/ontology/agrovoc_it_en.json"))
    parser.add_argument("--uri", default="bolt://localhost:7689")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="staging-kg-v2")
    parser.add_argument("--database", default=None)
    parser.add_argument("--max-alt", type=int, default=3,
                        help="English altLabels to add beyond the prefLabel")
    parser.add_argument("--apply", action="store_true", help="write to the graph")
    parser.add_argument("--revert", action="store_true", help="restore aliases_v1 and exit")
    parser.add_argument("--output-dir", default=str(REPO / "artifacts/kg_v2"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )

    if args.revert:
        with driver.session(database=args.database) as session:
            session.run(REVERT).consume()
            left = session.run(
                "MATCH (n) WHERE n.ontology_uri IS NOT NULL RETURN count(n) AS c"
            ).single()["c"]
        logger.info("reverted; nodes still carrying ontology_uri: %d", left)
        return 0

    it_index, en_index, concepts = load_lexicon(Path(args.lexicon))
    logger.info("lexicon: %d concepts, %d italian keys, %d english keys",
                len(concepts), len(it_index), len(en_index))

    nodes = fetch_nodes(driver, args.database)
    logger.info("graph: %d nodes with a name", len(nodes))

    plans = [p for p in (plan_node(n, it_index, en_index, concepts, args.max_alt)
                         for n in nodes) if p]
    with_aliases = [p for p in plans if p["added"]]

    by_id = {n["id"]: n for n in nodes}
    per_label: Counter = Counter()
    for plan in with_aliases:
        for lbl in by_id[plan["id"]]["labels"]:
            per_label[lbl] += 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_csv = out_dir / f"ontology_align_{stamp}.csv"
    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_name", "matched_form", "agrovoc_uris", "aliases_added",
                         "alias_conflict", "conflicting_uris"])
        for plan in plans:
            writer.writerow([plan["name"], plan["form"], " ".join(plan["uris"]),
                             " | ".join(plan["added"]), plan["conflict"],
                             " ".join(plan["conflict_uris"])])

    summary = {
        "measured": datetime.now().isoformat(timespec="seconds"),
        "applied": bool(args.apply),
        "nodes_scanned": len(nodes),
        "nodes_matched": len(plans),
        "nodes_gaining_aliases": len(with_aliases),
        "aliases_added": sum(len(p["added"]) for p in with_aliases),
        "nodes_with_alias_conflict": sum(1 for p in plans if p["conflict"]),
        "per_label": dict(per_label.most_common()),
        "report_csv": str(report_csv),
    }

    if args.apply:
        rows = [{k: p[k] for k in
                 ("id", "aliases", "search_text", "uris", "sources", "form", "conflict")}
                for p in plans]
        with driver.session(database=args.database) as session:
            for start in range(0, len(rows), 1000):
                session.run(WRITE, rows=rows[start:start + 1000]).consume()
                logger.info("  wrote %d/%d", min(start + 1000, len(rows)), len(rows))
        logger.info("applied to %d nodes", len(rows))

    (out_dir / f"ontology_align_{stamp}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
