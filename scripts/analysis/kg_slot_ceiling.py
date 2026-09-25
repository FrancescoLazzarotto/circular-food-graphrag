"""How many gold concept slots the graph can reach at all, by name.

Every change to the graph is judged by what it moves here, because this is the
number no retriever can beat: if a concept has no node reachable under any
accepted English surface form, no amount of retrieval tuning will return it for
an English question.

Two figures are reported, because the full-text index the retriever queries
covers ``name`` *and* ``search_text``, i.e. the aliases too:

``name``       gold forms against ``n.name`` only
``name+alias`` what retrieval can really reach — the one that moves when
               English labels are attached to Italian nodes

Buckets, per slot:

``en``         some non-Italian accepted form matches exactly
``it_only``    only an Italian ``alt_labels`` form matches
``substring``  no exact match, but some form appears inside a longer node name
``absent``     nothing at all

Comparison keys come from ``evalkit.normalisation.match_key``, the same
lowercase + accent-fold the scorer uses, so a slot counted here is a slot the
scorer would accept.

Usage::

    conda run -n graphllm python scripts/analysis/kg_slot_ceiling.py \\
        --uri bolt://localhost:7689 --password staging-kg-v2 --tag v2_baseline
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

from evalkit.metrics.resolver import fold_number  # noqa: E402
from evalkit.normalisation import match_key as _match_key  # noqa: E402
from kg_pipeline.utils import neo4j_env  # noqa: E402


def match_key(text: str) -> str:
    """The scorer's comparison key, number folding included.

    ``normalisation.match_key`` is only tier 2 of the resolver; tier 3 folds
    plurals through the explicit lexicon in ``metrics.resolver``. Measuring
    without that fold understates what the graph reaches, precisely on English
    labels taken from AGROVOC: the prefLabel for ``polifenoli`` is
    ``polyphenols`` while the gold slot is ``polyphenol``, and the scorer counts
    that as a match.
    """
    return fold_number(_match_key(text))

logger = logging.getLogger("kg_slot_ceiling")

FETCH = (
    "MATCH (n) WHERE n.name IS NOT NULL AND NOT n:NodeVec "
    "RETURN toString(n.name) AS name, coalesce(n.aliases, []) AS aliases, "
    "coalesce(n.aliases_v1, []) AS aliases_v1, n.aliases_v1 IS NOT NULL AS snapshotted "
    "SKIP $skip LIMIT $limit"
)


def load_graph_forms(driver, database: str | None,
                     baseline: bool = False) -> tuple[dict, dict, list[str]]:
    """Two lookup tables (name-only, name+alias) plus the raw name list.

    The tables map a match key to the node name that carries it. The raw list is
    what the substring bucket scans, since a substring hit has no key.

    ``baseline=True`` reads ``aliases_v1``, the snapshot taken before the first
    alias write, so the pre-intervention number can be recomputed on the current
    graph without reverting it.
    """
    names: list[str] = []
    by_name: dict[str, str] = {}
    by_any: dict[str, str] = {}
    skip = 0
    with driver.session(database=database) as session:
        while True:
            rows = list(session.run(FETCH, skip=skip, limit=5000))
            if not rows:
                break
            for row in rows:
                name = row["name"]
                names.append(name)
                key = match_key(name)
                if key:
                    by_name.setdefault(key, name)
                    by_any.setdefault(key, name)
                aliases = (row["aliases_v1"] if (baseline and row["snapshotted"])
                           else row["aliases"])
                for alias in aliases or []:
                    akey = match_key(str(alias))
                    if akey:
                        by_any.setdefault(akey, name)
            skip += 5000
    return by_name, by_any, names


def accepted_forms(entity: dict) -> tuple[list[str], list[str]]:
    """(non-Italian forms, Italian-only forms) for one gold slot.

    The gold set marks no language on ``alt_labels``. A slot counts as
    English-reachable when it matches ``label`` or ``normalised_label``, which
    are English by construction (§5.4 of the protocol), and as Italian-only when
    the sole match came from ``alt_labels``. The rule is stated here because the
    gold file does not carry it.
    """
    primary = [entity.get("normalised_label"), entity.get("label")]
    primary = [p for p in primary if p]
    alts = [a for a in (entity.get("alt_labels") or []) if a]
    primary_keys = {match_key(p) for p in primary}
    alt_only = [a for a in alts if match_key(a) not in primary_keys]
    return primary, alt_only


def classify(entity: dict, table: dict[str, str], names: list[str]) -> tuple[str, str | None, str | None]:
    """Bucket for one slot: (bucket, matched form, node name)."""
    primary, alt_only = accepted_forms(entity)
    for form in primary:
        hit = table.get(match_key(form))
        if hit:
            return "en", form, hit
    for form in alt_only:
        hit = table.get(match_key(form))
        if hit:
            return "it_only", form, hit
    for form in [*primary, *alt_only]:
        needle = match_key(form)
        if len(needle) < 4:
            continue  # 'pm10' is fine, two-letter needles match everything
        for name in names:
            if needle in match_key(name):
                return "substring", form, name
    return "absent", None, None


def main(argv: list[str] | None = None) -> int:
    """Measure the slot ceiling of a graph and write it to JSON.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(REPO / "evaluation" / "gold" / "gold_v3.json"))
    parser.add_argument("--uri", default="bolt://localhost:7689")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="staging-kg-v2")
    parser.add_argument("--database", default=None)
    parser.add_argument("--tag", default="unnamed", help="label for the output file")
    parser.add_argument("--baseline-aliases", action="store_true",
                        help="score against aliases_v1, the pre-intervention snapshot")
    parser.add_argument("--output-dir", default=str(REPO / "artifacts/kg_v2"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )
    by_name, by_any, names = load_graph_forms(driver, args.database, args.baseline_aliases)
    logger.info("graph: %d names, %d name keys, %d name+alias keys",
                len(names), len(by_name), len(by_any))

    slots: list[dict] = []
    for query in gold["queries"]:
        if query.get("scoring", {}).get("distractor_expected"):
            continue  # a distractor has no expected entity to reach
        for entity in query.get("expected_entities") or []:
            b_name, f_name, n_name = classify(entity, by_name, names)
            b_any, f_any, n_any = classify(entity, by_any, names)
            slots.append({
                "query_id": query["query_id"],
                "label": entity.get("label"),
                "uri": entity.get("uri"),
                "name_bucket": b_name,
                "name_form": f_name,
                "name_node": n_name,
                "any_bucket": b_any,
                "any_form": f_any,
                "any_node": n_any,
            })

    name_counts = Counter(s["name_bucket"] for s in slots)
    any_counts = Counter(s["any_bucket"] for s in slots)
    total = len(slots)

    def exact_ceiling(counts: Counter) -> float:
        """Share of slots whose concept has a node matching some form exactly.

        That is en + it_only.
        """
        return (counts["en"] + counts["it_only"]) / total if total else 0.0

    def ceiling(counts: Counter) -> float:
        """As above, plus the substring slots a name cleanup would recover."""
        return (counts["en"] + counts["it_only"] + counts["substring"]) / total if total else 0.0

    report = {
        "tag": args.tag,
        "measured": datetime.now().isoformat(timespec="seconds"),
        "uri": args.uri,
        "slots": total,
        "graph_names": len(names),
        "name_only": {**name_counts,
                      "exact_ceiling": round(exact_ceiling(name_counts), 4),
                      "ceiling": round(ceiling(name_counts), 4),
                      "en_reachable": round(name_counts["en"] / total, 4) if total else 0.0},
        "name_plus_alias": {**any_counts,
                            "exact_ceiling": round(exact_ceiling(any_counts), 4),
                            "ceiling": round(ceiling(any_counts), 4),
                            "en_reachable": round(any_counts["en"] / total, 4) if total else 0.0},
        "detail": slots,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"slot_ceiling_{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{total} gold slots (distractors excluded)\n")
    print(f"{'bucket':<12}{'name only':>12}{'name+alias':>12}")
    for bucket in ("en", "it_only", "substring", "absent"):
        print(f"{bucket:<12}{name_counts[bucket]:>12}{any_counts[bucket]:>12}")
    print(f"{'':<12}{'':>12}{'':>12}")
    print(f"{'EN reachable':<12}{name_counts['en']/total:>11.1%}{any_counts['en']/total:>12.1%}")
    print(f"{'exact ceil.':<12}{exact_ceiling(name_counts):>11.1%}{exact_ceiling(any_counts):>12.1%}")
    print(f"{'ceiling':<12}{ceiling(name_counts):>11.1%}{ceiling(any_counts):>12.1%}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
