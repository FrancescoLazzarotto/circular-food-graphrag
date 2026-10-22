"""Build the ontology from the sheets: checks, SKOS file and runtime JSON.

Reads ``sources/*.tsv``, runs the Python checks, SHACL (``rules/shapes.ttl``)
and Skosify, then writes

- ``build/ontology.ttl``: SKOS, for exchange and validation;
- ``build/ontology.json``: precomputed structures read by
  ``graphrag.ontology`` at start-up;
- ``build/report.md``: every problem found, readable.

Exits with status 1, and writes no ontology file, if any check fails.

With ``--check-only`` it writes nothing: it prints, as JSON, the number of
problems of every check and a few counts of the SKOS graph (mappings to
external vocabularies, resources without a name of their own), for the
nightly quality report.

Usage::

    /mnt/storage/flazzarotto/venvs/ontology/bin/python ontology/scripts/build.py [--check-only]
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import logging
import sys

import pyshacl
import skosify.check
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS, XSD

from checks import LIVE, QUOTE_IN_NOTES, check_all, member_key
from common import (
    BUILD_DIR,
    ONTOLOGY_DIR,
    SHEET_COLUMNS,
    graph_name_index,
    document_filename,
    graph_nodes_for,
    latest_backup,
    normalize,
    quote_on_page,
    read_sheet,
    split_list,
)

CEF = Namespace("urn:cef-ontology:")
ENGLISH_DOCUMENTS = {"f_fassio_diid_2021"}
# Statuses the application treats as settled; anything else is shown as
# "not yet reviewed" or used for name lookup only.
USABLE = {"auto_accepted", "checked", "approved", "corrected"}
RULES = ONTOLOGY_DIR / "rules" / "rules.tsv"
SOURCE = re.compile(r"^(\S+) p\.(\d+(?:-\d+)?)$")


def load_rules() -> list[dict]:
    """The live rows of ``rules/rules.tsv``.

    Returns:
        Rule rows, with ``document``, ``page`` and ``quote`` parsed from
        ``source`` and ``notes``.
    """
    with RULES.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r.get("status") in LIVE]
    for row in rows:
        source = SOURCE.match(row["source"])
        quote = QUOTE_IN_NOTES.search(row.get("notes", ""))
        row["document"], row["page"] = (source.group(1), source.group(2)) if source else ("", "")
        row["quote"] = quote.group(1) if quote else ""
    return rows


def rule_problems(rules: list[dict], concepts: set[str]) -> list[str]:
    """A rule needs a known family and a quote found on its source page."""
    problems = []
    for rule in rules:
        if rule["applies_to"] not in concepts:
            problems.append(f"{rule['id']}: famiglia sconosciuta {rule['applies_to']}")
        if not (rule["document"] and quote_on_page(rule["quote"], rule["document"], rule["page"])):
            problems.append(f"{rule['id']}: citazione non trovata in {rule['source']}")
    return problems


def load_sheets() -> dict[str, list[dict]]:
    """Read every source sheet.

    Returns:
        Sheet name → rows.
    """
    return {name: read_sheet(name) for name in SHEET_COLUMNS}


def position_order(position: str) -> int:
    """Sort key of a member position: ``R0``…``R9``, then ``R+``; ``1``…``3``.

    Args:
        position: Position as written in the sheet.

    Returns:
        An integer for sorting; additions without a number come last.
    """
    digits = "".join(ch for ch in position if ch.isdigit())
    return int(digits) if digits else 1000


def to_rdf(sheets: dict[str, list[dict]]) -> Graph:
    """Translate the live rows into SKOS plus the ``cef:`` properties.

    Args:
        sheets: Sheet name → rows.

    Returns:
        The RDF graph.
    """
    g = Graph()
    g.bind("skos", SKOS)
    g.bind("cef", CEF)
    g.bind("dcterms", DCTERMS)
    live = {s: [r for r in rows if r.get("status") in LIVE] for s, rows in sheets.items()}

    def framework_uri(fid: str) -> URIRef:
        return CEF[f"framework/{fid}"]

    def member_uri(ref: str) -> URIRef:
        fid, position = ref.split("#")
        return CEF[f"framework/{fid}/{position.replace('+', 'plus')}"]

    def concept_uri(cid: str) -> URIRef:
        return CEF[f"concept/{cid}"]

    concepts_scheme = CEF["concepts"]
    g.add((concepts_scheme, RDF.type, SKOS.ConceptScheme))
    g.add((concepts_scheme, SKOS.prefLabel, Literal("Concetti dei framework", lang="it")))

    for row in live["frameworks"]:
        uri = framework_uri(row["id"])
        lang = "en" if row["document"] in ENGLISH_DOCUMENTS else "it"
        g.add((uri, RDF.type, SKOS.ConceptScheme))
        g.add((uri, RDF.type, CEF.Framework))
        g.add((uri, SKOS.prefLabel, Literal(row["name"], lang=lang)))
        g.add((uri, CEF.version, Literal(row["version"])))
        g.add((uri, CEF.document, Literal(row["document"])))
        g.add((uri, CEF.pages, Literal(row["pages"])))
        g.add((uri, CEF.status, Literal(row["status"])))
        if row.get("author"):
            g.add((uri, DCTERMS.creator, Literal(row["author"])))
        if row.get("year"):
            g.add((uri, DCTERMS.date, Literal(row["year"], datatype=XSD.gYear)))
        for parent in split_list(row.get("derived_from", "")):
            g.add((uri, CEF.derivedFrom, framework_uri(parent)))

    for row in live["concepts"]:
        uri = concept_uri(row["id"])
        g.add((uri, RDF.type, SKOS.Concept))
        g.add((uri, RDF.type, CEF.SharedConcept))
        g.add((uri, SKOS.inScheme, concepts_scheme))
        g.add((uri, CEF.status, Literal(row["status"])))
        for lang in ("it", "en"):
            if row.get(f"label_{lang}"):
                g.add((uri, SKOS.prefLabel, Literal(row[f"label_{lang}"], lang=lang)))
            for synonym in split_list(row.get(f"synonyms_{lang}", "")):
                g.add((uri, SKOS.altLabel, Literal(synonym, lang=lang)))
        for target in split_list(row.get("broader", "")):
            g.add((uri, SKOS.broader, concept_uri(target)))
            g.add((concept_uri(target), SKOS.narrower, uri))
        for target in split_list(row.get("related", "")):
            g.add((uri, SKOS.related, concept_uri(target)))
            g.add((concept_uri(target), SKOS.related, uri))
        for name in split_list(row.get("graph_nodes", "")):
            g.add((uri, CEF.graphNode, Literal(name)))

    for row in live["framework_members"]:
        uri = member_uri(member_key(row))
        g.add((uri, RDF.type, SKOS.Concept))
        g.add((uri, RDF.type, CEF.Member))
        g.add((uri, SKOS.inScheme, framework_uri(row["framework_id"])))
        g.add((uri, CEF.position, Literal(row["position"])))
        g.add((uri, CEF.document, Literal(row["document"])))
        g.add((uri, CEF.page, Literal(row["page"])))
        g.add((uri, CEF.quote, Literal(row["quote"])))
        g.add((uri, CEF.status, Literal(row["status"])))
        for lang in ("it", "en"):
            if row.get(f"label_{lang}"):
                g.add((uri, SKOS.prefLabel, Literal(row[f"label_{lang}"], lang=lang)))
        if row.get("family"):
            g.add((uri, CEF.family, Literal(row["family"])))
        if row.get("concept_id"):
            g.add((uri, CEF.concept, concept_uri(row["concept_id"])))

    for row in live["definitions"]:
        node = BNode()
        g.add((node, RDF.type, CEF.Definition))
        g.add((node, RDF.value, Literal(row["quoted_definition"], lang=row["language"])))
        g.add((node, CEF.document, Literal(row["document"])))
        g.add((node, CEF.page, Literal(row["page"])))
        g.add((node, CEF.status, Literal(row["status"])))
        g.add((concept_uri(row["concept_id"]), CEF.definition, node))

    for row in live.get("cases", []):
        uri = CEF[f"case/{row['id']}"]
        g.add((uri, RDF.type, CEF.Case))
        g.add((uri, SKOS.prefLabel, Literal(row["name"], lang="it")))
        g.add((uri, CEF.document, Literal(row["document"])))
        g.add((uri, CEF.page, Literal(row["page"])))
        g.add((uri, CEF.status, Literal(row["status"])))
        for field, prop in (("territory", CEF.territory), ("actors", CEF.actors), ("supply_chain", CEF.supplyChain)):
            if row.get(field):
                g.add((uri, prop, Literal(row[field])))
        for code in split_list(row.get("strategies", "")):
            g.add((uri, CEF.appliesStrategy, member_uri(f"{row['framework_id']}#{code}")))

    for row in live["version_mappings"]:
        node = BNode()
        g.add((node, RDF.type, CEF.VersionMapping))
        if row.get("from"):
            g.add((node, CEF["from"], member_uri(row["from"])))
        g.add((node, CEF.to, member_uri(row["to"])))
        g.add((node, CEF.type, Literal(row["type"])))
        g.add((node, CEF.document, Literal(row["document"])))
        g.add((node, CEF.page, Literal(row["page"])))
        g.add((node, CEF.status, Literal(row["status"])))
        quote = QUOTE_IN_NOTES.search(row.get("notes", ""))
        if quote:
            g.add((node, CEF.quote, Literal(quote.group(1))))
    return g


def shacl_problems(g: Graph) -> list[str]:
    """Validate the graph against ``rules/shapes.ttl``.

    Args:
        g: The ontology graph.

    Returns:
        One readable line per violation.
    """
    shapes = Graph().parse(ONTOLOGY_DIR / "rules" / "shapes.ttl")
    conforms, results, _ = pyshacl.validate(g, shacl_graph=shapes, advanced=True)
    if conforms:
        return []
    SH = Namespace("http://www.w3.org/ns/shacl#")
    lines = []
    for result in results.subjects(RDF.type, SH.ValidationResult):
        focus = results.value(result, SH.focusNode)
        path = results.value(result, SH.resultPath)
        message = results.value(result, SH.resultMessage)
        lines.append(f"{focus} {path or ''}: {message}")
    return sorted(lines)


def skosify_problems(g: Graph) -> list[str]:
    """Run the Skosify checks without fixing anything.

    Skosify reports through ``logging``; a warning there is a problem here.

    Args:
        g: The ontology graph.

    Returns:
        One line per warning.
    """
    captured: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                captured.append(record.getMessage())

    handler = Collect()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        copy = Graph()
        for triple in g:
            copy.add(triple)
        skosify.check.hierarchy_cycles(copy, fix=False)
        skosify.check.disjoint_relations(copy, fix=False)
        skosify.check.hierarchical_redundancy(copy, fix=False)
        skosify.check.preflabel_uniqueness(copy, policy="shortest")
        skosify.check.label_overlap(copy, fix=False)
    finally:
        root.removeHandler(handler)
    return captured


def runtime_json(sheets: dict[str, list[dict]], rules: list[dict]) -> dict:
    """Precompute what the consultation module needs, so lookups are dict reads.

    Args:
        sheets: Sheet name → rows.
        rules: :func:`load_rules`.

    Returns:
        The content of ``ontology.json``.
    """
    live = {s: [r for r in rows if r.get("status") in LIVE] for s, rows in sheets.items()}
    concepts = {}
    for row in live["concepts"]:
        forms = [row.get("label_it", ""), row.get("label_en", "")]
        forms += split_list(row.get("synonyms_it", "")) + split_list(row.get("synonyms_en", ""))
        declared = split_list(row.get("graph_nodes", ""))
        concepts[row["id"]] = {
            "id": row["id"],
            "label_it": row.get("label_it", ""),
            "label_en": row.get("label_en", ""),
            "synonyms_it": split_list(row.get("synonyms_it", "")),
            "synonyms_en": split_list(row.get("synonyms_en", "")),
            "broader": split_list(row.get("broader", "")),
            "narrower": [],
            "related": split_list(row.get("related", "")),
            "graph_nodes": sorted(set(declared) | set(graph_nodes_for([f for f in forms if f]))),
            "status": row["status"],
            "notes": row.get("notes", ""),
            "definitions": [],
            "members": [],
        }
    for concept in concepts.values():
        for parent in concept["broader"]:
            concepts[parent]["narrower"].append(concept["id"])
        for other in concept["related"]:
            if concept["id"] not in concepts[other]["related"]:
                concepts[other]["related"].append(concept["id"])
    for concept in concepts.values():
        concept["ancestors"] = _ancestors(concepts, concept["id"])
    for row in live["definitions"]:
        concepts[row["concept_id"]]["definitions"].append(
            {k: row[k] for k in ("quoted_definition", "language", "document", "page", "status")}
        )

    frameworks = {}
    for row in live["frameworks"]:
        frameworks[row["id"]] = {
            **{k: row.get(k, "") for k in ("id", "name", "version", "author", "year", "document", "pages", "status", "notes")},
            "family": row["id"].split("_", 1)[0],
            "derived_from": split_list(row.get("derived_from", "")),
            "members": [],
        }
    for row in live["framework_members"]:
        member = {k: row.get(k, "") for k in SHEET_COLUMNS["framework_members"]}
        member["key"] = member_key(row)
        frameworks[row["framework_id"]]["members"].append(member)
        if row.get("concept_id") in concepts:
            concepts[row["concept_id"]]["members"].append(member["key"])
    for framework in frameworks.values():
        framework["members"].sort(key=lambda m: position_order(m["position"]))

    families = {}
    for concept_id, concept in concepts.items():
        versions = sorted(fid for fid, f in frameworks.items() if f["family"] == concept_id)
        if versions:
            families[concept_id] = {"concept": concept_id, "frameworks": versions}
    for rule in rules:
        if rule["type"] == "order" and rule["applies_to"] in families:
            families[rule["applies_to"]]["order"] = {
                k: rule[k] for k in ("id", "rule", "document", "page", "quote", "status")
            }

    mappings = [
        {k: row.get(k, "") for k in SHEET_COLUMNS["version_mappings"]}
        | {"quote": (QUOTE_IN_NOTES.search(row.get("notes", "")) or [None, ""])[1]}
        for row in live["version_mappings"]
    ]

    cases = {
        row["id"]: {
            **{k: row.get(k, "") for k in SHEET_COLUMNS["cases"]},
            "strategies": [f"{row['framework_id']}#{code}" for code in split_list(row.get("strategies", ""))],
        }
        for row in live.get("cases", [])
    }

    labels: dict[str, list[dict]] = {}

    def index(form: str, kind: str, ident: str) -> None:
        key = normalize(form)
        if key and {"kind": kind, "id": ident} not in labels.setdefault(key, []):
            labels[key].append({"kind": kind, "id": ident})

    for concept in concepts.values():
        for form in [concept["label_it"], concept["label_en"], *concept["synonyms_it"], *concept["synonyms_en"]]:
            index(form, "concept", concept["id"])
    for framework in frameworks.values():
        for form in (framework["id"], framework["name"], framework["version"]):
            index(form, "framework", framework["id"])
        for member in framework["members"]:
            for lang in ("it", "en"):
                if member[f"label_{lang}"] and member["concept_id"]:
                    index(member[f"label_{lang}"], "concept", member["concept_id"])
    for case in cases.values():
        index(case["name"], "case", case["id"])
        # "VIS MARIS - Mangime per pesci...": the name before the dash is how
        # the case is called in a question.
        short = case["name"].split(" - ")[0].strip(' "')
        if short != case["name"]:
            index(short, "case", case["id"])

    return {
        "meta": {
            "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "graph_backup": latest_backup().name,
            "usable_statuses": sorted(USABLE),
        },
        "families": families,
        "frameworks": frameworks,
        "concepts": concepts,
        "mappings": mappings,
        "cases": cases,
        "labels": labels,
        # Retrieved passages are labelled with the PDF file name, so the
        # ontology's evidence has to be too.
        "documents": {
            doc: document_filename(doc)
            for doc in sorted(
                {f["document"] for f in frameworks.values()}
                | {m["document"] for f in frameworks.values() for m in f["members"]}
                | {d["document"] for c in concepts.values() for d in c["definitions"]}
                | {m["document"] for m in mappings}
                | {c["document"] for c in cases.values()}
                | {f["order"]["document"] for f in families.values() if "order" in f}
            )
        },
    }


def _ancestors(concepts: dict, concept_id: str) -> list[str]:
    chain, frontier = [], list(concepts[concept_id]["broader"])
    while frontier:
        current = frontier.pop(0)
        if current in chain or current not in concepts:
            continue
        chain.append(current)
        frontier.extend(concepts[current]["broader"])
    return chain


def stale_graph_names(sheets: dict[str, list[dict]]) -> list[str]:
    """Node names written in ``graph_nodes`` that the latest backup no longer has.

    Args:
        sheets: Sheet name → rows.

    Returns:
        ``concept_id: name`` lines.
    """
    present = {name for names in graph_name_index().values() for name in names}
    return [
        f"{row['id']}: {name}"
        for row in sheets["concepts"]
        for name in split_list(row.get("graph_nodes", ""))
        if name not in present
    ]


def graph_counts(g: Graph) -> dict[str, int]:
    """Counts of the SKOS graph the quality report needs.

    Args:
        g: The graph built from the sheets.

    Returns:
        Triples, concepts, mappings to external vocabularies and subjects
        that are blank nodes (no identifier anyone else can point to).
    """
    matches = (SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch, SKOS.narrowMatch, SKOS.relatedMatch)
    subjects = set(g.subjects())
    return {
        "triples": len(g),
        "concepts": len(set(g.subjects(RDF.type, SKOS.Concept))),
        "external_mappings": sum(1 for p in matches for _ in g.triples((None, p, None))),
        "subjects": len(subjects),
        "blank_subjects": sum(isinstance(s, BNode) for s in subjects),
    }


def main(check_only: bool = False) -> int:
    sheets = load_sheets()
    rules = load_rules()
    graph = to_rdf(sheets)
    live_concepts = {r["id"] for r in sheets["concepts"] if r.get("status") in LIVE}
    sections = {
        "Controlli Python": [f"{p.sheet} {p.key}: {p.message}" for p in check_all(sheets)],
        "Regole": rule_problems(rules, live_concepts),
        "SHACL": shacl_problems(graph),
        "Skosify": skosify_problems(graph),
    }
    warnings = {"Nodi del grafo non più presenti nel backup": stale_graph_names(sheets)}
    failed = any(sections.values())
    if check_only:
        print(json.dumps({
            "problems": {title: len(problems) for title, problems in {**sections, **warnings}.items()},
            "graph": graph_counts(graph),
            "backup": latest_backup().name,
        }, ensure_ascii=False))
        return 1 if failed else 0

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    counts = {s: {} for s in sheets}
    for sheet, rows in sheets.items():
        for row in rows:
            counts[sheet][row["status"]] = counts[sheet].get(row["status"], 0) + 1
    lines = [
        "# Rapporto di build dell'ontologia",
        "",
        f"Esito: **{'FALLITA' if failed else 'riuscita'}** — backup del grafo {latest_backup().name}",
        "",
        "| foglio | " + " | ".join(sorted({s for c in counts.values() for s in c})) + " |",
    ]
    statuses = sorted({s for c in counts.values() for s in c})
    lines.append("|---" * (len(statuses) + 1) + "|")
    for sheet, by_status in counts.items():
        lines.append(f"| {sheet} | " + " | ".join(str(by_status.get(s, 0)) for s in statuses) + " |")
    for title, problems in {**sections, **warnings}.items():
        lines += ["", f"## {title}", ""]
        lines += [f"- {p}" for p in problems] or ["nessun problema"]
    (BUILD_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failed:
        print("build FALLITA, vedi ontology/build/report.md", file=sys.stderr)
        return 1
    graph.serialize(BUILD_DIR / "ontology.ttl", format="turtle")
    data = runtime_json(sheets, rules)
    (BUILD_DIR / "ontology.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"build riuscita: {len(data['frameworks'])} framework, {len(data['concepts'])} concetti, "
        f"{len(data['mappings'])} corrispondenze"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(check_only="--check-only" in sys.argv[1:]))
