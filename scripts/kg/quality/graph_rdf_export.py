"""Export the knowledge graph to RDF 1.2 from a backup, read it back, list what is lost.

Reads ``nodes.json`` and ``edges.json`` of a daily backup (never the database)
and writes N-Triples 1.2:

- a node is ``urn:cef-graph:node/<Label>/<name>``: Neo4j ids change at every
  restore, the label and the name do not. ``name`` becomes ``rdfs:label``,
  ``aliases`` one ``skos:altLabel`` each, any other property a typed literal;
- an edge is the asserted triple plus a named reifier
  (``urn:cef-graph:edge/<key>``, the restore-proof key of ``edge_signals``)
  that ``rdf:reifies`` the triple term and carries the edge properties. Two
  edges between the same nodes keep their own properties.

The file is then parsed with pyoxigraph, turned back into nodes and edges and
compared with the backup; the export is repeated from what was read back and
both are canonicalised with RDFC-1.0, so the hashes say whether the round trip
is a fixed point. ``report.json`` lists, with counts, everything that does not
survive.

Usage (pyoxigraph is needed only for the read-back)::

    python scripts/kg/quality/graph_rdf_export.py [--backup DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_signals import edge_key, latest_backup  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BASE = "urn:cef-graph:"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_REIFIES = "http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SKOS_ALT = "http://www.w3.org/2004/02/skos/core#altLabel"
XSD = "http://www.w3.org/2001/XMLSchema#"
# Properties with a standard term; every other one is urn:cef-graph:prop/<name>.
STANDARD = {"name": RDFS_LABEL, "aliases": SKOS_ALT}
EDGE_CLASS = f"{BASE}Edge"


def iri(kind: str, *parts: str) -> str:
    """An IRI of the graph namespace, each part percent-encoded."""
    return BASE + kind + "/" + "/".join(quote(str(p), safe="") for p in parts)


def node_iri(node: dict) -> str:
    """The identifier of a node: its first label and its name."""
    return iri("node", (node.get("labels") or ["Node"])[0], (node.get("props") or {}).get("name", node["id"]))


def _escape(text: str) -> str:
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20 or ch in "￾￿":
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return "".join(out)


def literal(value) -> str | None:
    """A value as an N-Triples literal, or None when RDF cannot hold it.

    Args:
        value: A property value (or one element of a list).

    Returns:
        The literal, typed by the Python type so it reads back the same.
    """
    if isinstance(value, bool):
        return f'"{str(value).lower()}"^^<{XSD}boolean>'
    if isinstance(value, int):
        return f'"{value}"^^<{XSD}integer>'
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return f'"{value!r}"^^<{XSD}double>'
    if isinstance(value, str):
        return f'"{_escape(value)}"'
    return None


def prop_iri(key: str) -> str:
    """The predicate of a property."""
    return STANDARD.get(key) or iri("prop", key)


def triples(nodes: list[dict], edges: list[dict]) -> tuple[list[str], Counter]:
    """The N-Triples lines of the graph, and what could not be written.

    Args:
        nodes: ``nodes.json`` of a backup.
        edges: ``edges.json`` of the same backup.

    Returns:
        The sorted, distinct lines, and counts of values left out by kind.
    """
    skipped: Counter = Counter()
    lines: set[str] = set()
    by_id = {}
    for node in nodes:
        subject = f"<{node_iri(node)}>"
        by_id[node["id"]] = subject
        for label in node.get("labels") or []:
            lines.add(f"{subject} <{RDF_TYPE}> <{iri('type', label)}> .")
        for key, value in (node.get("props") or {}).items():
            for item in value if isinstance(value, list) else [value]:
                term = literal(item)
                if term is None:
                    skipped[f"node property {key}: {type(item).__name__}"] += 1
                    continue
                lines.add(f"{subject} <{prop_iri(key)}> {term} .")
    for edge in edges:
        if edge["src"] not in by_id or edge["dst"] not in by_id:
            skipped["edge with an end missing from nodes.json"] += 1
            continue
        props = edge.get("props") or {}
        statement = f"{by_id[edge['src']]} <{iri('rel', edge['type'])}> {by_id[edge['dst']]}"
        reifier = f"<{iri('edge', edge_key(props, edge['type']))}>"
        lines.add(f"{statement} .")
        lines.add(f"{reifier} <{RDF_REIFIES}> <<( {statement} )>> .")
        lines.add(f"{reifier} <{RDF_TYPE}> <{EDGE_CLASS}> .")
        for key, value in props.items():
            for item in value if isinstance(value, list) else [value]:
                term = literal(item)
                if term is None:
                    skipped[f"edge property {key}: {type(item).__name__}"] += 1
                    continue
                lines.add(f"{reifier} <{prop_iri(key)}> {term} .")
    return sorted(lines), skipped


def _value(term):
    import pyoxigraph as ox

    if not isinstance(term, ox.Literal):
        return term.value
    datatype = term.datatype.value
    if datatype == f"{XSD}boolean":
        return term.value == "true"
    if datatype == f"{XSD}integer":
        return int(term.value)
    if datatype == f"{XSD}double":
        return float(term.value)
    return term.value


def _key(predicate: str) -> str:
    for key, standard in STANDARD.items():
        if predicate == standard:
            return key
    return unquote(predicate.removeprefix(BASE + "prop/"))


def read_back(path: Path) -> tuple[list[dict], list[dict]]:
    """Parse an export and rebuild nodes and edges in the backup's shape.

    Node and edge ids are the IRIs; list properties come back sorted,
    because RDF keeps a set of values, not their order.

    Args:
        path: The gzipped N-Triples file.

    Returns:
        ``(nodes, edges)``.
    """
    import pyoxigraph as ox

    with gzip.open(path, "rb") as handle:
        quads = list(ox.parse(handle, format=ox.RdfFormat.N_TRIPLES))
    nodes: dict[str, dict] = {}
    reified: dict[str, dict] = {}
    multi: dict[tuple[str, str], list] = {}
    for q in quads:
        s, p, o = q.subject.value, q.predicate.value, q.object
        if p == RDF_REIFIES:
            reified.setdefault(s, {"props": {}})["triple"] = o
            continue
        if s.startswith(BASE + "edge/"):
            if p != RDF_TYPE:
                multi.setdefault((s, _key(p)), []).append(_value(o))
            continue
        if not s.startswith(BASE + "node/"):
            continue
        if p == RDF_TYPE:
            nodes.setdefault(s, {"id": s, "labels": [], "props": {}})["labels"].append(unquote(o.value.removeprefix(BASE + "type/")))
        elif p.startswith(BASE + "rel/"):
            continue
        else:
            nodes.setdefault(s, {"id": s, "labels": [], "props": {}})
            multi.setdefault((s, _key(p)), []).append(_value(o))
    for (s, key), values in multi.items():
        target = reified.setdefault(s, {"props": {}}) if s.startswith(BASE + "edge/") else nodes[s]
        target["props"][key] = sorted(values, key=repr) if len(values) > 1 or key == "aliases" else values[0]
    edges = []
    for s, item in reified.items():
        triple = item["triple"]
        edges.append({
            "id": s,
            "src": triple.subject.value,
            "dst": triple.object.value,
            "type": unquote(triple.predicate.value.removeprefix(BASE + "rel/")),
            "props": item["props"],
        })
    return list(nodes.values()), edges


def canonical_hash(lines: list[str]) -> str:
    """SHA-256 of the RDFC-1.0 canonical N-Quads of some N-Triples lines."""
    import pyoxigraph as ox

    dataset = ox.Dataset(ox.parse("\n".join(lines) + "\n", format=ox.RdfFormat.N_TRIPLES))
    dataset.canonicalize(ox.CanonicalizationAlgorithm.RDFC_1_0)
    data = ox.serialize(dataset, format=ox.RdfFormat.N_QUADS)
    return hashlib.sha256(b"\n".join(sorted(data.splitlines()))).hexdigest()


def compare(nodes: list[dict], edges: list[dict], back_nodes: list[dict], back_edges: list[dict]) -> dict[str, int]:
    """What the read-back lost or changed, by kind.

    Args:
        nodes, edges: The backup.
        back_nodes, back_edges: :func:`read_back` of its export.

    Returns:
        Counts by kind of loss.
    """
    lost: Counter = Counter()
    back_by_iri = {n["id"]: n for n in back_nodes}
    seen: Counter = Counter(node_iri(n) for n in nodes)
    lost["nodes merged: same label and name"] = sum(c - 1 for c in seen.values() if c > 1)
    for node in nodes:
        back = back_by_iri.get(node_iri(node))
        if back is None:
            lost["nodes missing"] += 1
            continue
        if seen[node_iri(node)] > 1:
            continue
        for key, value in (node.get("props") or {}).items():
            got = back["props"].get(key)
            if isinstance(value, list):
                if sorted(value, key=repr) != got:
                    lost[f"list {key}: duplicates dropped" if len(set(map(repr, value))) < len(value) else f"list {key}: changed"] += 1
                elif value != got:
                    lost[f"list {key}: order lost"] += 1
            elif value != got:
                lost[f"node property {key}: changed"] += 1
    keys = Counter(edge_key(e.get("props") or {}, e["type"]) for e in edges)
    lost["edges merged: same key"] = sum(c - 1 for c in keys.values() if c > 1)
    back_edges_by_key = {e["id"].removeprefix(BASE + "edge/"): e for e in back_edges}
    for edge in edges:
        key = edge_key(edge.get("props") or {}, edge["type"])
        if keys[key] > 1:
            continue
        back = back_edges_by_key.get(key)
        if back is None:
            lost["edges missing"] += 1
            continue
        if back["type"] != edge["type"]:
            lost["edge type changed"] += 1
        for prop, value in (edge.get("props") or {}).items():
            if back["props"].get(prop) != value:
                lost[f"edge property {prop}: changed"] += 1
    # The two identifiers are where a loss would hide, so they are reported
    # even when nothing was merged.
    always = ("nodes merged: same label and name", "edges merged: same key")
    return {key: lost[key] for key in always} | {k: v for k, v in sorted(lost.items()) if k not in always and v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backup", type=Path, default=None, help="backup folder (default: the latest)")
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    args = parser.parse_args()
    backup = args.backup or latest_backup()
    out = args.out or ROOT / "artifacts" / "kg_eval_2026-10" / "rdf_export" / backup.name
    out.mkdir(parents=True, exist_ok=True)
    nodes = json.loads((backup / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((backup / "edges.json").read_text(encoding="utf-8"))
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((backup / "schema.json").read_text(encoding="utf-8"))

    lines, skipped = triples(nodes, edges)
    path = out / "graph.nt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    back_nodes, back_edges = read_back(path)
    lost = compare(nodes, edges, back_nodes, back_edges)
    again, _ = triples(back_nodes, back_edges)
    first, second = canonical_hash(lines), canonical_hash(again)
    report = {
        "backup": backup.name,
        "triples": len(lines),
        "nodes": {"backup": len(nodes), "read_back": len(back_nodes)},
        "edges": {"backup": len(edges), "read_back": len(back_edges)},
        "values_not_written": dict(skipped),
        "lost_in_round_trip": dict(lost),
        "not_exported": {
            "neo4j element ids (nodes + edges)": len(nodes) + len(edges),
            "indexes": len(schema.get("indexes", [])),
            "constraints": len(schema.get("constraints", [])),
            "vectors in the backup": 0 if not manifest.get("include_vectors") else "present, not exported",
        },
        "rdfc10_sha256": {"export": first, "export_of_read_back": second, "fixed_point": first == second},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
