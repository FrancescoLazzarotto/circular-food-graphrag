"""The RDF 1.2 export of a backup, on a hand-built graph of three nodes."""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts/kg/quality"))
_spec = importlib.util.spec_from_file_location("graph_rdf_export", _ROOT / "scripts/kg/quality/graph_rdf_export.py")
graph_rdf_export = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("graph_rdf_export", graph_rdf_export)
_spec.loader.exec_module(graph_rdf_export)

NODES = [
    {"id": "4:x:1", "labels": ["Organization"], "props": {"name": "FAO", "aliases": ["Food and Agriculture Organization", "F.A.O."]}},
    {"id": "4:x:2", "labels": ["Concept"], "props": {"name": 'riuso "a km 0"\nlocale'}},
    {"id": "4:x:3", "labels": ["DataValue"], "props": {"name": "40 %", "value": 40, "share": 0.4, "checked": True}},
]
EDGE = {"source_doc": "a.pdf", "subject": "FAO", "object": "riuso", "confidence": 0.8, "page_range": "3"}
EDGES = [
    {"id": "5:x:1", "src": "4:x:1", "dst": "4:x:2", "type": "PROMOTES", "props": {**EDGE, "chunk_id": "c1"}},
    {"id": "5:x:2", "src": "4:x:1", "dst": "4:x:2", "type": "PROMOTES", "props": {**EDGE, "chunk_id": "c2", "page_range": "9"}},
    {"id": "5:x:3", "src": "4:x:2", "dst": "4:x:3", "type": "HAS_VALUE", "props": {**EDGE, "chunk_id": "c1"}},
]


def test_a_node_is_named_by_label_and_name_not_by_its_neo4j_id():
    a = graph_rdf_export.node_iri(NODES[0])
    b = graph_rdf_export.node_iri({**NODES[0], "id": "4:y:99"})
    assert a == b == "urn:cef-graph:node/Organization/FAO"


def test_values_keep_their_type():
    assert graph_rdf_export.literal(True).endswith("#boolean>")
    assert graph_rdf_export.literal(40).endswith("#integer>")
    assert graph_rdf_export.literal(0.4) == '"0.4"^^<http://www.w3.org/2001/XMLSchema#double>'
    assert graph_rdf_export.literal('a "b"\nc') == '"a \\"b\\"\\nc"'
    assert graph_rdf_export.literal(None) is None


def test_two_edges_between_the_same_nodes_keep_their_own_properties():
    lines, skipped = graph_rdf_export.triples(NODES, EDGES)
    asserted = [line for line in lines if ":rel/PROMOTES>" in line and "<<(" not in line]
    reifiers = [line for line in lines if "rdf-syntax-ns#reifies" in line and "PROMOTES" in line]
    assert len(asserted) == 1 and len(reifiers) == 2
    assert sum('"9"' in line for line in lines) == 1
    assert not skipped


def test_the_round_trip_is_a_fixed_point(tmp_path):
    pytest.importorskip("pyoxigraph")
    lines, _ = graph_rdf_export.triples(NODES, EDGES)
    path = tmp_path / "graph.nt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    nodes, edges = graph_rdf_export.read_back(path)
    lost = graph_rdf_export.compare(NODES, EDGES, nodes, edges)
    assert lost == {"nodes merged: same label and name": 0, "edges merged: same key": 0, "list aliases: order lost": 1}
    again, _ = graph_rdf_export.triples(nodes, edges)
    assert graph_rdf_export.canonical_hash(lines) == graph_rdf_export.canonical_hash(again)
