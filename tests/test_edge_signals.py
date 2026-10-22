"""The per-edge schema signals, on a hand-built backup of four edges."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("edge_signals", _ROOT / "scripts/kg/quality/edge_signals.py")
edge_signals = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("edge_signals", edge_signals)
_spec.loader.exec_module(edge_signals)

NODES = [
    {"id": "n1", "labels": ["Organization"], "props": {"name": "FAO"}},
    {"id": "n2", "labels": ["Document"], "props": {"name": "Rapporto"}},
    {"id": "n3", "labels": ["Concept"], "props": {"name": "circolarità"}},
    {"id": "n4", "labels": ["Concept"], "props": {"name": "Person"}},
    {"id": "n5", "labels": ["Person"], "props": {"name": "Rossi"}},
]


def _edge(i: str, src: str, dst: str, edge_type: str, chunk: str, **extra) -> dict:
    props = {"chunk_id": chunk, "source_doc": "doc.pdf", "page_range": "1-1", "subject": src, "object": dst,
             "confidence": 0.8, "mention_count": 1, **extra}
    return {"id": i, "src": src, "dst": dst, "type": edge_type, "props": props}


EDGES = [
    _edge("e1", "n1", "n2", "PUBLISHED", "c1"),
    _edge("e2", "n2", "n1", "PUBLISHED", "c1"),
    _edge("e3", "n3", "n4", "RELATED_TO", "c1", predicate="IS_A"),
    _edge("e4", "n5", "n3", "HAS_VALUE", "c2"),
]


def _rows() -> dict[str, dict]:
    rows = edge_signals.signals(NODES, EDGES, {"PUBLISHED", "RELATED_TO", "HAS_VALUE"}, edge_signals.PROMPT_RULES)
    return {r["backup_id"]: r for r in rows}


def test_domain_and_range_rules():
    rows = _rows()
    assert rows["e1"]["violations"] == ""
    assert rows["e2"]["violations"] == "published_org_doc"
    assert rows["e4"]["violations"] == "has_value_datavalue"


def test_fallback_concept_ends_and_class_names():
    row = _rows()["e3"]
    assert (row["fallback"], row["fallback_predicate"], row["predicate"]) == (1, 1, "IS_A")
    assert row["concept_end"] == "both"
    assert row["class_name_object"] == 1
    assert row["family"] == "fallback"


def test_passage_size_and_stable_key():
    rows = _rows()
    assert rows["e1"]["chunk_edges"] == 3 and rows["e4"]["chunk_edges"] == 1
    moved = [dict(e, id="other") for e in EDGES]
    again = edge_signals.signals(NODES, moved, set(), edge_signals.PROMPT_RULES)
    assert [r["edge_key"] for r in again] == [rows[e["id"]]["edge_key"] for e in EDGES]


def test_summary_counts_exposure():
    rows = list(_rows().values())
    report = edge_signals.summary(rows, edge_signals.PROMPT_RULES, Path("x"))
    assert report["exposed_to_rules"] == 3
    assert report["edges_with_violation"] == 2
    assert report["duplicate_edge_keys"] == 0
