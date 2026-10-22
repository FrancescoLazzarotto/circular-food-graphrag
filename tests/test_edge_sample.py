"""The reading sample: probabilities that add up, supplements and repeats kept apart."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("edge_sample", _ROOT / "scripts/kg/quality/edge_sample.py")
edge_sample = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("edge_sample", edge_sample)
_spec.loader.exec_module(edge_sample)


def _rows() -> list[dict]:
    rng = random.Random(0)
    rows = []
    for c in range(60):
        for e in range(rng.randint(1, 12)):
            rows.append({
                "edge_key": f"k{c}_{e}", "chunk_id": f"c{c}", "type": "USES", "predicate": "",
                "violations": "published_org_doc" if e == 0 and c % 7 == 0 else "",
                "class_name_object": "0",
            })
    return rows


def test_main_sample_weights_add_up_to_the_graph():
    rows = _rows()
    readings = edge_sample.draw(rows, 10, 3, {}, 0.0, 2, seed=1)
    main = [r for r in readings if r["frame"] == "main"]
    assert sum(1 / r["pi_main"] for r in main) == pytest.approx(len(rows))
    assert len({r["chunk_id"] for r in main}) == 10


def test_supplements_and_repeats_are_marked():
    rows = _rows()
    readings = edge_sample.draw(rows, 10, 3, {"violation": 3}, 0.2, 2, seed=1)
    frames = [r["frame"] for r in readings]
    assert frames.count("supplement:violation") == 3
    repeated = [r for r in readings if r["frame"] == "double"]
    assert repeated and all(r["double_of"] == r["edge_key"] for r in repeated)
    keys = [r["edge_key"] for r in readings if r["frame"] != "double"]
    assert len(keys) == len(set(keys))
    assert {r["session"] for r in repeated} == {2}


def test_a_passage_larger_than_the_step_is_refused():
    with pytest.raises(ValueError):
        edge_sample.systematic_pps({"a": 50, "b": 1, "c": 1}, 2, random.Random(0))


def test_the_reader_sees_the_original_predicate():
    row = {"type": "RELATED_TO", "predicate": "PROVIDES"}
    assert edge_sample.relation_text(row) == "RELATED_TO (predicato originale: PROVIDES)"
