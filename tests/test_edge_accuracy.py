"""The accuracy estimate of a labelled reading sample."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("edge_accuracy", _ROOT / "scripts/kg/quality/edge_accuracy.py")
edge_accuracy = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("edge_accuracy", edge_accuracy)
_spec.loader.exec_module(edge_accuracy)


def test_wilson_matches_the_textbook_value():
    low, high = edge_accuracy.wilson(0.5, 100)
    assert (round(low, 3), round(high, 3)) == (0.404, 0.596)


def test_one_edge_per_passage_has_no_design_effect():
    rows = [{"y": i % 4 != 0, "w": 1.0, "cluster": f"c{i}"} for i in range(400)]
    est = edge_accuracy.cluster_estimate(rows)
    assert est["p"] == 0.75
    assert est["deff"] == pytest.approx(1.0, abs=0.01)


def test_errors_that_share_a_passage_widen_the_interval():
    # Same share of correct edges, but whole passages right or wrong.
    spread = [{"y": i % 4 != 0, "w": 1.0, "cluster": f"c{i // 5}"} for i in range(400)]
    lumped = [{"y": (i // 5) % 4 != 0, "w": 1.0, "cluster": f"c{i // 5}"} for i in range(400)]
    a, b = edge_accuracy.cluster_estimate(spread), edge_accuracy.cluster_estimate(lumped)
    assert a["p"] == b["p"] == 0.75
    assert b["deff"] > 3 * a["deff"]
    assert b["high"] - b["low"] > a["high"] - a["low"]


def test_weights_count():
    rows = [{"y": 1, "w": 3.0, "cluster": "a"}, {"y": 0, "w": 1.0, "cluster": "b"}]
    assert edge_accuracy.cluster_estimate(rows)["p"] == 0.75


def test_kappa_of_identical_and_of_opposite_readings():
    assert edge_accuracy.kappa([(1, 1), (0, 0), (1, 1), (0, 0)])["kappa"] == 1.0
    assert edge_accuracy.kappa([(1, 0), (0, 1)])["kappa"] == -1.0


def test_strata_that_separate_errors_need_fewer_edges():
    rows = [{"y": 1, "w": 1.0, "s": "good"} for _ in range(90)] + [{"y": 0, "w": 1.0, "s": "bad"} for _ in range(10)]
    need = edge_accuracy.neyman_need(rows, "s", 0.05, 1.0)
    assert need["edges_needed_neyman"] == 0 and need["edges_needed_srs"] > 100
