"""One expert budget across dimensions, on dimensions built by hand."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("expert_budget", _ROOT / "scripts/kg/quality/expert_budget.py")
expert_budget = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("expert_budget", expert_budget)
_spec.loader.exec_module(expert_budget)


def _units(stratum: str, count: int, labelled: int, rate: float) -> list[dict]:
    ones = round(labelled * rate)
    labels = [1] * ones + [0] * (labelled - ones)
    return [{"stratum": stratum, "y": labels[i] if i < labelled else None} for i in range(count)]


# A meter that separates right from wrong: its strata are almost pure.
SHARP = _units("judged_right", 800, 20, 0.95) + _units("judged_wrong", 200, 20, 0.10)
# A meter that says nothing: both strata have the same mix.
BLUNT = _units("a", 500, 20, 0.6) + _units("b", 500, 20, 0.6)
SPEC = {"dimensions": {"accuracy": {"cost_min": 1.0, "units": SHARP}, "coverage": {"cost_min": 2.0, "units": BLUNT}}}


def test_an_unlabelled_stratum_gets_the_widest_spread():
    table = expert_budget.strata([{"stratum": "x", "y": None}] * 3)
    assert table["x"]["S"] == 0.5 and table["x"]["p"] == 0.5


def test_a_pure_stratum_keeps_some_spread():
    table = expert_budget.strata([{"stratum": "x", "y": 1}] * 10)
    assert 0 < table["x"]["S"] < 0.3


def test_a_sharp_meter_narrows_the_interval_for_the_same_minutes():
    result = expert_budget.compare(SPEC, 120)["accuracy"]
    assert result["optimal"]["half_width"] < result["expert_only"]["half_width"]


def test_the_optimal_split_beats_the_uniform_one_on_the_summed_variance():
    result = expert_budget.compare(SPEC, 120)
    optimal = sum(d["optimal"]["half_width"] ** 2 for d in result.values())
    uniform = sum(d["uniform"]["half_width"] ** 2 for d in result.values())
    assert optimal <= uniform


def test_clustered_labels_need_more_minutes():
    clustered = {"dimensions": {d: {**v, "deff": 2.0} if d == "accuracy" else v for d, v in SPEC["dimensions"].items()}}
    assert expert_budget.minutes_for(clustered, 0.05) > expert_budget.minutes_for(SPEC, 0.05)


def test_the_minutes_for_a_target_reach_it():
    minutes = expert_budget.minutes_for(SPEC, 0.05)
    result = expert_budget.compare(SPEC, minutes)
    assert max(d["optimal"]["half_width"] for d in result.values()) <= 0.0501
    assert max(d["optimal"]["half_width"] for d in expert_budget.compare(SPEC, minutes * 0.9).values()) > 0.05
