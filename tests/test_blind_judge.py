"""The blind pairwise judgement: key, both orders, decision rule."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("blind_judge", _ROOT / "scripts/analysis/blind_judge.py")
blind_judge = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("blind_judge", blind_judge)
_spec.loader.exec_module(blind_judge)


def test_sign_test_matches_the_binomial():
    assert blind_judge.sign_test(0, 0) == 1.0
    assert blind_judge.sign_test(8, 2) == pytest.approx(0.1094, abs=1e-4)
    assert blind_judge.sign_test(9, 0) == pytest.approx(0.0039, abs=1e-4)


def _write(tmp_path: Path, key: dict, verdicts: list[dict], answers: list[dict]) -> argparse.Namespace:
    (tmp_path / "key.json").write_text(json.dumps(key))
    (tmp_path / "verdicts.jsonl").write_text("\n".join(json.dumps(v) for v in verdicts))
    path = tmp_path / "answers.jsonl"
    path.write_text("\n".join(json.dumps(a) for a in answers))
    return argparse.Namespace(answers=path, out=tmp_path)


def _answers(qid: str, group: str) -> list[dict]:
    return [{"id": qid, "arm": arm, "group": group, "intruders": [], "latency_s": 1.0} for arm in ("off", "on")]


def test_a_pair_counts_only_when_both_orders_agree(tmp_path):
    key = {"F1": {"A": "on", "B": "off"}, "F2": {"A": "off", "B": "on"}}
    verdicts = [
        # F1: "on" is first in AB and second in BA, and wins both times.
        {"id": "F1", "order": "AB", "winner": "first", "error": ""},
        {"id": "F1", "order": "BA", "winner": "second", "error": ""},
        # F2: the judge picks whatever comes first: a position effect, a tie.
        {"id": "F2", "order": "AB", "winner": "first", "error": ""},
        {"id": "F2", "order": "BA", "winner": "first", "error": ""},
    ]
    args = _write(tmp_path, key, verdicts, _answers("F1", "cq") + _answers("F2", "cq"))
    blind_judge.run_unblind(args)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["outcome"] == {"F1": "on", "F2": "tie"}
    assert result["framework"]["on"] == 1 and result["framework"]["tie"] == 1
    assert result["order_disagreements"] == 1
    # One win is not enough to switch the ontology on.
    assert not result["switch_on"]


def test_intruders_with_the_ontology_on_fail_the_rule(tmp_path):
    key = {"F1": {"A": "on", "B": "off"}}
    verdicts = [{"id": "F1", "order": o, "winner": "tie", "error": ""} for o in ("AB", "BA")]
    answers = _answers("F1", "cq")
    answers[1]["intruders"] = ["Rinnovare"]
    blind_judge.run_unblind(_write(tmp_path, key, verdicts, answers))
    result = json.loads((tmp_path / "result.json").read_text())
    assert not result["rule"]["2_intruders_not_more"]
