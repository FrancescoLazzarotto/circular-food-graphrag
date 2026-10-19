"""The small guarantees the analysis scripts give their readers.

A p95 one rank too low, an analysis that finds nothing and exits 0, statistics
of zeros for the wrong file: each reads as a result, not as a fault.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


analyze_matrix = _load("analyze_matrix", "scripts/analysis/analyze_matrix.py")
generate_questions = _load("generate_questions", "scripts/gold/generate_questions.py")


@pytest.mark.parametrize(("count", "expected"), [(20, 19.0), (30, 29.0), (1, 1.0), (100, 95.0)])
def test_p95_is_the_nearest_rank(count: int, expected: float) -> None:
    assert analyze_matrix._p95([float(v) for v in range(1, count + 1)]) == expected


def test_finding_no_run_is_a_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["analyze_matrix.py", str(tmp_path / "nothing")])
    with pytest.raises(SystemExit) as exit_info:
        analyze_matrix.main()
    assert exit_info.value.code == 1


def test_stats_refuses_a_file_that_is_not_a_suite(tmp_path) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"_meta": {}, "queries": [{"query": "x"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="questions"):
        generate_questions._stats(gold)


@pytest.mark.parametrize(("count", "expected"), [(20, 19.0), (30, 28.0), (1, 1.0), (100, 95.0)])
def test_the_floor_rank_gives_the_earlier_p95(count: int, expected: float) -> None:
    values = [float(v) for v in range(1, count + 1)]
    assert analyze_matrix._p95(values, floor_rank=True) == expected


def test_finding_no_run_can_exit_0_as_before(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["analyze_matrix.py", str(tmp_path / "nothing"), "--empty-ok"])
    with pytest.raises(SystemExit) as exit_info:
        analyze_matrix.main()
    assert exit_info.value.code == 0


def test_stats_can_read_a_file_without_questions_as_before(tmp_path, capsys) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"_meta": {}, "queries": [{"query": "x"}]}), encoding="utf-8")
    generate_questions._stats(gold, allow_missing=True)
    assert "0" in capsys.readouterr().out


def test_analyze_experiments_takes_the_same_p95_option() -> None:
    analyze_experiments = _load("analyze_experiments", "scripts/analysis/analyze_experiments.py")
    rows = [
        {"strategy": "s", "latency_ms": float(v), "kg_triples_used": 0, "kg_neighbors_used": 0,
         "kg_subgraph_triples_used": 0, "kg_shortest_path_triples_used": 0, "sub_questions": 0}
        for v in range(1, 31)
    ]
    assert analyze_experiments._aggregate(rows)["s"]["p95_latency_ms"] == 29.0
    assert analyze_experiments._aggregate(rows, p95_floor=True)["s"]["p95_latency_ms"] == 28.0
