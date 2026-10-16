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
