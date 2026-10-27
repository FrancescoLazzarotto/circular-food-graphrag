"""The budget input built from meters and labels, on a hand-built sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("budget_spec", _ROOT / "scripts/kg/quality/budget_spec.py")
budget_spec = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("budget_spec", budget_spec)
_spec.loader.exec_module(budget_spec)


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.write_text("\n".join("\t".join(r) for r in [header, *rows]) + "\n", encoding="utf-8")
    return path


def test_the_dimensions_take_their_strata_and_labels(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    _write(sample / "sheet.tsv", ["session", "passage", "edge", "edge_key", "label"],
           [["1", "1", "1", "e1", "corretto"], ["1", "1", "2", "e2", "falso"], ["1", "2", "1", "e3", "non giudicabile"]])
    _write(sample / "times.tsv", ["session", "passage", "start", "end"], [["1", "1", "10:00", "10:03"], ["1", "2", "10:03", ""]])
    _write(sample / "design.tsv", ["edge_key", "frame"], [["e1", "main"], ["e2", "supplement"], ["e3", "main"]])
    signals = _write(tmp_path / "signals.tsv", ["edge_key", "violations"], [["e1", ""], ["e2", "published_org_doc"], ["e3", ""], ["e4", ""]])
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(json.dumps({"edges": [{"edge_key": k, "verdict": "supported"} for k in ("e1", "e2", "e3")]}))
    mappings = _write(tmp_path / "mappings.tsv", ["concept_id", "match_type", "status"],
                      [["c1", "closeMatch", "checked"], ["c2", "exactMatch", "rejected"], ["c3", "closeMatch", "proposed"]])
    args = argparse.Namespace(signals=signals, verdicts=verdicts, sample=sample, labels=None, mappings=mappings,
                              grounded=None, deff_accuracy=2.0, cost_accuracy=1.5, cost_mapping=3.0, cost_answer=3.0)
    spec = budget_spec.build(args)
    dims = spec["dimensions"]
    # Three minutes for the two labelled edges of passage 1.
    assert spec["minutes_per_edge_measured"] == 1.5
    # Only main-frame labels count for accuracy; the unjudgeable one is left out.
    assert [u["y"] for u in dims["accuracy"]["units"]] == [1, None, None, None]
    assert dims["accuracy"]["units"][3]["stratum"] == "missing"
    # A violation on an edge read as false is a real error.
    assert dims["conformity"]["units"] == [{"stratum": "published_org_doc", "y": 1}]
    assert [u["y"] for u in dims["interoperability"]["units"]] == [1, 0, None]
