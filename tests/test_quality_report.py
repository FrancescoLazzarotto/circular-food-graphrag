"""The nightly quality report keeps its dimensions apart and every count with its denominator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts/kg/quality"))
_spec = importlib.util.spec_from_file_location("quality_report", _ROOT / "scripts/kg/quality/quality_report.py")
quality_report = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("quality_report", quality_report)
_spec.loader.exec_module(quality_report)


def test_a_share_keeps_numerator_and_denominator():
    report = quality_report.Report("2026-10-22T00:00:00Z")
    report.add("conformity", "fallback", 25, 100, "backup", "src")
    report.add("conformity", "problems", 3, None, "build", "src")
    report.add("accuracy", "edge_accuracy", None, None, "", "", "none yet")
    share, count, empty = report.rows
    assert (share["value"], share["numerator"], share["denominator"]) == (0.25, 25, 100)
    assert (count["value"], count["denominator"]) == (3, "")
    assert empty["value"] == "" and empty["note"] == "none yet"


def test_one_section_per_dimension_and_history(tmp_path):
    report = quality_report.Report("2026-10-22T00:00:00Z")
    for dimension in ("conformity", "interoperability", "coverage", "accuracy"):
        report.add(dimension, "m", 1, 2, "x", "y")
    run = quality_report.write(report, tmp_path, "backup1")
    quality_report.write(report, tmp_path, "backup2")
    text = (run / "report.md").read_text(encoding="utf-8")
    assert [line for line in text.splitlines() if line.startswith("## ")] == [
        "## conformity", "## interoperability", "## coverage", "## accuracy",
    ]
    assert len((tmp_path / "history.tsv").read_text(encoding="utf-8").splitlines()) == 1 + 2 * 4
