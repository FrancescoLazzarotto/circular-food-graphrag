"""The expert sheet holds what the cheaper readers could not settle, never their verdicts."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("expert_sheet", _ROOT / "scripts/kg/quality/expert_sheet.py")
expert_sheet = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("expert_sheet", expert_sheet)
_spec.loader.exec_module(expert_sheet)


def test_doubtful_answers_all_go_in_and_no_verdict_shows(tmp_path):
    answers = tmp_path / "answers.jsonl"
    grounded = tmp_path / "grounded.jsonl"
    questions = [f"q{i}" for i in range(6)]
    answers.write_text("\n".join(json.dumps({"question": q, "answer": f"risposta {q}"}) for q in questions))
    verdicts = [{"question": q, "answers": "yes", "supported": "partly" if i < 2 else "yes"} for i, q in enumerate(questions)]
    grounded.write_text("\n".join(json.dumps(v) for v in verdicts))
    rows = expert_sheet.answer_rows(answers, grounded, 1, random.Random(42))
    assert {r[0]["id"] for r in rows} >= {"q0", "q1"} and len(rows) == 3
    for shown, hidden, text in rows:
        assert "partly" not in json.dumps(shown) + text and "supported" not in json.dumps(shown) + text
        assert hidden["judge"].startswith("answers=")


def test_edges_where_person_and_judge_disagree_are_picked(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    header = "session\tpassage\tedge\tedge_key\tsubject\trelation\tobject\tlabel\terror_type\tnote\n"
    lines = [
        "1\t1\t1\te1\tA\tUSES\tB\tcorretto\t\t\n",
        "1\t1\t2\te2\tA\tUSES\tC\tfalso\t\t\n",
        "1\t10\t1\te3\tD\tUSES\tE\tnon giudicabile\t\t\n",
        "1\t10\t2\te4\tD\tUSES\tF\tcorretto\t\t\n",
    ]
    (sample / "sheet.tsv").write_text(header + "".join(lines))
    (sample / "passages.md").write_text("## Passaggio 1 (seduta 1)\ntesto uno\n\n## Passaggio 10 (seduta 1)\ntesto dieci\n")
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(json.dumps({"edges": [
        {"edge_key": "e1", "verdict": "not_supported"}, {"edge_key": "e2", "verdict": "not_supported"},
        {"edge_key": "e3", "verdict": "supported"}, {"edge_key": "e4", "verdict": "supported"},
    ]}))
    rows = expert_sheet.edge_rows(sample, verdicts, 10, random.Random(42))
    assert {r[0]["id"] for r in rows} == {"e1", "e3"}
    by_id = {r[0]["id"]: r for r in rows}
    assert "testo dieci" in by_id["e3"][2] and "testo uno" not in by_id["e3"][2]
    assert "supported" not in json.dumps(by_id["e1"][0])
