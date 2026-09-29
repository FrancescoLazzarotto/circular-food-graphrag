"""Tests for LLM-as-a-Judge: output parsing, score extraction, rubrics, caching."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
for p in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evalkit.judge.base import JudgeResult, extract_score, parse_judge_output
from evalkit.judge.llm_judge import LLMJudge
from evalkit.judge.rubrics import RUBRICS, get_rubric
from evalkit.models import EvalRow


class MockBackend:
    """Returns a fixed JSON response for any completion."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def complete(self, system: str, user: str) -> str:
        return json.dumps(self._payload)


def _make_row(skip_reason: str = "") -> EvalRow:
    """An evaluation row, skipped when `skip_reason` is set."""
    return EvalRow(
        run_dir="r1", strategy="default", framework="graphrag", model_id="m1",
        run_index="0", question_id="", question_type="factoid", difficulty="easy",
        notes="", question="What is aquafaba?",
        answer="Aquafaba is the liquid from canned chickpeas.",
        ground_truth="Aquafaba is the cooking liquid of chickpeas.",
        answer_variants=[],
        contexts=["Aquafaba is the water from canned chickpeas."],
        retrieved_triples=[{"subject": "aquafaba", "predicate": "IS", "object": "chickpea liquid"}],
        retrieved_entities=["aquafaba"],
        expected_entities=["aquafaba"],
        gold_triples=[],
        latency_ms=100.0,
        kg_triples_used=1, kg_neighbors_used=0,
        kg_subgraph_triples_used=0, kg_shortest_path_triples_used=0,
        sub_questions=1, insufficient=False, skip_reason=skip_reason,
    )


def test_parse_judge_output_direct_json() -> None:
    raw = '{"correctness_score": 0.8, "rationale": "Good answer."}'
    parsed, ok = parse_judge_output(raw)
    assert ok is True
    assert parsed["correctness_score"] == pytest.approx(0.8)


def test_parse_judge_output_embedded_json() -> None:
    raw = "Sure, here is the evaluation: {\"correctness_score\": 0.5, \"rationale\": \"OK\"} done."
    parsed, ok = parse_judge_output(raw)
    assert ok is True
    assert parsed["correctness_score"] == pytest.approx(0.5)


def test_parse_judge_output_empty() -> None:
    _, ok = parse_judge_output("")
    assert ok is False


def test_extract_score_normalises_likert() -> None:
    parsed = {"correctness_score": 4.0}
    score = extract_score(parsed, "correctness_score")
    assert score is not None
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.75)


def test_extract_score_0_to_1_passthrough() -> None:
    parsed = {"correctness_score": 0.7}
    score = extract_score(parsed, "correctness_score")
    assert score == pytest.approx(0.7)


def test_extract_score_missing() -> None:
    assert extract_score({}, "nonexistent") is None


def test_get_rubric_known() -> None:
    rubric = get_rubric("factual_correctness")
    assert rubric.name == "factual_correctness"
    assert rubric.score_field == "correctness_score"


def test_get_rubric_accepts_legacy_name() -> None:
    # The legacy name answer_correctness resolves to factual_correctness, so
    # older configs and artifacts still load.
    assert get_rubric("answer_correctness") is get_rubric("factual_correctness")


def test_get_rubric_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown rubric"):
        get_rubric("nonexistent_rubric")


def test_llm_judge_score_row() -> None:
    backend = MockBackend({
        "correctness_score": 0.9,
        "rationale": "Very accurate.",
    })
    judge = LLMJudge(backend=backend, rubric_names=["factual_correctness"])
    row = _make_row()
    results = judge.score_row(row)
    assert "factual_correctness" in results
    result = results["factual_correctness"]
    assert result.ok is True
    assert result.scores["factual_correctness"] == pytest.approx(0.9)
    assert "Very accurate" in result.rationale


def test_llm_judge_skips_row_with_skip_reason() -> None:
    backend = MockBackend({"correctness_score": 0.9, "rationale": "x"})
    judge = LLMJudge(backend=backend, rubric_names=["factual_correctness"])
    row = _make_row(skip_reason="no_gold")
    result = judge.score_dataset([row])
    assert result["rows_evaluated"] == 0
    assert result["rows_skipped"] == 1


def test_llm_judge_dataset_summary() -> None:
    backend = MockBackend({"correctness_score": 0.8, "rationale": "Fine."})
    judge = LLMJudge(backend=backend, rubric_names=["factual_correctness"])
    rows = [_make_row() for _ in range(3)]
    result = judge.score_dataset(rows)
    assert result["rows_evaluated"] == 3
    assert result["rows_skipped"] == 0
    summary = result["rubrics"]["factual_correctness"]
    assert summary["mean"] == pytest.approx(0.8)
    assert summary["n"] == 3


def test_llm_judge_caches_results() -> None:
    call_count = 0

    class CountingBackend:
        def complete(self, system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({"correctness_score": 0.7, "rationale": "ok"})

    judge = LLMJudge(backend=CountingBackend(), rubric_names=["factual_correctness"])
    row = _make_row()
    judge.score_row(row)
    judge.score_row(row)  # same row → should hit cache
    assert call_count == 1
