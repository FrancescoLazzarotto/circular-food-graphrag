"""Tests for batched judging: prompt grouping, checkpoint resume, fallback."""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.judge.batch import (
    build_batch_prompt,
    parse_batch_array,
    score_dataset_batched,
)
from evalkit.judge.compare import compare_judges
from evalkit.judge.rubrics import get_rubric
from evalkit.models import EvalRow


def _row(qid: str, question: str, answer: str = "an answer", gt: str = "the truth") -> EvalRow:
    """An evaluation row with one context."""
    return EvalRow(
        run_dir="run1", strategy="default", framework="graphrag", model_id="qwen",
        run_index="0", question_id=qid, question_type="factoid", difficulty="easy",
        notes="", question=question, answer=answer, ground_truth=gt,
        answer_variants=[], contexts=["ctx"], retrieved_triples=[], retrieved_entities=[],
        expected_entities=[], gold_triples=[], latency_ms=0.0, kg_triples_used=0,
        kg_neighbors_used=0, kg_subgraph_triples_used=0, kg_shortest_path_triples_used=0,
        sub_questions=0, insufficient=False, skip_reason="",
    )


class _FakeBatchBackend:
    """Returns a JSON array scoring each item; records the prompts it saw."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        # one item per "### Item N" header
        n = user.count("### Item ")
        arr = [
            {"id": i, "factual_correctness": 0.8, "groundedness": 1.0,
             "relevance": 0.9, "rationale": f"ok {i}"}
            for i in range(n)
        ]
        return json.dumps(arr)


def test_parse_batch_array_with_prose_wrapper():
    raw = 'Sure!\n[{"id": 0, "answer_correctness": 1}]\nDone.'
    parsed = parse_batch_array(raw)
    assert parsed == [{"id": 0, "answer_correctness": 1}]


def test_parse_batch_array_garbage_returns_empty():
    assert parse_batch_array("no json here") == []
    assert parse_batch_array("") == []


def test_build_batch_prompt_has_one_item_per_row():
    rows = [_row("q1", "Q one?"), _row("q2", "Q two?")]
    rubrics = [get_rubric("factual_correctness"), get_rubric("completeness")]
    system, user = build_batch_prompt(rows, rubrics)
    assert user.count("### Item ") == 2
    assert "factual_correctness" in system and "completeness" in system


def test_score_dataset_batched_scores_all_rows_in_one_call_per_rubric_group():
    rows = [_row(f"q{i}", f"Q {i}?") for i in range(5)]
    backend = _FakeBatchBackend()
    result = score_dataset_batched(
        rows, backend=backend, rubric_names=["factual_correctness", "groundedness", "relevance"],
        batch_size=5, n_bootstrap=50,
    )
    # 5 rows, batch 5 → one call per group of rubrics that can share a prompt:
    # groundedness/relevance must not see the ground truth, factual_correctness needs it.
    assert len(backend.calls) == 2
    reference_free, reference_based = backend.calls
    assert "groundedness" in reference_free[0] and "relevance" in reference_free[0]
    assert "the truth" not in reference_free[1]
    assert "factual_correctness" in reference_based[0]
    assert "the truth" in reference_based[1]

    assert result["rows_evaluated"] == 5
    assert result["rubrics"]["factual_correctness"]["n"] == 5
    assert abs(result["rubrics"]["factual_correctness"]["mean"] - 0.8) < 1e-9
    assert result["rubrics"]["groundedness"]["n"] == 5


def test_score_dataset_batched_skips_skip_rows():
    rows = [_row("q1", "Q1?"), _row("q2", "Q2?")]
    rows[1].skip_reason = "no_gold"
    result = score_dataset_batched(
        rows, backend=_FakeBatchBackend(), rubric_names=["factual_correctness"],
        batch_size=8, n_bootstrap=50,
    )
    assert result["rows_evaluated"] == 1
    assert result["rows_skipped"] == 1


def test_checkpoint_resume_skips_completed(tmp_path: Path):
    rows = [_row(f"q{i}", f"Q {i}?") for i in range(4)]
    backend = _FakeBatchBackend()
    # First pass: score 2 of 4 by writing a checkpoint, then resume.
    score_dataset_batched(
        rows[:2], backend=backend, rubric_names=["factual_correctness"],
        batch_size=2, out_dir=tmp_path, n_bootstrap=50,
    )
    assert (tmp_path / "judge_rows.jsonl").exists()
    calls_before = len(backend.calls)

    # Resume over all 4: the first 2 are already in the checkpoint.
    result = score_dataset_batched(
        rows, backend=backend, rubric_names=["factual_correctness"],
        batch_size=2, out_dir=tmp_path, resume=True, n_bootstrap=50,
    )
    assert result["rows_evaluated"] == 4
    # Only the 2 pending rows triggered a new call (1 batch of size 2).
    assert len(backend.calls) == calls_before + 1


def test_batch_miss_falls_back_to_single():
    rows = [_row("q1", "Q1?"), _row("q2", "Q2?")]

    class _PartialBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system: str, user: str) -> str:
            self.calls += 1
            if "### Item " in user and user.count("### Item ") > 1:
                # Batch call: only return item 0, omit item 1 → forces fallback.
                return json.dumps([{"id": 0, "factual_correctness": 0.5, "rationale": "x"}])
            # Single-row fallback prompt.
            return json.dumps({"correctness_score": 1.0, "rationale": "y"})

    backend = _PartialBackend()
    result = score_dataset_batched(
        rows, backend=backend, rubric_names=["factual_correctness"], batch_size=2, n_bootstrap=50,
    )
    assert result["rows_evaluated"] == 2
    scores = [rs["factual_correctness"] for rs in result["row_scores"]]
    assert 0.5 in scores and 1.0 in scores  # one batched, one fallback


def test_compare_judges_reports_agreement():
    a = {
        "rubrics": {"answer_correctness": {"mean": 0.8}},
        "row_scores": [
            {"run_dir": "r", "strategy": "s", "question": "q1", "answer_correctness": 0.8},
            {"run_dir": "r", "strategy": "s", "question": "q2", "answer_correctness": 0.6},
        ],
    }
    b = {
        "rubrics": {"answer_correctness": {"mean": 0.85}},
        "row_scores": [
            {"run_dir": "r", "strategy": "s", "question": "q1", "answer_correctness": 0.9},
            {"run_dir": "r", "strategy": "s", "question": "q2", "answer_correctness": 0.7},
        ],
    }
    cmp = compare_judges(a, b, "haiku", "sonnet")
    assert cmp["n_matched"] == 2
    m = cmp["rubrics"]["answer_correctness"]
    assert m["n_paired"] == 2
    assert abs(m["mean_abs_diff"] - 0.1) < 1e-9
    assert abs(m["pearson"] - 1.0) < 1e-9  # perfectly correlated (both +0.1, monotone)
