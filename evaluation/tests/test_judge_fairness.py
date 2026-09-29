"""Judge fairness guarantees.

These tests pin the properties that make a cross-pipeline comparison meaningful.
They are cheap and structural on purpose: they check what the judge is *shown*,
not what it answers, so they need no LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
for _p in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evalkit.judge.batch import (
    _row_key,
    build_batch_prompt,
    partition_by_ground_truth,
    score_dataset_batched,
)
from evalkit.judge.llm_judge import LLMJudge
from evalkit.judge.prompts import (
    EVIDENCE_CHAR_BUDGET,
    EVIDENCE_HEADER,
    NO_EVIDENCE_TEXT,
    build_prompt,
    render_evidence,
)
from evalkit.judge.rubrics import (
    CANONICAL_RUBRICS,
    RUBRIC_ALIASES,
    RUBRICS,
    canonical_rubric_name,
    get_rubric,
    resolve_rubrics,
    rubrics_for_row,
)
from evalkit.models import EvalRow, GoldQuery

QUESTION = "What is aquafaba used for?"
ANSWER = "Aquafaba is used as an egg-white substitute in vegan baking."
GROUND_TRUTH = "Aquafaba, the cooking liquid of chickpeas, replaces egg white in vegan recipes."

# The same fact, in the two native forms the pipelines expose it in.
FACT_AS_TRIPLE = {"subject": "aquafaba", "predicate": "SUBSTITUTES_FOR", "object": "egg white"}
FACT_AS_TEXT = "aquafaba — substitutes for — egg white"


def _row(
    *,
    contexts: list[str] | None = None,
    triples: list[dict] | None = None,
    pipeline: str = "",
    strategy: str = "default",
    ground_truth: str = GROUND_TRUTH,
    distractor: bool = False,
    question: str = QUESTION,
    answer: str = ANSWER,
) -> EvalRow:
    """An evaluation row carrying text contexts, triples, or both."""
    gold_query = None
    if distractor:
        gold_query = GoldQuery(
            query_id="Q14",
            query_type="distractor",
            query=question,
            expected_answer="Not answerable from the supplied corpus.",
            expected_entities=(),
            distractor_expected=True,
        )
    return EvalRow(
        run_dir="r1", strategy=strategy, framework="graphrag", model_id="m1",
        run_index="0", question_id="Q01", question_type="factoid", difficulty="easy",
        notes="", question=question, answer=answer, ground_truth=ground_truth,
        answer_variants=[], contexts=contexts or [], retrieved_triples=triples or [],
        retrieved_entities=[], expected_entities=[], gold_triples=[], latency_ms=1.0,
        kg_triples_used=0, kg_neighbors_used=0, kg_subgraph_triples_used=0,
        kg_shortest_path_triples_used=0, sub_questions=0, insufficient=False,
        skip_reason="", gold_query=gold_query, pipeline=pipeline,
    )


class _RecordingBackend:
    """Records every prompt it is asked to complete."""

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {
            "correctness_score": 0.8, "groundedness_score": 0.8,
            "relevance_score": 0.8, "completeness_score": 0.8,
            "abstention_score": 0.8, "rationale": "ok",
        }
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return json.dumps(self.payload)


# ── the prompt must not reveal the pipeline ───────────────────────────────────


def test_prompt_is_byte_identical_across_pipelines() -> None:
    """Same answer, same evidence content, different native form → same prompt."""
    text_rag = _row(contexts=[FACT_AS_TEXT], pipeline="text_rag", strategy="text_only")
    graph_rag = _row(triples=[FACT_AS_TRIPLE], pipeline="graph_rag", strategy="subgraph_2hop")

    for rubric_name in sorted(CANONICAL_RUBRICS):
        rubric = get_rubric(rubric_name)
        if rubric.applies_to != ("answerable",):
            continue
        assert build_prompt(text_rag, rubric) == build_prompt(graph_rag, rubric), rubric_name


def test_prompt_never_names_the_pipeline_or_strategy() -> None:
    row = _row(triples=[FACT_AS_TRIPLE], pipeline="ontology_grounded", strategy="subgraph_2hop")
    system, user = build_prompt(row, get_rubric("groundedness"))
    blob = system + user
    for leak in ("ontology_grounded", "subgraph_2hop", "graphrag", "Triple", "triples", "KG"):
        assert leak not in blob


def test_evidence_uses_one_header_whatever_the_row_carries() -> None:
    both = _row(contexts=["a chunk of prose"], triples=[FACT_AS_TRIPLE])
    rendered = render_evidence(both)
    assert rendered.count(EVIDENCE_HEADER) == 1
    assert "Retrieved Text Contexts" not in rendered
    assert "Retrieved KG Triples" not in rendered
    # Both elements survive, one line each, indistinguishable in form.
    assert rendered.splitlines()[1:] == ["- a chunk of prose", f"- {FACT_AS_TEXT}"]


def test_evidence_section_is_present_even_when_nothing_was_retrieved() -> None:
    rendered = render_evidence(_row())
    assert rendered == f"{EVIDENCE_HEADER}\n{NO_EVIDENCE_TEXT}"


def test_triple_is_serialised_as_a_statement() -> None:
    row = _row(triples=[{"subject": "whey", "predicate": "IS_BYPRODUCT_OF", "object": "cheese"}])
    assert "- whey — is byproduct of — cheese" in render_evidence(row)


def test_multiline_context_is_flattened_to_one_line() -> None:
    row = _row(contexts=["line one\nline two\n\nline three"])
    body = render_evidence(row).splitlines()[1:]
    assert body == ["- line one line two line three"]


# ── the evidence cap is a character budget, symmetric across pipelines ────────


def test_evidence_cap_is_character_based_not_element_based() -> None:
    """Element caps would keep 20 triples and 5 chunks: only the budget may bind."""
    many_triples = [
        {"subject": f"s{i}", "predicate": "RELATES_TO", "object": f"o{i}"} for i in range(40)
    ]
    rendered = render_evidence(_row(triples=many_triples))
    body = rendered[len(EVIDENCE_HEADER) + 1 :]
    # Far more than a 20-element cap would keep, because they all fit the budget.
    assert len(body.splitlines()) == 40
    assert len(body) <= EVIDENCE_CHAR_BUDGET

    many_contexts = [f"context number {i}" for i in range(40)]
    body_text = render_evidence(_row(contexts=many_contexts))[len(EVIDENCE_HEADER) + 1 :]
    assert len(body_text.splitlines()) == 40


def test_evidence_cap_is_symmetric_between_prose_and_triples() -> None:
    """A pipeline cannot buy more judge attention by being verbose."""
    fat_contexts = ["x" * 900 for _ in range(20)]
    fat_triples = [
        {"subject": "s" * 300, "predicate": "P" * 300, "object": "o" * 300} for _ in range(20)
    ]
    prose_body = render_evidence(_row(contexts=fat_contexts))[len(EVIDENCE_HEADER) + 1 :]
    triple_body = render_evidence(_row(triples=fat_triples))[len(EVIDENCE_HEADER) + 1 :]

    assert len(prose_body) <= EVIDENCE_CHAR_BUDGET
    assert len(triple_body) <= EVIDENCE_CHAR_BUDGET
    # Both are held to the same budget, so neither is starved relative to the other.
    assert abs(len(prose_body) - len(triple_body)) < EVIDENCE_CHAR_BUDGET * 0.1


def test_evidence_budget_truncates_a_single_oversized_element() -> None:
    body = render_evidence(_row(contexts=["y" * 10_000]), char_budget=500)
    body = body[len(EVIDENCE_HEADER) + 1 :]
    assert len(body) <= 500
    assert body.endswith("…")


def test_evidence_budget_is_honoured_exactly() -> None:
    for budget in (120, 500, 1000, 4000):
        row = _row(contexts=[f"context {i} " + "z" * 200 for i in range(50)])
        body = render_evidence(row, char_budget=budget)[len(EVIDENCE_HEADER) + 1 :]
        assert len(body) <= budget, budget


# ── the ground truth reaches only the rubrics that ask for it ─────────────────


def test_groundedness_prompt_has_no_ground_truth() -> None:
    row = _row(contexts=[FACT_AS_TEXT])
    system, user = build_prompt(row, get_rubric("groundedness"))
    assert GROUND_TRUTH not in user
    assert "Ground Truth" not in user
    # And the rubric does not have to ask the model to unsee it.
    assert "IGNORE the Ground Truth" not in user


def test_factual_correctness_prompt_has_the_ground_truth() -> None:
    row = _row(contexts=[FACT_AS_TEXT])
    _, user = build_prompt(row, get_rubric("factual_correctness"))
    assert "## Ground Truth Answer" in user
    assert GROUND_TRUTH in user


def test_reference_free_rubrics_are_declared_reference_free() -> None:
    assert get_rubric("groundedness").uses_ground_truth is False
    assert get_rubric("relevance").uses_ground_truth is False
    assert get_rubric("abstention").uses_ground_truth is False
    assert get_rubric("factual_correctness").uses_ground_truth is True
    assert get_rubric("completeness").uses_ground_truth is True


def test_batch_prompt_refuses_to_mix_reference_free_and_reference_based_rubrics() -> None:
    rows = [_row(contexts=[FACT_AS_TEXT])]
    mixed = [get_rubric("factual_correctness"), get_rubric("groundedness")]
    with pytest.raises(ValueError, match="disagree about the ground truth"):
        build_batch_prompt(rows, mixed)


def test_batch_prompt_hides_ground_truth_from_reference_free_rubrics() -> None:
    rows = [_row(contexts=[FACT_AS_TEXT])]
    _, user = build_batch_prompt(rows, [get_rubric("groundedness"), get_rubric("relevance")])
    assert GROUND_TRUTH not in user
    _, user_gt = build_batch_prompt(rows, [get_rubric("factual_correctness")])
    assert GROUND_TRUTH in user_gt


def test_partition_by_ground_truth_splits_the_default_set() -> None:
    groups = partition_by_ground_truth(
        resolve_rubrics(["factual_correctness", "completeness", "groundedness", "relevance"])
    )
    assert [[r.name for r in g] for g in groups] == [
        ["groundedness", "relevance"],
        ["factual_correctness", "completeness"],
    ]


# ── distractors are scored on abstention alone ────────────────────────────────


def test_distractor_rows_select_only_abstention() -> None:
    requested = resolve_rubrics(["factual_correctness", "completeness", "groundedness", "relevance"])
    selected = rubrics_for_row(_row(distractor=True), requested)
    assert [r.name for r in selected] == ["abstention"]


def test_answerable_rows_never_get_abstention() -> None:
    requested = resolve_rubrics(["factual_correctness", "groundedness", "relevance"])
    selected = rubrics_for_row(_row(contexts=[FACT_AS_TEXT]), requested)
    assert [r.name for r in selected] == ["factual_correctness", "groundedness", "relevance"]


def test_judge_runs_only_abstention_on_distractors() -> None:
    backend = _RecordingBackend()
    judge = LLMJudge(backend=backend, rubric_names=["factual_correctness", "groundedness"])
    results = judge.score_row(_row(distractor=True, contexts=["unrelated prose"]))

    assert list(results) == ["abstention"]
    assert len(backend.prompts) == 1
    assert "abstention_score" in backend.prompts[0][0]


def test_abstention_prompt_does_not_reveal_that_the_question_is_a_distractor() -> None:
    """The system under test had to work that out; the judge must not be told."""
    row = _row(distractor=True, contexts=["unrelated prose"])
    _, user = build_prompt(row, get_rubric("abstention"))
    assert "Not answerable from the supplied corpus." not in user
    assert "distractor" not in user.lower()


class _BatchBackend:
    """Scores every item on whatever rubric keys the system prompt asks for."""

    _KNOWN = ("factual_correctness", "completeness", "groundedness", "relevance", "abstention")

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        keys = [k for k in self._KNOWN if f'"{k}"' in system]
        n = user.count("### Item ")
        return json.dumps([{"id": i, **{k: 0.5 for k in keys}, "rationale": "r"} for i in range(n)])


def test_batched_path_scores_distractors_on_abstention_only() -> None:
    rows = [
        _row(contexts=[FACT_AS_TEXT], question="Answerable?"),
        _row(distractor=True, contexts=["unrelated prose"], question="Distractor?"),
    ]
    backend = _BatchBackend()
    result = score_dataset_batched(
        rows, backend=backend, rubric_names=["factual_correctness", "groundedness"],
        batch_size=8, n_bootstrap=50,
    )
    by_question = {rs["question"]: rs for rs in result["row_scores"]}

    assert by_question["Distractor?"]["abstention"] == pytest.approx(0.5)
    # The distractor row carries no answerable score at all.
    assert "groundedness" not in by_question["Distractor?"]
    assert "factual_correctness" not in by_question["Distractor?"]

    assert by_question["Answerable?"]["groundedness"] == pytest.approx(0.5)
    assert "abstention" not in by_question["Answerable?"]

    # abstention is summarised even though it was not requested.
    assert result["rubrics"]["abstention"]["n"] == 1
    assert result["rubrics"]["groundedness"]["n"] == 1


def test_batched_distractor_prompt_carries_no_ground_truth() -> None:
    rows = [_row(distractor=True, contexts=["unrelated prose"], question="Distractor?")]
    backend = _BatchBackend()
    score_dataset_batched(
        rows, backend=backend, rubric_names=["factual_correctness", "groundedness"],
        batch_size=8, n_bootstrap=50,
    )
    assert len(backend.prompts) == 1  # abstention only → one call
    system, user = backend.prompts[0]
    assert "Not answerable from the supplied corpus." not in user
    assert GROUND_TRUTH not in user


# ── answer_correctness → factual_correctness, without losing old runs ─────────


def test_legacy_rubric_name_resolves_to_factual_correctness() -> None:
    assert canonical_rubric_name("answer_correctness") == "factual_correctness"
    assert get_rubric("answer_correctness") is CANONICAL_RUBRICS["factual_correctness"]
    assert RUBRIC_ALIASES["answer_correctness"] == "factual_correctness"


def test_legacy_rubric_name_still_enumerable_for_reading_old_artifacts() -> None:
    """report/aggregate.py builds its metric list from RUBRICS keys."""
    assert "answer_correctness" in RUBRICS
    assert "factual_correctness" in RUBRICS
    assert RUBRICS["answer_correctness"] is RUBRICS["factual_correctness"]


def test_resolve_rubrics_deduplicates_a_name_and_its_alias() -> None:
    resolved = resolve_rubrics(["answer_correctness", "factual_correctness", "groundedness"])
    assert [r.name for r in resolved] == ["factual_correctness", "groundedness"]


def test_factual_correctness_no_longer_double_counts_completeness() -> None:
    description = get_rubric("factual_correctness").description.lower()
    assert "score only" in description or "only" in description
    assert "coverage is scored separately" in description
    # completeness remains its own dimension, as the gold's judge_dimensions ask.
    assert "completeness" in CANONICAL_RUBRICS


def test_unknown_rubric_still_raises() -> None:
    with pytest.raises(ValueError, match="Unknown rubric"):
        get_rubric("nonexistent_rubric")


def test_resume_re_judges_rows_scored_under_the_legacy_rubric_name(tmp_path: Path) -> None:
    """A checkpoint scored under the legacy name is not merged into a new run.

    answer_correctness scored accuracy *and* completeness; reading those numbers
    back as factual_correctness would mix two different measurements under one
    name. The row is re-judged instead.
    """
    row = _row(contexts=[FACT_AS_TEXT], question="Answerable?")
    checkpoint = tmp_path / "judge_rows.jsonl"
    checkpoint.write_text(
        json.dumps({
            "run_dir": row.run_dir, "model_id": row.model_id, "framework": row.framework,
            "strategy": row.strategy, "question": row.question,
            "question_type": row.question_type, "skip_reason": "",
            "_key": _row_key(row), "answer_correctness": 0.2, "rationale": "scored last month",
        }) + "\n",
        encoding="utf-8",
    )

    backend = _BatchBackend()
    result = score_dataset_batched(
        [row], backend=backend, rubric_names=["factual_correctness"],
        batch_size=8, out_dir=tmp_path, resume=True, n_bootstrap=50,
    )

    assert len(backend.prompts) == 1  # re-judged rather than reused
    assert result["rows_evaluated"] == 1  # and the stale entry did not survive alongside it
    assert result["rubrics"]["factual_correctness"]["mean"] == pytest.approx(0.5)
    assert 0.2 not in [rs.get("factual_correctness") for rs in result["row_scores"]]
