"""Judging rubrics and the per-row rubric selection.

Each rubric declares two things the prompt builder needs in order to keep the
comparison fair across pipelines:

* ``uses_ground_truth`` — whether the gold answer may appear in the prompt at
  all. Reference-free rubrics never see it: instructing a model to ignore what
  is put in front of it is not a control.
* ``applies_to`` — the kind of row the rubric is meaningful on. On a distractor
  the only correct answer is an abstention, which by definition does not address
  the question, so the answerable rubrics would punish the right behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evalkit.models import EvalRow

logger = logging.getLogger("graphrag")

# Row kinds a rubric can apply to.
ROW_KIND_ANSWERABLE = "answerable"
ROW_KIND_DISTRACTOR = "distractor"

ABSTENTION = "abstention"


@dataclass(frozen=True)
class Rubric:
    """One judging dimension.

    Attributes:
        name: Canonical name; also the key the rubric's score is reported under.
        description: The rubric text shown to the judge.
        score_field: JSON key the judge is asked to emit in single-rubric mode.
        rationale_field: JSON key holding the judge's rationale.
        uses_ground_truth: Whether the prompt may show the gold answer. Defaults
            to False: a rubric has to ask for the reference to be given it.
        applies_to: Row kinds the rubric is scored on.
    """

    name: str
    description: str
    score_field: str
    rationale_field: str = "rationale"
    uses_ground_truth: bool = False
    applies_to: tuple[str, ...] = (ROW_KIND_ANSWERABLE,)


CANONICAL_RUBRICS: dict[str, Rubric] = {
    "factual_correctness": Rubric(
        name="factual_correctness",
        score_field="correctness_score",
        uses_ground_truth=True,
        description=(
            "Compare the generated answer against the ground truth answer and score ONLY "
            "factual accuracy: is what the answer asserts true? "
            "Score just the claims the answer actually makes. Do NOT lower the score because "
            "the answer omits something — coverage is scored separately by another rubric. "
            "An answer that makes three claims, all of them accurate, scores 1 even if the "
            "ground truth mentions five things. "
            "Score 1 = every claim is factually accurate, 0 = the claims contradict the "
            "ground truth."
        ),
    ),
    "completeness": Rubric(
        name="completeness",
        score_field="completeness_score",
        uses_ground_truth=True,
        description=(
            "Does the generated answer cover all relevant aspects of the ground truth answer? "
            "Score coverage ONLY: do not re-score factual accuracy, which another rubric "
            "handles. Penalise partial answers that omit important information present in "
            "the ground truth. "
            "Score 1 = every important element of the ground truth is covered, 0 = minimal "
            "or no useful content."
        ),
    ),
    "groundedness": Rubric(
        name="groundedness",
        score_field="groundedness_score",
        uses_ground_truth=False,
        description=(
            "Is every factual claim in the generated answer supported by the retrieved "
            "evidence? Work claim by claim. A claim is grounded when the evidence states it "
            "or directly entails it: verbatim overlap is NOT required, and evidence supplied "
            "as short factual statements legitimately has to be joined up into prose. "
            "IGNORE connective and framing material — introductions, transitions, "
            "restatements of the question, summarising phrases. It carries no factual load, "
            "so it can be neither grounded nor ungrounded, and an answer must not be "
            "penalised for reading fluently. "
            "Judge the answer against the evidence shown and nothing else: do not use your "
            "own knowledge of the subject. "
            "Score 1 = every factual claim is supported by the evidence, 0 = the answer's "
            "central claims are unsupported."
        ),
    ),
    "relevance": Rubric(
        name="relevance",
        score_field="relevance_score",
        uses_ground_truth=False,
        description=(
            "Does the generated answer directly address the question asked? "
            "Judge topical fit only — whether the answer is accurate or complete is scored "
            "by other rubrics. Penalise answers that are off-topic or that answer a "
            "different question. "
            "Score 1 = perfectly on-topic, 0 = completely irrelevant."
        ),
    ),
    ABSTENTION: Rubric(
        name=ABSTENTION,
        score_field="abstention_score",
        uses_ground_truth=False,
        applies_to=(ROW_KIND_DISTRACTOR,),
        description=(
            "The question may or may not be answerable from the retrieved evidence. Score "
            "whether the generated answer deals with that evidence honestly. "
            "Score 1 = the answer states that the supplied material does not contain the "
            "information (or is insufficient to answer) and asserts no specific facts beyond "
            "what the evidence supports. "
            "Score 0 = the answer presents specific facts — figures, names, dates, causal "
            "claims — that the evidence does not support, whatever confidence it expresses. "
            "Award partial credit to answers that hedge but still assert unsupported "
            "specifics. "
            "Do not reward fluency, length or politeness: score only whether unsupported "
            "specifics are asserted."
        ),
    ),
}

# Legacy rubric names → current names.
#
# ``answer_correctness`` maps to ``factual_correctness``, which scores accuracy
# only: the gold asks for accuracy and completeness apart
# (`scoring.judge_dimensions`), and ``completeness`` is its own rubric.
RUBRIC_ALIASES: dict[str, str] = {"answer_correctness": "factual_correctness"}

# Registry keyed by rubric name AND by every legacy name, so that callers which
# enumerate rubric names to read scores off an artifact — report/aggregate.py
# builds its metric list from these keys — keep resolving the names older
# judge artifacts were written under. Iterate CANONICAL_RUBRICS to visit each rubric
# exactly once.
RUBRICS: dict[str, Rubric] = {
    **CANONICAL_RUBRICS,
    **{legacy: CANONICAL_RUBRICS[current] for legacy, current in RUBRIC_ALIASES.items()},
}


def canonical_rubric_name(name: str) -> str:
    """Map a possibly legacy rubric name onto the name in use today.

    Args:
        name: Rubric name, current or legacy.

    Returns:
        The current name; unknown names are returned unchanged.
    """
    return RUBRIC_ALIASES.get(name, name)


def get_rubric(name: str) -> Rubric:
    """Return a Rubric by name, accepting legacy names.

    Args:
        name: Rubric name, current or legacy (see RUBRIC_ALIASES).

    Returns:
        The Rubric registered under the current name.

    Raises:
        ValueError: If the rubric name is unknown.
    """
    current = canonical_rubric_name(name)
    if current != name:
        logger.warning(
            "Rubric %r was renamed to %r and now scores accuracy only; scores are reported "
            "under the new name and are not comparable with %r scores from earlier runs.",
            name, current, name,
        )
    if current not in CANONICAL_RUBRICS:
        available = ", ".join(sorted(CANONICAL_RUBRICS))
        raise ValueError(f"Unknown rubric '{name}'. Available: {available}")
    return CANONICAL_RUBRICS[current]


def resolve_rubrics(names: Iterable[str]) -> list[Rubric]:
    """Resolve rubric names to Rubrics, de-duplicated, in the order given.

    Args:
        names: Rubric names, current or legacy. A name and its legacy alias
            resolve to the same rubric and are kept once.

    Returns:
        The resolved rubrics.

    Raises:
        ValueError: If any name is unknown.
    """
    resolved: list[Rubric] = []
    seen: set[str] = set()
    for name in names:
        rubric = get_rubric(name)
        if rubric.name not in seen:
            seen.add(rubric.name)
            resolved.append(rubric)
    return resolved


def row_kind(row: EvalRow) -> str:
    """Return the rubric-selection kind of a row.

    Args:
        row: The row about to be judged.

    Returns:
        ``ROW_KIND_DISTRACTOR`` when the gold expects an abstention, else
        ``ROW_KIND_ANSWERABLE``.
    """
    return ROW_KIND_DISTRACTOR if row.is_distractor else ROW_KIND_ANSWERABLE


def rubrics_for_kind(kind: str, rubrics: Sequence[Rubric]) -> list[Rubric]:
    """Select the rubrics that are meaningful for one kind of row.

    On distractor rows every answerable rubric is dropped — the expected answer
    ("not answerable from the supplied corpus") is off-topic by construction and
    has no facts to check against a gold answer that asserts none — and
    ``abstention`` is scored instead, whether or not the caller asked for it.

    Args:
        kind: ``ROW_KIND_ANSWERABLE`` or ``ROW_KIND_DISTRACTOR``.
        rubrics: The rubrics the caller requested.

    Returns:
        The subset to score, in the requested order.
    """
    selected = [r for r in rubrics if kind in r.applies_to]
    if kind == ROW_KIND_DISTRACTOR and not any(r.name == ABSTENTION for r in selected):
        selected.insert(0, CANONICAL_RUBRICS[ABSTENTION])
    return selected


def rubrics_for_row(row: EvalRow, rubrics: Sequence[Rubric]) -> list[Rubric]:
    """Select the rubrics that are meaningful for one row.

    Args:
        row: The row about to be judged.
        rubrics: The rubrics the caller requested.

    Returns:
        The subset to score on this row.
    """
    return rubrics_for_kind(row_kind(row), rubrics)
