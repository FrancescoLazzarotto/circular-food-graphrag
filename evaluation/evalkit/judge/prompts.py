"""Prompt construction for the LLM judge.

Single source of truth for every string the judge model sees: backends are handed
a (system, user) pair and never assemble prompt text themselves.

Two fairness properties are enforced here:

* **The prompt does not reveal which pipeline produced the answer.** All evidence
  goes through :func:`render_evidence`: one header, one line per element, triples
  serialised as statements, and one character budget for everybody. A header
  that depends on the kind of evidence a row carries would tell the judge which
  system it is scoring, and capping by element count (5 chunks vs 20 triples)
  would hand the text pipelines an order of magnitude more evidence.
* **The ground truth reaches only the rubrics that ask for it.** A rubric that
  declares ``uses_ground_truth = False`` never has the gold answer in its prompt.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from evalkit.judge.rubrics import Rubric
from evalkit.models import EvalRow

logger = logging.getLogger("graphrag")

EVIDENCE_HEADER = "## Retrieved Evidence"
NO_EVIDENCE_TEXT = "_No evidence was retrieved for this question._"

# Same budget for every pipeline, in characters rather than elements: five prose
# chunks and twenty triples are not comparable amounts of evidence.
EVIDENCE_CHAR_BUDGET = 4000

_BULLET = "- "
_TRIPLE_SEPARATOR = " — "
_TRUNCATION_SUFFIX = " …"
# Below this, the leftover budget cannot hold a readable fragment of an element,
# so the element is dropped rather than shown as a stub.
_MIN_FRAGMENT_CHARS = 80


SYSTEM_TEMPLATE = """\
You are an expert evaluator for question-answering systems.
You will be given a question, a generated answer, the evidence that was retrieved \
to answer it, possibly a reference answer, and a scoring rubric.
Apply the rubric exactly as written and return a JSON object with the score and a \
brief rationale. Judge every answer by the same standard, whatever form its \
evidence takes.

Respond ONLY with a JSON object. Do not add any text before or after the JSON.
Example format:
{{
  "{score_field}": <float between 0 and 1>,
  "rationale": "<one or two sentences explaining the score>"
}}
"""

USER_TEMPLATE = """\
{sections}

## Rubric
{rubric_description}

Return only the JSON object with keys "{score_field}" and "rationale".
"""


def _flatten(text: Any) -> str:
    """Collapse an evidence element onto a single line of normalised whitespace."""
    return " ".join(str(text).split())


def _render_triple(triple: Mapping[str, Any]) -> str:
    """Serialise a KG triple as one statement line.

    The predicate is de-cased (``HAS_BYPRODUCT`` → ``has byproduct``) so the line
    reads as a statement: a SCREAMING_SNAKE_CASE predicate would tell the judge
    the evidence came out of the graph.

    Args:
        triple: Mapping with subject/predicate/object (or s/p/o) keys.

    Returns:
        ``subject — predicate — object``, omitting empty parts.
    """
    subject = _flatten(triple.get("subject") or triple.get("s") or "")
    predicate = _flatten(triple.get("predicate") or triple.get("p") or "")
    obj = _flatten(triple.get("object") or triple.get("o") or "")
    predicate = predicate.replace("_", " ").lower()
    return _TRIPLE_SEPARATOR.join(part for part in (subject, predicate, obj) if part)


def evidence_items(row: EvalRow) -> list[str]:
    """Every retrieved evidence element of a row, as uniform one-line strings.

    Args:
        row: The row being judged.

    Returns:
        Retrieved text contexts followed by retrieved triples, each rendered to a
        single line and stripped of empties. Nothing distinguishes the two once
        rendered: that is the point.
    """
    items = [_flatten(context) for context in row.contexts]
    for triple in row.retrieved_triples:
        items.append(
            _render_triple(triple) if isinstance(triple, Mapping) else _flatten(triple)
        )
    return [item for item in items if item]


def render_evidence(row: EvalRow, char_budget: int = EVIDENCE_CHAR_BUDGET) -> str:
    """Render a row's retrieved evidence identically for every pipeline.

    Elements are emitted in order until the character budget is spent; an element
    that does not fit is truncated to what is left (or dropped, if that is less
    than a readable fragment) and the rest are cut. The rule is the same for a
    prose chunk and for a triple, so no pipeline is capped harder than another.

    Args:
        row: The row being judged.
        char_budget: Maximum characters of evidence, excluding the header.

    Returns:
        The evidence section, header included; a placeholder when the row
        retrieved nothing — the section is always present, since its absence
        would itself say something about the pipeline.
    """
    items = evidence_items(row)
    lines: list[str] = []
    used = 0
    for item in items:
        line = _BULLET + item
        remaining = char_budget - used
        if len(line) <= remaining:
            lines.append(line)
            used += len(line) + 1  # + the newline joining it to the next line
            continue
        if remaining >= _MIN_FRAGMENT_CHARS:
            keep = remaining - len(_TRUNCATION_SUFFIX)
            lines.append(line[:keep].rstrip() + _TRUNCATION_SUFFIX)
        logger.debug(
            "evidence truncated at %d chars (%d of %d elements shown)",
            char_budget, len(lines), len(items),
        )
        break

    body = "\n".join(lines) if lines else NO_EVIDENCE_TEXT
    return f"{EVIDENCE_HEADER}\n{body}"


def build_row_content(
    row: EvalRow,
    *,
    include_ground_truth: bool = True,
    char_budget: int = EVIDENCE_CHAR_BUDGET,
) -> str:
    """Assemble the per-row evaluation block (question, answer, references).

    Shared by the single-rubric prompt and the batched multi-row prompt so both
    feed the judge identical evidence.

    Args:
        row: The row being judged.
        include_ground_truth: Whether the gold answer may be shown. False for
            reference-free rubrics.
        char_budget: Maximum characters of evidence.

    Returns:
        A markdown string with the row's question, generated answer, the ground
        truth if allowed, and the retrieved evidence.
    """
    parts = [
        f"## Question\n{row.question}",
        f"## Generated Answer\n{row.answer}",
    ]
    if include_ground_truth and row.ground_truth.strip():
        parts.append(f"## Ground Truth Answer\n{row.ground_truth}")
    parts.append(render_evidence(row, char_budget=char_budget))
    return "\n\n".join(parts)


def build_prompt(
    row: EvalRow,
    rubric: Rubric,
    char_budget: int = EVIDENCE_CHAR_BUDGET,
) -> tuple[str, str]:
    """Build (system, user) prompt strings for a given row and rubric.

    Args:
        row: The row being judged.
        rubric: Rubric defining what to evaluate; its ``uses_ground_truth`` flag
            decides whether the gold answer appears in the prompt at all.
        char_budget: Maximum characters of evidence.

    Returns:
        (system_prompt, user_prompt)
    """
    system = SYSTEM_TEMPLATE.format(score_field=rubric.score_field)
    sections = build_row_content(
        row,
        include_ground_truth=rubric.uses_ground_truth,
        char_budget=char_budget,
    )
    user = USER_TEMPLATE.format(
        sections=sections,
        rubric_description=rubric.description,
        score_field=rubric.score_field,
    )
    return system, user
