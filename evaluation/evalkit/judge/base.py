"""Judge backend protocol, judge result, and parsing of the judge's JSON output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class JudgeResult:
    """Output of LLM-as-a-Judge for a single row + rubric.

    Attributes:
        scores: Score per field, normalised to [0, 1].
        rationale: The judge's explanation.
        raw: The raw completion.
        ok: False when the completion could not be parsed.
    """

    scores: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    raw: str = ""
    ok: bool = True


@runtime_checkable
class JudgeBackend(Protocol):
    """Minimal protocol for any LLM backend used as a judge."""

    def complete(self, system: str, user: str) -> str:
        """Return the raw completion string."""
        ...


def parse_judge_output(raw: str) -> tuple[dict[str, Any], bool]:
    """Extract the first valid JSON object from a judge completion.

    Returns:
        (parsed_dict, success_flag)
    """
    if not raw:
        return ({}, False)

    # Direct JSON parse
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict):
            return (parsed, True)
    except json.JSONDecodeError:
        pass

    # Extract first JSON block from prose output
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return (parsed, True)
        except json.JSONDecodeError:
            pass

    return ({}, False)


def extract_score(parsed: dict[str, Any], field_name: str) -> float | None:
    """Extract a numeric score from a parsed dict, normalising to [0, 1]."""
    value = parsed.get(field_name)
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    # If score looks like a 1-5 scale, normalise to [0, 1]
    if score > 1.0:
        score = (score - 1.0) / 4.0

    return max(0.0, min(1.0, score))
