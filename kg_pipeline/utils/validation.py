"""Parsing and validation of the JSON triples returned by the extraction LLM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kg_pipeline.models.types import KGTriple

_LOGGER = logging.getLogger("kg_pipeline")


def normalize_json_text(raw_text: str) -> str:
    """Strip whitespace and a surrounding Markdown code fence from LLM output.

    Args:
        raw_text: Raw model output; ``None`` is treated as empty.

    Returns:
        The text between the fences, or the stripped input when it is not
        fenced.
    """
    cleaned = (raw_text or "").strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_array(raw_text: str) -> list[dict[str, Any]]:
    """Parse LLM output that must be a JSON array.

    Args:
        raw_text: Raw model output, optionally wrapped in a code fence.

    Returns:
        The parsed array.

    Raises:
        json.JSONDecodeError: If the text is not valid JSON.
        ValueError: If the JSON value is not an array.
    """
    cleaned = normalize_json_text(raw_text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("LLM output is not a JSON array")
    return parsed


def validate_triples(
    raw_items: list[dict[str, Any]],
    allowed_predicates: list[str] | None = None,
) -> list[KGTriple]:
    """Validate raw items as ``KGTriple`` objects and enforce the vocabulary.

    A predicate outside ``allowed_predicates`` is not discarded: the triple is
    kept with relationship type ``RELATED_TO`` and the original predicate is
    stored in ``relationship_properties["predicate"]``, where the retriever
    reads it back. The type is remapped because indexes, type filters and the
    repair passes key off the relationship type, which must stay within the
    vocabulary.

    Args:
        raw_items: Items parsed from the LLM output.
        allowed_predicates: Relationship vocabulary, compared case-insensitively.
            ``None`` or empty disables the check.

    Returns:
        The validated triples, in input order.

    Raises:
        pydantic.ValidationError: If an item does not match the ``KGTriple``
            schema.
    """
    triples: list[KGTriple] = []
    allowed_set = None
    if allowed_predicates:
        allowed_set = {str(item).strip().upper() for item in allowed_predicates}
    for item in raw_items:
        triple = KGTriple.model_validate(item)
        if allowed_set is not None and triple.predicate not in allowed_set:
            _LOGGER.warning(
                "Off-vocab predicate '%s' remapped to RELATED_TO "
                "(subject=%r, object=%r)",
                triple.predicate,
                triple.subject,
                triple.object,
            )
            rel_props = dict(triple.relationship_properties)
            rel_props["predicate"] = triple.predicate
            triple.relationship_properties = rel_props
            triple.predicate = "RELATED_TO"
        triples.append(triple)
    return triples


def write_failed_chunk(
    failed_path: Path,
    chunk_metadata: dict[str, Any],
    attempt: int,
    error: str,
    raw_response: str,
) -> None:
    """Append one extraction failure to a JSON Lines file.

    Args:
        failed_path: JSONL file to append to; parent directories are created.
        chunk_metadata: The failing chunk, as a dict.
        attempt: 1-based attempt number (0 for per-item validation failures).
        error: Error message.
        raw_response: Model output that caused the failure.
    """
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "chunk_metadata": chunk_metadata,
        "attempt": attempt,
        "error": error,
        "raw_response": raw_response,
    }
    with failed_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
