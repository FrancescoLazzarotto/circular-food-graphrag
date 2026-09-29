"""Load the gold: the JSON gold keyed by query_id, and the legacy CSV gold."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from evalkit.models import MAPPING_EXACT, MAPPING_LOCAL, GoldEntity, GoldQuery
from evalkit.normalisation import normalise

logger = logging.getLogger("graphrag")

# The mapping statuses a gold entity may carry. Anything else is a gold typo
# and must not silently fall out of grounding-level scope.
KNOWN_MAPPING_STATUS = frozenset({MAPPING_EXACT, MAPPING_LOCAL})


def _remove_accents(text: str) -> str:
    """`text` with combining accents removed."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_question(text: str) -> str:
    """Normalize question text for join-key matching.

    Lowercases, collapses whitespace, removes trailing punctuation.
    Accent removal is on by default; disable with EVAL_NORMALIZE_REMOVE_ACCENTS=0.
    """
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]+$", "", normalized)
    remove_accents = (
        os.getenv("EVAL_NORMALIZE_REMOVE_ACCENTS", "1").strip().lower()
        not in {"0", "false", "no"}
    )
    if remove_accents:
        normalized = _remove_accents(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _parse_gold_list(raw: str) -> tuple[list[Any], bool]:
    """Parse a gold annotation field that may be JSON or pipe-separated."""
    if not raw or not raw.strip():
        return ([], False)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return (parsed, False)
    except json.JSONDecodeError:
        pass

    # Pipe-separated fallback (no valid JSON)
    fallback = [item.strip() for item in raw.split("|") if item.strip()]
    if fallback:
        return (fallback, False)

    return ([], True)


def load_gold(gold_path: Path) -> dict[str, dict[str, str]]:
    """Load a gold CSV and return a dict keyed by normalised question text.

    Args:
        gold_path: Path to gold CSV with at least ``question`` and one of
            ``ground_truth``, ``canonical_answer``, or ``gold_answer`` columns.

    Returns:
        Mapping of normalised question → raw row dict.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not gold_path.exists() or not gold_path.is_file():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    gold_by_question: dict[str, dict[str, str]] = {}
    duplicates = 0

    with gold_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            question = (row.get("question", "") or "").strip()
            if not question:
                continue
            key = normalize_question(question)
            if key in gold_by_question:
                duplicates += 1
                continue
            gold_by_question[key] = dict(row)

    if duplicates:
        logger.warning("Ignored %d duplicate questions in gold file: %s", duplicates, gold_path)

    return gold_by_question


# ─── JSON gold ───────────────────────────────────────────────────────────────
#
# The definitive gold is JSON with entities as OBJECTS (label / normalised_label /
# alt_labels / uri / mapping_status), keyed by query_id. The CSV loader above is
# kept for the silver set (evaluation/gold/gold_circular_v1_silver.csv), which is
# a dev set only and never produces reported numbers.


def is_json_gold(gold_path: Path) -> bool:
    """True when the gold file should be read with load_gold_json (not the CSV loader)."""
    return gold_path.suffix.lower() == ".json"


def _parse_gold_entity(raw: dict[str, Any], query_id: str) -> GoldEntity | None:
    """Parse one expected_entities[] object into a GoldEntity.

    Args:
        raw: Entity object from the gold JSON.
        query_id: Owning query id, for error messages.

    Returns:
        The parsed GoldEntity, or None when it carries no usable surface form.
    """
    label = str(raw.get("label", "") or "").strip()
    normalised_label = str(raw.get("normalised_label", "") or "").strip()

    if not normalised_label:
        if not label:
            logger.warning(
                "%s: expected entity with neither 'label' nor 'normalised_label'; "
                "dropped. Raw entry: %r",
                query_id,
                raw,
            )
            return None
        normalised_label = normalise(label)
        logger.warning(
            "%s: entity %r has no 'normalised_label'; derived %r from the label. "
            "The protocol (§4) requires the field to be explicit in the gold.",
            query_id,
            label,
            normalised_label,
        )

    raw_alts = raw.get("alt_labels") or []
    if not isinstance(raw_alts, list):
        logger.warning(
            "%s: entity %r has non-list 'alt_labels' (%s); treated as empty.",
            query_id,
            label or normalised_label,
            type(raw_alts).__name__,
        )
        raw_alts = []
    alt_labels = tuple(str(a).strip() for a in raw_alts if str(a).strip())

    raw_uri = raw.get("uri")
    uri = str(raw_uri).strip() if raw_uri else None

    mapping_status = str(raw.get("mapping_status", "") or "").strip()
    if mapping_status not in KNOWN_MAPPING_STATUS:
        logger.warning(
            "%s: entity %r has unknown mapping_status %r (expected one of %s). "
            "It will NOT count at grounding-level (§2b).",
            query_id,
            label or normalised_label,
            mapping_status,
            sorted(KNOWN_MAPPING_STATUS),
        )
    elif mapping_status == MAPPING_EXACT and not uri:
        logger.warning(
            "%s: entity %r is mapping_status=exact but has no URI — grounding-level "
            "scoring (§2b) has nothing to anchor it to.",
            query_id,
            label or normalised_label,
        )

    raw_aligned = raw.get("aligned_to")
    aligned_to = str(raw_aligned).strip() if raw_aligned else None

    # 'note' is deliberately dropped: informational prose, never scored.
    return GoldEntity(
        label=label or normalised_label,
        normalised_label=normalised_label,
        alt_labels=alt_labels,
        uri=uri,
        mapping_status=mapping_status,
        vocabulary=str(raw.get("vocabulary", "") or ""),
        aligned_to=aligned_to,
    )


def parse_gold_query(raw: dict[str, Any]) -> GoldQuery:
    """Parse one gold JSON query object into a GoldQuery.

    Args:
        raw: A single entry of the gold's ``queries`` array.

    Returns:
        The parsed GoldQuery.

    Raises:
        ValueError: If ``query_id`` is missing — it is the join key, and a query
            that cannot be keyed cannot be scored.
    """
    query_id = str(raw.get("query_id", "") or "").strip()
    if not query_id:
        raise ValueError(
            f"Gold query without 'query_id' (join key): {str(raw.get('query', ''))[:80]!r}"
        )

    raw_entities = raw.get("expected_entities") or []
    if not isinstance(raw_entities, list):
        raise ValueError(f"{query_id}: 'expected_entities' must be a list")
    entities = tuple(
        entity
        for entity in (
            _parse_gold_entity(e, query_id) for e in raw_entities if isinstance(e, dict)
        )
        if entity is not None
    )

    raw_relations = raw.get("expected_relations") or []
    relations = tuple(
        str(r).strip() for r in raw_relations if str(r).strip()
    ) if isinstance(raw_relations, list) else ()

    raw_sources = raw.get("source_verified") or []
    sources = tuple(s for s in raw_sources if isinstance(s, dict)) if isinstance(
        raw_sources, list
    ) else ()

    scoring = raw.get("scoring") or {}
    if not isinstance(scoring, dict):
        logger.warning("%s: 'scoring' is not an object; distractor_expected=False", query_id)
        scoring = {}

    return GoldQuery(
        query_id=query_id,
        query_type=str(raw.get("query_type", "") or ""),
        query=str(raw.get("query", "") or ""),
        expected_answer=str(raw.get("expected_answer", "") or ""),
        expected_entities=entities,
        expected_relations=relations,
        distractor_expected=bool(scoring.get("distractor_expected", False)),
        source_verified=sources,
    )


def load_gold_json(gold_path: Path) -> dict[str, GoldQuery]:
    """Load the JSON gold and return its queries keyed by query_id.

    Args:
        gold_path: Path to a gold JSON — either ``{"_meta": ..., "queries": [...]}``
            or a bare list of query objects.

    Returns:
        Mapping of query_id (``Q01``…``Q30``) → GoldQuery.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the payload is malformed, has no queries, or contains a
            duplicate query_id (a duplicate would silently drop a gold query).
    """
    if not gold_path.exists() or not gold_path.is_file():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    try:
        payload = json.loads(gold_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gold file is not valid JSON: {gold_path}: {exc}") from exc

    meta: dict[str, Any] = {}
    if isinstance(payload, dict):
        meta = payload.get("_meta") or {}
        raw_queries = payload.get("queries")
    elif isinstance(payload, list):
        raw_queries = payload
    else:
        raise ValueError(
            f"Gold JSON must be an object with 'queries' or a list: {gold_path}"
        )

    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError(f"Gold JSON has no 'queries' array: {gold_path}")

    gold_by_id: dict[str, GoldQuery] = {}
    for raw in raw_queries:
        if not isinstance(raw, dict):
            logger.warning("Skipping non-object entry in gold 'queries': %r", raw)
            continue
        query = parse_gold_query(raw)
        if query.query_id in gold_by_id:
            # Not a warning: keying by a duplicated id drops a query from the
            # benchmark without any downstream signal.
            raise ValueError(
                f"Duplicate query_id {query.query_id!r} in gold: {gold_path}"
            )
        gold_by_id[query.query_id] = query

    declared = meta.get("n_queries")
    if isinstance(declared, int) and declared != len(gold_by_id):
        logger.warning(
            "Gold _meta.n_queries=%d but %d queries parsed from %s — the gold file "
            "and its own metadata disagree.",
            declared,
            len(gold_by_id),
            gold_path,
        )

    n_distractors = sum(1 for q in gold_by_id.values() if q.distractor_expected)
    n_entities = sum(len(q.expected_entities) for q in gold_by_id.values())
    n_grounding = sum(len(q.grounding_entities) for q in gold_by_id.values())
    logger.info(
        "Loaded %d gold queries from %s (%d distractors, %d entities, "
        "%d grounding-level)",
        len(gold_by_id),
        gold_path,
        n_distractors,
        n_entities,
        n_grounding,
    )
    return gold_by_id


def gold_entity_to_dict(entity: GoldEntity) -> dict[str, Any]:
    """Serialise a GoldEntity to a JSON-safe dict (inverse of _parse_gold_entity)."""
    return {
        "label": entity.label,
        "normalised_label": entity.normalised_label,
        "alt_labels": list(entity.alt_labels),
        "uri": entity.uri,
        "vocabulary": entity.vocabulary,
        "mapping_status": entity.mapping_status,
        "aligned_to": entity.aligned_to,
    }


def gold_query_to_dict(query: GoldQuery) -> dict[str, Any]:
    """Serialise a GoldQuery to a JSON-safe dict, round-trippable by gold_query_from_dict."""
    return {
        "query_id": query.query_id,
        "query_type": query.query_type,
        "query": query.query,
        "expected_answer": query.expected_answer,
        "expected_entities": [gold_entity_to_dict(e) for e in query.expected_entities],
        "expected_relations": list(query.expected_relations),
        "source_verified": [dict(s) for s in query.source_verified],
        "scoring": {"distractor_expected": query.distractor_expected},
    }


def gold_query_from_dict(raw: dict[str, Any]) -> GoldQuery:
    """Rebuild a GoldQuery from gold_query_to_dict output (CSV round-trip)."""
    return parse_gold_query(raw)


def gold_entity_from_dict(raw: dict[str, Any], query_id: str = "<csv>") -> GoldEntity | None:
    """Rebuild a GoldEntity from gold_entity_to_dict output (CSV round-trip)."""
    return _parse_gold_entity(raw, query_id)


def looks_like_gold_entity(raw: Any) -> bool:
    """True when a raw dict has the gold entity shape (normalised label, mapping)."""
    return isinstance(raw, dict) and "normalised_label" in raw and "mapping_status" in raw


def pick_ground_truth(gold_row: dict[str, str]) -> str:
    """Return the canonical answer from a gold row, trying multiple field names."""
    for key in ("ground_truth", "canonical_answer", "gold_answer", "answer"):
        value = (gold_row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def extract_gold_annotations(
    gold_row: dict[str, str],
) -> tuple[list[Any], list[Any], list[Any]]:
    """Parse expected_entities, gold_triples, and answer_variants from a gold row.

    Returns:
        (expected_entities, gold_triples, answer_variants)
    """
    expected_raw = str(
        gold_row.get("expected_entities_json") or gold_row.get("expected_entities") or ""
    )
    triples_raw = str(
        gold_row.get("gold_triples_json") or gold_row.get("gold_triples") or ""
    )
    variants_raw = str(
        gold_row.get("answer_variants_json") or gold_row.get("answer_variants") or ""
    )

    expected_entities, _ = _parse_gold_list(expected_raw)
    gold_triples, _ = _parse_gold_list(triples_raw)
    answer_variants, _ = _parse_gold_list(variants_raw)
    return expected_entities, gold_triples, answer_variants
