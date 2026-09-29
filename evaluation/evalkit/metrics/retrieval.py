"""Ranked retrieval metrics over triples, and entity coverage."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from evalkit.models import EvalRow


# ─── Normalisation helpers ───────────────────────────────────────────────────

def _normalize_text(value: str) -> str:
    """Lowercased, whitespace-collapsed text."""
    text = value.strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_entity(item: Any) -> str:
    """The comparison label of one retrieved entity, string or dict."""
    # Key order matters and must match `metrics/entities.py`. With "id" first,
    # every node entity emitted by the runner would normalise to its Neo4j
    # elementId ("4:4eddd2a9-...:12680"), which can never equal a gold label,
    # and `entity_coverage` would silently measure the triple channel alone.
    if isinstance(item, str):
        return _normalize_text(item)
    if isinstance(item, dict):
        for key in ("label", "name", "id", "entity"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_text(value)
    return ""


def _normalize_triple(item: Any) -> str:
    """A triple as ``subject|predicate|object``, from a dict, list or string."""
    if isinstance(item, dict):
        subject = str(item.get("subject") or item.get("s") or "").strip()
        predicate = str(item.get("predicate") or item.get("p") or "").strip()
        obj = str(item.get("object") or item.get("o") or "").strip()
        if subject or predicate or obj:
            return "|".join([
                _normalize_text(subject),
                _normalize_text(predicate),
                _normalize_text(obj),
            ])
    if isinstance(item, list) and len(item) >= 3:
        return "|".join(_normalize_text(str(part)) for part in item[:3])
    if isinstance(item, str):
        value = item.strip()
        if "|" in value:
            parts = [part.strip() for part in value.split("|", 2)]
            while len(parts) < 3:
                parts.append("")
            return "|".join(_normalize_text(p) for p in parts[:3])
        return _normalize_text(value)
    return ""


def _unique_non_empty(items: list[str]) -> list[str]:
    """The non-empty items, first occurrence only, in order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


# ─── Core metric functions ───────────────────────────────────────────────────

def precision_at_k(relevant: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Precision with optional top-k cutoff."""
    if k is not None and k <= 0:
        return 0.0
    ranked = retrieved if k is None else retrieved[:k]
    if not ranked:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in ranked if item in relevant_set) / len(ranked)


def recall_at_k(relevant: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Recall with optional top-k cutoff."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    if k is not None and k <= 0:
        return 0.0
    ranked = retrieved if k is None else retrieved[:k]
    hits = {item for item in ranked if item in relevant_set}
    return len(hits) / len(relevant_set)


def hit_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """1.0 if any relevant item in top-k, else 0.0."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in retrieved[:k]) else 0.0


def ndcg_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Binary nDCG@k."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = retrieved[:k]
    if not ranked:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(ranked)
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), len(ranked))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevant: list[str], retrieved: list[str]) -> float:
    """Mean reciprocal rank for a single query."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / i
    return 0.0


def mean_average_precision(relevant: list[str], retrieved: list[str]) -> float:
    """Average precision for a single query."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    seen_relevant: set[str] = set()
    for i, item in enumerate(retrieved, start=1):
        if item in relevant_set and item not in seen_relevant:
            seen_relevant.add(item)
            hit_count += 1
            precision_sum += hit_count / i
    return precision_sum / len(relevant_set)


def entity_coverage(expected: list[Any], retrieved: list[Any]) -> float | None:
    """Fraction of expected entities found in retrieved entities.

    Returns None if no expected entities are labelled.
    """
    expected_norm = {_normalize_entity(e) for e in expected}
    expected_norm.discard("")
    if not expected_norm:
        return None
    retrieved_norm = {_normalize_entity(e) for e in retrieved}
    retrieved_norm.discard("")
    return len(expected_norm & retrieved_norm) / len(expected_norm)


# ─── Row-level computation ───────────────────────────────────────────────────

def compute_retrieval_row(row: EvalRow, k: int | None) -> dict[str, Any]:
    """Compute all retrieval metrics for a single EvalRow.

    Returns a flat dict with normalised triple-based metrics + entity_coverage.
    """
    relevant_triples = _unique_non_empty(
        [_normalize_triple(t) for t in row.gold_triples]
    )
    retrieved_triples = _unique_non_empty(
        [_normalize_triple(t) for t in row.retrieved_triples]
    )

    skip = bool(row.skip_reason)
    p: float | None = None
    r: float | None = None
    h: float | None = None
    ndc: float | None = None
    rr: float | None = None
    ap: float | None = None

    if not skip and relevant_triples:
        cutoff = k if k is not None else len(retrieved_triples)
        effective_k = max(cutoff, 1)
        p = precision_at_k(relevant_triples, retrieved_triples, k=k)
        r = recall_at_k(relevant_triples, retrieved_triples, k=k)
        h = hit_at_k(relevant_triples, retrieved_triples, k=effective_k)
        ndc = ndcg_at_k(relevant_triples, retrieved_triples, k=effective_k)
        rr = mrr(relevant_triples, retrieved_triples)
        ap = mean_average_precision(relevant_triples, retrieved_triples)

    ec = entity_coverage(row.expected_entities, row.retrieved_entities)

    return {
        "run_dir": row.run_dir,
        "model_id": row.model_id,
        "framework": row.framework,
        "strategy": row.strategy,
        "question": row.question,
        "question_type": row.question_type,
        "latency_ms": row.latency_ms,
        "has_gold_match": row.has_gold,
        "skip_reason": row.skip_reason,
        "entity_coverage": ec,
        "precision_at_k": p,
        "recall_at_k": r,
        "hit_at_k": h,
        "ndcg_at_k": ndc,
        "mrr": rr,
        "map": ap,
    }


RETRIEVAL_METRICS = [
    "entity_coverage",
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "ndcg_at_k",
    "mrr",
    "map",
    "latency_ms",
]
