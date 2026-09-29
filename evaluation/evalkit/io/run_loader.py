"""Read the raw result rows of an experiment run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("graphrag")

# Single source of truth lives in graphrag.llm.refusal.is_insufficient. Import it
# when graphrag is available; fall back to a self-contained copy so evalkit keeps
# working when used standalone (graphrag is an optional dependency here — see the
# lazy import in evalkit.judge.backends). Keep the fallback list in sync with
# graphrag.llm.refusal._INSUFFICIENT_MARKERS.
try:
    from graphrag.llm.refusal import is_insufficient as _is_insufficient
except ImportError:
    _INSUFFICIENT_MARKERS: tuple[str, ...] = (
        "the provided context does not contain",
        "the context does not contain",
        "does not contain enough information",
        "does not contain information",
        "i don't have enough information",
        "i cannot find",
        "cannot answer",
        "unable to answer",
        "not enough information",
        "no information available",
        "no relevant information",
        "non ho informazioni",
        "non posso rispondere",
        "il contesto fornito non contiene",
        "il contesto non contiene",
        "context is insufficient",
        "too sparse to build a reliable answer",
        "troppo scarno per costruire una risposta",
    )

    def _is_insufficient(answer: str) -> bool:
        """True for an empty answer or one that declares the context insufficient."""
        if not answer or not str(answer).strip():
            return True
        lower = str(answer).lower()
        return any(marker in lower for marker in _INSUFFICIENT_MARKERS)


def _safe_int(value: Any, default: int = 0) -> int:
    """`value` as an int, or `default` when it does not parse."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """`value` as a float, or `default` when it does not parse."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_list(raw: Any) -> list[Any]:
    """A list, or the JSON list in a string; empty otherwise."""
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _parse_json_dict(raw: Any) -> dict[str, Any]:
    """A dict, or the JSON object in a string; empty otherwise."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _raw_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """The rows of a JSONL file, skipping malformed lines with a warning."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
    return rows


def _raw_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    """The rows of a CSV file as dicts."""
    import csv

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_raw_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Load raw result rows from a run directory.

    Prefers ``results.jsonl`` (richer data) over ``results.csv``.

    Args:
        run_dir: Path to an experiment run directory (``artifacts/experiments/<tag>``).

    Returns:
        List of raw dicts, one per (question, strategy) row.

    Raises:
        FileNotFoundError: If neither results.jsonl nor results.csv exists.
    """
    jsonl = run_dir / "results.jsonl"
    csv_path = run_dir / "results.csv"

    if jsonl.exists():
        return _raw_rows_from_jsonl(jsonl)
    if csv_path.exists():
        return _raw_rows_from_csv(csv_path)

    raise FileNotFoundError(
        f"No results.jsonl or results.csv found in: {run_dir}"
    )


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    """Load summary.json from a run directory; returns {} if absent."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_resource_summary(run_dir: Path) -> dict[str, Any]:
    """Load resource_summary.json from a run directory; returns {} if absent."""
    path = run_dir / "resource_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def raw_row_to_partial(raw: dict[str, Any], run_dir_name: str) -> dict[str, Any]:
    """Normalise a raw result row into a consistent intermediate dict.

    Does NOT fill gold-side fields (ground_truth, expected_entities, etc.).
    Those are added by io.dataset.build_dataset.

    Args:
        raw: The row as the runner wrote it, JSONL or CSV.
        run_dir_name: Name of the run directory.

    Returns:
        A flat dict with typed fields ready for EvalRow construction.
    """
    metadata = _parse_json_dict(raw.get("metadata", raw.get("metadata_json", "")))

    strategy = str(raw.get("strategy", "") or "")
    question = str(raw.get("question", "") or "")
    answer = str(raw.get("answer", "") or "")

    contexts_raw = raw.get("contexts", raw.get("contexts_json", ""))
    contexts = _parse_json_list(contexts_raw)

    triples_raw = raw.get("retrieved_triples", raw.get("retrieved_triples_json", ""))
    retrieved_triples = _parse_json_list(triples_raw)

    entities_raw = raw.get("retrieved_entities", raw.get("retrieved_entities_json", ""))
    retrieved_entities = _parse_json_list(entities_raw)

    return {
        "run_dir": run_dir_name,
        "strategy": strategy,
        "framework": str(metadata.get("framework") or raw.get("framework") or "unknown"),
        "model_id": str(metadata.get("model_id") or raw.get("model_id") or "unknown"),
        "run_index": str(metadata.get("run_index") or raw.get("run_index") or "0"),
        "question": question,
        "answer": answer,
        "latency_ms": _safe_float(raw.get("latency_ms", 0)),
        "kg_triples_used": _safe_int(raw.get("kg_triples_used", 0)),
        "kg_neighbors_used": _safe_int(raw.get("kg_neighbors_used", 0)),
        "kg_subgraph_triples_used": _safe_int(raw.get("kg_subgraph_triples_used", 0)),
        "kg_shortest_path_triples_used": _safe_int(raw.get("kg_shortest_path_triples_used", 0)),
        "sub_questions": _safe_int(raw.get("sub_questions", 0)),
        # Always recomputed from the answer text rather than read from the
        # artifact: the stored flag depends on the runner version that wrote it,
        # and recomputing keeps every run comparable under one rule.
        "insufficient": _is_insufficient(answer),
        "contexts": [str(c) for c in contexts if str(c).strip()],
        "retrieved_triples": [t for t in retrieved_triples if isinstance(t, dict)],
        "retrieved_entities": retrieved_entities,
    }
