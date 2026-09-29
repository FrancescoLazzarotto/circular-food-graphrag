"""Join run results to the gold as :class:`EvalRow` objects; CSV round-trip."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.io.gold_loader import (
    extract_gold_annotations,
    gold_entity_from_dict,
    gold_entity_to_dict,
    gold_query_from_dict,
    gold_query_to_dict,
    is_json_gold,
    load_gold,
    load_gold_json,
    looks_like_gold_entity,
    normalize_question,
    pick_ground_truth,
)
from evalkit.io.run_loader import (
    _is_insufficient,
    _parse_json_dict,
    load_raw_rows,
    raw_row_to_partial,
)
from evalkit.models import EvalRow, GoldEntity, GoldQuery

logger = logging.getLogger("graphrag")

# How many distinct questions get their own text-fallback warning before the
# rest are folded into the aggregate report. The aggregate always fires.
_MAX_FALLBACK_WARNINGS = 5

REQUIRED_EVAL_COLUMNS = [
    "run_dir", "strategy", "framework", "model_id",
    "question", "answer", "ground_truth",
    "contexts_json", "retrieved_triples_json", "retrieved_entities_json",
    "expected_entities_json", "gold_triples_json",
    "has_gold_match", "skip_reason",
]


def _determine_skip_reason(
    question_key: str,
    has_gold: bool,
    malformed_json: bool,
    contexts: list[Any],
) -> str:
    """Why a row is left out of scoring; empty when it is scored."""
    if not question_key:
        return "question_not_found"
    if malformed_json:
        return "malformed_json"
    # `no_gold` is checked before `empty_context`: reporting a row with neither
    # as a context problem would send the investigation towards retrieval when
    # the gold is simply missing.
    if not has_gold:
        return "no_gold"
    if not contexts:
        return "empty_context"
    return ""


def _validate_schema(rows: list[EvalRow], min_coverage: float = 0.95) -> None:
    """Warn when a key column is filled on fewer than `min_coverage` of the rows."""
    if not rows:
        return
    total = len(rows)
    issues: list[str] = []
    for col in ("question", "answer", "ground_truth"):
        non_empty = sum(1 for r in rows if getattr(r, col, ""))
        if non_empty / total < min_coverage:
            issues.append(f"  {col}: {non_empty}/{total} non-empty")
    if issues:
        logger.warning(
            "Schema coverage below %.0f%% for:\n%s", min_coverage * 100, "\n".join(issues)
        )


def _pipeline_of(raw: dict[str, Any], partial: dict[str, Any]) -> str:
    """Return the pipeline label the run declared, or '' if it declared none.

    Deliberately pass-through. The pipeline taxonomy
    (ontology-grounded / graph-RAG / plain-text) is a property of the run being
    evaluated; inferring it from a strategy name here would invent a label the
    run never asserted.
    """
    metadata = _parse_json_dict(raw.get("metadata", raw.get("metadata_json", "")))
    declared = str(metadata.get("pipeline", "") or "").strip()
    if declared:
        return declared
    return str(raw.get("pipeline", "") or "").strip()


@dataclass
class GoldJoinStats:
    """Bookkeeping for the run↔gold join, reported loudly by _log_join_report.

    A join that fails quietly produces wrong numbers, so every degraded path is
    counted here and surfaced at WARNING level rather than left to the reader to
    notice.

    Attributes:
        total_rows: Rows seen.
        joined_by_id: Rows joined on ``query_id``.
        joined_by_text: Rows joined on the question text.
        unmatched: Rows joined to nothing.
        rows_without_query_id: Rows that carry no ``query_id``.
        unknown_query_ids: Ids the gold does not contain, with counts.
        fallback_questions: Questions joined by text.
        unmatched_questions: Questions joined to nothing.
        matched_ids: Gold ids some row joined to.
    """

    total_rows: int = 0
    joined_by_id: int = 0
    joined_by_text: int = 0
    unmatched: int = 0
    rows_without_query_id: int = 0
    unknown_query_ids: Counter[str] = field(default_factory=Counter)
    fallback_questions: set[str] = field(default_factory=set)
    unmatched_questions: set[str] = field(default_factory=set)
    matched_ids: set[str] = field(default_factory=set)

    @property
    def id_coverage(self) -> float:
        """Fraction of rows joined on query_id — the only non-degraded path."""
        if not self.total_rows:
            return 0.0
        return self.joined_by_id / self.total_rows


def _index_gold_by_question(gold_by_id: dict[str, GoldQuery]) -> dict[str, GoldQuery]:
    """Build the normalised-question → GoldQuery index used only for fallback.

    Questions that normalise to the same key are dropped from the index: a
    fallback that has to guess between two gold queries is worse than no
    fallback, because it silently attributes an answer to the wrong query.
    """
    index: dict[str, GoldQuery] = {}
    collisions: set[str] = set()
    for query in gold_by_id.values():
        key = normalize_question(query.query)
        if not key:
            continue
        if key in index:
            collisions.add(key)
            continue
        index[key] = query

    for key in collisions:
        clashing = sorted(
            q.query_id for q in gold_by_id.values() if normalize_question(q.query) == key
        )
        logger.warning(
            "Gold queries %s share the same normalised question text; they are "
            "excluded from the text fallback index (an ambiguous fallback would "
            "silently join to the wrong query).",
            clashing,
        )
        index.pop(key, None)
    return index


def _join_gold_query(
    raw: dict[str, Any],
    question: str,
    gold_by_id: dict[str, GoldQuery],
    gold_by_text: dict[str, GoldQuery],
    stats: GoldJoinStats,
) -> GoldQuery | None:
    """Resolve a raw result row to its gold query: by query_id first, text second.

    Args:
        raw: The raw result row as written by the experiment runner.
        question: The row's question text.
        gold_by_id: Primary index (query_id → GoldQuery).
        gold_by_text: Fallback index (normalised question → GoldQuery).
        stats: Mutated in place with the outcome of this row.

    Returns:
        The matching GoldQuery, or None when the row cannot be joined at all.
    """
    stats.total_rows += 1
    query_id = str(raw.get("query_id", "") or "").strip()

    if query_id and query_id in gold_by_id:
        stats.joined_by_id += 1
        stats.matched_ids.add(query_id)
        return gold_by_id[query_id]

    if query_id:
        # A run emitted an id that the gold does not contain: never fall back
        # silently, this means run and gold are out of sync.
        stats.unknown_query_ids[query_id] += 1
    else:
        stats.rows_without_query_id += 1

    key = normalize_question(question)
    gold_query = gold_by_text.get(key) if key else None
    if gold_query is not None:
        stats.joined_by_text += 1
        if question not in stats.fallback_questions:
            if len(stats.fallback_questions) < _MAX_FALLBACK_WARNINGS:
                logger.warning(
                    "GOLD JOIN FALLBACK: row has no usable query_id (%s) — joined to "
                    "gold %s by matching question TEXT instead. This is fragile: any "
                    "edit to the question wording silently unjoins the row. Emit "
                    "query_id from the runner (--questions-file with ids).",
                    f"query_id={query_id!r} not in gold" if query_id else "field absent",
                    gold_query.query_id,
                )
            stats.fallback_questions.add(question)
        stats.matched_ids.add(gold_query.query_id)
        return gold_query

    stats.unmatched += 1
    stats.unmatched_questions.add(question)
    return None


def _log_join_report(stats: GoldJoinStats, gold_by_id: dict[str, GoldQuery]) -> None:
    """Emit the aggregate join report; anything below a perfect join warns.

    Two independent coverages are checked, because either can be complete while
    the other is not: every row may join by id while gold queries are still
    missing from the run entirely (e.g. a run that crashed after 28/30).
    """
    if not stats.total_rows:
        return

    missing = sorted(set(gold_by_id) - stats.matched_ids)

    if stats.id_coverage >= 1.0 and not missing:
        logger.info(
            "Gold join: %d/%d rows joined by query_id (100%%), all %d gold "
            "queries covered.",
            stats.joined_by_id,
            stats.total_rows,
            len(gold_by_id),
        )
        return

    if missing:
        logger.warning(
            "GOLD COVERAGE: %d/%d gold queries were never matched by any row: %s. "
            "Any metric aggregated over this dataset silently excludes them.",
            len(missing),
            len(gold_by_id),
            missing,
        )

    if stats.id_coverage >= 1.0:
        return

    logger.warning(
        "GOLD JOIN NOT CLEAN: only %d/%d rows (%.1f%%) joined by query_id. "
        "text-fallback=%d, unmatched=%d, rows without query_id=%d.",
        stats.joined_by_id,
        stats.total_rows,
        stats.id_coverage * 100,
        stats.joined_by_text,
        stats.unmatched,
        stats.rows_without_query_id,
    )

    if stats.joined_by_text:
        extra = len(stats.fallback_questions) - _MAX_FALLBACK_WARNINGS
        logger.warning(
            "GOLD JOIN FALLBACK TOTAL: %d rows over %d distinct questions were "
            "joined by question text, not by id%s. Every metric computed on these "
            "rows depends on question wording matching the gold byte-for-byte.",
            stats.joined_by_text,
            len(stats.fallback_questions),
            f" ({extra} further questions not logged individually)" if extra > 0 else "",
        )

    if stats.unknown_query_ids:
        logger.warning(
            "GOLD JOIN: %d rows carry query_ids absent from the gold: %s. "
            "Run and gold are out of sync.",
            sum(stats.unknown_query_ids.values()),
            sorted(stats.unknown_query_ids),
        )

    if stats.unmatched:
        sample = sorted(stats.unmatched_questions)[:3]
        logger.warning(
            "GOLD JOIN: %d rows (%d distinct questions) matched NO gold query and "
            "carry no ground truth. Sample: %s",
            stats.unmatched,
            len(stats.unmatched_questions),
            [q[:70] for q in sample],
        )


def build_dataset(
    run_dirs: list[Path],
    gold_path: Path | None = None,
    tag_contains: str = "",
    min_schema_coverage: float = 0.95,
) -> list[EvalRow]:
    """Join run results with optional gold labels into a list of EvalRow.

    Args:
        run_dirs: One or more experiment run directories OR a single parent
            directory (artifacts/experiments) — in the latter case all
            subdirectories matching *tag_contains* are walked.
        gold_path: Optional gold for the ground-truth join. A ``.json`` gold is
            joined on ``query_id`` (primary) with a warned text fallback; any
            other suffix is read as the legacy CSV gold and joined on question
            text.
        tag_contains: Filter string applied to run-directory names.
        min_schema_coverage: Minimum non-empty ratio for key columns.

    Returns:
        List of EvalRow, one per (run, strategy, question).
    """
    resolved_dirs = _resolve_run_dirs(run_dirs, tag_contains)

    gold_map: dict[str, dict[str, str]] = {}
    gold_by_id: dict[str, GoldQuery] = {}
    gold_by_text: dict[str, GoldQuery] = {}
    json_gold = False

    if gold_path:
        json_gold = is_json_gold(gold_path)
        if json_gold:
            gold_by_id = load_gold_json(gold_path)
            gold_by_text = _index_gold_by_question(gold_by_id)
        else:
            gold_map = load_gold(gold_path)

    out: list[EvalRow] = []
    skip_counter: Counter[str] = Counter()
    stats = GoldJoinStats()

    for run_dir in resolved_dirs:
        raw_rows = load_raw_rows(run_dir)
        run_dir_name = run_dir.name

        for raw in raw_rows:
            partial = raw_row_to_partial(raw, run_dir_name)
            question = partial["question"]
            question_key = normalize_question(question)

            gold_row: dict[str, str] = {}
            gold_query: GoldQuery | None = None

            expected_entities: list[Any] = []
            gold_triples: list[Any] = []
            answer_variants: list[str] = []
            ground_truth = ""

            if json_gold:
                gold_query = _join_gold_query(
                    raw=raw,
                    question=question,
                    gold_by_id=gold_by_id,
                    gold_by_text=gold_by_text,
                    stats=stats,
                )
                if gold_query is not None:
                    ground_truth = gold_query.expected_answer
                    expected_entities = list(gold_query.expected_entities)
                    # expected_relations stay prose on gold_query: there is no
                    # scoring rule for them, and feeding them to the triple
                    # metrics would fabricate numbers.
            else:
                gold_row = gold_map.get(question_key, {})
                if gold_row:
                    ground_truth = pick_ground_truth(gold_row)
                    expected_entities, gold_triples, answer_variants = (
                        extract_gold_annotations(gold_row)
                    )

            has_gold = bool(gold_query) if json_gold else bool(gold_row)

            skip_reason = _determine_skip_reason(
                question_key=question_key,
                has_gold=has_gold,
                malformed_json=False,
                contexts=partial["contexts"],
            )
            if skip_reason:
                skip_counter[skip_reason] += 1

            out.append(
                EvalRow(
                    run_dir=run_dir_name,
                    strategy=partial["strategy"],
                    framework=partial["framework"],
                    model_id=partial["model_id"],
                    run_index=partial["run_index"],
                    question_id=(
                        gold_query.query_id
                        if gold_query is not None
                        else str(raw.get("query_id", "") or "")
                        if json_gold
                        else gold_row.get("question_id", "")
                    ),
                    question_type=(
                        gold_query.query_type
                        if gold_query is not None
                        else "" if json_gold else gold_row.get("question_type", "")
                    ),
                    difficulty="" if json_gold else gold_row.get("difficulty", ""),
                    notes="" if json_gold else gold_row.get("notes", ""),
                    question=question,
                    answer=partial["answer"],
                    ground_truth=ground_truth,
                    answer_variants=[str(v) for v in answer_variants],
                    contexts=partial["contexts"],
                    retrieved_triples=partial["retrieved_triples"],
                    retrieved_entities=partial["retrieved_entities"],
                    expected_entities=expected_entities,
                    gold_triples=gold_triples,
                    latency_ms=partial["latency_ms"],
                    kg_triples_used=partial["kg_triples_used"],
                    kg_neighbors_used=partial["kg_neighbors_used"],
                    kg_subgraph_triples_used=partial["kg_subgraph_triples_used"],
                    kg_shortest_path_triples_used=partial["kg_shortest_path_triples_used"],
                    sub_questions=partial["sub_questions"],
                    insufficient=partial["insufficient"],
                    skip_reason=skip_reason,
                    gold_query=gold_query,
                    pipeline=_pipeline_of(raw, partial),
                )
            )

    if skip_counter:
        logger.info("skip_reasons=%s", dict(skip_counter))

    if json_gold:
        _log_join_report(stats, gold_by_id)

    _validate_schema(out, min_schema_coverage)
    return out


def _resolve_run_dirs(run_dirs: list[Path], tag_contains: str) -> list[Path]:
    """Resolve a list of paths to concrete run directories.

    A single directory with results.jsonl/results.csv is used as-is.
    A parent directory without results files is walked to find subdirs.
    """
    resolved: list[Path] = []

    for path in run_dirs:
        if not path.exists():
            logger.warning("Run path not found, skipping: %s", path)
            continue

        if (path / "results.jsonl").exists() or (path / "results.csv").exists():
            resolved.append(path)
            continue

        # Parent directory — walk subdirs
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if tag_contains and tag_contains not in child.name:
                continue
            if (child / "results.jsonl").exists() or (child / "results.csv").exists():
                resolved.append(child)

    return resolved


# ─── CSV round-trip helpers ──────────────────────────────────────────────────

def rows_to_csv(rows: list[EvalRow], output_path: Path) -> None:
    """Write EvalRow list to a CSV compatible with the legacy build_eval_dataset format."""
    fieldnames = [
        "run_dir", "strategy", "framework", "model_id", "run_index",
        "question_id", "question_type", "difficulty", "notes",
        "question", "answer", "ground_truth", "answer_variants_json",
        "latency_ms", "kg_triples_used", "kg_neighbors_used",
        "kg_subgraph_triples_used", "kg_shortest_path_triples_used", "sub_questions",
        "contexts_json", "retrieved_triples_json", "retrieved_entities_json",
        "expected_entities_json", "gold_triples_json",
        "has_gold_match", "skip_reason",
        # Appended (never inserted) so existing DictReader consumers of the
        # legacy column order keep working.
        "pipeline", "gold_query_json",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))


def _entity_to_jsonable(entity: Any) -> Any:
    """Make an expected-entity JSON-safe: GoldEntity → dict, legacy items unchanged."""
    if isinstance(entity, GoldEntity):
        return gold_entity_to_dict(entity)
    return entity


def _row_to_dict(row: EvalRow) -> dict[str, str]:
    """The CSV row of an :class:`EvalRow`."""
    return {
        "run_dir": row.run_dir,
        "strategy": row.strategy,
        "framework": row.framework,
        "model_id": row.model_id,
        "run_index": row.run_index,
        "question_id": row.question_id,
        "question_type": row.question_type,
        "difficulty": row.difficulty,
        "notes": row.notes,
        "question": row.question,
        "answer": row.answer,
        "ground_truth": row.ground_truth,
        "answer_variants_json": json.dumps(row.answer_variants, ensure_ascii=False),
        "latency_ms": str(row.latency_ms),
        "kg_triples_used": str(row.kg_triples_used),
        "kg_neighbors_used": str(row.kg_neighbors_used),
        "kg_subgraph_triples_used": str(row.kg_subgraph_triples_used),
        "kg_shortest_path_triples_used": str(row.kg_shortest_path_triples_used),
        "sub_questions": str(row.sub_questions),
        "contexts_json": json.dumps(row.contexts, ensure_ascii=False),
        "retrieved_triples_json": json.dumps(row.retrieved_triples, ensure_ascii=False),
        "retrieved_entities_json": json.dumps(row.retrieved_entities, ensure_ascii=False),
        "expected_entities_json": json.dumps(
            [_entity_to_jsonable(e) for e in row.expected_entities], ensure_ascii=False
        ),
        "gold_triples_json": json.dumps(row.gold_triples, ensure_ascii=False),
        "has_gold_match": "1" if row.has_gold else "0",
        "skip_reason": row.skip_reason,
        "pipeline": row.pipeline,
        # Without this column the round-trip drops distractor_expected, and
        # EvalRow.is_distractor silently degrades to False on every reload.
        "gold_query_json": (
            json.dumps(gold_query_to_dict(row.gold_query), ensure_ascii=False)
            if row.gold_query is not None
            else ""
        ),
    }


def rows_from_csv(input_path: Path) -> list[EvalRow]:
    """Load EvalRow list from a CSV produced by rows_to_csv or build_eval_dataset."""
    rows: list[EvalRow] = []
    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            rows.append(_dict_to_row(raw))
    return rows


def _dict_to_row(d: dict[str, str]) -> EvalRow:
    """Rebuild an :class:`EvalRow` from a CSV row."""
    def _jlist(key: str) -> list[Any]:
        """The JSON list in column `key`, empty when absent or invalid."""
        raw = d.get(key, "")
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    def _f(key: str) -> float:
        """Column `key` as a float, 0 when absent or invalid."""
        try:
            return float(d.get(key, 0) or 0)
        except (ValueError, TypeError):
            return 0.0

    def _i(key: str) -> int:
        """Column `key` as an int, 0 when absent or invalid."""
        try:
            return int(d.get(key, 0) or 0)
        except (ValueError, TypeError):
            return 0

    gold_query = _gold_query_from_csv(d)
    if gold_query is not None:
        # The serialised gold_query is authoritative and already typed.
        expected_entities: list[Any] = list(gold_query.expected_entities)
    else:
        expected_entities = _rehydrate_entities(_jlist("expected_entities_json"))

    return EvalRow(
        run_dir=d.get("run_dir", ""),
        strategy=d.get("strategy", ""),
        framework=d.get("framework", ""),
        model_id=d.get("model_id", ""),
        run_index=d.get("run_index", "0"),
        question_id=d.get("question_id", ""),
        question_type=d.get("question_type", ""),
        difficulty=d.get("difficulty", ""),
        notes=d.get("notes", ""),
        question=d.get("question", ""),
        answer=d.get("answer", ""),
        ground_truth=d.get("ground_truth", ""),
        answer_variants=_jlist("answer_variants_json"),
        contexts=_jlist("contexts_json"),
        retrieved_triples=_jlist("retrieved_triples_json"),
        retrieved_entities=_jlist("retrieved_entities_json"),
        expected_entities=expected_entities,
        gold_triples=_jlist("gold_triples_json"),
        latency_ms=_f("latency_ms"),
        kg_triples_used=_i("kg_triples_used"),
        kg_neighbors_used=_i("kg_neighbors_used"),
        kg_subgraph_triples_used=_i("kg_subgraph_triples_used"),
        kg_shortest_path_triples_used=_i("kg_shortest_path_triples_used"),
        sub_questions=_i("sub_questions"),
        # Recomputed from the answer text (same rule as raw_row_to_partial), so
        # the CSV round-trip keeps the insufficiency signal.
        insufficient=_is_insufficient(d.get("answer", "")),
        skip_reason=d.get("skip_reason", ""),
        gold_query=gold_query,
        pipeline=d.get("pipeline", ""),
    )


def _rehydrate_entities(raw_items: list[Any]) -> list[Any]:
    """Rebuild GoldEntity objects from a CSV round-trip, leaving legacy items as-is."""
    out: list[Any] = []
    for item in raw_items:
        if looks_like_gold_entity(item):
            entity = gold_entity_from_dict(item)
            if entity is not None:
                out.append(entity)
                continue
        out.append(item)
    return out


def _gold_query_from_csv(d: dict[str, str]) -> GoldQuery | None:
    """Rebuild EvalRow.gold_query from the gold_query_json column, if present."""
    raw = d.get("gold_query_json", "")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Malformed gold_query_json for question %r; the row loses its gold "
            "query (distractor handling and entity scoring degrade for it).",
            d.get("question", "")[:70],
        )
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        return gold_query_from_dict(parsed)
    except ValueError as exc:
        logger.warning("Could not rebuild gold_query from CSV: %s", exc)
        return None
