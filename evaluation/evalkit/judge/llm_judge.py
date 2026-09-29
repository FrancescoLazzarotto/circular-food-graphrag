"""LLM-as-a-Judge over evaluation rows, with per-row rubric selection and caching."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

from evalkit.judge.base import JudgeBackend, JudgeResult, extract_score, parse_judge_output
from evalkit.judge.prompts import build_prompt
from evalkit.judge.rubrics import (
    ROW_KIND_ANSWERABLE,
    ROW_KIND_DISTRACTOR,
    Rubric,
    resolve_rubrics,
    row_kind,
    rubrics_for_kind,
    rubrics_for_row,
)
from evalkit.metrics.stats import aggregate, bootstrap_ci
from evalkit.models import EvalRow

logger = logging.getLogger("graphrag")

# factual_correctness and completeness are the gold's own judge_dimensions, and
# they are both here because they are scored apart: factual_correctness scores
# accuracy only, so a run that drops completeness loses that dimension
# entirely.
DEFAULT_RUBRIC_NAMES: tuple[str, ...] = (
    "factual_correctness",
    "completeness",
    "groundedness",
    "relevance",
)


def summary_rubric_names(rows: list[EvalRow], rubrics: list[Rubric]) -> list[str]:
    """Rubric names a dataset's summary should carry.

    The requested rubrics, plus any rubric the row kinds present in the data pull
    in on their own — ``abstention`` is scored on distractor rows whether or not
    it was requested.

    Args:
        rows: The dataset about to be scored.
        rubrics: The requested rubrics.

    Returns:
        Rubric names, requested ones first, each once.
    """
    names = [r.name for r in rubrics]
    kinds_present = {row_kind(row) for row in rows}
    for kind in (ROW_KIND_ANSWERABLE, ROW_KIND_DISTRACTOR):
        if kind not in kinds_present:
            continue
        for rubric in rubrics_for_kind(kind, rubrics):
            if rubric.name not in names:
                names.append(rubric.name)
    return names


def summarize_row_scores(
    row_scores: list[dict[str, Any]],
    rubric_names: list[str],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    rows_skipped: int = 0,
) -> dict[str, Any]:
    """Aggregate per-row judge scores into per-rubric summaries with bootstrap CI.

    Shared by the sequential (LLMJudge.score_dataset) and batched judging paths
    so both emit an identical result schema.

    Args:
        row_scores: Per-row dicts, each carrying a numeric value per rubric name.
        rubric_names: Rubrics to summarise.
        n_bootstrap: Bootstrap resamples for the CI.
        ci: Confidence level.
        seed: Bootstrap seed.
        rows_skipped: Count of rows excluded before scoring (for reporting).

    Returns:
        Dict with rows_evaluated, rows_skipped, per-rubric summaries, row_scores.
    """
    from statistics import mean, stdev

    summaries: dict[str, dict[str, Any]] = {}
    for rubric_name in rubric_names:
        values = [
            float(rs[rubric_name])
            for rs in row_scores
            if rs.get(rubric_name) is not None
        ]
        if values:
            ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
            summaries[rubric_name] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(values),
            }
        else:
            summaries[rubric_name] = {
                "mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0,
            }

    return {
        "rows_evaluated": len(row_scores),
        "rows_skipped": rows_skipped,
        "rubrics": summaries,
        "row_scores": row_scores,
    }


class _JudgeCache:
    """Simple LRU cache keyed by (rubric_name, prompt_hash)."""

    def __init__(self, maxsize: int = 256) -> None:
        """Start an empty cache holding at most `maxsize` results."""
        self._cache: OrderedDict[str, JudgeResult] = OrderedDict()
        self._maxsize = maxsize

    def _key(self, rubric_name: str, system: str, user: str) -> str:
        """SHA-256 of the rubric name and both prompts."""
        raw = f"{rubric_name}::{system}::{user}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, rubric_name: str, system: str, user: str) -> JudgeResult | None:
        """The cached result for these prompts; ``None`` on a miss."""
        key = self._key(rubric_name, system, user)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, rubric_name: str, system: str, user: str, result: JudgeResult) -> None:
        """Cache a result, evicting the least recently used past `maxsize`."""
        key = self._key(rubric_name, system, user)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


class LLMJudge:
    """Evaluate EvalRows using LLM-as-a-Judge with configurable rubrics.

    Which rubrics run is decided per row, not once for the dataset: a distractor
    row is scored on ``abstention`` alone.

    Args:
        backend: A JudgeBackend instance (VLLMBackend, LocalHFBackend, or APIBackend).
        rubric_names: Rubric names to apply per row; legacy names are accepted
            and resolved. Defaults to DEFAULT_RUBRIC_NAMES.
        cache_size: Maximum number of cached judge results.
    """

    def __init__(
        self,
        backend: JudgeBackend,
        rubric_names: list[str] | None = None,
        cache_size: int = 256,
    ) -> None:
        self.backend = backend
        self.rubrics: list[Rubric] = resolve_rubrics(rubric_names or DEFAULT_RUBRIC_NAMES)
        self._cache = _JudgeCache(maxsize=cache_size)

    def score_row(self, row: EvalRow) -> dict[str, JudgeResult]:
        """Evaluate a single row against the rubrics that apply to it.

        Returns:
            Dict mapping rubric_name → JudgeResult.
        """
        results: dict[str, JudgeResult] = {}
        for rubric in rubrics_for_row(row, self.rubrics):
            system, user = build_prompt(row, rubric)
            cached = self._cache.get(rubric.name, system, user)
            if cached is not None:
                results[rubric.name] = cached
                continue

            raw = self.backend.complete(system, user)
            parsed, ok = parse_judge_output(raw)

            scores: dict[str, float] = {}
            rationale = str(parsed.get("rationale", "")) if parsed else ""
            if ok:
                score = extract_score(parsed, rubric.score_field)
                if score is not None:
                    scores[rubric.name] = score

            result = JudgeResult(scores=scores, rationale=rationale, raw=raw, ok=ok and bool(scores))
            self._cache.put(rubric.name, system, user, result)
            results[rubric.name] = result

        return results

    def score_dataset(
        self,
        rows: list[EvalRow],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Score all rows and return per-rubric summaries with bootstrap CI.

        Args:
            rows: List of EvalRow to evaluate.
            n_bootstrap: Bootstrap resamples for CI.
            ci: Confidence level.
            seed: Bootstrap seed.

        Returns:
            Dict with per-rubric summary stats and row-level scores.
        """
        row_scores: list[dict[str, Any]] = []
        skipped = 0

        for row in rows:
            if row.skip_reason:
                skipped += 1
                continue
            rubric_results = self.score_row(row)
            entry: dict[str, Any] = {
                "run_dir": row.run_dir,
                "model_id": row.model_id,
                "framework": row.framework,
                "strategy": row.strategy,
                "question": row.question,
                "question_type": row.question_type,
                "skip_reason": row.skip_reason,
            }
            for rubric_name, result in rubric_results.items():
                entry[rubric_name] = result.scores.get(rubric_name)
                entry[f"{rubric_name}_rationale"] = result.rationale
                if not result.ok:
                    logger.warning(
                        "Judge failed for rubric=%s question=%r raw=%r",
                        rubric_name, row.question[:60], result.raw[:80],
                    )
            row_scores.append(entry)

        return summarize_row_scores(
            row_scores,
            summary_rubric_names(rows, self.rubrics),
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            rows_skipped=skipped,
        )
