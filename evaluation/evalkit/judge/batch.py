"""Batched LLM judging, with a resumable JSONL checkpoint."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evalkit.judge.base import JudgeBackend, extract_score, parse_judge_output
from evalkit.judge.llm_judge import summarize_row_scores, summary_rubric_names
from evalkit.judge.prompts import EVIDENCE_CHAR_BUDGET, build_prompt, build_row_content
from evalkit.judge.rubrics import (
    ROW_KIND_ANSWERABLE,
    ROW_KIND_DISTRACTOR,
    Rubric,
    resolve_rubrics,
    row_kind,
    rubrics_for_kind,
)
from evalkit.models import EvalRow

logger = logging.getLogger("graphrag")

CHECKPOINT_FILE = "judge_rows.jsonl"


_BATCH_SYSTEM_TEMPLATE = """\
You are an expert evaluator for question-answering systems.
You will be given a numbered list of items. Each item has a question, a generated \
answer, the evidence that was retrieved to answer it, and possibly a reference \
answer.

For EACH item, evaluate the generated answer on the following rubrics. Every \
rubric is scored as a float between 0 and 1. Judge every answer by the same \
standard, whatever form its evidence takes.

{rubric_block}

Respond ONLY with a JSON array, one object per item, in the same order as given:
[{{"id": <item id>, {score_keys}, "rationale": "<one sentence>"}}, ...]
Do not add any text before or after the JSON array.
"""


def _row_key(row: EvalRow) -> str:
    """Stable identity for a row across runs (for checkpoint dedupe).

    Includes model_id and run_index: with runs_per_strategy > 1 (or multiple
    models in one run dir) the same (run_dir, strategy, question) occurs more
    than once, and a coarser key would make resume drop every repeat after the
    first. Old checkpoints keyed without these fields are simply re-judged.
    """
    raw = f"{row.run_dir}|{row.model_id}|{row.run_index}|{row.strategy}|{row.question}"
    return hashlib.sha256(raw.encode()).hexdigest()


def build_batch_prompt(
    rows: list[EvalRow],
    rubrics: list[Rubric],
    char_budget: int = EVIDENCE_CHAR_BUDGET,
) -> tuple[str, str]:
    """Build (system, user) prompts scoring a batch of rows across all rubrics.

    Args:
        rows: Rows in this batch; their position is the item ``id``.
        rubrics: Rubrics to score for every row. They must agree on whether the
            ground truth is needed — see Raises.
        char_budget: Maximum characters of evidence per row.

    Returns:
        (system_prompt, user_prompt). Item ``id`` is the row's index in *rows*.

    Raises:
        ValueError: If *rubrics* is empty, or mixes rubrics that need the ground
            truth with rubrics that must not see it — one prompt either
            shows the gold answer or it does not, so such a set has to be scored
            in separate calls (:func:`partition_by_ground_truth`).
    """
    if not rubrics:
        raise ValueError("build_batch_prompt needs at least one rubric")
    uses_ground_truth = {r.uses_ground_truth for r in rubrics}
    if len(uses_ground_truth) > 1:
        names = ", ".join(r.name for r in rubrics)
        raise ValueError(
            f"Cannot build one batch prompt for rubrics that disagree about the ground "
            f"truth ({names}): score them in separate calls."
        )

    include_ground_truth = uses_ground_truth.pop()
    rubric_block = "\n".join(f"- {r.name}: {r.description}" for r in rubrics)
    score_keys = ", ".join(f'"{r.name}": <float 0-1>' for r in rubrics)
    system = _BATCH_SYSTEM_TEMPLATE.format(rubric_block=rubric_block, score_keys=score_keys)

    items = [
        f"### Item {i}\n"
        + build_row_content(
            row,
            include_ground_truth=include_ground_truth,
            char_budget=char_budget,
        )
        for i, row in enumerate(rows)
    ]
    user = "\n\n".join(items)
    return system, user


def partition_by_ground_truth(rubrics: Sequence[Rubric]) -> list[list[Rubric]]:
    """Split rubrics into groups that can share one prompt.

    Args:
        rubrics: Rubrics to score for a batch.

    Returns:
        One group of reference-free rubrics and one of reference-based rubrics,
        empty groups dropped. Each group costs one backend call per batch: the
        gold answer cannot be both hidden from groundedness and shown to
        factual_correctness in a single prompt.
    """
    groups = [
        [r for r in rubrics if not r.uses_ground_truth],
        [r for r in rubrics if r.uses_ground_truth],
    ]
    return [g for g in groups if g]


def parse_batch_array(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array of objects from a batched judge completion.

    The base ``parse_judge_output`` only matches a flat object, so batched
    output (an array) needs its own extractor.

    Returns:
        List of objects (possibly empty if no array could be parsed).
    """
    if not raw:
        return []
    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    return [o for o in parsed if isinstance(o, dict)] if isinstance(parsed, list) else []


def _score_single(row: EvalRow, rubric: Rubric, backend: JudgeBackend) -> float | None:
    """Fallback: score one (row, rubric) with the single-item prompt."""
    system, user = build_prompt(row, rubric)
    parsed, ok = parse_judge_output(backend.complete(system, user))
    if not ok:
        return None
    return extract_score(parsed, rubric.score_field)


def _entry_for(row: EvalRow) -> dict[str, Any]:
    """The identity fields a checkpoint row starts from."""
    return {
        "run_dir": row.run_dir,
        "model_id": row.model_id,
        "framework": row.framework,
        "strategy": row.strategy,
        "question": row.question,
        "question_type": row.question_type,
        "skip_reason": row.skip_reason,
        "_key": _row_key(row),
    }


def _entry_covers(entry: Mapping[str, Any], rubrics: Sequence[Rubric]) -> bool:
    """Whether a checkpointed row already carries a score for every rubric.

    A rubric that ran and failed to parse leaves its key with a None value, so a
    *missing* key means the row was never scored on that rubric — the case for a
    checkpoint written under a legacy rubric name, or when a rubric is added to
    an existing checkpoint. Such rows are re-judged rather than merged:
    scores produced under a different rubric set are not the same measurement.

    Args:
        entry: A checkpoint row.
        rubrics: The rubrics this row is meant to be scored on.

    Returns:
        True when the entry can be reused as-is.
    """
    return all(rubric.name in entry for rubric in rubrics)


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Checkpoint rows keyed by row identity; malformed lines are skipped."""
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = entry.get("_key")
        if key:
            done[key] = entry
    return done


def score_dataset_batched(
    rows: list[EvalRow],
    backend: JudgeBackend,
    rubric_names: list[str],
    batch_size: int = 8,
    out_dir: Path | None = None,
    resume: bool = False,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Score rows in batches (one backend call per batch) with checkpointing.

    One call scores ``batch_size`` rows across the rubrics that can share a
    prompt, cutting call count from rows*rubrics to roughly
    ceil(active_rows / batch_size) per rubric group. Built for the
    subscription-driven ``claude_code`` backend (rate-limit friendly) but works
    with any backend. Per-batch results are appended to ``out_dir/judge_rows.jsonl``
    so an interrupted run can resume without re-spending quota.

    Rows are grouped before batching so that every call is coherent:

    * by row kind — distractor rows are scored on ``abstention`` alone;
    * by ground-truth need — reference-free rubrics are asked in their own call,
      with no gold answer in the prompt. A rubric set spanning both
      therefore costs two calls per batch.

    Args:
        rows: EvalRows to score (rows with a skip_reason are excluded).
        backend: Any JudgeBackend.
        rubric_names: Rubric names to score per row; legacy names are resolved.
        batch_size: Rows per backend call (>=1).
        out_dir: Where to write the checkpoint; required for ``resume``.
        resume: Reuse rows the checkpoint already scores on these rubrics.
        n_bootstrap, ci, seed: Bootstrap CI parameters.

    Returns:
        Same schema as ``LLMJudge.score_dataset``.
    """
    rubrics = resolve_rubrics(rubric_names)
    active = [r for r in rows if not r.skip_reason]
    skipped = len(rows) - len(active)

    ckpt_path = (out_dir / CHECKPOINT_FILE) if out_dir else None
    done: dict[str, dict[str, Any]] = {}
    if ckpt_path and resume:
        done = _load_checkpoint(ckpt_path)
        logger.info("resume: %d rows already scored in %s", len(done), ckpt_path)

    if ckpt_path:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    pending: list[EvalRow] = []
    stale_keys: set[str] = set()
    for row in active:
        key = _row_key(row)
        entry = done.get(key)
        if entry is not None and _entry_covers(entry, rubrics_for_kind(row_kind(row), rubrics)):
            continue
        if entry is not None:
            stale_keys.add(key)
        pending.append(row)
    if stale_keys:
        logger.warning(
            "resume: re-judging %d checkpointed rows that carry no score for every "
            "requested rubric (rubric renamed, or added since the checkpoint was written)",
            len(stale_keys),
        )

    # Seeded from the intersection with the rows actually being judged, not from
    # every checkpoint entry: re-judging a filtered subset against an existing
    # judge_rows.jsonl would otherwise fold the earlier, larger dataset back into
    # the summary.
    active_keys = {_row_key(row) for row in active}
    row_scores: list[dict[str, Any]] = [
        entry
        for key, entry in done.items()
        if key in active_keys and key not in stale_keys
    ]

    ckpt_fh = ckpt_path.open("a", encoding="utf-8") if ckpt_path else None
    try:
        bs = max(1, batch_size)
        for kind in (ROW_KIND_ANSWERABLE, ROW_KIND_DISTRACTOR):
            kind_rows = [r for r in pending if row_kind(r) == kind]
            if not kind_rows:
                continue
            kind_rubrics = rubrics_for_kind(kind, rubrics)
            if not kind_rubrics:
                logger.warning(
                    "no requested rubric applies to %d %s rows — leaving them unscored",
                    len(kind_rows), kind,
                )
                continue

            for start in range(0, len(kind_rows), bs):
                batch = kind_rows[start : start + bs]
                entries: list[dict[str, Any]] = [_entry_for(r) for r in batch]

                if bs == 1:
                    # Degenerate batch → single-item prompts (the sequential path).
                    for entry, row in zip(entries, batch):
                        rationale = ""
                        for rubric in kind_rubrics:
                            system, user = build_prompt(row, rubric)
                            parsed, ok = parse_judge_output(backend.complete(system, user))
                            entry[rubric.name] = (
                                extract_score(parsed, rubric.score_field) if ok else None
                            )
                            rationale = rationale or (
                                str(parsed.get("rationale", "")) if parsed else ""
                            )
                        entry["rationale"] = rationale
                else:
                    # One call per rubric group: a single prompt cannot both hide the
                    # gold answer from groundedness and show it to factual_correctness.
                    for group in partition_by_ground_truth(kind_rubrics):
                        system, user = build_batch_prompt(batch, group)
                        by_id = {
                            o.get("id"): o
                            for o in parse_batch_array(backend.complete(system, user))
                        }
                        for idx, (entry, row) in enumerate(zip(entries, batch)):
                            obj = by_id.get(idx)
                            rationale = str(obj.get("rationale", "")) if obj else ""
                            if rationale and not entry.get("rationale"):
                                entry["rationale"] = rationale
                            for rubric in group:
                                score: float | None = None
                                if obj is not None and rubric.name in obj:
                                    score = extract_score(obj, rubric.name)
                                if score is None:
                                    # Missing/malformed → re-score just this (row, rubric).
                                    logger.warning(
                                        "batch miss id=%d rubric=%s — single-row fallback",
                                        idx, rubric.name,
                                    )
                                    score = _score_single(row, rubric, backend)
                                entry[rubric.name] = score

                for entry in entries:
                    entry.setdefault("rationale", "")
                    row_scores.append(entry)
                    if ckpt_fh:
                        ckpt_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if ckpt_fh:
                    ckpt_fh.flush()
                logger.info("judged %d/%d rows", len(row_scores), len(active))
    finally:
        if ckpt_fh:
            ckpt_fh.close()

    return summarize_row_scores(
        row_scores,
        summary_rubric_names(active, rubrics),
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        rows_skipped=skipped,
    )
