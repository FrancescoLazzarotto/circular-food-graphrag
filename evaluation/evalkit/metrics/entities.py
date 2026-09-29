"""Two-level entity scorer.

Every entity comparison is reported at two levels that are NEVER merged into a
single number, because mixing them hides what differs between pipelines:

* **concept-level** — normalised surface forms vs ``GoldEntity.surface_forms``,
  over ALL expected entities. The fair, pipeline-agnostic retrieval measure.
* **grounding-level** — resolved canonical URIs, over ``mapping_status == exact``
  entities only. The interoperability / auditability measure.

The gap between the two levels is the interoperability finding and is
computed explicitly by :func:`level_gaps` rather than left to the reader.

This module deliberately exposes no aggregate that averages the two levels
together.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from evalkit.models import EvalRow, GoldEntity
from evalkit.normalisation import match_key

logger = logging.getLogger("graphrag")

# The shared resolver lives in evalkit.metrics.resolver and raises
# AmbiguousLabelError when a surface form resolves to more than one URI. This
# module is written against that contract but has to import cleanly without
# it, so the exception type is looked up defensively.
#
# Only ModuleNotFoundError is tolerated: if resolver.py exists but does not
# export AmbiguousLabelError, that is a contract violation and must fail loudly
# instead of silently degrading to an exception class that never matches.
try:  # pragma: no cover - exercised implicitly by whichever branch applies
    from evalkit.metrics.resolver import AmbiguousLabelError as _AmbiguousLabelError
except ModuleNotFoundError:  # pragma: no cover

    class _AmbiguousLabelError(Exception):
        """Stand-in used only when ``evalkit.metrics.resolver`` is absent."""


AMBIGUOUS_LABEL_ERROR: type[BaseException] = _AmbiguousLabelError

# Keys inspected, in order, when a pipeline reports entities as dicts rather
# than plain strings. Mirrors metrics/retrieval.py:_normalize_entity so both
# scorers read the same field from the same payload.
_ENTITY_LABEL_KEYS = ("label", "name", "id", "entity")

# Policies for a retrieved label that matches an expected concept the gold marks
# benchmark_local_extension. See grounding_level(). The default applies the
# exact-only grounding scope to both sides; either policy moves grounding
# precision for every pipeline equally.
OUT_OF_SCOPE = "out_of_scope"
FALSE_POSITIVE = "false_positive"

_lexical_default_warned = False


class LabelResolver(Protocol):
    """Structural type of the shared resolver.

    Matches ``evalkit.metrics.resolver.Resolver``. Injected rather than imported
    so the scorer stays testable against a fake and the resolver stays the single
    declared mapping applied symmetrically to every pipeline.
    """

    def resolve(self, label: str) -> str | None:
        """Return the canonical URI for ``label``, or None when unanchorable."""
        ...


class FabricationCheck(Protocol):
    """Decides whether an answer asserts fabricated content."""

    def __call__(self, row: EvalRow) -> bool:
        """Return True when ``row.answer`` states facts it cannot support."""
        ...


# ─── Result containers ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PRF:
    """Precision/recall/F1 held as raw counts so they pool losslessly.

    Storing counts rather than ratios lets micro-aggregation sum groups without
    re-deriving anything, and keeps the degenerate 0/0 cases explicit instead of
    silently collapsing to 0.0.

    Attributes:
        n_expected: Expected items in scope (true positives + false negatives).
        n_retrieved: Retrieved items in scope (true positives + false positives).
        n_correct: True positives.
        n_unresolved: Grounding-level only — distinct retrieved labels that the
            resolver could not anchor to any URI. Counted inside ``n_retrieved``.
        n_ambiguous: Grounding-level only — retrieved labels rejected as
            ambiguous. Subset of ``n_unresolved``, tracked for auditability.
        n_out_of_scope: Grounding-level only — retrieved labels matching an
            expected concept the gold itself declares unanchorable. Excluded
            from ``n_retrieved``; see :func:`grounding_level`.
    """

    n_expected: int
    n_retrieved: int
    n_correct: int
    n_unresolved: int = 0
    n_ambiguous: int = 0
    n_out_of_scope: int = 0

    @property
    def precision(self) -> float | None:
        """True positives over retrieved; None when nothing was retrieved."""
        if self.n_retrieved == 0:
            return None
        return self.n_correct / self.n_retrieved

    @property
    def recall(self) -> float | None:
        """True positives over expected; None when nothing was expected."""
        if self.n_expected == 0:
            return None
        return self.n_correct / self.n_expected

    @property
    def f1(self) -> float | None:
        """Count-based F1 (2TP / (2TP + FP + FN)); None when nothing to measure.

        Derived from counts rather than from precision and recall so that the
        one-sided degenerate cases stay defined: retrieving nothing when three
        entities were expected is F1 0.0, not undefined.
        """
        denominator = self.n_expected + self.n_retrieved
        if denominator == 0:
            return None
        return 2 * self.n_correct / denominator


@dataclass(frozen=True)
class MacroPRF:
    """Unweighted mean of per-row precision/recall/F1 (each query counts once).

    Attributes:
        precision: Mean precision over rows where it is defined.
        recall: Mean recall over rows where it is defined.
        f1: Mean F1 over rows where it is defined.
        n_rows: Rows averaged.
    """

    precision: float | None
    recall: float | None
    f1: float | None
    n_rows: int


@dataclass(frozen=True)
class RowScores:
    """Both levels plus abstention for a single (pipeline, query) row.

    Attributes:
        concept: Concept-level over ALL expected entities. None for
            distractors, which have no expected entities.
        grounding: Grounding-level over ``mapping_status == exact`` entities
            None when the query has no grounding-eligible entity, so a
            query with nothing to anchor never drags the grounding numbers.
        concept_on_grounding_subset: Concept-level restricted to exactly the
            entities in scope for ``grounding``. Not a reported metric — it is
            the like-for-like baseline for the gap (see :func:`level_gaps`).
        abstention: 1.0/0.0 for distractors, None otherwise.
    """

    query_id: str
    pipeline: str
    query_type: str
    is_distractor: bool
    concept: PRF | None
    grounding: PRF | None
    concept_on_grounding_subset: PRF | None
    abstention: float | None


@dataclass(frozen=True)
class LevelSummary:
    """Both levels aggregated over one group of rows, never merged.

    Attributes:
        keys: The grouping keys and their values.
        n_rows: Rows in the group.
        concept_micro: Concept-level pooled counts.
        concept_macro: Concept-level per-query mean.
        grounding_micro: Grounding-level pooled counts.
        grounding_macro: Grounding-level per-query mean.
        abstention_rate: Mean abstention over the distractor rows.
        n_distractor_rows: Distractor rows in the group.
    """

    keys: dict[str, str]
    n_rows: int
    concept_micro: PRF | None
    concept_macro: MacroPRF | None
    grounding_micro: PRF | None
    grounding_macro: MacroPRF | None
    abstention_rate: float | None
    n_distractor_rows: int


@dataclass(frozen=True)
class LevelGap:
    """Concept-minus-grounding gap for one pipeline — the interoperability finding.

    Two readings of the gap are reported because they answer different
    questions and both are defensible:

    * ``f1_gap`` — the direct reading: concept-level over ALL entities minus
      grounding-level over the exact-only subset. Pre-registered, but the two
      levels are measured over different entity populations, so part of the gap
      reflects that difference rather than anchoring cost.
    * ``f1_gap_like_for_like`` — both levels restricted to the SAME entities and
      the SAME rows. Isolates the anchoring loss and is the interpretable number.

    Attributes:
        pipeline: The pipeline.
        n_rows: Rows scored.
        concept_f1: Concept-level micro F1.
        grounding_f1: Grounding-level micro F1.
        f1_gap: ``concept_f1 - grounding_f1``.
        precision_gap: Concept minus grounding precision.
        recall_gap: Concept minus grounding recall.
        concept_f1_on_grounding_subset: Concept F1 on the grounding entities.
        f1_gap_like_for_like: That F1 minus ``grounding_f1``.
        n_unresolved: Retrieved labels the resolver could not anchor.
        n_ambiguous: Retrieved labels rejected as ambiguous.
    """

    pipeline: str
    n_rows: int
    concept_f1: float | None
    grounding_f1: float | None
    f1_gap: float | None
    precision_gap: float | None
    recall_gap: float | None
    concept_f1_on_grounding_subset: float | None
    f1_gap_like_for_like: float | None
    n_unresolved: int
    n_ambiguous: int


# ─── Label extraction ────────────────────────────────────────────────────────


def entity_label(item: Any) -> str:
    """Pull the surface label out of one retrieved entity.

    Pipelines report entities either as plain strings (graph-RAG today) or as
    dicts. Both are accepted so the same scorer reads every pipeline's payload.

    Args:
        item: One element of ``EvalRow.retrieved_entities``.

    Returns:
        The raw label, or empty string when no usable label is present.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in _ENTITY_LABEL_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def retrieved_labels(row: EvalRow) -> list[str]:
    """Return the raw labels of every entity a row retrieved.

    Args:
        row: The evaluation row.

    Returns:
        Raw labels, unnormalised and not deduplicated; empty entries dropped.
    """
    labels = [entity_label(item) for item in row.retrieved_entities]
    return [label for label in labels if label.strip()]


def _distinct_keys(retrieved: Sequence[str]) -> list[str]:
    """Deduplicate retrieved labels on their comparison key, preserving order.

    One distinct key is one retrieved item. Two different forms of the same
    concept ('grape pomace' and 'vinacce') stay two items: deduplicating them to
    one would require knowing they are the same concept, which is what the
    metric is measuring.
    """
    seen: set[str] = set()
    keys: list[str] = []
    for label in retrieved:
        key = match_key(label)
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


# ─── Concept level ───────────────────────────────────────────────────────────


def concept_level(expected: Sequence[GoldEntity], retrieved: Sequence[str]) -> PRF:
    """Score retrieved labels against expected surface forms.

    A retrieved label is correct when its comparison key is one of the expected
    entity's ``surface_forms`` (normalised_label + alt_labels). Uses ALL expected
    entities, both ``exact`` and ``benchmark_local_extension``: this is the fair,
    pipeline-agnostic retrieval measure, and an entity without a vocabulary URI
    is still a concept a pipeline can retrieve.

    Because alt_labels are multilingual, an Italian label emitted by a pipeline
    matches an English gold entity ('vinacce' -> 'grape pomace'). That is correct
    vocabulary use, not generosity.

    Args:
        expected: Expected entities of the gold query.
        retrieved: Raw labels reported by the pipeline.

    Returns:
        Counts and P/R/F1 at concept level.
    """
    scorable: list[GoldEntity] = []
    for entity in expected:
        if entity.surface_forms:
            scorable.append(entity)
        else:
            logger.warning(
                "gold entity %r has no usable surface form; excluded from the "
                "concept-level denominator (unmatchable by construction)",
                entity.label,
            )

    # The unit of this level is the CONCEPT, not the surface form: several forms
    # of one concept ('grape pomace' + 'vinacce') are one concept retrieved, not
    # two. Collapsing them keeps precision honest (emitting both forms of a
    # correct concept is not two hits) and mirrors the grounding level, where the
    # set of resolved URIs collapses the same way.
    matched: set[int] = set()
    n_spurious = 0
    for key in _distinct_keys(retrieved):
        hits = {i for i, entity in enumerate(scorable) if key in entity.surface_forms}
        if hits:
            matched |= hits
        else:
            n_spurious += 1

    return PRF(
        n_expected=len(scorable),
        n_retrieved=len(matched) + n_spurious,
        n_correct=len(matched),
    )


# ─── Grounding level ─────────────────────────────────────────────────────────


def _out_of_scope_forms(expected: Sequence[GoldEntity]) -> frozenset[str]:
    """Surface forms of expected concepts the gold declares unanchorable.

    A form shared with an anchorable concept stays in scope: the conservative
    choice, since excluding it could hide a real anchoring failure.

    Args:
        expected: The query's full expected entities.

    Returns:
        Comparison keys that are out of scope at grounding level.
    """
    anchorable: set[str] = set()
    unanchorable: set[str] = set()
    for entity in expected:
        if entity.counts_at_grounding_level:
            anchorable |= entity.surface_forms
        else:
            unanchorable |= entity.surface_forms
    return frozenset(unanchorable - anchorable)


def _resolve_key(resolver: LabelResolver, key: str) -> tuple[str | None, bool]:
    """Resolve one comparison key, absorbing ambiguity as non-anchorable.

    Args:
        resolver: The shared resolver.
        key: A normalised comparison key.

    Returns:
        (uri or None, was_ambiguous). An ambiguous label is reported rather than
        raised: a pipeline may emit any label, and one ambiguous form must not
        abort a whole evaluation run. It is not anchorable to *a* canonical
        identifier, so it is treated exactly like any other unresolved label.
    """
    try:
        return resolver.resolve(key), False
    except AMBIGUOUS_LABEL_ERROR:
        logger.warning(
            "retrieved label %r is ambiguous for the shared resolver; counted as "
            "unresolved at grounding level",
            key,
        )
        return None, True


def grounding_level(
    expected: Sequence[GoldEntity],
    retrieved: Sequence[str],
    resolver: LabelResolver,
    unanchorable_expected: str = OUT_OF_SCOPE,
) -> PRF:
    """Score resolved URIs against expected canonical URIs.

    Only entities with ``mapping_status == exact`` are in scope on the expected
    side: they are the ones with a real vocabulary URI to be anchored to. The
    filter is applied here rather than trusted from the caller.

    Mapping failures: a retrieved label that resolves to no URI
    still counts as a retrieval attempt — it stays in the precision denominator
    as a false positive, it is NOT dropped. Dropping it would let a pipeline
    emitting a hundred unanchorable labels and one correct URI score perfect
    grounding precision, erasing the very property this level measures ("an
    unanchorable entity is a real, measurable property, not discarded noise").

    That rule has one exception, controlled by ``unanchorable_expected``: a
    retrieved label matching an expected concept the gold ITSELF declares
    unanchorable (``benchmark_local_extension``).
    No pipeline can ever anchor such a concept, because no vocabulary term
    exists — charging it as a false positive would measure the benchmark's
    vocabulary gap, not the pipeline's interoperability. ``OUT_OF_SCOPE``
    applies the exact-only scope filter symmetrically to both sides and drops
    it from both denominators; ``FALSE_POSITIVE`` keeps the blanket
    false-positive rule.

    Note:
        Pass the query's FULL ``expected_entities``, not ``grounding_entities``:
        the local-extension entries are what identifies an out-of-scope
        retrieval. With a pre-filtered list nothing is out of scope and the
        function silently behaves as ``FALSE_POSITIVE``.

    Args:
        expected: The query's full expected entities; filtered to ``exact``
            internally for the expected side.
        retrieved: Raw labels reported by the pipeline.
        resolver: The shared resolver, applied identically to every pipeline.
        unanchorable_expected: ``OUT_OF_SCOPE`` (default) or ``FALSE_POSITIVE``.

    Returns:
        Counts and P/R/F1 at grounding level.

    Raises:
        ValueError: If ``unanchorable_expected`` is not one of the two policies.
    """
    if unanchorable_expected not in (OUT_OF_SCOPE, FALSE_POSITIVE):
        raise ValueError(
            f"unanchorable_expected must be {OUT_OF_SCOPE!r} or "
            f"{FALSE_POSITIVE!r}, got {unanchorable_expected!r}"
        )

    expected_uris: set[str] = set()
    for entity in expected:
        if not entity.counts_at_grounding_level:
            continue
        if not entity.uri:
            logger.warning(
                "gold entity %r is mapping_status=exact but carries no URI; "
                "excluded from the grounding-level denominator",
                entity.label,
            )
            continue
        expected_uris.add(entity.uri)

    out_of_scope = (
        _out_of_scope_forms(expected)
        if unanchorable_expected == OUT_OF_SCOPE
        else frozenset()
    )

    resolved_uris: set[str] = set()
    n_unresolved = 0
    n_ambiguous = 0
    n_out_of_scope = 0
    for key in _distinct_keys(retrieved):
        if key in out_of_scope:
            n_out_of_scope += 1
            continue
        uri, ambiguous = _resolve_key(resolver, key)
        if uri is None:
            n_unresolved += 1
            n_ambiguous += int(ambiguous)
        else:
            resolved_uris.add(uri)

    return PRF(
        n_expected=len(expected_uris),
        n_retrieved=len(resolved_uris) + n_unresolved,
        n_correct=len(resolved_uris & expected_uris),
        n_unresolved=n_unresolved,
        n_ambiguous=n_ambiguous,
        n_out_of_scope=n_out_of_scope,
    )


# ─── Abstention (distractors) ────────────────────────────────────────────────


def lexical_fabrication_check(row: EvalRow) -> bool:
    """Lexical fallback for the fabrication conjunct — KNOWN to be permissive.

    Treats an answer as fabricated when it carries no insufficiency marker at
    all (``EvalRow.insufficient``, from ``graphrag.llm.refusal.is_insufficient``).

    This is a substring test over the whole answer, so it cannot see a fabricated
    answer that also contains a hedging trailer: an answer defining 'eccedenze
    alimentari' entirely from parametric knowledge, with zero retrieved
    evidence, is flagged insufficient purely because its closing "Limiti e
    fiducia" section says "Il contesto fornito non contiene", and so scores
    as a correct abstention.

    It is therefore the FLOOR, not the decision: pass the judge-backed check to
    :func:`abstention` for reported numbers.

    Args:
        row: The evaluation row.

    Returns:
        True when the answer never signals insufficiency.
    """
    return not row.insufficient


def abstention(row: EvalRow, fabrication_check: FabricationCheck | None = None) -> float | None:
    """Score a distractor row: correct abstention is 1.0, anything else 0.0.

    Correct behaviour on a distractor is a conjunction:
    the retrieved entity set is EMPTY **and** no answer was fabricated.

    The first conjunct is decided here and is deterministic. The second needs a
    judgement about the answer's content and is injected; ``fabrication_check``
    defaults to :func:`lexical_fabrication_check`, which is documented-permissive
    and logs a warning the first time it is used.

    Args:
        row: The evaluation row.
        fabrication_check: Predicate returning True when the answer fabricates
            content. Defaults to the lexical fallback.

    Returns:
        1.0 for a correct abstention, 0.0 otherwise, None when the row is not a
        distractor (abstention is undefined for answerable queries).
    """
    global _lexical_default_warned

    if not row.is_distractor:
        return None

    # Spurious entities fail the first conjunct outright: no judgement about the
    # answer can rescue a row that retrieved things a distractor has no ground
    # truth for.
    if _distinct_keys(retrieved_labels(row)):
        return 0.0

    check = fabrication_check
    if check is None:
        check = lexical_fabrication_check
        if not _lexical_default_warned:
            _lexical_default_warned = True
            logger.warning(
                "abstention() is using the lexical fabrication fallback: it "
                "cannot detect a fabricated answer that also carries a hedging "
                "trailer. Inject a judge-backed FabricationCheck before "
                "reporting abstention in the paper."
            )
    return 0.0 if check(row) else 1.0


# ─── Row scoring ─────────────────────────────────────────────────────────────


def score_row(
    row: EvalRow,
    resolver: LabelResolver,
    fabrication_check: FabricationCheck | None = None,
    unanchorable_expected: str = OUT_OF_SCOPE,
) -> RowScores:
    """Score one row at both levels plus abstention.

    Distractors carry no expected entities, so concept and grounding are left
    undefined for them and abstention is their only entity metric — scoring a
    distractor's spurious entities as concept-level false positives too would
    count one failure twice.

    Args:
        row: The evaluation row; must carry its joined ``gold_query``.
        resolver: The shared resolver.
        fabrication_check: Optional injected fabrication predicate.
        unanchorable_expected: Grounding policy, see :func:`grounding_level`.

    Returns:
        The row's scores at both levels, never merged.

    Raises:
        ValueError: If the row has no ``gold_query`` — a silently unjoined row
            would score as an honest zero and corrupt the aggregates.
    """
    gold = row.gold_query
    if gold is None:
        raise ValueError(
            f"row for question {row.question_id!r} has no gold_query attached; "
            "refusing to score an unjoined row"
        )

    labels = retrieved_labels(row)
    pipeline = row.pipeline or row.strategy

    if gold.distractor_expected:
        return RowScores(
            query_id=gold.query_id,
            pipeline=pipeline,
            query_type=gold.query_type,
            is_distractor=True,
            concept=None,
            grounding=None,
            concept_on_grounding_subset=None,
            abstention=abstention(row, fabrication_check),
        )

    # A query with nothing anchorable is left out of the grounding aggregate
    # rather than scored 0: it has no grounding target to hit.
    grounding: PRF | None = None
    concept_subset: PRF | None = None
    if gold.grounding_entities:
        # Full expected list: grounding_level needs the local-extension entries
        # to recognise out-of-scope retrievals.
        grounding = grounding_level(
            gold.expected_entities, labels, resolver, unanchorable_expected
        )
        # The like-for-like baseline must drop the same out-of-scope retrievals
        # grounding_level drops, otherwise a correctly retrieved local-extension
        # concept is a false positive here and out of scope there, and the gap
        # stops measuring anchoring loss alone.
        out_of_scope = (
            _out_of_scope_forms(gold.expected_entities)
            if unanchorable_expected == OUT_OF_SCOPE
            else frozenset()
        )
        in_scope = [label for label in labels if match_key(label) not in out_of_scope]
        concept_subset = concept_level(gold.grounding_entities, in_scope)

    return RowScores(
        query_id=gold.query_id,
        pipeline=pipeline,
        query_type=gold.query_type,
        is_distractor=False,
        concept=concept_level(gold.expected_entities, labels),
        grounding=grounding,
        concept_on_grounding_subset=concept_subset,
        abstention=None,
    )


# ─── Aggregation ─────────────────────────────────────────────────────────────


def _sum_prf(items: Sequence[PRF]) -> PRF | None:
    """Pool counts across rows (micro-average)."""
    if not items:
        return None
    return PRF(
        n_expected=sum(i.n_expected for i in items),
        n_retrieved=sum(i.n_retrieved for i in items),
        n_correct=sum(i.n_correct for i in items),
        n_unresolved=sum(i.n_unresolved for i in items),
        n_ambiguous=sum(i.n_ambiguous for i in items),
        n_out_of_scope=sum(i.n_out_of_scope for i in items),
    )


def _mean(values: Sequence[float | None]) -> float | None:
    """Mean of the defined values; None when none are defined."""
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _macro_prf(items: Sequence[PRF]) -> MacroPRF | None:
    """Average per-row ratios (each query weighs the same)."""
    if not items:
        return None
    return MacroPRF(
        precision=_mean([i.precision for i in items]),
        recall=_mean([i.recall for i in items]),
        f1=_mean([i.f1 for i in items]),
        n_rows=len(items),
    )


def aggregate(
    scores: Sequence[RowScores],
    by: Sequence[str] = ("pipeline", "query_type"),
) -> list[LevelSummary]:
    """Aggregate both levels per group, reported side by side and never merged.

    Micro (pooled counts) and macro (per-query mean) are both returned: with ~3
    expected entities per query the per-row ratios are coarse, so micro is the
    stable headline while macro is the per-query-equal reading reviewers expect.

    Args:
        scores: Row scores, typically from :func:`score_row`.
        by: Row attributes forming the group key. Defaults to the reporting
            unit, pipeline x query_type.

    Returns:
        One summary per group, ordered by group key.

    Raises:
        AttributeError: If ``by`` names an attribute RowScores does not have.
    """
    groups: dict[tuple[str, ...], list[RowScores]] = {}
    for score in scores:
        key = tuple(str(getattr(score, field)) for field in by)
        groups.setdefault(key, []).append(score)

    summaries: list[LevelSummary] = []
    for key in sorted(groups):
        rows = groups[key]
        concept = [r.concept for r in rows if r.concept is not None]
        grounding = [r.grounding for r in rows if r.grounding is not None]
        abstentions = [r.abstention for r in rows if r.abstention is not None]
        summaries.append(
            LevelSummary(
                keys=dict(zip(by, key)),
                n_rows=len(rows),
                concept_micro=_sum_prf(concept),
                concept_macro=_macro_prf(concept),
                grounding_micro=_sum_prf(grounding),
                grounding_macro=_macro_prf(grounding),
                abstention_rate=_mean(abstentions),
                n_distractor_rows=sum(1 for r in rows if r.is_distractor),
            )
        )
    return summaries


def _gap(left: float | None, right: float | None) -> float | None:
    """Difference between two optional ratios."""
    if left is None or right is None:
        return None
    return left - right


def level_gaps(scores: Sequence[RowScores]) -> list[LevelGap]:
    """Compute the concept-minus-grounding gap per pipeline.

    This gap IS the interoperability finding, so it is a computed value rather
    than something the reader subtracts by hand. Both readings are returned; see
    :class:`LevelGap` for why the like-for-like one is the interpretable number.

    Args:
        scores: Row scores, typically from :func:`score_row`.

    Returns:
        One gap record per pipeline, ordered by pipeline name.
    """
    pipelines: dict[str, list[RowScores]] = {}
    for score in scores:
        pipelines.setdefault(score.pipeline, []).append(score)

    gaps: list[LevelGap] = []
    for pipeline in sorted(pipelines):
        rows = pipelines[pipeline]
        concept = _sum_prf([r.concept for r in rows if r.concept is not None])
        grounding = _sum_prf([r.grounding for r in rows if r.grounding is not None])
        subset = _sum_prf(
            [r.concept_on_grounding_subset for r in rows if r.concept_on_grounding_subset is not None]
        )

        concept_f1 = concept.f1 if concept else None
        grounding_f1 = grounding.f1 if grounding else None
        subset_f1 = subset.f1 if subset else None
        gaps.append(
            LevelGap(
                pipeline=pipeline,
                n_rows=len(rows),
                concept_f1=concept_f1,
                grounding_f1=grounding_f1,
                f1_gap=_gap(concept_f1, grounding_f1),
                precision_gap=_gap(
                    concept.precision if concept else None,
                    grounding.precision if grounding else None,
                ),
                recall_gap=_gap(
                    concept.recall if concept else None,
                    grounding.recall if grounding else None,
                ),
                concept_f1_on_grounding_subset=subset_f1,
                f1_gap_like_for_like=_gap(subset_f1, grounding_f1),
                n_unresolved=grounding.n_unresolved if grounding else 0,
                n_ambiguous=grounding.n_ambiguous if grounding else 0,
            )
        )
    return gaps
