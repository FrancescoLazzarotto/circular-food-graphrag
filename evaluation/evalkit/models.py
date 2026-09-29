"""Evaluation data model: gold queries, joined rows, metrics and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalkit.normalisation import match_key

# Mapping status values of a gold entity.
MAPPING_EXACT = "exact"
MAPPING_LOCAL = "benchmark_local_extension"


@dataclass(frozen=True)
class GoldEntity:
    """One expected entity of a gold query.

    Attributes:
        label: Human-readable label as written in the gold.
        normalised_label: Canonical form the shared resolver targets.
        alt_labels: Accepted concept-level variants (synonyms, plurals, IT forms).
        uri: Canonical vocabulary IRI, or a ``urn:ceff:`` local identifier, or None.
        mapping_status: ``exact`` (counts at grounding-level) or
            ``benchmark_local_extension`` (concept-level only).
        vocabulary: Source vocabulary name, informational.
        aligned_to: Broader CEON/AGROVOC class a local concept is aligned to.
            A proposed alignment, never asserted equivalence — never scored.
    """

    label: str
    normalised_label: str
    alt_labels: tuple[str, ...]
    uri: str | None
    mapping_status: str
    vocabulary: str = ""
    aligned_to: str | None = None

    @property
    def counts_at_grounding_level(self) -> bool:
        """True when the entity has a real vocabulary URI to be anchored to."""
        return self.mapping_status == MAPPING_EXACT

    @property
    def surface_forms(self) -> frozenset[str]:
        """Every form accepted as a concept-level hit, as comparison keys."""
        forms = [self.normalised_label, *self.alt_labels]
        return frozenset(match_key(f) for f in forms if match_key(f))


@dataclass(frozen=True)
class GoldQuery:
    """One gold query with its expected answer and entities.

    Attributes:
        query_id: Stable identifier runs are joined on.
        query_type: Question category, e.g. ``factual_simple`` or ``distractor``.
        query: The question text.
        expected_answer: Reference answer.
        expected_entities: Entities a correct answer relies on.
        expected_relations: Relations in prose; not scored.
        distractor_expected: True when the right behaviour is to abstain.
        source_verified: Source passages (document, page, evidence) behind the answer.
    """

    query_id: str
    query_type: str
    query: str
    expected_answer: str
    expected_entities: tuple[GoldEntity, ...]
    expected_relations: tuple[str, ...] = ()
    distractor_expected: bool = False
    source_verified: tuple[dict[str, Any], ...] = ()

    @property
    def grounding_entities(self) -> tuple[GoldEntity, ...]:
        """Entities in scope for grounding-level scoring: ``exact`` mappings only."""
        return tuple(e for e in self.expected_entities if e.counts_at_grounding_level)


@dataclass
class EvalRow:
    """One (run, strategy, question) tuple with all evaluation data.

    Attributes:
        run_dir: Name of the run directory.
        strategy: Retrieval strategy that answered.
        framework: ``graph_rag``, ``standard_rag`` or ``unknown``.
        model_id: Generator model.
        run_index: Repetition index within the run.
        question_id: Gold query id, or the run's own id when unjoined.
        question_type: Gold query type.
        difficulty: Difficulty label of a CSV gold.
        notes: Notes of a CSV gold.
        question: The question text.
        answer: The pipeline's answer.
        ground_truth: Reference answer; empty without a gold match.
        answer_variants: Accepted alternative answers of a CSV gold.
        contexts: Retrieved text passages.
        retrieved_triples: Retrieved graph triples.
        retrieved_entities: Retrieved graph entities.
        expected_entities: GoldEntity objects, or legacy CSV items.
        gold_triples: Expected triples of a CSV gold.
        latency_ms: End-to-end latency.
        kg_triples_used: Triples in the context.
        kg_neighbors_used: Neighbour rows in the context.
        kg_subgraph_triples_used: Subgraph triples in the context.
        kg_shortest_path_triples_used: Shortest-path triples in the context.
        sub_questions: Sub-questions from decomposition.
        insufficient: The answer declares the context insufficient.
        skip_reason: Why the row is left out of scoring; empty when scored.
        gold_query: The joined gold query, if any.
        pipeline: Pipeline label the run declared; empty when none.
    """

    run_dir: str
    strategy: str
    framework: str
    model_id: str
    run_index: str
    question_id: str
    question_type: str
    difficulty: str
    notes: str
    question: str
    answer: str
    ground_truth: str
    answer_variants: list[str]
    contexts: list[str]
    retrieved_triples: list[dict[str, Any]]
    retrieved_entities: list[Any]
    expected_entities: list[Any]
    gold_triples: list[dict[str, Any]]
    latency_ms: float
    kg_triples_used: int
    kg_neighbors_used: int
    kg_subgraph_triples_used: int
    kg_shortest_path_triples_used: int
    sub_questions: int
    insufficient: bool
    skip_reason: str
    gold_query: GoldQuery | None = None
    pipeline: str = ""

    @property
    def has_gold(self) -> bool:
        """True when the row has a reference answer."""
        return bool(self.ground_truth)

    @property
    def is_skipped(self) -> bool:
        """True when the row is left out of scoring."""
        return bool(self.skip_reason)

    @property
    def is_distractor(self) -> bool:
        """True when correct behaviour is abstention (the gold's ``scoring`` block)."""
        return bool(self.gold_query and self.gold_query.distractor_expected)


@dataclass
class MetricResult:
    """Uniform output of a single metric over a dataset.

    Attributes:
        name: Metric name.
        per_row: One value per row; ``None`` where the metric does not apply.
        extra: Metric-specific details.
    """

    name: str
    per_row: list[float | None] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float | None:
        """Mean of the non-``None`` per-row values, ``None`` when there is none."""
        values = [v for v in self.per_row if v is not None]
        if not values:
            return None
        return sum(values) / len(values)


@dataclass
class GroupSummary:
    """Aggregated metric stats for one (model_id, framework, strategy, segment).

    Attributes:
        keys: The grouping keys and their values.
        n_rows: Rows in the group.
        metrics: Metric name -> summary statistics.
    """

    keys: dict[str, str]
    n_rows: int
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class KGQualityResult:
    """Structural quality metrics of a knowledge graph.

    Attributes:
        n_entities: Nodes.
        n_triples: Relationships.
        n_predicates: Distinct relationship types.
        n_documents: Source documents.
        density: Relationships per node.
        avg_degree: Mean node degree.
        median_degree: Median node degree.
        n_components: Connected components.
        isolated_ratio: Share of nodes with degree at most 1.
        predicate_entropy: Shannon entropy (bits) of the type distribution.
        failed_chunks: Chunks stage 3 lost.
        failed_chunks_ratio: Lost chunks over attempted chunks.
        resolution_collapse_ratio: Share of entities merged away by resolution.
        entity_gold_coverage: Share of gold entities present in the graph.
        extra: Supporting details (top predicates, per-label counts, ...).
    """

    n_entities: int = 0
    n_triples: int = 0
    n_predicates: int = 0
    n_documents: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    median_degree: float = 0.0
    n_components: int = 0
    isolated_ratio: float = 0.0
    predicate_entropy: float = 0.0
    failed_chunks: int = 0
    failed_chunks_ratio: float = 0.0
    resolution_collapse_ratio: float = 0.0
    entity_gold_coverage: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionResult:
    """Comparison of current metric vs baseline.

    Attributes:
        metric: Metric name.
        baseline: Baseline value.
        current: Current value.
        delta: ``current - baseline``.
        status: ``improved``, ``stable`` or ``regressed``.
    """

    metric: str
    baseline: float
    current: float
    delta: float
    status: str  # "improved" | "stable" | "regressed"


@dataclass
class ReportModel:
    """Single input to all report renderers.

    Attributes:
        scope: ``experiment`` or ``project``.
        runs: Run directory names covered.
        groups: One summary per group.
        kg: Graph quality, when computed.
        regression: Comparison with the baseline, when computed.
        meta: Free-form report metadata.
    """

    scope: str  # "experiment" | "project"
    runs: list[str]
    groups: list[GroupSummary]
    kg: KGQualityResult | None = None
    regression: list[RegressionResult] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
