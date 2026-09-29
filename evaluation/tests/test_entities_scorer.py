"""Golden tests for the two-level entity scorer: concept level and grounding level.

Every expected P/R/F1 in this file is computed by hand in the test's own comment,
from the counts the case sets up. F1 is the count-based form 2TP / (2TP + FP + FN),
which for these cases equals the usual harmonic mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.metrics import entities
from evalkit.metrics.entities import (
    FALSE_POSITIVE,
    OUT_OF_SCOPE,
    PRF,
    abstention,
    aggregate,
    concept_level,
    grounding_level,
    level_gaps,
    score_row,
)
from evalkit.models import MAPPING_EXACT, MAPPING_LOCAL, EvalRow, GoldEntity, GoldQuery
from evalkit.normalisation import match_key

# ─── Fixtures ────────────────────────────────────────────────────────────────

URI_WHEY = "http://aims.fao.org/aos/agrovoc/c_8376"
URI_POMACE = "http://aims.fao.org/aos/agrovoc/c_16060"
URI_BRAN = "http://aims.fao.org/aos/agrovoc/c_6599"

# mapping_status=exact -> in scope at grounding level.
WHEY = GoldEntity(
    label="whey",
    normalised_label="whey",
    alt_labels=("milk whey", "siero"),
    uri=URI_WHEY,
    mapping_status=MAPPING_EXACT,
)
POMACE = GoldEntity(
    label="grape pomace",
    normalised_label="grape pomace",
    alt_labels=("vinacce", "vinaccia"),
    uri=URI_POMACE,
    mapping_status=MAPPING_EXACT,
)
BRAN = GoldEntity(
    label="rice bran",
    normalised_label="rice bran",
    alt_labels=("pula di riso", "pula"),
    uri=URI_BRAN,
    mapping_status=MAPPING_EXACT,
)
# mapping_status=benchmark_local_extension -> concept level only.
SCOTTA = GoldEntity(
    label="scotta",
    normalised_label="scotta",
    alt_labels=(),
    uri=None,
    mapping_status=MAPPING_LOCAL,
)
CAPITAL = GoldEntity(
    label="Capital",
    normalised_label="capital",
    alt_labels=("capitale",),
    uri="urn:ceff:Capital",
    mapping_status=MAPPING_LOCAL,
)


class FakeResolver:
    """Stand-in for evalkit.metrics.resolver.Resolver: resolve(label) -> URI | None."""

    def __init__(self, table: dict[str, str], ambiguous: tuple[str, ...] = ()) -> None:
        self._table = {match_key(k): v for k, v in table.items()}
        self._ambiguous = {match_key(k) for k in ambiguous}

    def resolve(self, label: str) -> str | None:
        key = match_key(label)
        if key in self._ambiguous:
            # Same (label, key, uris) signature as the real
            # evalkit.metrics.resolver.AmbiguousLabelError.
            raise entities.AMBIGUOUS_LABEL_ERROR(label, key, ["uri:a", "uri:b"])
        return self._table.get(key)


# Resolver whose vocabulary snapshot carries the multilingual AGROVOC altLabels.
MULTILINGUAL = FakeResolver(
    {
        "whey": URI_WHEY,
        "siero": URI_WHEY,
        "grape pomace": URI_POMACE,
        "vinacce": URI_POMACE,
        "rice bran": URI_BRAN,
    }
)
# Resolver that only knows the English prefLabels: an Italian label reaches
# concept level but fails to anchor: the mapping-failure case.
EN_ONLY = FakeResolver({"whey": URI_WHEY, "grape pomace": URI_POMACE, "rice bran": URI_BRAN})


def _gold(
    query_id: str = "Q01",
    query_type: str = "factual_simple",
    expected: tuple[GoldEntity, ...] = (),
    distractor: bool = False,
) -> GoldQuery:
    """A gold query with the given expected entities."""
    return GoldQuery(
        query_id=query_id,
        query_type=query_type,
        query="q?",
        expected_answer="a.",
        expected_entities=expected,
        distractor_expected=distractor,
    )


def _row(
    retrieved_entities: list | None = None,
    gold_query: GoldQuery | None = None,
    insufficient: bool = False,
    answer: str = "An answer.",
    pipeline: str = "graph_rag",
) -> EvalRow:
    """An evaluation row joined to `gold_query`, if given."""
    return EvalRow(
        run_dir="run",
        strategy="default",
        framework="graphrag",
        model_id="qwen",
        run_index="0",
        question_id=gold_query.query_id if gold_query else "Q01",
        question_type=gold_query.query_type if gold_query else "factual_simple",
        difficulty="",
        notes="",
        question="q?",
        answer=answer,
        ground_truth="a.",
        answer_variants=[],
        contexts=[],
        retrieved_triples=[],
        retrieved_entities=retrieved_entities if retrieved_entities is not None else [],
        expected_entities=list(gold_query.expected_entities) if gold_query else [],
        gold_triples=[],
        latency_ms=1.0,
        kg_triples_used=0,
        kg_neighbors_used=0,
        kg_subgraph_triples_used=0,
        kg_shortest_path_triples_used=0,
        sub_questions=0,
        insufficient=insufficient,
        skip_reason="",
        gold_query=gold_query,
        pipeline=pipeline,
    )


# ─── PRF arithmetic ──────────────────────────────────────────────────────────


def test_prf_undefined_when_nothing_expected_and_nothing_retrieved():
    prf = PRF(n_expected=0, n_retrieved=0, n_correct=0)
    assert prf.precision is None
    assert prf.recall is None
    assert prf.f1 is None


def test_prf_retrieving_nothing_is_zero_f1_not_undefined():
    # 3 expected, nothing retrieved: TP=0 FP=0 FN=3.
    # precision 0/0 -> None (you cannot be imprecise without guessing).
    # recall 0/3 = 0.0 ; F1 = 2*0/(3+0) = 0.0
    prf = PRF(n_expected=3, n_retrieved=0, n_correct=0)
    assert prf.precision is None
    assert prf.recall == 0.0
    assert prf.f1 == 0.0


# ─── Concept level ───────────────────────────────────────────────────────────


def test_resolvable_entity_found_counts_at_both_levels():
    # concept: expected 1, retrieved 1, correct 1 -> P=1/1=1.0 R=1/1=1.0 F1=2*1/(1+1)=1.0
    concept = concept_level([WHEY], ["whey"])
    assert (concept.precision, concept.recall, concept.f1) == (1.0, 1.0, 1.0)

    # grounding: expected URIs {whey}, resolved {whey}, correct 1
    #            -> P=1/1=1.0 R=1/1=1.0 F1=2*1/(1+1)=1.0
    grounding = grounding_level([WHEY], ["whey"], MULTILINGUAL)
    assert (grounding.precision, grounding.recall, grounding.f1) == (1.0, 1.0, 1.0)
    assert grounding.n_unresolved == 0


def test_found_but_unresolvable_counts_at_concept_not_grounding():
    # 'siero' is an alt_label of whey, so the concept was retrieved; the EN-only
    # resolver cannot anchor it. This is the mapping-failure case.
    # concept: expected 1, retrieved 1, correct 1 -> P=1.0 R=1.0 F1=1.0
    concept = concept_level([WHEY], ["siero"])
    assert (concept.precision, concept.recall, concept.f1) == (1.0, 1.0, 1.0)

    # grounding: expected URIs {whey}; 'siero' -> None so it is an unresolved
    # retrieval, kept in the denominator: TP=0 FP=1 FN=1
    #            -> P=0/1=0.0 R=0/1=0.0 F1=2*0/(1+1)=0.0
    grounding = grounding_level([WHEY], ["siero"], EN_ONLY)
    assert (grounding.precision, grounding.recall, grounding.f1) == (0.0, 0.0, 0.0)
    assert grounding.n_unresolved == 1


def test_local_extension_found_is_excluded_from_grounding_denominator():
    # 'scotta' is benchmark_local_extension: no vocabulary term exists, so no
    # pipeline could ever anchor it.
    # concept: expected 2, retrieved 2, correct 2 -> P=1.0 R=1.0 F1=2*2/(2+2)=1.0
    concept = concept_level([WHEY, SCOTTA], ["whey", "scotta"])
    assert (concept.precision, concept.recall, concept.f1) == (1.0, 1.0, 1.0)

    # grounding: only whey is in scope. 'scotta' is dropped from BOTH denominators.
    # expected URIs {whey}; retrieved {whey} -> TP=1 FP=0 FN=0
    #            -> P=1/1=1.0 R=1/1=1.0 F1=2*1/(1+1)=1.0
    grounding = grounding_level([WHEY, SCOTTA], ["whey", "scotta"], MULTILINGUAL)
    assert grounding.n_expected == 1
    assert grounding.n_retrieved == 1
    assert grounding.n_out_of_scope == 1
    assert (grounding.precision, grounding.recall, grounding.f1) == (1.0, 1.0, 1.0)


def test_local_extension_policy_false_positive_charges_it_instead():
    # The alternative policy, kept available: 'scotta' resolves to nothing, so
    # it is charged as a false positive.
    # TP=1 FP=1 FN=0 -> P=1/2=0.5 R=1/1=1.0 F1=2*1/(1+2)=0.6667
    grounding = grounding_level(
        [WHEY, SCOTTA], ["whey", "scotta"], MULTILINGUAL, FALSE_POSITIVE
    )
    assert grounding.n_retrieved == 2
    assert grounding.n_out_of_scope == 0
    assert grounding.precision == 0.5
    assert grounding.recall == 1.0
    assert grounding.f1 == pytest.approx(2 / 3)


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="unanchorable_expected"):
        grounding_level([WHEY], ["whey"], MULTILINGUAL, "drop_silently")


# ─── Cross-lingual matching — the central use case ───────────────────────────


def test_cross_lingual_italian_label_hits_english_gold_at_concept_level():
    # Gold is EN ('grape pomace'); our KG emits the IT node label ('vinacce').
    # concept: expected 1, retrieved 1, correct 1 -> P=1.0 R=1.0 F1=1.0
    concept = concept_level([POMACE], ["vinacce"])
    assert concept.n_correct == 1
    assert (concept.precision, concept.recall, concept.f1) == (1.0, 1.0, 1.0)


def test_cross_lingual_match_is_case_and_accent_insensitive():
    # 'Vinacce' (cased) and 'vinaccia' (variant) both reach the same concept.
    assert concept_level([POMACE], ["Vinacce"]).n_correct == 1
    assert concept_level([POMACE], ["  vinaccia. "]).n_correct == 1
    # 'ciclicità' vs 'ciclicita': accent folding, applied symmetrically.
    cyclicality = GoldEntity(
        label="Cyclicality",
        normalised_label="cyclicality",
        alt_labels=("ciclicita",),
        uri="urn:ceff:Cyclicality",
        mapping_status=MAPPING_LOCAL,
    )
    assert concept_level([cyclicality], ["ciclicità"]).n_correct == 1


def test_cross_lingual_label_anchors_when_resolver_has_multilingual_altlabels():
    # AGROVOC altLabels are multilingual, so the IT form anchors to the canonical
    # URI: the genuine strength of vocabulary grounding.
    # grounding: expected {pomace}, resolved {pomace} -> P=1.0 R=1.0 F1=1.0
    grounding = grounding_level([POMACE], ["vinacce"], MULTILINGUAL)
    assert (grounding.precision, grounding.recall, grounding.f1) == (1.0, 1.0, 1.0)


def test_several_surface_forms_of_one_concept_are_one_retrieval():
    # Emitting both the EN and IT form of a correct concept is not two hits:
    # the unit of the concept level is the concept.
    # expected 1, retrieved 1 (collapsed), correct 1 -> P=1.0 (not 0.5) R=1.0
    concept = concept_level([POMACE], ["grape pomace", "vinacce"])
    assert concept.n_retrieved == 1
    assert (concept.precision, concept.recall, concept.f1) == (1.0, 1.0, 1.0)


# ─── Asymmetric precision / recall ───────────────────────────────────────────


def test_many_spurious_entities_low_precision_high_recall():
    retrieved = ["whey", "vinacce"] + [f"junk_{i}" for i in range(8)]
    # concept: expected 2, retrieved 2 correct + 8 spurious = 10, correct 2
    #   P = 2/10 = 0.2 ; R = 2/2 = 1.0 ; F1 = 2*2/(2+10) = 4/12 = 0.3333
    concept = concept_level([WHEY, POMACE], retrieved)
    assert concept.n_expected == 2
    assert concept.n_retrieved == 10
    assert concept.precision == 0.2
    assert concept.recall == 1.0
    assert concept.f1 == pytest.approx(1 / 3)


def test_few_retrieved_entities_high_precision_low_recall():
    # concept: expected 5, retrieved 1, correct 1
    #   P = 1/1 = 1.0 ; R = 1/5 = 0.2 ; F1 = 2*1/(5+1) = 2/6 = 0.3333
    concept = concept_level([WHEY, POMACE, BRAN, SCOTTA, CAPITAL], ["whey"])
    assert concept.n_expected == 5
    assert concept.n_retrieved == 1
    assert concept.precision == 1.0
    assert concept.recall == 0.2
    assert concept.f1 == pytest.approx(1 / 3)


def test_unanchorable_junk_still_costs_grounding_precision():
    # Guards the rule that unresolvable labels are NOT discarded. A pipeline
    # emitting one correct URI plus nine unanchorable labels must not score
    # perfect grounding precision.
    # expected {whey}; resolved {whey}; unresolved 9 -> TP=1 FP=9 FN=0
    #   P = 1/10 = 0.1 ; R = 1/1 = 1.0 ; F1 = 2*1/(1+10) = 2/11 = 0.1818
    retrieved = ["whey"] + [f"unanchorable_{i}" for i in range(9)]
    grounding = grounding_level([WHEY], retrieved, MULTILINGUAL)
    assert grounding.n_unresolved == 9
    assert grounding.precision == 0.1
    assert grounding.recall == 1.0
    assert grounding.f1 == pytest.approx(2 / 11)


def test_ambiguous_label_is_counted_unresolved_not_raised():
    # A pipeline may emit any label; one ambiguous form must not abort the run.
    # It is not anchorable to *a* identifier, so it counts like any unresolved
    # label: TP=0 FP=1 FN=1 -> P=0.0 R=0.0 F1=0.0
    resolver = FakeResolver({"rice bran": URI_BRAN}, ambiguous=("pula",))
    grounding = grounding_level([BRAN], ["pula"], resolver)
    assert grounding.n_unresolved == 1
    assert grounding.n_ambiguous == 1
    assert (grounding.precision, grounding.recall, grounding.f1) == (0.0, 0.0, 0.0)


# ─── Abstention (distractors) ────────────────────────────────────────────────


def test_distractor_with_spurious_entities_scores_zero():
    gold = _gold("Q14", "distractor", expected=(), distractor=True)
    row = _row(retrieved_entities=["Economia Circolare"], gold_query=gold)
    assert abstention(row) == 0.0


def test_distractor_with_empty_entity_set_scores_one():
    gold = _gold("Q14", "distractor", expected=(), distractor=True)
    row = _row(
        retrieved_entities=[],
        gold_query=gold,
        insufficient=True,
        answer="The provided context does not contain this information.",
    )
    assert abstention(row) == 1.0


def test_distractor_entities_empty_but_answer_fabricated_scores_zero():
    gold = _gold("Q14", "distractor", expected=(), distractor=True)
    row = _row(retrieved_entities=[], gold_query=gold, insufficient=False,
               answer="The annual volume is 42,000 tonnes.")
    assert abstention(row) == 0.0


def test_blank_labels_do_not_count_as_retrieved_entities():
    gold = _gold("Q14", "distractor", expected=(), distractor=True)
    row = _row(retrieved_entities=["", "   "], gold_query=gold, insufficient=True,
               answer="Not answerable from the supplied corpus.")
    assert abstention(row) == 1.0


def test_abstention_is_none_for_answerable_queries():
    row = _row(retrieved_entities=["whey"], gold_query=_gold(expected=(WHEY,)))
    assert abstention(row) is None


def test_lexical_fallback_cannot_see_a_fabrication_behind_a_hedging_trailer():
    """Pins the known false positive of the lexical fabrication fallback.

    A no_retrieval answer defines the concept entirely from parametric knowledge
    with zero retrieved evidence, yet is flagged insufficient_answer=True purely
    because its closing section says "Il contesto fornito non contiene". The
    lexical check scores that fabrication as a correct abstention; only a
    judge-backed check sees it.
    """
    gold = _gold("Q14", "distractor", expected=(), distractor=True)
    fabricated = (
        "Le eccedenze alimentari si riferiscono a quantita di cibo prodotte che "
        "superano le necessita immediate di consumo. Possono essere recuperate "
        "tramite congelazione, distribuzione a banche del cibo o altri piatti.\n\n"
        "**Limiti e fiducia**: Il contesto fornito non contiene dettagli specifici."
    )
    row = _row(retrieved_entities=[], gold_query=gold, insufficient=True, answer=fabricated)

    # The lexical fallback is fooled: it awards a correct abstention.
    assert abstention(row) == 1.0
    # An injected judge-backed check sees the fabricated claims and rejects it.
    assert abstention(row, fabrication_check=lambda r: True) == 0.0


# ─── score_row ───────────────────────────────────────────────────────────────


def test_score_row_refuses_a_row_without_gold():
    row = _row(retrieved_entities=["whey"], gold_query=None)
    with pytest.raises(ValueError, match="no gold_query"):
        score_row(row, MULTILINGUAL)


def test_score_row_leaves_grounding_undefined_when_nothing_is_anchorable():
    # A query whose expected entities are all local extensions has no grounding
    # target: it is left out rather than scored zero.
    gold = _gold("Q01", expected=(SCOTTA, CAPITAL))
    scores = score_row(_row(retrieved_entities=["scotta"], gold_query=gold), MULTILINGUAL)
    assert scores.grounding is None
    assert scores.concept is not None
    assert scores.concept.recall == 0.5  # 1 of 2 concepts retrieved


def test_score_row_scores_distractor_on_abstention_only():
    gold = _gold("Q14", "distractor", expected=(), distractor=True)
    row = _row(retrieved_entities=[], gold_query=gold, insufficient=True,
               answer="The context does not contain this.")
    scores = score_row(row, MULTILINGUAL)
    assert scores.is_distractor is True
    assert scores.concept is None
    assert scores.grounding is None
    assert scores.abstention == 1.0


def test_score_row_reads_dict_shaped_entities():
    gold = _gold("Q01", expected=(WHEY,))
    row = _row(retrieved_entities=[{"label": "siero", "id": "n42"}], gold_query=gold)
    scores = score_row(row, MULTILINGUAL)
    assert scores.concept.n_correct == 1


# ─── Aggregation and the interoperability gap ────────────────────────────────


def _two_pipeline_scores() -> list:
    """Two pipelines over the same two queries, scored by the same resolver.

    The resolver knows English prefLabels only, so the graph-RAG pipeline (which
    emits the Italian KG node labels) retrieves every concept but anchors none,
    while the URI-native pipeline anchors everything. That contrast is the
    interoperability finding.
    """
    q01 = _gold("Q01", "factual_simple", expected=(WHEY, SCOTTA))
    q02 = _gold("Q02", "cross_vocab_relational", expected=(POMACE,))
    rows = [
        _row(["siero", "scotta"], q01, pipeline="graph_rag"),
        _row(["vinacce"], q02, pipeline="graph_rag"),
        _row(["whey", "scotta"], q01, pipeline="ontology"),
        _row(["grape pomace"], q02, pipeline="ontology"),
    ]
    return [score_row(r, EN_ONLY) for r in rows]


def test_aggregate_reports_both_levels_per_pipeline_and_query_type():
    summaries = aggregate(_two_pipeline_scores())
    keyed = {(s.keys["pipeline"], s.keys["query_type"]): s for s in summaries}
    assert set(keyed) == {
        ("graph_rag", "factual_simple"),
        ("graph_rag", "cross_vocab_relational"),
        ("ontology", "factual_simple"),
        ("ontology", "cross_vocab_relational"),
    }

    # graph_rag / factual_simple, Q01 expected {whey, scotta}, retrieved {siero, scotta}
    #   concept:   expected 2, retrieved 2, correct 2 -> P=1.0 R=1.0 F1=1.0
    #   grounding: expected {whey}; 'scotta' out of scope; 'siero' unresolved
    #              TP=0 FP=1 FN=1 -> P=0.0 R=0.0 F1=0.0
    group = keyed[("graph_rag", "factual_simple")]
    assert (group.concept_micro.precision, group.concept_micro.recall) == (1.0, 1.0)
    assert group.concept_micro.f1 == 1.0
    assert group.grounding_micro.f1 == 0.0
    assert group.grounding_micro.n_out_of_scope == 1

    # ontology / factual_simple: same concepts, but the labels anchor.
    #   grounding: expected {whey}, resolved {whey} -> P=1.0 R=1.0 F1=1.0
    group = keyed[("ontology", "factual_simple")]
    assert group.concept_micro.f1 == 1.0
    assert group.grounding_micro.f1 == 1.0


def test_aggregate_groups_distractors_on_abstention_only():
    gold = _gold("Q14", "distractor", expected=(), distractor=True)
    scores = [
        score_row(_row([], gold, insufficient=True, answer="cannot answer"), MULTILINGUAL),
        score_row(_row(["Blue Economy"], gold), MULTILINGUAL),
    ]
    summary = aggregate(scores)[0]
    assert summary.keys == {"pipeline": "graph_rag", "query_type": "distractor"}
    assert summary.n_distractor_rows == 2
    assert summary.abstention_rate == 0.5  # one correct, one with spurious entities
    # Distractors never enter the P/R/F1 tables.
    assert summary.concept_micro is None
    assert summary.grounding_micro is None


def test_level_gap_is_the_interoperability_finding():
    gaps = {g.pipeline: g for g in level_gaps(_two_pipeline_scores())}

    # graph_rag, pooled over Q01 + Q02:
    #   concept   : expected 2+1=3, retrieved 2+1=3, correct 3 -> F1 = 2*3/(3+3) = 1.0
    #   grounding : expected 1+1=2, retrieved 1+1=2, correct 0 -> F1 = 2*0/(2+2) = 0.0
    #   gap = 1.0 - 0.0 = 1.0  -> retrieves every concept, anchors none.
    graph = gaps["graph_rag"]
    assert graph.concept_f1 == 1.0
    assert graph.grounding_f1 == 0.0
    assert graph.f1_gap == 1.0
    assert graph.n_unresolved == 2

    # Like-for-like: concept restricted to the same entities and rows as
    # grounding -> expected 1+1=2, retrieved 2, correct 2 -> F1 = 1.0
    #   gap = 1.0 - 0.0 = 1.0 (pure anchoring loss, no population difference)
    assert graph.concept_f1_on_grounding_subset == 1.0
    assert graph.f1_gap_like_for_like == 1.0

    # ontology: URI-native, so both levels agree and the gap vanishes.
    onto = gaps["ontology"]
    assert onto.concept_f1 == 1.0
    assert onto.grounding_f1 == 1.0
    assert onto.f1_gap == 0.0
    assert onto.f1_gap_like_for_like == 0.0
    assert onto.n_unresolved == 0


def test_like_for_like_gap_is_not_polluted_by_out_of_scope_retrievals():
    # Q01 expects {whey (exact), scotta (local)}. A pipeline retrieving both
    # anchors whey correctly; 'scotta' must not be a false positive in the
    # like-for-like baseline, or the gap would show an anchoring loss that
    # did not happen.
    q01 = _gold("Q01", "factual_simple", expected=(WHEY, SCOTTA))
    scores = [score_row(_row(["whey", "scotta"], q01, pipeline="p"), MULTILINGUAL)]
    gap = level_gaps(scores)[0]
    assert gap.concept_f1_on_grounding_subset == 1.0
    assert gap.grounding_f1 == 1.0
    assert gap.f1_gap_like_for_like == 0.0


def test_no_metric_merges_the_two_levels():
    # Mixing the levels into one number hides what differs between
    # pipelines. Nothing in the public surface may expose a combined score.
    summary = aggregate(_two_pipeline_scores())[0]
    fields = set(vars(summary)) | set(vars(level_gaps(_two_pipeline_scores())[0]))
    for name in fields:
        assert "combined" not in name and "overall" not in name
    assert not hasattr(summary, "entity_score")
