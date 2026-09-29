"""Tests for the gazetteer answer-channel extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalkit.io.gold_loader import load_gold_json
from evalkit.metrics.entities import concept_level, retrieved_labels
from evalkit.metrics.mentions import Gazetteer, answer_channel_row
from evalkit.models import GoldEntity

GOLD_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "gold" / "gold.json"


def _entity(label: str, alts: tuple[str, ...] = ()) -> GoldEntity:
    """A benchmark-local gold entity with the given alt labels."""
    return GoldEntity(
        label=label,
        normalised_label=label.lower(),
        alt_labels=alts,
        uri=None,
        mapping_status="benchmark_local_extension",
    )


def test_whole_word_only_no_substring_hits():
    gaz = Gazetteer.from_entities([_entity("capital", ("capitale",))])
    # 'capital' must NOT fire inside the Italian 'capitalizzazione'.
    assert gaz.extract("la capitalizzazione della societa") == []
    assert gaz.extract("il capitale naturale") == ["capitale"]


def test_accent_folding_matches_both_directions():
    gaz = Gazetteer.from_entities([_entity("cyclicality", ("ciclicità",))])
    # Text with the accent matches the folded form and vice versa.
    assert gaz.extract("La ciclicità dei processi") == ["ciclicita"]
    assert gaz.extract("la ciclicita dei processi") == ["ciclicita"]


def test_multiword_form_tolerates_whitespace():
    gaz = Gazetteer.from_entities([_entity("anaerobic digestion")])
    assert gaz.extract("uses anaerobic  digestion for biogas") == [
        "anaerobic digestion"
    ]
    assert gaz.extract("anaerobic co-digestion") == []


def test_empty_text_yields_nothing():
    gaz = Gazetteer.from_entities([_entity("biogas")])
    assert gaz.extract("") == []


@pytest.mark.skipif(not GOLD_PATH.exists(), reason="gold.json not present")
def test_answer_channel_scores_text_pipeline_above_zero():
    """The point of the answer channel: a text answer naming the concepts scores.

    Without this extractor, text_only scores 0 by construction because it
    reports no retrieved_entities. Here a synthetic answer naming two expected
    concepts of a real gold query gets a non-zero concept recall through the
    UNCHANGED scoring path.
    """
    by_id = load_gold_json(GOLD_PATH)
    gaz = Gazetteer.from_gold(list(by_id.values()))

    q16 = by_id["Q16"]
    answer = (
        "The Via del Campo plant is fed with animal manure and crop biomass, "
        "producing biogas by anaerobic digestion."
    )

    import dataclasses

    from evalkit.models import EvalRow

    row = EvalRow(
        run_dir="synthetic",
        strategy="text_only",
        framework="",
        model_id="",
        run_index="",
        question_id="Q16",
        question_type="",
        difficulty="",
        notes="",
        question=q16.query,
        answer=answer,
        ground_truth="",
        answer_variants=[],
        contexts=[],
        retrieved_triples=[],
        retrieved_entities=[],
        expected_entities=[],
        gold_triples=[],
        latency_ms=0.0,
        kg_triples_used=0,
        kg_neighbors_used=0,
        kg_subgraph_triples_used=0,
        kg_shortest_path_triples_used=0,
        sub_questions=0,
        insufficient=False,
        skip_reason="",
        gold_query=q16,
    )

    # Original row: zero by construction.
    zero = concept_level(q16.expected_entities, retrieved_labels(row))
    assert zero.n_correct == 0

    scored = answer_channel_row(row, gaz)
    prf = concept_level(q16.expected_entities, retrieved_labels(scored))
    assert prf.n_correct >= 2
    assert row.retrieved_entities == []  # original untouched


@pytest.mark.skipif(not GOLD_PATH.exists(), reason="gold.json not present")
def test_global_gazetteer_exposes_cross_query_leakage():
    queries = load_gold_json(GOLD_PATH)
    gaz = Gazetteer.from_gold(list(queries.values()))
    # 'scotta' belongs to another query's expected entities: the global
    # gazetteer must still see it in free text.
    hits = gaz.extract("il siero produce la scotta durante la caseificazione")
    assert "scotta" in hits
