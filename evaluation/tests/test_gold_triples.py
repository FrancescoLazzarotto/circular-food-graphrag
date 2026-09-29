"""Tests for gold-triple candidates: node matching, triple scoring, the review step."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.kg.gold_triples import (
    CANDIDATE_FIELDS,
    apply_review,
    match_entities_to_nodes,
    score_triple,
)


# ─── match_entities_to_nodes ─────────────────────────────────────────────────

def test_match_exact_normalized():
    nodes = ["Economia Circolare", "Città di Torino"]
    matches = match_entities_to_nodes(["economia circolare"], nodes)
    assert matches["economia circolare"] == ["Economia Circolare"]


def test_match_accent_insensitive():
    matches = match_entities_to_nodes(["citta di torino"], ["Città di Torino"])
    assert matches["citta di torino"] == ["Città di Torino"]


def test_match_substring_both_directions():
    nodes = ["Economia Circolare ittica", "RePoPP"]
    matches = match_entities_to_nodes(["economia circolare", "progetto RePoPP"], nodes)
    assert matches["economia circolare"] == ["Economia Circolare ittica"]
    assert matches["progetto RePoPP"] == ["RePoPP"]


def test_match_short_entity_no_partial():
    # < 4 chars: exact only, no substring explosion
    matches = match_entities_to_nodes(["cib"], ["cibo", "Circular"])
    assert matches["cib"] == []


def test_match_unmatched_entity_empty():
    matches = match_entities_to_nodes(["blockchain"], ["Economia Circolare"])
    assert matches["blockchain"] == []


# ─── score_triple ────────────────────────────────────────────────────────────

def test_score_anchored_and_lexical():
    triple = {"subject": "RePoPP", "predicate": "PROMOTED_BY", "object": "Città di Torino"}
    ref = {"repopp", "torino", "progetto", "gestisce"}
    matched = {"repopp", "citta di torino"}
    score = score_triple(triple, ref, matched)
    # both endpoints matched (0.2+0.2) + some lexical overlap
    assert score > 0.4


def test_score_zero_for_unrelated():
    triple = {"subject": "Kenya", "predicate": "LOCATED_IN", "object": "Africa"}
    assert score_triple(triple, {"economia", "circolare"}, set()) == 0.0


def test_score_bounded():
    triple = {"subject": "a b", "predicate": "REL", "object": "c"}
    ref = {"rel"}
    score = score_triple(triple, ref, {"a b", "c"})
    assert 0.0 <= score <= 1.0


# ─── apply_review ────────────────────────────────────────────────────────────

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write `rows` as a CSV with the given header."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_apply_review_populates_gold_triples(tmp_path):
    gold = tmp_path / "gold.csv"
    _write_csv(
        gold,
        ["question_id", "question", "canonical_answer", "gold_triples"],
        [
            {"question_id": "q1", "question": "Chi gestisce RePoPP?", "canonical_answer": "x", "gold_triples": "[]"},
            {"question_id": "q2", "question": "Altro?", "canonical_answer": "y", "gold_triples": "[]"},
        ],
    )
    candidates = tmp_path / "cands.csv"
    _write_csv(
        candidates,
        CANDIDATE_FIELDS,
        [
            {"question_id": "q1", "question": "Chi gestisce RePoPP?", "rank": 1,
             "subject": "RePoPP", "predicate": "PROMOTED_BY", "object": "Città di Torino",
             "matched_entities": "RePoPP", "source": "one_hop", "score": 0.9, "keep": "x"},
            # duplicate kept row → deduped
            {"question_id": "q1", "question": "Chi gestisce RePoPP?", "rank": 2,
             "subject": "RePoPP", "predicate": "PROMOTED_BY", "object": "Città di Torino",
             "matched_entities": "RePoPP", "source": "bridge", "score": 0.8, "keep": "sì"},
            # not kept
            {"question_id": "q1", "question": "Chi gestisce RePoPP?", "rank": 3,
             "subject": "RePoPP", "predicate": "RELATED_TO", "object": "Kenya",
             "matched_entities": "RePoPP", "source": "one_hop", "score": 0.1, "keep": ""},
        ],
    )
    out = tmp_path / "gold_out.csv"
    summary = apply_review(gold, candidates, out)

    assert summary["questions_updated"] == 1
    assert summary["triples_kept"] == 1

    with out.open() as fh:
        rows = {r["question_id"]: r for r in csv.DictReader(fh)}
    assert json.loads(rows["q1"]["gold_triples"]) == [
        {"subject": "RePoPP", "predicate": "PROMOTED_BY", "object": "Città di Torino"}
    ]
    # untouched question keeps its original value
    assert rows["q2"]["gold_triples"] == "[]"


def test_apply_review_refuses_overwrite(tmp_path):
    gold = tmp_path / "gold.csv"
    _write_csv(gold, ["question_id", "question", "gold_triples"], [])
    candidates = tmp_path / "cands.csv"
    _write_csv(candidates, CANDIDATE_FIELDS, [])
    with pytest.raises(ValueError, match="differ"):
        apply_review(gold, candidates, gold)
