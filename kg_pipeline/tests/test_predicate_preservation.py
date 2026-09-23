"""An off-vocabulary predicate is remapped, not thrown away.

A sizeable share of extracted triples carry a predicate outside the vocabulary
and are remapped to RELATED_TO. The remapping itself is right: the model
invents hundreds of distinct predicate names, and letting each become a
relationship type is what the controlled vocabulary exists to prevent.
Destroying the name is not.

Every triple-returning query in the retriever reads
``coalesce(properties(r)['predicate'], type(r))``, so the model's own
predicate is kept in that property.
"""

from __future__ import annotations

import pytest

from kg_pipeline.stages import linking
from kg_pipeline.stages.neo4j_ingestion import _sanitize_props
from kg_pipeline.utils.validation import validate_triples

_VOCAB = ["USES", "PRODUCES", "RELATED_TO"]


def _raw(predicate: str, **rel) -> dict:
    """A raw extracted triple with `predicate`; kwargs join its properties."""
    return {
        "subject": "Rice husk",
        "predicate": predicate,
        "object": "substrate",
        "subject_labels": ["Material"],
        "object_labels": ["Material"],
        "subject_properties": {"name": "Rice husk"},
        "object_properties": {"name": "substrate"},
        "relationship_properties": {"source_doc": "a.pdf", **rel},
    }


def test_the_predicate_the_model_wrote_survives_the_remapping():
    (triple,) = validate_triples([_raw("USED_FOR")], allowed_predicates=_VOCAB)

    # The type is the structural part and stays inside the vocabulary.
    assert triple.predicate == "RELATED_TO"
    # The name is the part a reader needs, and it is kept.
    assert triple.relationship_properties["predicate"] == "USED_FOR"


def test_a_predicate_in_the_vocabulary_is_left_exactly_alone():
    (triple,) = validate_triples([_raw("USES")], allowed_predicates=_VOCAB)

    assert triple.predicate == "USES"
    # No redundant property: the type already says it.
    assert "predicate" not in triple.relationship_properties


def test_the_other_relationship_properties_are_not_disturbed():
    (triple,) = validate_triples(
        [_raw("HAS_ROLE", chunk_id="c1", mention_count=3)], allowed_predicates=_VOCAB
    )

    assert triple.relationship_properties["chunk_id"] == "c1"
    assert triple.relationship_properties["mention_count"] == 3
    assert triple.relationship_properties["source_doc"] == "a.pdf"
    assert triple.relationship_properties["predicate"] == "HAS_ROLE"


def test_no_vocabulary_configured_changes_nothing():
    (triple,) = validate_triples([_raw("WHATEVER")], allowed_predicates=None)

    assert triple.predicate == "WHATEVER"
    assert "predicate" not in triple.relationship_properties


@pytest.mark.parametrize("predicate", ["USED_FOR", "HAS_ROLE", "WORKS_AT", "ORGANIZED_BY"])
def test_the_name_reaches_the_property_the_retriever_reads(predicate):
    # `SET r += row.r_props` in stage 6 is what puts it on the edge, and the
    # sanitiser is the last thing that could drop it.
    (triple,) = validate_triples([_raw(predicate)], allowed_predicates=_VOCAB)

    props = _sanitize_props(triple.relationship_properties)

    assert props["predicate"] == predicate


def test_stage_five_leaves_the_preserved_name_in_place():
    (triple,) = validate_triples([_raw("USED_FOR")], allowed_predicates=_VOCAB)

    linked = linking.add_cross_document_links([triple], {}, [], include_mentioned_in=False)

    assert linked[0].relationship_properties["predicate"] == "USED_FOR"
    assert linked[0].relationship_properties["mention_count"] == 1
