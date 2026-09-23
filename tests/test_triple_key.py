"""One key function, not three copies of it.

The agent, the retriever and the experiment runner key triples through one
`triple_key`. Three copies of a de-duplication key would be three chances
for "the same triple" to mean something different inside one turn.
"""

from __future__ import annotations

import pytest

from graphrag.agent.core import KGRAGAgent
from graphrag.experiments.runner import ExperimentRunner
from graphrag.kg.retriever import KGRetriever
from graphrag.types import triple_key

CASI = [
    {"subject": "Coevoluzione", "predicate": "PART_OF", "object": "CEFF"},
    # Ids win when both are present: two nodes sharing a name are not one fact.
    {"subject": "A", "predicate": "Rel", "object": "B", "subject_id": "1", "object_id": "2"},
    # One id alone is not enough to identify the pair.
    {"subject": "A", "predicate": "Rel", "object": "B", "subject_id": "1"},
    # Surface forms are trimmed and lowercased, so spacing and case collapse.
    {"subject": "  Coevoluzione ", "predicate": " part_of ", "object": "CEFF"},
    {},
]


@pytest.mark.parametrize("triple", CASI)
def test_every_caller_computes_the_same_key(triple: dict) -> None:
    atteso = triple_key(triple)

    assert KGRAGAgent._triple_key(triple) == atteso
    assert KGRetriever._triple_key(triple) == atteso
    assert ExperimentRunner._triple_key(triple) == atteso


def test_ids_identify_the_pair_when_both_are_present() -> None:
    con_id = {"subject": "A", "predicate": "R", "object": "B", "subject_id": "1", "object_id": "2"},
    altro_nome = {"subject": "X", "predicate": "R", "object": "Y", "subject_id": "1", "object_id": "2"}

    assert triple_key(con_id[0]) == triple_key(altro_nome)


def test_the_same_name_on_different_nodes_is_not_the_same_fact() -> None:
    uno = {"subject": "A", "predicate": "R", "object": "B", "subject_id": "1", "object_id": "2"}
    due = {"subject": "A", "predicate": "R", "object": "B", "subject_id": "3", "object_id": "4"}

    assert triple_key(uno) != triple_key(due)


def test_case_and_spacing_collapse_without_ids() -> None:
    assert triple_key({"subject": " A ", "predicate": "REL", "object": "b"}) == (
        "a",
        "rel",
        "b",
    )
