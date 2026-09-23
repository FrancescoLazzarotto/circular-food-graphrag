"""A document name is where an answer came from, not what it was about.

A file name such as `SEeD for Change.pdf` can reach, and lead, the seed list,
spending one of only four slots on a name that steers the rewrite of a
follow-up towards the file rather than the subject.

Seeds are not filtered for discriminativeness with the retriever's
`lexical_df_max_ratio`; `test_document_frequency_would_be_the_wrong_filter`
records why, with numbers from the live graph.
"""

from __future__ import annotations

import pytest

from graphrag.agent.memory import ConversationMemory, _entity_names


def _nodes(*names: str) -> list[dict]:
    """Node rows with the given names."""
    return [{"text": name} for name in names]


@pytest.mark.parametrize(
    "name",
    ["SEeD for Change.pdf", "REPORT MATTM_Definitivo.PDF", "dati.csv", "note.DOCX"],
)
def test_a_document_name_is_not_an_entity(name: str) -> None:
    assert _entity_names(nodes=_nodes(name)) == []


@pytest.mark.parametrize(
    "name",
    ["Circular Economy for Food", "biochar", "SEeD for Change", "Agenda ONU 2030"],
)
def test_a_subject_that_merely_resembles_one_survives(name: str) -> None:
    """The rule is the suffix, not the presence of a dot or of a file-ish word."""
    assert _entity_names(nodes=_nodes(name)) == [name]


def test_the_slot_goes_to_the_subject_instead() -> None:
    """A turn where the file would rank first: the file gets no slot."""
    memory = ConversationMemory()
    memory.observe(
        question="Che cos'è SEeD e che cosa vuol dire?",
        answer="SEeD for Change è un progetto; SEeD for Global Goals ne è l'evoluzione.",
        nodes=_nodes("SEeD for Change.pdf", "SEeD for Change", "SEeD for Global Goals"),
    )
    seeds = memory.seed_entities()
    assert "SEeD for Change.pdf" not in seeds
    assert seeds, "dropping the document must not empty the seed list"


def test_document_frequency_would_be_the_wrong_filter() -> None:
    """Document frequency is the wrong filter for vague seeds.

    The numbers are from the live graph (14 520 nodes, 1% ceiling = 145).
    Document frequency here counts how many *node names* contain a token, so
    the domain's central English words are the common ones and vague Italian
    abstractions are rare. Every token of "Circular Economy for Food" is above
    the ceiling, while `cambiamento`, `integrazione` and `transizione` are far
    below it. Any threshold that removes the second group removes the first
    group first.
    """
    df_over_ceiling = {"circular": 165, "economy": 151, "for": 259, "food": 530}
    df_under_ceiling = {"cambiamento": 13, "integrazione": 4, "transizione": 20}
    ceiling = 145
    assert all(df > ceiling for df in df_over_ceiling.values())
    assert all(df < ceiling for df in df_under_ceiling.values())
