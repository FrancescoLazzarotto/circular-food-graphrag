"""Front and back matter must not be extracted as if it were domain knowledge.

Read as content, a citation list turns into AUTHORED_BY and PUBLISHED triples
and makes :Document one of the most common labels in a knowledge graph about
food.

The filter is deliberately two rules rather than one. Unambiguous headings match
anywhere in the title — no section about circular food is called
"acknowledgements". Ambiguous single words match only when they are the whole
heading, because this corpus is an early phase and will grow: "Fonti" is a
sources list, "Fonti rinnovabili" is content, and a substring rule would drop
the second one in silence.
"""

from __future__ import annotations

import pytest

from kg_pipeline.models.types import ChunkRecord
from kg_pipeline.stages.llm_extraction import _should_skip_chunk


def _chunk(section_title: str) -> ChunkRecord:
    """A chunk under the given section title."""
    return ChunkRecord(
        doc_id="d",
        filename="d.pdf",
        chunk_id="c1",
        page_range="1-2",
        section_title=section_title,
        chunk_index=1,
        text="Some text.",
    )


@pytest.mark.parametrize(
    "title",
    [
        "References",
        "REFERENCES",
        "5. References",
        "Bibliografia",
        "Bibliografia e Sitografia",
        "Bibliografia/link di approfondimento",
        "Acknowledgements",
        "Ringraziamenti",
        "Editorial Board",
        "Comitato Scientifico",
        "Table of contents",
        "List of figures",
        "Conflicts of interest",
        "Conflict of interest",
        "Conflitto di interessi",
        "Author contributions",
        "Contributi degli autori",
        "Data availability",
        "Appendix A. Supplementary data",
    ],
)
def test_unambiguous_back_matter_is_skipped_wherever_it_appears(title):
    assert _should_skip_chunk(_chunk(title)) is True


@pytest.mark.parametrize(
    "title",
    ["Fonti", "fonti", "Riferimenti", "Funding", "INDICE", "Abbreviazioni", "Glossary", "Sources"],
)
def test_an_ambiguous_word_alone_is_back_matter(title):
    assert _should_skip_chunk(_chunk(title)) is True


@pytest.mark.parametrize(
    "title",
    [
        "Fonti rinnovabili di energia",
        "Le fonti del cibo circolare",
        "Funding models for the circular economy",
        "Riferimenti normativi per la filiera",
        "Circularity Measurement Indices",
        "Sources of food loss in the supply chain",
    ],
)
def test_the_same_word_inside_a_real_heading_is_not(title):
    # The corpus will grow. A rule that reads "fonti" anywhere would take a
    # section about renewable sources out of the graph and say nothing.
    assert _should_skip_chunk(_chunk(title)) is False


@pytest.mark.parametrize(
    "title, normalised",
    [
        ("5. Fonti", "fonti"),
        ("5.1 Riferimenti", "riferimenti"),
        ("A. Glossary", "glossary"),
        ("— Note", "note"),
        ("**Fonti**", "fonti"),
        ("  FONTI  ", "fonti"),
    ],
)
def test_numbering_and_punctuation_do_not_hide_a_heading(title, normalised):
    from kg_pipeline.stages.llm_extraction import _normalise_section_title

    assert _normalise_section_title(title) == normalised
    assert _should_skip_chunk(_chunk(title)) is True


@pytest.mark.parametrize(
    "title",
    ["1. Introduction", "Descrizione dell'iniziativa", "Results", "KEYWORDS", "SmallDoc", ""],
)
def test_ordinary_sections_are_extracted(title):
    assert _should_skip_chunk(_chunk(title)) is False
