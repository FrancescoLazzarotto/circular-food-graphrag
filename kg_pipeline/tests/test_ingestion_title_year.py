"""The title and year on a :Document node are what the expert reads in a citation.

Two things go wrong without care. `pymupdf4llm` renders a bold heading as
`## **Title**`, and the markup would travel into the node. And the first
`19xx|20xx` found anywhere in the first three pages is not the year — it can
be a line number, an ISSN or a cited work, which dates a 2019 paper to 1943.
"""

from __future__ import annotations

import time

import pytest

from kg_pipeline.models.types import PageChunkRecord
from kg_pipeline.stages import ingestion


def _pages(*texts: str) -> list[PageChunkRecord]:
    """Page records numbered from 1."""
    return [PageChunkRecord(page_number=i, text=t) for i, t in enumerate(texts, start=1)]


def _meta(year: int | None = None, key: str = "creationDate") -> dict[str, str]:
    """PDF metadata declaring `year` under `key`, or none."""
    return {key: f"D:{year}0517103000+02'00'"} if year else {}


# --- markup ----------------------------------------------------------------


@pytest.mark.parametrize(
    "heading, expected",
    [
        ("**Energia & Materia**", "Energia & Materia"),
        ("_A Circular Economy Approach_", "A Circular Economy Approach"),
        ("__Bold__ and *italic*", "Bold and italic"),
        ("`Economia circolare`", "Economia circolare"),
        ("**Materia,   energia**", "Materia, energia"),
    ],
)
def test_the_title_reaches_the_node_without_its_markup(heading, expected):
    title, _ = ingestion._extract_title_and_year(
        _pages(f"# {heading}\n\nBody text."), fallback_title="file"
    )
    assert title == expected


def test_section_titles_lose_the_markup_too():
    # `section_title` is pasted into the extraction prompt and is what the
    # back-matter filter matches on, so the markup is not only cosmetic there.
    sections = ingestion._extract_sections(
        _pages("## **Introduction**\n\ntext", "## _Methods_\n\ntext")
    )
    assert [s.title for s in sections] == ["Introduction", "Methods"]


def test_a_heading_that_is_only_markup_is_not_a_section():
    sections = ingestion._extract_sections(_pages("## ****\n\ntext"))
    # No usable heading at all, so the document is one span rather than a
    # section titled with the empty string.
    assert [s.title for s in sections] == ["Full Document"]


def test_a_title_that_strips_to_nothing_falls_back_to_the_filename():
    title, _ = ingestion._extract_title_and_year(
        _pages("# ``\n\nBody."), fallback_title="Kenya Report_Full version"
    )
    assert title == "Kenya Report_Full version"


# --- year ------------------------------------------------------------------


def test_the_date_the_file_declares_beats_a_number_found_in_the_text():
    # A real case: a page carrying "1943" as a line number, in a paper the file
    # itself dates to 2019.
    _, year = ingestion._extract_title_and_year(
        _pages("# Paper\n\nTA1085 1943 Tourist events"),
        fallback_title="f",
        metadata=_meta(2019),
    )
    assert year == 2019


def test_a_report_whose_text_carries_no_year_still_gets_one():
    # Like `REPORT MATTM_Definitivo.pdf`, whose text dates nothing: the file's
    # metadata has to supply the year.
    _, year = ingestion._extract_title_and_year(
        _pages("# Economia circolare\n\nNessuna data qui."),
        fallback_title="f",
        metadata=_meta(2022),
    )
    assert year == 2022


def test_the_text_is_still_used_when_the_file_declares_nothing():
    _, year = ingestion._extract_title_and_year(
        _pages("# Paper\n\nPublished 2021 in Systems."), fallback_title="f", metadata={}
    )
    assert year == 2021


def test_a_declared_date_outside_the_plausible_window_is_ignored():
    # A far-future date is a broken timestamp, not a publication year.
    _, year = ingestion._extract_title_and_year(
        _pages("# Paper\n\nPublished 2021."), fallback_title="f", metadata=_meta(2099)
    )
    assert year == 2021


def test_a_scanned_year_outside_the_window_is_skipped_for_the_next_one():
    _, year = ingestion._extract_title_and_year(
        _pages("# Paper\n\nRef 2098 — published 2021."), fallback_title="f", metadata={}
    )
    assert year == 2021


def test_no_year_anywhere_stays_unknown():
    # Better an absent year than an invented one in a citation.
    _, year = ingestion._extract_title_and_year(
        _pages("# Paper\n\nNo date at all."), fallback_title="f", metadata={}
    )
    assert year is None


def test_the_modification_date_is_used_when_there_is_no_creation_date():
    _, year = ingestion._extract_title_and_year(
        _pages("# Paper\n\nNo date."), fallback_title="f", metadata=_meta(2020, "modDate")
    )
    assert year == 2020


def test_next_year_is_still_plausible():
    # Journals date an issue ahead of the calendar; the year after this one is
    # a real publication year, not a broken timestamp.
    ahead = time.localtime().tm_year + 1
    _, year = ingestion._extract_title_and_year(
        _pages("# Paper\n\nNo date."), fallback_title="f", metadata=_meta(ahead)
    )
    assert year == ahead


def test_a_disagreement_between_the_file_and_the_text_is_visible(caplog):
    # A re-saved PDF declares the day it was re-saved. The declared date still
    # wins, but the operator can see why the citation says what it says.
    with caplog.at_level("DEBUG", logger="kg_pipeline"):
        _, year = ingestion._extract_title_and_year(
            _pages("# Paper\n\nPublished 2015."),
            fallback_title="the-file",
            metadata=_meta(2024),
        )
    assert year == 2024
    assert "the-file" in caplog.text and "2015" in caplog.text
