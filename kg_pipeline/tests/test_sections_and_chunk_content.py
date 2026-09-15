"""A page carrying three headings used to become one section, and lost the rest.

`_extract_sections` broke out of the page after the first heading, and a running
header broke out of the page entirely. On the production corpus that discarded
1 418 of 2 251 headings — 64 % — and `section_title` is both what the extraction
prompt is given and what the back-matter filter matches on.

Recovering them is not a matter of deleting the two `break`s: sections were
page-granular and the chunker takes every page in a section's range, so two
sections starting on the same page both claimed all of it. Measured before
changing it: 34 % of the corpus's pages would have been chunked more than once,
one catalogue page seventeen times, inflating the graph with duplicate triples
and with them the mention counts the retriever ranks on. So a section now
carries an offset into its first and last page.
"""

from __future__ import annotations

import pytest

from kg_pipeline.models.types import DocumentRecord, PageChunkRecord
from kg_pipeline.stages import chunking
from kg_pipeline.stages.ingestion import _extract_sections


def _pages(*texts: str) -> list[PageChunkRecord]:
    return [PageChunkRecord(page_number=i, text=t) for i, t in enumerate(texts, start=1)]


def _doc(pages: list[PageChunkRecord]) -> DocumentRecord:
    doc = DocumentRecord(
        doc_id="d",
        filename="d.pdf",
        page_count=len(pages),
        markdown_text="\n\n".join(p.text for p in pages),
        page_chunks=pages,
    )
    doc.sections = _extract_sections(pages)
    return doc


_CFG = {
    "chunking": {
        "small_max_pages": 0,
        "medium_max_pages": 80,
        "small_min_tokens": 200,
        "small_max_tokens": 400,
        "medium_window_tokens": 512,
        "medium_overlap_tokens": 0,
        "large_window_tokens": 1024,
        "large_overlap_tokens": 0,
    }
}


# --- every heading becomes a section ---------------------------------------


def test_three_headings_on_one_page_are_three_sections():
    pages = _pages(
        "## First\n\nBody of the first one.\n\n"
        "## Second\n\nBody of the second one.\n\n"
        "## Third\n\nBody of the third one."
    )
    assert [s.title for s in _extract_sections(pages)] == ["First", "Second", "Third"]


def test_a_running_header_does_not_hide_the_headings_under_it():
    # The magazine masthead opens every page. It used to abort the scan of the
    # whole page, so the real headings below it were never seen.
    pages = _pages(
        "# MATERIA RINNOVABILE\n\n## Editorial\n\nText.",
        "# MATERIA RINNOVABILE\n\n## Oceans\n\nText.",
    )
    titles = [s.title for s in _extract_sections(pages)]
    assert "Editorial" in titles and "Oceans" in titles


def test_a_title_that_recurs_with_other_sections_between_is_kept_each_time():
    # A catalogue repeats "Results" under each of its cases. Those are real
    # section boundaries, not a running header: the corpus has one document that
    # repeats the same four headings for 56 separate initiatives.
    pages = _pages(
        "## Case A\n\nText.\n\n## Results\n\nText.",
        "## Case B\n\nText.\n\n## Results\n\nText.",
    )
    assert [s.title for s in _extract_sections(pages)] == [
        "Case A", "Results", "Case B", "Results",
    ]


def test_a_masthead_repeated_with_nothing_between_opens_one_section():
    pages = _pages("# MATERIA RINNOVABILE\n\nText.", "# MATERIA RINNOVABILE\n\nMore text.")
    assert [s.title for s in _extract_sections(pages)] == ["MATERIA RINNOVABILE"]


# --- the offsets that stop the duplication ---------------------------------


def test_two_sections_on_one_page_do_not_both_claim_all_of_it():
    doc = _doc(_pages("## First\n\nAlpha alpha alpha.\n\n## Second\n\nBeta beta beta."))

    chunks = chunking.chunk_documents([doc], _CFG)

    by_title = {c.section_title: c.text for c in chunks}
    assert "Alpha alpha alpha." in by_title["First"]
    assert "Beta beta beta." not in by_title["First"]
    assert "Alpha alpha alpha." not in by_title["Second"]


def test_the_text_above_a_mid_page_heading_belongs_to_the_section_before_it():
    # The old rule ended a section on the previous page, so text sitting above
    # the next heading was attributed to whichever section began that page.
    doc = _doc(
        _pages(
            "## First\n\nAlpha alpha alpha.",
            "Tail of the first section.\n\n## Second\n\nBeta beta beta.",
        )
    )

    chunks = chunking.chunk_documents([doc], _CFG)

    first = "\n\n".join(c.text for c in chunks if c.section_title == "First")
    second = "\n\n".join(c.text for c in chunks if c.section_title == "Second")
    assert "Tail of the first section." in first
    assert "Tail of the first section." not in second


def test_the_heading_line_itself_does_not_end_up_in_the_body():
    doc = _doc(_pages("## **A Heading**\n\nSome body text here for the chunk."))

    chunks = chunking.chunk_documents([doc], _CFG)

    assert chunks
    assert all("##" not in c.text for c in chunks)
    assert chunks[0].section_title == "A Heading"


def test_a_document_still_yields_every_paragraph_once():
    doc = _doc(
        _pages(
            "## One\n\nParagraph one here.\n\n## Two\n\nParagraph two here.",
            "## Three\n\nParagraph three here.",
        )
    )

    joined = "\n\n".join(c.text for c in chunking.chunk_documents([doc], _CFG))

    for paragraph in ("Paragraph one here.", "Paragraph two here.", "Paragraph three here."):
        assert joined.count(paragraph) == 1


# --- nothing to extract ----------------------------------------------------


@pytest.mark.parametrize("noise", ["5", "_Cont._", "50\n\n51", "**Table 1.** _Cont._"])
def test_a_chunk_that_cannot_hold_a_triple_is_not_sent_to_the_model(noise):
    # A subject, a predicate and an object need three words. Below that it is a
    # page footer or a table continuation marker, and it costs a full LLM call.
    doc = _doc(_pages(f"## Section\n\n{noise}\n\n## Real\n\nRice husk is used as a substrate."))

    chunks = chunking.chunk_documents([doc], _CFG)

    assert [c.section_title for c in chunks] == ["Real"]


def test_a_document_made_only_of_noise_is_still_ingested():
    # Dropping every chunk would remove the document from the graph silently,
    # which is the failure this guard exists to prevent.
    doc = _doc(_pages("## Section\n\n5\n\n## Other\n\n7"))

    chunks = chunking.chunk_documents([doc], _CFG)

    assert chunks, "a document must not vanish because its chunks are thin"


def test_a_short_but_real_sentence_survives():
    doc = _doc(_pages("## Section\n\nRice husk is a substrate."))

    chunks = chunking.chunk_documents([doc], _CFG)

    assert [c.text for c in chunks] == ["Rice husk is a substrate."]


# --- artifacts written before the offsets existed --------------------------


def test_a_section_without_offsets_still_means_the_whole_page():
    # Stage 1 must keep working on a stage 0 artifact from an earlier run.
    doc = DocumentRecord(
        doc_id="d",
        filename="d.pdf",
        page_count=1,
        markdown_text="x",
        page_chunks=_pages("Alpha alpha alpha.\n\nBeta beta beta."),
        sections=[{"title": "Old", "level": 1, "start_page": 1, "end_page": 1}],
    )

    chunks = chunking.chunk_documents([doc], _CFG)

    joined = "\n\n".join(c.text for c in chunks)
    assert "Alpha alpha alpha." in joined and "Beta beta beta." in joined


# --- a paragraph bigger than the window ------------------------------------
#
# A markdown table's rows are separated by single newlines, so the whole table
# is one paragraph and walked past the token budget untouched: 42 of the 55
# oversized paragraphs in the corpus are tables, the largest 2 526 tokens
# against a budget of 512. The other 13 are prose rendered as a single line.


def _table(rows: int) -> str:
    head = "|**Variable**|**Category**|**n**|\n|---|---|---|"
    body = "\n".join(f"|variable number {i}|category {i}|{i}|" for i in range(rows))
    return head + "\n" + body


def test_a_table_is_split_into_row_groups(tmp_path):
    doc = _doc(_pages("## Data\n\n" + _table(120)))

    chunks = chunking.chunk_documents([doc], _CFG)

    assert len(chunks) > 1
    assert all(chunking._token_count(c.text) <= _CFG["chunking"]["medium_window_tokens"] * 1.3
               for c in chunks)


def test_every_group_of_a_split_table_carries_the_header():
    groups = chunking._split_table(_table(120), max_tokens=200)

    assert len(groups) > 1
    for group in groups:
        # Rows without their column names are unreadable, to a model as to a
        # person.
        assert group.startswith("|**Variable**|**Category**|**n**|")
        assert "|---|---|---|" in group


def test_a_split_table_keeps_every_row_exactly_once():
    rows = [f"|variable number {i}|category {i}|{i}|" for i in range(120)]
    groups = chunking._split_table(_table(120), max_tokens=200)

    body = "\n".join(groups)
    for row in rows:
        assert body.count(row) == 1


def test_a_table_that_already_fits_is_left_alone():
    small = _table(3)
    assert chunking._split_long_text(small, max_tokens=1000) == [small]


def test_a_long_paragraph_with_no_line_breaks_is_split_on_sentences():
    prose = " ".join(f"This is sentence number {i} of a very long paragraph." for i in range(200))

    pieces = chunking._split_long_text(prose, max_tokens=100)

    assert len(pieces) > 1
    assert all(chunking._token_count(p) <= 120 for p in pieces)
    # Nothing invented and nothing dropped.
    assert " ".join(pieces).split() == prose.split()


def test_a_paragraph_with_no_sentence_end_is_still_broken_up():
    # A run-on line with no punctuation must not come back as one oversized
    # piece: it would land in the graph as a single unusable chunk.
    prose = " ".join(f"word{i}" for i in range(2000))

    pieces = chunking._split_long_text(prose, max_tokens=100)

    assert len(pieces) > 1


def test_a_table_is_recognised_and_prose_is_not():
    assert chunking._is_table(_table(10)) is True
    assert chunking._is_table("Just a sentence.\nAnd another one.\nAnd a third.") is False
    # Two lines are not enough to call it a table.
    assert chunking._is_table("|a|b|\n|---|---|") is False


# --- stage 1 says what it produced -----------------------------------------


def test_stage_one_names_a_document_that_produced_nothing(caplog):
    # `chunking.py` had no logger at all: a document that yielded no chunk was
    # simply absent from the graph, and nothing in the run said so.
    empty = DocumentRecord(
        doc_id="empty", filename="empty.pdf", page_count=1, markdown_text="",
        page_chunks=_pages(""), sections=[],
    )
    ok = _doc(_pages("## Section\n\nRice husk is used as a substrate."))

    with caplog.at_level("WARNING", logger="kg_pipeline"):
        chunks = chunking.chunk_documents([empty, ok], _CFG)

    assert [c.doc_id for c in chunks] == ["d"]
    assert "empty.pdf produced no chunks at all" in caplog.text
    assert "1 of 2 documents produced no chunks" in caplog.text


def test_stage_one_says_so_when_nothing_was_lost(caplog):
    doc = _doc(_pages("## Section\n\nRice husk is used as a substrate."))

    with caplog.at_level("INFO", logger="kg_pipeline"):
        chunking.chunk_documents([doc], _CFG)

    assert "no document lost" in caplog.text


def test_stage_one_names_a_document_that_kept_only_a_sliver_of_its_words(caplog):
    # The alarm above fires only at zero chunks, so REPORT MATTM entered the
    # graph as 95 words out of 103 650 in silence. Here the pages hold one
    # sentence while the extracted text holds a thousand words.
    pages = _pages("## Section\n\nRice husk is used as a substrate.")
    doc = DocumentRecord(
        doc_id="thin",
        filename="thin.pdf",
        page_count=1,
        markdown_text="Rice husk is used as a substrate. "
        + "circular economy for food " * 200,
        page_chunks=pages,
    )
    doc.sections = _extract_sections(pages)

    with caplog.at_level("WARNING", logger="kg_pipeline"):
        chunks = chunking.chunk_documents([doc], _CFG)

    assert chunks, "the document is thin, not empty: it still produces chunks"
    assert "thin.pdf kept" in caplog.text
    assert "1 of 1 documents kept less than 50 % of their words" in caplog.text


def test_stage_one_reports_the_lowest_coverage_when_no_document_is_thin(caplog):
    doc = _doc(_pages("## Section\n\nRice husk is used as a substrate."))

    with caplog.at_level("INFO", logger="kg_pipeline"):
        chunking.chunk_documents([doc], _CFG)

    assert "no document lost" in caplog.text
    assert "lowest word coverage" in caplog.text


def test_a_document_with_no_text_is_reported_empty_and_not_also_thin(caplog):
    empty = DocumentRecord(
        doc_id="empty", filename="empty.pdf", page_count=1, markdown_text="",
        page_chunks=_pages(""), sections=[],
    )

    with caplog.at_level("WARNING", logger="kg_pipeline"):
        chunking.chunk_documents([empty], _CFG)

    assert "empty.pdf produced no chunks at all" in caplog.text
    assert "kept less than" not in caplog.text
