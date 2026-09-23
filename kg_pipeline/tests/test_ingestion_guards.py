"""Stage 0 must refuse to hand the pipeline a corpus it did not actually read.

Without the guards, `--single-doc typo.pdf` passes every check: the path list
is non-empty, the loop skips the file that is not there, and stage 0 returns
zero documents. Stages 1-4 then run to completion on nothing and the run looks
successful.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz
import pytest

from kg_pipeline.stages.ingestion import ingest_documents


def _write_pdf(path: Path, pages: list[str]) -> None:
    """Write a PDF with one page per text."""
    with fitz.open() as doc:
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        doc.save(path)


def test_a_misspelled_single_doc_is_an_error_not_an_empty_corpus(tmp_path):
    _write_pdf(tmp_path / "real.pdf", ["# Report\n\nCircular economy."])

    with pytest.raises(FileNotFoundError) as excinfo:
        ingest_documents(tmp_path, single_doc="reale.pdf")

    # The message has to name what the operator typed, or they cannot fix it.
    assert "reale.pdf" in str(excinfo.value)


def test_a_correct_single_doc_still_works(tmp_path):
    pytest.importorskip("pymupdf4llm")
    _write_pdf(tmp_path / "real.pdf", ["# Report\n\nCircular economy."])
    _write_pdf(tmp_path / "other.pdf", ["# Other\n\nSomething else."])

    docs = ingest_documents(tmp_path, single_doc="real.pdf")

    assert [d.filename for d in docs] == ["real.pdf"]


def test_a_pdf_with_no_text_layer_is_reported_not_swallowed(tmp_path, caplog):
    pytest.importorskip("pymupdf4llm")
    _write_pdf(tmp_path / "scan.pdf", ["", ""])
    _write_pdf(tmp_path / "real.pdf", ["# Report\n\nCircular economy."])

    with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
        docs = ingest_documents(tmp_path)

    # The scan is still returned — the operator decides what to do about it —
    # but the run says out loud that it will contribute nothing.
    assert len(docs) == 2
    assert "scan.pdf" in caplog.text
    assert "no text" in caplog.text
    assert "1 of 2 documents parsed to no text" in caplog.text


def test_a_corpus_that_yields_nothing_stops_the_run(tmp_path, monkeypatch):
    pytest.importorskip("pymupdf4llm")
    _write_pdf(tmp_path / "a.pdf", ["Some text."])

    # A parser that returns nothing for every file: the records exist but
    # carry no text, which is the case an empty-list check does not catch.
    from kg_pipeline.stages import ingestion

    monkeypatch.setattr(ingestion, "_read_page_chunks", lambda _p: [])

    with pytest.raises(ValueError) as excinfo:
        ingest_documents(tmp_path)

    assert "No readable document" in str(excinfo.value)


def test_a_healthy_corpus_reports_no_loss(tmp_path, caplog):
    pytest.importorskip("pymupdf4llm")
    _write_pdf(tmp_path / "a.pdf", ["# A\n\nFirst document."])
    _write_pdf(tmp_path / "b.pdf", ["# B\n\nSecond document."])

    with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
        docs = ingest_documents(tmp_path)

    assert len(docs) == 2
    assert "parsed to no text" not in caplog.text


def test_a_missing_input_directory_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_documents(tmp_path / "nope")
