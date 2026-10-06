"""Stage 0 reads a file once: later runs take its reading from the cache.

Reading the PDFs is the slow part of stage 0, and a corpus that grows by a few
documents at a time must not pay for reading every document again. The cache
is keyed by what decides the text — the file's content and how it is read —
so it may serve a reading only when reading again would give the same one.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from pathlib import Path

import fitz
import pytest

pytest.importorskip("pymupdf4llm")

from kg_pipeline.stages import ingestion  # noqa: E402
from kg_pipeline.utils import corpus_registry  # noqa: E402
from kg_pipeline.utils.corpus_registry import RegistryRow  # noqa: E402


def _write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a PDF with one page per text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        doc.save(path)
    return path


@pytest.fixture()
def reads(monkeypatch) -> list[str]:
    """The names of the files stage 0 actually reads, in order."""
    seen: list[str] = []
    real = ingestion._read_pdf

    def counting(pdf_path: Path, is_ocr_copy: bool):
        seen.append(pdf_path.name)
        return real(pdf_path, is_ocr_copy)

    monkeypatch.setattr(ingestion, "_read_pdf", counting)
    return seen


def test_a_second_run_reads_nothing_and_gives_the_same_records(tmp_path, reads):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["# Primo\n\nTesto del primo.", "Seconda pagina 2021."])
    _write_pdf(corpus / "b.pdf", ["Testo senza titoli."])
    cache = tmp_path / "cache"

    first = ingestion.ingest_documents(corpus, cache_dir=cache)
    second = ingestion.ingest_documents(corpus, cache_dir=cache)

    assert sorted(reads) == ["a.pdf", "b.pdf"]
    assert [d.model_dump() for d in second] == [d.model_dump() for d in first]
    assert [d.model_dump() for d in first] == [
        d.model_dump() for d in ingestion.ingest_documents(corpus)
    ]


def test_adding_a_document_reads_only_that_document(tmp_path, reads):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["Primo."])
    cache = tmp_path / "cache"
    ingestion.ingest_documents(corpus, cache_dir=cache)
    reads.clear()

    _write_pdf(corpus / "b.pdf", ["Secondo."])
    docs = ingestion.ingest_documents(corpus, cache_dir=cache)

    assert reads == ["b.pdf"]
    assert [d.filename for d in docs] == ["a.pdf", "b.pdf"]


def test_a_changed_file_is_read_again(tmp_path, reads):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["Versione uno."])
    cache = tmp_path / "cache"
    ingestion.ingest_documents(corpus, cache_dir=cache)

    _write_pdf(corpus / "a.pdf", ["Versione due."])
    docs = ingestion.ingest_documents(corpus, cache_dir=cache)

    assert reads == ["a.pdf", "a.pdf"]
    assert "Versione due" in docs[0].markdown_text


def test_a_renamed_file_is_not_read_again_but_takes_its_new_name(tmp_path, reads):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "vecchio.pdf", ["Testo senza titoli."])
    cache = tmp_path / "cache"
    ingestion.ingest_documents(corpus, cache_dir=cache)

    (corpus / "vecchio.pdf").rename(corpus / "nuovo.pdf")
    docs = ingestion.ingest_documents(corpus, cache_dir=cache)

    assert reads == ["vecchio.pdf"]
    # The name and the id come from the corpus, not from the cache.
    assert (docs[0].filename, docs[0].doc_id) == ("nuovo.pdf", "nuovo")


def test_a_different_reader_does_not_reuse_the_entry(tmp_path, monkeypatch, reads):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["Testo."])
    cache = tmp_path / "cache"
    ingestion.ingest_documents(corpus, cache_dir=cache)

    def other_reader(pdf_path: Path) -> list:
        """A reader with other options: same file, different text."""
        return [ingestion.PageChunkRecord(page_number=1, text="letto in un altro modo")]

    monkeypatch.setattr(ingestion, "_read_page_chunks", other_reader)
    docs = ingestion.ingest_documents(corpus, cache_dir=cache)

    assert reads == ["a.pdf", "a.pdf"]
    assert docs[0].markdown_text == "letto in un altro modo"


def test_an_unreadable_entry_is_read_again(tmp_path, reads, caplog):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["Testo."])
    cache = tmp_path / "cache"
    ingestion.ingest_documents(corpus, cache_dir=cache)
    for entry in cache.glob("*.json"):
        entry.write_text("{troncato", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
        docs = ingestion.ingest_documents(corpus, cache_dir=cache)

    assert reads == ["a.pdf", "a.pdf"]
    assert "Testo" in docs[0].markdown_text
    assert "unreadable" in caplog.text
    # The entry is rewritten whole, and no temporary file is left behind.
    assert [p.suffix for p in cache.iterdir()] == [".json"]


def test_the_saved_stage_zero_file_is_byte_identical(tmp_path):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["# Caffè · economia\n\nTesto – con “virgolette” e 2021."])
    cache = tmp_path / "cache"
    ingestion.ingest_documents(corpus, cache_dir=cache)

    ingestion.save_documents(tmp_path / "fresh.json", ingestion.ingest_documents(corpus))
    ingestion.save_documents(
        tmp_path / "cached.json", ingestion.ingest_documents(corpus, cache_dir=cache)
    )

    assert (tmp_path / "fresh.json").read_bytes() == (tmp_path / "cached.json").read_bytes()


def test_an_ocr_copy_is_cached_by_its_own_content(tmp_path, reads):
    corpus = tmp_path / "corpus"
    original = _write_pdf(corpus / "scan.pdf", [""])
    row = RegistryRow(
        id_documento="scan",
        percorso="scan.pdf",
        impronta=corpus_registry.file_digest(original),
        ocr=True,
    )
    registry = tmp_path / "registro.csv"
    corpus_registry.save_registry(registry, [row])
    ocr_dir = tmp_path / "ocr"
    copy = _write_pdf(corpus_registry.ocr_copy(ocr_dir, row), ["prima lettura OCR"])
    cache = tmp_path / "cache"

    def run():
        return ingestion.ingest_documents(
            corpus, registry_path=registry, ocr_dir=ocr_dir, cache_dir=cache
        )

    run()
    run()
    # OCR redone: same original, new copy, new reading.
    _write_pdf(copy, ["seconda lettura OCR"])
    docs = run()

    assert reads == [copy.name, copy.name]
    assert "seconda lettura" in docs[0].markdown_text


def test_a_cache_that_cannot_be_written_does_not_stop_stage_zero(tmp_path, caplog):
    if os.geteuid() == 0:
        pytest.skip("permissions do not bind root")
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["Testo."])
    cache = tmp_path / "cache"
    cache.mkdir()
    cache.chmod(0o500)
    try:
        with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
            docs = ingestion.ingest_documents(corpus, cache_dir=cache)
    finally:
        cache.chmod(0o700)

    assert "Testo" in docs[0].markdown_text
    assert "not saved" in caplog.text
    assert list(cache.iterdir()) == []


def _original():
    """A reader, as first written."""

    def reader(path):
        """One wording."""
        return [path]

    return reader


def _reworded():
    """The same reader with another docstring and a comment."""

    def reader(path):
        """Another wording, longer."""
        # A comment.
        return [path]

    return reader


def _changed():
    """The same reader doing something else."""

    def reader(path):
        """One wording."""
        return [path, path]

    return reader


def test_rewording_the_reader_keeps_the_cache_and_changing_its_code_does_not():
    assert ingestion._code_of(_original()) == ingestion._code_of(_reworded())
    assert ingestion._code_of(_original()) != ingestion._code_of(_changed())


def test_the_layout_engine_is_part_of_the_key(monkeypatch):
    # With pymupdf-layout installed, pymupdf4llm lays pages out with another
    # engine, and the same file reads differently.
    real = importlib.metadata.version
    ingestion._reader_signature.cache_clear()
    before = ingestion._reader_signature(ingestion._read_page_chunks)

    def without_layout(package):
        if package == "pymupdf-layout":
            raise importlib.metadata.PackageNotFoundError(package)
        return real(package)

    monkeypatch.setattr(importlib.metadata, "version", without_layout)
    ingestion._reader_signature.cache_clear()
    try:
        assert ingestion._reader_signature(ingestion._read_page_chunks) != before
    finally:
        ingestion._reader_signature.cache_clear()
