from __future__ import annotations

import logging
from pathlib import Path

import fitz
import pytest

import kg_pipeline
from kg_pipeline.models.types import DocumentRecord, PageChunkRecord, SectionRecord
from kg_pipeline.stages.chunking import chunk_documents
from kg_pipeline.stages.ingestion import discover_pdfs, ingest_documents
from kg_pipeline.utils.validation import validate_triples


def test_schema_validation_accepts_valid_triple():
    payload = [
        {
            "subject": "Europe",
            "predicate": "HAS_VALUE",
            "object": "2.7 C",
            "subject_labels": ["Region"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "Europe"},
            "object_properties": {"name": "2.7 C"},
            "relationship_properties": {
                "source_doc": "demo.pdf",
                "extraction_method": "llm",
                "value": 2.7,
                "unit": "C",
                "year": 2025,
            },
        }
    ]
    triples = validate_triples(payload)
    assert len(triples) == 1
    assert triples[0].predicate == "HAS_VALUE"


def test_schema_validation_rejects_bad_predicate():
    payload = [
        {
            "subject": "Europe",
            "predicate": "1_BAD",
            "object": "2.7 C",
            "subject_labels": ["Region"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "Europe"},
            "object_properties": {"name": "2.7 C"},
            "relationship_properties": {
                "source_doc": "demo.pdf",
                "extraction_method": "llm",
            },
        }
    ]
    with pytest.raises(Exception):
        validate_triples(payload)


def test_schema_validation_normalizes_predicate():
    payload = [
        {
            "subject": "Europe",
            "predicate": "has_value",
            "object": "2.7 C",
            "subject_labels": ["Region"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "Europe"},
            "object_properties": {"name": "2.7 C"},
            "relationship_properties": {
                "source_doc": "demo.pdf",
                "extraction_method": "llm",
            },
        }
    ]
    triples = validate_triples(payload)
    assert triples[0].predicate == "HAS_VALUE"


def test_chunking_metadata_fields_present():
    doc = DocumentRecord(
        doc_id="demo_doc",
        filename="demo.pdf",
        page_count=2,
        markdown_text="# Intro\n\nParagraph one.\n\nParagraph two.",
        sections=[SectionRecord(title="Intro", level=1, start_page=1, end_page=2)],
        page_chunks=[
            PageChunkRecord(page_number=1, text="# Intro\n\nParagraph one."),
            PageChunkRecord(page_number=2, text="Paragraph two."),
        ],
        title="Intro",
        publication_year=2025,
    )
    config = {
        "chunking": {
            "small_max_pages": 10,
            "medium_max_pages": 80,
            "small_min_tokens": 1,
            "small_max_tokens": 400,
            "medium_window_tokens": 512,
            "medium_overlap_tokens": 128,
            "large_window_tokens": 1024,
            "large_overlap_tokens": 256,
        }
    }
    chunks = chunk_documents([doc], config)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id
        assert chunk.filename
        assert chunk.chunk_id
        assert chunk.page_range
        assert chunk.section_title


def test_ingestion_reads_pdf(tmp_path: Path):
    pytest.importorskip("pymupdf4llm")

    pdf_path = tmp_path / "mini.pdf"
    with fitz.open() as doc:
        page1 = doc.new_page()
        page1.insert_text((72, 72), "# Test Report\n\nThis is page one.")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "This is page two.")
        doc.save(pdf_path)

    docs = ingest_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].page_count == 2
    assert docs[0].filename == "mini.pdf"


def test_discover_pdfs_takes_uppercase_suffix_and_leaves_subfolders(tmp_path: Path, caplog):
    """A .PDF is a PDF; a PDF in a subfolder is not silently ingested.

    The corpus keeps ``excluded/`` and ``pilot/`` folders next to the documents,
    so a recursive scan would build the graph from a different corpus than the
    one configured. It must warn instead.
    """
    (tmp_path / "flat.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "SHOUTED.PDF").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "excluded").mkdir()
    (tmp_path / "excluded" / "left_out.pdf").write_bytes(b"%PDF-1.4\n")

    with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
        found = discover_pdfs(tmp_path)

    assert [path.name for path in found] == ["SHOUTED.PDF", "flat.pdf"]
    assert "excluded/left_out.pdf" in caplog.text

    assert discover_pdfs(tmp_path, warn=False) == found


def test_chunking_large_doc_with_heading_only_level1_sections():
    """Level-1 sections spanning only their heading page must not drop the
    document body (regression: 303-page report reduced to 10 tiny chunks)."""
    pages = [
        PageChunkRecord(page_number=i, text=f"Body paragraph on page {i}. " * 30)
        for i in range(1, 101)
    ]
    sections = [
        SectionRecord(title=f"Chapter {i}", level=1, start_page=p, end_page=p)
        for i, p in enumerate([1, 40, 70], start=1)
    ] + [
        SectionRecord(title="Sub 1", level=2, start_page=1, end_page=39),
        SectionRecord(title="Sub 2", level=2, start_page=40, end_page=69),
        SectionRecord(title="Sub 3", level=2, start_page=70, end_page=100),
    ]
    doc = DocumentRecord(
        doc_id="big_doc",
        filename="big.pdf",
        page_count=100,
        markdown_text="",
        sections=sections,
        page_chunks=pages,
    )
    config = {
        "chunking": {
            "small_max_pages": 10,
            "medium_max_pages": 80,
            "small_min_tokens": 200,
            "small_max_tokens": 400,
            "medium_window_tokens": 512,
            "medium_overlap_tokens": 128,
            "large_window_tokens": 1024,
            "large_overlap_tokens": 256,
        }
    }
    chunks = chunk_documents([doc], config)
    covered_pages = set()
    for chunk in chunks:
        first, _, last = chunk.page_range.partition("-")
        covered_pages.update(range(int(first), int(last or first) + 1))
    assert len(covered_pages) >= 95, f"only {len(covered_pages)} pages covered"


def test_masked_environment_is_refused_with_the_cure(tmp_path: Path, monkeypatch):
    """The user-site torch must be named as the cause, with the fix."""
    fake_user_site = tmp_path / "user-site"
    (fake_user_site / "torch").mkdir(parents=True)
    monkeypatch.setattr(kg_pipeline.site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(kg_pipeline.site, "getusersitepackages", lambda: str(fake_user_site))

    with pytest.raises(RuntimeError, match="PYTHONNOUSERSITE=1"):
        kg_pipeline._refuse_masked_environment()

    monkeypatch.setattr(kg_pipeline.site, "ENABLE_USER_SITE", False)
    kg_pipeline._refuse_masked_environment()
