from __future__ import annotations

import logging
from pathlib import Path

import fitz
import pytest

import kg_pipeline
import kg_pipeline.main
from kg_pipeline.prompts.extraction_prompt import build_extraction_prompt
from kg_pipeline.models.types import (
    ChunkRecord,
    DocumentRecord,
    PageChunkRecord,
    SectionRecord,
)
from kg_pipeline.stages.chunking import chunk_documents
from kg_pipeline.stages.ingestion import discover_pdfs, ingest_documents
from kg_pipeline.stages.llm_extraction import _name_invented_words
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


def test_post_passes_are_named_when_they_do_not_run(caplog):
    """Stage 6 must not end in silence: a graph without them is not the product."""
    with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
        ran = kg_pipeline.main._run_post_passes(run=False)

    assert ran == []
    for fragment in ("densification", "search_text", "vector index"):
        assert fragment in caplog.text
    assert "scripts/kg/kg_densify.py" in caplog.text


def test_run_post_runs_the_indexes_but_never_densification(monkeypatch):
    """Densification is hours of GPU and a model choice: it stays the operator's."""
    calls: list[str] = []

    class _Result:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1])
        return _Result()

    monkeypatch.setattr(kg_pipeline.main.subprocess, "run", fake_run)
    ran = kg_pipeline.main._run_post_passes(run=True)

    assert calls == ["scripts/kg/kg_search_index.py", "scripts/kg/kg_vector_index.py"]
    assert "densification" not in ran


def test_the_index_pass_is_pointed_at_this_run_not_at_production(monkeypatch):
    """kg_search_index defaults its env file to the one naming the hosted graph
    and loads it with override=True: without forwarding, stage 6 would write
    staging and then rebuild the index on the demo's graph."""
    commands: list[list[str]] = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(
        kg_pipeline.main.subprocess, "run",
        lambda cmd, **kw: (commands.append(cmd), _Result())[1],
    )
    kg_pipeline.main._run_post_passes(
        run=True, config_path=Path("run/config.yaml"), env_file=Path("run/rebuild.env")
    )

    search = next(c for c in commands if c[1].endswith("kg_search_index.py"))
    assert search[2:] == ["--config", "run/config.yaml", "--env-file", "run/rebuild.env"]


def test_a_failing_post_pass_stops_the_run(monkeypatch):
    class _Result:
        returncode = 1

    monkeypatch.setattr(kg_pipeline.main.subprocess, "run", lambda cmd, **kw: _Result())
    with pytest.raises(RuntimeError, match="search_text"):
        kg_pipeline.main._run_post_passes(run=True)


def test_a_translated_proper_name_is_rejected():
    """`nel nome del pane` came back as `Nel name del pane`, with 25 triples on it."""
    source = "Az. Agr. Nel nome del pane di Cappelletti Fabio, Dovadola (FC)."

    assert _name_invented_words("Az. Agr. Nel name del pane", ["Organization"], source) == ["name"]
    assert _name_invented_words("Az. Agr. Nel nome del pane", ["Organization"], source) == []


def test_a_composed_indicator_name_is_left_alone():
    """The prompt asks for a year-scoped Indicator name, so its words may be new."""
    source = "Food waste per capita was 67 kg in Italy."

    assert _name_invented_words("food waste per capita Italy 2022", ["Indicator"], source) == []
    assert _name_invented_words("Totally New Company", ["Organization"], source) == []


def test_the_prompt_asks_to_connect_the_entities_of_the_chunk():
    """Measured on 20 chunks with Qwen3-32B: 318 -> 392 triples, local degree
    1.87 -> 2.02, share of triples hanging off the ten busiest subjects
    52.8% -> 37.0%. It widens the graph instead of thickening its hubs."""
    prompt = build_extraction_prompt(
        ChunkRecord(
            doc_id="d", filename="d.pdf", chunk_id="d_chunk_1", page_range="1-1",
            section_title="s", chunk_index=1, text="Il compostaggio ricicla gli scarti organici.",
        ),
        [],
        ["Concept", "Process"],
        relation_vocab=["RECYCLES"],
    )

    assert "Connect the entities of this passage to each other" in prompt
    assert "prefer reusing an entity you already named" in prompt
