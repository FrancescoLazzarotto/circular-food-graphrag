"""The corpus registry decides which files stage 0 reads, and under which id.

The corpus nests documents in theme folders next to duplicates, drafts and
files in other formats. A recursive scan would ingest all of them; the
registry lists each file once, with the reason it is excluded, and gives every
document an id that does not change when two file names reduce to the same
slug. These tests pin the file format curators edit by hand, the checks that
keep two documents from collapsing into one, and how stage 0 follows the
registry.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz
import pytest

from kg_pipeline import main as pipeline_main
from kg_pipeline.stages.ingestion import ingest_documents
from kg_pipeline.utils import corpus_registry
from kg_pipeline.utils.corpus_registry import RegistryRow


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


def _row(corpus: Path, relative: str, doc_id: str, **fields) -> RegistryRow:
    """A registry row for a file under ``corpus``, hashed when the file exists."""
    path = corpus / relative
    digest = corpus_registry.file_digest(path) if path.is_file() else ""
    return RegistryRow(id_documento=doc_id, percorso=relative, impronta=digest, **fields)


# --- the file curators edit ------------------------------------------------


def test_a_saved_registry_reads_back_unchanged(tmp_path):
    rows = [
        RegistryRow(
            id_documento="rapporto",
            percorso="Tema/Sotto/Rapporto; finale.pdf",
            impronta="ab" * 32,
            tema="Tema",
            sottotema="Sotto",
            lingua="it",
            pagine=12,
            pagine_senza_testo=0,
            livello=2,
            note="contiene ; e \" virgolette",
        ),
        RegistryRow(id_documento="bozza", percorso="Tema/bozza.pdf", escluso=True, doppione_di="rapporto"),
    ]
    path = tmp_path / "registro.csv"
    corpus_registry.save_registry(path, rows)

    # A byte-order mark and semicolons: what a spreadsheet in an Italian
    # locale opens correctly with a double click.
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8-sig").splitlines()[0].split(";")[0] == "id_documento"

    loaded = {row.id_documento: row for row in corpus_registry.load_registry(path)}
    assert loaded == {row.id_documento: row for row in rows}


def test_yes_no_cells_accept_what_people_type(tmp_path):
    path = tmp_path / "registro.csv"
    header = ";".join(corpus_registry.COLUMNS)
    lines = [header]
    for doc_id, escluso in (("a", "Sì"), ("b", "si"), ("c", "x"), ("d", "NO"), ("e", "")):
        cells = {c: "" for c in corpus_registry.COLUMNS}
        cells.update(id_documento=doc_id, percorso=f"{doc_id}.pdf", escluso=escluso)
        lines.append(";".join(cells[c] for c in corpus_registry.COLUMNS))
    path.write_text("\n".join(lines), encoding="utf-8")

    flags = {row.id_documento: row.escluso for row in corpus_registry.load_registry(path)}

    assert flags == {"a": True, "b": True, "c": True, "d": False, "e": False}
    # An empty level is the default level, text only.
    assert {row.livello for row in corpus_registry.load_registry(path)} == {1}


def test_a_registry_saved_with_commas_still_reads(tmp_path):
    path = tmp_path / "registro.csv"
    cells = {c: "" for c in corpus_registry.COLUMNS}
    cells.update(id_documento="a", percorso="a.pdf", escluso="no")
    path.write_text(
        ",".join(corpus_registry.COLUMNS) + "\n" + ",".join(cells[c] for c in corpus_registry.COLUMNS),
        encoding="utf-8",
    )

    assert [row.id_documento for row in corpus_registry.load_registry(path)] == ["a"]


def test_an_added_column_is_refused_rather_than_dropped_on_the_next_save(tmp_path):
    path = tmp_path / "registro.csv"
    path.write_text(";".join(corpus_registry.COLUMNS) + ";commento\n", encoding="utf-8")

    with pytest.raises(ValueError, match="commento"):
        corpus_registry.load_registry(path)


def test_an_unreadable_cell_is_reported_with_its_line(tmp_path):
    path = tmp_path / "registro.csv"
    cells = {c: "" for c in corpus_registry.COLUMNS}
    cells.update(id_documento="a", percorso="a.pdf", escluso="forse", livello="3")
    path.write_text(
        ";".join(corpus_registry.COLUMNS) + "\n" + ";".join(cells[c] for c in corpus_registry.COLUMNS),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        corpus_registry.load_registry(path)

    assert "riga 2, colonna escluso" in str(excinfo.value)
    assert "riga 2, colonna livello" in str(excinfo.value)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([RegistryRow("a", "x/a.pdf"), RegistryRow("a", "y/b.pdf")], "ripetuto"),
        ([RegistryRow("a", "a.pdf"), RegistryRow("b", "a.pdf")], "percorso ripetuto"),
        ([RegistryRow("Doc-1", "a.pdf")], "solo lettere minuscole"),
        ([RegistryRow("a", "slide.pptx")], "solo i PDF"),
        ([RegistryRow("a", "x/report.pdf"), RegistryRow("b", "y/report.pdf")], "stesso nome di file"),
        ([RegistryRow("a", "a.pdf", doppione_di="z", escluso=True)], "non è un id"),
        ([RegistryRow("a", "a.pdf", ocr=True)], "senza impronta"),
    ],
)
def test_validation_names_the_problem(rows, expected):
    problems = corpus_registry.validate(rows)

    assert any(expected in problem for problem in problems), problems


def test_excluded_files_may_be_any_format_and_share_a_name():
    rows = [
        RegistryRow("a", "x/report.pdf"),
        RegistryRow("b", "y/report.pdf", escluso=True),
        RegistryRow("c", "slide.pptx", escluso=True),
    ]

    assert corpus_registry.validate(rows) == []


# --- the files stage 0 reads ------------------------------------------------


def test_a_moved_or_missing_file_stops_stage_zero(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    row = RegistryRow("a", "a.pdf", impronta="0" * 64)

    with pytest.raises(FileNotFoundError, match="a.pdf"):
        corpus_registry.documents_to_ingest([row], corpus, None)


def test_a_file_changed_since_the_registry_stops_stage_zero(tmp_path):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["first version"])
    row = _row(corpus, "a.pdf", "a")
    _write_pdf(corpus / "a.pdf", ["second version"])

    with pytest.raises(ValueError, match="changed"):
        corpus_registry.documents_to_ingest([row], corpus, None)


def test_an_ocr_row_reads_its_copy_and_fails_without_it(tmp_path):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "scan.pdf", [""])
    row = _row(corpus, "scan.pdf", "scan", ocr=True)
    ocr_dir = tmp_path / "ocr"

    with pytest.raises(FileNotFoundError, match="OCR copy"):
        corpus_registry.documents_to_ingest([row], corpus, ocr_dir)

    copy = _write_pdf(corpus_registry.ocr_copy(ocr_dir, row), ["recognised text"])
    assert corpus_registry.documents_to_ingest([row], corpus, ocr_dir) == [(copy, row)]


def test_stage_zero_follows_the_registry(tmp_path, caplog):
    pytest.importorskip("pymupdf4llm")
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "Tema A" / "economia circolare.pdf", ["# Primo\n\nTesto del primo."])
    _write_pdf(corpus / "Tema B" / "Economia-circolare.pdf", ["# Secondo\n\nTesto del secondo."])
    _write_pdf(corpus / "Tema B" / "bozza.pdf", ["# Bozza\n\nTesto."])
    _write_pdf(corpus / "Tema B" / "scansione.pdf", [""])
    _write_pdf(corpus / "nuovo.pdf", ["# Nuovo\n\nNon ancora registrato."])
    rows = [
        _row(corpus, "Tema A/economia circolare.pdf", "economia_circolare"),
        # Same slug: the registry gives it its own id.
        _row(corpus, "Tema B/Economia-circolare.pdf", "economia_circolare_2"),
        _row(corpus, "Tema B/bozza.pdf", "bozza", escluso=True, doppione_di="economia_circolare"),
        _row(corpus, "Tema B/scansione.pdf", "scansione", ocr=True),
    ]
    registry = tmp_path / "registro.csv"
    corpus_registry.save_registry(registry, rows)
    ocr_dir = tmp_path / "ocr"
    _write_pdf(corpus_registry.ocr_copy(ocr_dir, rows[3]), ["# Riconosciuto\n\nTesto letto dall'OCR."])

    with caplog.at_level(logging.WARNING, logger="kg_pipeline"):
        docs = ingest_documents(corpus, registry_path=registry, ocr_dir=ocr_dir)

    by_id = {doc.doc_id: doc for doc in docs}
    assert set(by_id) == {"economia_circolare", "economia_circolare_2", "scansione"}
    # The record keeps the original file name, not the OCR copy's.
    assert by_id["scansione"].filename == "scansione.pdf"
    assert "OCR" in by_id["scansione"].markdown_text
    assert by_id["economia_circolare_2"].filename == "Economia-circolare.pdf"
    # A file nobody registered is reported, not read.
    assert "nuovo.pdf" in caplog.text


def test_without_a_registry_two_files_with_one_id_are_refused(tmp_path):
    pytest.importorskip("pymupdf4llm")
    _write_pdf(tmp_path / "economia circolare.pdf", ["# Primo\n\nTesto."])
    _write_pdf(tmp_path / "Economia-circolare.pdf", ["# Secondo\n\nTesto."])

    with pytest.raises(ValueError) as excinfo:
        ingest_documents(tmp_path)

    message = str(excinfo.value)
    assert "economia circolare.pdf" in message and "Economia-circolare.pdf" in message


def test_the_stage_zero_fingerprint_follows_the_registry(tmp_path):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", ["a"])
    _write_pdf(corpus / "b.pdf", ["b"])
    rows = [_row(corpus, "a.pdf", "a"), _row(corpus, "b.pdf", "b")]
    registry = tmp_path / "registro.csv"
    corpus_registry.save_registry(registry, rows)

    first = pipeline_main._corpus_fingerprint(corpus, None, registry)
    assert first == pipeline_main._corpus_fingerprint(corpus, None, registry)
    # Curators editing a note do not invalidate stage 0; excluding a file does.
    rows[0].note = "rivisto"
    corpus_registry.save_registry(registry, rows)
    assert pipeline_main._corpus_fingerprint(corpus, None, registry) == first
    rows[1].escluso = True
    corpus_registry.save_registry(registry, rows)
    assert pipeline_main._corpus_fingerprint(corpus, None, registry) != first
    # And it is not the fingerprint of the same folder read without a registry.
    assert pipeline_main._corpus_fingerprint(corpus, None) != first


def test_clutter_is_not_a_document(tmp_path):
    assert corpus_registry.is_ignored(Path(".DS_Store"))
    assert corpus_registry.is_ignored(Path("Tema/.DS_Store"))
    assert corpus_registry.is_ignored(Path("Tema/~$bozza.docx"))
    assert not corpus_registry.is_ignored(Path("Tema/report.pdf"))




def test_the_invisible_text_of_an_ocr_copy_is_read(tmp_path):
    """OCR puts the recognised text in an invisible layer, which the Markdown renderer skips."""
    pytest.importorskip("pymupdf4llm")
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "scan.pdf", [""])
    row = _row(corpus, "scan.pdf", "scan", ocr=True)
    registry = tmp_path / "registro.csv"
    corpus_registry.save_registry(registry, [row])
    copy = corpus_registry.ocr_copy(tmp_path / "ocr", row)
    copy.parent.mkdir(parents=True)
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "testo riconosciuto dalla scansione", render_mode=3)
        doc.save(copy)

    docs = ingest_documents(corpus, registry_path=registry, ocr_dir=tmp_path / "ocr")

    assert "testo riconosciuto dalla scansione" in docs[0].markdown_text
