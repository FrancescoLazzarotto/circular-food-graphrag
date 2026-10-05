"""Refreshing the corpus registry must never undo what a curator decided.

The registry is rebuilt from the folder whenever files arrive, and the same
file holds columns a person fills in by hand. A refresh that rewrote them would
silently re-include an excluded draft or drop a note. These tests pin which
columns a refresh recomputes, how duplicates are proposed and which copy is
kept, and the repair of folder names damaged by a zip extractor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import fitz

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "corpus" / "build_registry.py"
_spec = importlib.util.spec_from_file_location("build_registry", _PATH)
build_registry = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("build_registry", build_registry)
_spec.loader.exec_module(build_registry)


def _letters(number: int) -> str:
    """Spell a number in letters, so it reads as a word and not as digits."""
    out = ""
    while True:
        number, digit = divmod(number, 26)
        out += chr(97 + digit)
        if not number:
            return out


def _page_text(seed: int, words: int = 60) -> str:
    """A page of distinct made-up words, different for each seed."""
    return " ".join(f"par{_letters(seed * 1000 + i)}" for i in range(words))


def _write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a PDF with one page per text, wrapped so it stays on the page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_textbox(fitz.Rect(36, 36, 560, 800), text, fontsize=8)
        doc.save(path)
    return path


def test_damaged_folder_names_read_as_typed():
    assert build_registry.readable("Che cosa e╠Ç l'economia circolare") == "Che cosa è l'economia circolare"
    assert build_registry.readable("7┬░ RAPPORTO SULLÔÇÖECONOMIA") == "7° RAPPORTO SULL’ECONOMIA"
    # Names that are not damaged come back unchanged.
    assert build_registry.readable("System Dynamics") == "System Dynamics"
    assert build_registry.readable("sostenibilità") == "sostenibilità"


def test_a_slug_already_taken_gets_a_hash_suffix():
    taken: set[str] = set()

    first = build_registry.propose_id("economia circolare.pdf", "a" * 64, taken)
    second = build_registry.propose_id("Economia-circolare.pdf", "b" * 64, taken)

    assert first == "economia_circolare"
    assert second == "economia_circolare_bbbbbbbb"


def test_duplicates_are_proposed_and_the_graph_copy_is_kept(tmp_path):
    corpus = tmp_path / "corpus"
    pages = [_page_text(i) for i in range(5)]
    _write_pdf(corpus / "Tema" / "finale.pdf", pages)
    # The draft holds four of the final's five pages.
    _write_pdf(corpus / "Tema" / "bozza.pdf", pages[:4])
    _write_pdf(corpus / "Tema" / "altro.pdf", [_page_text(i) for i in range(10, 13)])
    graph = {build_registry.corpus_registry.file_digest(corpus / "Tema" / "finale.pdf")}

    rows, _ = build_registry.build(corpus, [], graph, None)

    by_name = {row.filename: row for row in rows}
    assert by_name["finale.pdf"].livello == 2
    assert not by_name["finale.pdf"].escluso
    assert by_name["bozza.pdf"].escluso
    assert by_name["bozza.pdf"].doppione_di == by_name["finale.pdf"].id_documento
    assert not by_name["altro.pdf"].escluso
    assert by_name["finale.pdf"].tema == "Tema"


def test_a_graph_document_inside_a_larger_volume_is_not_dropped(tmp_path):
    corpus = tmp_path / "corpus"
    article = [_page_text(i) for i in range(3)]
    _write_pdf(corpus / "articolo.pdf", article)
    _write_pdf(corpus / "volume.pdf", article + [_page_text(i) for i in range(20, 30)])
    graph = {build_registry.corpus_registry.file_digest(corpus / "articolo.pdf")}

    rows, _ = build_registry.build(corpus, [], graph, None)

    assert not any(row.escluso for row in rows)


def test_a_refresh_keeps_the_curators_columns_and_flags_a_missing_file(tmp_path):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "Tema" / "a.pdf", [_page_text(1)])
    _write_pdf(corpus / "Tema" / "b.pdf", [_page_text(2)])
    rows, _ = build_registry.build(corpus, [], set(), None)
    by_name = {row.filename: row for row in rows}
    by_name["a.pdf"].genere = "rapporto"
    by_name["a.pdf"].priorita = "1"
    by_name["a.pdf"].lingua = "it"
    by_name["a.pdf"].escluso = True
    by_name["a.pdf"].note = "escluso dai colleghi"
    original_id = by_name["a.pdf"].id_documento

    # a.pdf moves to another folder; b.pdf disappears; c.pdf arrives.
    (corpus / "Altro").mkdir()
    (corpus / "Tema" / "a.pdf").rename(corpus / "Altro" / "a.pdf")
    (corpus / "Tema" / "b.pdf").unlink()
    _write_pdf(corpus / "Tema" / "c.pdf", [_page_text(3)])

    refreshed, report = build_registry.build(corpus, rows, set(), None)

    by_name = {row.filename: row for row in refreshed}
    a = by_name["a.pdf"]
    assert (a.id_documento, a.genere, a.priorita, a.lingua, a.escluso, a.note) == (
        original_id,
        "rapporto",
        "1",
        "it",
        True,
        "escluso dai colleghi",
    )
    # What describes the file follows the file.
    assert a.percorso == "Altro/a.pdf" and a.tema == "Altro"
    assert by_name["b.pdf"].escluso and "file non trovato" in by_name["b.pdf"].note
    assert not by_name["c.pdf"].escluso
    assert any("spostato" in line for line in report)


def test_a_file_mostly_without_text_is_marked_for_ocr_and_other_formats_excluded(tmp_path):
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "scansione.pdf", ["", "", _page_text(1)])
    (corpus / "slide.pptx").write_bytes(b"PK\x03\x04")
    (corpus / ".DS_Store").write_bytes(b"\x00")

    rows, _ = build_registry.build(corpus, [], set(), None)

    by_name = {row.filename: row for row in rows}
    assert set(by_name) == {"scansione.pdf", "slide.pptx"}
    assert by_name["scansione.pdf"].ocr and by_name["scansione.pdf"].pagine_senza_testo == 2
    assert by_name["slide.pptx"].escluso
