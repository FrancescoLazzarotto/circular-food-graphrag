"""Passage ids of the text index name the document, not its position in a run.

Ids are keys: the dense index maps them back to stored vectors and the
evidence list deduplicates on them. An id built from a document's position
repeats across runs — the second document of one run and the second of the
next share it — and changes when a run gains or loses a document.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from graphrag import cli


def _doc(doc_id: str, filename: str, pages: list[str]) -> dict:
    """One stage0 document with the given page texts."""
    return {
        "doc_id": doc_id,
        "filename": filename,
        "markdown_text": "\n\n".join(pages),
        "page_chunks": [{"page_number": n, "text": text} for n, text in enumerate(pages, start=1)],
    }


def _write_run(root: Path, name: str, docs: list[dict]) -> None:
    """A run directory holding only a stage0 artifact."""
    run = root / "kg_pipeline" / "artifacts" / name
    run.mkdir(parents=True)
    (run / "stage0_documents.json").write_text(json.dumps(docs), encoding="utf-8")


def _index(runs: str) -> dict[str, str]:
    """``{chunk_id: source}`` of the lexical index built from ``runs``."""
    args = argparse.Namespace(
        text_retriever_backend="tfidf", text_docs_dir="", text_stage0_runs=runs
    )
    pipeline = cli._build_text_pipeline(args)
    return {chunk.chunk_id: chunk.source for chunk in pipeline.retriever.chunks}


_TEXT = "Testo di una pagina abbastanza lungo da diventare un passaggio. " * 3


def test_ids_are_unique_across_runs_and_do_not_depend_on_run_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, "run_a", [_doc("alfa", "alfa.pdf", [_TEXT, _TEXT])])
    _write_run(tmp_path, "run_b", [_doc("beta", "beta.pdf", [_TEXT])])

    one_way = _index("run_a,run_b")
    other_way = _index("run_b,run_a")

    assert one_way == other_way
    assert set(one_way) == {"alfa-p1-c0001", "alfa-p2-c0001", "beta-p1-c0001"}


def test_a_document_keeps_its_ids_when_the_run_gains_a_document(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, "run_old", [_doc("beta", "beta.pdf", [_TEXT])])
    _write_run(
        tmp_path, "run_new", [_doc("alfa", "alfa.pdf", [_TEXT]), _doc("beta", "beta.pdf", [_TEXT])]
    )

    before = _index("run_old")
    after = _index("run_new")

    assert before.items() <= after.items()


def test_two_files_with_one_doc_id_are_both_indexed_under_distinct_ids(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, "run_a", [_doc("economia_circolare", "economia circolare.pdf", [_TEXT])])
    _write_run(tmp_path, "run_b", [_doc("economia_circolare", "Economia-circolare.pdf", [_TEXT])])

    with caplog.at_level(logging.WARNING, logger="graphrag.cli"):
        index = _index("run_a,run_b")

    assert sorted(index.values()) == [
        "Economia-circolare.pdf#page=1#chunk=1",
        "economia circolare.pdf#page=1#chunk=1",
    ]
    assert len(set(index)) == 2
    assert "economia_circolare-p1-c0001" in index
    assert "Economia-circolare.pdf" in caplog.text
    # Stable: the second file gets the same id on every build.
    assert _index("run_a,run_b") == index


def test_a_document_without_pages_or_id_still_gets_unique_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    no_pages = {"doc_id": "alfa", "filename": "alfa.pdf", "markdown_text": _TEXT * 8}
    no_id = {"filename": "beta.pdf", "markdown_text": _TEXT}
    _write_run(tmp_path, "run_a", [no_pages, no_id])

    index = _index("run_a")

    assert {"alfa-c0001", "alfa-c0002", "beta.pdf-c0001"} <= set(index)
    assert index["alfa-c0001"] == "alfa.pdf"


# --- unreadable text layers --------------------------------------------------

# What a font without a Unicode map gives stage 0: control characters where
# the letters should be.
_UNREADABLE = "".join(chr(c) for c in range(2, 20) if chr(c) not in "\t\n\r") * 12


def test_unreadable_passages_are_left_out_and_the_others_keep_their_ids(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, "run_a", [_doc("alfa", "alfa.pdf", [_TEXT, _UNREADABLE, _TEXT])])

    with caplog.at_level(logging.WARNING, logger="graphrag.cli"):
        index = _index("run_a")

    assert set(index) == {"alfa-p1-c0001", "alfa-p3-c0001"}
    assert "alfa.pdf: 1 of 3 passages" in caplog.text


def test_a_wholly_unreadable_document_gives_no_passages(tmp_path, monkeypatch):
    # Not even from its markdown text, which is the same characters.
    monkeypatch.chdir(tmp_path)
    _write_run(
        tmp_path,
        "run_a",
        [_doc("alfa", "alfa.pdf", [_UNREADABLE]), _doc("beta", "beta.pdf", [_TEXT])],
    )

    assert set(_index("run_a")) == {"beta-p1-c0001"}


def test_a_table_of_symbols_is_still_read(tmp_path, monkeypatch, caplog):
    # Symbol fonts put ticks on private-use code points: a passage can hold a
    # tenth of them and be a perfectly readable table.
    monkeypatch.chdir(tmp_path)
    table = "|Paris (France)|\uf0fc|\uf0fc|\uf0fc|\uf0fc|Circular plan|\n" * 6
    _write_run(tmp_path, "run_a", [_doc("alfa", "alfa.pdf", [table])])

    with caplog.at_level(logging.WARNING, logger="graphrag.cli"):
        index = _index("run_a")

    assert set(index) == {"alfa-p1-c0001"}
    assert "unreadable" not in caplog.text
