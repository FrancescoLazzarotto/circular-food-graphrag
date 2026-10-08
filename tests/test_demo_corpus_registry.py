"""The demo can take its list of documents from the corpus registry.

Without `DEMO_CORPUS_REGISTRY` the demo reads what `DEMO_TEXT_STAGE0_RUNS`
holds, as before. With it, a document the curators excluded is neither
searched nor counted, and a file whose name was damaged on disk is shown under
its name as the author typed it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from kg_pipeline.utils import corpus_registry
from kg_pipeline.utils.corpus_registry import RegistryRow
from product import config

_TEXT = "Testo di una pagina abbastanza lungo da diventare un passaggio. " * 3
_DAMAGED = "Scenari fra sostenibilita╠Ç e innovazione.pdf"


def _doc(doc_id: str, filename: str, title: str = "") -> dict:
    """One stage0 document with one page of text."""
    return {
        "doc_id": doc_id,
        "filename": filename,
        "title": title,
        "markdown_text": _TEXT,
        "page_chunks": [{"page_number": 1, "text": _TEXT}],
    }


@pytest.fixture()
def corpus(tmp_path, monkeypatch) -> Path:
    """A stage0 run of three documents and a registry that excludes one of them."""
    run = tmp_path / "kg_pipeline" / "artifacts" / "run_corpus"
    run.mkdir(parents=True)
    (run / "stage0_documents.json").write_text(
        json.dumps(
            [
                _doc("alfa", "alfa.pdf", "Il titolo vero del documento alfa"),
                _doc("scenari", _DAMAGED),
                _doc("bozza", "bozza.pdf", "Una bozza da non mostrare"),
            ]
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registro.csv"
    corpus_registry.save_registry(
        registry,
        [
            RegistryRow(id_documento="alfa", percorso="Tema/alfa.pdf"),
            RegistryRow(id_documento="scenari", percorso=f"Tema/{_DAMAGED}"),
            RegistryRow(id_documento="bozza", percorso="Tema/bozza.pdf", escluso=True),
        ],
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_ARTIFACTS", tmp_path / "kg_pipeline" / "artifacts")
    monkeypatch.setattr(config, "TEXT_STAGE0_RUNS", "run_corpus")
    monkeypatch.setattr(config, "CATALOG_FILE", catalog)
    monkeypatch.setattr(config, "CORPUS_REGISTRY", "")
    return registry


def test_without_a_registry_the_demo_reads_the_runs_as_before(corpus):
    assert config.registry_documents() is None
    assert config.corpus_manifest()["count"] == 3
    assert config.document_titles() == {
        "alfa.pdf": "Il titolo vero del documento alfa",
        "bozza.pdf": "Una bozza da non mostrare",
    }
    pipeline = config.build_text_pipeline(backend="tfidf")
    assert {c.chunk_id.split("-")[0] for c in pipeline.retriever.chunks} == {
        "alfa",
        "scenari",
        "bozza",
    }


def test_with_a_registry_an_excluded_document_is_neither_searched_nor_counted(
    corpus, monkeypatch
):
    monkeypatch.setattr(config, "CORPUS_REGISTRY", str(corpus))

    pipeline = config.build_text_pipeline(backend="tfidf")

    assert {c.chunk_id.split("-")[0] for c in pipeline.retriever.chunks} == {"alfa", "scenari"}
    manifest = config.corpus_manifest()
    assert manifest["count"] == 2
    assert _DAMAGED in manifest["documents"]


def test_with_a_registry_a_damaged_file_name_is_never_shown(corpus, monkeypatch):
    monkeypatch.setattr(config, "CORPUS_REGISTRY", str(corpus))

    titles = config.document_titles()

    assert titles["alfa.pdf"] == "Il titolo vero del documento alfa"
    assert titles[_DAMAGED] == "Scenari fra sostenibilità e innovazione"


def test_a_registered_document_the_runs_lack_is_reported(corpus, monkeypatch, caplog):
    rows = corpus_registry.load_registry(corpus)
    rows.append(RegistryRow(id_documento="nuovo", percorso="Tema/nuovo.pdf"))
    corpus_registry.save_registry(corpus, rows)
    monkeypatch.setattr(config, "CORPUS_REGISTRY", str(corpus))

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        config.build_text_pipeline(backend="tfidf")

    assert "Tema/nuovo.pdf" in caplog.text


def test_an_unreadable_registry_stops_the_demo(corpus, monkeypatch):
    # Falling back to another list would show a corpus nobody chose.
    monkeypatch.setattr(config, "CORPUS_REGISTRY", str(corpus.with_name("assente.csv")))

    with pytest.raises(FileNotFoundError):
        config.build_text_pipeline(backend="tfidf")


# --- the whole corpus, offered next to the demo's own documents --------------


def test_the_whole_corpus_is_offered_once_its_run_exists(corpus, monkeypatch):
    monkeypatch.setattr(config, "TEXT_STAGE0_RUNS", "run_iniziale")
    monkeypatch.setattr(config, "FULL_CORPUS_RUNS", "run_corpus")
    monkeypatch.setattr(config, "FULL_CORPUS_REGISTRY", str(corpus))

    found = config.collections()

    assert list(found) == ["base", "full"]
    assert (found["full"].runs, found["full"].registry) == ("run_corpus", str(corpus))


@pytest.mark.parametrize(
    ("runs", "registry"),
    [
        ("run_mai_fatto", None),  # update_corpus.py has not been run
        ("run_corpus", "assente.csv"),  # its registry is gone
    ],
)
def test_the_whole_corpus_is_not_offered_without_its_files(corpus, monkeypatch, runs, registry):
    monkeypatch.setattr(config, "TEXT_STAGE0_RUNS", "run_iniziale")
    monkeypatch.setattr(config, "FULL_CORPUS_RUNS", runs)
    monkeypatch.setattr(
        config, "FULL_CORPUS_REGISTRY", str(corpus) if registry is None else str(corpus.with_name(registry))
    )

    assert list(config.collections()) == ["base"]


def test_a_demo_already_on_the_whole_corpus_offers_no_second_copy(corpus, monkeypatch):
    monkeypatch.setattr(config, "FULL_CORPUS_RUNS", "run_corpus")

    assert list(config.collections()) == ["base"]


def test_each_collection_is_searched_counted_and_named_on_its_own(corpus, monkeypatch):
    whole = config.Collection("full", "run_corpus", str(corpus))

    pipeline = config.build_text_pipeline(backend="tfidf", collection=whole)

    assert {c.chunk_id.split("-")[0] for c in pipeline.retriever.chunks} == {"alfa", "scenari"}
    assert config.corpus_manifest(whole)["count"] == 2
    assert config.document_titles(whole)[_DAMAGED] == "Scenari fra sostenibilità e innovazione"
    # The demo's own documents are untouched by the choice.
    assert config.corpus_manifest()["count"] == 3
    assert _DAMAGED not in config.document_titles()


def test_the_wording_follows_the_collection(corpus, monkeypatch):
    corpus_registry.save_registry(
        corpus, [RegistryRow(id_documento="alfa", percorso="Bioeconomia/alfa.pdf", tema="Bioeconomia")]
    )
    whole = config.Collection("full", "run_corpus", str(corpus))
    for name in ("DEMO_PRODUCT_TAGLINE", "DEMO_EXAMPLE_QUESTIONS"):
        monkeypatch.delenv(name, raising=False)

    assert config.collection_topics() == ()
    assert "economia circolare del cibo" in config.product_tagline()
    assert config.collection_topics(whole) == ("Bioeconomia",)
    assert "Bioeconomia" in config.product_tagline(whole)
    assert "Bioeconomia" in config.product_tagline(whole, "en")
    assert config.example_questions(whole) == ("Che cosa dicono i documenti su «Bioeconomia»?",)
    agent_config = config.build_agent_config("hybrid", whole)
    assert agent_config.collection_topics == ("Bioeconomia",)
    assert agent_config.example_questions == config.example_questions(whole)


def test_the_same_runs_written_differently_are_still_one_collection(corpus, monkeypatch):
    monkeypatch.setattr(config, "TEXT_STAGE0_RUNS", "run_corpus")
    monkeypatch.setattr(config, "FULL_CORPUS_RUNS", " run_corpus, ")

    assert list(config.collections()) == ["base"]
