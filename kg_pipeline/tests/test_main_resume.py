"""The orchestrator decides what to recompute.

Every stage in the build passes through `kg_pipeline/main.py`, and what it
decides is one thing repeated six times: whether an artifact on disk stands
in for running the stage again. Stage 3 alone is seven hours, so getting that
wrong in either direction is expensive — recomputing what is already there,
or skipping a stage whose artifact is half-written or was produced from other
inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg_pipeline import main as pipeline_main
from kg_pipeline.models.types import ChunkRecord, DocumentRecord, KGTriple


def _doc(filename: str = "a.pdf") -> DocumentRecord:
    """A document record for `filename`."""
    return DocumentRecord(
        doc_id=filename.removesuffix(".pdf"),
        filename=filename,
        page_count=4,
        markdown_text="text",
    )


def _chunk(chunk_id: str = "c1") -> ChunkRecord:
    """A chunk of document `a.pdf`."""
    return ChunkRecord(
        doc_id="a",
        filename="a.pdf",
        chunk_id=chunk_id,
        page_range="1-2",
        section_title="Intro",
        chunk_index=1,
        text="Rice husk is used as a substrate.",
    )


def _triple() -> KGTriple:
    """One resolved triple."""
    return KGTriple.model_validate(
        {
            "subject": "Rice husk",
            "predicate": "USED_AS",
            "object": "substrate",
            "subject_labels": ["Material"],
            "object_labels": ["Material"],
            "subject_properties": {"name": "Rice husk"},
            "object_properties": {"name": "substrate"},
            "relationship_properties": {"source_doc": "a.pdf"},
        }
    )


@pytest.fixture()
def paths(tmp_path: Path) -> dict[str, Path]:
    """The stage artifact paths under a temporary run directory."""
    return pipeline_main._stage_output_paths(tmp_path)


# --- the artifact names other tools read -----------------------------------


def test_the_artifact_names_are_the_ones_the_analysers_open(tmp_path: Path):
    # The evaluation toolkit and the repair scripts open these by name. A rename
    # here is a silent "no data" downstream, not an error.
    names = {k: v.name for k, v in pipeline_main._stage_output_paths(tmp_path).items()}

    assert names == {
        "documents": "stage0_documents.json",
        "chunks": "stage1_chunks.json",
        "ner": "stage2_ner.json",
        "triples_raw": "stage3_triples_raw.json",
        "acronyms": "stage3_acronyms.json",
        "triples_resolved": "stage4_triples_resolved.json",
        "registry": "stage4_registry.json",
        "merge_cache": "stage4_merge_approved.json",
        "triples_linked": "stage5_triples_linked.json",
        "failed_chunks": "failed_chunks.jsonl",
        "new_labels_log": "new_labels.log",
        "neo4j_summary": "stage6_neo4j_summary.json",
    }
    assert all(p.parent == tmp_path for p in pipeline_main._stage_output_paths(tmp_path).values())


# --- resume: an artifact stands in for the stage ---------------------------


def test_an_existing_document_artifact_is_not_re_ingested(paths, tmp_path, monkeypatch):
    from kg_pipeline.stages import ingestion

    ingestion.save_documents(paths["documents"], [_doc()])
    monkeypatch.setattr(
        ingestion, "ingest_documents", lambda **_: pytest.fail("stage 0 re-ran")
    )

    docs, _ = pipeline_main._load_or_run_documents(
        paths, {"paths": {"input_dir": str(tmp_path)}}, None, tmp_path
    )

    assert [d.filename for d in docs] == ["a.pdf"]


def test_a_missing_document_artifact_is_ingested_and_saved(paths, tmp_path, monkeypatch):
    from kg_pipeline.stages import ingestion

    monkeypatch.setattr(ingestion, "ingest_documents", lambda **_: [_doc("b.pdf")])

    docs, _ = pipeline_main._load_or_run_documents(
        paths, {"paths": {"input_dir": str(tmp_path)}}, None, tmp_path
    )

    assert [d.filename for d in docs] == ["b.pdf"]
    # Saved, so the next invocation resumes instead of re-reading the PDFs.
    assert paths["documents"].exists()
    assert [d.filename for d in ingestion.load_documents(paths["documents"])] == ["b.pdf"]


def test_an_existing_chunk_artifact_is_not_re_chunked(paths, tmp_path, monkeypatch):
    from kg_pipeline.stages import chunking

    chunking.save_chunks(paths["chunks"], [_chunk()])
    monkeypatch.setattr(
        chunking, "chunk_documents", lambda *a, **k: pytest.fail("stage 1 re-ran")
    )

    chunks, _ = pipeline_main._load_or_run_chunks(paths, {}, [_doc()], tmp_path, "up")
    assert [c.chunk_id for c in chunks] == ["c1"]


def test_a_missing_chunk_artifact_is_chunked_and_saved(paths, tmp_path, monkeypatch):
    from kg_pipeline.stages import chunking

    monkeypatch.setattr(chunking, "chunk_documents", lambda *a, **k: [_chunk("c7")])

    chunks, _ = pipeline_main._load_or_run_chunks(paths, {}, [_doc()], tmp_path, "up")

    assert [c.chunk_id for c in chunks] == ["c7"]
    assert paths["chunks"].exists()


def test_stage_three_resumes_only_when_both_its_artifacts_are_there(paths, tmp_path, monkeypatch):
    # Triples and acronyms are written by two separate calls. A crash between
    # them leaves triples without acronyms, and that is not a finished stage.
    from kg_pipeline.stages import llm_extraction

    llm_extraction.save_triples(paths["triples_raw"], [_triple()])
    ran = {"n": 0}

    def _extract(**_kwargs):
        ran["n"] += 1
        return [_triple()], {"RH": "Rice husk"}

    monkeypatch.setattr(llm_extraction, "extract_triples", _extract)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:9/v1")

    config = {
        "llm": {
            "temperature": 0.0,
            "max_retries_per_chunk": 1,
            "use_structured_output": True,
            "checkpoint_every": 0,
        },
        "ontology": {"labels": ["Material"]},
    }
    pipeline_main._load_or_run_raw_triples(
        paths, config, [_chunk()], {}, seed=42, relation_vocab=None,
        run_dir=tmp_path, upstream="up",
    )

    assert ran["n"] == 1, "a half-written stage 3 must be redone, not resumed"
    assert paths["acronyms"].exists()

    # With both present, the stage is skipped.
    monkeypatch.setattr(
        llm_extraction, "extract_triples", lambda **_: pytest.fail("stage 3 re-ran")
    )
    triples, acronyms, _ = pipeline_main._load_or_run_raw_triples(
        paths, config, [_chunk()], {}, seed=42, relation_vocab=None,
        run_dir=tmp_path, upstream="up",
    )
    assert [t.object for t in triples] == ["substrate"]
    assert acronyms == {"RH": "Rice husk"}


def test_an_existing_linked_artifact_is_not_re_linked(paths, monkeypatch):
    from kg_pipeline.stages import linking

    linking.save_triples(paths["triples_linked"], [_triple()])
    monkeypatch.setattr(
        linking, "add_cross_document_links", lambda **_: pytest.fail("stage 5 re-ran")
    )

    out = pipeline_main._load_or_run_linking(paths, [], {}, [_doc()], {})

    assert [t.predicate for t in out] == ["USED_AS"]


def test_the_document_edge_switch_reaches_stage_five(paths):
    # `include_mentioned_in: false` is a deliberate setting for this corpus. If
    # the wiring breaks, the graph quietly grows a :Document edge per mention
    # and nothing says so.
    out = pipeline_main._load_or_run_linking(
        paths, [_triple()], {}, [_doc()], {"linking": {"include_mentioned_in": False}}
    )
    assert [t.predicate for t in out] == ["USED_AS"]

    paths["triples_linked"].unlink()
    out = pipeline_main._load_or_run_linking(
        paths, [_triple()], {}, [_doc()], {"linking": {"include_mentioned_in": True}}
    )
    assert "MENTIONED_IN" in [t.predicate for t in out]

    paths["triples_linked"].unlink()
    # An absent key keeps the default, which writes the edges.
    out = pipeline_main._load_or_run_linking(paths, [_triple()], {}, [_doc()], {})
    assert "MENTIONED_IN" in [t.predicate for t in out]


# --- the relation vocabulary ------------------------------------------------


def test_a_relative_vocab_path_resolves_next_to_the_config(tmp_path: Path):
    (tmp_path / "vocab.json").write_text(json.dumps(["used_as", " contains "]))
    config_path = tmp_path / "config.yaml"

    vocab = pipeline_main._load_relation_vocab(
        {"llm": {"relation_vocab_path": "vocab.json"}}, config_path
    )

    # Upper-cased and trimmed, because that is what the extractor compares against.
    assert vocab == ["USED_AS", "CONTAINS"]


def test_no_vocab_configured_is_not_an_error(tmp_path: Path):
    assert pipeline_main._load_relation_vocab({}, tmp_path / "config.yaml") is None


def test_a_configured_vocab_that_is_missing_stops_the_run(tmp_path: Path):
    # Silently extracting with no vocabulary would remap every predicate to
    # RELATED_TO and look like a successful build.
    with pytest.raises(FileNotFoundError):
        pipeline_main._load_relation_vocab(
            {"llm": {"relation_vocab_path": "gone.json"}}, tmp_path / "config.yaml"
        )


def test_a_vocab_that_is_not_a_list_stops_the_run(tmp_path: Path):
    (tmp_path / "vocab.json").write_text(json.dumps({"USED_AS": 1}))
    with pytest.raises(ValueError):
        pipeline_main._load_relation_vocab(
            {"llm": {"relation_vocab_path": "vocab.json"}}, tmp_path / "config.yaml"
        )


# --- run metadata -----------------------------------------------------------


def test_the_run_snapshots_the_config_it_ran_with(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    (config_dir / "vocab.json").write_text(json.dumps(["USED_AS"]), encoding="utf-8")

    pipeline_main._write_run_metadata(
        run_dir,
        config_path,
        {"llm": {"relation_vocab_path": "vocab.json"}, "gliner": {"model_name": "g"}},
        seed=42,
    )

    # The config and the vocabulary travel with the run: without them the
    # artifacts cannot be traced back to the settings that made them.
    assert (run_dir / "config.yaml").read_text(encoding="utf-8") == "seed: 42\n"
    assert json.loads((run_dir / "vocab.json").read_text(encoding="utf-8")) == ["USED_AS"]

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert metadata["config_path"] == str(config_path.resolve())
    assert metadata["gliner_model"] == "g"


# --- the fingerprint that says an artifact still means what it meant ---------
#
# A file existing does not say it still fits: re-running stage 1 with a
# different window would leave stage 3 resuming against chunk indices that no
# longer mean the same thing, and the run would report success. The
# fingerprint ties a stage's output to the settings and the upstream artifacts
# that produced it.


_CHUNK_CFG_A = {"chunking": {"medium_window_tokens": 512}}
_CHUNK_CFG_B = {"chunking": {"medium_window_tokens": 1024}}


def _run_chunks(paths, tmp_path, config, upstream="up"):
    """Resume or run stage 1 with `config` and an upstream fingerprint."""
    return pipeline_main._load_or_run_chunks(paths, config, [_doc()], tmp_path, upstream)


def test_the_same_settings_resume_normally(paths, tmp_path, monkeypatch):
    from kg_pipeline.stages import chunking

    monkeypatch.setattr(chunking, "chunk_documents", lambda *a, **k: [_chunk("c1")])
    _run_chunks(paths, tmp_path, _CHUNK_CFG_A)

    monkeypatch.setattr(
        chunking, "chunk_documents", lambda *a, **k: pytest.fail("stage 1 re-ran")
    )
    chunks, _ = _run_chunks(paths, tmp_path, _CHUNK_CFG_A)

    assert [c.chunk_id for c in chunks] == ["c1"]


def test_changed_settings_stop_the_run_instead_of_reusing_the_artifact(
    paths, tmp_path, monkeypatch
):
    from kg_pipeline.stages import chunking

    monkeypatch.setattr(chunking, "chunk_documents", lambda *a, **k: [_chunk("c1")])
    _run_chunks(paths, tmp_path, _CHUNK_CFG_A)

    with pytest.raises(SystemExit) as exit_info:
        _run_chunks(paths, tmp_path, _CHUNK_CFG_B)

    message = str(exit_info.value)
    # Silently redoing seven hours is as surprising as silently skipping it, so
    # the message has to say what to do rather than pick for the operator.
    assert "stage1_chunks.json" in message
    assert "delete" in message and "--run-dir" in message


def test_a_changed_corpus_invalidates_everything_downstream(paths, tmp_path, monkeypatch):
    from kg_pipeline.stages import chunking

    monkeypatch.setattr(chunking, "chunk_documents", lambda *a, **k: [_chunk("c1")])
    _run_chunks(paths, tmp_path, _CHUNK_CFG_A, upstream="corpus-v1")

    # The stage's own settings are unchanged; its input is not.
    with pytest.raises(SystemExit):
        _run_chunks(paths, tmp_path, _CHUNK_CFG_A, upstream="corpus-v2")


def test_the_corpus_fingerprint_follows_the_files(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"one")
    first = pipeline_main._corpus_fingerprint(tmp_path, None)

    (tmp_path / "b.pdf").write_bytes(b"two")
    assert pipeline_main._corpus_fingerprint(tmp_path, None) != first

    (tmp_path / "b.pdf").unlink()
    assert pipeline_main._corpus_fingerprint(tmp_path, None) == first

    # A different --single-doc is a different corpus.
    assert pipeline_main._corpus_fingerprint(tmp_path, "a.pdf") != first


def test_an_artifact_from_before_fingerprints_is_taken_once_and_stamped(
    paths, tmp_path, monkeypatch, caplog
):
    from kg_pipeline.stages import chunking

    chunking.save_chunks(paths["chunks"], [_chunk("old")])
    monkeypatch.setattr(
        chunking, "chunk_documents", lambda *a, **k: pytest.fail("stage 1 re-ran")
    )

    with caplog.at_level("INFO", logger="kg_pipeline"):
        chunks, _ = _run_chunks(paths, tmp_path, _CHUNK_CFG_A)

    assert [c.chunk_id for c in chunks] == ["old"]
    assert "no recorded fingerprint" in caplog.text
    # Stamped, so the next run is checked like any other.
    assert "chunks" in pipeline_main._load_fingerprints(tmp_path)


def test_an_operational_knob_does_not_invalidate_seven_hours_of_work():
    # `checkpoint_every` and `batch_size` change how the work is done, not what
    # comes out of it.
    base = {"llm": {"temperature": 0.0, "use_structured_output": True, "checkpoint_every": 50},
            "ontology": {"labels": ["Material"]}}
    other = {"llm": {"temperature": 0.0, "use_structured_output": True, "checkpoint_every": 10},
             "ontology": {"labels": ["Material"]}}

    inputs = pipeline_main._STAGE_INPUTS["triples_raw"]
    assert inputs(base) == inputs(other)

    warmer = {"llm": {"temperature": 0.7, "use_structured_output": True, "checkpoint_every": 50},
              "ontology": {"labels": ["Material"]}}
    assert inputs(base) != inputs(warmer)


# --- run metadata -----------------------------------------------------------


def test_a_resume_does_not_rewrite_when_the_run_started(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")

    pipeline_main._write_run_metadata(run_dir, config_path, {}, seed=42)
    first = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    pipeline_main._write_run_metadata(run_dir, config_path, {}, seed=42)
    second = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    # Overwritten on every invocation, the date would make a resumed run claim
    # to have started after its own extraction finished.
    assert second["started_at"] == first["started_at"]
    assert second["invocations"] == 2
    assert second["last_run_at"] >= first["last_run_at"]
