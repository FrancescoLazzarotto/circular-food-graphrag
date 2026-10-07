"""The one-command corpus update changes what the demo reads only when every step passed.

The demo reads the run and the registry this command writes. A step that fails
halfway must leave both as they were, a folder the command did not create must
never be replaced, documents must not leave the demo because the folder was
mistyped, and the reading step must not be able to reach a graph.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from kg_pipeline.utils import corpus_registry
from kg_pipeline.utils.corpus_registry import RegistryRow

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "corpus" / "update_corpus.py"
_spec = importlib.util.spec_from_file_location("update_corpus", _PATH)
update_corpus = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("update_corpus", update_corpus)
_spec.loader.exec_module(update_corpus)


class _Steps:
    """Stands in for the four scripts; records each call and fails the step asked to.

    The registry step adds the row ``nuovo`` and, when ``drop`` names a row,
    marks it excluded as if its file had gone from the folder.
    """

    def __init__(self, fail: str = "", drop: str = "", during=None):
        self.fail = fail
        self.drop = drop
        self.during = during
        self.calls: list[tuple[str, list[str], dict[str, str]]] = []

    def __call__(self, command, env):
        command = list(command)
        joined = " ".join(command)
        step = next(
            name
            for name in ("build_registry", "ocr_scanned", "kg_pipeline.main", "build_text_index")
            if name in joined
        )
        self.calls.append((step, command, env))
        if step == self.fail:
            return 1
        if step == "build_registry":
            path = Path(command[command.index("--registry") + 1])
            rows = corpus_registry.load_registry(path) if path.exists() else []
            if not any(row.id_documento == "nuovo" for row in rows):
                rows.append(RegistryRow(id_documento="nuovo", percorso="Tema/nuovo.pdf", lingua="en"))
            for row in rows:
                if row.id_documento == self.drop:
                    row.escluso = True
                    row.note = "file non trovato nella cartella"
            corpus_registry.save_registry(path, rows)
        elif step == "kg_pipeline.main":
            run_dir = Path(command[command.index("--run-dir") + 1])
            (run_dir / "stage0_documents.json").write_text("[]", encoding="utf-8")
            (run_dir / "pipeline.log").write_text(
                "Stage 0: 1 of 2 documents reused from cache, 1 read\n", encoding="utf-8"
            )
            if self.during:
                self.during()
        return 0

    def steps(self) -> list[str]:
        return [step for step, _, _ in self.calls]


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    registry = tmp_path / "registro.csv"
    if not registry.exists():
        corpus_registry.save_registry(registry, [RegistryRow(id_documento="vecchio", percorso="Tema/vecchio.pdf")])
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"seed": 7, "paths": {"input_dir": "x"}, "neo4j": {"database": "588fe1bc"}}),
        encoding="utf-8",
    )
    (tmp_path / "corpus").mkdir(exist_ok=True)
    values = {
        "corpus_dir": tmp_path / "corpus",
        "registry": registry,
        "run": "run_corpus_demo",
        "ocr_dir": tmp_path / "ocr",
        "stage0_cache": tmp_path / "cache",
        "vector_index_dir": tmp_path / "vettori",
        "pipeline_config": config,
        "gpu": None,
        "allow_removals": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _ids(registry: Path) -> list[str]:
    return [row.id_documento for row in corpus_registry.load_registry(registry)]


def test_a_full_update_replaces_the_run_and_the_registry(tmp_path):
    artifacts = tmp_path / "artifacts"
    args = _args(tmp_path)
    steps = _Steps()

    assert update_corpus.update(args, artifacts=artifacts, run=steps) == 0

    run = artifacts / "run_corpus_demo"
    summary = (run / update_corpus.MARKER).read_text(encoding="utf-8")
    assert "Tema/nuovo.pdf" in summary
    assert "letti ora: 1; dalla cache: 1 di 2" in summary
    assert sorted(_ids(args.registry)) == ["nuovo", "vecchio"]
    assert not (artifacts / "run_corpus_demo.nuovo").exists()
    # No OCR row: the OCR step is not run at all.
    assert steps.steps() == ["build_registry", "kg_pipeline.main", "build_text_index"]


def test_a_second_update_replaces_the_first_run(tmp_path):
    artifacts = tmp_path / "artifacts"
    update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps())
    (artifacts / "run_corpus_demo" / "segno").write_text("primo", encoding="utf-8")

    assert update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps()) == 0

    assert not (artifacts / "run_corpus_demo" / "segno").exists()
    assert not (artifacts / "run_corpus_demo.vecchio").exists()


def test_the_reading_step_cannot_reach_a_graph(tmp_path):
    artifacts = tmp_path / "artifacts"
    args = _args(tmp_path)
    steps = _Steps()

    update_corpus.update(args, artifacts=artifacts, run=steps)

    _, command, env = next(call for call in steps.calls if call[0] == "kg_pipeline.main")
    # Without its own env file the pipeline loads kg_pipeline/.env, which names the live graph.
    env_file = Path(command[command.index("--env-file") + 1])
    assert command[command.index("--stage") + 1] == "ingestion"
    assert env_file.name == "run.env"
    run = artifacts / "run_corpus_demo"
    assert "127.0.0.1:1" in (run / "run.env").read_text(encoding="utf-8")
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert config["neo4j"]["database"] != "588fe1bc"
    assert Path(config["paths"]["input_dir"]).is_absolute()
    assert env["PYTHONHASHSEED"] == "7"


@pytest.mark.parametrize("failing", ["build_registry", "kg_pipeline.main", "build_text_index"])
def test_a_failed_step_leaves_the_run_and_the_registry_as_they_were(tmp_path, failing):
    artifacts = tmp_path / "artifacts"
    update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps())
    run_before = (artifacts / "run_corpus_demo" / update_corpus.MARKER).read_text(encoding="utf-8")
    corpus_registry.save_registry(
        tmp_path / "registro.csv", [RegistryRow(id_documento="vecchio", percorso="Tema/vecchio.pdf")]
    )
    registry_before = (tmp_path / "registro.csv").read_bytes()

    status = update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps(fail=failing))

    assert status == 1
    assert (artifacts / "run_corpus_demo" / update_corpus.MARKER).read_text(encoding="utf-8") == run_before
    assert (tmp_path / "registro.csv").read_bytes() == registry_before


def test_a_retry_after_a_failure_still_names_the_new_documents(tmp_path):
    artifacts = tmp_path / "artifacts"
    update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps(fail="kg_pipeline.main"))

    update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps())

    assert "righe nuove nel registro: 1" in (artifacts / "run_corpus_demo" / update_corpus.MARKER).read_text(
        encoding="utf-8"
    )


def test_documents_leave_the_demo_only_when_allowed(tmp_path):
    artifacts = tmp_path / "artifacts"
    registry_before = _args(tmp_path).registry.read_bytes()
    refused = _Steps(drop="vecchio")

    assert update_corpus.update(_args(tmp_path), artifacts=artifacts, run=refused) == 1
    assert refused.steps() == ["build_registry"]
    assert (tmp_path / "registro.csv").read_bytes() == registry_before

    assert update_corpus.update(_args(tmp_path, allow_removals=True), artifacts=artifacts, run=_Steps(drop="vecchio")) == 0
    summary = (artifacts / "run_corpus_demo" / update_corpus.MARKER).read_text(encoding="utf-8")
    assert "documenti usciti dalla demo: 1" in summary


def test_a_missing_corpus_folder_is_refused_before_any_step(tmp_path):
    steps = _Steps()

    with pytest.raises(SystemExit):
        update_corpus.update(_args(tmp_path, corpus_dir=tmp_path / "manca"), artifacts=tmp_path / "a", run=steps)

    assert steps.calls == []


def test_a_registry_edited_during_the_update_is_not_overwritten(tmp_path):
    args = _args(tmp_path)

    def curator_edits():
        corpus_registry.save_registry(args.registry, [RegistryRow(id_documento="vecchio", percorso="Tema/vecchio.pdf", genere="libro")])

    status = update_corpus.update(args, artifacts=tmp_path / "artifacts", run=_Steps(during=curator_edits))

    assert status == 1
    assert corpus_registry.load_registry(args.registry)[0].genere == "libro"
    assert not (tmp_path / "artifacts" / "run_corpus_demo").exists()


def test_a_folder_the_command_did_not_create_is_never_replaced(tmp_path):
    artifacts = tmp_path / "artifacts"
    (artifacts / "run_corpus_20261005").mkdir(parents=True)
    steps = _Steps()

    with pytest.raises(SystemExit):
        update_corpus.update(_args(tmp_path, run="run_corpus_20261005"), artifacts=artifacts, run=steps)

    assert steps.calls == []
    assert (artifacts / "run_corpus_20261005").is_dir()


def test_a_staging_folder_the_command_did_not_create_is_not_deleted(tmp_path):
    artifacts = tmp_path / "artifacts"
    (artifacts / "run_corpus_demo.nuovo").mkdir(parents=True)

    with pytest.raises(SystemExit):
        update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps())

    assert (artifacts / "run_corpus_demo.nuovo").is_dir()


@pytest.mark.parametrize("name", ["../product", "run_corpus_demo.nuovo", ".nascosta"])
def test_a_run_name_that_is_not_a_plain_folder_name_is_refused(tmp_path, name):
    with pytest.raises(SystemExit):
        update_corpus.update(_args(tmp_path, run=name), artifacts=tmp_path, run=_Steps())


def test_a_swap_interrupted_between_its_renames_is_recovered(tmp_path):
    artifacts = tmp_path / "artifacts"
    update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps())
    (artifacts / "run_corpus_demo").rename(artifacts / "run_corpus_demo.vecchio")

    # Fails at once: what is left is the recovered run, not a missing one.
    update_corpus.update(_args(tmp_path), artifacts=artifacts, run=_Steps(fail="build_registry"))

    assert (artifacts / "run_corpus_demo" / update_corpus.MARKER).is_file()
    assert not (artifacts / "run_corpus_demo.vecchio").exists()


def test_two_updates_of_one_run_do_not_overlap(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with (artifacts / "run_corpus_demo.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        steps = _Steps()
        with pytest.raises(SystemExit):
            update_corpus.update(_args(tmp_path), artifacts=artifacts, run=steps)

    assert steps.calls == []


def _scanned(tmp_path: Path, escluso: bool) -> argparse.Namespace:
    args = _args(tmp_path)
    corpus_registry.save_registry(
        args.registry,
        [RegistryRow(id_documento="scan", percorso="Tema/scan.pdf", impronta="ab" * 32, ocr=True, escluso=escluso)],
    )
    return args


def test_ocr_runs_for_a_scanned_file_without_a_copy(tmp_path):
    steps = _Steps()

    update_corpus.update(_scanned(tmp_path, escluso=False), artifacts=tmp_path / "a", run=steps)

    assert "ocr_scanned" in steps.steps()


def test_ocr_is_skipped_when_the_copy_exists_or_the_file_is_excluded(tmp_path):
    copied, excluded_dir = tmp_path / "copia", tmp_path / "escluso"
    copied.mkdir()
    excluded_dir.mkdir()
    args = _scanned(copied, escluso=False)
    args.ocr_dir.mkdir()
    (args.ocr_dir / f"{'ab' * 32}.pdf").write_bytes(b"%PDF")
    with_copy = _Steps()
    update_corpus.update(args, artifacts=copied / "a", run=with_copy)

    excluded = _Steps()
    update_corpus.update(_scanned(excluded_dir, escluso=True), artifacts=excluded_dir / "a", run=excluded)

    assert "ocr_scanned" not in with_copy.steps()
    assert "ocr_scanned" not in excluded.steps()


def test_the_encoder_stays_off_the_gpu_unless_a_device_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    on_cpu = _Steps()
    update_corpus.update(_args(tmp_path), artifacts=tmp_path / "a", run=on_cpu)
    on_gpu = _Steps()
    update_corpus.update(_args(tmp_path, gpu="1", run="altro"), artifacts=tmp_path / "a", run=on_gpu)

    assert all(env["CUDA_VISIBLE_DEVICES"] == "" for _, _, env in on_cpu.calls)
    assert all(env["CUDA_VISIBLE_DEVICES"] == "1" for _, _, env in on_gpu.calls)
    assert all(env["PYTHONNOUSERSITE"] == "1" for _, _, env in on_cpu.calls + on_gpu.calls)
