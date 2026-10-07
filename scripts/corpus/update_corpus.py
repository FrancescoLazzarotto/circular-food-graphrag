#!/usr/bin/env python3
"""Bring the documents the demo searches up to date with the corpus folder, in one command.

Four steps, each with its own script, run in order: refresh the registry from
the folder (``build_registry.py``), make the OCR copies of new scanned files
(``ocr_scanned.py``), read the included documents (stage 0 of the pipeline),
and encode their new passages into the text index, checking that every
included document has passages (``build_text_index.py``).

Every step works on a new folder next to the demo's run, with its own copy of
the registry. The run and the registry are replaced only when every step has
passed, so a failure leaves what the demo reads as it was. A document whose
file has gone from the folder would leave the demo; that happens only with
``--allow-removals``, because a mistyped or unmounted folder looks the same.
Readings and passage vectors are cached by content: only new files are read
and only new passages encoded. The encoder runs on the CPU unless ``--gpu``
names a device. Nothing is written to any graph: stage 0 runs with an
unreachable Neo4j address.

    PYTHONNOUSERSITE=1 python scripts/corpus/update_corpus.py --corpus-dir "<corpus folder>"

Then restart the demo with ``DEMO_CORPUS_REGISTRY`` set to the registry and
``DEMO_TEXT_STAGE0_RUNS`` to the run.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
# The kg_pipeline package lives at the repo root and is not pip-installed.
sys.path.insert(0, str(ROOT))

from kg_pipeline.utils import corpus_registry  # noqa: E402
from kg_pipeline.utils.corpus_registry import RegistryRow  # noqa: E402

ARTIFACTS = ROOT / "kg_pipeline" / "artifacts"
SCRIPTS = Path(__file__).resolve().parent
# Written into every folder this command makes, from the moment it makes it;
# a folder without it is someone else's and is never replaced or deleted.
MARKER = "AGGIORNAMENTO.txt"
# Stage 0 needs no graph; an address nothing listens on makes certain that a
# mistake in the pipeline cannot reach one.
_NO_GRAPH_ENV = "NEO4J_URL=bolt://127.0.0.1:1\nNEO4J_URI=bolt://127.0.0.1:1\n"
_STAGE0_LOG = re.compile(
    r"Stage 0: (\d+) of (\d+) documents reused from .*?, (\d+) read"
)

Runner = Callable[[Sequence[str], dict[str, str]], int]


def _run(command: Sequence[str], env: dict[str, str]) -> int:
    """Run one step from the repository root and return its exit status."""
    return subprocess.run(list(command), cwd=ROOT, env=env, check=False).returncode


def stage0_config(
    base: dict[str, Any],
    corpus_dir: Path,
    registry: Path,
    ocr_dir: Path,
    cache_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """The pipeline configuration that reads this corpus into ``run_dir``.

    Args:
        base: The pipeline's own configuration, left unchanged.
        corpus_dir: Corpus folder.
        registry: Corpus registry.
        ocr_dir: Folder of the OCR copies.
        cache_dir: Stage 0 reading cache.
        run_dir: Run folder the readings go to.

    Returns:
        A copy of ``base`` with absolute paths and no graph database named.
    """
    config = copy.deepcopy(base)
    config.setdefault("paths", {}).update(
        {
            "input_dir": str(corpus_dir.resolve()),
            "output_dir": str(run_dir.resolve()),
            "registry": str(registry.resolve()),
            "ocr_dir": str(ocr_dir.resolve()),
            "stage0_cache": str(cache_dir.resolve()),
        }
    )
    # The base names the production database; stage 0 must not carry it.
    config.setdefault("neo4j", {})["database"] = "non_usato_stadio_0"
    return config


def new_rows(before: list[RegistryRow], after: list[RegistryRow]) -> list[RegistryRow]:
    """Rows of ``after`` whose id ``before`` does not have."""
    known = {row.id_documento for row in before}
    return [row for row in after if row.id_documento not in known]


def newly_excluded(
    before: list[RegistryRow], after: list[RegistryRow]
) -> list[RegistryRow]:
    """Rows included in ``before`` and excluded in ``after``: documents leaving the demo."""
    included = {row.id_documento for row in before if not row.escluso}
    return [row for row in after if row.escluso and row.id_documento in included]


def _describe(row: RegistryRow) -> str:
    """One line on a registry row, saying whether it is searched and why not."""
    state = "ESCLUSO" if row.escluso else "incluso"
    note = f" — {row.note}" if row.note else ""
    return f"  {state}: {row.percorso} (id {row.id_documento}, lingua {row.lingua or '?'}){note}"


def _digest(path: Path) -> str:
    """SHA-256 of a file's bytes; empty when it does not exist."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _remove_ours(folder: Path) -> None:
    """Delete a folder this command made.

    Args:
        folder: The folder; nothing happens when it does not exist.

    Raises:
        SystemExit: If the folder exists without the marker.
    """
    if not folder.exists():
        return
    if not (folder / MARKER).is_file():
        raise SystemExit(
            f"{folder} non è stata creata da questo comando: non la tocco."
        )
    shutil.rmtree(folder)


def _replace_run(staged: Path, target: Path) -> None:
    """Put ``staged`` in place of ``target``, keeping the old run until the new one is in place.

    Args:
        staged: The new run.
        target: The run the demo reads; may not exist yet.
    """
    old = target.with_name(target.name + ".vecchio")
    _remove_ours(old)
    if target.exists():
        target.rename(old)
    staged.rename(target)
    _remove_ours(old)


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive lock: two updates of one run would delete each other's folders.

    Args:
        lock_path: Lock file, one per run.

    Raises:
        SystemExit: If another update holds the lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"Un altro aggiornamento di {lock_path.stem} è in corso."
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def update(
    args: argparse.Namespace, artifacts: Path = ARTIFACTS, run: Runner = _run
) -> int:
    """Run the four steps; replace the demo's run and registry only when all of them pass.

    Args:
        args: Parsed command line.
        artifacts: Folder the pipeline runs live in.
        run: Runs one step and returns its exit status.

    Returns:
        The exit status: 0 when the run and the registry were replaced, 1 when
        a step failed or documents would leave the demo without
        ``--allow-removals``.

    Raises:
        SystemExit: If ``--run`` is not a plain folder name or names a folder
            this command did not create, if the corpus folder does not exist,
            or if another update of the same run is in progress.
    """
    # No dots: "<run>.nuovo", "<run>.vecchio" and "<run>.lock" belong to the command.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        raise SystemExit(
            f"--run deve essere un nome di cartella semplice, non {args.run!r}"
        )
    # Every step runs from the repository root: a relative path given from
    # elsewhere would point at another folder there.
    corpus_dir, registry, ocr_dir = (
        args.corpus_dir.resolve(),
        args.registry.resolve(),
        args.ocr_dir.resolve(),
    )
    if not corpus_dir.is_dir():
        raise SystemExit(f"Cartella del corpus non trovata: {corpus_dir}")
    target = artifacts / args.run
    staged = artifacts / f"{args.run}.nuovo"
    old = artifacts / f"{args.run}.vecchio"

    with _exclusive(artifacts / f"{args.run}.lock"):
        if not target.exists() and (old / MARKER).is_file():
            # An earlier update stopped between the two renames of the swap.
            old.rename(target)
        if target.exists() and not (target / MARKER).is_file():
            raise SystemExit(
                f"{target} esiste e non è stata creata da questo comando: "
                "non la sostituisco. Scegliere un altro --run."
            )
        _remove_ours(staged)
        staged.mkdir(parents=True)
        (staged / MARKER).write_text("aggiornamento in corso\n", encoding="utf-8")
        return _update_in(args, corpus_dir, registry, ocr_dir, staged, target, run)


def _update_in(
    args: argparse.Namespace,
    corpus_dir: Path,
    registry: Path,
    ocr_dir: Path,
    staged: Path,
    target: Path,
    run: Runner,
) -> int:
    """The four steps inside ``staged``, then the swap; see :func:`update`."""
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env["CUDA_VISIBLE_DEVICES"] = args.gpu or ""
    python = sys.executable
    unchanged = f"la demo legge ancora {target} e il registro non è cambiato"

    # The steps change a copy: the registry the demo and the curators use is
    # replaced only at the end, with the run.
    work = staged / "registry.csv"
    registry_at_start = _digest(registry)
    if registry.exists():
        shutil.copyfile(registry, work)
    before = corpus_registry.load_registry(work) if work.exists() else []

    print("== 1/4 registro", flush=True)
    files = [
        "--corpus-dir",
        str(corpus_dir),
        "--registry",
        str(work),
        "--ocr-dir",
        str(ocr_dir),
    ]
    if run([python, str(SCRIPTS / "build_registry.py"), *files], env):
        print(f"ERRORE nell'aggiornamento del registro: {unchanged}.")
        return 1
    after = corpus_registry.load_registry(work)
    added = new_rows(before, after)
    removed = newly_excluded(before, after)
    if removed and not args.allow_removals:
        print(
            "\n".join(
                [
                    f"{len(removed)} documenti oggi inclusi uscirebbero dalla demo:",
                    *(_describe(row) for row in removed[:20]),
                    "La cartella del corpus è quella giusta? Se i file sono stati tolti "
                    "apposta, rilanciare con --allow-removals.",
                    f"Niente è stato cambiato: {unchanged}.",
                ]
            )
        )
        return 1

    print("== 2/4 copie OCR dei PDF scansionati", flush=True)
    missing_ocr = [
        row
        for row in after
        if row.ocr
        and not row.escluso
        and not corpus_registry.ocr_copy(ocr_dir, row).is_file()
    ]
    if missing_ocr and run([python, str(SCRIPTS / "ocr_scanned.py"), *files], env):
        print(f"ERRORE nell'OCR: {unchanged}.")
        return 1

    print("== 3/4 lettura dei documenti (stadio 0)", flush=True)
    base = yaml.safe_load(args.pipeline_config.read_text(encoding="utf-8"))
    config = stage0_config(base, corpus_dir, work, ocr_dir, args.stage0_cache, staged)
    (staged / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (staged / "run.env").write_text(_NO_GRAPH_ENV, encoding="utf-8")
    pipeline_env = dict(env)
    if "seed" in base:
        # The pipeline's seed only fixes set iteration when the interpreter
        # starts with it.
        pipeline_env["PYTHONHASHSEED"] = str(base["seed"])
    stage0 = [
        python,
        "-m",
        "kg_pipeline.main",
        "--config",
        str(staged / "config.yaml"),
        "--env-file",
        str(staged / "run.env"),
        "--run-dir",
        str(staged),
        "--stage",
        "ingestion",
    ]
    if run(stage0, pipeline_env):
        print(f"ERRORE nella lettura: {unchanged}; log in {staged}.")
        return 1

    print("== 4/4 indice del testo e controllo di copertura", flush=True)
    index = [
        python,
        str(SCRIPTS / "build_text_index.py"),
        "--stage0-runs",
        staged.name,
        "--registry",
        str(work),
        "--vector-index-dir",
        str(args.vector_index_dir.resolve()),
        "--report",
        str(staged / "copertura.json"),
    ]
    if run(index, env):
        print(
            f"ERRORE nell'indice o nella copertura: {unchanged}; "
            f"vedere {staged / 'copertura.json'}."
        )
        return 1

    if _digest(registry) != registry_at_start:
        print(
            f"Il registro {registry} è stato modificato durante l'aggiornamento: "
            f"non lo sovrascrivo. Rilanciare il comando; {unchanged}."
        )
        return 1

    log_file = staged / "pipeline.log"
    log = (
        log_file.read_text(encoding="utf-8", errors="replace")
        if log_file.exists()
        else ""
    )
    reading = _STAGE0_LOG.findall(log)
    reused, total, read = reading[-1] if reading else ("?", "?", "?")
    included = [row for row in after if not row.escluso]
    summary = [
        f"Aggiornato il {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC da scripts/corpus/update_corpus.py",
        f"cartella del corpus: {corpus_dir}",
        f"registro: {registry}",
        f"documenti inclusi: {len(included)}; letti ora: {read}; dalla cache: {reused} di {total}",
        f"righe nuove nel registro: {len(added)}",
        *(_describe(row) for row in added),
        f"documenti usciti dalla demo: {len(removed)}",
        *(_describe(row) for row in removed),
    ]
    (staged / MARKER).write_text("\n".join(summary) + "\n", encoding="utf-8")

    _replace_run(staged, target)
    replacement = registry.with_name(registry.name + ".tmp")
    shutil.copyfile(target / "registry.csv", replacement)
    os.replace(replacement, registry)
    print(
        "\n".join(
            [
                "",
                *summary,
                "",
                "Fatto. Controllare le righe nuove nel registro (genere, livello, priorità, escluso),",
                "poi riavviare la demo con:",
                f"  DEMO_CORPUS_REGISTRY={registry} DEMO_TEXT_STAGE0_RUNS={target.name}",
            ]
        )
    )
    return 0


def main() -> int:
    """Parse the command line and run the update; return the exit status."""
    base = yaml.safe_load(
        (ROOT / "kg_pipeline" / "config.yaml").read_text(encoding="utf-8")
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "product" / "corpus_registry.csv"
    )
    parser.add_argument(
        "--run",
        default="run_corpus_demo",
        help="run folder under kg_pipeline/artifacts the demo reads (DEMO_TEXT_STAGE0_RUNS)",
    )
    parser.add_argument("--ocr-dir", type=Path, default=ARTIFACTS / "corpus_ocr")
    parser.add_argument(
        "--stage0-cache", type=Path, default=ROOT / base["paths"]["stage0_cache"]
    )
    parser.add_argument(
        "--vector-index-dir",
        type=Path,
        default=ROOT / "artifacts" / "vector_index",
        help="passage-vector cache the demo reads",
    )
    parser.add_argument(
        "--pipeline-config", type=Path, default=ROOT / "kg_pipeline" / "config.yaml"
    )
    parser.add_argument(
        "--gpu",
        metavar="DEVICE",
        default=None,
        help="encode on this GPU (a CUDA_VISIBLE_DEVICES value) instead of the CPU",
    )
    parser.add_argument(
        "--allow-removals",
        action="store_true",
        help="let documents whose files have gone from the folder leave the demo",
    )
    return update(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
