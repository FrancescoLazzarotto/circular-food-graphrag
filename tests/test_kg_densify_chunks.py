"""Densification must read the chunks of the run that built the graph.

Chunk ids are unique only within a run: a rebuild with different chunking reuses
the same ids for other passages. Densifying its graph from another run's folder
stamps every new edge with the id of a passage that says something else, so the
folder is an argument, and the default stays the runs the current graph came
from.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kg" / "kg_densify.py"
_spec = importlib.util.spec_from_file_location("kg_densify", _PATH)
kg_densify = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("kg_densify", kg_densify)
_spec.loader.exec_module(kg_densify)


class _Stop(Exception):
    """Raised by the patched loader to end main() once the folders are known."""


class _Session:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **kwargs):
        return []


class _Driver:
    def session(self, **kwargs):
        return _Session()


def _write_chunks(directory: Path, *chunk_ids: str) -> None:
    directory.mkdir()
    rows = [{"chunk_id": cid, "text": f"text of {cid}"} for cid in chunk_ids]
    (directory / "stage1_chunks.json").write_text(json.dumps(rows), encoding="utf-8")


def test_load_chunks_reads_the_folders_it_is_given(tmp_path) -> None:
    _write_chunks(tmp_path / "a", "a_1", "a_2")
    _write_chunks(tmp_path / "b", "b_1")

    chunks = kg_densify.load_chunks([tmp_path / "a", tmp_path / "missing", tmp_path / "b"])

    assert [c["chunk_id"] for c in chunks] == ["a_1", "a_2", "b_1"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--chunks-dir", "run_x"], [Path("run_x")]),
        (["--chunks-dir", "run_x", "--chunks-dir", "run_y"], [Path("run_x"), Path("run_y")]),
        ([], kg_densify.CHUNK_DIRS),
    ],
)
def test_main_passes_chunks_dir_to_the_loader(monkeypatch, argv, expected) -> None:
    seen: list[list[Path]] = []

    def fake_load(directories):
        seen.append(list(directories))
        raise _Stop

    monkeypatch.setattr(kg_densify.neo4j_env, "connect", lambda target: _Driver())
    monkeypatch.setattr(kg_densify, "load_chunks", fake_load)

    with pytest.raises(_Stop):
        kg_densify.main(argv)

    assert seen == [list(expected)]
