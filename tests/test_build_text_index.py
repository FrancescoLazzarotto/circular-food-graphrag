"""The coverage check of the prebuilt text index names what the demo could not cite.

A document the registry includes but the index does not hold is invisible to
every answer, and nothing else reports it: the demo starts and answers anyway.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from kg_pipeline.utils import corpus_registry
from kg_pipeline.utils.corpus_registry import RegistryRow

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "corpus" / "build_text_index.py"
_spec = importlib.util.spec_from_file_location("build_text_index", _PATH)
build_text_index = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("build_text_index", build_text_index)
_spec.loader.exec_module(build_text_index)


def _registry(tmp_path: Path) -> Path:
    """Three documents, one of them excluded."""
    path = tmp_path / "registro.csv"
    corpus_registry.save_registry(
        path,
        [
            RegistryRow(id_documento="alfa", percorso="A/alfa.pdf"),
            RegistryRow(id_documento="beta", percorso="A/beta.pdf"),
            RegistryRow(id_documento="bozza", percorso="A/bozza.pdf", escluso=True),
        ],
    )
    return path


def test_a_complete_index_passes(tmp_path):
    report = build_text_index.coverage(
        ["alfa-p1-c0001", "beta-p1-c0001"],
        {"alfa": "alfa.pdf", "beta": "beta.pdf"},
        _registry(tmp_path),
    )

    assert report["repeated_ids"] == 0
    assert report["registry_included"] == 2
    assert report["registry_without_passages"] == []
    assert report["indexed_not_in_registry"] == []


def test_an_included_document_without_passages_is_named(tmp_path):
    report = build_text_index.coverage(
        ["alfa-p1-c0001", "vecchio-p1-c0001"],
        {"alfa": "alfa.pdf", "vecchio": "vecchio.pdf"},
        _registry(tmp_path),
    )

    assert report["registry_without_passages"] == [
        {"id_documento": "beta", "percorso": "A/beta.pdf"}
    ]
    assert report["indexed_not_in_registry"] == ["vecchio"]


def test_a_repeated_passage_id_is_reported(tmp_path):
    report = build_text_index.coverage(
        ["alfa-p1-c0001", "alfa-p1-c0001"], {"alfa": "alfa.pdf"}, None
    )

    assert report["repeated_ids"] == 1
    assert report["repeated_id_examples"] == ["alfa-p1-c0001"]
    assert "registry_included" not in report
