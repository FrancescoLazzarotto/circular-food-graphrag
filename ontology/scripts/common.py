"""Shared paths, corpus access and sheet I/O for the ontology scripts.

The scripts run in the separate ``ontology`` virtualenv, not in ``graphllm``.
Quote matching must behave exactly like the answer-time quote gate, so the
normaliser is loaded straight from ``src/graphrag/agent/evidence.py``, which
depends on the standard library only.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

ONTOLOGY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ONTOLOGY_DIR.parent
SOURCES_DIR = ONTOLOGY_DIR / "sources"
DRAFTS_DIR = ONTOLOGY_DIR / "drafts"
BUILD_DIR = ONTOLOGY_DIR / "build"
REVIEW_DIR = ONTOLOGY_DIR / "review"

# The corpus text and the graph backups are large and live outside git; a
# worktree has neither, so both default to the main checkout and storage.
STAGE0_PATH = Path(
    os.environ.get(
        "ONTOLOGY_STAGE0",
        "/srv/projects/graphllm/experiments/graphRAGPipelineExp1/"
        "kg_pipeline/artifacts/run_corpus_20261005/stage0_documents.json",
    )
)
BACKUP_ROOT = Path(
    os.environ.get(
        "ONTOLOGY_BACKUP_ROOT", "/mnt/storage/flazzarotto/graph_backups/daily"
    )
)

SHEET_COLUMNS = {
    "frameworks": [
        "id", "name", "version", "derived_from", "author", "year",
        "document", "pages", "status", "notes",
    ],
    "framework_members": [
        "framework_id", "position", "concept_id", "label_it", "label_en",
        "family", "document", "page", "quote", "status", "notes",
    ],
    "concepts": [
        "id", "label_it", "label_en", "synonyms_it", "synonyms_en",
        "broader", "related", "graph_nodes", "status", "notes",
    ],
    "definitions": [
        "concept_id", "quoted_definition", "language", "document", "page",
        "status", "notes",
    ],
    "version_mappings": ["from", "to", "type", "document", "page", "status", "notes"],
    "cases": [
        "id", "name", "document", "page", "supply_chain", "territory", "actors",
        "input", "output", "strategies", "framework_id", "concepts_3c", "status", "notes",
    ],
}

STATUSES = {"proposed", "auto_accepted", "approved", "corrected", "rejected"}
MAPPING_TYPES = {"same", "merged", "new", "renamed"}


def _load_evidence_module():
    path = REPO_ROOT / "src" / "graphrag" / "agent" / "evidence.py"
    spec = importlib.util.spec_from_file_location("_ontology_evidence", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


normalize = _load_evidence_module()._normalize_for_quote_match


@lru_cache(maxsize=1)
def _stage0() -> list[dict]:
    with STAGE0_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _documents() -> dict[str, dict[int, str]]:
    return {
        doc["doc_id"]: {int(p["page_number"]): p["text"] for p in doc["page_chunks"]}
        for doc in _stage0()
    }


def document_filename(document: str) -> str:
    """File name of a corpus document, as the retrieved passages carry it.

    Args:
        document: ``doc_id`` in ``stage0_documents.json``.

    Returns:
        The PDF file name, or an empty string for an unknown document.
    """
    return next((doc["filename"] for doc in _stage0() if doc["doc_id"] == document), "")


def page_text(document: str, page: int) -> str:
    """Return the text of one page of a corpus document.

    Args:
        document: ``doc_id`` in ``stage0_documents.json``.
        page: 1-based PDF page number.

    Returns:
        The page text, or an empty string when the page does not exist.
    """
    return _documents().get(document, {}).get(int(page), "")


def parse_pages(pages: str | int) -> list[int]:
    """Expand a page field such as ``"8"``, ``"8-9"`` or ``"5, 13"``.

    Args:
        pages: Page field as written in the sheets.

    Returns:
        The page numbers, in order.
    """
    result: list[int] = []
    for part in re.split(r"[,;]\s*", str(pages).strip()):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def quote_on_page(quote: str, document: str, pages: str | int) -> bool:
    """Check that a quote occurs verbatim on the stated page(s).

    Unlike the answer-time gate, no ellipsis is accepted: a source quote in
    the sheets must be one contiguous passage. A quote that runs across a page
    break is matched against the listed pages joined in order.

    Args:
        quote: The quoted passage.
        document: ``doc_id`` of the source.
        pages: Page field, e.g. ``"8"`` or ``"8-9"``.

    Returns:
        True when the normalised quote is a substring of the normalised pages.
    """
    needle = normalize(quote)
    if len(needle) < 3:
        return False
    try:
        page_list = parse_pages(pages)
    except ValueError:
        return False
    haystack = " ".join(normalize(page_text(document, p)) for p in page_list)
    return needle in haystack


def contains_term(text: str, term: str) -> bool:
    """Whole-word containment after normalisation.

    Args:
        text: Passage to search.
        term: Label or synonym.

    Returns:
        True when the normalised term occurs in the passage on word boundaries.
    """
    needle = normalize(term)
    return bool(needle) and f" {needle} " in f" {normalize(text)} "


def read_sheet(name: str) -> list[dict[str, str]]:
    """Read a TSV sheet from ``sources/``.

    Args:
        name: Sheet name without extension.

    Returns:
        One dict per row, keyed by column.
    """
    path = SOURCES_DIR / f"{name}.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_sheet(name: str, rows: list[dict[str, str]]) -> None:
    """Write a TSV sheet to ``sources/`` with its fixed columns.

    Args:
        name: Sheet name without extension.
        rows: Rows to write; missing columns are left empty.
    """
    write_tsv(SOURCES_DIR / f"{name}.tsv", SHEET_COLUMNS[name], rows)


def write_tsv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write rows as a tab-separated file.

    Args:
        path: Destination.
        columns: Column order.
        rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _cell(row.get(col, "")) for col in columns})


def _cell(value) -> str:
    # A tab or newline inside a cell would break the sheet for every reader.
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def latest_backup() -> Path:
    """Return the most recent daily graph backup directory.

    Returns:
        Path of the newest ``<date>`` folder that holds a ``nodes.json``.
    """
    candidates = sorted(
        p for p in BACKUP_ROOT.iterdir() if (p / "nodes.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"no nodes.json under {BACKUP_ROOT}")
    return candidates[-1]


@lru_cache(maxsize=1)
def graph_name_index() -> dict[str, list[str]]:
    """Map normalised node names and aliases to node names.

    Node ids change at every restore, so the link is the name.

    Returns:
        Normalised form → sorted distinct node names carrying it.
    """
    with (latest_backup() / "nodes.json").open(encoding="utf-8") as handle:
        nodes = json.load(handle)
    index: dict[str, set[str]] = {}
    for node in nodes:
        props = node.get("props") or {}
        name = props.get("name")
        if not name:
            continue
        for form in [name, *(props.get("aliases") or [])]:
            key = normalize(form)
            if key:
                index.setdefault(key, set()).add(name)
    return {key: sorted(names) for key, names in index.items()}


def graph_nodes_for(forms: list[str]) -> list[str]:
    """Names of the extracted nodes whose name or alias equals one of the forms.

    Args:
        forms: Labels and synonyms of a concept.

    Returns:
        Sorted distinct node names.
    """
    index = graph_name_index()
    found: set[str] = set()
    for form in forms:
        found.update(index.get(normalize(form), []))
    return sorted(found)


def split_list(value: str) -> list[str]:
    """Split a ``;``-separated cell.

    Args:
        value: Cell content.

    Returns:
        Non-empty stripped items.
    """
    return [item.strip() for item in str(value or "").split(";") if item.strip()]
