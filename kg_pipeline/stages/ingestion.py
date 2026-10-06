"""Stage 0: parse PDFs into Markdown pages, sections, title and publication year."""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import re
import textwrap
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz
import pymupdf4llm
from tqdm import tqdm

from kg_pipeline.models.types import DocumentRecord, PageChunkRecord, SectionRecord
from kg_pipeline.utils import corpus_registry


LOGGER = logging.getLogger("kg_pipeline")

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# pymupdf4llm renders a bold heading as `## **Title**`, so the emphasis markup
# ends up in the captured heading text. Document and section titles reach the
# graph and the extraction prompt, so the markup is stripped.
_EMPHASIS_RE = re.compile(r"\*\*|__|[*_`]")
# Years before this, or after next year, are parse artifacts, not dates.
_MIN_PUBLICATION_YEAR = 1900
# PDF metadata dates look like `D:20240517103000+02'00'`.
_PDF_DATE_RE = re.compile(r"D:(\d{4})")
# Packages whose presence or version changes the text a PDF is read into. With
# pymupdf-layout installed, pymupdf4llm lays pages out with a model run by
# onnxruntime instead of its own rules.
_READING_PACKAGES = ("pymupdf4llm", "pymupdf", "pymupdf-layout", "onnxruntime")


def _strip_markup(text: str) -> str:
    """Remove Markdown emphasis from a heading and collapse whitespace."""
    return " ".join(_EMPHASIS_RE.sub("", text).split()).strip()


def _doc_id_from_filename(filename: str) -> str:
    """Derive a ``doc_id`` from a file name.

    Args:
        filename: PDF file name.

    Returns:
        The lower-cased stem with every run of non-alphanumerics replaced by
        ``_``, or ``"document"`` when nothing is left.
    """
    stem = Path(filename).stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return cleaned or "document"


def _require_unique(sources: list[tuple[Path, str, str, bool]]) -> None:
    """Refuse two documents that would share a ``doc_id`` or a file name.

    Chunk ids are built from the ``doc_id`` and later stages key documents on
    the file name, so a shared value silently merges two documents into one.

    Args:
        sources: ``(file to read, doc_id, file name, is an OCR copy)`` per
            document.

    Raises:
        ValueError: Naming every clash and the files involved.
    """
    clashes: list[str] = []
    for position, label in ((1, "doc_id"), (2, "file name")):
        seen: dict[str, Path] = {}
        for source in sources:
            key = unicodedata.normalize("NFC", source[position])
            if key in seen:
                clashes.append(f"{seen[key]} and {source[0]} share the {label} {key!r}")
            else:
                seen[key] = source[0]
    if clashes:
        raise ValueError(
            "Two documents would become one: "
            + "; ".join(clashes)
            + ". Give them distinct ids in a corpus registry (paths.registry), or rename one."
        )


def _read_page_chunks(pdf_path: Path) -> list[PageChunkRecord]:
    """Render every page of a PDF as Markdown, from its text layer only.

    Uses ``pymupdf4llm``'s page-chunk mode when available, and otherwise
    renders one page at a time.

    The library's own OCR is turned off. Whenever an OCR engine is installed on
    the host it runs by itself, in English, on every page it judges unreadable,
    and replaces an existing OCR layer with its own: the text of a document
    would then depend on what else is installed on the machine. A scanned file
    is read from the OCR copy the corpus registry points to instead.

    Args:
        pdf_path: PDF to read.

    Returns:
        One record per page, in page order.
    """
    chunks: list[PageChunkRecord] = []

    try:
        raw = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, use_ocr=False)
    except TypeError:
        # A library version without page chunks, or without the option; such a
        # version has no OCR to turn off.
        raw = None

    if isinstance(raw, list) and raw:
        for idx, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                meta = item.get("metadata", {})
                page_num = int(meta.get("page", idx))
                text = str(item.get("text", ""))
            else:
                page_num = idx
                text = str(item)
            chunks.append(PageChunkRecord(page_number=page_num, text=text))
        return chunks

    with fitz.open(pdf_path) as doc:
        for page_no in range(1, len(doc) + 1):
            text = str(pymupdf4llm.to_markdown(str(pdf_path), pages=[page_no - 1]))
            chunks.append(PageChunkRecord(page_number=page_no, text=text))

    return chunks


def _read_ocr_copy(pdf_path: Path) -> list[PageChunkRecord]:
    """Read every page of an OCR copy as plain text.

    The recognised text of a scanned page is an invisible layer over the
    image, and ``pymupdf4llm`` leaves invisible text out of its Markdown, so
    the copy is read with PyMuPDF directly. There are no headings to detect:
    the document is one section.

    Args:
        pdf_path: OCR copy to read.

    Returns:
        One record per page, in page order.
    """
    with fitz.open(pdf_path) as doc:
        return [
            PageChunkRecord(page_number=number, text=page.get_text())
            for number, page in enumerate(doc, start=1)
        ]


def _read_pdf(
    pdf_path: Path, is_ocr_copy: bool
) -> tuple[int, dict[str, Any], list[PageChunkRecord]]:
    """Read a PDF's page count, metadata and per-page text.

    Args:
        pdf_path: PDF to read.
        is_ocr_copy: Read it as an OCR copy (see :func:`_read_ocr_copy`).

    Returns:
        ``(page count, PDF metadata, one record per page)``.
    """
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        pdf_metadata = dict(doc.metadata or {})
    page_chunks = _read_ocr_copy(pdf_path) if is_ocr_copy else _read_page_chunks(pdf_path)
    return page_count, pdf_metadata, page_chunks


def _code_of(function: Callable[..., Any]) -> str:
    """A function's syntax tree without its docstring, as text.

    Comments are not part of the tree and the docstring is dropped, so
    rewording either leaves the stage 0 cache valid, while any change to the
    code itself invalidates it.

    Args:
        function: A function defined in a source file.

    Returns:
        The dump of its syntax tree.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


@functools.cache
def _reader_signature(reader: Callable[[Path], list[PageChunkRecord]]) -> str:
    """Everything besides the file that decides what reading it returns.

    The code that reads, options included, and the versions of the packages
    the text depends on: a change to any of them can change the reading, so a
    cache entry filled before it must not be served after it.

    Args:
        reader: :func:`_read_page_chunks` or :func:`_read_ocr_copy`.

    Returns:
        A hex digest.
    """
    parts = [_code_of(_read_pdf), _code_of(reader)]
    for package in _READING_PACKAGES:
        try:
            parts.append(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package}=none")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _read_pdf_cached(
    pdf_path: Path, is_ocr_copy: bool, cache_dir: Path
) -> tuple[tuple[int, dict[str, Any], list[PageChunkRecord]], bool]:
    """Like :func:`_read_pdf`, reusing what an earlier run read from the same file.

    Entries are keyed by the content of the file and by how it is read, never
    by its name or place in the corpus: adding a document to the corpus then
    costs reading that document, and a renamed or moved file is not read again.
    Only the reading is cached; sections, title and year are derived again on
    every run.

    Args:
        pdf_path: PDF to read.
        is_ocr_copy: Read it as an OCR copy.
        cache_dir: Folder of the cache entries; created when missing.

    Returns:
        The result of :func:`_read_pdf`, and whether it came from the cache.
    """
    reader = _read_ocr_copy if is_ocr_copy else _read_page_chunks
    key = hashlib.sha256(
        json.dumps([corpus_registry.file_digest(pdf_path), _reader_signature(reader)]).encode(
            "utf-8"
        )
    ).hexdigest()
    entry = cache_dir / f"{key}.json"
    if entry.is_file():
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
            return (
                int(payload["page_count"]),
                dict(payload["pdf_metadata"]),
                [PageChunkRecord.model_validate(page) for page in payload["page_chunks"]],
            ), True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOGGER.warning(
                "Stage 0 cache entry %s unreadable (%s); reading %s again",
                entry.name,
                exc,
                pdf_path.name,
            )

    page_count, pdf_metadata, page_chunks = _read_pdf(pdf_path, is_ocr_copy)
    payload = {
        "page_count": page_count,
        "pdf_metadata": pdf_metadata,
        "page_chunks": [page.model_dump() for page in page_chunks],
    }
    # Written whole or not at all: an interrupted run must not leave an entry
    # that a later run would serve as a complete reading.
    tmp = entry.with_name(f"{entry.name}.{os.getpid()}.tmp")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(entry)
    except OSError as exc:
        # The reading is done and correct; losing it from the cache costs a
        # later run time, failing the stage would cost this one everything.
        LOGGER.warning("Stage 0 cache entry for %s not saved: %s", pdf_path.name, exc)
        tmp.unlink(missing_ok=True)
    return (page_count, pdf_metadata, page_chunks), False


def _extract_sections(page_chunks: list[PageChunkRecord]) -> list[SectionRecord]:
    """Split a document into sections at its Markdown headings.

    A heading repeated immediately after itself (a running header) does not
    open a new section. Each section starts after its own heading line and
    ends where the next heading line begins, possibly mid-page.

    Args:
        page_chunks: Per-page Markdown text, in page order.

    Returns:
        The sections in document order, or a single ``"Full Document"``
        section spanning every page when there are no headings.
    """
    # (page, level, title, where the heading line starts, where its body starts)
    starts: list[tuple[int, int, str, int, int]] = []

    for page in page_chunks:
        offset = 0
        for raw_line in page.text.splitlines(keepends=True):
            line_start = offset
            offset += len(raw_line)
            match = _HEADER_RE.match(raw_line.strip())
            if not match:
                continue
            level = len(match.group(1))
            title = _strip_markup(match.group(2))
            if not title:
                continue
            # Running headers (a magazine repeating its issue title on every
            # page) must not open a section per page. Only consecutive repeats
            # are skipped: a title that recurs with other sections in between
            # is a real recurring heading, such as a catalogue that repeats the
            # same heading for each case.
            if starts and starts[-1][2].strip().lower() == title.lower():
                continue
            # A section ends where the next heading line begins and starts
            # after its own heading line, so the heading markup never enters
            # the chunk text and a heading with no body yields no text.
            starts.append((page.page_number, level, title, line_start, offset))

    if not starts:
        return [
            SectionRecord(
                title="Full Document",
                level=1,
                start_page=1,
                end_page=max(1, page_chunks[-1].page_number if page_chunks else 1),
            )
        ]

    sections: list[SectionRecord] = []
    last_page = max(1, page_chunks[-1].page_number if page_chunks else 1)
    for idx, (start_page, level, title, _heading_start, start_offset) in enumerate(starts):
        if idx < len(starts) - 1:
            # A section ends exactly where the next one begins, which may be
            # part-way down a page it shares with it, so the text above a
            # mid-page heading stays with the preceding section.
            end_page = max(start_page, starts[idx + 1][0])
            end_offset: int | None = starts[idx + 1][3]
        else:
            end_page = max(start_page, last_page)
            end_offset = None
        sections.append(
            SectionRecord(
                title=title,
                level=level,
                start_page=start_page,
                end_page=end_page,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        )
    return sections


def _year_from_pdf_metadata(metadata: dict[str, str] | None) -> int | None:
    """Read a plausible publication year from the PDF metadata.

    Args:
        metadata: PDF metadata dict, as returned by PyMuPDF.

    Returns:
        The year of ``creationDate``, else of ``modDate``, when it falls
        between ``_MIN_PUBLICATION_YEAR`` and next year; otherwise ``None``.
    """
    for key in ("creationDate", "modDate"):
        match = _PDF_DATE_RE.match(str((metadata or {}).get(key, "") or ""))
        if not match:
            continue
        year = int(match.group(1))
        if _MIN_PUBLICATION_YEAR <= year <= _now_year() + 1:
            return year
    return None


def _now_year() -> int:
    """Return the current local year."""
    return time.localtime().tm_year


def _extract_title_and_year(
    page_chunks: list[PageChunkRecord],
    fallback_title: str,
    metadata: dict[str, str] | None = None,
) -> tuple[str, int | None]:
    """Detect a document's title and publication year from its first pages.

    The title is the first level-1 heading in the first three pages, else the
    longest heading of any level, else the first non-empty line. The year is
    the one declared in the PDF metadata, else the first plausible year in the
    text of the first three pages.

    Args:
        page_chunks: Per-page Markdown text, in page order.
        fallback_title: Title used when nothing better is found.
        metadata: PDF metadata dict, if available.

    Returns:
        ``(title, publication_year)``; the title has Markdown emphasis removed
        and the year is ``None`` when none was found.
    """
    title = fallback_title
    publication_year: int | None = None

    head_text = "\n".join(
        chunk.text for chunk in page_chunks[: min(3, len(page_chunks))]
    )

    # Prefer a level-1 header; failing that, the longest header of any level in
    # the first pages (books often expose the real title as a level-2 header,
    # while the first text line is a preface author or colophon fragment).
    headers: list[tuple[int, str]] = []
    first_line = ""
    for line in head_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not first_line:
            first_line = line
        match = _HEADER_RE.match(line)
        if match:
            headers.append((len(match.group(1)), match.group(2).strip()))

    level1 = [text for level, text in headers if level == 1]
    if level1:
        title = level1[0]
    elif headers:
        # Heading levels come from font size, so the most prominent heading on
        # a first page is often a journal masthead or a word like "Article".
        # The longest heading is the more reliable title.
        title = max((text for _, text in headers), key=len)
    elif first_line:
        title = first_line

    # The date declared in the metadata is preferred: the text scan takes the
    # first `19xx|20xx` in the first three pages, which may be a line number,
    # an ISSN or the year of a cited work.
    declared_year = _year_from_pdf_metadata(metadata)
    scanned_year: int | None = None
    for candidate in _YEAR_RE.finditer(head_text):
        year = int(candidate.group(0))
        if _MIN_PUBLICATION_YEAR <= year <= _now_year() + 1:
            scanned_year = year
            break

    publication_year = declared_year if declared_year is not None else scanned_year
    if (
        declared_year is not None
        and scanned_year is not None
        and abs(declared_year - scanned_year) > 1
    ):
        # Not an error: a re-saved PDF declares the date it was re-saved. Logged
        # because this year ends up in citations.
        LOGGER.debug(
            "%s: file declares %d, first year in the text is %d; using %d",
            fallback_title,
            declared_year,
            scanned_year,
            declared_year,
        )

    return _strip_markup(title) or fallback_title, publication_year


def discover_pdfs(input_dir: Path, *, warn: bool = True) -> list[Path]:
    """List the PDFs directly inside ``input_dir``.

    The scan is not recursive: subfolders may hold documents deliberately
    excluded from the corpus or copies of documents already in it. PDFs found
    in subfolders are reported with a warning instead. The suffix match is
    case-insensitive.

    Args:
        input_dir: Corpus directory.
        warn: Log a warning when subfolders contain PDFs.

    Returns:
        The PDF paths, sorted.
    """
    pdfs = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    nested = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf" and path.parent != input_dir
    )
    if nested and warn:
        LOGGER.warning(
            "%d PDF(s) live in subfolders of %s and are NOT ingested: %s%s. "
            "Move them up one level to include them.",
            len(nested),
            input_dir,
            ", ".join(str(p.relative_to(input_dir)) for p in nested[:5]),
            " …" if len(nested) > 5 else "",
        )
    return pdfs


def ingest_documents(
    input_dir: Path,
    single_doc: str | None = None,
    registry_path: Path | None = None,
    ocr_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> list[DocumentRecord]:
    """Parse the corpus PDFs into document records.

    Without a registry, the PDFs directly inside ``input_dir`` are read (see
    :func:`discover_pdfs`). With one, its included rows are read wherever they
    sit under ``input_dir``, with the registry's ids, and the OCR copy stands in
    for a file marked ``ocr``. Documents without a text layer are kept but
    reported.

    Args:
        input_dir: Corpus directory.
        single_doc: File name of a single PDF to ingest instead of the whole
            corpus; with a registry, also its path or id.
        registry_path: Corpus registry (``paths.registry``), if any.
        ocr_dir: Folder of the OCR copies (``paths.ocr_dir``), if any.
        cache_dir: Folder where each file's reading is kept for later runs
            (``paths.stage0_cache``, see :func:`_read_pdf_cached`); ``None``
            reads every file.

    Returns:
        One record per PDF, in file-name order without a registry and in
        registry order with one.

    Raises:
        FileNotFoundError: If ``input_dir`` or ``single_doc`` does not exist, or
            a registry file is missing.
        ValueError: If there is no PDF, no PDF yields any text, two documents
            would share an id or a file name, or the registry is invalid or
            out of date.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # (file to read, doc_id, file name kept in the record, is an OCR copy)
    sources: list[tuple[Path, str, str, bool]]
    if registry_path is not None:
        rows = corpus_registry.load_registry(registry_path)
        sources = [
            (path, row.id_documento, row.filename, row.ocr)
            for path, row in corpus_registry.documents_to_ingest(
                rows, input_dir, ocr_dir, single_doc
            )
        ]
        unlisted = corpus_registry.unregistered_files(rows, input_dir)
        if unlisted:
            LOGGER.warning(
                "%d file(s) under %s are not in the registry and are NOT ingested: %s%s. "
                "Add them with scripts/corpus/build_registry.py.",
                len(unlisted),
                input_dir,
                ", ".join(unlisted[:5]),
                " …" if len(unlisted) > 5 else "",
            )
    else:
        if single_doc:
            # A named document that does not exist is an error, not an empty
            # corpus.
            pdf_paths = [input_dir / single_doc]
            if not pdf_paths[0].exists():
                raise FileNotFoundError(
                    f"--single-doc {single_doc!r} not found in {input_dir}"
                )
        else:
            pdf_paths = discover_pdfs(input_dir)
        sources = [
            (path, _doc_id_from_filename(path.name), path.name, False) for path in pdf_paths
        ]

    if not sources:
        raise ValueError(f"No PDF files found in {input_dir}")
    _require_unique(sources)

    docs: list[DocumentRecord] = []
    empty_docs: list[str] = []
    from_cache = 0

    for pdf_path, doc_id, filename, is_ocr_copy in tqdm(
        sources, desc="Stage 0 Ingestion", unit="doc"
    ):
        if not pdf_path.exists():
            # Only reachable if the file is removed between discovery and
            # opening.
            LOGGER.warning("Skipping %s: it vanished during ingestion", pdf_path)
            continue

        if cache_dir is None:
            page_count, pdf_metadata, page_chunks = _read_pdf(pdf_path, is_ocr_copy)
        else:
            (page_count, pdf_metadata, page_chunks), cached = _read_pdf_cached(
                pdf_path, is_ocr_copy, cache_dir
            )
            from_cache += cached
        markdown_text = "\n\n".join(chunk.text for chunk in page_chunks)
        sections = _extract_sections(page_chunks)
        title, publication_year = _extract_title_and_year(
            page_chunks, fallback_title=Path(filename).stem, metadata=pdf_metadata
        )

        # A PDF without a text layer (a scan, an image-only report) parses
        # without error and yields nothing to extract from. It is not fatal,
        # but it must not pass for an ingested document.
        if not markdown_text.strip():
            empty_docs.append(filename)
            LOGGER.warning(
                "%s parsed to no text at all over %d pages: no text layer? "
                "It will contribute nothing to the graph",
                filename,
                page_count,
            )
        else:
            LOGGER.debug(
                "%s: %d pages, %d characters (%.0f per page)",
                filename,
                page_count,
                len(markdown_text),
                len(markdown_text) / max(1, page_count),
            )

        docs.append(
            DocumentRecord(
                doc_id=doc_id,
                filename=filename,
                page_count=page_count,
                markdown_text=markdown_text,
                sections=sections,
                page_chunks=page_chunks,
                title=title,
                publication_year=publication_year,
            )
        )

    if cache_dir is not None:
        LOGGER.info(
            "Stage 0: %d of %d documents reused from %s, %d read",
            from_cache,
            len(docs),
            cache_dir,
            len(docs) - from_cache,
        )
    if empty_docs:
        LOGGER.warning(
            "%d of %d documents parsed to no text: %s",
            len(empty_docs),
            len(sources),
            ", ".join(empty_docs),
        )
    # One unreadable document among many is tolerated; if none has text, the
    # later stages would run on nothing.
    if not docs or len(empty_docs) == len(docs):
        raise ValueError(
            f"No readable document in {input_dir}: "
            f"{len(sources)} candidate file(s) yielded no text. "
            "Check the PDFs have a text layer (scans need OCR first)"
        )

    return docs


def save_documents(path: Path, docs: list[DocumentRecord]) -> None:
    """Write document records to a JSON file.

    Args:
        path: Output file; parent directories are created.
        docs: Records to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.model_dump() for doc in docs]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_documents(path: Path) -> list[DocumentRecord]:
    """Read document records written by :func:`save_documents`.

    Args:
        path: JSON file to read.

    Returns:
        The validated records.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DocumentRecord.model_validate(item) for item in payload]


def _cli() -> None:
    """Run stage 0 standalone from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--single-doc", default=None)
    args = parser.parse_args()

    docs = ingest_documents(Path(args.input_dir), single_doc=args.single_doc)
    save_documents(Path(args.output_json), docs)


if __name__ == "__main__":
    _cli()
