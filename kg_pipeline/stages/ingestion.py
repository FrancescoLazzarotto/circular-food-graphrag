"""Stage 0: parse PDFs into Markdown pages, sections, title and publication year."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import fitz
import pymupdf4llm
from tqdm import tqdm

from kg_pipeline.models.types import DocumentRecord, PageChunkRecord, SectionRecord


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


def _read_page_chunks(pdf_path: Path) -> list[PageChunkRecord]:
    """Render every page of a PDF as Markdown.

    Uses ``pymupdf4llm``'s page-chunk mode when available, and otherwise
    renders one page at a time.

    Args:
        pdf_path: PDF to read.

    Returns:
        One record per page, in page order.
    """
    chunks: list[PageChunkRecord] = []

    try:
        raw = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    except TypeError:
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
    input_dir: Path, single_doc: str | None = None
) -> list[DocumentRecord]:
    """Parse the corpus PDFs into document records.

    Documents without a text layer are kept but reported.

    Args:
        input_dir: Corpus directory.
        single_doc: File name of a single PDF in ``input_dir`` to ingest
            instead of the whole directory.

    Returns:
        One record per PDF, in file-name order.

    Raises:
        FileNotFoundError: If ``input_dir`` or ``single_doc`` does not exist.
        ValueError: If there is no PDF, or no PDF yields any text.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

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

    if not pdf_paths:
        raise ValueError(f"No PDF files found in {input_dir}")

    docs: list[DocumentRecord] = []
    empty_docs: list[str] = []

    for pdf_path in tqdm(pdf_paths, desc="Stage 0 Ingestion", unit="doc"):
        if not pdf_path.exists():
            # Only reachable if the file is removed between discovery and
            # opening.
            LOGGER.warning("Skipping %s: it vanished during ingestion", pdf_path)
            continue

        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
            pdf_metadata = dict(doc.metadata or {})

        page_chunks = _read_page_chunks(pdf_path)
        markdown_text = "\n\n".join(chunk.text for chunk in page_chunks)
        sections = _extract_sections(page_chunks)
        title, publication_year = _extract_title_and_year(
            page_chunks, fallback_title=pdf_path.stem, metadata=pdf_metadata
        )

        # A PDF without a text layer (a scan, an image-only report) parses
        # without error and yields nothing to extract from. It is not fatal,
        # but it must not pass for an ingested document.
        if not markdown_text.strip():
            empty_docs.append(pdf_path.name)
            LOGGER.warning(
                "%s parsed to no text at all over %d pages: no text layer? "
                "It will contribute nothing to the graph",
                pdf_path.name,
                page_count,
            )
        else:
            LOGGER.debug(
                "%s: %d pages, %d characters (%.0f per page)",
                pdf_path.name,
                page_count,
                len(markdown_text),
                len(markdown_text) / max(1, page_count),
            )

        docs.append(
            DocumentRecord(
                doc_id=_doc_id_from_filename(pdf_path.name),
                filename=pdf_path.name,
                page_count=page_count,
                markdown_text=markdown_text,
                sections=sections,
                page_chunks=page_chunks,
                title=title,
                publication_year=publication_year,
            )
        )

    if empty_docs:
        LOGGER.warning(
            "%d of %d documents parsed to no text: %s",
            len(empty_docs),
            len(pdf_paths),
            ", ".join(empty_docs),
        )
    # One unreadable document among many is tolerated; if none has text, the
    # later stages would run on nothing.
    if not docs or len(empty_docs) == len(docs):
        raise ValueError(
            f"No readable document in {input_dir}: "
            f"{len(pdf_paths)} candidate file(s) yielded no text. "
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
