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
# pymupdf4llm renders a bold heading as `## **Title**`, so the markup travels
# inside the captured heading text. It reached the `:Document` nodes verbatim —
# 16 of 22 titles in the production corpus carry `**` or `_` — and from there
# into the citations the expert reads. Section titles carry it too, and those
# are pasted into the extraction prompt.
_EMPHASIS_RE = re.compile(r"\*\*|__|[*_`]")
# A publication year outside this window is a parse artifact, not a date: the
# text scan takes the first `19xx|20xx` it meets in the first three pages, which
# on one paper was a line number and produced 1943.
_MIN_PUBLICATION_YEAR = 1900
# PDF metadata dates look like `D:20240517103000+02'00'`.
_PDF_DATE_RE = re.compile(r"D:(\d{4})")


def _strip_markup(text: str) -> str:
    """Remove markdown emphasis from a heading, leaving the words alone."""
    return " ".join(_EMPHASIS_RE.sub("", text).split()).strip()


def _doc_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return cleaned or "document"


def _read_page_chunks(pdf_path: Path) -> list[PageChunkRecord]:
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
            # Running headers (magazines repeat the issue title on every page)
            # must not open a section per page: skip a title identical to the
            # one that opened the previous section. Only consecutive repeats —
            # a title that recurs with other sections in between is a real
            # recurring heading, and this corpus is full of them: one catalogue
            # repeats "Descrizione dell'iniziativa" for each of its 56 cases.
            if starts and starts[-1][2].strip().lower() == title.lower():
                continue
            # Two offsets, because they answer different questions: the section
            # *ends* where the next heading line begins, and *begins* after its
            # own heading line. Starting a section at its own heading put the
            # raw `## **Title**` into the chunk text, and a heading with no body
            # under it became a chunk that was nothing but that line — 115 of
            # them, 33 characters each, on the production corpus.
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
            # part-way down a page it shares with it. The old rule ended it on
            # the previous page, so the text above a mid-page heading was
            # attributed to whichever section happened to start that page.
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
    """The publication year the file declares, if it declares a plausible one."""
    for key in ("creationDate", "modDate"):
        match = _PDF_DATE_RE.match(str((metadata or {}).get(key, "") or ""))
        if not match:
            continue
        year = int(match.group(1))
        if _MIN_PUBLICATION_YEAR <= year <= _now_year() + 1:
            return year
    return None


def _now_year() -> int:
    return time.localtime().tm_year


def _extract_title_and_year(
    page_chunks: list[PageChunkRecord],
    fallback_title: str,
    metadata: dict[str, str] | None = None,
) -> tuple[str, int | None]:
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
        # The longest heading, not the most prominent one. Measured against the
        # alternative on the production corpus: heading levels come from font
        # size, so the largest text on a first page is the journal masthead or
        # the word "Article", and picking by level replaced 17 of 22 titles with
        # those. Length is the cruder rule and the better one.
        title = max((text for _, text in headers), key=len)
    elif first_line:
        title = first_line

    # A declared date beats a date scraped out of running text. Every file in
    # the corpus carries one (63 of 63), while the text scan takes the first
    # `19xx|20xx` in the first three pages — a line number, an ISSN, a cited
    # work — and got 1943 for a 2019 paper and nothing at all for a 2022 report.
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
        # Not an error: a re-saved PDF declares the day it was re-saved. Worth
        # seeing, because it is the number that ends up in a citation.
        LOGGER.debug(
            "%s: file declares %d, first year in the text is %d; using %d",
            fallback_title,
            declared_year,
            scanned_year,
            declared_year,
        )

    return _strip_markup(title) or fallback_title, publication_year


def discover_pdfs(input_dir: Path, *, warn: bool = True) -> list[Path]:
    """The PDFs of ``input_dir`` itself, whatever the case of the suffix.

    Deliberately NOT recursive. The audit asked for a recursive scan so that a
    file dropped in a subfolder would not be missed in silence; measured against
    the real corpus, recursion does something worse. ``documents/test 1``
    carries a ``pilot/`` folder that repeats three documents of the corpus and
    an ``excluded/`` folder holding three that were left out on purpose, so a
    recursive scan would quietly ingest the excluded ones and duplicate the
    pilot ones — changing what the graph is built from, without a line in the
    log.

    So: the same set of files as before, plus ``.PDF`` spellings, plus a warning
    when a subfolder holds PDFs. Nothing is added in silence and nothing is
    dropped in silence; which of the two the operator wants stays their call.
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
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if single_doc:
        # A misspelled --single-doc used to pass every check: the list was
        # non-empty, the loop skipped the missing file, and stage 0 returned an
        # empty document set that the later stages happily processed. The
        # operator asked for one named document; not finding it is an error,
        # not a document count of zero.
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
            # Only reachable now if the file disappears between the glob and
            # the open. Say so instead of skipping in silence.
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

        # A PDF whose pages carry no text layer (a scan, an image-only report)
        # parses without error and yields nothing to extract from. It is not
        # fatal — a growing corpus will contain some — but it must not pass for
        # an ingested document.
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
    # A record per file is not a corpus if none of them carries text. One
    # unreadable document among many is the operator's call; all of them
    # unreadable means the later stages would run on nothing at all.
    if not docs or len(empty_docs) == len(docs):
        raise ValueError(
            f"No readable document in {input_dir}: "
            f"{len(pdf_paths)} candidate file(s) yielded no text. "
            "Check the PDFs have a text layer (scans need OCR first)"
        )

    return docs


def save_documents(path: Path, docs: list[DocumentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.model_dump() for doc in docs]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_documents(path: Path) -> list[DocumentRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DocumentRecord.model_validate(item) for item in payload]


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--single-doc", default=None)
    args = parser.parse_args()

    docs = ingest_documents(Path(args.input_dir), single_doc=args.single_doc)
    save_documents(Path(args.output_json), docs)


if __name__ == "__main__":
    _cli()
