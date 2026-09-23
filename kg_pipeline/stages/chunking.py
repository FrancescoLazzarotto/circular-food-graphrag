"""Stage 1: split documents into token-bounded chunks for NER and extraction."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

from kg_pipeline.models.types import ChunkRecord, DocumentRecord

LOGGER = logging.getLogger("kg_pipeline")


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# A word, for the purpose of asking whether there is anything here to extract:
# at least two letters, digits excluded.
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
# Extraction produces triples. A subject, a predicate and an object cannot be
# stated in fewer than three words, so a chunk below that is a page footer, a
# table's "Cont." marker or a stray number — and it costs a full LLM call.
_MIN_WORDS_PER_CHUNK = 3
# Minimum share of a document's words that must survive chunking. Healthy
# documents land near or above 100 % (medium and large windows overlap); below
# this, most of the document never reaches the graph.
_MIN_DOC_WORD_COVERAGE = 0.5
_TABLE_LINE_RE = re.compile(r"^\s*\|")
# `|---|---|` under the header row.
_TABLE_RULE_RE = re.compile(r"^\s*\|[\s|:-]+$")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


class ParagraphUnit(NamedTuple):
    """A paragraph with the page and section it comes from.

    Attributes:
        text: Paragraph text.
        page_number: 1-based page number.
        section_title: Title of the enclosing section.
    """

    text: str
    page_number: int
    section_title: str


def _token_count(text: str) -> int:
    """Count word and punctuation tokens in ``text``."""
    return len(_TOKEN_RE.findall(text))


def _is_table(text: str) -> bool:
    """Whether ``text`` is a Markdown table.

    Args:
        text: Paragraph text.

    Returns:
        True when it has at least three non-blank lines and at least 60 % of
        them start with ``|``.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    return sum(1 for line in lines if _TABLE_LINE_RE.match(line)) >= 0.6 * len(lines)


def _split_table(text: str, max_tokens: int) -> list[str]:
    """Split a Markdown table into row groups, each repeating the header.

    A table is a single paragraph (its rows are separated by single
    newlines), so paragraph splitting never breaks it up. Repeating the header
    keeps the column names with every group.

    Args:
        text: Markdown table.
        max_tokens: Token budget per group, header included.

    Returns:
        The row groups, or ``[text]`` when the table has no body rows.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header_len = 2 if len(lines) > 2 and _TABLE_RULE_RE.match(lines[1]) else 1
    header, body = lines[:header_len], lines[header_len:]
    if not body:
        return [text]

    header_text = "\n".join(header)
    budget = max(1, max_tokens - _token_count(header_text))
    groups: list[str] = []
    current: list[str] = []
    used = 0
    for row in body:
        cost = _token_count(row)
        if current and used + cost > budget:
            groups.append("\n".join(header + current))
            current, used = [], 0
        current.append(row)
        used += cost
    if current:
        groups.append("\n".join(header + current))
    return groups


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """Break one oversized paragraph into pieces that fit the window.

    Tables are split into row groups; prose is split at sentence ends, and a
    sentence that is still too long is split on whitespace.

    Args:
        text: Paragraph text.
        max_tokens: Token budget per piece.

    Returns:
        The non-blank pieces, or ``[text]`` when it already fits.
    """
    if _token_count(text) <= max_tokens:
        return [text]
    if _is_table(text):
        return _split_table(text, max_tokens)

    # Prose with no blank line in it — a whole page rendered as one line is
    # common. Sentences are the natural seam; a paragraph with no sentence end
    # left in it is cut on whitespace so it still reaches the model.
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for part in _SENTENCE_END_RE.split(text):
        cost = _token_count(part)
        if current and used + cost > max_tokens:
            pieces.append(" ".join(current))
            current, used = [], 0
        current.append(part)
        used += cost
    if current:
        pieces.append(" ".join(current))

    # A run-on line with no sentence end anywhere — a caption block, an OCR
    # artefact — would come back as one oversized piece and land in the graph as
    # a single unusable chunk. Cut it on whitespace so it still fits.
    sized: list[str] = []
    for piece in pieces:
        if _token_count(piece) <= max_tokens:
            sized.append(piece)
            continue
        words, run, used = piece.split(), [], 0
        for word in words:
            cost = _token_count(word)
            if run and used + cost > max_tokens:
                sized.append(" ".join(run))
                run, used = [], 0
            run.append(word)
            used += cost
        if run:
            sized.append(" ".join(run))
    return [piece for piece in sized if piece.strip()] or [text]


def _window_text(window: list[ParagraphUnit]) -> str:
    """Join a window's paragraphs with blank lines."""
    return "\n\n".join(p.text for p in window)


def _has_extractable_content(window: list[ParagraphUnit]) -> bool:
    """Whether a window has at least ``_MIN_WORDS_PER_CHUNK`` words."""
    return len(_WORD_RE.findall(_window_text(window))) >= _MIN_WORDS_PER_CHUNK


def _drop_empty_windows(
    windows: list[tuple[str, list[ParagraphUnit]]],
) -> list[tuple[str, list[ParagraphUnit]]]:
    """Drop windows too short to hold a triple, unless that empties the document.

    The guard is per document, not per section: a section whose only window is
    a page footer loses it, and only a document that would otherwise vanish
    from the graph keeps its noise.

    Args:
        windows: ``(section_title, window)`` pairs for one whole document.

    Returns:
        The windows with extractable content, or all of ``windows`` when none
        has any.
    """
    kept = [(title, win) for title, win in windows if _has_extractable_content(win)]
    return kept if kept else windows


def _split_paragraphs(text: str) -> list[str]:
    """Split text at blank lines, dropping empty paragraphs."""
    parts = [part.strip() for part in text.split("\n\n")]
    return [part for part in parts if part]


def _paragraphs_for_range(
    doc: DocumentRecord,
    start_page: int,
    end_page: int,
    section_title: str,
    start_offset: int = 0,
    end_offset: int | None = None,
) -> list[ParagraphUnit]:
    """Collect the paragraphs between two positions in a document.

    The offsets keep two sections that share a page from both claiming all of
    it, which would chunk the page twice and duplicate its triples.

    Args:
        doc: Source document.
        start_page: First page, 1-based.
        end_page: Last page, 1-based and inclusive.
        section_title: Title attached to every paragraph.
        start_offset: Character offset in ``start_page`` where the range
            begins.
        end_offset: Character offset in ``end_page`` where the range ends, or
            ``None`` for the end of the page.

    Returns:
        The paragraphs in reading order.
    """
    units: list[ParagraphUnit] = []
    for page in doc.page_chunks:
        if page.page_number < start_page or page.page_number > end_page:
            continue
        text = page.text
        lo = start_offset if page.page_number == start_page else 0
        hi = (
            end_offset
            if page.page_number == end_page and end_offset is not None
            else len(text)
        )
        if hi <= lo:
            continue
        for paragraph in _split_paragraphs(text[lo:hi]):
            units.append(ParagraphUnit(paragraph, page.page_number, section_title))
    return units


def _window_paragraphs(
    paragraphs: list[ParagraphUnit],
    max_tokens: int,
    overlap_tokens: int,
) -> list[list[ParagraphUnit]]:
    """Pack consecutive paragraphs into overlapping token windows.

    Oversized paragraphs are split first. Each window holds as many whole
    paragraphs as fit in ``max_tokens``; the next window starts far enough back
    to repeat about ``overlap_tokens`` tokens, but always advances by at least
    half of the paragraphs of the window just emitted (rounded down, minimum
    one).

    Args:
        paragraphs: Paragraphs in reading order.
        max_tokens: Token budget per window.
        overlap_tokens: Tokens to repeat at the start of the next window.

    Returns:
        The windows, each a non-empty list of paragraphs.
    """
    windows: list[list[ParagraphUnit]] = []
    if not paragraphs:
        return windows

    # A single paragraph over budget cannot be windowed: split it first. Units
    # that already fit are passed through as they are, not rebuilt — that is the
    # overwhelming majority of them.
    expanded: list[ParagraphUnit] = []
    for unit in paragraphs:
        pieces = _split_long_text(unit.text, max_tokens)
        if len(pieces) == 1:
            expanded.append(unit)
            continue
        expanded.extend(
            ParagraphUnit(piece, unit.page_number, unit.section_title)
            for piece in pieces
        )
    paragraphs = expanded

    idx = 0
    while idx < len(paragraphs):
        current: list[ParagraphUnit] = []
        token_budget = 0
        j = idx

        while j < len(paragraphs):
            p_tokens = _token_count(paragraphs[j].text)
            if current and token_budget + p_tokens > max_tokens:
                break
            current.append(paragraphs[j])
            token_budget += p_tokens
            j += 1

        if not current:
            current = [paragraphs[idx]]
            j = idx + 1

        windows.append(current)
        if j >= len(paragraphs):
            break

        overlap = 0
        back = j - 1
        while back > idx and overlap < overlap_tokens:
            overlap += _token_count(paragraphs[back].text)
            back -= 1
        # Guarantee real progress. When a run of short paragraphs sums to less
        # than the overlap budget, the walk-back reaches `idx` and the next
        # window would repeat almost the same content, making the chunk count
        # quadratic. The overlap may claim at most half of the window just
        # emitted.
        min_next = idx + max(1, (j - idx) // 2)
        idx = max(back + 1, min_next)

    return windows


def _build_chunk(
    doc: DocumentRecord,
    chunk_index: int,
    section_title: str,
    paragraphs: list[ParagraphUnit],
) -> ChunkRecord:
    """Build a chunk record from one window.

    Args:
        doc: Source document.
        chunk_index: 1-based position of the chunk within the document.
        section_title: Section the chunk is attributed to.
        paragraphs: The window's paragraphs.

    Returns:
        The chunk, with a page range spanning its paragraphs.
    """
    page_numbers = [p.page_number for p in paragraphs]
    start_page = min(page_numbers)
    end_page = max(page_numbers)
    text = _window_text(paragraphs)

    return ChunkRecord(
        doc_id=doc.doc_id,
        filename=doc.filename,
        chunk_id=f"{doc.doc_id}_chunk_{chunk_index:05d}",
        page_range=f"{start_page}-{end_page}",
        section_title=section_title,
        chunk_index=chunk_index,
        text=text,
    )


def chunk_documents(docs: list[DocumentRecord], config: dict) -> list[ChunkRecord]:
    """Split every document into chunks, with a strategy chosen by page count.

    * Small documents (up to ``small_max_pages``) are packed whole into
      windows of ``small_max_tokens`` without overlap; windows under
      ``small_min_tokens`` are dropped unless that would drop them all.
    * Medium documents (up to ``medium_max_pages``) are windowed section by
      section with ``medium_window_tokens`` / ``medium_overlap_tokens``.
    * Large documents are windowed by level-1 section with
      ``large_window_tokens`` / ``large_overlap_tokens``, or by every section
      when level-1 sections cover less than half of the pages.

    Windows too short to hold a triple are then dropped, unless the document
    would be left with none. Documents that yield no chunk, or keep less than
    ``_MIN_DOC_WORD_COVERAGE`` of their words, are logged as warnings.

    Args:
        docs: Parsed documents.
        config: Pipeline configuration; the ``chunking`` section supplies the
            limits above.

    Returns:
        The chunks of all documents, in document order.

    Raises:
        KeyError: If a ``chunking`` setting is missing.
    """
    chunk_cfg = config["chunking"]

    small_max_pages = int(chunk_cfg["small_max_pages"])
    medium_max_pages = int(chunk_cfg["medium_max_pages"])
    small_min_tokens = int(chunk_cfg["small_min_tokens"])
    small_max_tokens = int(chunk_cfg["small_max_tokens"])
    medium_window = int(chunk_cfg["medium_window_tokens"])
    medium_overlap = int(chunk_cfg["medium_overlap_tokens"])
    large_window = int(chunk_cfg["large_window_tokens"])
    large_overlap = int(chunk_cfg["large_overlap_tokens"])

    chunks: list[ChunkRecord] = []
    empty_docs: list[str] = []
    thin_docs: list[str] = []
    coverages: list[float] = []

    for doc in tqdm(docs, desc="Stage 1 Chunking", unit="doc"):
        next_chunk_idx = 1
        doc_first_chunk = len(chunks)
        # (section_title, window) for the whole document, so the "never empty a
        # document" guard can be applied once, at the end, over all of them.
        pending: list[tuple[str, list[ParagraphUnit]]] = []

        if doc.page_count <= small_max_pages:
            paragraphs = _paragraphs_for_range(
                doc=doc,
                start_page=1,
                end_page=doc.page_count,
                section_title="SmallDoc",
            )
            # Short paragraphs are packed together up to ``small_max_tokens``,
            # so a document made only of short paragraphs (e.g. picture-heavy
            # briefs) still produces chunks.
            windows = _window_paragraphs(
                paragraphs, max_tokens=small_max_tokens, overlap_tokens=0
            )
            kept = [
                win
                for win in windows
                if _token_count(_window_text(win)) >= small_min_tokens
            ]
            # Never drop an entire document: if no window clears the noise floor
            # but there is text, keep the packed windows so the doc still enters
            # the KG.
            if not kept and windows:
                kept = windows
            pending = [(win[0].section_title, win) for win in kept]

        elif doc.page_count <= medium_max_pages:
            for section in doc.sections:
                paragraphs = _paragraphs_for_range(
                    doc=doc,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    section_title=section.title,
                    start_offset=section.start_offset,
                    end_offset=section.end_offset,
                )
                windows = _window_paragraphs(
                    paragraphs, max_tokens=medium_window, overlap_tokens=medium_overlap
                )
                pending.extend((section.title, win) for win in windows)

        else:
            top_sections = [s for s in doc.sections if s.level == 1] or doc.sections
            # Some PDFs expose heading-only level-1 metadata (start_page ==
            # end_page for every section), so windowing those ranges would drop
            # nearly the whole document. If level-1 ranges cover less than half
            # of the pages, fall back to all sections — the same path used by
            # documents that have no level-1 sections at all.
            covered_pages = {
                page
                for s in top_sections
                for page in range(s.start_page, s.end_page + 1)
            }
            if len(covered_pages) < 0.5 * doc.page_count:
                top_sections = doc.sections
            for section in top_sections:
                paragraphs = _paragraphs_for_range(
                    doc=doc,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    section_title=section.title,
                    start_offset=section.start_offset,
                    end_offset=section.end_offset,
                )
                windows = _window_paragraphs(
                    paragraphs, max_tokens=large_window, overlap_tokens=large_overlap
                )
                pending.extend((section.title, win) for win in windows)


        for section_title, win in _drop_empty_windows(pending):
            chunks.append(_build_chunk(doc, next_chunk_idx, section_title, win))
            next_chunk_idx += 1

        doc_words = len(_WORD_RE.findall(doc.markdown_text))
        chunk_words = sum(
            len(_WORD_RE.findall(c.text)) for c in chunks[doc_first_chunk:]
        )
        if doc_words and next_chunk_idx > 1:
            coverage = chunk_words / doc_words
            coverages.append(coverage)
            if coverage < _MIN_DOC_WORD_COVERAGE:
                thin_docs.append(doc.filename)
                LOGGER.warning(
                    "%s kept %d of its %d words (%.1f %%) in %d chunks: most of "
                    "the document will not reach the graph",
                    doc.filename,
                    chunk_words,
                    doc_words,
                    coverage * 100,
                    next_chunk_idx - 1,
                )

        if next_chunk_idx == 1:
            # The guards above keep a document with text from producing no
            # chunks; this reports it if it happens anyway.
            empty_docs.append(doc.filename)
            LOGGER.warning(
                "%s produced no chunks at all (%d pages, %d sections, "
                "%d characters of text): it will not be in the graph",
                doc.filename,
                doc.page_count,
                len(doc.sections),
                len(doc.markdown_text),
            )

    if empty_docs:
        LOGGER.warning(
            "Stage 1: %d of %d documents produced no chunks: %s",
            len(empty_docs),
            len(docs),
            ", ".join(empty_docs),
        )
    if thin_docs:
        LOGGER.warning(
            "Stage 1: %d of %d documents kept less than %.0f %% of their words: "
            "%s",
            len(thin_docs),
            len(docs),
            _MIN_DOC_WORD_COVERAGE * 100,
            ", ".join(thin_docs),
        )
    if not empty_docs and not thin_docs:
        LOGGER.info(
            "Stage 1: %d chunks from %d documents, no document lost, "
            "lowest word coverage %.1f %%",
            len(chunks),
            len(docs),
            min(coverages, default=0.0) * 100,
        )

    return chunks


def save_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    """Write chunk records to a JSON file.

    Args:
        path: Output file; parent directories are created.
        chunks: Records to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump() for chunk in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Read chunk records written by :func:`save_chunks`.

    Args:
        path: JSON file to read.

    Returns:
        The validated records.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ChunkRecord.model_validate(item) for item in payload]


def _cli() -> None:
    """Run stage 1 standalone from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-json", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    docs_payload = json.loads(Path(args.documents_json).read_text(encoding="utf-8"))
    docs = [DocumentRecord.model_validate(item) for item in docs_payload]
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))

    chunks = chunk_documents(docs, config)
    save_chunks(Path(args.output_json), chunks)


if __name__ == "__main__":
    _cli()
