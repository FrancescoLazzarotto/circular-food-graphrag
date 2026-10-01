"""Document loading, character chunking and retrieval for text-only RAG."""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from graphrag.text_rag.manager import TextChunk, TextRAGManager

try:
    import fitz  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - runtime dependency check
    fitz = None

@lru_cache(maxsize=8)
def _accepts_mmr(retriever_type: type) -> bool:
    """Whether a retriever's ``retrieve_with_scores`` takes ``mmr_lambda``."""
    try:
        parameters = inspect.signature(retriever_type.retrieve_with_scores).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return "mmr_lambda" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


_WHITESPACE_RE = re.compile(r"\s+")
_SUPPORTED_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".csv",
}
_DEFAULT_DISCOVERY_PATTERNS = (
    "*.pdf",
    "*.txt",
    "*.md",
    "*.markdown",
)


# A page or a range in a citation label: "70", "59-60", "128-128".
_PAGE_SPAN_RE = re.compile(r"(\d+)(?:\s*[-–]\s*(\d+))?")
# Wider than any span a single claim cites; a longer one is a malformed label
# and would promote half the document.
_MAX_PAGE_SPAN = 20


def _page_numbers(label: str) -> set[int]:
    """Every page a label such as ``p. 32, 11-13`` names."""
    pages: set[int] = set()
    for start, end in _PAGE_SPAN_RE.findall(str(label or "")):
        first = int(start)
        last = int(end) if end else first
        if first <= last <= first + _MAX_PAGE_SPAN:
            pages.update(range(first, last + 1))
        else:
            pages.add(first)
    return pages


def _normalize_text(text: str) -> str:
    """Collapse every whitespace run to one space and strip."""
    return _WHITESPACE_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class RetrievedTextChunk:
    """A chunk returned by retrieval, with its score.

    Attributes:
        chunk_id: Chunk identifier.
        source: Where the chunk comes from.
        content: Chunk text.
        score: Backend score; higher is more relevant.
    """

    chunk_id: str
    source: str | None
    content: str
    score: float


class StandardTextRAGPipeline:
    """Basic document retrieval pipeline for standard (text-only) RAG."""

    def __init__(
        self,
        retriever: TextRAGManager | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
        min_chunk_chars: int = 80,
    ) -> None:
        """Create the pipeline.

        Args:
            retriever: Chunk index; defaults to a lexical ``TextRAGManager``.
            chunk_size: Characters per chunk, at least 128.
            chunk_overlap: Characters shared by consecutive chunks; smaller
                than ``chunk_size``.
            min_chunk_chars: Chunks shorter than this are discarded.

        Raises:
            ValueError: If a size argument is out of range.
        """
        if chunk_size < 128:
            raise ValueError("chunk_size must be >= 128")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if min_chunk_chars < 1:
            raise ValueError("min_chunk_chars must be >= 1")

        self.retriever = retriever or TextRAGManager()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    @property
    def indexed_chunks(self) -> int:
        """Number of chunks in the index."""
        return self.retriever.size

    def clear(self) -> None:
        """Empty the index."""
        self.retriever.clear()

    def index_paths(
        self,
        paths: Sequence[str | Path],
        discovery_patterns: Sequence[str] | None = None,
    ) -> int:
        """Load, chunk and index files.

        PDFs are split per page and text files kept whole before chunking;
        each chunk's source is ``<path>[#page=<n>]#chunk=<n>``.

        Args:
            paths: Files, or directories searched recursively.
            discovery_patterns: Glob patterns for directories; defaults to
                PDF, text and Markdown files.

        Returns:
            How many chunks were added.

        Raises:
            FileNotFoundError: If a path does not exist.
            ValueError: If no file is found.
            RuntimeError: If a PDF is found and PyMuPDF is not installed.
        """
        files_to_index = self._resolve_paths(
            paths, discovery_patterns=discovery_patterns
        )
        prepared_chunks: list[TextChunk] = []

        for doc_index, file_path in enumerate(files_to_index, start=1):
            sections = self._load_sections_from_path(file_path)
            for section_index, (source_tag, section_text) in enumerate(
                sections, start=1
            ):
                chunk_texts = self._split_into_chunks(section_text)
                for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
                    chunk_id = (
                        f"d{doc_index:04d}-s{section_index:04d}-c{chunk_index:04d}"
                    )
                    chunk_source = f"{source_tag}#chunk={chunk_index}"
                    prepared_chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            source=chunk_source,
                        )
                    )

        return self.retriever.add_chunks(prepared_chunks)

    def index_directory(
        self,
        root: str | Path,
        discovery_patterns: Sequence[str] | None = None,
    ) -> int:
        """Index every matching file under ``root``; see :meth:`index_paths`."""
        return self.index_paths([root], discovery_patterns=discovery_patterns)

    def chunks_from(self, document_label: str, page: str = "") -> list[Any]:
        """The indexed chunks of one document, the cited page first.

        A citation names a document and a page; this returns the passage it
        points at. It is deliberately not a search — the question that quotes a
        claim is phrased in the reader's words, not the source's, so ranking
        cannot be relied on to surface the document the claim came from.

        Args:
            document_label: The short label as it appears in an answer, e.g.
                ``REPORT MATTM``.
            page: Page label such as ``p. 70``, ``p. 59-60`` or ``p. 32,
                11-13``; when given, chunks from those pages come first.

        Returns:
            Matching chunks, cited page first, empty when the label matches no
            indexed document.
        """
        from graphrag.agent.evidence import parse_chunk_source, short_doc_label

        wanted = document_label.strip().lower()
        if not wanted:
            return []
        indexed = getattr(self.retriever, "chunks", None)
        if indexed is None:
            return []

        cited = _page_numbers(page)
        on_page: list[Any] = []
        elsewhere: list[Any] = []
        for chunk in indexed:
            document, chunk_page = parse_chunk_source(str(getattr(chunk, "source", "")))
            if short_doc_label(document).strip().lower() != wanted:
                continue
            if cited & _page_numbers(chunk_page):
                on_page.append(chunk)
            else:
                elsewhere.append(chunk)
        return on_page + elsewhere

    def chunks_with_terms(
        self, terms: Sequence[str], query: str = "", max_df_ratio: float = 0.02
    ) -> list[Any]:
        """The indexed chunks containing ``terms``, the most relevant first.

        A lookup by exact form, not a similarity search: it reaches the
        passages an embedding cannot place, such as a code ("3C", "10R") the
        encoder never learnt. A term found in more than ``max_df_ratio`` of the
        chunks says nothing about which of them is relevant and is ignored.

        Args:
            terms: The forms to look for, matched exactly on word boundaries:
                "SEeD" is a project, "seed" is not.
            query: When given and the backend can score stored passages, the
                matches are ordered by similarity to it: among the passages
                naming "3C", the one about the three C's rather than the
                bibliography entry that names them twice.
            max_df_ratio: Largest share of chunks a term may occur in.

        Returns:
            Matching chunks, by similarity to ``query`` when it can be scored,
            otherwise by the sum over the terms they contain of
            log(chunks / chunks with the term); empty when nothing matches.
        """
        indexed = getattr(self.retriever, "chunks", None)
        if not indexed or not terms:
            return []
        total = len(indexed)
        patterns = [
            re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
            for term in dict.fromkeys(str(t).strip() for t in terms)
            if term
        ]
        present = [[bool(p.search(str(c.content))) for p in patterns] for c in indexed]
        weights = []
        for column in range(len(patterns)):
            df = sum(1 for row in present if row[column])
            weights.append(
                math.log(total / df) if 0 < df <= max(1, int(total * max_df_ratio)) else 0.0
            )
        matched = [
            (sum(w for hit, w in zip(row, weights) if hit), index)
            for index, row in enumerate(present)
        ]
        matched = [item for item in matched if item[0] > 0]
        similarity = getattr(self.retriever, "similarity", None)
        if query and callable(similarity):
            scores = similarity(query, [indexed[index] for _, index in matched])
            matched = [(score, index) for score, (_, index) in zip(scores, matched)]
        ranked = sorted(matched, key=lambda item: -item[0])
        return [indexed[index] for _, index in ranked]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mmr_lambda: float | None = None,
        fetch_k: int | None = None,
    ) -> list[RetrievedTextChunk]:
        """Retrieve the top chunks for a query.

        Args:
            query: The retrieval query.
            top_k: How many chunks to return.
            mmr_lambda: MMR diversification factor, passed to backends whose
                ``retrieve_with_scores`` accepts it and ignored by the others.
            fetch_k: Candidate pool for MMR, passed with ``mmr_lambda``.

        Returns:
            The retrieved chunks with their scores.
        """
        kwargs: dict[str, Any] = {}
        # Asked of the signature rather than discovered by catching TypeError: a
        # backend raising TypeError for its own reasons would otherwise be
        # retried silently and look like a backend without MMR.
        if mmr_lambda is not None and _accepts_mmr(type(self.retriever)):
            kwargs["mmr_lambda"] = mmr_lambda
            if fetch_k:
                kwargs["fetch_k"] = fetch_k
        items = self.retriever.retrieve_with_scores(query=query, top_k=top_k, **kwargs)
        return [
            RetrievedTextChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                content=chunk.content,
                score=score,
            )
            for chunk, score in items
        ]

    def build_context(
        self,
        query: str,
        top_k: int = 4,
        include_sources: bool = True,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Render the ``top_k`` best chunks as one context string.

        Args:
            query: The retrieval query.
            top_k: How many chunks to include.
            include_sources: Prefix each chunk with ``Source: <source>``.
            separator: String placed between chunks.

        Returns:
            The context.
        """
        retrieved = self.retrieve(query=query, top_k=top_k)
        if include_sources:
            rendered = []
            for item in retrieved:
                if item.source:
                    rendered.append(f"Source: {item.source}\n{item.content}")
                else:
                    rendered.append(item.content)
            return separator.join(rendered)

        return separator.join(item.content for item in retrieved)

    def _resolve_paths(
        self,
        paths: Sequence[str | Path],
        discovery_patterns: Sequence[str] | None,
    ) -> list[Path]:
        """Expand files and directories into a sorted, de-duplicated file list.

        Raises:
            FileNotFoundError: If a path does not exist.
            ValueError: If no file is found.
        """
        patterns = (
            tuple(discovery_patterns)
            if discovery_patterns
            else _DEFAULT_DISCOVERY_PATTERNS
        )
        resolved: list[Path] = []

        for raw_path in paths:
            current = Path(raw_path).expanduser().resolve()
            if not current.exists():
                raise FileNotFoundError(f"Path does not exist: {current}")

            if current.is_file():
                resolved.append(current)
                continue

            for pattern in patterns:
                resolved.extend(current.rglob(pattern))

        unique_files = sorted({path.resolve() for path in resolved if path.is_file()})
        if not unique_files:
            raise ValueError("No files discovered for indexing")
        return unique_files

    def _load_sections_from_path(self, file_path: Path) -> list[tuple[str, str]]:
        """Read a file as ``(source_tag, text)`` sections.

        PDFs give one section per page; supported text files give one section;
        any other file gives none.
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf_sections(file_path)

        if suffix in _SUPPORTED_TEXT_SUFFIXES:
            text = _normalize_text(
                file_path.read_text(encoding="utf-8", errors="ignore")
            )
            if not text:
                return []
            return [(str(file_path), text)]

        return []

    def _load_pdf_sections(self, file_path: Path) -> list[tuple[str, str]]:
        """Read the non-empty pages of a PDF as ``(path#page=<n>, text)``.

        Raises:
            RuntimeError: If PyMuPDF is not installed.
        """
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is required for PDF ingestion. Install with: pip install pymupdf"
            )

        sections: list[tuple[str, str]] = []
        with fitz.open(file_path) as document:
            for page_number, page in enumerate(document, start=1):
                page_text = _normalize_text(page.get_text("text"))
                if not page_text:
                    continue
                source_tag = f"{file_path}#page={page_number}"
                sections.append((source_tag, page_text))
        return sections

    def _split_into_chunks(self, text: str) -> list[str]:
        """Cut text into overlapping character windows of ``chunk_size``.

        Windows shorter than ``min_chunk_chars`` are dropped, as is a text
        shorter than that.
        """
        normalized = _normalize_text(text)
        if len(normalized) < self.min_chunk_chars:
            return []

        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(normalized):
            end = min(len(normalized), start + self.chunk_size)
            candidate = normalized[start:end].strip()
            if len(candidate) >= self.min_chunk_chars:
                chunks.append(candidate)
            if end >= len(normalized):
                break
            start += step

        return chunks
