"""In-memory lexical chunk retrieval ranked by Okapi BM25."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into lower-cased word tokens."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class TextChunk:
    """An indexed passage of a document.

    Attributes:
        chunk_id: Unique identifier.
        content: Passage text.
        source: Where the passage comes from, e.g. ``path#page=3#chunk=1``.
    """

    chunk_id: str
    content: str
    source: str | None = None


class TextRAGManager:
    """In-memory manager for standard textual RAG retrieval, ranked by BM25."""

    # Okapi BM25 defaults. k1 bounds how fast term frequency saturates, b how
    # strongly length is normalised; 1.5/0.75 are the standard settings and the
    # ones Lucene uses, which keeps this channel comparable to the Neo4j
    # full-text channel.
    _BM25_K1 = 1.5
    _BM25_B = 0.75

    def __init__(self) -> None:
        """Create an empty index."""
        self._chunks: list[TextChunk] = []
        self._chunk_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._idf_dirty = False

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    @property
    def chunks(self) -> list[TextChunk]:
        """Every indexed chunk, for lookups that are not a ranking.

        Following a citation back to the passage it names is a lookup by
        document and page, not a search: the words of a follow-up question are
        not the words of the source, so no amount of ranking is guaranteed to
        surface it.
        """
        return list(self._chunks)

    def clear(self) -> None:
        """Remove every chunk from the index."""
        self._chunks.clear()
        self._chunk_tokens.clear()
        self._idf.clear()
        self._avg_doc_len = 0.0
        self._idf_dirty = False

    def add_chunks(self, chunks: Iterable[TextChunk]) -> int:
        """Index chunks, skipping those without any word token.

        Args:
            chunks: Chunks to add.

        Returns:
            How many chunks were added.
        """
        added = 0
        for chunk in chunks:
            tokens = _tokenize(chunk.content)
            if not tokens:
                continue
            self._chunks.append(chunk)
            self._chunk_tokens.append(tokens)
            added += 1

        if added:
            # Defer the O(corpus) IDF rebuild to the first retrieval so that
            # repeated add_chunks calls don't recompute it per batch.
            self._idf_dirty = True

        return added

    def add_documents(
        self, documents: Iterable[str], source_prefix: str = "doc"
    ) -> int:
        """Index whole strings as chunks, one per non-blank document.

        Args:
            documents: Document texts.
            source_prefix: Source of every chunk and prefix of its id
                (``<prefix>-<n>``).

        Returns:
            How many chunks were added.
        """
        prepared_chunks: list[TextChunk] = []
        for index, content in enumerate(documents, start=1):
            text = content.strip()
            if not text:
                continue
            prepared_chunks.append(
                TextChunk(
                    chunk_id=f"{source_prefix}-{index}",
                    content=text,
                    source=source_prefix,
                )
            )
        return self.add_chunks(prepared_chunks)

    def retrieve_with_scores(
        self, query: str, top_k: int = 5, mmr_lambda: float | None = None
    ) -> list[tuple[TextChunk, float]]:
        """Rank chunks against the query, optionally diversifying the result.

        Args:
            query: The question.
            top_k: How many chunks to return.
            mmr_lambda: Relevance/diversity trade-off in ``[0, 1]``; 1.0 is pure
                relevance. ``None`` disables diversification.
                ``StandardTextRAGPipeline`` passes it only to backends whose
                signature accepts it.

        Returns:
            ``(chunk, score)`` pairs, best first.
        """
        if top_k <= 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        if self._idf_dirty:
            self._rebuild_idf()
            self._idf_dirty = False

        query_tf: Counter[str] = Counter(query_tokens)
        scored: list[tuple[TextChunk, float, set[str]]] = []

        for chunk, tokens in zip(self._chunks, self._chunk_tokens):
            score = self._score(query_tf, tokens)
            if score > 0:
                scored.append((chunk, score, set(tokens)))

        scored.sort(key=lambda item: item[1], reverse=True)

        if mmr_lambda is None or not scored:
            return [(chunk, score) for chunk, score, _ in scored[:top_k]]

        return self._mmr_select(scored, top_k=top_k, lambda_=mmr_lambda)

    @staticmethod
    def _mmr_select(
        scored: list[tuple[TextChunk, float, set[str]]],
        top_k: int,
        lambda_: float,
    ) -> list[tuple[TextChunk, float]]:
        """Maximal Marginal Relevance over the ranked candidates.

        Similarity between chunks is Jaccard over their token sets — the same
        surface signal the ranking uses, which keeps the trade-off interpretable
        and needs no second model.

        Args:
            scored: Candidates as ``(chunk, score, tokens)``, best first.
            top_k: How many to keep.
            lambda_: Weight on relevance; ``1 - lambda_`` weights novelty.

        Returns:
            The selected ``(chunk, score)`` pairs, in selection order.
        """
        weight = min(1.0, max(0.0, float(lambda_)))
        best_score = scored[0][1] or 1.0

        selected: list[tuple[TextChunk, float, set[str]]] = [scored[0]]
        remaining = list(scored[1:])

        while remaining and len(selected) < top_k:
            best_index = 0
            best_value = float("-inf")
            for index, (_, score, tokens) in enumerate(remaining):
                redundancy = max(
                    (
                        len(tokens & chosen) / len(tokens | chosen)
                        if (tokens | chosen)
                        else 0.0
                    )
                    for _, _, chosen in selected
                )
                value = weight * (score / best_score) - (1.0 - weight) * redundancy
                if value > best_value:
                    best_value = value
                    best_index = index
            selected.append(remaining.pop(best_index))

        return [(chunk, score) for chunk, score, _ in selected]

    def retrieve(self, query: str, top_k: int = 5) -> list[TextChunk]:
        """Return the ``top_k`` best chunks for ``query``, best first."""
        return [
            chunk for chunk, _ in self.retrieve_with_scores(query=query, top_k=top_k)
        ]

    def build_context(
        self, query: str, top_k: int = 4, separator: str = "\n\n---\n\n"
    ) -> str:
        """Join the contents of the ``top_k`` best chunks with ``separator``."""
        chunks = self.retrieve(query=query, top_k=top_k)
        return separator.join(chunk.content for chunk in chunks)

    def _rebuild_idf(self) -> None:
        """Recompute the IDF table and the average chunk length."""
        doc_count = len(self._chunk_tokens)
        document_frequency: Counter[str] = Counter()

        for tokens in self._chunk_tokens:
            document_frequency.update(set(tokens))

        # BM25 IDF in the always-non-negative form: the classic
        # log((N - df + 0.5) / (df + 0.5)) goes negative for a token present in
        # more than half the corpus, which on a small corpus would make common
        # domain words *subtract* from the score.
        self._idf = {
            token: math.log(
                1.0 + (doc_count - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }
        total_tokens = sum(len(tokens) for tokens in self._chunk_tokens)
        self._avg_doc_len = (total_tokens / doc_count) if doc_count else 0.0

    def _score(self, query_tf: Counter[str], document_tokens: list[str]) -> float:
        """Okapi BM25 score of one chunk against the query.

        Args:
            query_tf: Term frequencies of the query.
            document_tokens: Tokens of the candidate chunk.

        Returns:
            The BM25 score; higher is more relevant.
        """
        document_tf: Counter[str] = Counter(document_tokens)
        doc_len = len(document_tokens)
        avg_len = self._avg_doc_len or float(doc_len or 1)
        length_norm = self._BM25_K1 * (
            1.0 - self._BM25_B + self._BM25_B * (doc_len / avg_len)
        )

        score = 0.0
        for token in query_tf:
            term_frequency = document_tf.get(token, 0)
            if term_frequency == 0:
                continue
            idf = self._idf.get(token, 0.0)
            score += idf * (
                term_frequency
                * (self._BM25_K1 + 1.0)
                / (term_frequency + length_norm)
            )

        return score
