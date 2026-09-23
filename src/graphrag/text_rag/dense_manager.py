"""Dense chunk retrieval over a FAISS index cached on disk."""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.embeddings import Embeddings

from graphrag.text_rag.manager import TextChunk

logger = logging.getLogger("graphrag")


class _PrefixedEmbeddings(Embeddings):
    """LangChain ``Embeddings`` wrapper that prepends query/passage prefixes.

    Required for multilingual-e5-style models. An empty prefix is skipped.
    """

    def __init__(
        self,
        inner: Embeddings,
        query_prefix: str,
        passage_prefix: str,
    ) -> None:
        """Wrap ``inner`` with the given prefixes."""
        self._inner = inner
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages with the passage prefix."""
        if self._passage_prefix:
            texts = [f"{self._passage_prefix}{t}" for t in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the query prefix."""
        if self._query_prefix:
            text = f"{self._query_prefix}{text}"
        return self._inner.embed_query(text)


def _build_embeddings(
    model_name: str,
    query_prefix: str,
    passage_prefix: str,
    normalize: bool,
    device: str,
) -> _PrefixedEmbeddings:
    """Load a HuggingFace sentence encoder wrapped with the e5 prefixes.

    Args:
        model_name: HuggingFace model id.
        query_prefix: Prefix of queries.
        passage_prefix: Prefix of passages.
        normalize: L2-normalise the embeddings.
        device: ``"auto"`` (CUDA when available), ``"cpu"`` or ``"cuda"``.

    Returns:
        The prefixed embeddings.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    resolved_device = device
    if device == "auto":
        try:
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"

    inner = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": resolved_device},
        encode_kwargs={"normalize_embeddings": normalize},
    )
    return _PrefixedEmbeddings(inner, query_prefix=query_prefix, passage_prefix=passage_prefix)


def _embedding_env_signature(model_name: str) -> str:
    """Model name plus embedding-stack versions, for the index cache key.

    A library upgrade can change the embedding space, so it must invalidate
    cached FAISS indices.

    Args:
        model_name: Encoder id.

    Returns:
        ``model|sentence-transformers=<v>|transformers=<v>``.
    """
    parts = [model_name]
    for pkg in ("sentence-transformers", "transformers"):
        try:
            parts.append(f"{pkg}={importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{pkg}=none")
    return "|".join(parts)


def _fingerprint_hasher(model_name: str) -> "hashlib._Hash":
    """Start a SHA-256 corpus fingerprint seeded with the embedding stack."""
    h = hashlib.sha256()
    h.update(_embedding_env_signature(model_name).encode())
    return h


def _update_fingerprint(h: "hashlib._Hash", chunks: Iterable[TextChunk]) -> None:
    """Feed each chunk's id, content and source into the fingerprint."""
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.content.encode())
        # Provenance belongs in the key: otherwise a re-index that only changes
        # page tags reuses the cached index with its stale metadata, and
        # citations point at the old page labels.
        h.update((c.source or "").encode())


def _corpus_fingerprint(model_name: str, chunks: list[TextChunk]) -> str:
    """First 16 hex digits of the fingerprint of ``chunks`` under ``model_name``."""
    h = _fingerprint_hasher(model_name)
    _update_fingerprint(h, chunks)
    return h.hexdigest()[:16]


def _model_slug(model_name: str) -> str:
    """Make a model id usable in a directory name."""
    return model_name.replace("/", "-").replace(":", "-")


class DenseTextRAGManager:
    """FAISS-backed vector store manager with cosine similarity retrieval.

    Drop-in replacement for TextRAGManager — identical public interface.
    Embeddings use ``intfloat/multilingual-e5-base`` by default, which works
    for both English and Italian queries with ``query: ``/``passage: `` prefixes.

    The FAISS index is persisted to ``vector_index_dir`` keyed by a fingerprint
    of the model name, the embedding library versions and the chunks;
    subsequent runs with the same corpus skip re-encoding and load from cache.
    """

    def __init__(
        self,
        embedding_model: str = "intfloat/multilingual-e5-base",
        vector_index_dir: str = "artifacts/vector_index",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        normalize: bool = True,
        device: str = "auto",
    ) -> None:
        """Create an empty manager; the encoder is loaded on first use.

        Args:
            embedding_model: HuggingFace encoder id.
            vector_index_dir: Directory of the cached FAISS indices.
            query_prefix: Prefix of queries.
            passage_prefix: Prefix of passages.
            normalize: L2-normalise the embeddings.
            device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        """
        self._embedding_model = embedding_model
        self._vector_index_dir = Path(vector_index_dir)
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._normalize = normalize
        self._device = device
        self._chunks: list[TextChunk] = []
        self._store = None  # FAISS | None
        self._embeddings: _PrefixedEmbeddings | None = None
        # Incremental corpus fingerprint: avoids re-hashing the whole corpus
        # on every add_chunks call (O(n^2) for progressive indexing).
        self._hasher = _fingerprint_hasher(embedding_model)

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    @property
    def chunks(self) -> list[TextChunk]:
        """Every indexed chunk, for lookups that are not a ranking.

        Same contract as the lexical backend: following a citation back to the
        passage it names is a lookup by document and page, and a vector search
        on the question's words cannot guarantee to reach it.
        """
        return list(self._chunks)

    def clear(self) -> None:
        """Drop every chunk and the in-memory index; the disk cache is kept."""
        self._chunks.clear()
        self._store = None
        self._hasher = _fingerprint_hasher(self._embedding_model)

    def _get_embeddings(self) -> _PrefixedEmbeddings:
        """Return the encoder, loading it on first use."""
        if self._embeddings is None:
            self._embeddings = _build_embeddings(
                model_name=self._embedding_model,
                query_prefix=self._query_prefix,
                passage_prefix=self._passage_prefix,
                normalize=self._normalize,
                device=self._device,
            )
        return self._embeddings
    

    def add_chunks(self, chunks: Iterable[TextChunk]) -> int:
        """Add chunks and (re)build the index over every chunk added so far.

        The index is loaded from the disk cache when one exists for the
        current corpus fingerprint; otherwise all chunks are encoded and the
        index is saved there.

        Args:
            chunks: Chunks to add; blank ones are skipped.

        Returns:
            How many chunks were added.
        """
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        chunk_list = [c for c in chunks if c.content.strip()]
        if not chunk_list:
            return 0

        self._chunks.extend(chunk_list)
        _update_fingerprint(self._hasher, chunk_list)
        fingerprint = self._hasher.copy().hexdigest()[:16]
        cache_dir = self._vector_index_dir / f"{_model_slug(self._embedding_model)}-{fingerprint}"
        embeddings = self._get_embeddings()

        if cache_dir.exists():
            logger.info("DenseTextRAGManager: loading index from cache %s", cache_dir)
            self._store = FAISS.load_local(
                str(cache_dir),
                embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        else:
            
            logger.info(
                "DenseTextRAGManager: building FAISS index for %d chunks", len(self._chunks)
            )
            texts = [c.content for c in self._chunks]
            metadatas = [
                {"chunk_id": c.chunk_id, "source": c.source or ""}
                for c in self._chunks
            ]
            self._store = FAISS.from_texts(
                texts,
                embeddings,
                metadatas=metadatas,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(cache_dir))
            logger.info("DenseTextRAGManager: index saved to %s", cache_dir)

        return len(chunk_list)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        mmr_lambda: float | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[TextChunk, float]]:
        """Retrieve the top chunks, optionally diversified with MMR.

        Args:
            query: The retrieval query.
            top_k: How many chunks to return.
            mmr_lambda: ``None`` keeps pure similarity ranking. A value in
                ``[0, 1]`` switches to Maximal Marginal Relevance: 1.0 is again
                pure similarity, lower values trade similarity for coverage.
            fetch_k: Candidate pool MMR selects from; never less than
                ``4 * top_k``.

        Returns:
            ``(chunk, score)`` pairs, most relevant first.
        """
        if self._store is None or top_k <= 0:
            return []

        if mmr_lambda is None:
            hits = self._store.similarity_search_with_score(query, k=top_k)
        else:
            pool = max(int(fetch_k or 0), top_k * 4)
            embedding = self._get_embeddings().embed_query(query)
            hits = self._store.max_marginal_relevance_search_with_score_by_vector(
                embedding,
                k=top_k,
                fetch_k=pool,
                lambda_mult=max(0.0, min(1.0, float(mmr_lambda))),
            )
        # MAX_INNER_PRODUCT: score is inner product (== cosine for normalised embs),
        # higher means more similar. Results already ordered descending by FAISS.
        result: list[tuple[TextChunk, float]] = []
        for doc, score in hits:
            meta = doc.metadata
            chunk = TextChunk(
                chunk_id=meta.get("chunk_id", ""),
                content=doc.page_content,
                source=meta.get("source") or None,
            )
            result.append((chunk, float(score)))

        return result

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mmr_lambda: float | None = None,
        fetch_k: int | None = None,
    ) -> list[TextChunk]:
        """Like :meth:`retrieve_with_scores`, without the scores."""
        return [
            chunk
            for chunk, _ in self.retrieve_with_scores(
                query=query, top_k=top_k, mmr_lambda=mmr_lambda, fetch_k=fetch_k
            )
        ]

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

    def build_context(
        self, query: str, top_k: int = 4, separator: str = "\n\n---\n\n"
    ) -> str:
        """Join the contents of the ``top_k`` best chunks with ``separator``."""
        chunks = self.retrieve(query=query, top_k=top_k)
        return separator.join(chunk.content for chunk in chunks)
