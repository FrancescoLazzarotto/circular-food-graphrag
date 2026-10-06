"""Dense chunk retrieval over a FAISS index built from passage vectors cached on disk."""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable

import numpy as np
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


def _model_slug(model_name: str) -> str:
    """Make a model id usable in a directory name."""
    return model_name.replace("/", "-").replace(":", "-")


# Passages encoded per call, and per commit to the vector cache: an interrupted
# indexing run keeps every slice it finished.
_ENCODE_SLICE = 256
# Keys per lookup query, below SQLite's limit on bound parameters.
_LOOKUP_SLICE = 900


class _PassageVectors:
    """Passage vectors on disk, keyed by the exact text that was encoded.

    One SQLite file per encoder and embedding stack. A passage is encoded once,
    whichever corpus, run or document id it comes with, so an index over a
    corpus that grew by one document encodes only that document's passages.
    Ids and sources are never stored here: they come from the chunks of the
    current build, so a re-index that only changes provenance serves the new
    labels.

    Attributes:
        path: The SQLite file.
    """

    def __init__(self, path: Path) -> None:
        """Create the cache at ``path`` when it does not exist yet."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS passages (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        """A connection for one operation.

        Never kept open: a connection belongs to the thread that opened it, and
        the demo builds and rebuilds its index from more than one thread. The
        long timeout covers an indexing job and the demo writing at once.
        """
        return sqlite3.connect(str(self.path), timeout=60)

    @staticmethod
    def key(text: str) -> str:
        """Cache key of an encoded text, prefix included."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Look up cached vectors.

        Args:
            keys: Keys from :meth:`key`.

        Returns:
            ``{key: vector}`` for the keys the cache holds; the others are absent.
        """
        found: dict[str, np.ndarray] = {}
        with closing(self._connect()) as db:
            for start in range(0, len(keys), _LOOKUP_SLICE):
                batch = keys[start : start + _LOOKUP_SLICE]
                marks = ",".join("?" * len(batch))
                for key, blob in db.execute(
                    f"SELECT key, vector FROM passages WHERE key IN ({marks})", batch
                ):
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put(self, items: list[tuple[str, list[float]]]) -> None:
        """Store vectors in one transaction; a key already present is kept.

        Args:
            items: ``(key, vector)`` pairs.
        """
        with closing(self._connect()) as db, db:
            db.executemany(
                "INSERT OR IGNORE INTO passages (key, vector) VALUES (?, ?)",
                [(key, np.asarray(vector, dtype=np.float32).tobytes()) for key, vector in items],
            )


class DenseTextRAGManager:
    """FAISS-backed vector store manager with cosine similarity retrieval.

    Drop-in replacement for TextRAGManager — identical public interface.
    Embeddings use ``intfloat/multilingual-e5-base`` by default, which works
    for both English and Italian queries with ``query: ``/``passage: `` prefixes.

    Passage vectors are cached under ``vector_index_dir``, one cache per model,
    embedding library versions and normalisation; only passages the cache does
    not hold are encoded, and the FAISS index is assembled from the vectors.
    Adding chunks extends the index instead of rebuilding it.
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
            vector_index_dir: Directory of the passage-vector cache.
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
        # Index row of each chunk id, built on the first `similarity` call.
        self._rows: dict[str, int] | None = None
        self._embeddings: _PrefixedEmbeddings | None = None
        self._vectors: _PassageVectors | None = None

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
        self._rows = None

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

    def _get_vectors(self) -> _PassageVectors:
        """Return the passage-vector cache of this encoder, opening it on first use.

        The file name carries the model, the embedding library versions and
        the normalisation, so an upgrade or another encoder never reads
        vectors from a different embedding space. The device is left out on
        purpose: vectors encoded on CPU and on GPU differ only by rounding.
        """
        if self._vectors is None:
            signature = f"{_embedding_env_signature(self._embedding_model)}|normalize={self._normalize}"
            digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
            self._vectors = _PassageVectors(
                self._vector_index_dir
                / "passages"
                / f"{_model_slug(self._embedding_model)}-{digest}.sqlite"
            )
        return self._vectors

    def _passage_vectors(self, texts: list[str]) -> list[np.ndarray]:
        """Vectors of ``texts``, encoding only those the cache does not hold.

        Args:
            texts: Passage texts, without the passage prefix.

        Returns:
            One vector per text, in order.
        """
        cache = self._get_vectors()
        keys = [cache.key(f"{self._passage_prefix}{text}") for text in texts]
        unique = list(dict.fromkeys(keys))
        found = cache.get(unique)
        missing = list({key: text for key, text in zip(keys, texts) if key not in found}.items())
        if missing:
            logger.info(
                "DenseTextRAGManager: %d of %d distinct passages cached, encoding %d into %s",
                len(found),
                len(unique),
                len(missing),
                cache.path,
            )
            embeddings = self._get_embeddings()
            for start in range(0, len(missing), _ENCODE_SLICE):
                batch = missing[start : start + _ENCODE_SLICE]
                vectors = embeddings.embed_documents([text for _, text in batch])
                cache.put([(key, vector) for (key, _), vector in zip(batch, vectors)])
                found.update(
                    (key, np.asarray(vector, dtype=np.float32))
                    for (key, _), vector in zip(batch, vectors)
                )
                logger.info(
                    "DenseTextRAGManager: encoded %d/%d new passages",
                    min(start + _ENCODE_SLICE, len(missing)),
                    len(missing),
                )
        return [found[key] for key in keys]

    def add_chunks(self, chunks: Iterable[TextChunk]) -> int:
        """Add chunks to the index, encoding only passages never encoded before.

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

        texts = [c.content for c in chunk_list]
        pairs = list(zip(texts, self._passage_vectors(texts)))
        metadatas = [{"chunk_id": c.chunk_id, "source": c.source or ""} for c in chunk_list]
        if self._store is None:
            self._store = FAISS.from_embeddings(
                pairs,
                self._get_embeddings(),
                metadatas=metadatas,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        else:
            self._store.add_embeddings(pairs, metadatas=metadatas)
        self._chunks.extend(chunk_list)
        self._rows = None
        logger.info("DenseTextRAGManager: %d chunks indexed", len(self._chunks))
        return len(chunk_list)

    def similarity(self, query: str, chunks: Iterable[TextChunk]) -> list[float]:
        """Similarity of ``query`` to each of ``chunks``, from the stored vectors.

        For ordering passages found some other way: the vectors are read back
        from the index, so nothing is encoded but the query.

        Args:
            query: The retrieval query.
            chunks: Indexed chunks.

        Returns:
            One score per chunk, in order; 0.0 for a chunk not in the index.
        """
        chunk_list = list(chunks)
        if self._store is None or not chunk_list:
            return [0.0] * len(chunk_list)
        if self._rows is None:
            self._rows = {
                self._store.docstore.search(doc_id).metadata.get("chunk_id", ""): row
                for row, doc_id in self._store.index_to_docstore_id.items()
            }
        vector = self._get_embeddings().embed_query(query)
        scores: list[float] = []
        for chunk in chunk_list:
            row = self._rows.get(chunk.chunk_id)
            if row is None:
                scores.append(0.0)
                continue
            stored = self._store.index.reconstruct(row)
            scores.append(float(sum(a * b for a, b in zip(vector, stored))))
        return scores

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
