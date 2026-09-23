"""Factory for text-only retrieval pipelines."""

from __future__ import annotations

from graphrag.text_rag.manager import TextRAGManager
from graphrag.text_rag.pipeline import StandardTextRAGPipeline

_DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"
_DEFAULT_QUERY_PREFIX = "query: "
_DEFAULT_PASSAGE_PREFIX = "passage: "
_DEFAULT_VECTOR_INDEX_DIR = "artifacts/vector_index"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180
DEFAULT_MIN_CHUNK_CHARS = 80


def make_text_pipeline(
    backend: str = "tfidf",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    *,
    embedding_model: str = _DEFAULT_DENSE_MODEL,
    vector_index_dir: str = _DEFAULT_VECTOR_INDEX_DIR,
    query_prefix: str = _DEFAULT_QUERY_PREFIX,
    passage_prefix: str = _DEFAULT_PASSAGE_PREFIX,
    normalize: bool = True,
    device: str = "auto",
) -> StandardTextRAGPipeline:
    """Create a StandardTextRAGPipeline with the given retrieval backend.

    Args:
        backend: ``"tfidf"`` for lexical BM25 retrieval (default, no extra
            dependencies), ``"dense"`` for cosine similarity over FAISS with
            HuggingFace embeddings.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        min_chunk_chars: Chunks shorter than this are discarded.
        embedding_model: HuggingFace model ID. Ignored for ``"tfidf"``.
        vector_index_dir: Directory for persisted FAISS index cache. Ignored for ``"tfidf"``.
        query_prefix: Prepended to queries at embed time (e.g. ``"query: "`` for e5 models).
        passage_prefix: Prepended to passages at embed time (e.g. ``"passage: "`` for e5 models).
        normalize: Whether to L2-normalise embeddings before indexing.
        device: ``"auto"`` (cuda if available, else cpu), ``"cpu"``, or ``"cuda"``.

    Returns:
        Fully configured ``StandardTextRAGPipeline`` ready for ``index_paths`` / ``retrieve``.

    Raises:
        ValueError: If ``backend`` is not ``"tfidf"`` or ``"dense"``.
    """
    if backend == "tfidf":
        retriever = TextRAGManager()
    elif backend == "dense":
        from graphrag.text_rag.dense_manager import DenseTextRAGManager

        retriever = DenseTextRAGManager(
            embedding_model=embedding_model,
            vector_index_dir=vector_index_dir,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            normalize=normalize,
            device=device,
        )
    else:
        raise ValueError(f"Unknown text retrieval backend '{backend}'. Use 'tfidf' or 'dense'.")

    return StandardTextRAGPipeline(
        retriever=retriever,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
