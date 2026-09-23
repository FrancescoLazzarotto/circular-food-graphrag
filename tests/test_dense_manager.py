"""Unit tests for the dense (FAISS) text retriever and its on-disk index cache."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings

from graphrag.text_rag.dense_manager import DenseTextRAGManager, _corpus_fingerprint, _model_slug
from graphrag.text_rag.manager import TextChunk


# ---------------------------------------------------------------------------
# Fake embeddings: deterministic unit vectors, proper Embeddings subclass
# so LangChain FAISS calls embed_query (not the object as a callable).
# ---------------------------------------------------------------------------

class _FakeEmbeddings(Embeddings):
    """4-dim normalised embeddings keyed by text length for determinism."""

    def __init__(self) -> None:
        self.embed_documents_call_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_call_count += 1
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> list[float]:
        raw = [len(text) / 100.0, 0.1, 0.1, 0.1]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


def _make_fake_embeddings() -> _FakeEmbeddings:
    """A fresh fake embedder."""
    return _FakeEmbeddings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 5) -> list[TextChunk]:
    """`n` text chunks of different lengths, one per document."""
    return [
        TextChunk(
            chunk_id=f"c{i:04d}",
            content=f"Document chunk number {i}. " * (10 + i),  # varying lengths
            source=f"doc_{i}.txt",
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_add_chunks_returns_count(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    mgr = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    chunks = _make_chunks(5)
    added = mgr.add_chunks(chunks)
    assert added == 5
    assert mgr.size == 5


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_retrieve_with_scores_sorted_descending(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    mgr = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    mgr.add_chunks(_make_chunks(5))

    results = mgr.retrieve_with_scores("chunk number", top_k=3)
    assert len(results) == 3
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "scores must be descending"


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_retrieve_returns_text_chunks(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    mgr = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    mgr.add_chunks(_make_chunks(3))

    chunks = mgr.retrieve("chunk", top_k=2)
    assert len(chunks) == 2
    for c in chunks:
        assert isinstance(c, TextChunk)
        assert c.chunk_id
        assert c.content


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_empty_retrieve_before_index(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    mgr = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    assert mgr.retrieve_with_scores("anything", top_k=5) == []


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_clear_resets_state(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    mgr = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    mgr.add_chunks(_make_chunks(3))
    assert mgr.size == 3
    mgr.clear()
    assert mgr.size == 0
    assert mgr.retrieve_with_scores("anything") == []


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_index_persisted_to_cache_dir(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    chunks = _make_chunks(3)
    mgr = DenseTextRAGManager(
        embedding_model="test/model",
        vector_index_dir=str(tmp_path),
    )
    mgr.add_chunks(chunks)

    fp = _corpus_fingerprint("test/model", chunks)
    slug = _model_slug("test/model")
    cache_dir = tmp_path / f"{slug}-{fp}"
    assert cache_dir.exists(), "FAISS index not saved to cache directory"


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_index_reloaded_from_cache(mock_build, tmp_path):
    """Second add_chunks on same corpus should hit cache (embed_documents called once)."""
    shared_emb = _make_fake_embeddings()
    mock_build.return_value = shared_emb

    chunks = _make_chunks(3)

    mgr1 = DenseTextRAGManager(embedding_model="test/model", vector_index_dir=str(tmp_path))
    mgr1.add_chunks(chunks)
    first_call_count = shared_emb.embed_documents_call_count

    # Second manager, same corpus → should load from cache, not re-embed
    mgr2 = DenseTextRAGManager(embedding_model="test/model", vector_index_dir=str(tmp_path))
    mgr2.add_chunks(chunks)
    second_call_count = shared_emb.embed_documents_call_count

    assert second_call_count == first_call_count, (
        "embed_documents should not be called again when cache hit"
    )


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_build_context_returns_str(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    mgr = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    mgr.add_chunks(_make_chunks(3))
    ctx = mgr.build_context("chunk", top_k=2)
    assert isinstance(ctx, str)
    assert len(ctx) > 0


@pytest.mark.parametrize("model_name,expected_slug", [
    ("intfloat/multilingual-e5-base", "intfloat-multilingual-e5-base"),
    ("BAAI/bge-m3", "BAAI-bge-m3"),
    ("model:v1", "model-v1"),
])
def test_model_slug(model_name, expected_slug):
    assert _model_slug(model_name) == expected_slug


def test_corpus_fingerprint_changes_with_content():
    chunks_a = [TextChunk(chunk_id="c1", content="hello world", source=None)]
    chunks_b = [TextChunk(chunk_id="c1", content="different content", source=None)]
    fp_a = _corpus_fingerprint("model", chunks_a)
    fp_b = _corpus_fingerprint("model", chunks_b)
    assert fp_a != fp_b


def test_corpus_fingerprint_changes_with_model():
    chunks = [TextChunk(chunk_id="c1", content="hello world", source=None)]
    fp_a = _corpus_fingerprint("model-a", chunks)
    fp_b = _corpus_fingerprint("model-b", chunks)
    assert fp_a != fp_b
