"""Unit tests for the dense (FAISS) text retriever and its on-disk passage-vector cache."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings

from graphrag.text_rag.dense_manager import DenseTextRAGManager, _model_slug
from graphrag.text_rag.manager import TextChunk


# ---------------------------------------------------------------------------
# Fake embeddings: deterministic unit vectors, proper Embeddings subclass
# so LangChain FAISS calls embed_query (not the object as a callable).
# ---------------------------------------------------------------------------

class _FakeEmbeddings(Embeddings):
    """4-dim normalised embeddings keyed by text length for determinism."""

    def __init__(self) -> None:
        self.embed_documents_call_count = 0
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_call_count += 1
        self.embedded.extend(texts)
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
def test_vectors_persisted_to_cache_dir(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    mgr = DenseTextRAGManager(
        embedding_model="test/model",
        vector_index_dir=str(tmp_path),
    )
    mgr.add_chunks(_make_chunks(3))

    stores = list((tmp_path / "passages").glob(f"{_model_slug('test/model')}-*.sqlite"))
    assert len(stores) == 1, "passage vectors not saved to the cache directory"


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
def test_adding_chunks_encodes_only_the_new_ones(mock_build, tmp_path):
    fake = _make_fake_embeddings()
    mock_build.return_value = fake
    chunks = _make_chunks(5)
    mgr = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    mgr.add_chunks(chunks[:3])
    fake.embedded.clear()

    mgr.add_chunks(chunks[3:])

    assert fake.embedded == [c.content for c in chunks[3:]]
    assert mgr.size == 5
    assert {c.chunk_id for c in mgr.retrieve("chunk", top_k=5)} == {c.chunk_id for c in chunks}


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_a_grown_corpus_encodes_only_its_new_passages(mock_build, tmp_path):
    # The demo rebuilds its index in a new process over the whole corpus.
    fake = _make_fake_embeddings()
    mock_build.return_value = fake
    chunks = _make_chunks(5)
    DenseTextRAGManager(vector_index_dir=str(tmp_path)).add_chunks(chunks[:4])
    fake.embedded.clear()

    DenseTextRAGManager(vector_index_dir=str(tmp_path)).add_chunks(chunks)

    assert fake.embedded == [chunks[4].content]


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_an_index_built_in_pieces_answers_like_one_built_at_once(mock_build, tmp_path):
    mock_build.return_value = _make_fake_embeddings()
    chunks = _make_chunks(8)
    whole = DenseTextRAGManager(vector_index_dir=str(tmp_path / "whole"))
    whole.add_chunks(chunks)
    pieces = DenseTextRAGManager(vector_index_dir=str(tmp_path / "pieces"))
    for start in range(0, 8, 3):
        pieces.add_chunks(chunks[start : start + 3])

    for query in ("chunk", "Document chunk number 4. " * 14, "x"):
        assert [
            (c.chunk_id, round(s, 6)) for c, s in whole.retrieve_with_scores(query, top_k=8)
        ] == [(c.chunk_id, round(s, 6)) for c, s in pieces.retrieve_with_scores(query, top_k=8)]
    assert whole.similarity("chunk", chunks) == pieces.similarity("chunk", chunks)


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_new_provenance_is_served_without_encoding_again(mock_build, tmp_path):
    # Citations are built from the chunk's source: a re-index that only
    # changes page tags or ids must not serve the old ones.
    fake = _make_fake_embeddings()
    mock_build.return_value = fake
    chunks = _make_chunks(2)
    DenseTextRAGManager(vector_index_dir=str(tmp_path)).add_chunks(chunks)
    fake.embedded.clear()
    relabelled = [
        TextChunk(chunk_id=f"new-{c.chunk_id}", content=c.content, source=f"new#{c.source}")
        for c in chunks
    ]

    mgr = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    mgr.add_chunks(relabelled)

    assert fake.embedded == []
    hits = mgr.retrieve("chunk", top_k=2)
    assert {c.chunk_id for c in hits} == {c.chunk_id for c in relabelled}
    assert {c.source for c in hits} == {c.source for c in relabelled}


@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_another_encoder_does_not_reuse_the_vectors(mock_build, tmp_path):
    fake = _make_fake_embeddings()
    mock_build.return_value = fake
    chunks = _make_chunks(2)
    DenseTextRAGManager(embedding_model="model-a", vector_index_dir=str(tmp_path)).add_chunks(chunks)
    fake.embedded.clear()

    DenseTextRAGManager(embedding_model="model-b", vector_index_dir=str(tmp_path)).add_chunks(chunks)

    assert len(fake.embedded) == 2


def test_another_passage_prefix_does_not_reuse_the_vectors(tmp_path):
    # The prefix is part of what the encoder reads, so it is part of the key.
    from graphrag.text_rag.dense_manager import _PrefixedEmbeddings

    fake = _make_fake_embeddings()
    chunks = _make_chunks(2)
    for prefix in ("passage: ", "passage: ", ""):
        with patch(
            "graphrag.text_rag.dense_manager._build_embeddings",
            return_value=_PrefixedEmbeddings(fake, "query: ", prefix),
        ):
            DenseTextRAGManager(
                vector_index_dir=str(tmp_path), passage_prefix=prefix
            ).add_chunks(chunks)

    assert fake.embedded == [f"passage: {c.content}" for c in chunks] + [
        c.content for c in chunks
    ]


@patch("graphrag.text_rag.dense_manager._LOOKUP_SLICE", 3)
@patch("graphrag.text_rag.dense_manager._ENCODE_SLICE", 4)
@patch("graphrag.text_rag.dense_manager._build_embeddings")
def test_an_interrupted_encoding_resumes_from_the_last_finished_slice(mock_build, tmp_path):
    class _Failing(_FakeEmbeddings):
        """Fails on its third call, as a killed or crashed run would stop."""

        def embed_documents(self, texts):
            if self.embed_documents_call_count == 2:
                raise RuntimeError("encoder gone")
            return super().embed_documents(texts)

    chunks = _make_chunks(10)
    mock_build.return_value = _Failing()
    with pytest.raises(RuntimeError):
        DenseTextRAGManager(vector_index_dir=str(tmp_path)).add_chunks(chunks)

    fake = _make_fake_embeddings()
    mock_build.return_value = fake
    resumed = DenseTextRAGManager(vector_index_dir=str(tmp_path))
    resumed.add_chunks(chunks)

    # Two slices of four were committed before the failure.
    assert fake.embedded == [c.content for c in chunks[8:]]
    assert resumed.size == 10


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
