"""Stage 2 calls GLiNER in batches, not one chunk at a time.

Batching yields the same span sets; the only difference is confidence noise
from padding (below 1e-4, with no entity that close to the 0.45 threshold — so
no span can cross it). These tests keep the wiring honest without a GPU: the
batch call is used when available, its results are handed back to the right
chunks, and a failing batch falls back instead of losing the chunks.
"""

from __future__ import annotations

from typing import Any

import pytest

from kg_pipeline.models.types import ChunkRecord
from kg_pipeline.stages import ner


class _FakeGLiNER:
    """Returns one entity per text, naming the text it came from."""

    def __init__(self, *, fail_batches: bool = False) -> None:
        self.batch_calls: list[list[str]] = []
        self.single_calls: list[str] = []
        self.fail_batches = fail_batches

    def _entity(self, text: str) -> dict[str, Any]:
        return {
            "text": text.strip(),
            "label": "concept",
            "start": 0,
            "end": len(text),
            "score": 0.9,
        }

    def inference(self, texts, labels, **kwargs) -> list[list[dict[str, Any]]]:
        if self.fail_batches:
            raise RuntimeError("CUDA out of memory")
        self.batch_calls.append(list(texts))
        assert kwargs["flat_ner"] is True, "must keep predict_entities' defaults"
        assert kwargs["multi_label"] is False
        return [[self._entity(text)] for text in texts]

    def predict_entities(self, text, labels, **kwargs) -> list[dict[str, Any]]:
        self.single_calls.append(text)
        return [self._entity(text)]

    def to(self, _device):
        return self


class _OldGLiNER(_FakeGLiNER):
    """Older gliner releases expose only predict_entities."""

    inference = None


def _chunks(count: int) -> list[ChunkRecord]:
    """`count` chunks of one document."""
    return [
        ChunkRecord(
            doc_id="doc",
            filename="doc.pdf",
            chunk_id=f"chunk_{i:03d}",
            page_range="1-1",
            section_title="",
            chunk_index=i + 1,  # 1-based in the model
            text=f"testo del chunk {i}",
        )
        for i in range(count)
    ]


@pytest.fixture
def patched(monkeypatch):
    """An installer that makes `GLiNER.from_pretrained` return a fake."""
    def _install(model: _FakeGLiNER) -> _FakeGLiNER:
        monkeypatch.setattr(
            ner.GLiNER, "from_pretrained", staticmethod(lambda _name: model)
        )
        return model

    return _install


def test_chunks_are_sent_in_batches_not_one_at_a_time(patched):
    model = patched(_FakeGLiNER())
    result = ner.run_ner(_chunks(10), "fake", ["Concept"], 0.45, batch_size=4)

    assert [len(call) for call in model.batch_calls] == [4, 4, 2]
    assert not model.single_calls
    assert len(result) == 10


def test_each_chunk_keeps_its_own_entities(patched):
    """The batch answer is a list per input; pairing it up wrong is silent."""
    patched(_FakeGLiNER())
    result = ner.run_ner(_chunks(7), "fake", ["Concept"], 0.45, batch_size=3)

    for index in range(7):
        entities = result[f"chunk_{index:03d}"]
        assert [entity.text_span for entity in entities] == [f"testo del chunk {index}"]


def test_labels_are_mapped_back_to_their_configured_casing(patched):
    patched(_FakeGLiNER())
    result = ner.run_ner(_chunks(1), "fake", ["Concept"], 0.45, batch_size=4)

    assert result["chunk_000"][0].entity_label == "Concept"


def test_a_failed_batch_falls_back_to_one_chunk_at_a_time(patched):
    model = patched(_FakeGLiNER(fail_batches=True))
    result = ner.run_ner(_chunks(5), "fake", ["Concept"], 0.45, batch_size=5)

    assert len(model.single_calls) == 5
    assert len(result) == 5


def test_an_older_gliner_without_inference_still_works(patched):
    model = patched(_OldGLiNER())
    result = ner.run_ner(_chunks(5), "fake", ["Concept"], 0.45, batch_size=5)

    assert len(model.single_calls) == 5
    assert len(result) == 5


def test_batch_size_one_restores_the_old_behaviour(patched):
    model = patched(_FakeGLiNER())
    ner.run_ner(_chunks(3), "fake", ["Concept"], 0.45, batch_size=1)

    assert not model.batch_calls
    assert len(model.single_calls) == 3


def test_the_batch_size_comes_from_the_environment(patched, monkeypatch):
    monkeypatch.setenv("KG_NER_BATCH_SIZE", "2")
    model = patched(_FakeGLiNER())
    ner.run_ner(_chunks(5), "fake", ["Concept"], 0.45)

    assert [len(call) for call in model.batch_calls] == [2, 2]
    assert model.single_calls == ["testo del chunk 4"], "a lone chunk needs no batch"
