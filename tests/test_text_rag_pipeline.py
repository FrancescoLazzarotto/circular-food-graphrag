"""How the corpus is cut before it is indexed, and what comes back out.

This is the dense/text half of retrieval, the one every campaign measures
with. `_split_into_chunks` decides how the corpus is cut before it is
indexed, so a defect there moves every number produced after it.

The real `TextRAGManager` is used where it is cheap (BM25 over a handful of
strings) and faked where the point is what the pipeline asks of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from graphrag.text_rag.manager import TextChunk, TextRAGManager
from graphrag.text_rag.pipeline import (
    RetrievedTextChunk,
    StandardTextRAGPipeline,
    _accepts_mmr,
    _normalize_text,
)


class _Recorder:
    """A retriever that records what it was given and what it was asked."""

    def __init__(self, results: list[tuple[TextChunk, float]] | None = None) -> None:
        self.chunks: list[TextChunk] = []
        self.results = results or []
        self.asked: list[dict[str, Any]] = []
        self.cleared = 0

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks) -> int:
        added = list(chunks)
        self.chunks.extend(added)
        return len(added)

    def clear(self) -> None:
        self.cleared += 1
        self.chunks = []

    def retrieve_with_scores(self, query: str, top_k: int = 5, **kwargs: Any):
        self.asked.append({"query": query, "top_k": top_k, **kwargs})
        return list(self.results)[:top_k]


class _NoMMRRecorder(_Recorder):
    """Signature without `mmr_lambda` and without `**kwargs`, like TF-IDF."""

    def retrieve_with_scores(self, query: str, top_k: int = 5):
        self.asked.append({"query": query, "top_k": top_k})
        return list(self.results)[:top_k]


class _MMRRecorder(_Recorder):
    """Recorder whose signature names `mmr_lambda` and `fetch_k`."""

    def retrieve_with_scores(
        self, query: str, top_k: int = 5, mmr_lambda: float | None = None,
        fetch_k: int | None = None,
    ):
        self.asked.append(
            {"query": query, "top_k": top_k, "mmr_lambda": mmr_lambda, "fetch_k": fetch_k}
        )
        return list(self.results)[:top_k]


def _pipeline(retriever=None, **kwargs: Any) -> StandardTextRAGPipeline:
    """A `StandardTextRAGPipeline` over `retriever`, a `_Recorder` by default."""
    return StandardTextRAGPipeline(retriever=retriever or _Recorder(), **kwargs)


# --- the settings that decide the cut --------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"chunk_size": 127}, "chunk_size"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"chunk_size": 200, "chunk_overlap": 200}, "smaller than chunk_size"),
        ({"chunk_size": 200, "chunk_overlap": 300}, "smaller than chunk_size"),
        ({"min_chunk_chars": 0}, "min_chunk_chars"),
    ],
)
def test_a_setting_that_would_produce_nonsense_is_refused(kwargs, message):
    # An overlap at or above the size makes the window never advance.
    with pytest.raises(ValueError, match=message):
        StandardTextRAGPipeline(retriever=_Recorder(), **kwargs)


def test_the_defaults_are_the_ones_the_campaigns_ran_with():
    pipeline = _pipeline()

    assert (pipeline.chunk_size, pipeline.chunk_overlap, pipeline.min_chunk_chars) == (
        1200,
        180,
        80,
    )


# --- cutting the corpus ----------------------------------------------------


def test_text_shorter_than_the_window_is_one_chunk():
    pipeline = _pipeline(chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    assert pipeline._split_into_chunks("a" * 150) == ["a" * 150]


def test_text_exactly_the_window_is_still_one_chunk():
    pipeline = _pipeline(chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    assert len(pipeline._split_into_chunks("a" * 200)) == 1


def test_text_below_the_minimum_is_dropped_rather_than_indexed():
    # A page of nothing but a header would otherwise become a chunk that can
    # be retrieved and cited.
    pipeline = _pipeline(chunk_size=200, chunk_overlap=20, min_chunk_chars=50)

    assert pipeline._split_into_chunks("a" * 49) == []


def test_consecutive_chunks_overlap_by_exactly_the_configured_amount():
    # The overlap is what keeps a sentence spanning a boundary retrievable
    # from either side.
    pipeline = _pipeline(chunk_size=160, chunk_overlap=40, min_chunk_chars=10)
    text = "".join(f"{i:03d}-" for i in range(100))  # 400 chars, no repeats

    chunks = pipeline._split_into_chunks(text)

    assert chunks[0][-40:] == chunks[1][:40]


def test_the_window_advances_by_size_minus_overlap():
    pipeline = _pipeline(chunk_size=160, chunk_overlap=40, min_chunk_chars=10)
    text = "".join(f"{i:03d}-" for i in range(100))  # 400 chars, no repeats

    chunks = pipeline._split_into_chunks(text)

    assert text.index(chunks[1]) == 120  # 160 - 40


def test_the_whole_text_is_covered_with_no_hole():
    pipeline = _pipeline(chunk_size=160, chunk_overlap=40, min_chunk_chars=10)
    text = "".join(f"{i:03d}-" for i in range(100))  # 400 chars, no repeats

    chunks = pipeline._split_into_chunks(text)

    rebuilt = chunks[0]
    for chunk in chunks[1:]:
        rebuilt += chunk[40:]
    assert rebuilt == text


def test_a_trailing_fragment_below_the_minimum_is_not_indexed():
    pipeline = _pipeline(chunk_size=160, chunk_overlap=10, min_chunk_chars=50)
    text = "".join(f"{i:03d}-" for i in range(76))[:305]

    chunks = pipeline._split_into_chunks(text)

    assert all(len(chunk) >= 50 for chunk in chunks)


def test_whitespace_is_collapsed_before_anything_is_measured():
    # Otherwise a PDF page's line breaks count toward the window and the
    # chunk boundary lands somewhere different than the settings say.
    assert _normalize_text("  due\n\n  parole\t ") == "due parole"


def test_a_page_of_only_whitespace_produces_nothing():
    pipeline = _pipeline(chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    assert pipeline._split_into_chunks("   \n\t  ") == []


# --- indexing --------------------------------------------------------------


def test_each_chunk_is_named_after_its_document_section_and_position(tmp_path):
    doc = tmp_path / "uno.txt"
    doc.write_text("x" * 500, encoding="utf-8")
    recorder = _Recorder()
    pipeline = _pipeline(recorder, chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    pipeline.index_paths([doc])

    ids = [c.chunk_id for c in recorder.chunks]
    assert ids[0] == "d0001-s0001-c0001"
    assert ids == sorted(ids)


def test_every_chunk_carries_the_source_a_citation_is_built_from(tmp_path):
    doc = tmp_path / "uno.txt"
    doc.write_text("x" * 500, encoding="utf-8")
    recorder = _Recorder()
    pipeline = _pipeline(recorder, chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    pipeline.index_paths([doc])

    assert all(c.source.endswith(f"#chunk={i}") for i, c in enumerate(recorder.chunks, 1))
    assert all(str(doc) in c.source for c in recorder.chunks)


def test_indexing_reports_how_many_chunks_landed(tmp_path):
    doc = tmp_path / "uno.txt"
    doc.write_text("x" * 500, encoding="utf-8")
    pipeline = _pipeline(chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    assert pipeline.index_paths([doc]) == pipeline.indexed_chunks


def test_a_directory_is_walked_for_the_suffixes_that_carry_text(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "uno.txt").write_text("a" * 200, encoding="utf-8")
    (tmp_path / "sub" / "due.md").write_text("b" * 200, encoding="utf-8")
    (tmp_path / "tre.png").write_bytes(b"\x89PNG")
    recorder = _Recorder()
    pipeline = _pipeline(recorder, chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    pipeline.index_directory(tmp_path)

    sources = " ".join(c.source for c in recorder.chunks)
    assert "uno.txt" in sources and "due.md" in sources
    assert "tre.png" not in sources


def test_a_narrower_discovery_pattern_is_honoured(tmp_path):
    (tmp_path / "uno.txt").write_text("a" * 200, encoding="utf-8")
    (tmp_path / "due.md").write_text("b" * 200, encoding="utf-8")
    recorder = _Recorder()
    pipeline = _pipeline(recorder, chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    pipeline.index_paths([tmp_path], discovery_patterns=["*.md"])

    assert all("due.md" in c.source for c in recorder.chunks)


def test_a_path_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _pipeline().index_paths([tmp_path / "assente"])


def test_a_directory_with_nothing_to_index_is_an_error_not_a_silent_zero(tmp_path):
    # Indexing nothing and reporting success is how a campaign runs against an
    # empty corpus and still produces numbers.
    (tmp_path / "solo.png").write_bytes(b"\x89PNG")

    with pytest.raises(ValueError, match="No files discovered"):
        _pipeline().index_paths([tmp_path])


def test_the_same_file_reached_twice_is_indexed_once(tmp_path):
    doc = tmp_path / "uno.txt"
    doc.write_text("a" * 200, encoding="utf-8")
    recorder = _Recorder()
    pipeline = _pipeline(recorder, chunk_size=200, chunk_overlap=20, min_chunk_chars=10)

    pipeline.index_paths([doc, tmp_path])

    assert len({c.source for c in recorder.chunks}) == len(recorder.chunks)


def test_an_unsupported_suffix_contributes_no_sections(tmp_path):
    other = tmp_path / "foglio.xlsx"
    other.write_bytes(b"PK\x03\x04")

    assert _pipeline()._load_sections_from_path(other) == []


def test_an_empty_text_file_contributes_no_sections(tmp_path):
    empty = tmp_path / "vuoto.txt"
    empty.write_text("   \n  ", encoding="utf-8")

    assert _pipeline()._load_sections_from_path(empty) == []


def test_a_file_with_broken_encoding_is_read_rather_than_refused(tmp_path):
    doc = tmp_path / "uno.txt"
    doc.write_bytes(b"caff\xe8 e scotta " + b"x" * 200)

    sections = _pipeline()._load_sections_from_path(doc)

    assert sections and "scotta" in sections[0][1]


def test_clearing_empties_the_index():
    recorder = _Recorder()
    pipeline = _pipeline(recorder)

    pipeline.clear()

    assert recorder.cleared == 1


# --- finding the passage a citation points at ------------------------------


def _indexed(*sources: str) -> _Recorder:
    """A recorder already holding one chunk per source."""
    recorder = _Recorder()
    recorder.chunks = [
        TextChunk(chunk_id=f"c{i}", content=f"testo {i}", source=src)
        for i, src in enumerate(sources, start=1)
    ]
    return recorder


def test_the_cited_page_comes_first():
    # A citation names a document and a page; this returns the passage it
    # points at, and ranking cannot be relied on to surface it — the question
    # is phrased in the reader's words, not the source's.
    recorder = _indexed(
        "/corpus/REPORT MATTM.pdf#page=3#chunk=1",
        "/corpus/REPORT MATTM.pdf#page=70#chunk=1",
    )

    found = _pipeline(recorder).chunks_from("REPORT MATTM", page="p. 70")

    assert found[0].source.endswith("page=70#chunk=1")


def test_the_rest_of_the_document_follows_the_cited_page():
    recorder = _indexed(
        "/corpus/REPORT MATTM.pdf#page=3#chunk=1",
        "/corpus/REPORT MATTM.pdf#page=70#chunk=1",
    )

    assert len(_pipeline(recorder).chunks_from("REPORT MATTM", page="p. 70")) == 2


def test_another_document_is_never_returned():
    recorder = _indexed(
        "/corpus/REPORT MATTM.pdf#page=70#chunk=1",
        "/corpus/Altro.pdf#page=1#chunk=1",
    )

    found = _pipeline(recorder).chunks_from("REPORT MATTM")

    assert len(found) == 1


def test_a_label_that_matches_nothing_returns_nothing():
    recorder = _indexed("/corpus/REPORT MATTM.pdf#page=70#chunk=1")

    assert _pipeline(recorder).chunks_from("Documento Inesistente") == []


@pytest.mark.parametrize("label", ["", "   "])
def test_an_empty_label_returns_nothing(label):
    assert _pipeline(_indexed("/corpus/a.pdf#page=1")).chunks_from(label) == []


def test_a_retriever_that_does_not_keep_its_chunks_returns_nothing():
    class _Opaque:
        size = 0

        def retrieve_with_scores(self, query, top_k=5, **kwargs):
            return []

    assert StandardTextRAGPipeline(retriever=_Opaque()).chunks_from("qualcosa") == []


# --- retrieving ------------------------------------------------------------


def _results(n: int = 3) -> list[tuple[TextChunk, float]]:
    """`n` scored chunks, one page each, in descending score."""
    return [
        (TextChunk(chunk_id=f"c{i}", content=f"passaggio {i}", source=f"doc.pdf#page={i}"), 1.0 / i)
        for i in range(1, n + 1)
    ]


def test_what_comes_back_carries_its_score_and_provenance():
    recorder = _Recorder(_results(2))

    found = _pipeline(recorder).retrieve("cos'e' la scotta?", top_k=2)

    assert all(isinstance(item, RetrievedTextChunk) for item in found)
    assert found[0].score == 1.0
    assert found[0].source == "doc.pdf#page=1"


def test_the_budget_is_passed_to_the_backend():
    recorder = _Recorder(_results(5))

    _pipeline(recorder).retrieve("q", top_k=2)

    assert recorder.asked[0]["top_k"] == 2


def test_diversification_reaches_a_backend_that_understands_it():
    recorder = _MMRRecorder(_results(3))

    _pipeline(recorder).retrieve("q", top_k=2, mmr_lambda=0.7, fetch_k=20)

    assert recorder.asked[0]["mmr_lambda"] == 0.7
    assert recorder.asked[0]["fetch_k"] == 20


def test_diversification_is_not_forced_on_a_backend_without_an_embedding_space():
    # The TF-IDF retriever has nothing to diversify in. Asked of the signature
    # rather than discovered by catching TypeError: a backend raising it for
    # its own reasons would look like a backend without MMR.
    recorder = _NoMMRRecorder(_results(3))

    _pipeline(recorder).retrieve("q", top_k=2, mmr_lambda=0.7)

    assert "mmr_lambda" not in recorder.asked[0]


def test_a_candidate_pool_without_a_lambda_is_not_sent():
    recorder = _MMRRecorder(_results(3))

    _pipeline(recorder).retrieve("q", top_k=2, fetch_k=20)

    assert recorder.asked[0]["mmr_lambda"] is None


def test_a_backend_taking_kwargs_is_treated_as_understanding_mmr():
    assert _accepts_mmr(_Recorder) is True  # **kwargs
    assert _accepts_mmr(_MMRRecorder) is True  # named parameter
    assert _accepts_mmr(_NoMMRRecorder) is False  # neither


def test_something_that_is_not_a_retriever_is_not_asked_for_mmr():
    class _NoMethod:
        pass

    assert _accepts_mmr(_NoMethod) is False


# --- rendering the context -------------------------------------------------


def test_the_context_names_the_source_of_each_passage():
    recorder = _Recorder(_results(2))

    context = _pipeline(recorder).build_context("q", top_k=2)

    assert context.count("Source: ") == 2
    assert "---" in context


def test_the_context_can_be_built_without_provenance():
    recorder = _Recorder(_results(2))

    context = _pipeline(recorder).build_context("q", top_k=2, include_sources=False)

    assert "Source: " not in context
    assert "passaggio 1" in context


def test_a_passage_with_no_source_is_still_included():
    recorder = _Recorder([(TextChunk(chunk_id="c1", content="senza fonte", source=None), 1.0)])

    context = _pipeline(recorder).build_context("q")

    assert context == "senza fonte"


def test_an_empty_index_yields_an_empty_context():
    assert _pipeline(_Recorder([])).build_context("q") == ""


# --- end to end over the real BM25 backend ---------------------------------


def test_a_real_index_finds_the_page_that_answers(tmp_path):
    (tmp_path / "scotta.txt").write_text(
        "La scotta e' il residuo liquido della lavorazione del formaggio. " * 6,
        encoding="utf-8",
    )
    (tmp_path / "packaging.txt").write_text(
        "Il packaging alimentare riguarda i materiali di imballaggio. " * 6,
        encoding="utf-8",
    )
    pipeline = StandardTextRAGPipeline(
        retriever=TextRAGManager(), chunk_size=200, chunk_overlap=20, min_chunk_chars=40
    )
    pipeline.index_directory(tmp_path)

    found = pipeline.retrieve("cos'e' la scotta?", top_k=1)

    assert "scotta" in found[0].content.lower()
    assert "scotta.txt" in found[0].source
