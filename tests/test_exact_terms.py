"""Codes the encoder cannot place reach the text context by their exact form.

"Parlami delle 3C" embeds nowhere near the passages on the three C's, and the
similarity ranking fills the context with whatever is closest. The passages
that name the code word for word are looked up and put first, up to half the
context; a question without codes is retrieved exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever, _code_terms
from graphrag.text_rag.manager import TextChunk
from graphrag.text_rag.pipeline import StandardTextRAGPipeline


# --- which words are codes -------------------------------------------------


def test_a_digit_code_is_looked_up_joined_and_spaced():
    assert _code_terms("Parlami delle 3C") == ["3C", "3 C"]
    assert _code_terms("Quali sono le 10 R?") == ["10R", "10 R"]


def test_a_reader_typing_lower_case_still_reaches_the_code():
    assert _code_terms("Cosa sono le 3c?") == ["3C", "3 C"]


def test_words_with_two_capitals_are_codes():
    assert _code_terms("Che cos'è SEeD? E il report MATTM?") == ["SEeD", "MATTM"]


def test_units_years_and_ordinary_words_are_not_codes():
    assert _code_terms("20 kg di vinaccia nel 2021, biochar e Torino") == []


# --- the lookup ------------------------------------------------------------


class _Index:
    """A backend holding fixed chunks, optionally able to score them."""

    def __init__(self, chunks: list[TextChunk], scores: dict[str, float] | None = None):
        self.chunks = chunks
        if scores is not None:
            self.similarity = lambda query, items: [scores[c.chunk_id] for c in items]


def _pipeline(chunks: list[TextChunk], scores: dict[str, float] | None = None):
    return StandardTextRAGPipeline(retriever=_Index(chunks, scores))


def _chunk(chunk_id: str, content: str) -> TextChunk:
    return TextChunk(chunk_id=chunk_id, content=content, source=f"{chunk_id}.pdf#page=1")


FILLER = [_chunk(f"f{i}", "Testo che non nomina alcun codice.") for i in range(200)]


def test_the_match_is_exact_so_a_homonym_word_does_not_count():
    seed = _chunk("seed", "Il progetto SEeD misura gli eventi.")
    plant = _chunk("plant", "The seed of the coffee plant is dried.")

    found = _pipeline([seed, plant, *FILLER]).chunks_with_terms(["SEeD"])

    assert [c.chunk_id for c in found] == ["seed"]


def test_a_term_found_almost_everywhere_ranks_nothing():
    common = [_chunk(f"c{i}", "economia circolare CE") for i in range(10)]

    assert _pipeline([*common, *FILLER[:20]]).chunks_with_terms(["CE"]) == []


def test_with_a_query_the_matches_follow_similarity():
    bibliography = _chunk("bib", "Fassio (2021). The 3 C's. The 3 C's of the CEFF.")
    definition = _chunk("def", "Le 3C sono Capitale, Ciclicità e Coevoluzione.")
    scores = {"bib": 0.2, "def": 0.9, **{c.chunk_id: 0.0 for c in FILLER}}

    found = _pipeline([bibliography, definition, *FILLER], scores).chunks_with_terms(
        ["3C", "3 C"], query="Parlami delle 3C"
    )

    assert [c.chunk_id for c in found] == ["def", "bib"]


# --- placement in the text ranking -----------------------------------------


@dataclass
class _Hit:
    content: str
    source: str = ""
    chunk_id: str = ""
    score: float = 0.0


class _Pipeline:
    """Similarity ranking fixed in advance, plus an exact-form lookup."""

    def __init__(self, ranking: list[_Hit], exact: list[_Hit]):
        self.ranking, self.exact = ranking, exact

    def retrieve(self, query, top_k=5, mmr_lambda=None, fetch_k=None):
        return self.ranking[:top_k]

    def chunks_with_terms(self, terms, query=""):
        return list(self.exact)


def _retriever(ranking, exact, **overrides) -> KGRetriever:
    config = AgentConfig(use_text_retriever=True, text_retriever_top_k=4, **overrides)
    return KGRetriever(kg_store=None, config=config, text_pipeline=_Pipeline(ranking, exact))


RANKING = [_Hit(f"vicino {i}", f"r{i}.pdf#page=1", f"r{i}") for i in range(6)]
EXACT = [_Hit(f"3C {i}", f"e{i}.pdf#page=1", f"e{i}") for i in range(5)]


def test_exact_matches_take_up_to_half_the_context_first():
    retriever = _retriever(RANKING, EXACT, text_retriever_exact_terms=True)

    result = retriever._retrieve_text_chunks("Parlami delle 3C")

    assert [h.chunk_id for h in result] == ["e0", "e1", "r0", "r1"]


def test_a_passage_in_both_rankings_appears_once():
    ranking = [EXACT[0], *RANKING]
    retriever = _retriever(ranking, EXACT, text_retriever_exact_terms=True)

    result = retriever._retrieve_text_chunks("Parlami delle 3C")

    assert [h.chunk_id for h in result] == ["e0", "e1", "r0", "r1"]


def test_without_codes_the_ranking_is_untouched():
    retriever = _retriever(RANKING, EXACT, text_retriever_exact_terms=True)

    result = retriever._retrieve_text_chunks("Che cos'è il biochar?")

    assert [h.chunk_id for h in result] == ["r0", "r1", "r2", "r3"]


def test_off_by_default():
    retriever = _retriever(RANKING, EXACT)

    result = retriever._retrieve_text_chunks("Parlami delle 3C")

    assert [h.chunk_id for h in result] == ["r0", "r1", "r2", "r3"]


def test_an_empty_dense_index_scores_every_chunk_zero():
    from graphrag.text_rag.dense_manager import DenseTextRAGManager

    manager = DenseTextRAGManager()

    assert manager.similarity("3C", [_chunk("a", "Le 3C")]) == [0.0]
