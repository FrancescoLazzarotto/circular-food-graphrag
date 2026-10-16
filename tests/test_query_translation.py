"""The text channel also searches with the question in the corpus's other language.

The multilingual encoder ranks passages in the question's language first:
asked in Italian, a source that exists only in English is rarely reached, and
the other way round. The translation is one short model call; its ranking is
interleaved with the question's, never merged by score, because scores of two
queries in two languages are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager


@dataclass
class _Hit:
    content: str
    source: str = ""
    chunk_id: str = ""
    score: float = 0.0


class _Pipeline:
    """One fixed ranking per query."""

    def __init__(self, rankings: dict[str, list[_Hit]]):
        self.rankings = rankings
        self.queries: list[str] = []

    def retrieve(self, query, top_k=5, mmr_lambda=None, fetch_k=None):
        self.queries.append(query)
        return self.rankings.get(query, [])[:top_k]


def _hits(prefix: str, n: int) -> list[_Hit]:
    return [_Hit(f"{prefix} {i}", f"{prefix}{i}.pdf#page=1", f"{prefix}{i}") for i in range(n)]


def _retriever(pipeline) -> KGRetriever:
    config = AgentConfig(use_text_retriever=True, text_retriever_top_k=4)
    return KGRetriever(kg_store=None, config=config, text_pipeline=pipeline)


def test_the_two_rankings_alternate_the_question_first():
    italian, english = _hits("it", 4), _hits("en", 4)
    pipeline = _Pipeline({"Quali sono le 3C?": italian, "What are the 3Cs?": english})

    result = _retriever(pipeline)._retrieve_text_chunks("Quali sono le 3C?", translated_query="What are the 3Cs?")

    assert [h.chunk_id for h in result] == ["it0", "en0", "it1", "en1"]


def test_a_passage_both_rankings_return_appears_once():
    shared = _Hit("comune", "c.pdf#page=1", "c")
    pipeline = _Pipeline({"q": [shared, *_hits("it", 3)], "t": [shared, *_hits("en", 3)]})

    result = _retriever(pipeline)._retrieve_text_chunks("q", translated_query="t")

    assert [h.chunk_id for h in result] == ["c", "it0", "en0", "it1"]


def test_without_a_translation_only_the_question_is_searched():
    pipeline = _Pipeline({"q": _hits("it", 6)})

    result = _retriever(pipeline)._retrieve_text_chunks("q")

    assert pipeline.queries == ["q"]
    assert [h.chunk_id for h in result] == ["it0", "it1", "it2", "it3"]


class _Answer:
    def __init__(self, content: str):
        self.content = content


def _manager(reply) -> LLMManager:
    manager = LLMManager.__new__(LLMManager)
    manager.load_llm = lambda model_id=None: object()

    def invoke(model, prompt):
        if isinstance(reply, Exception):
            raise reply
        return _Answer(reply)

    manager._invoke_with_retry = invoke
    return manager


def test_the_translation_is_the_first_line_without_reasoning_or_quotes():
    manager = _manager('<think>easy</think>\n"What are the three Cs?"\nNote: kept CEFF.')
    assert manager.translate_question("Quali sono le tre C?", "en") == "What are the three Cs?"


def test_a_failed_or_runaway_translation_is_dropped():
    assert _manager(RuntimeError("down")).translate_question("Cos'è la scotta?", "en") == ""
    assert _manager("x" * 500).translate_question("Cos'è la scotta?", "en") == ""


def test_the_switch_off_asks_for_no_translation():
    class _LLM:
        def translate_question(self, question, language):
            raise AssertionError("asked")

    agent = KGRAGAgent.__new__(KGRAGAgent)
    agent.config = AgentConfig()
    agent.llm = _LLM()
    assert agent._translated_query("Cos'è la scotta?") == ""


def test_an_italian_question_is_translated_into_english_and_back():
    asked = []

    class _LLM:
        def translate_question(self, question, language):
            asked.append(language)
            return "translated"

    agent = KGRAGAgent.__new__(KGRAGAgent)
    agent.config = AgentConfig(translate_query_for_retrieval=True)
    agent.llm = _LLM()
    agent._translated_query("Quali sono gli impieghi delle sanse di oliva?")
    agent._translated_query("What are the uses of olive pomace?")
    assert asked == ["en", "it"]
    assert KGRAGAgent._translation_kwargs("t", True) == {"translated_query": "t"}
    assert KGRAGAgent._translation_kwargs("t", False) == {} == KGRAGAgent._translation_kwargs("", True)
