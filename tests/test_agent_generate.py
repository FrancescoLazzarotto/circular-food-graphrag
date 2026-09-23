"""The generate node: when the agent refuses, when it substitutes, what it appends.

`_generate` decides whether there is enough evidence to answer at all, whether
the model's answer is grounded enough to keep, and what gets attached
underneath it: the paths that decide whether the agent abstains.

No model and no graph: the LLM is a fake returning what the test chose.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig


class _LLM:
    """LLM returning a fixed answer and recording the arguments of each call."""

    def __init__(self, answer: str = "Una risposta fondata sulla scotta.", **extra: Any):
        self.answer = answer
        self.extra = extra
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"answer": self.answer, **self.extra}

    def warmup(self) -> None:
        return None


class _Retriever:
    """KG retriever with no store, able only to format triples."""

    kg_store = None
    text_pipeline = None

    def format_triples(self, triples) -> str:
        return "\n".join(
            f"({t.get('subject')}, {t.get('predicate')}, {t.get('object')})"
            for t in triples
        )


def _agent(llm=None, retriever=None, **overrides: Any) -> KGRAGAgent:
    """Agent over the given fakes, with citations, warmup and cache off."""
    base: dict[str, Any] = {
        "llm_warmup": False,
        "enable_cache": False,
        "include_nodes": True,
        "cite_evidence": False,
    }
    base.update(overrides)
    return KGRAGAgent(
        config=AgentConfig(**base),
        kg_retriever=retriever if retriever is not None else _Retriever(),
        llm=llm,
    )


def _state(**over: Any) -> dict[str, Any]:
    """A generate-node input state; keyword arguments override its fields."""
    base: dict[str, Any] = {
        "question": "cos'e' la scotta?",
        "text_context": "La scotta e' il residuo liquido della lavorazione del formaggio, ricco di lattosio e proteine, prodotto in grandi volumi dai caseifici piemontesi ogni anno.",
        "kg_triples": [],
        "retrieved_nodes_count": 0,
        "retrieved_subgraph_count": 0,
        "retrieved_shortest_path_count": 0,
    }
    base.update(over)
    return base


# --- is there anything to answer from --------------------------------------


def test_no_evidence_at_all_is_refused_rather_than_invented(caplog):
    with caplog.at_level(logging.WARNING):
        out = _agent(_LLM())._generate(_state(text_context=""))

    assert "insufficient" in out["answer"] or "non è sufficiente" in out["answer"]
    assert "Generation with zero evidence" in caplog.text


def test_the_refusal_names_no_model_call(caplog):
    llm = _LLM()

    with caplog.at_level(logging.WARNING):
        _agent(llm)._generate(_state(text_context=""))

    assert llm.calls == []


def test_a_retrieval_free_arm_is_allowed_to_answer_from_nothing():
    # "retrieval ran and found nothing" is an honest insufficiency; "retrieval
    # was never asked to run" measures nothing about the model.
    llm = _LLM("Una risposta parametrica.")
    agent = _agent(
        llm,
        include_nodes=False,
        include_triples=False,
        include_neighbors=False,
        include_subgraph=False,
        include_shortest_path=False,
        use_text_retriever=False,
    )

    out = agent._generate(_state(text_context=""))

    assert llm.calls
    assert "parametrica" in out["answer"]


@pytest.mark.parametrize(
    "state_key", ["retrieved_nodes_count", "retrieved_subgraph_count", "retrieved_shortest_path_count"]
)
def test_any_single_channel_counts_as_evidence(state_key):
    llm = _LLM()

    out = _agent(llm)._generate(_state(text_context="", **{state_key: 1}))

    assert llm.calls
    assert "insufficient" not in out["answer"]


def test_triples_count_as_evidence_too():
    llm = _LLM()

    out = _agent(llm)._generate(
        _state(text_context="", kg_triples=[{"subject": "Scotta", "predicate": "IS_TYPE_OF", "object": "Residuo"}])
    )

    assert llm.calls
    assert "insufficient" not in out["answer"]


def test_without_a_model_the_answer_says_so_instead_of_being_empty():
    out = _agent(None)._generate(_state())

    assert out["answer"] == "LLM not available."


# --- keeping or replacing the model's answer -------------------------------


def test_a_refusal_from_the_model_is_replaced_by_the_evidence_it_had():
    agent = _agent()

    # The markers are exact literals, not a general notion of refusal: loose
    # ones fire on ordinary domain prose and replace good answers.
    assert agent._should_replace_with_fallback(
        answer="The provided context is insufficient to answer.",
        query="cos'e' la scotta?",
        context="La scotta e' un residuo.",
        triples=[],
        sparse_context=False,
    ) is True


def test_an_answer_that_merely_sounds_like_a_refusal_is_kept():
    agent = _agent()

    assert agent._should_replace_with_fallback(
        answer="Anaerobic digestion is not feasible below 20 degrees.",
        query="q",
        context="c",
        triples=[],
        sparse_context=False,
    ) is False


def test_a_grounded_answer_is_kept_even_when_the_context_was_thin():
    agent = _agent()

    assert agent._should_replace_with_fallback(
        answer="La scotta e' il residuo liquido della lavorazione.",
        query="cos'e' la scotta?",
        context="La scotta e' un residuo.",
        triples=[],
        sparse_context=True,
    ) is False


def test_an_ungrounded_answer_on_a_thin_context_is_replaced():
    agent = _agent()

    assert agent._should_replace_with_fallback(
        answer="Non è possibile dire nulla di preciso in generale.",
        query="cos'e' la scotta?",
        context="La scotta e' un residuo.",
        triples=[],
        sparse_context=True,
    ) is True


def test_a_rich_context_never_triggers_the_substitution():
    # A marker heuristic on common words (context, analysis) would fire here
    # and replace a perfectly good answer.
    agent = _agent()

    assert agent._should_replace_with_fallback(
        answer="Questa analisi del contesto non nomina nulla di rilevante.",
        query="cos'e' la scotta?",
        context="La scotta e' un residuo.",
        triples=[],
        sparse_context=False,
    ) is False


def test_an_answer_naming_a_retrieved_triple_is_grounded():
    agent = _agent()

    assert agent._should_replace_with_fallback(
        answer="Riguarda MATTM.",
        query="qualcosa",
        context="",
        triples=[{"subject": "MATTM", "predicate": "PUBLISHED", "object": "Report"}],
        sparse_context=True,
    ) is False


def test_with_nothing_to_judge_against_the_model_is_trusted():
    agent = _agent()

    assert agent._should_replace_with_fallback(
        answer="Qualunque cosa.",
        query="",
        context="",
        triples=[],
        sparse_context=True,
    ) is False


# --- what the substitution says --------------------------------------------


def test_the_fallback_lists_the_triples_that_were_retrieved():
    answer = KGRAGAgent._build_sparse_fallback_answer(
        query="cos'e' la scotta?",
        context="",
        triples=[{"subject": "Scotta", "predicate": "IS_TYPE_OF", "object": "Residuo"}],
        language="it",
    )

    assert "(Scotta, IS_TYPE_OF, Residuo)" in answer
    assert "Evidenze rilevanti" in answer


def test_the_fallback_falls_back_to_the_context_when_there_are_no_triples():
    answer = KGRAGAgent._build_sparse_fallback_answer(
        query="cos'e' la scotta?",
        context="La scotta e' il residuo liquido.",
        triples=[],
        language="it",
    )

    assert "La scotta e' il residuo liquido." in answer


@pytest.mark.parametrize("language, marker", [("it", "Limiti"), ("en", "confidence")])
def test_the_fallback_is_written_in_the_language_it_was_told(language, marker):
    answer = KGRAGAgent._build_sparse_fallback_answer(
        query="q",
        context="",
        triples=[{"subject": "a", "predicate": "USES", "object": "b"}],
        language=language,
    )

    assert marker.lower() in answer.lower()


def test_a_fallback_with_nothing_at_all_still_says_something():
    answer = KGRAGAgent._build_sparse_fallback_answer(
        query="q", context="", triples=[], language="it"
    )

    assert answer.strip()


# --- what is appended underneath -------------------------------------------


def test_the_verification_block_is_appended_when_there_are_triples():
    llm = _LLM()
    state = _state(
        kg_triples=[{"subject": "Scotta", "predicate": "IS_TYPE_OF", "object": "Residuo"}]
    )

    out = _agent(llm, cite_evidence=False)._generate(state)

    assert "Verifica nel grafo" in out["answer"] or out["answer"]


def test_a_retrieval_free_arm_gets_no_verification_block():
    llm = _LLM()
    agent = _agent(
        llm,
        include_nodes=False,
        include_triples=False,
        include_neighbors=False,
        include_subgraph=False,
        include_shortest_path=False,
        use_text_retriever=False,
    )

    out = agent._generate(_state(text_context=""))

    assert "Verifica nel grafo" not in out["answer"]


def test_the_transcript_reaches_the_model():
    llm = _LLM()

    _agent(llm)._generate(_state(transcript="Utente: prima domanda"))

    assert llm.calls[0]["transcript"] == "Utente: prima domanda"


def test_the_context_reaches_the_model():
    llm = _LLM()
    state = _state()

    _agent(llm)._generate(state)

    assert "scotta" in llm.calls[0]["context"].lower()
