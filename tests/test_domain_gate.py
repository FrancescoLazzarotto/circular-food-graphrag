"""Unit tests for the out-of-domain gate.

Without the gate, "scrivi una funzione python che costruisca una rete neurale"
is answered with a Keras function carrying three circular-economy PDFs as
sources. The cause is structural, not a bad model: the dense retriever has no
score floor, so `_grade` always sees evidence, and `grade_condition` routes
every question to `generate` after three rewrites. The gate is the terminal
state that abstains.

These tests pin the pieces that must not silently regress: the refusal is
reachable and terminal, it is off by default so experiment baselines are
untouched, the gate reads the typed question rather than the memory-rewritten
one, and a gate failure fails open.

The scope wording itself is measured, not asserted here — see
`scripts/domain_gate/eval_domain_gate_llm.py` and `scripts/domain_gate/eval_domain_gate_heldout.py`.
"""

from __future__ import annotations

import pytest

from graphrag.agent.core import KGRAGAgent, _proper_noun_terms
from graphrag.config import AgentConfig
from graphrag.llm.prompts import PromptLibrary


@pytest.fixture(autouse=True)
def scope_mode(monkeypatch):
    """Run these against the scope gate, which is not the default.

    The default evidence gate is tested in `test_evidence_gate.py`; the scope
    one stays reachable with GRAPHRAG_GATE_MODE=scope and keeps the behaviour
    pinned below.
    """
    monkeypatch.setenv("GRAPHRAG_GATE_MODE", "scope")


class _StubLLM:
    """Records what the gate was asked and answers with a fixed verdict."""

    def __init__(self, verdict: bool = True, explode: bool = False) -> None:
        self.verdict = verdict
        self.explode = explode
        self.seen: list[str] = []
        self.seen_entities: list[list[str]] = []

    def classify_in_domain(self, question: str, config, known_entities=()) -> bool:
        self.seen.append(question)
        self.seen_entities.append(list(known_entities))
        if self.explode:
            raise RuntimeError("gate backend down")
        return self.verdict

    def warmup(self) -> None:  # pragma: no cover - never called, llm_warmup is off
        pass


class _StubKGStore:
    """Answers the gate's name lookup with a fixed node list."""

    def __init__(self, names: list[str] | None = None, unavailable: bool = False) -> None:
        self.names = names or []
        self.unavailable = unavailable
        self.searched: list[list[str]] = []

    def fulltext_search_nodes(self, terms, labels=None, limit=None, boosts=None):
        self.searched.append(list(terms))
        if self.unavailable:
            return None
        return [{"text": name} for name in self.names]


class _StubRetriever:
    """Retriever exposing only a KG store."""

    def __init__(self, kg_store: _StubKGStore) -> None:
        self.kg_store = kg_store

def _agent(llm: _StubLLM | None, retriever=None, **config_kwargs) -> KGRAGAgent:
    """Agent over the given stubs, with warmup off."""
    config = AgentConfig(llm_warmup=False, **config_kwargs)
    return KGRAGAgent(config=config, kg_retriever=retriever, llm=llm)


def test_gate_refuses_out_of_domain_question_without_generating():
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    state = agent._scope_gate({"question": "scrivi una rete neurale in python"})
    assert state["in_domain"] is False

    refusal = agent._refuse_out_of_scope({"question": "scrivi una rete neurale in python"})
    assert refusal["out_of_scope"] is True
    assert refusal["answer"]
    # No evidence index exists on this path, so no source list can be attached.
    assert "Fonti" not in refusal["answer"]
    assert "Sources" not in refusal["answer"]


def test_refusal_answers_in_the_language_of_the_question():
    agent = _agent(_StubLLM(verdict=False), enable_domain_gate=True)

    italian = agent._refuse_out_of_scope({"question": "Qual è la capitale dell'Australia?"})
    english = agent._refuse_out_of_scope({"question": "What is the capital of Australia?"})

    assert "non rispondo" in italian["answer"]
    assert "not answering" in english["answer"]


def test_refusal_names_what_the_collection_covers():
    """A bare refusal gives the expert nothing to rephrase towards."""
    agent = _agent(_StubLLM(verdict=False), enable_domain_gate=True)

    answer = agent._refuse_out_of_scope({"question": "Chi ha scritto la Divina Commedia?"})["answer"]
    assert "La raccolta copre" in answer
    assert "circular economy" in answer


def test_custom_scope_replaces_the_default_in_prompt_and_refusal():
    agent = _agent(_StubLLM(verdict=False), enable_domain_gate=True, domain_scope="marine biology")

    answer = agent._refuse_out_of_scope({"question": "Chi ha scritto la Divina Commedia?"})["answer"]
    assert "marine biology" in answer
    assert "circular economy principles" not in answer

    rendered = str(PromptLibrary.domain_gate_prompt("marine biology"))
    assert "marine biology" in rendered


def test_gate_is_off_by_default_so_baselines_are_unchanged():
    llm = _StubLLM(verdict=False)
    agent = _agent(llm)

    assert agent._scope_gate({"question": "anything at all"})["in_domain"] is True
    assert llm.seen == []


def test_gate_failure_fails_open():
    """A broken classifier must not silence a working demo."""
    from graphrag.llm.manager import LLMManager

    manager = LLMManager.__new__(LLMManager)

    def _boom():
        raise RuntimeError("vLLM unreachable")

    manager.load_llm = _boom  # type: ignore[method-assign]
    assert manager.classify_in_domain("qualsiasi cosa", AgentConfig()) is True


def test_gate_reads_the_typed_question_not_the_memory_rewrite():
    """A follow-up inherits entities from the previous answer.

    Gating the rewritten question would let "and write me the code for it"
    through on the strength of the subject carried over from the turn before.
    """
    llm = _StubLLM(verdict=True)
    agent = _agent(llm, enable_domain_gate=True)

    agent._scope_gate(
        {
            "question": "e scrivimi il codice",
            "rewritten_question": "e scrivimi il codice sulla ciclicità del sistema alimentare",
        }
    )
    assert llm.seen == ["e scrivimi il codice"]


def test_follow_ups_skip_the_gate():
    """A terse follow-up carries no domain of its own.

    Follow-ups such as "e quindi?", "in che senso?" and "perché?" read as out
    of domain to the classifier. Refusing those breaks the conversation, and
    the topic they continue was gated when it was introduced.
    """
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    state = agent._scope_gate({"question": "e quindi?", "follow_up": True})

    assert state["in_domain"] is True
    assert llm.seen == []


def test_the_barest_continuations_skip_the_gate_without_memory():
    """Memory fills only from KG entities, so the flag cannot be the only test.

    On an answer built entirely from the text channel — the common case — no
    entity is observed, `follow_up` stays False, and these would be refused.
    """
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    for question in ("in che senso?", "perché?", "non ho capito", "fammi un esempio"):
        assert agent._scope_gate({"question": question})["in_domain"] is True
    assert llm.seen == []


def test_the_floor_does_not_exempt_a_short_out_of_domain_question():
    """Four words is a topic, three is a continuation.

    "Spiegami la relatività generale" is exactly four and must still be gated.
    """
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    assert agent._scope_gate({"question": "Spiegami la relatività generale"})["in_domain"] is False
    assert llm.seen == ["Spiegami la relatività generale"]


def test_a_first_question_is_still_gated():
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    state = agent._scope_gate({"question": "scrivi una rete neurale", "follow_up": False})

    assert state["in_domain"] is False
    assert llm.seen == ["scrivi una rete neurale"]


def test_empty_question_is_not_refused():
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    assert agent._scope_gate({"question": "   "})["in_domain"] is True
    assert llm.seen == []


def test_closing_prompt_line_matches_the_grounding_rule():
    """The last line is the one models follow; it must not contradict the top.

    Calling the context insufficient only when it is empty is exactly wrong when
    an out-of-domain question arrives with a full context of
    unrelated-but-factual chunks.
    """
    strict = str(PromptLibrary.answer_prompt(AgentConfig(allow_parametric_fallback=False)))
    assert "insufficient whenever it does not cover what was asked" in strict
    assert "insufficient only when context is empty" not in strict

    marked = str(PromptLibrary.answer_prompt(AgentConfig(allow_parametric_fallback=True)))
    assert "not in the retrieved evidence" in marked


# ---------------------------------------------------------------- #
# names the model cannot know
# ---------------------------------------------------------------- #


def test_capitalised_tokens_are_candidates_and_question_words_are_not():
    """"Chi è Barilla?" must look up Barilla, not "Chi"."""
    assert _proper_noun_terms("Chi è Barilla?") == ["Barilla"]
    assert _proper_noun_terms("Che cos'è SeED?") == ["SeED"]
    assert _proper_noun_terms("What is SEeD?") == ["SEeD"]
    # Nothing capitalised beyond the opening word: no lookup worth making.
    assert _proper_noun_terms("Quante calorie ha una mela?") == []


def test_names_the_graph_holds_reach_the_classifier():
    llm = _StubLLM(verdict=True)
    store = _StubKGStore(["SEeD", "SEeD project"])
    agent = _agent(llm, retriever=_StubRetriever(store), enable_domain_gate=True)

    agent._scope_gate({"question": "Che cos'è SeED?"})

    assert store.searched == [["SeED"]]
    assert llm.seen_entities == [["SEeD", "SEeD project"]]


def test_a_name_that_only_looks_like_a_match_is_dropped():
    """The full-text index tokenises, so it returns more than it was asked.

    "Potatoes, Curd, and Linseed Oil" comes back for SEeD because "Linseed"
    stems to it. Telling the gate the collection contains that is worse than
    telling it nothing.
    """
    llm = _StubLLM(verdict=True)
    store = _StubKGStore(["Potatoes, Curd, and Linseed Oil", "SEeD project"])
    agent = _agent(llm, retriever=_StubRetriever(store), enable_domain_gate=True)

    agent._scope_gate({"question": "Che cos'è SeED?"})

    assert llm.seen_entities == [["SEeD project"]]


def test_the_lookup_never_decides_the_verdict():
    """A node named Torino does not make a restaurant question in domain.

    The graph says which names exist; the model says whether the question is
    one this collection answers. Collapsing the two is the failure mode this
    whole path has to avoid.
    """
    llm = _StubLLM(verdict=False)
    store = _StubKGStore(["Torino", "Politecnico di Torino"])
    agent = _agent(llm, retriever=_StubRetriever(store), enable_domain_gate=True)

    state = agent._scope_gate({"question": "Consigliami un ristorante a Torino"})

    assert llm.seen_entities == [["Torino", "Politecnico di Torino"]]
    assert state["in_domain"] is False


def test_an_unavailable_index_leaves_the_prompt_as_validated():
    """`fulltext_search_nodes` returns None while the index is disabled."""
    llm = _StubLLM(verdict=True)
    agent = _agent(
        llm,
        retriever=_StubRetriever(_StubKGStore(unavailable=True)),
        enable_domain_gate=True,
    )

    agent._scope_gate({"question": "Che cos'è SeED?"})

    assert llm.seen_entities == [[]]


def test_a_failed_lookup_does_not_fail_the_gate():
    class _Exploding(_StubKGStore):
        def fulltext_search_nodes(self, *args, **kwargs):
            raise RuntimeError("neo4j unreachable")

    llm = _StubLLM(verdict=True)
    agent = _agent(llm, retriever=_StubRetriever(_Exploding()), enable_domain_gate=True)

    assert agent._scope_gate({"question": "Che cos'è SeED?"})["in_domain"] is True
    assert llm.seen_entities == [[]]


def test_without_a_retriever_the_gate_still_runs():
    llm = _StubLLM(verdict=True)
    agent = _agent(llm, enable_domain_gate=True)

    agent._scope_gate({"question": "Che cos'è SeED?"})

    assert llm.seen_entities == [[]]


def test_the_prompt_is_byte_identical_when_no_name_matched():
    """The validated wording must survive untouched on the no-match path."""
    plain = str(PromptLibrary.domain_gate_prompt())
    assert str(PromptLibrary.domain_gate_prompt(known_entities=[])) == plain
    assert str(PromptLibrary.domain_gate_prompt(known_entities=["  "])) == plain
    assert str(PromptLibrary.domain_gate_prompt(known_entities=["SEeD"])) != plain


def test_listed_names_are_offered_as_evidence_not_as_a_verdict():
    rendered = str(PromptLibrary.domain_gate_prompt(known_entities=["SEeD", "SEeD"]))

    assert rendered.count("SEeD") == 1, "duplicates must collapse"
    assert "judge the question itself" in rendered
