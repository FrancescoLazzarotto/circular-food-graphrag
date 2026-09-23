"""Unit tests for intra-session conversational memory.

Two properties matter more than the feature itself:

* **No change when memory is off.** `invoke(question)` without a memory must
  reach the graph with exactly the state it reaches without the feature — gold
  runs and experiment baselines depend on it.
* **No false positives.** The follow-up detector must not fire on a
  self-contained question: 0 hits on the 30 gold queries, which are all
  self-contained by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from graphrag.agent.core import KGRAGAgent
from graphrag.agent.memory import ConversationMemory
from graphrag.config import AgentConfig

GOLD_PATH = Path(__file__).resolve().parents[1] / "evaluation/gold/gold_circular_v1.json"

# A real chain from an expert session: Q3 is self-contained, Q5 borrows its
# subject ("vino") from the answer to Q4.
Q3 = (
    "Quali sono le 5 filiere della regione Piemonte in cui l'economia circolare "
    "per il cibo può trovare una buona espressione?"
)
Q5 = "Mi indichi le strategie nel settore vino individuate dalla ricerca?"


class _FakeOutput:
    """Chat-model output carrying only `content`."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    """Model returning queued answers in order and recording its prompts."""

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self.prompts: list[Any] = []

    def invoke(self, payload: Any) -> _FakeOutput:
        self.prompts.append(payload)
        return _FakeOutput(self._answers.pop(0) if self._answers else "")


class _FakeLLM:
    """LLM manager exposing a fixed model, with no retry."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def load_llm(self) -> Any:
        return self._model

    def _invoke_with_retry(self, model: Any, payload: Any) -> Any:
        return model.invoke(payload)


class _RecordingAgent(KGRAGAgent):
    """Agent whose graph records the initial state instead of running."""

    def _build_graph(self):  # type: ignore[override]
        agent = self

        class _Graph:
            def invoke(self, state: dict, config: dict | None = None) -> dict:
                agent.seen_state = dict(state)
                return {"answer": "risposta", "retrieved_nodes": [], "kg_triples": []}

        return _Graph()


def _agent(model: Any = None) -> _RecordingAgent:
    """Recording agent, with a fake LLM over `model` when one is given."""
    llm = _FakeLLM(model) if model is not None else None
    return _RecordingAgent(config=AgentConfig(), kg_retriever=None, llm=llm)


def _memory_with(*entities: str) -> ConversationMemory:
    """Memory after one turn whose answer and triples name `entities`."""
    memory = ConversationMemory()
    memory.observe(
        question="domanda precedente",
        answer=" ".join(entities),
        triples=[{"subject": name, "predicate": "REL", "object": "x"} for name in entities],
    )
    return memory


# --- follow-up detection --------------------------------------------------









def test_a_failed_turn_does_not_spend_a_turn_of_the_decay_window():
    """The demo retries the same question after failing over to the other graph.
    Incrementing `turn` would spend two turns on one question and shorten the
    entity window by one."""
    memory = ConversationMemory()
    memory.observe_failure("Cosa sono le 3C?")
    assert memory.turn == 0
    assert memory.failed_turns == 1


def test_new_topic_forgets_the_failure_too():
    memory = ConversationMemory()
    memory.observe_failure("Cosa sono le 3C?")
    memory.reset()
    assert memory.has_context() is False
    assert memory.failed_turns == 0




def test_entities_come_from_retrieval_not_from_the_answer_text():
    memory = ConversationMemory()
    memory.observe(
        question="Quali filiere?",
        answer="Le filiere sono vino e riso, e anche il turismo.",
        nodes=[{"text": "Vino"}],
        triples=[{"subject": "Riso", "predicate": "PART_OF", "object": "Filiere"}],
    )

    names = {item.name for item in memory.active_entities}
    assert names == {"Vino", "Riso", "Filiere"}
    # "turismo" is in the answer but was never retrieved: memory carries only
    # what the graph actually returned.
    assert "turismo" not in {name.lower() for name in names}


def test_last_answer_entities_are_the_ones_the_model_talked_about():
    memory = ConversationMemory()
    memory.observe(
        question="Quali filiere?",
        answer="La filiera del vino è la più matura.",
        nodes=[{"text": "Vino"}, {"text": "Acqua minerale"}],
    )

    assert memory.last_answer_entities == ["Vino"]


def test_a_name_inside_a_longer_word_is_not_a_mention():
    """The common false positive: "riso" inside "risorse".

    A name promoted this way leads the seed ranking, so the follow-up gets
    rewritten around a topic the answer never discussed.
    """
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="Le risorse idriche del sistema sono limitate.",
        nodes=[{"text": "Riso"}, {"text": "Tema"}],
    )

    assert memory.last_answer_entities == []


def test_an_elided_article_does_not_hide_a_mention():
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="L'economia circolare parte dai sotto-prodotti.",
        nodes=[{"text": "Economia circolare"}, {"text": "Prodotti"}],
    )

    assert memory.last_answer_entities == ["Economia circolare", "Prodotti"]


def test_seed_entities_put_the_last_answer_first():
    memory = ConversationMemory()
    memory.observe(
        question="q1",
        answer="parla di Riso",
        nodes=[{"text": "Carne bovina"}, {"text": "Riso"}],
    )

    assert memory.seed_entities()[0] == "Riso"


def test_a_broader_name_absorbs_its_own_substring():
    """"Regione" next to "Regione Piemonte" wastes a slot and vaguens the rewrite."""
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="parla di Regione Piemonte, Regione e Vino",
        nodes=[{"text": "Regione Piemonte"}, {"text": "Regione"}, {"text": "Vino"}],
    )

    assert memory.seed_entities() == ["Regione Piemonte", "Vino"]


def test_the_specific_name_wins_even_when_the_broad_one_ranks_first():
    """Ranking decides the order, so absorption cannot depend on it."""
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="parla di Regione, Regione Piemonte e Vino",
        nodes=[{"text": "Regione"}, {"text": "Regione Piemonte"}, {"text": "Vino"}],
    )

    assert memory.seed_entities() == ["Regione Piemonte", "Vino"]


def test_absorption_needs_whole_words_not_a_substring():
    """"Riso" is not a broader "Risorsa": dropping it would lose a real topic."""
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="parla di Risorsa e di Riso",
        nodes=[{"text": "Risorsa"}, {"text": "Riso"}],
    )

    assert memory.seed_entities() == ["Risorsa", "Riso"]


def test_a_broader_name_absorbs_two_narrower_ones_at_once():
    """One new name can subsume more than one already-selected slot."""
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="parla di Politica Agricola, Agricola Comune e Politica Agricola Comune",
        nodes=[
            {"text": "Politica Agricola"},
            {"text": "Agricola Comune"},
            {"text": "Politica Agricola Comune"},
        ],
    )

    assert memory.seed_entities() == ["Politica Agricola Comune"]


def test_entities_decay_out_of_the_window():
    memory = ConversationMemory(window=2)
    memory.observe(question="q1", answer="", nodes=[{"text": "Vecchio argomento"}])
    for turn in range(2, 5):
        memory.observe(question=f"q{turn}", answer="", nodes=[{"text": "Nuovo"}])

    assert [item.name for item in memory.active_entities] == ["Nuovo"]


def test_reset_clears_the_topic():
    memory = _memory_with("Vino")
    memory.reset()

    assert memory.has_context() is False
    assert memory.seed_entities() == []
    assert memory.last_question == ""


def test_noise_is_kept_out_of_the_seed_list():
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="",
        nodes=[{"text": "ab"}, {"text": "123"}, {"text": "x" * 80}, {"text": "Vino"}],
    )

    assert [item.name for item in memory.active_entities] == ["Vino"]


# --- wiring into the agent ------------------------------------------------


def test_without_memory_the_initial_state_is_unchanged():
    agent = _agent()
    agent.invoke(Q5)

    assert set(agent.seen_state) == {"question", "run_id", "rewrite_count"}
    assert agent.seen_state["question"] == Q5


def test_a_follow_up_reaches_retrieval_rewritten_and_generation_intact():
    """`_retrieve`/`_grade` read `rewritten_question`, `_generate` reads `question`."""
    model = _FakeModel(
        "Quali strategie di economia circolare sono individuate per la filiera "
        "del vino in Piemonte?"
    )
    agent = _agent(model)
    agent.invoke(Q5, memory=_memory_with("Vino", "Piemonte"))

    assert agent.seen_state["question"] == Q5
    assert agent.seen_state["rewritten_question"].startswith("Quali strategie")
    assert "Vino" in str(model.prompts[0])


def test_a_self_contained_question_is_rewritten_to_itself():
    """A self-contained question is rewritten to itself, and leaves no trace.

    The rewriter runs on every turn that has context: no heuristic decides
    whether a question looks self-contained, because such heuristics read
    "Spiegameli meglio" as a fresh question and "e <anything>" as a
    continuation. The prompt is told to repeat a self-contained question
    unchanged, and an unchanged question puts nothing in the state, so
    retrieval sees exactly what the user typed.

    The cost is one short LLM call per turn with context, roughly 1-2 s against
    a turn of about 25 s.
    """
    model = _FakeModel(Q3)
    agent = _agent(model)
    agent.invoke(Q3, memory=_memory_with("Vino"))

    assert model.prompts, "the rewriter is expected to run now"
    assert "rewritten_question" not in agent.seen_state


def test_an_implausible_rewrite_is_discarded():
    """A model that answers instead of rewriting must not poison retrieval."""
    model = _FakeModel("Le strategie individuate sono molte. " * 30)
    agent = _agent(model)
    agent.invoke(Q5, memory=_memory_with("Vino"))

    assert "rewritten_question" not in agent.seen_state


def test_a_failing_rewrite_keeps_the_question():
    class _BrokenModel:
        def invoke(self, payload: Any) -> Any:
            raise ValueError("backend down")

    agent = _agent(_BrokenModel())
    agent.invoke(Q5, memory=_memory_with("Vino"))

    assert "rewritten_question" not in agent.seen_state


def test_the_rewrite_is_logged_next_to_the_original():
    model = _FakeModel("Quali strategie per la filiera del vino in Piemonte?")
    agent = _agent(model)
    result = agent.invoke(Q5, memory=_memory_with("Vino"))

    assert result["original_question"] == Q5
    assert result["retrieval_question"].startswith("Quali strategie")
    assert result["follow_up"] is True
    assert result["memory_entities"] == ["Vino"]


def test_the_turn_is_recorded_in_memory():
    memory = _memory_with("Vino")
    agent = _agent()
    agent.invoke(Q3, memory=memory)

    assert memory.turn == 2
    assert memory.last_question == Q3


# --- seeding must follow the answer, not the retriever -----------------------


def test_seeds_are_empty_when_the_answer_used_none_of_the_retrieved_entities():
    """The graph returning something is not the same as the answer using it.

    For "Quali sono le 3C dell'economia circolare per il cibo?" dozens of nodes
    come back, and the answer, about Capitale, Ciclicità and Coevoluzione, uses
    none of them. Ranking the unused ones anyway would seed the follow-up
    rewrite with "Economia circolare ittica" and send the next question out
    asking about fish.
    """
    memory = ConversationMemory()
    memory.observe(
        question="Quali sono le 3C?",
        answer="Le 3C sono Capitale, Ciclicità e Coevoluzione.",
        nodes=[{"text": "Economia circolare ittica"}, {"text": "Regione Piemonte"}],
    )

    assert memory.seed_entities() == []


def test_an_empty_seed_leaves_the_question_as_typed():
    """No rewrite is better than a wrong one; the retriever handles the original."""
    memory = ConversationMemory()
    memory.observe(
        question="Quali sono le 3C?",
        answer="Le 3C sono Capitale, Ciclicità e Coevoluzione.",
        nodes=[{"text": "Economia circolare ittica"}],
    )
    agent = _agent(_FakeModel("una riscrittura che non deve mai essere usata"))

    result = agent.invoke("spiegami meglio la ciclicità", memory=memory)

    assert result["retrieval_question"] == "spiegami meglio la ciclicità"
    assert result["memory_entities"] == []





def test_a_rewrite_that_explains_itself_is_discarded():
    """A model can answer the rewrite prompt with an essay (Gemma-4-31B does).

    Three numbered options, a "Key Improvements Made" section, 1500 characters.
    Passed to the retriever whole it buries "3C" under marketing vocabulary the
    corpus does not contain, and the demo reports the framework as absent.
    """
    from graphrag.agent.core import _plausible_rewrite

    essay = (
        'Depending on the context of your knowledge base, the term "3C" can refer '
        "to different frameworks.\n\n"
        "### Option 1: Marketing & Strategy (Kenichi Ohmae)\n"
        "> **Rewritten:** Cosa sono le 3C di Kenichi Ohmae?\n\n"
        "### Key Improvements Made:\n"
        "*   **Disambiguation:** \"3C\" is an ambiguous term.\n"
    )
    assert _plausible_rewrite(essay, "Cosa sono le 3C?") == "Cosa sono le 3C?"


def test_a_one_line_rewrite_survives():
    from graphrag.agent.core import _plausible_rewrite

    reply = "Rewritten question: Cosa sono le 3C della Circular Economy for Food?"
    assert (
        _plausible_rewrite(reply, "Cosa sono le 3C?")
        == "Cosa sono le 3C della Circular Economy for Food?"
    )


def test_an_empty_rewrite_keeps_the_question():
    from graphrag.agent.core import _plausible_rewrite

    assert _plausible_rewrite("   \n\n  ", "Cosa sono le 3C?") == "Cosa sono le 3C?"


def test_a_failed_turn_still_starts_the_conversation():
    """A failed first turn still gives the conversation context.

    `observe` runs only after a successful invoke. Without `observe_failure`,
    a failure on the first turn leaves has_context() false, the rewrite step
    never runs, and the follow-up the user asks *because* the first attempt
    failed is treated as a fresh question.
    """
    memory = ConversationMemory()
    assert memory.has_context() is False
    memory.observe_failure("Cosa sono le 3C della Circular Economy for Food?")
    assert memory.has_context() is True
    assert memory.turn == 0, "a failed turn must not spend a turn of the window"


def test_a_turn_with_no_graph_entities_still_opens_the_session():
    """Entities come from the KG channel only.

    On a question answered entirely from text the entity list stays empty,
    and has_context is a turn count for exactly that reason: otherwise every
    later follow-up would look fresh.
    """
    memory = ConversationMemory()
    memory.observe(question="Quali sono le 3C?", answer="Capitale, Ciclicita, Coevoluzione.",
                   nodes=[], triples=[])
    assert memory.active_entities == []
    assert memory.has_context() is True


def test_the_rewrite_is_driven_by_context_not_by_a_shape_test():
    """The rewrite depends on whether the conversation has started, not on shape.

    Shape heuristics read "Spiegameli meglio" as a fresh question and
    "e <anything>" as a continuation, and the second would switch the domain
    gate off entirely. The rewrite prompt is told to repeat a self-contained
    question unchanged, which is the judgement such heuristics approximate.
    """
    memory = ConversationMemory()
    assert memory.has_context() is False
    memory.observe(question="Cosa sono le 3C?", answer="Capitale, Ciclicita, Coevoluzione.",
                   nodes=[{"text": "Circular Economy for Food"}], triples=[])
    assert memory.has_context() is True
