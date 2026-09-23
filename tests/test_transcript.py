"""The assistant has to recognise its own words when the user quotes them.

When the reader quotes a sentence the assistant wrote ("hai scritto grafiche
di molti packaging alimentari, quali?"), the assistant must not answer that
the premise is unsupported by the evidence, or that it is factually wrong.
Without its own prose carried forward, a reference to it arrives as a bare
claim and the grounding rule turns that into a denial.

Two properties are load-bearing here:

* **The transcript is never evidence.** Reference tags are stripped out of it,
  so a claim the model made earlier cannot come back wearing a source id.
* **Nothing changes when memory is off.** No memory means no transcript slot and
  the template without it, byte for byte, which is what gold runs see.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphrag.agent.memory import ConversationMemory
from graphrag.config import AgentConfig
from graphrag.llm.prompts import PromptLibrary

# A real answer fragment and the follow-up that quoted it, from a recorded
# session.
ANSWER_2026_09_03 = (
    "Un esempio concreto di questa dinamica è rappresentato dalle grafiche di "
    "molti packaging alimentari presenti sul mercato, che riportano immagini, "
    "nomi ed iconografie sviluppate da persone comuni [REPORT MATTM, p. 70].\n\n"
    "Limiti e affidabilità\nL'affidabilità è elevata [MR37, p. 26].\n\n"
    "Fonti:\n"
    "- **REPORT MATTM_Definitivo.pdf**\n"
    "  - passaggi citati: p. 70\n"
)
FOLLOW_UP_2026_09_03 = "Hai scritto grafiche di molti packaging alimentari, quali?"


def _memory(*turns: tuple[str, str]) -> ConversationMemory:
    """Memory after the given (question, answer) turns."""
    memory = ConversationMemory()
    for question, answer in turns:
        memory.observe(question=question, answer=answer)
    return memory


def test_the_transcript_carries_what_the_assistant_said() -> None:
    memory = _memory(("Spiegami la Coevoluzione", ANSWER_2026_09_03))
    transcript = memory.transcript()

    assert "Spiegami la Coevoluzione" in transcript
    # The exact span the expert quoted back has to be findable in it.
    assert "grafiche di molti packaging alimentari presenti sul mercato" in transcript
    assert transcript.startswith("User: ")
    assert "\nAssistant: " in transcript


def test_reference_tags_never_reach_the_transcript() -> None:
    """A tag in the transcript is a source id the model can recite."""
    memory = _memory(("q", ANSWER_2026_09_03))
    transcript = memory.transcript()

    assert "[REPORT MATTM, p. 70]" not in transcript
    assert "[MR37, p. 26]" not in transcript
    # No bracketed span survives at all: the tag is the citable part, and a
    # partial strip that leaves "REPORT MATTM, p. 70" bare is just as reusable.
    assert "MATTM" not in transcript
    assert "MR37" not in transcript
    assert "[" not in transcript


@pytest.mark.parametrize(
    "tag",
    ["[S1]", "[T12]", "[MR37, p. 26; REPORT MATTM, p. 68]", "[Materia 45, p. 4-4]"],
)
def test_every_tag_shape_the_prompt_produces_is_stripped(tag: str) -> None:
    memory = _memory(("q", f"Una affermazione {tag} e la sua coda."))
    transcript = memory.transcript()

    assert tag not in transcript
    assert "Una affermazione e la sua coda." in transcript


def test_the_omission_marker_survives() -> None:
    """`[...]` is the elision the definitional prompt asks for, not a tag."""
    memory = _memory(("q", "«una definizione [...] abbreviata»"))

    assert "[...]" in memory.transcript()


def test_the_generated_source_list_is_dropped() -> None:
    memory = _memory(("q", ANSWER_2026_09_03))
    transcript = memory.transcript()

    assert "Fonti:" not in transcript
    assert "REPORT MATTM_Definitivo.pdf" not in transcript
    # The prose before it stays.
    assert "persone comuni" in transcript


def test_the_budget_drops_the_oldest_turn_first() -> None:
    memory = ConversationMemory()
    memory.max_transcript_chars = 400
    for i in range(6):
        memory.observe(question=f"domanda {i}", answer="x" * 150)

    transcript = memory.transcript()
    assert "domanda 5" in transcript
    assert "domanda 0" not in transcript
    assert len(transcript) < 1000


def test_a_single_oversized_turn_is_still_kept() -> None:
    """Trimming to nothing would lose exactly the turn most likely quoted."""
    memory = ConversationMemory()
    memory.max_transcript_chars = 100
    memory.observe(question="domanda", answer="y" * 5000)

    assert "domanda" in memory.transcript()


def test_reset_clears_the_transcript() -> None:
    memory = _memory(("q", ANSWER_2026_09_03))
    memory.reset()

    assert memory.transcript() == ""


def test_no_turn_yet_means_no_transcript() -> None:
    assert ConversationMemory().transcript() == ""


def test_the_prompt_gains_a_slot_only_when_asked() -> None:
    config = AgentConfig()

    without = PromptLibrary.answer_prompt(config)
    with_transcript = PromptLibrary.answer_prompt(config, transcript=True)

    assert set(without.input_variables) == {"question", "context"}
    assert set(with_transcript.input_variables) == {"question", "context", "transcript"}


def test_the_prompt_without_a_transcript_is_unchanged() -> None:
    """Gold runs and experiment baselines see the template without the slot."""
    config = AgentConfig()
    rendered = PromptLibrary.answer_prompt(config).invoke(
        {"question": "q", "context": "c"}
    )
    text = str(rendered)

    assert "Conversation so far" not in text
    assert "never cite them" not in text


def test_the_transcript_is_marked_as_not_evidence() -> None:
    config = AgentConfig()
    rendered = PromptLibrary.answer_prompt(config, transcript=True).invoke(
        {"question": "q", "context": "c", "transcript": "User: a\n\nAssistant: b"}
    )
    text = str(rendered)

    assert "not evidence" in text
    assert "Never cite them" in text
    # The instruction the transcript exists for.
    assert "do not deny it" in text


def test_the_transcript_sits_before_the_question_and_the_context() -> None:
    """Ordering is what lets the served prefix cache reuse it across turns."""
    config = AgentConfig()
    rendered = PromptLibrary.answer_prompt(config, transcript=True).invoke(
        {"question": "QUESTIONMARK", "context": "CONTEXTMARK", "transcript": "TRMARK"}
    )
    text = str(rendered)

    assert text.index("TRMARK") < text.index("QUESTIONMARK") < text.index("CONTEXTMARK")


class _FakeOutput:
    """Chat-model output with `content` and no metadata."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.response_metadata: dict[str, Any] = {}


class _FakeModel:
    """Model that records its prompts and answers in Italian."""

    def __init__(self) -> None:
        self.prompts: list[Any] = []

    def invoke(self, payload: Any) -> _FakeOutput:
        self.prompts.append(payload)
        return _FakeOutput("una risposta in italiano")


def test_generate_puts_the_transcript_in_the_rendered_prompt() -> None:
    from graphrag.llm.manager import LLMManager

    manager = LLMManager.__new__(LLMManager)
    model = _FakeModel()
    manager.load_llm = lambda: model  # type: ignore[method-assign]
    manager._invoke_with_retry = lambda m, payload: m.invoke(payload)  # type: ignore[method-assign]

    config = AgentConfig()
    config.enforce_language = False
    manager.generate(
        query="Hai scritto grafiche, quali?",
        context="contesto",
        config=config,
        transcript="User: q\n\nAssistant: grafiche di molti packaging alimentari",
    )

    assert model.prompts, "the model was never invoked"
    sent = str(model.prompts[0])
    assert "grafiche di molti packaging alimentari" in sent
    assert "Conversation so far" in sent


def test_generate_without_a_transcript_sends_no_transcript_block() -> None:
    from graphrag.llm.manager import LLMManager

    manager = LLMManager.__new__(LLMManager)
    model = _FakeModel()
    manager.load_llm = lambda: model  # type: ignore[method-assign]
    manager._invoke_with_retry = lambda m, payload: m.invoke(payload)  # type: ignore[method-assign]

    config = AgentConfig()
    config.enforce_language = False
    manager.generate(query="q", context="contesto", config=config)

    assert "Conversation so far" not in str(model.prompts[0])
