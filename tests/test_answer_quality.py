"""Unit tests for answer granularity and answer language.

A HIGH complexity answer drops the "1-2 short paragraphs" cap and asks for
concrete data, the answer language is pinned to the question language with a
directive written in that language, and a wrong-language answer triggers
exactly one retry. Also guards the invariant that both features leave the
prompt untouched when off, so gold runs and experiment baselines stay
comparable.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphrag.agent.evidence import build_evidence_index, render_reference_list
from graphrag.config import AgentConfig, OUTPUT_COMPLEXITY
from graphrag.llm.manager import LLMManager
from graphrag.llm.prompts import PromptLibrary


class _FakeOutput:
    """Chat-model output carrying only `content`."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    """Returns the queued answers in order and records the prompts it saw."""

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self.prompts: list[Any] = []

    def invoke(self, payload: Any) -> _FakeOutput:
        self.prompts.append(payload)
        return _FakeOutput(self._answers.pop(0) if self._answers else "")


def _manager() -> LLMManager:
    """An `LLMManager` on the vLLM path, for tests that swap in a fake model."""
    return LLMManager(model_id="test-model", use_vllm=True)


# --- granularity ----------------------------------------------------------


def test_medium_complexity_keeps_the_short_answer_cap():
    """Low and medium complexity keep the short-answer cap and no specificity rule."""
    for complexity in (OUTPUT_COMPLEXITY.LOW, OUTPUT_COMPLEXITY.MEDIUM):
        rendered = str(PromptLibrary.answer_prompt(AgentConfig(complexity=complexity)))

        assert "1-2 short paragraphs" in rendered
        assert "Stay concrete" not in rendered


def test_high_complexity_drops_the_cap_and_asks_for_specifics():
    rendered = str(
        PromptLibrary.answer_prompt(AgentConfig(complexity=OUTPUT_COMPLEXITY.HIGH))
    )

    assert "1-2 short paragraphs" not in rendered
    assert "across several paragraphs" in rendered
    # The rule that attacks the "fuffa": prefer the datum over the adjective.
    assert "never generalise when a specific one is available" in rendered
    assert "Provide a thorough, multi-paragraph analysis." in rendered


def test_high_complexity_composes_with_citations():
    config = AgentConfig(complexity=OUTPUT_COMPLEXITY.HIGH, cite_evidence=True)
    rendered = str(PromptLibrary.answer_prompt(config))

    assert "Stay concrete" in rendered
    assert "[S1], [S2]" in rendered


# --- language -------------------------------------------------------------


def test_prompt_unchanged_when_language_is_not_enforced():
    config = AgentConfig()
    assert str(PromptLibrary.answer_prompt(config)) == str(
        PromptLibrary.answer_prompt(config, language=None)
    )
    assert "Rispondi SEMPRE in italiano" not in str(PromptLibrary.answer_prompt(config))


def test_language_directive_is_written_in_the_target_language():
    italian = PromptLibrary.language_directive("it")
    english = PromptLibrary.language_directive("en")

    assert italian.startswith("Rispondi SEMPRE in italiano")
    assert english.startswith("ALWAYS answer in English")
    assert PromptLibrary.language_directive("de") == ""


def test_reinforced_directive_names_the_previous_failure():
    reinforced = PromptLibrary.language_directive("it", reinforced=True)

    assert reinforced.startswith("VINCOLO ASSOLUTO")
    assert "Rispondi SEMPRE in italiano" in reinforced


def test_limits_section_title_follows_the_answer_language():
    config = AgentConfig(always_include_limits=True)

    assert "Limiti e affidabilità" in str(
        PromptLibrary.answer_prompt(config, language="it")
    )
    assert "Limits and confidence" in str(
        PromptLibrary.answer_prompt(config, language="en")
    )
    assert "Limits and confidence" in str(PromptLibrary.answer_prompt(config))


def test_language_constraint_appears_twice_in_the_prompt():
    """Once in the system message, once as the last line before generation."""
    rendered = str(PromptLibrary.answer_prompt(AgentConfig(), language="it"))

    assert rendered.count("Rispondi SEMPRE in italiano") == 2


def test_detects_short_italian_questions():
    for question in (
        "Cos'è SEeD?",
        "Che cos'è la coevoluzione?",
        "Definizione di capitale relazionale?",
        "Mi indichi le strategie nel settore vino?",
        "Approfondisci l'economia circolare in Piemonte",
    ):
        assert LLMManager._detect_query_language(question) == "it", question


def test_an_imperative_with_the_pronoun_attached_is_italian():
    """These carry no function word at all, so both marker counts are zero and
    the tie would go to English, answering an Italian speaker in English.
    """
    for question in (
        "Spiegameli meglio",
        "Spiegamelo meglio",
        "Mostrameli",
        "Elencameli tutti",
        "Dammene un esempio",
        "Parlami del micelio",
    ):
        assert LLMManager._detect_query_language(question) == "it", question


ARTICULATED_PREPOSITIONS = [
    "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "del", "dello", "della", "dei", "degli", "delle",
]


@pytest.mark.parametrize("preposition", ARTICULATED_PREPOSITIONS)
def test_every_articulated_preposition_is_an_italian_signal(preposition: str) -> None:
    """Every articulated preposition counts, not some of each family.

    A question whose only Italian signal is a missing one scores zero on both
    sides, and the tie goes to English. The phrase carries no other marker, so
    the preposition is what is being tested.
    """
    assert LLMManager._detect_query_language(f"{preposition} system thinking") == "it"


def test_a_phrase_with_no_italian_signal_is_still_english():
    """The control for the test above.

    Without the preposition there is no evidence either way, and the tie must
    keep going to English: flipping it breaks more questions than it repairs.
    """
    assert LLMManager._detect_query_language("qqq system thinking") == "en"


def test_the_recorded_question_that_was_answered_in_the_wrong_language():
    """A real question from the session logs, which must be detected as Italian."""
    assert LLMManager._detect_query_language("Sai dirmi qualcosa sul system thinking?") == "it"


def test_short_english_questions_without_function_words_stay_english():
    """These score zero on both sides too, and are the reason the tie defaults
    to English."""
    for question in (
        "Biochar production yield?",
        "SEeD definition?",
        "Circular economy metrics?",
        "Industrial symbiosis examples",
        "Grape stalks percentage",
        "Food waste recovery strategies",
    ):
        assert LLMManager._detect_query_language(question) == "en", question


def test_english_questions_are_not_flipped():
    for question in (
        "What is SEeD?",
        "Which supply chains are covered by the Piedmont strategy?",
        "How does the CEFF framework relate to the SDGs?",
        "Define relational capital.",
    ):
        assert LLMManager._detect_query_language(question) == "en", question


def test_answer_language_ignores_reference_tags_and_source_list():
    answer = (
        "SEeD è un progetto di design sistemico degli eventi [S1]. "
        "Il modello riduce l'impatto del 65% [T2].\n\n"
        "Fonti:\n- [S1] SEeD for Change.pdf | p. 3\n"
        "- [T2] (SEeD, IMPLEMENTS, UNI ISO 20121) — SEeD for Change.pdf | p. 4"
    )

    assert LLMManager._detect_text_language(answer) == "it"


def test_wrong_language_answer_is_regenerated_once():
    manager = _manager()
    model = _FakeModel("SEeD is a systemic event design project for the fairs.")
    config = AgentConfig(enforce_language=True)

    answer = manager._enforce_answer_language(
        model=model,
        query="Che cos'è SEeD?",
        context="…",
        config=config,
        answer="SEeD is a systemic event design project for the fairs.",
        target_language="it",
    )

    # The queue is exhausted, so the retry returns "" and the first answer wins;
    # what matters here is that exactly one extra call was made.
    assert len(model.prompts) == 1
    assert answer == "SEeD is a systemic event design project for the fairs."


def test_retry_answer_replaces_the_original_when_the_language_is_fixed():
    manager = _manager()
    model = _FakeModel(
        "SEeD è un progetto di design sistemico degli eventi, nato nel 2005."
    )
    config = AgentConfig(enforce_language=True)

    answer = manager._enforce_answer_language(
        model=model,
        query="Che cos'è SEeD?",
        context="…",
        config=config,
        answer="SEeD is a systemic event design project.",
        target_language="it",
    )

    assert answer.startswith("SEeD è un progetto")
    assert "VINCOLO ASSOLUTO" in str(model.prompts[0])


def test_matching_language_skips_the_retry():
    manager = _manager()
    model = _FakeModel("mai chiamato")

    answer = manager._enforce_answer_language(
        model=model,
        query="Che cos'è SEeD?",
        context="…",
        config=AgentConfig(enforce_language=True),
        answer="SEeD è un progetto di design sistemico degli eventi.",
        target_language="it",
    )

    assert model.prompts == []
    assert answer == "SEeD è un progetto di design sistemico degli eventi."


def test_retry_failure_keeps_the_original_answer():
    class _BrokenModel:
        def invoke(self, payload: Any) -> Any:
            raise ValueError("backend down")

    manager = _manager()
    answer = manager._enforce_answer_language(
        model=_BrokenModel(),
        query="Che cos'è SEeD?",
        context="…",
        config=AgentConfig(enforce_language=True),
        answer="SEeD is a systemic event design project.",
        target_language="it",
    )

    assert answer == "SEeD is a systemic event design project."


# --- token cap ------------------------------------------------------------


def test_answer_cut_by_the_token_cap_loses_only_the_fragment():
    answer = (
        "SEeD è un progetto di design sistemico degli eventi. "
        "Riduce l'impatto del 65%. "
        "Pertanto, mentre l'interpretazione è ben fondata, la specifica"
    )
    trimmed = LLMManager._trim_to_last_sentence(answer)

    assert trimmed.endswith("Riduce l'impatto del 65%.")
    assert "la specifica" not in trimmed


def test_complete_answer_is_left_alone():
    answer = "Prima frase. Seconda frase."

    assert LLMManager._trim_to_last_sentence(answer) == answer


def test_trimming_never_reduces_the_answer_to_a_stub():
    # One long sentence followed by a fragment: cutting back would throw away
    # most of the answer to hide a few words.
    answer = "Punto. " + "parola " * 60

    assert LLMManager._trim_to_last_sentence(answer) == answer.rstrip()


def test_token_limit_is_read_from_the_backend_metadata():
    class _Output:
        response_metadata = {"finish_reason": "length"}

    class _Complete:
        response_metadata = {"finish_reason": "stop"}

    assert LLMManager._hit_token_limit(_Output()) is True
    assert LLMManager._hit_token_limit(_Complete()) is False
    assert LLMManager._hit_token_limit(object()) is False


# --- source list ordering -------------------------------------------------


def test_reference_list_shows_text_passages_before_triples():
    """Text items carry document *and* page: they must survive the cap."""
    evidence = build_evidence_index(
        text_chunks=[
            {"content": f"passo {i}", "source": f"doc{i}.pdf#page={i}#chunk=1", "chunk_id": f"c{i}"}
            for i in range(1, 4)
        ],
        triples=[
            {"subject": f"A{i}", "predicate": "REL", "object": f"B{i}"} for i in range(1, 10)
        ],
    )
    # Citation order puts the triples first, as the model usually does.
    cited = [f"T{i}" for i in range(1, 10)] + ["S1", "S2", "S3"]
    rendered = render_reference_list(evidence, cited_refs=cited, max_items=8)

    lines = [line for line in rendered.splitlines() if line.startswith("- [")]
    assert [line[3:5] for line in lines[:3]] == ["S1", "S2", "S3"]
    assert "(+4 altre evidenze citate)" in rendered
