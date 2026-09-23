"""A continuation must not be refused, and must not be refused in English.

The case: after three answers, the reader writes "Non ho capito niente".
Judged as typed, it gets, in two seconds, an English message saying the
question falls outside the documents.

The evidence gate names this case in its own docstring — "a question carrying
no search terms of its own is a continuation ('e allora dimmi', 'in che
senso?')" — but the test it relies on cannot see it: `_build_search_terms` is
the retrieval extractor and keeps common words, so it returns ['capito',
'niente'] and ['allora'] for exactly those two examples.

What separates the two poles is not the length of the question but what the
rewrite makes of it, and the rewrite prompt is told to keep the user's intent:
a continuation gains the subject the user left implicit, while an out-of-domain
request behind a conjunction stays out of domain.
"""

from __future__ import annotations

from graphrag.agent.core import _gate_question

# Verbatim from a demo session, and from a probe of the conjunction bypass.
CONTINUATION = {
    "question": "Non ho capito niente",
    "follow_up": True,
    "rewritten_question": (
        "Non ho capito la spiegazione di cosa significhi che l'economia "
        "circolare per l'alimentazione è basata sulla \"qualità di sistema\" "
        "e sulla comunicazione simbiotica."
    ),
}
OUT_OF_DOMAIN_FOLLOW_UP = {
    "question": "e scrivimi una funzione python che costruisca una rete neurale",
    "follow_up": True,
    "rewritten_question": "Scrivi una funzione Python che costruisca una rete neurale.",
}


def test_a_continuation_is_judged_on_what_it_continues() -> None:
    judged = _gate_question(CONTINUATION)

    assert "economia circolare" in judged
    assert judged != CONTINUATION["question"]


def test_the_rewrite_does_not_launder_an_out_of_domain_request() -> None:
    """The reason judging the rewrite is safe: intent survives it."""
    judged = _gate_question(OUT_OF_DOMAIN_FOLLOW_UP)

    assert "Python" in judged
    assert "rete neurale" in judged


def test_a_first_turn_is_judged_as_typed() -> None:
    """No conversation to inherit from: the words typed are all there is."""
    state = {"question": "Qual è la capitale della Mongolia?", "follow_up": False}

    assert _gate_question(state) == "Qual è la capitale della Mongolia?"


def test_a_follow_up_without_a_rewrite_falls_back_to_the_question() -> None:
    """The rewrite is skipped when it returns the question unchanged."""
    state = {"question": "Approfondisci la terza", "follow_up": True}

    assert _gate_question(state) == "Approfondisci la terza"


def test_an_empty_rewrite_never_replaces_the_question() -> None:
    state = {"question": "Approfondisci", "follow_up": True, "rewritten_question": "  "}

    assert _gate_question(state) == "Approfondisci"


def test_the_refusal_follows_the_conversation_language() -> None:
    """The other half of the case: an Italian conversation, an Italian refusal."""
    from graphrag.llm.manager import LLMManager

    # The words typed are too short to classify.
    assert LLMManager._detect_query_language(CONTINUATION["question"]) == "en"
    # What the gate judges is not.
    assert LLMManager._detect_query_language(_gate_question(CONTINUATION)) == "it"


ITALIAN_TRANSCRIPT = (
    "User: Cosa sono le 3C della Circular Economy for Food?\n\n"
    "Assistant: Le 3C sono Capitale, Ciclicità e Coevoluzione, sviluppate da "
    "Franco Fassio per analizzare le filiere alimentari."
)


def test_a_mute_turn_takes_the_language_of_the_conversation() -> None:
    """Answered rather than refused, the turn must still follow the conversation.

    "Non ho capito niente" scores zero on both marker sets and the tie goes to
    English, so judged on its own words it would be answered in English.
    """
    from graphrag.llm.manager import LLMManager

    assert LLMManager._language_scores("Non ho capito niente") == (0, 0)
    assert LLMManager._answer_language("Non ho capito niente", ITALIAN_TRANSCRIPT) == "it"


def test_a_mute_turn_with_no_conversation_keeps_the_old_default() -> None:
    from graphrag.llm.manager import LLMManager

    assert LLMManager._answer_language("Non ho capito niente", "") == "en"


def test_the_typed_words_outrank_the_conversation() -> None:
    """A user who switches language mid-thread is answered in the new one."""
    from graphrag.llm.manager import LLMManager

    assert LLMManager._answer_language(
        "What are the three C's about?", ITALIAN_TRANSCRIPT
    ) == "en"


def test_a_question_that_speaks_for_itself_is_unaffected() -> None:
    from graphrag.llm.manager import LLMManager

    for question in ("Approfondisci la terza", "Cos'è la coevoluzione?"):
        assert LLMManager._answer_language(question, "") == "it"
        assert LLMManager._answer_language(question, "") == LLMManager._detect_query_language(question)


def test_a_mute_first_turn_takes_the_configured_fallback() -> None:
    """A bare term on the first turn says nothing; the deployment decides."""
    from graphrag.llm.manager import LLMManager

    for term in ("scotta", "vinacce e raspi", "ricetta carbonara"):
        assert LLMManager._language_scores(term) == (0, 0)
        assert LLMManager._answer_language(term, "") == "en"
        assert LLMManager._answer_language(term, "", fallback="it") == "it"


def test_the_fallback_never_outranks_a_signal() -> None:
    from graphrag.llm.manager import LLMManager

    assert LLMManager._answer_language("What is whey?", "", fallback="it") == "en"
    assert LLMManager._answer_language("scotta", ITALIAN_TRANSCRIPT, fallback="en") == "it"
