"""Whether the agent answers at all, and what it judges before deciding.

The deciding part of `agent/core.py`: which form of a question the gate
judges, what evidence it is shown, whether retrieval counts as relevant, and
which terms all of that turns on.

These paths decide whether a legitimate follow-up is refused or an
out-of-domain question waved through, so what is pinned is the rule, not the
verdict. No graph, no LLM, no encoder: the retriever, its store and the model
are fakes that answer what the test chose.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from graphrag.agent.core import (
    KGRAGAgent,
    _content_terms,
    _gate_mode,
    _gate_question,
    _plausible_rewrite,
    _proper_noun_terms,
    _term_matches,
)
from graphrag.config import AgentConfig


# --- fakes -----------------------------------------------------------------


class _Chunk:
    """Text chunk as the text pipeline returns it."""

    def __init__(self, text: str, source: str = "report.pdf#page=3") -> None:
        self.text = text
        self.content = text
        self.source = source


class _Pipeline:
    """Text pipeline returning fixed chunks, or raising, and recording each call."""

    def __init__(self, chunks: list[_Chunk] | None = None, raises: Exception | None = None):
        self.chunks = chunks or []
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int = 3):
        self.calls.append((question, top_k))
        if self.raises is not None:
            raise self.raises
        return list(self.chunks)


class _Store:
    """Graph store whose fulltext search returns fixed nodes, or raises."""

    def __init__(self, nodes: Any = (), raises: Exception | None = None) -> None:
        self.nodes = nodes
        self.raises = raises
        self.calls: list[tuple[Any, int]] = []

    def fulltext_search_nodes(self, terms, limit: int = 0, **kwargs: Any):
        self.calls.append((list(terms), limit))
        if self.raises is not None:
            raise self.raises
        return self.nodes


class _Retriever:
    """KG retriever with fixed search terms, or a failing term extractor."""

    def __init__(self, store: _Store, terms: list[str] | None = None, pipeline=None):
        self.kg_store = store
        self.text_pipeline = pipeline
        self._terms = terms
        self.term_calls: list[str] = []

    def _build_search_terms(self, query_text: str, configured_entity: str) -> list[str]:
        self.term_calls.append(query_text)
        if self._terms is None:
            raise RuntimeError("term extraction unavailable")
        return list(self._terms)


class _LLM:
    """LLM with fixed gate verdicts that records what each gate was shown."""

    def __init__(self, answerable: bool = True, in_domain: bool = True) -> None:
        self.answerable = answerable
        self.in_domain = in_domain
        self.answerable_calls: list[tuple[str, list[str], list[str], list[str]]] = []
        self.domain_calls: list[tuple[str, list[str]]] = []

    def classify_answerable(self, question, names=(), passages=(), sources=()):
        self.answerable_calls.append((question, list(names), list(passages), list(sources)))
        return self.answerable

    def classify_in_domain(self, question, config, known=()):
        self.domain_calls.append((question, list(known)))
        return self.in_domain

    def warmup(self) -> None:
        return None


def _agent(retriever=None, llm=None, **overrides: Any) -> KGRAGAgent:
    """Agent over the given fakes, with warmup and cache off."""
    base: dict[str, Any] = {"llm_warmup": False, "enable_cache": False}
    base.update(overrides)
    return KGRAGAgent(config=AgentConfig(**base), kg_retriever=retriever, llm=llm)


# --- which wording the gate judges -----------------------------------------


def test_a_question_that_is_not_a_follow_up_is_judged_as_typed():
    state = {"question": "cos'e' la scotta?", "rewritten_question": "qualcos'altro"}

    assert _gate_question(state) == "cos'e' la scotta?"


def test_a_follow_up_is_judged_on_its_rewritten_form():
    # "Non ho capito niente" carries no subject, yet its search terms are
    # ['capito', 'niente'], so the "no terms of its own" exemption does not
    # apply: judged as typed, it would be refused.
    state = {
        "question": "Non ho capito niente",
        "follow_up": True,
        "rewritten_question": "Non ho capito la scotta",
    }

    assert _gate_question(state) == "Non ho capito la scotta"


def test_a_follow_up_with_no_rewrite_falls_back_to_what_was_typed():
    state = {"question": "e quindi?", "follow_up": True, "rewritten_question": "  "}

    assert _gate_question(state) == "e quindi?"


@pytest.mark.parametrize(
    "env, expected",
    [(None, "evidence"), ("evidence", "evidence"), ("scope", "evidence"), ("SCOPE", "evidence")],
)
def test_the_gate_mode_is_read_per_call(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("GRAPHRAG_GATE_MODE", raising=False)
    else:
        monkeypatch.setenv("GRAPHRAG_GATE_MODE", env)

    mode = _gate_mode()

    assert mode in {"scope", "evidence"}
    if env and env.strip().lower() == "scope":
        assert mode == "scope"
    else:
        assert mode == "evidence"


# --- the terms the gate looks up -------------------------------------------


def test_an_interrogative_is_not_offered_to_the_gate_as_a_name():
    # Left in, "Chi è Barilla?" would offer the gate "Chi-squared tests"
    # alongside the name that matters.
    assert _proper_noun_terms("Chi è Barilla?") == ["Barilla"]
    assert _proper_noun_terms("Cosa fa la Regione Piemonte?") == ["Regione", "Piemonte"]


def test_a_question_with_no_capitalised_content_offers_no_names():
    assert _proper_noun_terms("cosa contiene la scotta?") == []


def test_the_gate_looks_for_the_same_terms_retrieval_would():
    # A gate searching for different terms than retrieval would judge evidence
    # the answer is not going to be built from.
    retriever = _Retriever(_Store(), terms=["biochar", "digestato"])

    assert _content_terms(retriever, "cos'e' il biochar?") == ["biochar", "digestato"]


def test_a_lowercase_subject_reaches_the_gate_through_the_retriever():
    # `_proper_noun_terms` reads only capitalised tokens, so "biochar" alone
    # would give the gate nothing and it would decide on world knowledge.
    retriever = _Retriever(_Store(), terms=["biochar"])

    assert _content_terms(retriever, "cos'e' il biochar?") == ["biochar"]
    assert _proper_noun_terms("cos'e' il biochar?") == []


def test_a_broken_term_extractor_falls_back_to_capitalised_tokens():
    retriever = _Retriever(_Store(), terms=None)

    assert _content_terms(retriever, "Cosa fa Barilla?") == ["Barilla"]


def test_without_a_retriever_the_capitalised_tokens_are_all_there_is():
    assert _content_terms(None, "Cosa fa Barilla?") == ["Barilla"]


@pytest.mark.parametrize(
    "term, haystack, expected",
    [
        ("rice", "rice husk", True),
        ("rice", "price list", False),
        ("ceff", "ceffpolicy", False),
        ("rice husk", "the rice husk residue", True),
        ("", "anything", False),
        ("rice", "", False),
    ],
)
def test_a_term_matches_only_on_word_boundaries(term, haystack, expected):
    # Plain `term in haystack` lets short terms match inside unrelated words,
    # which inflates every relevance and coverage count that uses it.
    assert _term_matches(term, haystack) is expected


# --- the evidence gate -----------------------------------------------------


def test_a_question_with_no_terms_of_its_own_is_let_through():
    # "e allora dimmi" has no subject to look up, and refusing it ends the
    # conversation the demo exists to hold.
    store = _Store(nodes=[])
    llm = _LLM(answerable=False)
    agent = _agent(_Retriever(store, terms=[]), llm)

    assert agent._evidence_gate("e allora?") == {"in_domain": True}
    assert llm.answerable_calls == []
    assert store.calls == []


def test_without_a_retriever_nothing_is_refused():
    llm = _LLM(answerable=False)

    assert _agent(None, llm)._evidence_gate("qualunque cosa") == {"in_domain": True}
    assert llm.answerable_calls == []


def test_a_failed_lookup_leaves_the_question_in(caplog):
    # A gate that fails must not refuse.
    store = _Store(raises=RuntimeError("index down"))
    llm = _LLM(answerable=False)
    agent = _agent(_Retriever(store, terms=["scotta"]), llm)

    with caplog.at_level(logging.WARNING):
        assert agent._evidence_gate("cos'e' la scotta?") == {"in_domain": True}

    assert "Evidence lookup failed" in caplog.text
    assert llm.answerable_calls == []


def test_no_fulltext_index_leaves_the_question_in():
    # The fallback is a full scan, not worth paying for a hint.
    store = _Store(nodes=None)
    llm = _LLM(answerable=False)
    agent = _agent(_Retriever(store, terms=["scotta"]), llm)

    assert agent._evidence_gate("cos'e' la scotta?") == {"in_domain": True}
    assert llm.answerable_calls == []


def test_the_names_the_collection_holds_are_shown_to_the_model():
    store = _Store(nodes=[{"text": "Scotta"}, {"text": "Siero di latte"}])
    llm = _LLM(answerable=True)
    agent = _agent(_Retriever(store, terms=["scotta"]), llm)

    agent._evidence_gate("cos'e' la scotta?")

    assert llm.answerable_calls[0][1] == ["Scotta", "Siero di latte"]


def test_the_same_name_twice_is_shown_once():
    store = _Store(nodes=[{"text": "Scotta"}, {"text": "  scotta  "}])
    llm = _LLM()
    agent = _agent(_Retriever(store, terms=["scotta"]), llm)

    agent._evidence_gate("cos'e' la scotta?")

    assert llm.answerable_calls[0][1] == ["Scotta"]


def test_the_list_of_names_is_capped():
    store = _Store(nodes=[{"text": f"Nodo {i}"} for i in range(30)])
    llm = _LLM()
    agent = _agent(_Retriever(store, terms=["nodo"]), llm)

    agent._evidence_gate("q")

    assert len(llm.answerable_calls[0][1]) == 8


def test_passages_are_shown_alongside_the_names():
    # Shown only node names, the model refuses most specific questions: a name
    # cannot carry the figure a specific question asks for.
    pipeline = _Pipeline([_Chunk("La scotta e' il residuo liquido", "report.pdf#page=3")])
    store = _Store(nodes=[{"text": "Scotta"}])
    llm = _LLM()
    agent = _agent(_Retriever(store, terms=["scotta"], pipeline=pipeline), llm)

    agent._evidence_gate("quanta scotta si produce?")

    _, names, passages, sources = llm.answerable_calls[0]
    assert names == ["Scotta"]
    assert passages == ["La scotta e' il residuo liquido"]
    assert sources == ["report.pdf"]


def test_the_documents_name_themselves_rather_than_the_prompt():
    pipeline = _Pipeline(
        [_Chunk("a", "uno.pdf#page=1"), _Chunk("b", "uno.pdf#page=9"), _Chunk("c", "due.pdf")]
    )
    llm = _LLM()
    agent = _agent(_Retriever(_Store(nodes=[]), terms=["x"], pipeline=pipeline), llm)

    agent._evidence_gate("q")

    assert llm.answerable_calls[0][3] == ["uno.pdf", "due.pdf"]


def test_a_broken_passage_channel_does_not_stop_the_gate(caplog):
    pipeline = _Pipeline(raises=RuntimeError("dense index missing"))
    store = _Store(nodes=[{"text": "Scotta"}])
    llm = _LLM(answerable=True)
    agent = _agent(_Retriever(store, terms=["scotta"], pipeline=pipeline), llm)

    with caplog.at_level(logging.WARNING):
        assert agent._evidence_gate("q") == {"in_domain": True}

    assert "Evidence passages unavailable" in caplog.text
    assert llm.answerable_calls[0][2] == []


def test_the_model_can_refuse_a_question_the_collection_does_not_cover():
    store = _Store(nodes=[])
    llm = _LLM(answerable=False)
    agent = _agent(_Retriever(store, terms=["carbonara"]), llm)

    assert agent._evidence_gate("come si fa la carbonara?") == {"in_domain": False}


def test_only_a_few_passages_are_asked_for():
    pipeline = _Pipeline([_Chunk("a")])
    agent = _agent(_Retriever(_Store(nodes=[]), terms=["x"], pipeline=pipeline), _LLM())

    agent._evidence_gate("q")

    assert pipeline.calls[0][1] == 3


# --- the scope gate, and what it exempts -----------------------------------


def test_the_gate_is_skipped_entirely_when_it_is_disabled():
    llm = _LLM(in_domain=False)
    agent = _agent(None, llm, enable_domain_gate=False)

    assert agent._scope_gate({"question": "qualunque cosa"}) == {"in_domain": True}
    assert llm.domain_calls == []


def test_a_gate_without_a_model_cannot_refuse():
    agent = _agent(None, None, enable_domain_gate=True)

    assert agent._scope_gate({"question": "qualunque cosa"}) == {"in_domain": True}


def test_an_empty_question_is_not_judged():
    llm = _LLM(in_domain=False)
    agent = _agent(None, llm, enable_domain_gate=True)

    assert agent._scope_gate({"question": "   "}) == {"in_domain": True}
    assert llm.domain_calls == []


def test_in_scope_mode_the_follow_up_flag_no_longer_exempts_anything(monkeypatch):
    # A follow-up is not exempt as such: the only exemption is the word floor
    # below, whatever the follow-up flag says.
    monkeypatch.setenv("GRAPHRAG_GATE_MODE", "scope")
    llm = _LLM(in_domain=False)
    agent = _agent(None, llm, enable_domain_gate=True)

    verdict = agent._scope_gate(
        {"question": "e scrivimi una funzione python", "follow_up": True}
    )

    assert verdict == {"in_domain": False}
    assert llm.domain_calls[0][0] == "e scrivimi una funzione python"


@pytest.mark.parametrize("question", ["in che senso?", "perché?", "non ho capito"])
def test_in_scope_mode_a_question_too_short_to_judge_is_exempt(monkeypatch, question):
    # Three words or fewer carry no topic of their own. The floor stands in for
    # a follow-up exemption, and it does not care what the memory thinks.
    monkeypatch.setenv("GRAPHRAG_GATE_MODE", "scope")
    llm = _LLM(in_domain=False)
    agent = _agent(None, llm, enable_domain_gate=True)

    assert agent._scope_gate({"question": question}) == {"in_domain": True}
    assert llm.domain_calls == []


def test_in_evidence_mode_a_follow_up_is_judged_like_anything_else(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_GATE_MODE", "evidence")
    store = _Store(nodes=[])
    llm = _LLM(answerable=False)
    agent = _agent(
        _Retriever(store, terms=["funzione", "python"]), llm, enable_domain_gate=True
    )

    verdict = agent._scope_gate(
        {"question": "e scrivimi una funzione python", "follow_up": True}
    )

    assert verdict == {"in_domain": False}


# --- discarding an implausible rewrite -------------------------------------


def test_a_one_line_rewrite_is_taken_as_it_is():
    assert _plausible_rewrite("Cosa sono le 3C?", "3C?") == "Cosa sono le 3C?"


def test_a_label_on_its_own_line_is_tolerated():
    raw = "Rewritten question:\nCosa sono le 3C del Circular Economy for Food?"

    assert _plausible_rewrite(raw, "3C?").startswith("Cosa sono le 3C")


@pytest.mark.parametrize("label", ["Rewritten question:", "Rewritten:"])
def test_an_inline_label_is_stripped(label):
    assert _plausible_rewrite(f"{label} Cosa sono le 3C?", "3C?") == "Cosa sono le 3C?"


def test_an_essay_is_discarded_in_favour_of_the_question(caplog):
    # A model can answer with 1500 characters of markdown offering numbered
    # options and a "Key Improvements Made" section (Gemma-4-31B does). Fed to
    # the retriever whole, it buries the question.
    raw = "Here are three options:\n1. Una\n2. Due\n3. Tre\n\nKey Improvements Made:\n- ..."

    with caplog.at_level(logging.WARNING):
        assert _plausible_rewrite(raw, "Cosa sono le 3C?") == "Cosa sono le 3C?"

    assert "implausible rewrite" in caplog.text


def test_a_line_that_introduces_something_is_not_a_rewrite():
    assert _plausible_rewrite("Ecco la domanda riscritta:", "3C?") == "3C?"


def test_an_empty_reply_keeps_the_question():
    assert _plausible_rewrite("", "3C?") == "3C?"
    assert _plausible_rewrite("   \n  ", "3C?") == "3C?"


def test_a_rewrite_far_longer_than_the_question_is_discarded():
    raw = "x" * 500

    assert _plausible_rewrite(raw, "3C?") == "3C?"


def test_markdown_decoration_is_stripped():
    assert _plausible_rewrite("> **Cosa sono le 3C?**", "3C?").endswith("3C?**")


def test_quotes_around_the_rewrite_are_removed():
    assert _plausible_rewrite('"Cosa sono le 3C?"', "3C?") == "Cosa sono le 3C?"


# --- grading what came back ------------------------------------------------


def _graded(**state: Any) -> str:
    """The relevance verdict `_grade` gives a hand-built state."""
    return _agent()._grade(state)["relevance"]


def test_nothing_retrieved_is_never_relevant():
    assert _graded(question="cos'e' la scotta?") == "not_relevant"


def test_evidence_that_mentions_the_question_is_relevant():
    assert (
        _graded(
            question="cos'e' la scotta?",
            kg_triples=[
                {"subject": "Scotta", "predicate": "IS_TYPE_OF", "object": "Residuo"},
                {"subject": "Scotta", "predicate": "USES", "object": "Siero"},
            ],
        )
        == "relevant"
    )


def test_evidence_about_something_else_is_not_relevant():
    assert (
        _graded(
            question="cos'e' la scotta?",
            kg_triples=[
                {"subject": "Barilla", "predicate": "PRODUCES", "object": "Pasta"},
                {"subject": "Milano", "predicate": "LOCATED_IN", "object": "Italia"},
                {"subject": "Torino", "predicate": "LOCATED_IN", "object": "Italia"},
                {"subject": "Roma", "predicate": "LOCATED_IN", "object": "Italia"},
            ],
        )
        == "not_relevant"
    )


def test_one_match_in_a_pile_of_evidence_is_not_enough():
    triples = [
        {"subject": "Scotta", "predicate": "IS_TYPE_OF", "object": "Residuo"},
    ] + [
        {"subject": f"Altro {i}", "predicate": "X", "object": "Y"} for i in range(9)
    ]

    assert _graded(question="cos'e' la scotta?", kg_triples=triples) == "not_relevant"


def test_one_match_out_of_two_units_is_enough():
    assert (
        _graded(
            question="cos'e' la scotta?",
            kg_triples=[
                {"subject": "Scotta", "predicate": "IS_TYPE_OF", "object": "Residuo"},
                {"subject": "Altro", "predicate": "X", "object": "Y"},
            ],
        )
        == "relevant"
    )


def test_text_alone_needs_only_one_match():
    # With no KG evidence the ratio rule cannot apply: one hit in the passage
    # is the whole of the evidence.
    assert (
        _graded(question="cos'e' la scotta?", text_context="La scotta e' un residuo.")
        == "relevant"
    )


def test_text_about_something_else_is_not_relevant():
    assert (
        _graded(question="cos'e' la scotta?", text_context="Il packaging alimentare.")
        == "not_relevant"
    )


def test_a_matching_node_counts_as_evidence():
    assert (
        _graded(
            question="cos'e' la scotta?",
            retrieved_nodes_count=1,
            retrieved_nodes=[{"text": "Scotta"}],
        )
        == "relevant"
    )


def test_the_subgraph_is_graded_too():
    assert (
        _graded(
            question="cos'e' la scotta?",
            retrieved_subgraph_count=1,
            retrieved_subgraph=[
                {"subject": "Scotta", "predicate": "PART_OF", "object": "Siero"}
            ],
        )
        == "relevant"
    )


def test_the_shortest_path_is_graded_too():
    assert (
        _graded(
            question="cos'e' la scotta?",
            retrieved_shortest_path_count=1,
            retrieved_shortest_path=[
                {"subject": "Scotta", "predicate": "PART_OF", "object": "Siero"}
            ],
        )
        == "relevant"
    )


def test_grading_uses_the_rewritten_question_when_there_is_one():
    assert (
        _graded(
            question="e quindi?",
            rewritten_question="cos'e' la scotta?",
            kg_triples=[
                {"subject": "Scotta", "predicate": "IS_TYPE_OF", "object": "Residuo"},
                {"subject": "Scotta", "predicate": "USES", "object": "Siero"},
            ],
        )
        == "relevant"
    )


def test_a_term_inside_a_longer_word_does_not_count_as_a_match():
    assert (
        _graded(
            question="what about rice?",
            kg_triples=[
                {"subject": "Price list", "predicate": "X", "object": "Y"},
                {"subject": "Prices", "predicate": "X", "object": "Z"},
            ],
        )
        == "not_relevant"
    )


# --- the terms grading turns on --------------------------------------------


def test_an_acronym_comes_before_a_proper_noun_before_a_content_word():
    terms = KGRAGAgent._extract_salient_terms_from_text("Cosa dice il MATTM su Piemonte e scotta?")

    assert terms.index("mattm") < terms.index("piemonte")
    assert "scotta" in terms


def test_a_question_with_no_capitalisation_still_yields_terms():
    # Extracting only capitalised terms returns nothing here, which would send
    # grading to the Italian-only fallback and make it a no-op on English.
    terms = KGRAGAgent._extract_salient_terms_from_text("what does rice husk contain?")

    assert "rice" in terms and "husk" in terms


def test_stopwords_never_become_salient_terms():
    terms = KGRAGAgent._extract_salient_terms_from_text("the and for with that")

    assert terms == []


def test_the_term_list_is_capped():
    text = " ".join(f"Parola{i}" for i in range(40))

    assert len(KGRAGAgent._extract_salient_terms_from_text(text)) == 16


def test_terms_are_mined_from_triples_as_acronyms_only():
    terms = KGRAGAgent._extract_salient_terms_from_triples(
        [{"subject": "MATTM", "predicate": "PUBLISHED", "object": "Report annuale"}]
    )

    assert "mattm" in terms
    assert "report" not in terms


def test_the_context_contributes_terms_too():
    terms = KGRAGAgent._extract_salient_terms(
        query="cos'e'?", context="La SCOTTA e' un residuo di Piemonte"
    )

    assert "scotta" in terms and "piemonte" in terms


def test_highlights_keep_the_lines_and_their_order():
    # The salient terms are mined from the query *and* the context, so in
    # practice almost every context line carries one: this selects order and
    # count far more than relevance. Pinned as it behaves, not as it reads.
    context = "La scotta e' un residuo.\nIl packaging e' altro.\nAncora scotta qui."

    highlights = KGRAGAgent._extract_context_highlights("cos'e' la scotta?", context)

    assert highlights == [
        "La scotta e' un residuo.",
        "Il packaging e' altro.",
        "Ancora scotta qui.",
    ]


def test_a_line_sharing_nothing_with_query_or_context_is_left_out():
    context = "La scotta e' un residuo.\n???"

    highlights = KGRAGAgent._extract_context_highlights("scotta", context)

    assert highlights == ["La scotta e' un residuo."]


def test_highlights_fall_back_to_the_first_lines_when_nothing_matches():
    context = "Prima riga.\nSeconda riga."

    highlights = KGRAGAgent._extract_context_highlights("xyzzy", context, limit=1)

    assert highlights == ["Prima riga."]


def test_highlights_are_capped():
    context = "\n".join(f"La scotta riga {i}" for i in range(10))

    assert len(KGRAGAgent._extract_context_highlights("scotta", context, limit=3)) == 3


def test_a_triple_summary_prefers_the_triples_the_question_is_about():
    summaries = KGRAGAgent._triple_summaries(
        [
            {"subject": "Barilla", "predicate": "PRODUCES", "object": "Pasta"},
            {"subject": "Scotta", "predicate": "IS_TYPE_OF", "object": "Residuo"},
        ],
        query="cos'e' la scotta?",
    )

    assert summaries == ["(Scotta, IS_TYPE_OF, Residuo)"]


def test_a_triple_summary_falls_back_to_whatever_there_is():
    summaries = KGRAGAgent._triple_summaries(
        [{"subject": "Barilla", "predicate": "PRODUCES", "object": "Pasta"}],
        query="xyzzy?",
    )

    assert summaries == ["(Barilla, PRODUCES, Pasta)"]


def test_summarising_nothing_produces_nothing():
    assert KGRAGAgent._triple_summaries([], query="q") == []


# --- merging what several retrieval queries returned -----------------------


def test_a_node_seen_twice_is_kept_once():
    agent = _agent()
    seen: set[tuple[str, str]] = set()
    existing: list[dict[str, Any]] = []

    agent._merge_nodes(existing, [{"text": "Scotta"}], seen, limit=10)
    agent._merge_nodes(existing, [{"text": "Scotta"}], seen, limit=10)

    assert len(existing) == 1


def test_the_merge_cap_is_checked_before_the_append():
    # Checked after the append, every merge could finish one item over the cap,
    # and with decomposition the limit would be exceeded by up to three.
    agent = _agent()
    seen: set[tuple[str, str]] = set()
    existing: list[dict[str, Any]] = []

    for batch in range(4):
        agent._merge_nodes(
            existing, [{"text": f"n{batch}{i}"} for i in range(5)], seen, limit=6
        )

    assert len(existing) == 6


def test_a_non_list_is_ignored_rather_than_crashing():
    agent = _agent()
    existing = [{"text": "Scotta"}]

    assert agent._merge_nodes(existing, "not a list", set(), limit=10) == existing
    assert agent._merge_triples(existing, None, set(), limit=10) == existing


def test_a_non_dict_item_is_skipped():
    agent = _agent()
    existing: list[dict[str, Any]] = []

    agent._merge_nodes(existing, ["stringa", {"text": "Scotta"}], set(), limit=10)

    assert existing == [{"text": "Scotta"}]


def test_triples_merge_on_their_three_parts():
    agent = _agent()
    seen: set[tuple[str, str, str]] = set()
    existing: list[dict[str, Any]] = []
    triple = {"subject": "a", "predicate": "USES", "object": "b"}

    agent._merge_triples(existing, [triple, dict(triple)], seen, limit=10)

    assert len(existing) == 1


def test_the_same_section_from_two_queries_appears_once():
    merged = KGRAGAgent._merge_context_sections(
        ["Matched nodes:\nScotta", "  Matched nodes:\nScotta  ", "Subgraph:\nx"]
    )

    assert merged.count("Matched nodes") == 1
    assert "Subgraph" in merged


def test_an_empty_section_contributes_nothing():
    assert KGRAGAgent._merge_context_sections(["", "   ", "reale"]) == "reale"


# --- the language of the messages the agent writes itself ------------------


class _Retrieving:
    """A retriever that exists, so the run is not the LLM-only baseline."""

    kg_store = None
    text_pipeline = None


def _zero_evidence_state(question: str, transcript: str = "") -> dict[str, Any]:
    """State of a turn whose retrieval found nothing."""
    return {
        "question": question,
        "transcript": transcript,
        "text_context": "",
        "kg_triples": [],
        "retrieved_nodes_count": 0,
        "retrieved_subgraph_count": 0,
        "retrieved_shortest_path_count": 0,
    }


def test_a_refusal_for_lack_of_evidence_follows_the_conversation(caplog):
    # The generated answer picks its language with the transcript behind it,
    # and the fixed strings must too: a continuation carries no marker of its
    # own, so judged on the question alone an Italian conversation that runs
    # out of evidence would be told so in English.
    agent = _agent(_Retrieving(), None, include_nodes=True)
    italian = "Utente: Cosa contiene la scotta?\nAssistente: E' il residuo liquido."

    with caplog.at_level(logging.WARNING):
        out = agent._generate(_zero_evidence_state("Non ho capito niente", italian))

    assert "Il contesto disponibile non è sufficiente" in out["answer"]


def test_the_same_turn_in_an_english_conversation_is_told_in_english(caplog):
    agent = _agent(_Retrieving(), None, include_nodes=True)
    english = "User: What does rice husk contain?\nAssistant: It is the outer shell."

    with caplog.at_level(logging.WARNING):
        out = agent._generate(_zero_evidence_state("Non ho capito niente", english))

    assert "The provided context is insufficient" in out["answer"]


def test_a_question_that_states_its_own_language_is_not_overridden(caplog):
    agent = _agent(_Retrieving(), None, include_nodes=True)
    english = "User: What does rice husk contain?\nAssistant: It is the outer shell."

    with caplog.at_level(logging.WARNING):
        out = agent._generate(
            _zero_evidence_state("Cosa contiene la scotta prodotta dai caseifici?", english)
        )

    assert "Il contesto disponibile non è sufficiente" in out["answer"]


def test_with_no_conversation_behind_it_the_message_is_english(caplog):
    agent = _agent(_Retrieving(), None, include_nodes=True)

    with caplog.at_level(logging.WARNING):
        out = agent._generate(_zero_evidence_state("Non ho capito niente"))

    assert "The provided context is insufficient" in out["answer"]
