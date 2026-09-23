"""Unit tests for definitional retrieval and source diversification.

The features themselves matter less than the three ways they could make the
system worse, which is what most of these tests pin down:

* **No change when the flags are off.** With the defaults, retrieval and the
  answer prompt are exactly what they are without these features, down to the
  byte-identical answer prompt.
* **No fabricated quotes.** Asking for a verbatim definition is also an
  invitation to invent one, and a made-up quote can carry a valid `[S2]`, so the
  quoted string is checked against the evidence itself.
* **No truncated enumerations.** The per-document cap diversifies sources, but
  an enumeration lives in one list on contiguous pages of one document: the cap
  must not be what cuts it short.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphrag.agent.core import KGRAGAgent
from graphrag.agent.evidence import EvidenceItem, verify_quotes
from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.llm.prompts import PromptLibrary
from graphrag.questions import (
    definition_score,
    definition_sentence,
    definitional_term,
    is_definitional,
    is_enumerative,
)

SEED_DEFINITION = (
    "SEeD (Systemic Event Design) è un approccio progettuale che applica la "
    "teoria dei sistemi alla progettazione di eventi."
)
SEED_MENTION = "Il progetto SEeD ha coinvolto dodici imprese piemontesi nel 2023."


@dataclass
class _Chunk:
    """A retrieved text chunk."""

    content: str
    source: str = ""
    chunk_id: str = ""
    score: float = 0.0


class _FakePipeline:
    """Text pipeline that returns a fixed pool and records how it was called."""

    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query, top_k=5, mmr_lambda=None, fetch_k=None):
        self.calls.append(
            {"query": query, "top_k": top_k, "mmr_lambda": mmr_lambda, "fetch_k": fetch_k}
        )
        return list(self._chunks[:top_k])


def _retriever(chunks: list[_Chunk], **overrides) -> tuple[KGRetriever, _FakePipeline]:
    """A `KGRetriever` over a fake text pipeline serving `chunks`."""
    config = AgentConfig(use_text_retriever=True, **overrides)
    pipeline = _FakePipeline(chunks)
    return KGRetriever(kg_store=None, config=config, text_pipeline=pipeline), pipeline


def _agent_with(**overrides) -> KGRAGAgent:
    """Agent with the given config and no retriever or LLM."""
    return KGRAGAgent(config=AgentConfig(**overrides), kg_retriever=None, llm=None)


def _text_evidence(*passages: str) -> list[EvidenceItem]:
    """Text evidence items S1..Sn, all from one document."""
    return [
        EvidenceItem(ref_id=f"S{index}", kind="text", text=passage, source_doc="doc.pdf")
        for index, passage in enumerate(passages, start=1)
    ]


# --- question typing ------------------------------------------------------


def test_definitional_questions_are_recognised_in_both_languages():
    for question, term in (
        ("Che cos'è SEeD?", "SEeD"),
        ("Cosa si intende per economia circolare?", "economia circolare"),
        ("Dammi la definizione delle 3C", "3C"),
        ("Come si definisce il capitale relazionale?", "capitale relazionale"),
        ("What is scotta and how does it differ from whey?", "scotta"),
        ("Define metabolization", "metabolization"),
    ):
        assert definitional_term(question) == term, question


def test_an_opener_without_a_concept_is_not_a_definition():
    """"Cosa vuol dire questo" points at the conversation, not at a term."""
    for question in (
        "Cosa vuol dire questo per le imprese?",
        "Che cosa succede se il compost non è maturo?",
        "Quali sono le 5 filiere della regione Piemonte?",
        "Mi indichi le strategie nel settore vino individuate dalla ricerca?",
    ):
        assert is_definitional(question) is False, question


def test_enumerations_are_recognised():
    for question in (
        "Quali sono le 5 filiere della regione Piemonte?",
        "Elenca i principi dell'economia circolare",
        "Quante imprese hanno partecipato?",
        "Which are the three pillars of the model?",
        "List the main barriers",
    ):
        assert is_enumerative(question) is True, question


def test_a_definition_is_not_an_enumeration():
    assert is_enumerative("Che cos'è SEeD?") is False


# --- definitional scoring -------------------------------------------------


def test_an_acronym_expansion_outscores_a_mention():
    assert definition_score(SEED_DEFINITION, "SEeD") > definition_score(
        SEED_MENTION, "SEeD"
    )


def test_the_expansion_counts_in_both_directions():
    reversed_form = "Systemic Event Design (SEeD) was launched in 2019."
    assert definition_score(reversed_form, "SEeD") >= 3.0


def test_definitional_wording_is_anchored_to_the_term():
    """"è un" matches half of Italian prose; it must count only next to the term."""
    about_something_else = (
        "La filiera del vino è un settore maturo, e il progetto SEeD lo conferma."
    )
    assert definition_score(about_something_else, "SEeD") == 0.5


def test_a_missing_term_scores_zero():
    assert definition_score("Testo che non parla dell'argomento.", "SEeD") == 0.0


# --- retrieval: definitional boost ---------------------------------------


def test_the_defining_chunk_is_promoted_for_a_definitional_question():
    chunks = [_Chunk(SEED_MENTION, "a.pdf#page=9"), _Chunk(SEED_DEFINITION, "a.pdf#page=3")]
    retriever, _ = _retriever(chunks, prefer_verbatim_definitions=True)

    result = retriever._retrieve_text_chunks("Che cos'è SEeD?")

    assert result[0].content == SEED_DEFINITION


def test_retrieval_order_is_untouched_when_nothing_defines_the_term():
    chunks = [_Chunk(SEED_MENTION, "a.pdf#page=9"), _Chunk("Altro testo.", "b.pdf#page=1")]
    retriever, _ = _retriever(chunks, prefer_verbatim_definitions=True)

    result = retriever._retrieve_text_chunks("Che cos'è SEeD?")

    assert [chunk.content for chunk in result] == [chunk.content for chunk in chunks]


def test_a_non_definitional_question_never_reorders():
    chunks = [_Chunk(SEED_MENTION, "a.pdf#page=9"), _Chunk(SEED_DEFINITION, "a.pdf#page=3")]
    retriever, _ = _retriever(chunks, prefer_verbatim_definitions=True)

    result = retriever._retrieve_text_chunks("Quali strategie adotta SEeD?")

    assert result[0].content == SEED_MENTION


# --- retrieval: source diversification -----------------------------------


def test_the_cap_limits_how_much_one_document_contributes():
    chunks = [
        _Chunk("uno", "a.pdf#page=1"),
        _Chunk("due", "a.pdf#page=2"),
        _Chunk("tre", "a.pdf#page=3"),
        _Chunk("quattro", "b.pdf#page=1"),
    ]
    retriever, _ = _retriever(
        chunks, text_retriever_top_k=3, text_retriever_max_per_doc=2
    )

    sources = [chunk.source for chunk in retriever._retrieve_text_chunks("domanda")]

    assert sources == ["a.pdf#page=1", "a.pdf#page=2", "b.pdf#page=1"]


def test_an_enumeration_gets_twice_the_per_document_budget():
    """The 5-filiere case: one list, contiguous pages, one document."""
    chunks = [_Chunk(f"filiera {index}", f"a.pdf#page={index}") for index in range(1, 5)]
    retriever, _ = _retriever(
        chunks, text_retriever_top_k=4, text_retriever_max_per_doc=2
    )

    result = retriever._retrieve_text_chunks("Quali sono le 5 filiere del Piemonte?")

    assert [chunk.content for chunk in result] == [
        "filiera 1",
        "filiera 2",
        "filiera 3",
        "filiera 4",
    ]


def test_the_cap_demotes_instead_of_dropping_when_the_corpus_has_nothing_else():
    """A truncated context is worse than a single-source one."""
    chunks = [_Chunk(f"p{index}", f"a.pdf#page={index}") for index in range(1, 4)]
    retriever, _ = _retriever(
        chunks, text_retriever_top_k=3, text_retriever_max_per_doc=1
    )

    result = retriever._retrieve_text_chunks("domanda")

    assert len(result) == 3


def test_pages_of_the_same_document_share_one_cap():
    assert KGRetriever._document_key("a.pdf#page=3#chunk=2") == "a.pdf"
    assert KGRetriever._document_key("") == ""


def test_mmr_is_off_and_no_pool_is_fetched_by_default():
    """The default config issues a plain top-k query: no MMR, no wider pool."""
    retriever, pipeline = _retriever([_Chunk("x", "a.pdf")], text_retriever_top_k=5)

    retriever._retrieve_text_chunks("domanda")

    assert pipeline.calls == [
        {"query": "domanda", "top_k": 5, "mmr_lambda": None, "fetch_k": None}
    ]


def test_reordering_features_fetch_a_candidate_pool():
    retriever, pipeline = _retriever(
        [_Chunk("x", "a.pdf")],
        text_retriever_top_k=5,
        text_retriever_max_per_doc=2,
        text_retriever_mmr=True,
    )

    retriever._retrieve_text_chunks("domanda")

    assert pipeline.calls[0]["top_k"] == 20
    assert pipeline.calls[0]["mmr_lambda"] == 0.7


# --- the quote gate -------------------------------------------------------


def test_a_faithful_quote_survives_reflowing():
    """PDF text arrives hyphenated and re-wrapped; that is not a fabrication."""
    evidence = _text_evidence("L'economia circo-\nlare per il cibo è un modello\nsistemico.")
    answer = "Il documento parla di «economia circolare per il cibo è un modello sistemico» [S1]."

    report = verify_quotes(answer=answer, evidence=evidence)

    assert report.unverified_quotes == []
    assert report.answer == answer


def test_an_invented_quote_loses_its_guillemets():
    evidence = _text_evidence(SEED_MENTION)
    answer = "SEeD è definito come «il primo modello sistemico europeo di eventi» [S1]."

    report = verify_quotes(answer=answer, evidence=evidence)

    assert report.unverified_quotes == ["il primo modello sistemico europeo di eventi"]
    assert "«" not in report.answer
    # The sentence stays: what is removed is the claim that these are the
    # source's words, not the content.
    assert "il primo modello sistemico europeo di eventi" in report.answer
    assert report.unverified_rate == 1.0


def test_an_elided_quote_is_verified_fragment_by_fragment():
    evidence = _text_evidence(
        "La metabolizzazione trasforma gli scarti in risorse attraverso "
        "processi biologici controllati che restituiscono nutrienti al suolo."
    )
    answer = "«La metabolizzazione trasforma gli scarti in risorse [...] nutrienti al suolo» [S1]."

    report = verify_quotes(answer=answer, evidence=evidence)

    assert report.unverified_quotes == []


def test_a_single_quoted_term_is_not_a_passage_to_verify():
    """«scotta» is emphasis, not a claim about someone's wording."""
    report = verify_quotes(answer="Il sottoprodotto «scotta» è citato.", evidence=_text_evidence("x"))

    assert report.total_quotes == 0
    assert report.answer == "Il sottoprodotto «scotta» è citato."


def test_the_quote_gate_ignores_ordinary_quotation_marks():
    """Straight and curly quotes carry titles and emphasis, not verbatim claims."""
    answer = 'Il rapporto "Circular Food" non definisce il termine in questo modo.'

    report = verify_quotes(answer=answer, evidence=_text_evidence("x"))

    assert report.answer == answer
    assert report.total_quotes == 0


def test_a_translated_quote_is_not_a_quote():
    """The corpus is bilingual: translating a passage stops it being verbatim.

    This is why the definitional prompt tells the model to quote in the source's
    language and translate outside the guillemets: terms such as SEeD, CEFF and
    metabolizzazione are defined in English documents.
    """
    evidence = _text_evidence(
        "SEeD, an acronym for Systemic Event Design, a systemic and circular "
        "sustainability project developed for application at Slow Food's events."
    )
    answer = "«SEeD, un acronimo per Systemic Event Design, è un progetto di sostenibilità sistemica» [S1]."

    report = verify_quotes(answer=answer, evidence=evidence)

    assert len(report.unverified_quotes) == 1


def test_a_verbatim_quote_in_the_source_language_passes():
    evidence = _text_evidence(
        "SEeD, an acronym for Systemic Event Design, a systemic and circular "
        "sustainability project developed for application at Slow Food's events."
    )
    answer = (
        "Il documento definisce SEeD come «an acronym for Systemic Event Design, "
        "a systemic and circular sustainability project» [S1], cioè un progetto "
        "di sostenibilità sistemica e circolare."
    )

    report = verify_quotes(answer=answer, evidence=evidence)

    assert report.unverified_quotes == []


def test_an_english_quotation_does_not_flip_the_answer_language():
    """Quoting an English source must not trigger the wrong-language retry."""
    answer = (
        "Il documento definisce il progetto come «a systemic and circular "
        "sustainability project developed for application at Slow Food's "
        "principal cultural events», cioè un progetto che applica la "
        "sostenibilità sistemica agli eventi culturali della rete."
    )

    assert LLMManager._detect_text_language(answer) == "it"


def test_a_quote_is_matched_against_passages_not_against_graph_facts():
    """Triples are a paraphrase of the source; quoting them is not quoting it."""
    evidence = [
        EvidenceItem(
            ref_id="T1", kind="triple", text="SEeD IMPLEMENTS systemic sustainability"
        )
    ]
    answer = "«SEeD IMPLEMENTS systemic sustainability» [T1]."

    report = verify_quotes(answer=answer, evidence=evidence)

    assert len(report.unverified_quotes) == 1


# --- the verbatim definition, extracted instead of asked for --------------


def test_the_defining_sentence_is_picked_out_of_the_chunk():
    chunk = (
        "Terra Madre Salone del Gusto ha coinvolto 3000 persone. "
        + SEED_DEFINITION
        + " Il progetto prosegue nel 2024."
    )

    assert definition_sentence(chunk, "SEeD") == SEED_DEFINITION


def test_a_chunk_that_only_mentions_the_term_yields_no_sentence():
    assert definition_sentence(SEED_MENTION, "SEeD") == ""


def test_a_markdown_heading_is_not_a_sentence_to_quote():
    """Headings carry no terminal punctuation and glue themselves to the
    paragraph below, so a naive split quotes one, bold markers included.
    """
    chunk = "## Introduction: The Systemic Event Design Project (SEeD)**\n\nFood matters."

    assert definition_sentence(chunk, "SEeD") == ""


def test_a_fragment_left_by_a_chunk_boundary_is_not_quoted():
    """Chunking cuts mid-word, so a chunk's first "sentence" can start there."""
    chunk = "pati dal Systemic Food Design Lab è nata la Circular Economy for Food (CEFF)."

    assert definition_sentence(chunk, "CEFF") == ""


def test_a_figure_caption_is_not_a_definition():
    """It reads like one and defines nothing; the caption glues itself to the
    paragraph below because it has no terminal punctuation either."""
    chunk = (
        "Fig. 28 - Rappresentazione delle 3C della Circular Economy for Food "
        "(F.Fassio) Progettando i flussi di materia si ottiene un sistema chiuso."
    )

    assert definition_sentence(chunk, "3C") == ""


def test_a_narrative_preamble_is_elided():
    """The corpus states the definition of SEeD at the end of a long sentence."""
    chunk = (
        "More than 16 years of research, eight editions of Terra Madre Salone "
        "del Gusto and 3000 people involved are the facts and figures behind "
        "SEeD, an acronym for Systemic Event Design, a circular sustainability "
        "project developed for Slow Food's principal cultural events."
    )

    sentence = definition_sentence(chunk, "SEeD")

    assert sentence.startswith("[...] SEeD, an acronym for Systemic Event Design")
    # Still verbatim: what is kept has to survive the quote gate.
    assert verify_quotes(
        answer=f"«{sentence}» [S1]", evidence=_text_evidence(chunk)
    ).unverified_quotes == []


def test_the_definition_that_opens_with_the_term_wins():
    """A definition starts with what it defines; a narrative mentions it late."""
    chunk = (
        "More than 16 years of research and eight editions of Terra Madre are "
        "the facts and figures behind SEeD, an acronym for Systemic Event Design. "
        + SEED_DEFINITION
    )

    assert definition_sentence(chunk, "SEeD") == SEED_DEFINITION


def test_the_source_definition_opens_the_answer():
    """Extracted here, not asked of the model.

    Asked to quote, the model translates an English source into Italian, which
    is accurate prose and not a quotation.
    """
    agent = _agent_with(prefer_verbatim_definitions=True)
    evidence = _text_evidence(SEED_MENTION, SEED_DEFINITION)

    answer, ref = agent._prepend_source_definition(
        answer="SEeD è un progetto sviluppato a Pollenzo.",
        question="Che cos'è SEeD?",
        evidence=evidence,
        language="it",
    )

    assert answer.startswith(f"**Dalla fonte:** «{SEED_DEFINITION}» [S2]")
    assert "SEeD è un progetto sviluppato a Pollenzo." in answer
    # The caller adds it to the source list: a quotation is a citation.
    assert ref == "S2"


def test_the_source_definition_is_quoted_in_the_source_language():
    agent = _agent_with(prefer_verbatim_definitions=True)
    english = (
        "SEeD is an acronym for Systemic Event Design, a systemic and circular "
        "sustainability project developed for Slow Food's cultural events."
    )

    answer, _ = agent._prepend_source_definition(
        answer="SEeD è un progetto di sostenibilità sistemica.",
        question="Che cos'è SEeD?",
        evidence=_text_evidence(english),
        language="it",
    )

    assert f"«{english}»" in answer


def test_nothing_is_prepended_without_a_defining_passage():
    agent = _agent_with(prefer_verbatim_definitions=True)

    assert agent._prepend_source_definition(
        answer="Risposta.",
        question="Che cos'è SEeD?",
        evidence=_text_evidence(SEED_MENTION),
        language="it",
    ) == ("Risposta.", None)


def test_nothing_is_prepended_when_the_flag_is_off_or_the_question_is_not_definitional():
    evidence = _text_evidence(SEED_DEFINITION)

    off = _agent_with(prefer_verbatim_definitions=False)._prepend_source_definition(
        answer="Risposta.", question="Che cos'è SEeD?", evidence=evidence, language="it"
    )
    other = _agent_with(prefer_verbatim_definitions=True)._prepend_source_definition(
        answer="Risposta.",
        question="Quali strategie adotta SEeD?",
        evidence=evidence,
        language="it",
    )

    assert off == ("Risposta.", None)
    assert other == ("Risposta.", None)


def test_the_definition_is_not_quoted_twice():
    """The model gets there on its own often enough for this to matter."""
    agent = _agent_with(prefer_verbatim_definitions=True)
    answer = f"Secondo il documento, {SEED_DEFINITION} Il progetto nasce nel 2005."

    assert agent._prepend_source_definition(
        answer=answer,
        question="Che cos'è SEeD?",
        evidence=_text_evidence(SEED_DEFINITION),
        language="it",
    ) == (answer, None)


# --- prompt: no regression when the flag is off ---------------------------


def test_the_answer_prompt_is_unchanged_for_a_non_definitional_question():
    config = AgentConfig()
    baseline = PromptLibrary.answer_prompt(config).messages[1].prompt.template
    with_flag = (
        PromptLibrary.answer_prompt(config, definitional=False).messages[1].prompt.template
    )

    assert with_flag == baseline


def test_the_definitional_prompt_asks_to_quote_before_paraphrasing():
    template = (
        PromptLibrary.answer_prompt(AgentConfig(), definitional=True)
        .messages[1]
        .prompt.template
    )

    assert "«guillemets»" in template
    assert "copy the source word for word" in template
    # The bilingual corpus clause: quote in the source's language, translate
    # outside the guillemets, or the quote can never be verbatim.
    assert "copy the\npassage in the language the source wrote it in".replace(
        "\n", " "
    ) in template.replace("\n", " ")
    # The escape hatch is what keeps the instruction from becoming a licence to
    # invent a definition when the corpus has none.
    assert "when no passage defines the term, use no guillemets at all" in template
    assert "Never invent a definition to quote" in template
    assert "never as a replacement" in template


def test_definitional_defaults_are_off():
    config = AgentConfig()

    assert config.prefer_verbatim_definitions is False
    assert config.text_retriever_mmr is False
    assert config.text_retriever_max_per_doc == 0
    assert config.text_retriever_top_k == 5
