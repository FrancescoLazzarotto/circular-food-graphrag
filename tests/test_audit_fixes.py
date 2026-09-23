"""Regression tests for the defects catalogued in docs/code_audit_2026-08-15.md.

Each test names the audit section it locks down. They are grouped here rather
than spread across the suite because they share one property: each defect is
silent, so without these tests the suite gives no signal at all.
"""

from __future__ import annotations

from graphrag.agent.compression import ContextCompressor
from graphrag.agent.core import KGRAGAgent, _term_matches
from graphrag.agent.evidence import (
    build_evidence_index,
    refs_present_in,
    render_cited_context,
    verify_citations,
)
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.llm.manager import LLMManager
from graphrag.llm.refusal import is_insufficient, looks_like_refusal
from graphrag.text_rag.manager import TextRAGManager


def _triple(subject: str, predicate: str, obj: str, **kwargs: object) -> dict:
    """A triple row."""
    return {"subject": subject, "predicate": predicate, "object": obj, **kwargs}


# --- §1.1 the context must not echo the question --------------------------


def test_cited_context_omits_the_question():
    context = render_cited_context(
        evidence=build_evidence_index(triples=[_triple("A", "REL", "B")]),
    )
    assert "Query:" not in context


def test_context_is_empty_when_nothing_was_retrieved():
    assert render_cited_context(evidence=[]) == ""


# --- §1.2 salient terms are bilingual and boundary-matched ----------------


def test_salient_terms_survive_an_english_question_without_acronyms():
    terms = KGRAGAgent._extract_salient_terms_from_text(
        "What are the objectives of the circular economy for food?"
    )
    # Function words carry no signal and must not become salient terms.
    assert "the" not in terms
    assert "are" not in terms
    assert "objectives" in terms
    assert "circular" in terms


def test_salient_terms_keep_acronyms_first():
    terms = KGRAGAgent._extract_salient_terms_from_text(
        "How does CEFF relate to the SDGs?"
    )
    assert terms[:2] == ["ceff", "sdgs"]


def test_term_matching_respects_word_boundaries():
    assert _term_matches("rice", "rice straw is a residue")
    # Substring matching would accept every one of these.
    assert not _term_matches("rice", "the market price per tonne")
    assert not _term_matches("ceff", "ceffpolicy in sustainability")


# --- §1.3 compression must not leave half-rendered evidence ---------------


def test_compression_cuts_on_block_boundaries():
    blocks = [f"[T{i}] (subject{i}, REL, object{i}) <doc.pdf | p. {i}>" for i in range(60)]
    text = "\n\n".join(blocks)
    compressed = ContextCompressor(max_tokens=60, ratio=0.25).compress(text)

    assert "[... context trimmed ...]" in compressed
    head, tail = compressed.split("\n\n[... context trimmed ...]\n\n")
    # Every surviving line is a whole block, not a truncated one.
    for line in (*head.split("\n\n"), *tail.split("\n\n")):
        assert line.startswith("[T") and line.endswith(">"), line


def test_compression_keeps_a_single_oversized_block():
    text = "x" * 4000
    compressed = ContextCompressor(max_tokens=100, ratio=0.25).compress(text)
    # No boundary exists; the compressor must still return content.
    assert compressed.replace("[... context trimmed ...]", "").strip()


# --- §1.3 the citation gate judges only what the model saw ----------------


def test_citation_to_a_compressed_away_block_is_phantom():
    evidence = build_evidence_index(
        triples=[_triple(f"S{i}", "REL", f"O{i}") for i in range(5)]
    )
    report = verify_citations(
        answer="The answer draws on [T5].",
        evidence=evidence,
        visible_refs={"T1", "T2"},
    )
    assert report.phantom_refs == ["T5"]


def test_citation_gate_without_visible_refs_uses_the_full_index():
    evidence = build_evidence_index(triples=[_triple("A", "REL", "B")])
    report = verify_citations(answer="Grounded in [T1].", evidence=evidence)
    assert report.cited_refs == ["T1"]
    assert not report.phantom_refs


def test_refs_present_in_reads_block_headers():
    assert refs_present_in("[S1] passage\n\n[T12] (A, REL, B)") == {"S1", "T12"}


# --- §1.4 / §1.5 refusal detection ----------------------------------------


def test_domain_prose_is_not_a_refusal():
    assert not looks_like_refusal(
        "Anaerobic digestion is not feasible below 20 t/day, so the plant "
        "was sized for composting instead."
    )
    assert not looks_like_refusal(
        "The indicator set was challenging to construct because the sources "
        "disagree on system boundaries."
    )


def test_a_genuine_refusal_is_still_detected():
    assert looks_like_refusal("The context is insufficient to answer.")
    assert looks_like_refusal("")


def test_trailing_caveat_on_a_long_answer_is_not_an_abstention():
    answer = (
        "The three C's of the Circular Economy for Food are Capital, Cycles and "
        "Co-evolution. " + "The framework is described at length in the source. " * 12
        + "The provided context does not contain the publication year."
    )
    assert not is_insufficient(answer)


def test_a_short_no_evidence_answer_is_an_abstention():
    assert is_insufficient("The provided context does not contain that figure.")


def test_a_leading_abstention_in_a_long_answer_still_counts():
    answer = (
        "The provided context does not contain information about the market "
        "price. " + "It discusses grape pomace only in qualitative terms. " * 12
    )
    assert is_insufficient(answer)


# --- §1.12 language detection is symmetric --------------------------------


def test_an_accented_noun_does_not_make_an_english_question_italian():
    assert (
        LLMManager._detect_query_language(
            "What is the role of café waste in the circular economy?"
        )
        == "en"
    )


def test_short_italian_questions_still_detect_as_italian():
    assert LLMManager._detect_query_language("Cos'è la coevoluzione?") == "it"
    assert LLMManager._detect_query_language("Definizione di capitale relazionale?") == "it"


# --- §2.1 one bad Lucene query must not disable the index -----------------


def _manager_without_db() -> KnowledgeGraphManager:
    """A `KnowledgeGraphManager` with fulltext state set and no driver."""
    manager = object.__new__(KnowledgeGraphManager)
    manager.fulltext_index = "node_search"
    manager._fulltext_available = None
    manager._fulltext_retry_at = 0.0
    manager._fulltext_failures = 0
    return manager


_MISSING_INDEX_ERROR = Exception(
    "Failed to invoke procedure `db.index.fulltext.queryNodes`: Caused by: "
    "java.lang.IllegalArgumentException: There is no such fulltext schema "
    "index: node_search"
)


def test_lucene_parse_error_leaves_fulltext_enabled():
    manager = _manager_without_db()
    exc = Exception(
        "Failed to invoke procedure `db.index.fulltext.queryNodes`: Caused by: "
        "org.apache.lucene.queryparser.classic.ParseException: Cannot parse "
        "'name:(': Encountered \"<EOF>\""
    )
    assert manager._handle_fulltext_error(exc) is False
    assert manager._fulltext_available is None


def test_missing_index_disables_fulltext():
    manager = _manager_without_db()
    exc = Exception(
        "Failed to invoke procedure `db.index.fulltext.queryNodes`: Caused by: "
        "java.lang.IllegalArgumentException: There is no such fulltext schema "
        "index: node_search"
    )
    assert manager._handle_fulltext_error(exc) is True
    assert manager._fulltext_available is False


def test_disabled_fulltext_is_retried_once_its_backoff_expires(monkeypatch):
    """A transient failure must not downgrade retrieval for the whole process."""
    manager = _manager_without_db()
    now = 1_000.0
    monkeypatch.setattr(
        "graphrag.kg.manager.time.monotonic", lambda: now, raising=False
    )
    assert manager._handle_fulltext_error(_MISSING_INDEX_ERROR) is True
    assert manager._fulltext_ready() is False

    now += manager._FULLTEXT_RETRY_BACKOFF_SEC[0] + 1
    assert manager._fulltext_ready() is True
    assert manager._fulltext_available is None


def test_repeated_failures_back_off_instead_of_probing_at_a_fixed_rate():
    """A genuinely missing index must not cost a failed query every 30 s."""
    manager = _manager_without_db()
    delays = []
    for _ in range(len(manager._FULLTEXT_RETRY_BACKOFF_SEC) + 2):
        manager._handle_fulltext_error(_MISSING_INDEX_ERROR)
        delays.append(manager._fulltext_retry_delay_sec())
    assert delays == sorted(delays)
    assert delays[0] == manager._FULLTEXT_RETRY_BACKOFF_SEC[0]
    assert delays[-1] == manager._FULLTEXT_RETRY_BACKOFF_SEC[-1]


# --- §5.7 / §5.9 the lexical text channel ---------------------------------


def _manager_with(*documents: str) -> TextRAGManager:
    """A `TextRAGManager` indexing the given documents."""
    manager = TextRAGManager()
    manager.add_documents(documents)
    return manager


def test_bm25_prefers_the_chunk_that_is_about_the_query():
    manager = _manager_with(
        "grape pomace is a by-product of winemaking rich in polyphenols",
        "the report lists many by-products of many chains " * 12,
        "rice husk is a by-product of rice milling",
    )
    hits = manager.retrieve_with_scores("grape pomace polyphenols", top_k=3)
    assert hits
    assert "grape pomace" in hits[0][0].content


def test_bm25_does_not_reward_padding():
    # A long chunk that merely repeats a query term must not outrank a short
    # chunk that answers it: this is what the length normalisation is for.
    focused = "capital is one of the three C's of the CEFF framework"
    padded = "capital " * 200
    manager = _manager_with(focused, padded)
    hits = manager.retrieve_with_scores("three C's capital CEFF", top_k=2)
    assert hits[0][0].content == focused


def test_mmr_drops_a_near_duplicate_chunk():
    duplicate = "grape pomace is a winemaking by-product rich in polyphenols"
    manager = _manager_with(
        duplicate,
        duplicate + " indeed",
        # A real alternative: it shares one query term, so it is a candidate,
        # but says something else. A zero-score chunk is never a candidate and
        # MMR cannot promote it.
        "polyphenols require ethanol extraction at controlled temperature",
    )
    plain = manager.retrieve_with_scores("grape pomace polyphenols", top_k=2)
    diverse = manager.retrieve_with_scores(
        "grape pomace polyphenols", top_k=2, mmr_lambda=0.3
    )
    # Pure relevance keeps both near-identical chunks; MMR replaces the second.
    assert plain[1][0].content.startswith(duplicate)
    assert not diverse[1][0].content.startswith(duplicate)


# --- §1.10 merge caps ------------------------------------------------------


def test_merge_never_exceeds_the_limit():
    agent = object.__new__(KGRAGAgent)
    existing: list[dict] = []
    seen: set = set()
    incoming = [_triple(f"S{i}", "REL", f"O{i}") for i in range(10)]
    KGRAGAgent._merge_triples(agent, existing, incoming, seen, limit=4)
    assert len(existing) == 4
    # A second call on a list already at the cap must add nothing.
    KGRAGAgent._merge_triples(agent, existing, incoming, seen, limit=4)
    assert len(existing) == 4


# --- §4.4 partial_match ----------------------------------------------------


def test_partial_match_needs_whole_tokens_and_ignores_duplicates():
    from evalkit.metrics.text import partial_match

    # "co" must not match inside "cost"; the repeated "the" must not inflate.
    assert partial_match("the cost of the plant", "co") == 0.0
    assert partial_match("capital cyclicality co-evolution", "capital the the") == 0.5


# --- §1.6 domain gate parsing ---------------------------------------------


class _FakeOutput:
    """Chat-model output carrying only `content`."""

    def __init__(self, content: str) -> None:
        self.content = content


def _gate_verdict(text: str) -> bool:
    """Run only the verdict parsing of `classify_in_domain` on `text`."""
    manager = object.__new__(LLMManager)
    manager.load_llm = lambda: object()  # type: ignore[method-assign]
    manager._invoke_with_retry = lambda _model, _prompt: _FakeOutput(text)  # type: ignore[method-assign]

    class _Config:
        domain_scope = "circular economy for food"

    return LLMManager.classify_in_domain(manager, "una domanda", _Config())


def test_reasoning_preamble_does_not_flip_an_out_of_domain_verdict():
    assert _gate_verdict("OUT") is False
    # Prose around the verdict must not read as "in domain".
    assert _gate_verdict("The question is OUT of domain") is False
    assert _gate_verdict("<think>Hmm, cars are unrelated.</think>OUT") is False


def test_in_domain_verdicts_stay_in_domain():
    assert _gate_verdict("IN") is True
    assert _gate_verdict("<think>This is about food waste.</think>IN") is True
    # A restatement of the options followed by the conclusion.
    assert _gate_verdict("Either OUT or IN — the answer is IN") is True


# --- §5.5 the latent prompt crash -----------------------------------------


def test_multihop_steer_prompt_can_be_built():
    from graphrag.llm.prompts import PromptLibrary

    prompt = PromptLibrary.multihop_steer_prompt()
    rendered = prompt.invoke({"hop_history": "A -> B", "question": "why?"})
    assert "enough" in str(rendered)


# --- §1.11 the retrieval cache must not hand out its own objects ----------


def test_retrieval_cache_returns_a_copy():
    from graphrag.agent.cache import LRUCache

    cache = LRUCache()
    cache.put("q", "HYBRID", {"triples": [{"subject": "A"}]})
    first = cache.get("q", "HYBRID")
    first["triples"][0]["subject"] = "MUTATED"
    second = cache.get("q", "HYBRID")
    assert second["triples"][0]["subject"] == "A"


# --- §1.8 definitional detection ------------------------------------------


def test_a_counted_list_is_not_a_definition():
    from graphrag import questions

    assert not questions.is_definitional(
        "What are the four implementation cycles of metabolisation?"
    )


def test_a_second_clause_opener_does_not_define_a_possessive():
    from graphrag import questions

    assert not questions.is_definitional(
        "What valuable compounds can be extracted from grape pomace, and what "
        "are their applications?"
    )


def test_a_real_definition_is_still_detected():
    from graphrag import questions

    assert questions.is_definitional("What is scotta and how does it differ from whey?")


# --- §3.9 chunk windowing must make progress ------------------------------


def test_windowing_does_not_degenerate_on_short_paragraphs():
    from kg_pipeline.stages.chunking import _window_paragraphs

    class _P:
        def __init__(self, text: str, page: int = 1) -> None:
            self.text = text
            self.page_number = page

    # 40 short paragraphs: their total is far below the overlap budget, which is
    # what can make the window advance one paragraph at a time.
    paragraphs = [_P("breve paragrafo numero %d." % i) for i in range(40)]
    windows = _window_paragraphs(paragraphs, max_tokens=40, overlap_tokens=512)
    # Advancing one paragraph at a time gives ~len(paragraphs) windows; linear
    # windowing gives far fewer.
    assert len(windows) < len(paragraphs) // 2
    assert windows
