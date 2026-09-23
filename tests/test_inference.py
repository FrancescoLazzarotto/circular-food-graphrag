"""Unit tests for the GraphRAG inference path (retrieval + generation glue).

Covered here:
- context compression keeps a realistic token budget (no over-trimming)
- shared refusal detection
- agent fallback only fires on genuine refusal / ungrounded sparse answers
- triple ranker weights reflect the signals actually available
- the experiment runner emits no always-constant metric columns
- a single shared strategy-preset applier
- runs emit query_id so the evaluator can join to the gold by id
- decomposition parsing, ranker weight validation and full-text query
  building
"""

from __future__ import annotations

import argparse
import json

import pytest

from graphrag.agent.compression import ContextCompressor
from graphrag.config import AgentConfig
from graphrag.experiments.runner import ExperimentResult, ExperimentRunner, Question
from graphrag.kg.retriever import KGRetriever, _REQUEST_VERBS_IT
from graphrag.llm.refusal import looks_like_refusal
from graphrag.strategies import STRATEGY_PRESETS, apply_strategy


# --------------------------------------------------------------------------- #
# ContextCompressor
# --------------------------------------------------------------------------- #
def test_compressor_keeps_short_context_untrimmed():
    comp = ContextCompressor(max_tokens=1000)
    text = "x" * 3000  # ~750 estimated tokens (< 1000) -> untouched
    assert comp.compress(text) == text


def test_compressor_trims_only_when_over_budget_and_keeps_more_than_old_ratio():
    comp = ContextCompressor(max_tokens=1000)
    text = "x" * 8000  # ~2000 estimated tokens -> trimmed
    out = comp.compress(text)
    assert "[... context trimmed ...]" in out
    # The budget is ~4000 chars; an inverted tokens-per-char ratio would keep
    # only ~1333.
    assert len(out) > 3000


def test_compressor_ratio_not_inverted_regression():
    # The ratio is tokens per char, not chars per token.
    assert ContextCompressor(max_tokens=1000).ratio <= 0.3


# --------------------------------------------------------------------------- #
# refusal detection
# --------------------------------------------------------------------------- #
def test_refusal_empty_is_refusal():
    assert looks_like_refusal("") is True
    assert looks_like_refusal("   ") is True


def test_refusal_phrase_detected_en_and_it():
    assert looks_like_refusal("The context is insufficient to answer.") is True
    assert looks_like_refusal("Mi dispiace, contesto insufficiente.") is True


def test_refusal_substantive_answer_not_flagged():
    answer = "Regulation 178/2002 establishes the European Food Safety Authority."
    assert looks_like_refusal(answer) is False


def test_refusal_common_words_not_flagged():
    # Common words like "context"/"information"/"analysis" are not a
    # refusal; only the refusal phrases are.
    answer = (
        "The available information and analysis in this context show that the "
        "policy affects food security indicators."
    )
    assert looks_like_refusal(answer) is False


# --------------------------------------------------------------------------- #
# shared strategy presets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", STRATEGY_PRESETS)
def test_apply_strategy_known_labels_return_config(label):
    assert isinstance(apply_strategy(AgentConfig(), label), AgentConfig)


def test_apply_strategy_text_only_disables_kg_channels():
    cfg = apply_strategy(AgentConfig(), "text_only")
    assert cfg.include_nodes is False
    assert cfg.include_triples is False
    assert cfg.include_subgraph is False
    assert cfg.include_shortest_path is False


def test_apply_strategy_subgraph_2hop_sets_min_two_hops():
    assert apply_strategy(AgentConfig(hops=1), "subgraph_2hop").hops >= 2


def test_apply_strategy_does_not_mutate_base():
    base = AgentConfig()
    apply_strategy(base, "text_only")
    assert base.include_nodes is True  # deep-copied, base untouched


def test_apply_strategy_unknown_raises():
    with pytest.raises(ValueError):
        apply_strategy(AgentConfig(), "does_not_exist")


# --------------------------------------------------------------------------- #
# experiment runner: no dead metric columns
# --------------------------------------------------------------------------- #
class _StubAgent:
    """Agent returning a fixed minimal result."""

    def invoke(self, question: str) -> dict:
        return {"answer": "ok", "latency_ms": 1.0, "kg_triples": [], "run_id": "r1"}


def test_experiment_result_has_no_dead_fields():
    fields = set(ExperimentResult.__dataclass_fields__)
    assert "confidence" not in fields
    assert "reflection_passed" not in fields


def test_runner_csv_has_no_dead_metric_columns(tmp_path):
    runner = ExperimentRunner(questions=["q1"])
    runner.run_agent(agent=_StubAgent(), label="default")
    csv_path = tmp_path / "results.csv"
    runner.export_csv(str(csv_path))
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "confidence" not in header
    assert "reflection_passed" not in header


def test_runner_summary_stats_has_no_dead_metrics():
    runner = ExperimentRunner(questions=["q1"])
    runner.run_agent(agent=_StubAgent(), label="default")
    stats = runner.summary_stats()["default"]
    assert "avg_confidence" not in stats
    assert "reflection_pass_rate" not in stats


# --------------------------------------------------------------------------- #
# query_id propagation into run artifacts
# --------------------------------------------------------------------------- #
def test_runner_accepts_plain_strings_for_backwards_compat():
    """The legacy list[str] call must keep working, with an empty query_id."""
    runner = ExperimentRunner(questions=["q1", "q2"])
    batch = runner.run_agent(agent=_StubAgent(), label="default")

    assert [q.text for q in runner.questions] == ["q1", "q2"]
    assert [r.question for r in batch] == ["q1", "q2"]
    assert all(r.query_id == "" for r in batch)


def test_runner_emits_query_id_in_jsonl(tmp_path):
    runner = ExperimentRunner(
        questions=[Question(text="What is whey?", query_id="Q01")]
    )
    runner.run_agent(agent=_StubAgent(), label="default")

    path = tmp_path / "results.jsonl"
    runner.export_jsonl(str(path))
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert row["query_id"] == "Q01"
    assert row["question"] == "What is whey?"


def test_runner_emits_query_id_in_csv(tmp_path):
    runner = ExperimentRunner(questions=[Question(text="q1", query_id="Q07")])
    runner.run_agent(agent=_StubAgent(), label="default")

    path = tmp_path / "results.csv"
    runner.export_csv(str(path))
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0].split(",")[0] == "query_id"
    assert lines[1].split(",")[0] == "Q07"


def _args(questions_file: str = "", question: str = "q?") -> argparse.Namespace:
    """A namespace with the question arguments."""
    return argparse.Namespace(questions_file=questions_file, question=question)


def test_load_questions_plain_text_is_backwards_compatible(tmp_path):
    from graphrag.cli import _load_questions

    path = tmp_path / "questions.txt"
    path.write_text("First question?\nSecond question?\n\n", encoding="utf-8")

    questions = _load_questions(_args(str(path)))

    assert [q.text for q in questions] == ["First question?", "Second question?"]
    assert all(q.query_id == "" for q in questions)


def test_load_questions_text_with_tab_ids(tmp_path):
    from graphrag.cli import _load_questions

    path = tmp_path / "questions.txt"
    path.write_text("Q01\tFirst question?\nQ02\tSecond question?\n", encoding="utf-8")

    questions = _load_questions(_args(str(path)))

    assert [(q.query_id, q.text) for q in questions] == [
        ("Q01", "First question?"),
        ("Q02", "Second question?"),
    ]


def test_load_questions_from_gold_json(tmp_path):
    """A gold file can be handed straight to --questions-file."""
    from graphrag.cli import _load_questions

    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "_meta": {"n_queries": 2},
                "queries": [
                    {"query_id": "Q01", "query": "What is whey?"},
                    {"query_id": "Q02", "query": "What is scotta?"},
                ],
            }
        ),
        encoding="utf-8",
    )

    questions = _load_questions(_args(str(path)))

    assert [(q.query_id, q.text) for q in questions] == [
        ("Q01", "What is whey?"),
        ("Q02", "What is scotta?"),
    ]


def test_load_questions_from_jsonl(tmp_path):
    from graphrag.cli import _load_questions

    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"query_id": "Q01", "query": "First?"}\n'
        '{"query_id": "Q02", "question": "Second?"}\n',
        encoding="utf-8",
    )

    questions = _load_questions(_args(str(path)))

    assert [(q.query_id, q.text) for q in questions] == [
        ("Q01", "First?"),
        ("Q02", "Second?"),
    ]


def test_load_questions_from_csv(tmp_path):
    from graphrag.cli import _load_questions

    path = tmp_path / "questions.csv"
    path.write_text("query_id,query\nQ01,First?\nQ02,Second?\n", encoding="utf-8")

    questions = _load_questions(_args(str(path)))

    assert [(q.query_id, q.text) for q in questions] == [
        ("Q01", "First?"),
        ("Q02", "Second?"),
    ]


def test_load_questions_rejects_duplicate_ids(tmp_path):
    from graphrag.cli import _load_questions

    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"query_id": "Q01", "query": "First?"}\n'
        '{"query_id": "Q01", "query": "Second?"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate query_id"):
        _load_questions(_args(str(path)))


def test_load_questions_empty_file_raises(tmp_path):
    from graphrag.cli import _load_questions

    path = tmp_path / "questions.txt"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        _load_questions(_args(str(path)))


def test_load_questions_missing_file_raises():
    from graphrag.cli import _load_questions

    with pytest.raises(FileNotFoundError):
        _load_questions(_args("/nonexistent/questions.txt"))


def test_load_questions_warns_when_no_ids(tmp_path, caplog):
    """A run that cannot be joined by id must say so before it burns GPU hours."""
    import logging

    from graphrag.cli import _load_questions

    path = tmp_path / "questions.txt"
    path.write_text("First question?\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="graphrag.cli"):
        _load_questions(_args(str(path)))

    assert "declares no query_id" in caplog.text


# --------------------------------------------------------------------------- #
# agent fallback gating + localization
# --------------------------------------------------------------------------- #
def _make_agent() -> "object":
    """Agent with the default config and no retriever or LLM."""
    from graphrag.agent.core import KGRAGAgent

    return KGRAGAgent(config=AgentConfig())


def test_agent_keeps_grounded_answer_under_sparse_context():
    agent = _make_agent()
    query = "What does Regulation establish about EFSA?"
    context = "Regulation 178/2002 established EFSA."
    answer = (
        "The context and information here, by analysis, show that Regulation "
        "178/2002 established EFSA as the food safety authority."
    )
    # Three or more meta words are not enough to replace an answer that
    # references salient terms.
    assert (
        agent._should_replace_with_fallback(
            answer=answer,
            query=query,
            context=context,
            triples=[],
            sparse_context=True,
        )
        is False
    )


def test_agent_replaces_genuine_refusal():
    agent = _make_agent()
    assert (
        agent._should_replace_with_fallback(
            answer="The context is insufficient to answer.",
            query="q",
            context="c",
            triples=[],
            sparse_context=True,
        )
        is True
    )


def test_agent_sparse_fallback_text_is_localized():
    from graphrag.agent.core import KGRAGAgent

    en = KGRAGAgent._build_sparse_fallback_answer(
        query="What is EFSA?", context="", triples=[], language="en"
    )
    assert "too sparse" in en
    assert "Limiti e fiducia" not in en

    it = KGRAGAgent._build_sparse_fallback_answer(
        query="Cos'e EFSA?", context="", triples=[], language="it"
    )
    assert "troppo scarno" in it


# --------------------------------------------------------------------------- #
# triple ranker
# --------------------------------------------------------------------------- #
def test_ranker_weights_reflect_available_signals():
    cfg = AgentConfig()
    assert cfg.ranker_weight_confidence == 0.0
    assert abs((cfg.ranker_weight_lexical + cfg.ranker_weight_mention) - 1.0) < 1e-9


def test_rank_triples_prefers_query_overlap():
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    triples = [
        {"subject": "Banana", "predicate": "GROWN_IN", "object": "Ecuador"},
        {"subject": "EFSA", "predicate": "ESTABLISHED", "object": "Authority"},
    ]
    ranked = retriever._rank_triples(triples, "What established EFSA?")
    assert ranked[0]["subject"] == "EFSA"


# --------------------------------------------------------------------------- #
# decomposition parsing, ranker weight validation
# --------------------------------------------------------------------------- #
class _StubDecomposeLLM:
    """LLM whose model answers every prompt with `text`."""

    def __init__(self, text: str) -> None:
        self._text = text

    def load_llm(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            invoke=lambda _payload: SimpleNamespace(content=self._text)
        )


def test_decompose_parses_json_array():
    agent = _make_agent()
    agent.config.enable_decomposition_step = True
    agent.llm = _StubDecomposeLLM('["What is X?", "What is Y?"]')
    out = agent._decompose({"question": "complex question?"})
    assert out["sub_questions"] == ["What is X?", "What is Y?"]


def test_decompose_fallback_strips_numbering_and_bullets():
    agent = _make_agent()
    agent.config.enable_decomposition_step = True
    agent.llm = _StubDecomposeLLM("1. What is X?\n2) What is Y?\n- What is Z?")
    out = agent._decompose({"question": "complex question?"})
    assert out["sub_questions"] == ["What is X?", "What is Y?", "What is Z?"]


def test_decompose_fallback_never_returns_empty():
    agent = _make_agent()
    agent.config.enable_decomposition_step = True
    agent.llm = _StubDecomposeLLM("   \n  ")
    out = agent._decompose({"question": "complex question?"})
    assert out["sub_questions"] == ["complex question?"]


def test_ranker_weight_sum_warning_on_misconfiguration(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        AgentConfig(ranker_weight_lexical=1.0, ranker_weight_mention=0.5)
    assert any("Ranker weights sum" in record.message for record in caplog.records)


def test_ranker_weight_no_warning_when_normalized(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        AgentConfig()
    assert not any("Ranker weights sum" in record.message for record in caplog.records)


# --------------------------------------------------------------------------- #
# Full-text retrieval: Lucene query building, indexed path, CONTAINS fallback
# --------------------------------------------------------------------------- #
def test_lucene_query_quotes_phrases_and_escapes_specials():
    from graphrag.kg.manager import KnowledgeGraphManager

    query = KnowledgeGraphManager._lucene_query(
        ["circular economy", "UNISG", 'waste:type', "", "  "]
    )
    assert query == '"circular economy" OR UNISG OR waste\\:type'


def test_lucene_query_empty_terms_yield_empty_query():
    from graphrag.kg.manager import KnowledgeGraphManager

    assert KnowledgeGraphManager._lucene_query(["", None, "  "]) == ""


class _StubKGStore:
    """KG store stub: indexed rows when available, CONTAINS rows otherwise."""

    def __init__(self, indexed_nodes):
        self.indexed_nodes = indexed_nodes
        self.contains_calls = 0

    def fulltext_search_nodes(self, terms, labels=None, limit=None, boosts=None):
        return self.indexed_nodes

    def fulltext_search_triples(
        self, terms, labels=None, relationship_types=None, limit=None
    ):
        return None

    def extract_nodes(self, text=None, labels=None, limit=None):
        self.contains_calls += 1
        return [{"node_id": f"contains-{text}", "labels": [], "properties": {}, "text": text}]


def test_collect_nodes_uses_single_indexed_query_and_dedups():
    node = {"node_id": "n1", "labels": ["Concept"], "properties": {}, "text": "X"}
    store = _StubKGStore(indexed_nodes=[node, dict(node), dict(node)])
    retriever = KGRetriever(kg_store=store, config=AgentConfig())

    result = retriever._collect_nodes(["term a", "term b"], limit=10)

    assert len(result) == 1
    assert store.contains_calls == 0


def test_collect_nodes_falls_back_to_contains_scan_when_index_missing():
    store = _StubKGStore(indexed_nodes=None)
    retriever = KGRetriever(kg_store=store, config=AgentConfig())

    result = retriever._collect_nodes(["term a", "term b"], limit=10)

    assert store.contains_calls == 2
    assert len(result) == 2


def test_search_terms_ignore_italian_elision_apostrophes():
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    terms = retriever._build_search_terms(
        query_text="Che cos'è l'economia circolare nel sistema alimentare?",
        configured_entity="",
    )
    assert "è l" not in terms
    assert "Che" not in terms
    lowered = [t.lower() for t in terms]
    assert "economia" in lowered
    assert "circolare" in lowered


def test_quoted_entities_still_extracted_from_real_quotes():
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    candidates = retriever._extract_entity_candidates(
        'What does "Materia Rinnovabile" say about \'Mimica Touch\'?'
    )
    assert "Materia Rinnovabile" in candidates
    assert "Mimica Touch" in candidates


def test_title_phrase_keeps_lowercase_connectors_whole():
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    candidates = retriever._extract_entity_candidates(
        "In the Via del Campo biogas case, what two input materials feed the plant?"
    )
    assert "Via del Campo" in candidates
    # The sentence-initial function words must be trimmed off the phrase edge
    # and the fragments must not survive as standalone candidates.
    assert "In the Via del Campo" not in candidates
    assert "Via" not in candidates
    assert "Campo" not in candidates


def test_content_keywords_survive_entity_candidates():
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    terms = retriever._build_search_terms(
        query_text=(
            "In the Via del Campo biogas case, what two input materials "
            "feed the plant and in what proportions?"
        ),
        configured_entity="",
    )
    lowered = [t.lower() for t in terms]
    # A capitalised match ("Via", "Campo") must not silence the lowercase
    # content terms, or the Lucene query drifts to homonym nodes.
    assert "via del campo" in lowered
    assert "biogas" in lowered
    assert "what" not in lowered
    assert "via" not in lowered  # bare fragment must not be a term


def test_english_question_without_capitals_yields_keywords():
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    terms = retriever._build_search_terms(
        query_text="What pharmaceutical properties have been found in rice bran (pula)?",
        configured_entity="",
    )
    lowered = [t.lower() for t in terms]
    assert "rice" in lowered
    assert "bran" in lowered
    assert "pula" in lowered
    assert "what" not in lowered
    assert "been" not in lowered


def test_italian_title_phrase_still_extracted():
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    candidates = retriever._extract_entity_candidates(
        "Quali strategie propone la Regione Piemonte per gli scarti alimentari?"
    )
    assert "Regione Piemonte" in candidates


def test_request_verb_does_not_become_the_entity():
    """A request verb opening the question is not an entity.

    "Parlami delle 3C" must not search the graph for `Parlami`: the verb
    opens the question, so it is capitalised, so `_SINGLE_TOKEN_ENTITY_RE`
    reads it as a proper noun and anchors retrieval on it rather than on what
    the question asks about.
    """
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    for question in (
        "Parlami del micelio come elemento di studio",
        "Spiegami meglio la ciclicità",
        "Dammi la definizione delle 3C",
    ):
        candidates = retriever._extract_entity_candidates(question)
        assert not [c for c in candidates if c.lower() in _REQUEST_VERBS_IT], (
            question,
            candidates,
        )


def test_request_verbs_do_not_swallow_a_real_name():
    """The verbs are dropped, the entity beside them is not."""
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    candidates = retriever._extract_entity_candidates("Parlami di Terra Madre")
    assert "Terra Madre" in candidates


def test_numeric_acronym_becomes_a_search_term():
    """"Cosa sono le 3C?" must search for the acronym.

    Every other token is either a stopword or shorter than the three
    characters the keyword filter requires, so without the acronym the query
    falls back to the raw question string -- which matches no node. With it,
    the question reaches "3C (Capitale, Ciclicita e Coevoluzione)".
    """
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    for question, expected in (
        ("Cosa sono le 3C?", "3C"),
        ("Parlami delle 3C", "3C"),
        ("Cosa sono le 3c e cosa rappresentano?", "3c"),
        ("Quali strategie coprono le 10R?", "10R"),
    ):
        candidates = retriever._extract_entity_candidates(question)
        assert expected in candidates, (question, candidates)


def test_numeric_acronym_is_searched_in_both_spellings():
    """The corpus writes the framework both ways -- "3C (Capitale, ...)" and
    "3 C di CEFF" -- and the full-text index tokenises the two differently.
    Emitting only the compact form misses the nodes written the other way.
    """
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    candidates = retriever._extract_entity_candidates("Quali sono le 3C?")
    assert "3C" in candidates
    assert "3 C" in candidates


def test_english_ordinals_are_not_read_as_acronyms():
    """"21st Century", "2nd report" and "1st Award" all occur inside node names.
    They share the digit+letter shape and name nothing, so the pattern must not
    anchor retrieval on them.
    """
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    for question in (
        "What does Doughnut Economics say about the 21st century?",
        "Chi ha vinto il 1st Ideas4Change Award?",
        "Cosa dice il 2nd report sui sottoprodotti del caffe?",
    ):
        candidates = retriever._extract_entity_candidates(question)
        assert not [
            c for c in candidates if c.lower().rstrip("s") in
            {"21st", "1st", "2nd", "3rd", "11th", "12th"}
        ], (question, candidates)


def test_acronym_extraction_leaves_quantities_alone():
    """`_NUMERIC_TERM_RE` already owns quantities with an explicit unit; the
    acronym pattern must not turn "50 kg" into a second, different candidate.
    """
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    candidates = retriever._extract_entity_candidates(
        "Quante tonnes di scarto produce la filiera nel 2019?"
    )
    assert "2019" in candidates
