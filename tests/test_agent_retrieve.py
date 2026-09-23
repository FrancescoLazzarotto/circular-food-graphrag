"""The retrieve node: which channels are asked, what is merged, what is cached.

`_retrieve` picks between four retrieval modes, merges the results of up to
four queries under per-channel caps, and decides what the cache key is — a
wrong key there serves one turn's evidence to another, which is invisible in
the answer and fatal to a measurement.

The retriever is a fake that records what it was asked; no graph, no encoder.
"""

from __future__ import annotations

from typing import Any

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig


def _node(text: str) -> dict[str, Any]:
    """A retrieved node row."""
    return {"text": text, "node_id": text, "labels": ["Concept"]}


def _triple(subject: str = "a", predicate: str = "USES", obj: str = "b") -> dict[str, Any]:
    """A retrieved triple row."""
    return {"subject": subject, "predicate": predicate, "object": obj}


class _Retriever:
    """Answers `retrieve`, `multi_hop` and `resolve_entity_seed`, and remembers."""

    def __init__(self, **rows: Any) -> None:
        self.rows = rows
        self.queries: list[tuple[str, tuple[str, ...]]] = []
        self.hops: list[str] = []
        self.seeds_asked: list[str] = []
        self.ranked: list[str] = []

    def format_triples(self, triples) -> str:
        return "\n".join(
            f"({t.get('subject')}, {t.get('predicate')}, {t.get('object')})"
            for t in triples
        )

    def retrieve(self, query: str, prefer_documents=()) -> dict[str, Any]:
        self.queries.append((query, tuple(prefer_documents)))
        if self.rows.get("nodes_per_query"):
            # A distinct batch per query, so merging is tested rather than
            # deduplication.
            index = len(self.queries) - 1
            nodes = [_node(f"q{index}n{i}") for i in range(self.rows["nodes_per_query"])]
        else:
            nodes = list(self.rows.get("nodes", []))
        payload = {
            "query": query,
            "context_text": self.rows.get("context", f"contesto per {query}"),
            "nodes": nodes,
            "triples": list(self.rows.get("triples", [])),
            "neighbors": list(self.rows.get("neighbors", [])),
            "subgraph": list(self.rows.get("subgraph", [])),
            "shortest_path": list(self.rows.get("shortest_path", [])),
            "text_sources": list(self.rows.get("text_sources", [])),
            "text_chunks": list(self.rows.get("text_chunks", [])),
        }
        return payload

    def multi_hop(self, entity: str, hops: int = 1, limit: int = 200):
        self.hops.append(entity)
        return list(self.rows.get("multi_hop", []))

    def resolve_entity_seed(self, query: str) -> str:
        self.seeds_asked.append(query)
        seeds = self.rows.get("seeds", {})
        return seeds.get(query, "")

    def rank_triples(self, triples, query_text):
        self.ranked.append(query_text)
        return list(reversed(list(triples)))


def _agent(retriever=None, **overrides: Any) -> KGRAGAgent:
    """Agent over the given retriever, with no LLM, warmup and cache off."""
    base: dict[str, Any] = {"llm_warmup": False, "enable_cache": False}
    base.update(overrides)
    return KGRAGAgent(config=AgentConfig(**base), kg_retriever=retriever, llm=None)


# --- which query is retrieved on -------------------------------------------


def test_the_rewritten_question_is_what_gets_retrieved():
    retriever = _Retriever()

    _agent(retriever)._retrieve(
        {"question": "e quindi?", "rewritten_question": "cos'e' la scotta?"}
    )

    assert retriever.queries[0][0] == "cos'e' la scotta?"


def test_without_a_rewrite_the_question_is_used_as_typed():
    retriever = _Retriever()

    _agent(retriever)._retrieve({"question": "cos'e' la scotta?"})

    assert retriever.queries[0][0] == "cos'e' la scotta?"


def test_sub_questions_are_ignored_unless_decomposition_is_on():
    retriever = _Retriever()

    _agent(retriever, enable_decomposition_step=False)._retrieve(
        {"question": "q", "sub_questions": ["una", "due"]}
    )

    assert len(retriever.queries) == 1


def test_decomposition_retrieves_once_per_sub_question():
    retriever = _Retriever()

    _agent(retriever, enable_decomposition_step=True)._retrieve(
        {"question": "q", "sub_questions": ["una", "due"]}
    )

    assert [q for q, _ in retriever.queries] == ["q", "una", "due"]


def test_a_sub_question_repeating_the_question_is_not_retrieved_twice():
    retriever = _Retriever()

    _agent(retriever, enable_decomposition_step=True)._retrieve(
        {"question": "q", "sub_questions": ["  Q  ", "altra"]}
    )

    assert [q for q, _ in retriever.queries] == ["q", "altra"]


def test_the_number_of_retrieval_queries_is_capped():
    agent = _agent(_Retriever(), enable_decomposition_step=True)

    queries = agent._build_retrieval_queries(
        query="q", sub_questions=[f"s{i}" for i in range(10)]
    )

    assert len(queries) == 4


def test_a_quoted_source_is_passed_to_every_retrieval():
    retriever = _Retriever()

    _agent(retriever)._retrieve(
        {"question": "q", "quoted_sources": ["REPORT MATTM, p. 70", "  "]}
    )

    assert retriever.queries[0][1] == ("REPORT MATTM, p. 70",)


# --- the four modes --------------------------------------------------------


def test_text_mode_keeps_only_the_text_channel():
    retriever = _Retriever(nodes=[_node("Scotta")], triples=[_triple()])

    out = _agent(retriever)._retrieve(
        {"question": "q", "chosen_retrieval_mode": "TEXT"}
    )

    assert out["retrieved_nodes"] == []
    assert out["kg_triples"] == []
    assert out["text_context"]


def test_kg_and_hybrid_keep_every_graph_channel():
    retriever = _Retriever(
        nodes=[_node("Scotta")],
        triples=[_triple()],
        neighbors=[_node("Siero")],
        subgraph=[_triple("c", "PART_OF", "d")],
        shortest_path=[_triple("e", "USES", "f")],
    )

    out = _agent(retriever)._retrieve(
        {"question": "q", "chosen_retrieval_mode": "HYBRID"}
    )

    assert out["retrieved_nodes_count"] == 1
    assert len(out["kg_triples"]) == 1
    assert out["retrieved_neighbors_count"] == 1
    assert out["retrieved_subgraph_count"] == 1
    assert out["retrieved_shortest_path_count"] == 1


def test_kg_mode_renders_the_triples_when_there_is_no_context():
    retriever = _Retriever(context="", triples=[_triple("Scotta", "IS_TYPE_OF", "Residuo")])

    out = _agent(retriever)._retrieve({"question": "q", "chosen_retrieval_mode": "KG"})

    assert "Scotta" in out["text_context"]


def test_multihop_expands_from_the_seeds_the_retriever_resolved():
    retriever = _Retriever(
        seeds={"q": "Scotta"}, multi_hop=[_triple("Scotta", "PART_OF", "Siero")]
    )

    out = _agent(retriever)._retrieve(
        {"question": "q", "chosen_retrieval_mode": "MULTIHOP"}
    )

    assert retriever.hops == ["Scotta"]
    assert out["retrieved_subgraph_count"] == 1
    assert out["retrieved_nodes"] == []


def test_multihop_expands_from_at_most_two_seeds():
    retriever = _Retriever(
        seeds={"q": "A", "una": "B", "due": "C"}, multi_hop=[_triple()]
    )

    _agent(retriever, enable_decomposition_step=True)._retrieve(
        {
            "question": "q",
            "sub_questions": ["una", "due"],
            "chosen_retrieval_mode": "MULTIHOP",
        }
    )

    assert retriever.hops == ["A", "B"]


def test_multihop_does_not_expand_from_the_same_seed_twice():
    retriever = _Retriever(seeds={"q": "A", "una": "  a  "}, multi_hop=[_triple()])

    _agent(retriever, enable_decomposition_step=True)._retrieve(
        {"question": "q", "sub_questions": ["una"], "chosen_retrieval_mode": "MULTIHOP"}
    )

    assert retriever.hops == ["A"]


def test_an_unknown_mode_falls_back_to_asking_the_retriever_once():
    retriever = _Retriever(nodes=[_node("Scotta")])

    out = _agent(retriever)._retrieve(
        {"question": "q", "chosen_retrieval_mode": "SOMETHING_ELSE"}
    )

    assert len(retriever.queries) == 1
    assert out["retrieved_nodes_count"] == 1


def test_without_a_retriever_nothing_is_retrieved_and_nothing_breaks():
    out = _agent(None)._retrieve({"question": "q"})

    assert out["retrieved_nodes_count"] == 0
    assert out["text_context"] == ""


# --- merging several queries -----------------------------------------------


def test_the_same_node_from_two_queries_is_kept_once():
    retriever = _Retriever(nodes=[_node("Scotta")])

    out = _agent(retriever, enable_decomposition_step=True)._retrieve(
        {"question": "q", "sub_questions": ["altra"]}
    )

    assert out["retrieved_nodes_count"] == 1


def test_each_channel_keeps_its_own_cap_across_queries():
    # The cap is checked before the append: checked after it, four retrieval
    # queries could finish three items over it.
    retriever = _Retriever(nodes_per_query=5)

    out = _agent(
        retriever, enable_decomposition_step=True, nodes_limit=6
    )._retrieve({"question": "q", "sub_questions": ["a", "b", "c"]})

    assert len(retriever.queries) == 4  # four distinct batches of five
    assert out["retrieved_nodes_count"] == 6  # not 7, 8 or 9


def test_merged_results_are_reranked_only_when_asked_and_only_for_several_queries():
    retriever = _Retriever(triples=[_triple("x"), _triple("y")])

    _agent(
        retriever,
        enable_decomposition_step=True,
        rerank_merged_results=True,
        rank_triples=True,
    )._retrieve({"question": "q", "sub_questions": ["altra"]})

    assert retriever.ranked  # ranked against the original question


def test_a_single_query_is_not_reranked():
    retriever = _Retriever(triples=[_triple()])

    _agent(retriever, rerank_merged_results=True, rank_triples=True)._retrieve(
        {"question": "q"}
    )

    assert retriever.ranked == []


# --- the cache key ---------------------------------------------------------


def test_the_same_question_is_retrieved_once():
    retriever = _Retriever(nodes=[_node("Scotta")])
    agent = _agent(retriever, enable_cache=True)

    agent._retrieve({"question": "q"})
    agent._retrieve({"question": "q"})

    assert len(retriever.queries) == 1


def test_a_different_mode_is_a_different_cache_entry():
    retriever = _Retriever()
    agent = _agent(retriever, enable_cache=True)

    agent._retrieve({"question": "q", "chosen_retrieval_mode": "KG"})
    agent._retrieve({"question": "q", "chosen_retrieval_mode": "TEXT"})

    assert len(retriever.queries) == 2


def test_a_turn_that_quotes_a_source_does_not_reuse_one_that_did_not():
    # The preferred documents reorder the text channel, so serving a cached
    # entry from a turn that quoted nothing would drop the preference without
    # a trace.
    retriever = _Retriever()
    agent = _agent(retriever, enable_cache=True)

    agent._retrieve({"question": "q"})
    agent._retrieve({"question": "q", "quoted_sources": ["REPORT MATTM, p. 70"]})

    assert len(retriever.queries) == 2
    assert retriever.queries[1][1] == ("REPORT MATTM, p. 70",)


def test_the_same_quoted_sources_in_another_order_hit_the_same_entry():
    retriever = _Retriever()
    agent = _agent(retriever, enable_cache=True)

    agent._retrieve({"question": "q", "quoted_sources": ["A", "B"]})
    agent._retrieve({"question": "q", "quoted_sources": ["B", "A"]})

    assert len(retriever.queries) == 1


def test_a_rewrite_that_changes_the_text_is_retrieved_again():
    retriever = _Retriever()
    agent = _agent(retriever, enable_cache=True)

    agent._retrieve({"question": "q"})
    agent._retrieve({"question": "q", "rewritten_question": "q riformulata"})

    assert len(retriever.queries) == 2


def test_with_the_cache_off_every_turn_retrieves():
    retriever = _Retriever()
    agent = _agent(retriever, enable_cache=False)

    agent._retrieve({"question": "q"})
    agent._retrieve({"question": "q"})

    assert len(retriever.queries) == 2


# --- the routing decision --------------------------------------------------


def test_routing_off_always_chooses_hybrid():
    agent = _agent(_Retriever(), enable_adaptive_routing_step=False)

    assert agent._adaptive_route({"question": "q"}) == {"chosen_retrieval_mode": "HYBRID"}


def test_routing_without_a_model_chooses_hybrid_rather_than_dropping_the_graph():
    agent = _agent(_Retriever(), enable_adaptive_routing_step=True)

    assert agent._adaptive_route({"question": "q"}) == {"chosen_retrieval_mode": "HYBRID"}
