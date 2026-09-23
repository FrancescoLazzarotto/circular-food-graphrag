"""The retriever decides what the model is allowed to see.

It decides which channels fire, which seed becomes the anchor the graph is
walked from, how the channels are merged, and what reaches the prompt.
Judging the outcome needs the live graph, but the deciding itself is
arithmetic over rows, and rows can be handed to it.

So the graph is a fake that records what it was asked. Nothing here reaches
Neo4j or an encoder; what is pinned is the choice, not the data.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphrag import embeddings
from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever


# --- the fake graph --------------------------------------------------------


def _node(text: str, node_id: str = "", score: float = 1.0) -> dict[str, Any]:
    """A retrieved node row."""
    return {"text": text, "node_id": node_id, "score": score, "labels": ["Concept"]}


def _triple(
    subject: str = "Rice husk",
    predicate: str = "USES",
    obj: str = "Substrate",
    **rel: Any,
) -> dict[str, Any]:
    """A retrieved triple row; extra kwargs become relationship properties."""
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "subject_id": rel.pop("subject_id", ""),
        "object_id": rel.pop("object_id", ""),
        "relationship_properties": rel,
    }


class _Store:
    """Answers every call the retriever makes, and remembers each one.

    Defaults are empty so a test states only the channel it is about; anything
    it does not set returns nothing rather than a surprise.
    """

    def __init__(self, **rows: Any) -> None:
        self.nodes = rows.get("nodes", [])
        self.triples = rows.get("triples", [])
        self.vector_nodes = rows.get("vector_nodes", [])
        self.vector_triples = rows.get("vector_triples", [])
        self.neighbors = rows.get("neighbors", [])
        self.subgraph = rows.get("subgraph", [])
        self.path = rows.get("path", [])
        self.df = rows.get("df", ({}, 0))
        self.fulltext_available = rows.get("fulltext_available", True)
        self.df_raises = rows.get("df_raises", None)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _log(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))

    def fulltext_search_nodes(self, **kwargs: Any):
        self._log("fulltext_search_nodes", **kwargs)
        return list(self.nodes) if self.fulltext_available else None

    def extract_nodes(self, **kwargs: Any):
        self._log("extract_nodes", **kwargs)
        return list(self.nodes)

    def vector_search_nodes(self, **kwargs: Any):
        self._log("vector_search_nodes", **kwargs)
        return list(self.vector_nodes)

    def fulltext_search_triples(self, **kwargs: Any):
        self._log("fulltext_search_triples", **kwargs)
        return list(self.triples) if self.fulltext_available else None

    def extract_triples(self, **kwargs: Any):
        self._log("extract_triples", **kwargs)
        return list(self.triples)

    def vector_search_triples(self, **kwargs: Any):
        self._log("vector_search_triples", **kwargs)
        return list(self.vector_triples)

    def get_neighbors(self, **kwargs: Any):
        self._log("get_neighbors", **kwargs)
        return list(self.neighbors)

    def extract_subgraph(self, **kwargs: Any):
        self._log("extract_subgraph", **kwargs)
        return list(self.subgraph)

    def get_shortest_path(self, **kwargs: Any):
        self._log("get_shortest_path", **kwargs)
        return list(self.path)

    def token_document_frequency(self, **kwargs: Any):
        self._log("token_document_frequency", **kwargs)
        if self.df_raises is not None:
            raise self.df_raises
        return self.df

    def nodes_to_text(self, nodes) -> str:
        return "\n".join(str(n.get("text", "")) for n in nodes)

    def triples_to_text(self, triples) -> str:
        return "\n".join(
            f"{t.get('subject')} {t.get('predicate')} {t.get('object')}" for t in triples
        )

    def named(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for call, kwargs in self.calls if call == name]


def _config(**overrides: Any) -> AgentConfig:
    """An `AgentConfig` with every channel off; kwargs turn channels on."""
    base: dict[str, Any] = {
        "include_nodes": False,
        "include_triples": False,
        "include_neighbors": False,
        "include_subgraph": False,
        "include_shortest_path": False,
        "vector_retrieval": False,
        "use_text_retriever": False,
        "rank_triples": False,
    }
    base.update(overrides)
    return AgentConfig(**base)


def _retriever(store: _Store, **overrides: Any) -> KGRetriever:
    """A `KGRetriever` over the fake store."""
    return KGRetriever(kg_store=store, config=_config(**overrides))


# --- which channels fire ---------------------------------------------------


def test_a_channel_that_is_off_is_never_asked():
    store = _Store(nodes=[_node("Rice husk")], triples=[_triple()])

    result = _retriever(store).retrieve("cos'e' la scotta?")

    assert store.calls == []
    assert result["nodes"] == []
    assert result["triples"] == []
    assert result["context_sections"] == []


def test_the_query_is_never_echoed_into_the_context():
    # Prepending it would make `context_text` non-empty for every query,
    # silently disabling the zero-evidence branch and the whole no_retrieval
    # baseline.
    store = _Store()

    result = _retriever(store).retrieve("una domanda molto specifica")

    assert result["context_text"] == ""
    assert "una domanda molto specifica" not in result["context_text"]


def test_a_configured_entity_survives_into_the_result():
    store = _Store()

    result = _retriever(store, entity="Rice husk").retrieve("cos'e'?")

    assert result["entity"] == "Rice husk"


@pytest.mark.parametrize("placeholder", ["entity a", "entita a", "entità a"])
def test_a_placeholder_entity_is_treated_as_no_entity(placeholder):
    # These are the prompt's own slot names coming back as if they were an
    # answer: a model asked to name an entity echoing the instruction.
    assert KGRetriever._sanitize_entity_name(placeholder) == ""
    assert KGRetriever._sanitize_entity_name(f"  {placeholder.upper()}  ") == ""


@pytest.mark.parametrize("value", ["unknown", "none", "n/a"])
def test_a_word_that_only_looks_like_a_placeholder_is_kept(value):
    # Pinned as it stands: the list is exactly the three prompt slots, so a
    # node genuinely named "unknown" still reaches the graph.
    assert KGRetriever._sanitize_entity_name(value) == value


def test_a_real_entity_keeps_its_original_casing():
    assert KGRetriever._sanitize_entity_name("  Rice Husk  ") == "Rice Husk"


# --- which seed becomes the anchor ----------------------------------------


def test_the_anchor_comes_from_what_the_graph_returned_not_from_question_words():
    # Anchoring on search terms asks the graph for neighbours of "valuable",
    # which matches no node: three channels return nothing while looking as if
    # they had run.
    store = _Store(
        nodes=[_node("Circular Economy for Food", node_id="4:abc:1")],
        neighbors=[_node("Rice husk")],
    )

    _retriever(
        store, include_nodes=True, include_neighbors=True, seed_from_retrieved=True
    ).retrieve("what is valuable in the implementation?")

    assert store.named("get_neighbors")[0]["entity"] == "4:abc:1"


def test_an_element_id_is_preferred_over_a_name_as_anchor():
    # Matching by name compares six lowercased properties on every candidate,
    # which is a scan; the id lookup is direct.
    anchors = KGRetriever._graph_anchors(
        KGRetriever(_Store(), _config()),
        [_node("Rice husk", node_id="4:abc:1")],
        [_triple(subject="Straw", subject_id="4:abc:2")],
    )

    assert anchors[:2] == ["4:abc:1", "4:abc:2"]


def test_a_triple_endpoint_without_an_id_falls_back_to_its_name():
    anchors = KGRetriever._graph_anchors(
        KGRetriever(_Store(), _config()), [], [_triple(subject="Straw", obj="Compost")]
    )

    assert anchors == ["Straw", "Compost"]


def test_turning_anchor_verification_off_seeds_from_the_question_instead():
    store = _Store(neighbors=[_node("x")])

    _retriever(
        store, include_neighbors=True, entity="Rice husk", verify_anchor_exists=False
    ).retrieve("cosa contiene?")

    assert store.named("get_neighbors")[0]["entity"] == "Rice husk"


def test_the_configured_entity_is_the_first_seed():
    retriever = _retriever(_Store(), entity="Rice husk")

    seeds = retriever._seed_entities(
        query_text="q", nodes=[_node("Straw")], triples=[], search_terms=["straw"]
    )

    assert seeds[0] == "Rice husk"


def test_retrieved_names_outrank_question_words_when_configured():
    retriever = _retriever(_Store(), seed_from_retrieved=True)

    seeds = retriever._seed_entities(
        query_text="q", nodes=[_node("Rice husk")], triples=[], search_terms=["valuable"]
    )

    assert seeds == ["Rice husk", "valuable"]


def test_the_old_order_puts_question_words_first():
    retriever = _retriever(_Store(), seed_from_retrieved=False)

    seeds = retriever._seed_entities(
        query_text="q", nodes=[_node("Rice husk")], triples=[], search_terms=["valuable"]
    )

    assert seeds == ["valuable", "Rice husk"]


def test_a_question_that_yields_no_seed_at_all_seeds_on_itself():
    retriever = _retriever(_Store())

    seeds = retriever._seed_entities(
        query_text="???", nodes=[], triples=[], search_terms=[]
    )

    assert seeds == ["???"]


def test_resolving_a_seed_does_not_touch_the_graph():
    store = _Store()

    seed = _retriever(store, entity="Rice husk").resolve_entity_seed("q")

    assert seed == "Rice husk"
    assert store.named("get_neighbors") == []


def test_resolving_a_seed_with_nothing_to_go_on_returns_empty():
    assert _retriever(_Store()).resolve_entity_seed("") == ""


# --- the subgraph budget ---------------------------------------------------


def test_the_subgraph_budget_is_split_across_the_anchors_it_expands_from():
    store = _Store(
        nodes=[_node("A", node_id="1"), _node("B", node_id="2"), _node("C", node_id="3")],
        subgraph=[_triple()],
    )

    _retriever(
        store,
        include_nodes=True,
        include_subgraph=True,
        seed_from_retrieved=True,
        subgraph_seed_count=3,
        subgraph_limit=90,
        adaptive_hops=False,
    ).retrieve("q")

    calls = store.named("extract_subgraph")
    assert [c["entity"] for c in calls] == ["1", "2", "3"]
    assert {c["limit"] for c in calls} == {30}


def test_one_anchor_keeps_the_whole_subgraph_budget():
    store = _Store(nodes=[_node("A", node_id="1")], subgraph=[_triple()])

    _retriever(
        store,
        include_nodes=True,
        include_subgraph=True,
        seed_from_retrieved=True,
        subgraph_seed_count=1,
        subgraph_limit=90,
        adaptive_hops=False,
    ).retrieve("q")

    assert store.named("extract_subgraph")[0]["limit"] == 90


def test_the_same_triple_reached_from_two_anchors_is_kept_once():
    store = _Store(
        nodes=[_node("A", node_id="1"), _node("B", node_id="2")],
        subgraph=[_triple()],
    )

    result = _retriever(
        store,
        include_nodes=True,
        include_subgraph=True,
        seed_from_retrieved=True,
        subgraph_seed_count=2,
        adaptive_hops=False,
    ).retrieve("q")

    assert len(store.named("extract_subgraph")) == 2
    assert len(result["subgraph"]) == 1


def test_adaptive_hops_stops_as_soon_as_it_has_enough():
    store = _Store(subgraph=[_triple(obj=f"o{i}") for i in range(5)])
    retriever = _retriever(store, adaptive_hops=True, min_subgraph_triples=3, max_hops=4)

    collected = retriever._adaptive_subgraph("seed", hops=1, limit=10, relationship_types=None)

    assert len(collected) == 5
    assert len(store.named("extract_subgraph")) == 1  # one hop was enough


def test_adaptive_hops_widens_until_the_ceiling_when_it_never_has_enough():
    store = _Store(subgraph=[_triple()])
    retriever = _retriever(store, adaptive_hops=True, min_subgraph_triples=50, max_hops=3)

    retriever._adaptive_subgraph("seed", hops=1, limit=10, relationship_types=None)

    assert [c["hops"] for c in store.named("extract_subgraph")] == [1, 2, 3]


def test_asking_for_no_minimum_still_runs_every_hop():
    store = _Store(subgraph=[_triple()])
    retriever = _retriever(store, adaptive_hops=True, min_subgraph_triples=0, max_hops=2)

    retriever._adaptive_subgraph("seed", hops=1, limit=10, relationship_types=None)

    assert len(store.named("extract_subgraph")) == 2


# --- the shortest path -----------------------------------------------------


def test_the_shortest_path_needs_two_distinct_anchors():
    store = _Store(nodes=[_node("A", node_id="1")], path=[_triple()])

    _retriever(
        store, include_nodes=True, include_shortest_path=True, seed_from_retrieved=True
    ).retrieve("q")

    assert store.named("get_shortest_path") == []


def test_two_anchors_produce_a_shortest_path_query():
    store = _Store(
        nodes=[_node("A", node_id="1"), _node("B", node_id="2")], path=[_triple()]
    )

    _retriever(
        store, include_nodes=True, include_shortest_path=True, seed_from_retrieved=True
    ).retrieve("q")

    call = store.named("get_shortest_path")[0]
    assert (call["entity_a"], call["entity_b"]) == ("1", "2")


def test_configured_endpoints_win_over_retrieved_anchors():
    store = _Store(nodes=[_node("A", node_id="1")], path=[_triple()])

    _retriever(
        store,
        include_nodes=True,
        include_shortest_path=True,
        entity_a="Straw",
        entity_b="Compost",
    ).retrieve("q")

    call = store.named("get_shortest_path")[0]
    assert (call["entity_a"], call["entity_b"]) == ("Straw", "Compost")


# --- merging the channels --------------------------------------------------


def test_the_vector_channel_is_read_before_the_lexical_one():
    store = _Store(
        vector_nodes=[_node("spreco alimentare", node_id="v1")],
        nodes=[_node("food waste", node_id="l1")],
    )
    retriever = _retriever(store, include_nodes=True, vector_retrieval=True)

    result = retriever._collect_nodes(["food"], limit=10, query_vector=[0.1, 0.2])

    assert [n["node_id"] for n in result] == ["v1", "l1"]


def test_a_node_both_channels_return_is_kept_once():
    same = _node("food waste", node_id="n1")
    store = _Store(vector_nodes=[same], nodes=[same])
    retriever = _retriever(store, include_nodes=True, vector_retrieval=True)

    result = retriever._collect_nodes(["food"], limit=10, query_vector=[0.1])

    assert len(result) == 1


def test_the_node_budget_is_a_hard_stop():
    store = _Store(nodes=[_node(f"n{i}", node_id=str(i)) for i in range(10)])
    retriever = _retriever(store, include_nodes=True)

    assert len(retriever._collect_nodes(["x"], limit=3)) == 3


def test_a_zero_budget_asks_nothing():
    store = _Store(nodes=[_node("x")])
    retriever = _retriever(store, include_nodes=True)

    assert retriever._collect_nodes(["x"], limit=0) == []
    assert store.calls == []


def test_without_a_fulltext_index_the_legacy_scan_runs_once_per_term():
    store = _Store(nodes=[_node("x", node_id="1")], fulltext_available=False)
    retriever = _retriever(store, include_nodes=True)

    retriever._collect_nodes(["alpha", "beta"], limit=10)

    assert [c["text"] for c in store.named("extract_nodes")] == ["alpha", "beta"]


def test_the_legacy_scan_stops_at_the_budget_without_asking_the_rest():
    store = _Store(
        nodes=[_node(f"n{i}", node_id=str(i)) for i in range(3)], fulltext_available=False
    )
    retriever = _retriever(store, include_nodes=True)

    retriever._collect_nodes(["alpha", "beta", "gamma"], limit=2)

    assert len(store.named("extract_nodes")) == 1


def test_triples_merge_the_same_way_as_nodes():
    store = _Store(
        vector_triples=[_triple(obj="vettoriale")],
        triples=[_triple(obj="lessicale")],
    )
    retriever = _retriever(store, include_triples=True, vector_retrieval=True)

    result = retriever._collect_triples(["x"], limit=10, query_vector=[0.1])

    assert [t["object"] for t in result] == ["vettoriale", "lessicale"]


def test_the_triple_fallback_scan_also_respects_the_budget():
    store = _Store(
        triples=[_triple(obj=f"o{i}") for i in range(4)], fulltext_available=False
    )
    retriever = _retriever(store, include_triples=True)

    assert len(retriever._collect_triples(["a", "b"], limit=2)) == 2


# --- what never reaches the model -----------------------------------------


def test_a_dropped_predicate_never_reaches_the_context():
    store = _Store(
        triples=[_triple(predicate="MENTIONED_IN"), _triple(predicate="USES")]
    )

    result = _retriever(
        store, include_triples=True, drop_predicates=["mentioned_in"]
    ).retrieve("q")

    assert [t["predicate"] for t in result["triples"]] == ["USES"]


def test_dropping_happens_after_retrieval_so_ranking_is_unaffected():
    # Filtering in Cypher would change which seeds scored; filtering here keeps
    # the ranking of what remains the one the retriever intended.
    store = _Store(triples=[_triple(predicate="MENTIONED_IN")])

    _retriever(store, include_triples=True, drop_predicates=["mentioned_in"]).retrieve("q")

    assert store.named("fulltext_search_triples")[0]["limit"] > 0


def test_no_drop_list_leaves_every_predicate_alone():
    retriever = _retriever(_Store(), drop_predicates=[])
    triples = [_triple(predicate="MENTIONED_IN")]

    assert retriever._drop_empty_predicates(triples) == triples


# --- the ranker ------------------------------------------------------------


def test_a_triple_that_shares_words_with_the_question_ranks_above_one_that_does_not():
    retriever = _retriever(
        _Store(), ranker_weight_lexical=1.0, ranker_weight_mention=0.0,
        ranker_weight_confidence=0.0,
    )
    triples = [
        _triple(subject="Compost", obj="Terreno"),
        _triple(subject="Rice husk", obj="Substrate"),
    ]

    ranked = retriever._rank_triples(triples, "what does rice husk become?")

    assert ranked[0]["subject"] == "Rice husk"


def test_a_question_with_only_stopwords_leaves_the_order_alone():
    retriever = _retriever(_Store())
    triples = [_triple(obj="a"), _triple(obj="b")]

    assert retriever._rank_triples(triples, "e quindi?") == triples


def test_ranking_nothing_returns_nothing():
    assert _retriever(_Store())._rank_triples([], "q") == []


def test_a_system_link_is_pushed_down_the_ranking():
    retriever = _retriever(
        _Store(), ranker_weight_lexical=1.0, ranker_weight_mention=0.0,
        ranker_weight_confidence=0.0, ranker_system_link_penalty=0.9,
    )
    triples = [
        _triple(subject="Rice husk", predicate="MENTIONED_IN", obj="doc.pdf"),
        _triple(subject="Rice husk", predicate="USES", obj="Substrate"),
    ]

    ranked = retriever._rank_triples(triples, "rice husk")

    assert ranked[0]["predicate"] == "USES"


@pytest.mark.parametrize(
    "value, expected", [(5, 5), ("3", 3), (0, 1), (-2, 1), (None, 1), ("x", 1)]
)
def test_a_mention_count_is_never_below_one(value, expected):
    assert KGRetriever._mention_count(_triple(mention_count=value)) == expected


def test_mentions_only_separate_triples_when_they_actually_differ():
    # Only a minority of linked triples carry more than one mention; when they
    # are all 1 the term is constant and cannot rank anything.
    retriever = _retriever(
        _Store(), ranker_weight_lexical=0.0, ranker_weight_mention=1.0,
        ranker_weight_confidence=0.0,
    )
    flat = [_triple(obj=f"o{i}", mention_count=1) for i in range(3)]

    assert retriever._rank_triples(flat, "rice husk substrate") == flat


def test_a_triple_mentioned_more_often_outranks_one_mentioned_once():
    retriever = _retriever(
        _Store(), ranker_weight_lexical=0.0, ranker_weight_mention=1.0,
        ranker_weight_confidence=0.0,
    )
    triples = [_triple(obj="raro", mention_count=1), _triple(obj="frequente", mention_count=8)]

    assert retriever._rank_triples(triples, "rice husk")[0]["object"] == "frequente"


@pytest.mark.parametrize("text, expected", [("Rice husk", {"rice", "husk"}), ("e di un", set())])
def test_tokenising_drops_stopwords_and_short_tokens(text, expected):
    assert KGRetriever._tokenize(text) == expected


# --- lexical specificity ---------------------------------------------------


def test_a_keyword_that_matches_most_of_the_graph_is_dropped():
    store = _Store(df=({"food": 900, "scotta": 3}, 1000))
    retriever = _retriever(store, lexical_specificity=True, lexical_df_max_ratio=0.1)

    assert retriever._rank_keywords_by_specificity(["food", "scotta"]) == ["scotta"]


def test_the_rarest_keyword_goes_first():
    # Both under the 1 % ceiling, so what is being tested is the order alone.
    store = _Store(df=({"siero": 2, "latte": 9}, 1000))
    retriever = _retriever(store, lexical_specificity=True)

    assert retriever._rank_keywords_by_specificity(["latte", "siero"]) == ["siero", "latte"]


def test_a_keyword_the_graph_has_never_seen_ranks_last():
    # df 0 is not evidence of specificity: it matches nothing on its own.
    store = _Store(df=({"latte": 9}, 1000))
    retriever = _retriever(store, lexical_specificity=True)

    assert retriever._rank_keywords_by_specificity(["ignoto", "latte"]) == ["latte", "ignoto"]


def test_without_frequencies_the_keywords_are_left_as_they_came():
    store = _Store(df=({}, 0))
    retriever = _retriever(store, lexical_specificity=True)

    assert retriever._rank_keywords_by_specificity(["b", "a"]) == ["b", "a"]


def test_an_unavailable_frequency_table_disables_specificity_instead_of_failing():
    store = _Store(df_raises=RuntimeError("no cache"))
    retriever = _retriever(store, lexical_specificity=True)

    assert retriever._rank_keywords_by_specificity(["a"]) == ["a"]
    assert retriever._token_df() == ({}, 0)


def test_the_frequency_table_is_loaded_once_per_retriever():
    store = _Store(df=({"a": 1}, 10))
    retriever = _retriever(store, lexical_specificity=True)

    retriever._token_df()
    retriever._token_df()

    assert len(store.named("token_document_frequency")) == 1


def test_a_phrase_is_boosted_above_single_tokens():
    store = _Store(df=({"latte": 40}, 1000))
    retriever = _retriever(store, lexical_specificity=True, lexical_phrase_boost=4.0)

    boosts = retriever._term_boosts(["siero di latte", "latte"])

    assert boosts["siero di latte"] == 4.0
    assert boosts["latte"] < 4.0


def test_a_rarer_token_is_boosted_more_than_a_common_one():
    store = _Store(df=({"siero": 1, "latte": 100}, 1000))
    retriever = _retriever(store, lexical_specificity=True)

    boosts = retriever._term_boosts(["siero", "latte"])

    assert boosts["siero"] > boosts["latte"]


def test_boosts_are_capped():
    store = _Store(df=({"unicum": 1}, 10**9))
    retriever = _retriever(store, lexical_specificity=True, lexical_max_token_boost=3.0)

    assert retriever._term_boosts(["unicum"])["unicum"] == 3.0


def test_specificity_off_means_no_boosts_at_all():
    store = _Store(df=({"latte": 40}, 1000))
    retriever = _retriever(store, lexical_specificity=False)

    assert retriever._term_boosts(["siero di latte", "latte"]) == {}


# --- the vector channel and its degradation -------------------------------


def test_the_query_is_embedded_once_and_reused(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        embeddings, "encode_query", lambda text: calls.append(text) or [0.1, 0.2]
    )
    retriever = _retriever(_Store(), vector_retrieval=True)

    assert retriever._query_vector("q") == [0.1, 0.2]
    assert retriever._query_vector("q") == [0.1, 0.2]
    assert calls == ["q"]


def test_a_different_question_is_embedded_again(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        embeddings, "encode_query", lambda text: calls.append(text) or [0.1]
    )
    retriever = _retriever(_Store(), vector_retrieval=True)

    retriever._query_vector("una")
    retriever._query_vector("altra")

    assert calls == ["una", "altra"]


def test_the_vector_channel_off_never_calls_the_encoder(monkeypatch):
    def _boom(text):
        raise AssertionError("the encoder should not be reached")

    monkeypatch.setattr(embeddings, "encode_query", _boom)

    assert _retriever(_Store(), vector_retrieval=False)._query_vector("q") == []


def test_an_empty_question_is_never_embedded(monkeypatch):
    monkeypatch.setattr(
        embeddings, "encode_query", lambda text: (_ for _ in ()).throw(AssertionError())
    )

    assert _retriever(_Store(), vector_retrieval=True)._query_vector("") == []


def test_a_dead_encoder_stops_the_run_rather_than_changing_the_method(monkeypatch):
    # Degrading quietly makes a campaign model-asymmetric: some generators lose
    # the cross-lingual channel on some queries and the run still looks
    # complete.
    monkeypatch.delenv("GRAPHRAG_VECTOR_ALLOW_DEGRADED", raising=False)
    monkeypatch.setattr(
        embeddings,
        "encode_query",
        lambda text: (_ for _ in ()).throw(embeddings.EmbeddingUnavailable("down")),
    )
    retriever = _retriever(_Store(), vector_retrieval=True)

    with pytest.raises(embeddings.EmbeddingUnavailable, match="start the encoder|Start the encoder"):
        retriever._query_vector("q")


@pytest.mark.parametrize("flag", ["1", "true", "yes"])
def test_an_interactive_caller_can_accept_lexical_only(monkeypatch, flag):
    monkeypatch.setenv("GRAPHRAG_VECTOR_ALLOW_DEGRADED", flag)
    monkeypatch.setattr(
        embeddings,
        "encode_query",
        lambda text: (_ for _ in ()).throw(embeddings.EmbeddingUnavailable("down")),
    )
    retriever = _retriever(_Store(), vector_retrieval=True)

    assert retriever._query_vector("q") == []
    assert retriever.vector_skips == 1


def test_a_degraded_answer_can_be_told_from_a_healthy_one(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_VECTOR_ALLOW_DEGRADED", "1")
    monkeypatch.setattr(
        embeddings,
        "encode_query",
        lambda text: (_ for _ in ()).throw(embeddings.EmbeddingUnavailable("down")),
    )
    retriever = _retriever(_Store(), vector_retrieval=True)

    assert retriever.vector_skips == 0
    retriever._query_vector("una")
    retriever._query_vector("altra")
    assert retriever.vector_skips == 2


def test_the_vector_score_floor_is_passed_to_the_graph(monkeypatch):
    monkeypatch.setattr(embeddings, "encode_query", lambda text: [0.1])
    store = _Store(vector_nodes=[_node("x")])
    retriever = _retriever(store, include_nodes=True, vector_retrieval=True, vector_min_score=0.8)

    retriever.retrieve("q")

    assert store.named("vector_search_nodes")[0]["min_score"] == 0.8


# --- what the model is shown ----------------------------------------------


def test_each_channel_gets_its_own_labelled_section():
    retriever = _retriever(_Store(), include_triple_metadata=False)

    sections = retriever._build_context_sections(
        nodes=[_node("Rice husk")],
        triples=[_triple()],
        neighbors=[_node("Straw")],
        subgraph=[_triple(obj="Compost")],
        shortest_path=[_triple(obj="Terreno")],
        text_chunks=["un paragrafo"],
    )

    assert [s.split(":")[0] for s in sections] == [
        "Retrieved text",
        "Matched nodes",
        "Matched triples",
        "Neighbors",
        "Subgraph",
        "Shortest path",
    ]


def test_an_empty_channel_contributes_no_section():
    retriever = _retriever(_Store())

    assert retriever._build_context_sections([], [], [], [], [], []) == []


def test_text_comes_before_the_graph_in_the_context():
    retriever = _retriever(_Store())

    sections = retriever._build_context_sections(
        nodes=[_node("Rice husk")], triples=[], neighbors=[], subgraph=[],
        shortest_path=[], text_chunks=["paragrafo"],
    )

    assert sections[0].startswith("Retrieved text")


def test_a_triple_carries_its_provenance_when_metadata_is_on():
    retriever = _retriever(_Store(), include_triple_metadata=True)

    line = retriever._format_triples(
        [_triple(source_doc="report.pdf", page_range="12-13", mention_count=4, confidence=0.9)]
    )

    assert "source=report.pdf" in line
    assert "pages=12-13" in line
    assert "mentions=4" in line
    assert "conf=0.90" in line


def test_a_single_mention_is_not_worth_printing():
    retriever = _retriever(_Store(), include_triple_metadata=True)

    assert "mentions=" not in retriever._format_triples([_triple(mention_count=1)])


def test_metadata_off_falls_back_to_the_graphs_own_rendering():
    store = _Store()
    retriever = _retriever(store, include_triple_metadata=False)

    assert retriever._format_triples([_triple()]) == "Rice husk USES Substrate"


def test_formatting_nothing_produces_nothing():
    assert _retriever(_Store(), include_triple_metadata=True)._format_triples([]) == ""


def test_the_public_formatter_is_the_private_one():
    retriever = _retriever(_Store(), include_triple_metadata=True)
    triples = [_triple()]

    assert retriever.format_triples(triples) == retriever._format_triples(triples)


# --- multi_hop -------------------------------------------------------------


def test_multi_hop_without_an_entity_asks_nothing():
    store = _Store()

    assert _retriever(store).multi_hop() == []
    assert store.calls == []


def test_multi_hop_falls_back_to_the_configured_entity():
    store = _Store(subgraph=[_triple()])

    _retriever(store, entity="Rice husk").multi_hop()

    assert store.named("extract_subgraph")[0]["entity"] == "Rice husk"


def test_multi_hop_arguments_override_the_configuration():
    store = _Store(subgraph=[_triple()])

    _retriever(store, entity="Rice husk", hops=1, subgraph_limit=10).multi_hop(
        entity="Straw", hops=3, limit=99
    )

    call = store.named("extract_subgraph")[0]
    assert (call["entity"], call["hops"], call["limit"]) == ("Straw", 3, 99)


# --- degradation warnings --------------------------------------------------


def test_a_text_strategy_without_a_pipeline_says_so(caplog):
    # A "hybrid" run that silently drops to KG-only looks valid while measuring
    # the wrong strategy.
    import logging

    with caplog.at_level(logging.WARNING):
        KGRetriever(kg_store=_Store(), config=_config(use_text_retriever=True))

    assert "no text_pipeline was provided" in caplog.text


def test_a_kg_only_strategy_warns_about_nothing(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        KGRetriever(kg_store=_Store(), config=_config(use_text_retriever=False))

    assert caplog.text == ""
