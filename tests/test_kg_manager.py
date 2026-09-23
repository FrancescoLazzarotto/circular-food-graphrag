"""The layer between the agent and Neo4j: retries, outages, and what it counts.

`kg/manager.py` decides whether a failed query is worth asking again, whether
the graph is down at all, and how the lexical channel weighs a token. Those are
the paths that can turn one unreachable graph into a demo that waits five
minutes.

No Neo4j: `Neo4jGraph` is replaced by a stub that answers or raises what the
test chose. `test_graph_resilience.py` covers the driver's own settings; this
covers what the manager does with what comes back.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from graphrag.config import KGConfig
from graphrag.kg import manager as manager_module
from graphrag.kg.manager import KnowledgeGraphManager


class _Graph:
    """Stands in for Neo4jGraph. Each query returns, or raises, in order."""

    def __init__(self, outcomes: list[Any] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = 0
        self._driver = self

    def query(self, cypher: str, params: dict[str, Any] | None = None):
        self.calls.append((" ".join(cypher.split()), dict(params or {})))
        if not self.outcomes:
            return []
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed += 1


def _manager(graph: _Graph, **config: Any) -> KnowledgeGraphManager:
    """A `KnowledgeGraphManager` over the stub graph; kwargs override the config."""
    base = {
        "url": "bolt://localhost:7689",
        "username": "neo4j",
        "password": "pw",
        "database": "neo4j",
    }
    base.update(config)
    return KnowledgeGraphManager(config=KGConfig(**base), graph=graph)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retries are about the decision, not the wall clock."""
    monkeypatch.setattr(manager_module.time, "sleep", lambda _s: None)


# --- which failures are worth asking again ---------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "SessionExpired: the session was closed",
        "ServiceUnavailable: could not connect",
        "TransientError: please retry",
        "failed to read from defunct connection",
        "connection reset by peer",
        "the operation timed out",
        "network is unreachable",
    ],
)
def test_a_transport_failure_is_worth_another_attempt(message):
    assert KnowledgeGraphManager._is_retryable_query_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    ["SyntaxError: invalid cypher", "constraint already exists", "unknown function apoc.x"],
)
def test_a_refusal_from_the_server_is_not_retried(message):
    assert KnowledgeGraphManager._is_retryable_query_error(ValueError(message)) is False


@pytest.mark.parametrize(
    "message",
    [
        "ServiceUnavailable: nobody home",
        "could not connect to the server",
        "Unable to retrieve routing information",
        "cannot resolve address 588fe1bc.databases.neo4j.io",
    ],
)
def test_an_unanswered_call_counts_as_an_outage(message):
    assert KnowledgeGraphManager._is_outage(RuntimeError(message)) is True


def test_a_server_asking_to_be_asked_again_is_not_an_outage():
    # A TransientError means the graph is reachable, so the breaker must not
    # trip: it is narrower than the retry test on purpose.
    assert KnowledgeGraphManager._is_outage(RuntimeError("TransientError: retry")) is False


# --- the retry loop --------------------------------------------------------


def test_a_query_that_works_is_asked_once():
    graph = _Graph([[{"n": 1}]])

    assert _manager(graph).run_query("MATCH (n) RETURN n") == [{"n": 1}]
    assert len(graph.calls) == 1


def test_a_transient_failure_is_retried_until_it_works():
    graph = _Graph([RuntimeError("connection reset"), [{"n": 1}]])

    assert _manager(graph).run_query("MATCH (n) RETURN n") == [{"n": 1}]
    assert len(graph.calls) == 2


def test_a_permanent_failure_is_raised_at_once():
    graph = _Graph([ValueError("SyntaxError: invalid cypher")])

    with pytest.raises(ValueError, match="invalid cypher"):
        _manager(graph).run_query("BROKEN")

    assert len(graph.calls) == 1


def test_the_retries_run_out():
    manager = _manager(_Graph([RuntimeError("connection reset")] * 20))
    manager.query_retry_attempts = 3

    with pytest.raises(RuntimeError, match="connection reset"):
        manager.run_query("MATCH (n) RETURN n")

    assert len(manager.graph.calls) == 3


def test_a_retry_closes_the_driver_it_is_replacing(monkeypatch):
    # On a flaky link the loop builds a new Neo4jGraph per attempt; every
    # previous driver, and its connection pool, must be closed.
    first = _Graph([RuntimeError("connection reset")])
    replacements = [_Graph([[{"n": 1}]])]
    monkeypatch.setattr(
        KnowledgeGraphManager, "_build_graph", lambda self: replacements.pop(0)
    )

    _manager(first).run_query("MATCH (n) RETURN n")

    assert first.closed == 1


def test_a_driver_that_will_not_close_does_not_mask_the_retry(monkeypatch, caplog):
    class _Stubborn(_Graph):
        def close(self) -> None:
            raise RuntimeError("already closed")

    first = _Stubborn([RuntimeError("connection reset")])
    monkeypatch.setattr(
        KnowledgeGraphManager, "_build_graph", lambda self: _Graph([[{"n": 1}]])
    )

    with caplog.at_level(logging.DEBUG):
        assert _manager(first).run_query("MATCH (n) RETURN n") == [{"n": 1}]


# --- the outage breaker ----------------------------------------------------


def test_an_outage_is_remembered_instead_of_rediscovered(caplog):
    graph = _Graph([RuntimeError("ServiceUnavailable: nobody home")])
    manager = _manager(graph)
    manager.query_retry_attempts = 1

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            manager.run_query("MATCH (n) RETURN n")

    assert "Neo4j unreachable" in caplog.text

    # The next query fails without touching the network at all.
    with pytest.raises(RuntimeError, match="nobody home"):
        manager.run_query("MATCH (n) RETURN n")

    assert len(graph.calls) == 1


def test_the_breaker_opens_again_once_its_memory_expires(monkeypatch):
    graph = _Graph([RuntimeError("ServiceUnavailable: nobody home"), [{"n": 1}]])
    manager = _manager(graph)
    manager.query_retry_attempts = 1

    with pytest.raises(RuntimeError):
        manager.run_query("MATCH (n) RETURN n")

    manager._outage_until = 0.0  # as if the memory window had passed

    assert manager.run_query("MATCH (n) RETURN n") == [{"n": 1}]


def test_a_working_query_clears_a_remembered_outage():
    manager = _manager(_Graph([[{"n": 1}]]))
    manager._outage_error = RuntimeError("stale")
    manager._outage_until = 0.0

    manager.run_query("MATCH (n) RETURN n")

    assert manager._outage_error is None


def test_a_refusal_does_not_trip_the_breaker():
    graph = _Graph([ValueError("SyntaxError: invalid cypher"), [{"n": 1}]])
    manager = _manager(graph)

    with pytest.raises(ValueError):
        manager.run_query("BROKEN")

    assert manager.run_query("MATCH (n) RETURN n") == [{"n": 1}]


# --- document frequency ----------------------------------------------------


def _df_graph(names: list[str], total: int | None = None) -> _Graph:
    """A stub graph answering the node count, then the name scan."""
    return _Graph([[{"total": total if total is not None else len(names)}],
                   [{"name": n} for n in names]])


def test_a_token_is_counted_once_per_name_not_once_per_occurrence():
    # This is a document frequency, not a term frequency.
    graph = _df_graph(["rice rice rice husk", "rice bran"])

    counts, total = _manager(graph).token_document_frequency()

    assert counts["rice"] == 2
    assert counts["husk"] == 1
    assert total == 2


def test_the_frequency_is_measured_over_every_name_property():
    # Built from `n.name` alone, a token common in titles but absent from names
    # would get no demotion, and the weighting would silently favour it.
    graph = _df_graph(["x"])

    _manager(graph, node_name_properties=("name", "title")).token_document_frequency()

    assert "n.title" in graph.calls[1][0]
    assert "n.name" in graph.calls[1][0]


def test_the_table_is_computed_once_per_process():
    graph = _df_graph(["rice husk"])
    manager = _manager(graph)

    manager.token_document_frequency()
    manager.token_document_frequency()

    assert len(graph.calls) == 2  # the count and the scan, not four queries


def test_a_cache_matching_the_graph_is_used_instead_of_a_scan(tmp_path):
    path = tmp_path / "df.json"
    path.write_text(
        json.dumps({"node_count": 2, "properties": ["name"], "token_df": {"rice": 2}}),
        encoding="utf-8",
    )
    graph = _Graph([[{"total": 2}]])

    counts, total = _manager(
        graph, node_name_properties=("name",)
    ).token_document_frequency(cache_path=path)

    assert (counts, total) == ({"rice": 2}, 2)
    assert len(graph.calls) == 1  # the count only: no scan


def test_a_cache_from_a_graph_of_a_different_size_is_recomputed(tmp_path, caplog):
    path = tmp_path / "df.json"
    path.write_text(
        json.dumps({"node_count": 99, "properties": ["name"], "token_df": {"rice": 2}}),
        encoding="utf-8",
    )
    graph = _df_graph(["rice husk"], total=1)

    with caplog.at_level(logging.INFO):
        counts, _ = _manager(
            graph, node_name_properties=("name",)
        ).token_document_frequency(cache_path=path)

    assert "stale" in caplog.text
    assert counts == {"rice": 1, "husk": 1}


def test_a_cache_built_over_other_properties_is_recomputed(tmp_path):
    # The property list is part of the key: a cache built from `n.name` alone
    # counts different things than one built from all of them, at the same
    # node count.
    path = tmp_path / "df.json"
    path.write_text(
        json.dumps({"node_count": 1, "properties": ["name"], "token_df": {"rice": 1}}),
        encoding="utf-8",
    )
    graph = _df_graph(["rice husk"], total=1)

    counts, _ = _manager(
        graph, node_name_properties=("name", "title")
    ).token_document_frequency(cache_path=path)

    assert counts == {"rice": 1, "husk": 1}


def test_an_unreadable_cache_is_recomputed_rather_than_fatal(tmp_path, caplog):
    path = tmp_path / "df.json"
    path.write_text("{ not json", encoding="utf-8")
    graph = _df_graph(["rice husk"], total=1)

    with caplog.at_level(logging.WARNING):
        counts, _ = _manager(
            graph, node_name_properties=("name",)
        ).token_document_frequency(cache_path=path)

    assert "unreadable token DF cache" in caplog.text
    assert counts == {"rice": 1, "husk": 1}


def test_the_table_is_written_back_to_the_cache(tmp_path):
    path = tmp_path / "nested" / "df.json"
    graph = _df_graph(["rice husk"], total=1)

    _manager(graph, node_name_properties=("name",)).token_document_frequency(
        cache_path=path
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["node_count"] == 1
    assert payload["properties"] == ["name"]
    assert payload["token_df"] == {"rice": 1, "husk": 1}


def test_an_unwritable_cache_does_not_lose_the_table(tmp_path, caplog):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")
    graph = _df_graph(["rice husk"], total=1)

    with caplog.at_level(logging.WARNING):
        counts, _ = _manager(
            graph, node_name_properties=("name",)
        ).token_document_frequency(cache_path=blocked / "sub" / "df.json")

    assert counts == {"rice": 1, "husk": 1}
    assert "could not write token DF cache" in caplog.text


def test_no_cache_path_means_no_file_is_touched(tmp_path):
    graph = _df_graph(["rice husk"], total=1)

    counts, _ = _manager(graph).token_document_frequency(cache_path=None)

    assert counts
    assert list(tmp_path.iterdir()) == []


def test_an_empty_graph_reports_no_tokens():
    graph = _Graph([[{"total": 0}], []])

    assert _manager(graph).token_document_frequency() == ({}, 0)


# --- identifiers and element ids -------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("uses", "USES"),
        ("part of", "PART_OF"),
        ("a--b", "A_B"),
        ("  __weird__  ", "WEIRD"),
        ("2024report", "_2024REPORT"),
    ],
)
def test_a_relationship_type_is_sanitised_before_it_reaches_a_query(value, expected):
    # It is interpolated into the query text, so anything that is not an
    # identifier is a syntax error. Note this one upper-cases, unlike the label
    # sanitiser in the ingestion stage: these are relationship types.
    assert KnowledgeGraphManager._safe_identifier(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "!!!", "___"])
def test_a_type_that_survives_as_nothing_becomes_the_fallback(value):
    assert KnowledgeGraphManager._safe_identifier(value) == "RELATED_TO"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("4:c9865134-1234-5678-9abc-def012345678:42", True),
        ("Rice husk", False),
        ("", False),
        ("4:notauuid:1", False),
    ],
)
def test_an_element_id_is_told_apart_from_a_name(value, expected):
    # Passing an id where a name is expected asks the graph "which node names
    # contain this uuid?", which is a full scan for nothing.
    assert KnowledgeGraphManager.is_element_id(value) is expected


def test_the_connection_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("NEO4J_URL", "bolt://localhost:7689")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("NEO4J_DATABASE", "staging")
    monkeypatch.setattr(KnowledgeGraphManager, "_build_graph", lambda self: _Graph())

    manager = KnowledgeGraphManager.from_env()

    assert manager.config.url == "bolt://localhost:7689"
    assert manager.config.database == "staging"


def test_a_missing_connection_variable_is_reported(monkeypatch):
    for name in ("NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(KeyError):
        KnowledgeGraphManager.from_env()


# --- rendering rows --------------------------------------------------------


def test_a_row_becomes_a_node():
    node = KnowledgeGraphManager._row_to_node(
        {"text": "Rice husk", "labels": ["Material"], "score": 1.5}
    )

    assert node["text"] == "Rice husk"


def test_a_row_becomes_a_triple():
    triple = KnowledgeGraphManager._row_to_triple(
        {"subject": "Rice husk", "predicate": "USES", "object": "Substrate"}
    )

    assert (triple["subject"], triple["predicate"], triple["object"]) == (
        "Rice husk",
        "USES",
        "Substrate",
    )


def test_clearing_the_graph_is_one_query():
    graph = _Graph([[]])

    _manager(graph).clear()

    assert "DETACH DELETE" in graph.calls[0][0]
