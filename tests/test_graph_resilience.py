"""Two ways a failing graph must not pass for a working one.

The driver's own retry window is the first: at its 30 s default, every query
in a retrieval burns the window independently and the manager's retry loop
multiplies it, so one unreachable graph costs minutes of waiting.

The postprocess pass is the second: every step collects its failures into an
`errors` list, and `main()` must not exit 0 unless that list is empty.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphrag.kg import manager as manager_module
from kg_pipeline.stages.neo4j_postprocess import _collect_errors


class _Recorder:
    """Stands in for Neo4jGraph and keeps the kwargs it was built with."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs


@pytest.fixture
def built(monkeypatch) -> dict:
    """The arguments `_build_graph` passes to a recorded Neo4jGraph."""
    monkeypatch.setattr(manager_module, "Neo4jGraph", _Recorder)
    bare = manager_module.KnowledgeGraphManager.__new__(manager_module.KnowledgeGraphManager)
    bare.config = manager_module.KGConfig(
        url="bolt://localhost:7687", username="neo4j", password="x", database="neo4j"
    )
    bare._build_graph()
    return _Recorder.last_kwargs


def test_the_retry_window_is_shorter_than_the_drivers_default(built) -> None:
    """At 30 s per query per retry, a failover takes minutes."""
    assert built["driver_config"]["max_transaction_retry_time"] < 30.0


def test_connecting_gives_up_before_a_person_does(built) -> None:
    assert built["driver_config"]["connection_timeout"] <= 10.0


def test_a_query_may_run_well_past_the_slowest_one_measured(built) -> None:
    """Most queries finish well under a second, but a slow one can take ~24 s.

    The cap has to clear those, or it turns a slow answer into no answer.
    """
    assert built["timeout"] > 24.3


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "2.5", 2.5),
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "0", 8.0),
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "-1", 8.0),
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "abc", 8.0),
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "", 8.0),
    ],
)
def test_an_unusable_override_falls_back_instead_of_crashing(
    monkeypatch, name: str, value: str, expected: float
) -> None:
    """An operator typo must not stop the demo from connecting at all."""
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(manager_module, "Neo4jGraph", _Recorder)
    bare = manager_module.KnowledgeGraphManager.__new__(manager_module.KnowledgeGraphManager)
    bare.config = manager_module.KGConfig(
        url="bolt://localhost:7687", username="neo4j", password="x", database="neo4j"
    )
    bare._build_graph()
    assert _Recorder.last_kwargs["driver_config"]["max_transaction_retry_time"] == expected


def test_errors_are_found_however_deep_the_report_nests_them() -> None:
    report = {
        "step1": {"renamed": 3, "errors": ["rename A -> B: apoc missing"]},
        "aura": {"sub": [{"errors": []}, {"errors": ["merge failed"]}]},
        "step5": {"created": ["c"], "errors": []},
    }
    assert _collect_errors(report) == ["rename A -> B: apoc missing", "merge failed"]


def test_a_clean_report_reports_nothing() -> None:
    assert _collect_errors({"step1": {"errors": []}, "step2": {"renamed": 2}}) == []


class _Flaky:
    """A graph that fails a fixed number of times, then answers."""

    def __init__(self, error: BaseException | None, failures: int = 99) -> None:
        self.error = error
        self.remaining = failures
        self.calls = 0

    def query(self, cypher, params=None):  # noqa: ANN001 - test double
        self.calls += 1
        if self.error is not None and self.remaining > 0:
            self.remaining -= 1
            raise self.error
        return [{"ok": 1}]


def _manager(graph: _Flaky) -> manager_module.KnowledgeGraphManager:
    """A `KnowledgeGraphManager` over `graph`, two attempts, no backoff."""
    bare = manager_module.KnowledgeGraphManager.__new__(manager_module.KnowledgeGraphManager)
    bare.graph = graph
    bare.query_retry_attempts = 2
    bare.query_retry_backoff_sec = 0.0
    bare._outage_until = 0.0
    bare._outage_error = None
    bare._reconnect = lambda: None  # type: ignore[method-assign]
    return bare


def _service_unavailable() -> BaseException:
    """The driver's error for an unreachable server."""
    from neo4j.exceptions import ServiceUnavailable

    return ServiceUnavailable("Couldn't connect to 127.0.0.1:7687")


def test_an_unreachable_graph_is_established_once_not_per_query() -> None:
    """A retrieval issues several queries; only the first pays to find the outage."""
    graph = _Flaky(_service_unavailable())
    kg = _manager(graph)
    for _ in range(4):
        with pytest.raises(Exception):
            kg.run_query("RETURN 1")
    # The first query spends its retries; the next three cost nothing.
    assert graph.calls == 2


def test_the_breaker_forgets(monkeypatch) -> None:
    """It only has to cover the retrieval in progress, not the session."""
    graph = _Flaky(_service_unavailable(), failures=2)
    kg = _manager(graph)
    with pytest.raises(Exception):
        kg.run_query("RETURN 1")
    now = [1000.0]
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now[0])
    kg._outage_until = now[0] + 5.0
    with pytest.raises(Exception):
        kg.run_query("RETURN 1")
    now[0] += 6.0
    assert kg.run_query("RETURN 1") == [{"ok": 1}]


def test_a_server_that_answers_does_not_trip_it() -> None:
    """TransientError means the server asked to be asked again: it is there."""
    from neo4j.exceptions import TransientError

    graph = _Flaky(TransientError("deadlock detected"), failures=1)
    kg = _manager(graph)
    assert kg.run_query("RETURN 1") == [{"ok": 1}]
    assert kg._outage_error is None


def test_a_query_error_does_not_trip_it() -> None:
    """A bad Cypher statement is not an outage, and must not silence the next."""
    graph = _Flaky(ValueError("Invalid input 'X'"))
    kg = _manager(graph)
    with pytest.raises(ValueError):
        kg.run_query("RETURN 1")
    assert kg._outage_error is None
    assert graph.calls == 1


class _SeedSession:
    """Records every statement; the exact seed query always returns nothing."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def query(self, cypher, params=None, *a, **k):  # noqa: ANN001 - test double
        self.statements.append(" ".join(str(cypher).split()))
        return []

    def fallbacks(self) -> list[str]:
        return [s for s in self.statements if "CONTAINS" in s]


class _Result:
    """Driver result over fixed rows."""

    def __init__(self, rows): self._rows = rows
    def data(self): return self._rows
    def consume(self): return None
    def __iter__(self): return iter(self._rows)


def _store(session: _SeedSession) -> manager_module.KnowledgeGraphManager:
    """A `KnowledgeGraphManager` over the recording session, one attempt."""
    bare = manager_module.KnowledgeGraphManager.__new__(manager_module.KnowledgeGraphManager)
    bare.config = manager_module.KGConfig(
        url="bolt://x", username="u", password="p", database="neo4j"
    )
    bare.graph = session
    bare.query_retry_attempts = 1
    bare.query_retry_backoff_sec = 0.0
    bare._outage_until = 0.0
    bare._outage_error = None
    bare._vector_skips = 0
    bare._fulltext_available = False
    return bare


ELEMENT_ID = "4:c9865134-b1d8-40ef-9861-2c15e0a3a3d1:1234"


@pytest.mark.parametrize("method,kwargs", [
    ("get_neighbors", {"entity": ELEMENT_ID, "limit": 10}),
    ("extract_subgraph", {"entity": ELEMENT_ID, "hops": 1, "limit": 10}),
])
def test_an_id_anchor_never_falls_back_to_a_name_scan(method: str, kwargs: dict) -> None:
    """`_graph_anchors` passes elementIds on purpose.

    Matching by name compares six lowercased properties on every node. The
    exact query respects that with id_only; a fallback that drops it asks which
    node *names* contain a UUID, which is a full scan that can only answer no.
    """
    session = _SeedSession()
    getattr(_store(session), method)(**kwargs)
    assert not session.fallbacks(), (
        f"{method} issued a CONTAINS scan for an elementId: {session.fallbacks()[0][:120]}"
    )


@pytest.mark.parametrize("method,kwargs", [
    ("get_neighbors", {"entity": "biochar", "limit": 10}),
    ("extract_subgraph", {"entity": "biochar", "hops": 1, "limit": 10}),
])
def test_a_name_anchor_still_gets_the_fallback(method: str, kwargs: dict) -> None:
    """With a name there is something to broaden, and that is what it is for."""
    session = _SeedSession()
    getattr(_store(session), method)(**kwargs)
    assert session.fallbacks(), f"{method} lost the fallback for a name anchor"


def test_the_destructive_pass_can_be_pointed_somewhere_else(tmp_path, monkeypatch) -> None:
    """An exported NEO4J_URL must win over the env file in neo4j_postprocess.

    With override=True, an operator who exports NEO4J_URL to reach staging is
    silently returned to the file's value -- the demo's live Aura instance --
    by the one pass that merges, relabels and deletes. Two Neo4j instances run
    on this host, so "localhost" does not identify the target either.
    """
    import inspect
    from kg_pipeline.stages import neo4j_postprocess

    source = inspect.getsource(neo4j_postprocess.main)
    calls = re.findall(r"load_dotenv\([^)]*\)", source)
    assert calls, "main no longer loads an env file; this guard needs rewriting"
    for call in calls:
        assert "override=True" not in call, (
            f"{call}: the env file must not override an exported NEO4J_URL — "
            "that is how a cleanup lands on the wrong graph"
        )
        assert "override=False" in call, f"{call}: state the override explicitly"
