"""Two repair passes that batch their queries instead of one per node or pair.

They are destructive passes over a real graph, so what is pinned here is not
"it is faster" but "it does the same thing": same edges bridged, same nodes
merged and deleted, same per-group cap, in the same order.
"""

from __future__ import annotations

import re
from typing import Any

from kg_pipeline.stages.neo4j_postprocess import (
    _bridge_duplicate_name_groups,
    _cleanup_isolated_nodes,
)


class _Result:
    """Driver result over fixed rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def data(self) -> list[dict[str, Any]]:
        return self._rows

    def consume(self) -> None:
        return None


class _FakeSession:
    """Answers the handful of Cypher shapes these two passes send.

    Matching is on distinctive fragments rather than the whole statement, so a
    reformatted query still routes to the right handler; an unrecognised query
    raises instead of silently returning nothing.
    """

    def __init__(
        self,
        connected_pairs: set[tuple[int, int]] | None = None,
        degrees: list[dict[str, Any]] | None = None,
        isolated: list[dict[str, Any]] | None = None,
    ) -> None:
        self.connected_pairs = connected_pairs or set()
        self.degrees = degrees or []
        self.isolated = isolated or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, cypher: str, **params: Any) -> _Result:
        flat = re.sub(r"\s+", " ", cypher).strip()
        self.calls.append((flat, params))

        if "count(r) AS c" in flat and "UNWIND $pairs" in flat:
            rows = []
            for pair in params["pairs"]:
                key = (pair["a"], pair["b"])
                if key in self.connected_pairs or (key[1], key[0]) in self.connected_pairs:
                    rows.append({"a": pair["a"], "b": pair["b"], "c": 1})
            return _Result(rows)
        if "MERGE (a)-[:RELATED_TO]->(b)" in flat:
            return _Result([])
        if "WHERE NOT (n)--()" in flat:
            return _Result(list(self.isolated))
        if "RETURN id(m) AS id, m.name AS name, degree" in flat:
            return _Result(list(self.degrees))
        if "apoc.refactor.mergeNodes" in flat:
            return _Result([{"id": params["primary"]}])
        if "DETACH DELETE n" in flat:
            return _Result([])
        raise AssertionError(f"unexpected query: {flat}")

    def queries_matching(self, fragment: str) -> list[tuple[str, dict[str, Any]]]:
        return [call for call in self.calls if fragment in call[0]]


def _group(normalized: str, nodes: list[tuple[int, str, int]]) -> dict[str, Any]:
    """A duplicate-name group of (id, name, degree) nodes."""
    return {
        "normalized": normalized,
        "nodes": [
            {"id": node_id, "name": name, "degree": degree}
            for node_id, name, degree in nodes
        ],
    }


def _bridge(monkeypatch, session, groups, *, cap=0, dry_run=False):
    """Run the bridging pass over `groups` with the given cap."""
    monkeypatch.setattr(
        "kg_pipeline.stages.neo4j_postprocess._find_duplicate_groups",
        lambda _session: groups,
    )
    return _bridge_duplicate_name_groups(
        session=session, dry_run=dry_run, max_edges_per_group=cap
    )


# --- bridging duplicate-name groups ---------------------------------------


def test_bridging_creates_one_edge_per_unconnected_pair(monkeypatch):
    session = _FakeSession()
    groups = [_group("riso", [(1, "Riso", 5), (2, "riso", 2), (3, "RISO", 1)])]
    report = _bridge(monkeypatch, session, groups)

    assert report["edges_created"] == 2
    assert report["candidate_pairs"] == 2
    merges = session.queries_matching("MERGE (a)-[:RELATED_TO]->(b)")
    assert len(merges) == 1, "one round-trip, not one per pair"
    assert merges[0][1]["pairs"] == [{"a": 1, "b": 2}, {"a": 1, "b": 3}]


def test_already_connected_pairs_are_not_bridged_again(monkeypatch):
    session = _FakeSession(connected_pairs={(1, 2)})
    groups = [_group("riso", [(1, "Riso", 5), (2, "riso", 2), (3, "RISO", 1)])]
    report = _bridge(monkeypatch, session, groups)

    assert report["skipped_already_connected"] == 1
    assert report["edges_created"] == 1
    assert session.queries_matching("MERGE (a)-[:RELATED_TO]->(b)")[0][1]["pairs"] == [
        {"a": 1, "b": 3}
    ]


def test_the_per_group_cap_counts_created_edges_only(monkeypatch):
    """A pair that was already connected must not consume a slot of the cap."""
    session = _FakeSession(connected_pairs={(1, 2)})
    groups = [
        _group("riso", [(1, "Riso", 9), (2, "riso", 3), (3, "RISO", 2), (4, "rIso", 1)])
    ]
    report = _bridge(monkeypatch, session, groups, cap=2)

    assert report["skipped_already_connected"] == 1
    assert report["edges_created"] == 2
    assert report["skipped_group_limit"] == 0


def test_the_cap_applies_to_each_group_separately(monkeypatch):
    session = _FakeSession()
    groups = [
        _group("riso", [(1, "Riso", 9), (2, "riso", 3), (3, "RISO", 2)]),
        _group("paglia", [(10, "Paglia", 9), (11, "paglia", 3), (12, "PAGLIA", 2)]),
    ]
    report = _bridge(monkeypatch, session, groups, cap=1)

    assert report["edges_created"] == 2
    assert report["skipped_group_limit"] == 2


def test_a_failed_existence_check_does_not_create_duplicate_edges(monkeypatch):
    """Unknown must be treated as connected: a double edge is unrecoverable."""

    class _FailingSession(_FakeSession):
        def run(self, cypher: str, **params: Any):
            if "count(r) AS c" in re.sub(r"\s+", " ", cypher):
                raise RuntimeError("connection reset")
            return super().run(cypher, **params)

    session = _FailingSession()
    groups = [_group("riso", [(1, "Riso", 5), (2, "riso", 2)])]
    report = _bridge(monkeypatch, session, groups)

    assert report["edges_created"] == 0
    assert report["errors"]
    assert not session.queries_matching("MERGE (a)-[:RELATED_TO]->(b)")


def test_dry_run_reports_the_edges_without_writing(monkeypatch):
    session = _FakeSession()
    groups = [_group("riso", [(1, "Riso", 5), (2, "riso", 2)])]
    report = _bridge(monkeypatch, session, groups, dry_run=True)

    assert report["edges_created"] == 1
    assert not session.queries_matching("MERGE (a)-[:RELATED_TO]->(b)")


# --- isolated nodes -------------------------------------------------------


def test_isolated_nodes_with_a_namesake_are_merged_into_the_highest_degree_one():
    session = _FakeSession(
        isolated=[{"id": 100, "name": "Riso"}],
        degrees=[
            {"id": 1, "name": "riso", "degree": 2},
            {"id": 2, "name": "RISO", "degree": 7},
        ],
    )
    report = _cleanup_isolated_nodes(session=session, dry_run=False, apoc_available=True)

    assert report["matched"] == 1
    assert report["deleted_nodes"] == 0
    merge = session.queries_matching("apoc.refactor.mergeNodes")[0]
    assert merge[1]["primary"] == 2, "highest degree wins"


def test_the_lowest_id_breaks_a_degree_tie():
    session = _FakeSession(
        isolated=[{"id": 100, "name": "Riso"}],
        degrees=[
            {"id": 9, "name": "riso", "degree": 4},
            {"id": 4, "name": "RISO", "degree": 4},
        ],
    )
    _cleanup_isolated_nodes(session=session, dry_run=False, apoc_available=True)

    assert session.queries_matching("apoc.refactor.mergeNodes")[0][1]["primary"] == 4


def test_isolated_nodes_without_a_namesake_are_deleted_in_one_query():
    session = _FakeSession(
        isolated=[
            {"id": 100, "name": "Orfano A"},
            {"id": 101, "name": "Orfano B"},
            {"id": 102, "name": ""},
        ],
        degrees=[{"id": 1, "name": "Riso", "degree": 3}],
    )
    report = _cleanup_isolated_nodes(session=session, dry_run=False, apoc_available=True)

    assert report["deleted_nodes"] == 3
    deletes = session.queries_matching("DETACH DELETE n")
    assert len(deletes) == 1, "one round-trip, not one per node"
    assert deletes[0][1]["ids"] == [100, 101, 102]


def test_no_candidates_means_no_further_queries():
    session = _FakeSession(isolated=[])
    report = _cleanup_isolated_nodes(session=session, dry_run=False, apoc_available=True)

    assert report["candidates"] == 0
    assert len(session.calls) == 1


def test_dry_run_neither_merges_nor_deletes():
    session = _FakeSession(
        isolated=[{"id": 100, "name": "Riso"}, {"id": 101, "name": "Orfano"}],
        degrees=[{"id": 1, "name": "riso", "degree": 3}],
    )
    report = _cleanup_isolated_nodes(session=session, dry_run=True, apoc_available=True)

    assert report["matched"] == 1
    assert report["deleted_nodes"] == 1
    assert not session.queries_matching("apoc.refactor.mergeNodes")
    assert not session.queries_matching("DETACH DELETE n")


def test_without_apoc_the_merge_is_skipped_and_reported():
    session = _FakeSession(
        isolated=[{"id": 100, "name": "Riso"}],
        degrees=[{"id": 1, "name": "riso", "degree": 3}],
    )
    report = _cleanup_isolated_nodes(session=session, dry_run=False, apoc_available=False)

    assert report["skipped"] == 1
    assert report["errors"]
    assert not session.queries_matching("apoc.refactor.mergeNodes")
