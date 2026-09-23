"""A cleanup with thousands of things to delete is not the cleanup it claims.

Without the `:NodeVec` and `name IS NOT NULL` guards, the isolated-node
query returns about 14 500 candidates on the demo graph — nearly all of them
the vector carriers that hold the cross-lingual index — and the pass would
delete them. With the guards it returns a few dozen. A cap turns the
interesting middle into a question for a person instead of a DETACH DELETE.

The session is a double that records every statement, so these assert on what
would actually reach the database.
"""

from __future__ import annotations

import pytest

from kg_pipeline.stages.neo4j_postprocess import (
    _cleanup_isolated_nodes,
    _isolated_delete_cap,
)


class _Result:
    """Driver result over fixed rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def data(self) -> list[dict]:
        return self._rows

    def consume(self) -> None:
        return None

    def __iter__(self):
        return iter(self._rows)


class _Session:
    """Answers the two read queries; records everything it is asked to run."""

    def __init__(self, candidates: int) -> None:
        self.statements: list[str] = []
        self._candidates = [
            {"id": i, "name": f"orfano {i}"} for i in range(candidates)
        ]

    def run(self, cypher: str, *args, **kwargs) -> _Result:
        self.statements.append(" ".join(cypher.split()))
        if "NOT (n)--()" in cypher:
            return _Result(self._candidates)
        if "count(r) AS degree" in cypher:
            return _Result([])          # no namesake anywhere: everything is a delete
        return _Result([])

    def deletes(self) -> list[str]:
        return [s for s in self.statements if "DETACH DELETE" in s]


def test_the_default_cap_sits_between_the_two_measured_numbers() -> None:
    """41 candidates with the guards, 14 561 without: the cap separates them."""
    assert 41 < _isolated_delete_cap() < 14561


@pytest.mark.parametrize("raw,expected", [("2000", 2000), ("0", 500), ("-1", 500),
                                          ("abc", 500), ("", 500)])
def test_an_unusable_override_falls_back(monkeypatch, raw: str, expected: int) -> None:
    monkeypatch.setenv("KG_ISOLATED_DELETE_MAX", raw)
    assert _isolated_delete_cap() == expected


def test_a_normal_run_still_deletes() -> None:
    session = _Session(candidates=41)
    report = _cleanup_isolated_nodes(session, dry_run=False, apoc_available=True)
    assert report["candidates"] == 41
    assert report["deleted_nodes"] == 41
    assert session.deletes(), "the pass must still do its job under the cap"
    assert not report["errors"]


def test_a_run_over_the_cap_writes_nothing() -> None:
    """Thousands of candidates, the shape of a missing guard: nothing is deleted."""
    session = _Session(candidates=14561)
    report = _cleanup_isolated_nodes(session, dry_run=False, apoc_available=True)
    assert not session.deletes(), "nothing may reach the database over the cap"
    assert report["errors"], "and the run must not look clean"
    assert "14561" in report["errors"][0]


def test_a_refused_run_still_says_what_it_would_have_done() -> None:
    """Refusing by returning early would leave the operator with no evidence."""
    session = _Session(candidates=14561)
    report = _cleanup_isolated_nodes(session, dry_run=False, apoc_available=True)
    assert report["candidates"] == 14561
    assert report["deleted_nodes"] == 14561
    assert report["samples"], "the sample is what the person reads before raising the cap"


def test_the_cap_does_not_touch_an_explicit_dry_run() -> None:
    session = _Session(candidates=14561)
    report = _cleanup_isolated_nodes(session, dry_run=True, apoc_available=True)
    assert not session.deletes()
    assert not report["errors"], "a dry run over the cap is not a failure, it is a dry run"
