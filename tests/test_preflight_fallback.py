"""The preflight must not refuse to launch in the outage the fallback is for.

Aura Free suspends itself after three idle days; the demo falls back to the
local mirror, and `start_demo.sh` exits 1 when the preflight fails, so a
preflight that checks only the primary stops a demo the mirror could serve.

The probe itself is stubbed here — whether a given graph answers is what the
live checks cover — so these exercise the decision made on top of it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "smoke_check", ROOT / "scripts" / "smoke" / "smoke_check.py"
)
smoke_check = importlib.util.module_from_spec(_spec)
sys.modules["smoke_check"] = smoke_check
_spec.loader.exec_module(smoke_check)

PRIMARY = {
    "NEO4J_URL": "neo4j+s://primary.example",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "x",
    "NEO4J_DATABASE": "primarydb",
}
FALLBACK = {
    "DEMO_NEO4J_FALLBACK_URL": "bolt://localhost:7689",
    "DEMO_NEO4J_FALLBACK_USERNAME": "neo4j",
    "DEMO_NEO4J_FALLBACK_PASSWORD": "y",
}


@pytest.fixture
def env(monkeypatch):
    """Unset every primary and fallback Neo4j variable."""
    for key in list(PRIMARY) + list(FALLBACK) + ["DEMO_NEO4J_FALLBACK_DATABASE"]:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _probe(results: dict[str, tuple[bool, str]]):
    """Answer per uri, and record what each probe was called with."""
    seen: list[tuple[str, str, str, str]] = []

    def probe(uri, username, password, database):
        seen.append((uri, username, password, database))
        return results.get(uri, (False, f"unreachable: {uri}"))

    probe.seen = seen  # type: ignore[attr-defined]
    return probe


def test_a_healthy_primary_is_used_and_named(env, monkeypatch) -> None:
    for key, value in {**PRIMARY, **FALLBACK}.items():
        env.setenv(key, value)
    monkeypatch.setattr(
        smoke_check, "_probe_graph", _probe({PRIMARY["NEO4J_URL"]: (True, "29040 nodes")})
    )
    ok, detail = smoke_check._check_neo4j_connectivity()
    assert ok and detail.startswith("primary ")


def test_a_suspended_primary_passes_on_the_fallback(env, monkeypatch) -> None:
    """The whole point: the launch must not stop here."""
    for key, value in {**PRIMARY, **FALLBACK}.items():
        env.setenv(key, value)
    monkeypatch.setattr(
        smoke_check,
        "_probe_graph",
        _probe({FALLBACK["DEMO_NEO4J_FALLBACK_URL"]: (True, "29040 nodes")}),
    )
    ok, detail = smoke_check._check_neo4j_connectivity()
    assert ok
    assert "PRIMARY DOWN" in detail, "passing on the mirror must not read as healthy"
    assert "fallback" in detail


def test_both_graphs_down_still_fails(env, monkeypatch) -> None:
    for key, value in {**PRIMARY, **FALLBACK}.items():
        env.setenv(key, value)
    monkeypatch.setattr(smoke_check, "_probe_graph", _probe({}))
    ok, detail = smoke_check._check_neo4j_connectivity()
    assert not ok
    assert "primary" in detail and "fallback" in detail


def test_no_fallback_configured_says_which_variables(env, monkeypatch) -> None:
    for key, value in PRIMARY.items():
        env.setenv(key, value)
    monkeypatch.setattr(smoke_check, "_probe_graph", _probe({}))
    ok, detail = smoke_check._check_neo4j_connectivity()
    assert not ok
    assert "DEMO_NEO4J_FALLBACK_URL" in detail


def test_missing_primary_variables_do_not_hide_a_working_fallback(env, monkeypatch) -> None:
    """A half-configured primary is a reason to try the mirror, not to stop."""
    for key, value in FALLBACK.items():
        env.setenv(key, value)
    monkeypatch.setattr(
        smoke_check,
        "_probe_graph",
        _probe({FALLBACK["DEMO_NEO4J_FALLBACK_URL"]: (True, "29040 nodes")}),
    )
    ok, detail = smoke_check._check_neo4j_connectivity()
    assert ok
    assert "NEO4J_URL" in detail, "the missing variables still have to be named"


def test_the_fallback_database_does_not_inherit_auras_name(env, monkeypatch) -> None:
    """Unset is not 'no database': the driver would read NEO4J_DATABASE and the
    local mirror would be probed under Aura's database name, which it lacks."""
    for key, value in {**PRIMARY, **FALLBACK}.items():
        env.setenv(key, value)
    probe = _probe({FALLBACK["DEMO_NEO4J_FALLBACK_URL"]: (True, "ok")})
    monkeypatch.setattr(smoke_check, "_probe_graph", probe)
    smoke_check._check_neo4j_connectivity()
    assert probe.seen[-1][3] == "neo4j"
    assert probe.seen[0][3] == "primarydb"
