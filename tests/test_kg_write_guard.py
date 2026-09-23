"""A repair pass must not write to the wrong graph because --yes was habit.

The repair passes take no arguments and read their target from
`kg_pipeline/.env`, which points at the hosted instance the demo serves, so a
careless run can cost the demo graph vector carriers, entities and whole
relationship types. The confirmation covers the action; this covers the
target: `--yes` says "do it", and on a remote instance it does not say "to
that one".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "kg"))

import write_guard  # noqa: E402

_LOCAL = "bolt://localhost:7689"
_HOSTED = "neo4j+s://588fe1bc.databases.neo4j.io"


def _run(uri: str, argv: list[str]) -> None:
    """Ask `write_guard` to confirm a repair pass against `uri` with `argv`."""
    write_guard.require_confirmation(
        title="KG Repair 3",
        what_it_does="Deletes orphan nodes.",
        uri=uri,
        database=None,
        argv=argv,
    )


def test_no_confirmation_prints_the_plan_and_writes_nothing(capsys):
    with pytest.raises(SystemExit) as exit_info:
        _run(_LOCAL, [])

    # Exit 0: being asked for a plan and getting one is a success.
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "Nothing has been written" in out
    assert _LOCAL in out


def test_help_never_runs_the_pass(capsys):
    # A script with no argument parser would otherwise run on `--help`.
    with pytest.raises(SystemExit) as exit_info:
        _run(_LOCAL, ["--help", "--yes"])

    assert exit_info.value.code == 0
    assert "help wins" in capsys.readouterr().out


def test_a_local_target_runs_on_yes_alone():
    _run(_LOCAL, ["--yes"])  # returns, does not raise


def test_a_hosted_target_refuses_yes_alone(capsys):
    with pytest.raises(SystemExit) as exit_info:
        _run(_HOSTED, ["--yes"])

    # Exit 1: this one is a refusal, not a plan.
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "refusing to write to a hosted graph" in out
    # It has to say what to type, and name the host it wants named.
    assert "588fe1bc.databases.neo4j.io" in out


def test_a_hosted_target_runs_when_its_host_is_named(monkeypatch):
    monkeypatch.setenv("KG_ALLOW_HOSTED_WRITES", "588fe1bc.databases.neo4j.io")

    _run(_HOSTED, ["--yes"])


def test_naming_a_different_host_does_not_authorise_this_one(capsys, monkeypatch):
    # The failure this exists for: the variable left over from another session.
    monkeypatch.setenv("KG_ALLOW_HOSTED_WRITES", "other.databases.neo4j.io")

    with pytest.raises(SystemExit) as exit_info:
        _run(_HOSTED, ["--yes"])

    assert exit_info.value.code == 1
    assert "which is not" in capsys.readouterr().out


def test_naming_a_host_is_not_a_substitute_for_confirming(monkeypatch, capsys):
    monkeypatch.setenv("KG_ALLOW_HOSTED_WRITES", "588fe1bc.databases.neo4j.io")

    with pytest.raises(SystemExit) as exit_info:
        _run(_HOSTED, [])

    assert exit_info.value.code == 0
    assert "Nothing has been written" in capsys.readouterr().out


@pytest.mark.parametrize(
    "uri, host",
    [
        ("neo4j+s://588fe1bc.databases.neo4j.io", "588fe1bc.databases.neo4j.io"),
        ("bolt://localhost:7689", "localhost"),
        ("neo4j://user:pw@example.org:7687/db", "example.org"),
        ("", ""),
    ],
)
def test_the_host_is_read_out_of_the_uri(uri, host):
    assert write_guard._host_of(uri) == host


@pytest.mark.parametrize("uri", ["bolt://localhost:7689", "bolt://127.0.0.1:7688", "bolt://[::1]:7687"])
def test_local_instances_are_not_hosted(uri):
    assert write_guard._looks_hosted(uri) is False
