"""The one resolver that says which graph a process is about to write to.

Ports 7688 and 7689 are both live instances and the hosted graph serves the
demo, so "which target did this script pick, and from which variable" is the
question worth pinning, and this module is the one place that answers it.
"""

from __future__ import annotations

import pytest

from kg_pipeline.utils import neo4j_env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Unset every Neo4j variable the resolver reads."""
    for name in (
        neo4j_env.URI_VARS
        + neo4j_env.USER_VARS
        + neo4j_env.PASSWORD_VARS
        + neo4j_env.DATABASE_VARS
    ):
        monkeypatch.delenv(name, raising=False)


def _set(monkeypatch, **values):
    """Set the given environment variables."""
    for name, value in values.items():
        monkeypatch.setenv(name, value)


# --- every spelling in use is understood -----------------------------------


@pytest.mark.parametrize("uri_var", ["NEO4J_URI", "NEO4J_URL"])
@pytest.mark.parametrize("user_var", ["NEO4J_USER", "NEO4J_USERNAME"])
def test_either_spelling_of_a_variable_is_read(monkeypatch, uri_var, user_var):
    _set(
        monkeypatch,
        **{uri_var: "bolt://localhost:7689", user_var: "neo4j", "NEO4J_PASSWORD": "pw"},
    )

    target = neo4j_env.resolve_target()

    assert (target.uri, target.user, target.password) == (
        "bolt://localhost:7689",
        "neo4j",
        "pw",
    )


@pytest.mark.parametrize("db_var", ["NEO4J_DATABASE", "NEO4J_DB"])
def test_either_spelling_of_the_database_is_read(monkeypatch, db_var):
    _set(
        monkeypatch,
        NEO4J_URI="bolt://localhost:7689",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw",
        **{db_var: "staging"},
    )

    assert neo4j_env.resolve_target().database == "staging"


def test_the_first_spelling_wins_when_both_are_set(monkeypatch):
    _set(
        monkeypatch,
        NEO4J_URI="bolt://localhost:7689",
        NEO4J_URL="neo4j+s://aura.example",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw",
    )

    assert neo4j_env.resolve_target().uri == "bolt://localhost:7689"


def test_a_variable_set_to_whitespace_counts_as_unset(monkeypatch):
    _set(
        monkeypatch,
        NEO4J_URI="   ",
        NEO4J_URL="bolt://localhost:7689",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw",
    )

    assert neo4j_env.resolve_target().uri == "bolt://localhost:7689"


def test_surrounding_whitespace_is_trimmed(monkeypatch):
    _set(
        monkeypatch,
        NEO4J_URI="  bolt://localhost:7689\n",
        NEO4J_USER=" neo4j ",
        NEO4J_PASSWORD="pw",
    )

    target = neo4j_env.resolve_target()

    assert (target.uri, target.user) == ("bolt://localhost:7689", "neo4j")


def test_no_database_means_the_servers_default_not_an_empty_name(monkeypatch):
    _set(
        monkeypatch,
        NEO4J_URI="bolt://localhost:7689",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw",
    )

    target = neo4j_env.resolve_target()

    assert target.database is None
    assert target.session_kwargs() == {}


def test_a_named_database_reaches_the_session(monkeypatch):
    _set(
        monkeypatch,
        NEO4J_URI="bolt://localhost:7689",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw",
        NEO4J_DATABASE="neo4j",
    )

    assert neo4j_env.resolve_target().session_kwargs() == {"database": "neo4j"}


# --- what a missing setting says -------------------------------------------


def test_a_missing_setting_names_every_variable_that_would_have_worked(monkeypatch):
    _set(monkeypatch, NEO4J_PASSWORD="pw")

    with pytest.raises(ValueError) as excinfo:
        neo4j_env.resolve_target()

    message = str(excinfo.value)
    assert "NEO4J_URI or NEO4J_URL" in message
    assert "NEO4J_USER or NEO4J_USERNAME" in message
    assert "NEO4J_PASSWORD" not in message  # that one was set


def test_a_missing_password_is_reported_rather_than_sent_as_an_empty_string(monkeypatch):
    # Defaulting to `""` would hand an empty password to the driver, and the
    # failure would arrive as an authentication error from the server.
    _set(monkeypatch, NEO4J_URI="bolt://localhost:7689", NEO4J_USER="neo4j")

    with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
        neo4j_env.resolve_target()


def test_a_report_only_caller_can_ask_without_raising(monkeypatch):
    target = neo4j_env.resolve_target(require=False)

    assert target.uri == ""
    assert target.database is None


# --- explicit arguments beat the environment -------------------------------


def test_an_explicit_argument_overrides_the_environment(monkeypatch):
    _set(
        monkeypatch,
        NEO4J_URI="neo4j+s://aura.example",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="from-env",
    )

    target = neo4j_env.resolve_target(uri="bolt://localhost:7689", password="from-flag")

    assert target.uri == "bolt://localhost:7689"
    assert target.password == "from-flag"
    assert target.user == "neo4j"  # not overridden, still from the environment


def test_an_empty_argument_falls_back_rather_than_blanking_the_setting(monkeypatch):
    # argparse hands `""` for a flag that was declared but not passed.
    _set(
        monkeypatch,
        NEO4J_URI="bolt://localhost:7689",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw",
    )

    assert neo4j_env.resolve_target(uri="", database="").uri == "bolt://localhost:7689"


# --- knowing where you are pointing ----------------------------------------


@pytest.mark.parametrize(
    "uri, host, local",
    [
        ("bolt://localhost:7689", "localhost", True),
        ("bolt://localhost:7688", "localhost", True),
        ("bolt://127.0.0.1:7687", "127.0.0.1", True),
        ("neo4j+s://588fe1bc.databases.neo4j.io", "588fe1bc.databases.neo4j.io", False),
        ("neo4j://kg.example.org:7687", "kg.example.org", False),
        ("localhost:7687", "localhost", True),
    ],
)
def test_a_target_knows_its_host_and_whether_it_is_local(uri, host, local):
    target = neo4j_env.Neo4jTarget(uri=uri, user="neo4j", password="pw", database=None)

    assert target.host == host
    assert target.is_local is local


def test_the_two_local_ports_are_different_targets_with_the_same_host():
    # 7688 and 7689 are both live: "it is localhost" is never the check.
    a = neo4j_env.Neo4jTarget("bolt://localhost:7688", "neo4j", "pw", None)
    b = neo4j_env.Neo4jTarget("bolt://localhost:7689", "neo4j", "pw", None)

    assert a.host == b.host
    assert a.uri != b.uri
    assert a != b


def test_describing_a_target_never_prints_the_password():
    # A literal that is obviously not a real credential: this file is tracked,
    # and a test fixture is one more place a rotation would have to find.
    secret = "not-the-real-password"
    target = neo4j_env.Neo4jTarget("bolt://localhost:7689", "neo4j", secret, "neo4j")

    described = target.describe()

    assert secret not in described
    assert "bolt://localhost:7689" in described
    assert "neo4j" in described


def test_a_target_without_a_database_says_so_when_described():
    target = neo4j_env.Neo4jTarget("bolt://localhost:7689", "neo4j", "pw", None)

    assert "<default>" in target.describe()


def test_the_auth_pair_is_what_the_driver_expects():
    target = neo4j_env.Neo4jTarget("bolt://localhost:7689", "neo4j", "pw", None)

    assert target.auth == ("neo4j", "pw")


# --- the driver factory ----------------------------------------------------


def test_connecting_passes_the_resolved_target_to_the_driver(monkeypatch):
    seen: dict[str, object] = {}

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri, auth=None, **kwargs):
            seen.update({"uri": uri, "auth": auth})
            return "driver"

    monkeypatch.setattr(neo4j_env, "GraphDatabase", _FakeGraphDatabase)
    _set(
        monkeypatch,
        NEO4J_URI="bolt://localhost:7689",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw",
    )

    assert neo4j_env.connect() == "driver"
    assert seen == {"uri": "bolt://localhost:7689", "auth": ("neo4j", "pw")}


def test_connecting_with_a_prepared_target_does_not_touch_the_environment(monkeypatch):
    seen: dict[str, object] = {}

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri, auth=None, **kwargs):
            seen.update({"uri": uri, "auth": auth})
            return "driver"

    monkeypatch.setattr(neo4j_env, "GraphDatabase", _FakeGraphDatabase)
    target = neo4j_env.Neo4jTarget("bolt://localhost:7688", "neo4j", "pw", None)

    neo4j_env.connect(target)

    assert seen["uri"] == "bolt://localhost:7688"


def test_connecting_without_settings_refuses_before_reaching_the_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("no driver should be built")

    class _FakeGraphDatabase:
        driver = staticmethod(_boom)

    monkeypatch.setattr(neo4j_env, "GraphDatabase", _FakeGraphDatabase)

    with pytest.raises(ValueError, match="Missing Neo4j connection settings"):
        neo4j_env.connect()
