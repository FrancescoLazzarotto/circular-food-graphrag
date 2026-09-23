"""Stage 6 writes the graph.

The parts that matter here are the ones that decide what reaches Neo4j and what
is quietly dropped: the identifier sanitiser that turns a model-invented label
into a Cypher token, the property sanitiser that has to flatten anything Neo4j
cannot store, and the batch path that falls back to one triple at a time and
counts only what actually landed.

No Neo4j: the driver is faked. These are about the decisions, not the database.
"""

from __future__ import annotations

import logging

import pytest

from kg_pipeline.models.types import KGTriple
from kg_pipeline.stages import neo4j_ingestion
from kg_pipeline.utils import neo4j_env


def _triple(subject="Rice husk", predicate="USES", obj="substrate", **rel) -> KGTriple:
    """A valid triple between two materials; kwargs join its properties."""
    return KGTriple.model_validate(
        {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "subject_labels": ["Material"],
            "object_labels": ["Material"],
            "subject_properties": {"name": subject},
            "object_properties": {"name": obj},
            "relationship_properties": {"source_doc": "a.pdf", **rel},
        }
    )


# --- identifiers -----------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Material", "Material"),
        ("Food Product", "Food_Product"),
        ("città", "citt"),
        ("a--b", "a_b"),
        ("__weird__", "weird"),
        ("2024 report", "_2024_report"),
    ],
)
def test_a_label_becomes_a_usable_cypher_token(value, expected):
    # These are interpolated straight into the query text, so anything that is
    # not an identifier is a syntax error at best.
    assert neo4j_ingestion._safe_identifier(value, "Concept") == expected


@pytest.mark.parametrize("value", ["", "   ", "!!!", "___"])
def test_a_label_that_survives_as_nothing_falls_back(value):
    assert neo4j_ingestion._safe_identifier(value, "Concept") == "Concept"


# --- properties ------------------------------------------------------------


def test_primitives_go_through_untouched():
    props = {"name": "Rice husk", "count": 3, "score": 0.5, "ok": True}

    assert neo4j_ingestion._sanitize_props(props) == props


def test_a_null_is_dropped_rather_than_written():
    assert neo4j_ingestion._sanitize_props({"a": 1, "b": None}) == {"a": 1}


def test_a_nested_structure_is_flattened_to_something_neo4j_can_store():
    out = neo4j_ingestion._sanitize_props({"meta": {"page": 3, "tags": ["a"]}})

    # Neo4j has no map properties: it is this or the write fails.
    assert isinstance(out["meta"], str)
    assert "page" in out["meta"]


def test_a_mixed_list_becomes_strings_rather_than_failing():
    # Neo4j list properties must be homogeneous.
    out = neo4j_ingestion._sanitize_props({"xs": [1, "two", True]})

    assert isinstance(out["xs"], list)
    assert all(isinstance(x, str) for x in out["xs"])


def test_a_homogeneous_list_keeps_its_type():
    out = neo4j_ingestion._sanitize_props({"xs": [1, 2, 3]})

    assert out["xs"] == [1, 2, 3]


def test_the_predicate_kept_from_the_remapping_reaches_the_edge():
    # `validate_triples` puts the model's own predicate here when it remaps to
    # RELATED_TO, and the retriever reads it back with
    # `coalesce(properties(r)['predicate'], type(r))`.
    triple = _triple(predicate="RELATED_TO")
    triple.relationship_properties["predicate"] = "USED_FOR"
    _, row = neo4j_ingestion._triple_cypher_parts(triple)

    assert row["r_props"]["predicate"] == "USED_FOR"


def test_a_predicate_that_is_not_screaming_snake_case_is_refused_at_the_model():
    # The identifier sanitiser is the last line, not the first: a predicate that
    # would need sanitising does not get past KGTriple.
    with pytest.raises(Exception):
        _triple(predicate="USED FOR")


def test_the_query_uses_the_sanitised_label():
    triple = _triple()
    triple.subject_labels = ["Food Product"]

    query, row = neo4j_ingestion._triple_cypher_parts(triple)

    assert ":Food_Product" in query
    assert "[r:USES" in query
    # Names travel as parameters, never interpolated.
    assert row["s_name"] == "Rice husk"
    assert "Rice husk" not in query


# --- the batch path --------------------------------------------------------


class _FakeSession:
    """Runs the write functions, and can be told which batches to fail."""

    def __init__(self, failing_queries: set[str] | None = None, failing_triples=()):
        self.failing_queries = failing_queries or set()
        self.failing_triples = set(failing_triples)
        self.batches: list[list[dict]] = []
        self.singles: list[KGTriple] = []

    def execute_write(self, fn, *args):
        if fn is neo4j_ingestion._merge_triples_batch:
            query, rows = args
            if query in self.failing_queries:
                raise RuntimeError("batch rejected")
            self.batches.append(rows)
            return None
        triple = args[0]
        if triple.subject in self.failing_triples:
            raise RuntimeError("triple rejected")
        self.singles.append(triple)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeDriver:
    """Driver handing out one session and recording the database asked for."""

    def __init__(self, session: _FakeSession):
        self._session = session

    def session(self, database=None):
        self.database = database
        return self._session

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture()
def fake_driver(monkeypatch):
    """An installer that makes every driver in the repo a fake one."""
    def _install(session: _FakeSession) -> _FakeDriver:
        driver = _FakeDriver(session)
        # Every driver in the repo is built in one place; that is the seam to
        # replace.
        monkeypatch.setattr(
            neo4j_env.GraphDatabase, "driver", lambda *a, **k: driver
        )
        return driver

    return _install


def test_triples_sharing_a_shape_are_sent_as_one_batch(fake_driver):
    session = _FakeSession()
    fake_driver(session)

    written = neo4j_ingestion.ingest_triples(
        [_triple(obj=f"o{i}") for i in range(5)],
        uri="bolt://x", user="u", password="p", batch_size=100,
    )

    assert written == 5
    assert len(session.batches) == 1, "one Cypher shape, one round trip"
    assert len(session.batches[0]) == 5


def test_batches_respect_the_batch_size(fake_driver):
    session = _FakeSession()
    fake_driver(session)

    neo4j_ingestion.ingest_triples(
        [_triple(obj=f"o{i}") for i in range(7)],
        uri="bolt://x", user="u", password="p", batch_size=3,
    )

    assert [len(b) for b in session.batches] == [3, 3, 1]


def test_a_failed_batch_is_retried_one_triple_at_a_time(fake_driver, caplog):
    session = _FakeSession()
    fake_driver(session)
    session.failing_queries = {
        neo4j_ingestion._triple_cypher_parts(_triple())[0]
    }

    with caplog.at_level(logging.WARNING):
        written = neo4j_ingestion.ingest_triples(
            [_triple(obj=f"o{i}") for i in range(3)],
            uri="bolt://x", user="u", password="p", batch_size=10,
        )

    assert written == 3
    assert len(session.singles) == 3
    assert "retrying" in caplog.text


def test_only_the_triples_that_landed_are_counted(fake_driver, caplog):
    session = _FakeSession()
    fake_driver(session)
    session.failing_queries = {neo4j_ingestion._triple_cypher_parts(_triple())[0]}
    session.failing_triples = {"Rice husk"}

    with caplog.at_level(logging.WARNING):
        written = neo4j_ingestion.ingest_triples(
            [_triple(obj=f"o{i}") for i in range(3)],
            uri="bolt://x", user="u", password="p", batch_size=10,
        )

    # Counting a failed batch whose retries were all skipped would report
    # writing triples that are not in the graph.
    assert written == 0
    assert "are NOT in the graph" in caplog.text


def test_an_empty_input_writes_nothing_and_says_zero(fake_driver):
    session = _FakeSession()
    fake_driver(session)

    assert neo4j_ingestion.ingest_triples(
        [], uri="bolt://x", user="u", password="p"
    ) == 0
    assert session.batches == []


def test_the_database_name_is_passed_to_the_session(fake_driver):
    session = _FakeSession()
    driver = fake_driver(session)

    neo4j_ingestion.ingest_triples(
        [_triple()], uri="bolt://x", user="u", password="p", database="neo4j",
    )

    assert driver.database == "neo4j"


# --- environment -----------------------------------------------------------


def test_either_spelling_of_the_env_vars_is_accepted(monkeypatch):
    for name in ("NEO4J_URI", "NEO4J_URL", "NEO4J_USER", "NEO4J_USERNAME",
                 "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEO4J_URL", "bolt://localhost:7689")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")

    assert neo4j_ingestion._resolve_neo4j_env() == (
        "bolt://localhost:7689", "neo4j", "pw", None,
    )


def test_a_missing_credential_names_what_is_missing(monkeypatch):
    for name in ("NEO4J_URI", "NEO4J_URL", "NEO4J_USER", "NEO4J_USERNAME",
                 "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEO4J_URL", "bolt://localhost:7689")

    with pytest.raises(ValueError) as excinfo:
        neo4j_ingestion._resolve_neo4j_env()

    message = str(excinfo.value)
    assert "NEO4J_PASSWORD" in message and "NEO4J_USER" in message
