"""The checks that run after the graph has been written, and the per-triple fallback.

`run_quality_checks` is what tells you whether an ingestion went well —
out-of-vocabulary predicates, duplicate names, nodes with no neighbours.
`_merge_triple` is the other half: the fallback that writes one triple at a
time when a batch fails, so it is the path that runs precisely when something
has already gone wrong.

No Neo4j: the driver is faked and records the queries it was asked.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from neo4j.exceptions import CypherTypeError

from kg_pipeline.models.types import KGTriple
from kg_pipeline.stages import neo4j_ingestion
from kg_pipeline.utils import neo4j_env


def _triple(subject="Rice husk", predicate="USES", obj="Substrate", **rel) -> KGTriple:
    """A valid triple between two nodes; kwargs join its properties."""
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


class _Result:
    """Driver result over fixed rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def data(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def consume(self) -> None:
        return None


class _Session:
    """Session answering by query fragment and recording every call."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, cypher: str, **params: Any):
        flat = " ".join(cypher.split())
        self.calls.append((flat, params))
        for fragment, rows in self.answers.items():
            if fragment in flat:
                if isinstance(rows, BaseException):
                    raise rows
                return _Result(rows)
        return _Result([])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Driver:
    """Driver handing out one session and recording the database asked for."""

    def __init__(self, session: _Session) -> None:
        self._session = session
        self.databases: list[Any] = []

    def session(self, database=None, **kwargs: Any):
        self.databases.append(database)
        return self._session

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def checks(monkeypatch, tmp_path):
    """Runs `run_quality_checks` against a faked graph, returns (session, report)."""

    def _run(answers=None, vocab=None, database=None, report_name="report.json"):
        session = _Session(answers)
        monkeypatch.setattr(neo4j_env, "connect", lambda target: _Driver(session))
        path = tmp_path / report_name
        neo4j_ingestion.run_quality_checks(
            uri="bolt://localhost:7689",
            user="neo4j",
            password="pw",
            report_path=path,
            database=database,
            relation_vocab=vocab,
        )
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return session, report, path

    return _run


# --- which checks run ------------------------------------------------------


def test_the_three_standing_checks_always_run(checks):
    _, report, _ = checks()

    assert set(report["queries"]) == {
        "duplicate_nodes_by_name",
        "sparsely_connected_nodes",
    }


def test_the_vocabulary_check_runs_only_when_a_vocabulary_is_given(checks):
    _, with_vocab, _ = checks(vocab=["USES", "PART_OF"])
    _, without, _ = checks(report_name="senza.json")

    assert "predicates_out_of_vocab" in with_vocab["queries"]
    assert "predicates_out_of_vocab" not in without["queries"]


def test_the_system_predicates_are_always_allowed(checks):
    # SAME_AS and MENTIONED_IN are written by the pipeline itself; flagging
    # them as out of vocabulary would report the pipeline to itself.
    session, _, _ = checks(vocab=["USES"])

    allowed = next(
        params["allowed_predicates"]
        for _, params in session.calls
        if "allowed_predicates" in params
    )
    assert "SAME_AS" in allowed and "MENTIONED_IN" in allowed


def test_the_vocabulary_is_normalised_before_it_is_compared(checks):
    session, _, _ = checks(vocab=["uses", "  part_of  ", ""])

    allowed = next(
        params["allowed_predicates"]
        for _, params in session.calls
        if "allowed_predicates" in params
    )
    assert "USES" in allowed and "PART_OF" in allowed
    assert "" not in allowed


def test_an_empty_vocabulary_is_the_same_as_none(checks):
    _, report, _ = checks(vocab=[])

    assert "predicates_out_of_vocab" not in report["queries"]


# --- what the report says --------------------------------------------------


def test_what_the_graph_answered_reaches_the_report(checks):
    _, report, _ = checks(
        answers={
            "WHERE c > 1": [{"name": "Scotta", "labelSets": [["Concept"]], "c": 3}],
            "size((n)--()) <= 1": [{"labels": ["Concept"], "name": "Orfano"}],
        }
    )

    assert report["queries"]["duplicate_nodes_by_name"][0]["c"] == 3
    assert report["queries"]["sparsely_connected_nodes"][0]["name"] == "Orfano"


def test_an_out_of_vocabulary_predicate_is_reported_with_its_count(checks):
    _, report, _ = checks(
        answers={"NOT type(r) IN $allowed_predicates": [{"outOfVocab": "MITIGATES", "n": 12}]},
        vocab=["USES"],
    )

    assert report["queries"]["predicates_out_of_vocab"] == [
        {"outOfVocab": "MITIGATES", "n": 12}
    ]


def test_a_check_that_fails_is_recorded_rather_than_stopping_the_rest(checks):
    # One broken query must not cost the other checks: the report is what
    # somebody reads after an ingestion, and a missing section reads as "fine".
    _, report, _ = checks(answers={"WHERE c > 1": RuntimeError("index missing")})

    assert report["queries"]["duplicate_nodes_by_name"] == {"error": "index missing"}
    assert "sparsely_connected_nodes" in report["queries"]


def test_the_report_is_written_where_it_was_asked_for(checks, tmp_path):
    _, _, path = checks()

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["queries"]


def test_a_missing_directory_is_created(monkeypatch, tmp_path):
    session = _Session()
    monkeypatch.setattr(neo4j_env, "connect", lambda target: _Driver(session))
    path = tmp_path / "nested" / "deeper" / "report.json"

    neo4j_ingestion.run_quality_checks(
        uri="bolt://x", user="neo4j", password="pw", report_path=path
    )

    assert path.exists()


def test_an_unwritable_report_does_not_raise(monkeypatch, tmp_path):
    # Best-effort by design: the checks have already run and losing the file
    # must not turn a completed ingestion into a crash.
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")
    session = _Session()
    monkeypatch.setattr(neo4j_env, "connect", lambda target: _Driver(session))
    monkeypatch.chdir(tmp_path)

    neo4j_ingestion.run_quality_checks(
        uri="bolt://x",
        user="neo4j",
        password="pw",
        report_path=blocked / "sub" / "report.json",
    )

    assert (tmp_path / "kg_quality_report.txt").exists()


def test_the_database_reaches_the_session(monkeypatch, tmp_path):
    session = _Session()
    driver = _Driver(session)
    monkeypatch.setattr(neo4j_env, "connect", lambda target: driver)

    neo4j_ingestion.run_quality_checks(
        uri="bolt://x",
        user="neo4j",
        password="pw",
        report_path=tmp_path / "r.json",
        database="staging",
    )

    assert driver.databases == ["staging"]


# --- the one-triple-at-a-time fallback -------------------------------------


class _Tx:
    """Transaction that records its queries, or raises `error`."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, cypher: str, **params: Any):
        self.calls.append((cypher, params))
        if self.error is not None:
            raise self.error
        return _Result([])


def test_a_triple_that_writes_cleanly_is_written_once():
    tx = _Tx()

    neo4j_ingestion._merge_triple(tx, _triple())

    assert len(tx.calls) == 1
    assert tx.calls[0][1]["rows"][0]["s_name"] == "Rice husk"


def test_a_type_neo4j_refuses_is_skipped_so_the_rest_can_land(caplog, monkeypatch, tmp_path):
    # This is the fallback after a batch already failed; raising here would
    # lose every remaining triple of the run.
    monkeypatch.setattr(
        neo4j_ingestion, "__file__", str(tmp_path / "stages" / "neo4j_ingestion.py")
    )
    tx = _Tx(CypherTypeError("Property values can only be of primitive types"))

    with caplog.at_level(logging.ERROR):
        neo4j_ingestion._merge_triple(tx, _triple())

    assert "CypherTypeError" in caplog.text


def test_the_refused_triple_is_written_down_so_it_can_be_found(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    stages.mkdir()
    monkeypatch.setattr(neo4j_ingestion, "__file__", str(stages / "neo4j_ingestion.py"))
    tx = _Tx(CypherTypeError("bad property"))

    neo4j_ingestion._merge_triple(tx, _triple(subject="Scotta"))

    logged = (tmp_path / "logs" / "problematic_triples.jsonl").read_text(encoding="utf-8")
    assert "Scotta" in logged
    assert "s_props_sanitized" in logged


def test_an_unexpected_failure_is_also_survived(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    stages.mkdir()
    monkeypatch.setattr(neo4j_ingestion, "__file__", str(stages / "neo4j_ingestion.py"))
    tx = _Tx(RuntimeError("connection reset"))

    neo4j_ingestion._merge_triple(tx, _triple())

    logged = (tmp_path / "logs" / "problematic_triples.jsonl").read_text(encoding="utf-8")
    assert "connection reset" in logged


def test_a_log_that_cannot_be_written_does_not_lose_the_ingestion(monkeypatch, tmp_path, caplog):
    blocked = tmp_path / "stages"
    blocked.mkdir()
    (tmp_path / "logs").write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(neo4j_ingestion, "__file__", str(blocked / "neo4j_ingestion.py"))
    tx = _Tx(CypherTypeError("bad property"))

    with caplog.at_level(logging.ERROR):
        neo4j_ingestion._merge_triple(tx, _triple())

    assert "Failed to write problematic triple" in caplog.text
