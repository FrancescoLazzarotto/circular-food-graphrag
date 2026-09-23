"""A skipped vector channel is skipped for a reason, and the reason is logged.

Every failure of `db.index.vector.queryNodes` names the procedure, so
matching "not found" or the procedure's own name would report any failure as
a missing index, with advice ("run kg_vector_index.py to build it") that is
wrong for every cause except that one.

The error texts below are the real ones, taken from a Neo4j 5 Aura instance by
provoking each fault, not paraphrases.
"""

from __future__ import annotations

import logging

import pytest

from graphrag.kg.manager import KnowledgeGraphManager

MISSING_INDEX = (
    "{code: Neo.ClientError.Procedure.ProcedureCallFailed} {message: Failed to "
    "invoke procedure `db.index.vector.queryNodes`: Caused by: "
    "java.lang.IllegalArgumentException: There is no such vector schema index: "
    "node_embedding}"
)
WRONG_DIMENSION = (
    "{code: Neo.ClientError.Statement.TypeError} {message: Vector index "
    "'node_embedding' has a configured dimensionality of 768, but the provided "
    "vector has dimension 3.}"
)
PROCEDURE_ABSENT = (
    "{code: Neo.ClientError.Procedure.ProcedureNotFound} {message: There is no "
    "procedure with the name `db.index.vector.queryNodes` registered for this "
    "database instance.}"
)


@pytest.fixture
def manager() -> KnowledgeGraphManager:
    """A manager with no connection: only the error classifier is exercised."""
    return KnowledgeGraphManager.__new__(KnowledgeGraphManager)


def _handle(manager: KnowledgeGraphManager, message: str) -> bool:
    """Run the vector-error classifier on an error carrying `message`."""
    manager._vector_skips = getattr(manager, "_vector_skips", 0)
    return manager._handle_vector_error(RuntimeError(message), "node_embedding")


@pytest.mark.parametrize(
    "message", [MISSING_INDEX, WRONG_DIMENSION, PROCEDURE_ABSENT],
    ids=["missing-index", "wrong-dimension", "procedure-absent"],
)
def test_the_question_survives_every_cause(manager, message: str) -> None:
    """Losing one channel must never cost the answer: the others still work."""
    assert _handle(manager, message) is True


def test_a_missing_index_is_a_warning_naming_the_fix(manager, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="graphrag.kg.manager"):
        _handle(manager, MISSING_INDEX)
    record = caplog.records[-1]
    assert record.levelno == logging.WARNING
    assert "kg_vector_index.py" in record.getMessage()


@pytest.mark.parametrize(
    "message", [WRONG_DIMENSION, PROCEDURE_ABSENT],
    ids=["wrong-dimension", "procedure-absent"],
)
def test_any_other_cause_is_an_error_that_does_not_misdirect(manager, caplog, message) -> None:
    """Calling these "missing index" sends the operator to the wrong script.

    A dimension mismatch means the index was built with another encoder;
    rebuilding it blindly is not the fix, reading the error is.
    """
    with caplog.at_level(logging.WARNING, logger="graphrag.kg.manager"):
        _handle(manager, message)
    record = caplog.records[-1]
    assert record.levelno == logging.ERROR
    assert "kg_vector_index.py" not in record.getMessage()


def test_skips_are_counted_so_the_demo_can_say_so(manager) -> None:
    """The count is what puts the notice on the affected answer."""
    manager._vector_skips = 0
    for message in (MISSING_INDEX, WRONG_DIMENSION, PROCEDURE_ABSENT):
        _handle(manager, message)
    assert manager.vector_skips == 3
