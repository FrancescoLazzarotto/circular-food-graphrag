"""Every campaign log line says which run it belongs to.

A long campaign, or several arms started into one nohup file, produces
interleaved lines: without an id on each, a warning cannot be attributed to
the run whose results it explains. The id is the run's identity — the
`<timestamp>_<tag>` directory its results land in.
"""

from __future__ import annotations

import logging

import pytest

from graphrag import cli


@pytest.fixture(autouse=True)
def clean_logging():
    """Restore the root logger and clear the run id after each test."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    cli.set_run_id("")


def _record(name: str = "graphrag") -> logging.LogRecord:
    """An INFO record from the given logger."""
    return logging.LogRecord(name, logging.INFO, __file__, 1, "un messaggio", None, None)


def _formatted(handler: logging.Handler, record: logging.LogRecord) -> str:
    """Run the handler's filters on the record, then format it."""
    for filt in handler.filters:
        filt.filter(record)
    return handler.format(record)


def _handler() -> logging.Handler:
    """A stream handler with the CLI's log format and run-id filter."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(cli._LOG_FORMAT))
    cli._attach_run_id(handler)
    return handler


# --- the id on the line ----------------------------------------------------


def test_a_line_carries_the_run_it_belongs_to():
    cli.set_run_id("20260911_120000_hybrid")

    assert "20260911_120000_hybrid" in _formatted(_handler(), _record())


def test_before_a_run_is_named_the_field_is_a_placeholder():
    # Import-time and argument-parsing lines happen before the output
    # directory exists; they still have to format.
    cli.set_run_id("")

    assert " - | " in _formatted(_handler(), _record())


def test_naming_a_new_run_changes_the_lines_that_follow():
    handler = _handler()
    cli.set_run_id("primo")
    first = _formatted(handler, _record())
    cli.set_run_id("secondo")
    second = _formatted(handler, _record())

    assert "primo" in first
    assert "secondo" in second


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_empty_name_restores_the_placeholder(value):
    cli.set_run_id("qualcosa")
    cli.set_run_id(value if value is not None else "")

    assert cli._RUN_ID["value"] == "-"


def test_the_name_is_trimmed():
    cli.set_run_id("  20260911_120000_hybrid  ")

    assert cli._RUN_ID["value"] == "20260911_120000_hybrid"


# --- the filter has to reach every record ----------------------------------


def test_a_third_party_logger_does_not_break_the_formatter():
    # The filter is attached to handlers, not loggers: a record from neo4j or
    # httpx would otherwise reach the formatter without the attribute and
    # raise while formatting it.
    cli.set_run_id("run")

    assert "run" in _formatted(_handler(), _record(name="neo4j.notifications"))


def test_the_filter_is_attached_once_however_often_it_is_asked():
    handler = logging.StreamHandler()

    cli._attach_run_id(handler)
    cli._attach_run_id(handler)

    assert sum(isinstance(f, cli._RunIdFilter) for f in handler.filters) == 1


def test_a_record_that_already_names_a_run_is_overwritten_by_the_current_one():
    cli.set_run_id("vero")
    record = _record()
    record.run_id = "stale"  # type: ignore[attr-defined]

    assert "vero" in _formatted(_handler(), record)


# --- what _configure_logging leaves behind ---------------------------------


def test_configuring_logging_stamps_the_console_handler(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_LOG_FILE", raising=False)
    root = logging.getLogger()
    root.handlers[:] = []

    cli._configure_logging()

    assert root.handlers
    assert all(
        any(isinstance(f, cli._RunIdFilter) for f in h.filters) for h in root.handlers
    )


def test_a_file_log_carries_the_run_id_too(monkeypatch, tmp_path):
    path = tmp_path / "nested" / "campaign.log"
    monkeypatch.setenv("GRAPHRAG_LOG_FILE", str(path))
    root = logging.getLogger()
    root.handlers[:] = []

    cli._configure_logging()
    cli.set_run_id("20260911_120000_hybrid")
    logging.getLogger("graphrag").warning("attenzione")

    assert "20260911_120000_hybrid" in path.read_text(encoding="utf-8")


def test_the_file_handler_is_not_added_twice(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_LOG_FILE", str(tmp_path / "campaign.log"))
    root = logging.getLogger()
    root.handlers[:] = []

    cli._configure_logging()
    cli._configure_logging()

    assert sum(
        getattr(h, "_graphrag_file_handler", False) for h in root.handlers
    ) == 1


def test_every_line_still_carries_the_timestamp_it_gained_in_august():
    cli.set_run_id("run")

    line = _formatted(_handler(), _record())

    assert line.count("|") >= 4  # ts | level | run | name | message
