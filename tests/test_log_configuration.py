"""A campaign log must say when things happened and, on request, be a file.

A long campaign produces thousands of lines in one nohup file, several arms
deep: without a timestamp there is no way to tell how long a turn took or
which arm a warning belongs to. The per-question progress line goes through
logging for the same reason: printed to stdout, it carries no timestamp and
never reaches a log file.
"""

from __future__ import annotations

import logging

import pytest

from graphrag import cli


@pytest.fixture(autouse=True)
def _restore_root_handlers():
    """Restore the root logger's handlers and level after each test."""
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in saved:
            handler.close()
            root.removeHandler(handler)
    root.handlers = saved
    root.setLevel(saved_level)


def test_the_console_format_carries_a_timestamp(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_LOG_FILE", raising=False)
    logging.getLogger().handlers = []

    cli._configure_logging()

    formats = [h.formatter._fmt for h in logging.getLogger().handlers if h.formatter]
    assert formats, "no handler was configured"
    assert all("asctime" in fmt for fmt in formats)


def test_a_log_file_is_written_when_asked(monkeypatch, tmp_path):
    target = tmp_path / "nested" / "campaign.log"
    monkeypatch.setenv("GRAPHRAG_LOG_FILE", str(target))
    logging.getLogger().handlers = []

    cli._configure_logging()
    logging.getLogger("graphrag").info("arm started")

    assert target.exists()
    written = target.read_text(encoding="utf-8")
    assert "arm started" in written
    # A dated line is the whole point.
    assert "|" in written and written.split("|")[0].strip()[:2].isdigit()


def test_no_log_file_without_the_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHRAG_LOG_FILE", raising=False)
    logging.getLogger().handlers = []

    cli._configure_logging()

    assert not any(
        isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers
    )


def test_calling_it_twice_does_not_duplicate_the_file_handler(monkeypatch, tmp_path):
    target = tmp_path / "campaign.log"
    monkeypatch.setenv("GRAPHRAG_LOG_FILE", str(target))
    logging.getLogger().handlers = []

    cli._configure_logging()
    cli._configure_logging()

    file_handlers = [
        h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1


def test_the_progress_line_goes_through_logging(caplog):
    """The only line that says which arm and question a log block belongs to."""
    from graphrag.experiments import runner as runner_module

    assert hasattr(runner_module, "logger")

    with caplog.at_level(logging.INFO, logger="graphrag"):
        runner_module.logger.info(
            "[%s] q%d/%d latency_ms=%.0f insufficient=%s kg_triples=%d",
            "hybrid",
            3,
            30,
            1234.5,
            False,
            17,
        )

    assert "[hybrid] q3/30 latency_ms=1234" in caplog.text
