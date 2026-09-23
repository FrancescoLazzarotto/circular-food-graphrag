"""The health check must not read the ingestion pin as a serving requirement.

`VLLM_MODEL_NAME` in `kg_pipeline/.env` names the model the ingestion pipeline
extracts with — the one the current graph was built by — which need not be
the model being served. Read as "the model that must be served", it makes
the documented preflight report FAILED on a demo that is answering questions
correctly.

Editing the pin is the wrong fix: it would silently change which model a future
rebuild extracts with. The check simply must not make that inference — the demo
probes its endpoints and answers with whatever is up.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke" / "smoke_check.py"
_spec = importlib.util.spec_from_file_location("smoke_check", _PATH)
smoke_check = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("smoke_check", smoke_check)
_spec.loader.exec_module(smoke_check)


def _args(**kwargs) -> argparse.Namespace:
    """A namespace with the smoke check's model arguments; kwargs override."""
    base = {"llm_base_url": None, "llm_model": None, "timeout_sec": 1.0}
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_the_ingestion_pin_is_not_a_serving_requirement(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct-AWQ")
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(
        smoke_check, "_served_models", lambda url, timeout: ["RedHatAI/Qwen3.8-27B-INT4"]
    )

    ok, detail = smoke_check._check_generators(_args())

    assert ok, detail
    assert "Qwen3.8-27B-INT4" in detail


def test_an_explicit_expectation_is_still_enforced(monkeypatch) -> None:
    """Someone reproducing a campaign can still demand a specific model."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(
        smoke_check, "_served_models", lambda url, timeout: ["RedHatAI/Qwen3.8-27B-INT4"]
    )

    ok, detail = smoke_check._check_generators(_args(llm_model="Qwen/Qwen2.5-32B-Instruct-AWQ"))

    assert not ok
    assert "Qwen/Qwen2.5-32B-Instruct-AWQ" in detail


def test_an_endpoint_serving_nothing_still_fails(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(smoke_check, "_served_models", lambda url, timeout: [])

    ok, _ = smoke_check._check_generators(_args())

    assert not ok
