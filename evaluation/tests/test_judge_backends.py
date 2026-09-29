"""Tests for judge backend construction and failure semantics."""

from __future__ import annotations

import pytest

from evalkit.judge.backends import APIBackend, make_backend


@pytest.mark.parametrize("backend", ["vllm", "local_hf", "api"])
def test_make_backend_rejects_empty_model_id(backend):
    # JudgeConfig defaults model_id to "": misconfiguration must fail at
    # construction, not as a per-row 404 at scoring time.
    with pytest.raises(ValueError, match="model_id"):
        make_backend(backend, model_id="")


def test_make_backend_claude_code_defaults_model():
    judge = make_backend("claude_code", model_id="")
    assert judge.model_id == "sonnet"


def test_make_backend_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        make_backend("banana", model_id="x")


def test_api_backend_stores_temperature_for_both_providers():
    # Without temperature the provider samples at its default (1.0) and judge
    # scores stop being reproducible. The constructor must retain it so both
    # provider calls can pass it through.
    judge = APIBackend(model_id="m", provider="anthropic", temperature=0.0)
    assert judge.temperature == 0.0


def test_api_backend_unknown_provider():
    judge = APIBackend(model_id="m", provider="mistero")
    with pytest.raises(ValueError, match="Unknown API provider"):
        judge.complete("s", "u")
