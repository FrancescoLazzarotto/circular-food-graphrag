"""The edge judge's prompt and the probabilities read from its tokens."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("edge_judge", _ROOT / "scripts/kg/quality/edge_judge.py")
edge_judge = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("edge_judge", edge_judge)
_spec.loader.exec_module(edge_judge)


def _token(text: str, alternatives: dict[str, float] | None = None) -> dict:
    alts = [{"token": t, "logprob": math.log(p)} for t, p in (alternatives or {text: 1.0}).items()]
    return {"token": text, "logprob": alts[0]["logprob"], "top_logprobs": alts}


def test_probabilities_follow_each_verdict_whatever_the_quote_tokens():
    tokens = [
        _token('{"edges":[{"n":1,"evidence":"x","verdict":"'),
        _token("supported", {"supported": 0.8, "partial": 0.15, "not": 0.05}),
        _token('"},{"n":2,"evidence":"y","verdict'),
        _token('":"'),
        _token("partial", {"partial": 0.6, "supported": 0.3, "not": 0.1}),
        _token('"},{"n":3,"evidence":"z","verdict": '),
        _token('"not', {'"not': 0.9, '"supported': 0.1}),
        _token('_supported"}]}'),
    ]
    assert edge_judge.verdict_probabilities(tokens) == pytest.approx([0.8, 0.3, 0.1])


def test_no_logprobs_gives_no_probabilities():
    assert edge_judge.verdict_probabilities([]) == []


def test_prompt_numbers_edges_and_shows_the_original_predicate():
    edges = [
        {"subject": "PTC", "type": "RELATED_TO", "predicate": "PROVIDES", "object": "technical support"},
        {"subject": "FAO", "type": "PUBLISHED", "predicate": "", "object": "report"},
    ]
    text = edge_judge.prompt("Some passage.", edges)
    assert "1. PTC — RELATED_TO (original predicate: PROVIDES) → technical support" in text
    assert "2. FAO — PUBLISHED → report" in text
    assert edge_judge.schema(2)["properties"]["edges"]["minItems"] == 2
