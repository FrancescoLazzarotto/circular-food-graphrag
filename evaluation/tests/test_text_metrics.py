"""Tests for the text similarity metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
for p in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evalkit.metrics.text import (
    bleu,
    exact_match,
    partial_match,
    rouge_l,
    token_f1,
)


def test_exact_match_identical() -> None:
    assert exact_match("hello world", "hello world") == pytest.approx(1.0)


def test_exact_match_case_insensitive() -> None:
    assert exact_match("Hello World", "hello world") == pytest.approx(1.0)


def test_exact_match_different() -> None:
    assert exact_match("hello", "world") == pytest.approx(0.0)


def test_partial_match_partial() -> None:
    # "banana" is not in prediction; "cat" is → 1/2 tokens found
    score = partial_match("the cat sat on the mat", "cat banana")
    assert 0.0 < score < 1.0


def test_partial_match_full() -> None:
    assert partial_match("the cat sat", "cat") == pytest.approx(1.0)


def test_token_f1_perfect() -> None:
    assert token_f1("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)


def test_token_f1_zero() -> None:
    assert token_f1("apple", "banana") == pytest.approx(0.0)


def test_token_f1_partial() -> None:
    score = token_f1("the cat sat on the mat", "cat mat")
    assert 0.0 < score < 1.0


def test_rouge_l_identical() -> None:
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_rouge_l_partial() -> None:
    score = rouge_l("the cat sat", "cat sat on the mat")
    assert 0.0 < score < 1.0


def test_rouge_l_empty() -> None:
    assert rouge_l("", "reference") == pytest.approx(0.0)
    assert rouge_l("prediction", "") == pytest.approx(0.0)


def test_bleu_identical() -> None:
    score = bleu("the quick brown fox", "the quick brown fox")
    assert score > 0.9


def test_bleu_no_overlap() -> None:
    score = bleu("apple orange", "banana grape")
    assert score == pytest.approx(0.0)


def test_bleu_partial() -> None:
    score = bleu("the cat sat on the mat", "the cat sat on the mat and looked around")
    assert 0.0 < score <= 1.0
