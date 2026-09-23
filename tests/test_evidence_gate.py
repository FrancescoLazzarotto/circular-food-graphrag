"""The gate that asks the collection instead of describing it.

The scope gate names the domain in its prompt — food, crops, the three C's,
ecodesign. That description is a maintenance trap: the day a document about
something else is added, questions about the new material are refused, and
refused silently. This one is shown what the collection returned and judges
against that, so it widens on its own as documents arrive.

These tests are the structural guards. What the gate actually decides is
measured on labelled questions, not asserted here.
"""

from __future__ import annotations

import pytest

from graphrag.agent.core import _gate_mode
from graphrag.llm.manager import LLMManager
from graphrag.llm.prompts import PromptLibrary

DOMAIN_WORDS = [
    "food", "crop", "circular", "ecodesign", "metabolisation", "symbiosis",
    "by-product", "supply chain", "capital", "cyclicality", "co-evolution",
]


def _rendered(**kwargs) -> str:
    """The evidence-gate prompt, rendered to text."""
    return str(PromptLibrary.evidence_gate_prompt(**kwargs))


def test_the_prompt_names_no_domain() -> None:
    """The whole point. A domain written here is wrong the day the collection
    grows, and wrong in the direction that refuses the new documents."""
    text = _rendered(
        entity_names=["biochar", "siero di latte"],
        passages=["Il biochar migliora la fertilita del suolo."],
        sources=["MR37-ita.pdf"],
    ).lower()
    # The evidence itself may contain anything; the instructions must not.
    instructions = text.split("entries the collection holds")[0]
    leaked = [w for w in DOMAIN_WORDS if w in instructions]
    assert not leaked, f"the prompt describes the domain: {leaked}"


def test_a_brace_in_a_node_name_does_not_raise() -> None:
    """Names come from the graph and the template parses braces. In the scope
    gate a name containing "{" raises KeyError, and the caller swallows it by
    returning "in domain" — the gate switches itself off in silence."""
    prompt = PromptLibrary.evidence_gate_prompt(
        entity_names=["azienda {agricola}", "co{de"],
        passages=["testo con {parentesi} graffe"],
        sources=["report {2024}.pdf"],
    )
    rendered = prompt.invoke({"question": "Che cos'e il biochar?"})
    assert "biochar" in str(rendered)


def test_empty_evidence_still_renders() -> None:
    """A question with no match must still reach the model, not crash it."""
    prompt = PromptLibrary.evidence_gate_prompt()
    assert "nothing" in str(prompt.invoke({"question": "x"})).lower()


def test_the_evidence_mode_is_the_default(monkeypatch) -> None:
    """It scores as well as the scope gate, closes the conjunction bypass, and
    needs no domain written into the prompt.
    """
    monkeypatch.delenv("GRAPHRAG_GATE_MODE", raising=False)
    assert _gate_mode() == "evidence"


@pytest.mark.parametrize(
    "value,expected",
    [("evidence", "evidence"), ("SCOPE", "scope"), (" scope ", "scope"),
     ("scope", "scope"), ("qualunque", "evidence"), ("", "evidence")],
)
def test_the_mode_is_read_per_call(monkeypatch, value: str, expected: str) -> None:
    """Read per call, not at import, so the two can be compared in one process."""
    monkeypatch.setenv("GRAPHRAG_GATE_MODE", value)
    assert _gate_mode() == expected


@pytest.mark.parametrize(
    "completion,expected",
    [("IN", True), ("OUT", False), ("<think>hmm</think> OUT", False),
     ("<think>maybe OUT</think> IN", True), ("The answer is OUT.", False),
     ("garbage", True)],
)
def test_the_verdict_survives_a_reasoning_block(completion: str, expected: bool) -> None:
    """Reasoning models open with <think>, and reading only the first three
    characters would flip refusals into acceptances.
    """
    assert LLMManager._read_gate_verdict(completion, "q") is expected
