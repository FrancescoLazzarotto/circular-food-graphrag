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

from graphrag.agent.core import _GATE_SNIPPET_CHARS, _gate_mode, _gate_snippet
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


# --- which part of a passage the gate reads -------------------------------

LOVINS_PASSAGE = (
    "L’evolversi del pensiero circolare non poteva a questo punto che tornare "
    "circolarmente alle origini della materia prima. Sono gli anni del capitale "
    "naturale, in cui si dimostra la relazione tra la produzione di beni e il "
    "consumo di risorse, e la necessità di non compromettere i rapporti con “il "
    "miglior fornitore di materia prima che il genere umano conosca” (Lovins, "
    "et al., 1999). A partire dall’ipotesi di Gaia si arriva alla biomimesi."
)
LOVINS_TERMS = ["miglior", "fornitore", "materia", "genere", "umano", "conosca"]


def test_the_snippet_is_where_the_question_terms_are() -> None:
    """The sentence that answers sits past the head of the chunk; the head
    alone reads as history of economics and the question gets refused."""
    snippet = _gate_snippet(LOVINS_PASSAGE, LOVINS_TERMS)

    assert "miglior fornitore di materia prima che il genere umano conosca" in snippet
    assert snippet.startswith("… ")
    assert len(snippet) <= _GATE_SNIPPET_CHARS


def test_a_short_passage_is_shown_whole() -> None:
    assert _gate_snippet("Il biochar  è un\ncarbone vegetale.", ["biochar"]) == (
        "Il biochar è un carbone vegetale."
    )


def test_without_a_matching_term_the_head_is_shown() -> None:
    assert _gate_snippet(LOVINS_PASSAGE, ["carbonara"]) == " ".join(
        LOVINS_PASSAGE.split()
    )[:_GATE_SNIPPET_CHARS]


def test_a_term_near_the_end_still_gets_a_full_window() -> None:
    """Started at the hit, the window would run out of passage and show a
    stub; it is pulled back so the gate reads a full snippet."""
    passage = "parola " * 60 + "fine con biochar."
    snippet = _gate_snippet(passage, ["biochar"])

    assert snippet.endswith("biochar.")
    assert len(snippet) >= _GATE_SNIPPET_CHARS - 10


def test_a_term_inside_a_word_does_not_count() -> None:
    passage = "x " * 150 + "materiale materiale materiale " + "y " * 150 + "la materia prima"
    snippet = _gate_snippet(passage, ["materia"])

    assert "materia prima" in snippet
