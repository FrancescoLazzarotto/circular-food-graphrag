"""`DEMO_PROFILE=prima` runs the demo as it ran before the ontology and the
reasoning changes, and every one of those changes has a switch of its own.

The demo turns each change on; the library leaves each off; `prima` turns
them all off at once, and a variable set explicitly still wins.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig
from graphrag.llm.prompts import PromptLibrary

# The switches `prima` turns off, as (variable, product constant, config field).
_SWITCHES = (
    ("DEMO_TRANSLATE_QUERY", "TRANSLATE_QUERY", "translate_query_for_retrieval"),
    ("DEMO_GATE_TERM_PASSAGES", "GATE_TERM_PASSAGES", "gate_term_passages"),
    ("DEMO_STRICT_DEFINITIONS", "STRICT_DEFINITIONS", "strict_definition_subject"),
    ("DEMO_TEXT_EXACT_DEFINITIONS", "TEXT_EXACT_DEFINITIONS", "exact_term_definitions"),
    ("DEMO_GAP_REMINDER", "GAP_REMINDER", "gap_reminder"),
    ("DEMO_ONTOLOGY_SWITCH", "ONTOLOGY_SWITCH", None),
    ("DEMO_SHARED_TEXT_INDEX", "SHARED_TEXT_INDEX", None),
)

_GAP_REMINDER = "for the reader: say what is missing about the subject"


@pytest.fixture
def product_config(monkeypatch):
    """A freshly imported product.config with the given environment."""

    def _load(**env: str):
        for name, _, _ in _SWITCHES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("DEMO_PROFILE", raising=False)
        monkeypatch.delenv("DEMO_ONTOLOGY", raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        import product.config as module

        return importlib.reload(module)

    yield _load
    monkeypatch.undo()
    import product.config as module

    importlib.reload(module)


def test_the_demo_runs_every_change_by_default(product_config):
    settings = product_config()
    config = settings.build_agent_config()
    for _, constant, field in _SWITCHES:
        assert getattr(settings, constant) is True, constant
        if field:
            assert getattr(config, field) is True, field
    assert settings.ONTOLOGY is False


def test_prima_turns_every_change_off(product_config):
    settings = product_config(DEMO_PROFILE="prima")
    config = settings.build_agent_config()
    for _, constant, field in _SWITCHES:
        assert getattr(settings, constant) is False, constant
        if field:
            assert getattr(config, field) is False, field
    assert config.enable_ontology is False


def test_a_variable_set_explicitly_wins_over_the_profile(product_config):
    settings = product_config(DEMO_PROFILE="prima", DEMO_TRANSLATE_QUERY="1")
    assert settings.TRANSLATE_QUERY is True
    assert settings.GATE_TERM_PASSAGES is False

    settings = product_config(DEMO_GAP_REMINDER="0")
    assert settings.GAP_REMINDER is False
    assert settings.TRANSLATE_QUERY is True


def test_an_unknown_profile_stops_the_demo(product_config):
    with pytest.raises(ValueError, match="DEMO_PROFILE"):
        product_config(DEMO_PROFILE="vecchio")


def test_the_library_keeps_every_change_off():
    default = AgentConfig()
    for _, _, field in _SWITCHES:
        if field:
            assert getattr(default, field) is False, field
    assert default.enable_ontology is False


def test_the_gap_reminder_is_in_the_prompt_only_when_switched_on():
    on = str(PromptLibrary.answer_prompt(AgentConfig(cite_evidence=True, gap_reminder=True)))
    off = str(PromptLibrary.answer_prompt(AgentConfig(cite_evidence=True)))
    assert _GAP_REMINDER in on
    assert _GAP_REMINDER not in off
    assert off == on.replace(
        " Write for the reader: say what is missing about the subject itself ('the yield "
        "per hectare is not given'), never about the context, the evidence, the passages "
        "or the knowledge graph.",
        "",
    )


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text
        self.source = "doc.pdf"


class _Pipeline:
    """Ranks one passage that never names the term; the exact lookup finds it."""

    def __init__(self) -> None:
        self.asked: list[str] | None = None

    def retrieve(self, query, top_k=3):
        return [_Chunk("La filiera del riso comprende diverse fasi di lavorazione.")]

    def chunks_with_terms(self, terms, query=""):
        self.asked = list(terms)
        return [_Chunk("La sbramatura toglie la lolla dal risone.")]


def _gate_passages(**config) -> tuple[list[str], _Pipeline]:
    pipeline = _Pipeline()
    retriever = SimpleNamespace(
        _build_search_terms=lambda **_: ["sbramatura"],
        kg_store=SimpleNamespace(fulltext_search_nodes=lambda terms, limit: []),
        text_pipeline=pipeline,
    )
    shown: list[list[str]] = []
    llm = SimpleNamespace(
        classify_answerable=lambda question, names, passages, sources: shown.append(passages) or True
    )
    agent = KGRAGAgent(config=AgentConfig(**config), kg_retriever=retriever, llm=llm)
    agent._evidence_gate("Cos'è la sbramatura?")
    return shown[0], pipeline


def test_the_gate_reads_the_term_passages_only_when_switched_on():
    passages, pipeline = _gate_passages(gate_term_passages=True)
    assert any("sbramatura" in p for p in passages)

    passages, pipeline = _gate_passages()
    assert pipeline.asked is None
    assert not any("sbramatura" in p for p in passages)
