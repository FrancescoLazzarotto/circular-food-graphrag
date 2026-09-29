"""The demo's config has to name the retriever the demo actually used.

`AgentConfig.text_retriever_backend` is the one field that records which text
retriever answered, so `build_agent_config` must set it to what
`build_text_pipeline` builds: left at its default, every session log and
every bug report read from it would name the wrong backend.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def config(monkeypatch):
    """A freshly imported product.config, so env overrides are read."""

    def _load(**env: str):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        import product.config as module

        return importlib.reload(module)

    yield _load
    import product.config as module

    importlib.reload(module)


def test_the_config_names_the_backend_that_will_be_built(config):
    settings = config()

    built = settings.TEXT_RETRIEVER_BACKEND
    recorded = settings.build_agent_config().text_retriever_backend

    assert recorded == built


@pytest.mark.parametrize("backend", ["dense", "tfidf"])
def test_whichever_backend_is_chosen_is_the_one_recorded(config, backend):
    settings = config(DEMO_TEXT_RETRIEVER_BACKEND=backend)

    assert settings.build_agent_config().text_retriever_backend == backend


def test_the_default_is_dense_not_the_dataclass_default(config):
    # `AgentConfig.text_retriever_backend` defaults to "tfidf"; the demo runs
    # dense and must say so.
    settings = config()
    monkeypatched_default = settings.build_agent_config().text_retriever_backend

    assert settings.TEXT_RETRIEVER_BACKEND == "dense"
    assert monkeypatched_default == "dense"


def test_the_embedding_model_is_named_once_for_both_uses(config):
    # One constant feeds both, so the model the pipeline loads is the one the
    # config reports.
    settings = config()

    assert (
        settings.build_agent_config().dense_embedding_model
        == settings.DENSE_EMBEDDING_MODEL
    )


def test_the_embedding_model_can_be_overridden(config):
    settings = config(DEMO_DENSE_EMBEDDING_MODEL="intfloat/multilingual-e5-large")

    assert (
        settings.build_agent_config().dense_embedding_model
        == "intfloat/multilingual-e5-large"
    )


def test_the_vector_index_directory_is_recorded_too(config):
    settings = config()

    recorded = settings.build_agent_config().vector_index_dir

    assert recorded.endswith("artifacts/vector_index")


def test_a_bare_term_is_answered_in_the_interface_language(config):
    assert config().build_agent_config().fallback_language == "it"
    assert config(DEMO_UI_LANGUAGE="en").build_agent_config().fallback_language == "en"


def _demo_agent(settings):
    """A demo agent on stand-in parts: nothing here talks to a server."""
    from graphrag.agent.core import KGRAGAgent
    from graphrag.kg.retriever import KGRetriever

    store, pipeline, llm = object(), object(), object()
    config = settings.build_agent_config()
    retriever = KGRetriever(kg_store=store, config=config, text_pipeline=pipeline)
    return KGRAGAgent(config=config, kg_retriever=retriever, llm=llm)


def test_the_deep_agent_answers_long_from_more_passages(config):
    settings = config()
    short = _demo_agent(settings)

    deep = settings.build_deep_agent(short)

    assert short.config.lead_with_answer is True
    assert deep.config.lead_with_answer is False
    assert deep.config.focused_answer is True
    assert deep.kg_retriever.config.text_retriever_top_k == settings.DEEP_TEXT_TOP_K
    assert settings.DEEP_TEXT_TOP_K > short.config.text_retriever_top_k
    # The question it deepens already passed the gate.
    assert deep.config.enable_domain_gate is False
    assert deep.config.answer_meta_questions is False


def test_the_deep_agent_reuses_the_demo_agents_parts(config):
    settings = config()
    short = _demo_agent(settings)

    deep = settings.build_deep_agent(short)

    assert deep.llm is short.llm
    assert deep.kg_retriever.kg_store is short.kg_retriever.kg_store
    assert deep.kg_retriever.text_pipeline is short.kg_retriever.text_pipeline
    # Its own config: the short agent is untouched.
    assert short.config.text_retriever_top_k != deep.config.text_retriever_top_k
