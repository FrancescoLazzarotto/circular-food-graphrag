"""What the demo says the collection covers comes from the collection.

The refusal, the reply to "chi sei?", the line under the product name and the
example questions were written for a corpus about the circular economy of
food. With a corpus registry they name the registry's themes, so the wording
follows the corpus as it grows; without one they stay word for word as they
were. The topics are display only: the gate's own description of the domain
is never touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig
from graphrag.llm.prompts import PromptLibrary
from kg_pipeline.utils import corpus_registry
from kg_pipeline.utils.corpus_registry import RegistryRow

_ROOT = Path(__file__).resolve().parents[1]
_TOPICS = ("Systems Thinking e Design Sistemico", "Bioeconomia")


def _agent(**overrides: Any) -> KGRAGAgent:
    """Agent with no retriever or LLM, warmup and cache off."""
    base: dict[str, Any] = {"llm_warmup": False, "enable_cache": False}
    base.update(overrides)
    return KGRAGAgent(config=AgentConfig(**base), kg_retriever=None, llm=None)


def test_the_introduction_names_the_topics_when_it_has_them():
    for language in ("it", "en"):
        text = PromptLibrary.identity_message(language, (), _TOPICS)
        assert "Systems Thinking e Design Sistemico; Bioeconomia" in text
        assert "food" not in text and "cibo" not in text


def test_without_topics_the_introduction_is_unchanged():
    it = PromptLibrary.identity_message("it", ())
    en = PromptLibrary.identity_message("en", ("Una domanda?",))

    assert it.startswith("Sono un assistente sull'economia circolare applicata al cibo. ")
    assert it.endswith("progetti territoriali descritti nei documenti.")
    assert en.startswith("I am an assistant on the circular economy applied to food. I answer")
    assert en.endswith("You could ask, for example:\n- Una domanda?")


def test_the_refusal_names_the_topics_in_both_languages():
    agent = _agent(collection_topics=_TOPICS)

    for question in ("Chi ha scritto la Divina Commedia?", "Who wrote the Divine Comedy?"):
        answer = agent._refuse_out_of_scope({"question": question})["answer"]
        assert answer.endswith("Systems Thinking e Design Sistemico; Bioeconomia.")
        assert PromptLibrary.DEFAULT_DOMAIN_SCOPE not in answer


def test_the_reply_to_chi_sei_names_the_topics():
    agent = _agent(collection_topics=_TOPICS)

    answer = agent._refuse_out_of_scope(
        {"question": "chi sei?", "meta_question": True, "meta_language": "it"}
    )["answer"]

    assert "Systems Thinking e Design Sistemico; Bioeconomia" in answer


def test_an_operator_scope_still_wins_over_the_topics():
    agent = _agent(collection_topics=_TOPICS, domain_scope="solo vini piemontesi")

    answer = agent._refuse_out_of_scope({"question": "Who wrote the Divine Comedy?"})["answer"]

    assert answer.endswith("solo vini piemontesi.")


def _demo_settings(env: dict[str, str]) -> dict[str, Any]:
    """The demo's wording and agent settings, imported fresh under ``env``."""
    code = (
        "import json; from product import config as c; a = c.build_agent_config();"
        "print(json.dumps({'topics': list(c.COLLECTION_TOPICS), 'tagline': c.PRODUCT_TAGLINE,"
        "'tagline_en': c.PRODUCT_TAGLINE_EN, 'examples': list(c.EXAMPLE_QUESTIONS),"
        "'agent_topics': list(a.collection_topics), 'domain_scope': a.domain_scope}))"
    )
    clean = {k: v for k, v in os.environ.items() if not k.startswith("DEMO_")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env={**clean, **env, "PYTHONPATH": f"{_ROOT}{os.pathsep}{_ROOT / 'src'}"},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_without_a_registry_the_demo_wording_is_unchanged():
    settings = _demo_settings({})

    assert settings["topics"] == [] and settings["agent_topics"] == []
    assert settings["tagline"] == (
        "Risponde sull'economia circolare del cibo citando i documenti da cui prende "
        "ogni affermazione."
    )
    assert settings["examples"][0] == "Che cos'è l'economia circolare applicata al cibo?"


def test_with_a_registry_the_demo_wording_follows_its_themes(tmp_path):
    registry = tmp_path / "registro.csv"
    corpus_registry.save_registry(
        registry,
        [
            RegistryRow(id_documento="a", percorso="Bioeconomia/a.pdf", tema="Bioeconomia"),
            RegistryRow(id_documento="b", percorso="Bioeconomia/b.pdf", tema="Bioeconomia"),
            RegistryRow(
                id_documento="c",
                percorso="System Dynamics/c.pdf",
                tema="System Dynamics",
            ),
            RegistryRow(
                id_documento="d", percorso="Escluso/d.pdf", tema="Escluso", escluso=True
            ),
        ],
    )

    settings = _demo_settings({"DEMO_CORPUS_REGISTRY": str(registry)})

    # Largest theme first; an excluded document brings no theme.
    assert settings["topics"] == ["Bioeconomia", "System Dynamics"]
    assert settings["agent_topics"] == settings["topics"]
    assert "Bioeconomia; System Dynamics" in settings["tagline"]
    assert "Bioeconomia; System Dynamics" in settings["tagline_en"]
    assert settings["examples"] == [
        "Che cosa dicono i documenti su «Bioeconomia»?",
        "Che cosa dicono i documenti su «System Dynamics»?",
    ]
    # The gate's description of the domain is never set from the registry.
    assert settings["domain_scope"] == ""
