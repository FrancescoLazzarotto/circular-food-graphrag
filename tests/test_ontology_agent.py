"""The framework ontology inside the agent: when it is consulted, what the
model is shown, and what the check after generation says.

No graph and no model: the retriever is a fake, and the nodes are called
directly on hand-built states. The ontology is the repository build.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphrag.agent.core import KGRAGAgent
from graphrag.agent.evidence import evidence_from_dicts
from graphrag.config import AgentConfig
from graphrag.ontology import consult as ontology_consult
from graphrag.ontology import hooks, load
from graphrag.types import RAGState

MATTM_LIST = [
    "Rifiutare", "Ripensare", "Ridurre", "Riusare", "Riparare",
    "Rinnovare", "Rifabbricare", "Riqualificare", "Riciclare", "Recuperare",
]


class _Retriever:
    """KG retriever returning one fixed passage."""

    kg_store = None
    text_pipeline = None

    def retrieve(self, query: str, prefer_documents=()) -> dict[str, Any]:
        return {
            "context_text": "passaggio recuperato",
            "text_chunks": [{"content": "passaggio recuperato", "source": "altro.pdf#page=2", "chunk_id": "c1"}],
            "text_sources": [],
            "nodes": [],
            "triples": [],
        }


@pytest.fixture(scope="module")
def onto():
    try:
        return load()
    except ontology_consult.OntologyNotBuilt:
        pytest.skip("ontology/build/ontology.json assente")


def _agent(**overrides: Any) -> KGRAGAgent:
    base: dict[str, Any] = {"llm_warmup": False, "enable_cache": False, "enable_ontology": True, "cite_evidence": True}
    base.update(overrides)
    return KGRAGAgent(config=AgentConfig(**base), kg_retriever=_Retriever(), llm=None)


def test_switched_off_the_graph_is_unchanged(onto):
    nodes = set(_agent(enable_ontology=False).graph.get_graph().nodes)
    assert "ontology" not in nodes and "verify" not in nodes
    assert {"ontology", "verify"} <= set(_agent().graph.get_graph().nodes)


def test_the_state_declares_the_ontology_keys():
    # A key a node returns but RAGState does not declare is dropped.
    assert {"ontology_evidence", "ontology_match", "ontology_check"} <= set(RAGState.__annotations__)


@pytest.mark.parametrize(
    "question",
    ["Come si ricicla il vetro?", "Qual è la capitale della Mongolia?", "Come ridurre gli sprechi nelle mense?"],
)
def test_ordinary_questions_do_not_consult_the_ontology(onto, question):
    match, items = hooks.consult(question, onto)
    assert not match and not items


def test_a_framework_without_version_brings_every_version(onto):
    match, items = hooks.consult("Le 10R sono le stesse in tutti i documenti?", onto)
    assert match.frameworks == ["r_potting_2017", "r_errando", "r_nodes_2024"]
    assert not match.named_version
    assert [i["chunk_id"] for i in items][-1] == "ontology:differences"


def test_a_version_named_by_author_or_year(onto):
    match, _ = hooks.consult("Che differenza c'è tra le 9R di Potting e le 10 R di eRRRando?", onto)
    assert match.frameworks == ["r_potting_2017", "r_errando"]


def test_the_ontology_evidence_comes_first_with_document_and_page(onto):
    agent = _agent()
    state: dict[str, Any] = {"question": "Quali sono le 10 R di eRRRando?", "chosen_retrieval_mode": "HYBRID"}
    state.update(agent._consult_ontology(state))
    result = agent._retrieve(state)
    first = evidence_from_dicts(result["evidence_index"])[0]
    assert first.ref_id == "S1"
    assert first.chunk_id == "ontology:r_errando"
    assert first.source_doc == "IT_Circular-City-Selfie.pdf"
    assert "Ricontenere" in first.text
    assert "S1" in result["visible_evidence_refs"]


def test_a_wrong_version_in_the_answer_gets_a_note(onto):
    agent = _agent()
    answer = "Le 10 R di eRRRando sono:\n" + "\n".join(f"- R{i} – {name}" for i, name in enumerate(MATTM_LIST))
    state = {"question": "Quali sono le 10 R di eRRRando?", "answer": answer}
    state.update(agent._consult_ontology(state))
    result = agent._verify_against_ontology(state)
    assert result["ontology_check"]["checked"] and not result["ontology_check"]["ok"]
    assert "Ricontenere" in result["answer"].split("\n\n")[-1]


def test_no_false_extra_while_a_version_is_incomplete(onto):
    # R0 of eRRRando is not in the ontology yet: "Rifiutare" may be it.
    match, _ = hooks.consult("Quali sono le 10 R di eRRRando?", onto)
    report = hooks.check_answer("Sono " + ", ".join(MATTM_LIST) + ".", match, onto)
    assert report["checked"]
    assert "Rifiutare" not in report["extra"]
    assert {"Ricontenere", "Ricondizionare"} <= set(report["missing"])


def test_variants_of_a_name_match(onto):
    result = ontology_consult.verifica(["Riciclo", "Recupero energetico", "Riusare"], "r_potting_2017", ontology=onto)
    assert {m["position"] for m in result["matched"]} == {"R8", "R9", "R3"}


def test_an_answer_without_a_framework_list_is_left_alone(onto):
    agent = _agent()
    state = {"question": "Parlami delle 3C", "answer": "Le 3C sono un quadro concettuale."}
    state.update(agent._consult_ontology(state))
    result = agent._verify_against_ontology(state)
    assert result == {"ontology_check": {"checked": False, "ok": True, "framework": "", "family": "", "missing": [], "extra": []}}


class _RefusingLLM:
    """Model whose gates reject everything, recording whether it was asked."""

    def __init__(self) -> None:
        self.asked = 0

    def classify_answerable(self, *args, **kwargs):
        self.asked += 1
        return False

    def classify_in_domain(self, *args, **kwargs):
        self.asked += 1
        return False

    def warmup(self) -> None:
        return None


@pytest.mark.parametrize("question", ["Le 10 R di errando", "Che cos'è la [R+]?"])
def test_a_framework_question_passes_the_gates(onto, question):
    llm = _RefusingLLM()
    agent = KGRAGAgent(
        config=AgentConfig(llm_warmup=False, enable_cache=False, enable_ontology=True,
                           enable_domain_gate=True, answer_meta_questions=True),
        kg_retriever=_Retriever(),
        llm=llm,
    )
    assert agent._scope_gate({"question": question}) == {"in_domain": True}
    assert llm.asked == 0


def test_switched_off_the_gates_do_not_consult_the_ontology(onto, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("ontology consulted with the switch off")

    monkeypatch.setattr(hooks, "recognise", fail)
    agent = KGRAGAgent(
        config=AgentConfig(llm_warmup=False, enable_cache=False, enable_domain_gate=True),
        kg_retriever=_Retriever(),
        llm=_RefusingLLM(),
    )
    agent._scope_gate({"question": "Le 10 R di errando"})


def test_a_named_case_comes_first(onto):
    match, items = hooks.consult("Che strategie R applica l'Accademia della Vigna?", onto)
    assert match.cases == ["case_accademia_della_vigna"]
    assert items[0]["chunk_id"] == "ontology:case_accademia_della_vigna"
    assert "R+" in items[0]["content"]


def test_cases_by_strategy_and_territory(onto):
    assert sorted(c["name"] for c in ontology_consult.casi("R4", ontology=onto)) == ["Bottiglie Rese", "Rendivetro"]
    for case in ontology_consult.casi(territorio="Cuneo", ontology=onto):
        assert "cuneo" in case["territory"].lower()
        assert case["document"] and case["page"]


def test_a_question_the_ontology_does_not_know_is_not_checked(onto):
    agent = _agent()
    state = {"question": "Come si ricicla il vetro?", "answer": "- a\n- b\n- c"}
    state.update(agent._consult_ontology(state))
    assert agent._verify_against_ontology(state) == {}


def test_a_question_on_the_order_of_two_strategies_gets_the_rule_first(onto):
    match, items = hooks.consult("Nella gerarchia delle R viene prima riusare o riciclare?", onto)
    assert match.concepts == ["r_riusare", "r_riciclare"]
    assert items[0]["chunk_id"] == "ontology:order_r"
    assert "R3 Riusare viene prima di R8 Riciclare" in items[0]["content"]
    assert items[0]["source"].endswith("#page=5")


def test_two_strategies_without_an_order_word_get_no_rule(onto):
    _, items = hooks.consult("Che cosa hanno in comune riusare e riciclare nelle 10R?", onto)
    assert all(item["chunk_id"] != "ontology:order_r" for item in items)


def test_a_family_list_is_checked_against_its_family(onto):
    answer = (
        "Le 10 R di eRRRando sono R0 Rifiutare, R1 Ripensare, R2 Ridurre, R3 Riusare, R4 Ricontenere, "
        "R5 Riparare, R6 Ricondizionare, R7 Riconvertire, R8 Riciclo e R9 Recupero energetico. "
        "La famiglia R Vita comprende Riusare, Ricontenere, Riparare e Ricondizionare."
    )
    match = hooks.recognise("Quali sono le 10 R di eRRRando?", onto)
    report = hooks.check_answer(answer, match, onto)
    assert (report["ok"], report["family"], report["missing"]) == (False, "R Vita", ["Riconvertire"])
    assert "«R Vita»" in hooks.check_note(report, onto, "it")


def test_a_stretch_of_a_scale_without_families_is_not_judged(onto):
    answer = (
        "Le 10 R sono R0 Rifiutare, R1 Ripensare, R2 Ridurre, R3 Riusare, R4 Riparare, R5 Rinnovare, "
        "R6 Rifabbricare, R7 Riqualificare, R8 Riciclare e R9 Recuperare. "
        "Le R centrali (Riusare, Riparare, Rinnovare, Rifabbricare, Riqualificare) allungano la vita del prodotto."
    )
    match = hooks.recognise("Quali sono le 10R del paradigma di Potting?", onto)
    assert hooks.check_answer(answer, match, onto)["ok"]

