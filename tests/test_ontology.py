"""Competency questions of the framework ontology, run against the build.

The expected answers are read from
``ontology/competency_questions/competency_questions.tsv``, written from the
sources; the ontology passes a question only if the build reproduces them.
Questions outside the first batch (priority, cases) are not run here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from graphrag.agent.evidence import _normalize_for_quote_match
from graphrag.ontology import consult
from graphrag.ontology import casi, concetto, confronta, framework, load, piu_specifici, verifica

ROOT = Path(__file__).resolve().parents[1]
CQ_PATH = ROOT / "ontology" / "competency_questions" / "competency_questions.tsv"


def _questions() -> dict[str, dict]:
    with CQ_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def _fields(expected: str) -> dict[str, str]:
    fields = {}
    for part in expected.split(" | "):
        key, _, value = part.partition("=")
        fields[key.strip()] = value.strip()
    return fields


def _items(value: str) -> list[tuple[str, list[str]]]:
    """``"R0 Rifiutare, R4 Ricarica/Refill"`` → ``[("R0", ["Rifiutare"]), ...]``."""
    items = []
    for chunk in value.split(", "):
        position, _, labels = chunk.strip().partition(" ")
        # A slash separates alternative names, except inside a name that has
        # one ("Refurbish/Remanufacture"), so the whole form is a name too.
        items.append((position, [label.strip() for label in labels.split("/")] + [labels.strip()]))
    return items


def _same(a: str, b: str) -> bool:
    return consult._compact(a) == consult._compact(b)


def _member_names(onto, member: dict) -> list[str]:
    concept = onto.concepts.get(member["concept_id"], {})
    return [
        member["label_it"], member["label_en"], concept.get("label_it", ""),
        concept.get("label_en", ""), *concept.get("synonyms_it", []), *concept.get("synonyms_en", []),
    ]


QUESTIONS = _questions()


def _gap(reason: str):
    # A competency question the current batch does not answer yet. Strict, so
    # the mark has to go as soon as the ontology answers it.
    return pytest.mark.xfail(reason=reason, strict=True)


ERRANDO_R0 = _gap("r_errando R0 in sospeso: una bozza cita una frase senza il nome, l'altra indica p. 14 invece di 13")


def test_normalize_matches_quote_gate():
    samples = ["Ciclicità", "co-\nevoluzione", "«R0 = RIFIUTO»", "3 C’s", "Ri-contenere, [R+]"]
    for text in samples:
        assert consult.normalize(text) == _normalize_for_quote_match(text)


def test_competency_questions_use_known_operations():
    operations = {"list_framework", "compare_versions", "rank_priority", "broaden_narrow", "classify_case", "check_answer"}
    assert 25 <= len(QUESTIONS) <= 30
    for row in QUESTIONS.values():
        assert row["type"] in operations, row["id"]
        assert row["expected_answer"], row["id"]


@pytest.fixture(scope="module")
def onto():
    try:
        return load()
    except consult.OntologyNotBuilt:
        pytest.skip("ontology/build/ontology.json assente: eseguire ontology/scripts/build.py")


@pytest.mark.parametrize(
    "cq",
    [
        "CQ01",
        "CQ02",
        pytest.param("CQ03", marks=ERRANDO_R0),
        "CQ05",
        pytest.param("CQ06", marks=_gap("R7 accettata come 'Riproduzione riservata', artefatto della traduzione automatica a p. 24")),
    ],
)
def test_list_framework(onto, cq):
    fields = _fields(QUESTIONS[cq]["expected_answer"])
    result = framework(fields["framework"], ontology=onto)
    assert [v["id"] for v in result["versions"]] == [fields["framework"]]
    version = result["versions"][0]
    by_position = {m["position"]: m for m in version["members"]}
    expected = _items(fields["members"])
    assert sorted(by_position) == sorted(position for position, _ in expected)
    for position, labels in expected:
        names = _member_names(onto, by_position[position])
        assert any(_same(label, name) for label in labels for name in names if name), (cq, position, labels)
    for member in version["members"]:
        assert member["document"] and member["page"] and member["quote"]


def test_name_without_version_returns_every_version(onto):
    expected = _fields(QUESTIONS["CQ04"]["expected_answer"])["versions"].split(", ")
    for name in ("10 R", "10R", "le strategie R"):
        result = framework(name, ontology=onto)
        assert result["ambiguous"]
        assert sorted(v["id"] for v in result["versions"]) == sorted(expected)


@ERRANDO_R0
def test_errando_families(onto):
    version = framework("eRRRando", ontology=onto)["versions"][0]
    families = {group["family"]: group["positions"] for group in version["families"]}
    assert families == {
        "R Metodo": ["R0", "R1", "R2"],
        "R Vita": ["R3", "R4", "R5", "R6", "R7"],
        "R Trasformazione": ["R8", "R9"],
    }


def test_r4_differs_across_versions(onto):
    expected = {
        part.split(" R4=")[0]: part.split(" R4=")[1].replace(" (", "/").rstrip(")").split("/")
        for part in QUESTIONS["CQ08"]["expected_answer"].split(" | ")
        if " R4=" in part
    }
    for version in framework("10R", ontology=onto)["versions"]:
        member = next(m for m in version["members"] if m["position"] == "R4")
        names = [n for n in _member_names(onto, member) if n]
        assert any(_same(label, name) for label in expected[version["id"]] for name in names), version["id"]


def _changes(result: dict, kind: str) -> list[dict]:
    return [c for c in result["changes"] if c["type"] == kind]


@_gap("le unioni R5+R6 sono state fermate dal controllo di coerenza: le contraddice l'unione sbagliata eRRRando→2024")
def test_errando_against_potting(onto):
    result = confronta("r_potting_2017", "r_errando", ontology=onto)
    merged = _changes(result, "merged")
    assert {c["from"]["position"] for c in merged} == {"R5", "R6"}
    assert {c["to"]["position"] for c in merged} == {"R6"}
    assert [c["to"]["position"] for c in _changes(result, "new")] == ["R4"]
    moved = [c for c in result["changes"] if c.get("moved") and c["type"] == "same"]
    assert any(c["from"]["position"] == "R4" and c["to"]["position"] == "R5" for c in moved)
    for change in merged + _changes(result, "new"):
        assert all(e["document"] and e["page"] and e["quote"] for e in change["evidence"])


@_gap("entrambe le bozze dicono che la R4 del 2024 è nuova rispetto a eRRRando, non rinominata")
def test_nodes_against_errando(onto):
    result = confronta("r_nodes_2024", "r_errando", ontology=onto)
    assert result["from"]["id"] == "r_errando"
    renamed = _changes(result, "renamed")
    assert [(c["from"]["position"], c["to"]["position"]) for c in renamed] == [("R4", "R4")]
    assert [c["to"]["position"] for c in _changes(result, "new")] == ["R+"]


def test_r_plus_belongs_only_to_2024(onto):
    names = ["Relazionarsi con cura", "Relate with care"]
    found = next(filter(None, (concetto(n, ontology=onto) for n in names)), None)
    assert found is not None
    assert [(m["framework"], m["position"]) for m in found["members"]] == [("r_nodes_2024", "R+")]
    numbered = [m for m in framework("r_nodes_2024", ontology=onto)["versions"][0]["members"] if m["position"] != "R+"]
    assert [m["position"] for m in numbered] == [f"R{i}" for i in range(10)]


def test_ricontenere_origin(onto):
    found = concetto("Ricontenere", ontology=onto)
    assert ("r_errando", "R4") in {(m["framework"], m["position"]) for m in found["members"]}
    assert all(m["document"] and m["page"] for m in found["members"])


@pytest.mark.parametrize(
    "cq",
    [
        pytest.param("CQ15", marks=_gap("rinnovabilità: citazione a cavallo fra p. 8 e 9, interrotta da didascalia e piè di pagina")),
        pytest.param("CQ16", marks=_gap("capitale culturale ed economico: citazioni a cavallo fra p. 7 e 8, interrotte dal piè di pagina")),
        "CQ17",
        pytest.param("CQ18", marks=_gap("i quattro cicli della metabolizzazione sono in una sola bozza")),
    ],
)
def test_narrower(onto, cq):
    fields = _fields(QUESTIONS[cq]["expected_answer"])
    start = fields["concept"].split("/")[0]
    expected = [label.split("/") for label in fields["narrower"].split(", ")]
    found = piu_specifici(start, 1, ontology=onto)
    names = [[c["label_it"], c["label_en"]] for c in found]
    for labels in expected:
        assert any(_same(label, name) for label in labels for pair in names for name in pair if name), (cq, labels)
    for concept in found:
        assert concept["definitions"], (cq, concept["id"])


def test_symbiosis_belongs_to_coevolution(onto):
    found = concetto("simbiosi", ontology=onto)
    assert [b["label_it"] for b in found["broader"]] == ["Coevoluzione"]


def test_mattm_list_flagged_as_errando(onto):
    items = ["Rifiutare", "Ripensare", "Ridurre", "Riusare", "Riparare", "Rinnovare", "Rifabbricare", "Riqualificare", "Riciclare", "Recuperare"]
    result = verifica(items, "le 10 R di eRRRando", ontology=onto)
    assert result["framework"] == "r_errando"
    assert not result["ok"]
    assert {"R4", "R6"} <= {m["position"] for m in result["missing"]}
    # Extra only once the version is complete; until then they are unjudged.
    flagged = set(result["extra"]) if result["complete"] else set(result["unjudged"])
    assert {"Rinnovare", "Rifabbricare"} <= flagged
    assert result["best_match"] == "r_potting_2017"


def test_case_strategy(onto):
    fields = _fields(QUESTIONS["CQ24"]["expected_answer"])
    framework_id, code = fields["strategies"].split(" ")
    case = next(c for c in casi(ontology=onto) if _same(c["name"], fields["case"]))
    assert [(s["framework"], s["position"]) for s in case["strategies"]] == [(framework_id, code)]
    assert case["document"] and case["page"]


def test_wrong_3c_flagged(onto):
    result = verifica(["Capitale", "Ciclicità", "Comunicazione"], "3C", ontology=onto)
    assert [m["position"] for m in result["missing"]] == ["3"]
    assert result["extra"] == ["Comunicazione"]


def test_every_result_has_a_source(onto):
    for version in framework("10R", ontology=onto)["versions"] + framework("3C", ontology=onto)["versions"]:
        assert version["document"] and version["pages"]
        assert not version["reviewed"] or version["status"] in consult.SETTLED
