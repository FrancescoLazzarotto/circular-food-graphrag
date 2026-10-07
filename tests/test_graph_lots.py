"""A lot enters a graph next to what the graph holds, without touching it.

The checks a lot runs before anything is written: which documents may form a
lot, what configuration extracts it, which lot entities are nodes the graph
already has, and what the write plan does with each.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kg_pipeline import lots
from kg_pipeline.models.types import CanonicalEntityRecord, KGTriple
from kg_pipeline.utils import corpus_registry
from kg_pipeline.utils.corpus_registry import RegistryRow


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registro.csv"
    corpus_registry.save_registry(
        path,
        [
            RegistryRow(id_documento="nuovo", percorso="A/nuovo.pdf"),
            RegistryRow(id_documento="altro", percorso="A/altro.pdf"),
            RegistryRow(id_documento="escluso", percorso="A/escluso.pdf", escluso=True),
            RegistryRow(id_documento="nel_grafo", percorso="A/nel_grafo.pdf", livello=2),
        ],
    )
    return path


def _base_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "paths": {"input_dir": "x", "output_dir": "y", "stage0_cache": "cache"},
                "llm": {"relation_vocab_path": "vocab.json"},
                "neo4j": {"database": "588fe1bc"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _prepare(tmp_path: Path, name: str = "lotto_a", docs: list[str] | None = None) -> lots.LotFiles:
    return lots.prepare(
        name=name,
        doc_ids=docs or ["nuovo"],
        registry=_registry(tmp_path),
        corpus_dir=tmp_path / "corpus",
        graph=lots.graph_key("bolt://localhost:7690"),
        llm_base_url="http://localhost:8000/v1",
        llm_model="Qwen/Qwen3-32B-AWQ",
        base_config=_base_config(tmp_path),
        lots_dir=tmp_path / "lotti",
    )


def test_a_lot_gets_its_own_registry_configuration_and_ledger_entry(tmp_path):
    files = _prepare(tmp_path)

    assert [r.id_documento for r in corpus_registry.load_registry(files.registry)] == ["nuovo"]
    config = yaml.safe_load(files.config.read_text(encoding="utf-8"))
    assert config["paths"]["registry"] == str(files.registry.resolve())
    assert config["paths"]["output_dir"] == str(files.dir.resolve())
    # The base names the production database; extraction must not carry it.
    assert config["neo4j"]["database"] != "588fe1bc"
    assert Path(config["llm"]["relation_vocab_path"]) == (tmp_path / "vocab.json").resolve()
    assert "127.0.0.1:1" in files.extract_env.read_text(encoding="utf-8")
    ledger = lots.Ledger.load(tmp_path / "lotti")
    assert ledger.lots["lotto_a"]["documents"] == ["nuovo"]


@pytest.mark.parametrize("doc", ["escluso", "nel_grafo", "manca"])
def test_a_document_that_cannot_enter_refuses_the_lot(tmp_path, doc):
    with pytest.raises(ValueError):
        _prepare(tmp_path, docs=["nuovo", doc])

    assert not (tmp_path / "lotti" / "lotto_a").exists()


def test_a_document_already_written_with_a_lot_cannot_enter_the_same_graph_again(tmp_path):
    _prepare(tmp_path)
    ledger = lots.Ledger.load(tmp_path / "lotti")
    ledger.lots["lotto_a"]["status"] = "scritto"
    ledger.save()

    with pytest.raises(ValueError, match="già nel grafo"):
        _prepare(tmp_path, name="lotto_b", docs=["nuovo", "altro"])


def test_a_lot_name_is_a_plain_folder_name(tmp_path):
    with pytest.raises(ValueError):
        _prepare(tmp_path, name="../fuori")


def _record(name: str, aliases: list[str], label: str = "Concept", docs: list[str] | None = None) -> CanonicalEntityRecord:
    return CanonicalEntityRecord(
        canonical_name=name,
        aliases=aliases,
        labels=[label],
        merged_properties={"name": name},
        alias_sources={a: docs or ["d1.pdf"] for a in aliases},
    )


def _node(element_id: str, name: str, label: str = "Concept", aliases: tuple[str, ...] = (), degree: int = 1):
    return lots.GraphNode(element_id, label, name, aliases, degree)


def test_an_entity_the_graph_already_names_is_matched_as_stage_four_would_merge_it():
    entries = {
        "Food Waste": _record("Food Waste", ["Food Waste"]),
        "WHEY": _record("WHEY", ["WHEY"], label="Material"),
        "PAC": _record("PAC", ["PAC"]),
        "something new": _record("something new", ["something new"]),
    }
    nodes = [
        _node("1", "food-waste"),
        _node("2", "whey", label="Product"),
        _node("3", "Politica agricola comune", aliases=("politica agricola comune",)),
    ]

    matches = lots.exact_matches(entries, nodes, {"PAC": "Politica agricola comune"})

    # Same label and the same letters and digits.
    assert matches["Food Waste"].node.element_id == "1"
    # Any label when the names differ only in case.
    assert matches["WHEY"].node.element_id == "2"
    # Acronyms are expanded with the lot's acronym map.
    assert matches["PAC"].node.element_id == "3"
    assert "something new" not in matches


def test_accents_are_folded_and_numbers_are_never_matched():
    entries = {
        "Perù": _record("Perù", ["Perù"], label="Place"),
        "50%": _record("50%", ["50%"], label="DataValue"),
    }
    nodes = [_node("1", "Peru", label="Place"), _node("2", "per", label="Place"), _node("3", "50%", label="DataValue")]

    matches = lots.exact_matches(entries, nodes, {})

    assert matches["Perù"].node.element_id == "1"
    assert "50%" not in matches


def test_a_lot_name_equal_to_an_alias_of_a_node_is_that_node_whatever_the_labels():
    entries = {"spreco alimentare": _record("spreco alimentare", ["spreco alimentare"], label="Process")}
    nodes = [_node("1", "food waste", label="Concept", aliases=("Spreco alimentare",))]

    assert lots.exact_matches(entries, nodes, {})["spreco alimentare"].node.element_id == "1"


def test_among_nodes_with_the_same_name_the_one_with_the_entity_label_wins():
    entries = {"Milano": _record("Milano", ["Milano"], label="Place")}
    nodes = [_node("1", "Milano", label="Organization", degree=50), _node("2", "Milano", label="Place", degree=2)]

    assert lots.exact_matches(entries, nodes, {})["Milano"].node.element_id == "2"


def _triple(s: str, p: str, o: str, **props) -> KGTriple:
    return KGTriple(
        subject=s, predicate=p, object=o,
        subject_labels=["Concept"], object_labels=["Concept"],
        subject_properties={"name": s}, object_properties={"name": o},
        relationship_properties={"source_doc": "d1.pdf", "chunk_id": "d1_chunk_00001", **props},
    )


def test_the_plan_reuses_matched_nodes_and_creates_the_others():
    registry = {
        "spreco alimentare": _record("spreco alimentare", ["spreco alimentare", "sprechi alimentari"]),
        "compost": _record("compost", ["compost"]),
    }
    matches = {"spreco alimentare": lots.Match(_node("e1", "food waste"), "giudice", 0.9)}
    plan = lots.plan_writes([_triple("compost", "REDUCES", "spreco alimentare")], registry, matches)

    assert plan.new_nodes == {("Concept", "compost"): {}}
    assert plan.edges[0][0] == ("new", "Concept", "compost")
    assert plan.edges[0][2] == ("id", "e1")
    # The node keeps its name; the lot's names become its aliases.
    assert plan.aliases[("id", "e1")] == ["spreco alimentare", "sprechi alimentari"]


def test_same_as_triples_become_aliases_and_self_loops_are_dropped():
    registry = {
        "compost": _record("compost", ["compost", "composto"], docs=["d1.pdf", "d2.pdf"]),
        "humus": _record("humus", ["humus"]),
    }
    matches = {"humus": lots.Match(_node("e2", "humus"), "nome")}
    triples = [
        _triple("composto", "SAME_AS", "compost"),
        _triple("humus", "RELATED_TO", "humus"),
        _triple("compost", "PRODUCES", "humus"),
    ]

    plan = lots.plan_writes(triples, registry, matches)

    assert plan.aliases[("new", "Concept", "compost")] == ["composto"]
    assert plan.skipped_self_loops == 1
    assert [(s, p, o) for s, p, o, _ in plan.edges] == [(("new", "Concept", "compost"), "PRODUCES", ("id", "e2"))]
    # A node matched by its own name gets no alias from it.
    assert ("id", "e2") not in plan.aliases


def test_judged_matches_reuse_stored_verdicts_and_keep_the_closest_node(tmp_path):
    entries = {"spreco": _record("spreco", ["spreco"])}
    candidates = [("spreco", _node("1", "food waste"), 0.91), ("spreco", _node("2", "waste"), 0.95)]
    asked: list[list[tuple[str, str]]] = []

    def judge(pairs):
        asked.append(pairs)
        return [pair[1] == "food waste" for pair in pairs]

    store = tmp_path / "giudizi.json"
    first = lots.judged_matches(candidates, entries, judge, store)
    second = lots.judged_matches(candidates, entries, judge, store)

    assert first["spreco"].node.element_id == "1"
    assert second["spreco"].node.element_id == "1"
    assert len(asked) == 1
    assert json.loads(store.read_text(encoding="utf-8")) == {"spreco||food waste": True, "spreco||waste": False}


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class _Session:
    """Answers the node query of :func:`lots.propose_unions` with fixed rows."""

    def __init__(self, rows):
        self.rows = rows

    def run(self, query, **params):
        return _Rows(self.rows)


def test_a_union_keeps_the_graph_node_as_centre_and_never_chains(tmp_path):
    rows = [
        {"id": "g1", "name": "food waste", "ours": False, "deg": 3},
        {"id": "l1", "name": "spreco alimentare", "ours": True, "deg": 40},
        {"id": "l2", "name": "sprechi alimentari", "ours": True, "deg": 5},
    ]
    # Every name close to every other: the judge decides.
    encode = lambda names: [[1.0, 0.0] for _ in names]  # noqa: E731
    asked = []

    def judge(pairs):
        asked.append(pairs)
        return [True for _ in pairs]

    store = tmp_path / "giudizi.json"
    unions = lots.propose_unions(_Session(rows), "L", encode, judge, store, min_degree=1)
    again = lots.propose_unions(_Session(rows), "L", encode, judge, store, min_degree=1)

    # The graph's node is the centre even though a lot node is more connected,
    # and the second lot node is not merged into a node that was merged.
    assert [(u["centro"], u["unito"]) for u in unions] == [
        ("food waste", "spreco alimentare"),
        ("food waste", "sprechi alimentari"),
    ]
    assert again == unions
    assert len(asked) == 1


def test_a_prepared_lot_already_holds_its_documents_until_removed(tmp_path):
    _prepare(tmp_path)

    with pytest.raises(ValueError, match="già nel grafo"):
        _prepare(tmp_path, name="lotto_b", docs=["nuovo"])

    ledger = lots.Ledger.load(tmp_path / "lotti")
    ledger.lots["lotto_a"]["status"] = "rimosso"
    ledger.save()
    _prepare(tmp_path, name="lotto_c", docs=["nuovo"])


def test_one_graph_has_one_key_whatever_the_loopback_spelling():
    assert lots.graph_key("bolt://127.0.0.1:7690/") == lots.graph_key("BOLT://localhost:7690")
    assert lots.graph_key("bolt://localhost:7690", "neo4j") != lots.graph_key("bolt://localhost:7690", "altro")


def test_a_matched_node_whose_id_now_names_another_node_stops_the_write():
    class Session:
        def run(self, query, **params):
            return [{"id": "1", "name": "food waste", "labels": ["Concept"]},
                    {"id": "2", "name": "qualcos'altro", "labels": ["Concept"]},
                    {"id": "3", "name": None, "labels": None}]

    problems = lots.check_matched_nodes(
        Session(), {"1": ("Concept", "food waste"), "2": ("Concept", "whey"), "3": ("Place", "Bra")}
    )

    assert len(problems) == 2
    assert problems[0].startswith("2:") and problems[1].startswith("3:")
