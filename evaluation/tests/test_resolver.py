"""Tests for the shared label -> URI resolver, against the real gold and snapshot."""

from __future__ import annotations

import ast
import json
import socket
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
for p in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evalkit.metrics.resolver import (
    _NUMBER_LEXICON_FROM_GOLD,
    DEFAULT_VOCAB_PATH,
    SOURCE_GOLD,
    SOURCE_VOCABULARY,
    AmbiguousLabelError,
    LabelRecord,
    Resolver,
    fold_number,
)
from evalkit.normalisation import match_key

RESOLVER_SRC = Path(__file__).resolve().parents[1] / "evalkit" / "metrics" / "resolver.py"

# Gold locations, all under `evaluation/gold/`.
_GOLD_CANDIDATES = (
    EVAL_DIR / "gold" / "gold_circular_v1.json",
    EVAL_DIR / "gold" / "gold.json",
)

AGROVOC = "http://aims.fao.org/aos/agrovoc/c_"
GRAPE_POMACE = f"{AGROVOC}909cff15"
RICE_HUSKS = f"{AGROVOC}24892"
RICE_BRAN = f"{AGROVOC}77d35680"
WINE_LEES = f"{AGROVOC}28445"
BIOGAS = f"{AGROVOC}9262"
WHEY = f"{AGROVOC}8376"
POLYPHENOL = "http://purl.obolibrary.org/obo/CHEBI_26195"
CO_EVOLUTION = "urn:ceff:CoEvolution"
CYCLICALITY = "urn:ceff:Cyclicality"


def _gold_path() -> Path:
    """The first gold file present; fails the test when there is none."""
    for candidate in _GOLD_CANDIDATES:
        if candidate.exists():
            return candidate
    pytest.fail(f"gold not found at any of: {[str(c) for c in _GOLD_CANDIDATES]}")


@pytest.fixture(scope="module")
def gold_json() -> dict:
    """The raw gold JSON."""
    return json.loads(_gold_path().read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def resolver() -> Resolver:
    """The real resolver, built once from the gold + snapshot."""
    return Resolver.from_gold(_gold_path())


# --- The required cases ------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected", "why"),
    [
        ("vinacce", GRAPE_POMACE, "IT plural, AGROVOC prefLabel"),
        ("vinaccia", GRAPE_POMACE, "IT singular via the number lexicon"),
        ("grape pomace", GRAPE_POMACE, "EN prefLabel"),
        ("lolla", RICE_HUSKS, "IT corpus term for husks"),
        ("lolla di riso", RICE_HUSKS, "IT multiword, AGROVOC altLabel"),
        ("rice husks", RICE_HUSKS, "EN prefLabel"),
        ("rice husk", RICE_HUSKS, "EN singular"),
        ("pula", RICE_BRAN, "corpus usage: pula is BRAN, not husks (known conflict)"),
        ("crusca di riso", RICE_BRAN, "AGROVOC IT prefLabel for bran"),
        ("coevoluzione", CO_EVOLUTION, "urn:ceff: concept from gold alt_labels"),
        ("co-evolution", CO_EVOLUTION, "urn:ceff: normalised_label"),
        ("Ciclicità", CYCLICALITY, "accented IT form, mixed case"),
        ("ciclicita", CYCLICALITY, "accent-folded IT form"),
        ("polifenoli", POLYPHENOL, "ChEBI concept via IT plural"),
        ("siero", WHEY, "IT altLabel, cross-lingual (protocol §5)"),
        ("banana split", None, "unknown form resolves to nothing"),
        ("", None, "empty input"),
        ("   ", None, "whitespace-only input"),
        (".", None, "punctuation-only input normalises to empty"),
    ],
)
def test_resolve_table(resolver: Resolver, label: str, expected: str | None, why: str) -> None:
    assert resolver.resolve(label) == expected, why


def test_pula_and_lolla_stay_distinct_concepts(resolver: Resolver) -> None:
    """The husks/bran conflict must not collapse the two concepts into one."""
    assert resolver.resolve("pula") != resolver.resolve("lolla")


# --- No blind -s strip -------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected", "why"),
    [
        ("lees", WINE_LEES, "plurale tantum: must resolve as-is"),
        ("lee", None, "a blind -s strip would wrongly make 'lees' reachable from 'lee'"),
        ("biogas", BIOGAS, "singular ending in -s: a blind strip would yield 'bioga'"),
        ("bioga", None, "the blind-strip artefact must not resolve"),
        ("fecce", WINE_LEES, "IT plural folded via lexicon"),
        ("feccia", WINE_LEES, "IT singular"),
    ],
)
def test_number_folding_is_lexicon_only(
    resolver: Resolver, label: str, expected: str | None, why: str
) -> None:
    assert resolver.resolve(label) == expected, why


def test_fold_number_leaves_unknown_tokens_untouched() -> None:
    assert fold_number("lees") == "lees"
    assert fold_number("biogas") == "biogas"
    assert fold_number("vinacce") == "vinaccia"
    assert fold_number("rice husks") == "rice husk"


def test_from_gold_lexicon_pairs_are_attested_in_the_data(gold_json: dict) -> None:
    """Every _NUMBER_LEXICON_FROM_GOLD entry must have both forms in gold/snapshot.

    Machine-checks the provenance claim in the resolver's lexicon: the lexicon is
    sourced from pairs the data already contains, not invented to fit a result.
    """
    vocab = json.loads(DEFAULT_VOCAB_PATH.read_text(encoding="utf-8"))
    tokens: set[str] = set()
    for query in gold_json["queries"]:
        for entity in query["expected_entities"]:
            for form in [entity["normalised_label"], *entity["alt_labels"]]:
                tokens.update(match_key(form).split())
    for entry in vocab.values():
        for form in [*entry.get("pref", []), *entry.get("alt", [])]:
            tokens.update(match_key(form).split())

    unattested = {
        plural: singular
        for plural, singular in _NUMBER_LEXICON_FROM_GOLD.items()
        if plural not in tokens or singular not in tokens
    }
    assert unattested == {}, f"lexicon pairs claimed from gold but not attested: {unattested}"


# --- prefLabel AND altLabel --------------------------------------------------


def test_resolves_from_vocabulary_altlabels_absent_from_the_gold(resolver: Resolver) -> None:
    """AGROVOC/ChEBI altLabels extend coverage beyond the gold's own forms."""
    assert resolver.resolve("rice hulls") == RICE_HUSKS
    assert resolver.resolve("cholesterin") == "http://purl.obolibrary.org/obo/CHEBI_16113"
    assert resolver.resolve("tartar cream") == "http://purl.obolibrary.org/obo/CHEBI_32034"


def test_resolves_ceff_concepts_from_gold_alt_labels_only(resolver: Resolver) -> None:
    """The urn:ceff: concepts have no external vocabulary."""
    assert resolver.resolve("simbiosi industriale") == "urn:ceff:IndustrialSymbiosis"
    assert resolver.resolve("siero di ricotta") == "urn:ceff:Scotta"
    assert resolver.resolve("capitale culturale") == "urn:ceff:CulturalCapital"


# --- Cross-source conflict: gold wins, and only where pre-registered ---------


def test_pula_di_riso_conflict_is_the_only_suppression(resolver: Resolver) -> None:
    """AGROVOC calls 'Pula di riso' husks; the corpus calls it bran. Gold wins.

    Pinned deliberately: a NEW undeclared conflict from a future snapshot must
    fail this test rather than silently move the published numbers.
    """
    assert [c.key for c in resolver.conflicts] == ["pula di riso"]
    conflict = resolver.conflicts[0]
    assert conflict.kept_uri == RICE_BRAN
    assert conflict.kept_source == SOURCE_GOLD
    assert conflict.dropped_uris == (RICE_HUSKS,)
    assert resolver.resolve("pula di riso") == RICE_BRAN


def test_the_suppression_is_documented_in_the_pre_registered_gold(
    resolver: Resolver, gold_json: dict
) -> None:
    """Every suppression must be a declared decision, not a resolver-local choice."""
    documented = gold_json["_meta"]["alt_labels_provenance"]["known_terminology_conflicts"]
    blob = " ".join(documented).lower()
    for conflict in resolver.conflicts:
        assert conflict.key in blob, f"undeclared conflict {conflict.key!r}"


def test_gold_precedence_beats_vocabulary_for_the_same_form() -> None:
    records = [
        LabelRecord(form="shared form", uri="urn:x:FromGold", source=SOURCE_GOLD),
        LabelRecord(form="shared form", uri="urn:x:FromVocab", source=SOURCE_VOCABULARY),
    ]
    resolver = Resolver(records)
    assert resolver.resolve("shared form") == "urn:x:FromGold"
    assert resolver.conflicts[0].dropped_uris == ("urn:x:FromVocab",)


# --- Ambiguity is an explicit error, never an arbitrary pick -----------------


def test_ambiguous_form_raises_rather_than_picking_one() -> None:
    records = [
        LabelRecord(form="scotta", uri="urn:ceff:ConceptA", source=SOURCE_GOLD),
        LabelRecord(form="scotta", uri="urn:ceff:ConceptB", source=SOURCE_GOLD),
    ]
    resolver = Resolver(records)
    with pytest.raises(AmbiguousLabelError) as excinfo:
        resolver.resolve("Scotta")
    assert excinfo.value.uris == ("urn:ceff:ConceptA", "urn:ceff:ConceptB")
    assert excinfo.value.key == "scotta"
    assert "urn:ceff:ConceptA" in str(excinfo.value)
    assert resolver.ambiguous_forms == {"scotta": ("urn:ceff:ConceptA", "urn:ceff:ConceptB")}


def test_exact_match_wins_over_a_folded_ambiguity() -> None:
    """Tier order: an exactly-indexed form is never made ambiguous by folding."""
    records = [
        LabelRecord(form="vinaccia", uri="urn:ceff:Singular", source=SOURCE_GOLD),
        LabelRecord(form="vinacce", uri="urn:ceff:Plural", source=SOURCE_GOLD),
    ]
    resolver = Resolver(records)
    assert resolver.resolve("vinaccia") == "urn:ceff:Singular"
    assert resolver.resolve("vinacce") == "urn:ceff:Plural"


def test_ambiguity_created_by_number_folding_also_raises() -> None:
    """Folding must not silently merge two concepts.

    Reachable only when the queried form misses the exact tier yet folds onto an
    ambiguous key: here a mixed-number multiword form ('acidi fenolico'), which
    is exactly the kind of output a lossy extractor produces.
    """
    records = [
        LabelRecord(form="acidi fenolici", uri="urn:ceff:A", source=SOURCE_GOLD),
        LabelRecord(form="acido fenolico", uri="urn:ceff:B", source=SOURCE_GOLD),
    ]
    resolver = Resolver(records)
    with pytest.raises(AmbiguousLabelError) as excinfo:
        resolver.resolve("acidi fenolico")
    assert excinfo.value.uris == ("urn:ceff:A", "urn:ceff:B")


def test_ambiguity_surfaces_through_the_from_gold_factory(tmp_path: Path) -> None:
    gold = {
        "queries": [
            {
                "query_id": "Q01",
                "expected_entities": [
                    {
                        "normalised_label": "duplicated",
                        "alt_labels": [],
                        "uri": "urn:ceff:One",
                        "mapping_status": "benchmark_local_extension",
                    },
                    {
                        "normalised_label": "other",
                        "alt_labels": ["duplicated"],
                        "uri": "urn:ceff:Two",
                        "mapping_status": "benchmark_local_extension",
                    },
                ],
            }
        ]
    }
    gold_file = tmp_path / "gold_synthetic.json"
    gold_file.write_text(json.dumps(gold), encoding="utf-8")
    vocab_file = tmp_path / "vocab_synthetic.json"
    vocab_file.write_text(json.dumps({}), encoding="utf-8")

    resolver = Resolver.from_gold(gold_file, vocab_file)
    with pytest.raises(AmbiguousLabelError):
        resolver.resolve("duplicated")
    assert resolver.resolve("other") == "urn:ceff:Two"


def test_real_gold_has_no_ambiguous_forms(resolver: Resolver) -> None:
    assert resolver.ambiguous_forms == {}


# --- Offline, deterministic --------------------------------------------------


def test_resolver_module_imports_nothing_that_can_reach_the_network() -> None:
    """Static check: no network import anywhere, including inside functions."""
    forbidden = {
        "socket",
        "ssl",
        "urllib",
        "urllib.request",
        "http",
        "http.client",
        "httplib",
        "ftplib",
        "requests",
        "httpx",
        "aiohttp",
        "SPARQLWrapper",
        "rdflib",
    }
    tree = ast.parse(RESOLVER_SRC.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {f.split(".")[0] for f in forbidden}), f"network import: {imported}"


def test_build_and_resolve_work_with_sockets_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime proof: a fresh build + resolve never touches the network."""

    def _no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("resolver attempted network access")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)

    offline_resolver = Resolver.from_gold(_gold_path())
    assert offline_resolver.resolve("vinacce") == GRAPE_POMACE
    assert offline_resolver.resolve("banana split") is None


def test_resolution_is_deterministic_across_rebuilds() -> None:
    labels = ["vinacce", "pula", "lolla", "coevoluzione", "banana split", "lees"]
    first = Resolver.from_gold(_gold_path())
    second = Resolver.from_gold(_gold_path())
    assert [first.resolve(x) for x in labels] == [second.resolve(x) for x in labels]
    assert [first.resolve(x) for x in labels] == [first.resolve(x) for x in labels]


def test_no_fuzzy_matching(resolver: Resolver) -> None:
    """No fuzzy matching: a near-miss must NOT resolve, however close it looks."""
    for typo in ("vinacc", "vinacces", "polifenol", "coevoluzion", "rice husck", "wheyy"):
        assert resolver.resolve(typo) is None, f"{typo!r} resolved: fuzzy matching leaked in"


# --- Symmetric application, URI-native pipeline is a no-op -------------------


def test_known_uri_passes_through_unchanged(resolver: Resolver) -> None:
    """Ontology-grounded entities are already URIs: identity, for symmetry."""
    assert resolver.resolve(GRAPE_POMACE) == GRAPE_POMACE
    assert resolver.resolve(CO_EVOLUTION) == CO_EVOLUTION
    assert resolver.resolve("  " + RICE_BRAN + "  ") == RICE_BRAN


def test_unknown_uri_does_not_pass_through(resolver: Resolver) -> None:
    """Identity is granted only to URIs the gold declares — no invented grounding."""
    assert resolver.resolve("http://example.org/not/in/the/gold") is None


# --- Coverage sanity ---------------------------------------------------------


def test_every_gold_entity_resolves_to_its_own_uri(resolver: Resolver, gold_json: dict) -> None:
    """Each gold form must resolve to that entity's URI: the resolver's floor."""
    failures: list[tuple[str, str, str | None]] = []
    for query in gold_json["queries"]:
        for entity in query["expected_entities"]:
            uri = entity["uri"]
            if not uri:
                continue
            for form in [entity["label"], entity["normalised_label"], *entity["alt_labels"]]:
                got = resolver.resolve(form)
                if got != uri:
                    failures.append((form, uri, got))
    assert failures == [], f"gold forms not resolving to their own URI: {failures}"


def test_resolver_indexes_both_sources(resolver: Resolver) -> None:
    assert len(resolver) > 250
    assert len(resolver.known_uris) == 68
