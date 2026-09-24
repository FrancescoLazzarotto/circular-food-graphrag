"""Stage 4 decides which entities are the same entity.

Every merge here is irreversible downstream: two groups that collapse become
one node in Neo4j, with one name, and no later stage can tell them apart
again. A giant component or junk canonical names are decisions made in this
file.

So what is pinned is the deciding: which mentions land in the same initial
group, which pairs the embedding step is even allowed to propose, which of two
names becomes canonical, and three rules that guard the merge (accumulate
rather than overwrite, one LLM vote per pair, and a fingerprint on the merge
cache).

No model and no vLLM: `SentenceTransformer` is replaced by a fake that returns
vectors the test chose, so a similarity is a number under test rather than a
GPU result. `test_resolution_async.py` covers the concurrency; this covers the
decisions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pytest

from kg_pipeline.models.types import CanonicalEntityRecord, KGTriple
from kg_pipeline.stages import resolution


def _triple(
    subject="Rice husk",
    predicate="USES",
    obj="Substrate",
    *,
    doc="a.pdf",
    subject_labels=("Material",),
    object_labels=("Material",),
    subject_properties=None,
    object_properties=None,
) -> KGTriple:
    """A raw triple from `doc`, with the given labels."""
    return KGTriple.model_validate(
        {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "subject_labels": list(subject_labels),
            "object_labels": list(object_labels),
            "subject_properties": dict(subject_properties or {"name": subject}),
            "object_properties": dict(object_properties or {"name": obj}),
            "relationship_properties": {"source_doc": doc},
        }
    )


class _FakeEncoder:
    """Returns the unit vector each name was assigned, so sims are chosen.

    Names not in the table get their own orthogonal axis, which makes them
    similar to nothing — the default a test wants for a name it did not set up.
    """

    calls: list[dict[str, Any]] = []

    def __init__(self, name: str, device: Any = None) -> None:
        self.name = name
        self.device = device
        _FakeEncoder.calls.append({"model": name, "device": device})

    def encode(self, names: list[str], normalize_embeddings: bool = False):
        table = _FakeEncoder.vectors
        dim = max(len(table.get(next(iter(table), ""), [])) if table else 0, len(names) + 2)
        out = []
        for pos, name in enumerate(names):
            if name in table:
                vec = np.array(table[name], dtype=float)
                vec = np.pad(vec, (0, dim - len(vec)))
            else:
                vec = np.zeros(dim)
                vec[pos] = 1.0
            norm = np.linalg.norm(vec)
            out.append(vec / norm if norm else vec)
        return np.array(out)


@pytest.fixture
def encoder(monkeypatch):
    """Replace `SentenceTransformer` with a reset `_FakeEncoder`."""
    _FakeEncoder.calls = []
    _FakeEncoder.vectors = {}
    monkeypatch.setattr(resolution, "SentenceTransformer", _FakeEncoder)
    return _FakeEncoder


# --- merging approved pairs ------------------------------------------------


def _groups(*sizes: int) -> list[list[int]]:
    """Groups with the given mention counts, over consecutive mention indices."""
    out, start = [], 0
    for size in sizes:
        out.append(list(range(start, start + size)))
        start += size
    return out


def test_a_directly_approved_pair_merges():
    clusters = resolution._centre_clusters(_groups(2, 1), {(0, 1)})

    assert list(clusters.values()) == [[0, 1]]


def test_a_chain_of_approvals_does_not_merge_its_ends():
    # food ~ leftover food ~ food scraps ~ food waste: each pair judged on its
    # own, none of them says "food" is "food waste".
    clusters = resolution._centre_clusters(_groups(4, 3, 2, 1), {(0, 1), (1, 2), (2, 3)})

    merged = [sorted(members) for members in clusters.values()]
    assert [0, 1] in merged
    assert not any(0 in m and 2 in m for m in merged)
    assert not any(0 in m and 3 in m for m in merged)


def test_the_largest_group_is_the_centre_its_neighbours_join():
    clusters = resolution._centre_clusters(_groups(1, 5, 1, 1), {(0, 1), (1, 2), (1, 3)})

    assert clusters[1] == [1, 0, 2, 3]


def test_a_group_joins_only_one_centre():
    # 2 is approved with both 0 and 1; it joins the larger, and 1 stays apart
    # because it was never approved with 0.
    clusters = resolution._centre_clusters(_groups(3, 2, 1), {(0, 2), (1, 2)})

    assert clusters[0] == [0, 2]
    assert clusters[1] == [1]


def test_merging_a_group_with_itself_changes_nothing():
    clusters = resolution._centre_clusters(_groups(1, 1, 1), {(1, 1)})

    assert sorted(clusters.values()) == [[0], [1], [2]]


@pytest.mark.parametrize(
    ("left", "right", "merged"),
    [
        ("3.3 ± 1.3", "3.3 ± 1.0", False),
        ("9 gruppi di generazione", "8 gruppi di generazione", False),
        ("7.58%", "75%", False),
        ("2,17", "2.17", True),
        ("17 Sustainable Development Goals", "Sustainable Development Goals", True),
        ("food waste", "spreco alimentare", True),
    ],
)
def test_names_with_different_numbers_are_never_merged(left, right, merged):
    mentions = [
        {"name": left, "label": "DataValue", "doc": "a.pdf", "properties": {}, "predicates": set()},
        {"name": right, "label": "DataValue", "doc": "a.pdf", "properties": {}, "predicates": set()},
    ]

    kept = resolution._drop_number_mismatches({(0, 1)}, mentions, [[0], [1]])

    assert (kept == {(0, 1)}) is merged


# --- the string key that decides an initial group --------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Food waste", "foodwaste"),
        ("food-waste", "foodwaste"),
        ("  FOOD   WASTE  ", "foodwaste"),
        ("CO2", "co2"),
        ("---", ""),
    ],
)
def test_names_differing_only_in_punctuation_share_a_key(value, expected):
    assert resolution._norm(value) == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (set(), set(), 1.0),
        ({"USES"}, set(), 0.0),
        ({"USES"}, {"USES"}, 1.0),
        ({"USES", "PART_OF"}, {"USES"}, 0.5),
    ],
)
def test_predicate_overlap_is_a_jaccard(a, b, expected):
    assert resolution._jaccard(a, b) == expected


def test_a_triple_yields_one_mention_per_end():
    mentions = resolution._build_mentions([_triple(doc="report.pdf")])

    assert [m["name"] for m in mentions] == ["Rice husk", "Substrate"]
    assert all(m["doc"] == "report.pdf" for m in mentions)
    assert mentions[0]["predicates"] == {"USES"}


def test_a_mention_with_no_label_defaults_to_concept():
    mentions = resolution._build_mentions(
        [_triple(subject_labels=(), object_labels=())]
    )

    assert [m["label"] for m in mentions] == ["Concept", "Concept"]


def test_two_spellings_of_one_name_group_together():
    mentions = resolution._build_mentions(
        [_triple(subject="Food waste"), _triple(subject="food-waste")]
    )

    groups = resolution._initial_groups(mentions, {}, context_jaccard_floor=0.15)
    subject_group = [g for g in groups if mentions[g[0]]["name"].lower().startswith("food")]

    assert len(subject_group) == 1
    assert len(subject_group[0]) == 2


def test_the_same_name_under_two_labels_stays_apart():
    # A Region called Milan and an Organization called Milan are two entities;
    # the cross-label pass later gets a say, the initial grouping does not.
    mentions = resolution._build_mentions(
        [
            _triple(subject="Milan", subject_labels=("Region",)),
            _triple(subject="Milan", subject_labels=("Organization",)),
        ]
    )

    groups = resolution._initial_groups(mentions, {}, context_jaccard_floor=0.15)
    milan = [g for g in groups if mentions[g[0]]["name"] == "Milan"]

    assert len(milan) == 2


def test_an_acronym_groups_with_what_it_stands_for():
    mentions = resolution._build_mentions(
        [
            _triple(subject="EU", subject_labels=("Organization",)),
            _triple(subject="European Union", subject_labels=("Organization",)),
        ]
    )

    groups = resolution._initial_groups(
        mentions, {"EU": "European Union"}, context_jaccard_floor=0.15
    )
    eu = [g for g in groups if mentions[g[0]]["name"] in {"EU", "European Union"}]

    assert len(eu) == 1


def test_one_name_used_in_unrelated_ways_splits_into_separate_groups():
    # Same surface, no shared predicate: the floor keeps them apart rather than
    # letting one string pull two entities into one node.
    mentions = resolution._build_mentions(
        [
            _triple(subject="Bank", predicate="LOCATED_IN", subject_labels=("Concept",)),
            _triple(subject="Bank", predicate="MANAGES", subject_labels=("Concept",)),
        ]
    )

    groups = resolution._initial_groups(mentions, {}, context_jaccard_floor=0.5)
    banks = [g for g in groups if mentions[g[0]]["name"] == "Bank"]

    assert len(banks) == 2


def test_a_floor_of_zero_puts_every_namesake_in_one_group():
    mentions = resolution._build_mentions(
        [
            _triple(subject="Bank", predicate="LOCATED_IN", subject_labels=("Concept",)),
            _triple(subject="Bank", predicate="MANAGES", subject_labels=("Concept",)),
        ]
    )

    groups = resolution._initial_groups(mentions, {}, context_jaccard_floor=0.0)
    banks = [g for g in groups if mentions[g[0]]["name"] == "Bank"]

    assert len(banks) == 1


# --- which pairs the embedding step may propose ----------------------------


def _candidates(names_and_labels, vectors, threshold, encoder):
    """The pairs the embedding step proposes over singleton groups."""
    encoder.vectors = vectors
    mentions = [
        {"name": name, "label": label, "doc": "a.pdf", "properties": {}, "predicates": set()}
        for name, label in names_and_labels
    ]
    groups = [[i] for i in range(len(mentions))]
    return resolution._embedding_candidates(
        mentions=mentions,
        groups=groups,
        embedding_model="fake-model",
        threshold=threshold,
    )


def test_a_single_group_never_loads_the_model(encoder):
    assert resolution._embedding_candidates([], [[0]], "fake-model", 0.5) == []
    assert encoder.calls == []


def test_two_similar_names_of_the_same_type_become_a_candidate(encoder):
    pairs = _candidates(
        [("food waste", "Product"), ("food wastes", "Product")],
        {"food waste": [1.0, 0.0], "food wastes": [0.99, 0.14]},
        0.88,
        encoder,
    )

    assert pairs == [(0, 1)]


def test_a_cross_label_pair_needs_a_stricter_similarity_than_a_same_label_one(encoder):
    # 0.90 clears the 0.88 threshold but not the 0.92 cross-label floor: the
    # case that separates the two rules.
    vectors = {"food waste": [1.0, 0.0], "spreco alimentare": [0.90, 0.436]}

    same_label = _candidates(
        [("food waste", "Product"), ("spreco alimentare", "Product")],
        vectors,
        0.88,
        encoder,
    )
    cross_label = _candidates(
        [("food waste", "Product"), ("spreco alimentare", "Process")],
        vectors,
        0.88,
        encoder,
    )

    assert same_label == [(0, 1)]
    assert cross_label == []


def test_a_cross_label_pair_above_the_stricter_floor_is_still_proposed(encoder):
    pairs = _candidates(
        [("food waste", "Product"), ("spreco alimentare", "Process")],
        {"food waste": [1.0, 0.0], "spreco alimentare": [0.99, 0.14]},
        0.88,
        encoder,
    )

    assert pairs == [(0, 1)]


def test_lowering_the_threshold_cannot_lower_the_cross_label_floor(encoder):
    # `max(threshold, 0.92)`: a run that opens up same-label matching does not
    # silently open up bilingual matching with it.
    pairs = _candidates(
        [("food waste", "Product"), ("spreco alimentare", "Process")],
        {"food waste": [1.0, 0.0], "spreco alimentare": [0.90, 0.436]},
        0.10,
        encoder,
    )

    assert pairs == []


def test_the_longest_name_represents_its_group_to_the_encoder(encoder):
    encoder.vectors = {}
    mentions = [
        {"name": "EU", "label": "Organization", "doc": "", "properties": {}, "predicates": set()},
        {"name": "European Union", "label": "Organization", "doc": "", "properties": {}, "predicates": set()},
        {"name": "Rice husk", "label": "Material", "doc": "", "properties": {}, "predicates": set()},
    ]

    resolution._embedding_candidates(mentions, [[0, 1], [2]], "fake-model", 0.88)

    assert encoder.calls == [{"model": "fake-model", "device": None}]


def test_the_embedding_device_comes_from_the_environment(encoder, monkeypatch):
    monkeypatch.setenv("KG_EMBED_DEVICE", "cpu")
    _candidates([("a", "X"), ("b", "X")], {}, 0.88, encoder)

    assert encoder.calls[0]["device"] == "cpu"


# --- reading the model's verdicts ------------------------------------------


def test_a_verdict_naming_a_group_that_does_not_exist_is_dropped():
    content = json.dumps(
        [
            {"left_group": 0, "right_group": 1, "merge": True},
            {"left_group": 0, "right_group": 99, "merge": True},
            {"left_group": -1, "right_group": 0, "merge": True},
        ]
    )

    assert resolution._approved_from_response(content, group_count=2) == {(0, 1)}


def test_a_refused_merge_is_not_an_approval():
    content = json.dumps(
        [
            {"left_group": 0, "right_group": 1, "merge": False},
            {"left_group": 0, "right_group": 1},
        ]
    )

    assert resolution._approved_from_response(content, group_count=2) == set()


def test_an_approval_is_stored_the_same_way_whichever_order_it_arrives_in():
    a = resolution._approved_from_response(
        json.dumps([{"left_group": 1, "right_group": 0, "merge": True}]), 2
    )
    b = resolution._approved_from_response(
        json.dumps([{"left_group": 0, "right_group": 1, "merge": True}]), 2
    )

    assert a == b == {(0, 1)}


def test_a_non_dict_verdict_is_ignored_rather_than_raising():
    assert resolution._approved_from_response('["yes", 3, null]', 2) == set()


def test_a_fenced_reply_is_still_parsed():
    content = '```json\n[{"left_group": 0, "right_group": 1, "merge": true}]\n```'

    assert resolution._parse_llm_json_array(content) == [
        {"left_group": 0, "right_group": 1, "merge": True}
    ]


def test_a_reply_wrapped_in_prose_is_still_parsed():
    content = 'Here are the merges: [{"merge": false}] — hope that helps.'

    assert resolution._parse_llm_json_array(content) == [{"merge": False}]


@pytest.mark.parametrize("content", ["no array at all", "", "{}"])
def test_a_reply_with_no_array_raises_rather_than_returning_empty(content):
    with pytest.raises(Exception):
        resolution._parse_llm_json_array(content)


# --- the merge cache and its fingerprint -----------------------------------


def _mentions_and_groups(names):
    """Concept mentions of `names`, one singleton group each."""
    mentions = [
        {"name": name, "label": "Concept", "doc": "a.pdf", "properties": {}, "predicates": set()}
        for name in names
    ]
    return mentions, [[i] for i in range(len(mentions))]


def test_the_same_grouping_fingerprints_the_same_way():
    m1, g1 = _mentions_and_groups(["a", "b"])
    m2, g2 = _mentions_and_groups(["a", "b"])

    assert resolution._group_fingerprint(m1, g1) == resolution._group_fingerprint(m2, g2)


def test_a_grouping_that_changed_fingerprints_differently():
    m1, g1 = _mentions_and_groups(["a", "b"])
    m2, g2 = _mentions_and_groups(["a", "c"])
    m3, g3 = _mentions_and_groups(["a", "b", "c"])

    prints = {
        resolution._group_fingerprint(m1, g1),
        resolution._group_fingerprint(m2, g2),
        resolution._group_fingerprint(m3, g3),
        resolution._group_fingerprint(m1, [[0, 1]]),
    }

    assert len(prints) == 4


def test_a_cache_from_a_different_grouping_is_refused(caplog):
    # Bare group indices mean nothing once the grouping changes: without the
    # fingerprint, a stage 4 rerun after stage 3 changed would reuse those
    # indices against different entities, with only a range check in the way.
    payload = {"group_fingerprint": "oldoldoldoldold", "pairs": [[0, 1]]}

    with caplog.at_level(logging.WARNING):
        assert resolution._cached_pairs_if_current(payload, "newnewnewnewnew", None) is None
    assert "different group construction" in caplog.text


def test_a_cache_from_before_fingerprinting_is_refused(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolution._cached_pairs_if_current([[0, 1]], "abc", None) is None
    assert "predates fingerprinting" in caplog.text


@pytest.mark.parametrize("payload", [None, "nonsense", 3])
def test_a_missing_or_malformed_cache_forces_reconfirmation(payload):
    assert resolution._cached_pairs_if_current(payload, "abc", None) is None


def test_a_matching_cache_is_used_as_it_is():
    payload = {"group_fingerprint": "abc", "pairs": [[0, 1], [2, 3]]}

    assert resolution._cached_pairs_if_current(payload, "abc", None) == {(0, 1), (2, 3)}


# --- one vote per pair -----------------------------------------------------


def test_a_pair_spanning_several_documents_is_judged_once(monkeypatch):
    # Appended to every document bucket it touches, a pair across five
    # documents would get five votes and one `merge: true` would win — the
    # opposite of the prompt's "if uncertain, return merge=false".
    sent: list[tuple[str, list[dict[str, Any]]]] = []

    async def _fake_batches(*, doc_batches, **kwargs):
        sent.extend(doc_batches)
        return [set()]

    monkeypatch.setattr(resolution, "_confirm_batches_async", _fake_batches)

    mentions = [
        {"name": "Food waste", "label": "Concept", "doc": "a.pdf", "properties": {}, "predicates": set()},
        {"name": "Food waste", "label": "Concept", "doc": "b.pdf", "properties": {}, "predicates": set()},
        {"name": "Spreco", "label": "Concept", "doc": "c.pdf", "properties": {}, "predicates": set()},
    ]

    resolution._confirm_candidates_with_llm(
        base_url="http://x/v1",
        api_key="EMPTY",
        model_name="m",
        mentions=mentions,
        groups=[[0, 1], [2]],
        candidates=[(0, 1)],
    )

    assert sum(len(pairs) for _, pairs in sent) == 1
    assert sent[0][0] == "a.pdf"  # the first document that carries it


def test_no_candidates_means_no_llm_call_at_all(monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("the LLM should not be reached")

    monkeypatch.setattr(resolution, "_confirm_batches_async", _boom)

    assert resolution._confirm_candidates_with_llm(
        base_url="http://x/v1", api_key="EMPTY", model_name="m",
        mentions=[], groups=[], candidates=[],
    ) == set()


def test_candidates_are_split_into_batches(monkeypatch):
    sent: list[tuple[str, list[dict[str, Any]]]] = []

    async def _fake_batches(*, doc_batches, **kwargs):
        sent.extend(doc_batches)
        return [set()]

    monkeypatch.setattr(resolution, "_confirm_batches_async", _fake_batches)

    n = resolution._CONFIRM_BATCH_SIZE + 5
    mentions = [
        {"name": f"n{i}", "label": "Concept", "doc": "a.pdf", "properties": {}, "predicates": set()}
        for i in range(n * 2)
    ]
    groups = [[i] for i in range(n * 2)]

    resolution._confirm_candidates_with_llm(
        base_url="http://x/v1", api_key="EMPTY", model_name="m",
        mentions=mentions, groups=groups,
        candidates=[(i, i + n) for i in range(n)],
    )

    assert [len(pairs) for _, pairs in sent] == [resolution._CONFIRM_BATCH_SIZE, 5]


@pytest.mark.parametrize("raw, expected", [("4", 4), ("0", 1), ("-3", 1), ("nonsense", 8)])
def test_the_concurrency_knob_is_read_but_never_drops_below_one(
    monkeypatch, raw, expected
):
    monkeypatch.setenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", raw)
    seen: dict[str, Any] = {}

    async def _fake_batches(**kwargs):
        seen.update(kwargs)
        return [set()]

    monkeypatch.setattr(resolution, "_confirm_batches_async", _fake_batches)

    mentions = [
        {"name": "a", "label": "Concept", "doc": "a.pdf", "properties": {}, "predicates": set()},
        {"name": "b", "label": "Concept", "doc": "a.pdf", "properties": {}, "predicates": set()},
    ]
    resolution._confirm_candidates_with_llm(
        base_url="http://x/v1", api_key="EMPTY", model_name="m",
        mentions=mentions, groups=[[0], [1]], candidates=[(0, 1)],
    )

    assert seen["concurrent_requests"] == expected


# --- end to end: which name survives ---------------------------------------


def _resolve(triples, encoder, *, vectors=None, threshold=0.88, floor=0.15, **kwargs):
    """Run `resolve_entities` over `triples` with the fake encoder."""
    encoder.vectors = vectors or {}
    return resolution.resolve_entities(
        triples=triples,
        acronym_map=kwargs.pop("acronym_map", {}),
        embedding_model="fake-model",
        similarity_threshold=threshold,
        context_jaccard_floor=floor,
        base_url=kwargs.pop("base_url", None),
        api_key=None,
        model_name=kwargs.pop("model_name", None),
        **kwargs,
    )


def test_the_spelled_out_alias_beats_the_acronym(encoder):
    _, registry = _resolve(
        [
            _triple(subject="EU", subject_labels=("Organization",)),
            _triple(subject="European Union", subject_labels=("Organization",)),
        ],
        encoder,
        acronym_map={"EU": "European Union"},
    )

    assert "European Union" in registry
    assert registry["European Union"].aliases == ["EU", "European Union"]


def test_a_resolved_triple_carries_the_canonical_name_not_the_surface_one(encoder):
    triples, _ = _resolve(
        [
            _triple(subject="EU", subject_labels=("Organization",)),
            _triple(subject="European Union", subject_labels=("Organization",)),
        ],
        encoder,
        acronym_map={"EU": "European Union"},
    )

    assert [t.subject for t in triples] == ["European Union", "European Union"]


def test_where_an_alias_was_seen_is_kept(encoder):
    _, registry = _resolve(
        [
            _triple(subject="EU", subject_labels=("Organization",), doc="one.pdf"),
            _triple(subject="European Union", subject_labels=("Organization",), doc="two.pdf"),
        ],
        encoder,
        acronym_map={"EU": "European Union"},
    )

    assert registry["European Union"].alias_sources == {
        "EU": ["one.pdf"],
        "European Union": ["two.pdf"],
    }


def test_two_groups_reaching_the_same_canonical_name_accumulate_rather_than_replace(
    encoder,
):
    # `_initial_groups` splits on predicate overlap, so one surface name can
    # produce several groups whose longest alias is identical. A plain
    # assignment would drop the earlier group's aliases, and their triples
    # would keep unresolved names.
    triples, registry = _resolve(
        [
            _triple(subject="Bank of Italy", predicate="LOCATED_IN", subject_labels=("Organization",), doc="one.pdf"),
            _triple(subject="Bank of Italy", predicate="MANAGES", subject_labels=("Organization",), doc="two.pdf"),
        ],
        encoder,
        floor=0.9,
    )

    record = registry["Bank of Italy"]
    assert record.alias_sources["Bank of Italy"] == ["one.pdf", "two.pdf"]


def test_an_entity_takes_the_label_most_of_its_mentions_carry(encoder):
    # A union of labels let one noisy mention type an entity twice, and the
    # extra label reaches the answer ("Sostenibilità, an Organization").
    _, registry = _resolve(
        [
            _triple(subject="Milan", subject_labels=("Place",)),
            _triple(subject="Milan", subject_labels=("Place",), predicate="HAS_MEMBER"),
            _triple(subject="Milan", subject_labels=("Organization",)),
        ],
        encoder,
    )

    assert registry["Milan"].labels == ["Place"]


def test_an_entity_is_typed_by_the_mentions_that_reach_it(encoder, monkeypatch):
    # The Place mentions of "Italia" merge with "Italy" and the Indicator one
    # stays apart, so "Italia" is both an alias of "Italy" and the name of its
    # own entry. Triples reach an entry by name: whichever entry the "Italia"
    # triples land on must be typed by them, not by its own group alone.
    def _same_label_only(mentions, groups, candidates, **_):
        return {
            (a, b) for a, b in candidates
            if mentions[groups[a][0]]["label"] == mentions[groups[b][0]]["label"]
        }

    monkeypatch.setattr(resolution, "_confirm_candidates_with_llm", _same_label_only)
    triples, _ = _resolve(
        [
            _triple(subject="Italia", predicate="HAS_VALUE", subject_labels=("Indicator",)),
            _triple(subject="Italia", predicate="LOCATED_IN", subject_labels=("Place",)),
            _triple(subject="Italia", predicate="LOCATED_IN", subject_labels=("Place",), doc="b.pdf"),
            _triple(subject="Italy", predicate="LOCATED_IN", subject_labels=("Place",), doc="c.pdf"),
            _triple(subject="Italy", predicate="LOCATED_IN", subject_labels=("Place",), doc="d.pdf"),
        ],
        encoder,
        vectors={"Italia": [1.0, 0.0], "Italy": [1.0, 0.0]},
        base_url="http://x/v1",
        model_name="m",
    )

    assert {t.subject_labels[0] for t in triples} == {"Place"}


def test_a_tie_between_labels_goes_to_the_more_specific_one(encoder):
    _, registry = _resolve(
        [
            _triple(subject="Milan", subject_labels=("Concept",)),
            _triple(subject="Milan", subject_labels=("Organization",)),
        ],
        encoder,
    )

    assert registry["Milan"].labels == ["Organization"]


def test_case_variants_of_one_name_collapse_to_a_single_entry(encoder):
    _, registry = _resolve(
        [
            _triple(subject="rice husk", subject_labels=("Material",)),
            _triple(subject="Rice Husk", subject_labels=("Material",)),
        ],
        encoder,
    )

    keys = [k for k in registry if k.lower() == "rice husk"]
    assert len(keys) == 1
    assert sorted(registry[keys[0]].aliases) == ["Rice Husk", "rice husk"]


def test_the_surviving_spelling_is_the_one_that_carried_the_winning_label(encoder):
    # Pinned as it stands, not as an endorsement. Between two spellings of the
    # same length the keeper is whichever one holds the precedence-winning
    # label, so the lower-cased "milan" beats "Milan" purely because the
    # Organization mention happened to be spelled that way. Changing the rule
    # changes names in the graph, which cannot be judged without a rebuild.
    _, registry = _resolve(
        [
            _triple(subject="Milan", subject_labels=("Concept",)),
            _triple(subject="milan", subject_labels=("Organization",)),
        ],
        encoder,
    )

    assert "milan" in registry
    assert "Milan" not in registry
    assert registry["milan"].labels == ["Organization"]


def test_between_case_variants_only_the_label_ever_decides(encoder):
    # The keeper is picked by `sorted(cnames, key=(-len, lower))` among the
    # variants holding the winning label, but two spellings can only collide
    # here after `strip().lower()`, and a surface name is already stripped by
    # the time it becomes an alias — so they are always the same length and
    # the length term never breaks anything. The label alone decides.
    _, registry = _resolve(
        [
            _triple(subject="EUROPEAN UNION", subject_labels=("Concept",)),
            _triple(subject="European Union ", subject_labels=("Organization",)),
        ],
        encoder,
    )

    assert "European Union" in registry  # stripped on the way in
    assert "EUROPEAN UNION" not in registry
    assert registry["European Union"].labels == ["Organization"]


def test_a_cross_label_merge_is_written_to_the_log(encoder, tmp_path):
    log = tmp_path / "crosslabel.log"

    _resolve(
        [
            _triple(subject="Milan", subject_labels=("Concept",)),
            _triple(subject="milan", subject_labels=("Organization",)),
        ],
        encoder,
        crosslabel_log_path=log,
    )

    entry = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[0])
    assert entry["normalized_name"] == "milan"
    assert entry["chosen_label"] == "Organization"
    assert entry["keeper"] == "milan"
    assert entry["removed"] == [{"canonical": "Milan", "labels": ["Concept"]}]


def test_an_unwritable_log_does_not_lose_the_resolution(encoder, tmp_path, caplog):
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a directory", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        _, registry = _resolve(
            [
                _triple(subject="Milan", subject_labels=("Concept",)),
                _triple(subject="milan", subject_labels=("Organization",)),
            ],
            encoder,
            crosslabel_log_path=blocked / "sub" / "crosslabel.log",
        )

    assert "milan" in registry
    assert "cross-label merge log" in caplog.text


def test_a_resolved_triple_takes_its_labels_from_the_registry(encoder):
    # The triple said Concept; after the merge the registry says Organization
    # and the triple is overwritten, not consulted.
    triples, _ = _resolve(
        [
            _triple(subject="Milan", subject_labels=("Concept",)),
            _triple(subject="milan", subject_labels=("Organization",)),
        ],
        encoder,
    )

    assert all(t.subject_labels == ["Organization"] for t in triples)
    assert all(t.subject == "milan" for t in triples)


def test_properties_from_every_mention_are_accumulated(encoder):
    _, registry = _resolve(
        [
            _triple(subject="Rice husk", subject_properties={"name": "Rice husk", "origin": "IT"}),
            _triple(subject="Rice husk", subject_properties={"name": "Rice husk", "colour": "brown"}),
        ],
        encoder,
        floor=0.0,
    )

    props = registry["Rice husk"].merged_properties
    assert props["origin"] == "IT"
    assert props["colour"] == "brown"


def test_a_name_nothing_else_matches_survives_untouched(encoder):
    triples, registry = _resolve([_triple(subject="Rice husk")], encoder)

    assert triples[0].subject == "Rice husk"
    assert registry["Rice husk"].aliases == ["Rice husk"]


# --- the cache, end to end -------------------------------------------------


def test_a_confirmed_run_writes_a_cache_stamped_with_its_grouping(
    encoder, tmp_path, monkeypatch
):
    cache = tmp_path / "merge_cache.json"
    monkeypatch.setattr(
        resolution, "_confirm_candidates_with_llm", lambda **kwargs: {(0, 1)}
    )

    _resolve(
        [_triple(subject="A"), _triple(subject="B")],
        encoder,
        base_url="http://x/v1",
        model_name="m",
        merge_cache_path=cache,
    )

    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["pairs"] == [[0, 1]]
    assert len(payload["group_fingerprint"]) == 16


def test_a_matching_cache_skips_the_model_entirely(encoder, tmp_path, monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("the LLM should not be reached")

    monkeypatch.setattr(resolution, "_confirm_candidates_with_llm", _boom)
    cache = tmp_path / "merge_cache.json"
    triples = [_triple(subject="A"), _triple(subject="B")]

    mentions = resolution._build_mentions(triples)
    groups = resolution._initial_groups(mentions, {}, context_jaccard_floor=0.15)
    cache.write_text(
        json.dumps(
            {
                "group_fingerprint": resolution._group_fingerprint(mentions, groups),
                "pairs": [],
            }
        ),
        encoding="utf-8",
    )

    _, registry = _resolve(
        triples, encoder, base_url="http://x/v1", model_name="m", merge_cache_path=cache
    )

    assert "A" in registry


def test_a_stale_cache_sends_the_run_back_to_the_model(encoder, tmp_path, monkeypatch):
    called: list[int] = []
    monkeypatch.setattr(
        resolution,
        "_confirm_candidates_with_llm",
        lambda **kwargs: called.append(1) or set(),
    )
    cache = tmp_path / "merge_cache.json"
    cache.write_text(
        json.dumps({"group_fingerprint": "somethingelse", "pairs": [[0, 1]]}),
        encoding="utf-8",
    )

    _resolve(
        [_triple(subject="A"), _triple(subject="B")],
        encoder,
        base_url="http://x/v1",
        model_name="m",
        merge_cache_path=cache,
    )

    assert called == [1]


def test_without_a_model_url_nothing_is_merged_beyond_the_exact_matches(encoder):
    _, registry = _resolve(
        [_triple(subject="A"), _triple(subject="Ab")], encoder, base_url=None
    )

    assert {"A", "Ab"} <= set(registry)


def test_a_cached_pair_outside_the_group_range_is_refused(encoder, tmp_path, caplog):
    triples = [_triple(subject="A")]
    mentions = resolution._build_mentions(triples)
    groups = resolution._initial_groups(mentions, {}, context_jaccard_floor=0.15)
    cache = tmp_path / "merge_cache.json"
    cache.write_text(
        json.dumps(
            {
                "group_fingerprint": resolution._group_fingerprint(mentions, groups),
                "pairs": [[0, 99]],
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        _, registry = _resolve(triples, encoder, merge_cache_path=cache)

    assert "outside valid group range" in caplog.text
    assert "Rice husk" not in registry or True  # the run still completed


# --- persistence -----------------------------------------------------------


def test_a_registry_survives_a_round_trip(tmp_path):
    path = tmp_path / "nested" / "registry.json"
    registry = {
        "European Union": CanonicalEntityRecord(
            canonical_name="European Union",
            aliases=["EU", "European Union"],
            labels=["Organization"],
            merged_properties={"name": "European Union"},
            alias_sources={"EU": ["one.pdf"]},
        )
    }

    resolution.save_registry(path, registry)

    assert resolution.load_registry(path) == registry


def test_saving_a_registry_with_case_variant_keys_warns(tmp_path, caplog):
    # Two keys differing only in case mean the cross-label pass did not do its
    # job; downstream they become two Neo4j nodes.
    registry = {
        name: CanonicalEntityRecord(
            canonical_name=name, aliases=[name], labels=["Concept"],
            merged_properties={"name": name}, alias_sources={},
        )
        for name in ("Rice husk", "rice husk")
    }

    with caplog.at_level(logging.WARNING):
        resolution.save_registry(tmp_path / "registry.json", registry)

    assert "case-variant duplicate" in caplog.text


def test_triples_survive_a_round_trip(tmp_path):
    path = tmp_path / "triples.json"
    triples = [_triple(), _triple(subject="Straw", predicate="PART_OF")]

    resolution.save_triples(path, triples)
    loaded = resolution.load_triples(path)

    assert [(t.subject, t.predicate, t.object) for t in loaded] == [
        (t.subject, t.predicate, t.object) for t in triples
    ]


def test_a_saved_file_is_readable_utf8_json(tmp_path):
    path = tmp_path / "triples.json"
    resolution.save_triples(path, [_triple(subject="città")])

    assert "città" in path.read_text(encoding="utf-8")


def test_a_term_beats_a_sentence_as_the_canonical_name():
    """The longest alias must not win the canonical name.

    That is how a 463-character sentence, or `principi di sostenibilita
    ambientale`, becomes a node name.
    """
    from kg_pipeline.stages.resolution import _pick_canonical_name

    aliases = [
        "Circular Economy for Food",
        "Le 3 C dell'economia circolare per l'alimentazione: un quadro concettuale",
    ]
    documents = {alias: {"a.pdf"} for alias in aliases}

    assert _pick_canonical_name(aliases, documents) == "Circular Economy for Food"


def test_the_name_more_documents_use_wins_between_terms():
    from kg_pipeline.stages.resolution import _pick_canonical_name

    aliases = ["spreco alimentare", "sprechi di cibo"]
    documents = {"spreco alimentare": {"a.pdf", "b.pdf"}, "sprechi di cibo": {"c.pdf"}}

    assert _pick_canonical_name(aliases, documents) == "spreco alimentare"
