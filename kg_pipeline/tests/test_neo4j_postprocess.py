"""The pass that rewrites the graph after it has been built.

`neo4j_postprocess.py` is the largest file in the pipeline and the only one
that deletes. It renames relationship types, merges nodes it believes are the
same thing, and detaches whatever it decides is an artifact — on a graph that
is already the delivered product. When a guard stops working, one run can cost
the graph its vector carriers, entities and whole relationship types.

So what is pinned here is the deciding, not the plumbing: which relation type
folds into which, which node of a duplicate group survives, and — above all —
the three places where the pass is supposed to refuse: no APOC, over the
safety cap, above the rare-type threshold.

No Neo4j and no LLM: both are faked. Every test is a decision the code makes
before anything is written.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kg_pipeline.stages import neo4j_postprocess as pp


# --- fakes -----------------------------------------------------------------


class _Result:
    """Driver result over fixed rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)

    def single(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def data(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def consume(self) -> None:
        return None


class _Session:
    """Routes a query to the first route whose fragment appears in it.

    Matching on a fragment rather than the whole statement so a reformatted
    query still lands on the right handler; an unrouted query raises, because
    a test that silently answers nothing to a query it did not expect proves
    nothing. Order matters: put the more specific fragment first.
    """

    def __init__(self, routes: list[tuple[str, Any]] | None = None) -> None:
        self.routes = list(routes or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, cypher: str, **params: Any) -> _Result:
        flat = " ".join(cypher.split())
        self.calls.append((flat, params))
        for fragment, handler in self.routes:
            if fragment in flat:
                if isinstance(handler, BaseException):
                    raise handler
                rows = handler(params) if callable(handler) else handler
                return _Result(rows)
        raise AssertionError(f"unexpected query: {flat}")

    def ran(self, fragment: str) -> list[tuple[str, dict[str, Any]]]:
        return [call for call in self.calls if fragment in call[0]]


class _FakeLLM:
    """Enough of the OpenAI client for `_llm_json_array`."""

    def __init__(self, replies: list[Any] | None = None) -> None:
        self._replies = list(replies or [])
        self.prompts: list[str] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, *, model: str, temperature: float, messages: list[dict[str, str]]):
        self.prompts.append(messages[0]["content"])
        reply = self._replies.pop(0) if self._replies else "[]"
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


CANONICAL = list(pp._CANONICAL_RELATION_TYPES)
CANONICAL_SET = {pp._normalize_rel_type(item) for item in CANONICAL}
CANONICAL_TOKENS = {
    pp._normalize_rel_type(item): pp._reltype_tokens(item) for item in CANONICAL
}


def _target(source: str) -> str:
    """The deterministic relation target of `source` in the test vocabulary."""
    return pp._deterministic_relation_target(source, CANONICAL_SET, CANONICAL_TOKENS)


# --- normalisers -----------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("The European Union", "european union"),
        ("  A   Circular   Economy  ", "circular economy"),
        ("An Indicator", "indicator"),
        ("Bio-based materials", "bio based materials"),
        ("CO2", "co2"),
        # Not an article: no space after it.
        ("A.M.A.", "a m a"),
        ("!!!", ""),
    ],
)
def test_names_that_are_the_same_thing_normalise_to_the_same_key(value, expected):
    # This key is what decides duplicate groups, so two spellings that should
    # merge have to land on one string and two that should not, must not.
    assert pp._normalize_name(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("produced by", "PRODUCED_BY"),
        ("has--component", "HAS_COMPONENT"),
        ("__leading__", "LEADING"),
        ("  uses  ", "USES"),
        ("è_valido", "VALIDO"),
    ],
)
def test_a_relation_type_becomes_a_cypher_safe_token(value, expected):
    assert pp._normalize_rel_type(value) == expected


def test_a_quote_in_a_name_cannot_close_the_literal_early():
    # These go into query text by interpolation, not as a parameter.
    assert pp._cypher_string_literal("O'Brien") == "'O''Brien'"
    assert pp._cypher_string_literal("x' OR '1'='1") == "'x'' OR ''1''=''1'"


def test_a_backtick_in_a_label_cannot_close_the_identifier_early():
    assert pp._sanitize_label("  Con`cept  ") == "Concept"


@pytest.mark.parametrize("value, expected", [("", ""), ("  ", ""), ("RICE HUSK", "Rice Husk")])
def test_title_case_of_an_empty_name_is_empty_not_a_crash(value, expected):
    assert pp._to_title_case(value) == expected


def test_a_non_positive_chunk_size_means_one_chunk_not_an_infinite_loop():
    assert list(pp._chunked([1, 2, 3], 0)) == [[1, 2, 3]]
    assert list(pp._chunked([1, 2, 3], -5)) == [[1, 2, 3]]
    assert list(pp._chunked([], 0)) == []
    assert list(pp._chunked([1, 2, 3], 2)) == [[1, 2], [3]]


# --- LLM output ------------------------------------------------------------


def test_a_json_array_is_found_inside_the_prose_a_model_wraps_it_in():
    text = 'Here you go:\n```json\n[{"source": "A", "target": "B"}]\n```\nHope that helps.'

    assert pp._extract_first_json_array(text) == '[{"source": "A", "target": "B"}]'


def test_a_bracket_inside_a_string_does_not_end_the_array_early():
    text = 'prefix [{"source": "a]b", "target": "C"}] suffix'

    assert json.loads(pp._extract_first_json_array(text))[0]["source"] == "a]b"


def test_an_escaped_quote_does_not_end_the_string_early():
    text = r'[{"source": "a\"]", "target": "C"}]'

    assert json.loads(pp._extract_first_json_array(text))[0]["source"] == 'a"]'


def test_a_nested_array_is_returned_whole():
    assert pp._extract_first_json_array("x [[1, 2], [3]] y") == "[[1, 2], [3]]"


@pytest.mark.parametrize("text", ["no array here", "", '[{"unterminated": 1}'])
def test_output_with_no_complete_array_yields_nothing_rather_than_a_guess(text):
    assert pp._extract_first_json_array(text) == ""


def test_a_model_reply_that_needs_unwrapping_is_still_parsed():
    client = _FakeLLM(['Sure!\n[{"source": "X", "target": "USES"}]'])

    rows = pp._llm_json_array(client, "m", "prompt")

    assert rows == [{"source": "X", "target": "USES"}]


def test_a_model_reply_with_no_array_at_all_raises_rather_than_returning_empty():
    # Callers treat an empty list as "the model had nothing to map", which is a
    # different thing from "the model did not answer".
    client = _FakeLLM(["I cannot do that."])

    with pytest.raises(Exception):
        pp._llm_json_array(client, "m", "prompt")


# --- which relation type folds into which ----------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("IS_PART_OF", {"PART"}),
        ("HAS_COMPONENT", {"COMPONENT"}),
        ("HAS_CO2_LEVEL", {"CO2", "LEVEL"}),
        # A bare letter carries nothing; a bare digit might.
        ("X_RAY", {"RAY"}),
        ("PHASE_2", {"PHASE", "2"}),
        ("IS_A_TO_OF", set()),
    ],
)
def test_stop_words_are_dropped_before_types_are_compared(value, expected):
    assert pp._reltype_tokens(value) == expected


def test_a_canonical_type_maps_to_itself():
    assert pp._fallback_relation_target("uses", CANONICAL_SET) == "USES"
    assert _target("USES") == "USES"


@pytest.mark.parametrize("source", ["AFFECTED", "PRODUCING", "CAUSING", "UTILISED"])
def test_the_fallback_suffix_stripper_almost_never_fires_on_this_vocabulary(source):
    # It strips S/ES/ED/ING and keeps the result only if the stem is itself
    # canonical. This vocabulary is built from inflected verbs (AFFECTS,
    # PRODUCES, CAUSES), so the stem never is: everything falls to RELATED_TO.
    # Documented rather than fixed — the token matcher below is the path that
    # actually does the work.
    assert pp._fallback_relation_target(source, CANONICAL_SET) == "RELATED_TO"


def test_the_suffix_stripper_does_fire_when_the_stem_is_canonical():
    assert pp._fallback_relation_target("PUBLISHEDS", CANONICAL_SET) == "PUBLISHED"


@pytest.mark.parametrize(
    "source, expected",
    [
        # Two shared tokens is the strong signal.
        ("USES_METHOD_FOR", "USES_METHOD"),
        ("HAS_MINIMUM_LEVEL_OF", "HAS_MINIMUM_LEVEL"),
        # One shared token, but it is the whole of both sides.
        ("HAS_COMPONENT_OF", "HAS_COMPONENT"),
        ("IS_LOCATED_IN_REGION", "LOCATED_IN"),
    ],
)
def test_an_off_vocabulary_type_folds_into_the_canonical_one_it_shares_words_with(
    source, expected
):
    assert _target(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "MITIGATES_RISK",
        "HAS_ROLE",
        "SUPPORTS",
        "REPLACES",
        "EMPLOYS_TECHNIQUE_DURING_HARVEST",
    ],
)
def test_a_type_with_no_real_overlap_stays_related_to_rather_than_guessing(source):
    # HAS_ROLE is here on purpose: a synonym map would fold it into
    # HAS_COMPONENT because both start with HAS_. The stop-word list is what
    # stops that, and this is the test that catches it.
    assert _target(source) == "RELATED_TO"


def test_a_type_that_normalises_to_nothing_is_related_to():
    assert _target("___") == "RELATED_TO"
    assert _target("IS_OF_THE") == "RELATED_TO"


# --- the deterministic compaction pass -------------------------------------


_TYPE_COUNTS = "RETURN type(r) AS type, count(r) AS count"
_RENAME = "apoc.refactor.rename.type"


def _compact(rows, *, dry_run=False, apoc=True, rare_threshold=10, extra=None):
    """Run the deterministic compaction over the type counts in `rows`."""
    session = _Session([(_TYPE_COUNTS, rows), (_RENAME, extra if extra is not None else [])])
    report = pp._compact_relation_types_deterministic(
        session=session,
        canonical=CANONICAL,
        dry_run=dry_run,
        apoc_available=apoc,
        rare_threshold=rare_threshold,
    )
    return session, report


def test_compaction_refuses_to_write_without_apoc():
    session, report = _compact([{"type": "HAS_COMPONENT_OF", "count": 3}], apoc=False)

    assert report["errors"] == [
        "APOC unavailable, cannot rename relationship types for compaction"
    ]
    assert report["renamed_types"] == 0
    assert session.ran(_RENAME) == []


def test_a_frequent_off_vocabulary_type_is_reported_not_renamed():
    # Renaming a type that carries thousands of edges on a guess is how a graph
    # loses a distinction it needs. Above the threshold the pass only reports.
    session, report = _compact(
        [{"type": "HAS_COMPONENT_OF", "count": 5000}], rare_threshold=10
    )

    assert report["skipped_high_freq_noncanonical"] == [
        {"source": "HAS_COMPONENT_OF", "count": 5000}
    ]
    assert report["renamed_types"] == 0
    assert session.ran(_RENAME) == []


def test_a_rare_off_vocabulary_type_is_folded_and_counted():
    session, report = _compact(
        [
            {"type": "USES", "count": 900},
            {"type": "USES_METHOD_FOR", "count": 4},
            {"type": "MITIGATES_RISK", "count": 2},
        ]
    )

    assert report["already_canonical_types"] == 1
    assert report["renamed_types"] == 2
    assert report["renamed_edges"] == 6
    assert report["collapsed_to_related_to_edges"] == 2
    assert report["total_edges"] == 906
    assert [call[1] for call in session.ran(_RENAME)] == [
        {"old": "USES_METHOD_FOR", "new": "USES_METHOD"},
        {"old": "MITIGATES_RISK", "new": "RELATED_TO"},
    ]


def test_a_dry_run_reports_the_same_renames_without_making_them():
    session, report = _compact([{"type": "USES_METHOD_FOR", "count": 4}], dry_run=True)

    assert report["renamed_types"] == 1
    assert report["renamed_samples"] == [
        {"source": "USES_METHOD_FOR", "target": "USES_METHOD", "count": 4}
    ]
    assert session.ran(_RENAME) == []


def test_a_dry_run_needs_no_apoc():
    _, report = _compact(
        [{"type": "USES_METHOD_FOR", "count": 4}], dry_run=True, apoc=False
    )

    assert report["errors"] == []
    assert report["renamed_types"] == 1


def test_a_rename_that_fails_is_recorded_and_not_counted_as_done():
    session, report = _compact(
        [{"type": "USES_METHOD_FOR", "count": 4}],
        extra=RuntimeError("no such type"),
    )

    assert report["renamed_types"] == 0
    assert report["renamed_edges"] == 0
    assert "compaction rename failed for USES_METHOD_FOR -> USES_METHOD" in report["errors"][0]


def test_a_blank_type_is_skipped_rather_than_renamed_to_something():
    _, report = _compact([{"type": "   ", "count": 3}, {"type": None, "count": 1}])

    assert report["renamed_types"] == 0
    assert report["errors"] == []


# --- the LLM-driven mapping ------------------------------------------------


_COUNT = "RETURN count(r) AS c"


def _apply_mapping(items, replies, *, dry_run=False, batch_size=10):
    """Run the LLM-driven mapping of `items` with scripted `replies`."""
    session = _Session([(_COUNT, [{"c": 7}]), (_RENAME, [])])
    client = _FakeLLM(replies)
    report = pp._apply_relation_mapping(
        session=session,
        relation_items=items,
        canonical=CANONICAL,
        client=client,
        model_name="m",
        dry_run=dry_run,
        batch_size=batch_size,
    )
    return session, client, report


def test_a_type_already_in_the_vocabulary_never_reaches_the_model():
    session, client, report = _apply_mapping(
        [{"type": "USES"}, {"type": "LOCATED_IN"}], []
    )

    assert client.prompts == []
    assert {row["source"] for row in report["skipped"]} == {"USES", "LOCATED_IN"}
    assert session.ran(_RENAME) == []


def test_a_model_answer_outside_the_vocabulary_is_replaced_by_the_fallback():
    _, _, report = _apply_mapping(
        [{"type": "PUBLISHEDS"}],
        ['[{"source": "PUBLISHEDS", "target": "INVENTED_TYPE"}]'],
    )

    assert report["renamed"] == [
        {"source": "PUBLISHEDS", "target": "PUBLISHED", "count": 7}
    ]


def test_when_the_model_fails_every_type_still_gets_a_mapping():
    # The pass has to leave the graph in the vocabulary either way; a dead vLLM
    # must not mean "half the types keep their invented names".
    _, _, report = _apply_mapping(
        [{"type": "MITIGATES_RISK"}], [RuntimeError("connection refused")]
    )

    assert "llm mapping failed" in report["errors"][0]
    assert report["renamed"] == [
        {"source": "MITIGATES_RISK", "target": "RELATED_TO", "count": 7}
    ]


def test_a_type_the_model_did_not_answer_about_is_not_left_behind():
    _, _, report = _apply_mapping(
        [{"type": "MITIGATES_RISK"}, {"type": "SUPPORTS"}],
        ['[{"source": "MITIGATES_RISK", "target": "AFFECTS"}]'],
    )

    renamed = {row["source"]: row["target"] for row in report["renamed"]}
    assert renamed == {"MITIGATES_RISK": "AFFECTS", "SUPPORTS": "RELATED_TO"}


def test_types_are_sent_to_the_model_in_batches():
    items = [{"type": f"WEIRD_TYPE_{i}"} for i in range(5)]

    _, client, _ = _apply_mapping(items, ["[]", "[]", "[]"], batch_size=2)

    assert len(client.prompts) == 3


def test_a_failed_count_does_not_stop_the_rename():
    session = _Session([(_COUNT, RuntimeError("boom")), (_RENAME, [])])
    report = pp._apply_relation_mapping(
        session=session,
        relation_items=[{"type": "MITIGATES_RISK"}],
        canonical=CANONICAL,
        client=_FakeLLM(["[]"]),
        model_name="m",
        dry_run=False,
        batch_size=10,
    )

    assert "count failed for MITIGATES_RISK" in report["errors"][0]
    assert report["renamed"] == [
        {"source": "MITIGATES_RISK", "target": "RELATED_TO", "count": 0}
    ]
    assert len(session.ran(_RENAME)) == 1


def test_a_dry_run_mapping_reports_without_renaming():
    session, _, report = _apply_mapping(
        [{"type": "MITIGATES_RISK"}], ["[]"], dry_run=True
    )

    assert report["renamed"][0]["target"] == "RELATED_TO"
    assert session.ran(_RENAME) == []


# --- inverting and renaming ------------------------------------------------


_INVERSE_MERGE = "MERGE (b)-[r2:"


def test_an_inverse_rewrite_with_no_edges_writes_nothing():
    session = _Session([(_COUNT, [{"c": 0}])])

    report = pp._rewrite_inverse_relationships(
        session, [{"from": "USED_BY", "to": "USES"}], dry_run=False
    )

    assert report["pairs"] == [
        {"source": "USED_BY", "target": "USES", "count": 0, "rewritten": 0}
    ]
    assert session.ran(_INVERSE_MERGE) == []


def test_an_inverse_rewrite_flips_the_edges_it_finds():
    session = _Session(
        [(_INVERSE_MERGE, [{"rewritten": 12}]), (_COUNT, [{"c": 12}])]
    )

    report = pp._rewrite_inverse_relationships(
        session, [{"from": "used by", "to": "uses"}], dry_run=False
    )

    assert report["pairs"] == [
        {"source": "USED_BY", "target": "USES", "count": 12, "rewritten": 12}
    ]


def test_an_inverse_rewrite_in_dry_run_counts_but_does_not_flip():
    session = _Session([(_COUNT, [{"c": 12}])])

    report = pp._rewrite_inverse_relationships(
        session, [{"from": "USED_BY", "to": "USES"}], dry_run=True
    )

    assert report["pairs"][0]["rewritten"] == 0
    assert session.ran(_INVERSE_MERGE) == []


@pytest.mark.parametrize(
    "item", [{"from": "USES", "to": "USES"}, {"from": "", "to": "USES"}, {"from": "USES", "to": ""}]
)
def test_a_degenerate_rewrite_pair_is_ignored(item):
    session = _Session([])

    assert pp._rewrite_inverse_relationships(session, [item], dry_run=False)["pairs"] == []
    assert pp._rename_relation_types(session, [item], dry_run=False, apoc_available=True)["pairs"] == []
    assert pp._absorb_micro_relation_types(session, [item], dry_run=False)["pairs"] == []
    assert session.calls == []


def test_renaming_uses_apoc_when_it_is_there():
    session = _Session([(_RENAME, []), (_COUNT, [{"c": 30}])])

    report = pp._rename_relation_types(
        session, [{"from": "OLD_NAME", "to": "USES"}], dry_run=False, apoc_available=True
    )

    assert report["pairs"] == [
        {"source": "OLD_NAME", "target": "USES", "count": 30, "updated": 30}
    ]


def test_renaming_falls_back_to_merge_and_delete_without_apoc():
    session = _Session(
        [("MERGE (a)-[r2:", [{"updated": 30}]), (_COUNT, [{"c": 30}])]
    )

    report = pp._rename_relation_types(
        session, [{"from": "OLD_NAME", "to": "USES"}], dry_run=False, apoc_available=False
    )

    assert report["pairs"][0]["updated"] == 30
    assert session.ran(_RENAME) == []


def test_a_rename_whose_count_fails_is_skipped_not_attempted():
    session = _Session([(_COUNT, RuntimeError("boom"))])

    report = pp._rename_relation_types(
        session, [{"from": "OLD_NAME", "to": "USES"}], dry_run=False, apoc_available=True
    )

    assert report["pairs"] == []
    assert "rename count failed for OLD_NAME" in report["errors"][0]


def test_published_points_from_the_organization_to_the_document():
    session = _Session(
        [
            ("MERGE (o)-[r2:PUBLISHED]->(d)", [{"rewritten": 5}]),
            ("(d:Document)-[r:PUBLISHED]->(o:Organization) RETURN count(r)", [{"c": 5}]),
        ]
    )

    report = pp._invert_published_direction(session, dry_run=False)

    assert report == {"count": 5, "rewritten": 5, "errors": []}


def test_published_inversion_in_dry_run_writes_nothing():
    session = _Session(
        [("(d:Document)-[r:PUBLISHED]->(o:Organization) RETURN count(r)", [{"c": 5}])]
    )

    report = pp._invert_published_direction(session, dry_run=True)

    assert report == {"count": 5, "rewritten": 0, "errors": []}


def test_published_inversion_gives_up_when_it_cannot_count():
    session = _Session(
        [("(d:Document)-[r:PUBLISHED]->(o:Organization)", RuntimeError("boom"))]
    )

    report = pp._invert_published_direction(session, dry_run=False)

    assert report["rewritten"] == 0
    assert "published count failed" in report["errors"][0]


def test_micro_relation_types_are_absorbed_by_count():
    session = _Session([(_RENAME, []), (_COUNT, [{"c": 3}])])

    report = pp._absorb_micro_relation_types(
        session, [{"from": "TINY_TYPE", "to": "RELATED_TO"}], dry_run=False
    )

    assert report["pairs"] == [
        {"source": "TINY_TYPE", "target": "RELATED_TO", "count": 3, "updated": 3}
    ]


def test_a_micro_type_with_no_edges_is_reported_and_left_alone():
    session = _Session([(_COUNT, [{"c": 0}])])

    report = pp._absorb_micro_relation_types(
        session, [{"from": "TINY_TYPE", "to": "RELATED_TO"}], dry_run=False
    )

    assert report["pairs"][0]["updated"] == 0
    assert session.ran(_RENAME) == []


def test_counting_a_relationship_type_normalises_it_first():
    session = _Session([(_COUNT, [{"c": 4}])])

    assert pp._count_relationships(session, "used by") == 4
    assert "`USED_BY`" in session.calls[0][0]


# --- which duplicate survives ----------------------------------------------


_DUPES = "RETURN id(n) AS id, n.name AS name, labels(n) AS labels, degree"
_MERGE_NODES = "apoc.refactor.mergeNodes"


def test_only_names_that_collide_after_normalisation_form_a_group():
    session = _Session(
        [
            (
                _DUPES,
                [
                    {"id": 1, "name": "The Circular Economy", "labels": ["Concept"], "degree": 4},
                    {"id": 2, "name": "circular   economy", "labels": ["Concept"], "degree": 1},
                    {"id": 3, "name": "Rice husk", "labels": ["Material"], "degree": 9},
                    {"id": 4, "name": "   ", "labels": ["Concept"], "degree": 0},
                ],
            )
        ]
    )

    groups = pp._find_duplicate_groups(session)

    assert [group["normalized"] for group in groups] == ["circular economy"]
    assert {node["id"] for node in groups[0]["nodes"]} == {1, 2}


def test_the_best_connected_node_of_a_group_is_the_one_that_survives():
    session = _Session([(_MERGE_NODES, [{"id": 7}])])
    groups = [
        {
            "normalized": "circular economy",
            "nodes": [
                {"id": 3, "name": "circular economy", "labels": ["Concept"], "degree": 1},
                {"id": 7, "name": "Circular Economy", "labels": ["Concept"], "degree": 40},
            ],
        }
    ]

    report = pp._merge_duplicate_groups(session, groups, dry_run=False, label_mode="overlap")

    assert report["merged_nodes"] == 1
    assert report["samples"][0]["primary"] == "Circular Economy"
    assert session.ran(_MERGE_NODES)[0][1] == {"ids": [7, 3], "primary": 7}


def test_a_tie_on_degree_is_broken_by_the_older_node():
    session = _Session([(_MERGE_NODES, [{"id": 3}])])
    groups = [
        {
            "normalized": "x",
            "nodes": [
                {"id": 9, "name": "X", "labels": ["Concept"], "degree": 2},
                {"id": 3, "name": "x", "labels": ["Concept"], "degree": 2},
            ],
        }
    ]

    pp._merge_duplicate_groups(session, groups, dry_run=False, label_mode="overlap")

    assert session.ran(_MERGE_NODES)[0][1]["primary"] == 3


def test_a_namesake_with_a_different_type_is_not_merged_into_it():
    # "Milan" the Region and "Milan" the Organization are not the same node.
    session = _Session([])
    groups = [
        {
            "normalized": "milan",
            "nodes": [
                {"id": 1, "name": "Milan", "labels": ["Region"], "degree": 5},
                {"id": 2, "name": "Milan", "labels": ["Organization"], "degree": 3},
            ],
        }
    ]

    report = pp._merge_duplicate_groups(session, groups, dry_run=False, label_mode="overlap")

    assert report["skipped_incompatible"] == 1
    assert report["merged_nodes"] == 0
    assert session.calls == []


def test_a_duplicate_merge_in_dry_run_writes_nothing():
    session = _Session([])
    groups = [
        {
            "normalized": "x",
            "nodes": [
                {"id": 1, "name": "X", "labels": ["Concept"], "degree": 5},
                {"id": 2, "name": "x", "labels": ["Concept"], "degree": 3},
            ],
        }
    ]

    report = pp._merge_duplicate_groups(session, groups, dry_run=True, label_mode="overlap")

    assert report["merged_nodes"] == 1
    assert session.calls == []


def test_a_failed_merge_does_not_stop_the_remaining_groups():
    session = _Session([(_MERGE_NODES, RuntimeError("deadlock"))])
    groups = [
        {
            "normalized": str(n),
            "nodes": [
                {"id": n, "name": str(n), "labels": ["Concept"], "degree": 5},
                {"id": n + 100, "name": str(n), "labels": ["Concept"], "degree": 1},
            ],
        }
        for n in (1, 2)
    ]

    report = pp._merge_duplicate_groups(session, groups, dry_run=False, label_mode="overlap")

    assert report["groups"] == 2
    assert len(report["errors"]) == 2


@pytest.mark.parametrize(
    "primary, secondary, mode, expected",
    [
        (["Concept"], ["Organization"], "any", True),
        (["Concept"], ["Organization"], "overlap", False),
        (["Concept", "Material"], ["Concept"], "overlap", True),
        (["Concept", "Material"], ["Concept"], "exact", False),
        (["Concept"], ["Concept"], "exact", True),
        ([], [], "exact", True),
        ([], ["Concept"], "overlap", False),
    ],
)
def test_label_compatibility_modes(primary, secondary, mode, expected):
    assert pp._labels_compatible(primary, secondary, mode) is expected


# --- the isolated-node safety cap ------------------------------------------


_ISOLATED = "WHERE NOT (n)--() AND NOT n:NodeVec"
_DEGREES = "RETURN id(m) AS id, m.name AS name, degree"
_DETACH = "WHERE id(n) IN $ids DETACH DELETE n"


def _isolated_session(count: int) -> _Session:
    """A session with `count` isolated candidates and no namesakes."""
    rows = [{"id": i, "name": f"orphan {i}"} for i in range(count)]
    return _Session([(_ISOLATED, rows), (_DEGREES, []), (_DETACH, [])])


def test_a_cleanup_over_the_cap_refuses_and_becomes_a_dry_run(monkeypatch):
    # Unguarded, this pass can delete the vector carriers and whole relationship
    # types. The cap is the last thing standing between a changed graph shape
    # and a DETACH DELETE.
    monkeypatch.setenv("KG_ISOLATED_DELETE_MAX", "5")
    session = _isolated_session(9)

    report = pp._cleanup_isolated_nodes(session, dry_run=False, apoc_available=True)

    assert "refusing to delete 9 isolated nodes" in report["errors"][0]
    assert report["deleted_nodes"] == 9  # what it would have deleted
    assert session.ran(_DETACH) == []  # what it actually deleted


def test_a_cleanup_under_the_cap_goes_through(monkeypatch):
    monkeypatch.setenv("KG_ISOLATED_DELETE_MAX", "5")
    session = _isolated_session(3)

    report = pp._cleanup_isolated_nodes(session, dry_run=False, apoc_available=True)

    assert report["errors"] == []
    assert session.ran(_DETACH)[0][1] == {"ids": [0, 1, 2]}


def test_an_isolated_node_with_a_connected_namesake_is_merged_not_deleted():
    session = _Session(
        [
            (_ISOLATED, [{"id": 1, "name": "Rice Husk"}]),
            (_DEGREES, [{"id": 2, "name": "rice husk", "degree": 12}]),
            (_MERGE_NODES, [{"id": 2}]),
        ]
    )

    report = pp._cleanup_isolated_nodes(session, dry_run=False, apoc_available=True)

    assert report["matched"] == 1
    assert report["deleted_nodes"] == 0
    assert session.ran(_MERGE_NODES)[0][1] == {"ids": [2, 1], "primary": 2}


def test_an_isolated_node_that_would_be_merged_is_left_alone_without_apoc():
    session = _Session(
        [
            (_ISOLATED, [{"id": 1, "name": "Rice Husk"}]),
            (_DEGREES, [{"id": 2, "name": "rice husk", "degree": 12}]),
        ]
    )

    report = pp._cleanup_isolated_nodes(session, dry_run=False, apoc_available=False)

    assert report["skipped"] == 1
    assert report["errors"] == ["APOC unavailable, cannot merge isolated nodes"]


def test_nothing_isolated_means_no_further_queries():
    session = _Session([(_ISOLATED, [])])

    report = pp._cleanup_isolated_nodes(session, dry_run=False, apoc_available=True)

    assert report["candidates"] == 0
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "raw, expected", [("", 500), ("nonsense", 500), ("0", 500), ("-1", 500), ("2000", 2000)]
)
def test_a_broken_cap_setting_falls_back_to_the_default(monkeypatch, raw, expected):
    monkeypatch.setenv("KG_ISOLATED_DELETE_MAX", raw)

    assert pp._isolated_delete_cap() == expected


# --- ALL-CAPS concepts -----------------------------------------------------


_ALLCAPS = "n.name = toUpper(n.name)"
_NAMESAKE = "WHERE id(m) <> $id AND m.name = $name"
_REL_COUNT = "WHERE id(n) = $id MATCH (n)-[r]-()"
_SET_NAME = "SET n.name = $name"


def test_an_all_caps_concept_with_no_namesake_is_renamed_in_place():
    session = _Session(
        [
            (_ALLCAPS, [{"id": 1, "name": "CIRCULAR ECONOMY"}]),
            (_NAMESAKE, []),
            (_SET_NAME, []),
        ]
    )

    report = pp._normalize_all_caps_concepts(session, dry_run=False, apoc_available=True)

    assert report["renamed_nodes"] == 1
    assert session.ran(_SET_NAME)[0][1] == {"id": 1, "name": "Circular Economy"}


def test_an_all_caps_concept_with_a_namesake_is_merged_into_it():
    session = _Session(
        [
            (_ALLCAPS, [{"id": 1, "name": "CIRCULAR ECONOMY"}]),
            (_NAMESAKE, [{"id": 2, "name": "Circular Economy", "degree": 8}]),
            (_REL_COUNT, [{"c": 3}]),
            (_MERGE_NODES, [{"id": 2}]),
        ]
    )

    report = pp._normalize_all_caps_concepts(session, dry_run=False, apoc_available=True)

    assert report["merged_nodes"] == 1
    assert report["edges_modified"] == 3
    assert session.ran(_MERGE_NODES)[0][1] == {"ids": [2, 1], "primary": 2}


def test_a_concept_whose_title_case_is_itself_is_left_alone():
    session = _Session([(_ALLCAPS, [{"id": 1, "name": "2024"}])])

    report = pp._normalize_all_caps_concepts(session, dry_run=False, apoc_available=True)

    assert report["candidates"] == 1
    assert report["renamed_nodes"] == 0
    assert report["merged_nodes"] == 0
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "acronym, becomes", [("CO2", "Co2"), ("EU", "Eu"), ("ISO", "Iso"), ("GHG", "Ghg")]
)
def test_an_acronym_concept_is_title_cased_like_any_other_name(acronym, becomes):
    # Not an endorsement: `str.title()` cannot tell an acronym from a shouted
    # phrase, so this pass renames CO2 to Co2. Pinned as it stands because
    # changing it means deciding what an acronym is, which needs a measurement
    # on the graph and not a guess here.
    session = _Session(
        [(_ALLCAPS, [{"id": 1, "name": acronym}]), (_NAMESAKE, []), (_SET_NAME, [])]
    )

    report = pp._normalize_all_caps_concepts(session, dry_run=False, apoc_available=True)

    assert report["renamed_nodes"] == 1
    assert session.ran(_SET_NAME)[0][1] == {"id": 1, "name": becomes}


def test_an_all_caps_merge_without_apoc_is_recorded_as_skipped():
    session = _Session(
        [
            (_ALLCAPS, [{"id": 1, "name": "CIRCULAR ECONOMY"}]),
            (_NAMESAKE, [{"id": 2, "name": "Circular Economy", "degree": 8}]),
            (_REL_COUNT, [{"c": 3}]),
        ]
    )

    report = pp._normalize_all_caps_concepts(session, dry_run=False, apoc_available=False)

    assert report["skipped"] == 1
    assert report["errors"] == ["APOC unavailable, cannot merge Concept nodes"]


def test_an_all_caps_pass_in_dry_run_writes_nothing():
    session = _Session(
        [
            (_ALLCAPS, [{"id": 1, "name": "CIRCULAR ECONOMY"}, {"id": 3, "name": "RICE HUSK"}]),
            (_NAMESAKE, lambda params: [{"id": 2, "name": params["name"], "degree": 8}] if params["id"] == 1 else []),
            (_REL_COUNT, [{"c": 3}]),
        ]
    )

    report = pp._normalize_all_caps_concepts(session, dry_run=True, apoc_available=True)

    assert (report["merged_nodes"], report["renamed_nodes"]) == (1, 1)
    assert session.ran(_MERGE_NODES) == []
    assert session.ran(_SET_NAME) == []


# --- region artifacts ------------------------------------------------------


_REGION_FIND = "MATCH (n:Region)"
_REGION_DELETE = "MATCH (n:Region) WHERE id(n) = $id DETACH DELETE n"


def test_a_region_artifact_with_a_clean_twin_is_merged_into_it():
    session = _Session(
        [
            (_REL_COUNT, [{"c": 4}]),
            (_MERGE_NODES, [{"id": 2}]),
            (_REGION_FIND, [{"id": 1, "name": "ITALY / EUROPE", "match_id": 2, "match_name": "Italy"}]),
        ]
    )

    report = pp._cleanup_region_artifacts(session, dry_run=False)

    assert (report["matched"], report["rewired_relationships"]) == (1, 4)
    assert report["deleted_nodes"] == 0
    assert session.ran(_MERGE_NODES)[0][1] == {"match_id": 2, "bad_id": 1}


def test_a_region_artifact_with_no_twin_is_detached_and_deleted():
    session = _Session(
        [
            (_REGION_DELETE, []),
            (_REL_COUNT, [{"c": 2}]),
            (_REGION_FIND, [{"id": 1, "name": "SEE FIGURE 3 / MAP", "match_id": None, "match_name": None}]),
        ]
    )

    report = pp._cleanup_region_artifacts(session, dry_run=False)

    assert (report["deleted_nodes"], report["deleted_relationships"]) == (1, 2)
    assert session.ran(_REGION_DELETE)[0][1] == {"id": 1}


def test_a_region_cleanup_in_dry_run_neither_merges_nor_deletes():
    session = _Session(
        [
            (_REL_COUNT, [{"c": 2}]),
            (
                _REGION_FIND,
                [
                    {"id": 1, "name": "A / B", "match_id": 2, "match_name": "B"},
                    {"id": 3, "name": "C * D", "match_id": None, "match_name": None},
                ],
            ),
        ]
    )

    report = pp._cleanup_region_artifacts(session, dry_run=True)

    assert (report["matched"], report["deleted_nodes"]) == (1, 1)
    assert session.ran(_MERGE_NODES) == []
    assert session.ran("DETACH DELETE") == []


# --- MENTIONED_IN ----------------------------------------------------------


_MENTION_COUNT = "`MENTIONED_IN`"
_CONVERT = "SET e[$prop_name] = docs"
_MENTION_DELETE = "MATCH ()-[r:MENTIONED_IN]->() DELETE r"


def test_converting_mentions_writes_the_properties_then_drops_the_edges():
    session = _Session(
        [
            (_MENTION_COUNT, [{"c": 900}]),
            (_CONVERT, [{"nodes_updated": 120}]),
            (_MENTION_DELETE, [{"c": 900}]),
        ]
    )

    report = pp._cleanup_mentioned_in(
        session, "convert", False, "documents", "document_count", "mention_count"
    )

    assert report["nodes_updated"] == 120
    assert report["edges_deleted"] == 900
    assert session.ran(_CONVERT)[0][1] == {
        "prop_name": "documents",
        "count_prop": "document_count",
        "mentions_prop": "mention_count",
    }


def test_dropping_mentions_without_converting_loses_the_provenance():
    # Pinned because it is a real fork: any mode other than "convert" deletes
    # the edges and writes nothing in their place, so where an entity was said
    # is gone. mention_count on the surviving nodes comes from stage 5, not
    # from here.
    session = _Session([(_MENTION_COUNT, [{"c": 900}]), (_MENTION_DELETE, [{"c": 900}])])

    report = pp._cleanup_mentioned_in(
        session, "drop", False, "documents", "document_count", "mention_count"
    )

    assert report["nodes_updated"] == 0
    assert report["edges_deleted"] == 900
    assert session.ran(_CONVERT) == []


def test_a_mentions_dry_run_never_deletes():
    session = _Session([(_MENTION_COUNT, [{"c": 900}]), (_CONVERT, [{"nodes_updated": 120}])])

    report = pp._cleanup_mentioned_in(
        session, "convert", True, "documents", "document_count", "mention_count"
    )

    assert report["nodes_updated"] == 120
    assert report["edges_deleted"] == 0
    assert session.ran(_MENTION_DELETE) == []


def test_no_mention_edges_means_the_pass_stops_immediately():
    session = _Session([(_MENTION_COUNT, [{"c": 0}])])

    report = pp._cleanup_mentioned_in(
        session, "convert", False, "documents", "document_count", "mention_count"
    )

    assert report["edges_deleted"] == 0
    assert len(session.calls) == 1


def test_a_mentions_count_that_fails_stops_the_pass_before_the_delete():
    session = _Session([(_MENTION_COUNT, RuntimeError("boom"))])

    report = pp._cleanup_mentioned_in(
        session, "convert", False, "documents", "document_count", "mention_count"
    )

    assert "count failed" in report["errors"][0]
    assert session.ran(_MENTION_DELETE) == []


# --- configuration ---------------------------------------------------------


def test_the_default_vocabulary_always_offers_a_fallback_type():
    vocab = pp._load_relation_vocab("")

    assert "RELATED_TO" in vocab
    assert vocab == list(pp._CANONICAL_RELATION_TYPES)


def test_a_custom_vocabulary_is_upper_cased_and_gets_related_to(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(["uses", " affects ", ""]), encoding="utf-8")

    assert pp._load_relation_vocab(str(path)) == ["RELATED_TO", "USES", "AFFECTS"]


def test_a_vocabulary_that_is_not_a_list_is_rejected(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"uses": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array"):
        pp._load_relation_vocab(str(path))


def test_a_property_schema_entry_that_is_not_a_mapping_is_dropped(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps({"Region": {"country": "the country"}, "Broken": ["nope"]}),
        encoding="utf-8",
    )

    assert pp._load_property_schema(str(path)) == {"Region": {"country": "the country"}}


def test_a_property_schema_that_is_not_an_object_is_rejected(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(["nope"]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        pp._load_property_schema(str(path))


def test_the_default_property_schema_is_a_copy_not_the_module_global():
    schema = pp._load_property_schema("")
    schema["Region"] = {}

    assert pp._DEFAULT_PROPERTY_SCHEMA["Region"] != {}


def test_a_yaml_config_is_read_as_a_mapping(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("neo4j:\n  database: neo4j\n", encoding="utf-8")

    assert pp._load_yaml(path) == {"neo4j": {"database": "neo4j"}}


# --- properties ------------------------------------------------------------


def test_property_values_neo4j_cannot_store_are_flattened_or_dropped():
    out = pp._sanitize_props(
        {
            "name": "Rice husk",
            "count": 3,
            "ok": True,
            "nothing": None,
            "blank": "   ",
            "tags": ["a", None, "b"],
            "obj": {"page": 3},
        }
    )

    assert out == {
        "name": "Rice husk",
        "count": 3,
        "ok": True,
        "tags": ["a", "b"],
        "obj": "{'page': 3}",
    }


def test_a_set_becomes_a_list_because_neo4j_has_no_sets():
    assert sorted(pp._coerce_value({"b", "a"})) == ["a", "b"]


# --- APOC probe and error collection ---------------------------------------


def test_the_apoc_probe_reports_false_instead_of_raising():
    assert pp._has_apoc(_Session([("apoc.version", RuntimeError("unknown function"))])) is False
    assert pp._has_apoc(_Session([("apoc.version", [{"version": "5.20"}])])) is True


def test_every_error_anywhere_in_a_nested_report_is_collected():
    report = {
        "compaction": {"errors": ["a"], "nested": {"errors": ["b"]}},
        "passes": [{"errors": ["c"]}, {"errors": []}],
        "count": 3,
        "errors": ["d"],
    }

    assert sorted(pp._collect_errors(report)) == ["a", "b", "c", "d"]


def test_a_report_with_nothing_wrong_collects_nothing():
    assert pp._collect_errors({"count": 3, "pairs": [{"source": "A"}]}) == []


# --- promoting RELATED_TO into a real predicate ----------------------------

# These two passes are what decides whether an edge the extractor could not
# name stays anonymous or becomes USES / PART_OF / LOCATED_IN. They act on
# relationship ids the model hands back, so what matters is what they refuse.

_RELATED_IDS = "MATCH ()-[r:RELATED_TO]->() RETURN id(r) AS id"
_RELATED_CTX = "MATCH (s)-[r:RELATED_TO]->(t)"
_SET_TYPE = "apoc.refactor.setType"


def _refine(ids, replies, *, dry_run=False, updated=None):
    """Run the RELATED_TO refinement over `ids` with scripted `replies`."""
    session = _Session(
        [
            (_SET_TYPE, [{"updated": len(ids) if updated is None else updated}]),
            (_RELATED_CTX, [{"id": i} for i in ids]),
            (_RELATED_IDS, [{"id": i} for i in ids]),
        ]
    )
    report = pp._refine_related_to_relationships(
        session=session,
        canonical=CANONICAL,
        client=_FakeLLM(replies),
        model_name="m",
        dry_run=dry_run,
    )
    return session, report


def test_an_anonymous_edge_the_model_can_name_is_retyped():
    session, report = _refine([1, 2], ['[{"id": 1, "type": "USES"}, {"id": 2, "type": "PART_OF"}]'])

    assert report["updated"] == 2
    assert report["type_counts"] == {"USES": 1, "PART_OF": 1}
    assert session.ran(_SET_TYPE)[0][1]["updates"] == [
        {"id": 1, "type": "USES"},
        {"id": 2, "type": "PART_OF"},
    ]


def test_an_edge_the_model_leaves_anonymous_is_counted_not_rewritten():
    session, report = _refine([1], ['[{"id": 1, "type": "RELATED_TO"}]'])

    assert (report["updated"], report["skipped"]) == (0, 1)
    assert session.ran(_SET_TYPE) == []


def test_a_type_outside_the_vocabulary_leaves_the_edge_anonymous():
    session, report = _refine([1], ['[{"id": 1, "type": "MITIGATES_RISK"}]'])

    assert report["type_counts"] == {"RELATED_TO": 1}
    assert session.ran(_SET_TYPE) == []


def test_a_relationship_id_the_batch_never_asked_about_is_refused():
    # The update matches on id alone, so an invented id would retype whatever
    # edge happens to carry it — including one that was never RELATED_TO.
    session, report = _refine([1], ['[{"id": 999, "type": "USES"}]'])

    assert report["skipped"] == 1
    assert report["updated"] == 0
    assert session.ran(_SET_TYPE) == []


@pytest.mark.parametrize("row", ['{"id": "abc", "type": "USES"}', '{"type": "USES"}', '{"id": -3, "type": "USES"}'])
def test_a_row_with_no_usable_id_is_skipped(row):
    _, report = _refine([1], [f"[{row}]"])

    assert report["skipped"] == 1
    assert report["updated"] == 0


def test_a_refinement_dry_run_names_nothing():
    session, report = _refine([1], ['[{"id": 1, "type": "USES"}]'], dry_run=True)

    assert report["updated"] == 0
    assert report["type_counts"] == {"USES": 1}  # what it would have written
    assert session.ran(_SET_TYPE) == []


def test_a_graph_with_no_anonymous_edges_ends_the_pass_at_once():
    session, report = _refine([], [])

    assert report == {
        "total_related_to": 0,
        "updated": 0,
        "skipped": 0,
        "type_counts": {},
        "batches": 0,
        "errors": [],
    }
    assert len(session.calls) == 1


def test_a_model_failure_during_refinement_leaves_the_batch_untouched():
    session, report = _refine([1], [RuntimeError("connection refused")])

    assert "RELATED_TO refinement failed" in report["errors"][0]
    assert report["updated"] == 0
    assert session.ran(_SET_TYPE) == []


def test_a_failed_refinement_write_is_recorded_not_counted():
    session = _Session(
        [
            (_SET_TYPE, RuntimeError("no such relationship")),
            (_RELATED_CTX, [{"id": 1}]),
            (_RELATED_IDS, [{"id": 1}]),
        ]
    )

    report = pp._refine_related_to_relationships(
        session=session,
        canonical=CANONICAL,
        client=_FakeLLM(['[{"id": 1, "type": "USES"}]']),
        model_name="m",
        dry_run=False,
    )

    assert report["updated"] == 0
    assert "RELATED_TO update failed" in report["errors"][0]


# --- the generic reclassifier ----------------------------------------------


def _reclass(ids, replies, *, rel_type="HAS_COMPONENT", skip=None, dry_run=False, batch_size=50):
    """Run the generic reclassifier over `ids` with scripted `replies`."""
    session = _Session(
        [
            (_SET_TYPE, [{"updated": len(ids)}]),
            (f"[r:`{rel_type}`]->(t)", [{"id": i, "current_type": rel_type} for i in ids]),
        ]
    )
    report = pp._reclassify_relationships(
        session=session,
        rel_ids=ids,
        rel_type=rel_type,
        allowed=CANONICAL,
        client=_FakeLLM(replies),
        model_name="m",
        dry_run=dry_run,
        batch_size=batch_size,
        skip_when_type=skip,
    )
    return session, report


def test_reclassification_ignores_ids_outside_the_batch():
    session, report = _reclass([1], ['[{"id": 1, "type": "USES"}, {"id": 999, "type": "PART_OF"}]'])

    assert session.ran(_SET_TYPE)[0][1]["updates"] == [{"id": 1, "type": "USES"}]
    assert report["type_counts"] == {"USES": 1}


def test_an_edge_the_model_never_answered_about_counts_as_skipped():
    _, report = _reclass([1, 2], ['[{"id": 1, "type": "USES"}]'])

    assert report["skipped"] == 1


def test_reclassification_can_be_told_to_leave_one_type_alone():
    session, report = _reclass(
        [1, 2],
        ['[{"id": 1, "type": "RELATED_TO"}, {"id": 2, "type": "USES"}]'],
        rel_type="RELATED_TO",
        skip="RELATED_TO",
    )

    assert report["skipped"] == 1
    assert session.ran(_SET_TYPE)[0][1]["updates"] == [{"id": 2, "type": "USES"}]


def test_reclassification_splits_into_batches_and_survives_one_failing():
    session = _Session(
        [
            (_SET_TYPE, [{"updated": 1}]),
            ("[r:`HAS_COMPONENT`]->(t)", lambda p: [{"id": i} for i in p["ids"]]),
        ]
    )

    report = pp._reclassify_relationships(
        session=session,
        rel_ids=[1, 2, 3, 4],
        rel_type="HAS_COMPONENT",
        allowed=CANONICAL,
        client=_FakeLLM(
            [RuntimeError("timeout"), '[{"id": 3, "type": "USES"}, {"id": 4, "type": "USES"}]']
        ),
        model_name="m",
        dry_run=False,
        batch_size=2,
    )

    assert report["batches"] == 2
    assert len(report["errors"]) == 1
    assert report["updated"] == 1  # the second batch still landed
    # The lost batch shows up in `errors` and nowhere else: a failed batch is
    # not counted as skipped, so total_candidates > updated + skipped whenever
    # one dies. Reading the report without reading `errors` overstates the run.
    assert report["skipped"] == 0
    assert report["total_candidates"] == 4


def test_reclassification_of_an_empty_candidate_set_queries_nothing():
    session = _Session([])

    report = pp._reclassify_relationships(
        session=session,
        rel_ids=[],
        rel_type="HAS_COMPONENT",
        allowed=CANONICAL,
        client=_FakeLLM([]),
        model_name="m",
        dry_run=False,
        batch_size=50,
    )

    assert report["total_candidates"] == 0
    assert session.calls == []


def test_the_anomaly_pass_reclassifies_only_the_two_shapes_it_looks_for():
    session = _Session(
        [
            (_SET_TYPE, [{"updated": 1}]),
            ("[r:`HAS_COMPONENT`]->(t)", [{"id": 5, "current_type": "HAS_COMPONENT"}]),
            ("(s:Organization)-[r:HAS_COMPONENT]->(t:Concept)", [{"id": 5}]),
        ]
    )

    report = pp._reclassify_has_component_anomalies(
        session=session, client=_FakeLLM(['[{"id": 5, "type": "USES"}]']),
        model_name="m", dry_run=False, batch_size=50,
    )

    assert report["updated"] == 1
    assert "UNION" in session.calls[0][0]


def test_the_second_related_to_pass_never_rewrites_an_edge_back_to_related_to():
    session = _Session(
        [
            (_SET_TYPE, [{"updated": 1}]),
            ("[r:`RELATED_TO`]->(t)", [{"id": 7, "current_type": "RELATED_TO"}]),
            (_RELATED_IDS, [{"id": 7}]),
        ]
    )

    report = pp._reclassify_related_to_second_pass(
        session=session, client=_FakeLLM(['[{"id": 7, "type": "RELATED_TO"}]']),
        model_name="m", allowed=CANONICAL, dry_run=False, batch_size=50,
    )

    assert report["skipped"] == 1
    assert session.ran(_SET_TYPE) == []


# --- prompts ---------------------------------------------------------------


def test_every_prompt_states_the_vocabulary_it_expects_back():
    # A prompt that forgets to list the allowed types gets free-text back, and
    # everything downstream then folds it to RELATED_TO.
    items = [{"type": "WEIRD", "count": 3, "patterns": []}]
    rows = [{"id": 1, "source": {"name": "a", "labels": ["Concept"]},
             "target": {"name": "b", "labels": ["Concept"]},
             "source_context": [], "target_context": []}]

    assert "USES" in pp._relation_mapping_prompt(CANONICAL, items)
    assert "USES" in pp._related_to_refinement_prompt(CANONICAL, rows)
    assert "USES" in pp._relation_reclass_prompt(CANONICAL, rows)
    assert "Concept" in pp._classify_concepts_prompt(["Concept"], [{"id": 1, "name": "x"}])
