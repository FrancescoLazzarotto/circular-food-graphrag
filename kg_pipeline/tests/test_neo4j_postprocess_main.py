"""Which passes run, in what order, and what a dry run is allowed to touch.

The individual passes are covered in `test_neo4j_postprocess.py`. This covers
the orchestration above them that decides *which* of them runs: a pass running
when it should not is as destructive as a pass behaving badly.

No Neo4j, no LLM: the driver, the model client and every pass are replaced by
recorders, so what the tests see is the sequence of decisions `main()` makes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kg_pipeline.stages import neo4j_postprocess as pp

# Every pass `main()` can dispatch to. Replacing all of them at once means a
# test names only the flags; anything that runs shows up in `calls`.
_PASSES = [
    "_apply_relation_mapping",
    "_rewrite_inverse_relationships",
    "_invert_published_direction",
    "_cleanup_named_region_nodes",
    "_find_duplicate_groups",
    "_merge_duplicate_groups",
    "_bridge_duplicate_name_groups",
    "_cleanup_isolated_nodes",
    "_normalize_all_caps_concepts",
    "_classify_concepts",
    "_enrich_properties",
    "_refine_related_to_relationships",
    "_reclassify_has_component_anomalies",
    "_reclassify_related_to_second_pass",
    "_rename_relation_types",
    "_absorb_micro_relation_types",
    "_cleanup_region_artifacts",
    "_cleanup_mentioned_in",
    "_run_aura_issues",
    "_run_verbose_relation_cleanup",
    "_run_cleanup_pass3",
    "_run_semantic_compaction",
    "_compact_relation_types_deterministic",
    "_apply_constraints",
]


class _Session:
    """Session answering the APOC version probe, and nothing else."""

    def run(self, cypher: str, **params: Any):
        class _R:
            def single(self_inner):
                return {"version": "5.20"}

            def data(self_inner):
                return []

            def consume(self_inner):
                return None

            def __iter__(self_inner):
                return iter(())

        return _R()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Driver:
    """Driver handing out sessions and recording the database asked for."""

    def __init__(self) -> None:
        self.databases: list[Any] = []

    def session(self, database=None, **kwargs: Any):
        self.databases.append(database)
        return _Session()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def run_main(monkeypatch, tmp_path, capsys):
    """Runs `main()` with everything below it replaced, and reports what ran."""
    calls: list[str] = []
    dry_runs: dict[str, Any] = {}
    driver = _Driver()

    config = tmp_path / "config.yaml"
    config.write_text(
        "ontology:\n  labels: [Concept, Material]\nllm:\n  relation_vocab_path: ''\n",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")

    # The orchestrator reads specific fields out of each result to log them,
    # and reads `skipped` as a list in one place and an int in another. A
    # recorder returning the wrong shape fails on the logging line rather than
    # on the thing under test, so the shapes are spelled out.
    _COUNTED = {
        "count": 0, "rewritten": 0, "candidates": 0, "matched": 0,
        "deleted_nodes": 0, "deleted_relationships": 0, "rewired_relationships": 0,
        "total_candidates": 0, "updated": 0, "relabeled": 0, "updated_nodes": 0,
        "groups": 0, "merged_nodes": 0, "skipped_incompatible": 0,
        "errors": [],
    }
    _LISTED = {
        "errors": [], "pairs": [], "renamed": [], "created": [], "skipped": [],
        "groups": 0, "merged_nodes": 0, "skipped_incompatible": 0,
        "candidates": 0, "relabeled": 0, "updated_nodes": 0,
    }
    _AURA = {
        "step1_published_direction_fix": dict(_COUNTED),
        "step2_region_garbage_cleanup": dict(_COUNTED),
        "step3_inverse_pairs": {"pairs": [], "errors": []},
        "step4_has_component_reclass": dict(_COUNTED, skipped=0),
        "step5_related_to_reclass": dict(_COUNTED, skipped=0),
        "errors": [],
    }
    # `skipped` is a list for the mapping and constraint steps and a count for
    # the reclassification ones — the orchestrator logs it both ways.
    _RETURNS = {
        "_run_aura_issues": _AURA,
        "_reclassify_has_component_anomalies": dict(_COUNTED, skipped=0),
        "_reclassify_related_to_second_pass": dict(_COUNTED, skipped=0),
        "_refine_related_to_relationships": dict(_COUNTED, skipped=0),
    }

    def _recorder(name: str):
        def _call(*args: Any, **kwargs: Any):
            calls.append(name)
            if "dry_run" in kwargs:
                dry_runs[name] = kwargs["dry_run"]
            return _RETURNS.get(name, dict(_LISTED))

        return _call

    for name in _PASSES:
        monkeypatch.setattr(pp, name, _recorder(name), raising=False)

    monkeypatch.setattr(pp, "_has_apoc", lambda session: True)
    monkeypatch.setattr(
        pp, "_resolve_neo4j_env", lambda: ("bolt://localhost:7689", "neo4j", "pw", "neo4j")
    )
    monkeypatch.setattr(
        pp, "_resolve_llm_env", lambda: ("http://localhost:8000/v1", "m", "EMPTY")
    )
    monkeypatch.setattr(pp, "_build_llm_client", lambda base_url, api_key: object())
    monkeypatch.setattr(pp.neo4j_env, "connect", lambda target: driver)
    monkeypatch.setattr(pp, "_setup_logging", lambda path: None)
    monkeypatch.setattr(pp, "_fetch_relation_types", lambda session, max_patterns: [])
    monkeypatch.setattr(pp, "_fetch_related_to_ids", lambda session: [])
    monkeypatch.setattr(pp, "_fetch_has_component_anomaly_ids", lambda session: [])

    confirmations: list[dict[str, Any]] = []
    monkeypatch.setattr(
        pp,
        "_confirm_db_changes",
        lambda **kwargs: confirmations.append(kwargs),
    )

    def _go(*flags: str):
        calls.clear()
        argv = [
            "neo4j_postprocess",
            "--config", str(config),
            "--env-file", str(env_file),
            "--log-file", str(tmp_path / "pp.log"),
            *flags,
        ]
        monkeypatch.setattr("sys.argv", argv)
        pp.main()
        out = capsys.readouterr().out
        report = json.loads(out[out.index("{") :]) if "{" in out else {}
        return report

    _go.calls = calls
    _go.dry_runs = dry_runs
    _go.driver = driver
    _go.confirmations = confirmations
    return _go


# --- one task at a time ----------------------------------------------------


@pytest.mark.parametrize(
    "fix, expected",
    [
        ("aura-issues", "_run_aura_issues"),
        ("mentioned-in", "_cleanup_mentioned_in"),
        ("region-artifacts", "_cleanup_region_artifacts"),
        ("micro-types", "_absorb_micro_relation_types"),
        ("related-to", "_refine_related_to_relationships"),
        ("cleanup-pass3", "_run_cleanup_pass3"),
        ("compact-semantic", "_run_semantic_compaction"),
    ],
)
def test_asking_for_one_task_runs_that_one(run_main, fix, expected):
    run_main("--fix", fix, "--dry-run", "--yes")

    assert expected in run_main.calls


@pytest.mark.parametrize(
    "fix", ["aura-issues", "mentioned-in", "region-artifacts", "micro-types"]
)
def test_asking_for_one_task_does_not_run_the_whole_pipeline(run_main, fix):
    # `--fix` says "skip the default pipeline". A task flag that still ran the
    # rest is how a targeted repair becomes a full rewrite of the graph.
    run_main("--fix", fix, "--dry-run", "--yes")

    assert "_apply_constraints" not in run_main.calls
    assert "_merge_duplicate_groups" not in run_main.calls


def test_an_unknown_task_is_refused_by_the_parser(run_main):
    with pytest.raises(SystemExit):
        run_main("--fix", "invent-a-pass", "--dry-run", "--yes")


# --- the default pipeline --------------------------------------------------


def test_without_a_task_the_whole_pipeline_runs(run_main):
    run_main("--dry-run", "--yes")

    assert "_apply_relation_mapping" in run_main.calls
    assert "_apply_constraints" in run_main.calls


def test_the_pipeline_keeps_its_order(run_main):
    # Steps are numbered in the report for a reason: dedup before relabel
    # before enrichment before constraints. A reordering changes what each
    # step sees.
    run_main("--dry-run", "--yes")

    order = [c for c in run_main.calls if c in {
        "_apply_relation_mapping",
        "_merge_duplicate_groups",
        "_enrich_properties",
        "_apply_constraints",
    }]
    assert order == [
        "_apply_relation_mapping",
        "_merge_duplicate_groups",
        "_enrich_properties",
        "_apply_constraints",
    ]


def test_the_report_names_every_step_that_ran(run_main):
    report = run_main("--dry-run", "--yes")

    assert report["dry_run"] is True
    assert report["apoc_available"] is True
    assert any(key.startswith("step1") for key in report)


# --- a dry run is a dry run ------------------------------------------------


def test_a_dry_run_tells_every_pass_it_is_one(run_main):
    run_main("--dry-run", "--yes")

    assert run_main.dry_runs
    assert all(value is True for value in run_main.dry_runs.values())


def test_a_real_run_tells_every_pass_it_is_one(run_main):
    run_main("--yes")

    assert run_main.dry_runs
    assert all(value is False for value in run_main.dry_runs.values())


def test_the_dry_run_flag_reaches_a_single_task_too(run_main):
    run_main("--fix", "mentioned-in", "--dry-run", "--yes")

    assert run_main.dry_runs["_cleanup_mentioned_in"] is True


# --- which graph, and who said so ------------------------------------------


def test_the_target_is_confirmed_before_anything_runs(run_main):
    run_main("--dry-run", "--yes")

    assert run_main.confirmations
    assert run_main.confirmations[0]["uri"] == "bolt://localhost:7689"
    assert run_main.confirmations[0]["assume_yes"] is True


def test_the_database_flag_wins_over_the_environment(run_main):
    run_main("--database", "staging", "--dry-run", "--yes")

    assert run_main.driver.databases[0] == "staging"
    assert run_main.confirmations[0]["database"] == "staging"


def test_without_the_flag_the_environment_names_the_database(run_main):
    run_main("--dry-run", "--yes")

    assert run_main.driver.databases[0] == "neo4j"


# --- what a task needs -----------------------------------------------------


def test_a_task_that_needs_no_model_does_not_build_one(run_main, monkeypatch):
    def _boom():
        raise AssertionError("the LLM settings should not be read")

    monkeypatch.setattr(pp, "_resolve_llm_env", _boom)

    run_main("--fix", "mentioned-in", "--dry-run", "--yes")

    assert "_cleanup_mentioned_in" in run_main.calls


@pytest.mark.parametrize("fix", ["related-to", "aura-issues", "cleanup-pass3"])
def test_a_task_that_needs_a_model_reads_its_settings(run_main, monkeypatch, fix):
    asked: list[int] = []
    monkeypatch.setattr(
        pp,
        "_resolve_llm_env",
        lambda: asked.append(1) or ("http://localhost:8000/v1", "m", "EMPTY"),
    )

    run_main("--fix", fix, "--dry-run", "--yes")

    assert asked == [1]


def test_a_missing_model_setting_stops_the_run(run_main, monkeypatch):
    monkeypatch.setattr(
        pp,
        "_resolve_llm_env",
        lambda: (_ for _ in ()).throw(ValueError("Missing VLLM_MODEL_NAME")),
    )

    with pytest.raises(ValueError, match="VLLM_MODEL_NAME"):
        run_main("--dry-run", "--yes")


# --- the exit status -------------------------------------------------------


def test_a_run_whose_steps_all_worked_exits_quietly(run_main):
    run_main("--dry-run", "--yes")  # no SystemExit


def test_a_step_that_reported_errors_makes_the_run_fail(run_main, monkeypatch):
    # Every step collects its failures into `errors`; exiting 0 regardless would
    # let a pass that renamed nothing, because every APOC call failed, look
    # clean to anything reading the status.
    def _broken(*args: Any, **kwargs: Any):
        return {"errors": ["APOC unavailable"], "pairs": [], "renamed": []}

    monkeypatch.setattr(pp, "_apply_constraints", _broken)

    with pytest.raises(SystemExit) as excinfo:
        run_main("--dry-run", "--yes")

    assert excinfo.value.code == 1
