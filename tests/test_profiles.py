"""The named profiles must reproduce what the campaigns actually ran.

`THESIS_CAMPAIGN` is checked against `tests/fixtures/thesis_campaign_config.json`,
which is the `strategy_configs` block written by `graphrag.cli --experiment` for
a real thesis run (gemma3_27b_awq, 2026-08-18). If a profile field drifts from
what that campaign was measured with, these tests fail -- which is the point:
the profile is only useful if switching a runner over to it cannot move a
published number.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from graphrag.config import AgentConfig
from graphrag.profiles import (
    DEMO,
    PROFILES,
    RESEARCH_BASELINE,
    THESIS_CAMPAIGN,
    build_config,
)
from graphrag.strategies import STRATEGY_PRESETS

FIXTURE = Path(__file__).parent / "fixtures" / "thesis_campaign_config.json"

# Per-run values, not part of any profile: the fixture has them stripped.
PER_RUN_FIELDS = ("query", "entity")


def _recorded() -> dict[str, dict]:
    """The `strategy_configs` block of the recorded campaign fixture."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["strategy_configs"]


def _serialise(config: AgentConfig) -> dict:
    """Render a config the way `graphrag.cli` writes it into config.json."""
    raw = dataclasses.asdict(config)
    for field in PER_RUN_FIELDS:
        raw.pop(field, None)
    # cli.py dumps with `default=str`, which is what turns enums into
    # "OUTPUT_TONE.TECHNICAL" and tuples into lists.
    return json.loads(json.dumps(raw, default=str))


@pytest.mark.parametrize("strategy", sorted(_recorded()))
def test_thesis_campaign_reproduces_the_recorded_run(strategy: str) -> None:
    """Every field of every strategy matches the campaign that was published."""
    expected = _recorded()[strategy]
    actual = _serialise(build_config("thesis_campaign", strategy=strategy))

    missing = sorted(set(expected) - set(actual))
    assert not missing, f"fields the recorded run had and the profile no longer sets: {missing}"

    # A field added to `AgentConfig` after that run did not exist in its
    # config.json, and the fixture is the record of what ran — editing it in
    # would falsify the record. What has to hold instead is that such a field is
    # inert: at its dataclass default it cannot move a published number, which is
    # the guarantee this test exists for. A new field that is *not* at its
    # default changes the campaign and fails here.
    defaults = _serialise(AgentConfig())
    added = {
        field: actual[field]
        for field in set(actual) - set(expected)
        if actual[field] != defaults[field]
    }
    assert not added, f"{strategy}: fields added since the recorded run are not inert: {added}"

    differing = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    assert not differing, f"{strategy}: profile differs from the recorded run: {differing}"


def test_fixture_covers_every_strategy() -> None:
    """The contract is only as strong as its coverage of the preset list."""
    assert set(_recorded()) == set(STRATEGY_PRESETS)


def test_profiles_only_declare_real_fields() -> None:
    """A typo in a profile key would otherwise surface as a silent no-op."""
    fields = {f.name for f in AgentConfig.__dataclass_fields__.values()}
    for name, profile in PROFILES.items():
        unknown = sorted(set(profile) - fields)
        assert not unknown, f"profile '{name}' names non-existent field(s): {unknown}"


def test_profiles_only_declare_actual_overrides() -> None:
    """A profile entry equal to the library default is noise that will rot.

    The campaign shell scripts pass several flags whose values are already the
    default; carrying them here would make the profile look like it controls
    something it does not.
    """
    default = AgentConfig()
    for name, profile in PROFILES.items():
        redundant = [k for k, v in profile.items() if getattr(default, k) == v]
        assert not redundant, f"profile '{name}' restates library defaults: {redundant}"


def test_research_baseline_is_the_library_default() -> None:
    assert RESEARCH_BASELINE == {}
    assert _serialise(build_config("research_baseline")) == _serialise(AgentConfig())


def test_demo_differs_from_the_campaign_where_it_should() -> None:
    """The demo is not the campaign renamed; the differences are deliberate."""
    # A demo may abstain and may answer from parametric knowledge; a measurement
    # run may do neither.
    assert DEMO["enable_domain_gate"] is True
    assert DEMO["allow_parametric_fallback"] is True
    assert "enable_domain_gate" not in THESIS_CAMPAIGN
    assert "allow_parametric_fallback" not in THESIS_CAMPAIGN
    # The demo does not narrow the answer or re-seed graph expansion, so a demo
    # answer is not a retrieval measurement.
    for field in ("focused_answer", "seed_from_retrieved", "subgraph_seed_count"):
        assert field in THESIS_CAMPAIGN
        assert field not in DEMO


def test_build_config_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        build_config("does_not_exist")


def test_build_config_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unknown AgentConfig field"):
        build_config("demo", nonexistent_field=1)


def test_build_config_does_not_mutate_the_profile() -> None:
    before = dict(THESIS_CAMPAIGN)
    build_config("thesis_campaign", strategy="hybrid", query="x")
    assert THESIS_CAMPAIGN == before


def test_overrides_win_over_profile_and_strategy() -> None:
    config = build_config("thesis_campaign", strategy="text_only", include_nodes=True)
    # text_only switches include_nodes off; the explicit override comes last.
    assert config.include_nodes is True
    assert build_config("thesis_campaign", strategy="text_only").include_nodes is False


def test_demo_profile_matches_the_product() -> None:
    """The profile that claims to describe the demo has to describe it.

    Without `text_retriever_backend`, the profile would resolve to the library
    default `tfidf` while `product/config.py` builds the text pipeline with
    `dense`. This test is the comparison that catches it.
    """
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from product import config as product_config

    # Every field the profile declares, against the constant the demo runs on.
    declared_by_the_demo = {
        "complexity": product_config.COMPLEXITY,
        "always_include_limits": product_config.ALWAYS_LIMITS,
        "cite_evidence": product_config.CITE_EVIDENCE,
        "citation_display": product_config.CITATION_DISPLAY,
        "enforce_language": product_config.ENFORCE_LANGUAGE,
        "prefer_verbatim_definitions": product_config.VERBATIM_DEFINITIONS,
        "vector_retrieval": product_config.VECTOR_RETRIEVAL,
        "enable_domain_gate": product_config.DOMAIN_GATE,
        "answer_meta_questions": product_config.META_REPLY,
        "allow_parametric_fallback": product_config.PARAMETRIC_FALLBACK,
        "text_retriever_top_k": product_config.TEXT_TOP_K,
        "text_retriever_mmr": product_config.TEXT_MMR,
        "text_retriever_max_per_doc": product_config.TEXT_MAX_PER_DOC,
        "text_retriever_backend": product_config.TEXT_RETRIEVER_BACKEND,
    }
    differing = {
        field: {"profile": DEMO.get(field, "<assente>"), "demo": value}
        for field, value in declared_by_the_demo.items()
        if DEMO.get(field) != value
    }

    assert not differing, f"the demo profile diverges from product/config.py: {differing}"
