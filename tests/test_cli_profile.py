"""`--profile` must be a pure abbreviation of the flags it replaces.

The point of the flag is to let a runner drop sixteen arguments. That is only
worth doing if the short form and the long form resolve to the same
configuration, so these tests compare them field by field rather than trusting
the substitution.

Precedence is argparse's own: `_parse_args` installs the profile with
`set_defaults` and parses the same argv again, so a flag the caller typed wins
over the profile. The alternative -- working out which flags were typed -- is a
distinction argparse does not expose, and every way of reconstructing it treats
"the user passed the default value on purpose" as indistinguishable from "the
user passed nothing".
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from graphrag.cli import _build_arg_parser, _build_base_config, _parse_args
from graphrag.config import AgentConfig
from graphrag.profiles import PROFILES, build_config

# What a campaign passes that a profile cannot carry: question set, model,
# endpoints, output location.
INFRASTRUCTURE_ARGS = [
    "--experiment",
    "--questions-file", "evaluation/gold/gold_v3.json",
    "--strategies", "default,hybrid",
    "--llm", "--vllm",
    "--vllm-base-url", "http://localhost:8000/v1",
    "--model-id", "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "--max-new-tokens", "1024",
    "--text-docs-dir", "artifacts/corpus_circular22",
    "--output-dir", "/tmp/does-not-need-to-exist",
    "--experiment-tag", "probe",
]

# The long form: the configuration flags `--profile` stands in for.
LONG_FORM_ARGS = INFRASTRUCTURE_ARGS + [
    "--max-context-tokens", "6000",
    "--complexity", "medium",
    "--enforce-language",
    "--cite-evidence",
    "--citation-policy", "mark",
    "--citation-display", "label",
    "--prefer-verbatim-definitions",
    "--text-retriever-backend", "tfidf",
    "--text-retriever-mmr",
    "--text-retriever-mmr-lambda", "0.7",
    "--text-retriever-max-per-doc", "2",
    "--vector-retrieval",
    "--seed-from-retrieved",
    "--subgraph-seed-count", "3",
    "--focused-answer",
    "--evidence-max-triple-items", "30",
]

SHORT_FORM_ARGS = ["--profile", "thesis_campaign"] + INFRASTRUCTURE_ARGS

PER_RUN_FIELDS = ("query", "entity")


def _config(argv: list[str]) -> dict:
    """The run configuration argv resolves to, without per-run fields, as JSON."""
    args = _parse_args(_build_arg_parser(), argv)
    raw = dataclasses.asdict(_build_base_config(args))
    for field in PER_RUN_FIELDS:
        raw.pop(field, None)
    return json.loads(json.dumps(raw, default=str))


def test_short_form_equals_the_flags_it_replaces() -> None:
    """The substitution the runner scripts make is an identity."""
    long_form = _config(LONG_FORM_ARGS)
    short_form = _config(SHORT_FORM_ARGS)
    differing = {
        key: {"long": long_form[key], "short": short_form[key]}
        for key in long_form
        if long_form[key] != short_form[key]
    }
    assert not differing, f"--profile is not equivalent to the long form: {differing}"


def test_short_form_equals_the_profile_itself() -> None:
    """The CLI path and the library path agree."""
    from_cli = _config(SHORT_FORM_ARGS)
    from_library = json.loads(
        json.dumps(dataclasses.asdict(build_config("thesis_campaign")), default=str)
    )
    for field in PER_RUN_FIELDS:
        from_library.pop(field, None)
    assert from_cli == from_library


def test_explicit_flag_beats_the_profile() -> None:
    args = _parse_args(
        _build_arg_parser(),
        ["--profile", "thesis_campaign", "--subgraph-seed-count", "7"],
    )
    assert args.subgraph_seed_count == 7


def test_profile_applies_where_no_flag_was_given() -> None:
    args = _parse_args(_build_arg_parser(), ["--profile", "thesis_campaign"])
    assert args.subgraph_seed_count == 3
    assert args.vector_retrieval is True
    assert args.citation_display == "label"


def test_without_a_profile_nothing_changes() -> None:
    """The regression guard: the flag must be inert when unused.

    Every run that does not pass `--profile` goes through this path.
    """
    args = _parse_args(_build_arg_parser(), [])
    assert args.profile is None
    assert args.subgraph_seed_count == 1
    assert args.vector_retrieval is False
    assert args.citation_display == "id"

    default = json.loads(
        json.dumps(dataclasses.asdict(AgentConfig()), default=str)
    )
    produced = _config([])
    for field in PER_RUN_FIELDS:
        default.pop(field, None)
    assert produced == default


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse_args(_build_arg_parser(), ["--profile", "nope"])


def test_profile_with_no_flag_for_a_field_fails_loudly() -> None:
    """`demo` sets two fields the CLI cannot express; dropping them silently
    would hand the caller a configuration that is not the profile they named."""
    with pytest.raises(SystemExit):
        _parse_args(_build_arg_parser(), ["--profile", "demo"])


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_is_either_expressible_or_rejected(profile: str) -> None:
    """No profile may half-apply: it either resolves fully or exits."""
    try:
        args = _parse_args(_build_arg_parser(), ["--profile", profile])
    except SystemExit:
        return
    for field, value in PROFILES[profile].items():
        dest = "max_context_tokens" if field == "max_content_tokens" else field
        expected = value.value if hasattr(value, "value") else value
        assert getattr(args, dest) == expected, f"{profile}: {field} did not apply"
