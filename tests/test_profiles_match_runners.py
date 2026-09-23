"""The profile must equal what the runner scripts actually pass.

`test_profiles.py` checks `THESIS_CAMPAIGN` against a configuration recorded by a
run that already happened. This file checks it against the scripts that will
produce the next one: it reads the real flag block out of each shell runner, lets
bash expand the variable defaults, feeds the result through the CLI's own parser
and config builder, and compares field by field.

Together the two files close the loop from both ends. That is what makes
switching a runner over to `--profile thesis_campaign` a verifiable change rather
than a hopeful one: if the substitution would move any of the 86 fields, this
test says which.

The tests shell out to bash because bash is the authority on its own `${VAR:-default}`
syntax, and reparsing it in Python would only prove that the reimplementation
agrees with itself.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path

import pytest

from graphrag.cli import _build_arg_parser, _build_base_config, _parse_args
from graphrag.profiles import build_config

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    "scripts/runners/run_gold_variant.sh",
    "scripts/runners/run_italian_arm.sh",
    "scripts/runners/run_abstention_arms.sh",
)

# Per-run values a profile cannot carry.
PER_RUN_FIELDS = ("query", "entity")

# Supplied so `set -u` style `${VAR:?...}` defaults resolve during expansion.
STUB_ENV = "VARIANT=probe\nARM=probe\n"


def _cli_invocation(script: str) -> str:
    """Return the `graphrag.cli` command line, line continuations joined."""
    text = (REPO_ROOT / script).read_text(encoding="utf-8")
    start = text.index("python -m graphrag.cli")
    lines: list[str] = []
    for line in text[start:].splitlines():
        lines.append(line)
        if not line.rstrip().endswith("\\"):
            break
    joined = " ".join(line.rstrip().rstrip("\\").strip() for line in lines)
    return joined.replace("python -m graphrag.cli", "", 1)


def _shell_cli_args(script: str) -> list[str]:
    """Expand the script's flag block the way bash would, and split it."""
    text = (REPO_ROOT / script).read_text(encoding="utf-8")
    assignments = "\n".join(
        line for line in text.splitlines() if re.match(r"^[A-Z_]+=", line)
    )
    program = f"{STUB_ENV}{assignments}\nprintf '%s\\n' {_cli_invocation(script)}\n"
    completed = subprocess.run(
        ["bash", "-c", program], capture_output=True, text=True, check=True
    )
    return [arg for arg in completed.stdout.splitlines() if arg]


def _normalise(config) -> dict:
    """A config as JSON would store it, without per-run fields."""
    raw = dataclasses.asdict(config)
    for field in PER_RUN_FIELDS:
        raw.pop(field, None)
    return json.loads(json.dumps(raw, default=str))


@pytest.mark.parametrize("script", RUNNERS)
def test_runner_flags_resolve_to_the_profile(script: str) -> None:
    """Every field the shell script produces matches THESIS_CAMPAIGN."""
    # `_parse_args`, not `parse_args`: it resolves --profile too, so this test
    # holds whether the script spells the configuration out or names it.
    args = _parse_args(_build_arg_parser(), _shell_cli_args(script))
    from_shell = _normalise(_build_base_config(args))
    from_profile = _normalise(build_config("thesis_campaign"))

    assert set(from_shell) == set(from_profile)
    differing = {
        key: {"profile": from_profile[key], "script": from_shell[key]}
        for key in from_profile
        if from_profile[key] != from_shell[key]
    }
    assert not differing, f"{script} diverges from the profile: {differing}"


@pytest.mark.parametrize("script", RUNNERS)
def test_runner_still_passes_a_recognised_flag_block(script: str) -> None:
    """A renamed or dropped CLI flag must fail here, not mid-campaign.

    `parse_args` exits non-zero on an unknown flag, so this also guards against a
    runner drifting away from the CLI it calls.
    """
    args = _shell_cli_args(script)
    assert args, f"{script}: no CLI arguments extracted"
    assert "--experiment" in args
    _parse_args(_build_arg_parser(), args)


def test_abstention_arms_only_adds_documented_per_arm_flags() -> None:
    """The abstention runner's arms differ from the base block on purpose.

    Its shared invocation matches the profile like the others; the two extra
    flags are applied per arm by `run_arm`, and they are exactly the two that
    the abstention experiment is about.
    """
    text = (REPO_ROOT / "scripts/runners/run_abstention_arms.sh").read_text(
        encoding="utf-8"
    )
    per_arm = set(re.findall(r"^run_arm \S+ (--[a-z-]+)", text, re.M))
    assert per_arm == {"--legacy-insufficiency-wording", "--enable-domain-gate"}
