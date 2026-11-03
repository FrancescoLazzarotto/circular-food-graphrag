# Testing

```bash
pytest -q     # 2282 tests: 1459 agent/retrieval, 575 KG pipeline, 248 evaluation
```

The paths come from `[tool.pytest.ini_options]` in `pyproject.toml`, so the bare
command works from any working directory.

34 of them live in `tests/test_audit_fixes.py` and each locks one finding from the
August 2026 code audit. Every one of them passed *before* its fix — which is why
the suite gave no signal at all, and why they are worth keeping.

CI runs three jobs on every push to `main`/`master` and on every pull request:

| Job | What it proves |
|---|---|
| `syntax` | `compileall` under Python 3.10 — the floor declared in `pyproject.toml` still parses |
| `test` | `pip install -e ".[dev]"` from `pyproject.toml` alone, then the full suite |
| `lint` | `ruff check src product` — the rule set is written out in `pyproject.toml` so it cannot drift with a ruff release |

> [!NOTE]
> The suite runs on Python 3.12 only. 3.10 is the floor `pyproject.toml`
> declares and the `syntax` job proves the sources still parse under it — it
> does not prove they still *behave* under it.

The `test` job installs from `pyproject.toml` and never from a requirements file,
so a dependency declared in only one of the two shows up as a CI failure rather
than in a user's clean install.

Health checks and smoke scripts: see
[docs/troubleshooting.md](troubleshooting.md#health-checks).
