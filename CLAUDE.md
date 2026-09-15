# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

## Read AGENTS.md first

**[AGENTS.md](AGENTS.md) is the canonical repository guide** — what the project
is, how to read it, the execution model, working conventions, data models, code
conventions, anti-patterns and validation habits. It is not Claude-specific and
it is not duplicated here.

Reference documentation, in the order you will usually want it:

| Document | Covers |
|---|---|
| [README.md](README.md) | Orientation, install, architecture, known limitations, reproducibility |
| [docs/cli.md](docs/cli.md) | Every `graphrag.cli` flag with its real default |
| [docs/configuration.md](docs/configuration.md) | Every environment variable, and how `.env` is loaded per entry point |
| [docs/experiments.md](docs/experiments.md) | Reference sets, runner choice, campaign drivers, run layout |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptoms and what actually caused them |
| [COMMANDS.md](COMMANDS.md) | Task recipes |
| [evaluation/README.md](evaluation/README.md) | Gold sets and the two-channel, two-level scorer |

This file holds only what is specific to running a Claude Code session here.

---

## Environment

Always work inside the `graphllm` Conda environment. Use `conda run -n graphllm …`
for anything scripted, so the invocation is reproducible.

```bash
conda activate graphllm
pip install -e .                    # pyproject carries the full runtime dependency set
pip install -e ".[demo,dev]"        # extras: demo · eval · gpu · dev
```

> `import vllm` is broken inside `graphllm`. Every (re)start of a served model
> must go through the `vllm-serve` virtualenv — use the
> `scripts/serving/start_vllm*.sh` wrappers, and do not try to repair the Conda
> environment instead.

Credentials and endpoints come from the environment. `.env` handling differs per
entry point — see
[docs/configuration.md](docs/configuration.md#how-env-is-loaded).

---

## The commands worth knowing by heart

```bash
# preflight: imports, graph + both indexes ONLINE, generator, encoder
python scripts/smoke/smoke_check.py

# is the vector index alive, or only present?
python scripts/kg/check_vector_index.py --min-resolving 1000

# the whole demo up, with a preflight
bash scripts/serving/start_demo.sh --list
bash scripts/serving/start_demo.sh qwen25-32b
bash scripts/serving/stop_demo.sh

# one question
conda run -n graphllm python -m graphrag.cli --question "What is X?" --entity "Y"

# the full suite, from any working directory
pytest -q                          # 1924 tests: 1156 agent/retrieval, 520 KG pipeline, 248 evaluation
pytest evaluation/tests/test_metrics.py -v
pytest kg_pipeline/tests/test_pipeline.py::test_schema_validation_accepts_valid_triple -v
```

Everything else is a recipe in [COMMANDS.md](COMMANDS.md).

---

## Validation after an edit

Use the smallest check that can falsify the change.

| Changed | Run |
|---|---|
| Documentation only | `git diff --check -- '*.md'`, then confirm every relative link still resolves |
| Python logic | The smallest relevant smoke script |
| CLI or pipeline | The smallest command that exercises the touched path |
| Retrieval or scoring | `pytest tests/ evaluation/tests/ -q` |
| Experiment code | Inspect a recent artifact folder and confirm the output names still match the analysis scripts |

---

## Traps this repository sets

These have each cost a measurement here. The full list is in
[AGENTS.md](AGENTS.md#anti-patterns); these are the ones a session tends to hit
first.

- **Two gold files.** `evaluation/gold/gold.json` shares the 30 `query_id`s of `gold_v3.json` but differs in `expected_entities` on 7 of them, and `score_gold_run.py` defaults to the older one. Always pass `--gold evaluation/gold/gold_v3.json`.
- **`run_retrieval_matrix.py` takes `--graph-strategies` / `--standard-strategies`.** There is no `--strategies` and no `--models` on that runner.
- **`kg_postprocess.py --passes` defaults to `1,2,3,4`.** Pass 5 exists and is opt-in.
- **`--stage` is inclusive-up-to, not isolating.** `--stage ner` also runs ingestion and chunking.
- **A key returned by an agent node but not declared in `RAGState` is silently dropped.**
- **Changing `graphrag.config` or `graphrag.strategies` makes future runs incomparable** with the campaigns already reported. Demo behaviour belongs in `product/config.py`.
- **The audit write-up `docs/code_audit_2026-08-15.md` is not tracked by git.** Internal plans, audits, worklogs and probes under `docs/` are deliberately local — do not commit them, and do not link to them from tracked files. `tests/test_audit_fixes.py` is the tracked record of those findings.
