# AGENTS.md

This file is the repository guide for coding agents. Read it first, then work from the nearest implementation file.

## What this repository is

`graphRAGPipelineExp1` is an experiment-oriented GraphRAG codebase. It has four real execution paths:

1. Knowledge graph construction from documents into Neo4j
2. Retrieval + answer generation through `graphrag-demo` / `python -m graphrag.cli`
3. Experiment and evaluation workflows that compare strategies, models, and resource use
4. Two interactive demos used for domain-expert sessions

The repository is used for practical experiments, not only for demos. Many commands write
artifacts that are later analyzed or included in paper-style reports, so output names and
schemas are part of the contract.

## How to read the project

Start from the owning file for the behavior you want to change:

- [README.md](README.md): orientation, install, architecture, limitations, reproducibility
- [docs/cli.md](docs/cli.md): every `graphrag.cli` flag with its real default — the single source of truth for the option surface
- [docs/configuration.md](docs/configuration.md): every environment variable, and how `.env` is loaded per entry point
- [docs/experiments.md](docs/experiments.md): reference sets, runner choice, campaign drivers, run output layout
- [docs/troubleshooting.md](docs/troubleshooting.md): symptoms seen in this project and what actually caused them
- [COMMANDS.md](COMMANDS.md): task recipes
- [tests/test_audit_fixes.py](tests/test_audit_fixes.py): 34 tests, one per finding of the August 2026 audit — the tracked record of the known logic defects
- [pyproject.toml](pyproject.toml): package metadata, dependencies, CLI entry point
- [src/graphrag/cli.py](src/graphrag/cli.py): public command-line interface and experiment orchestration
- [src/graphrag/config.py](src/graphrag/config.py): `AgentConfig` / `KGConfig` — every tunable, each with the measurement that motivated its default
- [src/graphrag/strategies.py](src/graphrag/strategies.py): the eight retrieval presets, single source of truth
- [src/graphrag/profiles.py](src/graphrag/profiles.py): the named configurations that are actually run — `thesis_campaign`, `demo`, `research_baseline`
- [src/graphrag/agent/core.py](src/graphrag/agent/core.py): the LangGraph state machine
- [src/graphrag/agent/evidence.py](src/graphrag/agent/evidence.py): numbered evidence, citation gate, quote gate, reference lists
- [src/graphrag/agent/memory.py](src/graphrag/agent/memory.py): intra-session conversational memory (demo only)
- [src/graphrag/kg/retriever.py](src/graphrag/kg/retriever.py): search terms, channels, anchors, triple ranking
- [src/graphrag/kg/manager.py](src/graphrag/kg/manager.py): Cypher, full-text and vector lookups, retries
- [src/graphrag/llm/manager.py](src/graphrag/llm/manager.py): local HF / vLLM backends, language detection, domain gate
- [src/graphrag/llm/prompts.py](src/graphrag/llm/prompts.py): every prompt in the system
- [src/graphrag/text_rag/](src/graphrag/text_rag/): TF-IDF and dense (FAISS) text channels
- [src/graphrag/embeddings.py](src/graphrag/embeddings.py): the shared multilingual encoder client
- [src/graphrag/experiments/](src/graphrag/experiments/): experiment runner and resource monitoring
- [kg_pipeline/main.py](kg_pipeline/main.py): KG pipeline entry point with staged checkpointing
- [kg_pipeline/config.yaml](kg_pipeline/config.yaml): pipeline defaults, stage settings, paths
- [scripts/runners/run_retrieval_matrix.py](scripts/runners/run_retrieval_matrix.py): Standard-RAG vs GraphRAG matrix with telemetry
- [scripts/analysis/analyze_experiments.py](scripts/analysis/analyze_experiments.py): per-run analysis
- [scripts/analysis/analyze_resource_usage.py](scripts/analysis/analyze_resource_usage.py): telemetry aggregation across runs
- [evaluation/README.md](evaluation/README.md): gold dataset building, retrieval metrics, judge, optional RAGAS
- [evaluation/scripts/score_gold_run.py](evaluation/scripts/score_gold_run.py): the paper scorer (two channels, two levels)

## Repository structure

- `src/graphrag/`: main package (`cli`, `agent/`, `kg/`, `llm/`, `text_rag/`, `experiments/`)
- `kg_pipeline/`: document-to-KG pipeline with checkpointed stages
- `scripts/`: smoke checks, experiment runners, KG repair passes, index builders, demos, SLURM templates
- `evaluation/`: `evalkit` toolkit, gold datasets, fixtures, evaluation tests
- `tests/`: core unit tests for the agent, evidence, memory, domain gate, definitions
- `docs/`: cluster guidance, plans, audits, worklogs
- `artifacts/`, `exp_results*/`: generated experiment and evaluation outputs
- `kg_pipeline/artifacts/`: pipeline run directories, stage outputs, logs
- `logs/`: ad hoc run logs and matrix logs

## Execution model

### 1) GraphRAG CLI

`python -m graphrag.cli` (or `graphrag-demo`) for single-question or batched runs. It is the
**only** runner that exposes the full `AgentConfig` surface.

Important behavior from `src/graphrag/cli.py`:

- `--llm` enables generation; without it the agent runs retrieval-only paths.
- `--vllm` switches from local HF loading to an OpenAI-compatible vLLM endpoint.
- `--strategies` selects presets; in single-question mode only the first is applied.
- `--experiment` runs a batch over questions and strategies and exports structured artifacts.
- `--questions-file` accepts `.txt` (optionally `Q01<TAB>question`), `.json` (the gold's
  `{"queries": [...]}` shape), `.jsonl` and `.csv`. **Declaring `query_id` is what lets the
  evaluator join by id instead of by question text** — the text join is fragile and warns loudly.
- Retrieval and answer behaviour flags: `--vector-retrieval`, `--seed-from-retrieved`,
  `--subgraph-seed-count`, `--subgraph-limit`, `--drop-predicates`, `--cite-evidence`,
  `--prefer-verbatim-definitions`, `--enable-domain-gate`, `--allow-parametric-fallback`,
  `--focused-answer`, `--complexity`, `--enforce-language`, `--legacy-insufficiency-wording`.
- `--enable-decomposition-step` and `--enable-adaptive-routing-step` add extra LLM calls before
  retrieval; enable only when the experiment needs them.
- `--profile` starts from a named configuration in `graphrag.profiles` instead of spelling it
  out. A flag given explicitly still wins: the profile is installed with `set_defaults` and the
  command line is parsed a second time, so precedence is argparse's own. A profile that sets a
  field with no corresponding flag (`demo` does) is refused rather than half-applied.

### 2) KG pipeline

`kg_pipeline/main.py` runs stages 0–6 and checkpoints outputs so it can resume.
`--stage <name>` runs everything **up to and including** that stage, reusing earlier artifacts.

Stage outputs inside the run directory:

- `stage0_documents.json`
- `stage1_chunks.json`
- `stage2_ner.json`
- `stage3_triples_raw.json`, `stage3_acronyms.json`
- `stage3_checkpoint.json`, `stage3_checkpoint_info.json`
- `stage4_triples_resolved.json`, `stage4_registry.json`, `stage4_merge_approved.json`
- `stage5_triples_linked.json`
- `stage6_neo4j_summary.json`
- `failed_chunks.jsonl`, `new_labels.log`, `pipeline.log`
- `run_metadata.json` plus snapshots of `config.yaml` and the relation vocab

Post-ingestion, in order: `scripts/kg/kg_postprocess.py --passes 1,2,3,4,5`, then
`scripts/kg/kg_search_index.py` (full-text), then `scripts/kg/kg_vector_index.py` (vector carriers).
Retrieval quality depends on the last two having been run against the live graph.

### 3) Experiments and analysis

Two runners write the same artifact shape but do **not** offer the same configuration surface:

| | `python -m graphrag.cli --experiment` | `scripts/runners/run_retrieval_matrix.py` |
|---|---|---|
| GraphRAG strategies | yes | yes |
| Standard-RAG baselines | no | yes |
| Resource telemetry | no | yes |
| `query_id` in results | yes | **no** |
| Full `AgentConfig` (vector, citations, gate, …) | yes | **no** — it builds the config once, with 10 of `AgentConfig`'s 90 fields on the graph arm and 2 on the standard arm, and never mutates it |

Expected run outputs:

- `results.jsonl`, `results.csv`, `summary.txt`, `summary.json`, `config.json`
- `resource_samples.jsonl`, `resource_summary.json` (matrix runner only)

### 4) Evaluation

`graphrag-eval <subcommand>` — `build-dataset`, `retrieval`,
`text`, `judge`, `judge-compare`, `ragas`, `kg`, `gold-triples`, `report-experiment`,
`report-project`, `baseline-update`.

For paper numbers use `evaluation/scripts/score_gold_run.py`, which scores both the retrieval
channel and the answer-text channel at concept and grounding level. The two levels are never
merged into one number — that separation is the finding, not an implementation detail.

### 5) Demos

- `streamlit run product/app.py` — multi-chat expert console with intra-session memory,
  citations, verbatim definitions and the domain gate.
- `python product/console.py --strategy hybrid` — the same stack, console-only.

Both build their own `AgentConfig` inline; they do not read the CLI flags.

## Working conventions

- Prefer the Conda environment `graphllm` for local work; `conda run -n graphllm ...` for automation.
- The CLI entry point is `graphrag-demo`; `python -m graphrag.cli` is also valid and more reliable.
- Do not edit generated artifacts unless the task explicitly asks for it.
- Preserve existing outputs and logs unless you are fixing those files directly.
- Keep changes local to the controlling code path; avoid broad refactors during a bug fix.
- Prefer root-cause fixes over surface patches.
- Prompts live only in `PromptLibrary`. A prompt string in a backend breaks the invariant that
  vLLM and local HF are comparable.
- Retrieval presets live only in `strategies.py`. Both runners import them; keeping a second
  list anywhere is how they drift.

## Known repository details

- `scripts/smoke/smoke_check.py` fills missing variables from `--env-file` (default
  `kg_pipeline/.env`) and then a local `.env`, both non-overriding — anything already exported
  wins. Every check runs by default and a failure is a non-zero exit; waive one with
  `--skip-neo4j` / `--skip-llm` / `--skip-encoder`.
- `graphrag-demo` is the installed console script; if it exits 126 the shim is from a stale
  install, so re-run `pip install -e .`. `python -m graphrag.cli` never depends on the shim.
- `import vllm` is broken inside `graphllm`; serve models from the `vllm-serve` virtualenv.
- Retrieval needs APOC on the Neo4j instance (`apoc.map.removeKey` in every node projection).
- `RAGState` is a LangGraph channel schema: a key returned by a node but not declared there is
  silently dropped from the final state.
- `pyproject.toml` carries the full runtime dependency set, so `pip install -e .` alone is a
  working install; the requirements files exist for the pinned CPU/GPU cluster targets. Extras:
  `demo`, `eval`, `gpu`, `dev`.
- CI runs three jobs: `syntax` (`compileall` under Python 3.10, the declared floor), `test`
  (`pip install -e ".[dev]"` from `pyproject.toml` alone, then the full suite) and `lint`
  (`ruff check src product`, scoped to those two trees — the script trees are out). `ruff` is in
  the `dev` extra, so the lint gate is runnable locally with the same command.
- Two gold files coexist. `evaluation/gold/gold.json` shares the 30 `query_id`s of
  `gold_v3.json` but differs in `expected_entities` on 7 of them, and
  `evaluation/scripts/score_gold_run.py` defaults to the older one. Always pass
  `--gold evaluation/gold/gold_v3.json` explicitly.
- `scripts/runners/run_retrieval_matrix.py` takes `--graph-strategies` and
  `--standard-strategies`. It has no `--strategies` and no `--models`.
- `scripts/kg/kg_postprocess.py --passes` defaults to `1,2,3,4`. Pass 5 exists and is opt-in.
- Real sample outputs exist under `artifacts/experiments/`, `exp_results*/` and
  `kg_pipeline/artifacts/`; use them when you need concrete examples of a schema.

## Data models

Core Pydantic models live in `kg_pipeline/models/types.py`. `RAGState`, `KGNode`, `Triple` and
`ProvenanceRecord` are TypedDicts in `src/graphrag/types.py`.

`RAGState` is the LangGraph channel schema: **a key a node returns but does not declare there is
silently dropped**. `retrieved_neighbors` and `visible_evidence_refs` are declared there; check the
schema before returning any new key.

`KGTriple` predicates must be `SCREAMING_SNAKE_CASE` (validated by regex). Entity names are **not**
unique before stage-4 resolution — use `CanonicalEntityRecord` after stage 4.

## Code conventions

- **Type hints**: always; union types with `|` (Python 3.10+)
- **Docstrings**: Google-style — one-liner plus Args/Returns/Raises on public and non-trivial private functions; a one-liner is enough for an obvious helper; classes list their fields under Attributes
- **Logging**: module-level `logger = logging.getLogger("graphrag")` or `"kg_pipeline"`; INFO for milestones, DEBUG for traces, WARNING for recoverable issues
- **Imports**: stdlib → third-party → local, separated by blank lines
- **Pydantic**: `ConfigDict(extra="forbid")`; `field_validator` for normalisation
- **Cypher**: always parameterised — never f-string user input into a query
- **Neo4j writes**: UNWIND + MERGE for batches; never loop with individual queries
- **Property access in Cypher**: `properties(node)['key']` avoids the `UnknownPropertyKey` warnings
- **Comments explain why, not what, in the present tense.** State the reason for a non-obvious
  choice or a safety constraint. No history: no "used to" or "the old code", no past incidents or
  dates, no run or corpus measurements, no references to audits, plans or issue ids — that belongs
  in the commit message. A comment restating the line above it is noise.

## Anti-patterns

Each of these has cost this project a measurement or a run.

- Bare `except:` or a silent `except Exception:` — catch the specific exception
- Querying Neo4j inside a loop
- Assuming entity names are unique before stage-4 resolution
- Calling an LLM without retry logic — `LLMManager` already handles it
- Hardcoding model paths or credentials
- Skipping `validate_triples()` after parsing LLM JSON
- Mixing async and sync without coordinating `LLMManager._load_lock`
- Appending experiment results without `run_id` / timestamp
- Assuming vLLM is available — health-check first
- Ignoring checkpoint files: re-running stage 3 without clearing resumes from the checkpoint
- Adding a generic phrase to `_REFUSAL_MARKERS` or `_INSUFFICIENT_MARKERS`
  (`src/graphrag/llm/refusal.py`) — they are substring-matched over the whole answer, feed both an
  answer-replacement path and a published metric, and a generic phrase silently rewrites correct answers
- Returning a new key from an agent node without declaring it in `RAGState`
- Changing `PromptLibrary.DEFAULT_DOMAIN_SCOPE` or the domain-gate wording without rerunning all
  three suites: `scripts/domain_gate/eval_domain_gate_llm.py`, `eval_domain_gate_heldout.py` and
  `eval_domain_gate_entities.py` (the last needs the graph as well as the model)
- Copying a prompt out of `PromptLibrary` into a script that measures it — the eval suite did, the
  two drifted, and it scored a prompt nobody ran
- Editing `graphrag.config` or `graphrag.strategies` to change demo behaviour. Those are what the
  campaigns were measured with; demo settings live in `product/config.py`
- Editing a profile in `graphrag.profiles`, or the flag block of a runner under
  `scripts/runners/`, without running `tests/test_profiles.py` and
  `tests/test_profiles_match_runners.py`. They pin `thesis_campaign` from both ends — against the
  configuration a real campaign recorded, and against what the runners resolve to today — so a
  change that would move a published number fails there and names the field
- Adding a field to a profile without a matching CLI flag. `--profile` refuses such a profile
  outright rather than applying it in part, so the failure is at startup, not in the results

## Validation habits

After an edit, use the smallest check that can falsify the change:

- Documentation only: `git diff --check -- '*.md'`, then confirm every relative link still resolves
- Python logic changes: targeted smoke script or a narrow module run
- CLI or pipeline changes: the smallest relevant command that exercises the touched path
- Retrieval or metric changes: the unit tests, which are fast and cover the tricky parts

```bash
pytest -q     # 1924 tests; paths come from pyproject.toml, so any cwd works
python scripts/smoke/smoke_check.py
python scripts/smoke/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
python scripts/smoke/smoke_kg_retriever.py
conda run -n graphllm python -m graphrag.cli --help
conda run -n graphllm python -m kg_pipeline.main --config kg_pipeline/config.yaml --env-file kg_pipeline/.env --log-level INFO
```

If you touch experiment code, also inspect a recent artifact folder and confirm the output
names still match the analysis scripts.

## Repository-specific implementation notes

- Retrieval and experiment code is performance-sensitive; avoid adding extra LLM calls or broad
  abstractions unless required.
- In `scripts/runners/run_retrieval_matrix.py`, the runner may checkpoint or finalize outputs during long
  runs. Do not assume the output directory is empty until the process finishes. Note also that it
  cannot express the newer `AgentConfig` options.
- In `src/graphrag/agent/core.py`, do not reintroduce decomposition or routing steps unless the
  task explicitly requires them; they add latency and are often not needed.
- In `src/graphrag/llm/refusal.py`, the two marker lists are substring-matched over the whole
  answer and feed both an answer-replacement path and a published metric. Adding a generic phrase
  there silently rewrites correct answers.
- In `kg_pipeline/main.py`, the stage order matters because downstream stages read stage artifacts
  from earlier stages.
- In `kg_pipeline/stages/resolution.py`, the merge cache stores raw group indices; it is only valid
  for an unchanged stage-3 output.
- In `evaluation/`, the evaluation flow assumes run outputs already exist and a gold file has been
  prepared. A join that silently degrades to matching on question text has produced wrong numbers
  in this project before — the join report at WARNING level is there for that reason.

## Before you change behaviour

`tests/test_audit_fixes.py` locks 34 findings from the August 2026 code audit, one test each.
Every one of them passed *before* its fix, which is why the suite gave no signal at all. Read the
test that covers the area you are about to touch: several behaviours that look like bugs on first
reading are deliberate, and several that look correct are not.

The audit write-up itself (`docs/code_audit_2026-08-15.md`) is an internal working document and is
not tracked by git — it exists only in a local checkout that has it.

## If you are unsure

- Re-read the nearest implementation file, not the whole repo.
- Check the latest artifact folder or the corresponding analyzer before changing experiment code.
- If the change touches evaluation, consult [evaluation/README.md](evaluation/README.md) first.
