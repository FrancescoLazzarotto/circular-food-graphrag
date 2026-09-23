# GitHub Copilot — Repository Instructions

## Read AGENTS.md first

**[AGENTS.md](../AGENTS.md) is the canonical repository guide.** It carries what
the project is, how to read it, the execution model, working conventions, data
models, code conventions, anti-patterns and validation habits — none of which is
Copilot-specific, and none of which is duplicated here.

Reference documentation:

| Document | Covers |
|---|---|
| [README.md](../README.md) | Orientation, install, architecture, known limitations, reproducibility |
| [docs/cli.md](../docs/cli.md) | Every `graphrag.cli` flag with its real default |
| [docs/configuration.md](../docs/configuration.md) | Every environment variable |
| [docs/experiments.md](../docs/experiments.md) | Reference sets, runners, campaign drivers |
| [docs/troubleshooting.md](../docs/troubleshooting.md) | Symptoms and their real causes |
| [COMMANDS.md](../COMMANDS.md) | Task recipes |
| [evaluation/README.md](../evaluation/README.md) | Gold sets and the two-channel, two-level scorer |

---

## Read the types from the source, not from a copy

This file used to carry a hand-written copy of the pipeline data contracts. It
drifted: it listed `evaluation/run_ragas_eval.py`, `retrieval_metrics.py` and
`build_eval_dataset.py` long after they were replaced by `evalkit/`, and its
`RAGState` copy was missing `retrieved_neighbors` and `visible_evidence_refs` —
exactly the two keys the codebase warns about, because **a key an agent node
returns but does not declare in `RAGState` is silently dropped**.

Read the definitions where they live:

| Contract | File |
|---|---|
| `RAGState`, `Triple`, `KGNode`, `ProvenanceRecord` | [`src/graphrag/types.py`](../src/graphrag/types.py) |
| `AgentConfig`, `KGConfig` — every tunable, each with the measurement that motivated its default | [`src/graphrag/config.py`](../src/graphrag/config.py) |
| The 8 retrieval presets | [`src/graphrag/strategies.py`](../src/graphrag/strategies.py) |
| `DocumentRecord`, `ChunkRecord`, `KGTriple`, `CanonicalEntityRecord` | [`kg_pipeline/models/types.py`](../kg_pipeline/models/types.py) |
| Every prompt in the system | [`src/graphrag/llm/prompts.py`](../src/graphrag/llm/prompts.py) |
| Pipeline defaults, ontology, chunking profiles | [`kg_pipeline/config.yaml`](../kg_pipeline/config.yaml) |

---

## What matters most when suggesting code here

The full list is in [AGENTS.md](../AGENTS.md#anti-patterns). The rules that
Copilot completions break most often:

- **Type hints always**, union types with `|`. Google-style docstrings.
- **Never f-string user input into Cypher.** Parameterise every query.
- **Batch Neo4j writes with UNWIND + MERGE.** Never loop with individual queries.
- **No bare `except:`**, no silent `except Exception:` — catch the specific one.
- **Prompts live only in `PromptLibrary`.** A prompt string inlined in a backend or
  an eval script breaks the invariant that vLLM and local HF are comparable; it has
  already caused a suite to score a prompt nobody ran.
- **Retrieval presets live only in `strategies.py`.** Both runners import them; a
  second list anywhere is how they drift.
- **Entity names are not unique before stage-4 resolution.** Use
  `CanonicalEntityRecord` after stage 4.
- **`LLMManager` already handles retries.** Do not wrap it in another retry loop.
- **Comments explain why, not what, in the present tense.** State the reason for a
  non-obvious choice or a safety constraint. No history — no "used to", past
  incidents, dates, run measurements or references to audits and plans; that
  belongs in the commit message. A comment restating the line above it is noise.

---

## Before proposing a change to measured behaviour

`src/graphrag/config.py` and `src/graphrag/strategies.py` are what the reported
campaigns were measured with. Changing a default there makes future runs
incomparable with the results already published from this repository. Demo
behaviour belongs in `product/config.py` instead.
