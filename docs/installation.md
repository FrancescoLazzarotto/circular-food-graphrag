# Installation

## Prerequisites

`pip install -e .` installs the Python side and nothing else. Three things live
outside the package, and the [Quick start](../README.md#quick-start) cannot run until
they exist.

| What | Why it is required | How this project runs it |
|---|---|---|
| **Neo4j 5.x with the APOC plugin** | The graph store. APOC is a **hard dependency**, not an optimisation: every node and triple projection calls `apoc.map.removeKey` to strip the embedding vector before returning properties, and the KG pipeline calls `apoc.periodic.iterate`, `apoc.path.subgraphNodes`, `apoc.refactor.mergeNodes`, `apoc.refactor.setType` and `apoc.create.relationship`. There is no fallback projection — without APOC, retrieval raises | Community 5.26.0 as an unpacked tarball. The production graph runs as a systemd service on loopback, installed by [`deploy/graph/install.sh`](../deploy/graph/) (see [docs/graph_hosting.md](graph_hosting.md)); a throwaway staging instance starts with [`scripts/serving/start_neo4j_staging.sh`](../scripts/serving/start_neo4j_staging.sh). Neo4j 5 needs a **JDK 17 or newer**; the staging script carries its own JDK 21 because the host's system Java is 11 |
| **A generation server** | Every answer is produced by an LLM. Either an OpenAI-compatible endpoint (vLLM) at `VLLM_BASE_URL`, or a local Hugging Face model loaded in-process | [`scripts/serving/start_vllm.sh`](../scripts/serving/) and the per-model variants next to it; `--list` on `start_demo.sh` prints which models are servable |
| **A multilingual encoder** | Only for the vector retrieval channel — the one channel that crosses the Italian/English gap. Required whenever `--vector-retrieval` is on, which both demos and the reference campaign use. Without it the encoder call fails and retrieval **raises rather than degrading**, on purpose: a silent fallback to lexical-only makes two runs incomparable | [`scripts/serving/start_vllm_encoder.sh`](../scripts/serving/), `intfloat/multilingual-e5-base` at `GRAPHRAG_EMBED_BASE_URL` |

> [!IMPORTANT]
> **The repository ships no corpus and no graph dump.** `documents/` is
> git-ignored, and `kg_pipeline/config.yaml` points `input_dir` at a local path
> that is not in the clone. A fresh clone therefore has the code to build a
> knowledge graph but nothing to build one *from*: supply your own PDF/Markdown
> corpus, point `input_dir` at it, and run the
> [Knowledge Graph pipeline](kg_pipeline.md) before anything under
> Quick start will answer.

Everything else is environment variables: `cp .env.example .env` and see
**[docs/configuration.md](configuration.md)** for every variable and its
real default.

---

# Install

Recommended environment: Conda, named `graphllm`. Python 3.12 is the version the
project is tested on; `pyproject.toml` declares a 3.10 floor and CI checks that
the sources still parse under it.

```bash
conda create -n graphllm python=3.12 -y
conda activate graphllm
pip install -e .
```

That single command is enough for the CLI, the KG pipeline and the demos —
`pyproject.toml` carries every runtime dependency. Optional extras, all
`pip install -e ".[<name>]"`:

| Extra | Contents |
|---|---|
| `demo` | Streamlit, for the browser demo in `product/` |
| `eval` | RAGAS, pandas, datasets, ROUGE, matplotlib, numpy |
| `gpu` | `bitsandbytes`, `autoawq`, `vllm` — GPU nodes serving models locally |
| `dev` | pytest, pytest-cov and ruff — the linter CI gates on, so `ruff check src product` runs the same way here |

The requirements files remain for the pinned targets, where a resolver has to be
told exactly which wheel to take: `requirements.txt` (local, loose bounds),
`requirements-cpu.txt` (CPU cluster nodes), `requirements-gpu.txt` (GPU nodes,
CUDA 12.4, pins `torch==2.5.1+cu124`).

To reproduce an environment rather than describe one, install
`requirements.lock`: exact versions including transitive dependencies, captured
from the environment the measured results were produced in. Use it whenever a
number has to be comparable with a previous run.

```bash
pip install -r requirements.lock
```

> [!TIP]
> Without `sacrebleu`, evalkit falls back to a simplified local BLEU. Keep it
> installed so published metrics come from the reference implementation.
