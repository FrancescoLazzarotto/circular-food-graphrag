# scripts/

Operational entrypoints, grouped by job. Every script is run from the repo root
(`python scripts/<group>/<name>.py`, `bash scripts/<group>/<name>.sh`); each one
resolves the repo root from its own location, so the working directory only
matters for relative `--output` paths.

| Group | Contains |
|---|---|
| `kg/` | Graph lifecycle against Neo4j: backup/restore/wipe, the `kg_repair*` passes and their `kg_postprocess.py` driver, alias collapse, translation, densification, ontology alignment, search and vector indexes. |
| `kg/quality/` | Standalone graph cleanup passes and structural metrics (`pass1_cleanup`, `pass3_rename_merge`, `merge_same_as`, `kg_metrics`). |
| `gold/` | Gold-set construction: question generation, Italian gold build, annotation backfill, AGROVOC lexicon. |
| `domain_gate/` | Domain-scope threshold calibration and its held-out evaluation. Rerun both after any change to `PromptLibrary.DEFAULT_DOMAIN_SCOPE`. |
| `runners/` | Experiment drivers: retrieval matrix, A/B fast profile, gold variant, Italian and abstention arms. |
| `smoke/` | Reachability and end-to-end checks — the fastest way to tell whether Neo4j and the LLM endpoint are alive. |
| `analysis/` | Post-run analysis: result aggregation, answer diffs, provenance precision, KG variant comparison and significance, slot ceiling, visualisation. |
| `serving/` | vLLM, Neo4j staging and demo start/stop wrappers, plus the per-model `chat_templates/`. vLLM wrappers use the `vllm-serve` virtualenv, never `graphllm`. |
| `cluster/` | SLURM job templates and submission helpers. |

## Order matters in two places

After a KG build, run these three against the live graph, in this order:

```bash
python scripts/kg/kg_postprocess.py --passes 1,2,3,4,5   # repair rounds, not versions of one script
python scripts/kg/kg_search_index.py                     # full-text index — lexical retrieval
python scripts/kg/kg_vector_index.py                     # :NodeVec carriers + vector index — cross-lingual
```

Retrieval quality depends on the last two having been run. `--passes` defaults to
`1,2,3,4`; pass 5 exists and is opt-in, so name it explicitly.

The `kg_repair*.py` passes are driven through `kg_postprocess.py`, never called
directly, and each loads `kg_pipeline/.env` for its `NEO4J_*` and `VLLM_*`
variables.

## Two things the runners do not share

- `runners/run_retrieval_matrix.py` takes **`--graph-strategies`** and
  **`--standard-strategies`**. It has no `--strategies` and no `--models`; pass a
  single `--model-id`.
- Matrix runs carry no `query_id`, so the evaluator joins them to the gold by
  question text. Use `python -m graphrag.cli --experiment` for anything the gold
  scorer will read.

## What `serving/` reads from the environment

Every wrapper in `serving/` takes its settings from the environment, so nothing
in them has to be edited to move a port or try another checkpoint. They are the
one group whose variables are **not** listed in
[../docs/configuration.md](../docs/configuration.md), because they configure a
process being launched rather than the code being run.

Shared by every vLLM wrapper:

| Variable | Default | Effect |
|---|---|---|
| `VLLM_BIN` | `/mnt/storage/flazzarotto/venvs/vllm-serve/bin/vllm` | The `vllm-serve` virtualenv's binary. `import vllm` is broken inside `graphllm`; this is why these are shell wrappers and not a Python entry point |
| `VLLM_HOST` | `127.0.0.1` | Bind address. Loopback on purpose: these servers carry no authentication and two A40s. Export `VLLM_HOST=0.0.0.0` to open one deliberately |
| `HF_HOME` | `/mnt/storage/hf-cache` | Weight cache, exported to the server. One value for every wrapper, so a checkpoint is downloaded once |

Per generator, the names carry the model's own prefix, so two wrappers can never
fight over one variable:

| Demo key | Wrapper | Prefix | Port | GPU(s) |
|---|---|---|---|---|
| `qwen25-32b` | `start_vllm.sh` | `VLLM_` (`VLLM_MODEL_NAME`, `VLLM_PORT`, `VLLM_GPU`) | 8000 | 0 |
| `qwen3-32b` | `start_vllm_qwen3_32b.sh` | `VLLM_QWEN3_32B_` | 8000 | 0 |
| `qwen25-7b` | `start_vllm_qwen25_7b.sh` | `VLLM_QWEN25_7B_` | 8001 | 1 |
| `qwen3-30b-a3b` | `start_vllm_qwen3.sh` | `VLLM_QWEN3_` | 8001 | 1 |
| `qwen38-27b` | `start_vllm_qwen38_27b.sh` | `VLLM_QWEN38_` | 8001 | 1 |
| `qwen38-27b-bf16` | `start_vllm_qwen38_27b_bf16.sh` | `VLLM_QWEN38_BF16_` | 8000 | 0,1 |
| `gemma4-31b` | `start_vllm_gemma4_31b.sh` | `VLLM_GEMMA4_` | 8001 | 1 |
| `qwen25-72b` | `start_vllm_qwen25_72b.sh` | `VLLM_QWEN25_72B_` | 8000 | 0,1 |
| — (densification) | `start_vllm_densify.sh` | `VLLM_DENSIFY_` | 8001 | 1 |
| — (encoder) | `start_vllm_encoder.sh` | `EMBED_PORT`, `EMBED_GPU`, `EMBED_GPU_UTIL`, `GRAPHRAG_EMBED_MODEL` | 8002 | 1 |

Each prefix takes `_MODEL`, `_PORT` and `_GPU` (or `_GPUS` for the two that need
both cards); the newer wrappers also take `_UTIL` for
`--gpu-memory-utilization`, and the ones serving a reasoning model take
`_CHAT_TEMPLATE`, pointing at `serving/chat_templates/` where thinking is off by
default.

> **`VLLM_QWEN38_REVISION` is a pin, not a preference.** It defaults to
> `2fb0debc`, not `main`: the revision published on 2026-09-11 quantises the KV
> cache to FP8, the A40 is sm_86 and has no FP8 hardware, and the model then
> emits garbage from the first token. Override it only on Hopper or newer.

`start_demo.sh` and `stop_demo.sh` share `serving/_models.sh`, so both derive
every port from the table above:

| Variable | Default | Effect |
|---|---|---|
| `DEMO_UI_PORT` | `8501` | Streamlit port. **Export the same value for `stop_demo.sh`** — with a different one it finds nothing, reports success, and the next start says "already up" and serves the old build |
| `DEMO_UI_ADDRESS` | `0.0.0.0` | Streamlit bind address |
| `DEMO_DEFAULT_MODEL` | `qwen25-32b` | Generator started when no key is named |
| `DEMO_CONDA_ENV` | `graphllm` | Environment Streamlit runs in |
| `DEMO_BOOT_TIMEOUT_SEC` | `900` | How long a server may take to answer on its port. A 32B checkpoint from a cold page cache is minutes, not seconds |
| `DEMO_STOP_GRACE_SEC` | `30` | Wait after `SIGTERM` before `stop_demo.sh` escalates |
| `DEMO_LOG_DIR_RUNTIME` | `artifacts/demo_logs` | Where both scripts write and look for `<label>.log` and `<label>.pid` |
| `EMBED_PORT` | `8002` | Encoder port, passed through to the encoder wrapper |

The two Neo4j wrappers:

| Variable | Default | Effect |
|---|---|---|
| `BOLT_PORT` / `HTTP_PORT` | `7689` / `7476` | Ports for the KG v2 staging instance. Neither is a free choice: 7687 is the default and 7688 / 7475 belong to the July staging instance, which is still running and left alone |
| `NEO_HOME` / `JAVA_HOME` | the unpacked Community 5.26 tarball / the `neo4jrt` env | Where that instance and its JVM live. Docker is not usable on this host and the system JDK is 11, while Neo4j 5 needs 17+ |
| `STAGING_PASSWORD` | `staging-kg-v2` | Read by both wrappers |
| `STAGING_URL` / `STAGING_USER` | `bolt://localhost:7689`, `neo4j` | Read by `promote_staging_to_aura.sh` only |
| `CONFIRM` | `no` | `promote_staging_to_aura.sh` prints its plan and stops unless this is `yes`. It replaces the graph the expert demo answers from |

Fuller descriptions: [../docs/experiments.md](../docs/experiments.md) and
[../COMMANDS.md](../COMMANDS.md).
