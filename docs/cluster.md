# Cluster Deployment Guide

## 1. Prerequisites

- Python 3.10+
- Access to Neo4j endpoint
- Network access to Hugging Face model hub or pre-cached model files
- For server-backed inference mode: vLLM-capable GPU node (installed via requirements-gpu.txt)

## 2. Immediate Run (Recommended)

The provided SLURM scripts can bootstrap the environment automatically (create venv + install dependencies + run smoke checks).

Export Neo4j variables before submission:

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<your-username>"
export NEO4J_PASSWORD="<your-password>"
export NEO4J_DATABASE="<your-database>"
```

GPU run:

```bash
sbatch -p <gpu_partition> scripts/cluster/run_graphrag.sbatch
```

CPU run:

```bash
sbatch -p <cpu_partition> scripts/cluster/run_graphrag_cpu.sbatch
```

KG pipeline run that survives laptop disconnects:

```bash
sbatch -p <partition> scripts/cluster/run_kg_pipeline.sbatch
```

Before submission, export required Neo4j variables in your shell or directly in the script.

## 2.1 Local Validation Before Cluster Access

If you do not have cluster access yet, run the local preflight helper from Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/cluster/preflight.ps1
```

Optional Neo4j connectivity validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/cluster/preflight.ps1 -CheckNeo4j
```

## 3. Environment Variables

Required:

- NEO4J_URL
- NEO4J_USERNAME
- NEO4J_PASSWORD

Optional:

- NEO4J_DATABASE (default: neo4j)
- MODEL_ID (default: Qwen/Qwen2.5-7B-Instruct)
- HF_HOME / TRANSFORMERS_CACHE for model cache location
- USE_VLLM=1 to enable vLLM server mode in `run_graphrag.sbatch`
- VLLM_BASE_URL (default: http://127.0.0.1:8000/v1)
- VLLM_HOST / VLLM_PORT (defaults: 127.0.0.1 / 8000)
- VLLM_GPU_MEMORY_UTILIZATION (default: 0.90)
- VLLM_STARTUP_TIMEOUT_SEC (default: 180)
- CONDA_ENV (default: graphllm) for `scripts/cluster/run_kg_pipeline.sbatch`
- KG_CONFIG / KG_ENV_FILE / KG_RUN_DIR / KG_STAGE / KG_LOG_LEVEL for the KG pipeline launcher

## 4. Manual Install (Optional)

Use this only if you prefer pre-installing dependencies yourself.

CPU nodes:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
pip install -e . --no-deps
```

GPU nodes:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt
pip install -e . --no-deps
```

## 5. Smoke Check

```bash
python scripts/smoke/smoke_check.py
```

Imports, the graph and both its indexes, the generator and the encoder are all
checked, and any failure is a non-zero exit. Waive one with `--skip-neo4j`,
`--skip-llm` or `--skip-encoder`, or stop after the imports with
`--check-imports-only`.

`--check-neo4j` still parses, so old job scripts keep working, but it turns
nothing on: the graph check has been part of every run for a while now.

## 6. Job Script Behavior

Both scripts support these runtime overrides:

- INSTALL_DEPS=0: skip dependency installation
- VENV_PATH=/path/to/venv: custom virtual environment path
- QUESTION / ENTITY: override default prompt inputs
- MODEL_ID: set custom model (default `Qwen/Qwen2.5-7B-Instruct`)
- LLM_WARMUP=0 or 1: disable/enable warmup (GPU script default is 1, CPU script 0)
- RUN_LLM_ON_CPU=1: enable local LLM on CPU script (default is 0)
- USE_VLLM=1: in GPU script, start a local vLLM server and run `graphrag-demo --vllm`
- NEO4J_DATABASE: defaults to `neo4j` here, not to the empty value the rest of
  the project uses — a job left unset talks to the database named `neo4j`

The GPU script alone reads these, the `VLLM_*` ones only when USE_VLLM=1:

- VLLM_PORT / VLLM_HOST: where it starts the server (8000 on 127.0.0.1)
- VLLM_BASE_URL: the endpoint polled for readiness and handed to
  `graphrag-demo` (`http://127.0.0.1:8000/v1`). **Not derived from the two
  above** — move the port without moving this and the job starts a server it
  then fails to reach
- VLLM_GPU_MEMORY_UTILIZATION: `--gpu-memory-utilization` for that server (0.90)
- VLLM_STARTUP_TIMEOUT_SEC: attempts on `/models`, one per second, before the
  job gives up and dumps the server log (180). A dead server is noticed
  immediately, without waiting out the budget
- HF_HOME: weight cache, `$HOME/.cache/huggingface` unless the node has a
  scratch filesystem worth pointing it at. TRANSFORMERS_CACHE follows it

When `USE_VLLM=1` in `scripts/cluster/run_graphrag.sbatch`, the script:

- starts `vllm.entrypoints.openai.api_server` in background
- waits for endpoint readiness on `/models`
- runs `graphrag-demo` with `--vllm --vllm-base-url ...`
- cleans up the background server process on exit (trap)

Example:

```bash
INSTALL_DEPS=0 QUESTION="Who directed The Matrix?" sbatch -p <gpu_partition> scripts/cluster/run_graphrag.sbatch
```

## 7. Submit Job (SLURM)

```bash
sbatch scripts/cluster/run_graphrag.sbatch
```

To inspect live logs:

```bash
squeue -u $USER
tail -f logs/graphrag-gpu-<job_id>.out
tail -f logs/graphrag-gpu-<job_id>.err
```

## 7.1 Long Matrix Benchmark (2x A40)

For long-running comparisons across retrieval strategies and LLMs, use:

- `scripts/cluster/run_experiment_matrix_gpu.sbatch`

Recommended submission pattern for two GPUs:

```bash
MODELS_CSV="Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
STRATEGIES="default,text_only,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
RUNS_PER_STRATEGY=3 \
CONDA_ENV=graphllm \
sbatch -p <gpu_partition> --array=0-2%2 scripts/cluster/run_experiment_matrix_gpu.sbatch
```

Notes:

- `--array=0-2%2` runs up to 2 tasks concurrently (one model per task).
- Each task requests 1 GPU, 12 CPU, 64G RAM, 24h walltime by default.
- Override `QUESTIONS_FILE`, `OUTPUT_DIR`, and `EXPERIMENT_TAG_PREFIX` as needed.

Aggregate all produced run folders with:

```bash
python scripts/analysis/analyze_matrix.py artifacts/experiments --tag-contains matrix_long
```

## 8. Troubleshooting

- If CUDA mismatch occurs, install a compatible PyTorch wheel for your cluster.
- If model download fails on compute nodes, pre-populate HF_HOME cache on shared storage.
- If Neo4j connection fails, verify firewall/egress rules and credentials.
