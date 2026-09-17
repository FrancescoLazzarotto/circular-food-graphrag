# Command Recipes

Copy-paste command sequences, organised by job. Every command here was run
against this repository's current state.

This file holds **recipes**. It deliberately does not repeat the option tables:

- Every `graphrag.cli` flag and its default → **[docs/cli.md](docs/cli.md)**
- Every environment variable → **[docs/configuration.md](docs/configuration.md)**
- Campaign drivers and run layout → **[docs/experiments.md](docs/experiments.md)**
- When something misbehaves → **[docs/troubleshooting.md](docs/troubleshooting.md)**

> `graphrag-demo` and `python -m graphrag.cli` are the same entry point. If
> `graphrag-demo` exits 126 the console-script shim is stale; the module form
> never depends on it.

---

## 1. Ask one question

```bash
# retrieval only, no generation
conda run -n graphllm python -m graphrag.cli \
  --question "What is the relationship between X and Y?" \
  --entity "X" \
  --strategies default

# with a local Hugging Face model
conda run -n graphllm python -m graphrag.cli \
  --question "What is the relationship between X and Y?" \
  --llm --model-id meta-llama/Meta-Llama-3-8B-Instruct

# against a vLLM / OpenAI-compatible server
conda run -n graphllm python -m graphrag.cli \
  --question "What is the relationship between X and Y?" \
  --llm --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id "Qwen/Qwen2.5-32B-Instruct-AWQ"
```

In single-question mode only the **first** entry of `--strategies` is applied.

### A grounded, cited answer

```bash
conda run -n graphllm python -m graphrag.cli --llm --vllm \
  --strategies hybrid \
  --question "According to the Regulation, what is the definition of 'food business operator'?" \
  --cite-evidence --citation-policy mark --citation-display label \
  --prefer-verbatim-definitions \
  --enforce-language --focused-answer --complexity high
```

### Reaching Italian node names from an English question

The graph is largely Italian; English questions cannot reach Italian node names
lexically. The vector channel is **added** to the lexical one, never replacing it.

```bash
python scripts/kg/kg_vector_index.py    # once, after the KG is built

conda run -n graphllm python -m graphrag.cli --llm --vllm \
  --vector-retrieval --vector-nodes-limit 10 --vector-triples-limit 10 \
  --seed-from-retrieved --subgraph-seed-count 3 \
  --drop-predicates "RELATED_TO,PUBLISHED,AUTHORED_BY"
```

### Refusing out-of-domain questions

```bash
conda run -n graphllm python -m graphrag.cli --llm --vllm --enable-domain-gate
```

One classification call before retrieval. Without it the agent has no terminal
refusal state: the dense retriever has no score floor, so `grade` always sees
evidence and every question reaches `generate`.

---

## 2. Build the Knowledge Graph

```bash
# full pipeline
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --log-level INFO

# up to and including one stage, reusing earlier artifacts
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --stage ner

# one document
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --single-doc documento.pdf

# no writes to Neo4j
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml --dry-run

# resume an interrupted run
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --run-dir kg_pipeline/artifacts/run_<tag>
```

| Flag | Effect |
|---|---|
| `--config` | Configuration file (default `kg_pipeline/config.yaml`) |
| `--env-file` | `.env` with Neo4j credentials and endpoints. Pass it explicitly |
| `--run-dir` | Existing run directory to resume; empty creates a timestamped one |
| `--single-doc` | Process one document (filename or doc_id) |
| `--stage` | `all` `ingestion` `chunking` `ner` `llm` `resolution` `linking` `neo4j` |
| `--dry-run` | Run without writing to Neo4j |
| `--log-level` | `DEBUG` `INFO` `WARNING` (default `INFO`) |

> **`--stage` is inclusive-up-to, not isolating.** `--stage ner` runs ingestion,
> chunking and ner, reusing earlier artifacts where they exist. There is no flag
> that runs one stage alone.

> **Stage 3 checkpoint.** Progress is saved every `llm.checkpoint_every` chunks
> to `stage3_checkpoint.json`, written atomically. Re-running without deleting it
> resumes from the last saved chunk; triples from chunks past the last completed
> checkpoint are dropped on resume, so recovery never duplicates.

### Post-processing and indexes

Run in this order, after ingestion:

```bash
conda run -n graphllm python scripts/kg/kg_postprocess.py --passes 1,2,3,4,5
conda run -n graphllm python scripts/kg/kg_search_index.py      # full-text index
conda run -n graphllm python scripts/kg/kg_vector_index.py      # :NodeVec + vector index
```

> `--passes` defaults to `1,2,3,4`. Pass 5 exists (`kg_repair5.py`) and is **not**
> in the default — name it explicitly, as above, if you want it.

Each pass loads `kg_pipeline/.env` and needs the `NEO4J_*` and `VLLM_*`
variables. Run them through `kg_postprocess.py`, not by calling `kg_repair*.py`
directly.

### Re-run resolution without re-running extraction

Tunes similarity thresholds against an existing stage-3 output — no NER, no LLM
extraction.

```bash
python scripts/kg/remerge_entities.py \
  --run-dir kg_pipeline/artifacts/run_<tag> \
  --similarity-threshold 0.90 \
  --context-jaccard-floor 0.15
```

| Flag | Effect |
|---|---|
| `--run-dir` | Run directory holding stage-3 output (required) |
| `--output-dir` | Alternative destination for the stage 4/5 artifacts |
| `--embedding-model` | SentenceTransformer model used for resolution |
| `--similarity-threshold` | Cosine similarity threshold (default 0.88) |
| `--context-jaccard-floor` | Minimum context Jaccard (default 0.15) |
| `--base-url` / `--model-name` | vLLM endpoint and model for merge confirmation |

The merge cache stores raw group indices and is only valid for an unchanged
stage-3 output.

### Inspect the graph

```bash
python scripts/analysis/visualize_kg.py --out artifacts/tmp/kg_viz.html
python scripts/analysis/kg_evaluator.py      # structural report → artifacts/kg_reports/
```

---

## 3. Run a campaign

```bash
conda run -n graphllm python -m graphrag.cli --experiment \
  --questions-file evaluation/gold/gold_v3.json \
  --strategies "default,hybrid,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
  --llm --vllm --vllm-base-url http://localhost:8000/v1 \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --vector-retrieval --seed-from-retrieved \
  --cite-evidence --complexity medium --max-new-tokens 1024 \
  --output-dir exp_results --experiment-tag thesis_qwen25_32b
```

Prepared drivers that run a whole family of arms in one server session:

```bash
bash scripts/runners/run_abstention_arms.sh
bash scripts/runners/run_italian_arm.sh
VARIANT=v2_baseline bash scripts/runners/run_gold_variant.sh
```

See [docs/experiments.md](docs/experiments.md) for what each one measures.

---

## 4. Retrieval matrix — Standard RAG vs GraphRAG

`run_retrieval_matrix.py` is the only runner that produces resource telemetry and
Standard-RAG baselines. It takes **`--standard-strategies` and
`--graph-strategies`**, not `--strategies`, and it has no `--models` flag: pass a
single `--model-id`.

```bash
# smoke first — always
python scripts/runners/run_retrieval_matrix.py \
  --smoke \
  --questions-file artifacts/experiments/questions_smoke.txt \
  --documents docs/ README.md \
  --runs-per-strategy 1 \
  --output-dir artifacts/experiments \
  --experiment-tag retrieval_matrix_smoke

# full, vLLM-backed
python scripts/runners/run_retrieval_matrix.py \
  --llm --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --questions-file evaluation/fixtures/questions_matrix_long.txt \
  --graph-strategies "default,text_plus_triples,subgraph_2hop" \
  --runs-per-strategy 1 \
  --experiment-tag strategy_comparison

# GraphRAG only, no standard-RAG arm
python scripts/runners/run_retrieval_matrix.py \
  --questions-file evaluation/fixtures/questions.txt \
  --skip-standard \
  --graph-strategies "neighbors_focus,subgraph_2hop,shortest_path"
```

| Flag | Effect |
|---|---|
| `--questions-file` / `--question` | Question set, or a single question |
| `--entity` | Optional entity seed for graph traversal; empty auto-seeds from the question |
| `--graph-strategies` | Comma-separated GraphRAG presets |
| `--standard-strategies` | Comma-separated Standard-RAG presets |
| `--documents` / `--doc-patterns` | Corpus for the standard-RAG arm |
| `--skip-standard` / `--skip-graph` | Run only one side of the comparison |
| `--llm` / `--vllm` / `--vllm-base-url` / `--model-id` / `--llm-warmup` | Generation |
| `--performance-profile` | `auto` / `default` / `production_fast` |
| `--monitor-resources` / `--no-monitor-resources` / `--resource-sample-interval` | Telemetry |
| `--runs-per-strategy` / `--output-dir` / `--experiment-tag` | Run shape and destination |
| `--smoke` / `--smoke-questions` / `--smoke-graph-strategies` / `--smoke-standard-strategies` | Reduced test pass |
| `--dense-embedding-model` / `--vector-index-dir` / `--dense-device` | Dense text backend |
| `--max-new-tokens` / `--gpu-memory-fraction` / `--allow-large-model-fp16-fallback` | Generation limits |
| `--enable-decomposition-step` / `--enable-adaptive-routing-step` | Extra LLM steps |

Matrix runs carry **no `query_id`**, so the evaluator joins them to the gold by
question text. Use `graphrag.cli --experiment` for anything the gold scorer will
read.

### A/B a performance profile

```bash
python scripts/runners/run_ab_fast_profile.py \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --questions-file evaluation/fixtures/questions_matrix_long.txt \
  --questions-count 10 \
  --graph-strategies default \
  --vllm \
  --output-dir artifacts/experiments \
  --report-dir artifacts/evaluation
```

---

## 5. Generate a question suite

```bash
# from the newest KG run
conda run -n graphllm python scripts/gold/generate_questions.py generate

# from a specific run, in English, with a plain-text copy for the matrix runner
conda run -n graphllm python scripts/gold/generate_questions.py generate \
  --run-dir kg_pipeline/artifacts/run_<tag> \
  --question-language en \
  --output artifacts/tmp/graphrag_test_suite.json \
  --matrix-output artifacts/tmp/graphrag_test_suite_questions.txt

# one document only
conda run -n graphllm python scripts/gold/generate_questions.py generate \
  --doc my_document.pdf --output artifacts/tmp/suite_doc.json

# what came out
conda run -n graphllm python scripts/gold/generate_questions.py stats \
  --input artifacts/tmp/graphrag_test_suite.json
```

`--matrix-output` writes one question per line — no post-processing script
needed. A generated suite is for smoke and sizing work: nothing in it is
source-verified, and no reported number comes from one.

---

## 6. Analyse a run

```bash
# one run: its strategies ranked by latency
python scripts/analysis/analyze_experiments.py artifacts/experiments/<timestamp>_<tag> \
  --save-json results_ranked.json

# across runs: every results.csv one level under the root, aggregated
python scripts/analysis/analyze_matrix.py artifacts/experiments \
  --tag-contains strategy_comparison \
  --save-csv matrix_summary.csv

# GPU/CPU telemetry across runs
python scripts/analysis/analyze_resource_usage.py artifacts/experiments \
  --tag-contains confronto \
  --save-csv resource_report.csv

# what changed between two runs, answer by answer
python scripts/analysis/answer_diff.py --help
```

The input is positional in all three, and the two aggregators are not
interchangeable: `analyze_experiments.py` resolves **one** `results.csv` — a run
directory or the file itself — and raises if the path holds neither, while
`analyze_matrix.py` walks `<root>/*/results.csv` and is the one `--tag-contains`
filters. Output goes to `--save-json`, and to `--save-csv` on the two that have
it; `analyze_experiments.py` writes JSON only.

---

## 7. Score a run

The paper path — two channels, two levels:

```bash
conda run -n graphllm python evaluation/scripts/score_gold_run.py \
  --run-dir exp_results/<run_dir>/ \
  --gold evaluation/gold/gold_v3.json \
  --out-prefix artifacts/evaluation/<name>
```

> `--gold` defaults to `evaluation/gold/gold.json`, an **older set** that differs
> from `gold_v3.json` in `expected_entities` on 7 of the 30 questions. Always pass
> `--gold` explicitly for numbers you intend to report.

The evalkit toolkit, judge and RAGAS are documented in
**[evaluation/README.md](evaluation/README.md)**.

---

## 8. Demos

```bash
# everything up, with a preflight
bash scripts/serving/start_demo.sh --list        # what can be served
bash scripts/serving/start_demo.sh qwen25-32b
bash scripts/serving/stop_demo.sh

# one surface at a time, against servers already running
conda run -n graphllm streamlit run product/app.py
conda run -n graphllm python product/console.py --strategy hybrid --max-context-tokens 6000
```

To reach a remote Streamlit instance, bind and tunnel the **same** port:

```bash
# on the server
conda run -n graphllm streamlit run product/app.py --server.address 0.0.0.0 --server.port 8501
# on your machine
ssh -L 8501:localhost:8501 <user>@<server>    # then browse http://localhost:8501
```

Every exchange is logged to `artifacts/demo_sessions/`. Demo settings are
environment variables — see
[docs/configuration.md](docs/configuration.md#demo-settings).

---

## 9. Health checks and tests

```bash
python scripts/smoke/smoke_check.py                  # imports, graph + both indexes, generator, encoder
python scripts/kg/check_vector_index.py --min-resolving 1000
python scripts/smoke/smoke_kg_retriever.py
python scripts/smoke/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
python scripts/smoke/smoke_dense_rag.py
python scripts/smoke/run_pipeline_smoke_full.py

pytest -q                                            # 1924 tests, any working directory
pytest kg_pipeline/tests/test_pipeline.py -v
pytest evaluation/tests/test_metrics.py -v
```

`smoke_check.py` fills missing variables from `--env-file` (default
`kg_pipeline/.env`) and then a local `.env`; anything already exported wins.
Every check runs by default and a failure is a non-zero exit — waive one with
`--skip-neo4j`, `--skip-llm` or `--skip-encoder`.

Is the server actually up?

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool     # generator
curl -s http://localhost:8002/v1/models | python -m json.tool     # encoder
```

---

## 10. Cluster

```bash
export NEO4J_URL="neo4j+s://<instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"

sbatch scripts/cluster/run_kg_pipeline.sbatch            # detached KG build
sbatch -p <gpu_partition> scripts/cluster/run_graphrag.sbatch
sbatch -p <cpu_partition> scripts/cluster/run_graphrag_cpu.sbatch
sbatch scripts/cluster/run_experiment_matrix_gpu.sbatch
bash   scripts/cluster/submit_matrix_from_env.sh         # parameters from env vars
```

Install `requirements-cpu.txt` on CPU nodes and `requirements-gpu.txt` on GPU
nodes, then `pip install -e .`. Full guide: [docs/cluster.md](docs/cluster.md).

---

## 11. End to end

```bash
# 1. build the graph
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml --env-file kg_pipeline/.env

# 2. repair and index it — retrieval quality depends on the last two
conda run -n graphllm python scripts/kg/kg_postprocess.py --passes 1,2,3,4,5
conda run -n graphllm python scripts/kg/kg_search_index.py
conda run -n graphllm python scripts/kg/kg_vector_index.py

# 3. confirm the whole stack answers
python scripts/smoke/smoke_check.py
python scripts/kg/check_vector_index.py --min-resolving 1000

# 4. run the reference campaign
conda run -n graphllm python -m graphrag.cli --experiment \
  --questions-file evaluation/gold/gold_v3.json \
  --strategies "default,hybrid,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
  --llm --vllm --vllm-base-url http://localhost:8000/v1 \
  --model-id "Qwen/Qwen2.5-32B-Instruct-AWQ" \
  --vector-retrieval --seed-from-retrieved \
  --cite-evidence --complexity medium --max-new-tokens 1024 \
  --output-dir exp_results --experiment-tag campaign_v1

# 5. score it
conda run -n graphllm python evaluation/scripts/score_gold_run.py \
  --run-dir exp_results/<timestamp>_campaign_v1/ \
  --gold evaluation/gold/gold_v3.json \
  --out-prefix artifacts/evaluation/campaign_v1

# 6. read it
cat exp_results/<timestamp>_campaign_v1/summary.txt
cat artifacts/evaluation/campaign_v1.md
```
