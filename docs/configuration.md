# Configuration

Every environment variable the project reads, with the default the code actually
carries. Start from the template:

```bash
cp .env.example .env && $EDITOR .env
```

> Defaults below were read from the `os.getenv` call sites in `src/graphrag/`,
> `kg_pipeline/`, `product/`, `evaluation/` and `scripts/`. A dash means the
> variable has no default and the feature that needs it is off or fails without
> it. The per-model serving wrappers carry their own overrides, documented in
> [`scripts/README.md`](../scripts/README.md).

---

## Neo4j

| Variable | Required | Default | Description |
|---|:---:|---|---|
| `NEO4J_URL` | ✅ | — | Connection URI, e.g. `bolt://localhost:7687` or `neo4j+s://<instance>` |
| `NEO4J_USERNAME` | ✅ | — | Database user |
| `NEO4J_PASSWORD` | ✅ | — | Database password |
| `NEO4J_DATABASE` | — | `""` | Target database name. Empty means the server's default database, not a missing setting |
| `NEO4J_URI` | — | — | Accepted spelling of `NEO4J_URL` |
| `NEO4J_USER` | — | — | Accepted spelling of `NEO4J_USERNAME` |
| `NEO4J_DB` | — | — | Accepted spelling of `NEO4J_DATABASE` |

**Two readers, and only one of them knows the aliases.** Everything that
connects through [`kg_pipeline/utils/neo4j_env.py`](../kg_pipeline/utils/neo4j_env.py)
— the KG pipeline, the repair passes, the analysis scripts, 33 files in all —
reads both spellings of each name. The retrieval engine under `src/graphrag/`
does not: it reads `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` and
`NEO4J_DATABASE` and nothing else.

**When both spellings of a name are set, the first non-empty one in the
resolver's own order wins** — and that order is not the same for all three:
`NEO4J_URI` beats `NEO4J_URL`, `NEO4J_USER` beats `NEO4J_USERNAME`, but
`NEO4J_DATABASE` beats `NEO4J_DB`. A stale `NEO4J_URI` left in a `.env`
therefore points the pipeline at one graph while the engine answers from
another, with nothing in either log saying so. Ports 7688 and 7689 are both live
here and the hosted graph serves the demo, so set one spelling per name and
delete the other.

> **APOC is a hard dependency.** Every node and triple projection goes through
> `apoc.map.removeKey` to strip the embedding vector from the returned
> properties. There is no fallback projection — without APOC, retrieval raises.

---

## Generation endpoint

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM or other OpenAI-compatible endpoint |
| `VLLM_MODEL_NAME` | `""` | Model name served there |
| `VLLM_API_KEY` | falls back to `OPENAI_API_KEY`, then `EMPTY` | API key, where the endpoint wants one |
| `HF_TOKEN` | — | Hugging Face token for gated models. `HUGGINGFACE_HUB_TOKEN` is accepted as an alias |

---

## Embedding endpoint

The multilingual encoder behind the vector channel — the only retrieval channel
that crosses the Italian/English gap.

| Variable | Default | Description |
|---|---|---|
| `GRAPHRAG_EMBED_BASE_URL` | `http://localhost:8002/v1` | OpenAI-compatible `/embeddings` endpoint |
| `GRAPHRAG_EMBED_MODEL` | `intfloat/multilingual-e5-base` | Encoder id. **Must match the one the index was built with** |

```bash
bash scripts/serving/start_vllm_encoder.sh        # GPU 1, port 8002, pooling runner
```

The wrapper script exists because this command used to live only inside an abort
message, and a mistyped restart cost a campaign its vector channel on three of
six models. Changing `GRAPHRAG_EMBED_MODEL` means rebuilding the index with
`scripts/kg/kg_vector_index.py`.

---

## Retrieval and runtime knobs

All optional. Each is read from the environment at call time.

### Indexes and projections

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_FULLTEXT_INDEX` | `node_search` | Full-text index name |
| `GRAPHRAG_VECTOR_PROPERTY` | `embedding` | Property stripped from node projections |
| `GRAPHRAG_VECTOR_ALLOW_DEGRADED` | `""` (off) for the CLI, `1` for the demo | `1` lets a failed encoder degrade to lexical-only instead of raising. `product/config.py` sets it with `setdefault`, so both demos degrade and say so on the affected answer; the CLI keeps raising, because a campaign scored under two retrieval methods is not recoverable — see [Reproducibility](../README.md#reproducibility-notes) |
| `GRAPHRAG_TEXT_STAGE0_RUNS` | `""` | Default for `--text-stage0-runs` |

### Logging

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_LOG_FILE` | `""` (console only) | Path to a campaign log file, added beside the console handler by the CLI. An environment variable rather than a flag: every campaign flag is part of the experiment's identity and recorded in `config.json`, while where the log is written is not |
| `GRAPHRAG_LOG_PROMPT_TEXT` | `""` (off) | `1` puts the rendered prompt and the raw answer back at INFO. They are at DEBUG by default: those two lines were ~26 % of a campaign log, and the answer is written from the retrieved passages, verbatim when `--prefer-verbatim-definitions` is on, so a world-readable log carried third-party PDF text |

### Gate and conversation

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_GATE_MODE` | `evidence` | Which domain gate runs when the gate is on. `evidence` judges what retrieval actually returned; `scope` restores the older gate, which judged the question as typed against the prompt's `domain_scope`. Read per call rather than at import, so both can be compared in one process. `evidence` became the default after measurement on 79 labelled questions: same score (0 wrong refusals of 53, 19 correct of 23) with the conjunction bypass closed, and correct refusals on the regression harness went 3/4 to 4/4 with wrong refusals still 0/18 |
| `GRAPHRAG_TRANSCRIPT_MAX_CHARS` | `16000` | Character budget for the conversation transcript carried into a follow-up. Characters, not turns: an answer runs 2.5k–5k characters, so a turn budget would swing by a factor of two. 16k is about five stripped answers, comfortable inside a 32k window that also holds ~3k of retrieved context. A value that is not a positive integer falls back to the default |

Turning the gate **on or off** is separate: `--enable-domain-gate` on the CLI,
`DEMO_DOMAIN_GATE` in the demo. This variable only picks which of the two gates
runs once it is on.

### Retries and timeouts

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_EMBED_RETRIES` | `3` | Encoder retries before the channel gives up |
| `GRAPHRAG_EMBED_RETRY_BACKOFF_SEC` | `0.5` | Backoff between encoder retries |
| `GRAPHRAG_EMBED_MAX_CHARS` | `1700` | Truncation applied before the encoder's context window |
| `GRAPHRAG_NEO4J_QUERY_RETRIES` | `3` | Transient-error retries per Cypher query |
| `GRAPHRAG_NEO4J_QUERY_RETRY_BACKOFF_SEC` | `1.0` | Backoff between Cypher retries |
| `GRAPHRAG_NEO4J_QUERY_TIMEOUT_SEC` | `45` | Cap on one Cypher query. Measured on the live graph, 34 of 36 queries in a retrieval finish under 0.23 s and the two slow ones are the unindexed `CONTAINS` scan at ~24 s, so this clears the slowest observed query with room to spare |
| `GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC` | `8` | Driver retry window per query (its own default is 30). At 30, one unreachable graph cost **301 s** of waiting in a measured demo session, because every query in a retrieval burned the window independently; at 8 the same failure took 119 s |
| `GRAPHRAG_NEO4J_CONNECTION_TIMEOUT_SEC` | `5` | TCP connect timeout |
| `GRAPHRAG_NEO4J_ACQUISITION_TIMEOUT_SEC` | `10` | Wait for a pooled connection |
| `GRAPHRAG_LLM_GENERATE_RETRIES` | `2` | Transient-error retries per LLM call |
| `GRAPHRAG_LLM_GENERATE_RETRY_BACKOFF_SEC` | `1.0` | Backoff between LLM retries |
| `GRAPHRAG_LLM_HTTP_TIMEOUT_SEC` | `300` | Client timeout on the interactive path, where someone is waiting |
| `GRAPHRAG_VLLM_HEALTHCHECK_TIMEOUT_SEC` | `5` | Endpoint health-check timeout |
| `VLLM_HTTP_TIMEOUT` | `900` | OpenAI-client timeout inside the KG pipeline, where nobody is |

### Local model placement

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_OFFLOAD_DIR` | `/tmp/graphrag-offload` | Offload target for local models |
| `GRAPHRAG_CPU_OFFLOAD_GIB` | `64` | CPU offload budget |
| `GRAPHRAG_TORCH_COMPILE` | `""` (off) | Opt into `torch.compile` for local models |
| `GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK` | `""` (off) | Environment equivalent of `--allow-large-model-fp16-fallback` |

### KG pipeline

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_LLM_CONCURRENT_REQUESTS` | `8` | Concurrency in stage-3 extraction and stage-4 merge confirmation |
| `KG_EXTRACTION_MAX_TOKENS` | `4096` | Output cap per extraction call |
| `KG_EXTRACTION_MAX_TOKENS_CEILING` | `4 x KG_EXTRACTION_MAX_TOKENS` | How far the cap may be raised for one chunk that came back truncated. A chunk cut off at the cap is retried with a doubled budget up to this ceiling, then reported as lost rather than dropped in silence |
| `KG_EXTRACTION_RETRY_TEMPERATURE` | `0.3` | Temperature from the second attempt on. At temperature 0 vLLM decodes greedily and ignores the seed, so a retry that varied only the seed re-sent an identical request; the first attempt still runs at the configured temperature, so a chunk that succeeds first time stays deterministic |
| `KG_NER_BATCH_SIZE` | `16` | Chunks per GLiNER forward pass. `gliner.batch_size` in `config.yaml` wins over it |
| `KG_NER_DEVICE` | `""` | Device placement for GLiNER |
| `KG_EMBED_DEVICE` | — | Device placement for the resolution encoder |
| `KG_PIPELINE_DEBUG_OPENAI` | `""` (off) | Log raw extraction requests and responses |
| `KG_ISOLATED_DELETE_MAX` | `500` | Safety cap on the isolated-node cleanup in `neo4j_postprocess`. Over the cap the pass writes nothing, becomes a dry run and exits non-zero, so the sample can be read before anything is deleted. 500 is an order of magnitude above the 41 the guarded query returns on the demo graph and two below the 14 561 it returned unguarded — a run with thousands of candidates has stopped meaning what it meant. Raise it only after reading the sample |
| `PYTHONHASHSEED` | — | Export **before** launching if set-iteration order must be reproducible. CPython reads it at interpreter startup, so the pipeline cannot set it from inside; it warns when it is unset |

---

## Demo settings

The two demos in `product/` build their own config from `product/config.py`.
Every setting there is an environment variable with the value the demo ships
with, so nothing needs editing to try something else.

| Variable | Default | Effect |
|---|---|---|
| `DEMO_STRATEGY` | `hybrid` | Retrieval strategy |
| `DEMO_COMPLEXITY` | `high` | Answer depth |
| `DEMO_MAX_NEW_TOKENS` | `2048` | Generation cap |
| `DEMO_MAX_CONTEXT_TOKENS` | `6000` | Compressed-context cap |
| `DEMO_CITATION_POLICY` | `mark` | Invented-tag handling |
| `DEMO_CITATION_DISPLAY` | `label` | `[Document, p. 12]` instead of `[S1]` |
| `DEMO_TEXT_RETRIEVER_BACKEND` | `dense` | Text channel backend |
| `DEMO_DENSE_EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Encoder for the dense text backend; recorded in the resolved config |
| `DEMO_TEXT_TOP_K` | `8` | Text chunks retrieved |
| `DEMO_TEXT_MAX_PER_DOC` | `2` | Cap on chunks from one document |
| `DEMO_TEXT_MMR_LAMBDA` | `0.7` | MMR relevance/diversity balance |
| `DEMO_NEO4J_FALLBACK_URL` | `""` | Graph used when the primary one does not answer |
| `DEMO_ENV_FILE` | `kg_pipeline/.env` | Where the demo reads credentials |
| `DEMO_LOG_DIR` | `artifacts/demo_sessions` | Session transcripts — what people asked and were told, meant to become access-restricted |
| `DEMO_LOG_DIR_RUNTIME` | `artifacts/demo_logs` | Operational output: `graphrag.log`, and where `start_demo.sh` puts `streamlit.log`. Deliberately not `DEMO_LOG_DIR`, so restricting the transcripts does not hide the logs an operator needs |
| `DEMO_PRODUCT_NAME` | `Assistente CEFF` | Name on the page and in the browser tab |
| `DEMO_PRODUCT_TAGLINE` | (Italian line) | The sentence under the name |
| `DEMO_PRODUCT_TAGLINE_EN` | (English line) | Same, when the interface is in English |
| `DEMO_PRODUCT_ICON` | `🌾` | Browser-tab icon |
| `DEMO_CITATION_STYLE` | `dim` | `dim` sets the inline citations small, grey and italic; `plain` leaves the engine's `[Document, p. 12]` |
| `DEMO_CITATION_DOC_CHARS` | `60` | Longest document title inside an inline citation |
| `DEMO_TITLE_OVERRIDES` | `product/corpus_titles.json` | Titles for documents whose own first page does not give one |
| `DEMO_UI_LANGUAGE` | `it` | Interface language at startup; the reader can switch it |
| `DEMO_FALLBACK_LANGUAGE` | `DEMO_UI_LANGUAGE` | Answer language when neither the question nor the conversation marks one — a bare term such as `scotta` |
| `DEMO_DEBUG` | `0` | Show the strategy, the model id and the graph URL on the page |
| `DEMO_DOMAIN_GATE` | `1` | Judge a question against the collection before retrieving. `0` answers everything |
| `DEMO_META_REPLY` | `1` | Answer a greeting or a question about the assistant ("ciao", "chi sei?", "prova, sistema operativo?") with an introduction and the example questions, without retrieving. `0` sends them to retrieval |
| `DEMO_MEMORY` | `1` | Intra-session memory: follow-up rewriting and the conversation transcript |
| `DEMO_VECTOR_RETRIEVAL` | `1` | The embedding channel. `0` leaves retrieval lexical only |
| `DEMO_CITE_EVIDENCE` | `1` | Numbered evidence and reference tags on specific claims |
| `DEMO_ENFORCE_LANGUAGE` | `1` | Answer in the question's language, with one retry |
| `DEMO_PARAMETRIC_FALLBACK` | `1` | May answer from model knowledge when the context does not cover the question, marked as such |
| `DEMO_VERBATIM_DEFINITIONS` | `1` | A definitional question opens with the source's own wording |
| `DEMO_ALWAYS_LIMITS` | `1` | Close every answer with a limits section, not only sparse ones |
| `DEMO_SHOW_FULL_ANSWER` | `1` | Show the whole answer including the graph-evidence block |
| `DEMO_TEXT_STAGE0_RUNS` | two run names | Which `kg_pipeline/artifacts` runs feed the text index, most authoritative first |
| `DEMO_TEXT_MMR` | `1` | Diversify text chunks across documents instead of taking the top scores |
| `DEMO_NEO4J_FALLBACK_USERNAME` | — | Credentials for the fallback graph, used with `DEMO_NEO4J_FALLBACK_URL` |
| `DEMO_NEO4J_FALLBACK_PASSWORD` | — | idem |
| `DEMO_NEO4J_FALLBACK_DATABASE` | — | idem |
| `DEMO_VLLM_ENDPOINTS` | `:8000,:8001,:8003` | Endpoints probed for the model selector; unreachable ones disappear |
| `DEMO_EXAMPLE_QUESTIONS` | 3 questions, separated by `\|` | Offered when a question is refused as out of domain |

Every `DEMO_*` boolean above is read the same way: **`1` is on, anything
else is off** (`product/config.py`, `_flag`). They default to on, so the
demo's behaviour is the full configuration unless a variable turns a piece
of it off.

```bash
DEMO_STRATEGY=default DEMO_COMPLEXITY=medium \
  conda run -n graphllm streamlit run product/app.py
```

Change demo behaviour here, never in `graphrag.config` or `graphrag.strategies`:
those are what the campaigns were measured with, and editing them makes future
runs incomparable with the ones already reported.

---

## Evaluation and graph reports

Read by the scorer, the judge backends and `scripts/analysis/kg_evaluator.py`.
None of them is needed for a default run.

| Variable | Default | Effect |
|---|---|---|
| `EVAL_NORMALIZE_REMOVE_ACCENTS` | `1` | Strip accents when building the join key that matches a run's questions to the gold. `0`, `false` or `no` keeps them, so `caffè` and `caffe` stop matching. Changing it changes which answers are scored at all — leave it alone unless a gold set is deliberately accent-sensitive |
| `KG_EVALUATOR_SAMPLE_LIMIT` | `2000` | Nodes read for the degree distribution and the property coverage. Both are therefore a **sample**, not the whole graph, on any graph larger than this |
| `KG_EVALUATOR_RELTYPE_MAX_TYPES` | `250` | Relationship types checked for endpoint-label consistency, taken in the order the type list comes back |
| `KG_EVALUATOR_OUT` | `""` | Write the report here instead of `artifacts/kg_reports/kg_report_<timestamp>.json`. Parent directories are created; an existing file is overwritten, and a fixed path means successive runs stop being separately readable |

### The `claude_code` judge backend

`graphrag-eval --backend claude_code` drives the local `claude` binary under
Pro/Max subscription auth instead of a metered API key. The binary must be
installed and logged in; the prompt goes in on stdin, so its length is not
bounded by argument limits.

| Variable | Default | Effect |
|---|---|---|
| `CLAUDE_CODE_BIN` | `claude` | Path to the binary; `--claude-bin` wins over it. A missing binary is reported as "claude CLI not found", not as a judge failure |
| `CLAUDE_CODE_TIMEOUT` | `300` | Per-call timeout in seconds. A timeout is retried, three attempts with backoff |
| `CLAUDE_CODE_EXTRA_ARGS` | `""` | Extra CLI arguments, whitespace-separated, appended after the backend's own |

---

## How `.env` is loaded

Not uniform across entry points, and it has caused confusion before:

| Entry point | `.env` handling |
|---|---|
| `scripts/smoke/smoke_check.py` | Loads `--env-file` (default `kg_pipeline/.env`), then a local `.env`, both with `override=False` — anything already exported wins |
| `python -m kg_pipeline.main` | Loads the file given by `--env-file`. Pass it explicitly |
| `python -m graphrag.cli` | Reads exported variables. Source your `.env` or run under a wrapper that does |
| `product/app.py`, `product/console.py` | Load `DEMO_ENV_FILE` (default `kg_pipeline/.env`) |

---

## Verify the configuration

```bash
python scripts/smoke/smoke_check.py
```

Checks imports, the graph (node count plus **both** indexes `ONLINE`), the
generator and the encoder. Every check runs by default and a failure is a
non-zero exit; waive one with `--skip-neo4j`, `--skip-llm` or `--skip-encoder`.

A carrier count alone cannot tell a live vector index from one whose identifiers
went stale under a store reload. Check that carriers still resolve:

```bash
python scripts/kg/check_vector_index.py --min-resolving 1000
```
