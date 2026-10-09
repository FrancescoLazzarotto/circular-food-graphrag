# Command Recipes

Copy-paste command sequences, organised by job. Every flag here was checked
against the code; ports, model ids and paths are those of the project server.

This file holds **recipes**. It deliberately does not repeat the option tables:

- Every `graphrag.cli` flag and its default → **[docs/cli.md](docs/cli.md)**
- Every environment variable → **[docs/configuration.md](docs/configuration.md)**
- The local production graph: install, backups, loading → **[docs/graph_hosting.md](docs/graph_hosting.md)**
- Campaign drivers and run layout → **[docs/experiments.md](docs/experiments.md)**
- What every script under `scripts/` is for → **[scripts/README.md](scripts/README.md)**
- When something misbehaves → **[docs/troubleshooting.md](docs/troubleshooting.md)**

Python commands assume `conda activate graphllm` and `export PYTHONNOUSERSITE=1`
(`~/.local` holds a torch / sentence-transformers pair that shadows the
environment's and breaks the dense encoder). In a script, prefix them with
`conda run -n graphllm`.

> `graphrag-demo` and `python -m graphrag.cli` are the same entry point. If
> `graphrag-demo` exits 126 the console-script shim is stale; the module form
> never depends on it.

---

## 0. What runs where, and which graph a command touches

| Port | Service | Started by |
|---|---|---|
| 7687 | **Production graph** (Neo4j 5.26, systemd user service `graph-db`, loopback only) | `deploy/graph/install.sh` |
| 7689 | Staging graph, where a graph is rebuilt and measured; stopped when idle | `scripts/serving/start_neo4j_staging.sh` |
| 7690 | Lots graph, for trying a lot before production; stopped when idle | by hand |
| 8001 | Generator `RedHatAI/Qwen3.8-27B-INT4`, GPU 1 | `start_demo.sh`, `start_vllm_qwen38_27b.sh` |
| 8002 | Encoder `intfloat/multilingual-e5-base`, GPU 1 | `start_demo.sh`, `start_vllm_encoder.sh` |
| 8000 | The graph's extractor `Qwen/Qwen3-32B-AWQ`, only while extracting | `start_vllm_qwen3_32b.sh` |
| 8600 | Demo UI (Streamlit) | `start_demo.sh` |

GPU 0 runs a service that is not this project's: do not start anything on it.

> **The repository defaults do not name the production graph.**
> `kg_pipeline/.env` and the root `.env` name the hosted Aura instance
> (`588fe1bc`, no longer reachable), `kg_pipeline/config.yaml` names its
> database, and `kg_pipeline/.env` sets `VLLM_BASE_URL` / `VLLM_MODEL_NAME` to a
> generator on `:8000` that is not served. The production graph's settings are
> in `~/.config/graphrag/graph-db.env`, written by `install.sh` with every
> `NEO4J_*` spelling the scripts read. Load them into a shell with:
>
> ```bash
> set -a; source ~/.config/graphrag/graph-db.env; set +a
> ```

Whether a script follows what is exported depends on how it loads its `.env`:

| Follows exported `NEO4J_*` (a `.env` only fills gaps) | Overrides them with its `--env-file` (default `kg_pipeline/.env`): always pass it |
|---|---|
| `graphrag.cli`, `product/app.py`, `product/console.py`, `smoke_check.py`, `check_vector_index.py`, `kg_vector_index.py`, `kg_backup.py`, `relink_vectors.py`, `kg_repair*.py`, `visualize_kg.py`, `kg_evaluator.py`, `smoke_kg_retriever.py`, `run_abstention_arms.sh`, `run_italian_arm.sh` | `kg_pipeline.main`, `kg_search_index.py`, `kg_collapse_aliases.py`, `kg_wipe.py`, `kg_retry_failed.py` |

`kg_search_index.py`, `kg_collapse_aliases.py` and `kg_wipe.py` also take the
database name from `--config` before the env: pass a config whose
`neo4j.database` is `neo4j` (a run's copy), never the default one.

Scripts that take the target as arguments: `graph_lot.py` (`--neo4j-env`),
`kg_restore.py` (`--uri`), the curation passes and `kg_densify.py` (`--uri`,
default staging), `run_gold_variant.sh` (staging, always).

---

## 1. The demo

### Start, stop, restart

```bash
GRAPH_ENV_FILE=~/.config/graphrag/graph-db.env bash scripts/serving/start_demo.sh qwen38-27b
bash scripts/serving/start_demo.sh --list          # model keys, ports, GPUs
bash scripts/serving/stop_demo.sh                  # everything start_demo.sh started
bash scripts/serving/stop_demo.sh streamlit        # one component, by label
```

`start_demo.sh` starts the encoder, then the generators named, runs the
preflight, then the UI on `0.0.0.0:8600`. A server that already answers is
reused, so after a code change in `src/` or `product/` this restarts the UI
alone:

```bash
bash scripts/serving/stop_demo.sh streamlit
GRAPH_ENV_FILE=~/.config/graphrag/graph-db.env bash scripts/serving/start_demo.sh qwen38-27b
```

> Without `GRAPH_ENV_FILE` the demo reads the graph of `kg_pipeline/.env`: the
> preflight fails and the UI does not start. Options: `--no-encoder`, `--no-ui`,
> `--port`; several model keys put several generators in the model selector.
> Only `qwen38-27b`, `qwen25-7b`, `qwen3-30b-a3b` and `gemma4-31b` stay on GPU 1;
> the other keys `--list` shows use GPU 0, which is not this project's.

Logs and pids go to `artifacts/demo_logs/` (`<label>.log`, `streamlit.log`,
`graphrag.log`); every exchange is logged to `artifacts/demo_sessions/`. Demo
settings are `DEMO_*` environment variables — see
[docs/configuration.md](docs/configuration.md#demo-settings).

From your own machine:

```bash
ssh -L 8600:localhost:8600 <user>@<server>    # then browse http://localhost:8600
```

### One surface by hand

```bash
set -a; source ~/.config/graphrag/graph-db.env; set +a

# the UI: its own text encoder takes the first GPU it sees, so name GPU 1
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n graphllm \
  streamlit run product/app.py --server.address 0.0.0.0 --server.port 8600

# the console, for an expert at a terminal ('nuova' starts a new thread, 'esci' quits)
CUDA_VISIBLE_DEVICES=1 python product/console.py
CUDA_VISIBLE_DEVICES=1 python product/console.py --model-id RedHatAI/Qwen3.8-27B-INT4 --vllm-base-url http://localhost:8001/v1
```

### Is everything up?

```bash
curl -s http://localhost:8001/v1/models | python -m json.tool     # generator
curl -s http://localhost:8002/v1/models | python -m json.tool     # encoder

set -a; source ~/.config/graphrag/graph-db.env; set +a
python scripts/smoke/smoke_check.py --llm-base-url http://localhost:8001/v1
python scripts/kg/check_vector_index.py --min-resolving 1000
```

`smoke_check.py` checks imports, the graph with both indexes `ONLINE`, the
generator and the encoder; any failure is a non-zero exit. Without
`--llm-base-url` it probes `VLLM_BASE_URL` from `kg_pipeline/.env`. Waive a
check with `--skip-neo4j`, `--skip-llm` or `--skip-encoder`.
`check_vector_index.py` checks that the vector carriers still resolve to nodes:
after a reload they survive but point at nothing, and the vector channel goes
silent without an error.

### Serving one model at a time

```bash
nvidia-smi                                                       # who holds what, first
bash scripts/serving/start_vllm_encoder.sh                       # :8002, GPU 1; exits 0 if already up
bash scripts/serving/start_vllm_qwen38_27b.sh                    # :8001, GPU 1
VLLM_QWEN3_32B_GPU=1 bash scripts/serving/start_vllm_qwen3_32b.sh   # :8000, the extractor
```

Each wrapper runs `vllm serve` in the foreground: detach it (`setsid nohup …
&`) or use `start_demo.sh`, which does. Start the encoder before a generator,
so it claims its slice of GPU 1 first. GPU 1 holds one large model at a time:
stop the demo's generator (`stop_demo.sh qwen38-27b`) before starting the
extractor.

> `import vllm` is broken inside `graphllm`. The wrappers use the `vllm-serve`
> virtualenv; never run `conda run -n graphllm vllm`.

---

## 2. The production graph

```bash
systemctl --user status graph-db                 # running?
systemctl --user list-timers 'graph-db*'         # next backups and checks
journalctl --user -u graph-db -n 100             # database log
bash deploy/graph/health.sh                      # check now; also writes <backups>/health.json
bash deploy/graph/backup.sh daily                # an extra online backup, no downtime
```

A graph reaches production only through `load.sh`, which dumps the current one
first as the rollback point, or through a lot (§4), which backs it up before
writing:

```bash
bash deploy/graph/load.sh --from-home <stopped neo4j home>                 # dry run: prints the plan
CONFIRM=yes bash deploy/graph/load.sh --from-home <stopped neo4j home>
CONFIRM=yes bash deploy/graph/load.sh --from-dump <backups>/weekly/<stamp>  # back to a weekly dump
CONFIRM=yes bash deploy/graph/load.sh --from-dump <backups>/pre_load/<stamp> # undo a load
```

`load.sh` runs `relink_vectors.py` and the health check itself. Install,
timers, retention and what the health check tests:
[docs/graph_hosting.md](docs/graph_hosting.md).

The staging graph, where a graph is built before it is loaded:

```bash
bash scripts/serving/start_neo4j_staging.sh                     # bolt 7689
JAVA_HOME=<jdk 17+> <staging neo4j home>/bin/neo4j stop          # load.sh --from-home needs it stopped
```

---

## 3. Keep the demo's documents up to date

One command refreshes the registry from the corpus folder, makes OCR copies of
new scanned PDFs, reads the new documents (stage 0) and encodes their passages
into the text index. It writes to a new folder and swaps it in only when every
step passed; nothing is written to any graph.

```bash
python scripts/corpus/update_corpus.py --corpus-dir "<corpus folder>"
python scripts/corpus/update_corpus.py --corpus-dir "<corpus folder>" --gpu 1   # encode on GPU 1, not the CPU
```

The result is the run `kg_pipeline/artifacts/run_corpus_demo` with the registry
`product/corpus_registry.csv`, which the demo offers in its advanced settings as
the whole corpus (`DEMO_FULL_CORPUS_RUNS`). Restart the UI to pick it up. A
document whose file left the folder stays in the demo unless you pass
`--allow-removals`.

The steps one by one, for debugging. Run by hand they write the registry in
place, without the staged swap; stage 0 (reading the documents) runs between
OCR and indexing, as `kg_pipeline.main --stage ingestion` with a graph address
that answers nothing.

```bash
python scripts/corpus/build_registry.py --corpus-dir "<corpus>" --registry <registry copy> --dry-run
python scripts/corpus/ocr_scanned.py --corpus-dir "<corpus>" --registry <registry copy>
CUDA_VISIBLE_DEVICES=1 python scripts/corpus/build_text_index.py --stage0-runs <run> \
  --registry <registry copy> --report coverage.json
```

Document cards — proposed title, authors and year of each document, for the
curators to check; nothing is applied:

```bash
python scripts/corpus/document_cards.py --registry product/corpus_registry.csv \
  --stage0-run <run> --out-dir <folder>          # generator on :8001
```

---

## 4. Add a lot of documents to the graph

A lot is a set of registry documents added to a graph that already exists,
without rebuilding it. Each step is resumable and leaves its decisions in
`kg_pipeline/artifacts/graph_lots/<lot>/`; `ledger.json` next to the lots
records which documents each graph holds.

```bash
python scripts/kg/graph_lot.py prepare --lot <lot> --docs <id1,id2,...> \
    --corpus-dir "<corpus>" --graph bolt://localhost:7687
python scripts/kg/graph_lot.py extract --lot <lot>                    # stages 0-3, extractor on :8000
python scripts/kg/graph_lot.py resolve --lot <lot> --neo4j-env <env>  # stages 4-5 + matching against the graph
python scripts/kg/graph_lot.py write   --lot <lot> --neo4j-env <env>  # backup, then write
python scripts/kg/graph_lot.py curate  --lot <lot> --neo4j-env <env>  # stops once for reading the unions
python scripts/kg/graph_lot.py index   --lot <lot> --neo4j-env <env>
python scripts/kg/graph_lot.py check   --lot <lot> --neo4j-env <env>
python scripts/kg/graph_lot.py status                                 # the ledger
python scripts/kg/graph_lot.py remove  --lot <lot> --neo4j-env <env>  # takes the lot out again
```

`<env>` holds the `NEO4J_*` settings of the target graph:
`~/.config/graphrag/graph-db.env` for production. A remote graph and the
staging graph on 7689 are refused without `--allow-remote` / `--allow-staging`.
`extract` uses the model the graph was extracted with (`prepare` defaults:
`--llm-base-url http://localhost:8000/v1 --llm-model Qwen/Qwen3-32B-AWQ`), so
start the extractor first (§1). The graph's own nodes keep their name, label and
properties: a lot adds nodes and edges tagged `lotto`, and aliases to existing
nodes, which `remove` restores. `remove` refuses while a later lot is still in
the graph, unless `--force`.

---

## 5. Rebuild the graph (staging)

A rebuild runs in its own run folder holding a copy of `config.yaml` with
`neo4j.database: neo4j` and a `rebuild.env` that points at staging. The
repository defaults point elsewhere (§0), so always pass `--config` and
`--env-file` of the run. `rebuild.env` is loaded over the environment and needs:

```bash
NEO4J_URL=bolt://localhost:7689
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<staging password>
NEO4J_DATABASE=neo4j
VLLM_BASE_URL=http://localhost:8000/v1          # the extractor, stage 3
VLLM_MODEL_NAME=Qwen/Qwen3-32B-AWQ
```

```bash
RUN=kg_pipeline/artifacts/run_<tag>
export PYTHONHASHSEED=42      # = `seed` in config.yaml; read only at interpreter start

# stages 0-5: reads documents, extracts, resolves; does not touch the graph
python -m kg_pipeline.main --config $RUN/config.yaml --env-file $RUN/rebuild.env \
  --run-dir $RUN --stage linking

# detached from the terminal, with a log you can tail
setsid nohup env PYTHONNOUSERSITE=1 PYTHONHASHSEED=42 python -m kg_pipeline.main \
  --config $RUN/config.yaml --env-file $RUN/rebuild.env --run-dir $RUN --stage linking \
  >> $RUN/rebuild.out 2>&1 < /dev/null &

# one document (with the default --stage all it is written to staging); or no writes at all
python -m kg_pipeline.main --config $RUN/config.yaml --env-file $RUN/rebuild.env --single-doc documento.pdf
python -m kg_pipeline.main --config $RUN/config.yaml --env-file $RUN/rebuild.env --dry-run
```

| Flag | Effect |
|---|---|
| `--config` | Configuration file (default `kg_pipeline/config.yaml`) |
| `--env-file` | `.env` with Neo4j credentials and endpoints, loaded **over** the environment (default `kg_pipeline/.env`) |
| `--run-dir` | Existing run directory to resume; empty creates a timestamped one |
| `--single-doc` | Process one document (filename or doc_id) |
| `--stage` | `all` `ingestion` `chunking` `ner` `llm` `resolution` `linking` `neo4j` |
| `--run-post` | After stage 6, rebuild the full-text and vector indexes on the graph just written (not densification) |
| `--dry-run` | Run without writing to Neo4j |
| `--log-level` | `DEBUG` `INFO` `WARNING` (default `INFO`) |

> **`--stage` is inclusive-up-to, not isolating.** `--stage ner` runs ingestion,
> chunking and ner, reusing earlier artifacts where they exist. There is no flag
> that runs one stage alone. Resuming with the default `--stage all` goes on into
> stage 6 and writes the graph named in `--env-file`.

> **`PYTHONHASHSEED` must match `seed` in `config.yaml` (42).** Stage 4 picks
> among spellings that differ only in case in set order, which follows the
> string hash: with another seed two runs name those nodes differently. Python
> reads the variable only at startup, so the pipeline can only warn.

> **Stage 3 checkpoint.** Progress is saved every `llm.checkpoint_every` chunks
> to `stage3_checkpoint.json`, written atomically. Re-running without deleting it
> resumes from the last saved chunk; triples from chunks past the last completed
> checkpoint are dropped on resume, so recovery never duplicates. Stage 4 has no
> checkpoint and starts over.

Chunks that failed extraction are listed in `failed_chunks.jsonl`. Retry them
before stage 4 runs (`--stage llm` stops after stage 3):

```bash
python scripts/kg/kg_retry_failed.py --run-dir $RUN --config $RUN/config.yaml --env-file $RUN/rebuild.env
```

If stage 4 already ran, delete the stage 4-5 artifacts and run again: a stage
whose inputs changed stops the run instead of being reused.

### Curate the rebuilt graph

Once stages 0-5 have run (`--stage linking` above: step 1 reads the stage-4
approvals), `replay_curation.sh` runs the curation that produced the current
graph, in order: strict re-judgement of the stage-4 merges, stages 4-6, alias
collapse, cleanup, edge rules, anaphoric nodes, two rounds of bilingual unions,
isolated nodes, indexes.

```bash
bash scripts/kg/curation/replay_curation.sh $RUN
START_STEP=9 bash scripts/kg/curation/replay_curation.sh $RUN      # resume
```

The run's `rebuild.env` must point at the staging graph; the script refuses
otherwise. Step 3 wipes the staging graph. Every model decision is a file in the
run directory (`merge_verdicts.json`, `bilingual_proposals_round{1,2}.json`,
`bilingual_excluded.json`): with them in place the replay makes no model call.
Without them, steps 1 and 9-10 need `Qwen/Qwen3-32B-AWQ` on
`CURATION_ENDPOINTS` (default `:8000,:8003`); step 12 needs the encoder on
`:8002`. When a bilingual round has no proposals yet, the step writes them and
stops: list the wrong ones in `bilingual_excluded.json`, then resume.

Then measure the staging graph, stop it, and load it into production (§2).

### Re-run resolution without re-running extraction

Tunes similarity thresholds against an existing stage-3 output — no NER, no LLM
extraction.

```bash
python scripts/kg/remerge_entities.py \
  --run-dir $RUN \
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
| `--base-url` / `--model-name` | vLLM endpoint and model for merge confirmation (default `VLLM_BASE_URL` / `VLLM_MODEL_NAME`) |
| `--exclude-mentioned-in` | No `MENTIONED_IN` edges in the linked output |

The merge cache stores raw group indices and is only valid for an unchanged
stage-3 output.

### Densify

Adds edges between entities the graph already has, chunk by chunk; it creates
no entity. Two passes: extract candidates to JSONL, then `--apply`. Default
target: the staging graph; default generator: `:8001`.

```bash
python scripts/kg/kg_densify.py --chunks-dir $RUN --output $RUN/densify.jsonl
python scripts/kg/kg_densify.py --chunks-dir $RUN --output $RUN/densify.jsonl --apply
```

Always pass `--chunks-dir` (the default reads the chunks of older runs, whose
ids collide with different text) and keep `--output` in the run folder:
`--apply` applies every `densify*.jsonl` next to it. Every edge it writes
carries `extraction_method: 'densification'`.

### Indexes

```bash
python scripts/kg/kg_search_index.py --config $RUN/config.yaml --env-file <target env>   # full-text index node_search
python scripts/kg/kg_vector_index.py                              # :NodeVec + vector index; needs the encoder on :8002
python scripts/kg/kg_vector_index.py --only-missing               # after adding nodes
python scripts/kg/check_vector_index.py --min-resolving 1000
```

`kg_search_index.py` loads its `--env-file` over the environment and takes the
database name from `--config` before the env, so pass a config whose
`neo4j.database` is `neo4j`; the other two follow the exported `NEO4J_*`. Pass `--recreate` to `kg_search_index.py` when
new labels appear.

### Back up, restore, wipe

```bash
python scripts/kg/kg_backup.py --output-dir artifacts/kg_backups/<name> --include-vectors   # exported NEO4J_*
python scripts/kg/kg_restore.py --backup-dir <dir> --uri bolt://localhost:7689 --password <pw> --wipe
python scripts/kg/kg_wipe.py --config $RUN/config.yaml --env-file $RUN/rebuild.env            # counts only
python scripts/kg/kg_wipe.py --config $RUN/config.yaml --env-file $RUN/rebuild.env --yes      # wipes
```

A JSON restore changes record ids, so the vectors come back pointing at
nothing. After it: `kg_search_index.py`, then `kg_vector_index.py --drop`,
`kg_vector_index.py`, `check_vector_index.py`. After a `neo4j-admin` load,
`relink_vectors.py` repairs the pointers instead (`load.sh` runs it).

### Older passes

`kg_postprocess.py` runs the repair rounds `kg_repair.py` … `kg_repair5.py` of
the July graph; the curation above does not use them. `--passes` defaults to
`1,2,3,4`; pass 5 is opt-in. Each round reads `VLLM_BASE_URL` and
`VLLM_MODEL_NAME` from `kg_pipeline/.env` unless exported, and only prints its
plan unless `KG_REPAIR_CONFIRM=yes` (a hosted graph also needs
`KG_ALLOW_HOSTED_WRITES=<host>`):

```bash
KG_REPAIR_CONFIRM=yes \
VLLM_BASE_URL=http://localhost:8001/v1 VLLM_MODEL_NAME=RedHatAI/Qwen3.8-27B-INT4 \
  python scripts/kg/kg_postprocess.py --passes 1,2,3,4,5
```

### Inspect the graph

```bash
python scripts/analysis/visualize_kg.py --out artifacts/tmp/kg_viz.html
python scripts/analysis/kg_evaluator.py      # structural report → artifacts/kg_reports/; no flags
```

---

## 6. Ask one question

```bash
set -a; source ~/.config/graphrag/graph-db.env; set +a     # the graph to ask

# retrieval only, no generation
python -m graphrag.cli \
  --question "What is food waste?" --entity "spreco alimentare" --strategies default

# a grounded, cited answer from the demo's own documents
CUDA_VISIBLE_DEVICES=1 python -m graphrag.cli \
  --question "What is food waste?" --strategies hybrid \
  --llm --vllm --vllm-base-url http://localhost:8001/v1 --model-id RedHatAI/Qwen3.8-27B-INT4 \
  --cite-evidence --citation-display label --prefer-verbatim-definitions \
  --enforce-language --focused-answer --complexity high \
  --vector-retrieval --enable-domain-gate \
  --text-retriever-backend dense --text-retriever-mmr --text-retriever-max-per-doc 2 \
  --text-stage0-runs run_fix2docs_20260710,run_full_circular_20260707
```

- `--vllm-base-url` defaults to `:8000` and `--model-id` to
  `Qwen/Qwen2.5-7B-Instruct`, neither served: pass both with `--llm --vllm`.
- The second recipe is close to the demo, not identical: the demo profile also
  sets fields the CLI has no flag for, so `--profile demo` is refused (use it
  through `graphrag.profiles`). `--profile thesis_campaign` and
  `--profile research_baseline` work; a flag given explicitly wins over the
  profile.
- `--text-stage0-runs` defaults to `GRAPHRAG_TEXT_STAGE0_RUNS`, and without it
  to the newest run only; the two runs above are the demo's own documents
  (`DEMO_TEXT_STAGE0_RUNS`).
- The dense text backend encodes on the first GPU the process sees: name GPU 1,
  or `CUDA_VISIBLE_DEVICES=""` for the CPU. GPU 0 is not this project's.
- In single-question mode only the **first** entry of `--strategies` is applied.
- `--vector-retrieval` adds the multilingual vector channel beside the lexical
  one, so an English question reaches Italian node names. It needs the encoder
  on `:8002` and the vector index (§5).
- `--enable-domain-gate` makes one classification call before retrieval and
  refuses out-of-domain questions; without it every question reaches
  generation.

---

## 7. Run a campaign

Preconditions: generator on `:8001`, encoder on `:8002`, the target graph
exported, and `check_vector_index.py` passing. Prefer the drivers: they carry
the fixed flag block and preflight themselves.

```bash
VARIANT=<name> bash scripts/runners/run_gold_variant.sh   # one graph variant, on the staging graph (7689)
bash scripts/runners/run_abstention_arms.sh                # arms A0/A1/A2, on the exported graph
bash scripts/runners/run_italian_arm.sh                    # after the arms, in the same server session
```

The same block by hand:

```bash
python -m graphrag.cli --experiment \
  --questions-file evaluation/gold/gold_v3.json \
  --strategies "default,hybrid,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
  --llm --vllm --vllm-base-url http://localhost:8001/v1 --model-id RedHatAI/Qwen3.8-27B-INT4 \
  --profile thesis_campaign --max-new-tokens 1024 \
  --text-docs-dir artifacts/corpus_circular22 --evidence-max-triple-items 30 \
  --output-dir exp_results_<family> --experiment-tag <tag>
```

The output is `<output-dir>/<timestamp>_<tag>/` with `results.jsonl`,
`results.csv`, `summary.txt`, `summary.json` and `config.json` (the CLI args
and the resolved configuration per strategy). The campaign profile keeps the
text channel on `tfidf`, the library default; the demo runs `dense` — add
`--text-retriever-backend dense` to measure what the demo runs.
`--legacy-insufficiency-wording` restores the closing line of the answer prompt
used by the thesis campaigns E1-E8. See [docs/experiments.md](docs/experiments.md) for what each
driver measures.

---

## 8. Retrieval matrix — Standard RAG vs GraphRAG

`run_retrieval_matrix.py` is the only runner that produces resource telemetry and
Standard-RAG baselines. It takes **`--standard-strategies` and
`--graph-strategies`**, not `--strategies`, and it has no `--models` flag: pass a
single `--model-id`. It builds its configuration from a handful of fields, so
its numbers are not comparable with `graphrag.cli --experiment`.

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
  --vllm-base-url http://localhost:8001/v1 \
  --model-id RedHatAI/Qwen3.8-27B-INT4 \
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
| `--llm` / `--vllm` / `--vllm-base-url` / `--model-id` / `--llm-warmup` | Generation; the endpoint defaults to `:8000` |
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
  --model-id RedHatAI/Qwen3.8-27B-INT4 \
  --vllm --vllm-base-url http://localhost:8001/v1 \
  --questions-file evaluation/fixtures/questions_matrix_long.txt \
  --questions-count 10 \
  --graph-strategies default \
  --output-dir artifacts/experiments \
  --report-dir artifacts/evaluation
```

---

## 9. Generate a question suite

The generator reads `VLLM_BASE_URL` (default `:8000`) and `VLLM_MODEL_NAME`
(required) from the environment only:

```bash
export VLLM_BASE_URL=http://localhost:8001/v1 VLLM_MODEL_NAME=RedHatAI/Qwen3.8-27B-INT4

# from the newest KG run
python scripts/gold/generate_questions.py generate

# from a specific run, in English, with a plain-text copy for the matrix runner
python scripts/gold/generate_questions.py generate \
  --run-dir kg_pipeline/artifacts/run_<tag> \
  --question-language en \
  --output artifacts/tmp/graphrag_test_suite.json \
  --matrix-output artifacts/tmp/graphrag_test_suite_questions.txt

# one document only
python scripts/gold/generate_questions.py generate \
  --doc my_document.pdf --output artifacts/tmp/suite_doc.json

# what came out
python scripts/gold/generate_questions.py stats \
  --input artifacts/tmp/graphrag_test_suite.json
```

`--matrix-output` writes one question per line — no post-processing script
needed. A generated suite is for smoke and sizing work: nothing in it is
source-verified, and no reported number comes from one.

---

## 10. Analyse a run

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

# how far each strategy's answers drift from a baseline strategy's, in one run
python scripts/analysis/answer_diff.py \
  --results <run>/results.jsonl --baseline text_only --output-csv answer_diff.csv
python scripts/analysis/answer_diff.py \
  --results <run>/results.jsonl --baseline text_only --side-by-side answers.md --top 15
```

The input is positional in the first three, and the two aggregators are not
interchangeable: `analyze_experiments.py` resolves **one** `results.csv` — a run
directory or the file itself — and raises if the path holds neither, while
`analyze_matrix.py` walks `<root>/*/results.csv` and is the one `--tag-contains`
filters. Output goes to `--save-json`, and to `--save-csv` on the two that have
it; `analyze_experiments.py` writes JSON only.

---

## 11. Score a run

The paper path — two channels, two levels:

```bash
python evaluation/scripts/score_gold_run.py \
  --run-dir exp_results/<run_dir>/ \
  --gold evaluation/gold/gold_v3.json \
  --out-prefix artifacts/evaluation/<name>       # writes <name>.json and <name>.md
```

> `--gold` defaults to `evaluation/gold/gold.json`, an **older set** that differs
> from `gold_v3.json` in `expected_entities` on 7 of the 30 questions. Always pass
> `--gold` explicitly for numbers you intend to report.

Result tables, significance, the evalkit toolkit, judge and RAGAS are
documented in **[evaluation/README.md](evaluation/README.md)**.

---

## 12. Tests and smoke checks

```bash
pytest -q                                            # 2188 tests, from the repository root
pytest kg_pipeline/tests/test_pipeline.py -v
pytest evaluation/tests/test_metrics.py -v
ruff check src product                               # what CI lints

python scripts/smoke/smoke_kg_retriever.py           # the exported graph
python scripts/smoke/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
CUDA_VISIBLE_DEVICES=1 python scripts/smoke/smoke_dense_rag.py docs/   # takes the files or folders to index
python scripts/smoke/smoke_check.py --check-imports-only
```

Run `pytest` from the repository root: given the repository folder as an
argument, it also collects the copies under `.claude/worktrees/`.

---

## 13. Cluster

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

## 14. End to end

### New documents, into the demo

```bash
# 1. read and index them
python scripts/corpus/update_corpus.py --corpus-dir "<corpus folder>"

# 2. optionally, into the graph as a lot (§4), with the extractor on :8000

# 3. restart the UI and check the stack
bash scripts/serving/stop_demo.sh streamlit
GRAPH_ENV_FILE=~/.config/graphrag/graph-db.env bash scripts/serving/start_demo.sh qwen38-27b
```

`start_demo.sh` runs the preflight (`smoke_check.py`) before it starts the UI
and stops if it fails.

### A rebuilt graph, into production

```bash
RUN=kg_pipeline/artifacts/run_<tag>      # config.yaml + rebuild.env pointing at 7689
export PYTHONHASHSEED=42

# 1. stages 0-5 (extractor on :8000), then the curation, which also writes and indexes the graph
bash scripts/serving/start_neo4j_staging.sh
python -m kg_pipeline.main --config $RUN/config.yaml --env-file $RUN/rebuild.env --run-dir $RUN --stage linking
bash scripts/kg/curation/replay_curation.sh $RUN

# 2. measure it on staging (§7, run_gold_variant.sh), then stop staging and load it
JAVA_HOME=<jdk 17+> <staging neo4j home>/bin/neo4j stop
CONFIRM=yes bash deploy/graph/load.sh --from-home <staging neo4j home>

# 3. restart the UI and check the stack
bash scripts/serving/stop_demo.sh streamlit
GRAPH_ENV_FILE=~/.config/graphrag/graph-db.env bash scripts/serving/start_demo.sh qwen38-27b
```
