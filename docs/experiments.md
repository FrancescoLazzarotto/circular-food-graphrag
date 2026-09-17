# Running Experiments

From a populated graph to a scored table. For the option surface see
[cli.md](cli.md); for scoring see
[evaluation/README.md](../evaluation/README.md).

---

## Reference sets

Two frozen sets ship with the repository, 30 questions each, same annotation:

| File | Language | Notes |
|---|---|---|
| [`gold_v3.json`](../evaluation/gold/gold_v3.json) | English | The reference set every thesis number is measured on |
| [`gold_v3_it.json`](../evaluation/gold/gold_v3_it.json) | Italian | Same expected entities, relations, expected answer and scoring block; only `query` changes, and `query_en` carries the original. Built by `scripts/gold/build_gold_it.py` |

Each entry carries `query_id`, `query_type`, `query`, `expected_answer`,
`expected_entities`, `expected_relations`, `source_verified` and `scoring`; the
Italian set adds `query_en`. The file's `_meta.field_docs` block documents every
annotation field in place.

> **`evaluation/gold/gold.json` is an older set, not an alias.** It shares the
> same 30 `query_id`s but differs from `gold_v3.json` in `expected_entities` on
> Q10, Q17, Q18, Q25, Q26, Q27 and Q28. Several tools default to it. Pass
> `--gold evaluation/gold/gold_v3.json` explicitly on anything whose numbers you
> intend to report.

Passing a `.json` gold straight to `--questions-file` makes the run emit
`query_id`, so the evaluator joins by id rather than by question text.

---

## Which runner to use

| | `python -m graphrag.cli --experiment` | `scripts/runners/run_retrieval_matrix.py` |
|---|:---:|:---:|
| GraphRAG strategies | ✅ | ✅ |
| Standard-RAG baselines — tfidf / dense presets | ❌ | ✅ |
| Resource telemetry — CPU/RAM/GPU | ❌ | ✅ |
| `query_id` carried into `results.jsonl` | ✅ | ❌ |
| Vector channel, citations, domain gate, complexity, … | ✅ | ❌ |

Use the CLI for anything the gold evaluation will score. Use the matrix runner
for Standard-RAG comparisons and sizing studies, and read its numbers as stock
defaults — it cannot express the newer `AgentConfig` options.

---

## A campaign through the CLI

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

---

## Prepared campaign drivers

Each script runs a whole family of arms in **one server session**, so the
comparison is within-session and the cross-session noise band does not apply.
All three preflight the generator, the encoder, and — critically — that the
vector index still *resolves*.

| Script | What it measures |
|---|---|
| [`scripts/runners/run_abstention_arms.sh`](../scripts/runners/run_abstention_arms.sh) | Three arms isolating the abstention path: `a0` pre-repair prompt wording, `a1` repaired wording, `a2` repaired wording plus domain gate |
| [`scripts/runners/run_italian_arm.sh`](../scripts/runners/run_italian_arm.sh) | The same 30 questions asked in Italian. Its control is the `a1` arm above; 44% of expected concept slots exist in the graph only under an Italian name, against 22% reachable under an English one |
| [`scripts/runners/run_gold_variant.sh`](../scripts/runners/run_gold_variant.sh) | One gold campaign per KG variant against the local staging graph — comparable to each other, not to hosted-instance runs |

```bash
bash scripts/runners/run_abstention_arms.sh
bash scripts/runners/run_italian_arm.sh
VARIANT=v2_baseline bash scripts/runners/run_gold_variant.sh
```

All three pass `--profile thesis_campaign` rather than spelling the configuration
out. That is the same configuration the earlier campaigns ran with, verified
field by field against a recorded run — see
[Profiles](cli.md#profiles). What stays on each command line is what a profile
cannot carry: question set, model, endpoints, corpus and output location. The
abstention arms add `--legacy-insufficiency-wording` and `--enable-domain-gate`
per arm, which is what that experiment varies.

> [!WARNING]
> A carrier count cannot tell a live vector index from one whose identifiers went
> stale under a store reload — the count passes, the channel silently degrades to
> lexical, and the campaign looks complete. Measured once, that cost 0.03–0.06
> concept F1 on every graph strategy. Guard with
> `python scripts/kg/check_vector_index.py --min-resolving 1000`, which counts
> carriers that still resolve to a node.

---

## Retrieval matrices

```bash
# smoke matrix — always run this before a long job
python scripts/runners/run_retrieval_matrix.py \
  --smoke \
  --questions-file artifacts/experiments/questions_smoke.txt \
  --documents docs/ README.md \
  --runs-per-strategy 1 \
  --output-dir artifacts/experiments \
  --experiment-tag retrieval_matrix_smoke

# full vLLM-backed matrix
python scripts/runners/run_retrieval_matrix.py \
  --llm --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --questions-file evaluation/fixtures/questions_matrix_long.txt \
  --graph-strategies default \
  --runs-per-strategy 1
```

`--questions-file` accepts plain text (one question per line) and JSON suites
from `scripts/gold/generate_questions.py`. Confirm that `summary.json` and
`results.jsonl` appear in the output directory before committing to a long run.

---

## Generating a question suite

```bash
conda run -n graphllm python scripts/gold/generate_questions.py generate
conda run -n graphllm python scripts/gold/generate_questions.py generate --question-language en
conda run -n graphllm python scripts/gold/generate_questions.py stats --input artifacts/tmp/graphrag_test_suite.json
```

Defaults to the most recent `kg_pipeline/artifacts/run_*` directory and writes to
`artifacts/tmp/graphrag_test_suite.json` unless `--output` is given.
`--matrix-output` emits one question per line as text.

A generated suite is a convenience set for smoke and sizing work. It is not a
gold set: nothing in it is source-verified, and no thesis number comes from one.

---

## Run output layout

```text
<output-dir>/<timestamp>_<tag>/
├── results.jsonl           # one record per question/strategy/run
├── results.csv             # tabular version
├── summary.txt             # fast human-readable check
├── summary.json            # structured statistics per strategy
├── config.json             # CLI args + graph_target + resolved AgentConfig per strategy
├── resource_samples.jsonl  # raw telemetry samples (matrix runner)
└── resource_summary.json   # peak and average resource usage (matrix runner)
```

`config.json` is what makes a metric traceable to its configuration. Its
`graph_target` block records the Neo4j URL and database and the embedding
endpoint and model actually used — the password is deliberately not recorded — so
"was this run against staging or against the hosted instance?" is answerable from
the outputs alone.

---

## Analysis

| Script | Purpose |
|---|---|
| `scripts/analysis/analyze_experiments.py` | Analyse a single run directory |
| `scripts/analysis/analyze_matrix.py` | Aggregate multiple runs into CSV/JSON summaries |
| `scripts/analysis/analyze_resource_usage.py` | Sizing and resource comparison across runs |
| `scripts/analysis/answer_diff.py` | Side-by-side answer comparison between runs |
| `scripts/analysis/provenance_precision.py` | Attribute retrieved text back to its origin documents |
| `scripts/analysis/kg_variant_significance.py` | Significance testing across KG variants |
| `scripts/analysis/kg_evaluator.py` | Structural report on the live graph — labels, degrees, property coverage — written to `artifacts/kg_reports/` |
| `evaluation/scripts/build_results_tables.py` | Build the paper's result tables |
| `evaluation/scripts/hard_subset.py` | Isolate the hard subset of the reference set |

---

## See also

- [cli.md](cli.md) — every flag and its default
- [../evaluation/README.md](../evaluation/README.md) — scoring a finished run
- [cluster.md](cluster.md) — SLURM submission
- [troubleshooting.md](troubleshooting.md) — when a run does not behave
