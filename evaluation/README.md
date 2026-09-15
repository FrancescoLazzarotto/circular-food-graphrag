# Evaluation

Everything that turns a finished run into a number. Two layers, and they answer
different questions:

| Layer | Tool | Question it answers |
|---|---|---|
| **Gold scoring** | `evaluation/scripts/score_gold_run.py` | Did the pipeline surface and state the concepts the gold set says the answer needs? This is the paper path |
| **Toolkit** | `graphrag-eval` / `python -m evalkit.cli` | Retrieval metrics with CIs, text similarity, LLM-as-a-Judge, RAGAS, KG quality, project reports |

Everything here assumes a run directory already exists. See
[docs/experiments.md](../docs/experiments.md) for producing one.

---

## 1. The reference sets

Two frozen sets, 30 questions each, identical annotation:

| File | Language |
|---|---|
| `gold/gold_v3.json` | English — the reference set every thesis number is measured on |
| `gold/gold_v3_it.json` | Italian — same annotation, only `query` changes; `query_en` carries the original |

The Italian set is built by `scripts/gold/build_gold_it.py`. Expected answers and
all annotation stay in English in both, so the cross-lingual arm changes exactly
one variable: the language of the question.

Each entry:

```jsonc
{
  "query_id": "Q01",
  "query_type": "factual_simple",
  "query": "What are the three C's of the Circular Economy for Food framework?",
  "expected_answer": "...",
  "expected_entities": [
    {
      "label": "Capital",
      "normalised_label": "capital",        // concept-level match key
      "alt_labels": ["capitale"],           // accepted variants, incl. Italian forms
      "uri": "urn:ceff:Capital",            // null for benchmark-local concepts
      "vocabulary": "CEFF benchmark extension",
      "mapping_status": "benchmark_local_extension",
      "aligned_to": "http://w3id.org/CEON/ontology/value/Value",
      "note": "..."
    }
  ],
  "expected_relations": [ ... ],
  "source_verified": true,
  "scoring": { ... }
}
```

The file's own `_meta.field_docs` block documents every field in place — read it
there rather than trusting a copy.

> **`gold/gold.json` is an older set, not an alias.** It shares the same 30
> `query_id`s but differs from `gold_v3.json` in `expected_entities` on Q10, Q17,
> Q18, Q25, Q26, Q27 and Q28. `score_gold_run.py` **defaults to it**. Pass
> `--gold evaluation/gold/gold_v3.json` explicitly for anything you intend to
> report, or two people will score the same run and disagree.

`gold/gold_circular_v1.json` (with its `.sha256`) is the earlier frozen circular
set, kept for provenance. `gold/gold_questions_template.csv` and
`gold/gold_schema.md` are for building a gold set on a new corpus.

---

## 2. Gold scoring — the paper path

```bash
conda run -n graphllm python evaluation/scripts/score_gold_run.py \
  --run-dir exp_results/<run_dir>/ \
  --gold evaluation/gold/gold_v3.json \
  --out-prefix artifacts/evaluation/<name>
```

Writes `<prefix>.json` (full counts) and `<prefix>.md` (a compact table).

It scores **two channels**:

- **retrieval channel** — `retrieved_entities` as reported by the run: what the retriever surfaced from the KG. Text-RAG reports none, by design.
- **answer channel** — gold surface forms found in the generated answer by a deterministic gazetteer (`evalkit.metrics.mentions`): what the answer actually says. Symmetric across pipelines, and the only channel where `text_only` and `no_retrieval` can score at all.

at **two levels**:

- **concept level** — normalised surface forms, over all expected entities. `surface_forms` is derived at load time from `normalised_label` + `alt_labels`; it is not a field in the JSON. The pipeline-agnostic retrieval measure.
- **grounding level** — resolved canonical URIs, over `mapping_status == "exact"` entities only. The interoperability measure.

Both channels go through the identical path: shared normalisation, shared
resolver, both levels reported side by side and **never averaged into one
number**. The gap between them is itself a result.

Why the alignment step exists at all: the pipelines expose entities in different
native forms — canonical vocabulary URIs, internal Neo4j node ids, or nothing at
all until they are extracted from the answer text. Comparing raw internal ids
against gold URIs would make every non-URI-native pipeline fail by construction,
not because it missed the concept but because its identifier has a different
name. One shared resolver, declared before any pipeline ran, applied
symmetrically.

### Related scorers

| Script | Purpose |
|---|---|
| `scripts/score_citations.py` · `scripts/collect_citation_coverage.py` | Citation gate behaviour across a run |
| `scripts/answer_text_metrics.py` | Answer-text similarity metrics |
| `scripts/hard_subset.py` · `scripts/collect_hard_subset.py` | Isolate the hard subset of the reference set |
| `scripts/build_results_tables.py` | Build the paper's result tables |
| `scripts/mixed_effects.py` | Mixed-effects modelling across runs |
| `scripts/retrieval_bench.py` | Retrieval-only benchmarking |
| `scripts/plot_thesis_figures.py` · `plot_hop_reach.py` · `plot_kg_subgraph.py` | Figures |

---

## 3. The evalkit toolkit

```bash
graphrag-eval <subcommand>          # or: python -m evalkit.cli <subcommand>
```

| Subcommand | Purpose |
|---|---|
| `build-dataset` | Join run results with gold labels into one row-level CSV |
| `retrieval` | Retrieval metrics with bootstrap confidence intervals |
| `text` | Text similarity metrics |
| `judge` | LLM-as-a-Judge |
| `judge-compare` | Agreement between two judge runs |
| `ragas` | RAGAS generative metrics |
| `kg` | KG quality metrics |
| `gold-triples` | Extract gold triple candidates from Neo4j, or apply reviewed ones to a gold CSV |
| `report-experiment` | Full report for one run |
| `report-project` | Project-level report across runs |
| `baseline-update` | Update `baselines/baseline_metrics.json` from a report |

### Join a run to the gold

```bash
conda run -n graphllm graphrag-eval build-dataset \
  --input exp_results/<run_dir> \
  --gold-file evaluation/gold/gold_v3.json \
  --output artifacts/evaluation/eval_dataset.csv
```

`--tag-contains` filters which runs under `--input` are picked up; `--smoke` and
`--smoke-size` cut it down for a fast pass.

> If the join reports `GOLD JOIN FALLBACK`, the run carried no `query_id` and the
> join matched on question text. That has produced wrong numbers in this project
> before, which is why the warning is loud. Re-run through
> `graphrag.cli --experiment` with a `.json` or `.csv` gold as `--questions-file`.

### Retrieval metrics

```bash
conda run -n graphllm graphrag-eval retrieval \
  --input artifacts/evaluation/eval_dataset.csv \
  --k 10 --n-bootstrap 1000 --ci 0.95 --seed 0 \
  --save-json artifacts/evaluation/retrieval_summary.json \
  --save-csv  artifacts/evaluation/retrieval_summary.csv
```

Reports `entity_coverage`, and `precision_at_k` / `recall_at_k` / `mrr` where
triple labels are available, each with a bootstrap CI.

> A metric with **zero observations reports `None`, not `0.0`**. A printed zero
> reads as "the system scored zero" when it means "never measured".

---

## 4. LLM-as-a-Judge

Scores answers on `factual_correctness`, `completeness`, `groundedness` and
`relevance` — a JSON score plus a rationale per row, with bootstrap CIs.
`abstention` is applied automatically to distractor rows and need not be listed
in `--rubrics`; `answer_correctness` still resolves as a legacy alias for
`factual_correctness`, with a deprecation warning.

Two fairness properties are built in and worth not breaking:

- `groundedness` and `relevance` never see the ground truth (`uses_ground_truth=False`).
- Every pipeline's evidence is rendered through the same `## Retrieved Evidence` block, so the judge cannot tell pipelines apart by format.

| `--backend` | Auth | Cost | Reproducible | When |
|---|---|---|---|---|
| `claude_code` | Claude Code subscription (OAuth) | no extra cost | ✗ — tied to your account | Day-to-day iteration on your own machine |
| `api` | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | pay per call | ✓ — pin model and run | Final published numbers, and anyone reusing the project with their own key |
| `vllm` / `local_hf` | local model | free | ✓ | No Claude access; your own model as judge |

```bash
# subscription backend, batched and resumable
conda run -n graphllm graphrag-eval judge \
  --input artifacts/evaluation/eval_dataset.csv \
  --backend claude_code --model haiku --batch-size 8 --resume \
  --out artifacts/evaluation/judge_haiku

# the same via API, pinned and reproducible
conda run -n graphllm python -m pip install anthropic
conda run -n graphllm graphrag-eval judge \
  --input artifacts/evaluation/eval_dataset.csv \
  --backend api --provider anthropic --model claude-sonnet-4-6 \
  --batch-size 8 --out artifacts/evaluation/judge_api

# agreement between two judges — the robustness table
conda run -n graphllm graphrag-eval judge-compare \
  --a artifacts/evaluation/judge_haiku --b artifacts/evaluation/judge_sonnet \
  --label-a haiku --label-b sonnet \
  --out artifacts/evaluation/judge_compare
```

`--batch-size > 1` (always on for `claude_code`) scores N rows per call, and
`--resume` recovers an interrupted run from `<out>/judge_rows.jsonl` without
re-spending quota. `--claude-bin` points at a `claude` CLI elsewhere on the path.

> The `claude_code` backend drives the coding subscription as a batch judge. It
> is not reproducible by a reviewer who does not have your account. Regenerate
> anything you publish with `--backend api`.

---

## 5. RAGAS (optional)

```bash
conda run -n graphllm python -m pip install -r evaluation/requirements.txt

conda run -n graphllm graphrag-eval ragas \
  --input artifacts/evaluation/eval_dataset.csv \
  --metrics faithfulness,answer_relevancy,answer_correctness,context_precision,context_recall \
  --judge-model Qwen/Qwen2.5-14B-Instruct \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --save-row-csv artifacts/evaluation/ragas_rows.csv \
  --save-summary-json artifacts/evaluation/ragas_summary.json
```

Where run outputs carry no explicit contexts or triples, the context metrics are
skipped and the run reports which ones and why.

---

## 6. Building a gold set for a new corpus

```bash
cp evaluation/gold/gold_questions_template.csv my_gold.csv
```

Required columns: `question`, `ground_truth`. Recommended:
`expected_entities_json`, `gold_triples_json` (both JSON lists).
`gold/gold_schema.md` is the full schema, and `scripts/gold/` holds the helpers —
question generation, Italian build, annotation backfill, AGROVOC lexicon.

`graphrag-eval gold-triples` pulls candidate triples out of Neo4j for review and
applies the reviewed set back into the CSV.

---

## Layout

```text
evaluation/
├── evalkit/          # the toolkit — metrics, judge, KG, reports (python -m evalkit.cli)
├── gold/             # frozen reference sets, template, schema, build scripts
├── scripts/          # score_gold_run.py, results tables, hard subset, figures
├── fixtures/         # question sets for matrix runs and smoke passes
├── baselines/        # baseline_metrics.json, updated via `graphrag-eval baseline-update`
├── tests/            # 248 of the repository's 1924 tests
└── requirements.txt  # RAGAS, ROUGE and plotting extras
```
