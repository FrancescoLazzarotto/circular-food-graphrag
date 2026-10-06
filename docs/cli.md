# CLI Reference — `graphrag.cli`

Every flag of `python -m graphrag.cli` (installed as `graphrag-demo`), with the
default the parser actually carries. This page is the single source of truth for
the option surface; [COMMANDS.md](../COMMANDS.md) holds the task recipes that use
it, and neither repeats the other.

> Defaults below are read from `_build_arg_parser()` in
> [`src/graphrag/cli.py`](../src/graphrag/cli.py). `graphrag-demo --help` prints
> the same list; if the two ever disagree, the parser wins.

Exit code 126 from `graphrag-demo` means the console-script shim is stale from an
older install. `python -m graphrag.cli` never depends on the shim.

---

## Modes

The CLI has two modes and one option surface. Without `--experiment` it answers a
single `--question` and prints to stdout. With `--experiment` it runs
`--questions-file` across `--strategies` and writes a run directory.

In single-question mode only the **first** entry of `--strategies` is applied.

---

## Profiles

| Flag | Default | Effect |
|---|---|---|
| `--profile` | none | Start from a named configuration in [`graphrag.profiles`](../src/graphrag/profiles.py): `thesis_campaign`, `demo`, `research_baseline` |

A profile is a set of overrides against the defaults on this page, so it replaces
flags rather than adding behaviour. `--profile thesis_campaign` is exactly the
sixteen configuration flags the campaign runners used to spell out — ten that
changed something, six that restated a default.

**Precedence.** A flag given explicitly wins over the profile. The profile is
installed with `set_defaults` and the command line is parsed a second time, so
this is argparse's own ordering rather than a rule reimplemented here:

```bash
graphrag-demo --profile thesis_campaign --subgraph-seed-count 7   # 7, not the profile's 3
```

A profile that sets a field with no flag on this page is **refused at startup**,
not applied in part — `demo` is one, because `always_include_limits` and
`text_retriever_top_k` have no flags. Use it through `graphrag.profiles` instead.

Two test suites pin `thesis_campaign` from both ends: `tests/test_profiles.py`
against the configuration a real campaign recorded, and
`tests/test_profiles_match_runners.py` against what the runner scripts resolve to
today. A profile edit that would move a published number fails there, naming the
field.

---

## Question and model

| Flag | Default | Effect |
|---|---|---|
| `--question` | `"Quali sono gli obiettivi della strategia Farm to Fork?"` | The question to answer. The default is a placeholder, not a suggestion |
| `--entity` | `""` | Seed entity for graph traversal. Empty means auto-seeding from the question |
| `--llm` | off | Enable generation. Without it only the retrieval path runs |
| `--model-id` | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face or vLLM model identifier |
| `--vllm` | off | Use an OpenAI-compatible endpoint instead of loading local weights |
| `--vllm-base-url` | `http://localhost:8000/v1` | Endpoint for `--vllm` |
| `--llm-warmup` | off | Preload the model at startup instead of on the first call |

---

## Graph retrieval channels

| Flag | Default | Effect |
|---|---|---|
| `--strategies` | `default` | Comma-separated presets — see [Retrieval strategies](#retrieval-strategies) |
| `--seed-from-retrieved` | off | Anchor the neighbour, subgraph and shortest-path channels on node names retrieval actually returned, instead of on raw question words |
| `--subgraph-seed-count` | `1` | How many anchors the subgraph channel expands from. `1` = best anchor only |
| `--subgraph-limit` | `200` | Triples the subgraph channel may pull per anchor set **before** ranking. Applied in Cypher, so it truncates in graph order and the ranker never sees what was cut |
| `--drop-predicates` | `""` | Comma-separated predicates removed from retrieved triples, e.g. `RELATED_TO,PUBLISHED,AUTHORED_BY` |
| `--evidence-max-triple-items` | `30` | Cap on numbered triple evidence blocks placed in the context |

### The vector channel

Added **beside** the lexical channel, never replacing it. It is the only channel
that crosses the Italian/English gap.

| Flag | Default | Effect |
|---|---|---|
| `--vector-retrieval` | off | Enable the multilingual vector channel |
| `--vector-index` | `node_embedding` | Neo4j vector index name |
| `--vector-nodes-limit` | `10` | Nodes taken from the vector channel per query |
| `--vector-triples-limit` | `10` | Triples taken from the vector channel per query |

Requires `scripts/kg/kg_vector_index.py` to have been run against the live graph
and an embedding endpoint reachable at `GRAPHRAG_EMBED_BASE_URL`. See
[configuration.md](configuration.md#embedding-endpoint).

---

## Text channel

| Flag | Default | Effect |
|---|---|---|
| `--text-retriever-backend` | `tfidf` | `tfidf` — lexical, ranked with Okapi BM25 (k1 1.5, b 0.75); `dense` — cosine over FAISS |
| `--text-docs-dir` | `""` | Documents to index. Empty auto-discovers from the newest `kg_pipeline/artifacts` stage-0 output |
| `--text-stage0-runs` | `$GRAPHRAG_TEXT_STAGE0_RUNS` or `""` | Which stage-0 runs feed the index, most authoritative first — a later run cannot shadow an earlier one on the same filename. Empty means every run, newest first |
| `--text-retriever-mmr` | off | Maximal Marginal Relevance instead of pure top-k, trading query similarity for source coverage. Applies to **both** backends |
| `--text-retriever-mmr-lambda` | `0.7` | MMR relevance/diversity balance; `1.0` is pure similarity |
| `--text-retriever-max-per-doc` | `0` | Cap on chunks from one document; `0` disables it. Enumerative questions get twice this budget |
| `--dense-embedding-model` | `intfloat/multilingual-e5-base` | Model for `--text-retriever-backend dense`; ignored for `tfidf` |
| `--vector-index-dir` | `artifacts/vector_index` | Cache of passage vectors for the dense backend (`passages/<encoder>-<hash>.sqlite`): only passages it does not hold are encoded; ignored for `tfidf` |

---

## Grounding and citations

| Flag | Default | Effect |
|---|---|---|
| `--cite-evidence` | off | Number the evidence, ask for `[S1]`/`[T1]` tags, and verify every tag against the **visible** index — the refs that survived context compression |
| `--citation-policy` | `mark` | What happens to a tag absent from the index: `mark` flags it in place, `strip` deletes it |
| `--citation-display` | `id` | `id` keeps `[S1]`; `label` rewrites it as `[Document, p. 12]` after the gate |
| `--prefer-verbatim-definitions` | off | On a definitional question, rank the defining passage first and open the answer with it, quoted between guillemets and checked against the retrieved text |
| `--allow-parametric-fallback` | off | Permit the model's own knowledge where evidence does not cover the question, marked `(not in the retrieved evidence)` so grounded and ungrounded stay separable |

---

## Answer shape

| Flag | Default | Effect |
|---|---|---|
| `--complexity` | `medium` | `low` / `medium` / `high`. `high` drops the "1–2 short paragraphs" cap and adds the specificity rule (figures, names, article numbers) |
| `--focused-answer` | off | Name only what answers the question, not every related concept the evidence carries |
| `--enforce-language` | off | Pin the answer to the language detected on the question, with one retry on a mismatch |
| `--enable-domain-gate` | off | One classification call before retrieval; refuses out-of-domain questions instead of answering them from an unrelated context |
| `--legacy-insufficiency-wording` | off | Restore the pre-repair closing line of the answer prompt, which allowed a declaration of insufficiency only for an empty context. For reproducing campaigns E1–E8 |

> Without `--enable-domain-gate` the agent has no terminal refusal state: the
> dense retriever has no score floor, so `grade` always sees evidence and every
> question reaches `generate`.

---

## Agent and runtime

| Flag | Default | Effect |
|---|---|---|
| `--enable-decomposition-step` | off | Extra LLM call decomposing the question before retrieval |
| `--enable-adaptive-routing-step` | off | Extra LLM call routing the question before retrieval |
| `--max-new-tokens` | `256` | Cap on generated tokens per response |
| `--max-context-tokens` | `6000` | Cap on the compressed prompt context |
| `--recursion-limit` | `50` | Maximum LangGraph steps before aborting |
| `--gpu-memory-fraction` | `0.92` | Fraction of each GPU reserved for model placement |
| `--allow-large-model-fp16-fallback` | off | For models ≥ 30B, allow fp16 when 4-bit loading fails. Off by default: enable only with the memory/precision trade-off understood |

---

## Batch experiments

| Flag | Default | Effect |
|---|---|---|
| `--experiment` | off | Batch mode: run every question across every strategy and persist a run directory |
| `--questions-file` | — | See the input formats below |
| `--strategies` | `default` | Comma-separated presets; all of them are applied in batch mode |
| `--runs-per-strategy` | `1` | Repetitions per strategy |
| `--output-dir` | `artifacts/experiments` | Where the run directory is created |
| `--experiment-tag` | `""` | Identifier appended to the run directory name |

### `--questions-file` formats

| Extension | Shape | Carries `query_id` |
|---|---|---|
| `.txt` | One question per line, optionally `Q01<TAB>question` | Only with the tab form |
| `.json` | The gold's `{"queries": [{"query_id", "query", …}]}` shape | ✅ |
| `.jsonl` | One such object per line | ✅ |
| `.csv` | `query_id` + `query` columns | ✅ |

**Declaring `query_id` is what lets the evaluator join results to the gold by id
instead of by question text.** The text join is fragile and warns loudly
(`GOLD JOIN FALLBACK`). Passing `evaluation/gold/gold_v3.json` straight to
`--questions-file` is the supported path.

---

## Retrieval strategies

Defined once in [`src/graphrag/strategies.py`](../src/graphrag/strategies.py) and
imported by both the CLI and the matrix runner. Presets toggle only the channel
flags; cardinality limits and ranking options come from the base `AgentConfig`,
and the fully resolved per-strategy config is serialised into every run's
`config.json`.

| Strategy | Nodes | Triples | Neighbours | Subgraph | Shortest path | Text |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `default` | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| `hybrid` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `text_only` | — | — | — | — | — | ✅ |
| `no_retrieval` | — | — | — | — | — | — |
| `text_plus_triples` | ✅ | ✅ | — | — | — | — |
| `neighbors_focus` | — | ✅ | ✅ | — | — | — |
| `subgraph_2hop` | — | ✅ | — | ✅ | — | — |
| `shortest_path` | — | ✅ | — | — | ✅ | — |

`text_only` is the sparse-retrieval baseline; `no_retrieval` is the zero-shot LLM
baseline. Everything else is a GraphRAG variant.

### How many hops the subgraph really uses

The base config carries `hops = 1`, `adaptive_hops = True` and `max_hops = 4`.
With adaptive hops on, the subgraph channel starts at `hops` and widens one hop
at a time until it has collected `min_subgraph_triples` or reaches `max_hops`.

`subgraph_2hop` is not "the 2-hop strategy" against 1-hop siblings — it raises the
**starting floor** to 2 (`hops = max(2, hops)`) and drops the node, neighbour and
shortest-path channels. `default` can reach the same depth or further when a
sparse neighbourhood forces the expansion.

`--text-docs-dir` matters only for the two presets with a text channel: `hybrid`
and `text_only`. On `hybrid` the text pipeline must be passed to the retriever,
otherwise the channel is silently skipped.

---

## Failure behaviour, by channel

This differs on purpose.

| Missing | Behaviour |
|---|---|
| Full-text index | Falls back to a per-term `CONTAINS` scan, with a warning |
| Vector index not built | Lexical only, with a warning naming `kg_vector_index.py` |
| Vector index present but unqueryable (wrong dimensionality, procedure absent) | Lexical only, logged at ERROR with the real cause. Not reported as a missing index: rebuilding would not fix a dimension mismatch, which means the index was built with a different encoder |
| Embedding endpoint failing after its retries | **Raises.** A silent, model-asymmetric change of retrieval method mid-comparison is worse than a stopped run. `GRAPHRAG_VECTOR_ALLOW_DEGRADED=1` restores degradation for interactive use only, and the demos set it |

---

## See also

- [configuration.md](configuration.md) — every environment variable
- [experiments.md](experiments.md) — runners, campaign drivers, output layout
- [../COMMANDS.md](../COMMANDS.md) — task recipes
- [../evaluation/README.md](../evaluation/README.md) — scoring a run
