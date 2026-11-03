# Retrieval

How the `KGRetriever` turns a question into evidence, and the eight strategies
that switch its channels on and off. The overall architecture is shown in the
[README](../README.md).

## Indexes

Retrieval reads two indexes: a lexical full-text index, which carries in-language
lookup, and a vector index, the only channel that crosses the Italian/English
gap. **Both are built by scripts, not by ingestion**; see
[Knowledge Graph pipeline](kg_pipeline.md#post-processing-and-indexes).

## Per-query flow

For each query the `KGRetriever`:

1. extracts entity candidates — quoted spans, capitalised phrases, numeric terms — and content keywords, optionally weighted by node-name document frequency (`lexical_specificity`);
2. runs one **lexical** full-text query, a Lucene OR-query with per-term boosts, for nodes and for triples;
3. optionally runs a **vector** query against the `:NodeVec` carriers;
4. picks anchors, by default only names retrieval actually returned (`verify_anchor_exists`), which avoids a full graph scan on an anchor that matches nothing;
5. expands neighbours, the subgraph and the shortest path;
6. drops uninformative predicates and ranks triples — lexical overlap · mention count · confidence, penalised for system links;
7. optionally retrieves raw text (BM25 or dense FAISS), capped per document and re-ranked for definitional questions.

## Strategies

| Strategy | Evidence used |
|---|---|
| `default` | All KG channels: nodes, triples, neighbourhoods, subgraph, shortest paths |
| `hybrid` | All KG channels **plus** raw-text retrieval |
| `text_only` | Text retrieval only — the sparse-retrieval baseline |
| `no_retrieval` | No retrieval channel — the LLM-only baseline |
| `text_plus_triples` | Entity nodes and triples only — no graph traversal |
| `neighbors_focus` | Triples plus local entity neighbourhoods |
| `subgraph_2hop` | Triples plus subgraph expansion, starting floor raised to 2 hops |
| `shortest_path` | Triples plus shortest paths between entities |

Defined once in [`src/graphrag/strategies.py`](../src/graphrag/strategies.py) and
imported by both the CLI and the matrix runner. Presets toggle only the channel
flags; cardinality limits and ranking options come from the base `AgentConfig`,
and the fully resolved per-strategy config is serialised into every run's
`config.json`.

> The subgraph channel is **adaptive**: it starts at `hops` (base default `1`) and
> widens one hop at a time until it has `min_subgraph_triples` or reaches
> `max_hops` (`4`). `subgraph_2hop` raises the starting floor to 2 and drops the
> node, neighbour and shortest-path channels — it is not "the 2-hop arm" against
> 1-hop siblings. Full detail in [docs/cli.md](cli.md#retrieval-strategies).
