# Reproducibility notes

Several behaviours changed on **2026-08-17**. Runs produced before that date are
not directly comparable to runs produced after it.

| Change | Effect on earlier runs |
|---|---|
| The retrieved context no longer echoes `Query: <question>` | `context_text` was never empty while a query existed. The zero-evidence guard was unreachable, the relevance grader matched the question against itself so no rewrite ever fired, and `no_retrieval` received the question as its context under a prompt saying to use ONLY the provided context |
| The lexical text backend ranks with **Okapi BM25** (k1 1.5, b 0.75) | Earlier runs used a formula that was neither tf-idf cosine nor BM25 and was biased by chunk length |
| **MMR applies to both text backends** | `--text-retriever-mmr` was silently inert on the default lexical backend; earlier runs passed the flag and got plain relevance ranking |
| The citation gate validates against `visible_evidence_refs` | Earlier runs let a tag pointing at a compression-dropped block pass |
| The vector channel **raises** instead of degrading | Earlier runs dropped the channel on 3 queries in 3 of 6 compared models and 0 in the other 3 — a model-asymmetric change of retrieval method mid-comparison |
| Salient-term extraction is bilingual and word-boundary matched | An Italian-only stopword list made `the`, `and`, `are` salient on every English question, and substring matching then accepted every triple |
| Entity resolution accumulates aliases | Registry entries used to overwrite each other on a canonical-name collision: 17 333 initial groups → 11 792 registry entries in the July run |
| Merge confirmation votes once per pair | A pair was judged once per document bucket it touched, and a single `merge:true` out of five approved |
| `config.json` records `graph_target` | Earlier runs do not record which Neo4j instance they read |

Measured effect of the fixes — same model (Qwen2.5-7B), same graph, same 30
questions, answer channel:

| Pipeline | Concept F1 before | after | Δ |
|---|---|---|---|
| `hybrid` | 0.620 | **0.667** | +0.047 |
| `text_only` | 0.590 | 0.630 | +0.041 |
| `default` | 0.550 | 0.560 | +0.010 |
| `no_retrieval` | 0.531 | 0.544 | +0.014 |

The baseline moving by only +0.014 is the number that matters most: the
`hybrid` − `no_retrieval` gap is **+0.159 concept recall before and after**, so the
advantage was never an artefact of a damaged control.
