# Known limitations

Documented so results are read correctly. Each was checked against the artifacts
of real runs.

**Accepted, with the measurement that justifies accepting them**

- **Edge provenance is single-valued.** A triple attested in several documents keeps the `source_doc` / `page_range` of the last ingestion. Measured: **106 of 13 058 edges (0.8%)** carry more than one document — a schema change to list-valued provenance was not worth the risk to the citation layer.
- **APOC is a hard dependency**, with no whitelist fallback for node projection.
- **Entity resolution materialises a full similarity matrix** (~576 MB at 12k groups). A scaling limit on larger corpora, not a defect on this one.
- **`PYTHONHASHSEED` cannot be set from inside the pipeline** — CPython reads it at interpreter startup. The pipeline warns instead of claiming to seed it; export it before launching if set-iteration order must be reproducible.

**Gaps in the matrix runner** — outside the thesis path, since every reported number comes from `graphrag.cli --experiment`

- **`run_retrieval_matrix.py` cannot express the newer options.** Vector retrieval, citations, the domain gate, `--complexity`, `--drop-predicates` and the rest are CLI-only. Matrix runs measure stock defaults.
- **Matrix runs carry no `query_id`,** so the evaluator joins them to the gold by question text.

**Interpretation**

- **Two gold files coexist.** `evaluation/gold/gold.json` shares the 30 `query_id`s of `gold_v3.json` but differs in `expected_entities` on 7 of them, and several tools default to it. Pass `--gold evaluation/gold/gold_v3.json` explicitly for anything you intend to report.
- **`stage6_neo4j_summary.json`** reports `triples_sent` — triples that reached the database, an upper bound on edges because MERGE deduplicates. The authoritative edge count is the one read back from Neo4j in the same file. The old `relationships_written` key is kept for existing readers and holds the same value.
- **`--subgraph-limit` truncates in Cypher, before ranking.** On a sparse graph the cap never binds; on a dense one it decides the answer.
- **The demos do not enable the vector channel,** so a demo answer is not a retrieval measurement.
