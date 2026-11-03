# Knowledge Graph pipeline

The pipeline lives in [`kg_pipeline/`](../kg_pipeline/) and writes checkpointed
stage artifacts to a run directory. Defaults come from
[`kg_pipeline/config.yaml`](../kg_pipeline/config.yaml).

```bash
PYTHONUNBUFFERED=1 conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --log-level INFO
```

Stages run sequentially with JSON checkpoint recovery, each reading the artifacts
of the previous one. `--stage <name>` runs everything **up to and including** that
stage, reusing earlier artifacts where they exist; it does not run one stage in
isolation. Reuse the same `--run-dir` to resume.

| `--stage` | Description | Main artifact |
|---|---|---|
| `ingestion` | Load raw documents — PDF → markdown, page chunks, sections | `stage0_documents.json` |
| `chunking` | Token-windowed paragraph chunks, three size profiles by page count | `stage1_chunks.json` |
| `ner` | Named Entity Recognition — GLiNER, multilingual | `stage2_ner.json` |
| `llm` | LLM triple extraction — async, batched, checkpointed | `stage3_triples_raw.json`, `stage3_acronyms.json` |
| `resolution` | Entity resolution — embeddings + predicate Jaccard + LLM confirmation | `stage4_triples_resolved.json`, `stage4_registry.json`, `stage4_merge_approved.json` |
| `linking` | `SAME_AS` alias edges, optional `MENTIONED_IN` | `stage5_triples_linked.json` |
| `neo4j` | Graph ingestion — UNWIND + MERGE per label/predicate signature | `stage6_neo4j_summary.json` |

`--dry-run` skips Neo4j ingestion, `--single-doc <name>` processes one document.
Stage 3 checkpoints every `llm.checkpoint_every` chunks with atomic writes; the
run directory also holds `run_metadata.json` (seed, models, git commit, endpoint),
a snapshot of the config and relation vocabulary, and `failed_chunks.jsonl`.

## Corpus registry

Without further configuration stage 0 reads the PDFs directly inside
`paths.input_dir`, and subfolders are left out on purpose. A corpus organised
in theme folders, with duplicates, drafts and files in other formats next to
the documents, is read through a **registry** instead: one row per file, with a
stable document id, the content hash, the theme folders, page counts, and the
columns curators fill in by hand (duplicate of, kind, level, priority,
excluded, OCR, notes). It is a semicolon-separated CSV that a spreadsheet
opens directly; the format is described in
[`kg_pipeline/utils/corpus_registry.py`](../kg_pipeline/utils/corpus_registry.py).

```yaml
paths:
  input_dir: <corpus folder>
  registry: ./product/corpus_registry.csv
  ocr_dir: ./kg_pipeline/artifacts/corpus_ocr
  stage0_cache: ./kg_pipeline/artifacts/stage0_cache
```

With `paths.registry` set, stage 0 reads every included row wherever it sits
under `input_dir`, under the registry's id, and stops if a file is missing or
changed since the registry was written. A file marked `ocr` is read from its
OCR copy in `paths.ocr_dir`. Files present but not registered are reported and
not read. [`scripts/corpus/build_registry.py`](../scripts/corpus/build_registry.py)
creates the registry and refreshes it when files arrive, keeping every column a
curator edited; [`scripts/corpus/ocr_scanned.py`](../scripts/corpus/ocr_scanned.py)
makes the OCR copies with Tesseract, and
[`scripts/corpus/document_cards.py`](../scripts/corpus/document_cards.py) proposes
each document's title, authors and year for a curator to check. To bring the
demo's documents up to date after files arrive, one command refreshes the
registry, makes the OCR copies, reads the new files and encodes their passages
into the text index, replacing what the demo reads only when every step has
passed; nothing is written to any graph:

```bash
PYTHONNOUSERSITE=1 python scripts/corpus/update_corpus.py --corpus-dir "<corpus folder>"
```

With `paths.stage0_cache` set, the reading of each file is kept there, keyed by
the file's content and by how it is read (reader, its options, library
versions, original or OCR copy). A later run reads only the files no earlier run
has read: adding a document to the corpus costs reading that document. A
renamed or moved file is not read again; a changed one is.

## Post-processing and indexes

After Neo4j ingestion, **in this order**:

```bash
python scripts/kg/kg_postprocess.py --passes 1,2,3,4,5   # repair passes kg_repair.py .. kg_repair5.py
python scripts/kg/kg_search_index.py                     # full-text index — lexical retrieval
python scripts/kg/kg_vector_index.py                     # :NodeVec carriers + vector index — cross-lingual
```

The passes are distinct ordered repair rounds, not versions of one script, and
`--passes` defaults to `1,2,3,4` — pass 5 exists and is opt-in, so name it
explicitly as above. **Retrieval quality depends on the last two commands having
been run against the live graph.** Other graph utilities are listed in
[scripts/README.md](../scripts/README.md).

## Adding documents to an existing graph

A rebuild is not the only way to grow a graph. [`scripts/kg/graph_lot.py`](../scripts/kg/graph_lot.py)
adds a **lot** — a set of registry documents — to a graph that already exists,
without touching what is in it. Each lot has its own folder under
`kg_pipeline/artifacts/graph_lots/<lot>/` and a ledger records which lots went
into which graph, so a document never enters one graph twice.

```bash
python scripts/kg/graph_lot.py prepare --lot L --docs id1,id2 --corpus-dir "<corpus>" --graph bolt://localhost:7690
python scripts/kg/graph_lot.py extract --lot L                      # stages 0-3, resumable
python scripts/kg/graph_lot.py resolve --lot L --neo4j-env <file>   # stage 4-5 + match against the graph
python scripts/kg/graph_lot.py write   --lot L --neo4j-env <file>   # backup, then write
python scripts/kg/graph_lot.py curate  --lot L --neo4j-env <file>   # stops once for reading the unions
python scripts/kg/graph_lot.py index   --lot L --neo4j-env <file>
python scripts/kg/graph_lot.py check   --lot L --neo4j-env <file>
python scripts/kg/graph_lot.py remove  --lot L --neo4j-env <file>   # takes the lot out again
```

Existing nodes only gain aliases; new nodes and edges carry the lot's name, which
is what `remove` uses to take them out again. A remote graph and the staging
instance on 7689 are refused unless asked for explicitly. Logic in
[`kg_pipeline/lots.py`](../kg_pipeline/lots.py).
