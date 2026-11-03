# Repository structure

```text
.
├── src/graphrag/            # main package
│   ├── cli.py               #   CLI + experiment orchestration
│   ├── config.py            #   AgentConfig / KGConfig
│   ├── strategies.py        #   the 8 retrieval presets — single source of truth
│   ├── profiles.py          #   named configurations behind --profile
│   ├── questions.py         #   question-file parsing (txt / json / jsonl / csv)
│   ├── embeddings.py        #   shared multilingual encoder client
│   ├── types.py             #   RAGState and the retrieval contract
│   ├── agent/               #   LangGraph state machine, evidence/citations, memory, cache, compression
│   ├── kg/                  #   Neo4j manager and retriever
│   ├── llm/                 #   backends, prompt library, refusal markers
│   ├── text_rag/            #   BM25 lexical and dense (FAISS) text channels
│   └── experiments/         #   experiment runner, standard-RAG presets, resource monitor
├── kg_pipeline/             # KG construction pipeline
│   ├── config.yaml          #   ontology, chunking profiles, model choices
│   ├── main.py              #   stage orchestration and checkpoint recovery
│   ├── lots.py              #   add or remove a lot of documents in an existing graph
│   ├── stages/              #   ingestion → chunking → ner → llm → resolution → linking → neo4j
│   ├── utils/               #   corpus registry, acronym map, Neo4j env, validation
│   └── tests/
├── evaluation/              # evaluation workspace
│   ├── evalkit/             #   metrics, LLM judge, reports (CLI: python -m evalkit.cli)
│   ├── gold/                #   gold QA datasets, templates, schema
│   │   ├── gold_v3.json     #     frozen reference set, English
│   │   └── gold_v3_it.json  #     frozen reference set, Italian
│   ├── scripts/             #   score_gold_run.py, build_results_tables.py, hard_subset.py, figures
│   ├── fixtures/            #   question sets for matrix runs
│   ├── baselines/           #   regression baselines
│   └── tests/
├── product/                 # the two demo surfaces over the engine, plus the corpus registry and catalog
├── deploy/graph/            # local production graph: install, systemd units, backup, health, load
├── scripts/                 # operational entrypoints, grouped by job — see scripts/README.md
│   ├── kg/ · corpus/ · gold/ · domain_gate/ · runners/
│   └── smoke/ · analysis/ · serving/ · cluster/
├── tests/                   # core unit tests, incl. test_audit_fixes.py
├── docs/                    # configuration, CLI, experiments, troubleshooting, cluster, graph hosting
├── AGENTS.md                # repository guide for coding agents
├── COMMANDS.md              # task recipes
├── CITATION.cff
├── pyproject.toml
├── requirements.txt         # + requirements-cpu.txt / requirements-gpu.txt
├── requirements.lock        # exact versions the results were produced with
└── .env.example             # configuration template
```

Not tracked by git, created at runtime: `documents/` (source corpus), `artifacts/`
(experiment and evaluation outputs), `logs/`, `kg_pipeline/artifacts/`. Campaign
outputs (`exp_results*/`) default to a path outside the clone, and are ignored
where a run writes them inside it. The internal working documents — audits,
plans, worklogs and probes under `docs/` — are ignored too, and stay local.
