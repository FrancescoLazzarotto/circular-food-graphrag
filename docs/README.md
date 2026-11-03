# Documentation

## Getting started

| Page | Covers |
|---|---|
| [Installation](installation.md) | Prerequisites, extras, pinned environments |
| [Knowledge Graph pipeline](kg_pipeline.md) | Stages, corpus registry, indexes, adding documents to a graph |
| [Retrieval](retrieval.md) | Indexes, the per-query flow, the eight strategies |

## Reference

| Page | Covers |
|---|---|
| [CLI](cli.md) | Every `graphrag.cli` option and its default |
| [Configuration](configuration.md) | Every environment variable and how `.env` is loaded |
| [Repository structure](repository_structure.md) | Where each part of the code lives |

## Operations

| Page | Covers |
|---|---|
| [Experiments](experiments.md) | Reference sets, runners, campaigns, output layout, analysis |
| [Graph hosting](graph_hosting.md) | The local production graph: service, backups, health checks |
| [Cluster](cluster.md) | SLURM templates and node-specific installs |
| [Troubleshooting](troubleshooting.md) | Symptoms, with the cause that produced them |

## Quality

| Page | Covers |
|---|---|
| [Testing](testing.md) | The suite and the three CI jobs |
| [Known limitations](limitations.md) | What is accepted, with the measurement behind it |
| [Reproducibility](reproducibility.md) | What changed between runs, and what that does to comparisons |

Elsewhere in the repository: [evaluation](../evaluation/README.md) for gold sets
and scoring, [COMMANDS.md](../COMMANDS.md) for task recipes,
[scripts](../scripts/README.md) for the script groups.
