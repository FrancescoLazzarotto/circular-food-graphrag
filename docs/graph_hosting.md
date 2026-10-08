# Hosting the knowledge graph locally

The production graph runs on this host as a Neo4j Community 5.26 (LTS) instance
managed by systemd, reachable only on loopback, with nightly and weekly backups
and a health check on timers. Everything is in [`deploy/graph/`](../deploy/graph/).

| piece | what it does |
|---|---|
| `install.sh` | Unpacks Neo4j, adds APOC, writes the managed block of `neo4j.conf`, sets the password, installs and starts the systemd user units. Safe to run again: it re-applies the configuration and the units |
| `neo4j.conf.fragment` | Loopback-only listeners, fixed heap, page cache, transaction-log retention, APOC allowed, `ExitOnOutOfMemoryError` |
| `systemd/graph-db.service` | The database, restarted on failure |
| `graph-db-backup.timer` | Every night at 02:30: online JSON export (`scripts/kg/kg_backup.py`), checked (counts, no repeated node or edge, every edge's ends exported); keeps 7 |
| `graph-db-dump.timer` | Every Sunday at 03:00: `neo4j-admin database dump`, with the database stopped for about a minute; skipped, and reported by the health check a week later, if a transaction is open; keeps 4 |
| `graph-db-health.timer` | Every 15 minutes (at :07, :22, :37, :52, away from the dump): `scripts/kg/graph_health.py` |
| `load.sh` | Replaces the graph with another one, from a dump or from a stopped instance, after dumping the current one as a rollback point |

Backups, loads and installs take one lock (`<backups>/.lock`): a timer job
that finds it taken skips its run and logs it, a load or an install refuses to
start. Every backup is written to `<stamp>.partial` and renamed once checked,
and pruning counts finished backups only.

## Settings and secrets

`install.sh` writes `~/.config/graphrag/graph-db.env` (mode 600): paths, ports,
memory, and the generated password, with every `NEO4J_*` spelling the
repository's scripts read. Nothing of it is in the repository. Every script in
`deploy/graph/` reads it; pass the file to anything else that needs the graph,
for instance `--neo4j-env` of `scripts/kg/graph_lot.py`.

The service survives logout and reboot because the account has
`loginctl enable-linger`.

## First install

```bash
bash deploy/graph/install.sh --home /mnt/storage/<user>/neo4j_prod/neo4j-community-5.26.0 \
    --backups /mnt/storage/<user>/graph_backups \
    --tarball neo4j-community-5.26.0-unix.tar.gz --apoc apoc-5.26.0-core.jar \
    --java-home <JDK 17+> --python <python of the graphllm env>
```

Defaults: bolt 7687, http 7474, heap 4g, page cache 4g (`--bolt-port`,
`--http-port`, `--heap`, `--pagecache`). Ports 7689 and 7690 are refused: they
belong to the staging and lots instances.

## Day to day

```bash
systemctl --user status graph-db                 # running?
systemctl --user list-timers 'graph-db*'         # next backups and checks
journalctl --user -u graph-db -n 100             # database log
bash deploy/graph/health.sh                      # check now; also writes <backups>/health.json
cat <backups>/backup.log                         # what every backup did
```

`graph_health.py` fails when the database does not answer a query, the graph is
empty, an index is not ONLINE, the full-text or vector index is missing, a node
has no vector or a vector points at no node, a disk has less than 10 GB free,
or a nightly (30 h) or weekly (8 days) backup is missing or too old. A check that
cannot finish is recorded as a failure. Failures show in
`systemctl --user --failed` and the journal; nothing sends a notification.

## Putting a graph in production

A graph is built and measured elsewhere (the staging instance, or a lots
instance, see `graph_lot.py` in [COMMANDS.md](../COMMANDS.md)), then loaded:

```bash
bash deploy/graph/load.sh --from-home <stopped neo4j home>        # dry run: prints the plan
CONFIRM=yes bash deploy/graph/load.sh --from-home <stopped neo4j home>
CONFIRM=yes bash deploy/graph/load.sh --from-dump <backups>/weekly/<stamp>   # back to a weekly dump
```

A neo4j-admin dump copies the store as it is, so record ids, indexes,
constraints and vectors come across unchanged; only the database id differs,
and `scripts/kg/relink_vectors.py` (run by `load.sh`) rewrites the vector
carriers' pointers to it. It refuses carriers that come from more than one
store, and checks each one against the name of its node (`of_name`, written by
`kg_vector_index.py`). Before stopping production, `load.sh` checks the disk
space and dumps the current graph; on any failure it prints the command that
goes back to that dump. A JSON export (`kg_backup.py` / `kg_restore.py`)
changes record ids, so after it the vector index is rebuilt with
`kg_vector_index.py`, which needs the encoder.

## Pointing the demo at it

```bash
GRAPH_ENV_FILE=~/.config/graphrag/graph-db.env bash scripts/serving/start_demo.sh qwen38-27b
```

Without `GRAPH_ENV_FILE` the demo keeps the graph named in `kg_pipeline/.env`.
