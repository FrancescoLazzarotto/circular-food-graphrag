#!/usr/bin/env python3
"""Is the local production graph up, complete and backed up?

Checks, against the graph named by the NEO4J_* environment:

- the database answers on bolt (waiting up to ``--wait`` seconds);
- it holds nodes and edges, unless ``--empty-ok``;
- every index is ONLINE, and the full-text and vector indexes retrieval needs exist;
- every named node has a vector carrier and no carrier points at a missing node;
- the data and backup disks have room;
- the last nightly and weekly backups exist and are recent, unless ``--empty-ok``.

Writes the result to ``<backups>/health.json`` and exits 1 when anything fails,
so a systemd timer shows the failure in ``systemctl --user list-timers`` and
the journal.

    python scripts/kg/graph_health.py --backups <dir> --data-dir <neo4j home>/data
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

ROOT = Path(__file__).resolve().parents[2]
# The kg_pipeline package lives at the repo root and is not pip-installed.
sys.path.insert(0, str(ROOT))

from kg_pipeline.utils import neo4j_env  # noqa: E402

REQUIRED_INDEXES = {"node_search": "FULLTEXT", "node_embedding": "VECTOR"}
MIN_FREE_BYTES = 10 * 1024**3
MAX_AGE_HOURS = {"daily": 30, "weekly": 8 * 24}


def _connect(target: neo4j_env.Neo4jTarget, wait: float):
    """A driver whose database answers a query, retrying for ``wait`` seconds.

    The server accepts connections before the database has opened; a query is
    what tells the two apart.
    """
    deadline = time.monotonic() + wait
    while True:
        driver = GraphDatabase.driver(target.uri, auth=target.auth)
        try:
            with driver.session(**target.session_kwargs()) as session:
                session.run("RETURN 1").consume()
            return driver
        except (ServiceUnavailable, OSError, Neo4jError) as exc:
            driver.close()
            if isinstance(exc, AuthError) or time.monotonic() >= deadline:
                raise
            time.sleep(5)


def _backup_age(backups: Path, mode: str) -> tuple[float | None, str]:
    """Hours since the last backup of ``mode`` and where it is."""
    record = backups / f"last_{mode}.json"
    if not record.exists():
        return None, ""
    data = json.loads(record.read_text(encoding="utf-8"))
    at = datetime.fromisoformat(data["at"].replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - at).total_seconds() / 3600, data["path"]


def check(args: argparse.Namespace) -> dict:
    """Run every check; return the report with its list of problems."""
    target = neo4j_env.resolve_target()
    problems: list[str] = []
    report: dict = {"at": datetime.now(timezone.utc).isoformat(), "graph": target.uri}
    try:
        driver = _connect(target, args.wait)
    except (ServiceUnavailable, OSError, Neo4jError) as exc:
        report["problems"] = [f"il database non risponde su {target.uri}: {exc}"]
        return report
    with driver, driver.session(**target.session_kwargs()) as session:
        row = session.run(
            "CALL () { MATCH (n) WHERE NOT n:NodeVec RETURN count(n) AS nodes } "
            "CALL () { MATCH ()-[r]->() RETURN count(r) AS edges } RETURN nodes, edges"
        ).single()
        report["nodes"], report["edges"] = row["nodes"], row["edges"]
        if not report["nodes"] and not args.empty_ok:
            problems.append("il grafo è vuoto")
        indexes = session.run("SHOW INDEXES YIELD name, type, state").data()
        report["indexes"] = {i["name"]: i["state"] for i in indexes}
        problems += [f"indice {i['name']} {i['state']}" for i in indexes if i["state"] != "ONLINE"]
        if report["nodes"]:
            have = {i["name"]: i["type"] for i in indexes}
            problems += [
                f"manca l'indice {name} ({kind})" for name, kind in REQUIRED_INDEXES.items() if have.get(name) != kind
            ]
            unvectored = session.run(
                "MATCH (n) WHERE n.name IS NOT NULL AND NOT n:NodeVec "
                "AND NOT EXISTS { MATCH (v:NodeVec {of: elementId(n)}) } RETURN count(n) AS c"
            ).single()["c"]
            dangling = session.run(
                "MATCH (v:NodeVec) WHERE NOT EXISTS { MATCH (n) WHERE elementId(n) = v.of } RETURN count(v) AS c"
            ).single()["c"]
            report["nodes_without_vector"], report["dangling_vectors"] = unvectored, dangling
            if unvectored:
                problems.append(f"{unvectored} nodi senza vettore")
            if dangling:
                problems.append(f"{dangling} vettori che puntano a nodi inesistenti")

    for label, path in (("dati", args.data_dir), ("backup", args.backups)):
        free = shutil.disk_usage(path).free
        report[f"free_gb_{label}"] = round(free / 1024**3, 1)
        if free < MIN_FREE_BYTES:
            problems.append(f"poco spazio sul disco dei {label}: {free / 1024**3:.1f} GB")
    for mode, limit in MAX_AGE_HOURS.items():
        age, path = _backup_age(args.backups, mode)
        report[f"last_{mode}"] = {"hours": None if age is None else round(age, 1), "path": path}
        if age is None and not args.empty_ok:
            problems.append(f"nessun backup {mode} registrato")
        elif age is not None and age > limit:
            problems.append(f"ultimo backup {mode} di {age:.0f} ore fa")
    report["problems"] = problems
    return report


def active_transactions() -> int:
    """Transactions open on the graph besides the one asking."""
    target = neo4j_env.resolve_target()
    with neo4j_env.connect(target) as driver, driver.session(**target.session_kwargs()) as session:
        rows = session.run("SHOW TRANSACTIONS YIELD transactionId, currentQuery").data()
    return sum(1 for r in rows if "SHOW TRANSACTIONS" not in (r["currentQuery"] or ""))


def main() -> int:
    """Check, record and report; return 1 when a check fails."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backups", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--active-transactions", action="store_true",
                        help="only print how many transactions are open, for the weekly dump")
    parser.add_argument("--wait", type=float, default=0, help="seconds to wait for the database to answer")
    parser.add_argument("--empty-ok", action="store_true", help="an empty graph is not a failure (first install)")
    args = parser.parse_args()
    if args.active_transactions:
        print(active_transactions())
        return 0
    if args.backups is None or args.data_dir is None:
        parser.error("--backups and --data-dir are required")
    try:
        report = check(args)
    except (Neo4jError, OSError, ValueError, KeyError) as exc:
        # A check that cannot finish is a failure on record, not a stale success.
        report = {"at": datetime.now(timezone.utc).isoformat(), "problems": [f"controllo interrotto: {exc!r}"]}
    args.backups.mkdir(parents=True, exist_ok=True)
    (args.backups / "health.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
