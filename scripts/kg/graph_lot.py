#!/usr/bin/env python3
"""Add a lot of documents to an existing knowledge graph, or take one out, step by step.

    python scripts/kg/graph_lot.py prepare --lot L --docs id1,id2 --corpus-dir "<corpus>" --graph bolt://localhost:7690
    python scripts/kg/graph_lot.py extract --lot L                       # stages 0-3, resumable
    python scripts/kg/graph_lot.py resolve --lot L --neo4j-env <file>    # stage 4-5 + match against the graph
    python scripts/kg/graph_lot.py write   --lot L --neo4j-env <file>    # backup, then write
    python scripts/kg/graph_lot.py curate  --lot L --neo4j-env <file>    # edge rules, unions (stops for reading)
    python scripts/kg/graph_lot.py index   --lot L --neo4j-env <file>    # search text and the missing vectors
    python scripts/kg/graph_lot.py check   --lot L --neo4j-env <file>
    python scripts/kg/graph_lot.py remove  --lot L --neo4j-env <file>
    python scripts/kg/graph_lot.py status

``--neo4j-env`` names a file with the ``NEO4J_*`` settings of the graph the lot
was prepared for; a remote graph (the hosted one) and the staging graph on
7689 are refused unless asked for explicitly. Every step reads and writes the
lot folder ``kg_pipeline/artifacts/graph_lots/<lot>/``; see
``kg_pipeline/lots.py`` for what each one does.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
CURATION = ROOT / "scripts" / "kg" / "curation"
# The kg_pipeline package lives at the repo root and is not pip-installed.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CURATION))

from kg_pipeline import lots  # noqa: E402
from kg_pipeline.utils import neo4j_env  # noqa: E402

UNION_PROPOSALS = "unioni_proposte.json"
UNION_EXCLUDED = "unioni_escluse.json"
# Lot entities whose match with a graph node was found wrong on reading
# ({"escluse": [<lot entity name>, ...]}): they are written as new nodes.
MATCHES_EXCLUDED = "corrispondenze_escluse.json"


def _files(args: argparse.Namespace) -> lots.LotFiles:
    files = lots.LotFiles(args.lots_dir / args.lot)
    if not files.meta.exists():
        raise SystemExit(f"il lotto {args.lot} non esiste in {args.lots_dir}")
    return files


def _meta(files: lots.LotFiles) -> dict:
    return json.loads(files.meta.read_text(encoding="utf-8"))


def _llm(files: lots.LotFiles) -> tuple[str, str]:
    env = dotenv_values(files.extract_env)
    return env["VLLM_BASE_URL"], env["VLLM_MODEL_NAME"]


def _set_status(args: argparse.Namespace, status: str, **extra: str) -> None:
    ledger = lots.Ledger.load(args.lots_dir)
    ledger.lots[args.lot].update(status=status, **extra)
    ledger.save()


def _target(args: argparse.Namespace, files: lots.LotFiles) -> tuple[neo4j_env.Neo4jTarget, dict[str, str]]:
    """The graph to work on, from ``--neo4j-env``, checked against the lot and the guards."""
    values = {k: v for k, v in dotenv_values(args.neo4j_env).items() if v is not None}
    target = neo4j_env.resolve_target(
        uri=values.get("NEO4J_URI") or values.get("NEO4J_URL"),
        user=values.get("NEO4J_USER") or values.get("NEO4J_USERNAME"),
        password=values.get("NEO4J_PASSWORD"),
        database=values.get("NEO4J_DATABASE") or values.get("NEO4J_DB") or "neo4j",
    )
    if lots.graph_key(target.uri, target.database) != _meta(files)["grafo"]:
        raise SystemExit(f"il lotto è per {_meta(files)['grafo']}, --neo4j-env punta a {target.uri}")
    if not target.is_local and not args.allow_remote:
        raise SystemExit(f"{target.uri} non è locale (il grafo ospitato?): serve --allow-remote")
    if target.uri.rstrip("/").endswith(":7689") and not args.allow_staging:
        raise SystemExit("7689 è lo staging di riferimento: serve --allow-staging")
    # Every NEO4J_* spelling, so a script that reads another one cannot fall
    # back to a default that names a different graph.
    env = dict(
        os.environ,
        PYTHONNOUSERSITE="1",
        NEO4J_URI=target.uri,
        NEO4J_URL=target.uri,
        NEO4J_USER=target.user,
        NEO4J_USERNAME=target.user,
        NEO4J_PASSWORD=target.password,
        NEO4J_DATABASE=target.database or "neo4j",
        NEO4J_DB=target.database or "neo4j",
    )
    return target, env


def _graph_flags(target: neo4j_env.Neo4jTarget) -> list[str]:
    return ["--uri", target.uri, "--user", target.user, "--password", target.password,
            "--database", target.database or "neo4j"]


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    status = subprocess.run(command, cwd=ROOT, env=env or dict(os.environ, PYTHONNOUSERSITE="1"), check=False).returncode
    if status:
        raise SystemExit(f"passo fallito ({status}): {' '.join(command[:4])} …")


def _pipeline(files: lots.LotFiles, stage: str) -> None:
    command, env = lots.extract_command(files, sys.executable)
    command[command.index("--stage") + 1] = stage
    _run(command, env)


def _prepare(args: argparse.Namespace) -> int:
    """Create the lot folder and its ledger entry."""
    files = lots.prepare(
        name=args.lot,
        doc_ids=[d.strip() for d in args.docs.split(",") if d.strip()],
        registry=args.registry,
        corpus_dir=args.corpus_dir,
        graph=lots.graph_key(args.graph, args.database),
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        base_config=args.base_config,
        lots_dir=args.lots_dir,
    )
    print(f"lotto preparato in {files.dir}")
    return 0


def _extract(args: argparse.Namespace) -> int:
    """Stages 0-3 on the lot's documents; the same command resumes an interrupted run."""
    files = _files(args)
    _pipeline(files, "llm")
    _set_status(args, "estratto")
    return 0


# Statuses in which the lot is (partly) in the graph.
IN_GRAPH = {"scritto", "in curatela", "da leggere", "curato"}
REJUDGED = "rigiudizio_fatto"


def _later_lots(args: argparse.Namespace) -> list[str]:
    """Lots written to the same graph after this one and still in it."""
    ledger = lots.Ledger.load(args.lots_dir)
    mine = ledger.lots[args.lot]
    return [
        name for name, rec in ledger.lots.items()
        if name != args.lot and rec.get("graph") == mine.get("graph")
        and rec.get("status") in IN_GRAPH and rec.get("written", "") > mine.get("written", "")
    ]


def _resolve(args: argparse.Namespace) -> int:
    """Stages 4-5 within the lot, then match the lot's entities against the graph.

    Stage 4 runs as in a full build, its merges are re-judged by the strict
    judge, and stages 4-5 run again from the approvals that survive. The
    re-judgement always runs until it has finished once: it replays its stored
    verdicts, so a run interrupted halfway never goes on with merges nobody
    confirmed.
    """
    from merge_judge import judge_in_slices

    from kg_pipeline.stages import resolution

    files = _files(args)
    status = lots.Ledger.load(args.lots_dir).lots[args.lot]["status"]
    if status in IN_GRAPH:
        raise SystemExit(f"il lotto è {status}, quindi nel grafo: toglierlo con remove prima di rifare resolve")
    run = files.dir
    target, _ = _target(args, files)
    base_url, model = _llm(files)
    config = yaml.safe_load(files.config.read_text(encoding="utf-8"))

    if not (run / REJUDGED).exists():
        if not ((run / "stage4_merge_approved.json").exists() or (run / "stage4_merge_approved_unfiltered.json").exists()):
            _pipeline(files, "resolution")
        _run([sys.executable, str(CURATION / "rejudge_merges.py"), "--run-dir", str(run),
              "--endpoints", base_url, "--model", model,
              "--context-jaccard-floor", str(config["resolution"]["context_jaccard_floor"])])
        for name in ("stage4_triples_resolved.json", "stage4_registry.json", "stage5_triples_linked.json"):
            (run / name).unlink(missing_ok=True)
        stamps = run / "stage_fingerprints.json"
        recorded = json.loads(stamps.read_text(encoding="utf-8"))
        for stage in ("triples_resolved", "triples_linked"):
            recorded.pop(stage, None)
        stamps.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
        (run / REJUDGED).write_text(lots._now() + "\n", encoding="utf-8")
    if not (run / "stage5_triples_linked.json").exists():
        _pipeline(files, "linking")

    registry = resolution.load_registry(run / "stage4_registry.json")
    acronyms = json.loads((run / "stage3_acronyms.json").read_text(encoding="utf-8"))
    with neo4j_env.connect(target) as driver, driver.session(**target.session_kwargs()) as session:
        nodes = lots.read_graph_nodes(session)
    exact = lots.exact_matches(registry, nodes, acronyms)
    rest = {name: record for name, record in registry.items() if name not in exact}
    vectors = lots.NameVectors(args.lots_dir / "vettori_nomi", config["resolution"]["embedding_model"])
    candidates = lots.similar_candidates(rest, nodes, vectors, float(config["resolution"]["similarity_threshold"]))

    def judge(pairs: list[tuple[str, str]]) -> list[bool | None]:
        return judge_in_slices(pairs, [base_url], model) if pairs else []

    judged = lots.judged_matches(candidates, registry, judge, run / "giudizi_grafo.json")
    matches = {**judged, **exact}
    out = {
        name: {
            "id": m.node.element_id, "label": m.node.label, "name": m.node.name,
            "aliases": list(m.node.aliases), "degree": m.node.degree,
            "come": m.how, "similarita": round(m.similarity, 4),
        }
        for name, m in sorted(matches.items())
    }
    (run / "corrispondenze.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"entità del lotto {len(registry)}; uguali a un nodo del grafo {len(exact)}; "
        f"candidate per somiglianza {len(candidates)}, confermate dal giudice {len(judged)}; "
        f"nuove {len(registry) - len(matches)}"
    )
    _set_status(args, "risolto")
    return 0


def _matches(files: lots.LotFiles) -> dict[str, lots.Match]:
    """The matches of ``resolve``, without those excluded on reading.

    Only a match the judge made can be excluded: an entity with the same name
    and label as a node would be merged into it again by the write.
    """
    raw = json.loads((files.dir / "corrispondenze.json").read_text(encoding="utf-8"))
    excluded_file = files.dir / MATCHES_EXCLUDED
    if excluded_file.exists():
        excluded = set(json.loads(excluded_file.read_text(encoding="utf-8"))["escluse"])
        unknown = excluded - set(raw)
        if unknown:
            raise SystemExit(f"{MATCHES_EXCLUDED} nomina entità senza corrispondenza: {sorted(unknown)}")
        by_name = sorted(name for name in excluded if raw[name]["come"] != "giudice")
        if by_name:
            raise SystemExit(f"{MATCHES_EXCLUDED}: si escludono solo le unioni del giudice, non {by_name}")
        raw = {name: m for name, m in raw.items() if name not in excluded}
    return {
        name: lots.Match(
            lots.GraphNode(m["id"], m["label"], m["name"], tuple(m["aliases"]), m["degree"]),
            m["come"],
            m["similarita"],
        )
        for name, m in raw.items()
    }


def _write(args: argparse.Namespace) -> int:
    """Back up the graph, check the lot may still enter it, and write it."""
    from kg_pipeline.stages import linking, resolution

    files = _files(args)
    target, env = _target(args, files)
    ledger = lots.Ledger.load(args.lots_dir)
    mine = ledger.lots[args.lot]
    if mine["status"] not in {"risolto", "scritto", "rimosso"}:
        raise SystemExit(f"il lotto è {mine['status']}: si scrive dopo resolve (o per riprendere una scrittura interrotta)")
    clash = {d: lot for d, lot in ledger.documents_in(mine["graph"], exclude=args.lot).items() if d in mine["documents"]}
    if clash:
        raise SystemExit(f"documenti già in altri lotti per questo grafo: {clash}")
    backup = files.dir / "backup_prima"
    if not (backup / "manifest.json").exists():
        _run([sys.executable, str(ROOT / "scripts" / "kg" / "kg_backup.py"), "--output-dir", str(backup)], env)
    triples = linking.load_triples(files.dir / "stage5_triples_linked.json")
    registry = resolution.load_registry(files.dir / "stage4_registry.json")
    matches = _matches(files)
    plan = lots.plan_writes(triples, registry, matches)
    expected = {m.node.element_id: (m.node.label, m.node.name) for m in matches.values()}
    with neo4j_env.connect(target) as driver, driver.session(**target.session_kwargs()) as session:
        try:
            record = lots.write_lot(session, args.lot, plan, files.dir / "scrittura.json", expected)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    print(json.dumps({k: v for k, v in record.items() if k != "alias_prima"}, ensure_ascii=False, indent=1))
    _set_status(args, "scritto", written=record["scritto"])
    return 0


def _curate(args: argparse.Namespace) -> int:
    """Edge rules and node removals within the lot, then the unions, which stop once for reading."""
    from merge_judge import judge_in_slices

    from graphrag.embeddings import QUERY_PREFIX, encode

    files = _files(args)
    status = lots.Ledger.load(args.lots_dir).lots[args.lot]["status"]
    if status not in IN_GRAPH:
        raise SystemExit(f"il lotto è {status}: si cura dopo write")
    later = _later_lots(args)
    if later:
        raise SystemExit(f"lotti scritti dopo questo sono nel grafo: {later}; le unioni potrebbero toccarli")
    target, env = _target(args, files)
    flags = _graph_flags(target)
    base_url, model = _llm(files)
    # From here a new write would bring back what the rules delete.
    _set_status(args, "in curatela")
    if not (files.dir / "regole_archi_fatte").exists():
        _run([sys.executable, str(CURATION / "edge_rules.py"), "--chunks-dir", str(files.dir), *flags,
              "--lot", args.lot, "--log", str(files.dir / "edge_rules_log.jsonl"), "--apply"], env)
        _run([sys.executable, str(CURATION / "drop_nodes.py"), "--anaphoric", *flags, "--lot", args.lot,
              "--log", str(files.dir / "anaphoric_deleted.jsonl"), "--apply"], env)
        (files.dir / "regole_archi_fatte").write_text("", encoding="utf-8")

    proposals = files.dir / UNION_PROPOSALS
    with neo4j_env.connect(target) as driver, driver.session(**target.session_kwargs()) as session:
        if not proposals.exists():
            unions = lots.propose_unions(
                session, args.lot,
                encode=lambda names: encode(names, QUERY_PREFIX),
                judge=lambda pairs: judge_in_slices(pairs, [base_url], model) if pairs else [],
                verdicts_path=files.dir / "giudizi_unioni.json",
            )
            proposals.write_text(json.dumps(unions, ensure_ascii=False, indent=1), encoding="utf-8")
        excluded_file = files.dir / UNION_EXCLUDED
        if not excluded_file.exists():
            print(f"Unioni proposte in {proposals}: leggerle e scrivere quelle sbagliate in {excluded_file} "
                  '({"escluse": [{"unito": ..., "centro": ...}]}, anche vuoto), poi rilanciare curate.')
            _set_status(args, "da leggere")
            return 2
        excluded = {(x["unito"], x["centro"]) for x in json.loads(excluded_file.read_text(encoding="utf-8"))["escluse"]}
        done = lots.apply_unions(session, json.loads(proposals.read_text(encoding="utf-8")), excluded,
                                 files.dir / "scrittura.json")
    _run([sys.executable, str(CURATION / "drop_nodes.py"), "--isolated", *flags, "--lot", args.lot, "--apply"], env)
    print(f"unioni applicate: {done}")
    _set_status(args, "curato")
    return 0


def _graph_config(files: lots.LotFiles, target: neo4j_env.Neo4jTarget) -> Path:
    """A configuration naming the lot's graph, for the scripts that read the database from one."""
    config = yaml.safe_load(files.config.read_text(encoding="utf-8"))
    config["neo4j"]["database"] = target.database or "neo4j"
    path = files.dir / "grafo.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _neo4j_env_file(files: lots.LotFiles, env: dict[str, str]) -> Path:
    """An env file naming the lot's graph, for the scripts that read one; readable by the owner only."""
    path = files.dir / "grafo.env"
    keys = ("NEO4J_URI", "NEO4J_URL", "NEO4J_USER", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE", "NEO4J_DB")
    path.write_text("".join(f"{k}={env[k]}\n" for k in keys), encoding="utf-8")
    path.chmod(0o600)
    return path


def _refresh_indexes(files: lots.LotFiles, target: neo4j_env.Neo4jTarget, env: dict[str, str]) -> None:
    """Search text of the changed nodes, vectors of the nodes without one, and the vector check."""
    _run([sys.executable, str(ROOT / "scripts" / "kg" / "kg_search_index.py"),
          "--config", str(_graph_config(files, target)), "--env-file", str(_neo4j_env_file(files, env))], env)
    _run([sys.executable, str(ROOT / "scripts" / "kg" / "kg_vector_index.py"), "--only-missing"], env)
    _run([sys.executable, str(ROOT / "scripts" / "kg" / "check_vector_index.py"), "--min-resolving", "1"], env)


def _index(args: argparse.Namespace) -> int:
    """Refresh the indexes after a write, a curation or a removal."""
    files = _files(args)
    target, env = _target(args, files)
    _refresh_indexes(files, target, env)
    return 0


def _check(args: argparse.Namespace) -> int:
    """The graph's own nodes and edges are as many as before the lot, and every node has a vector."""
    files = _files(args)
    target, _ = _target(args, files)
    record = json.loads((files.dir / "scrittura.json").read_text(encoding="utf-8"))
    with neo4j_env.connect(target) as driver, driver.session(**target.session_kwargs()) as session:
        now = lots.graph_counts(session)
        names = session.run(
            "UNWIND keys($before) AS id OPTIONAL MATCH (n) WHERE elementId(n) = id "
            "RETURN id, n.name AS name", before=record["alias_prima"]).data()
        lot_counts = session.run(
            "CALL () { MATCH (n) WHERE n.lotto = $lot RETURN count(n) AS nodes } "
            "CALL () { MATCH ()-[r]->() WHERE r.lotto = $lot RETURN count(r) AS edges } RETURN nodes, edges",
            lot=args.lot).single()
        unvectored = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL AND NOT EXISTS { MATCH (v:NodeVec {of: elementId(n)}) } "
            "RETURN count(n) AS c").single()["c"]
    before = record["prima"]
    problems = []
    if now["nodes"] - now["lot_nodes"] != before["nodes"] - before["lot_nodes"]:
        problems.append("i nodi che non sono di un lotto sono cambiati di numero")
    if now["edges"] - now["lot_edges"] != before["edges"] - before["lot_edges"]:
        problems.append("gli archi che non sono di un lotto sono cambiati di numero")
    renamed = [r for r in names if r["name"] != record["alias_prima"][r["id"]]["name"]]
    if renamed:
        problems.append(f"{len(renamed)} nodi del grafo hanno cambiato nome o non ci sono più")
    if record.get("archi_scritti", 0) != record.get("archi_del_piano", record.get("archi_scritti", 0)):
        problems.append("non tutti gli archi del piano sono stati scritti")
    if unvectored:
        problems.append(f"{unvectored} nodi senza vettore: lanciare index")
    report = {"prima": before, "adesso": now, "lotto": dict(lot_counts), "problemi": problems}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    (files.dir / "controllo.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if problems else 0


def _remove(args: argparse.Namespace) -> int:
    """Take the lot out of the graph, archive what it decided, refresh the indexes.

    Lots come out in the reverse order they went in; a run interrupted after
    the removal only refreshes the indexes again.
    """
    files = _files(args)
    target, env = _target(args, files)
    status = lots.Ledger.load(args.lots_dir).lots[args.lot]["status"]
    if status == "rimosso":
        _refresh_indexes(files, target, env)
        print("il lotto era già tolto: indici aggiornati")
        return 0
    if status not in IN_GRAPH:
        raise SystemExit(f"il lotto è {status}: non è nel grafo")
    later = _later_lots(args)
    if later and not args.force:
        raise SystemExit(f"lotti scritti dopo questo e ancora nel grafo: {later}; toglierli prima (o --force)")
    with neo4j_env.connect(target) as driver, driver.session(**target.session_kwargs()) as session:
        report = lots.remove_lot(session, args.lot, files.dir / "scrittura.json")
    # What the write and the curation decided refers to nodes that are gone:
    # a later write starts again from a clean record, and the unions are
    # proposed and read again.
    archive = files.dir / f"rimozione_{report['rimosso'].replace(':', '')}"
    archive.mkdir()
    for name in ("scrittura.json", "regole_archi_fatte", UNION_PROPOSALS, UNION_EXCLUDED,
                 "edge_rules_log.jsonl", "anaphoric_deleted.jsonl", "controllo.json"):
        if (files.dir / name).exists():
            (files.dir / name).rename(archive / name)
    (archive / "rimozione.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    _set_status(args, "rimosso")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    _refresh_indexes(files, target, env)
    return 0


def _status(args: argparse.Namespace) -> int:
    """Print the ledger."""
    print(json.dumps(lots.Ledger.load(args.lots_dir).lots, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """Parse the command line and run one step; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lots-dir", type=Path, default=lots.LOTS_DIR)
    steps = parser.add_subparsers(dest="step", required=True)

    prepare = steps.add_parser("prepare", help="create the lot folder")
    prepare.add_argument("--lot", required=True)
    prepare.add_argument("--docs", required=True, help="comma-separated registry ids")
    prepare.add_argument("--corpus-dir", type=Path, required=True)
    prepare.add_argument("--graph", required=True, help="bolt URI of the graph the lot is for")
    prepare.add_argument("--database", default="neo4j", help="database of that graph")
    prepare.add_argument("--registry", type=Path, default=ROOT / "product" / "corpus_registry.csv")
    prepare.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    prepare.add_argument("--llm-model", default="Qwen/Qwen3-32B-AWQ",
                         help="the model the graph was extracted with")
    prepare.add_argument("--base-config", type=Path, default=ROOT / "kg_pipeline" / "config.yaml")
    prepare.set_defaults(run=_prepare)

    for name, handler, help_text in (
        ("extract", _extract, "stages 0-3 on the lot's documents"),
        ("resolve", _resolve, "stages 4-5 in the lot, then match against the graph"),
        ("write", _write, "back up the graph, then write the lot"),
        ("curate", _curate, "edge rules, anaphoric nodes, unions to read"),
        ("index", _index, "search text and vectors of what changed"),
        ("check", _check, "the graph's own nodes and edges are untouched"),
        ("remove", _remove, "take the lot out of the graph"),
    ):
        step = steps.add_parser(name, help=help_text)
        step.add_argument("--lot", required=True)
        if name != "extract":
            step.add_argument("--neo4j-env", type=Path, required=True)
            step.add_argument("--allow-remote", action="store_true")
            step.add_argument("--allow-staging", action="store_true")
        if name == "remove":
            step.add_argument("--force", action="store_true")
        step.set_defaults(run=handler)

    status = steps.add_parser("status", help="print the ledger")
    status.set_defaults(run=_status)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
