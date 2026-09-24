"""Merge nodes that are the same entity but that resolution left apart.

Typically Italian/English pairs of central concepts ("food waste" and "spreco
alimentare"): the star merge of stage 4 leaves them apart when nobody ever
compared them directly.

``propose`` (on the live graph):

1. candidates: for every node of degree >= --min-degree, its --top-k nearest
   nodes by the e5 encoder (on :8002), above --min-cos. The encoder alone does
   not separate ("food" ~ "food waste" 0.91, higher than "food waste" ~
   "spreco alimentare" 0.88): it only proposes;
2. the strict judge with reasoning decides (`merge_judge`);
3. approved pairs become star unions, no chains: the node with the higher
   degree is the centre. Written to --out as a JSON list.

The proposals are meant to be read before applying: on 24/09 23 of 245 were
wrong ("materiale rinnovabile" into the magazine "Materia Rinnovabile").

``apply``: merges each proposed union into its centre
(``apoc.refactor.mergeNodes``, centre keeps name and label, the other's name
and aliases become aliases), skipping the pairs listed in --exclude
(``{"escluse": [{"unito", "centro", "giro"}...]}``; with --round only that
round's entries count). Applied unions go to --log.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from merge_judge import judge_in_slices  # noqa: E402

from graphrag.embeddings import QUERY_PREFIX, encode  # noqa: E402

SKIP = ["Person", "Document", "DataValue", "NodeVec"]


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s)


def propose(a, s) -> None:
    nodes = s.run(
        "MATCH (n) WHERE NOT any(l IN labels(n) WHERE l IN $skip) AND n.name IS NOT NULL "
        "WITH n, COUNT { (n)--() } AS deg WHERE deg >= $k "
        "RETURN elementId(n) AS id, n.name AS name, deg, labels(n) AS labels, "
        "coalesce(n.aliases, []) AS aliases", skip=SKIP, k=a.min_degree).data()
    names = [x["name"] for x in nodes]
    print(f"nodes of degree >= {a.min_degree}: {len(nodes)}", flush=True)
    vec = np.array(encode(names, QUERY_PREFIX), dtype=np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    sims = vec @ vec.T
    np.fill_diagonal(sims, -1)
    cands = set()
    for i in range(len(nodes)):
        for j in np.argsort(-sims[i])[: a.top_k]:
            if sims[i, j] < a.min_cos:
                break
            if fold(names[i]) == fold(names[int(j)]):
                continue
            cands.add(tuple(sorted((i, int(j)))))
    cands = sorted(cands)
    print(f"candidates: {len(cands)}", flush=True)
    endpoints = [u.strip() for u in a.endpoints.split(",") if u.strip()]
    verdicts = judge_in_slices([(names[i], names[j]) for i, j in cands], endpoints, a.model)
    approved = [(i, j) for (i, j), v in zip(cands, verdicts) if v is True]
    print(f"approved by the judge: {len(approved)}", flush=True)

    order = sorted(range(len(nodes)), key=lambda i: -nodes[i]["deg"])
    nb: dict[int, set[int]] = {}
    for i, j in approved:
        nb.setdefault(i, set()).add(j)
        nb.setdefault(j, set()).add(i)
    assigned: set[int] = set()
    merges = []
    for c in order:
        if c in assigned or c not in nb:
            continue
        assigned.add(c)
        for m in sorted(nb[c], key=lambda i: -nodes[i]["deg"]):
            if m not in assigned:
                assigned.add(m)
                merges.append((c, m))
    out = [{"centro": names[c], "grado_centro": nodes[c]["deg"], "unito": names[m],
            "grado_unito": nodes[m]["deg"]} for c, m in merges]
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"star unions: {len(merges)} (see {a.out})", flush=True)


def apply(a, s) -> None:
    props = json.loads(a.proposals.read_text())
    excluded = set()
    if a.exclude:
        for x in json.loads(a.exclude.read_text())["escluse"]:
            if a.round is None or x.get("giro") == a.round:
                excluded.add((x["unito"], x["centro"]))
    done = missing = 0
    log = a.log.open("w", encoding="utf-8") if a.log else None
    for x in props:
        if (x["unito"], x["centro"]) in excluded:
            continue
        ids = s.run("OPTIONAL MATCH (k {name: $k}) WHERE NOT k:NodeVec OPTIONAL MATCH (o {name: $o}) WHERE NOT o:NodeVec "
                    "RETURN collect(DISTINCT elementId(k)) AS k, collect(DISTINCT elementId(o)) AS o",
                    k=x["centro"], o=x["unito"]).single()
        if len(ids["k"]) != 1 or len(ids["o"]) != 1 or ids["k"][0] == ids["o"][0]:
            missing += 1
            continue
        s.run(
            "MATCH (k) WHERE elementId(k) = $k MATCH (o) WHERE elementId(o) = $o "
            "WITH k, o, [x IN coalesce(k.aliases, []) + [o.name] + coalesce(o.aliases, []) WHERE x <> k.name] AS al, "
            "     labels(k) AS keep "
            "SET k.aliases = apoc.coll.toSet(al) "
            "WITH k, o, keep "
            "CALL apoc.refactor.mergeNodes([k, o], {properties: 'discard', mergeRels: true}) YIELD node "
            "WITH node, [l IN labels(node) WHERE NOT l IN keep] AS extra "
            "CALL apoc.create.removeLabels(node, extra) YIELD node AS n2 RETURN count(*)",
            k=ids["k"][0], o=ids["o"][0])
        if log:
            log.write(json.dumps(x, ensure_ascii=False) + "\n")
        done += 1
    loops = s.run("MATCH (n)-[r]->(n) DELETE r RETURN count(r) AS c").single()["c"]
    if log:
        log.close()
    print(f"unions applied {done}, skipped (node missing or ambiguous) {missing}, self-loops removed {loops}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("propose", "apply"):
        q = sub.add_parser(name)
        q.add_argument("--uri", default="bolt://localhost:7689")
        q.add_argument("--user", default="neo4j")
        q.add_argument("--password", default="staging-kg-v2")
        q.add_argument("--database", default="neo4j")
    pr = sub.choices["propose"]
    pr.add_argument("--min-degree", type=int, default=4)
    pr.add_argument("--top-k", type=int, default=5)
    pr.add_argument("--min-cos", type=float, default=0.86)
    pr.add_argument("--endpoints", default="http://localhost:8000/v1,http://localhost:8003/v1")
    pr.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    pr.add_argument("--out", type=Path, required=True)
    ap = sub.choices["apply"]
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--round", type=int, help="only the exclusions of this round count")
    ap.add_argument("--log", type=Path)
    a = p.parse_args()
    with GraphDatabase.driver(a.uri, auth=(a.user, a.password)) as d, d.session(database=a.database) as s:
        (propose if a.cmd == "propose" else apply)(a, s)


if __name__ == "__main__":
    main()
