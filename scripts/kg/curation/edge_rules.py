"""Fix two recurring edge errors that the extraction prompt already forbids.

1. ``AUTHORED_BY`` backwards. ``X AUTHORED_BY Y`` reads "X is authored by Y".
   When the subject is a person or a citation ("Saba et al.") and the object is
   not:
   - a ``Document`` object: the edge is reversed (7 of 7 read became true);
   - any other object (a concept): the edge is deleted (18 of 25 read stay
     false even reversed: "Chin AUTHORED_BY convergent validity").
2. ``HAS_VALUE`` whose object is a bare number (47 % false in 30 read):
   - deleted when the number does not occur in the text of the edge's chunk;
   - deleted when the subject is a questionnaire item code ("S4").
   In the sample the false share drops to 26 % and every useful value stays.
   Number nodes left without edges are deleted.

Without ``--apply`` it only counts. Every edge it touches is appended to
``--log`` with its properties, so it can be put back.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from neo4j import GraphDatabase

CITATION = re.compile(r"(et al\.?|,\s*[A-Z]\.|\b[A-Z]\.\s*[A-Z]?\.?$|&)")
BARE_NUMBER = re.compile(r"^[\s\-–+~≈<>≤≥]*[\d.,]+\s*(%|‰)?\s*$")
ITEM_CODE = re.compile(r"^[A-Z]{1,3}\d{1,2}[A-Z]?$")
DIGITS = re.compile(r"\d+(?:[.,]\d+)*")


def is_person(name: str, labels: list[str]) -> bool:
    return "Person" in labels or bool(CITATION.search(name or ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks-dir", type=Path, required=True,
                   help="run folder holding the stage1_chunks.json the graph was built from")
    p.add_argument("--uri", default="bolt://localhost:7689")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="staging-kg-v2")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--log", type=Path, help="default: <chunks-dir>/edge_rules_log.jsonl")
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    log_path = a.log or a.chunks_dir / "edge_rules_log.jsonl"
    chunks = {c["chunk_id"]: re.sub(r"\s+", " ", c["text"])
              for c in json.loads((a.chunks_dir / "stage1_chunks.json").read_text())}
    log = log_path.open("a", encoding="utf-8") if a.apply else None

    with GraphDatabase.driver(a.uri, auth=(a.user, a.password)) as d, d.session(database=a.database) as s:
        rows = s.run(
            "MATCH (a)-[r:AUTHORED_BY]->(b) RETURN elementId(r) AS id, a.name AS s, labels(a) AS sl, "
            "b.name AS o, labels(b) AS ol, properties(r) AS props").data()
        flip = [x for x in rows if is_person(x["s"], x["sl"]) and not is_person(x["o"], x["ol"]) and "Document" in x["ol"]]
        drop_auth = [x for x in rows if is_person(x["s"], x["sl"]) and not is_person(x["o"], x["ol"]) and "Document" not in x["ol"]]

        hv = s.run(
            "MATCH (a)-[r:HAS_VALUE]->(b) RETURN elementId(r) AS id, a.name AS s, b.name AS o, "
            "r.chunk_id AS chunk, properties(r) AS props").data()
        drop_hv = []
        for x in hv:
            o = (x["o"] or "").strip()
            if not BARE_NUMBER.match(o):
                continue
            text = chunks.get(x["chunk"] or "", "")
            numbers = DIGITS.findall(o)
            missing = not numbers or any(n not in text for n in numbers)
            if missing or ITEM_CODE.match((x["s"] or "").strip()):
                drop_hv.append(x)

        print(f"AUTHORED_BY: {len(rows)} | to reverse {len(flip)} | to delete {len(drop_auth)}")
        print(f"HAS_VALUE: {len(hv)} | bare numbers to delete {len(drop_hv)}")
        if not a.apply:
            print("dry run: nothing written")
            return

        for x in flip:
            log.write(json.dumps({"action": "reversed", **x}, ensure_ascii=False) + "\n")
            s.run(
                "MATCH (a)-[r]->(b) WHERE elementId(r) = $id "
                "MERGE (b)-[n:AUTHORED_BY {subject: b.name, object: a.name}]->(a) "
                "ON CREATE SET n += apoc.map.removeKeys(properties(r), ['subject', 'object']), n.direzione_corretta = true "
                "DELETE r", id=x["id"])
        for x in drop_auth + drop_hv:
            log.write(json.dumps({"action": "deleted", **x}, ensure_ascii=False) + "\n")
        ids = [x["id"] for x in drop_auth + drop_hv]
        for k in range(0, len(ids), 2000):
            s.run("MATCH ()-[r]->() WHERE elementId(r) IN $ids DELETE r", ids=ids[k:k + 2000])
        orphans = s.run(
            "MATCH (n:DataValue) WHERE NOT (n)--() WITH n, n.name AS name DELETE n RETURN count(*) AS c").single()["c"]
        print(f"applied: {len(flip)} reversed, {len(ids)} edges deleted, {orphans} number nodes left alone and deleted")
    log.close()


if __name__ == "__main__":
    main()
