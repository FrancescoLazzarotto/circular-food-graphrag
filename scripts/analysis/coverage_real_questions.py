#!/usr/bin/env python3
"""Coverage of the questions people really asked, at the schema and data levels.

The questions are those of the demo session logs
(``analysis/assistente_2026-10/sessioni_domande.json``), counted once each and
weighted by how often they were asked. Each is answered alone, without
memory, by the demo agent as configured (``product.config.build_demo_agent``).

- schema: the ontology recognises a framework, a concept or a case in the
  question (``graphrag.ontology.hooks.recognise``);
- data: at least one graph fact is cited in the answer, not only retrieved
  (the facts that really enter the answers);
- answer: whether the answer is adequate needs a judge; ``answers.jsonl``
  holds what the judge reads, and nothing here scores it.

A follow-up asked alone has lost the turn it refers to, so follow-ups and
questions about the assistant are reported apart.

Usage:
    python scripts/analysis/coverage_real_questions.py --out artifacts/kg_eval_2026-10/coverage/$(date +%Y%m%d_%H%M)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SESSIONS = ROOT / "analysis" / "assistente_2026-10" / "sessioni_domande.json"
Z = 1.959964


def wilson(successes: float, n: float) -> list[float]:
    """Wilson 95 % interval of a proportion."""
    if n <= 0:
        return [0.0, 1.0]
    p = successes / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def distinct_questions(path: Path) -> list[dict]:
    """The logged questions, once each, with how often they were asked.

    Args:
        path: The session questions file.

    Returns:
        Rows with ``question`` (first form seen), ``weight``, ``follow_up``
        and ``meta`` (true if any asking was).
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_key: dict[str, dict] = {}
    for row in rows:
        if not row.get("q"):
            continue
        key = " ".join(row["q"].lower().split())
        item = by_key.setdefault(key, {"question": row["q"].strip(), "weight": 0, "follow_up": False, "meta": False})
        item["weight"] += 1
        item["follow_up"] |= row.get("follow_up") in (True, "True")
        item["meta"] |= row.get("meta") in (True, "True")
    return sorted(by_key.values(), key=lambda r: r["question"].lower())


def share(rows: list[dict], field: str) -> dict:
    """A level's share over distinct questions and over askings, with intervals."""
    hits = [r for r in rows if r[field]]
    asked = sum(r["weight"] for r in rows)
    asked_hits = sum(r["weight"] for r in hits)
    return {
        "distinct": [len(hits), len(rows)],
        "distinct_ci95": wilson(len(hits), len(rows)),
        # Askings of the same question are not independent, so the interval
        # stays on the distinct questions.
        "weighted": [asked_hits, asked],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help="re-read the follow-up and meta flags, rewrite answers.jsonl and recompute summary.json",
    )
    parser.add_argument(
        "--answerable-only", action="store_true", help="skip follow-ups and questions about the assistant"
    )
    parser.add_argument(
        "--save-evidence", action="store_true", help="keep the passages and graph facts the model was shown"
    )
    args = parser.parse_args()

    if args.summarise_only:
        flags = {r["question"]: r for r in distinct_questions(SESSIONS)}
        results = [json.loads(line) for line in (args.out / "answers.jsonl").open(encoding="utf-8")]
        for result in results:
            result["follow_up"], result["meta"] = flags[result["question"]]["follow_up"], flags[result["question"]]["meta"]
        with (args.out / "answers.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        old = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
        write_summary(args.out, results, {"model": old["model"], "graph": old["graph"]})
        return

    from graphrag.ontology import hooks, load
    from product.config import build_demo_agent

    # The corpus paths of the text index are relative to the repository, as
    # in the app.
    os.chdir(ROOT)
    with urllib.request.urlopen(f"{args.base_url}/models", timeout=5) as response:
        model_id = json.load(response)["data"][0]["id"]
    agent, graph_label = build_demo_agent(args.base_url, model_id)
    onto = load()
    rows = distinct_questions(SESSIONS)
    if args.answerable_only:
        rows = [r for r in rows if not (r["follow_up"] or r["meta"])]
    rows = rows[: args.limit or None]
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.out / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            match = hooks.recognise(row["question"], onto)
            started = time.perf_counter()
            state = agent.invoke(row["question"])
            cited = (state.get("citation_report") or {}).get("cited_refs") or []
            result = {
                **row,
                "answer": str(state.get("answer", "")),
                "latency_s": round(time.perf_counter() - started, 2),
                "schema": bool(match.frameworks or match.concepts or match.cases),
                "recognised": {"frameworks": match.frameworks, "concepts": match.concepts, "cases": match.cases},
                "triples_retrieved": len(state.get("kg_triples") or []),
                "triples_cited": sum(ref.startswith("T") for ref in cited),
                "passages_cited": sum(ref.startswith("S") for ref in cited),
                "out_of_scope": bool(state.get("out_of_scope")),
                "meta_question": bool(state.get("meta_question")),
            }
            result["data"] = result["triples_cited"] > 0
            if args.save_evidence:
                visible = set(state.get("visible_evidence_refs") or [])
                result["evidence"] = [
                    {k: item.get(k) for k in ("ref_id", "kind", "text", "source_doc", "pages")}
                    for item in state.get("evidence_index") or []
                    if not visible or item.get("ref_id") in visible
                ]
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {result['latency_s']}s schema={result['schema']} data={result['data']}", file=sys.stderr)

    write_summary(args.out, results, {"model": model_id, "graph": graph_label})


def write_summary(out: Path, results: list[dict], header: dict) -> None:
    """Write ``summary.json``: each level over all questions and over the answerable ones.

    Answerable: in scope, not a follow-up, not about the assistant.

    Args:
        out: The run folder.
        results: The rows of ``answers.jsonl``.
        header: Model and graph of the run.
    """
    answerable = [r for r in results if not (r["follow_up"] or r["meta"] or r["out_of_scope"])]
    latencies = sorted(r["latency_s"] for r in results)
    summary = {
        **header,
        "questions": len(results),
        "askings": sum(r["weight"] for r in results),
        "follow_up": sum(r["follow_up"] for r in results),
        "meta": sum(r["meta"] for r in results),
        "out_of_scope": sum(r["out_of_scope"] for r in results),
        "all": {level: share(results, level) for level in ("schema", "data")},
        "answerable": {level: share(answerable, level) for level in ("schema", "data")},
        "answerable_schema_and_data": share([{**r, "both": r["schema"] and r["data"]} for r in answerable], "both"),
        "graph_retrieved_not_cited": sum(1 for r in results if r["triples_retrieved"] and not r["data"]),
        "latency_s": {"median": latencies[len(latencies) // 2], "max": latencies[-1]} if latencies else {},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
