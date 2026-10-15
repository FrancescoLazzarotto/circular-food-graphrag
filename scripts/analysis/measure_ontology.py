#!/usr/bin/env python3
"""Answer the framework questions with the ontology off and on, side by side.

Both arms are the demo agent (``product.config.build_demo_agent``); the only
difference is ``enable_ontology``. The questions are the competency questions
of the first ontology batch (lists, versions, components) plus the forms the
session logs show, each answered alone, without memory.

Scored automatically against the expected answers written in
``ontology/competency_questions/competency_questions.tsv``:

- ``expected_found``: share of the expected names (members, components) the
  answer contains;
- ``intruders``: names of members of *other* versions of the same framework
  that the answer contains although the question asked for one version (the
  error of a list presented under the wrong version);
- latency.

The side-by-side markdown is for reading: the automatic score says whether a
name is there, not whether the answer is right.

Usage:
    python scripts/analysis/measure_ontology.py --out artifacts/ontology_measure/$(date +%Y%m%d_%H%M)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

CQ_PATH = ROOT / "ontology" / "competency_questions" / "competency_questions.tsv"
IN_BATCH = {"list_framework", "compare_versions", "broaden_narrow", "classify_case"}
# How the same questions arrive in the session logs: short, without a verb.
LOG_FORMS = [
    "Le 10R",
    "Le 10 R di errando",
    "Parlami delle 10R",
    "Cosa sono le 3C?",
    "Spiegami meglio la Coevoluzione",
]


def questions() -> list[dict]:
    """The questions with their expected names.

    Returns:
        Rows with ``id``, ``question``, ``expected`` (lists of alternative
        names, one list per expected item) and ``framework`` (the version
        asked for, when one is).
    """
    rows = []
    with CQ_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["type"] not in IN_BATCH:
                continue
            fields = dict(
                (part.partition("=")[0].strip(), part.partition("=")[2].strip())
                for part in row["expected_answer"].split(" | ")
            )
            expected = []
            for key in ("members", "narrower"):
                value = fields.get(key, "")
                # "come CQ17" points to another row: nothing to look for here.
                if value.startswith("come "):
                    continue
                for chunk in filter(None, value.split(", ")):
                    words = chunk.split(" ", 1)
                    names = words[1] if key == "members" and len(words) > 1 else chunk
                    expected.append([n.strip() for n in names.split("/")] + [names.strip()])
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected": expected,
                    "framework": fields.get("framework", "") if "members" in fields else "",
                }
            )
    rows += [{"id": f"LOG{i}", "question": q, "expected": [], "framework": ""} for i, q in enumerate(LOG_FORMS, 1)]
    return rows


def score(answer: str, row: dict, onto) -> dict:
    """Expected names found and intruders, as described in the module docstring."""
    from graphrag.ontology.consult import _compact, _stem

    words = f" {_compact(answer)} "
    stems = {_stem(w) for w in _compact(answer).split()}

    def present(name: str) -> bool:
        return f" {_compact(name)} " in words or (len(_compact(name)) >= 6 and _stem(name) in stems)

    found = sum(any(present(n) for n in names if n) for names in row["expected"])
    intruders: list[str] = []
    if row["framework"] in onto.frameworks:
        family = onto.frameworks[row["framework"]]["family"]
        own = {_stem(n) for names in row["expected"] for n in names if n}
        for fid, framework in onto.frameworks.items():
            if fid == row["framework"] or framework["family"] != family:
                continue
            for member in framework["members"]:
                label = member["label_it"] or member["label_en"]
                if label and _stem(label) not in own and present(label) and label not in intruders:
                    intruders.append(label)
    return {
        "expected_found": round(found / len(row["expected"]), 3) if row["expected"] else None,
        "intruders": intruders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from graphrag.agent.core import KGRAGAgent
    from graphrag.ontology import load
    from product.config import build_demo_agent

    # The corpus paths of the text index are relative to the repository, as
    # in the app.
    os.chdir(ROOT)
    with urllib.request.urlopen(f"{args.base_url}/models", timeout=5) as response:
        model_id = json.load(response)["data"][0]["id"]
    off, graph_label = build_demo_agent(args.base_url, model_id)
    off = KGRAGAgent(
        config=dataclasses.replace(off.config, enable_ontology=False), kg_retriever=off.kg_retriever, llm=off.llm
    )
    on = KGRAGAgent(
        config=dataclasses.replace(off.config, enable_ontology=True), kg_retriever=off.kg_retriever, llm=off.llm
    )
    onto = load()
    rows = questions()[: args.limit or None]
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(rows, 1):
        for arm, agent in (("off", off), ("on", on)):
            started = time.perf_counter()
            state = agent.invoke(row["question"])
            answer = str(state.get("answer", ""))
            results.append(
                {
                    **row,
                    "arm": arm,
                    "answer": answer,
                    "latency_s": round(time.perf_counter() - started, 2),
                    "ontology_items": len(state.get("ontology_evidence") or []),
                    "ontology_check": state.get("ontology_check"),
                    **score(answer, row, onto),
                }
            )
            print(f"[{index}/{len(rows)}] {arm} {row['id']} {results[-1]['latency_s']}s", file=sys.stderr)

    with (args.out / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {"model": model_id, "graph": graph_label, "questions": len(rows)}
    for arm in ("off", "on"):
        mine = [r for r in results if r["arm"] == arm]
        scored = [r["expected_found"] for r in mine if r["expected_found"] is not None]
        summary[arm] = {
            "expected_found_mean": round(sum(scored) / len(scored), 3) if scored else None,
            "answers_with_all_expected": sum(1 for s in scored if s == 1.0),
            "answers_with_intruders": sum(1 for r in mine if r["intruders"]),
            "latency_median_s": sorted(r["latency_s"] for r in mine)[len(mine) // 2],
            "checks_failed": sum(1 for r in mine if (r["ontology_check"] or {}).get("checked") and not r["ontology_check"]["ok"]),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# Ontologia spenta / accesa — {model_id}", ""]
    for row in rows:
        pair = {r["arm"]: r for r in results if r["id"] == row["id"]}
        lines += [f"## {row['id']} — {row['question']}", ""]
        for arm in ("off", "on"):
            r = pair[arm]
            lines += [
                f"### {arm} — attese trovate {r['expected_found']}, intrusi {', '.join(r['intruders']) or '-'}, {r['latency_s']} s",
                "",
                r["answer"],
                "",
            ]
    (args.out / "side_by_side.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
