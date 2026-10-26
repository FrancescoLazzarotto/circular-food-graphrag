#!/usr/bin/env python3
"""Blind judgements of the assistant's answers by a model of another family.

Three commands:

- ``pairs``: for every question answered with the ontology off and on
  (``answers.jsonl`` of ``measure_ontology.py``), the two answers are shown
  as "first" and "second" without saying which is which, once in each
  order. Which arm is "A" is drawn at random (seed 42) and written to
  ``key.json``, which the judging never reads. When the question comes from
  a competency question, its expected answer is given as the reference.
- ``unblind``: joins the verdicts with the key. A pair is won only if the
  judge picks the same arm in both orders; otherwise it is a tie. Then the
  decision rule fixed before the measurement: on the framework questions
  the "on" wins exceed the "off" wins with an exact two-sided sign test
  p < 0.05; answers with an intruder are no more with the ontology on; on
  the control questions "off" wins at most once.
- ``adequacy``: one verdict per answer (adequate, partial, inadequate) for
  the answers of ``coverage_real_questions.py`` that can be answered alone.

The judge sees no documents, so it cannot check facts the reference does not
give; it judges what can be judged from the question and the answers.

Usage:
    python scripts/analysis/blind_judge.py pairs --answers <onoff>/answers.jsonl --out <dir> --model <served model>
    python scripts/analysis/blind_judge.py unblind --answers <onoff>/answers.jsonl --out <dir>
    python scripts/analysis/blind_judge.py adequacy --answers <coverage>/answers.jsonl --out <dir> --model <served model>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CQ_PATH = ROOT / "ontology" / "competency_questions" / "competency_questions.tsv"
FRAMEWORK_GROUPS = {"cq", "paraphrase", "log", "log_missed"}

PAIR_SYSTEM = (
    "You compare two answers an assistant gave to the same question. The assistant answers people of a "
    "Department of Gastronomic Sciences from a collection of documents on the circular economy of food; answers "
    "may be in Italian or English. Decide which answer is better for the person who asked. In order of weight:\n"
    "1. Correctness. When a reference is given, the names, number and order of the items asked for must match "
    "it; an item from another version of the same framework, or an invented one, is an error.\n"
    "2. It answers what was asked: the version or framework named in the question, not another one.\n"
    "3. Completeness on what was asked.\n"
    "4. It says where its statements come from.\n"
    "Do not prefer an answer for being longer. Style counts only when it gets in the way of understanding. If "
    "the two are equally good or equally bad, the winner is tie. First write the reason in at most 60 words, "
    "then the winner and whether the difference is clear or slight."
)
PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "winner": {"type": "string", "enum": ["first", "second", "tie"]},
        "margin": {"type": "string", "enum": ["clear", "slight"]},
    },
    "required": ["reason", "winner", "margin"],
}

ADEQUACY_SYSTEM = (
    "You judge one answer an assistant gave to a question. The assistant answers people of a Department of "
    "Gastronomic Sciences from a collection of documents on the circular economy of food; answers may be in "
    "Italian or English. You cannot see the documents, so do not judge facts you cannot check. Judge whether "
    "the answer is adequate for the person who asked:\n"
    "- adequate: it answers the question that was asked, coherently, and says where its statements come from;\n"
    "- partial: it answers only part of the question, drifts to a different topic for much of it, or is too "
    "vague to be useful;\n"
    "- inadequate: it does not answer the question asked, contradicts itself, or refuses a question that is "
    "about the documents' subject.\n"
    "First write the reason in at most 60 words, then the verdict."
)
ADEQUACY_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["adequate", "partial", "inadequate"]},
    },
    "required": ["reason", "verdict"],
}


def references() -> dict[str, str]:
    """The expected answer of each competency question, without its source pages."""
    with CQ_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        row["id"]: " | ".join(p for p in row["expected_answer"].split(" | ") if not p.startswith("source="))
        for row in rows
    }


def call(endpoint: str, model: str, system: str, user: str, schema: dict, timeout: int) -> dict:
    """One judgement; ``error`` is set when the call failed."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 400,
        "structured_outputs": {"json": schema},
    }
    started = time.time()
    # A failed judgement would count as a tie, so it is tried once more.
    for _ in range(2):
        try:
            request = urllib.request.Request(
                endpoint, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            result = json.loads(payload["choices"][0]["message"]["content"])
            result["finish_reason"] = payload["choices"][0].get("finish_reason", "")
            result["error"] = ""
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
    result["latency_s"] = round(time.time() - started, 2)
    return result


def pair_prompt(question: str, reference: str, first: str, second: str) -> str:
    """The user message of a pair."""
    head = f"Question:\n{question}\n\n"
    if reference:
        head += f"Reference (what a correct answer must contain):\n{reference}\n\n"
    return head + f"First answer:\n\"\"\"\n{first}\n\"\"\"\n\nSecond answer:\n\"\"\"\n{second}\n\"\"\""


def run_pairs(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.answers.open(encoding="utf-8")]
    arms: dict[str, dict] = {}
    for row in rows:
        arms.setdefault(row["id"], {})[row["arm"]] = row
    rng = random.Random(42)
    key, jobs = {}, []
    refs = references()
    for qid in sorted(arms):
        pair = arms[qid]
        a, b = ("on", "off") if rng.random() < 0.5 else ("off", "on")
        key[qid] = {"A": a, "B": b}
        row = pair["off"]
        reference = refs.get(row.get("cq") or "", "")
        for order, (first, second) in (("AB", (a, b)), ("BA", (b, a))):
            jobs.append((qid, order, pair_prompt(row["question"], reference, pair[first]["answer"], pair[second]["answer"])))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda j: call(args.endpoint, args.model, PAIR_SYSTEM, j[2], PAIR_SCHEMA, args.timeout), jobs))
    with (args.out / "verdicts.jsonl").open("w", encoding="utf-8") as handle:
        for (qid, order, _), result in zip(jobs, results):
            handle.write(json.dumps({"id": qid, "order": order, "model": args.model, **result}, ensure_ascii=False) + "\n")
    errors = sum(1 for r in results if r["error"])
    print(f"{len(results)} judgements of {len(key)} pairs, {errors} errors")


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test on the pairs that are not ties."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def run_unblind(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.answers.open(encoding="utf-8")]
    by = {(r["id"], r["arm"]): r for r in rows}
    key = json.loads((args.out / "key.json").read_text(encoding="utf-8"))
    verdicts: dict[str, dict] = {}
    for line in (args.out / "verdicts.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        verdicts.setdefault(v["id"], {})[v["order"]] = v
    outcome = {}
    for qid, arms in key.items():
        picks = []
        for order in ("AB", "BA"):
            v = verdicts.get(qid, {}).get(order, {})
            winner = v.get("winner", "tie") if not v.get("error") else "tie"
            slot = {"first": order[0], "second": order[1]}.get(winner)
            picks.append(arms[slot] if slot else "tie")
        outcome[qid] = picks[0] if picks[0] == picks[1] and picks[0] != "tie" else "tie"

    def tally(ids: list[str]) -> dict:
        count = Counter(outcome[i] for i in ids)
        return {"on": count["on"], "off": count["off"], "tie": count["tie"], "n": len(ids),
                "sign_test_p": round(sign_test(count["on"], count["off"]), 4)}

    groups = {qid: by[(qid, "off")].get("group", "") for qid in key}
    framework = [q for q, g in groups.items() if g in FRAMEWORK_GROUPS]
    control = [q for q, g in groups.items() if g == "control"]
    intruders = {arm: sum(1 for (qid, a), r in by.items() if a == arm and r["intruders"]) for arm in ("off", "on")}
    fw, ctl = tally(framework), tally(control)
    rule = {
        "1_framework_on_beats_off": fw["on"] > fw["off"] and fw["sign_test_p"] < 0.05,
        "2_intruders_not_more": intruders["on"] <= intruders["off"],
        "3_controls_off_wins_at_most_1": ctl["off"] <= 1,
    }
    latency = {
        arm: sorted(r["latency_s"] for (q, a), r in by.items() if a == arm)[len(key) // 2] for arm in ("off", "on")
    }
    result = {
        "framework": fw,
        "control": ctl,
        "by_group": {g: tally([q for q, gg in groups.items() if gg == g]) for g in sorted(set(groups.values()))},
        "answers_with_intruders": intruders,
        "refused": {arm: sum(1 for (q, a), r in by.items() if a == arm and r.get("out_of_scope")) for arm in ("off", "on")},
        "latency_median_s": latency,
        "order_disagreements": sum(
            1 for qid in key
            if len({verdicts.get(qid, {}).get(o, {}).get("winner") for o in ("AB", "BA")}) > 1
        ),
        "rule": rule,
        "switch_on": all(rule.values()),
        "outcome": outcome,
    }
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "outcome"}, ensure_ascii=False, indent=1))


def run_adequacy(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.answers.open(encoding="utf-8")]
    todo = [r for r in rows if not (r["follow_up"] or r["meta"] or r["out_of_scope"])]
    prompts = [f"Question:\n{r['question']}\n\nAnswer:\n\"\"\"\n{r['answer']}\n\"\"\"" for r in todo]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda p: call(args.endpoint, args.model, ADEQUACY_SYSTEM, p, ADEQUACY_SCHEMA, args.timeout), prompts))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "adequacy.jsonl").open("w", encoding="utf-8") as handle:
        for row, result in zip(todo, results):
            handle.write(json.dumps({"question": row["question"], "schema": row["schema"], "data": row["data"],
                                     "model": args.model, **result}, ensure_ascii=False) + "\n")
    count = Counter(r.get("verdict", "error") for r in results)
    print(json.dumps({"answers": len(todo), **count}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["pairs", "unblind", "adequacy"])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--model", default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    {"pairs": run_pairs, "unblind": run_unblind, "adequacy": run_adequacy}[args.command](args)


if __name__ == "__main__":
    main()
