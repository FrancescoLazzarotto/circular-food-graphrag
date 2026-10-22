#!/usr/bin/env python3
"""Ask a model whether each edge is stated by the passage it was extracted from.

One request per passage with all its edges, so the passage is read once
(2 292 requests instead of 30 693 for the current graph). For every edge the
model returns the words of the passage that decide it and a verdict:
``supported``, ``partial`` or ``not_supported``. When the server returns
token log-probabilities, ``p_supported`` is the probability the model gave
to ``supported`` at the first token of the verdict.

The judge must come from another model family than the extractor, and its
verdicts must never reach the person who labels the sample: they are
written under ``--out`` only.

Resumes: passages already in ``verdicts.jsonl`` are skipped.

Usage:
    python scripts/kg/quality/edge_judge.py \\
        --signals artifacts/kg_eval_2026-10/signals/<backup>/edge_signals.tsv \\
        --chunks kg_pipeline/artifacts/run_rebuild_20260922_E/stage1_chunks.json \\
        --endpoint http://localhost:8001/v1/chat/completions --model <served model> \\
        --out artifacts/kg_eval_2026-10/judge/<model> [--limit 5]
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VERDICTS = ("supported", "partial", "not_supported")
SYSTEM = (
    "You check statements that another model extracted from a text. For each numbered statement, decide whether "
    "the passage asserts it. Judge only from the passage, never from what you know. The passage may be in Italian "
    "or English; a statement in the other language counts if it says the same thing.\n"
    "- supported: the passage states it, possibly in other words, with the right names and direction.\n"
    "- partial: the two things are linked in the passage, but the relation is wrong, reversed or too vague to be "
    "useful, or a name is only a fragment of the right one.\n"
    "- not_supported: the passage does not say it.\n"
    "For 'RELATED_TO (original predicate: X)' judge the relation X; for a bare RELATED_TO judge 'is related to'. "
    "For each statement, first copy at most 20 words of the passage that decide it (empty if none), then give the "
    "verdict."
)


def schema(count: int) -> dict:
    """JSON schema of the answer for a passage with ``count`` edges."""
    item = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "evidence": {"type": "string"},
            "verdict": {"type": "string", "enum": list(VERDICTS)},
        },
        "required": ["n", "evidence", "verdict"],
    }
    return {
        "type": "object",
        "properties": {"edges": {"type": "array", "items": item, "minItems": count, "maxItems": count}},
        "required": ["edges"],
    }


def relation(row: dict) -> str:
    """The relation as the judge sees it."""
    if row["type"] == "RELATED_TO" and row["predicate"]:
        return f"RELATED_TO (original predicate: {row['predicate']})"
    return row["type"]


def prompt(text: str, edges: list[dict]) -> str:
    """The user message: the passage, then its numbered edges."""
    lines = [f"{i}. {e['subject']} — {relation(e)} → {e['object']}" for i, e in enumerate(edges, 1)]
    return f"Passage:\n\"\"\"\n{text.strip()}\n\"\"\"\n\nStatements:\n" + "\n".join(lines)


def verdict_probabilities(logprobs: list[dict]) -> list[float | None]:
    """``p_supported`` at each verdict, read from the token log-probabilities.

    The token holding the first letter of each verdict value decides which
    label is written; its top alternatives are mapped to the labels they
    begin. Positions are found on the generated text, because the quotes
    around the value may share a token with the letter or not.

    Args:
        logprobs: ``choices[0].logprobs.content`` of the response.

    Returns:
        One probability per verdict, in order; None where the alternatives
        do not map to the labels unambiguously.
    """
    starts, text = [], ""
    for token in logprobs:
        starts.append(len(text))
        text += token["token"]
    found = []
    for match in re.finditer(r'"verdict"\s*:\s*"', text):
        index = bisect.bisect_right(starts, match.end()) - 1
        mass = {label: 0.0 for label in VERDICTS}
        for alt in logprobs[index].get("top_logprobs", []):
            piece = alt["token"].lstrip(" \"':").lower()
            labels = [label for label in VERDICTS if piece and label.startswith(piece)]
            if len(labels) == 1:
                mass[labels[0]] += math.exp(alt["logprob"])
        total = sum(mass.values())
        found.append(mass["supported"] / total if total else None)
    return found


def judge(endpoint: str, model: str, chunk: str, text: str, edges: list[dict], logprobs: bool, timeout: int) -> dict:
    """Send one passage and parse the verdicts.

    Args:
        endpoint: OpenAI-compatible chat completions URL.
        model: Served model name.
        chunk: Passage id.
        text: Passage text.
        edges: Its edges, in the order they are numbered.
        logprobs: Ask for token log-probabilities.
        timeout: Seconds.

    Returns:
        A ``verdicts.jsonl`` record; ``error`` is set when the call failed.
    """
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(text, edges)}],
        "temperature": 0,
        "max_tokens": 200 + 120 * len(edges),
        "structured_outputs": {"json": schema(len(edges))},
    }
    if logprobs:
        body.update({"logprobs": True, "top_logprobs": 5})
    record = {"chunk_id": chunk, "model": model, "edges": [], "error": ""}
    started = time.time()
    try:
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        choice = payload["choices"][0]
        answer = json.loads(choice["message"]["content"])["edges"]
        probabilities = verdict_probabilities((choice.get("logprobs") or {}).get("content") or [])
        for i, edge in enumerate(edges):
            item = answer[i] if i < len(answer) else {}
            record["edges"].append({
                "edge_key": edge["edge_key"], "n": i + 1, "verdict": item.get("verdict", ""),
                "p_supported": probabilities[i] if i < len(probabilities) else None,
                "evidence": item.get("evidence", ""),
            })
        record["usage"] = payload.get("usage", {})
        record["finish_reason"] = choice.get("finish_reason", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["latency_s"] = round(time.time() - started, 2)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="judge only the first N passages (a trial)")
    parser.add_argument("--only", type=Path, help="file with one chunk_id per line to judge")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-logprobs", action="store_true")
    args = parser.parse_args()

    with args.signals.open(encoding="utf-8", newline="") as handle:
        by_chunk: dict[str, list[dict]] = defaultdict(list)
        for row in csv.DictReader(handle, delimiter="\t"):
            by_chunk[row["chunk_id"]].append(row)
    texts = {c["chunk_id"]: c["text"] for c in json.loads(args.chunks.read_text(encoding="utf-8"))}
    chunks = sorted(by_chunk)
    if args.only:
        wanted = set(args.only.read_text(encoding="utf-8").split())
        chunks = [c for c in chunks if c in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "verdicts.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["chunk_id"] for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip() and not json.loads(line)["error"]}
    todo = [c for c in chunks if c not in done][: args.limit]
    print(f"{len(todo)} passages to judge ({len(done)} already done)", flush=True)
    started = time.time()
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(judge, args.endpoint, args.model, c, texts[c],
                               sorted(by_chunk[c], key=lambda r: r["edge_key"]), not args.no_logprobs, args.timeout)
                   for c in todo]
        for count, future in enumerate(as_completed(futures), 1):
            record = future.result()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if count % 50 == 0 or count == len(todo):
                print(f"{count}/{len(todo)} in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
