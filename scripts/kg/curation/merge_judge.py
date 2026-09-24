"""Strict judge for entity merges: are two names the SAME entity?

Stage 4 confirms merge candidates with a lenient prompt, and with Qwen3-32B as
the confirmer about a quarter of the merges it keeps are wrong ("food" into
"food waste", "milk" into "whey"). This judge applies stricter rules — broader,
narrower, a part, an action on, or a different number is not the same entity —
and is used to re-judge merges after stage 4 (`rejudge_merges.py`) and the
bilingual unions after ingestion (`bilingual_merges.py`).

Two modes: batches without reasoning (fast), or one pair per request with
reasoning (``think=True``). Calibrated on 300 hand-labelled merges, reasoning
keeps 97.6 % correct merges against 88.4 % without, at the cost of recall.

``--calibrate FILE`` judges a labelled file (a JSON list of objects with
``alias``, ``canonico``, ``corretta`` and optionally ``fonte``) and prints
precision and recall against the labels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from openai import AsyncOpenAI

RULES = """Two names refer to the SAME entity only if they are interchangeable: one could
replace the other in any sentence of a document without changing what is being
talked about. Translations (Italian/English), spelling variants, singular/plural,
abbreviations with their expansion, and the same person or organisation written
differently ARE the same.

They are NOT the same when:
- one is broader or narrower than the other ("food" vs "food waste",
  "agriculture" vs "sustainable agriculture", "packaging" vs "plastic packaging");
- one is a part, component, aspect, property or measure of the other
  ("soil" vs "soil fertility", "milk" vs "whey", "rice straw" vs "rice husk");
- one is an action, process, goal or strategy about the other
  ("food waste" vs "food waste reduction", "waste" vs "waste management");
- they are related but distinct things (a concept and a project, journal or
  organisation named after it; a method and its result; two different numbers,
  years, places, products or people);
- one is a generic word and the other a specific instance ("the project" vs
  "NODES project").
If in doubt, answer false."""

BATCH_PROMPT = RULES + """

For each numbered pair decide whether the two names are the SAME entity.
Answer with a JSON array only: [{"id": <number>, "same": true|false}, ...]

Pairs:
%s"""

SINGLE_PROMPT = RULES + """

Pair:
A: %s
B: %s

Think briefly, then end with a last line exactly "VERDICT: SAME" or "VERDICT: DIFFERENT"."""


async def _batch(client, model, sem, batch):
    listing = "\n".join(f'{i}. "{a}" || "{b}"' for i, (a, b) in batch)
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0, max_tokens=60 * len(batch) + 200,
                    messages=[{"role": "user", "content": BATCH_PROMPT % listing}])
                text = r.choices[0].message.content or ""
                m = re.search(r"\[.*\]", text, re.DOTALL)
                items = json.loads(m.group(0)) if m else []
                return {int(x["id"]): bool(x["same"]) for x in items if isinstance(x, dict) and "id" in x}
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
    return {}


async def _single(client, model, sem, i, a, b):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0, max_tokens=1500,
                    messages=[{"role": "user", "content": SINGLE_PROMPT % (a, b)}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}})
                text = r.choices[0].message.content or ""
                m = re.findall(r"VERDICT:\s*(SAME|DIFFERENT)", text)
                if m:
                    return i, m[-1] == "SAME"
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
    return i, None


async def judge_pairs_async(pairs, base_url, model, think=False, concurrency=16, batch_size=20):
    """Judge ``pairs`` of names; one verdict per pair, None when none came back."""
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=600)
    sem = asyncio.Semaphore(concurrency)
    indexed = list(enumerate(pairs))
    verdicts: dict[int, bool | None] = {}
    if think:
        for i, v in await asyncio.gather(*[_single(client, model, sem, i, a, b) for i, (a, b) in indexed]):
            verdicts[i] = v
    else:
        batches = [indexed[k:k + batch_size] for k in range(0, len(indexed), batch_size)]
        for res in await asyncio.gather(*[_batch(client, model, sem, b) for b in batches]):
            verdicts.update(res)
    await client.close()
    return [verdicts.get(i) for i in range(len(pairs))]


def live_endpoints(endpoints: list[str]) -> list[str]:
    """The endpoints that answer ``/models`` right now."""
    import urllib.request

    up = []
    for url in endpoints:
        try:
            with urllib.request.urlopen(f"{url}/models", timeout=3) as r:
                if b'"id"' in r.read():
                    up.append(url)
        except OSError:
            pass
    return up


async def _judge_split(pairs, endpoints, model):
    parts = [pairs[k::len(endpoints)] for k in range(len(endpoints))]
    results = await asyncio.gather(*[
        judge_pairs_async(h, url, model, think=True, concurrency=24)
        for h, url in zip(parts, endpoints)
    ])
    out = [None] * len(pairs)
    for k, res_k in enumerate(results):
        for j, v in enumerate(res_k):
            out[k + j * len(endpoints)] = v
    return out


def judge_in_slices(pairs, endpoints, model, size=600):
    """Judge with reasoning, in slices, re-checking which servers answer before each."""
    out = []
    for start in range(0, len(pairs), size):
        up = live_endpoints(endpoints)
        if not up:
            raise SystemExit("no model server answers")
        out.extend(asyncio.run(_judge_split(pairs[start:start + size], up, model)))
        print(f"   {min(start + size, len(pairs))}/{len(pairs)} (servers: {len(up)})", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    p.add_argument("--think", action="store_true")
    p.add_argument("--calibrate", type=Path, required=True, help="hand-labelled merges (JSON)")
    a = p.parse_args()
    data = json.loads(a.calibrate.read_text())
    verdicts = asyncio.run(judge_pairs_async([(x["alias"], x["canonico"]) for x in data],
                                             a.base_url, a.model, a.think))
    tp = sum(1 for x, v in zip(data, verdicts) if v and x["corretta"])
    fp = sum(1 for x, v in zip(data, verdicts) if v and not x["corretta"])
    fn = sum(1 for x, v in zip(data, verdicts) if v is False and x["corretta"])
    tn = sum(1 for x, v in zip(data, verdicts) if v is False and not x["corretta"])
    none = sum(1 for v in verdicts if v is None)
    print(f"think={a.think} | kept {tp + fp}: correct {tp}, wrong {fp} -> precision {100 * tp / max(1, tp + fp):.1f}% "
          f"| correct lost {fn} of {tp + fn} -> recall {100 * tp / max(1, tp + fn):.1f}% "
          f"| wrong removed {tn} of {tn + fp} | no verdict {none}")
    for src in sorted({x.get("fonte", "") for x in data}):
        idx = [k for k, x in enumerate(data) if x.get("fonte", "") == src]
        kept = [k for k in idx if verdicts[k]]
        good = sum(1 for k in kept if data[k]["corretta"])
        print(f"   {src or '-':6s}: precision after the filter {100 * good / max(1, len(kept)):.0f}% ({good}/{len(kept)}), "
              f"before {100 * sum(1 for k in idx if data[k]['corretta']) / len(idx):.0f}%")


if __name__ == "__main__":
    main()
