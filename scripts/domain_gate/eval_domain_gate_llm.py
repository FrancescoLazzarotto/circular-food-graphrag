#!/usr/bin/env python3
"""Measure an LLM topic gate on the same questions the score threshold failed on.

The dense top-1 similarity does not separate in-domain from out-of-domain
questions (see `calibrate_domain_gate.py`): e5 compresses everything into a
narrow band, the two ranges overlap and no single threshold works.

This evaluates the alternative: one short classification call to the serving
model before retrieval, at temperature 0, answering a single token. The gate is
only worth wiring if it clears both error classes on the same 50 questions:

* false refusal — an in-domain question judged out of domain. Expensive: the
  expert asks something legitimate and the demo stonewalls. Must be 0.
* false accept — an out-of-domain question judged in domain. Cheap by
  comparison: the answer still gets generated, marked as ungrounded.

Usage:
    conda run -n graphllm python scripts/domain_gate/eval_domain_gate_llm.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "domain_gate"))

from calibrate_domain_gate import IT_IN_DOMAIN, OUT_OF_DOMAIN, load_gold_questions  # noqa: E402
from graphrag.llm.prompts import PromptLibrary  # noqa: E402

VLLM_URL = "http://localhost:8000/v1/chat/completions"

# The scope description is the whole gate: it must name the domain widely enough
# to cover what the corpus actually holds (circular economy, food systems,
# agri-food by-products, sustainability policy, territorial projects) without
# turning into "anything to do with food", which would let a recipe through.
#
# Read from PromptLibrary, never copied: a copy drifts, and one clause of
# difference is enough to move held-out questions from OUT to IN. A suite that
# scores a prompt nobody runs is not a suite.


def gate_system(known_entities: Sequence[str] = ()) -> str:
    """The shipped domain-gate system message, optionally naming graph entities."""
    return PromptLibrary.domain_gate_prompt(
        known_entities=known_entities
    ).messages[0].prompt.template


GATE_SYSTEM = gate_system()


def classify(
    question: str, model_id: str, known_entities: Sequence[str] = ()
) -> tuple[str, float]:
    """Ask the served model for a verdict on one question.

    Args:
        question: The question as typed.
        model_id: Served model name.
        known_entities: Graph names to mention in the system message.

    Returns:
        ``"IN"`` or ``"OUT"``, and the seconds the call took.
    """
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": gate_system(known_entities)},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "max_tokens": 4,
    }
    request = urllib.request.Request(
        VLLM_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - start
    verdict = body["choices"][0]["message"]["content"].strip().upper()
    return ("OUT" if verdict.startswith("OUT") else "IN"), elapsed


def served_model() -> str:
    """The id of the model served on port 8000."""
    with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5) as response:
        return json.load(response)["data"][0]["id"]


def run(label: str, questions: list[str], expected: str, model_id: str) -> tuple[int, list[str]]:
    """Classify a question set and print the questions judged wrongly.

    Args:
        label: Heading of the printed block.
        questions: The questions.
        expected: ``"IN"`` or ``"OUT"``.
        model_id: Served model name.

    Returns:
        The number of errors and their printed lines.
    """
    errors: list[str] = []
    latencies: list[float] = []
    for question in questions:
        verdict, elapsed = classify(question, model_id)
        latencies.append(elapsed)
        if verdict != expected:
            errors.append(f"   {verdict} (atteso {expected})  {question}")
    mean_latency = sum(latencies) / max(1, len(latencies))
    ok = len(questions) - len(errors)
    print(f"\n## {label}: {ok}/{len(questions)} corrette  (latenza media {mean_latency:.2f}s)")
    for line in errors:
        print(line)
    return len(errors), errors


def main() -> int:
    """Run the three suites and print whether the gate can be used."""
    model_id = served_model()
    print(f"modello: {model_id}")

    gold = load_gold_questions(ROOT / "evaluation" / "gold" / "gold_circular_v1.json")

    false_refusals_en, _ = run("IN DOMINIO — gold EN (30)", gold, "IN", model_id)
    false_refusals_it, _ = run("IN DOMINIO — probe IT (10)", IT_IN_DOMAIN, "IN", model_id)
    false_accepts, _ = run("FUORI DOMINIO — probe (10)", OUT_OF_DOMAIN, "OUT", model_id)

    false_refusals = false_refusals_en + false_refusals_it
    print("\n" + "=" * 78)
    print(f"rifiuti falsi (in dominio giudicato OUT):  {false_refusals} / 40")
    print(f"accettazioni false (fuori dominio → IN):   {false_accepts} / 10")
    if false_refusals == 0 and false_accepts == 0:
        print("\nSEPARAZIONE PERFETTA sui 50 casi. Il gate è cablabile.")
    elif false_refusals == 0:
        print("\nNESSUN RIFIUTO FALSO. Il gate è cablabile: le accettazioni false")
        print("cadono nel percorso 'rispondi ma marca', che è il comportamento")
        print("previsto per il retrieval debole.")
    else:
        print("\nRIFIUTI FALSI PRESENTI: il gate bloccherebbe domande legittime.")
        print("Non cablare finché il prompt di scope non li azzera.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
