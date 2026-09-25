#!/usr/bin/env python3
"""Measure the gate on questions whose subject is a name, not a topic.

The two topic suites ask about by-products, packaging, indicators — which the
model can place from world knowledge alone — and contain no proper noun. A name
is where the gate goes wrong, and not through the model's judgement but through
its vocabulary: an acronym it has never seen, as in "Che cos'è SeED?", looks
like no domain at all, even while the graph holds nodes named after it.

So `_scope_gate` looks the proper nouns up in the graph and tells the model
which names the collection actually contains. This suite checks both halves of
that bargain:

* NAMED_IN — a name the graph holds, asked about plainly. Must be IN.
* NAMED_OUT_OF_DOMAIN — a name the graph also holds, asked about in a way the
  collection does not answer. Must stay OUT: the hint says a node exists, it
  does not say the question is in scope. This is the half that would break if
  the lookup were allowed to short-circuit the verdict.

Unlike its two sibling suites this one needs the graph as well as the model,
because the names come from the graph.

Usage:
    conda run -n graphllm python scripts/domain_gate/eval_domain_gate_entities.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Names the corpus holds, asked about the way an expert actually types them:
# no "the project", no context, just the name.
NAMED_IN = [
    "Che cos'è SeED?",
    "Che cos'è SEeD?",
    "What is SEeD?",
    "Parlami di Systemic Event Design",
    "Che cos'è Cartacrusca?",
    "Chi è Barilla?",
    "Che cos'è REPAiR?",
    "Cos'è il MATTM?",
    "Che cosa fa Environment Park?",
    "Che cos'è Slow Food?",
]

# The same graph, the wrong question. Every name here resolves to a node, so the
# gate is handed a hint on all of them — and must refuse anyway.
NAMED_OUT_OF_DOMAIN = [
    "Consigliami un ristorante a Torino",
    "Quanto dista Milano da Roma?",
    "Che tempo fa a Torino domani?",
    "Quanti abitanti ha Milano?",
    "Come arrivo al Politecnico di Torino in metro?",
    "Quanto costa un pacco di pasta Barilla al supermercato?",
    "In che anno è stata fondata Roma?",
    "Che partite gioca il Torino in casa?",
]


def main() -> int:
    """Run both name suites against the demo agent.

    Returns:
        The exit status: 0 when every verdict is right, 1 on errors, 2 when no
        model is reachable or the gate is off.
    """
    from product import config as settings

    options = settings.probe_vllm_endpoints()
    if not options:
        print("nessun endpoint vLLM raggiungibile", file=sys.stderr)
        return 2
    base_url, model_id = next(iter(options.values()))
    agent, graph_label = settings.build_demo_agent(base_url, model_id)
    if not agent.config.enable_domain_gate:
        print("il gate è disattivo in questa configurazione", file=sys.stderr)
        return 2
    print(f"modello: {model_id}")
    print(f"grafo:   {graph_label}")

    failures = 0
    for label, questions, expected in (
        ("NOMI IN DOMINIO", NAMED_IN, True),
        ("NOMI FUORI DOMINIO", NAMED_OUT_OF_DOMAIN, False),
    ):
        errors: list[str] = []
        latencies: list[float] = []
        print(f"\n## {label}")
        for question in questions:
            start = time.perf_counter()
            names = agent._known_entity_names(question)
            verdict = agent._scope_gate({"question": question})["in_domain"]
            latencies.append(time.perf_counter() - start)
            ok = verdict is expected
            print(
                f"  {'ok' if ok else 'XX'} {'IN' if verdict else 'OUT':>3}  "
                f"{question}\n        grafo: {names or '—'}"
            )
            if not ok:
                errors.append(question)
        mean = sum(latencies) / max(1, len(latencies))
        print(
            f"  → {len(questions) - len(errors)}/{len(questions)} corrette "
            f"(lookup + gate, latenza media {mean:.2f}s)"
        )
        failures += len(errors)

    print("\n" + "=" * 78)
    total = len(NAMED_IN) + len(NAMED_OUT_OF_DOMAIN)
    print(f"errori: {failures} / {total}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
