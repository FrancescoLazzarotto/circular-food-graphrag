#!/usr/bin/env python3
"""Validate the domain gate on questions it was not tuned on.

The scope description is tuned against the questions in
`eval_domain_gate_llm.py`, so a score there measures fit, not generalisation.
This runs the shipped scope text against a held-out set never used to edit it.

The out-of-domain half is deliberately adversarial for this deployment. The
demo is shown to a gastronomy department, so the realistic wrong question is
not "what is the capital of Australia" — it is a recipe, a nutrition question,
or a restaurant recommendation: food-related, and still outside a collection
about circular economy, by-product valorisation and sustainability policy.
Those are the cases a topic gate is most likely to wave through.

Usage:
    conda run -n graphllm python scripts/domain_gate/eval_domain_gate_heldout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "domain_gate"))

from eval_domain_gate_llm import run, served_model  # noqa: E402

# In domain, held out. Mix of EN and IT, including corpus topics that never came
# up in tuning: dairy, olive oil, packaging logistics, urban food policy.
HELDOUT_IN = [
    "Quali sono gli impieghi delle sanse di oliva?",
    "Che cosa succede al siero di latte dopo la produzione di formaggio?",
    "Come si misura l'impronta idrica di un prodotto alimentare?",
    "Quali politiche urbane favoriscono un sistema alimentare circolare?",
    "Che differenza c'è tra riciclo e riuso nella filiera alimentare?",
    "Quali sono i vantaggi ambientali della filiera corta?",
    "What by-products come from brewing beer, and how are they reused?",
    "How can food packaging be redesigned to reduce waste?",
    "What is the role of anaerobic digestion in food waste treatment?",
    "Which economic actors are involved in a territorial food system?",
    "Come si riduce la perdita di prodotto nella fase di trasporto alimentare?",
    "What indicators track the environmental impact of a supply chain?",
]

# Out of domain, held out, adversarially food-adjacent.
HELDOUT_OUT = [
    "Qual è la ricetta della carbonara?",
    "Quante calorie ha una mela?",
    "Consigliami un ristorante a Torino",
    "A che temperatura si cuoce il petto di pollo?",
    "Quanto tempo si conserva il latte aperto in frigorifero?",
    "What wine pairs well with grilled fish?",
    "How do I make sourdough starter from scratch?",
    "Sono allergico alle arachidi, cosa devo evitare?",
    "Come si configura un server nginx?",
    "Chi ha scritto la Divina Commedia?",
    "Qual è la differenza tra Python e Java?",
    "Come si allena la resistenza nella corsa?",
]


def main() -> int:
    """Run the held-out suites and print the error counts."""
    model_id = served_model()
    print(f"modello: {model_id}")
    print("scope: quello congelato in eval_domain_gate_llm.GATE_SYSTEM\n")

    false_refusals, _ = run("HELD-OUT in dominio", HELDOUT_IN, "IN", model_id)
    false_accepts, _ = run("HELD-OUT fuori dominio (food-adjacent)", HELDOUT_OUT, "OUT", model_id)

    print("\n" + "=" * 78)
    print(f"rifiuti falsi:      {false_refusals} / {len(HELDOUT_IN)}")
    print(f"accettazioni false: {false_accepts} / {len(HELDOUT_OUT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
