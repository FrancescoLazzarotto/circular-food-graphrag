#!/usr/bin/env python3
"""Calibrate the out-of-domain gate threshold on top-1 dense retrieval score.

The dense retriever has no score floor: it returns the top-k nearest chunks
whatever the question, so `_grade` always sees text evidence and, without a
gate, the agent has no path to abstain — a request for a Python neural network
is answered with circular-economy PDFs as its sources.

The score is usable as-is: the FAISS index is built with normalized embeddings
and MAX_INNER_PRODUCT, so `retrieve_with_scores` returns cosine similarity,
higher meaning more similar.

This script measures the top-1 similarity of three question sets against the
same index the demo queries, so the threshold is set from data rather than by
eye:

* the 30 frozen English gold questions (in domain, must never be refused)
* Italian in-domain probes (the demo's actual audience; the corpus is bilingual
  and Italian queries are known to behave differently)
* out-of-domain probes in both languages (must be refused)

A usable threshold sits strictly below the minimum in-domain score and above
the maximum out-of-domain one. When those two ranges overlap the script says so
instead of proposing a number: no single threshold separates them, and the gate
needs a second signal.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n graphllm python scripts/domain_gate/calibrate_domain_gate.py
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from graphrag import cli as graphrag_cli  # noqa: E402

# Mirrors the demo's default in product/config.py: the text channel must be the
# one the demo actually uses, otherwise the threshold is calibrated against a
# different index.
TEXT_STAGE0_RUNS = "run_fix2docs_20260710,run_full_circular_20260707"

# In-domain Italian probes. Written to match how the expert asks — short,
# specific, no English technical vocabulary — since the gold set is English
# only and the language asymmetry is the risk this calibration must expose.
IT_IN_DOMAIN = [
    "Quali sono le 3C dell'economia circolare per il cibo?",
    "Che cos'è l'ecodesign applicato al settore alimentare?",
    "Come si riducono gli sprechi alimentari nella filiera del vino?",
    "Quali sono i sottoprodotti della vinificazione?",
    "Che cosa si intende per simbiosi industriale nel settore agroalimentare?",
    "Quali strategie di economia circolare adotta la Regione Piemonte?",
    "Come vengono valorizzati gli scarti della lavorazione del riso?",
    "Che ruolo ha il compostaggio nella gestione dei rifiuti organici?",
    "Quali indicatori misurano la circolarità di un sistema alimentare?",
    "Che cos'è la bioeconomia circolare?",
]

# Out-of-domain probes. The first is the case the gate exists for.
OUT_OF_DOMAIN = [
    "scrivi una funzione python che costruisca una rete neurale",
    "write a python function that builds a neural network",
    "Qual è la capitale dell'Australia?",
    "What is the capital of Australia?",
    "Come si cura il mal di schiena?",
    "Spiegami la relatività generale",
    "Chi ha vinto il campionato di calcio nel 2020?",
    "Scrivi una query SQL per unire due tabelle",
    "What is the derivative of sin(x)?",
    "Consigliami un film da guardare stasera",
]


def build_pipeline(backend: str = "dense") -> object:
    """The demo's text pipeline, or ``None`` when it cannot be built."""
    ns = argparse.Namespace(
        text_retriever_backend=backend,
        dense_embedding_model="intfloat/multilingual-e5-base",
        vector_index_dir=str(ROOT / "artifacts" / "vector_index"),
        text_docs_dir="",
        text_stage0_runs=TEXT_STAGE0_RUNS,
    )
    return graphrag_cli._build_text_pipeline(ns)


def load_gold_questions(path: Path) -> list[str]:
    """The query texts of a gold file."""
    data = json.loads(path.read_text())
    return [q["query"] for q in data["queries"]]


def top1_scores(pipeline: object, questions: list[str]) -> list[tuple[str, float, str]]:
    """Return (question, top-1 score, source doc) triples.

    MMR is deliberately off: it reorders the tail for diversity, and the gate
    only reads the head. The plain top-1 is the maximum similarity in the
    index, which is the quantity the threshold is about.
    """
    rows: list[tuple[str, float, str]] = []
    for question in questions:
        hits = pipeline.retrieve(query=question, top_k=1)
        if not hits:
            rows.append((question, float("-inf"), "—"))
            continue
        rows.append((question, float(hits[0].score), hits[0].source))
    return rows


def describe(label: str, rows: list[tuple[str, float, str]]) -> dict[str, float]:
    """Print the score distribution of one question set, lowest score first.

    Args:
        label: Heading of the printed block.
        rows: As returned by `top1_scores`.

    Returns:
        Count, minimum, 5th percentile, median and maximum.
    """
    scores = [s for _, s, _ in rows]
    stats = {
        "n": len(scores),
        "min": min(scores),
        "p05": statistics.quantiles(scores, n=20)[0] if len(scores) >= 20 else min(scores),
        "median": statistics.median(scores),
        "max": max(scores),
    }
    print(f"\n## {label}  (n={stats['n']})")
    print(f"   min={stats['min']:.4f}  p05={stats['p05']:.4f}  "
          f"mediana={stats['median']:.4f}  max={stats['max']:.4f}")
    for question, score, source in sorted(rows, key=lambda r: r[1]):
        print(f"   {score:.4f}  {question[:66]:<66}  {Path(source).name[:34]}")
    return stats


def main() -> int:
    """Print the three distributions and whether one threshold separates them.

    Returns:
        The exit status: 1 when the text pipeline cannot be built.
    """
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("graphrag").setLevel(logging.ERROR)

    pipeline = build_pipeline()
    if pipeline is None:
        print("Text pipeline non costruita: indice o artifacts stage0 mancanti.")
        return 1

    gold = load_gold_questions(ROOT / "evaluation" / "gold" / "gold_circular_v1.json")

    en_rows = top1_scores(pipeline, gold)
    it_rows = top1_scores(pipeline, IT_IN_DOMAIN)
    ood_rows = top1_scores(pipeline, OUT_OF_DOMAIN)

    en = describe("IN DOMINIO — gold EN (30, congelate)", en_rows)
    it = describe("IN DOMINIO — probe IT", it_rows)
    ood = describe("FUORI DOMINIO — probe EN+IT", ood_rows)

    in_domain_min = min(en["min"], it["min"])
    ood_max = ood["max"]

    print("\n" + "=" * 78)
    print(f"minimo in dominio (EN+IT): {in_domain_min:.4f}")
    print(f"massimo fuori dominio:     {ood_max:.4f}")
    print(f"margine:                   {in_domain_min - ood_max:+.4f}")

    if in_domain_min > ood_max:
        threshold = (in_domain_min + ood_max) / 2
        print(f"\nSEPARABILE. Soglia proposta (punto medio): {threshold:.4f}")
        print("Zero rifiuti sulle gold e sulle probe IT per costruzione.")
    else:
        print("\nNON SEPARABILE: gli intervalli si sovrappongono.")
        print("Nessuna soglia singola divide i due insiemi — serve un secondo")
        print("segnale (es. copertura lessicale sul grafo) oltre alla similarità.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
