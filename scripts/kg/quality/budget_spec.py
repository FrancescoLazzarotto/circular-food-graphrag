#!/usr/bin/env python3
"""Build the input of expert_budget.py from the meters and the labels already collected.

One dimension per quality dimension, each unit in the stratum its free meter
puts it in, with the human label when there is one:

- accuracy: every edge of the graph, stratified by the judge's verdict;
  labels from the main frame of the reading sample (``corretto`` 1,
  ``falso`` 0, ``non giudicabile`` left out);
- conformity: the edges that break a rule, stratified by the rule; the label
  says whether the violation is a real error (the edge read as ``falso``);
- interoperability: the candidate mappings to external vocabularies,
  stratified by match type; the label is the reading of the vocabulary
  entry (kept 1, rejected 0);
- coverage: the answers to the real questions, stratified by the judge's
  two verdicts; no labels yet.

The minutes per accuracy label come from the reading times when given.

Usage:
    python scripts/kg/quality/budget_spec.py --signals <edge_signals.tsv> --verdicts <verdicts.jsonl> \\
        --sample <reading/sample_v1> --mappings ontology/external/mappings.tsv \\
        --grounded <grounded.jsonl> --out spec.json [--labels <labels.tsv>]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABEL = {"corretto": 1, "falso": 0}
KEPT = {"checked": 1, "approved": 1, "corrected": 1, "rejected": 0}


def _tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def reading_minutes(sample: Path, labels: list[dict]) -> float | None:
    """Minutes per labelled edge, from the passages with a start and an end time.

    Args:
        sample: The reading sample folder (``times.tsv``).
        labels: The labelled rows, with ``passage`` and ``label``.

    Returns:
        Minutes per edge, or None when no passage has both times.
    """
    def minutes(clock: str) -> int:
        hours, mins = clock.split(":")
        return int(hours) * 60 + int(mins)

    done = {r["passage"]: minutes(r["end"]) - minutes(r["start"]) for r in _tsv(sample / "times.tsv") if r["start"] and r["end"]}
    edges = sum(1 for r in labels if r["passage"] in done and r["label"].strip())
    return sum(done.values()) / edges if edges else None


def build(args: argparse.Namespace) -> dict:
    signals = _tsv(args.signals)
    verdict = {}
    for line in args.verdicts.open(encoding="utf-8"):
        for edge in json.loads(line).get("edges", []):
            verdict[edge["edge_key"]] = edge.get("verdict") or "missing"
    design = {r["edge_key"]: r for r in _tsv(args.sample / "design.tsv")}
    labels = _tsv(args.labels or args.sample / "sheet.tsv")
    label = {r["edge_key"]: LABEL.get(r["label"].strip()) for r in labels if r["label"].strip()}

    accuracy = [
        {"stratum": verdict.get(r["edge_key"], "missing"),
         "y": label.get(r["edge_key"]) if design.get(r["edge_key"], {}).get("frame") == "main" else None}
        for r in signals
    ]
    conformity = [
        {"stratum": r["violations"].split(";")[0],
         "y": None if label.get(r["edge_key"]) is None else 1 - label[r["edge_key"]]}
        for r in signals if r["violations"]
    ]
    interoperability = [
        {"stratum": r["match_type"], "y": KEPT.get(r["status"])} for r in _tsv(args.mappings)
    ]
    coverage = [
        {"stratum": f"{r.get('answers', '?')}/{r.get('supported', '?')}", "y": None}
        for r in map(json.loads, args.grounded.open(encoding="utf-8"))
    ] if args.grounded else []

    per_edge = reading_minutes(args.sample, labels)
    dimensions = {
        "accuracy": {"cost_min": round(per_edge or args.cost_accuracy, 2), "deff": args.deff_accuracy, "units": accuracy},
        "conformity": {"cost_min": round(per_edge or args.cost_accuracy, 2), "units": conformity},
        "interoperability": {"cost_min": args.cost_mapping, "units": interoperability},
    }
    if coverage:
        dimensions["coverage"] = {"cost_min": args.cost_answer, "units": coverage}
    return {"dimensions": dimensions, "minutes_per_edge_measured": per_edge}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="labels to use instead of the sample's sheet.tsv")
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--grounded", type=Path)
    parser.add_argument("--deff-accuracy", type=float, default=2.0)
    parser.add_argument("--cost-accuracy", type=float, default=1.5, help="minutes per edge when no time is recorded")
    parser.add_argument("--cost-mapping", type=float, default=3.0)
    parser.add_argument("--cost-answer", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = build(args)
    args.out.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({d: {"units": len(v["units"]), "labelled": sum(u["y"] is not None for u in v["units"]),
                          "cost_min": v["cost_min"]} for d, v in spec["dimensions"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
