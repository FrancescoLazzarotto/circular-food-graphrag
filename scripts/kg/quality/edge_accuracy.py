#!/usr/bin/env python3
"""Accuracy of the graph from a labelled reading sample, and what strata would save.

Reads a sample drawn by ``edge_sample.py`` once a person has filled
``sheet.tsv`` (``corretto`` / ``falso`` / ``non_giudicabile``), its hidden
``design.tsv``, ``times.tsv`` and, optionally, a judge's ``verdicts.jsonl``.

- Accuracy: share of correct edges on the main sample, weighted by the
  inverse of the inclusion probability, with a variance that treats the
  passage as the cluster (with-replacement linearisation) and a Wilson
  interval on the effective sample size (``edges / design effect``).
  Unjudgeable edges are left out and counted.
- Strata (one scheme per row): for each way of grouping edges (none, the
  ontology signals, the judge's verdict, both, and two controls: the
  extractor's confidence and the document) the within-stratum spread of the
  labels and the number of edges a Neyman allocation needs for a given
  margin, against simple random sampling. Strata rates use every labelled
  edge (main and supplements) with weights ``1 / pi``.
- Agreement: on the passages read twice, raw agreement and Cohen's kappa.
- Judge: agreement of the verdict with the person on the labelled edges.
- Cost: minutes per passage from ``times.tsv``.

Usage:
    python scripts/kg/quality/edge_accuracy.py --sample artifacts/kg_eval_2026-10/reading/sample_v1 \\
        [--verdicts artifacts/kg_eval_2026-10/judge/gemma4_31b/verdicts.jsonl] [--margin 0.05]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

LABELS = {"corretto": 1, "falso": 0}
Z = 1.959964


def wilson(p: float, n_eff: float, z: float = Z) -> tuple[float, float]:
    """Wilson interval for a proportion on an effective sample size."""
    if n_eff <= 0:
        return (0.0, 1.0)
    centre = (p + z * z / (2 * n_eff)) / (1 + z * z / n_eff)
    half = z * math.sqrt(p * (1 - p) / n_eff + z * z / (4 * n_eff * n_eff)) / (1 + z * z / n_eff)
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_estimate(rows: list[dict]) -> dict:
    """Weighted share of correct edges with a passage-clustered variance.

    Args:
        rows: Labelled edges with ``y`` (0/1), ``w`` (1/pi) and ``cluster``.

    Returns:
        ``p``, ``se``, ``deff``, ``n``, ``clusters``, ``n_eff``, ``low``, ``high``.
    """
    n = len(rows)
    total_w = sum(r["w"] for r in rows)
    if not n or not total_w:
        return {"p": None, "n": n}
    p = sum(r["w"] * r["y"] for r in rows) / total_w
    z_by: dict[str, float] = defaultdict(float)
    for r in rows:
        z_by[r["cluster"]] += r["w"] * (r["y"] - p)
    k = len(z_by)
    var = k / (k - 1) * sum(z * z for z in z_by.values()) / total_w ** 2 if k > 1 else float("nan")
    srs = p * (1 - p) / n
    deff = var / srs if srs > 0 else 1.0
    n_eff = n / deff if deff > 0 else n
    low, high = wilson(p, n_eff)
    return {"p": round(p, 4), "se": round(math.sqrt(var), 4), "deff": round(deff, 3), "n": n, "clusters": k,
            "n_eff": round(n_eff, 1), "low": round(low, 4), "high": round(high, 4)}


def neyman_need(rows: list[dict], key: str, margin: float, deff: float) -> dict:
    """Edges needed for ``margin`` at 95 % with Neyman allocation over the strata of ``key``.

    Args:
        rows: Labelled edges with ``y``, ``w`` and the stratum fields.
        key: Field naming the stratum.
        margin: Half-width wanted.
        deff: Design effect of the cluster sample, applied to every scheme.

    Returns:
        Strata sizes and rates, and the edges needed with and without strata.
    """
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[str(r[key])].append(r)
    total_w = sum(r["w"] for r in rows)
    p = sum(r["w"] * r["y"] for r in rows) / total_w
    spread = 0.0
    strata = {}
    for value, members in sorted(by.items()):
        w_h = sum(r["w"] for r in members) / total_w
        p_h = sum(r["w"] * r["y"] for r in members) / sum(r["w"] for r in members)
        spread += w_h * math.sqrt(p_h * (1 - p_h))
        strata[value] = {"share": round(w_h, 4), "accuracy": round(p_h, 4), "labelled": len(members)}
    factor = (Z / margin) ** 2 * deff
    return {
        "strata": strata,
        "edges_needed_srs": math.ceil(factor * p * (1 - p)),
        "edges_needed_neyman": math.ceil(factor * spread ** 2),
        "saving": round(1 - spread ** 2 / (p * (1 - p)), 4) if 0 < p < 1 else None,
    }


def kappa(pairs: list[tuple[int, int]]) -> dict:
    """Raw agreement and Cohen's kappa of two readings of the same edges."""
    if not pairs:
        return {"pairs": 0}
    agree = sum(a == b for a, b in pairs) / len(pairs)
    pa = sum(a for a, _ in pairs) / len(pairs)
    pb = sum(b for _, b in pairs) / len(pairs)
    chance = pa * pb + (1 - pa) * (1 - pb)
    return {"pairs": len(pairs), "agreement": round(agree, 4),
            "kappa": round((agree - chance) / (1 - chance), 4) if chance < 1 else None}


def confidence_band(value: str) -> str:
    """A coarse band of the extractor's confidence, a control signal."""
    try:
        c = float(value)
    except ValueError:
        return "?"
    return "<0.8" if c < 0.8 else ("0.8-0.85" if c <= 0.85 else ">0.85")


def load(sample: Path, verdicts: Path | None) -> tuple[list[dict], list[dict], dict]:
    """Join the sheet, the design and the judge into one row per reading.

    Args:
        sample: Folder of the sample.
        verdicts: Judge output, or None.

    Returns:
        Labelled readings, all readings, and the judge verdict per edge key.
    """
    def tsv(name: str) -> list[dict]:
        with (sample / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    sheet, design = tsv("sheet.tsv"), tsv("design.tsv")
    judge: dict[str, str] = {}
    if verdicts and verdicts.exists():
        for line in verdicts.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not record["error"]:
                for e in record["edges"]:
                    judge[e["edge_key"]] = e["verdict"]
    rows = []
    for s, d in zip(sheet, design):
        if s["edge_key"] != d["edge_key"] or s["passage"] != d["passage"]:
            raise ValueError(f"sheet and design are out of step at passage {s['passage']}")
        label = s["label"].strip().lower()
        rows.append({
            **d, "label": label, "error_type": s["error_type"].strip(), "y": LABELS.get(label),
            "w": 1 / float(d["pi"]) if d["frame"] != "double" else 0.0, "cluster": d["chunk_id"],
            "ontology": f"fallback={d['fallback']}|concept={d['concept_end'] != 'none'}|violation={bool(d['violations'])}",
            "judge": judge.get(d["edge_key"], "?"),
            "confidence_band": confidence_band(d["confidence"]),
            "document": d["chunk_id"].rsplit("_chunk", 1)[0],
            "none": "all",
        })
    for r in rows:
        r["combined"] = f"{r['ontology']}|judge={r['judge']}"
    labelled = [r for r in rows if r["y"] is not None and r["frame"] != "double"]
    return labelled, rows, judge


def minutes(sample: Path) -> dict:
    """Minutes per passage read, from ``times.tsv`` (HH:MM)."""
    spent = []
    with (sample / "times.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                start = datetime.strptime(row["start"].strip(), "%H:%M")
                end = datetime.strptime(row["end"].strip(), "%H:%M")
            except ValueError:
                continue
            spent.append((end - start).seconds / 60)
    if not spent:
        return {"passages_timed": 0}
    spent.sort()
    return {"passages_timed": len(spent), "minutes_total": round(sum(spent), 1),
            "minutes_median": spent[len(spent) // 2], "minutes_mean": round(sum(spent) / len(spent), 2)}


def analyse(sample: Path, verdicts: Path | None, margin: float) -> dict:
    """Every number the report gives, from one labelled sample."""
    labelled, rows, judge = load(sample, verdicts)
    main = [r for r in labelled if r["frame"] == "main"]
    estimate = cluster_estimate(main)
    deff = estimate.get("deff") or 1.0
    schemes = ["none", "ontology", "judge", "combined", "confidence_band", "document"]
    if not judge:
        schemes = [s for s in schemes if s not in ("judge", "combined")]
    first = {r["edge_key"]: r["y"] for r in rows if r["frame"] != "double" and r["y"] is not None}
    pairs = [(first[r["edge_key"]], r["y"]) for r in rows
             if r["frame"] == "double" and r["y"] is not None and r["edge_key"] in first]
    judged = [r for r in labelled if r["judge"] != "?"]
    return {
        "readings": len(rows),
        "labels": dict(Counter(r["label"] or "(vuota)" for r in rows if r["frame"] != "double")),
        "error_types": dict(Counter(r["error_type"] for r in labelled if r["y"] == 0)),
        "accuracy_main": estimate,
        "strata": {s: neyman_need(labelled, s, margin, deff) for s in schemes} if labelled else {},
        "repeat_agreement": kappa(pairs),
        "judge_vs_person": {
            "edges": len(judged),
            "agreement_supported_vs_correct": round(sum((r["judge"] == "supported") == (r["y"] == 1) for r in judged)
                                                    / len(judged), 4) if judged else None,
            "by_verdict": {v: {"edges": len(m), "person_correct": round(sum(r["y"] for r in m) / len(m), 4)}
                           for v, m in ((v, [r for r in judged if r["judge"] == v])
                                        for v in ("supported", "partial", "not_supported")) if m},
        },
        "cost": minutes(sample),
        "margin": margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--out", type=Path, help="JSON file for the result")
    args = parser.parse_args()
    result = analyse(args.sample, args.verdicts, args.margin)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
