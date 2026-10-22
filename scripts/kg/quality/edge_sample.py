#!/usr/bin/env python3
"""Draw the edges a person reads to estimate the accuracy of the graph.

Two-stage cluster sample, the passage being the cluster: an edge is read
with the passage it was extracted from, and every sampled edge of that
passage is judged in the same reading.

1. Passages are drawn with probability proportional to their number of
   edges (systematic sampling on a random order).
2. In each passage, at most ``--per-passage`` edges are drawn at random.
   With passages larger than that, every edge of the main sample has the
   same inclusion probability, ``passages * per_passage / edges``.
3. Supplements: ``--supplement name=k`` draws ``k`` more edges at random
   from a rare stratum (``violation``: an edge that breaks a domain or range
   rule; ``class_name_object``: an object named like a schema label),
   excluding edges already drawn. Each is read with its passage.
4. ``--double-share`` of the main passages come back in the last session,
   renumbered, to measure how often the same reading gives the same label.

The reader gets ``sheet.tsv`` (labels and error types to fill),
``passages.md`` (the text of each passage with its numbered edges) and
``times.tsv`` (start and end of each passage). Strata, signals and inclusion
probabilities go to ``design.tsv`` and ``design.json``, which the reader
does not open; no model verdict is in any file the reader sees.

Usage:
    python scripts/kg/quality/edge_sample.py \\
        --signals artifacts/kg_eval_2026-10/signals/<backup>/edge_signals.tsv \\
        --chunks kg_pipeline/artifacts/run_rebuild_20260922_E/stage1_chunks.json \\
        --out artifacts/kg_eval_2026-10/reading/sample_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SHEET = ["session", "passage", "edge", "edge_key", "subject", "relation", "object", "label", "error_type", "note"]
TIMES = ["session", "passage", "start", "end"]
DESIGN = [
    "edge_key", "session", "passage", "frame", "double_of", "chunk_id", "chunk_edges", "pi_passage",
    "pi_edge_in_passage", "pi_main", "pi_supplement", "pi", "type", "predicate", "family", "fallback",
    "fallback_predicate", "concept_end", "class_name_object", "violations", "confidence",
]
STRATA = {
    "violation": lambda row: bool(row["violations"]),
    "class_name_object": lambda row: row["class_name_object"] == "1",
}


def systematic_pps(sizes: dict[str, int], n: int, rng: random.Random) -> list[str]:
    """Draw ``n`` units with probability proportional to size, systematically.

    Args:
        sizes: Unit id → size.
        n: Units to draw; every size must be below ``total / n``.
        rng: Random generator.

    Returns:
        The drawn unit ids, in the random order used.

    Raises:
        ValueError: A unit would be drawn with certainty.
    """
    order = sorted(sizes)
    rng.shuffle(order)
    total = sum(sizes.values())
    step = total / n
    if max(sizes.values()) >= step:
        raise ValueError("a passage is larger than the sampling step: draw fewer passages")
    point, cumulative, drawn = rng.uniform(0, step), 0.0, []
    for unit in order:
        cumulative += sizes[unit]
        while point < cumulative and len(drawn) < n:
            drawn.append(unit)
            point += step
    return drawn


def relation_text(row: dict) -> str:
    """The relation as the reader sees it: the type, and the original predicate of a fallback edge."""
    if row["type"] == "RELATED_TO" and row["predicate"]:
        return f"RELATED_TO (predicato originale: {row['predicate']})"
    return row["type"]


def draw(rows: list[dict], passages: int, per_passage: int, supplements: dict[str, int], double_share: float,
         sessions: int, seed: int) -> list[dict]:
    """The reading plan: one dict per edge reading, with its design fields.

    Args:
        rows: Rows of ``edge_signals.tsv``.
        passages: Passages of the main sample.
        per_passage: Edges at most per passage.
        supplements: Stratum name → edges to add.
        double_share: Share of main passages read again in the last session.
        sessions: Reading sessions the main passages are split into.
        seed: Random seed.

    Returns:
        Readings in sheet order, each with ``session`` and ``passage`` set.
    """
    rng = random.Random(seed)
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_chunk[row["chunk_id"]].append(row)
    total = len(rows)
    sizes = {chunk: len(edges) for chunk, edges in by_chunk.items()}
    groups = []
    for chunk in systematic_pps(sizes, passages, rng):
        edges = sorted(by_chunk[chunk], key=lambda r: r["edge_key"])
        taken = rng.sample(edges, min(per_passage, len(edges)))
        pi_p = passages * sizes[chunk] / total
        pi_e = len(taken) / len(edges)
        groups.append([{**r, "frame": "main", "pi_passage": pi_p, "pi_edge_in_passage": pi_e, "pi_main": pi_p * pi_e}
                       for r in taken])
    drawn = {r["edge_key"] for g in groups for r in g}
    pi_main_of = {}
    for g in groups:
        for r in g:
            pi_main_of[r["edge_key"]] = r["pi_main"]

    supplement_groups = []
    for name, k in supplements.items():
        stratum = [r for r in rows if STRATA[name](r)]
        pool = sorted((r for r in stratum if r["edge_key"] not in drawn), key=lambda r: r["edge_key"])
        for r in rng.sample(pool, min(k, len(pool))):
            drawn.add(r["edge_key"])
            # An edge outside the main sample could still have entered it.
            pi_m = passages * per_passage / total if sizes[r["chunk_id"]] >= per_passage else passages * sizes[r["chunk_id"]] / total
            supplement_groups.append([{**r, "frame": f"supplement:{name}", "pi_passage": "", "pi_edge_in_passage": "",
                                       "pi_main": pi_m, "pi_supplement": k / len(stratum)}])
    for g in groups:
        for r in g:
            r["pi_supplement"] = 0.0
            for name, k in supplements.items():
                if STRATA[name](r):
                    r["pi_supplement"] = k / sum(STRATA[name](x) for x in rows)

    plan: list[list[dict]] = []
    per_session = -(-len(groups) // sessions)
    for s in range(sessions):
        chunk_groups = groups[s * per_session:(s + 1) * per_session]
        extra = supplement_groups[s::sessions]
        mixed = chunk_groups + extra
        rng.shuffle(mixed)
        plan += [[{**r, "session": s + 1} for r in g] for g in mixed]
    repeats = rng.sample(groups, round(double_share * len(groups)))
    plan += [[{**r, "session": sessions, "frame": "double", "double_of": r["edge_key"]} for r in rng.sample(g, len(g))]
             for g in repeats]

    readings = []
    for number, group in enumerate(plan, 1):
        for r in group:
            p_main, p_sup = float(r["pi_main"]), float(r["pi_supplement"])
            readings.append({**r, "passage": number, "double_of": r.get("double_of", ""),
                             "pi": p_main + p_sup - p_main * p_sup})
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True, help="stage1_chunks.json with the passage text")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passages", type=int, default=48)
    parser.add_argument("--per-passage", type=int, default=6)
    parser.add_argument("--supplement", action="append", default=[], help="stratum=edges, e.g. violation=15")
    parser.add_argument("--double-share", type=float, default=0.15)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists: a drawn sample is never redrawn in place")

    with args.signals.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    supplements = {name: int(k) for name, _, k in (s.partition("=") for s in args.supplement)}
    readings = draw(rows, args.passages, args.per_passage, supplements, args.double_share, args.sessions, args.seed)
    text = {c["chunk_id"]: c for c in json.loads(args.chunks.read_text(encoding="utf-8"))}

    args.out.mkdir(parents=True)
    by_passage: dict[int, list[dict]] = defaultdict(list)
    for r in readings:
        by_passage[r["passage"]].append(r)
    with (args.out / "sheet.tsv").open("w", encoding="utf-8", newline="") as sheet, \
            (args.out / "design.tsv").open("w", encoding="utf-8", newline="") as design, \
            (args.out / "times.tsv").open("w", encoding="utf-8", newline="") as times:
        s_out = csv.DictWriter(sheet, SHEET, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        d_out = csv.DictWriter(design, DESIGN, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        t_out = csv.DictWriter(times, TIMES, delimiter="\t", lineterminator="\n")
        for w in (s_out, d_out, t_out):
            w.writeheader()
        pages = ["# Passaggi da leggere", "", "Per ogni passaggio: leggi il testo, poi giudica ogni arco numerato nel foglio `sheet.tsv`.", ""]
        for number, group in sorted(by_passage.items()):
            chunk = text[group[0]["chunk_id"]]
            t_out.writerow({"session": group[0]["session"], "passage": number, "start": "", "end": ""})
            pages += [f"## Passaggio {number} (seduta {group[0]['session']})", "",
                      f"{chunk['filename']}, pagine {chunk['page_range']}", "", "```text", chunk["text"].strip(), "```", ""]
            for edge_no, r in enumerate(group, 1):
                relation = relation_text(r)
                s_out.writerow({**r, "edge": edge_no, "relation": relation, "label": "", "error_type": "", "note": ""})
                d_out.writerow(r)
                pages.append(f"{edge_no}. **{r['subject']}** — {relation} → **{r['object']}**")
            pages.append("")
        (args.out / "passages.md").write_text("\n".join(pages), encoding="utf-8")

    main_rows = [r for r in readings if r["frame"] == "main"]
    summary = {
        "seed": args.seed, "passages": args.passages, "per_passage": args.per_passage, "supplements": supplements,
        "double_share": args.double_share, "sessions": args.sessions,
        "signals": str(args.signals), "signals_sha256": hashlib.sha256(args.signals.read_bytes()).hexdigest(),
        "chunks": str(args.chunks), "edges_in_graph": len(rows),
        "readings": len(readings), "main_edges": len(main_rows),
        "supplement_edges": sum(r["frame"].startswith("supplement") for r in readings),
        "double_edges": sum(r["frame"] == "double" for r in readings),
        "passages_to_read": len(by_passage),
        "by_session": {s: sum(1 for g in by_passage.values() if g[0]["session"] == s) for s in range(1, args.sessions + 1)},
    }
    (args.out / "design.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
