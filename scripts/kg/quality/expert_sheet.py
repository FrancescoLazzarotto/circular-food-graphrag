#!/usr/bin/env python3
"""One review sheet for the expert, across the quality dimensions.

The expert's time is the scarcest, so the sheet holds only what the cheaper
readers could not settle, and never what they decided: the judge's verdict
and the developer's reading go to a key the expert does not see.

- answers (coverage): every answer the judge found only partly supported by
  its evidence, and a random few it found supported, to calibrate it;
- mappings (interoperability): every candidate mapping the developer
  rejected, and a random few it kept;
- edges (accuracy, conformity), when the person's labels are given: the
  edges where the person and the judge disagree, and those the person could
  not judge;
- the rows of an earlier sheet still undecided.

Writes ``sheet.tsv`` (the columns of ``ontology/review/sheet_template.tsv``
plus the minutes), ``items.md`` (question and answer of each answer row, the
passage of each edge row) and ``key.tsv`` (where each row comes from and what
the cheaper readers said; not for the expert).

Usage:
    python scripts/kg/quality/expert_sheet.py --answers <coverage>/answers.jsonl \\
        --grounded <coverage>/judge_gemma4_31b/grounded.jsonl --mappings ontology/external/mappings.tsv \\
        --previous ontology/review/expert_sheet_2026-10-16.tsv --out <dir> \\
        [--sample <reading/sample_v1> --verdicts <judge>/verdicts.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

COLUMNS = ["row", "sheet", "id", "proposal", "quote", "document", "page", "reason", "decision", "expert_note", "minutes"]
KEY = ["row", "sheet", "id", "source", "judge", "developer", "person"]


def _tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def answer_rows(answers: Path, grounded: Path, supported: int, rng: random.Random) -> list[tuple[dict, dict, str]]:
    """Answers judged partly supported, plus ``supported`` judged supported, drawn at random."""
    text = {r["question"]: r for r in map(json.loads, answers.open(encoding="utf-8"))}
    verdicts = [json.loads(line) for line in grounded.open(encoding="utf-8")]
    doubtful = [v for v in verdicts if v.get("supported") != "yes" or v.get("answers") != "yes"]
    sure = [v for v in verdicts if v not in doubtful]
    chosen = doubtful + rng.sample(sure, min(supported, len(sure)))
    rng.shuffle(chosen)
    rows = []
    for v in chosen:
        answer = text[v["question"]]
        rows.append((
            {"sheet": "risposta", "id": v["question"][:80],
             "proposal": "La risposta (in items.md) è corretta e utile per chi ha fatto la domanda?",
             "reason": "taratura del giudice delle risposte"},
            {"source": "coverage", "judge": f"answers={v.get('answers')} supported={v.get('supported')}"},
            f"**Domanda:** {v['question']}\n\n**Risposta:**\n\n{answer['answer']}",
        ))
    return rows


def mapping_rows(mappings: Path, kept: int, rng: random.Random) -> list[tuple[dict, dict, str]]:
    """Mappings the developer rejected, plus ``kept`` it kept, drawn at random."""
    rows = _tsv(mappings)
    rejected = [r for r in rows if r["status"] == "rejected"]
    others = [r for r in rows if r["status"] in ("checked", "approved", "corrected")]
    chosen = rejected + rng.sample(others, min(kept, len(others)))
    rng.shuffle(chosen)
    return [
        ({"sheet": "corrispondenza", "id": f"{r['concept_id']} ~ {r['vocabulary']}",
          "proposal": f"Il concetto {r['concept_id']} corrisponde a «{r['label']}» di {r['vocabulary']} ({r['uri']})? "
                      "Scrivi uguale, simile, più generale, più specifico oppure no.",
          "reason": "allineamento a un vocabolario di riferimento"},
         {"source": "mappings", "developer": f"{r['status']} {r['match_type']}"}, "")
        for r in chosen
    ]


def edge_rows(sample: Path, verdicts: Path, limit: int, rng: random.Random) -> list[tuple[dict, dict, str]]:
    """Edges where the person and the judge disagree, and those the person could not judge."""
    judge = {}
    for line in verdicts.open(encoding="utf-8"):
        for edge in json.loads(line).get("edges", []):
            judge[edge["edge_key"]] = edge.get("verdict", "")
    passages = (sample / "passages.md").read_text(encoding="utf-8")
    chosen = []
    for r in _tsv(sample / "sheet.tsv"):
        label, verdict = r["label"].strip(), judge.get(r["edge_key"], "")
        if not label:
            continue
        disagree = (label == "corretto" and verdict == "not_supported") or (label == "falso" and verdict == "supported")
        if disagree or label == "non giudicabile":
            chosen.append((r, verdict))
    rng.shuffle(chosen)
    rows = []
    for r, verdict in chosen[:limit]:
        start = passages.find(f"## Passaggio {r['passage']} (")
        excerpt = passages[start: passages.find("\n## ", start + 1)] if start >= 0 else ""
        rows.append((
            {"sheet": "arco", "id": r["edge_key"],
             "proposal": f"Il passaggio {r['passage']} (in items.md) afferma: {r['subject']} — {r['relation']} → {r['object']}?",
             "reason": "arco su cui i lettori non concordano"},
            {"source": "reading", "judge": verdict, "person": r["label"]},
            excerpt.strip(),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--grounded", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--previous", type=Path, help="earlier expert sheet whose undecided rows are carried over")
    parser.add_argument("--sample", type=Path, help="reading sample with the person's labels")
    parser.add_argument("--verdicts", type=Path, help="judge verdicts of the edges")
    parser.add_argument("--supported", type=int, default=5, help="answers judged supported to add")
    parser.add_argument("--kept", type=int, default=4, help="kept mappings to add")
    parser.add_argument("--edges", type=int, default=10, help="most edge rows")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(42)
    rows = answer_rows(args.answers, args.grounded, args.supported, rng)
    rows += mapping_rows(args.mappings, args.kept, rng)
    if args.sample and args.verdicts:
        rows += edge_rows(args.sample, args.verdicts, args.edges, rng)
    if args.previous:
        rows += [({k: r.get(k, "") for k in COLUMNS if k not in ("row", "decision", "expert_note", "minutes")},
                  {"source": args.previous.name}, "")
                 for r in _tsv(args.previous) if not r.get("decision", "").strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "sheet.tsv").open("w", encoding="utf-8", newline="") as sheet, \
            (args.out / "key.tsv").open("w", encoding="utf-8", newline="") as key:
        s = csv.DictWriter(sheet, COLUMNS, delimiter="\t", extrasaction="ignore")
        k = csv.DictWriter(key, KEY, delimiter="\t", extrasaction="ignore")
        s.writeheader()
        k.writeheader()
        items = ["# Righe da leggere per esteso", ""]
        for number, (row, hidden, text) in enumerate(rows, 1):
            s.writerow({**row, "row": number})
            k.writerow({**hidden, "row": number, "sheet": row["sheet"], "id": row["id"]})
            if text:
                items += [f"## Riga {number}", "", text, ""]
    (args.out / "items.md").write_text("\n".join(items) + "\n", encoding="utf-8")
    counts = {}
    for row, _, _ in rows:
        counts[row["sheet"]] = counts.get(row["sheet"], 0) + 1
    print(json.dumps({"rows": len(rows), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
