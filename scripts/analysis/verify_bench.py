#!/usr/bin/env python3
"""Measure the ontology check that runs after generation on a fixed bench.

The bench (``tests/fixtures/ontology_verify_bench.json``) holds answers in the
formats the demo writes (R codes before or after the name, lists inside a
sentence, bullet and numbered lists, citations in square brackets), in Italian
and English. Each case is either correct or carries one seeded error: wrong
version, missing item, intruder, wrong family, wrong order.

The check is the agent's own: the question goes through ``recognise`` and the
answer through ``check_answer``, as in the "verify" node. A case counts as
flagged when a list was judged and found different.

- errors found: flagged cases among those with a seeded error;
- false alarms: flagged cases among the correct ones;
- not judged: cases where no list was judged at all.

With ``--answers`` it also replays saved answers (the ``answers.jsonl`` of
``measure_ontology.py``), after removing the note the agent had already
appended, and lists what each judged list was flagged for. Those answers
have no ground truth: the output is for reading.

Usage:
    python scripts/analysis/verify_bench.py --out artifacts/verify_bench/$(date +%Y%m%d_%H%M)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from graphrag.ontology import hooks, load  # noqa: E402

BENCH = ROOT / "tests" / "fixtures" / "ontology_verify_bench.json"
# The note "verify" appends; a replayed answer must not be judged with it.
NOTE_MARKERS = ("\n\nControllo sull'elenco", "\n\nList check")


def judge(question: str, answer: str, onto) -> tuple[dict, float]:
    """Run the agent's check on one answer.

    Args:
        question: The question, as the agent would see it.
        answer: The generated answer.
        onto: The loaded ontology.

    Returns:
        The report of ``check_answer`` and the time it took in milliseconds.
    """
    start = time.perf_counter()
    match = hooks.recognise(question, onto)
    report = hooks.check_answer(answer, match, onto) if match.frameworks else {"checked": False, "ok": True}
    return report, (time.perf_counter() - start) * 1000


def score_bench(cases: list[dict], onto) -> dict:
    """Errors found, false alarms and unjudged cases, overall and by group.

    Args:
        cases: Bench cases.
        onto: The loaded ontology.

    Returns:
        Counts with their denominators, per-case outcomes and check latency.
    """
    rows, times = [], []
    for case in cases:
        report, ms = judge(case["question"], case["answer"], onto)
        times.append(ms)
        flagged = report["checked"] and not report["ok"]
        rows.append(
            {
                "id": case["id"],
                "lang": case["lang"],
                "format": case["format"],
                "expected": case["expected"],
                "error_type": case["error_type"],
                "checked": report["checked"],
                "flagged": flagged,
                "missing": report.get("missing", []),
                "extra": report.get("extra", []),
                "family": report.get("family", ""),
            }
        )

    def counts(selected: list[dict]) -> dict:
        errors = [r for r in selected if r["expected"] == "error"]
        correct = [r for r in selected if r["expected"] == "ok"]
        return {
            "errors_found": sum(r["flagged"] for r in errors),
            "errors": len(errors),
            "false_alarms": sum(r["flagged"] for r in correct),
            "correct": len(correct),
            "not_judged": sum(not r["checked"] for r in selected),
            "cases": len(selected),
        }

    groups: dict[str, dict] = {}
    for key in ("lang", "format", "error_type"):
        values = defaultdict(list)
        for r in rows:
            values[r[key]].append(r)
        groups[key] = {value: counts(selected) for value, selected in sorted(values.items())}
    times.sort()
    return {
        "overall": counts(rows),
        "by": groups,
        "check_ms_median": round(times[len(times) // 2], 3),
        "check_ms_max": round(times[-1], 3),
        "cases": rows,
    }


def replay(path: Path, onto) -> list[dict]:
    """Judge saved answers again with the current code.

    Args:
        path: An ``answers.jsonl`` written by ``measure_ontology.py``.
        onto: The loaded ontology.

    Returns:
        One row per answer: id, arm, whether a list was judged, and what it
        was flagged for.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        saved = json.loads(line)
        answer = saved["answer"]
        for marker in NOTE_MARKERS:
            answer = answer.split(marker)[0]
        report, _ = judge(saved["question"], answer, onto)
        rows.append(
            {
                "id": saved["id"],
                "arm": saved.get("arm", ""),
                "checked": report["checked"],
                "flagged": report["checked"] and not report["ok"],
                "missing": report.get("missing", []),
                "extra": report.get("extra", []),
                "family": report.get("family", ""),
            }
        )
    return rows


def _line(name: str, c: dict) -> str:
    return (
        f"| {name} | {c['errors_found']}/{c['errors']} | {c['false_alarms']}/{c['correct']} | "
        f"{c['not_judged']}/{c['cases']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--answers", type=Path, help="answers.jsonl of measure_ontology.py to replay")
    parser.add_argument("--out", type=Path, help="folder for summary.json and report.md")
    args = parser.parse_args()

    onto = load()
    cases = json.loads(args.bench.read_text(encoding="utf-8"))["cases"]
    result = score_bench(cases, onto)
    lines = [
        "| group | errors found | false alarms | not judged |",
        "| --- | --- | --- | --- |",
        _line("all", result["overall"]),
    ]
    for key, values in result["by"].items():
        lines += [_line(f"{key}={value}", c) for value, c in values.items()]
    lines.append("")
    lines.append(f"check latency: median {result['check_ms_median']} ms, max {result['check_ms_max']} ms")
    if args.answers:
        replayed = replay(args.answers, onto)
        result["replay"] = replayed
        judged = [r for r in replayed if r["checked"]]
        lines += ["", f"replay {args.answers}: judged {len(judged)}/{len(replayed)}, flagged {sum(r['flagged'] for r in judged)}"]
        lines += [f"- {r['id']} {r['arm']}: family={r['family']!r} missing={r['missing']} extra={r['extra']}" for r in judged]
    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        (args.out / "report.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
