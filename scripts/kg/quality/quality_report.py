#!/usr/bin/env python3
"""Quality of the graph and of the framework ontology, one row per measure.

Reads only the newest nightly JSON backup of the graph and the ontology
sources and build; never the live database. Every row is one measurement in
the shape of the W3C Data Quality Vocabulary: dimension, metric, value with
numerator and denominator, what it was computed on, when, and from what.
The four dimensions are reported apart and are never combined into one score.

- conformity: domain and range violations, exposure to the rules, fallback
  edges and default labels on the graph; Python checks, SHACL and Skosify on
  the ontology (``ontology/scripts/build.py --check-only``, in the ontology
  virtualenv, which writes nothing);
- interoperability: mappings to external vocabularies in the SKOS file and in
  ``ontology/external/mappings.tsv``, resources with an identifier of their
  own, graph nodes linked to an external identifier;
- coverage: competency questions with a passing test (``tests/test_ontology.py``)
  and, with ``--questions``, real questions the ontology recognises;
- accuracy: the last estimate given with ``--accuracy``, with its interval and
  date, or an empty row saying there is none.

Writes ``measurements.tsv``, ``measurements.json`` and ``report.md`` under
``--out/<backup>``, and appends the rows to ``--out/history.tsv``.

Usage:
    python scripts/kg/quality/quality_report.py --out artifacts/kg_eval_2026-10/nightly \\
        --questions analysis/assistente_2026-10/sessioni_domande.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import edge_signals  # noqa: E402

ONTOLOGY_PYTHON = Path("/mnt/storage/flazzarotto/venvs/ontology/bin/python")
CQ_SHEET = ROOT / "ontology" / "competency_questions" / "competency_questions.tsv"
MAPPINGS = ROOT / "ontology" / "external" / "mappings.tsv"
ONTOLOGY_JSON = ROOT / "ontology" / "build" / "ontology.json"
COLUMNS = ["dimension", "metric", "value", "numerator", "denominator", "computed_on", "date", "source", "note"]


class Report:
    """Measurements collected for one run.

    Attributes:
        rows: One dict per measurement, keys as in ``COLUMNS``.
        date: Timestamp written on every row (UTC, ISO 8601).
    """

    def __init__(self, date: str) -> None:
        self.rows: list[dict] = []
        self.date = date

    def add(self, dimension: str, metric: str, numerator: float | None, denominator: float | None,
            computed_on: str, source: str, note: str = "") -> None:
        """Record one measurement; the value is the share when there is a denominator, else the count."""
        if numerator is None:
            value = ""
        elif denominator:
            value = round(numerator / denominator, 4)
        else:
            value = numerator
        self.rows.append({
            "dimension": dimension, "metric": metric, "value": value,
            "numerator": "" if numerator is None else numerator,
            "denominator": "" if denominator is None else denominator,
            "computed_on": computed_on, "date": self.date, "source": source, "note": note,
        })


def graph_measures(report: Report, backup: Path) -> None:
    """Conformity and interoperability of the graph, from its backup.

    Args:
        report: Where the rows go.
        backup: Backup folder with ``nodes.json`` and ``edges.json``.
    """
    nodes = json.loads((backup / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((backup / "edges.json").read_text(encoding="utf-8"))
    vocab = set(json.loads(edge_signals.VOCAB.read_text(encoding="utf-8")))
    rows = edge_signals.signals(nodes, edges, vocab, edge_signals.PROMPT_RULES)
    s = edge_signals.summary(rows, edge_signals.PROMPT_RULES, backup)
    on = f"graph backup {backup.name}"
    src = "scripts/kg/quality/edge_signals.py"
    total = s["edges"]
    by_type = Counter(r["type"] for r in rows)
    for rule in edge_signals.PROMPT_RULES:
        report.add("conformity", f"violation_rate:{rule.rule}", s["violations"][rule.rule], by_type[rule.type], on, src, rule.source)
    report.add("conformity", "violation_rate:any_rule", s["edges_with_violation"], s["exposed_to_rules"], on, src,
               "denominator: edges of a type some rule constrains")
    report.add("conformity", "exposure", s["exposed_to_rules"], total, on, src, "edges at least one rule can fail on")
    report.add("conformity", "fallback_related_to", s["fallback"], total, on, src)
    report.add("conformity", "fallback_without_predicate", s["fallback"] - s["fallback_with_predicate"], s["fallback"], on, src)
    report.add("conformity", "edge_with_default_label_end", total - s["concept_end"].get("none", 0), total, on, src,
               "an end labelled Concept, the extraction default")
    report.add("conformity", "object_named_as_class", s["class_name_object"], total, on, src)
    report.add("conformity", "out_of_vocabulary_type", s["out_of_vocab"], total, on, src)
    linked = sum(1 for n in nodes if n["props"].get("ontology_uri"))
    report.add("interoperability", "graph_nodes_with_external_identifier", linked, len(nodes), on, "nodes.json: ontology_uri")


def ontology_measures(report: Report, python: Path) -> None:
    """Conformity and interoperability of the ontology.

    Args:
        report: Where the rows go.
        python: Interpreter of the ontology virtualenv.
    """
    on = "ontology sources " + dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in (ROOT / "ontology" / "sources").glob("*.tsv")), dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    src = "ontology/scripts/build.py --check-only"
    done = subprocess.run([str(python), str(ROOT / "ontology" / "scripts" / "build.py"), "--check-only"],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    try:
        checks = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        report.add("conformity", "ontology_build_checks_ran", 0, None, on, src, (done.stderr or done.stdout)[-300:])
        return
    graph = checks["graph"]
    for title, count in checks["problems"].items():
        report.add("conformity", f"ontology_problems:{title}", count, None, on, src)
    report.add("interoperability", "ontology_external_mappings_in_skos", graph["external_mappings"], graph["concepts"], on, src,
               "skos:*Match triples per SKOS concept")
    report.add("interoperability", "ontology_named_resources", graph["subjects"] - graph["blank_subjects"], graph["subjects"], on, src,
               "subjects with an IRI of their own; the rest are blank nodes")

    with MAPPINGS.open(encoding="utf-8", newline="") as handle:
        mappings = list(csv.DictReader(handle, delimiter="\t"))
    checked = [m for m in mappings if m["status"] in ("checked", "approved", "corrected")]
    report.add("interoperability", "mapping_candidates_kept", len(checked), len(mappings), on, "ontology/external/mappings.tsv",
               "read against the vocabulary entry; precision of the candidates")
    concepts = json.loads(ONTOLOGY_JSON.read_text(encoding="utf-8"))["concepts"]
    usable = {cid for cid, c in concepts.items() if c["status"] in ("auto_accepted", "checked", "approved", "corrected")}
    with_mapping = {m["concept_id"] for m in checked} & usable
    report.add("interoperability", "concepts_with_kept_mapping", len(with_mapping), len(usable), on,
               "ontology/external/mappings.tsv + ontology/build/ontology.json")


def coverage_measures(report: Report, questions: Path | None) -> None:
    """Competency questions passed, and real questions recognised.

    Args:
        report: Where the rows go.
        questions: The session questions file, or None to skip that row.
    """
    with CQ_SHEET.open(encoding="utf-8", newline="") as handle:
        cq_ids = [row["id"] for row in csv.DictReader(handle, delimiter="\t")]
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_ontology.py"),
                        f"--junitxml={junit}"], cwd=ROOT, capture_output=True, text=True, check=False)
        cases = ET.parse(junit).getroot().iter("testcase") if junit.exists() else []
        outcome: dict[str, list[bool]] = {}
        tests = passed = 0
        for case in cases:
            ok = not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
            tests += 1
            passed += ok
            name = case.get("name", "")
            for cq in cq_ids:
                if f"[{cq}]" in name:
                    outcome.setdefault(cq, []).append(ok)
    on = "ontology/build/ontology.json"
    src = "tests/test_ontology.py"
    report.add("coverage", "competency_questions_with_test", len(outcome), len(cq_ids), on, src,
               "a test parametrised by the question id")
    report.add("coverage", "competency_questions_passed", sum(all(v) for v in outcome.values()), len(outcome), on, src)
    report.add("coverage", "ontology_tests_passed", passed, tests, on, src, "strict xfail = a known gap")
    if questions is None:
        return
    from graphrag.ontology import hooks, load

    onto = load(ONTOLOGY_JSON)
    # Case and spacing do not make a question different.
    asked = [" ".join(row["q"].lower().split()) for row in json.loads(questions.read_text(encoding="utf-8")) if row.get("q")]
    recognised = {q: bool(hooks.recognise(q, onto)) for q in set(asked)}
    on_q = f"{questions.name} ({len(asked)} questions)"
    report.add("coverage", "real_questions_recognised_distinct", sum(recognised.values()), len(recognised), on_q,
               "graphrag.ontology.hooks.recognise", "schema level only: the ontology names something the question asks about")
    report.add("coverage", "real_questions_recognised_weighted", sum(recognised[q] for q in asked), len(asked), on_q,
               "graphrag.ontology.hooks.recognise")


def accuracy_measures(report: Report, accuracy: Path | None) -> None:
    """The last accuracy estimate, or a row saying there is none.

    Args:
        report: Where the rows go.
        accuracy: JSON with ``correct``, ``judged``, ``low``, ``high``,
            ``date``, ``method``, ``computed_on``; or None.
    """
    if accuracy is None or not accuracy.exists():
        report.add("accuracy", "edge_accuracy", None, None, "", "", "no estimate with a recorded sample yet")
        return
    est = json.loads(accuracy.read_text(encoding="utf-8"))
    report.add("accuracy", "edge_accuracy", est["correct"], est["judged"], est.get("computed_on", ""), str(accuracy),
               f"interval {est['low']}-{est['high']} ({est.get('method', '')}), estimated {est.get('date', '')}")


def write(report: Report, out: Path, backup: str) -> Path:
    """Write the run's files and append to the history.

    Args:
        report: The measurements.
        out: Root output folder.
        backup: Backup name, used as the run folder.

    Returns:
        The run folder.
    """
    run = out / backup
    run.mkdir(parents=True, exist_ok=True)
    with (run / "measurements.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(report.rows)
    (run / "measurements.json").write_text(json.dumps(report.rows, ensure_ascii=False, indent=1), encoding="utf-8")
    history = out / "history.tsv"
    new = not history.exists()
    with history.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows(report.rows)
    lines = [f"# Qualità del grafo e dell'ontologia — {report.date}", "",
             "Una riga per misura; le quattro dimensioni restano separate.", ""]
    for dimension in ("conformity", "interoperability", "coverage", "accuracy"):
        lines += [f"## {dimension}", "", "| metric | value | numerator | denominator | computed on | note |", "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {r['metric']} | {r['value']} | {r['numerator']} | {r['denominator']} | {r['computed_on']} | {r['note']} |"
                  for r in report.rows if r["dimension"] == dimension]
        lines.append("")
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="root folder; one subfolder per backup")
    parser.add_argument("--backup", type=Path, help="backup folder (default: the newest)")
    parser.add_argument("--questions", type=Path, help="session questions (sessioni_domande.json)")
    parser.add_argument("--accuracy", type=Path, help="JSON of the last accuracy estimate")
    parser.add_argument("--ontology-python", type=Path, default=ONTOLOGY_PYTHON)
    args = parser.parse_args()

    backup = args.backup or edge_signals.latest_backup()
    report = Report(dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    graph_measures(report, backup)
    ontology_measures(report, args.ontology_python)
    coverage_measures(report, args.questions)
    accuracy_measures(report, args.accuracy)
    run = write(report, args.out, backup.name)
    print((run / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
