#!/usr/bin/env python3
"""Answer the same questions with two served generators and lay them side by side.

Some qualities of an answer -- stilted Italian, grammar slips, a dated chatbot
tone -- belong to the *generator*, not to the retrieval pipeline, and the
choice of generator is not settled by measurement.

This script isolates that variable. Both arms are built with
``product.config.build_demo_agent``, so every other setting -- strategy, graph,
citation policy, complexity, token budget -- is byte-identical to what the demo
ships. The only difference between the arms is which vLLM endpoint answers.

It is deliberately *not* a scorer. There is no gold for the questions the expert
actually asks, and inventing one would measure the invention. What comes out is
a JSONL per arm (machine-readable, for later analysis) and a side-by-side
markdown that a domain expert can read and mark a preference on. Those marks are
the preference data any later DPO run would need.

Usage:
    python scripts/analysis/ab_generators.py \
        --questions evaluation/fixtures/questions_demo_it_expert.txt \
        --endpoints http://localhost:8000/v1,http://localhost:8001/v1 \
        --out artifacts/ab_generators/$(date +%Y%m%d_%H%M)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _model_id_at(base_url: str, timeout: float = 5.0) -> str:
    """Ask an endpoint what it serves, so the arms are named by the real model."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout) as r:
        return json.load(r)["data"][0]["id"]


def _read_questions(path: Path, limit: int | None) -> list[str]:
    """The non-empty, non-comment lines of a question file, up to `limit`."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return lines[:limit] if limit else lines


def _run_arm(base_url: str, questions: list[str], strategy: str) -> tuple[str, list[dict]]:
    """Answer every question with the generator served at `base_url`.

    Args:
        base_url: vLLM endpoint of this arm.
        questions: The questions, answered one by one without memory.
        strategy: Retrieval-strategy preset for the demo agent.

    Returns:
        The served model id and one row per question; a failed question is a
        row with ``error`` set, not an exception.
    """
    from product.config import build_demo_agent

    model_id = _model_id_at(base_url)
    # One agent for the whole arm: rebuilding it per question would reconnect to
    # the graph and reload the text pipeline every time for no benefit.
    agent, graph_label = build_demo_agent(base_url, model_id, strategy=strategy)
    print(f"[{model_id}] graph={graph_label} strategy={strategy}", file=sys.stderr)

    rows: list[dict] = []
    for index, question in enumerate(questions, start=1):
        start = time.perf_counter()
        try:
            # memory=None: each question stands alone. The fixture is a flat
            # list drawn from several sessions, so carrying memory across it
            # would let one arm's answer contaminate the next question.
            result = agent.invoke(question)
            answer = str(result.get("answer", "")).strip()
            error = None
        except Exception as exc:  # one bad question must not lose the arm
            result, answer, error = {}, "", f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "i": index,
                "question": question,
                "model_id": model_id,
                "strategy": strategy,
                "answer": answer,
                "error": error,
                "latency_s": round(elapsed, 2),
                "citation_report": result.get("citation_report"),
                "insufficient_answer": result.get("insufficient_answer"),
            }
        )
        mark = "!" if error else " "
        print(f"  {mark}{index:3d}/{len(questions)} {elapsed:6.1f}s  {question[:60]}", file=sys.stderr)
    return model_id, rows


def _summarise(arms: dict[str, list[dict]]) -> str:
    """Aggregate the two arms on the numbers that decide this comparison.

    ``phantom_rate`` is the direct hallucination measure: a reference the model
    emitted that points at no evidence actually placed in its context. Citation
    density and distinct sources answer the other two standing complaints --
    "corretto ma generico" and "diversificare le fonti" -- which is why they sit
    beside it rather than in a separate report.
    """
    lines = [
        "| metrica | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]

    def _col(fn) -> list[str]:
        """One cell per arm, computed by `fn` over that arm's rows."""
        return [fn(rows) for rows in arms.values()]

    def _mean(values: list[float]) -> float:
        """Mean of `values`, 0 when empty."""
        return sum(values) / len(values) if values else 0.0

    def _reports(rows: list[dict]) -> list[dict]:
        """The citation reports of the rows that have one."""
        return [r["citation_report"] for r in rows if r.get("citation_report")]

    rows_of = list(arms.values())
    metrics = [
        ("risposte", lambda rs: str(len(rs))),
        ("errori", lambda rs: str(sum(1 for r in rs if r["error"]))),
        ("latenza media (s)", lambda rs: f"{_mean([r['latency_s'] for r in rs]):.1f}"),
        ("caratteri medi", lambda rs: f"{_mean([len(r['answer']) for r in rs]):.0f}"),
        ("citazioni medie", lambda rs: f"{_mean([c['total_citations'] for c in _reports(rs)]):.1f}"),
        ("fonti distinte medie", lambda rs: f"{_mean([len(c['cited_refs']) for c in _reports(rs)]):.1f}"),
        ("phantom_rate medio", lambda rs: f"{_mean([c['phantom_rate'] for c in _reports(rs)]):.3f}"),
        (
            "risposte con >=1 fantasma",
            lambda rs: str(sum(1 for c in _reports(rs) if c["phantom_refs"])),
        ),
        (
            "risposte senza citazioni",
            lambda rs: str(sum(1 for c in _reports(rs) if not c["total_citations"])),
        ),
    ]
    for name, fn in metrics:
        lines.append(f"| {name} | " + " | ".join(fn(rs) for rs in rows_of) + " |")
    return "\n".join(lines)


def _write_side_by_side(path: Path, arms: dict[str, list[dict]]) -> None:
    """Render the arms as one markdown the expert can read and mark up."""
    names = list(arms)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Confronto generatori\n\n")
        fh.write("Stessa pipeline, stesso grafo, stessa strategia. Cambia solo il modello.\n\n")
        fh.write("Per ogni domanda, segna quale risposta preferisci e perche'.\n\n")
        for name in names:
            fh.write(f"- **{name}**\n")
        fh.write("\n" + _summarise(arms) + "\n\n---\n\n")
        for index in range(len(arms[names[0]])):
            question = arms[names[0]][index]["question"]
            fh.write(f"## {index + 1}. {question}\n\n")
            for name in names:
                row = arms[name][index]
                body = row["error"] or row["answer"] or "_(vuota)_"
                fh.write(f"### {name}  ({row['latency_s']}s)\n\n{body}\n\n")
            fh.write("**Preferisco:** ______   **Perche':** ______\n\n---\n\n")


def main() -> None:
    """Run every arm and write its JSONL plus the side-by-side markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument(
        "--endpoints",
        default="http://localhost:8000/v1,http://localhost:8001/v1",
        help="Comma-separated vLLM base URLs, one arm each",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--strategy", default=None, help="Default: the demo's own")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help="Re-read the JSONL already in --out and rewrite confronto.md",
    )
    args = parser.parse_args()

    if args.summarise_only:
        arms = {}
        for path in sorted(args.out.glob("*.jsonl")):
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
            if rows:
                arms[rows[0]["model_id"]] = rows
        if len(arms) < 2:
            parser.error(f"need two arms of JSONL in {args.out}")
        _write_side_by_side(args.out / "confronto.md", arms)
        print(_summarise(arms))
        return

    from product.config import STRATEGY

    strategy = args.strategy or STRATEGY
    questions = _read_questions(args.questions, args.limit)
    endpoints = [u.strip() for u in args.endpoints.split(",") if u.strip()]
    if len(endpoints) < 2:
        parser.error("two endpoints or more are needed for a comparison")

    args.out.mkdir(parents=True, exist_ok=True)
    arms: dict[str, list[dict]] = {}
    for base_url in endpoints:
        model_id, rows = _run_arm(base_url, questions, strategy)
        arms[model_id] = rows
        target = args.out / f"{model_id.replace('/', '__')}.jsonl"
        with target.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {target}", file=sys.stderr)

    _write_side_by_side(args.out / "confronto.md", arms)
    print(f"wrote {args.out / 'confronto.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
