#!/usr/bin/env python3
"""Replay labelled follow-up questions through the demo's conversation memory.

The memory has one job: turn a follow-up that depends on the previous turn
into a question retrieval can answer on its own, and leave every other
question alone. Whether it does that is a judgement per question, so the
bench is a file of labelled cases, not a gold set:

* each case is a recorded conversation (questions and the answers actually
  given) followed by the question under test;
* its label says whether that question must stay as typed (``keep``) or must
  become self-contained (``resolve``), which terms the rewrite must contain,
  which words of the user must survive, and which terms must not appear.

For every case the recorded turns are fed to a fresh ``ConversationMemory``
the way the demo feeds it (a turn's entities come from retrieving the question
that turn was retrieved with, on today's graph), then the question under test
goes through ``KGRAGAgent.invoke`` with the real model and the real graph —
domain gate, follow-up rewrite, retrieval and relevance loop — with generation
switched off. Nothing is written to the demo logs.

Retrieval is also measured against a reference: the hand-written standalone
form of the question for ``resolve`` cases, the question itself for ``keep``
cases, asked without memory. The overlap of the documents behind the first
passages shows what a rewrite does to the evidence, not only to the wording.

Usage:
    python scripts/analysis/replay_memory.py \\
        --cases artifacts/memory_bench/cases.json \\
        --out artifacts/memory_bench/runs/<name> \\
        [--baseline artifacts/memory_bench/runs/<other>]

    # labels edited, nothing re-run:
    python scripts/analysis/replay_memory.py --cases ... --out ... --rescore
"""

from __future__ import annotations

import argparse
import copy
import difflib
import inspect
import json
import logging
import re
import statistics
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger("graphrag")

# How many retrieved passages count as "what the answer is built from": the
# demo's text channel returns eight.
_TOP_PASSAGES = 8
# Shorter words carry no subject ("come", "cosa") once stopwords are gone.
_MIN_CONTENT_CHARS = 4
# Two spellings of one word ("sula" and "sulla", "approfondirlo" and
# "approfondire") are the same word for the added-words check.
_SAME_WORD_RATIO = 0.8


# ---------------------------------------------------------------------- #
# text matching
# ---------------------------------------------------------------------- #


def _fold(text: str) -> str:
    """Lowercase, strip accents, and reduce every non-word run to one space."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[\W_]+", " ", plain.lower()).split())


def _has_term(text: str, term: str) -> bool:
    """Whether ``term`` occurs in ``text`` as whole words.

    A trailing ``*`` matches any ending ("zootecn*" matches "zootecnica" and
    "zootecnici"). Accents, case and punctuation are ignored.
    """
    prefix = term.endswith("*")
    folded = _fold(term.rstrip("*"))
    if not folded:
        return False
    tail = r"\w*" if prefix else ""
    return re.search(rf"(?<!\w){re.escape(folded)}{tail}(?!\w)", _fold(text)) is not None


def _content_words(text: str) -> list[str]:
    """Words of ``text`` that can carry a subject."""
    from graphrag.agent.core import _GATE_EXTRA_STOPWORDS, _STOPWORDS

    stop = {_fold(word) for word in (*_STOPWORDS, *_GATE_EXTRA_STOPWORDS)}
    return [
        word
        for word in _fold(text).split()
        if len(word) >= _MIN_CONTENT_CHARS and word not in stop
    ]


def _same_word(word: str, others: list[str]) -> bool:
    """Whether ``word`` is a spelling of one of ``others``."""
    return any(
        difflib.SequenceMatcher(None, word, other).ratio() >= _SAME_WORD_RATIO
        for other in others
    )


def _added_words(question: str, rewrite: str) -> list[str]:
    """Content words of ``rewrite`` that are no spelling of a word typed."""
    typed = _content_words(question)
    added: list[str] = []
    for word in _content_words(rewrite):
        if word not in added and not _same_word(word, typed):
            added.append(word)
    return added


# ---------------------------------------------------------------------- #
# scoring
# ---------------------------------------------------------------------- #


def _documents(sources: list[str]) -> list[str]:
    """Distinct documents behind passage sources, in rank order."""
    seen: list[str] = []
    for source in sources:
        document = str(source).split("#page=")[0].split("#chunk=")[0].strip()
        if document and document not in seen:
            seen.append(document)
    return seen


def _previous_question(history: list[dict[str, Any]]) -> str:
    """The last question the memory observed in ``history``."""
    asked = [turn for turn in history if not turn.get("deepens")]
    return str(asked[-1]["question"]) if asked else ""


def score_case(
    case: dict[str, Any], row: dict[str, Any], previous_question: str = ""
) -> dict[str, Any]:
    """Judge one replayed question against its label.

    A question labelled ``keep`` is also contaminated by any word the rewrite
    added that comes from what the rewrite prompt was given — the seed
    entities and the previous question — because a forbidden-term list
    written in advance cannot name every way the previous topic leaks in.

    Args:
        case: The labelled case.
        row: What the replay recorded for it.
        previous_question: The question of the turn before, as observed.

    Returns:
        The verdicts: ``contaminated``, ``imported``, ``lost``,
        ``unresolved``, ``correct``, ``changed``, ``added_words`` and, when
        both sides retrieved, the overlap with the reference.
    """
    question = case["question"]
    rewrite = str(row.get("retrieval_question") or question)
    forbidden = [
        term
        for term in case.get("forbid", [])
        if _has_term(rewrite, term) and not _has_term(question, term)
    ]
    lost = [term for term in case.get("keep_terms", []) if not _has_term(rewrite, term)]
    missing = [
        group
        for group in case.get("need", [])
        if not any(_has_term(rewrite, term) for term in group)
    ]
    resolve = case["expect"] == "resolve"
    added = _added_words(question, rewrite)
    given = _content_words(" ".join([*(row.get("seeds") or []), previous_question]))
    imported = [] if resolve else [word for word in added if _same_word(word, given)]
    verdict: dict[str, Any] = {
        "contaminated": forbidden,
        "imported": imported,
        "lost": lost,
        "unresolved": missing if resolve else [],
        "changed": _fold(rewrite) != _fold(question),
        "added_words": added,
    }
    verdict["correct"] = not (forbidden or imported or lost or (resolve and missing))

    variant = row.get("passages")
    reference = row.get("reference_passages")
    if variant is not None and reference:
        ours, theirs = _documents(variant), _documents(reference)
        verdict["doc_recall"] = round(len(set(ours) & set(theirs)) / len(theirs), 3)
        verdict["passage_overlap"] = round(
            len(set(variant) & set(reference)) / len(reference), 3
        )
    expected_docs = case.get("expect_docs") or []
    if expected_docs and variant is not None:
        found = _documents(variant)
        verdict["expected_docs_found"] = [
            label for label in expected_docs if any(_has_term(doc, label) for doc in found)
        ]
    return verdict


def _mean(values: list[float]) -> float | None:
    """Mean rounded to three decimals, or None for no values."""
    return round(statistics.fmean(values), 3) if values else None


def summarise(cases: list[dict[str, Any]], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-case verdicts, overall and per origin.

    Args:
        cases: The labelled cases that were replayed.
        rows: Replay rows by case id, each carrying its ``verdict``.

    Returns:
        One block of counts per group (``all`` and each origin).
    """
    groups: dict[str, list[dict[str, Any]]] = {"all": []}
    for case in cases:
        if case["id"] not in rows:
            continue
        groups["all"].append(case)
        groups.setdefault(case.get("origin", "?"), []).append(case)

    summary: dict[str, Any] = {}
    for name, members in groups.items():
        keep = [c for c in members if c["expect"] == "keep"]
        resolve = [c for c in members if c["expect"] == "resolve"]

        def verdict(case: dict[str, Any]) -> dict[str, Any]:
            """The stored verdict of ``case``."""
            return rows[case["id"]]["verdict"]

        timed = [
            rows[c["id"]]["rewrite_s"] for c in members if rows[c["id"]].get("rewrite_llm_calls")
        ]
        summary[name] = {
            "cases": len(members),
            "keep_correct": f"{sum(verdict(c)['correct'] for c in keep)}/{len(keep)}",
            "resolve_correct": f"{sum(verdict(c)['correct'] for c in resolve)}/{len(resolve)}",
            "correct": sum(verdict(c)["correct"] for c in members),
            "contaminated": sum(
                bool(verdict(c)["contaminated"] or verdict(c)["imported"]) for c in members
            ),
            "lost_or_translated": sum(bool(verdict(c)["lost"]) for c in members),
            "unresolved": sum(bool(verdict(c)["unresolved"]) for c in resolve),
            "keep_changed": sum(verdict(c)["changed"] for c in keep),
            "keep_with_added_words": sum(bool(verdict(c)["added_words"]) for c in keep),
            "gate_refused": sum(bool(rows[c["id"]].get("out_of_scope")) for c in members),
            "relevance_loops": sum(bool(rows[c["id"]].get("rewrite_count")) for c in members),
            "rewrite_calls": len(timed),
            "rewrite_s_mean": _mean(timed),
            "rewrite_s_median": round(statistics.median(timed), 3) if timed else None,
            "rewrite_s_max": round(max(timed), 3) if timed else None,
            "doc_recall_resolve": _mean(
                [verdict(c)["doc_recall"] for c in resolve if "doc_recall" in verdict(c)]
            ),
            "doc_recall_keep": _mean(
                [verdict(c)["doc_recall"] for c in keep if "doc_recall" in verdict(c)]
            ),
            "passage_overlap_resolve": _mean(
                [verdict(c)["passage_overlap"] for c in resolve if "passage_overlap" in verdict(c)]
            ),
        }
    return summary


_SUMMARY_LABELS = {
    "cases": "casi",
    "keep_correct": "da lasciare com'è: corrette",
    "resolve_correct": "da rendere autonome: corrette",
    "correct": "corrette in tutto",
    "contaminated": "contaminazioni (termini estranei aggiunti)",
    "lost_or_translated": "parole dell'utente perse o tradotte",
    "unresolved": "riferimenti non risolti",
    "keep_changed": "da lasciare com'è: riscritte comunque",
    "keep_with_added_words": "da lasciare com'è: con parole aggiunte",
    "gate_refused": "rifiutate dal controllo di ambito",
    "relevance_loops": "con giri del controllo di pertinenza",
    "rewrite_calls": "chiamate di riscrittura",
    "rewrite_s_mean": "tempo riscrittura, media (s)",
    "rewrite_s_median": "tempo riscrittura, mediana (s)",
    "rewrite_s_max": "tempo riscrittura, massimo (s)",
    "doc_recall_resolve": "documenti del riferimento ritrovati (autonome)",
    "doc_recall_keep": "documenti del riferimento ritrovati (com'è)",
    "passage_overlap_resolve": "passaggi del riferimento ritrovati (autonome)",
}


def render_markdown(
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    baseline: dict[str, Any] | None = None,
    baseline_rows: dict[str, dict[str, Any]] | None = None,
) -> str:
    """The summary as a readable page: totals, changed verdicts, failures."""
    groups = list(summary)
    lines = ["# Banco della memoria", ""]
    header = ["misura", *groups]
    if baseline:
        header += [f"{g} (prima)" for g in groups]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key, label in _SUMMARY_LABELS.items():
        cells = [str(summary[g].get(key)) for g in groups]
        if baseline:
            cells += [str(baseline.get(g, {}).get(key)) for g in groups]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    if baseline_rows:
        flips = [
            case for case in cases
            if case["id"] in rows and case["id"] in baseline_rows
            and rows[case["id"]]["verdict"]["correct"]
            != baseline_rows[case["id"]]["verdict"]["correct"]
        ]
        lines += ["", "## Verdetti cambiati rispetto al confronto", ""]
        for case in flips:
            now = rows[case["id"]]
            before = baseline_rows[case["id"]]
            mark = "ora corretta" if now["verdict"]["correct"] else "ora sbagliata"
            lines.append(
                f"- **{case['id']}** ({mark}) {case['question']!r}\n"
                f"  - prima: {before.get('retrieval_question')!r}\n"
                f"  - ora: {now.get('retrieval_question')!r}"
            )
        if not flips:
            lines.append("nessuno")

    lines += ["", "## Casi sbagliati", ""]
    for case in cases:
        row = rows.get(case["id"])
        if not row or row["verdict"]["correct"]:
            continue
        v = row["verdict"]
        why = []
        if v["contaminated"]:
            why.append(f"aggiunge {v['contaminated']}")
        if v["imported"]:
            why.append(f"porta dal turno prima {v['imported']}")
        if v["lost"]:
            why.append(f"perde {v['lost']}")
        if v["unresolved"]:
            why.append(f"manca {v['unresolved']}")
        lines.append(
            f"- **{case['id']}** [{case['expect']}] {case['question']!r} → "
            f"{row.get('retrieval_question')!r} — {'; '.join(why)}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- #
# replay
# ---------------------------------------------------------------------- #


def _served_model(base_url: str) -> str:
    """The model an endpoint serves."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as resp:
        return json.load(resp)["data"][0]["id"]


def _build_bench_agent(base_url: str, model_id: str):
    """The demo agent, without generation and with the follow-up rewrite timed.

    Returns:
        ``(agent, graph_label)``.
    """
    from graphrag.agent.core import KGRAGAgent
    from product.config import build_demo_agent

    class BenchAgent(KGRAGAgent):
        """Demo agent that stops before generating and times the rewrite."""

        probe: dict[str, Any] = {}

        def _generate(self, state):  # noqa: ANN001, D401 - node signature
            """No answer: the bench measures what happens before generation."""
            return {"answer": ""}

        def _rewrite_with_memory(self, question, memory, *args, **kwargs):  # noqa: ANN001
            """Time the rewrite and count the model calls it makes."""
            calls = {"n": 0}
            original = self.llm._invoke_with_retry

            def counted(*args, **kwargs):
                """Count one model call."""
                calls["n"] += 1
                return original(*args, **kwargs)

            self.llm._invoke_with_retry = counted
            started = time.perf_counter()
            try:
                rewritten = super()._rewrite_with_memory(question, memory, *args, **kwargs)
            finally:
                self.llm._invoke_with_retry = original
            self.probe["rewrite_s"] = round(time.perf_counter() - started, 3)
            self.probe["rewrite_llm_calls"] = calls["n"]
            self.probe["rewrite_raw"] = rewritten
            return rewritten

        def _drops_named_subject(self, question, rewritten):  # noqa: ANN001
            """Record whether the named-subject guard fired."""
            dropped = super()._drops_named_subject(question, rewritten)
            self.probe["dropped_subject"] = dropped
            return dropped

    demo, graph_label = build_demo_agent(base_url, model_id)
    agent = BenchAgent(config=demo.config, kg_retriever=demo.kg_retriever, llm=demo.llm)
    return agent, graph_label


def _passages(state: dict[str, Any]) -> list[str]:
    """Sources of the first retrieved passages, in rank order."""
    return [
        str(item.get("source", ""))
        for item in (state.get("retrieved_text_sources") or [])[:_TOP_PASSAGES]
    ]


class _RetrievalCache:
    """Entities retrieved for recorded turns, and reference retrievals, on disk.

    Neither depends on the memory, so a second run of the bench reuses them
    and the only thing that varies between runs is the code under test.
    """

    def __init__(self, path: Path, graph_label: str, refresh: bool) -> None:
        self.path = path
        self.graph_label = graph_label
        self.data: dict[str, Any] = {"graph_label": graph_label, "turns": {}, "reference": {}}
        if path.exists() and not refresh:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("graph_label") != graph_label:
                logger.warning(
                    "Cache built on %s, running on %s: rebuilding it.",
                    stored.get("graph_label"), graph_label,
                )
            else:
                self.data = stored

    def save(self) -> None:
        """Write the cache back."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")

    def turn_entities(self, agent, question: str) -> dict[str, list[dict[str, str]]]:
        """Nodes and triples a recorded turn retrieves, reduced to their names."""
        hit = self.data["turns"].get(question)
        if hit is None:
            state = agent._retrieve(
                {"question": question, "chosen_retrieval_mode": "HYBRID", "sub_questions": []}
            )
            hit = {
                "nodes": [
                    {"text": str(n.get("text") or dict(n.get("properties") or {}).get("name") or "")}
                    for n in state.get("retrieved_nodes") or []
                ],
                "triples": [
                    {"subject": str(t.get("subject", "")), "object": str(t.get("object", ""))}
                    for t in [*(state.get("kg_triples") or []), *(state.get("retrieved_subgraph") or [])]
                ],
            }
            self.data["turns"][question] = hit
        return hit

    def reference(self, agent, question: str) -> dict[str, Any]:
        """Retrieval for the reference question, asked without memory."""
        hit = self.data["reference"].get(question)
        if hit is None:
            state = agent.invoke(question)
            hit = {"passages": _passages(state), "out_of_scope": bool(state.get("out_of_scope"))}
            self.data["reference"][question] = hit
        return hit


def replay_case(agent, cache: _RetrievalCache, case: dict[str, Any], history: list[dict]) -> dict:
    """Feed the recorded turns to a fresh memory and ask the question under test.

    The turns are fed as the demo feeds them: an "Approfondisci" answer takes
    the place of the one it expands where the memory can record it, and is
    skipped otherwise; a turn that failed only marks the conversation as
    started; a refused or introductory turn retrieved
    nothing, so it is observed with no entities and, where the memory takes
    it, with its kind.

    Args:
        agent: The bench agent.
        cache: Retrieval cache for the recorded turns and the reference.
        case: The labelled case.
        history: The recorded turns before the question.

    Returns:
        The replay row, without verdict.
    """
    from graphrag.agent.memory import ConversationMemory

    memory = ConversationMemory()
    # Older memories take no turn kind; the bench still has to run on them to
    # measure the code before a change.
    takes_kind = "kind" in inspect.signature(memory.observe).parameters
    typed_by_id = {turn.get("turn_id"): turn["question"] for turn in history if turn.get("turn_id")}
    for turn in history:
        question = turn["question"]
        if turn.get("deepens"):
            if hasattr(memory, "observe_deepening") and not turn.get("error"):
                entities = cache.turn_entities(agent, turn.get("retrieval_question") or question)
                memory.observe_deepening(
                    question=typed_by_id.get(turn["deepens"], question),
                    answer=str(turn.get("answer") or ""),
                    nodes=entities["nodes"],
                    triples=entities["triples"],
                )
            continue
        if turn.get("error"):
            memory.observe_failure(question)
            continue
        entities: dict[str, list] = {"nodes": [], "triples": []}
        if not (turn.get("out_of_scope") or turn.get("meta_question")):
            entities = cache.turn_entities(agent, turn.get("retrieval_question") or question)
        kind = {}
        if takes_kind:
            if turn.get("meta_question"):
                kind = {"kind": "meta"}
            elif turn.get("out_of_scope"):
                kind = {"kind": "refused"}
        memory.observe(
            question=question,
            answer=str(turn.get("answer") or ""),
            nodes=entities["nodes"],
            triples=entities["triples"],
            **kind,
        )

    agent.probe = {"rewrite_s": 0.0, "rewrite_llm_calls": 0, "dropped_subject": None}
    started = time.perf_counter()
    state = agent.invoke(case["question"], memory=copy.deepcopy(memory))
    row = {
        "id": case["id"],
        "question": case["question"],
        "retrieval_question": state.get("retrieval_question"),
        "seeds": state.get("memory_entities"),
        "follow_up": state.get("follow_up"),
        "quoted_sources": state.get("quoted_sources") or [],
        "out_of_scope": bool(state.get("out_of_scope")),
        "meta_question": bool(state.get("meta_question")),
        "rewrite_count": int(state.get("rewrite_count") or 0),
        "final_query": state.get("rewritten_question") or case["question"],
        "passages": _passages(state),
        "turn_s": round(time.perf_counter() - started, 2),
        **agent.probe,
    }
    reference_question = case.get("standalone") or case["question"]
    reference = cache.reference(agent, reference_question)
    row["reference_question"] = reference_question
    row["reference_passages"] = reference["passages"]
    row["reference_out_of_scope"] = reference["out_of_scope"]
    return row


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Replay rows by case id."""
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run or rescore the bench."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=Path, required=True, help="Labelled cases (JSON).")
    parser.add_argument("--out", type=Path, required=True, help="Run directory.")
    parser.add_argument("--baseline", type=Path, help="Run directory to compare against.")
    parser.add_argument("--only", default="", help="Comma-separated case ids.")
    parser.add_argument("--rescore", action="store_true",
                        help="Recompute verdicts from the rows already in --out.")
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="", help="Served model id (asked to the endpoint).")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/memory_bench/cache.json")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args(argv)

    bench = json.loads(args.cases.read_text(encoding="utf-8"))
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    cases = [case for case in bench["cases"] if not only or case["id"] in only]
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "results.jsonl"

    for case in cases:
        for term in case.get("keep_terms", []):
            if not _has_term(case["question"], term):
                print(f"warning: {case['id']}: keep term {term!r} is not in the question")

    if not args.rescore:
        handler = logging.FileHandler(args.out / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        model_id = args.model or _served_model(args.base_url)
        agent, graph_label = _build_bench_agent(args.base_url, model_id)
        cache = _RetrievalCache(args.cache, graph_label, args.refresh_cache)
        (args.out / "run.json").write_text(
            json.dumps({"model": model_id, "graph": graph_label, "cases": str(args.cases),
                        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1),
            encoding="utf-8",
        )
        with rows_path.open("w", encoding="utf-8") as fh:
            for number, case in enumerate(cases, 1):
                history = bench["histories"][case["history"]][: case["upto"]]
                row = replay_case(agent, cache, case, history)
                cache.save()
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{number}/{len(cases)}] {case['id']}: {case['question'][:50]!r} -> "
                      f"{str(row['retrieval_question'])[:80]!r} ({row['rewrite_s']}s)", flush=True)

    def rescore(rows: dict[str, dict[str, Any]]) -> None:
        """Attach a fresh verdict to every row of a replayed case."""
        for case in cases:
            if case["id"] in rows:
                previous = _previous_question(bench["histories"][case["history"]][: case["upto"]])
                rows[case["id"]]["verdict"] = score_case(case, rows[case["id"]], previous)

    rows = _load_rows(rows_path)
    rescore(rows)
    with rows_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            if case["id"] in rows:
                fh.write(json.dumps(rows[case["id"]], ensure_ascii=False) + "\n")

    summary = summarise(cases, rows)
    baseline = baseline_rows = None
    if args.baseline:
        baseline_rows = _load_rows(args.baseline / "results.jsonl")
        rescore(baseline_rows)
        baseline = summarise(cases, baseline_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    page = render_markdown(summary, cases, rows, baseline, baseline_rows)
    (args.out / "summary.md").write_text(page, encoding="utf-8")
    print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
