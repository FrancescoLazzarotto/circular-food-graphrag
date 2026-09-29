"""Score the citations an answer carries, against the evidence the model was shown.

Companion to ``score_gold_run.py``: that script asks whether the answer names the
right concepts, this one asks whether the answer says where each claim came from.
Four measurements, none of them a judgement of the claim itself:

* **coverage** — share of answers carrying at least one citation. The generator
  chooses this; nothing in the pipeline forces a citation to appear.
* **page specificity** — share of citations naming a page range rather than a
  document alone. A document-level citation does not survive the "open the page"
  test the thesis argues for.
* **unverified marks** — occurrences of the citation gate's marker in the
  delivered answer, one per reference id the model invented
  (``verify_citations`` with ``--citation-policy mark``).
* **index consistency** — citations whose document and pages appear in the
  evidence block for that same row. The gate enforces this before rendering, so
  a figure below 1.0 is an implementation defect, not a model property. It is
  reported as an audit of the gate, not as a result about the generator.

The support question ("does the cited page hold the claim?") needs a human.
``--annotation-sample`` writes a CSV of claim sentences paired with the text of
the passage they cite, ready for hand marking; ``--annotations`` reads the marked
file back and reports support accuracy with a Wilson interval.

Usage:
    python evaluation/scripts/score_citations.py \\
        --results-glob '/path/exp_results_fixed/*/*/results.jsonl' \\
        --exclude nothink \\
        --out-prefix artifacts/evaluation/citations_main

    python evaluation/scripts/score_citations.py \\
        --results-glob '...' --annotation-sample 50 --seed 42 \\
        --out-prefix artifacts/evaluation/citations_main

    python evaluation/scripts/score_citations.py \\
        --annotations artifacts/evaluation/citations_main_sample_marked.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from graphrag.agent.evidence import (  # noqa: E402
    UNVERIFIED_MARK_EN,
    UNVERIFIED_MARK_IT,
    short_doc_label,
)

logger = logging.getLogger("evalkit.score_citations")

# "[S1] (a, PRED, b) <REPORT MATTM_Definitivo.pdf | p. 170-170> [conf=…]" and the
# passage form "[S1] <doc | p. 170>".
EVIDENCE_RE = re.compile(r"\[([ST]\d+)\][^<\n]*<([^|>]+)\|\s*p\.\s*([^>]+)>")
# Rendered reader-facing citation: "[REPORT MATTM, p. 170-170, 163-165]", with
# several documents joined by "; ".
CITATION_RE = re.compile(r"\[([^\[\]\n]{3,240})\]")
PAGE_TAIL_RE = re.compile(r",\s*p\.\s*([0-9][0-9,\s–-]*)$")


def _pages(spec: str) -> set[int]:
    """Expand a page spec ("170-170, 163-165") into the pages it names."""
    out: set[int] = set()
    for part in re.split(r"[,;]", str(spec or "")):
        part = part.strip().replace("–", "-")
        if not part:
            continue
        bounds = [b for b in part.split("-") if b.strip().isdigit()]
        if not bounds:
            continue
        first, last = int(bounds[0]), int(bounds[-1])
        if last < first or last - first > 200:
            out.add(first)
            continue
        out.update(range(first, last + 1))
    return out


ABBREV_RE = re.compile(r"\b(pp?|cap|fig|tab|art|cfr|cf|es|e\.g|i\.e|no|nr|vol|ed)\.", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking at "p. 170" inside a citation."""
    masked = ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    parts = re.split(r"(?<=[.!?])\s+", masked)
    return [part.replace("\x00", ".").strip() for part in parts if part.strip()]


def evidence_index(row: dict[str, Any]) -> tuple[dict[str, set[int]], dict[str, str]]:
    """Map short document label -> pages shown, plus id -> evidence text."""
    pages_by_doc: dict[str, set[int]] = defaultdict(set)
    text_by_id: dict[str, str] = {}
    blocks = row.get("contexts") or []
    for block in blocks:
        for match in EVIDENCE_RE.finditer(block):
            ref_id, document, spec = match.group(1), match.group(2).strip(), match.group(3)
            pages_by_doc[short_doc_label(document)] |= _pages(spec)
        # Evidence text for the annotation sample: from the id to the next id.
        parts = re.split(r"\n(?=\[[ST]\d+\])", block)
        for part in parts:
            head = re.match(r"\[([ST]\d+)\]", part.strip())
            if head:
                text_by_id[head.group(1)] = part.strip()
    for source in row.get("retrieved_text_sources") or []:
        match = re.search(r"([^/]+\.pdf)#page=(\d+)", str(source.get("source", "")))
        if match:
            pages_by_doc[short_doc_label(match.group(1))].add(int(match.group(2)))
    return pages_by_doc, text_by_id


def citations(answer: str, known_docs: Iterable[str]) -> list[tuple[str, set[int]]]:
    """Pull rendered citations out of an answer.

    A bracket counts as a citation only when its leading text matches a document
    label from that row's own evidence, which keeps ordinary bracketed prose and
    enumeration markers out of the count.
    """
    known = {doc for doc in known_docs if doc}
    found: list[tuple[str, set[int]]] = []
    for match in CITATION_RE.finditer(answer or ""):
        for chunk in match.group(1).split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            tail = PAGE_TAIL_RE.search(chunk)
            label = chunk[: tail.start()].strip() if tail else chunk
            pages = _pages(tail.group(1)) if tail else set()
            if label in known:
                found.append((label, pages))
    return found


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which behaves at the proportions near 1 seen here."""
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def iter_rows(paths: list[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Every JSONL row of `paths`, with the file it came from."""
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield path, json.loads(line)


def score(paths: list[Path], skip_strategies: set[str]) -> dict[str, Any]:
    """Count citations per strategy; the four measurements plus the samples."""
    per_strategy: dict[str, Counter] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []

    for path, row in iter_rows(paths):
        strategy = str(row.get("strategy", ""))
        if strategy in skip_strategies:
            continue
        answer = str(row.get("answer") or "")
        pages_by_doc, text_by_id = evidence_index(row)
        found = citations(answer, pages_by_doc)
        counter = per_strategy[strategy]
        counter["answers"] += 1
        counter["citations"] += len(found)
        counter["unverified"] += answer.count(UNVERIFIED_MARK_EN) + answer.count(UNVERIFIED_MARK_IT)
        if found:
            counter["answers_cited"] += 1
            counter["docs_cited"] += len({doc for doc, _ in found})
        for doc, pages in found:
            if pages:
                counter["with_pages"] += 1
                shown = pages_by_doc.get(doc, set())
                if pages & shown:
                    counter["in_index"] += 1
            else:
                counter["doc_only"] += 1
        if found and text_by_id:
            samples.append(
                {
                    "run": path.parent.name,
                    "model": str((row.get("metadata") or {}).get("model_id", "")),
                    "strategy": strategy,
                    "query_id": str(row.get("query_id", "")),
                    "question": str(row.get("question", "")),
                    "answer": answer,
                    "evidence": text_by_id,
                    "docs": sorted(pages_by_doc),
                }
            )

    summary = {}
    for strategy, counter in sorted(per_strategy.items()):
        answers = counter["answers"]
        cites = counter["citations"]
        with_pages = counter["with_pages"]
        summary[strategy] = {
            "answers": answers,
            "answers_cited": counter["answers_cited"],
            "coverage": counter["answers_cited"] / answers if answers else 0.0,
            "coverage_ci": wilson(counter["answers_cited"], answers),
            "citations": cites,
            "citations_per_answer": cites / answers if answers else 0.0,
            "page_specificity": with_pages / cites if cites else 0.0,
            "doc_only": counter["doc_only"],
            "index_consistency": counter["in_index"] / with_pages if with_pages else 0.0,
            "unverified_marks": counter["unverified"],
            "docs_per_citing_answer": (
                counter["docs_cited"] / counter["answers_cited"] if counter["answers_cited"] else 0.0
            ),
        }
    return {"per_strategy": summary, "_samples": samples}


ORIGIN_RE = re.compile(r"<([^|>]+)\|\s*p\.\s*([^>]+)>")


def _evidence_origin(text: str) -> tuple[str, set[int]]:
    """Document label and pages an evidence item declares in its header."""
    match = ORIGIN_RE.search(text)
    if not match:
        return ("", set())
    return (short_doc_label(match.group(1).strip()), _pages(match.group(2)))


def write_sample(samples: list[dict[str, Any]], size: int, seed: int, path: Path) -> int:
    """Write claim/citation pairs for hand marking, one sentence per row."""
    rng = random.Random(seed)
    pool: list[dict[str, str]] = []
    for sample in samples:
        # The closing source list repeats every citation in one block; it is not
        # a claim and it would dominate the sample.
        body = re.split(r"\n\s*(?:Sources|Fonti)\s*:", sample["answer"])[0]
        for sentence in sentences(body):
            ids = re.findall(r"\[([ST]\d+)\]", sentence)
            cited = [sample["evidence"][ref] for ref in ids if ref in sample["evidence"]]
            if not cited:
                # Rendered labels replace the ids, so match the sentence back to
                # the evidence on document and pages together. Matching on pages
                # alone pairs a claim with a passage from a different document
                # that happens to share a page number.
                wanted = citations(sentence, sample["docs"])
                # One claim often cites two documents. Take evidence for each of
                # them rather than the first matches in reading order, or the
                # reader marks the claim against half of what it points at.
                per_doc: dict[str, list[str]] = {}
                for text in sample["evidence"].values():
                    item_doc, item_pages = _evidence_origin(text)
                    for doc, pages in wanted:
                        if doc == item_doc and (not pages or pages & item_pages):
                            per_doc.setdefault(doc, []).append(text)
                            break
                cited = [text for texts in per_doc.values() for text in texts[:2]]
            if cited:
                pool.append(
                    {
                        "model": sample["model"],
                        "strategy": sample["strategy"],
                        "query_id": sample["query_id"],
                        "claim": sentence.strip(),
                        "cited_evidence": "\n---\n".join(cited[:6])[:6000],
                        "supported": "",
                    }
                )
    rng.shuffle(pool)
    chosen = pool[:size]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "strategy", "query_id", "claim", "cited_evidence", "supported"],
        )
        writer.writeheader()
        writer.writerows(chosen)
    return len(chosen)


def read_annotations(path: Path) -> dict[str, Any]:
    """Read a marked sample: ``supported`` in {1,0,?}."""
    yes = no = unclear = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            mark = str(record.get("supported", "")).strip().lower()
            if mark in {"1", "y", "yes", "true"}:
                yes += 1
            elif mark in {"0", "n", "no", "false"}:
                no += 1
            elif mark:
                unclear += 1
    judged = yes + no
    return {
        "supported": yes,
        "unsupported": no,
        "unclear": unclear,
        "judged": judged,
        "support_accuracy": yes / judged if judged else 0.0,
        "support_ci": wilson(yes, judged),
    }


def markdown(result: dict[str, Any]) -> str:
    """Markdown table of the per-strategy measurements."""
    lines = [
        "| Strategy | Answers | Coverage | Citations/answer | Page-level | Docs/answer | Unverified |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, row in result["per_strategy"].items():
        lines.append(
            f"| {strategy} | {row['answers']} | {row['coverage']:.3f} | "
            f"{row['citations_per_answer']:.1f} | {row['page_specificity']:.3f} | "
            f"{row['docs_per_citing_answer']:.2f} | {row['unverified_marks']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments; score, write a sample, or read back annotations."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-glob", help="glob matching results.jsonl files")
    parser.add_argument(
        "--results",
        nargs="*",
        default=[],
        help="explicit results.jsonl paths, for a campaign whose scored runs are "
        "not the whole directory (the reasoning models are scored on their "
        "<think>-stripped copy)",
    )
    parser.add_argument("--exclude", default="", help="substring: skip paths containing it")
    parser.add_argument(
        "--skip-strategies",
        default="no_retrieval",
        help="comma-separated strategies to leave out (no evidence to cite)",
    )
    parser.add_argument("--out-prefix", help="write <prefix>.json and <prefix>.md")
    parser.add_argument("--annotation-sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations", help="marked CSV to read back")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    if args.annotations:
        stats = read_annotations(Path(args.annotations))
        low, high = stats["support_ci"]
        print(
            f"support accuracy {stats['support_accuracy']:.3f} "
            f"[{low:.3f}, {high:.3f}] on {stats['judged']} judged claims "
            f"({stats['unsupported']} unsupported, {stats['unclear']} unclear)"
        )
        return 0

    if not args.results_glob and not args.results:
        parser.error("--results-glob or --results is required unless --annotations is given")

    paths = [Path(p) for p in args.results]
    if args.results_glob:
        paths += [Path(p) for p in sorted(glob.glob(args.results_glob))]
    if args.exclude:
        paths = [p for p in paths if args.exclude not in str(p)]
    if not paths:
        parser.error(f"no results.jsonl matched {args.results_glob!r}")
    logger.info("scoring %d runs", len(paths))

    skip = {s.strip() for s in args.skip_strategies.split(",") if s.strip()}
    result = score(paths, skip)
    samples = result.pop("_samples")
    print(markdown(result))

    if args.out_prefix:
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        prefix.with_suffix(".md").write_text(markdown(result) + "\n", encoding="utf-8")
        logger.info("wrote %s.json and %s.md", prefix, prefix)

    if args.annotation_sample:
        out = Path(f"{args.out_prefix or 'citations'}_sample.csv")
        written = write_sample(samples, args.annotation_sample, args.seed, out)
        logger.info("wrote %d claims to %s (mark the 'supported' column 1/0)", written, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
