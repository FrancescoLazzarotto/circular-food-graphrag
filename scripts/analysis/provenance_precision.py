#!/usr/bin/env python3
"""Measure how much retrieved evidence comes from the gold documents.

Reads one or more ``results.jsonl`` files produced by
``scripts/runners/run_retrieval_matrix.py`` and, per strategy, reports the share of
retrieved evidence whose origin document belongs to the gold whitelist
(``provenance_precision``) versus evidence pulled from other (cross-document)
sources.

Two retrieval channels carry provenance:

* graph triples -> ``retrieved_triples[].source_doc`` (the edge's origin doc)
* text chunks   -> ``retrieved_text_sources[].source`` ("<path>#page=N#chunk=M")

Retrieved *entities* (bare graph nodes) carry no document provenance in this KG
(``MENTIONED_IN`` is not materialised), so they are reported only as a blind-spot
count, never as gold/non-gold.

This is a diagnostic, not a penalty: a correct answer that also cites a non-gold
document is not necessarily wrong. Report precision alongside correctness; do not
subtract it from the score.

Usage:
    # Build / verify the whitelist: list every source_doc seen in a run.
    python scripts/analysis/provenance_precision.py \
        --results artifacts/experiments/<run>/results.jsonl --list-docs

    # Score provenance precision per strategy against the whitelist.
    python scripts/analysis/provenance_precision.py \
        --results artifacts/experiments/<run>/results.jsonl \
        --gold-docs evaluation/gold/gold_source_docs.txt \
        --output-csv provenance.csv

    # Aggregate several runs at once.
    python scripts/analysis/provenance_precision.py \
        --results-dir artifacts/experiments --gold-docs evaluation/gold/gold_source_docs.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger("graphrag")

# Sentinel used for evidence that carries no source tag at all.
_UNKNOWN = "<unknown>"


def _normalise_doc(raw: str) -> str:
    """Reduce a source tag to a bare file basename.

    ``source_doc`` values are already basenames; text ``source`` tags look like
    ``/abs/path/Doc.pdf#page=3#chunk=2``. Both collapse to ``Doc.pdf``.

    Args:
        raw: Raw ``source_doc`` or text ``source`` string.

    Returns:
        The basename with any ``#...`` suffix stripped, or "" if empty.
    """
    head = (raw or "").split("#", 1)[0].strip()
    if not head:
        return ""
    return os.path.basename(head).strip()


def load_whitelist(path: Path) -> set[str]:
    """Load gold-document basenames from a whitelist file.

    Blank lines and ``#`` comment lines are ignored. Matching is case-insensitive,
    so basenames are lowercased.

    Args:
        path: Whitelist file, one basename per line.

    Returns:
        Set of lowercased gold-document basenames.

    Raises:
        FileNotFoundError: If the whitelist file does not exist.
        ValueError: If no usable entries are found.
    """
    docs: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        docs.add(_normalise_doc(stripped).lower())
    if not docs:
        raise ValueError(f"No gold documents found in whitelist: {path}")
    return docs


def resolve_results_paths(
    results: list[str], results_dir: str | None
) -> list[Path]:
    """Collect the ``results.jsonl`` files to analyse.

    Args:
        results: Explicit ``results.jsonl`` paths.
        results_dir: Directory to search recursively for ``results.jsonl``.

    Returns:
        Sorted, de-duplicated list of existing result files.

    Raises:
        FileNotFoundError: If nothing usable is found.
    """
    paths: set[Path] = set()
    for item in results:
        candidate = Path(item).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Results file not found: {candidate}")
        paths.add(candidate)
    if results_dir:
        root = Path(results_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Results dir not found: {root}")
        paths.update(p.resolve() for p in root.rglob("results.jsonl"))
    if not paths:
        raise FileNotFoundError("No results.jsonl files to analyse.")
    return sorted(paths)


def iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield JSON records from the given ``results.jsonl`` files.

    Malformed lines are skipped with a warning rather than aborting the run.

    Args:
        paths: Result files to read.

    Yields:
        One decoded record dict per non-empty line.
    """
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line %d in %s", line_no, path)
                    continue
                if isinstance(record, dict):
                    yield record


def _iter_docs(record: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield ``(channel, normalised_doc)`` for every provenance-bearing unit.

    ``channel`` is ``"graph"`` (from triples) or ``"text"`` (from chunks). Units
    with no source tag yield the ``_UNKNOWN`` sentinel so they can be counted
    separately from cross-document hits.

    Args:
        record: One results.jsonl record.

    Yields:
        ``(channel, doc_basename_or_unknown)`` tuples.
    """
    for triple in record.get("retrieved_triples", []) or []:
        if not isinstance(triple, dict):
            continue
        doc = _normalise_doc(str(triple.get("source_doc", "")))
        yield "graph", doc or _UNKNOWN
    for source in record.get("retrieved_text_sources", []) or []:
        if not isinstance(source, dict):
            continue
        doc = _normalise_doc(str(source.get("source", "")))
        yield "text", doc or _UNKNOWN


class ProvenanceTally:
    """Accumulates gold / non-gold / unknown counts per (group, channel).

    Attributes:
        whitelist: Lowercased gold-document basenames.
        counts: group -> channel -> bucket (``gold``, ``nongold``,
            ``unknown``) -> count.
        entities_no_prov: group -> retrieved entities, which carry no
            document provenance.
    """

    def __init__(self, whitelist: set[str]) -> None:
        """Start an empty tally against `whitelist`."""
        self.whitelist = whitelist
        # group -> channel -> {"gold", "nongold", "unknown"} -> count
        self.counts: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        # group -> number of retrieved entities (no provenance available)
        self.entities_no_prov: Counter = Counter()

    def add(self, group: str, record: dict[str, Any]) -> None:
        """Tally one record under the given group key (strategy or strategy+qid)."""
        for channel, doc in _iter_docs(record):
            if doc == _UNKNOWN:
                bucket = "unknown"
            elif doc.lower() in self.whitelist:
                bucket = "gold"
            else:
                bucket = "nongold"
            self.counts[group][channel][bucket] += 1
        entities = record.get("retrieved_entities", []) or []
        if isinstance(entities, list):
            self.entities_no_prov[group] += len(entities)

    def rows(self) -> list[dict[str, Any]]:
        """Flatten the tally into one row per (group, channel), plus 'combined'."""
        rows: list[dict[str, Any]] = []
        for group in sorted(self.counts):
            channels = self.counts[group]
            combined: Counter = Counter()
            for channel in ("graph", "text"):
                c = channels.get(channel, Counter())
                combined.update(c)
                rows.append(self._row(group, channel, c))
            rows.append(self._row(group, "combined", combined))
        return rows

    def _row(self, group: str, channel: str, c: Counter) -> dict[str, Any]:
        """One output row, with precision over the gold and non-gold units."""
        gold, nongold, unknown = c["gold"], c["nongold"], c["unknown"]
        scored = gold + nongold
        precision = (gold / scored) if scored else None
        return {
            "group": group,
            "channel": channel,
            "gold": gold,
            "nongold": nongold,
            "unknown": unknown,
            "scored": scored,
            "provenance_precision": precision,
            "entities_no_prov": self.entities_no_prov.get(group, 0)
            if channel == "combined"
            else "",
        }


def _fmt_precision(value: float | None) -> str:
    """Precision as a percentage, or ``n/a``."""
    return "  n/a" if value is None else f"{value:6.1%}"


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print the provenance table to stdout."""
    header = (
        f"{'group':<22} {'channel':<9} {'gold':>6} {'nongold':>8} "
        f"{'unknown':>8} {'precision':>10} {'ent_no_prov':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['group']:<22} {row['channel']:<9} {row['gold']:>6} "
            f"{row['nongold']:>8} {row['unknown']:>8} "
            f"{_fmt_precision(row['provenance_precision']):>10} "
            f"{str(row['entities_no_prov']):>12}"
        )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the provenance table as CSV."""
    fields = [
        "group",
        "channel",
        "gold",
        "nongold",
        "unknown",
        "scored",
        "provenance_precision",
        "entities_no_prov",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if out["provenance_precision"] is not None:
                out["provenance_precision"] = f"{out['provenance_precision']:.4f}"
            writer.writerow(out)


def list_docs(records: Iterable[dict[str, Any]], whitelist: set[str]) -> None:
    """Print every distinct source document seen, with counts and a gold tag.

    Use this to build or reconcile the whitelist: run it, eyeball the basenames,
    and confirm exactly which ones back the gold.

    Args:
        records: Result records to scan.
        whitelist: Current gold whitelist (may be empty when discovering).
    """
    per_channel: dict[str, Counter] = {"graph": Counter(), "text": Counter()}
    for record in records:
        for channel, doc in _iter_docs(record):
            per_channel[channel][doc] += 1
    for channel in ("graph", "text"):
        counter = per_channel[channel]
        print(f"\n=== {channel} channel: {len(counter)} distinct source(s) ===")
        for doc, count in counter.most_common():
            if doc == _UNKNOWN:
                tag = "UNKNOWN"
            elif doc.lower() in whitelist:
                tag = "GOLD"
            else:
                tag = "other"
            print(f"{count:8d}  [{tag:<7}]  {doc}")


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Measure provenance precision of retrieved evidence vs the gold docs.",
    )
    parser.add_argument(
        "--results",
        nargs="+",
        default=[],
        help="One or more results.jsonl files.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory searched recursively for results.jsonl files.",
    )
    parser.add_argument(
        "--gold-docs",
        default=None,
        help="Whitelist file of gold-document basenames (required unless --list-docs).",
    )
    parser.add_argument(
        "--by-query",
        action="store_true",
        help="Break the tally down per (strategy, query_id) instead of per strategy.",
    )
    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="List every source document seen (to build/verify the whitelist) and exit.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to also write the table as CSV.",
    )
    return parser


def main() -> None:
    """Tally provenance per strategy (or per query), or list the documents seen."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()

    paths = resolve_results_paths(args.results, args.results_dir)
    logger.info("Analysing %d result file(s).", len(paths))

    if args.list_docs:
        whitelist = load_whitelist(Path(args.gold_docs)) if args.gold_docs else set()
        list_docs(iter_records(paths), whitelist)
        return

    if not args.gold_docs:
        raise SystemExit("--gold-docs is required (or use --list-docs to discover them).")

    whitelist = load_whitelist(Path(args.gold_docs))
    logger.info("Gold whitelist: %d document(s).", len(whitelist))

    tally = ProvenanceTally(whitelist)
    n_records = 0
    for record in iter_records(paths):
        n_records += 1
        strategy = str(record.get("strategy", "") or "?")
        if args.by_query:
            qid = str(record.get("query_id", "") or "?")
            group = f"{strategy}:{qid}"
        else:
            group = strategy
        tally.add(group, record)
    logger.info("Tallied %d record(s).", n_records)

    rows = tally.rows()
    print_table(rows)
    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
        logger.info("Wrote CSV: %s", args.output_csv)


if __name__ == "__main__":
    main()
