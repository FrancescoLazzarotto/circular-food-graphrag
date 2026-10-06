#!/usr/bin/env python3
"""Encode the demo's text index ahead of time, outside the demo, and check its coverage.

The demo builds its dense text index at startup from the stage0 runs it is
given, encoding only the passages the passage-vector cache does not hold yet.
Run this first on the same runs and the demo finds every vector in the cache:
it starts without encoding anything, and the GPU it shares with the generator
is not used for indexing. Encoding is committed every few hundred passages, so
an interrupted run resumes where it stopped.

Then checks coverage: every document the registry includes has at least one
passage, and no passage id repeats. Exits with status 1 when either fails.

    python scripts/corpus/build_text_index.py --stage0-runs run_corpus_20261005 \
        --registry product/corpus_registry.csv --report coverage.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# The kg_pipeline and product packages live at the repo root and are not pip-installed.
sys.path.insert(0, str(ROOT))

from graphrag import cli as graphrag_cli  # noqa: E402
from kg_pipeline.utils import corpus_registry  # noqa: E402
from product.config import DENSE_EMBEDDING_MODEL  # noqa: E402

LOGGER = logging.getLogger("build_text_index")


def coverage(
    chunk_ids: list[str], indexed_docs: dict[str, str], registry: Path | None
) -> dict[str, object]:
    """Which registry documents the index holds, and whether any passage id repeats.

    Args:
        chunk_ids: Ids of every indexed passage.
        indexed_docs: ``doc_id -> file name`` of every indexed document.
        registry: Corpus registry, or ``None`` to skip the document check.

    Returns:
        The report: passage and document counts, repeated ids, and the
        included documents without any passage.
    """
    repeated = {cid: n for cid, n in Counter(chunk_ids).items() if n > 1}
    report: dict[str, object] = {
        "passages": len(chunk_ids),
        "documents_indexed": len(indexed_docs),
        "repeated_ids": len(repeated),
        "repeated_id_examples": sorted(repeated)[:10],
    }
    if registry is not None:
        included = [row for row in corpus_registry.load_registry(registry) if not row.escluso]
        missing = [
            {"id_documento": row.id_documento, "percorso": row.percorso}
            for row in included
            if row.id_documento not in indexed_docs
        ]
        report.update(
            {
                "registry": str(registry),
                "registry_included": len(included),
                "registry_without_passages": missing,
                "indexed_not_in_registry": sorted(
                    set(indexed_docs) - {row.id_documento for row in included}
                ),
            }
        )
    return report


def main() -> int:
    """Build the index, check it, write the report; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--stage0-runs",
        required=True,
        help="comma-separated run directories under kg_pipeline/artifacts, as DEMO_TEXT_STAGE0_RUNS",
    )
    parser.add_argument("--registry", type=Path, default=None, help="corpus registry to check against")
    parser.add_argument(
        "--vector-index-dir",
        type=Path,
        default=ROOT / "artifacts" / "vector_index",
        help="passage-vector cache the demo reads (default: the demo's)",
    )
    parser.add_argument("--report", type=Path, default=None, help="where to write the JSON report")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # The builder reads `kg_pipeline/artifacts` relative to the working directory.
    os.chdir(ROOT)

    started = time.monotonic()
    pipeline = graphrag_cli._build_text_pipeline(
        argparse.Namespace(
            text_retriever_backend="dense",
            dense_embedding_model=DENSE_EMBEDDING_MODEL,
            vector_index_dir=str(args.vector_index_dir),
            text_docs_dir="",
            text_stage0_runs=args.stage0_runs,
        )
    )
    if pipeline is None:
        LOGGER.error("Nothing indexed from %s", args.stage0_runs)
        return 1
    elapsed = time.monotonic() - started

    chunks = pipeline.retriever.chunks
    indexed_docs: dict[str, str] = {}
    for run in (part.strip() for part in args.stage0_runs.split(",") if part.strip()):
        stage0 = ROOT / "kg_pipeline" / "artifacts" / run / "stage0_documents.json"
        for doc in json.loads(stage0.read_text(encoding="utf-8")):
            indexed_docs.setdefault(str(doc.get("doc_id") or ""), str(doc.get("filename") or ""))
    with_passages = {chunk.chunk_id.split("-", 1)[0] for chunk in chunks}
    indexed_docs = {doc_id: name for doc_id, name in indexed_docs.items() if doc_id in with_passages}

    report = coverage([chunk.chunk_id for chunk in chunks], indexed_docs, args.registry)
    report.update(
        {
            "stage0_runs": args.stage0_runs,
            "vector_index_dir": str(args.vector_index_dir),
            "model": DENSE_EMBEDDING_MODEL,
            "build_seconds": round(elapsed, 1),
        }
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)

    ok = not report["repeated_ids"] and not report.get("registry_without_passages")
    if not ok:
        LOGGER.error("Coverage check failed: see repeated_ids and registry_without_passages")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
