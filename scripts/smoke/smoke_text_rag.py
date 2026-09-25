"""Smoke test of the standard (lexical) text pipeline: indexing and one query."""

from __future__ import annotations

import argparse
from pathlib import Path

from graphrag.text_rag.pipeline import StandardTextRAGPipeline


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Quick smoke test for standard text RAG pipeline"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input files or directories to index (PDF/TXT/MD)",
    )
    parser.add_argument(
        "--query", default="What are the main points in these documents?"
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    parser.add_argument("--min-chunk-chars", type=int, default=80)
    parser.add_argument("--min-indexed-chunks", type=int, default=1)
    parser.add_argument("--min-retrieved-chunks", type=int, default=1)
    parser.add_argument(
        "--patterns",
        default="*.pdf,*.txt,*.md,*.markdown",
        help="Comma-separated file patterns used when a path is a directory",
    )
    return parser


def main() -> int:
    """Index the given paths, retrieve for one query and check the counts."""
    args = _build_parser().parse_args()

    patterns = tuple(item.strip() for item in args.patterns.split(",") if item.strip())

    pipeline = StandardTextRAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
    )

    indexed_chunks = pipeline.index_paths(
        [Path(item) for item in args.paths],
        discovery_patterns=patterns,
    )

    retrieved = pipeline.retrieve(args.query, top_k=args.top_k)

    print("TEXT RAG SMOKE RESULT")
    print(f"- indexed_chunks: {indexed_chunks}")
    print(f"- retrieved_chunks: {len(retrieved)}")
    print(f"- query: {args.query}")

    for idx, item in enumerate(retrieved, start=1):
        preview = item.content[:180].replace("\n", " ")
        print(
            f"- hit_{idx}: score={item.score:.4f} source={item.source} preview={preview}"
        )

    failures: list[str] = []
    if indexed_chunks < max(0, int(args.min_indexed_chunks)):
        failures.append(
            f"indexed_chunks expected >= {args.min_indexed_chunks}, got {indexed_chunks}"
        )

    if len(retrieved) < max(0, int(args.min_retrieved_chunks)):
        failures.append(
            f"retrieved_chunks expected >= {args.min_retrieved_chunks}, got {len(retrieved)}"
        )

    if failures:
        print("TEXT RAG SMOKE FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    context_preview = pipeline.build_context(args.query, top_k=args.top_k)[
        :280
    ].replace("\n", " | ")
    print(f"- context_preview: {context_preview}")
    print("TEXT RAG SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
