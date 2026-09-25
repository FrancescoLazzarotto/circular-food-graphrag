"""Smoke test of the dense text pipeline: indexing, EN and IT queries, cache reload."""

from __future__ import annotations

import argparse
from pathlib import Path

from graphrag.text_rag.factory import make_text_pipeline


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Smoke test for dense (cosine/FAISS) standard RAG pipeline"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input files or directories to index (PDF/TXT/MD)",
    )
    parser.add_argument(
        "--query-en",
        default="What are the main topics discussed in these documents?",
        help="English query for retrieval smoke test",
    )
    parser.add_argument(
        "--query-it",
        default="Quali sono gli argomenti principali trattati in questi documenti?",
        help="Italian query for retrieval smoke test (verifies multilingual embedding)",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model ID for dense embeddings",
    )
    parser.add_argument(
        "--vector-index-dir",
        default="artifacts/vector_index",
        help="Directory for FAISS index cache",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--min-indexed-chunks", type=int, default=1)
    parser.add_argument("--min-retrieved-chunks", type=int, default=1)
    parser.add_argument(
        "--patterns",
        default="*.pdf,*.txt,*.md,*.markdown",
        help="Comma-separated file patterns used when a path is a directory",
    )
    return parser


def _run_query(pipeline, query: str, top_k: int, label: str) -> list:
    """Retrieve for `query` and print every hit."""
    retrieved = pipeline.retrieve(query, top_k=top_k)
    print(f"\n{label}")
    print(f"  query: {query}")
    for idx, item in enumerate(retrieved, start=1):
        preview = item.content[:160].replace("\n", " ")
        print(f"  hit_{idx}: score={item.score:.4f} source={item.source} preview={preview}")
    return retrieved


def main() -> int:
    """Index the given paths, query in both languages and check the results."""
    args = _build_parser().parse_args()

    patterns = tuple(item.strip() for item in args.patterns.split(",") if item.strip())

    pipeline = make_text_pipeline(
        backend="dense",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        vector_index_dir=args.vector_index_dir,
        device=args.device,
    )

    indexed_chunks = pipeline.index_paths(
        [Path(item) for item in args.paths],
        discovery_patterns=patterns,
    )

    print("DENSE RAG SMOKE RESULT")
    print(f"  indexed_chunks: {indexed_chunks}")
    print(f"  embedding_model: {args.embedding_model}")

    retrieved_en = _run_query(pipeline, args.query_en, args.top_k, "EN query")
    retrieved_it = _run_query(pipeline, args.query_it, args.top_k, "IT query")

    # Verify cache hit: re-indexing same corpus should reload from disk.
    pipeline2 = make_text_pipeline(
        backend="dense",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        vector_index_dir=args.vector_index_dir,
        device=args.device,
    )
    pipeline2.index_paths([Path(item) for item in args.paths], discovery_patterns=patterns)
    retrieved_cache = pipeline2.retrieve(args.query_en, top_k=args.top_k)
    print(f"\n  cache_hit_check: retrieved {len(retrieved_cache)} chunks on 2nd run (expect same)")

    failures: list[str] = []
    if indexed_chunks < args.min_indexed_chunks:
        failures.append(f"indexed_chunks {indexed_chunks} < {args.min_indexed_chunks}")
    if len(retrieved_en) < args.min_retrieved_chunks:
        failures.append(f"EN retrieved {len(retrieved_en)} < {args.min_retrieved_chunks}")
    if len(retrieved_it) < args.min_retrieved_chunks:
        failures.append(f"IT retrieved {len(retrieved_it)} < {args.min_retrieved_chunks}")

    # Cosine scores of normalised embeddings fall in [0, 1]; the margin absorbs
    # rounding, and a NaN fails the check.
    for label, hits in (("EN", retrieved_en), ("IT", retrieved_it)):
        for item in hits:
            if not (0.0 <= item.score <= 1.01):
                failures.append(f"{label} score out of range: {item.score:.4f}")

    if failures:
        print("\nDENSE RAG SMOKE FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nDENSE RAG SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
