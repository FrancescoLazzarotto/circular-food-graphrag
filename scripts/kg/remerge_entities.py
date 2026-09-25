#!/usr/bin/env python3
"""Re-run entity resolution and linking on a run's stage 3 outputs.

Writes new stage 4 and stage 5 artifacts to ``--output-dir``, by default a
timestamped ``remerge_*`` folder inside the run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg_pipeline.stages import linking, llm_extraction, resolution


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Re-run entity resolution and linking on existing stage3 outputs."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory with stage3 outputs")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for new stage4/5 artifacts",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for resolution",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--context-jaccard-floor", type=float, default=0.15)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", ""))
    parser.add_argument("--model-name", default=os.getenv("VLLM_MODEL_NAME", ""))
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
    )
    parser.add_argument(
        "--exclude-mentioned-in",
        action="store_true",
        help="Disable MENTIONED_IN edges in the linked output",
    )
    return parser


def _resolve_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    """The stage 3 triples, stage 3 acronyms and stage 0 documents of a run.

    Raises:
        FileNotFoundError: If any of them is missing.
    """
    triples_path = run_dir / "stage3_triples_raw.json"
    acronyms_path = run_dir / "stage3_acronyms.json"
    documents_path = run_dir / "stage0_documents.json"

    missing = [
        path.name for path in (triples_path, acronyms_path, documents_path) if not path.exists()
    ]
    if missing:
        missing_csv = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required stage files in {run_dir}: {missing_csv}"
        )

    return triples_path, acronyms_path, documents_path


def main() -> int:
    """Resolve and link the run's triples again and write the new artifacts."""
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()

    triples_path, acronyms_path, documents_path = _resolve_paths(run_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = run_dir / f"remerge_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    triples = llm_extraction.load_triples(triples_path)
    acronyms = llm_extraction.load_acronyms(acronyms_path)
    documents = linking.load_documents(documents_path)

    resolved_triples, registry = resolution.resolve_entities(
        triples=triples,
        acronym_map=acronyms,
        embedding_model=args.embedding_model,
        similarity_threshold=float(args.similarity_threshold),
        context_jaccard_floor=float(args.context_jaccard_floor),
        base_url=args.base_url or None,
        api_key=args.api_key or None,
        model_name=args.model_name or None,
    )

    linked = linking.add_cross_document_links(
        triples=resolved_triples,
        registry=registry,
        documents=documents,
        include_mentioned_in=not args.exclude_mentioned_in,
    )

    resolution.save_triples(output_dir / "stage4_triples_resolved.json", resolved_triples)
    resolution.save_registry(output_dir / "stage4_registry.json", registry)
    linking.save_triples(output_dir / "stage5_triples_linked.json", linked)

    print(f"output_dir={output_dir}")
    print(f"resolved_triples={len(resolved_triples)}")
    print(f"linked_triples={len(linked)}")
    print(f"registry_entities={len(registry)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
