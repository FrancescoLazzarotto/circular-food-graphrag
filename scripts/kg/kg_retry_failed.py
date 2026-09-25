#!/usr/bin/env python3
"""Re-extract triples for the chunks that failed during Stage 3.

Stage 3 logs every chunk it could not extract to ``failed_chunks.jsonl`` (LLM
timeouts, truncated/invalid JSON, validation errors). Those chunks contribute no
triples and, once Stage 3 has checkpointed past them, are never retried. This
script re-runs extraction for just those chunks — reusing the exact prompt,
parsing and validation path of the pipeline — with a generous timeout and output
budget, then appends the recovered triples to ``stage3_triples_raw.json``.

After running this, recompute the downstream stages so the graph picks up the
recovered triples:

    rm stageN  (4,5,6 artifacts) in the run-dir
    python -m kg_pipeline.main --run-dir <run-dir>      # resolution -> linking -> neo4j
    python scripts/kg/kg_postprocess.py --passes 1,2,3,4

It is safe to re-run: chunks that succeed are merged once; chunks that still fail
are written to ``failed_chunks_retry.jsonl`` and left out.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kg_pipeline.stages import ner, llm_extraction  # noqa: E402
from kg_pipeline.stages.chunking import load_chunks  # noqa: E402
from kg_pipeline.prompts.extraction_prompt import build_extraction_prompt  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("kg_retry_failed")


def _failed_chunk_ids(failed_path: Path) -> list[str]:
    """The distinct chunk ids of a ``failed_chunks.jsonl``, in order."""
    ids: list[str] = []
    seen: set[str] = set()
    for line in failed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        meta = json.loads(line).get("chunk_metadata")
        if isinstance(meta, str):
            try:
                meta = ast.literal_eval(meta)
            except (ValueError, SyntaxError):
                meta = {}
        cid = (meta or {}).get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def main() -> None:
    """Re-extract the failed chunks and append what is recovered to stage 3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "kg_pipeline" / "config.yaml"))
    parser.add_argument("--env-file", default=str(ROOT / "kg_pipeline" / ".env"))
    parser.add_argument("--timeout", type=float, default=400.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    load_dotenv(args.env_file, override=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    allowed_labels = config["ontology"]["labels"]
    allowed_label_set = set(allowed_labels)
    vocab_path = ROOT / "kg_pipeline" / config["llm"]["relation_vocab_path"]
    relation_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    # Lift the output cap for the retry so densely-packed chunks that truncated
    # at the default budget can complete.
    llm_extraction._MAX_OUTPUT_TOKENS = args.max_tokens

    chunks = {c.chunk_id: c for c in load_chunks(run_dir / "stage1_chunks.json")}
    ner_map = ner.load_ner(run_dir / "stage2_ner.json")
    failed_ids = _failed_chunk_ids(run_dir / "failed_chunks.jsonl")
    LOGGER.info("Unique failed chunks to retry: %d", len(failed_ids))

    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    model_name = os.getenv("VLLM_MODEL_NAME", "")
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
    temperature = float(config["llm"]["temperature"])
    seed = int(config.get("seed", 42))
    use_structured = bool(config["llm"]["use_structured_output"])

    retry_failed_path = run_dir / "failed_chunks_retry.jsonl"
    new_labels_path = run_dir / "new_labels.log"

    # Build (idx, chunk, prompt) tasks for every failed chunk we still have text
    # for, then run them through the pipeline's concurrent extractor.
    batch_tasks: list[tuple[int, object, str]] = []
    for idx, cid in enumerate(failed_ids):
        chunk = chunks.get(cid)
        if chunk is None:
            continue
        candidates = [e.model_dump() for e in ner_map.get(cid, [])]
        prompt = build_extraction_prompt(
            chunk, candidates, allowed_labels, relation_vocab=relation_vocab
        )
        batch_tasks.append((idx, chunk, prompt))

    LOGGER.info(
        "Retrying %d chunks (concurrency=%d, timeout=%ss, max_tokens=%d)",
        len(batch_tasks),
        args.concurrency,
        args.timeout,
        args.max_tokens,
    )

    async def _run() -> list:
        """Run the retry batch through the pipeline's concurrent extractor."""
        async with AsyncOpenAI(
            base_url=base_url, api_key=api_key or "EMPTY", timeout=args.timeout
        ) as client:
            return await llm_extraction._run_batch_async(
                batch_tasks=batch_tasks,
                client=client,
                concurrent_requests=args.concurrency,
                model_name=model_name,
                temperature=temperature,
                seed=seed,
                use_structured_output=use_structured,
                max_retries=args.max_retries,
                allowed_label_set=allowed_label_set,
                failed_chunks_path=retry_failed_path,
                new_label_log_path=new_labels_path,
                allowed_predicates=relation_vocab,
            )

    results = asyncio.run(_run())

    recovered: list = []
    recovered_chunks = 0
    still_failed = 0
    for _idx, triples, ok in results:
        if not ok:
            still_failed += 1
            continue
        if triples:
            recovered_chunks += 1
            for triple in triples:
                rel = dict(triple.relationship_properties)
                rel["extraction_method"] = "llm_retry"
                triple.relationship_properties = rel
            recovered.extend(triples)

    LOGGER.info(
        "Retry done: %d chunks recovered, %d triples, %d still failing",
        recovered_chunks,
        len(recovered),
        still_failed,
    )

    if recovered:
        raw_path = run_dir / "stage3_triples_raw.json"
        existing = llm_extraction.load_triples(raw_path)
        before = len(existing)
        existing.extend(recovered)
        llm_extraction.save_triples(raw_path, existing)
        LOGGER.info(
            "Appended to %s: %d -> %d triples", raw_path.name, before, len(existing)
        )
        LOGGER.info(
            "Next: rm %s/stage4_* %s/stage5_* %s/stage6_* and re-run "
            "`python -m kg_pipeline.main --run-dir %s` then kg_postprocess.",
            run_dir,
            run_dir,
            run_dir,
            run_dir,
        )


if __name__ == "__main__":
    main()
