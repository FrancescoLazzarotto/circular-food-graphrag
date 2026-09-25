#!/usr/bin/env python3
"""Ontology discovery for a new corpus: open-vocabulary triple extraction.

Samples chunks from the corpus (reusing kg_pipeline stage 0/1), asks the LLM to
extract triples WITHOUT any label or predicate vocabulary constraint, then
tallies the entity types and predicates the model proposes. The ranked output
is the raw material for curating the seed ontology labels and the relation
vocab of a new domain before running the full pipeline.

Run with:
  conda run -n graphllm python scripts/analysis/ontology_discovery.py \
    --input-dir "documents/test 1" --sample-target 60

Outputs (in --output-dir, default kg_pipeline/artifacts/discovery_<timestamp>):
  stage0_documents.json / stage1_chunks.json   cached ingestion+chunking
  sampled_chunks.json                          the sampled chunk ids + text
  raw_triples.jsonl                            every open-vocab triple extracted
  discovery_report.json                        machine-readable tallies
  discovery_report.md                          ranked tables for manual curation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg_pipeline.stages import chunking, ingestion  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("ontology_discovery")

# Section titles (EN + IT) that mark front/back matter: skipped when sampling
# because they yield editorial noise, not domain facts.
_SKIP_SECTION_RE = re.compile(
    r"(references|bibliograph|acknowledg|table of contents|list of (figures|tables|acronyms)"
    r"|copyright|colophon|editorial board|scientific (board|committee)"
    r"|bibliografia|sommario|indice|ringraziament|colofone|comitato scientifico)",
    re.IGNORECASE,
)

_PREDICATE_CLEAN_RE = re.compile(r"[^A-Z0-9_]+")
_TYPE_CLEAN_RE = re.compile(r"[^A-Za-z0-9 ]+")

_DISCOVERY_PROMPT = """You are helping design the ontology of a new knowledge graph. The domain is \
circular economy and food systems (papers, books, reports, magazines; text may be English or Italian).

Extract the factual triples stated in the chunk below. Return ONLY a JSON array of objects:
{{"subject": str, "subject_type": str, "predicate": str, "object": str, "object_type": str}}

Rules:
- subject_type / object_type: a short CapitalCase noun for the entity CLASS (e.g. Organization, \
Process, Material, Indicator). Invent the class that fits best - there is NO fixed list. Always \
in ENGLISH, even for Italian text. Prefer reusable general classes over one-off specific ones.
- predicate: SCREAMING_SNAKE_CASE verb phrase, 1-4 words, ENGLISH, expressing the relation \
direction subject -> object. Invent freely - NO fixed list. Prefer specific over generic.
- subject / object: keep the original language of the text, do not translate.
- Extract only facts actually stated in the chunk. No world knowledge. Skip vague fragments.
- If the chunk has no extractable facts (e.g. editorial boilerplate), return [].

Chunk (doc: {filename}, section: {section_title}):
{text}"""


def _norm_predicate(value: str) -> str:
    """SCREAMING_SNAKE_CASE form of a proposed predicate."""
    cleaned = _PREDICATE_CLEAN_RE.sub("_", value.strip().upper())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _norm_type(value: str) -> str:
    """CapitalCase form of a proposed entity type."""
    cleaned = _TYPE_CLEAN_RE.sub(" ", value.strip())
    cleaned = " ".join(part.capitalize() for part in cleaned.split())
    return cleaned.replace(" ", "")


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Parse a JSON array, tolerating markdown fences and surrounding prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("LLM output is not a JSON array")
    return [item for item in parsed if isinstance(item, dict)]


def _load_or_run_stage01(
    input_dir: Path, out_dir: Path, config: dict[str, Any]
) -> list[chunking.ChunkRecord]:
    """Chunk the corpus, reusing the stage 0/1 artifacts cached in `out_dir`.

    Args:
        input_dir: Directory of the corpus PDFs.
        out_dir: Output directory, where the artifacts are cached.
        config: Pipeline configuration, for chunking.

    Returns:
        The chunks.
    """
    docs_path = out_dir / "stage0_documents.json"
    chunks_path = out_dir / "stage1_chunks.json"

    if chunks_path.exists():
        LOGGER.info("Reusing cached chunks: %s", chunks_path)
        return chunking.load_chunks(chunks_path)

    if docs_path.exists():
        docs = ingestion.load_documents(docs_path)
    else:
        docs = ingestion.ingest_documents(input_dir=input_dir)
        ingestion.save_documents(docs_path, docs)

    chunks = chunking.chunk_documents(docs, config)
    chunking.save_chunks(chunks_path, chunks)
    return chunks


def _sample_chunks(
    chunks: list[chunking.ChunkRecord], target: int, seed: int
) -> list[chunking.ChunkRecord]:
    """Stratified sample of chunks, reproducible from `seed`.

    Each document gets a quota scaled by its size (larger documents up to
    double); front and back matter and each document's first chunk (the title
    page) are left out, and the sample is trimmed to `target`.

    Args:
        chunks: Every chunk of the corpus.
        target: Number of chunks wanted.
        seed: Seed of the sampling generator.

    Returns:
        The sampled chunks, sorted by id.
    """
    rng = random.Random(seed)

    by_doc: dict[str, list[chunking.ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    eligible_by_doc: dict[str, list[chunking.ChunkRecord]] = {}
    for doc_id, doc_chunks in by_doc.items():
        doc_chunks = sorted(doc_chunks, key=lambda c: c.chunk_index)
        eligible = [
            c
            for c in doc_chunks[1:]  # skip title-page chunk
            if not _SKIP_SECTION_RE.search(c.section_title or "")
        ]
        if not eligible:
            eligible = doc_chunks
        eligible_by_doc[doc_id] = eligible

    n_docs = len(eligible_by_doc)
    base_quota = max(1, target // max(1, n_docs))
    sampled: list[chunking.ChunkRecord] = []

    for doc_id in sorted(eligible_by_doc):
        eligible = eligible_by_doc[doc_id]
        # larger docs get up to double quota
        quota = base_quota if len(eligible) < 40 else base_quota * 2
        quota = min(quota, len(eligible))
        sampled.extend(rng.sample(eligible, quota))

    # trim to the target, reproducibly from the seed
    if len(sampled) > target:
        rng.shuffle(sampled)
        sampled = sampled[:target]
    return sorted(sampled, key=lambda c: c.chunk_id)


def _tally(
    triples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count the proposed entity types and predicates, with examples.

    Args:
        triples: Raw open-vocabulary triples.

    Returns:
        Type and predicate counts (most common first), up to eight example
        names per type, five example pairs per predicate, and the three most
        common type signatures per predicate.
    """
    type_counts: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    type_examples: dict[str, list[str]] = defaultdict(list)
    predicate_examples: dict[str, list[str]] = defaultdict(list)
    predicate_signatures: dict[str, Counter[str]] = defaultdict(Counter)

    for t in triples:
        s_type = _norm_type(str(t.get("subject_type", "")))
        o_type = _norm_type(str(t.get("object_type", "")))
        pred = _norm_predicate(str(t.get("predicate", "")))
        subj = str(t.get("subject", "")).strip()
        obj = str(t.get("object", "")).strip()
        if not (s_type and o_type and pred and subj and obj):
            continue

        for etype, name in ((s_type, subj), (o_type, obj)):
            type_counts[etype] += 1
            if len(type_examples[etype]) < 8 and name not in type_examples[etype]:
                type_examples[etype].append(name)

        predicate_counts[pred] += 1
        example = f"({subj}) -> ({obj})"
        if len(predicate_examples[pred]) < 5:
            predicate_examples[pred].append(example)
        predicate_signatures[pred][f"{s_type}->{o_type}"] += 1

    return {
        "type_counts": dict(type_counts.most_common()),
        "predicate_counts": dict(predicate_counts.most_common()),
        "type_examples": dict(type_examples),
        "predicate_examples": dict(predicate_examples),
        "predicate_signatures": {
            pred: dict(sig.most_common(3)) for pred, sig in predicate_signatures.items()
        },
    }


def _write_markdown_report(
    path: Path, tally: dict[str, Any], meta: dict[str, Any]
) -> None:
    """Write the ranked type and predicate tables for manual curation."""
    lines: list[str] = [
        "# Ontology discovery report",
        "",
        f"- chunks sampled: {meta['chunks_sampled']} (target {meta['sample_target']})",
        f"- chunks with triples: {meta['chunks_with_triples']}",
        f"- failed chunks: {meta['failed_chunks']}",
        f"- triples extracted: {meta['triples_total']}",
        f"- model: {meta['model']}",
        "",
        "## Entity types (by mention count)",
        "",
        "| # | Type | Count | Examples |",
        "|---|------|-------|----------|",
    ]
    for i, (etype, count) in enumerate(tally["type_counts"].items(), start=1):
        examples = "; ".join(tally["type_examples"].get(etype, [])[:4])
        lines.append(f"| {i} | {etype} | {count} | {examples} |")

    lines += [
        "",
        "## Predicates (by count)",
        "",
        "| # | Predicate | Count | Top signatures | Examples |",
        "|---|-----------|-------|----------------|----------|",
    ]
    for i, (pred, count) in enumerate(tally["predicate_counts"].items(), start=1):
        sigs = ", ".join(
            f"{sig} ({n})" for sig, n in tally["predicate_signatures"][pred].items()
        )
        examples = " · ".join(tally["predicate_examples"].get(pred, [])[:2])
        lines.append(f"| {i} | {pred} | {count} | {sigs} | {examples} |")

    lines += [
        "",
        "## Curation notes",
        "",
        "- Merge synonym types/predicates, drop one-offs, cap labels at ~12 and",
        "  predicates at ~30 for the pipeline config.",
        "- Types are English regardless of source language (prompt enforces it).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Sample the corpus, extract open-vocabulary triples and write the reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(ROOT / "documents" / "test 1"))
    parser.add_argument("--config", default=str(ROOT / "kg_pipeline" / "config.yaml"))
    parser.add_argument("--env-file", default=str(ROOT / "kg_pipeline" / ".env"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sample-target", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    load_dotenv(args.env_file, override=True)

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.output_dir.strip():
        out_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "kg_pipeline" / "artifacts" / f"discovery_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output dir: %s", out_dir)

    chunks = _load_or_run_stage01(Path(args.input_dir), out_dir, config)
    LOGGER.info("Corpus: %d chunks from %d docs", len(chunks), len({c.doc_id for c in chunks}))

    sampled = _sample_chunks(chunks, target=args.sample_target, seed=args.seed)
    (out_dir / "sampled_chunks.json").write_text(
        json.dumps([c.model_dump() for c in sampled], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Sampled %d chunks", len(sampled))

    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("VLLM_MODEL_NAME", "")
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
    if not model_name:
        raise ValueError("VLLM_MODEL_NAME missing")

    concurrency = max(1, int(os.getenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", "8")))

    async def _extract_one(
        client: AsyncOpenAI, semaphore: asyncio.Semaphore, chunk: chunking.ChunkRecord
    ) -> tuple[chunking.ChunkRecord, list[dict[str, Any]] | None]:
        """Extract from one chunk, retrying; ``None`` when every attempt fails."""
        prompt = _DISCOVERY_PROMPT.format(
            filename=chunk.filename,
            section_title=chunk.section_title,
            text=chunk.text,
        )
        for attempt in range(1, args.max_retries + 1):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=model_name,
                        temperature=0.0,
                        seed=args.seed,
                        max_tokens=2048,
                        messages=[{"role": "user", "content": prompt}],
                    )
                return chunk, _extract_json_array(
                    response.choices[0].message.content or "[]"
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "chunk %s attempt %d failed: %s", chunk.chunk_id, attempt, exc
                )
        return chunk, None

    async def _extract_all() -> list[tuple[chunking.ChunkRecord, list[dict[str, Any]] | None]]:
        """Extract from every sampled chunk, `concurrency` at a time."""
        semaphore = asyncio.Semaphore(concurrency)
        async with AsyncOpenAI(
            base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY", timeout=600
        ) as client:
            tasks = [_extract_one(client, semaphore, chunk) for chunk in sampled]
            results = []
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Discovery extraction",
                unit="chunk",
            ):
                results.append(await coro)
            return results

    results = asyncio.run(_extract_all())

    all_triples: list[dict[str, Any]] = []
    chunks_with_triples = 0
    failed = 0
    raw_path = out_dir / "raw_triples.jsonl"

    with raw_path.open("w", encoding="utf-8") as raw_fh:
        for chunk, triples in sorted(results, key=lambda pair: pair[0].chunk_id):
            if triples is None:
                failed += 1
                continue
            if triples:
                chunks_with_triples += 1
            for t in triples:
                t["chunk_id"] = chunk.chunk_id
                t["doc_id"] = chunk.doc_id
                raw_fh.write(json.dumps(t, ensure_ascii=False) + "\n")
            all_triples.extend(triples)

    tally = _tally(all_triples)
    meta = {
        "chunks_sampled": len(sampled),
        "sample_target": args.sample_target,
        "chunks_with_triples": chunks_with_triples,
        "failed_chunks": failed,
        "triples_total": len(all_triples),
        "model": model_name,
        "seed": args.seed,
        "input_dir": str(args.input_dir),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "discovery_report.json").write_text(
        json.dumps({"meta": meta, **tally}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(out_dir / "discovery_report.md", tally, meta)

    LOGGER.info(
        "Done: %d triples, %d entity types, %d predicates -> %s",
        len(all_triples),
        len(tally["type_counts"]),
        len(tally["predicate_counts"]),
        out_dir / "discovery_report.md",
    )


if __name__ == "__main__":
    main()
