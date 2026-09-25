"""Add edges between entities the graph already has, one chunk at a time.

The graph is a forest, not a network: about one edge per node, most nodes with
degree ≤ 1, a small two-hop neighbourhood. The cause is upstream of any repair —
stage 3 extracts triples chunk by chunk with no inventory of the entities
already in the graph, so almost every edge comes from a single chunk and
phrase-shaped entities never re-attach to anything.

This pass works on that directly. For each chunk it finds which canonical
entities are actually mentioned in the text, hands the model *that closed list*,
and asks only for relations between pairs drawn from it. No entity is created,
so no new fragmentation is possible; the only thing that can change is the
number of edges between nodes that already exist.

Every edge written carries ``extraction_method: 'densification'``, so the whole
pass can be deleted with one Cypher statement and the strategies can be measured
with and without it.

Two stages, like the translation pass: this script writes candidate triples to
JSONL, and ``--apply`` writes them to the graph after validation.

Usage::

    conda run -n graphllm python scripts/kg/kg_densify.py --base-url http://localhost:8001/v1
    conda run -n graphllm python scripts/kg/kg_densify.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from kg_pipeline.utils import neo4j_env  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

logger = logging.getLogger("kg_densify")

# Default chunk folders. Chunk ids are unique only within a run, and another run
# reuses them for other passages, so a graph built by another run has to be
# densified from that run's folder (--chunks-dir); otherwise every new edge
# cites a passage that says something else.
CHUNK_DIRS = [
    REPO / "kg_pipeline/artifacts/run_full_circular_20260707",
    REPO / "kg_pipeline/artifacts/run_fix2docs_20260710",
]
VOCAB_PATH = REPO / "kg_pipeline/relation_vocab_circular_v1_draft.json"

# People and documents are out of the inventory. With them in, most of the
# budget goes on degenerate author links — `Maria Piochi HAS_MEMBER Piochi, M.`,
# `Franceschini HAS_MEMBER Cinzia Franceschini` — which are two surface forms of
# one person, not a relation. Authorship edges already exist from stage 3
# anyway; the sparsity this pass targets is between concepts, processes and
# materials.
SKIP_LABELS = ["Person", "Document", "DataValue", "Dataset", "NodeVec"]

FETCH_ENTITIES = (
    "MATCH (n) WHERE n.name IS NOT NULL AND NONE(l IN labels(n) WHERE l IN $skip) "
    "RETURN elementId(n) AS id, toString(n.name) AS name, labels(n) AS labels, "
    "coalesce(n.aliases, []) AS aliases"
)

EXISTING_EDGES = (
    "MATCH (a)-[r]->(b) WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
    "RETURN toString(a.name) AS a, type(r) AS t, toString(b.name) AS b"
)

WRITE_TEMPLATE = (
    "UNWIND $rows AS row "
    "MATCH (a) WHERE elementId(a) = row.src "
    "MATCH (b) WHERE elementId(b) = row.dst "
    "MERGE (a)-[r:%s]->(b) "
    "ON CREATE SET r = row.props "
    "ON MATCH SET r.mention_count = coalesce(r.mention_count, 1) + 1"
)

SYSTEM = (
    "You extract relations from documents about the circular economy for food. "
    "You answer with JSON only."
)

INSTRUCTIONS = """\
Below is a passage and a closed list of entities that appear in it.

Return the relations the passage states between pairs of entities **from the
list**. Rules:

- both subject and object must be copied verbatim from the list; never invent an
  entity, never use a name that is not in the list
- the predicate must be one of: %s
- only relations the passage actually asserts. If the passage merely mentions
  two entities near each other, that is not a relation
- do not repeat the same triple twice

Return a JSON array of objects {"s": ..., "p": ..., "o": ...}, nothing else. An
empty array is a valid and expected answer.

ENTITIES:
%s

PASSAGE:
%s
"""


def scan_key(text: str) -> str:
    """Key for finding a name inside running text.

    Deliberately more aggressive than ``evalkit``'s ``match_key``: punctuation
    and hyphens become spaces so that ``ciclo a cascata`` is found in
    ``…il ciclo a cascata, che…``. Only used for mention detection, never for
    scoring.
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", folded)).strip()


def load_chunks(directories: list[Path]) -> list[dict]:
    """The stage 1 chunks of the given run folders; a missing file is skipped."""
    chunks: list[dict] = []
    for directory in directories:
        path = directory / "stage1_chunks.json"
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


def build_index(entities: list[dict], min_chars: int) -> dict[str, list[dict]]:
    """Scan key -> entities carrying it, from names and aliases alike."""
    index: dict[str, list[dict]] = {}
    for entity in entities:
        forms = [entity["name"], *[str(a) for a in entity["aliases"]]]
        for form in forms:
            key = scan_key(form)
            if len(key) < min_chars or not key:
                continue
            if len(key.split()) > 8:
                continue  # a sentence-shaped name is never found verbatim anyway
            index.setdefault(key, [])
            if entity not in index[key]:
                index[key].append(entity)
    return index


def mentions(text: str, index: dict[str, list[dict]], max_entities: int) -> list[dict]:
    """Entities whose name or alias occurs in the chunk, term-shaped names first.

    Scans every n-gram up to 8 words rather than searching thousands of names
    in the text: the chunk is a few hundred words, so this is the cheap
    direction.

    Args:
        text: The chunk text.
        index: As built by `build_index`.
        max_entities: Keep at most this many.

    Returns:
        The entities found, ranked.
    """
    words = scan_key(text).split()
    found: dict[str, dict] = {}
    for size in range(8, 0, -1):
        for start in range(0, len(words) - size + 1):
            key = " ".join(words[start:start + size])
            for entity in index.get(key, ()):
                found.setdefault(entity["id"], entity)
    # Term-shaped names first, sentence-shaped ones only as filler. Ranking by
    # raw length would fill all 25 slots with the phrase entities the extractor
    # produces ('una progettualità di…'), crowding out the concepts the pass is
    # meant to connect. Within each group, longer is more specific.
    def rank(entity: dict) -> tuple[int, int]:
        """Term-shaped names before sentence-shaped ones, then longest first."""
        words = len(entity["name"].split())
        return (1 if words > 5 else 0, -len(entity["name"]))

    return sorted(found.values(), key=rank)[:max_entities]


def parse_reply(text: str) -> list[dict]:
    """The ``{s, p, o}`` objects of the reply's JSON array; empty if there is none."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in items:
        if isinstance(item, dict) and item.get("s") and item.get("p") and item.get("o"):
            out.append({"s": str(item["s"]), "p": str(item["p"]), "o": str(item["o"])})
    return out


async def run_chunk(client: AsyncOpenAI, model: str, chunk: dict, entities: list[dict],
                    predicates: list[str], semaphore: asyncio.Semaphore,
                    max_chars: int, max_tokens: int, retries: int = 2) -> dict:
    """Ask for the relations one chunk states between its known entities.

    Subject and object must be names from the list and the predicate must be
    in the vocabulary; anything else, and any self-loop, is rejected.

    Args:
        client: The generator client.
        model: Served model name.
        chunk: The chunk record.
        entities: Its known entities.
        predicates: The relation vocabulary.
        semaphore: Bounds the concurrent requests.
        max_chars: Chunk text cut.
        max_tokens: Reply budget.
        retries: Extra attempts after a failed call.

    Returns:
        The chunk id, its document and pages, the valid triples and the
        number rejected.
    """
    listing = "\n".join(f"- {e['name']}" for e in entities)
    prompt = INSTRUCTIONS % (", ".join(predicates), listing, chunk["text"][:max_chars])
    by_name = {e["name"]: e for e in entities}
    allowed = set(predicates)

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                raw = parse_reply(response.choices[0].message.content or "")
                break
            except Exception as exc:
                logger.warning("chunk %s failed (%s)", chunk["chunk_id"], exc)
                await asyncio.sleep(3 * (attempt + 1))
        else:
            return {"chunk_id": chunk["chunk_id"], "triples": [], "rejected": 0}

    triples, rejected = [], 0
    for item in raw:
        subject, predicate, obj = item["s"].strip(), item["p"].strip().upper(), item["o"].strip()
        if subject not in by_name or obj not in by_name or predicate not in allowed:
            rejected += 1
            continue
        if subject == obj:
            rejected += 1  # self-loops: the graph has zero and should keep zero
            continue
        triples.append({
            "src": by_name[subject]["id"], "dst": by_name[obj]["id"],
            "subject": subject, "object": obj, "predicate": predicate,
        })
    return {
        "chunk_id": chunk["chunk_id"],
        "source_doc": chunk.get("filename"),
        "page_range": chunk.get("page_range"),
        "triples": triples,
        "rejected": rejected,
    }


async def extract(args, chunks, entities, predicates, out_path: Path) -> None:
    """Run every chunk with at least two known entities; append results to `out_path`.

    Resumes after the chunks already in the file unless ``--restart``, and
    takes only this shard's chunks (``--shard-index`` of ``--shard-count``).

    Args:
        args: Parsed command line.
        chunks: Chunk records.
        entities: Graph entities eligible for the inventory.
        predicates: The relation vocabulary.
        out_path: JSONL output of this shard.
    """
    index = build_index(entities, args.min_chars)
    logger.info("mention index: %d keys from %d entities", len(index), len(entities))

    done: set[str] = set()
    if out_path.exists() and not args.restart:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["chunk_id"])
        logger.info("resuming: %d chunks already processed", len(done))
    elif args.restart:
        out_path.unlink(missing_ok=True)

    work = []
    skipped_thin = 0
    for position, chunk in enumerate(chunks):
        if position % args.shard_count != args.shard_index:
            continue
        if chunk["chunk_id"] in done:
            continue
        found = mentions(chunk["text"], index, args.max_entities)
        if len(found) < 2:
            skipped_thin += 1
            continue
        work.append((chunk, found))
    logger.info("%d chunks to process, %d skipped for fewer than 2 known entities",
                len(work), skipped_thin)
    if not work:
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [run_chunk(client, args.model, chunk, found, predicates, semaphore,
                       args.max_chars, args.max_tokens)
             for chunk, found in work]

    kept = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for index_done, coro in enumerate(asyncio.as_completed(tasks), start=1):
            result = await coro
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            kept += len(result["triples"])
            if index_done % 25 == 0 or index_done == len(tasks):
                logger.info("chunk %d/%d, %d triples kept", index_done, len(tasks), kept)


def apply(args, driver, records: list[dict], existing: set[tuple]) -> dict:
    """Deduplicate the extracted triples and, with ``--apply``, write them.

    Args:
        args: Parsed command line.
        driver: Neo4j driver.
        records: JSONL records of every shard.
        existing: ``(subject, type, object)`` already in the graph.

    Returns:
        New, written and duplicate counts, and the new edges per type.
    """
    by_type: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    duplicates = 0
    for record in records:
        for triple in record["triples"]:
            signature = (triple["subject"], triple["predicate"], triple["object"])
            if signature in existing or signature in seen:
                duplicates += 1
                continue
            seen.add(signature)
            by_type.setdefault(triple["predicate"], []).append({
                "src": triple["src"], "dst": triple["dst"],
                "props": {
                    "subject": triple["subject"], "object": triple["object"],
                    "extraction_method": "densification",
                    "chunk_id": record["chunk_id"],
                    "source_doc": record.get("source_doc"),
                    "page_range": record.get("page_range"),
                    "confidence": 0.8,
                    "mention_count": 1,
                },
            })

    written = 0
    if args.apply:
        with driver.session(database=args.database) as session:
            for rel_type, rows in by_type.items():
                for start in range(0, len(rows), 500):
                    session.run(WRITE_TEMPLATE % rel_type,
                                rows=rows[start:start + 500]).consume()
                written += len(rows)
                logger.info("  %s: %d", rel_type, len(rows))
    return {
        "new_edges": sum(len(v) for v in by_type.values()),
        "written": written,
        "duplicates_skipped": duplicates,
        "per_type": {k: len(v) for k, v in sorted(by_type.items(), key=lambda x: -len(x[1]))},
    }


def main(argv: list[str] | None = None) -> int:
    """Extract candidate edges, then report or (with ``--apply``) write them.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="bolt://localhost:7689")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="staging-kg-v2")
    parser.add_argument("--database", default=None)
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-entities", type=int, default=25)
    parser.add_argument("--min-chars", type=int, default=5,
                        help="ignore names shorter than this when detecting mentions")
    parser.add_argument("--max-chars", type=int, default=6000, help="chunk text cut")
    parser.add_argument("--limit", type=int, default=0, help="process only N chunks (probe)")
    # Two generators are available (GPU 0 on 8000, GPU 1 on 8001). Splitting the
    # corpus by position lets both run at once; each shard keeps its own output
    # file and `apply` deduplicates by (subject, predicate, object) anyway.
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--apply", action="store_true", help="write the edges")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--output", default=str(REPO / "artifacts/kg_v2/densify.jsonl"))
    parser.add_argument("--chunks-dir", action="append", type=Path, default=None,
                        help="run folder holding the stage1_chunks.json the graph was "
                             "built from; repeatable (default: CHUNK_DIRS)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    predicates = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )
    with driver.session(database=args.database) as session:
        entities = [dict(r) for r in session.run(FETCH_ENTITIES, skip=SKIP_LABELS)]
        existing = {(r["a"], r["t"], r["b"]) for r in session.run(EXISTING_EDGES)}
    logger.info("%d entities, %d existing edges", len(entities), len(existing))

    chunks = load_chunks(args.chunks_dir or CHUNK_DIRS)
    if args.limit:
        chunks = chunks[: args.limit]
    logger.info("%d chunks", len(chunks))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.apply:
        asyncio.run(extract(args, chunks, entities, predicates, out_path))
    if args.extract_only:
        return 0

    shard_files = sorted(out_path.parent.glob(out_path.stem + "*.jsonl"))
    records = [json.loads(line) for path in shard_files
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("apply reads %d file(s): %s", len(shard_files),
                ", ".join(p.name for p in shard_files))
    stats = apply(args, driver, records, existing)
    stats.update({
        "measured": datetime.now().isoformat(timespec="seconds"),
        "applied": bool(args.apply),
        "chunks_processed": len(records),
        "chunks_with_triples": sum(1 for r in records if r["triples"]),
        "triples_rejected_by_validator": sum(r.get("rejected", 0) for r in records),
    })
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_path.parent / f"densify_{stamp}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "per_type"},
                     ensure_ascii=False, indent=1))
    print("per_type:", json.dumps(stats["per_type"], ensure_ascii=False))
    if not args.apply:
        print("\nno --apply: nothing written to the graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
