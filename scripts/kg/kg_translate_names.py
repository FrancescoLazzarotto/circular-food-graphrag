"""Give every Italian-named node an English alias, using the local generator.

AGROVOC covers the domain vocabulary but reaches only a fraction of the nodes,
which leaves most of the graph unreachable from an English question: extraction
produces names like ``filetto di alici`` or ``misuratore di nutrienti`` that no
thesaurus carries. This pass asks the local Qwen for the English form of each
remaining name, and for names that are whole phrases, also for the short head
term — the phrase names are the ``substring`` bucket of ``kg_slot_ceiling.py``,
where a node exists but its name buries the concept inside a sentence.

Nodes already carrying ``ontology_uri`` are skipped: they have a vocabulary
label, which is better than a translation.

Two stages, so a crash never costs more than the last batch:

1. this script writes one JSONL line per batch to ``--output``
2. ``scripts/kg/kg_apply_translations.py`` reads that file and writes the graph

Usage::

    conda run -n graphllm python scripts/kg/kg_translate_names.py --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from kg_pipeline.utils import neo4j_env  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

logger = logging.getLogger("kg_translate_names")

SKIP_LABELS = ["Person", "Document", "DataValue", "Dataset", "NodeVec"]

FETCH = """
MATCH (n) WHERE n.name IS NOT NULL
  AND NONE(l IN labels(n) WHERE l IN $skip)
  AND n.ontology_uri IS NULL
RETURN elementId(n) AS id, toString(n.name) AS name, labels(n) AS labels
ORDER BY id
"""

SYSTEM = (
    "You translate entity names from a knowledge graph built on Italian and "
    "English documents about the circular economy for food. You answer with "
    "JSON only."
)

INSTRUCTIONS = """\
For each numbered name below, return an object with these fields:

  "i"    the number, copied
  "en"   the English form of the name. If the name is already English, or is a
         proper name that should not be translated (a brand, a project, a
         company, a place with no English exonym), return the string "SAME".
  "head" ONLY when the name is a phrase or a sentence rather than a term: the
         short English head concept it is about, at most four words. Otherwise
         omit this field.

Translate the term, do not explain it and do not expand acronyms. Keep the
grammatical number: a plural stays plural. Return a JSON array, one object per
input, nothing else.

Names:
%s
"""


def build_prompt(batch: list[dict]) -> str:
    """The user message listing the batch's names, numbered from 1."""
    listing = "\n".join(f'{i + 1}. {row["name"]}' for i, row in enumerate(batch))
    return INSTRUCTIONS % listing


def parse_reply(text: str, batch: list[dict]) -> list[dict]:
    """Map the model's array back onto the batch, dropping anything unusable."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        logger.warning("no JSON array in reply (%d chars)", len(text))
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("unparseable JSON: %s", exc)
        return []

    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("i", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(batch):
            continue
        english = str(item.get("en") or "").strip()
        head = str(item.get("head") or "").strip()
        if english.upper() == "SAME":
            english = ""
        if not english and not head:
            continue
        out.append({
            "id": batch[index]["id"],
            "name": batch[index]["name"],
            "en": english,
            "head": head,
        })
    return out


async def run_batch(client: AsyncOpenAI, model: str, batch: list[dict],
                    semaphore: asyncio.Semaphore, retries: int = 3) -> list[dict]:
    """Translate one batch, retrying; empty when every attempt fails."""
    async with semaphore:
        for attempt in range(retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": build_prompt(batch)},
                    ],
                    temperature=0,
                    max_tokens=4096,
                )
                return parse_reply(response.choices[0].message.content or "", batch)
            except Exception as exc:  # vLLM restart, timeout, malformed stream
                logger.warning("batch failed (%s), attempt %d/%d", exc, attempt + 1, retries)
                await asyncio.sleep(3 * (attempt + 1))
        return []


async def amain(args) -> int:
    """Translate every eligible node not yet in the output file.

    Args:
        args: Parsed command line.

    Returns:
        The exit status.
    """
    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    )
    with driver.session(database=args.database) as session:
        nodes = [dict(r) for r in session.run(FETCH, skip=SKIP_LABELS)]
    logger.info("%d nodes to translate", len(nodes))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists() and not args.restart:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        logger.info("resuming: %d already translated", len(done))
    elif args.restart:
        out_path.unlink(missing_ok=True)

    todo = [n for n in nodes if n["id"] not in done]
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    logger.info("%d names left in %d batches", len(todo), len(batches))
    if not batches:
        return 0

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [run_batch(client, args.model, batch, semaphore) for batch in batches]

    written = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for index, coro in enumerate(asyncio.as_completed(tasks), start=1):
            for row in await coro:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            if index % 10 == 0 or index == len(batches):
                logger.info("batch %d/%d, %d translations written", index, len(batches), written)

    logger.info("done: %d translations in %s", written, out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the translation.
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
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--restart", action="store_true", help="ignore an existing output file")
    parser.add_argument("--output", default=str(REPO / "artifacts/kg_v2/translations.jsonl"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
