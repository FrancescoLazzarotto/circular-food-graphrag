#!/usr/bin/env python3
"""Propose the title, authors and year of every registry document, for curators to check.

The demo names a document by the title stage 0 found on its first pages, and in
a corpus of reports, books, slides and magazines that is often a running
header, a publisher's line or the first sentence of a page. This asks the
generator to read the first pages of each document the registry includes and
propose what a library card would say: the title as printed, the authors, the
year. Nothing is applied. The proposals go to a CSV the curators correct, next
to what stage 0 extracted, and to a JSON in the format of
`product/corpus_catalog.json`, which the demo reads once someone has checked it.

    python scripts/corpus/document_cards.py --registry product/corpus_registry.csv \
        --stage0-run run_corpus_20261005 --out-dir <folder>
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from openai import APIError, AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
# The kg_pipeline package lives at the repo root and is not pip-installed.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_registry import readable  # noqa: E402
from kg_pipeline.utils import corpus_registry  # noqa: E402

LOGGER = logging.getLogger("document_cards")

# The cover and the first pages carry title, authors and year; past them the
# text only dilutes what the model has to find.
_HEAD_CHARS = 6000
_MIN_YEAR, _MAX_YEAR = 1900, 2100
_RETRY_TEMPERATURE = 0.3
# What a model writes in a string when it means "not shown".
_ABSENT = frozenset({"null", "none", "n/a", "unknown"})

SYSTEM = (
    "You catalogue documents for a library. You are given a document's file name and "
    "the text of its first pages. Reply with the document's title as it is printed on "
    "the cover or the first page, its authors (the people named as authors; the "
    "organisation that issued it when no person is named), and its year of publication. "
    "Copy the title in its own language and wording: do not translate, shorten or "
    "complete it. A magazine issue is titled with the magazine's name and the issue. "
    "Use null for anything the pages do not show; never guess from the file name alone. "
    'Reply with JSON only: {"title": string or null, "authors": [string], "year": integer or null}'
)

FIELDS = (
    "id_documento",
    "percorso",
    "tema",
    "titolo_estratto",
    "titolo_proposto",
    "autori_proposti",
    "anno_estratto",
    "anno_proposto",
    "nota",
)


def head_text(doc: dict[str, Any]) -> str:
    """The first pages of a stage 0 record, up to ``_HEAD_CHARS`` characters."""
    pages = [str(p.get("text", "")) for p in doc.get("page_chunks") or []]
    text = "\n\n".join(pages) if pages else str(doc.get("markdown_text", ""))
    return text[:_HEAD_CHARS]


def _clean(value: Any) -> str | None:
    """Collapse whitespace; ``None`` for anything but text the pages could show."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return None if text.lower() in _ABSENT else text or None


def parse_card(raw: str) -> dict[str, Any]:
    """Read the model's reply into a card, discarding anything malformed.

    Args:
        raw: The completion text.

    Returns:
        ``title`` (str or None), ``authors`` (list of str) and ``year`` (int or
        None).

    Raises:
        ValueError: If the reply holds no JSON object.
    """
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in {raw[:120]!r}")
    data = json.loads(match.group(0))
    title = _clean(data.get("title"))
    authors = data.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    authors = [a for a in map(_clean, authors) if a]
    year = data.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    if year is not None and not _MIN_YEAR <= year <= _MAX_YEAR:
        year = None
    return {"title": title, "authors": authors, "year": year}


async def propose(
    client: AsyncOpenAI, model: str, filename: str, text: str, retries: int = 2
) -> tuple[dict[str, Any] | None, str]:
    """Ask the model for one document's card.

    Args:
        client: OpenAI-compatible client of the generator.
        model: Served model name.
        filename: The document's file name, readable.
        text: Its first pages.
        retries: Further attempts after a reply that cannot be read.

    Returns:
        The card, or ``None`` with the reason when no attempt gave one.
    """
    error = ""
    for attempt in range(retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                # Greedy decoding repeats a malformed reply verbatim, so only
                # the first attempt is greedy.
                temperature=0.0 if attempt == 0 else _RETRY_TEMPERATURE,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"File name: {filename}\n\nFirst pages:\n{text}"},
                ],
            )
            return parse_card(response.choices[0].message.content or ""), ""
        except (ValueError, json.JSONDecodeError) as exc:
            error = f"risposta illeggibile: {exc}"
        except APIError as exc:
            error = f"errore del server: {exc}"
    return None, error


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Propose a card for every included document of the registry."""
    rows = [r for r in corpus_registry.load_registry(args.registry) if not r.escluso]
    stage0 = ROOT / "kg_pipeline" / "artifacts" / args.stage0_run / "stage0_documents.json"
    docs = {d["doc_id"]: d for d in json.loads(stage0.read_text(encoding="utf-8"))}
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    gate = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(row: corpus_registry.RegistryRow) -> dict[str, Any]:
        nonlocal done
        doc = docs.get(row.id_documento)
        card: dict[str, Any] | None = None
        note = ""
        if doc is None:
            note = f"non è in {args.stage0_run}"
        elif not head_text(doc).strip():
            note = "nessun testo nelle prime pagine"
        else:
            async with gate:
                card, note = await propose(
                    client, args.model, readable(row.filename), head_text(doc)
                )
        done += 1
        if done % 10 == 0 or done == len(rows):
            LOGGER.info("%d/%d documents", done, len(rows))
        return {
            "id_documento": row.id_documento,
            "filename": row.filename,
            "percorso": readable(row.percorso),
            "tema": row.tema,
            "titolo_estratto": (doc or {}).get("title") or "",
            "anno_estratto": (doc or {}).get("publication_year") or "",
            "card": card,
            "nota": note,
        }

    return await asyncio.gather(*(one(row) for row in rows))


def write_outputs(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Write the CSV for the curators and the catalog-format JSON.

    Args:
        results: One entry per document, from :func:`run`.
        out_dir: Folder for ``schede_proposte.csv`` and ``catalogo_proposto.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "schede_proposte.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(FIELDS)
        for r in results:
            card = r["card"] or {}
            writer.writerow(
                [
                    r["id_documento"],
                    r["percorso"],
                    r["tema"],
                    r["titolo_estratto"],
                    card.get("title") or "",
                    " | ".join(card.get("authors") or []),
                    r["anno_estratto"],
                    card.get("year") or "",
                    r["nota"],
                ]
            )
    catalog = {
        "_README": (
            "Proposte del modello dalle prime pagine (scripts/corpus/document_cards.py), "
            "NON ancora controllate: da correggere in schede_proposte.csv prima dell'uso."
        ),
        "titles": {r["filename"]: r["card"]["title"] for r in results if r["card"] and r["card"]["title"]},
        "authors": {
            r["filename"]: r["card"]["authors"] for r in results if r["card"] and r["card"]["authors"]
        },
    }
    (out_dir / "catalogo_proposto.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Propose the cards and write them; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--stage0-run", required=True, help="run directory under kg_pipeline/artifacts")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    results = asyncio.run(run(args))
    write_outputs(results, args.out_dir)
    failed = [r for r in results if r["card"] is None]
    LOGGER.info(
        "%d cards proposed, %d without: %s",
        len(results) - len(failed),
        len(failed),
        ", ".join(f"{r['id_documento']} ({r['nota']})" for r in failed[:10]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
