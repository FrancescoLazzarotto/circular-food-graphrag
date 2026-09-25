"""Download AGROVOC's Italian/English label pairs into one local JSON lexicon.

Why not query the API per node: the graph has thousands of nodes and the public
Skosmos search endpoint answers one term per request. Paging the SPARQL endpoint
pulls every concept that carries both an Italian and an English label in a few
dozen requests, and the result is a file we can match against offline, exactly
and reproducibly, as many times as we want.

Output (``--output``, default ``artifacts/ontology/agrovoc_it_en.json``)::

    {
      "concepts": [
        {"uri": "http://aims.fao.org/aos/agrovoc/c_2810",
         "it": ["letame"], "en": ["manure", "farmyard manure"]},
        ...
      ],
      "meta": {...}
    }

Labels are kept verbatim; normalisation belongs to the matcher, not here.

Usage::

    conda run -n graphllm python scripts/gold/build_agrovoc_lexicon.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
ENDPOINT = "https://agrovoc.fao.org/sparql"
SKOS = "http://www.w3.org/2004/02/skos/core#"

logger = logging.getLogger("agrovoc_lexicon")

# One row per (concept, label). Both prefLabel and altLabel are collected: the
# alt labels are what catch the plural and the regional variant, which is
# exactly where a node name from free-text extraction tends to land.
QUERY = f"""
SELECT ?c ?lang ?label WHERE {{
  ?c a <{SKOS}Concept> .
  {{ ?c <{SKOS}prefLabel> ?label }} UNION {{ ?c <{SKOS}altLabel> ?label }}
  BIND(lang(?label) AS ?lang)
  FILTER(?lang IN ("it", "en"))
}}
ORDER BY ?c ?lang ?label
OFFSET %d
LIMIT %d
"""


def fetch_page(offset: int, limit: int, retries: int = 4) -> list[dict]:
    """One page of the paged query, as a list of SPARQL JSON bindings."""
    url = f"{ENDPOINT}?" + urlencode({"query": QUERY % (offset, limit)})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/sparql-results+json"})
            with urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["results"]["bindings"]
        except Exception as exc:  # network flake, 502 from the public endpoint
            last = exc
            wait = 5 * (attempt + 1)
            logger.warning("offset %d failed (%s), retry in %ds", offset, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"offset {offset} failed after {retries} attempts") from last


def main(argv: list[str] | None = None) -> int:
    """Download every concept with both an Italian and an English label.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO / "artifacts/ontology/agrovoc_it_en.json"))
    parser.add_argument("--page-size", type=int, default=10000)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=1.0, help="pause between pages")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    by_uri: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"it": [], "en": []})
    total_rows = 0
    for page in range(args.max_pages):
        offset = page * args.page_size
        rows = fetch_page(offset, args.page_size)
        if not rows:
            logger.info("page %d empty, done", page)
            break
        for row in rows:
            uri = row["c"]["value"]
            lang = row["lang"]["value"]
            label = row["label"]["value"]
            bucket = by_uri[uri][lang]
            if label not in bucket:
                bucket.append(label)
        total_rows += len(rows)
        logger.info("page %d: +%d rows, %d concepts so far", page, len(rows), len(by_uri))
        time.sleep(args.sleep)

    # A concept with labels in only one language cannot bridge anything.
    concepts = [
        {"uri": uri, "it": labels["it"], "en": labels["en"]}
        for uri, labels in sorted(by_uri.items())
        if labels["it"] and labels["en"]
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "concepts": concepts,
        "meta": {
            "endpoint": ENDPOINT,
            "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rows": total_rows,
            "concepts_seen": len(by_uri),
            "concepts_bilingual": len(concepts),
            "it_labels": sum(len(c["it"]) for c in concepts),
            "en_labels": sum(len(c["en"]) for c in concepts),
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", out)
    print(json.dumps(payload["meta"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
