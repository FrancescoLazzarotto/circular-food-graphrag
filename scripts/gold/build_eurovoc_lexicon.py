"""Download EuroVoc's Italian/English label pairs into one local JSON lexicon.

The same lexicon shape as ``build_agrovoc_lexicon.py``, so the ontology's
linker reads both alike. The Publications Office endpoint serves many schemes
besides EuroVoc; the query keeps the concepts of the EuroVoc thesaurus
(``http://eurovoc.europa.eu/100141``) and leaves out the deprecated ones,
whose labels survive in the data after a concept is retired.

Output (``--output``, default ``artifacts/ontology/eurovoc_it_en.json``)::

    {
      "concepts": [
        {"uri": "http://eurovoc.europa.eu/4038",
         "it": ["economia circolare"], "en": ["circular economy"]},
        ...
      ],
      "meta": {...}
    }

Usage::

    conda run -n graphllm python scripts/gold/build_eurovoc_lexicon.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from build_agrovoc_lexicon import SKOS, fetch_page

REPO = Path(__file__).resolve().parents[2]
ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
SCHEME = "http://eurovoc.europa.eu/100141"
OWL = "http://www.w3.org/2002/07/owl#"

logger = logging.getLogger("eurovoc_lexicon")

# Virtuoso sorts at most 10 000 rows per request, offset included, and EuroVoc
# has more labels than that: the concepts (fewer than 10 000) come first, then
# their labels in batches small enough to stay under the cap.
CONCEPTS = f"""
SELECT ?c WHERE {{
  ?c <{SKOS}inScheme> <{SCHEME}> .
  FILTER NOT EXISTS {{ ?c <{OWL}deprecated> true }}
}}
ORDER BY ?c
OFFSET %d
LIMIT %d
"""
LABELS = f"""
SELECT ?c ?lang ?label WHERE {{
  VALUES ?c {{ __URIS__ }}
  {{ ?c <{SKOS}prefLabel> ?label }} UNION {{ ?c <{SKOS}altLabel> ?label }}
  BIND(lang(?label) AS ?lang)
  FILTER(?lang IN ("it", "en"))
}}
ORDER BY ?c ?lang ?label
OFFSET %d
LIMIT %d
"""
CAP = 10000


def main(argv: list[str] | None = None) -> int:
    """Download every EuroVoc concept with both an Italian and an English label.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default=str(REPO / "artifacts/ontology/eurovoc_it_en.json"))
    parser.add_argument("--batch", type=int, default=100, help="concepts per label request")
    parser.add_argument("--sleep", type=float, default=1.0, help="pause between requests")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    uris = [row["c"]["value"] for row in fetch_page(0, CAP, endpoint=ENDPOINT, query=CONCEPTS)]
    if len(uris) >= CAP:
        raise RuntimeError(f"{len(uris)} concepts: the list reached the endpoint's cap")
    logger.info("%d concepts", len(uris))
    by_uri: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"it": [], "en": []})
    total_rows = 0
    for start in range(0, len(uris), args.batch):
        batch = " ".join(f"<{uri}>" for uri in uris[start:start + args.batch])
        rows = fetch_page(0, CAP, endpoint=ENDPOINT, query=LABELS.replace("__URIS__", batch))
        if len(rows) >= CAP:
            raise RuntimeError(f"batch at {start} reached the endpoint's cap: lower --batch")
        for row in rows:
            bucket = by_uri[row["c"]["value"]][row["lang"]["value"]]
            if row["label"]["value"] not in bucket:
                bucket.append(row["label"]["value"])
        total_rows += len(rows)
        logger.info("concepts %d-%d: +%d rows", start, start + args.batch, len(rows))
        time.sleep(args.sleep)

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
            "scheme": SCHEME,
            "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rows": total_rows,
            "concepts_seen": len(uris),
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
