"""Fetch multilingual labels (IT/EN) from AGROVOC and ChEBI for gold entity URIs.

Pipeline-independent: labels come from the external vocabularies only, never from
our own KG registry, so alt_labels cannot be tuned to any pipeline's output.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

GOLD = Path(__file__).resolve().parents[1] / "gold.json"
OUT = Path(__file__).with_name("vocab_labels.json")

AGROVOC_API = "https://agrovoc.fao.org/browse/rest/v1/agrovoc/data"
CHEBI_API = "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms"

LANGS = {"en", "it"}


def _get(url: str, timeout: int = 30) -> dict | None:
    """GET `url` as JSON; None on any error, which is printed."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"    ERR {exc} :: {url[:90]}")
        return None


def fetch_agrovoc(uri: str) -> dict[str, list[str]]:
    """Return {'pref': [...], 'alt': [...]} for IT+EN labels of an AGROVOC concept."""
    url = f"{AGROVOC_API}?uri={urllib.parse.quote(uri, safe='')}&format=application/json"
    data = _get(url)
    if not data:
        return {"pref": [], "alt": []}
    pref: list[str] = []
    alt: list[str] = []
    for node in data.get("graph", []):
        if node.get("uri") != uri:
            continue
        for key, bucket in (("prefLabel", pref), ("altLabel", alt), ("hiddenLabel", alt)):
            vals = node.get(key, [])
            if isinstance(vals, dict):
                vals = [vals]
            for v in vals:
                if isinstance(v, dict) and v.get("lang") in LANGS:
                    bucket.append(v["value"])
    return {"pref": pref, "alt": alt}


def fetch_chebi(uri: str) -> dict[str, list[str]]:
    """Return {'pref': [...], 'alt': [...]} for a ChEBI term via the OLS4 API.

    ChEBI is English-only; synonyms still matter (plurals, chemical variants).
    """
    double_enc = urllib.parse.quote(urllib.parse.quote(uri, safe=""), safe="")
    data = _get(f"{CHEBI_API}/{double_enc}")
    if not data:
        return {"pref": [], "alt": []}
    pref = [data["label"]] if data.get("label") else []
    alt = list(data.get("synonyms") or [])
    return {"pref": pref, "alt": alt}


def main() -> None:
    """Fetch labels for every external URI in the gold; write vocab_labels.json."""
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    uris: dict[str, str] = {}
    for q in gold["queries"]:
        for e in q["expected_entities"]:
            uri = e.get("uri") or ""
            if uri.startswith("http"):
                uris[uri] = e["label"]

    print(f"{len(uris)} distinct external URIs in gold")
    out: dict[str, dict] = {}
    for uri, label in sorted(uris.items()):
        if "agrovoc" in uri:
            src, res = "AGROVOC", fetch_agrovoc(uri)
        elif "obo/CHEBI" in uri or "chebi" in uri.lower():
            src, res = "ChEBI", fetch_chebi(uri)
        else:
            src, res = "OTHER", {"pref": [], "alt": []}
        out[uri] = {"gold_label": label, "source": src, **res}
        print(f"  [{src:7s}] {label:34s} pref={res['pref']} alt={res['alt'][:6]}")
        time.sleep(0.4)

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
