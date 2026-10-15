"""Propose AGROVOC identifiers for the ontology's concepts, by identical label.

Only a candidate: an identical label is not the same meaning (AGROVOC's
"transparency" is optical), so every row is ``proposed`` and waits for the
expert. ``exactMatch`` when the Italian and the English label both match the
same AGROVOC concept, ``closeMatch`` when only one does. Rows the expert has
already decided are kept.

The lexicon is ``artifacts/ontology/agrovoc_it_en.json`` (identifiers with
their Italian and English labels); ``ONTOLOGY_AGROVOC`` overrides the path.

Usage::

    /mnt/storage/flazzarotto/venvs/ontology/bin/python ontology/scripts/link_external.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from common import ONTOLOGY_DIR, normalize, read_sheet, split_list, write_tsv

AGROVOC_PATH = Path(
    os.environ.get(
        "ONTOLOGY_AGROVOC",
        "/srv/projects/graphllm/experiments/graphRAGPipelineExp1/artifacts/ontology/agrovoc_it_en.json",
    )
)
MAPPINGS = ONTOLOGY_DIR / "external" / "mappings.tsv"
COLUMNS = ["concept_id", "vocabulary", "uri", "label", "match_type", "status", "notes"]
LIVE = {"proposed", "auto_accepted", "approved", "corrected"}


def agrovoc_index() -> dict[str, dict[str, set[str]]]:
    """Normalised label → language → AGROVOC URIs carrying it.

    Returns:
        The index.
    """
    data = json.loads(AGROVOC_PATH.read_text(encoding="utf-8"))
    index: dict[str, dict[str, set[str]]] = {}
    for concept in data["concepts"]:
        for lang in ("it", "en"):
            for label in concept.get(lang) or []:
                index.setdefault(normalize(label), {}).setdefault(lang, set()).add(concept["uri"])
    return index


def candidates(concept: dict, index: dict) -> list[dict]:
    """AGROVOC candidates for one concept.

    Args:
        concept: A ``concepts.tsv`` row.
        index: Output of :func:`agrovoc_index`.

    Returns:
        Mapping rows, one per AGROVOC concept matched.
    """
    hits: dict[str, dict[str, str]] = {}
    for lang in ("it", "en"):
        forms = [concept.get(f"label_{lang}", "")] + split_list(concept.get(f"synonyms_{lang}", ""))
        for form in filter(None, forms):
            for uri in index.get(normalize(form), {}).get(lang, set()):
                hits.setdefault(uri, {}).setdefault(lang, form)
    rows = []
    for uri, langs in sorted(hits.items()):
        both = {"it", "en"} <= set(langs)
        rows.append(
            {
                "concept_id": concept["id"],
                "vocabulary": "AGROVOC",
                "uri": uri,
                "label": " / ".join(langs[lang] for lang in ("it", "en") if lang in langs),
                "match_type": "exactMatch" if both else "closeMatch",
                "status": "proposed",
                "notes": "etichetta identica in " + " e ".join(lang for lang in ("it", "en") if lang in langs)
                + ("" if len(hits) == 1 else f"; {len(hits)} concetti AGROVOC con la stessa etichetta"),
            }
        )
    return rows


def main() -> None:
    index = agrovoc_index()
    with MAPPINGS.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle, delimiter="\t"))
    decided = [r for r in existing if r.get("status") not in {"proposed", ""}]
    taken = {(r["concept_id"], r["uri"]) for r in decided}
    proposed = [
        row
        for concept in read_sheet("concepts")
        if concept.get("status") in LIVE
        for row in candidates(concept, index)
        if (row["concept_id"], row["uri"]) not in taken
    ]
    write_tsv(MAPPINGS, COLUMNS, decided + proposed)
    print(f"{len(proposed)} candidati AGROVOC per {len({r['concept_id'] for r in proposed})} concetti")


if __name__ == "__main__":
    main()
