"""Propose AGROVOC and EuroVoc identifiers for the ontology's concepts, by identical label.

Only a candidate: an identical label is not the same meaning (AGROVOC's
"transparency" is optical), so every row is ``proposed`` and waits for the
expert. ``exactMatch`` when the Italian and the English label both match the
same external concept, ``closeMatch`` when only one does. Rows the expert has
already decided are kept.

The lexicons are ``artifacts/ontology/agrovoc_it_en.json`` and
``artifacts/ontology/eurovoc_it_en.json`` (identifiers with their Italian and
English labels, from ``scripts/gold/build_agrovoc_lexicon.py`` and
``scripts/gold/build_eurovoc_lexicon.py``); ``ONTOLOGY_AGROVOC`` and
``ONTOLOGY_EUROVOC`` override the paths. A vocabulary whose lexicon is missing
is skipped with a message, and its proposed rows are kept as they are.

Usage::

    /mnt/storage/flazzarotto/venvs/ontology/bin/python ontology/scripts/link_external.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from common import ONTOLOGY_DIR, normalize, read_sheet, split_list, write_tsv

_LEXICONS = Path("/srv/projects/graphllm/experiments/graphRAGPipelineExp1/artifacts/ontology")
VOCABULARIES = {
    "AGROVOC": Path(os.environ.get("ONTOLOGY_AGROVOC", _LEXICONS / "agrovoc_it_en.json")),
    "EuroVoc": Path(os.environ.get("ONTOLOGY_EUROVOC", _LEXICONS / "eurovoc_it_en.json")),
}
MAPPINGS = ONTOLOGY_DIR / "external" / "mappings.tsv"
COLUMNS = ["concept_id", "vocabulary", "uri", "label", "match_type", "status", "notes"]
LIVE = {"proposed", "auto_accepted", "checked", "approved", "corrected"}


def lexicon_index(path: Path) -> dict[str, dict[str, set[str]]]:
    """Normalised label → language → URIs of the lexicon's concepts carrying it.

    Args:
        path: A lexicon file, ``{"concepts": [{"uri", "it", "en"}, ...]}``.

    Returns:
        The index.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, set[str]]] = {}
    for concept in data["concepts"]:
        for lang in ("it", "en"):
            for label in concept.get(lang) or []:
                index.setdefault(normalize(label), {}).setdefault(lang, set()).add(concept["uri"])
    return index


def candidates(concept: dict, index: dict, vocabulary: str = "AGROVOC") -> list[dict]:
    """External candidates for one concept in one vocabulary.

    Args:
        concept: A ``concepts.tsv`` row.
        index: Output of :func:`lexicon_index`.
        vocabulary: Name written in the ``vocabulary`` column.

    Returns:
        Mapping rows, one per external concept matched.
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
                "vocabulary": vocabulary,
                "uri": uri,
                "label": " / ".join(langs[lang] for lang in ("it", "en") if lang in langs),
                "match_type": "exactMatch" if both else "closeMatch",
                "status": "proposed",
                "notes": "etichetta identica in " + " e ".join(lang for lang in ("it", "en") if lang in langs)
                + ("" if len(hits) == 1 else f"; {len(hits)} concetti {vocabulary} con la stessa etichetta"),
            }
        )
    return rows


def main() -> None:
    with MAPPINGS.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle, delimiter="\t"))
    available = {name: path for name, path in VOCABULARIES.items() if path.exists()}
    for name in VOCABULARIES.keys() - available.keys():
        print(f"{name}: lessico assente in {VOCABULARIES[name]}, le sue proposte restano come sono")
    # A proposed row is rebuilt only for a vocabulary that is read now.
    kept = [
        r for r in existing
        if r.get("status") not in {"proposed", ""} or r.get("vocabulary") not in available
    ]
    taken = {(r["concept_id"], r["uri"]) for r in kept}
    concepts = [c for c in read_sheet("concepts") if c.get("status") in LIVE]
    proposed = []
    for name, path in available.items():
        index = lexicon_index(path)
        rows = [
            row
            for concept in concepts
            for row in candidates(concept, index, name)
            if (row["concept_id"], row["uri"]) not in taken
        ]
        print(f"{len(rows)} candidati {name} per {len({r['concept_id'] for r in rows})} concetti")
        proposed += rows
    write_tsv(MAPPINGS, COLUMNS, kept + proposed)


if __name__ == "__main__":
    main()
