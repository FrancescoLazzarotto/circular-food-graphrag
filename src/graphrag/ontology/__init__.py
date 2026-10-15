"""Exact lookups on the framework ontology (3C, R strategies and their versions).

The ontology is built offline from the sheets in ``ontology/sources/`` into
``ontology/build/ontology.json``; this package only reads that file. Every
result carries the document and page it comes from, and the review status of
each item, so a caller can say when something has not been checked by the
expert yet.
"""

from .consult import (
    Ontology,
    OntologyNotBuilt,
    concetto,
    confronta,
    framework,
    load,
    piu_specifici,
    verifica,
)

__all__ = [
    "Ontology",
    "OntologyNotBuilt",
    "concetto",
    "confronta",
    "framework",
    "load",
    "piu_specifici",
    "verifica",
]
