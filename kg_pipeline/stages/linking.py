"""Stage 5: add document-mention and cross-document alias links to the triples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from kg_pipeline.models.types import CanonicalEntityRecord, DocumentRecord, KGTriple


def _load_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented UTF-8 JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _document_props(doc: DocumentRecord) -> dict[str, Any]:
    """Node properties of a ``:Document`` node.

    Args:
        doc: Source document.

    Returns:
        ``name`` (the file name), ``filename``, ``page_count``, ``title`` and
        ``publication_year``.
    """
    return {
        "name": doc.filename,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "title": doc.title,
        "publication_year": doc.publication_year,
    }


def _normalize_key(value: str) -> str:
    """Lower-case ``value`` and collapse whitespace."""
    return " ".join(value.strip().lower().split())


def _triple_key(triple: KGTriple) -> tuple[str, str, str]:
    """Normalised ``(subject, predicate, object)`` key used to count repeats."""
    return (
        _normalize_key(triple.subject),
        _normalize_key(triple.predicate),
        _normalize_key(triple.object),
    )


def _apply_mention_counts(triples: list[KGTriple]) -> list[KGTriple]:
    """Set ``mention_count`` on each triple's relationship properties.

    The count is the number of triples in ``triples`` sharing the same
    normalised subject, predicate and object. An existing ``mention_count`` is
    left unchanged.

    Args:
        triples: Triples to annotate; modified in place.

    Returns:
        The same list.
    """
    counts = Counter(_triple_key(triple) for triple in triples)
    for triple in triples:
        key = _triple_key(triple)
        rel_props = dict(triple.relationship_properties)
        rel_props.setdefault("mention_count", int(counts.get(key, 1)))
        triple.relationship_properties = rel_props
    return triples


def add_cross_document_links(
    triples: list[KGTriple],
    registry: dict[str, CanonicalEntityRecord],
    documents: list[DocumentRecord],
    include_mentioned_in: bool = True,
) -> list[KGTriple]:
    """Add system-generated link triples and mention counts.

    Two kinds of triples are appended, both with
    ``extraction_method="system_linking"``:

    * ``MENTIONED_IN`` from each subject and object to its source
      ``:Document``, once per entity, document, chunk and page range
      (only when ``include_mentioned_in`` is set);
    * ``SAME_AS`` from each alias to its canonical name, for registry entities
      whose aliases come from at least two documents.

    Finally every triple receives a ``mention_count`` (see
    :func:`_apply_mention_counts`).

    Args:
        triples: Resolved triples; their relationship properties are updated
            in place.
        registry: Canonical entities from stage 4, keyed by canonical name.
        documents: Parsed documents, matched to triples by file name.
        include_mentioned_in: Whether to add ``MENTIONED_IN`` triples.

    Returns:
        The input triples followed by the added link triples.
    """
    doc_map = {doc.filename: doc for doc in documents}
    linked: list[KGTriple] = list(triples)
    seen_mention_edges: set[tuple[str, str, str, str]] = set()

    if include_mentioned_in:
        for triple in triples:
            source_doc = str(
                triple.relationship_properties.get("source_doc", "")
            ).strip()
            chunk_id = str(triple.relationship_properties.get("chunk_id", "")).strip()
            page_range = str(
                triple.relationship_properties.get("page_range", "")
            ).strip()

            if source_doc and source_doc in doc_map:
                doc_props = _document_props(doc_map[source_doc])

                for entity_name, entity_labels, entity_props in (
                    (triple.subject, triple.subject_labels, triple.subject_properties),
                    (triple.object, triple.object_labels, triple.object_properties),
                ):
                    edge_key = (entity_name, source_doc, chunk_id, page_range)
                    if edge_key in seen_mention_edges:
                        continue
                    seen_mention_edges.add(edge_key)

                    linked.append(
                        KGTriple.model_validate(
                            {
                                "subject": entity_name,
                                "predicate": "MENTIONED_IN",
                                "object": source_doc,
                                "subject_labels": entity_labels or ["Concept"],
                                "object_labels": ["Document"],
                                "subject_properties": dict(entity_props),
                                "object_properties": dict(doc_props),
                                "relationship_properties": {
                                    "source_doc": source_doc,
                                    "extraction_method": "system_linking",
                                    "chunk_id": chunk_id,
                                    "page_range": page_range,
                                },
                            }
                        )
                    )

    for canonical_name, record in registry.items():
        docs_for_concept = set()
        for alias, src_docs in record.alias_sources.items():
            for d in src_docs:
                docs_for_concept.add(d)

        if len(docs_for_concept) < 2:
            continue

        for alias in record.aliases:
            if alias == canonical_name:
                continue

            alias_docs = record.alias_sources.get(alias, [])
            source_doc = alias_docs[0] if alias_docs else "registry"

            linked.append(
                KGTriple.model_validate(
                    {
                        "subject": alias,
                        "predicate": "SAME_AS",
                        "object": canonical_name,
                        "subject_labels": record.labels or ["Concept"],
                        "object_labels": record.labels or ["Concept"],
                        "subject_properties": {"name": alias},
                        "object_properties": dict(record.merged_properties),
                        "relationship_properties": {
                            "source_doc": source_doc,
                            "extraction_method": "system_linking",
                        },
                    }
                )
            )

    return _apply_mention_counts(linked)


def save_triples(path: Path, triples: list[KGTriple]) -> None:
    """Write triples to a JSON file.

    Args:
        path: Output file; parent directories are created.
        triples: Triples to write.
    """
    _save_json(path, [triple.as_dict() for triple in triples])


def check_triples(path: Path, triples: list[KGTriple]) -> None:
    """Write a summary of triple and predicate counts to a JSON file.

    Args:
        path: Output file; parent directories are created.
        triples: Triples to summarise.
    """
    predicate_counts = Counter(triple.predicate for triple in triples)
    report = {
        "triples_count": len(triples),
        "distinct_predicates": len(predicate_counts),
        "predicate_counts": dict(sorted(predicate_counts.items())),
    }
    _save_json(path, report)


def load_triples(path: Path) -> list[KGTriple]:
    """Read triples written by :func:`save_triples`.

    Args:
        path: JSON file to read.

    Returns:
        The validated triples.
    """
    payload = _load_json(path)
    return [KGTriple.model_validate(item) for item in payload]


def load_documents(path: Path) -> list[DocumentRecord]:
    """Read the stage 0 document records.

    Args:
        path: JSON file to read.

    Returns:
        The validated records.
    """
    payload = _load_json(path)
    return [DocumentRecord.model_validate(item) for item in payload]


def load_registry(path: Path) -> dict[str, CanonicalEntityRecord]:
    """Read the stage 4 entity registry.

    Args:
        path: JSON file to read.

    Returns:
        Mapping from canonical name to validated record.
    """
    payload = _load_json(path)
    return {k: CanonicalEntityRecord.model_validate(v) for k, v in payload.items()}


def _cli() -> None:
    """Run stage 5 standalone from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples-json", required=True)
    parser.add_argument("--registry-json", required=True)
    parser.add_argument("--documents-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--exclude-mentioned-in", action="store_true")
    args = parser.parse_args()

    triples = load_triples(Path(args.triples_json))
    registry = load_registry(Path(args.registry_json))
    documents = load_documents(Path(args.documents_json))

    output = add_cross_document_links(
        triples,
        registry,
        documents,
        include_mentioned_in=not args.exclude_mentioned_in,
    )
    save_triples(Path(args.output_json), output)


if __name__ == "__main__":
    _cli()
