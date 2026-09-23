"""Stage 5 turns resolved triples into the edges the retriever ranks on.

`mention_count` — the term the KG retriever weights triples by — is computed
here and nowhere else, and a real share of linked triples carry a count above
1, so the value is load-bearing rather than decorative. The rules that produce
it are worth pinning: what counts as the same triple, what a document edge is
deduplicated by, and when an alias earns a SAME_AS.
"""

from __future__ import annotations

import json
from pathlib import Path

from kg_pipeline.models.types import CanonicalEntityRecord, DocumentRecord, KGTriple
from kg_pipeline.stages import linking


def _triple(
    subject: str = "Rice husk",
    predicate: str = "USED_AS",
    obj: str = "substrate",
    *,
    source_doc: str = "a.pdf",
    chunk_id: str = "c1",
    page_range: str = "1-2",
    **rel: object,
) -> KGTriple:
    """A resolved triple from one chunk and page."""
    return KGTriple.model_validate(
        {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "subject_labels": ["Material"],
            "object_labels": ["Material"],
            "subject_properties": {"name": subject},
            "object_properties": {"name": obj},
            "relationship_properties": {
                "source_doc": source_doc,
                "chunk_id": chunk_id,
                "page_range": page_range,
                "extraction_method": "llm",
                **rel,
            },
        }
    )


def _doc(filename: str = "a.pdf") -> DocumentRecord:
    """A document record for `filename`."""
    return DocumentRecord(
        doc_id=filename.removesuffix(".pdf"),
        filename=filename,
        page_count=10,
        markdown_text="...",
        title="A paper",
        publication_year=2024,
    )


def _registry(**kwargs) -> dict[str, CanonicalEntityRecord]:
    """Canonical entity records from keyword specs."""
    return {
        k: CanonicalEntityRecord.model_validate(v) for k, v in kwargs.items()
    }


def _predicates(triples: list[KGTriple]) -> list[str]:
    """The predicates of `triples`, in order."""
    return [t.predicate for t in triples]


def _mention_counts(triples: list[KGTriple]) -> dict[tuple[str, str, str], int]:
    """`mention_count` per (subject, predicate, object)."""
    return {
        (t.subject, t.predicate, t.object): t.relationship_properties["mention_count"]
        for t in triples
    }


# --- mention_count: the number the retriever ranks on ----------------------


def test_the_same_fact_stated_twice_is_counted_twice():
    # Two chunks, two pages, one fact. This is the case the ranker exists for.
    triples = [
        _triple(chunk_id="c1", page_range="1-2"),
        _triple(chunk_id="c9", page_range="7-8"),
        _triple("Straw", "USED_AS", "fuel"),
    ]

    out = linking.add_cross_document_links(triples, {}, [_doc()], include_mentioned_in=False)

    counts = _mention_counts(out)
    assert counts[("Rice husk", "USED_AS", "substrate")] == 2
    assert counts[("Straw", "USED_AS", "fuel")] == 1


def test_case_and_spacing_do_not_make_two_different_facts():
    triples = [
        _triple("Rice husk", "USED_AS", "substrate"),
        _triple("  RICE   HUSK ", "used_as", "Substrate"),
    ]

    out = linking.add_cross_document_links(triples, {}, [_doc()], include_mentioned_in=False)

    assert all(t.relationship_properties["mention_count"] == 2 for t in out)
    # Normalising is for counting only: the surface forms are left alone, because
    # they are what stage 6 writes and what a citation shows.
    assert [t.subject for t in out] == ["Rice husk", "RICE   HUSK"]


def test_a_count_already_on_the_triple_is_left_alone():
    # `setdefault`, not assignment. Nothing upstream sets `mention_count`, but
    # the model is free to put any key in `relationship_properties`, and if it
    # ever emits this one the computed value is discarded. Pinned so that stays
    # a decision rather than a surprise.
    triples = [_triple(mention_count=99), _triple(chunk_id="c2", mention_count=99)]

    out = linking.add_cross_document_links(triples, {}, [_doc()], include_mentioned_in=False)

    assert [t.relationship_properties["mention_count"] for t in out] == [99, 99]


# --- MENTIONED_IN ----------------------------------------------------------


def test_an_entity_named_twice_in_one_chunk_gets_one_document_edge():
    triples = [
        _triple("Rice husk", "USED_AS", "substrate", chunk_id="c1"),
        _triple("Rice husk", "CONTAINS", "silica", chunk_id="c1"),
    ]

    out = linking.add_cross_document_links(triples, {}, [_doc()])

    edges = [t for t in out if t.predicate == "MENTIONED_IN"]
    subjects = sorted(t.subject for t in edges)
    # One edge per (entity, doc, chunk, page): "Rice husk" appears in both
    # triples of the same chunk and must not be linked to the document twice.
    assert subjects == ["Rice husk", "silica", "substrate"]


def test_the_same_entity_in_another_chunk_is_another_edge():
    triples = [
        _triple("Rice husk", "USED_AS", "substrate", chunk_id="c1", page_range="1-2"),
        _triple("Rice husk", "USED_AS", "fuel", chunk_id="c4", page_range="5-6"),
    ]

    out = linking.add_cross_document_links(triples, {}, [_doc()])

    pages = sorted(
        t.relationship_properties["page_range"]
        for t in out
        if t.predicate == "MENTIONED_IN" and t.subject == "Rice husk"
    )
    assert pages == ["1-2", "5-6"]


def test_a_triple_from_an_unknown_document_gets_no_document_edge():
    # `source_doc` is authoritative pipeline metadata, but stage 5 is given the
    # document list separately: a mismatch must not invent a :Document node.
    triples = [_triple(source_doc="ghost.pdf")]

    out = linking.add_cross_document_links(triples, {}, [_doc("a.pdf")])

    assert _predicates(out) == ["USED_AS"]


def test_document_edges_can_be_switched_off():
    triples = [_triple()]

    out = linking.add_cross_document_links(triples, {}, [_doc()], include_mentioned_in=False)

    assert _predicates(out) == ["USED_AS"]


def test_a_document_edge_carries_the_document_it_points_at():
    out = linking.add_cross_document_links([_triple()], {}, [_doc()])

    edge = next(t for t in out if t.predicate == "MENTIONED_IN")
    assert edge.object_labels == ["Document"]
    assert edge.object_properties["title"] == "A paper"
    assert edge.object_properties["publication_year"] == 2024
    # Provenance says a machine made this edge, not the extractor.
    assert edge.relationship_properties["extraction_method"] == "system_linking"


# --- SAME_AS ---------------------------------------------------------------


def test_an_alias_seen_in_two_documents_earns_a_same_as():
    registry = _registry(
        **{
            "circular economy": {
                "canonical_name": "circular economy",
                "aliases": ["circular economy", "economia circolare"],
                "labels": ["Concept"],
                "merged_properties": {"name": "circular economy"},
                "alias_sources": {
                    "circular economy": ["a.pdf"],
                    "economia circolare": ["b.pdf"],
                },
            }
        }
    )

    out = linking.add_cross_document_links([], registry, [_doc()], include_mentioned_in=False)

    same_as = [t for t in out if t.predicate == "SAME_AS"]
    # The canonical name is not an alias of itself.
    assert [(t.subject, t.object) for t in same_as] == [
        ("economia circolare", "circular economy")
    ]


def test_an_alias_confined_to_one_document_earns_nothing():
    # The edge claims two documents talk about the same thing. One document is
    # not cross-document evidence, whatever the resolver merged.
    registry = _registry(
        **{
            "circular economy": {
                "canonical_name": "circular economy",
                "aliases": ["circular economy", "economia circolare"],
                "labels": ["Concept"],
                "merged_properties": {"name": "circular economy"},
                "alias_sources": {
                    "circular economy": ["a.pdf"],
                    "economia circolare": ["a.pdf"],
                },
            }
        }
    )

    out = linking.add_cross_document_links([], registry, [_doc()], include_mentioned_in=False)

    assert out == []


# --- artifacts -------------------------------------------------------------


def test_triples_survive_a_round_trip_through_the_artifact(tmp_path: Path):
    out = linking.add_cross_document_links([_triple()], {}, [_doc()])
    path = tmp_path / "stage5_triples_linked.json"

    linking.save_triples(path, out)
    back = linking.load_triples(path)

    assert [t.as_dict() for t in back] == [t.as_dict() for t in out]


def test_the_check_report_says_what_was_written(tmp_path: Path):
    out = linking.add_cross_document_links([_triple(), _triple(chunk_id="c2")], {}, [_doc()])
    path = tmp_path / "stage5_check.json"

    linking.check_triples(path, out)

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["triples_count"] == len(out)
    assert report["predicate_counts"]["USED_AS"] == 2
    assert report["distinct_predicates"] == len(report["predicate_counts"])


def test_the_registry_and_documents_load_back_as_records(tmp_path: Path):
    # main.py hands stage 5 whatever these two read; a shape change here is a
    # silent stage-5 no-op rather than an error.
    registry_path = tmp_path / "stage4_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "circular economy": {
                    "canonical_name": "circular economy",
                    "aliases": ["economia circolare"],
                    "labels": ["Concept"],
                    "merged_properties": {"name": "circular economy"},
                    "alias_sources": {"economia circolare": ["a.pdf"]},
                }
            }
        ),
        encoding="utf-8",
    )
    docs_path = tmp_path / "stage0_documents.json"
    docs_path.write_text(json.dumps([_doc().model_dump()]), encoding="utf-8")

    registry = linking.load_registry(registry_path)
    documents = linking.load_documents(docs_path)

    assert registry["circular economy"].aliases == ["economia circolare"]
    assert [d.filename for d in documents] == ["a.pdf"]
