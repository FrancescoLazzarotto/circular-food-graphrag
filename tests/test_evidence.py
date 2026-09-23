"""Unit tests for the numbered-evidence and citation-gate path.

Evidence gets stable ids and travels with its provenance, the model-facing
context carries document and page, invented reference tags are caught, and
the source list reflects what was actually cited. Also guards the invariant
that the answer prompt is unchanged when `cite_evidence` is off, so gold runs
and experiment baselines stay comparable.
"""

from __future__ import annotations

from graphrag.agent.evidence import (
    build_evidence_index,
    evidence_from_dicts,
    evidence_to_dicts,
    parse_chunk_source,
    render_cited_context,
    render_display_citations,
    render_grouped_reference_list,
    render_reference_list,
    short_doc_label,
    verify_citations,
)
from graphrag.config import AgentConfig
from graphrag.llm.prompts import PromptLibrary


def _chunk(content: str, source: str, chunk_id: str) -> dict[str, str]:
    """A retrieved text chunk row."""
    return {"content": content, "source": source, "chunk_id": chunk_id}


def _triple(subject: str, predicate: str, obj: str, **props: object) -> dict[str, object]:
    """A retrieved triple row with relationship properties."""
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "relationship_properties": dict(props),
    }


# --- provenance parsing ---------------------------------------------------


def test_parse_chunk_source_splits_document_and_page():
    doc, page = parse_chunk_source("documents/SEeD for Change.pdf#page=3#chunk=7")
    assert doc == "SEeD for Change.pdf"
    assert page == "p. 3"


def test_parse_chunk_source_tolerates_missing_parts():
    assert parse_chunk_source("") == ("", "")
    assert parse_chunk_source("/a/b/report.pdf") == ("report.pdf", "")
    assert parse_chunk_source("report.pdf#page=12") == ("report.pdf", "p. 12")


# --- index construction ---------------------------------------------------


def test_build_evidence_index_numbers_text_then_triples():
    evidence = build_evidence_index(
        text_chunks=[
            _chunk("SEeD (Systemic Event Design) e un progetto...", "a.pdf#page=3#chunk=1", "c1"),
            _chunk("La coevoluzione indica...", "b.pdf#page=9#chunk=2", "c2"),
        ],
        triples=[_triple("SEeD", "IMPLEMENTS", "UNI ISO 20121", source_doc="a.pdf", page_range="3-4")],
    )

    assert [item.ref_id for item in evidence] == ["S1", "S2", "T1"]
    assert evidence[0].source_doc == "a.pdf"
    assert evidence[0].pages == "p. 3"
    assert evidence[2].text == "(SEeD, IMPLEMENTS, UNI ISO 20121)"
    assert evidence[2].source_label() == "a.pdf | p. 3-4"


def test_build_evidence_index_deduplicates_across_subqueries():
    # The same chunk and the same triple retrieved by two sub-queries must not
    # become two citable ids: it would double-count one piece of evidence.
    evidence = build_evidence_index(
        text_chunks=[
            _chunk("stesso testo", "a.pdf#page=1#chunk=1", "c1"),
            _chunk("stesso testo", "a.pdf#page=1#chunk=1", "c1"),
        ],
        triples=[
            _triple("A", "REL", "B"),
            _triple("a", "rel", "b"),
        ],
    )

    assert [item.ref_id for item in evidence] == ["S1", "T1"]


def test_build_evidence_index_respects_caps_and_skips_empty():
    evidence = build_evidence_index(
        text_chunks=[
            _chunk("uno", "a.pdf#page=1#chunk=1", "c1"),
            _chunk("   ", "a.pdf#page=2#chunk=2", "c2"),
            _chunk("due", "a.pdf#page=3#chunk=3", "c3"),
        ],
        triples=[_triple("A", "REL", "B"), _triple("C", "REL", "D")],
        max_text_items=1,
        max_triple_items=1,
    )

    assert [item.ref_id for item in evidence] == ["S1", "T1"]


def test_evidence_round_trips_through_serialisation():
    evidence = build_evidence_index(
        text_chunks=[_chunk("testo", "a.pdf#page=1#chunk=1", "c1")],
        triples=[_triple("A", "REL", "B", source_doc="b.pdf", page_range="7")],
    )
    restored = evidence_from_dicts(evidence_to_dicts(evidence))

    assert [item.ref_id for item in restored] == ["S1", "T1"]
    assert restored[1].source_label() == "b.pdf | p. 7"


# --- context rendering ----------------------------------------------------


def test_render_cited_context_attaches_source_to_every_item():
    evidence = build_evidence_index(
        text_chunks=[_chunk("passaggio verbatim", "REPORT.pdf#page=129#chunk=4", "c1")],
        triples=[_triple("A", "REL", "B", source_doc="REPORT.pdf", page_range="129")],
    )
    context = render_cited_context(
        evidence=evidence,
        entity_sections=[("Entities in the graph (no source — do not cite):", "A, B")],
    )

    # The question belongs to the prompt's own slot, never to the context: an
    # echoed query would make the context non-empty with nothing retrieved.
    assert "Query:" not in context
    assert "[S1] <REPORT.pdf | p. 129>" in context
    assert "passaggio verbatim" in context
    assert "[T1] (A, REL, B) <REPORT.pdf | p. 129>" in context
    # Entity names have no provenance and must stay outside the citable blocks.
    assert "Entities in the graph" in context


# --- citation gate --------------------------------------------------------


def test_verify_citations_accepts_known_references():
    evidence = build_evidence_index(triples=[_triple("A", "REL", "B")])
    report = verify_citations("Il fatto vale [T1].", evidence)

    assert report.phantom_refs == []
    assert report.cited_refs == ["T1"]
    assert report.total_citations == 1
    assert report.answer == "Il fatto vale [T1]."


def test_verify_citations_marks_invented_reference():
    evidence = build_evidence_index(triples=[_triple("A", "REL", "B")])
    report = verify_citations("Affermazione inventata [T9].", evidence)

    assert report.phantom_refs == ["T9"]
    assert report.cited_refs == []
    assert "[T9]" not in report.answer
    assert "non verificato" in report.answer
    assert report.phantom_rate == 1.0


def test_verify_citations_strip_policy_removes_the_tag():
    evidence = build_evidence_index(triples=[_triple("A", "REL", "B")])
    report = verify_citations("Affermazione [S4].", evidence, policy="strip")

    assert report.phantom_refs == ["S4"]
    assert "S4" not in report.answer


def test_verify_citations_keeps_the_valid_half_of_a_group():
    evidence = build_evidence_index(triples=[_triple("A", "REL", "B")])
    report = verify_citations("Claim [T1, T5].", evidence)

    assert report.cited_refs == ["T1"]
    assert report.phantom_refs == ["T5"]
    assert "[T1]" in report.answer
    assert "T5" not in report.answer


def test_verify_citations_trims_stacked_ids():
    # Models stack "[T4, T5, T6]" on a single claim; the surplus turns both the
    # prose and the source list into noise.
    evidence = build_evidence_index(
        triples=[
            _triple("A", "REL", "B"),
            _triple("C", "REL", "D"),
            _triple("E", "REL", "F"),
        ]
    )
    report = verify_citations("Claim [T1, T2, T3].", evidence)

    assert report.cited_refs == ["T1", "T2"]
    assert report.phantom_refs == []
    assert "[T1, T2]" in report.answer
    assert "T3" not in report.answer


def test_verify_citations_collapses_adjacent_tags():
    # Some models (Qwen3-30B) sidestep a per-tag cap by closing and reopening
    # the brackets, "[T1], [T4], [T6]" on a single claim.
    evidence = build_evidence_index(
        triples=[_triple(f"S{i}", "REL", f"O{i}") for i in range(1, 7)]
    )
    report = verify_citations("Una affermazione [T1], [T4], [T6].", evidence)

    assert report.cited_refs == ["T1", "T4"]
    assert "[T1, T4]" in report.answer
    assert "T6" not in report.answer


def test_verify_citations_leaves_distant_tags_alone():
    evidence = build_evidence_index(
        triples=[_triple(f"S{i}", "REL", f"O{i}") for i in range(1, 4)]
    )
    report = verify_citations("Prima frase [T1]. Seconda frase [T2].", evidence)

    assert report.cited_refs == ["T1", "T2"]
    assert report.answer == "Prima frase [T1]. Seconda frase [T2]."


def test_render_reference_list_caps_long_lists():
    evidence = build_evidence_index(
        triples=[_triple(f"S{i}", "REL", f"O{i}") for i in range(1, 13)]
    )
    rendered = render_reference_list(
        evidence, cited_refs=[f"T{i}" for i in range(1, 13)], max_items=8
    )

    assert rendered.count("\n- ") == 9  # 8 items plus the summary line
    assert "[T8]" in rendered
    assert "[T9]" not in rendered
    assert "(+4 altre evidenze citate)" in rendered


def test_verify_citations_normalises_lowercase_tags():
    evidence = build_evidence_index(
        text_chunks=[_chunk("testo", "a.pdf#page=1#chunk=1", "c1")]
    )
    report = verify_citations("Claim [s1].", evidence)

    assert report.cited_refs == ["S1"]
    assert report.phantom_refs == []


def test_verify_citations_english_marker():
    evidence = build_evidence_index(triples=[_triple("A", "REL", "B")])
    report = verify_citations("Invented claim [T7].", evidence, language="en")

    assert "unverified reference" in report.answer


# --- reference list -------------------------------------------------------


def test_render_reference_list_shows_only_cited_items():
    evidence = build_evidence_index(
        text_chunks=[
            _chunk("uno", "a.pdf#page=1#chunk=1", "c1"),
            _chunk("due", "b.pdf#page=2#chunk=2", "c2"),
        ],
        triples=[_triple("A", "REL", "B", source_doc="c.pdf", page_range="5")],
    )
    rendered = render_reference_list(evidence, cited_refs=["S2", "T1"])

    assert rendered.startswith("Fonti:")
    assert "[S2] b.pdf | p. 2" in rendered
    assert "[T1] (A, REL, B) — c.pdf | p. 5" in rendered
    assert "[S1]" not in rendered


def test_render_reference_list_is_empty_when_nothing_was_cited():
    """An uncited answer gets no source list, evidence or no evidence.

    Falling back to the top evidence items would publish an off-topic answer
    under three circular-economy PDFs whenever the retriever returns its usual
    top-k for an out-of-domain question.
    """
    evidence = build_evidence_index(
        text_chunks=[_chunk("uno", "a.pdf#page=1#chunk=1", "c1")]
    )
    assert render_reference_list(evidence, cited_refs=[], language="en") == ""


def test_render_reference_list_empty_without_evidence():
    assert render_reference_list([], cited_refs=[]) == ""


# --- reader-facing citation labels ----------------------------------------


def test_short_doc_label_strips_filename_noise():
    assert short_doc_label("SEeD for Change.pdf") == "SEeD for Change"
    assert short_doc_label("sustainability-16-08960-v2 it.pdf") == "sustainability-16-08960"
    assert short_doc_label("Materia 45 ita web.pdf") == "Materia 45"
    assert short_doc_label("REPORT MATTM_Definitivo.pdf") == "REPORT MATTM"
    # Long titles cut at the comma, which drops the authors and keeps the title.
    assert short_doc_label("Circular Economy for Food, Fassio Tecco.pdf") == (
        "Circular Economy for Food"
    )
    assert short_doc_label("F.Fassio per Regione Piemonte su Sustainability (web) it.pdf") == (
        "F.Fassio per Regione Piemonte su…"
    )
    assert short_doc_label("") == ""


def test_display_citations_replace_ids_with_document_and_page():
    evidence = build_evidence_index(
        text_chunks=[_chunk("testo", "SEeD for Change.pdf#page=3#chunk=1", "c1")],
        triples=[_triple("SEeD", "IMPLEMENTS", "UNI ISO 20121", source_doc="SEeD for Change.pdf", page_range="3-4")],
    )
    rendered = render_display_citations(
        "SEeD nasce nel 2005 [S1]. Applica UNI ISO 20121 [T1].", evidence
    )

    assert "[SEeD for Change, p. 3]" in rendered
    assert "[SEeD for Change, p. 3-4]" in rendered
    assert "[S1]" not in rendered


def test_display_citations_merge_pages_of_the_same_document():
    evidence = build_evidence_index(
        text_chunks=[_chunk("testo", "SEeD for Change.pdf#page=3#chunk=1", "c1")],
        triples=[
            _triple("A", "REL", "B", source_doc="SEeD for Change.pdf", page_range="3-4")
        ],
    )
    rendered = render_display_citations("Affermazione [S1, T1].", evidence)

    assert rendered == "Affermazione [SEeD for Change, p. 3, 3-4]."


def test_display_citations_collapse_ids_sharing_a_label():
    # A passage and a triple from the same page must not print the same source
    # twice inside one tag.
    evidence = build_evidence_index(
        text_chunks=[_chunk("testo", "a.pdf#page=7#chunk=1", "c1")],
        triples=[_triple("A", "REL", "B", source_doc="a.pdf", page_range="7")],
    )
    rendered = render_display_citations("Affermazione [S1, T1].", evidence)

    assert rendered == "Affermazione [a, p. 7]."


def test_display_citations_leave_unknown_ids_untouched():
    evidence = build_evidence_index(triples=[_triple("A", "REL", "B")])

    assert render_display_citations("Claim [T9].", evidence) == "Claim [T9]."


def test_grouped_reference_list_keeps_every_document():
    evidence = build_evidence_index(
        text_chunks=[
            _chunk("uno", "SEeD for Change.pdf#page=1#chunk=1", "c1"),
            _chunk("due", "SEeD for Change.pdf#page=3#chunk=2", "c2"),
            _chunk("tre", "REPORT MATTM.pdf#page=13#chunk=3", "c3"),
        ],
        triples=[
            _triple(f"S{i}", "REL", f"O{i}", source_doc="SEeD for Change.pdf", page_range="3-4")
            for i in range(1, 10)
        ],
    )
    rendered = render_grouped_reference_list(
        evidence, cited_refs=[f"T{i}" for i in range(1, 10)] + ["S1", "S2", "S3"]
    )

    # A flat list capped at 8 entries would drop whole documents; grouping cannot.
    assert "- **SEeD for Change.pdf**" in rendered
    assert "- **REPORT MATTM.pdf**" in rendered
    assert "passaggi citati: p. 1, p. 3" in rendered
    assert "+5 altri" in rendered


def test_grouped_reference_list_is_empty_when_nothing_was_cited():
    evidence = build_evidence_index(
        text_chunks=[_chunk("uno", "a.pdf#page=1#chunk=1", "c1")]
    )
    assert render_grouped_reference_list(evidence, cited_refs=[], language="en") == ""


def test_grouped_reference_list_still_renders_what_was_cited():
    """The empty-list rule above applies only when nothing was cited."""
    evidence = build_evidence_index(
        text_chunks=[_chunk("uno", "a.pdf#page=1#chunk=1", "c1")]
    )
    rendered = render_grouped_reference_list(evidence, cited_refs=["S1"], language="en")

    assert rendered.startswith("Sources:")
    assert "cited passages: p. 1" in rendered


# --- prompt invariants ----------------------------------------------------


def test_answer_prompt_unchanged_when_citations_are_off():
    """With citations off the prompt carries no citation protocol, so baselines
    and gold runs stay comparable.
    """
    for always_limits in (False, True):
        config = AgentConfig(always_include_limits=always_limits, cite_evidence=False)
        rendered = str(PromptLibrary.answer_prompt(config))

        assert "Evidence in graph" in rendered
        assert "[S1]" not in rendered
        assert "1-2 short paragraphs" in rendered


def test_answer_prompt_requests_selective_citations_when_enabled():
    config = AgentConfig(always_include_limits=True, cite_evidence=True)
    rendered = str(PromptLibrary.answer_prompt(config))

    assert "[S1], [S2]" in rendered
    assert "at most one tag per sentence" in rendered
    # A ban on inline citations would contradict the citation protocol.
    assert "free of inline triple citations" not in rendered
