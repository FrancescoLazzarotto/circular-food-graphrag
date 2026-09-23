"""What the demo tells a reader, tested without starting Streamlit.

`product/ui.py` decides three things that fail quietly: where an answer's own
sections end, which document each piece of evidence belongs to, and what the
citation check is allowed to claim. None of the three raises when it goes wrong
— the page just shows a source list twice, or attributes a passage to the wrong
PDF, or tells a reader an answer is verified when it is not.

Every function under test is pure, so the whole surface is reachable from here.
"""

from __future__ import annotations

import pytest

from product import ui

# The shape the engine really produces: prose with reader-facing citation
# labels, the model's own limits section, then the source list appended by
# graphrag.agent.evidence.render_grouped_reference_list.
REAL_ANSWER_IT = (
    "Il biochar è un ammendante ottenuto per pirolisi [MR37, p. 35].\n\n"
    "Migliora la fertilità del suolo [Kenya Report, p. 79].\n\n"
    "Limiti e affidabilità\n"
    "Le evidenze non contengono una definizione formale.\n\n"
    "Fonti:\n"
    "- **MR37-ita.pdf**\n"
    "  - passaggi citati: p. 35\n"
)

EVIDENCE = [
    {
        "ref_id": "S1",
        "kind": "text",
        "text": "Il biochar è un ammendante ottenuto per pirolisi.",
        "source_doc": "MR37-ita.pdf",
        "pages": "p. 35",
        "chunk_id": "c1",
        "metadata": "",
    },
    {
        "ref_id": "S2",
        "kind": "text",
        "text": "Passaggio recuperato e mai citato.",
        "source_doc": "Kenya Report_Full version.pdf",
        "pages": "p. 79",
        "chunk_id": "c2",
        "metadata": "",
    },
    {
        "ref_id": "T1",
        "kind": "triple",
        "text": "(biochar, REDUCES, erosione del suolo)",
        "source_doc": "MR37-ita.pdf",
        "pages": "78-79",
        "chunk_id": "",
        "metadata": "",
    },
]


# --------------------------------------------------------------------------- #
# split_answer
# --------------------------------------------------------------------------- #


def test_split_answer_drops_the_engine_source_list():
    """The page rebuilds the sources from the evidence index.

    Leaving the engine's own list in the prose puts the same documents on
    screen twice, once as text and once as blocks.
    """
    parts = ui.split_answer(REAL_ANSWER_IT)
    assert "Fonti:" not in parts.body
    assert "MR37-ita.pdf" not in parts.body
    assert parts.body.startswith("Il biochar")


def test_split_answer_separates_the_limits_section():
    parts = ui.split_answer(REAL_ANSWER_IT)
    assert parts.limits == "Le evidenze non contengono una definizione formale."
    assert "Limiti e affidabilità" not in parts.body


def test_split_answer_handles_english_answers():
    answer = (
        "Biochar is a soil amendment [MR37, p. 35].\n\n"
        "Limits and confidence\nThe evidence is thin.\n\n"
        "Sources:\n- **MR37-ita.pdf**\n"
    )
    parts = ui.split_answer(answer)
    assert parts.body == "Biochar is a soil amendment [MR37, p. 35]."
    assert parts.limits == "The evidence is thin."


@pytest.mark.parametrize(
    "heading",
    ["Limiti e affidabilità", "**Limiti e affidabilità**", "## Limiti e affidabilità"],
)
def test_split_answer_tolerates_a_decorated_heading(heading):
    """The limits heading is written by the model, not by the renderer.

    It is asked for by name in the prompt, and models deliver it bold, as a
    heading, or bare. A splitter that knows only the bare form leaves the
    section inside the prose on most turns.
    """
    parts = ui.split_answer(f"Corpo della risposta.\n\n{heading}\nEvidenza debole.")
    assert parts.body == "Corpo della risposta."
    assert parts.limits == "Evidenza debole."


@pytest.mark.parametrize(
    "heading",
    [
        "**Limits and confidence**:",
        "**Limits and confidence:**",
        "Limiti e affidabilità:",
        "## Limiti e affidabilità:",
    ],
)
def test_split_answer_takes_a_limits_section_written_inline(heading):
    """Many real answers put the section text on the heading's own line.

    "**Limits and confidence**: the evidence is thin" as a single line is
    common in the recorded sessions. A pattern anchored to the end of the line
    matches none of those, so the box stays empty and the caveat stays buried
    in the prose.
    """
    parts = ui.split_answer(f"Corpo della risposta.\n\n{heading} Evidenza debole.")
    assert parts.body == "Corpo della risposta."
    assert parts.limits == "Evidenza debole."


def test_split_answer_does_not_cut_on_a_mid_sentence_mention():
    """Only a heading at the start of a line ends the prose."""
    answer = "La sezione Limiti e affidabilità di quel report è vuota, e prosegue."
    parts = ui.split_answer(answer)
    assert parts.body == answer
    assert parts.limits == ""


def test_split_answer_keeps_an_answer_without_sections_intact():
    parts = ui.split_answer("Una risposta breve, senza sezioni.")
    assert parts.body == "Una risposta breve, senza sezioni."
    assert parts.limits == ""


def test_split_answer_cuts_at_the_appended_list_not_at_a_mention():
    """The engine appends its list last, so the final heading is the real one.

    An answer that discusses its own sources — "le Fonti:" written mid-prose —
    must not lose everything after that line.
    """
    answer = (
        "Il documento elenca le sue Fonti:\n"
        "e prosegue con il ragionamento.\n\n"
        "Fonti:\n- **MR37-ita.pdf**\n"
    )
    parts = ui.split_answer(answer)
    assert "prosegue con il ragionamento" in parts.body
    assert "MR37-ita.pdf" not in parts.body


def test_split_answer_survives_an_empty_answer():
    parts = ui.split_answer("")
    assert parts.body == ""
    assert parts.limits == ""


# --------------------------------------------------------------------------- #
# readable_fact
# --------------------------------------------------------------------------- #


def test_readable_fact_parses_an_object_containing_commas():
    """The predicate is the anchor, not the comma count.

    Splitting on commas attributes half the object to the predicate whenever
    a node name carries one, which is common in this graph.
    """
    fact = ui.readable_fact("(biochar, REDUCES, erosione, dilavamento e perdita di suolo)")
    assert fact.subject == "biochar"
    assert fact.predicate == "reduces"
    assert fact.obj == "erosione, dilavamento e perdita di suolo"


def test_readable_fact_spells_out_an_underscored_relation():
    assert ui.readable_fact("(lolla, HAS_COMPONENT, silice)").predicate == "has component"


def test_readable_fact_falls_back_to_the_raw_form():
    """Anything that is not a vocabulary triple is shown as it arrived."""
    assert ui.readable_fact("(a, b, c)").sentence() == "(a, b, c)"


# --------------------------------------------------------------------------- #
# evidence_by_document
# --------------------------------------------------------------------------- #


def test_evidence_by_document_keeps_only_what_was_cited():
    documents = ui.evidence_by_document(EVIDENCE, ["S1", "T1"], only_cited=True)
    assert [entry.document for entry in documents] == ["MR37-ita.pdf"]
    entry = documents[0]
    assert entry.n_refs == 2
    assert entry.pages() == ["p. 35"]
    assert [fact["ref_id"] for fact in entry.facts] == ["T1"]


def test_evidence_by_document_can_show_everything_retrieved():
    """The evidence panel has to show what was left unused.

    An answer stands on what it cited; what the collection returned and the
    answer ignored is the part a reader needs to judge it.
    """
    documents = ui.evidence_by_document(EVIDENCE, ["S1"], only_cited=False)
    assert {entry.document for entry in documents} == {
        "MR37-ita.pdf",
        "Kenya Report_Full version.pdf",
    }
    uncited = [
        passage
        for entry in documents
        for passage in entry.passages
        if not passage["cited"]
    ]
    assert [passage["ref_id"] for passage in uncited] == ["S2"]


def test_evidence_by_document_matches_reference_ids_case_insensitively():
    """Models emit lowercase tags, and verify_citations normalises them.

    Comparing raw strings drops every citation on a turn where the model
    writes [s1] instead of [S1], so a correctly sourced answer shows no
    sources.
    """
    documents = ui.evidence_by_document(EVIDENCE, ["s1"], only_cited=True)
    assert documents and documents[0].n_refs == 1


def test_evidence_by_document_labels_evidence_without_a_document():
    orphan = [{"ref_id": "T9", "kind": "triple", "text": "(a, USES, b)", "source_doc": ""}]
    documents = ui.evidence_by_document(orphan, ["T9"], unnamed_label="senza documento")
    assert documents[0].document == "senza documento"


def test_evidence_by_document_ignores_malformed_rows():
    documents = ui.evidence_by_document([None, "S1", {}], ["S1"], only_cited=False)
    assert documents == [] or all(entry.n_refs >= 0 for entry in documents)


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #


def _stream(scrubber, pieces):
    """Feed `pieces` through the scrubber, then flush it."""
    return "".join(scrubber.feed(p) for p in pieces) + scrubber.flush()


def test_scrubber_hides_a_tag_split_across_chunks():
    """Tokens arrive in fragments: "[S" and "1]" are one tag, not two strings."""
    out = _stream(ui.StreamScrubber(), ["Le tre C ", "[S", "1", "]", " e la quarta."])
    assert out == "Le tre C  e la quarta."


def test_scrubber_hides_a_grouped_tag():
    assert _stream(ui.StreamScrubber(), ["Testo ", "[S2, T3]", "."]) == "Testo ."


def test_scrubber_leaves_ordinary_brackets_alone():
    text = ["Un elenco ", "[a, b, c]", " e una ", "[nota]", "."]
    assert _stream(ui.StreamScrubber(), text) == "Un elenco [a, b, c] e una [nota]."


def test_scrubber_releases_a_bracket_that_never_closes():
    """A held fragment must reach the reader, or the answer loses its tail."""
    assert _stream(ui.StreamScrubber(), ["testo [S", "1"]) == "testo [S1"


def test_scrubber_releases_a_long_bracket_before_it_swallows_the_answer():
    long_text = "[" + "x" * 60 + "]"
    assert long_text in _stream(ui.StreamScrubber(), [long_text])


def test_scrubber_reset_drops_a_discarded_retry():
    scrubber = ui.StreamScrubber()
    scrubber.feed("mezza risposta [S")
    scrubber.reset()
    assert scrubber.feed("nuova risposta") == "nuova risposta"
    assert scrubber.flush() == ""


# --------------------------------------------------------------------------- #
# inline citations
# --------------------------------------------------------------------------- #


def test_style_citations_sets_a_citation_apart_from_the_sentence():
    out = ui.style_citations("Le tre C sono Capitale e Ciclicità [MR37, p. 26].")
    assert out == "Le tre C sono Capitale e Ciclicità :gray[*(MR37, p. 26)*]."


def test_style_citations_shortens_a_long_document():
    out = ui.style_citations("Il quadro [A Review of Principal Measuring…, p. 16].", doc_chars=16)
    assert "A Review of Prin…, p. 16" in out
    assert "Principal Measuring" not in out


def test_style_citations_splits_a_grouped_citation():
    out = ui.style_citations("Vedi [MR37, p. 26; Kenya Report, p. 79].")
    assert out == "Vedi :gray[*(MR37, p. 26 · Kenya Report, p. 79)*]."


def test_style_citations_collapses_a_one_page_range():
    """"p. 18-18" is a range of one page, and it reads as a typo."""
    assert "p. 18)" in ui.style_citations("Testo [MR37, p. 18-18].")
    # A real range survives.
    assert "p. 22-24" in ui.style_citations("Testo [MR37, p. 22-24].")


def test_style_citations_leaves_other_brackets_alone():
    """Only brackets carrying a page marker are citations."""
    text = "Il modello ha scritto [una nota] e un elenco [a, b, c]."
    assert ui.style_citations(text) == text


def test_style_citations_can_be_asked_not_to_dim():
    out = ui.style_citations("Testo [MR37, p. 26].", dim=False)
    assert out == "Testo *(MR37, p. 26)*."


def test_style_citations_survives_an_empty_body():
    assert ui.style_citations("") == ""


# --------------------------------------------------------------------------- #
# the evidence panel
# --------------------------------------------------------------------------- #


def test_panel_evidence_puts_what_the_answer_used_first():
    evidence = [
        {"ref_id": "S1", "kind": "text", "text": "non citato", "source_doc": "a.pdf", "pages": "p. 1"},
        {"ref_id": "S2", "kind": "text", "text": "citato", "source_doc": "b.pdf", "pages": "p. 2"},
    ]
    panel = ui.panel_evidence(evidence, ["S2"])
    assert [row["ref_id"] for row in panel.passages] == ["S2", "S1"]
    assert [row["cited"] for row in panel.passages] == [True, False]


def test_panel_evidence_keeps_everything_retrieved():
    """Nothing is dropped: the remainder is what a reader judges the answer by."""
    panel = ui.panel_evidence(EVIDENCE, ["S1"])
    assert [row["ref_id"] for row in panel.passages] == ["S1", "S2"]
    assert [row["ref_id"] for row in panel.facts] == ["T1"]


def test_panel_evidence_leads_with_retrieval_order_when_nothing_was_cited():
    """Retrieval order is relevance order, so the box still opens on the best."""
    panel = ui.panel_evidence(EVIDENCE, [])
    assert [row["ref_id"] for row in panel.passages] == ["S1", "S2"]
    assert all(row["cited"] is False for row in panel.passages)




def test_fact_line_is_one_line_with_its_document():
    """Two lines per fact make the panel outgrow the answer."""
    line = ui.fact_line(
        {"text": "(biochar, REDUCES, erosione del suolo)", "document": "MR37-ita.pdf"}
    )
    assert line == "biochar · reduces · erosione del suolo · MR37"
    assert "\n" not in line


def test_fact_line_without_a_document():
    assert ui.fact_line({"text": "(a, USES, b)", "document": ""}) == "a · uses · b"


def test_passage_label_names_the_document_and_the_pages():
    assert ui.passage_label({"document": "MR37-ita.pdf", "pages": "p. 35"}) == "MR37-ita.pdf · p. 35"
    assert ui.passage_label({"document": "MR37-ita.pdf", "pages": ""}) == "MR37-ita.pdf"


# --------------------------------------------------------------------------- #
# the compact source line
# --------------------------------------------------------------------------- #


def test_compact_sources_line_names_documents_and_their_pages():
    line = ui.compact_sources_line(EVIDENCE, ["S1", "T1"], "it")
    assert line.startswith("Fonti: ")
    # The engine's own shortener, so a document is named here the way an
    # in-text citation names it.
    assert "MR37 (p. 35)" in line
    assert "1 fatto dal grafo" in line
    # Retrieved and never cited, so it is not part of the provenance.
    assert "Kenya" not in line


def test_compact_sources_line_is_empty_when_nothing_was_cited():
    assert ui.compact_sources_line(EVIDENCE, [], "it") == ""
    assert ui.compact_sources_line([], ["S1"], "en") == ""


def test_compact_sources_line_collects_the_pages_of_one_document():
    evidence = [
        {"ref_id": "S1", "kind": "text", "text": "a", "source_doc": "MR37-ita.pdf", "pages": "p. 35"},
        {"ref_id": "S2", "kind": "text", "text": "b", "source_doc": "MR37-ita.pdf", "pages": "p. 37"},
    ]
    assert "MR37 (p. 35, p. 37)" in ui.compact_sources_line(evidence, ["S1", "S2"], "it")


@pytest.mark.parametrize(
    ("lang", "key", "n", "expected"),
    [
        ("it", "meta_passages", 1, "1 passaggio"),
        ("it", "meta_passages", 8, "8 passaggi"),
        ("it", "meta_facts", 1, "1 fatto dal grafo"),
        ("en", "meta_documents", 1, "1 document"),
        ("en", "meta_documents", 4, "4 documents"),
    ],
)
def test_count_label_reads_right_at_one(lang, key, n, expected):
    assert ui.count_label(lang, key, n) == expected


# --------------------------------------------------------------------------- #
# counts and the citation check
# --------------------------------------------------------------------------- #


def test_retrieval_counts_read_the_state_not_the_prose():
    result = {
        "retrieved_text_sources": [{"content": "x"}] * 8,
        "kg_triples": [{"subject": "a"}] * 20,
        "evidence_index": EVIDENCE,
    }
    assert ui.retrieval_counts(result) == {
        "passages": 8,
        "facts": 20,
        "documents": 2,
    }


def test_retrieval_counts_on_an_empty_result():
    assert ui.retrieval_counts({}) == {"passages": 0, "facts": 0, "documents": 0}


def test_citation_summary_reports_a_clean_check():
    state, text = ui.citation_summary(
        {"total_citations": 31, "phantom_refs": [], "cited_refs": ["S1"]}, "it"
    )
    assert state == "clean"
    assert "31" in text


def test_citation_summary_reports_unverified_references():
    state, text = ui.citation_summary(
        {"total_citations": 12, "phantom_refs": ["S9", "T4"]}, "it"
    )
    assert state == "phantom"
    assert "2" in text


def test_citation_summary_says_nothing_when_nothing_was_cited():
    assert ui.citation_summary({}, "it")[0] == "none"
    assert ui.citation_summary(None, "en")[0] == "none"


def test_citation_summary_ignores_insufficient_answer():
    """`insufficient_answer` must never reach a reliability claim.

    It flags invented answers that hedge in the tail — so an answer it marks
    can be worse than one it does not. The citation check is the only signal
    allowed to speak here.
    """
    report = {"total_citations": 4, "phantom_refs": []}
    with_flag = ui.citation_summary({**report, "insufficient_answer": True}, "it")
    without = ui.citation_summary(report, "it")
    assert with_flag == without


# --------------------------------------------------------------------------- #
# model names and export
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("RedHatAI/Qwen3.8-27B-INT4", "Qwen3.8 27B"),
        ("Qwen/Qwen2.5-32B-Instruct-AWQ", "Qwen2.5 32B"),
        ("", ""),
    ],
)
def test_model_display_name_drops_the_vendor_and_the_artifact_suffix(model_id, expected):
    assert ui.model_display_name(model_id) == expected


def test_answer_markdown_carries_the_sources_with_the_text():
    """The export exists so a pasted answer keeps its provenance."""
    turn = {
        "question": "Che cos'è il biochar?",
        "body": "Il biochar è un ammendante [MR37, p. 35].",
        "limits": "Evidenza parziale.",
        "evidence_index": EVIDENCE,
        "cited_refs": ["S1", "T1"],
    }
    exported = ui.answer_markdown(turn, "it")
    assert "Che cos'è il biochar?" in exported
    assert "Fonti:" in exported
    assert "MR37-ita.pdf" in exported
    assert "biochar · reduces · erosione del suolo" in exported
    # Never cited, so it is not part of what the answer stands on.
    assert "Kenya Report_Full version.pdf" not in exported


def test_conversation_markdown_separates_the_turns():
    turns = [
        {"question": "prima", "body": "risposta uno", "evidence_index": [], "cited_refs": []},
        {"question": "seconda", "body": "risposta due", "evidence_index": [], "cited_refs": []},
    ]
    exported = ui.conversation_markdown("Sessione", turns, "it")
    assert exported.startswith("# Sessione")
    assert exported.count("---") == 1
    assert "risposta due" in exported


# --------------------------------------------------------------------------- #
# the language switch
# --------------------------------------------------------------------------- #


def test_both_languages_define_the_same_strings():
    """A half-translated switch shows English keys on an Italian page.

    The failure is silent: `t()` falls back to Italian, so an untranslated key
    reads as a working interface in the wrong language.
    """
    assert set(ui.STRINGS["it"]) == set(ui.STRINGS["en"])


def test_every_string_is_reachable_in_both_languages():
    for lang in ui.LANGUAGES:
        for key in ui.STRINGS[lang]:
            assert ui.t(lang, key, n=1, k=1, q="x", topics="y")


def test_t_falls_back_instead_of_raising():
    assert ui.t("de", "sources_title") == ui.STRINGS["it"]["sources_title"]
    assert ui.t("it", "chiave_inesistente") == "chiave_inesistente"
