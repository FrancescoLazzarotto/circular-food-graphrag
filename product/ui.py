#!/usr/bin/env python3
"""Interface strings and the pure helpers that turn one ``result`` into a page.

Split out of ``product/app.py`` for two reasons. The interface has to exist in
Italian and English, and a dictionary is the only form of that which stays
reviewable. And everything here is pure — no Streamlit, no agent — so the parts
that decide what a reader is *told* can be tested without starting either.

Nothing in this module reads the answer's prose to recover data. The evidence
blocks are rebuilt from ``evidence_index`` and ``citation_report``; the only
thing taken from the answer string is where its own sections begin, so the
engine's closing source list is not printed twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Read-only use of the engine's own filename shortener, so a document is named
# on screen the way it is named inside an answer's citations.
from graphrag.agent.evidence import short_doc_label

# --------------------------------------------------------------------------- #
# interface strings
# --------------------------------------------------------------------------- #

LANGUAGES: dict[str, str] = {"it": "Italiano", "en": "English"}

STRINGS: dict[str, dict[str, str]] = {
    "it": {
        "ask_placeholder": "Scrivi qui la tua domanda...",
        "thinking": "Sto pensando...",
        "answer_language_note": "Rispondo nella lingua della domanda.",
        # status
        "status_ok": "Sistema operativo",
        "status_reduced": "Modalità ridotta",
        "status_reduced_why": "Il grafo principale non risponde: sto usando la copia locale.",
        "status_degraded_why": "La ricerca cross-lingua non era disponibile per questa risposta.",
        # metadata bar
        "meta_passages": "{n} passaggi",
        "meta_passages_one": "1 passaggio",
        "meta_facts": "{n} fatti dal grafo",
        "meta_facts_one": "1 fatto dal grafo",
        "meta_documents": "{n} documenti",
        "meta_documents_one": "1 documento",
        "meta_seconds": "{n} s",
        # citations
        "cit_clean": "{n} citazioni, tutte verificate",
        "cit_phantom": "{n} citazioni, {k} non verificate",
        "cit_none": "Nessuna citazione in questa risposta",
        "cit_help": "Ogni riferimento è confrontato con le evidenze recuperate: "
                    "un riferimento che non corrisponde a nessuna viene segnalato.",
        # sections
        "sources_title": "Fonti",
        "limits_title": "Limiti e affidabilità",
        "evidence_title": "Evidenze",
        "evidence_of_last": "Evidenze dell'ultima risposta",
        "evidence_none": "Nessuna evidenza da mostrare per questa risposta.",
        "evidence_expander": "Evidenze di questa risposta",
        "passages": "Passaggi",
        "graph_facts": "Fatti dal grafo",
        "cited_passages": "passaggi citati",
        "not_cited": "recuperato, non citato",
        "also_retrieved": "Recuperato e non usato",
        # feedback
        "fb_useful": "Risposta utile",
        "fb_wrong": "Risposta sbagliata o inutile",
        "fb_thanks": "Grazie, registrato.",
        "fb_why": "Che cosa non andava?",
        "fb_reason_incomplete": "Incompleta",
        "fb_reason_offtarget": "Non risponde alla domanda",
        "fb_reason_sources": "Fonti sbagliate",
        "fb_note_placeholder": "Aggiungi un dettaglio (facoltativo)",
        "fb_send": "Invia",
        "fb_sent": "Registrato, grazie.",
        # conversations
        "conversations": "Conversazioni",
        "new_chat": "+ Nuova conversazione",
        "empty_chat": "Chat vuota",
        "delete": "Elimina conversazione",
        "delete_confirm": "Elimina definitivamente questa conversazione?",
        "delete_yes": "Sì, elimina",
        "delete_no": "Annulla",
        "thread_following": "Argomenti salvati in memoria: {topics}",
        "thread_reset": "Cancella memoria",
        # export
        "export": "Esporta",
        "copy_with_sources": "Copia con le fonti",
        "copy_hint": "Seleziona il testo e copialo.",
        "download_conversation": "Scarica la conversazione (Markdown)",
        # settings
        "advanced": "Impostazioni avanzate",
        "model": "Modello",
        "interface_language": "Lingua dell'interfaccia",
        # errors
        "err_service": "Il servizio non è raggiungibile in questo momento. "
                       "Riprova fra poco: la domanda non ha nulla che non va.",
        "err_question": "Questa domanda non è andata a buon fine. "
                        "Riprova, magari riformulandola.",
        # out of domain
        "oos_title": "Fuori dall'ambito coperto",
        # A greeting is not a refusal, and titling it as one tells whoever
        # typed "ciao" that they did something wrong.
        "meta_title": "Ecco di cosa mi occupo",
        "oos_covers": "Rispondo solo sull'economia circolare del cibo, "
                      "sulla base di {n} documenti.",
        "oos_try": "Prova per esempio:",
        # rewrite notice
        "rewritten_as": "Ho cercato nei documenti come: «{q}»",
        "rewrite_literal": "Rifai con la domanda letterale",
    },
    "en": {
        "ask_placeholder": "Type your question here...",
        "thinking": "Thinking...",
        "answer_language_note": "I answer in the language of the question.",
        "status_ok": "System operational",
        "status_reduced": "Reduced mode",
        "status_reduced_why": "The primary graph is not answering: using the local copy.",
        "status_degraded_why": "Cross-lingual search was unavailable for this answer.",
        "meta_passages": "{n} passages",
        "meta_passages_one": "1 passage",
        "meta_facts": "{n} graph facts",
        "meta_facts_one": "1 graph fact",
        "meta_documents": "{n} documents",
        "meta_documents_one": "1 document",
        "meta_seconds": "{n} s",
        "cit_clean": "{n} citations, all verified",
        "cit_phantom": "{n} citations, {k} unverified",
        "cit_none": "No citations in this answer",
        "cit_help": "Every reference is checked against the retrieved evidence: "
                    "a reference matching none of it is flagged.",
        "sources_title": "Sources",
        "limits_title": "Limits and confidence",
        "evidence_title": "Evidence",
        "evidence_of_last": "Evidence for the latest answer",
        "evidence_none": "No evidence to show for this answer.",
        "evidence_expander": "Evidence for this answer",
        "passages": "Passages",
        "graph_facts": "Graph facts",
        "cited_passages": "cited passages",
        "not_cited": "retrieved, not cited",
        "also_retrieved": "Retrieved, not used",
        "fb_useful": "Useful answer",
        "fb_wrong": "Wrong or useless answer",
        "fb_thanks": "Thanks, recorded.",
        "fb_why": "What went wrong?",
        "fb_reason_incomplete": "Incomplete",
        "fb_reason_offtarget": "Does not answer the question",
        "fb_reason_sources": "Wrong sources",
        "fb_note_placeholder": "Add a detail (optional)",
        "fb_send": "Send",
        "fb_sent": "Recorded, thank you.",
        "conversations": "Conversations",
        "new_chat": "+ New conversation",
        "empty_chat": "New conversation",
        "delete": "Delete conversation",
        "delete_confirm": "Delete this conversation for good?",
        "delete_yes": "Yes, delete",
        "delete_no": "Cancel",
        "thread_following": "Following the thread on: {topics}",
        "thread_reset": "Start again without the thread",
        "export": "Export",
        "copy_with_sources": "Copy with sources",
        "copy_hint": "Select the text and copy it.",
        "download_conversation": "Download the conversation (Markdown)",
        "advanced": "Advanced settings",
        "model": "Model",
        "interface_language": "Interface language",
        "err_service": "The service is unreachable right now. "
                       "Try again shortly: there is nothing wrong with your question.",
        "err_question": "This question did not go through. "
                        "Try again, perhaps rephrasing it.",
        "oos_title": "Outside the covered scope",
        "meta_title": "What I can help with",
        "oos_covers": "I only answer on the circular economy of food, "
                      "from {n} documents.",
        "oos_try": "Try for example:",
        "rewritten_as": "I searched the documents as: «{q}»",
        "rewrite_literal": "Redo with the literal question",
    },
}


def count_label(lang: str, key: str, n: int) -> str:
    """A counted noun that reads right at one.

    "1 passaggi · 1 fatti dal grafo · 1 documenti" is the sort of detail that
    makes an interface look unfinished, and every count on this page can be one.
    """
    singular = f"{key}_one"
    if int(n) == 1 and singular in (STRINGS.get(lang) or STRINGS["it"]):
        return t(lang, singular)
    return t(lang, key, n=n)


def t(lang: str, key: str, **kwargs: Any) -> str:
    """Look up an interface string, falling back to Italian then to the key."""
    table = STRINGS.get(lang) or STRINGS["it"]
    text = table.get(key) or STRINGS["it"].get(key) or key
    return text.format(**kwargs) if kwargs else text


# --------------------------------------------------------------------------- #
# answer sections
# --------------------------------------------------------------------------- #

# The engine appends its own closing source list (evidence.render_grouped_
# reference_list) and instructs the model to end with a limits section
# (llm/prompts.py). Both headings are fixed strings in the two answer
# languages. They are located here only to *split* the text: the source data
# itself is rebuilt from evidence_index, never parsed back out of the prose.
_SOURCES_RE = re.compile(r"^\s*(?:\*\*|#{1,6}\s*)?(?:Fonti|Sources)\s*:?\s*\*{0,2}\s*$", re.M)
# The limits heading is written by the model, not by the renderer: the prompt
# asks for a section with that title and leaves the formatting to it. Measured
# on 110 archived answers, it arrives bare on its own line, bold, as a heading —
# and, in 47 of them, inline with the section text after a colon
# ("**Limits and confidence**: the evidence is thin"). A pattern anchored to the
# end of the line missed exactly those, and the section stayed in the prose.
_LIMITS_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
    r"(?:Limiti e affidabilit[àa]|Limits and confidence)"
    r"[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*",
    re.M,
)


@dataclass(slots=True)
class AnswerParts:
    """The answer split into the pieces the page renders separately."""

    body: str = ""
    limits: str = ""


def split_answer(answer: str) -> AnswerParts:
    """Separate the prose, the limits section and the engine's source list.

    The source list is dropped rather than returned: the page rebuilds it from
    the evidence index, and printing both would show the same documents twice.

    Args:
        answer: The answer exactly as the engine produced it.

    Returns:
        The prose body and the limits section, either of which may be empty.
    """
    text = str(answer or "").strip()
    if not text:
        return AnswerParts()

    matches = list(_SOURCES_RE.finditer(text))
    if matches:
        # The engine appends it last, so the final heading is the real one; an
        # earlier "Fonti:" inside the prose stays where the model put it.
        text = text[: matches[-1].start()].rstrip()

    limits = ""
    limit_matches = list(_LIMITS_RE.finditer(text))
    if limit_matches:
        last = limit_matches[-1]
        # `end()` stops after the heading and its punctuation, so a section that
        # starts on the same line is kept whole.
        limits = text[last.end():].strip()
        text = text[: last.start()].rstrip()

    return AnswerParts(body=text, limits=limits)


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #

# A triple reaches the evidence index already rendered as "(subject, PREDICATE,
# object)". The predicate is always a vocabulary token — uppercase, no spaces
# (kg_pipeline/relation_vocab_circular_v1_draft.json) — which is what makes the
# subject and the object recoverable even when either contains a comma.
_TRIPLE_RE = re.compile(r"^\((.+?),\s*([A-Z][A-Z0-9_]+),\s*(.+)\)$", re.S)


@dataclass(slots=True)
class Fact:
    """One graph fact, in the three parts a reader can be shown."""

    subject: str = ""
    predicate: str = ""
    obj: str = ""
    raw: str = ""

    def sentence(self) -> str:
        """The fact as a line of text, or the raw form when it did not parse."""
        if not self.predicate:
            return self.raw
        return f"{self.subject} · {self.predicate} · {self.obj}"


def readable_fact(text: str) -> Fact:
    """Turn ``(a, PREDICATE, b)`` into its parts, with the relation lowercased.

    Relation names stay in the vocabulary's own English (``HAS_COMPONENT`` ->
    ``has component``): translating the 35 relation types is a decision about
    the vocabulary, not about the interface, and inventing one here would put
    words in the graph's mouth.
    """
    raw = " ".join(str(text or "").split())
    match = _TRIPLE_RE.match(raw)
    if not match:
        return Fact(raw=raw)
    subject, predicate, obj = match.groups()
    return Fact(
        subject=subject.strip(),
        predicate=predicate.replace("_", " ").lower().strip(),
        obj=obj.strip(),
        raw=raw,
    )


@dataclass(slots=True)
class DocumentEvidence:
    """Everything one document contributed to one answer."""

    document: str = ""
    passages: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_refs(self) -> int:
        """Distinct pieces of evidence from this document that were cited."""
        return len(self.passages) + len(self.facts)

    def pages(self) -> list[str]:
        """The cited pages, in order, without repeats."""
        seen = [str(p.get("pages", "") or "").strip() for p in self.passages]
        return list(dict.fromkeys(page for page in seen if page))


def evidence_by_document(
    evidence_index: Sequence[dict[str, Any]],
    cited_refs: Iterable[str] = (),
    *,
    only_cited: bool = True,
    unnamed_label: str = "documento non indicato",
) -> list[DocumentEvidence]:
    """Group evidence items by their source document.

    Args:
        evidence_index: ``result["evidence_index"]``, already serialised.
        cited_refs: ``result["citation_report"]["cited_refs"]``.
        only_cited: Keep just what the answer actually cited. False returns
            everything retrieved, which is what the evidence panel shows.
        unnamed_label: Stand-in for evidence with no document attached.

    Returns:
        One entry per document, passages before facts, in index order — which
        is retrieval order, so the strongest evidence comes first.
    """
    wanted = {str(ref).strip().upper() for ref in cited_refs if str(ref).strip()}
    grouped: dict[str, DocumentEvidence] = {}

    for item in evidence_index:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id", "") or "").strip().upper()
        if only_cited and ref_id not in wanted:
            continue
        document = str(item.get("source_doc", "") or "").strip() or unnamed_label
        entry = grouped.setdefault(document, DocumentEvidence(document=document))
        row = {
            "ref_id": ref_id,
            "text": str(item.get("text", "") or "").strip(),
            "pages": str(item.get("pages", "") or "").strip(),
            "chunk_id": str(item.get("chunk_id", "") or "").strip(),
            "cited": ref_id in wanted,
        }
        if str(item.get("kind", "")) == "triple":
            entry.facts.append(row)
        else:
            entry.passages.append(row)

    return list(grouped.values())


# The tags the model writes while it is still writing: "[S1]", "[T12]",
# "[S1, T2]". The engine swaps them for document labels once the answer is
# complete, so a reader watching the text arrive would otherwise see ids that
# mean nothing to them and then change under their eyes.
_RAW_TAG_RE = re.compile(r"^\[(?:[STst]\s?\d{1,3})(?:\s*[,;]\s*[STst]\s?\d{1,3})*\]$")
_TAG_PREFIX_RE = re.compile(r"^\[[STst\s\d,;]*$")


class StreamScrubber:
    """Hide the raw reference tags from text arriving a fragment at a time.

    A tag can be split across chunks — "[S", "1]" — so anything from an open
    bracket is held back until it is known to be a tag or not. Everything else
    passes through untouched and immediately.
    """

    __slots__ = ("_held",)

    # Longest thing that can still turn out to be a tag; past it, the bracket
    # belonged to the prose and the text is released.
    _MAX_HOLD = 48

    def __init__(self) -> None:
        self._held = ""

    def reset(self) -> None:
        self._held = ""

    def feed(self, piece: str) -> str:
        """Return the part of ``piece`` that can be shown now."""
        out: list[str] = []
        for char in str(piece or ""):
            if self._held:
                self._held += char
                if char == "]":
                    if not _RAW_TAG_RE.match(self._held):
                        out.append(self._held)
                    self._held = ""
                elif not _TAG_PREFIX_RE.match(self._held) or len(self._held) > self._MAX_HOLD:
                    out.append(self._held)
                    self._held = ""
            elif char == "[":
                self._held = char
            else:
                out.append(char)
        return "".join(out)

    def flush(self) -> str:
        """Release whatever is still held, at the end of the stream."""
        held, self._held = self._held, ""
        return "" if _RAW_TAG_RE.match(held) else held


# A citation the engine has already rendered for a reader: "[MR37, p. 35]", or
# several separated by ";". Only brackets carrying a page marker are touched, so
# square brackets the model wrote for its own reasons are left alone.
_INLINE_CITATION_RE = re.compile(r"\[([^\[\]]{0,300}?pp?\.[^\[\]]{0,80}?)\]")
_SAME_PAGE_RANGE_RE = re.compile(r"\bp\. (\d+)-\1\b")


def _shorten_citation_part(part: str, doc_chars: int) -> str:
    """Trim one "document, p. N" to something that fits inside a sentence."""
    text = " ".join(part.split())
    head, sep, pages = text.rpartition(", p")
    if not sep:
        return text
    head = head.rstrip(" ,")
    # short_doc_label has already cut the filename down to ~34 characters; a
    # citation sits mid-sentence, where that is still most of a line.
    if len(head) > doc_chars:
        head = head[:doc_chars].rstrip(" ,-–—") + "…"
    return f"{head}, p{pages}"


def style_citations(text: str, doc_chars: int = 16, dim: bool = True) -> str:
    """Make the citations recede without taking anything away from them.

    Measured over 572 citations in the archived sessions, the median one is 23
    characters and the longest 84 — set in the same weight and colour as the
    sentence around it, in square brackets, which is what breaks the line a
    reader is following. This shortens the document to a recognisable stub,
    collapses "p. 18-18" to "p. 18", and sets the whole thing small, grey and
    italic, in parentheses rather than brackets.

    Presentation only: the stored answer keeps its full labels, so what is
    copied or exported still names each document in full.
    """
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = [
            _shorten_citation_part(part, doc_chars)
            for part in inner.split(";")
            if part.strip()
        ]
        if not parts:
            return match.group(0)
        joined = _SAME_PAGE_RANGE_RE.sub(r"p. \1", " · ".join(parts))
        return f":gray[*({joined})*]" if dim else f"*({joined})*"

    return _INLINE_CITATION_RE.sub(replace, text)


@dataclass(slots=True)
class PanelEvidence:
    """One answer's evidence, ordered so what it used comes first."""

    passages: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)


def panel_evidence(
    evidence_index: Sequence[dict[str, Any]],
    cited_refs: Iterable[str] = (),
) -> PanelEvidence:
    """Order an answer's evidence for the box that holds it.

    Measured over 240 gold answers, a turn retrieves a median of 20 triples
    across 19 distinct subjects, so grouping them by entity compacts nothing —
    it would trade twenty lines for nineteen headings. What the reader needs is
    the distinction the panel exists to draw: an answer stands on what it
    cited, and the rest is the honest remainder, kept and marked as such.

    Args:
        evidence_index: ``result["evidence_index"]``.
        cited_refs: ``result["citation_report"]["cited_refs"]``.

    Returns:
        Passages and facts, cited ones first, each row carrying ``cited``.
    """
    wanted = {str(ref).strip().upper() for ref in cited_refs if str(ref).strip()}
    cited: dict[str, list[dict[str, Any]]] = {"text": [], "triple": []}
    spare: dict[str, list[dict[str, Any]]] = {"text": [], "triple": []}

    for item in evidence_index:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id", "") or "").strip().upper()
        kind = "triple" if str(item.get("kind", "")) == "triple" else "text"
        row = {
            "ref_id": ref_id,
            "text": str(item.get("text", "") or "").strip(),
            "pages": str(item.get("pages", "") or "").strip(),
            "document": str(item.get("source_doc", "") or "").strip(),
            "cited": ref_id in wanted,
        }
        (cited if row["cited"] else spare)[kind].append(row)

    # Retrieval order is relevance order, so an answer that cited nothing still
    # leads with the strongest thing the collection returned for it.
    return PanelEvidence(
        passages=cited["text"] + spare["text"],
        facts=cited["triple"] + spare["triple"],
    )


def fact_line(row: dict[str, Any]) -> str:
    """One graph fact on one line, document included.

    The panel used to spend two lines on each: the fact, then its document
    underneath. On a turn with twenty facts that is forty lines beside an answer
    of ten, which is what made the column outgrow the thing it was explaining.
    """
    sentence = readable_fact(row.get("text", "")).sentence()
    document = short_doc_label(str(row.get("document", "") or ""))
    return f"{sentence} · {document}" if document else sentence


def passage_label(row: dict[str, Any]) -> str:
    """The heading a passage is folded under: its document and its pages."""
    document = str(row.get("document", "") or "") or "?"
    pages = str(row.get("pages", "") or "")
    return f"{document} · {pages}" if pages else document


def compact_sources_line(
    evidence_index: Sequence[dict[str, Any]],
    cited_refs: Iterable[str] = (),
    lang: str = "it",
) -> str:
    """The answer's sources as one line: documents, their cited pages, a count.

    The first version gave every document a heading, a page caption and a row
    of buttons, which under a seven-paragraph answer was longer than some of
    the answers. The passages themselves stayed reachable — the evidence panel
    holds every one of them — so what belongs under the answer is the short
    statement of where it came from, not a second copy of the evidence.

    Returns:
        The line, or an empty string when the answer cited nothing.
    """
    documents = evidence_by_document(evidence_index, cited_refs, only_cited=True)
    if not documents:
        return ""

    bits: list[str] = []
    facts = 0
    for entry in documents:
        facts += len(entry.facts)
        label = short_doc_label(entry.document) or entry.document
        pages = entry.pages()
        bits.append(f"{label} ({', '.join(pages)})" if pages else label)
    if facts:
        bits.append(count_label(lang, "meta_facts", facts))
    return f"{t(lang, 'sources_title')}: " + " · ".join(bits)


def retrieval_counts(result: dict[str, Any]) -> dict[str, int]:
    """The three numbers the metadata bar shows, straight from ``result``."""
    text_sources = result.get("retrieved_text_sources") or []
    triples = result.get("kg_triples") or []
    evidence = result.get("evidence_index") or []
    documents = {
        str(item.get("source_doc", "") or "").strip()
        for item in evidence
        if isinstance(item, dict) and str(item.get("source_doc", "") or "").strip()
    }
    return {
        "passages": len(text_sources),
        "facts": len(triples),
        "documents": len(documents),
    }


def citation_summary(citation_report: dict[str, Any] | None, lang: str) -> tuple[str, str]:
    """Describe the citation check for a reader.

    Returns:
        ``(state, text)`` where state is ``"clean"``, ``"phantom"`` or
        ``"none"``. ``insufficient_answer`` deliberately plays no part: it is
        documented in this repository as flagging invented answers with hedging
        in the tail, so it cannot carry a reliability claim.
    """
    report = citation_report if isinstance(citation_report, dict) else {}
    total = int(report.get("total_citations", 0) or 0)
    phantom = len(report.get("phantom_refs") or [])
    if total <= 0:
        return "none", t(lang, "cit_none")
    if phantom > 0:
        return "phantom", t(lang, "cit_phantom", n=total, k=phantom)
    return "clean", t(lang, "cit_clean", n=total)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


def model_display_name(model_id: str) -> str:
    """A model name a reader can hold, without the vendor path or the port.

    ``RedHatAI/Qwen3.8-27B-INT4`` -> ``Qwen3.8 27B``. The exact id stays in the
    session log and in the debug caption, which is where it is needed.
    """
    name = str(model_id or "").split("/")[-1].strip()
    if not name:
        return ""
    # Quantisation and serving suffixes identify the artifact, not the model.
    name = re.sub(
        r"[-_.](?:awq|gptq|int4|int8|fp8|fp16|bf16|instruct|chat|hf)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(
        r"[-_.](?:awq|gptq|int4|int8|fp8|fp16|bf16|instruct|chat|hf)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return name.replace("-", " ").replace("_", " ").strip()


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def answer_markdown(turn: dict[str, Any], lang: str) -> str:
    """One answer with its sources, as text a reader can paste elsewhere.

    Rebuilt from the turn's evidence, so what is copied carries the same
    provenance the page shows — the complaint the export exists to answer is a
    pasted answer that has lost where it came from.
    """
    parts: list[str] = []
    question = str(turn.get("question", "") or "").strip()
    if question:
        parts.append(f"**{question}**")
    body = str(turn.get("body", "") or "").strip()
    if body:
        parts.append(body)
    limits = str(turn.get("limits", "") or "").strip()
    if limits:
        parts.append(f"_{t(lang, 'limits_title')}_\n\n{limits}")

    documents = evidence_by_document(
        turn.get("evidence_index") or [],
        turn.get("cited_refs") or [],
        only_cited=True,
    )
    if documents:
        lines = [f"{t(lang, 'sources_title')}:"]
        for entry in documents:
            lines.append(f"- **{entry.document}**")
            pages = entry.pages()
            if pages:
                lines.append(f"  - {t(lang, 'cited_passages')}: {', '.join(pages)}")
            for fact in entry.facts:
                lines.append(f"  - {readable_fact(fact['text']).sentence()}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts).strip()


def conversation_markdown(title: str, turns: Sequence[dict[str, Any]], lang: str) -> str:
    """The whole conversation as one Markdown document."""
    blocks = [answer_markdown(turn, lang) for turn in turns]
    body = "\n\n---\n\n".join(block for block in blocks if block)
    # The rule separates one exchange from the next; putting the heading in the
    # same join drew one straight under the title as well.
    return f"# {title}".strip() + ("\n\n" + body if body else "") + "\n"
