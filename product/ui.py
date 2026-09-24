#!/usr/bin/env python3
"""Interface strings and the pure helpers that turn one ``result`` into a page.

Kept apart from ``product/app.py`` for two reasons. The interface has to exist in
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
from typing import Any, Iterable, Mapping, Sequence

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
# asks for a section with that title and leaves the formatting to it. It
# arrives bare on its own line, bold, as a heading, or inline with the section
# text after a colon ("**Limits and confidence**: the evidence is thin"), so
# the pattern is not anchored to the end of the line.
_LIMITS_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
    r"(?:Limiti e affidabilit[àa]|Limits and confidence)"
    r"[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*",
    re.M,
)


@dataclass(slots=True)
class AnswerParts:
    """The answer split into the pieces the page renders separately.

    Attributes:
        body: The prose, without the limits section or the source list.
        limits: The text of the limits section, without its heading.
    """

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
    """One graph fact, in the three parts a reader can be shown.

    Attributes:
        subject: The subject entity.
        predicate: The relation, lowercased with spaces; empty when the text
            did not parse as a triple.
        obj: The object entity.
        raw: The fact as it arrived, whitespace-normalised.
    """

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
    ``has component``): translating the relation types is a decision about the
    vocabulary, not about the interface, and inventing one here would put words
    in the graph's mouth.

    Args:
        text: The fact as rendered in the evidence index.

    Returns:
        The parsed fact, or one carrying only ``raw`` when it does not parse.
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
    """Everything one document contributed to one answer.

    Attributes:
        document: The source document's name.
        passages: Text evidence rows from this document.
        facts: Graph fact rows from this document.
    """

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
        """Start with nothing held back."""
        self._held = ""

    def reset(self) -> None:
        """Forget what is held, when a retry discards the text written so far."""
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


def document_label(document: str, titles: Mapping[str, str] | None = None) -> str:
    """What to call a document on screen: its title, or its filename shortened.

    A citation naming "REPORT MATTM_Definitivo.pdf" names the file someone
    happened to save. The work is called "Economia Circolare nel sistema
    agroalimentare piemontese", and that is what a reader is looking for.
    """
    name = str(document or "").strip()
    if not name:
        return ""
    title = (titles or {}).get(name, "")
    return title or short_doc_label(name) or name


def citation_files(titles: Mapping[str, str]) -> dict[str, str]:
    """Map the label in the prose back to the file the document is kept in."""
    return {
        stub: str(filename)
        for filename in (titles or {})
        if (stub := short_doc_label(str(filename)))
    }


def citation_titles(titles: Mapping[str, str]) -> dict[str, str]:
    """Map the label the engine wrote into the prose to the document's title.

    The engine renders a citation as ``short_doc_label(filename)`` plus the
    page, so that stub is the only handle the prose gives back — keying on it
    resolves the title without reading anything else out of the answer.
    """
    resolved: dict[str, str] = {}
    for filename, title in (titles or {}).items():
        stub = short_doc_label(str(filename))
        if stub and title:
            resolved.setdefault(stub, str(title))
    return resolved


def fit_title(title: str, budget: int) -> str:
    """Cut a title to length at a seam a reader recognises.

    Titles in this corpus carry their subtitle after a full stop or a colon —
    "The 3 C's of the Circular Economy for Food. A Conceptual Framework for
    Circular Design in the Food System" — so the head of one is a title in its
    own right, where a cut mid-phrase is just damage.
    """
    text = " ".join(str(title or "").split())
    if len(text) <= budget:
        return text
    for seam in (". ", ": ", " — ", " - "):
        head = text.split(seam, 1)[0]
        if 0 < len(head) <= budget:
            return head
    clipped = text[:budget].rsplit(" ", 1)[0].rstrip(" ,;:-–—")
    return (clipped or text[:budget]) + "…"


@dataclass(slots=True)
class Reference:
    """One work in an answer's reference list."""

    number: int = 0
    title: str = ""
    document: str = ""

    def entry(self) -> str:
        """The line the reader gets: the work, and the file it is kept in."""
        if self.document and self.document != self.title:
            return f"{self.title} — {self.document}"
        return self.title


def number_citations(
    text: str,
    titles: Mapping[str, str] | None = None,
    files: Mapping[str, str] | None = None,
    dim: bool = False,
) -> tuple[str, list[Reference]]:
    """Replace each citation with a number, and return the list it points to.

    An answer cites a median of ten times but names only three works, so the
    titles were being written out ten times inside the prose. Numbering them
    the way a paper does puts each work once, in a list under the answer, and
    leaves a marker in the sentence small enough to read past. The page stays
    inline: the same work is cited at different pages, and a number alone would
    not say which.

    Args:
        text: The answer, with the citations the engine rendered into it.
        titles: Stub -> title, from :func:`citation_titles`.
        files: Stub -> filename, so the list can name the file to open.

    Returns:
        The answer with numbered markers, and the works in order of first use.
    """
    if not text:
        return text, []

    order: dict[str, Reference] = {}

    def register(stub: str) -> Reference:
        key = (titles or {}).get(stub, stub)
        if key not in order:
            order[key] = Reference(
                number=len(order) + 1,
                title=key,
                document=(files or {}).get(stub, ""),
            )
        return order[key]

    def replace(match: re.Match[str]) -> str:
        marked: list[str] = []
        for part in match.group(1).split(";"):
            head, sep, pages = " ".join(part.split()).rpartition(", p")
            if not sep:
                continue
            reference = register(head.rstrip(" ,"))
            pages = _SAME_PAGE_RANGE_RE.sub(r"p. \1", "p" + pages).strip()
            marked.append(f"{reference.number}, {pages}")
        if not marked:
            return match.group(0)
        marker = "[" + "; ".join(marked) + "]"
        return f":gray[{marker}]" if dim else marker

    return _INLINE_CITATION_RE.sub(replace, text), list(order.values())


def _shorten_citation_part(
    part: str, doc_chars: int, titles: Mapping[str, str] | None = None
) -> str:
    """Trim one "document, p. N" to something that fits inside a sentence."""
    text = " ".join(part.split())
    head, sep, pages = text.rpartition(", p")
    if not sep:
        return text
    head = head.rstrip(" ,")
    head = fit_title((titles or {}).get(head, head), doc_chars)
    return f"{head}, p{pages}"


def style_citations(
    text: str,
    doc_chars: int = 60,
    dim: bool = True,
    titles: Mapping[str, str] | None = None,
) -> str:
    """Make the citations recede without taking anything away from them.

    A citation set in the same weight and colour as the sentence around it, in
    square brackets, breaks the line a reader is following. This shortens the
    document to a recognisable stub, collapses "p. 18-18" to "p. 18", and sets
    the whole thing small, grey and italic, in parentheses rather than brackets.

    Presentation only: the stored answer keeps its full labels, so what is
    copied or exported still names each document in full.

    Args:
        text: The answer as Markdown.
        doc_chars: Longest document stub kept inside a citation.
        dim: Also grey the citation out.

    Returns:
        The text with every page-bearing citation restyled.
    """
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        """Restyle one bracketed citation, or return it untouched."""
        inner = match.group(1)
        parts = [
            _shorten_citation_part(part, doc_chars, titles)
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
    """One answer's evidence, ordered so what it used comes first.

    Attributes:
        passages: Text evidence rows, cited ones first.
        facts: Graph fact rows, cited ones first.
    """

    passages: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)


def panel_evidence(
    evidence_index: Sequence[dict[str, Any]],
    cited_refs: Iterable[str] = (),
) -> PanelEvidence:
    """Order an answer's evidence for the box that holds it.

    Facts are not grouped by entity: a turn's triples rarely share a subject,
    so grouping would trade each line for a heading. What the reader needs is
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


def fact_line(row: dict[str, Any], titles: Mapping[str, str] | None = None) -> str:
    """One graph fact on one line, document included.

    One line rather than two, the fact and then its document underneath, so a
    turn with many facts does not make the panel outgrow the answer it
    explains.
    """
    sentence = readable_fact(row.get("text", "")).sentence()
    document = fit_title(document_label(str(row.get("document", "") or ""), titles), 60)
    return f"{sentence} · {document}" if document else sentence


def passage_label(row: dict[str, Any], titles: Mapping[str, str] | None = None) -> str:
    """The heading a passage is folded under: its document and its pages."""
    document = document_label(str(row.get("document", "") or ""), titles) or "?"
    pages = str(row.get("pages", "") or "")
    return f"{document} · {pages}" if pages else document


def compact_sources_line(
    evidence_index: Sequence[dict[str, Any]],
    cited_refs: Iterable[str] = (),
    lang: str = "it",
    titles: Mapping[str, str] | None = None,
) -> str:
    """The answer's sources as one line: documents, their cited pages, a count.

    The passages themselves stay reachable — the evidence panel holds every one
    of them — so what belongs under the answer is the short statement of where
    it came from, not a second copy of the evidence.

    Args:
        evidence_index: ``result["evidence_index"]``.
        cited_refs: ``result["citation_report"]["cited_refs"]``.
        lang: Interface language.

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
        label = document_label(entry.document, titles)
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


def answer_markdown(
    turn: dict[str, Any], lang: str, titles: Mapping[str, str] | None = None
) -> str:
    """One answer with its sources, as text a reader can paste elsewhere.

    Rebuilt from the turn's evidence, so what is copied carries the same
    provenance the page shows: a pasted answer must not lose where it came
    from.

    Args:
        turn: The rendered turn payload.
        lang: Interface language.

    Returns:
        The question, answer, limits and cited sources as Markdown.
    """
    parts: list[str] = []
    question = str(turn.get("question", "") or "").strip()
    if question:
        parts.append(f"**{question}**")
    body, references = number_citations(
        str(turn.get("body", "") or "").strip(),
        citation_titles(titles or {}),
        citation_files(titles or {}),
    )
    if body:
        parts.append(body)
    limits = str(turn.get("limits", "") or "").strip()
    if limits:
        parts.append(f"_{t(lang, 'limits_title')}_\n\n{limits}")

    if references:
        # The numbers in the text point here, so what is pasted elsewhere
        # carries the same list the reader saw.
        lines = [f"{t(lang, 'sources_title')}:"]
        lines += [f"{ref.number}. {ref.entry()}" for ref in references]
        parts.append("\n".join(lines))

    facts = [
        fact
        for entry in evidence_by_document(
            turn.get("evidence_index") or [], turn.get("cited_refs") or [], only_cited=True
        )
        for fact in entry.facts
    ]
    if facts:
        lines = [f"{t(lang, 'graph_facts')}:"]
        lines += [f"- {readable_fact(fact['text']).sentence()}" for fact in facts]
        parts.append("\n".join(lines))

    return "\n\n".join(parts).strip()


def conversation_markdown(
    title: str,
    turns: Sequence[dict[str, Any]],
    lang: str,
    titles: Mapping[str, str] | None = None,
) -> str:
    """The whole conversation as one Markdown document."""
    blocks = [answer_markdown(turn, lang, titles) for turn in turns]
    body = "\n\n---\n\n".join(block for block in blocks if block)
    # The rule separates one exchange from the next; the heading stays out of
    # the join so no rule is drawn straight under the title.
    return f"# {title}".strip() + ("\n\n" + body if body else "") + "\n"
