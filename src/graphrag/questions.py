"""Deterministic question typing: definitional and enumerative questions.

Two question shapes need a different retrieval and a different answer:

* **definitional** — "che cos'è SEeD?". The answer *is* the author's wording, so
  the chunk carrying the verbatim definition must outrank the chunks that merely
  mention the term, and the answer must quote before it paraphrases.
* **enumerative** — "quali sono le 5 filiere?". The items usually live in one
  list inside one document, on contiguous pages, so the per-document cap that
  diversifies every other question actively hurts this one.

Both detectors are regex-based on purpose: an LLM classifier would add a call, a
failure mode and nondeterminism to every turn, for a decision a pattern makes
just as well and a test can pin down.

Nothing here imports from :mod:`graphrag.agent` or :mod:`graphrag.kg`: the
retriever, the prompt library and the agent all need it, and a shared leaf
module is what keeps that from becoming an import cycle.
"""

from __future__ import annotations

import re
import unicodedata

# The question openers that announce a definition. Ordered longest-first inside
# each language so "cosa si intende per X" is not consumed by a bare "cosa".
_DEFINITIONAL_OPENERS = (
    r"(?:che\s+)?cos(?:a|['’])?\s*[eè]\b",
    r"che\s+cosa\s+(?:sono|significa|significano|vuol\s+dire|vogliono\s+dire)\b",
    r"cosa\s+si\s+intende\s+(?:per|con)\b",
    r"che\s+cosa\s+si\s+intende\s+(?:per|con)\b",
    r"cosa\s+(?:significa|significano|vuol\s+dire|vogliono\s+dire)\b",
    r"(?:qual\s+[eè]\s+la\s+|dammi\s+la\s+|la\s+)?definizion[ei]\s+(?:di|del|dello|della|dei|degli|delle|d['’])\b",
    r"come\s+(?:si\s+definisce|viene\s+definit[oa])\b",
    r"what\s+(?:is|are)\s+meant\s+by\b",
    r"what\s+(?:is|are)\b",
    r"what\s+do(?:es)?\s+(?:the\s+term\s+)?",
    r"(?:the\s+)?definition\s+of\b",
    r"define\b",
    r"what\s+does\s+.{0,40}?\s+stand\s+for\b",
)
_DEFINITIONAL_RE = re.compile(
    r"(?:^|[\s,;:(])(?:" + "|".join(_DEFINITIONAL_OPENERS) + r")",
    re.IGNORECASE,
)

# Cut the term at the first boundary that starts a second question or a second
# clause: "what is scotta and how does it differ from whey?" defines "scotta".
_TERM_BOUNDARY_RE = re.compile(
    r"\?|[,;:]|\be\s+(?:che|come|quali|quale|quanto|perch[eé]|a\s+cosa)\b"
    r"|\band\s+(?:how|what|why|which|where|who)\b|\bmean(?:s)?\b",
    re.IGNORECASE,
)
_LEADING_ARTICLES_RE = re.compile(
    r"^(?:il|lo|la|i|gli|le|un|uno|una|un['’]|l['’]|the|a|an)\s+",
    re.IGNORECASE,
)
# A term made only of these refers back to the conversation, not to a concept:
# "cosa vuol dire questo per le imprese?" is not a definitional question.
_DEICTIC_TERMS = {
    "questo", "questa", "questi", "queste", "quello", "quella", "quelli",
    "quelle", "cio", "ciò", "this", "that", "these", "those", "it", "they",
    # Possessives point back at a noun introduced earlier in the question, so
    # the opener that produced them belongs to a second clause: "…extracted from
    # grape pomace, and what are their applications?" must not yield the term
    # "their applications".
    "their", "its", "his", "her", "our", "your", "loro", "sua", "sue", "suoi",
}
# Openers that leave a verb phrase behind instead of a concept.
_NON_TERM_STARTERS = {
    "succede", "succedeva", "successo", "accade", "capita", "significa",
    "happens", "happened", "means",
}
_MAX_TERM_TOKENS = 6
_MAX_TERM_CHARS = 60
_MIN_TERM_CHARS = 2

_ENUMERATIVE_RE = re.compile(
    r"\bquali\s+sono\b|\belenca\b|\belenco\s+d|\bfammi\s+(?:un\s+)?elenco\b"
    r"|\bquant[ei]\b|\btutt[ei]\s+(?:i|le|gli)\b|\bquali\s+(?:sono\s+)?le?\b"
    r"|\blist\b|\bwhich\s+are\b|\bhow\s+many\b|\benumerate\b|\ball\s+the\b"
    r"|\bnam[e]\s+the\b",
    re.IGNORECASE,
)
# "le 5 filiere", "the three pillars": an explicit count is a promise of a list.
_COUNTED_LIST_RE = re.compile(
    r"\b(?:\d{1,2}|due|tre|quattro|cinque|sei|sette|otto|nove|dieci"
    r"|two|three|four|five|six|seven|eight|nine|ten)\s+[a-zà-ÿ]{4,}",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    """Lower-case and strip accents, keeping punctuation and word order."""
    folded = unicodedata.normalize("NFKD", str(text or ""))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(folded.lower().split())


def _normalize(text: str) -> str:
    """Fold case, accents *and* punctuation so PDF text and prose compare."""
    return " ".join(re.sub(r"[^0-9a-zA-Z]+", " ", _fold(text)).split())


def _term_pattern(term: str) -> str:
    """Regex for ``term`` tolerant of the whitespace PDFs insert between words."""
    parts = [re.escape(part) for part in _fold(term).split() if part]
    return r"\s+".join(parts)


def definitional_term(question: str) -> str:
    """Extract the term a definitional question asks about.

    Args:
        question: The user question, in Italian or English.

    Returns:
        The term, or an empty string when the question is not definitional or
        when what follows the opener is a reference to the conversation
        ("cosa vuol dire questo?") rather than a concept.
    """
    text = " ".join(str(question or "").split())
    match = _DEFINITIONAL_RE.search(text)
    if match is None:
        return ""

    tail = text[match.end():].strip()
    boundary = _TERM_BOUNDARY_RE.search(tail)
    if boundary is not None:
        tail = tail[: boundary.start()]
    tail = _LEADING_ARTICLES_RE.sub("", tail.strip())
    term = tail.strip(" \t.?!\"'«»“”()")

    if len(term) < _MIN_TERM_CHARS or len(term) > _MAX_TERM_CHARS:
        return ""
    tokens = term.split()
    if not tokens or len(tokens) > _MAX_TERM_TOKENS:
        return ""
    first = tokens[0].lower().strip(".,")
    if first in _DEICTIC_TERMS or first in _NON_TERM_STARTERS:
        return ""
    if _normalize(term) in {_normalize(word) for word in _DEICTIC_TERMS}:
        return ""
    return term


def is_definitional(question: str) -> bool:
    """Whether the question asks what something is.

    A definitional opener alone is not enough on two counts. The term it
    introduces must look like a concept, otherwise the answer has nothing to
    quote a definition of. And the question must not be asking for a list: "What
    are the four implementation cycles of metabolisation?" opens like a
    definition but promises an enumeration, and sending it through the
    quote-then-explain prompt spends the answer's opening on a quotation of a
    list item.

    Args:
        question: The user question, in Italian or English.

    Returns:
        True when the question is not enumerative and yields a term.
    """
    if is_enumerative(question):
        return False
    return bool(definitional_term(question))


def is_enumerative(question: str) -> bool:
    """Whether the question asks for a list of items.

    Used to relax the per-document cap: an enumeration is usually a single list
    on contiguous pages of one document, and capping that document truncates the
    list — the opposite of what the cap is for.

    Args:
        question: The user question, in Italian or English.

    Returns:
        True when it contains a list opener ("quali sono", "list", "how
        many", ...) or an explicit count ("le 5 filiere").
    """
    text = " ".join(str(question or "").split())
    if not text:
        return False
    return bool(_ENUMERATIVE_RE.search(text) or _COUNTED_LIST_RE.search(text))


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Þ«\"'(])")
_MARKDOWN_NOISE_RE = re.compile(r"[*_`]{1,3}|^#{1,6}\s*|^\s*[-•*]\s+", re.MULTILINE)
_MIN_SENTENCE_CHARS = 40
_MAX_SENTENCE_CHARS = 420
_MIN_SENTENCE_WORDS = 8
# Captions carry no terminal punctuation either, so they glue themselves to the
# paragraph below exactly as headings do — and a caption reads like a definition
# ("Fig. 28 - Rappresentazione delle 3C della Circular Economy for Food") while
# defining nothing.
_CAPTION_START_RE = re.compile(
    r"^(?:fig|figura|figure|tab|tabella|table|graf|grafico|chart|box|foto|photo|"
    r"immagine|image|source|fonte)\b\.?\s*\d*\s*[-–—:.]?",
    re.IGNORECASE,
)


def _candidate_sentences(text: str) -> list[str]:
    """Split a chunk into sentences that can stand alone as a quotation.

    Chunks come out of PDFs as markdown: headings, bullet markers and bold
    runs. A heading has no terminal punctuation, so split by sentence alone it
    glues itself to the paragraph below ("Introduction: The Systemic Event
    Design Project (SEeD)** Food is characterized by…"). Markdown markers are
    removed and lines are split before sentences for that reason.

    Args:
        text: A retrieved chunk.

    Returns:
        The candidate sentences, in order.
    """
    cleaned = _MARKDOWN_NOISE_RE.sub(" ", str(text or ""))
    sentences: list[str] = []
    for line in cleaned.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        sentences.extend(_SENTENCE_SPLIT_RE.split(line))
    return sentences


def definition_sentence(text: str, term: str) -> str:
    """Return the sentence of ``text`` that defines ``term``, if there is one.

    Asking the model to copy a passage word for word does not work reliably
    across languages: with the answer language pinned to the question's, a
    definition written in English comes back translated, accurate and no longer
    a quotation. Extracting the sentence here removes the model from the loop —
    what gets quoted is the source, by construction.

    Args:
        text: A retrieved chunk.
        term: The term extracted by :func:`definitional_term`.

    Returns:
        The best defining sentence — with a long narrative preamble elided as
        ``[...]`` — or an empty string when no sentence of ``text`` defines the
        term.
    """
    tp = _term_pattern(term)
    best = ""
    best_score = 0.0
    for sentence in _candidate_sentences(text):
        candidate = sentence.strip(" \t–—-")
        if not (_MIN_SENTENCE_CHARS <= len(candidate) <= _MAX_SENTENCE_CHARS):
            continue
        # A heading, a caption or a table row is not a sentence to quote.
        if not candidate.endswith((".", "!", "?")):
            continue
        if _CAPTION_START_RE.match(candidate):
            continue
        # A chunk boundary cuts mid-word: the first "sentence" of a chunk can
        # start at "…svilup|pati dal Systemic Food Design Lab è nata la Circular
        # Economy for Food (CEFF)". A sentence to quote starts where a sentence
        # starts.
        if candidate[0].islower():
            continue
        if len(candidate.split()) < _MIN_SENTENCE_WORDS:
            continue
        # 0.5 is "the term merely occurs": a definition needs a pattern too.
        score = definition_score(candidate, term)
        if score <= 0.5:
            continue
        folded = _fold(candidate)
        # A definition opens with what it defines. Without this bonus a sentence
        # that reaches the term only after a long narrative ("More than 16 years
        # of research […] behind SEeD, an acronym for Systemic Event Design")
        # wins: verbatim, accurate, and a poor opening line.
        if re.match(rf"(?:per\s+|il\s+|lo\s+|la\s+|the\s+)?{tp}\b", folded):
            score += 1.5
        if len(candidate) > 250:
            score -= 1.0
        if score > best_score:
            best, best_score = candidate, score
    return _trim_to_definiendum(best, term)


def _trim_to_definiendum(sentence: str, term: str) -> str:
    """Drop a long narrative preamble sitting before the term.

    Some definitions close a sentence that opens on unrelated narrative. The
    ellipsis is the standard way to quote that faithfully: everything kept is
    still literally the source's, which is what the quote gate checks.

    Args:
        sentence: The chosen defining sentence, possibly empty.
        term: The defined term.

    Returns:
        ``"[...] <tail>"`` when the term starts at least 80 characters in and
        the tail from it is long enough to quote; otherwise ``sentence``.
    """
    if not sentence:
        return sentence
    match = re.search(_term_pattern(term), _fold(sentence))
    if match is None or match.start() < 80:
        return sentence
    tail = sentence[match.start():].lstrip()
    if len(tail) < _MIN_SENTENCE_CHARS:
        return sentence
    return f"[...] {tail}"


def definition_score(text: str, term: str) -> float:
    """Score how much ``text`` looks like a definition *of* ``term``.

    Every pattern is anchored to the term. That anchoring is the whole point:
    "è un" and "is a" match half of any prose, but "SEeD (Systemic Event
    Design)" and "per economia circolare si intende" only match where a
    definition is actually being given.

    Args:
        text: Candidate chunk content.
        term: The term extracted by :func:`definitional_term`.

    Returns:
        ``0.0`` when the term does not occur, otherwise a positive score whose
        magnitude reflects how definitional the surrounding wording is.
    """
    if not _normalize(term) or _normalize(term) not in _normalize(text):
        return 0.0

    # Folded but *not* stripped of punctuation: parentheses and colons are half
    # the definitional signal, and stripping them would hide "SEeD (Systemic
    # Event Design)" from the very pattern written to find it.
    folded = _fold(text)
    tp = _term_pattern(term)
    score = 0.5

    # Acronym expansion in either direction: "SEeD (Systemic Event Design)" or
    # "Systemic Event Design (SEeD)". The strongest signal there is.
    if re.search(rf"{tp}\s*\(\s*[a-z][a-z'’ .-]{{1,60}}\)", folded) or re.search(
        rf"[a-z]{{3,}}\s+[a-z]{{3,}}\s*\(\s*{tp}\s*\)", folded
    ):
        score += 3.0

    # An acronym often sits between the term and its copula — "l'economia
    # circolare per il cibo (CEFF) è un modello…" — and the copula patterns
    # below have to see across it.
    gap = r"(?:\s*\([^)]{1,40}\))?"
    definitional_after = (
        # "e'" is how a fair share of the corpus writes "è" after OCR.
        rf"{tp}{gap}\s+(?:e['’]?|sono)\s+(?:un|uno|una|il|lo|la|i|gli|le|quel)\b"
        # "Metabolizzazione, cioè la valorizzazione in ottica di upcycling…":
        # the corpus defines half its vocabulary in apposition like this.
        rf"|{tp},?\s+(?:cioe|ossia|ovvero|vale\s+a\s+dire|that\s+is|i\.e\.)\b"
        rf"|{tp}\s+si\s+definisce\b"
        rf"|{tp}\s+consiste\s+(?:in|nel|nella|nelle|nei)\b"
        rf"|{tp}\s+(?:indica|rappresenta|designa|denota)\b"
        rf"|{tp}\s+si\s+riferisce\s+a\b"
        rf"|{tp}\s+sta\s+per\b"
        rf"|{tp},?\s+(?:definit[oa]|inteso|intesa)\s+come\b"
        rf"|{tp}\s+(?:is|are)\s+(?:a|an|the)\b"
        rf"|{tp}\s+(?:is|are)\s+defined\s+as\b"
        rf"|{tp}\s+(?:refers|refer)\s+to\b"
        rf"|{tp}\s+(?:means|stands\s+for|denotes)\b"
        # "SEeD, an acronym for Systemic Event Design": how the corpus actually
        # writes the expansion when it is not in parentheses.
        rf"|{tp},?\s+(?:an?\s+)?acronym\s+(?:for|of)\b"
        rf"|{tp},?\s+(?:l['’])?acronimo\s+(?:di|per)\b"
        rf"|{tp},?\s+(?:short|abbreviation)\s+for\b"
    )
    if re.search(definitional_after, folded):
        score += 2.0

    # Appositive definition — "SEeD, a systemic sustainability project developed
    # for…" — weaker than an explicit copula, strong enough to outrank a bare
    # mention.
    if re.search(rf"{tp},\s+(?:a|an|the|un|uno|una|il|lo|la)\s+[a-z]{{3,}}", folded):
        score += 1.5

    definitional_before = (
        rf"(?:per|con)\s+{tp}\s+si\s+intende\b"
        rf"|si\s+intende\s+(?:per|con)\s+{tp}\b"
        rf"|definizione\s+di\s+{tp}\b"
        rf"|(?:by|term)\s+{tp}\s+(?:we\s+)?mean\b"
    )
    if re.search(definitional_before, folded):
        score += 2.0

    # A colon straight after the term, as in a glossary entry.
    if re.search(rf"(?:^|[\s.;»\"']){tp}\s*:\s+\S", folded):
        score += 1.0

    return score
