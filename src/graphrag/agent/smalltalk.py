"""Recognise the questions that are about the assistant, not about the corpus.

A demo session usually opens with "ciao" or "prova, sistema operativo?", not
with a subject question. Sent to retrieval, the first carries no search terms
and gets an empty context, so the model answers that it does not know; the
second is refused with a sentence about documents that never says what the
assistant is.

These are not domain questions and no gate can make them into one — there is
nothing to retrieve. They are answered before retrieval, with a fixed
sentence saying what this assistant covers plus the questions worth asking.

Detection is deterministic on purpose. An LLM classifier would cost a call on
every turn and, worse, would be free to read "come funziona la simbiosi
industriale?" as a question about itself. The patterns below match the whole
normalised question or a specific phrase that has no reading inside the domain,
under a word ceiling: a real question that happens to contain "chi sei" in a
subordinate clause is longer than anything here.
"""

from __future__ import annotations

import re
import unicodedata

# Above this many words a question carries a subject of its own, whatever
# phrase it contains. The patterns below are the first lock; this ceiling is
# the second.
_MAX_META_WORDS = 10

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Whole-question greetings and pings. Matched against the entire normalised
# string, never as a substring: "ciao, cos'e l'economia circolare del cibo?"
# is a domain question with a greeting glued to its front, and must not be
# answered with an introduction.
_WHOLE_IT = {
    "ciao", "ciao ciao", "salve", "buongiorno", "buon giorno", "buonasera",
    "buona sera", "buonpomeriggio", "buon pomeriggio", "ehi", "ehila", "hey",
    "ciao come stai", "come stai", "come va", "tutto bene", "prova",
    "prova prova", "sei attivo", "ci sei", "funzioni",
}
_WHOLE_EN = {
    "hi", "hi there", "hello", "hello there", "hey", "hey there", "yo",
    "good morning", "good afternoon", "good evening", "how are you",
    "are you there", "test", "testing", "ping", "are you working",
}

# Phrases with no reading inside the domain. Each is anchored on the verb or
# the possessive that makes it about the assistant: "che modello" alone matches
# "che modello di economia circolare descrive il documento?", "che modello sei"
# cannot.
_PHRASES_IT = (
    r"chi sei",
    r"chi e' che (mi )?risponde",
    r"(che )?cosa sei",
    r"che cos'?e'? (questo|sto) (assistente|sistema|servizio|chatbot|bot|strumento)",
    r"come ti chiami",
    r"(qual e'|qualè) il tuo nome",
    r"(che )?cosa (sai|puoi) fare$",
    r"cosa (mi )?sai dire di te",
    r"a (che )?cosa servi",
    r"come funzioni",
    r"come funziona (questo|il|la) (assistente|sistema|servizio|chat|chatbot|bot|demo|strumento)",
    r"di (che )?cosa (parli|ti occupi)",
    r"(che|quali) domande posso (farti|fare|porti)",
    r"(che )?cosa posso (chiederti|domandarti)",
    r"su (che )?cosa (posso chiederti|rispondi)",
    r"che tipo di domande",
    # No reading inside the domain: no document here is about an OS, so the
    # phrase is a probe of the machine whoever typed it thinks they are talking
    # to. Unanchored for the same reason.
    r"sistema operativo",
    r"(che|quale) modello (sei|usi|stai usando)",
    r"che (llm|intelligenza artificiale|ia) (sei|usi)",
    r"sei (chatgpt|un umano|una persona|un bot|un'?intelligenza artificiale)",
    r"^aiuto$",
)
# The `$` on the capability phrases is load-bearing: "what can you do with
# grape pomace?" is a domain question that opens with one of them, and must not
# be answered with an introduction.
_PHRASES_EN = (
    r"who are you",
    r"what are you",
    r"what is this (assistant|system|service|chatbot|bot|tool|demo)",
    r"what'?s your name",
    r"what can you do$",
    r"what do you do$",
    r"what are you for",
    r"how do you work",
    r"how does this (assistant|system|service|chat|chatbot|bot|demo|tool) work",
    r"what (questions )?(can|should) i ask",
    r"what can i ask you",
    r"what kind of questions",
    r"operating system",
    r"(what|which) model (are you|do you use|are you using)",
    r"(what|which) (llm|ai) (are you|do you use)",
    r"are you (chatgpt|human|a human|a bot|an ai)",
    r"^help$",
)

_RE_IT = tuple(re.compile(p) for p in _PHRASES_IT)
_RE_EN = tuple(re.compile(p) for p in _PHRASES_EN)


def _normalise(question: str) -> str:
    """Lowercase, unaccented, punctuation-free, single-spaced.

    Accents go because people write "qual e'", "qual è" and "qual e" in the
    same session, and the patterns would otherwise need three spellings each.

    Args:
        question: The question as typed.

    Returns:
        The normalised text; apostrophes are kept.
    """
    text = unicodedata.normalize("NFD", question.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # The apostrophe stays: it is what separates "cos'e" from "cose".
    text = re.sub(r"[^\w'\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


# Words that are not a subject: greetings, courtesy, copulas, the interrogative
# scaffolding both languages build a question out of. Not the retriever's
# stopword list — that one exists to score search terms and keeps "ciao" and
# "grazie", which is exactly what has to fall away here. Kept here rather than
# imported from `agent.core` because core imports this module.
_FILLER = {
    # greetings and courtesy
    "ciao", "salve", "buongiorno", "buonasera", "buondi", "giorno", "sera",
    "pomeriggio", "grazie", "mille", "prego", "scusa", "scusi", "hello",
    "hallo", "hey", "hiya", "thanks", "thank", "please", "sorry", "morning",
    "afternoon", "evening", "good",
    # copulas, pronouns, adverbs and the rest of the scaffolding
    "sono", "sei", "siete", "essere", "stai", "state", "come", "cosa", "che",
    "chi", "dove", "quando", "quanto", "quale", "quali", "perche", "questo",
    "questa", "questi", "queste", "tuo", "tua", "mio", "mia", "tutto", "tutti",
    "bene", "allora", "poi", "anche", "ancora", "adesso", "dimmi", "dirmi",
    "sapere", "senti", "senso", "niente", "nulla", "davvero", "ovvero",
    "insomma", "capito", "capisco", "sicuro", "certo", "vero", "amico",
    "amica", "gentile", "cortesia", "favore", "aiuto", "aiutare", "aiutarmi",
    "risposta", "rispondi", "domanda", "domande", "parlare", "parliamo",
    "conversazione", "chiedere", "chiederti",
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
    "this", "that", "these", "those", "your", "yours", "mine", "there",
    "here", "well", "then", "also", "still", "again", "really", "sure",
    "right", "friend", "help", "answer", "question", "questions", "ask",
    "tell", "talk", "know", "doing", "going", "just", "okay", "yeah",
    "please",
}
# Below this, a token carries nothing to search for once the filler is gone:
# "ore", "sai", "dai", "boh". Same threshold the retriever's own content-term
# filter uses, for the same reason.
_MIN_SUBJECT_LEN = 4


def has_searchable_subject(question: str) -> bool:
    """Whether anything is left to look up once the filler is removed.

    The retriever's own term builder cannot answer this: it falls back to the
    whole question when it finds no term, so "ciao" reaches the graph as the
    search term "ciao". This is the test that a question carries no subject
    at all.

    Args:
        question: The question as typed.

    Returns:
        False when every token is a greeting, a courtesy formula, a copula or
        interrogative scaffolding — "ciao come stai amico mio", "grazie
        mille!", "che ore sono?". True for anything naming something, which is
        every question that must keep reaching retrieval.
    """
    for token in _WORD_RE.findall(_normalise(question)):
        if token in _FILLER:
            continue
        # An acronym or a name is a subject whatever its length: SEeD, PNRR,
        # 3C. `_normalise` has already lowercased, so the test is on the
        # original.
        if len(token) >= _MIN_SUBJECT_LEN:
            return True
    for token in _WORD_RE.findall(question):
        if len(token) < _MIN_SUBJECT_LEN and token.isupper() and len(token) > 1:
            return True
    return False


# Which language a wordless opening is in. `LLMManager._detect_query_language`
# is a whole-text detector and reads "grazie mille" and "sei sveglio?" as
# English — on two words there is not enough text for it. These markers are
# only ever consulted for strings that carry no subject, so they never have to
# compete with the vocabulary of a real question.
_LANG_MARKERS_IT = {
    "ciao", "salve", "buongiorno", "buonasera", "buondi", "grazie", "prego",
    "scusa", "scusi", "sono", "sei", "stai", "come", "cosa", "che", "chi",
    "dove", "quando", "quale", "quali", "perche", "questo", "mio", "tuo",
    "tutto", "bene", "senti", "dimmi", "niente", "amico", "ore", "va", "mi",
    "ti", "ci", "un", "una", "il", "la", "lo", "gli", "le", "di", "e",
}
_LANG_MARKERS_EN = {
    "hello", "hey", "hi", "thanks", "thank", "please", "sorry", "good",
    "morning", "afternoon", "evening", "what", "which", "who", "how", "when",
    "where", "why", "this", "your", "you", "are", "is", "am", "do", "does",
    "the", "a", "an", "and", "there", "here", "well", "okay", "yeah",
}


def guess_language(question: str, default: str = "it") -> str:
    """Italian or English, decided on function words alone.

    Args:
        question: The question as typed.
        default: What to answer when nothing in it tells the two apart — a
            bare "ok", an emoji. Italian, because the surface that asks for
            this guess is opened in Italian and its expert users write in it.

    Returns:
        ``"it"`` or ``"en"``.
    """
    tokens = set(_WORD_RE.findall(_normalise(question)))
    italian = len(tokens & _LANG_MARKERS_IT)
    english = len(tokens & _LANG_MARKERS_EN)
    if italian > english:
        return "it"
    if english > italian:
        return "en"
    return default


def detect_meta_question(question: str) -> str | None:
    """Say whether this is a question about the assistant, and in which language.

    Args:
        question: The question as typed.

    Returns:
        ``"it"`` or ``"en"`` when the question is a greeting, a ping or a
        question about the assistant itself; ``None`` when it is anything else,
        which is every question that must keep reaching retrieval.
    """
    text = _normalise(question)
    if not text:
        return None
    # Language detection on two words is unreliable, and these two sets share
    # "hey": the tie goes to Italian, the language the demo is opened in.
    if text in _WHOLE_IT:
        return "it"
    if text in _WHOLE_EN:
        return "en"
    if len(_WORD_RE.findall(text)) > _MAX_META_WORDS:
        return None
    for pattern in _RE_IT:
        if pattern.search(text):
            return "it"
    for pattern in _RE_EN:
        if pattern.search(text):
            return "en"
    return None
