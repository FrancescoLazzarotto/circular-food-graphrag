"""Intra-session conversation memory for follow-up questions.

A reader often asks a follow-up — "mi indichi le strategie nel settore vino" —
whose topic comes from the previous answer, not from the question. Retrieval
receives that question in isolation and searches for the wrong thing.

This module holds the state needed to make such a question self-contained
again: the entities the conversation is actually about. Three hard boundaries:

* **Never a source of facts.** Memory carries *entities* and the plain text of
  what was already said, never citable claims. The groundedness of a turn is
  always computed against the evidence retrieved in that turn; a model that
  could cite something "because it was said earlier" would be self-confirming.
  Reference tags are stripped out of the transcript, so it holds no id for the
  model to reuse.
* **Retrieval only, with one exception.** The rewritten question steers
  retrieval; generation still answers the question the user literally typed.
  The exception is the transcript: a reader who writes "hai scritto X, quali?"
  is quoting the assistant, and without a record of its own prose the model
  reads that as an unsupported premise and contradicts its own earlier answer.
  The transcript is carried so the model can recognise its own words, and for
  nothing else.
* **Off unless asked.** Without a memory object the agent behaves exactly as
  if this module did not exist, so gold runs and experiment baselines stay
  comparable.

Every question after the first turn is treated as a possible follow-up
(:meth:`ConversationMemory.has_context`); the rewrite prompt then decides
whether it needs the conversation, and repeats it unchanged when it stands on
its own.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "ActiveEntity",
    "Exchange",
    "ConversationMemory",
]


# Entity names shorter than this, or made only of digits, are noise as retrieval
# seeds.
_MIN_ENTITY_CHARS = 3
_MAX_ENTITY_CHARS = 60
# A document is where an answer came from, not what it was about. The graph
# holds document nodes under their file name, and a name like "SEeD for
# Change.pdf" in the seed list spends one of only four slots steering the
# rewrite towards the file rather than the subject.
_DOCUMENT_SUFFIXES = (
    ".pdf", ".doc", ".docx", ".odt", ".rtf",
    ".xls", ".xlsx", ".ods", ".csv",
    ".ppt", ".pptx", ".odp",
    ".txt", ".md", ".json", ".xml", ".html", ".htm",
)

# Reference tags as the answer prompt writes them: "[S1]", "[T12]", and the
# document form "[REPORT MATTM, p. 70]" with its multi-source "[A, p. 1; B, p. 2]"
# variant. They are removed from the transcript so a claim the model made earlier
# cannot come back carrying an id and be recited as if a document supported it.
# "[...]" is the omission marker the definitional prompt asks for and stays.
_REFERENCE_TAG_RE = re.compile(r"\[(?!\.\.\.\])[^\[\]\n]{1,200}\]")

# The generated source list closing an answer: pure citation machinery, the part
# of the text with the highest tag density and the least conversational value.
_SOURCE_LIST_RE = re.compile(
    r"\n\s*\*{0,2}(?:Fonti|Sources)\*{0,2}\s*:\s*\n.*\Z",
    re.IGNORECASE | re.DOTALL,
)

# Characters, not turns: answer length varies by a factor of two, so a turn
# budget would too. 16k characters is roughly 4k tokens, a handful of stripped
# answers — comfortable inside a 32k window that also has to hold the retrieved
# context, and bounded because the context block grows with the corpus.
_DEFAULT_TRANSCRIPT_CHARS = 16_000


def _transcript_budget() -> int:
    """Character budget for the transcript, overridable per deployment."""
    raw = os.getenv("GRAPHRAG_TRANSCRIPT_MAX_CHARS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TRANSCRIPT_CHARS
    return value if value > 0 else _DEFAULT_TRANSCRIPT_CHARS


def _strip_references(answer: str) -> str:
    """Return the prose of an answer, without its source list and reference tags."""
    text = _SOURCE_LIST_RE.sub("", str(answer or ""))
    text = _REFERENCE_TAG_RE.sub("", text)
    # Removing an inline tag leaves " ." and doubled spaces behind.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    return text.strip()


_TOKEN_RE = re.compile(r"[\wÀ-ÿ'’-]+")

# Word units for entity matching. Narrower than `_TOKEN_RE` on purpose:
# apostrophes and hyphens separate here, so "l'economia" contains the word
# "economia" and "sotto-prodotti" contains "prodotti".
_WORD_RE = re.compile(r"[\wÀ-ÿ]+")


# Sentence boundaries, kept crude on purpose: the point is to attach a citation
# to the claim it follows, and a split that occasionally keeps two sentences
# together attaches one citation too many — a retrieval preference, not a wrong
# answer. Boundaries inside a bracket are skipped: a citation label ends in
# "p. 70", whose full stop is not the end of anything.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

# What the model is told to leave beside a claim once citations are rendered as
# labels: "REPORT MATTM, p. 70". The bare-id form "[S3]" is excluded because it
# names an evidence block of a turn that is over, not a document.
_SOURCE_LABEL_RE = re.compile(r"\[([^\[\]\n]{3,120})\]")
_BARE_REF_RE = re.compile(r"^[STst]\s?\d{1,3}(?:\s*[,;]\s*[STst]\s?\d{1,3})*$")

# A run this long, in words, is a quotation rather than a shared turn of
# phrase.
_QUOTE_MIN_WORDS = 5


def _is_source_label(label: str) -> bool:
    """Whether a bracketed group names a document rather than an evidence id."""
    text = label.strip()
    if not text or _BARE_REF_RE.match(text):
        return False
    # `verify_citations` writes these in place of a reference it could not
    # confirm. Following one back to a document is exactly what must not happen.
    return "non verificato" not in text.lower() and "unverified" not in text.lower()


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries that fall outside a bracketed citation."""
    if not text:
        return []
    protected = [match.span() for match in _SOURCE_LABEL_RE.finditer(text)]
    cuts = [0]
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        position = match.start()
        if any(start < position < end for start, end in protected):
            continue
        cuts.append(match.end())
    cuts.append(len(text))
    return [text[cuts[i] : cuts[i + 1]].strip() for i in range(len(cuts) - 1)]


def _sentence_sources(answer: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Pair each sentence of a raw answer with the sources cited inside it.

    The answer stored for the transcript has its tags stripped, which is right
    for the prompt and useless here: the tags are the only record of which
    document a sentence came from. Parsed once, when the turn is observed.

    Args:
        answer: The answer as generated, with its reference tags.

    Returns:
        ``(stripped sentence, source labels)`` for each sentence citing at
        least one document.
    """
    rows: list[tuple[str, tuple[str, ...]]] = []
    for sentence in _split_sentences(str(answer or "")):
        labels = tuple(
            dict.fromkeys(
                label.strip()
                for label in _SOURCE_LABEL_RE.findall(sentence)
                if _is_source_label(label)
            )
        )
        if not labels:
            continue
        rows.append((_strip_references(sentence), labels))
    return tuple(rows)


def _longest_common_run(outer: Sequence[str], inner: Sequence[str]) -> int:
    """Length of the longest run of words the two share, in order."""
    if not outer or not inner:
        return 0
    previous = [0] * (len(inner) + 1)
    best = 0
    for i in range(1, len(outer) + 1):
        current = [0] * (len(inner) + 1)
        for j in range(1, len(inner) + 1):
            if outer[i - 1] == inner[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


def _words(text: str) -> tuple[str, ...]:
    """Lowercased word units of `text`."""
    return tuple(match.lower() for match in _WORD_RE.findall(str(text or "")))


def _contains_span(outer: Sequence[str], inner: Sequence[str]) -> bool:
    """True when `inner` occurs inside `outer` as a run of whole words.

    Substring containment is not usable on entity names: with a 3-character
    floor, "Riso" sits inside "risorse", "Eni" inside "sostenibile" and "tema"
    inside "sistema".
    """
    if not inner or len(inner) > len(outer):
        return False
    span = tuple(inner)
    width = len(span)
    return any(
        tuple(outer[start : start + width]) == span
        for start in range(len(outer) - width + 1)
    )


@dataclass
class ActiveEntity:
    """A KG entity the conversation has touched, with its recency.

    Attributes:
        name: Entity name.
        turn: Last turn in which it was retrieved.
        mentions: How many turns retrieved it.
    """

    name: str
    turn: int
    mentions: int = 1


@dataclass
class Exchange:
    """One completed turn as plain conversational text.

    The answer is stored stripped of reference tags and of its generated source
    list: what stays is what the assistant said, not what it cited.
    """

    question: str
    answer: str
    # (sentence, source labels cited in it) for the sentences that carry a
    # citation. Not part of the transcript: it exists so a later question that
    # quotes this answer can be retrieved against the document the quoted
    # sentence came from, instead of against the words of the question.
    citations: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass
class ConversationMemory:
    """Entities in play for the current session.

    Lives in `st.session_state` for the demo and nowhere else: no persistence
    across sessions, no shared domain memory. Entities older than `window`
    turns are dropped — without decay the seed list grows until it describes
    half the graph and stops discriminating.

    Attributes:
        window: Turns an entity stays active after it was last retrieved.
        max_seed_entities: Default cap of :meth:`seed_entities`.
        turn: Answered turns so far.
        failed_turns: Turns that raised before producing an answer.
        active_entities: Entities retrieved within the window.
        last_answer_entities: Retrieved entities the last answer named.
        last_question: The last question, whitespace-normalised.
        exchanges: The conversation as text, oldest first.
        max_transcript_chars: Character budget of the transcript.
    """

    window: int = 3
    max_seed_entities: int = 4
    turn: int = 0
    # Turns that raised before producing an answer. Counted apart from `turn`
    # so a retry after a graph failover does not spend two turns of the decay
    # window on one question; see `observe_failure`.
    failed_turns: int = 0
    active_entities: list[ActiveEntity] = field(default_factory=list)
    last_answer_entities: list[str] = field(default_factory=list)
    last_question: str = ""
    # The conversation as text, oldest first. Unlike `active_entities` this is
    # not subject to the `window` decay: a user can refer to something said six
    # turns ago, and the character budget already bounds it.
    exchanges: list[Exchange] = field(default_factory=list)
    max_transcript_chars: int = field(default_factory=_transcript_budget)

    def reset(self) -> None:
        """Forget the current topic (the 'Nuovo argomento' button)."""
        self.turn = 0
        self.failed_turns = 0
        self.active_entities = []
        self.last_answer_entities = []
        self.last_question = ""
        self.exchanges = []

    def has_context(self) -> bool:
        """Whether anything has been said yet in this session.

        Turn count, not entity count. Entities are observed only from the KG
        channel, and on a question the graph answers with nothing (answered
        entirely from text) the entity list stays empty. The rewrite step
        exits on this flag before it runs, so an entity-based test would
        disable follow-up handling for the rest of the session.
        """
        return self.turn > 0 or self.failed_turns > 0

    def observe_failure(self, question: str) -> None:
        """Record that a turn happened even though it produced no answer.

        `observe` runs only after a successful `graph.invoke`, so without this
        a turn that raised leaves no trace: if it was the first turn of a
        session, `has_context()` stays false and the next follow-up is treated
        as a fresh question.

        Deliberately not `observe` with an empty answer: that would increment
        `turn`, and the demo retries the same question after rebuilding onto
        the fallback graph, so one question would consume two turns and shorten
        the entity decay window by one. The turn counter stays the truth about
        answered turns; this only records that the conversation has started.

        Args:
            question: The question that failed.
        """
        self.failed_turns += 1
        self.last_question = " ".join(str(question or "").split())

    def seed_entities(self, limit: int | None = None) -> list[str]:
        """Entities to resolve a follow-up against, most useful first.

        Only entities the previous answer actually named: they are what the
        expert just read, and therefore what an elliptical follow-up refers to.

        Retrieved-but-unused entities are deliberately excluded rather than
        ranked below: when the answer names none of the retrieved nodes, a
        fallback ranking would seed the rewrite with an unrelated node and
        send the follow-up somewhere else entirely. An empty seed list costs
        nothing: `_rewrite_with_memory` then keeps the question as typed.

        Within the answer's entities, the most recent and most mentioned come
        first, and a name contained in another selected name is merged into
        the more specific one.

        Args:
            limit: Maximum number of entities; defaults to
                ``max_seed_entities``.

        Returns:
            The selected entity names.
        """
        cap = self.max_seed_entities if limit is None else limit
        recent = {name.lower() for name in self.last_answer_entities}
        if not recent:
            return []
        ranked = sorted(
            (item for item in self.active_entities if item.name.lower() in recent),
            key=lambda item: (item.turn, item.mentions),
            reverse=True,
        )

        selected: list[str] = []
        selected_words: list[tuple[str, ...]] = []
        for item in ranked:
            if len(selected) >= cap:
                break
            words = _words(item.name)
            if not words:
                continue
            if any(_contains_span(chosen, words) for chosen in selected_words):
                continue
            # "Regione" next to "Regione Piemonte" wastes one of the few slots
            # and makes the rewrite vaguer, not richer. Either order can occur —
            # ranking decides which of the two is seen first — so the specific
            # name wins whether it arrives before or after the broader one.
            # One new entity can subsume more than one already-selected one
            # (e.g. "Politica Agricola Comune" absorbs both "Politica Agricola"
            # and "Agricola Comune"), so every absorbed slot is dropped, not
            # just the first found.
            absorbed = [
                pos for pos, chosen in enumerate(selected_words)
                if _contains_span(words, chosen)
            ]
            if absorbed:
                keep_at = absorbed[0]
                for pos in reversed(absorbed[1:]):
                    del selected[pos]
                    del selected_words[pos]
                # Replace in place: the slot keeps the rank it earned.
                selected[keep_at] = item.name
                selected_words[keep_at] = words
                continue
            selected.append(item.name)
            selected_words.append(words)
        return selected

    def observe(
        self,
        question: str,
        answer: str,
        nodes: Sequence[dict[str, Any]] = (),
        triples: Sequence[dict[str, Any]] = (),
    ) -> None:
        """Record one completed turn.

        Args:
            question: The question as typed.
            answer: The generated answer, used only to tell which retrieved
                entities the model actually talked about.
            nodes: Retrieved KG nodes for the turn.
            triples: Retrieved triples (and subgraph) for the turn.
        """
        self.turn += 1
        self.last_question = " ".join(str(question or "").split())

        retrieved = _entity_names(nodes=nodes, triples=triples)
        index = {item.name.lower(): item for item in self.active_entities}
        for name in retrieved:
            existing = index.get(name.lower())
            if existing is None:
                item = ActiveEntity(name=name, turn=self.turn)
                self.active_entities.append(item)
                index[name.lower()] = item
            else:
                existing.turn = self.turn
                existing.mentions += 1

        # Whole-word match: this list is the top of the seed ranking, so a name
        # that only happens to sit inside a longer word steers the rewrite
        # towards something the answer never discussed.
        answer_words = _words(answer)
        self.last_answer_entities = [
            name for name in retrieved if _contains_span(answer_words, _words(name))
        ]

        cutoff = self.turn - self.window
        self.active_entities = [
            item for item in self.active_entities if item.turn > cutoff
        ]

        self._record_exchange(question=self.last_question, answer=answer)

    def _record_exchange(self, question: str, answer: str) -> None:
        """Append the turn to the transcript and trim it to the budget.

        Oldest first out: a reference to what was just said is what breaks
        without a transcript, and the far end of a long conversation is the part
        the user is least likely to be quoting. The newest exchange is always
        kept.

        Args:
            question: The question, whitespace-normalised.
            answer: The answer as generated, with its reference tags.
        """
        prose = _strip_references(answer)
        if not question and not prose:
            return
        self.exchanges.append(
            Exchange(
                question=question,
                answer=prose,
                citations=_sentence_sources(answer),
            )
        )

        budget = self.max_transcript_chars
        while len(self.exchanges) > 1 and self._transcript_size() > budget:
            self.exchanges.pop(0)

    def sources_for_quote(self, question: str) -> list[str]:
        """Documents cited by the sentences this question quotes back.

        When an expert repeats a sentence the assistant wrote and asks about
        it, the words of the question are a poor retrieval query: they describe
        the claim, they do not come from the document that supports it. The
        claim's own citation does. A question quotes a sentence when they share
        a run of at least ``_QUOTE_MIN_WORDS`` words.

        Args:
            question: The question as typed.

        Returns:
            Source labels, most recent turn first, without repeats. Empty when
            nothing is quoted, which leaves retrieval unchanged.
        """
        asked = _words(question)
        if len(asked) < _QUOTE_MIN_WORDS:
            return []

        found: list[str] = []
        for exchange in reversed(self.exchanges):
            for sentence, labels in exchange.citations:
                if _longest_common_run(asked, _words(sentence)) < _QUOTE_MIN_WORDS:
                    continue
                for label in labels:
                    if label not in found:
                        found.append(label)
        return found

    def _transcript_size(self) -> int:
        """Total characters of the stored questions and answers."""
        return sum(len(item.question) + len(item.answer) for item in self.exchanges)

    def transcript(self) -> str:
        """The conversation so far, as text for the answer prompt.

        Empty until a turn has completed, so the first question of a session
        renders the prompt as if there were no memory.

        Labels are English because the prompt around them is: the model has to
        read these as speaker turns, not as retrieved material. The answers
        carry no reference tags, so nothing here can be cited.
        """
        parts: list[str] = []
        for item in self.exchanges:
            if item.question:
                parts.append(f"User: {item.question}")
            if item.answer:
                parts.append(f"Assistant: {item.answer}")
        return "\n\n".join(parts)


def _entity_names(
    nodes: Sequence[dict[str, Any]] = (),
    triples: Sequence[dict[str, Any]] = (),
) -> list[str]:
    """Canonical entity names from one turn's retrieval, in retrieval order.

    Retrieval order is relevance order, so the first names are the ones the turn
    was really about. Names that are too short or too long, contain no letter,
    or look like a file name are skipped.

    Args:
        nodes: Retrieved nodes; the name is ``text``, else ``properties.name``.
        triples: Retrieved triples; subject and object are both taken.

    Returns:
        The distinct names, compared case-insensitively.
    """
    names: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        """Append ``value`` as a name if it passes the filters above."""
        name = " ".join(str(value or "").split())
        if not (_MIN_ENTITY_CHARS <= len(name) <= _MAX_ENTITY_CHARS):
            return
        if not any(char.isalpha() for char in name):
            return
        if name.lower().endswith(_DOCUMENT_SUFFIXES):
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(name)

    for node in _as_dicts(nodes):
        add(node.get("text") or dict(node.get("properties", {}) or {}).get("name"))

    for triple in _as_dicts(triples):
        add(triple.get("subject"))
        add(triple.get("object"))

    return names


def _as_dicts(items: Iterable[Any]) -> list[dict[str, Any]]:
    """Keep only the dict items of ``items``; ``None`` gives an empty list."""
    return [item for item in (items or []) if isinstance(item, dict)]
