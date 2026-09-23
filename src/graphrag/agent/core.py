"""The GraphRAG agent: a LangGraph pipeline from question to cited answer.

Nodes run in this order: scope (meta questions and the domain gate), refuse
(terminal), decompose, route, retrieve, grade, rewrite (back to retrieve, at
most three times) and generate.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Sequence

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from graphrag import questions
from graphrag.agent.cache import LRUCache
from graphrag.agent.compression import ContextCompressor
from graphrag.agent.evidence import (
    EvidenceItem,
    build_evidence_index,
    evidence_from_dicts,
    evidence_to_dicts,
    refs_present_in,
    render_cited_context,
    render_display_citations,
    render_grouped_reference_list,
    render_reference_list,
    verify_citations,
    verify_quotes,
)
from graphrag.agent.memory import ConversationMemory
from graphrag.agent.smalltalk import (
    detect_meta_question,
    guess_language,
    has_searchable_subject,
)
from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.llm.prompts import PromptLibrary
from graphrag.llm.refusal import looks_like_refusal
from graphrag.types import RAGState, triple_key

logger = logging.getLogger("graphrag")

# Domain-gate ellipsis floor: at or below this many words a question is a
# continuation of the previous turn, not a topic of its own, and the scope gate
# has nothing to judge. See `KGRAGAgent._scope_gate`.
_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Three, not four: "Spiegami la relatività generale" is exactly four words and
# must still be refused, while continuations ("fammi un esempio", "non ho
# capito", "in che senso") sit at three or below.
_MIN_GATED_TOKENS = 3

# Bilingual function words. The corpus is mixed Italian/English and questions
# can be English, so an Italian-only list would leave `the`, `and`, `for`,
# `are` scoring as salient terms and let `_grade` accept any triple at all.
_STOPWORDS_IT = {
    "al", "alla", "alle", "agli", "che", "come", "con", "cosa", "cui", "da",
    "dal", "dalla", "degli", "dei", "del", "della", "delle", "di", "dove", "e",
    "ed", "gli", "i", "il", "in", "la", "le", "lo", "loro", "ma", "nel",
    "nella", "nelle", "non", "o", "parlami", "per", "più", "quale", "quali",
    "quando", "quanto", "si", "sono", "su", "sue", "sui", "suo", "sul", "sulla",
    "tra", "un", "una", "uno",
}
_STOPWORDS_EN = {
    "about", "all", "and", "any", "are", "as", "at", "be", "been", "between",
    "both", "but", "by", "can", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "in", "into", "is", "it", "its", "list", "many",
    "may", "much", "not", "of", "on", "or", "our", "over", "that", "the",
    "their", "them", "there", "these", "they", "this", "those", "to", "under",
    "was", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "within", "would",
}
_STOPWORDS = _STOPWORDS_IT | _STOPWORDS_EN

# Lowercase content words shorter than this carry no discriminative power once
# stopwords are removed ("use", "aim", "key") and, matched as substrings, hit
# almost every triple.
_MIN_CONTENT_TERM_LEN = 4

# A name the model has never seen is the one thing the domain gate cannot judge:
# asked "Che cos'è SEeD?" or "Cos'è il MATTM?", the model refuses because it does
# not recognise the acronym, not because of the topic, while the graph holds
# nodes named after them.
#
# So the graph answers the half the model cannot — which names this collection
# contains — and the model keeps the verdict. That split matters: a node named
# "Torino" must not turn "consigliami un ristorante a Torino" into an in-domain
# question, and it does not, because the model still reads the question.
#
# Any token carrying an uppercase letter is a candidate; the bilingual stopword
# list drops the sentence-initial "Che"/"What" that every question starts with.
_PROPER_NOUN_RE = re.compile(r"\b\w*[^\W\d_a-zà-öø-ÿ]\w*\b", re.UNICODE)
# Shorter than this an "acronym" is noise ("UE" aside, two letters match half
# the graph once the full-text index tokenises them).
_MIN_PROPER_NOUN_LEN = 3


def _plausible_rewrite(raw: str, question: str) -> str:
    """Take the rewritten query out of a model's reply, or give up on it.

    Both rewrite paths ask for one line and get whatever the served model feels
    like producing: sometimes the query plus a parenthetical guess, sometimes
    paragraphs of markdown offering numbered options and an explanation. Fed
    to the retriever whole, such a reply buries the question under the
    vocabulary of whichever domain the model guessed.

    Length alone does not separate the two: the first line of such an essay
    can be a plausible length. What separates them is shape. A rewrite is one
    line, because that is what both prompts ask for; a reply that runs to
    several is a model explaining the rewrite instead of producing one, and the
    original question retrieves better than an explanation of how it might be
    rewritten.

    Args:
        raw: The model's reply, as text.
        question: The question that was sent for rewriting.

    Returns:
        The rewritten question, or `question` when the reply is not usable.
    """
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # One line is the contract; a label on its own line ahead of the rewrite is
    # the only variation worth tolerating.
    if len(lines) > 2:
        logger.warning(
            "Discarding an implausible rewrite (%d lines); keeping the question.",
            len(lines),
        )
        return question

    candidate = ""
    for line in lines:
        stripped = line.strip("\"'")
        for label in ("Rewritten question:", "Rewritten:"):
            stripped = stripped.removeprefix(label)
        stripped = stripped.lstrip("#>*-").strip().strip("\"'")
        if stripped:
            candidate = stripped
            break

    # A trailing colon means the line introduced something rather than asking it.
    if (
        not candidate
        or candidate.endswith(":")
        or len(candidate) > max(400, len(question) * 4)
    ):
        logger.warning(
            "Discarding an implausible rewrite (%d chars); keeping the question.",
            len(candidate),
        )
        return question
    return candidate


# Interrogatives open a question and are capitalised there, but they are not in
# `_STOPWORDS` because `_grade` needs that list for a different job. Left in,
# "Chi è Barilla?" would offer the gate "Chi-squared tests" and "Tappo a chi?"
# alongside the name that matters. `cos` and `qual` are what the elision regex
# leaves of "cos'è" and "qual è".
_GATE_EXTRA_STOPWORDS = {
    "chi", "cos", "qual", "quale", "quali", "come", "dove", "quando", "quanto",
    "quanta", "quanti", "quante", "perche", "perché", "parlami", "spiegami",
    "dimmi", "elenca", "descrivi", "why", "who", "whose", "whom",
}
# A question mentioning more names than this is prose, not a lookup; and the
# gate prompt must not grow into a context of its own.
_MAX_GATE_ENTITY_TERMS = 6
_MAX_GATE_ENTITY_NAMES = 8
# Passages shown to the evidence gate. Three is enough to show what the
# collection is about without paying for the whole context twice.
_MAX_GATE_PASSAGES = 3


def _gate_question(state: RAGState) -> str:
    """The form of the question the evidence gate should judge.

    A continuation is judged on its rewritten form, not on the words typed.
    The gate exempts "a question carrying no search terms of its own", but that
    test rarely fires: `_build_search_terms` is a retrieval extractor and keeps
    common words, so it returns ['capito', 'niente'] for "Non ho capito niente"
    and ['allora'] for "e allora dimmi", and the bare continuation would be
    refused.

    Judging the rewrite is safe in the direction that matters. The rewrite
    prompt is told to keep the user's intent, so it does not launder an
    out-of-domain question into the domain: "e scrivimi una funzione python che
    costruisca una rete neurale" rewrites to "Scrivi una funzione Python che
    costruisca una rete neurale." and is still refused. What it does supply is
    the subject a bare continuation left implicit.

    Args:
        state: The graph state, carrying ``question``, ``follow_up`` and, when
            memory rewrote it, ``rewritten_question``.

    Returns:
        The question to judge: the rewrite on a follow-up that has one, the
        question as typed otherwise.
    """
    question = str(state.get("question", "") or "").strip()
    if not state.get("follow_up"):
        return question
    return str(state.get("rewritten_question", "") or "").strip() or question


def _proper_noun_terms(question: str) -> list[str]:
    """Capitalised tokens from ``question`` that might name a graph entity.

    Args:
        question: The question as typed.

    Returns:
        Distinct candidate names, at most :data:`_MAX_GATE_ENTITY_TERMS`, in the
        order they appear.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for token in _PROPER_NOUN_RE.findall(question):
        lowered = token.lower()
        if len(token) < _MIN_PROPER_NOUN_LEN or lowered in seen:
            continue
        if lowered in _STOPWORDS or lowered in _GATE_EXTRA_STOPWORDS:
            continue
        seen.add(lowered)
        terms.append(token)
        if len(terms) >= _MAX_GATE_ENTITY_TERMS:
            break
    return terms


def _gate_mode() -> str:
    """Which gate runs: the scope description, or the retrieved evidence.

    Read per call rather than at import, so the two can be compared side by
    side in one process. The evidence gate is the default;
    ``GRAPHRAG_GATE_MODE=scope`` selects the scope gate.

    Returns:
        ``"scope"`` or ``"evidence"``.
    """
    mode = os.getenv("GRAPHRAG_GATE_MODE", "evidence").strip().lower()
    return "scope" if mode == "scope" else "evidence"


def _content_terms(retriever, question: str) -> list[str]:
    """The terms retrieval itself would search for, or the capitalised ones.

    Content words, not just capitalised ones: with capitalised tokens only,
    "biochar", "eccedenze alimentari" and every lowercased subject would
    produce no evidence, and the gate would decide on the model's world
    knowledge alone. The retriever's own builder already pairs capitalised
    candidates with lowercase content keywords, and a gate that looked for
    different terms than retrieval would judge evidence the answer is not
    built from.

    Args:
        retriever: The agent's ``KGRetriever``, or ``None``.
        question: The question to judge.

    Returns:
        The retriever's search terms, or :func:`_proper_noun_terms` when there
        is no retriever or it yields nothing.
    """
    if retriever is not None:
        try:
            terms = retriever._build_search_terms(query_text=question, configured_entity="")
        except Exception:  # noqa: BLE001 - the hint is optional, the gate is not
            terms = []
        if terms:
            return terms
    return _proper_noun_terms(question)


def _term_matches(term: str, haystack: str) -> bool:
    """Whether ``term`` occurs in ``haystack`` on word boundaries.

    Plain ``term in haystack`` would let short terms match inside unrelated
    words (`rice` in `price`, `ceff` in `ceffpolicy`) and inflate every
    relevance and coverage count that uses it.

    Args:
        term: Lowercase search term.
        haystack: Lowercase text to search.

    Returns:
        True when the term appears as a whole word (or whole phrase).
    """
    if not term or not haystack:
        return False
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", haystack) is not None


class KGRAGAgent:
    """Answers questions from the knowledge graph through a LangGraph pipeline.

    One instance can serve several sessions (the Streamlit demo shares one),
    so per-invocation state is kept thread-local or in the graph state.
    """

    def __init__(
        self,
        config: AgentConfig,
        kg_retriever: KGRetriever | None = None,
        llm: LLMManager | None = None,
    ) -> None:
        """Build the agent and compile its graph.

        Args:
            config: Agent configuration.
            kg_retriever: Retriever; ``None`` retrieves nothing.
            llm: LLM backend; ``None`` skips every LLM step and answers
                "LLM not available.".
        """
        self.config = config
        self.kg_retriever = kg_retriever
        self.llm = llm
        self.compressor = ContextCompressor(
            config.max_content_tokens, config.token_estimator_ratio
        )
        self.cache = LRUCache(config.cache_maxsize) if config.enable_cache else None
        # Set for the length of one invoke(). A LangGraph node is handed the
        # state and nothing else, so this is how the generate node reaches the
        # caller's token sink; one agent answers one question at a time.
        self._on_token: Callable[[str | None], None] | None = None

        if self.llm is not None and self.config.llm_warmup:
            self.llm.warmup()

        # One agent serves every session, so the running totals cannot live on
        # the instance: two browser tabs would add their stages together.
        self._stage_clock = threading.local()

        self.graph = self._build_graph()

    def _timed(self, name: str, node: Callable[[RAGState], dict]):
        """Wrap a graph node so its wall time is recorded per invocation.

        `latency_ms` alone says how long a turn took and nothing about where
        the time went; per-node timings make the split a property of every
        turn, so a regression is visible in the session log.

        A node can run more than once — `grade` sends the question back to
        `rewrite` up to three times — so the times accumulate rather than
        replace.

        Args:
            name: Node name, the key in ``stage_timings_ms``.
            node: The node function.

        Returns:
            The wrapped node.
        """

        def _run(state: RAGState) -> dict:
            """Run the node and add its wall time to this thread's timings."""
            started = time.perf_counter()
            try:
                return node(state)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                timings = getattr(self._stage_clock, "timings", None)
                if timings is not None:
                    timings[name] = round(timings.get(name, 0.0) + elapsed_ms, 1)

        return _run

    def _build_graph(self):
        """Wire the nodes and edges of the agent graph and compile it."""
        builder = StateGraph(RAGState)

        builder.add_node("scope", self._timed("scope", self._scope_gate))
        builder.add_node("refuse", self._timed("refuse", self._refuse_out_of_scope))
        builder.add_node("decompose", self._timed("decompose", self._decompose))
        builder.add_node("route", self._timed("route", self._adaptive_route))
        builder.add_node("retrieve", self._timed("retrieve", self._retrieve))
        builder.add_node("grade", self._timed("grade", self._grade))
        builder.add_node("rewrite", self._timed("rewrite", self._rewrite))
        builder.add_node("generate", self._timed("generate", self._generate))

        builder.add_edge(START, "scope")

        def scope_condition(state: RAGState):
            """Refuse a rejected question, otherwise continue to decompose."""
            return "refuse" if state.get("in_domain") is False else "decompose"

        builder.add_conditional_edges("scope", scope_condition)
        builder.add_edge("refuse", END)
        builder.add_edge("decompose", "route")
        # The retrieval mode chosen in `route` is read directly by `_retrieve`
        # from the state, so the edge is unconditional.
        builder.add_edge("route", "retrieve")
        builder.add_edge("retrieve", "grade")

        def grade_condition(state: RAGState):
            """Generate when relevant or after three rewrites, else rewrite."""
            if int(state.get("rewrite_count", 0) or 0) >= 3:
                return "generate"
            if state.get("relevance") == "relevant":
                return "generate"
            return "rewrite"

        builder.add_conditional_edges("grade", grade_condition)
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", END)

        return builder.compile()

    def _scope_gate(self, state: RAGState) -> dict:
        """Decide whether the question is worth retrieving for.

        With ``answer_meta_questions``, a greeting or a question about the
        assistant is routed to the identity answer first. Then, with
        ``enable_domain_gate`` and an LLM, one of two gates runs, picked by
        `_gate_mode`. The default is the evidence gate, which judges what the
        collection returns rather than a description of the domain, and reads
        the question through `_gate_question` — so a continuation is judged on
        its rewritten form, because its own words carry no subject. The scope
        gate judges the question as typed against `domain_scope`.

        Neither exempts a follow-up as such, so a continuation that changes
        subject ("e scrivimi una funzione python") is still judged. The scope
        gate lets through questions of at most `_MIN_GATED_TOKENS` words,
        which carry no topic of their own; the evidence gate's exemptions are
        in `_evidence_gate`.

        Args:
            state: Graph state with ``question`` (and ``transcript``,
                ``follow_up`` and ``rewritten_question`` when memory is on).

        Returns:
            ``in_domain``, plus ``meta_question`` and ``meta_language`` for a
            question about the assistant.
        """
        question = state.get("question", "").strip()

        # Before the gates, and independent of them: a greeting or a question
        # about the assistant has nothing to retrieve, so neither gate can
        # judge it and the answer path has no context to build on. Costs one
        # regex match on a question that is about to cost a model call anyway.
        if self.config.answer_meta_questions and question:
            language = detect_meta_question(question)
            if language is None and self._is_opening_turn(state):
                # The pattern list cannot be complete — "buondì", "grazie
                # mille!", "mi senti?" are openings nobody enumerated — and what
                # they have in common is not their wording but that they name
                # nothing to look up. On the first turn that is decisive: a
                # question with no subject and no previous turn to continue
                # cannot be answered from any collection, and sent to
                # retrieval it gets a "non so".
                #
                # Only on the first turn, because from the second one the same
                # emptiness is what a continuation looks like ("in che senso?"),
                # and those must keep reaching the answer path.
                if not has_searchable_subject(question):
                    language = guess_language(question)
            if language is not None:
                logger.info("Meta question answered without retrieval: %s", question[:100])
                return {
                    "in_domain": False,
                    "meta_question": True,
                    "meta_language": language,
                }

        if not self.config.enable_domain_gate or self.llm is None:
            return {"in_domain": True}

        if not question:
            return {"in_domain": True}

        if _gate_mode() == "evidence":
            return self._evidence_gate(_gate_question(state))

        # A short continuation is exempt: "in che senso?", "perché?", "non ho
        # capito" carry no topic of their own, and refusing them ends the
        # conversation the demo exists to hold. The test is the word floor
        # alone; the `follow_up` flag is deliberately not read, since it would
        # also wave through a continuation that changes subject.
        if len(_WORD_RE.findall(question)) <= _MIN_GATED_TOKENS:
            return {"in_domain": True}

        known = self._known_entity_names(question)
        in_domain = self.llm.classify_in_domain(question, self.config, known)
        if not in_domain:
            logger.info(
                "Domain gate refused: %s (graph names offered: %s)",
                question[:100],
                known or "none",
            )
        return {"in_domain": in_domain}

    @staticmethod
    def _is_opening_turn(state: RAGState) -> bool:
        """Whether nothing has been said in this conversation yet.

        Two tests, because neither covers the other: `transcript` is filled by
        the demo's intra-session memory and stays empty everywhere else, and
        `follow_up` comes from memory's KG entities, which stay empty on a turn
        answered entirely from the text channel.
        """
        return not str(state.get("transcript", "") or "").strip() and not state.get(
            "follow_up"
        )

    def _evidence_gate(self, question: str) -> dict:
        """Judge the question against what the collection returns for it.

        The scope gate decides from a description of the domain written into
        the prompt, which goes stale silently as documents about something
        else are added. This one asks the collection instead — matching node
        names and the top text passages with their documents — so it widens on
        its own as documents arrive.

        Two exemptions, and only two:

        * A question carrying no search terms of its own is a continuation
          ("e allora dimmi", "in che senso?"). It has no subject to look up,
          and refusing it ends the conversation the demo exists to hold.
        * Anything the lookup itself cannot do — no retriever, index down —
          leaves the question in, because a gate that fails must not refuse.

        Starting with a conjunction is deliberately *not* an exemption: what
        matters is whether the question brings a subject the collection knows.

        Args:
            question: The question to judge (see `_gate_question`).

        Returns:
            ``{"in_domain": bool}``.
        """
        retriever = self.kg_retriever
        terms = _content_terms(retriever, question)
        if not terms:
            return {"in_domain": True}
        if retriever is None:
            return {"in_domain": True}

        names: list[str] = []
        try:
            nodes = retriever.kg_store.fulltext_search_nodes(
                terms, limit=_MAX_GATE_ENTITY_NAMES * 2
            )
        except Exception as exc:  # noqa: BLE001 - the gate must not break the demo
            logger.warning("Evidence lookup failed (%s); leaving the question in", exc)
            return {"in_domain": True}
        if nodes is None:
            # Full-text index unavailable: the caller's fallback is a full scan,
            # which is not worth paying for a hint. Leave the question in.
            return {"in_domain": True}
        seen: set[str] = set()
        for node in nodes:
            name = " ".join(str(node.get("text", "") or "").split())
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
            if len(names) >= _MAX_GATE_ENTITY_NAMES:
                break

        # Names alone are not enough: a name cannot carry the figure a specific
        # question asks for ("the annual production volume of grape pomace"),
        # and shown only names the model reads thin evidence as absence.
        # Passages carry the subject matter, and they are the same channel the
        # answer is built from, so the gate judges what the answer would use.
        passages: list[str] = []
        sources: list[str] = []
        pipeline = getattr(retriever, "text_pipeline", None)
        if pipeline is not None:
            try:
                chunks = pipeline.retrieve(question, top_k=_MAX_GATE_PASSAGES)
                passages = [
                    str(getattr(c, "text", "") or getattr(c, "content", "") or "")
                    for c in chunks
                ]
                # The document names are the collection describing itself, from
                # its own data. It is the one way to tell the model what this
                # collection is without writing it into the prompt — which is
                # exactly what goes stale the day a document about something
                # else is added.
                for chunk in chunks:
                    name = str(getattr(chunk, "source", "") or "").split("#", 1)[0].strip()
                    if name and name not in sources:
                        sources.append(name)
            except Exception as exc:  # noqa: BLE001 - the hint is optional
                logger.warning("Evidence passages unavailable (%s)", exc)

        in_domain = self.llm.classify_answerable(question, names, passages, sources)
        if not in_domain:
            logger.info(
                "Evidence gate refused: %s (collection returned: %s)",
                question[:100],
                names or "nothing",
            )
        return {"in_domain": in_domain}

    def _known_entity_names(self, question: str) -> list[str]:
        """Names the graph holds for the proper nouns in ``question``.

        One indexed lookup, never a scan: when the full-text index is
        unavailable ``fulltext_search_nodes`` returns None and the gate runs on
        exactly the wording it was validated with, which is the behaviour to
        degrade to.

        The index tokenises, so it answers "documents mentioning these terms",
        not "nodes named this". The word-boundary filter afterwards is what
        makes the answer a name: without it a question about SEeD would be told
        the collection contains "Potatoes, Curd, and Linseed Oil".

        It also inherits the index's own blind spot. Lucene's tokeniser keeps
        an underscore inside a token, so the node "REPORT MATTM_Definitivo.pdf"
        is unreachable by the term MATTM and no hint is produced for it — the
        gate then runs on the model's own judgement. The fix belongs in what
        `scripts/kg/kg_search_index.py` feeds the index, not here.

        Args:
            question: The question as typed.

        Returns:
            Matching node names, at most :data:`_MAX_GATE_ENTITY_NAMES`.
        """
        if self.kg_retriever is None:
            return []
        terms = _proper_noun_terms(question)
        if not terms:
            return []
        try:
            nodes = self.kg_retriever.kg_store.fulltext_search_nodes(
                terms, limit=_MAX_GATE_ENTITY_NAMES * 4
            )
        except Exception as exc:  # noqa: BLE001 - a gate hint is never worth a failure
            logger.warning("Domain-gate entity lookup failed (%s); continuing", exc)
            return []
        if not nodes:
            return []

        lowered_terms = [term.lower() for term in terms]
        names: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            name = str(node.get("text") or "").strip()
            key = name.lower()
            if not name or key in seen:
                continue
            if not any(_term_matches(term, key) for term in lowered_terms):
                continue
            seen.add(key)
            names.append(name)
            if len(names) >= _MAX_GATE_ENTITY_NAMES:
                break
        return names

    def _refuse_out_of_scope(self, state: RAGState) -> dict:
        """Terminal state for a rejected question.

        The one path in the graph that reaches END without generating. Nothing
        is retrieved, so no evidence index exists and no source list can be
        rendered under it.

        Args:
            state: Graph state after the scope gate.

        Returns:
            The fixed ``answer`` (identity message or refusal) with
            ``out_of_scope``, and ``meta_question`` for the identity answer.
        """
        if state.get("meta_question"):
            # Not a refusal: the question was about the assistant, and it gets
            # an answer — who it is and what is worth asking it. The language
            # comes from the pattern that matched, because `_detect_query_language`
            # reads "ciao" as English and answering a greeting in the wrong
            # language is the first thing a reader notices.
            return {
                "answer": PromptLibrary.identity_message(
                    language=str(state.get("meta_language") or "en"),
                    examples=self.config.example_questions,
                ),
                "out_of_scope": True,
                "meta_question": True,
            }

        # Detected on the form the gate judged, not on the words typed. A bare
        # continuation is too short to classify ("Non ho capito niente" scores
        # as English); the rewrite carries the conversation's own language.
        question = _gate_question(state) or state.get("question", "")
        language = LLMManager._answer_language(
            question, fallback=self.config.fallback_language
        )
        # The refusal names what the collection does cover: the reader's next
        # move is to rephrase, and a bare "out of scope" gives them nothing to
        # aim at.
        scope_hint = self.config.domain_scope.strip() or PromptLibrary.DEFAULT_DOMAIN_SCOPE
        return {
            "answer": PromptLibrary.out_of_scope_message(
                language=language, scope_hint=scope_hint
            ),
            "out_of_scope": True,
        }

    def _decompose(self, state: RAGState) -> dict:
        """Split the question into sub-questions when decomposition is on.

        The model is asked for a JSON array; other output is parsed line by
        line with bullets and numbering stripped.

        Returns:
            ``sub_questions``; just the question when the step is off or no
            LLM is available.
        """
        question = state.get("question", "").strip()
        if not question:
            return {"sub_questions": []}

        if not self.config.enable_decomposition_step:
            return {"sub_questions": [question]}

        prompt = PromptLibrary.decomposition_prompt(self.config)
        rendered = prompt.invoke({"question": question})

        if self.llm is None:
            return {"sub_questions": [question]}

        model = self.llm.load_llm()
        output = model.invoke(rendered)
        text = output.content if hasattr(output, "content") else str(output)
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return {
                    "sub_questions": [
                        str(item).strip() for item in parsed if str(item).strip()
                    ]
                }
        except json.JSONDecodeError:
            pass

        # Fallback for non-JSON output: strip bullets and "1." / "2)" numbering
        # so malformed sub-questions don't flow into retrieval.
        logger.warning(
            "Decomposition output was not a JSON array; falling back to "
            "line-based parsing (first 200 chars): %s",
            text[:200],
        )
        sub_questions = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                sub_questions.append(cleaned)
        return {"sub_questions": sub_questions or [question]}

    def _rewrite(self, state: RAGState) -> dict:
        """Rewrite the question for another retrieval round.

        A rewrite identical to the query just tried cannot change retrieval,
        since generation is deterministic, so it ends the loop.

        Returns:
            ``rewritten_question`` and the incremented ``rewrite_count``.
        """
        question = state.get("question", "").strip()
        if not question:
            return {"rewritten_question": ""}

        prompt = PromptLibrary.rewrite_prompt(self.config)
        rendered = prompt.invoke({"question": question})

        if self.llm is None:
            return {"rewritten_question": question}

        model = self.llm.load_llm()
        output = model.invoke(rendered)
        raw = str(output.content if hasattr(output, "content") else output)
        # Sanitised like the follow-up rewrite: a model that explains its
        # rewrite instead of producing one must not reach the retriever.
        rewritten = _plausible_rewrite(raw, question)
        rewrite_count = state.get("rewrite_count", 0) + 1
        # Generation is deterministic (temperature 0 / do_sample=False), so a
        # rewrite equal to the query already tried can never change retrieval:
        # cap the counter to exit the loop instead of burning identical
        # LLM + retrieval rounds.
        previous = str(
            state.get("rewritten_question") or state.get("question", "")
        ).strip()
        if rewritten and rewritten == previous:
            logger.info(
                "Rewrite produced an identical query; short-circuiting the rewrite loop."
            )
            rewrite_count = max(rewrite_count, 3)
        # The grade -> generate edge already short-circuits once rewrite_count
        # reaches 3, so a forced relevance flag here would never be read. Just
        # bump the counter.
        return {"rewritten_question": rewritten, "rewrite_count": rewrite_count}

    def _adaptive_route(self, state: RAGState) -> dict:
        """Pick the retrieval mode: TEXT, KG, HYBRID or MULTIHOP.

        Returns:
            ``chosen_retrieval_mode``; ``HYBRID`` when routing is off, no LLM
            is available, or the answer is not one of the four modes.
        """
        question = state.get("question", "").strip()
        if not self.config.enable_adaptive_routing_step:
            return {"chosen_retrieval_mode": "HYBRID"}

        if not self.llm:
            # Without an LLM router available, prefer HYBRID to avoid dropping KG evidence.
            return {"chosen_retrieval_mode": "HYBRID"}

        prompt = PromptLibrary.adaptive_router_prompt(self.config)
        rendered = prompt.invoke({"question": question})
        model = self.llm.load_llm()
        output = model.invoke(rendered)
        mode = (
            str(output.content if hasattr(output, "content") else output)
            .strip()
            .upper()
        )

        if mode not in ["TEXT", "KG", "HYBRID", "MULTIHOP"]:
            mode = "HYBRID"

        return {"chosen_retrieval_mode": mode}

    def _retrieve(self, state: RAGState) -> dict:
        """Retrieve evidence for the (rewritten) question and its sub-questions.

        Results of several retrieval queries are merged and de-duplicated per
        channel within the configured limits. With ``cite_evidence`` the
        merged evidence is numbered and the context re-rendered with
        provenance. The context is then compressed to the token budget, and
        the result is cached per query and mode.

        Returns:
            ``text_context``, ``evidence_index``, ``visible_evidence_refs``,
            ``kg_triples``, ``retrieved_text_sources`` and the
            ``retrieved_*`` lists with their counts.
        """
        query = str(state.get("rewritten_question") or state.get("question", "")).strip()
        mode = state.get("chosen_retrieval_mode", "HYBRID")
        sub_questions = state.get("sub_questions", []) or []
        retrieval_queries = self._build_retrieval_queries(
            query=query,
            sub_questions=sub_questions if isinstance(sub_questions, list) else [],
        )
        raw_preferred = state.get("quoted_sources", []) or []
        preferred_documents = [
            str(item).strip() for item in raw_preferred if str(item).strip()
        ]
        cache_query = " || ".join(retrieval_queries) if retrieval_queries else query
        # The key already contains the (possibly rewritten) query text, and
        # retrieval is deterministic for a fixed query and graph. Adding the
        # rewrite counter here would only force a re-retrieval when a rewrite
        # produces the exact same text — pure wasted Neo4j round-trips.
        #
        # The preferred documents do belong in the key: they reorder the text
        # channel, so without them a cached entry from a turn that quoted
        # nothing would be served for a turn that quotes, and the preference
        # would vanish without a trace.
        cache_mode = str(mode)
        if preferred_documents:
            cache_mode = f"{cache_mode}|docs={'~'.join(sorted(preferred_documents))}"

        if self.cache:
            hit = self.cache.get(cache_query, cache_mode)
            if hit is not None:
                return hit

        retrieved_data: dict[str, Any] = {
            "query": query,
            "context_text": "",
            "nodes": [],
            "triples": [],
            "neighbors": [],
            "subgraph": [],
            "shortest_path": [],
        }
        context = ""

        if self.kg_retriever:
            mm = str(mode).upper() if mode is not None else "HYBRID"
            if mm == "TEXT":
                text_sections: list[str] = []
                text_sources: list[dict[str, Any]] = []
                text_chunks: list[dict[str, Any]] = []
                for candidate_query in retrieval_queries:
                    # retrieve() (not retrieve_context()) so the text chunks'
                    # provenance tags travel with the context for later analysis.
                    batch = self.kg_retriever.retrieve(
                        candidate_query, prefer_documents=preferred_documents
                    )
                    value = str(batch.get("context_text", "")).strip()
                    if value:
                        text_sections.append(value)
                    text_sources.extend(batch.get("text_sources", []) or [])
                    text_chunks.extend(batch.get("text_chunks", []) or [])

                context = self._merge_context_sections(text_sections)
                retrieved_data["context_text"] = context
                retrieved_data["text_sources"] = text_sources
                retrieved_data["text_chunks"] = text_chunks

            elif mm in {"KG", "HYBRID"}:
                nodes: list[dict[str, Any]] = []
                triples: list[dict[str, Any]] = []
                neighbors: list[dict[str, Any]] = []
                subgraph: list[dict[str, Any]] = []
                shortest_path: list[dict[str, Any]] = []
                context_sections: list[str] = []
                text_sources: list[dict[str, Any]] = []
                text_chunks: list[dict[str, Any]] = []

                node_seen: set[tuple[str, str]] = set()
                neighbor_seen: set[tuple[str, str]] = set()
                triple_seen: set[tuple[str, str, str]] = set()
                subgraph_seen: set[tuple[str, str, str]] = set()
                shortest_path_seen: set[tuple[str, str, str]] = set()

                for candidate_query in retrieval_queries:
                    batch = self.kg_retriever.retrieve(
                        candidate_query, prefer_documents=preferred_documents
                    )

                    nodes = self._merge_nodes(
                        existing=nodes,
                        incoming=batch.get("nodes", []),
                        seen=node_seen,
                        limit=max(1, int(self.config.nodes_limit)),
                    )
                    triples = self._merge_triples(
                        existing=triples,
                        incoming=batch.get("triples", []),
                        seen=triple_seen,
                        limit=max(1, int(self.config.triples_limit)),
                    )
                    neighbors = self._merge_nodes(
                        existing=neighbors,
                        incoming=batch.get("neighbors", []),
                        seen=neighbor_seen,
                        limit=max(1, int(self.config.neighbors_limit)),
                    )
                    subgraph = self._merge_triples(
                        existing=subgraph,
                        incoming=batch.get("subgraph", []),
                        seen=subgraph_seen,
                        limit=max(1, int(self.config.subgraph_limit)),
                    )
                    shortest_path = self._merge_triples(
                        existing=shortest_path,
                        incoming=batch.get("shortest_path", []),
                        seen=shortest_path_seen,
                        limit=max(1, int(self.config.subgraph_limit)),
                    )

                    candidate_context = str(batch.get("context_text", "")).strip()
                    if candidate_context:
                        context_sections.append(candidate_context)
                    text_sources.extend(batch.get("text_sources", []) or [])
                    text_chunks.extend(batch.get("text_chunks", []) or [])

                if (
                    self.config.rerank_merged_results
                    and self.config.rank_triples
                    and len(retrieval_queries) > 1
                ):
                    # Merged multi-query results keep arrival order by default;
                    # re-rank globally against the original question.
                    triples = self.kg_retriever.rank_triples(triples, query)
                    subgraph = self.kg_retriever.rank_triples(subgraph, query)

                context = self._merge_context_sections(context_sections)
                if mm == "KG" and not context:
                    context = self._format_triples_for_context(triples + subgraph + shortest_path)

                retrieved_data = {
                    "query": query,
                    "context_text": context,
                    "nodes": nodes,
                    "triples": triples,
                    "neighbors": neighbors,
                    "subgraph": subgraph,
                    "shortest_path": shortest_path,
                    "text_sources": text_sources,
                    "text_chunks": text_chunks,
                }

            elif mm == "MULTIHOP":
                subgraph: list[dict[str, Any]] = []
                triple_seen: set[tuple[str, str, str]] = set()
                seed_entities: list[str] = []
                seed_seen: set[str] = set()

                for candidate_query in retrieval_queries:
                    seed = self.kg_retriever.resolve_entity_seed(candidate_query)
                    normalized = seed.strip().lower()
                    if seed and normalized not in seed_seen:
                        seed_seen.add(normalized)
                        seed_entities.append(seed)

                for seed in seed_entities[:2]:
                    batch = self.kg_retriever.multi_hop(
                        entity=seed,
                        hops=self.config.hops,
                        limit=self.config.subgraph_limit,
                    )
                    subgraph = self._merge_triples(
                        existing=subgraph,
                        incoming=batch,
                        seen=triple_seen,
                        limit=max(1, int(self.config.subgraph_limit)),
                    )

                context = self._format_triples_for_context(subgraph)
                retrieved_data = {
                    "query": query,
                    "context_text": context,
                    "nodes": [],
                    "triples": [],
                    "neighbors": [],
                    "subgraph": subgraph,
                    "shortest_path": [],
                }

            else:
                retrieved_data = self.kg_retriever.retrieve(
                    query, prefer_documents=preferred_documents
                )
                context = str(retrieved_data.get("context_text", ""))

        # Renumber the merged evidence and re-render the context so every
        # citable unit reaches the model with its document and page attached.
        # Runs after the merge, never per batch: ids must be unique across all
        # retrieval queries of the turn.
        evidence_items: list[EvidenceItem] = []
        if self.config.cite_evidence and isinstance(retrieved_data, dict):
            evidence_items = build_evidence_index(
                text_chunks=retrieved_data.get("text_chunks", []) or [],
                triples=[
                    *(retrieved_data.get("triples", []) or []),
                    *(retrieved_data.get("subgraph", []) or []),
                    *(retrieved_data.get("shortest_path", []) or []),
                ],
                max_text_items=self.config.evidence_max_text_items,
                max_triple_items=self.config.evidence_max_triple_items,
            )
            if evidence_items:
                context = render_cited_context(
                    evidence=evidence_items,
                    entity_sections=self._entity_sections(retrieved_data),
                )

        compressed_context = self.compressor.compress(context)
        # Which evidence blocks the model will actually see. Compression drops
        # the middle of the context, and the citation gate must judge tags
        # against that, not against the full index.
        visible_refs = sorted(refs_present_in(compressed_context))
        triples = (
            retrieved_data.get("triples", [])
            if isinstance(retrieved_data, dict)
            else []
        )
        nodes = (
            retrieved_data.get("nodes", []) if isinstance(retrieved_data, dict) else []
        )
        neighbors = (
            retrieved_data.get("neighbors", [])
            if isinstance(retrieved_data, dict)
            else []
        )
        subgraph = (
            retrieved_data.get("subgraph", [])
            if isinstance(retrieved_data, dict)
            else []
        )
        shortest_path = (
            retrieved_data.get("shortest_path", [])
            if isinstance(retrieved_data, dict)
            else []
        )
        text_sources = (
            retrieved_data.get("text_sources", [])
            if isinstance(retrieved_data, dict)
            else []
        )

        result = {
            "text_context": compressed_context,
            "evidence_index": evidence_to_dicts(evidence_items),
            "visible_evidence_refs": visible_refs,
            "kg_triples": triples if isinstance(triples, list) else [],
            "retrieved_text_sources": text_sources
            if isinstance(text_sources, list)
            else [],
            "retrieved_nodes": nodes if isinstance(nodes, list) else [],
            "retrieved_nodes_count": len(nodes) if isinstance(nodes, list) else 0,
            "retrieved_neighbors": neighbors if isinstance(neighbors, list) else [],
            "retrieved_neighbors_count": len(neighbors)
            if isinstance(neighbors, list)
            else 0,
            "retrieved_subgraph": subgraph if isinstance(subgraph, list) else [],
            "retrieved_subgraph_count": len(subgraph)
            if isinstance(subgraph, list)
            else 0,
            "retrieved_shortest_path": shortest_path
            if isinstance(shortest_path, list)
            else [],
            "retrieved_shortest_path_count": len(shortest_path)
            if isinstance(shortest_path, list)
            else 0,
        }

        if self.cache:
            self.cache.put(cache_query, cache_mode, result)

        return result

    def _build_retrieval_queries(
        self,
        query: str,
        sub_questions: list[object],
        max_queries: int = 4,
    ) -> list[str]:
        """The distinct queries to retrieve for: the question, then sub-questions.

        Sub-questions are added only with decomposition on, up to
        ``max_queries`` queries in total.
        """
        queries: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            """Append a whitespace-normalised query unless empty or repeated."""
            value = " ".join(str(candidate).split()).strip()
            if not value:
                return
            key = value.lower()
            if key in seen:
                return
            seen.add(key)
            queries.append(value)

        add(query)
        if self.config.enable_decomposition_step:
            for sub in sub_questions:
                add(str(sub))
                if len(queries) >= max_queries:
                    break

        return queries or ([query] if query else [])

    @staticmethod
    def _node_key(node: dict[str, Any]) -> tuple[str, str]:
        """De-duplication key of a node: its elementId, else its name."""
        node_id = str(node.get("node_id", "")).strip()
        if node_id:
            return ("id", node_id)
        return ("text", str(node.get("text", "")).strip().lower())

    @staticmethod
    def _triple_key(triple: dict[str, Any]) -> tuple[str, str, str]:
        """De-duplication key of a triple; see :func:`graphrag.types.triple_key`."""
        return triple_key(triple)

    def _merge_nodes(
        self,
        existing: list[dict[str, Any]],
        incoming: object,
        seen: set[tuple[str, str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Append unseen nodes from ``incoming`` to ``existing`` up to ``limit``.

        Args:
            existing: Nodes merged so far; extended in place.
            incoming: Nodes of one retrieval batch; ignored unless a list.
            seen: Keys already merged; updated in place.
            limit: Total cap on ``existing``.

        Returns:
            ``existing``.
        """
        if not isinstance(incoming, list):
            return existing

        # Checked before the append, and on entry, so that repeated merges
        # (one per retrieval query) never exceed the limit.
        for item in incoming:
            if len(existing) >= limit:
                break
            if not isinstance(item, dict):
                continue
            key = self._node_key(item)
            if key in seen:
                continue
            seen.add(key)
            existing.append(item)

        return existing

    def _merge_triples(
        self,
        existing: list[dict[str, Any]],
        incoming: object,
        seen: set[tuple[str, str, str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Append unseen triples from ``incoming`` to ``existing`` up to ``limit``.

        Same contract as :meth:`_merge_nodes`.
        """
        if not isinstance(incoming, list):
            return existing

        # Cap checked before the append — see `_merge_nodes`.
        for item in incoming:
            if len(existing) >= limit:
                break
            if not isinstance(item, dict):
                continue
            key = self._triple_key(item)
            if key in seen:
                continue
            seen.add(key)
            existing.append(item)

        return existing

    @staticmethod
    def _merge_context_sections(sections: list[str]) -> str:
        """Join distinct non-empty sections (whitespace- and case-insensitive)."""
        merged: list[str] = []
        seen: set[str] = set()
        for section in sections:
            value = section.strip()
            if not value:
                continue
            key = " ".join(value.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
        return "\n\n".join(merged)

    def _entity_sections(
        self, retrieved_data: dict[str, Any]
    ) -> list[tuple[str, str]]:
        """Build the non-citable context blocks (matched nodes, neighbours).

        Entity names carry no document provenance, so they must not receive a
        reference id: a claim tagged with a bare node name would look sourced
        without being traceable to a document.

        Args:
            retrieved_data: The merged retrieval payload for this turn.

        Returns:
            ``(title, body)`` pairs, skipping empty or unformattable sections.
        """
        if not self.kg_retriever:
            return []

        sections: list[tuple[str, str]] = []
        for title, key in (
            ("Entities in the graph (no source — do not cite):", "nodes"),
            ("Neighbouring entities (no source — do not cite):", "neighbors"),
        ):
            rows = retrieved_data.get(key, []) or []
            if not isinstance(rows, list) or not rows:
                continue
            try:
                body = str(self.kg_retriever.kg_store.nodes_to_text(rows))
            except (AttributeError, TypeError, ValueError):
                logger.warning(
                    "Node formatting failed; dropping the %r context section.",
                    key,
                    exc_info=True,
                )
                continue
            if body.strip():
                sections.append((title, body))
        return sections

    def _format_triples_for_context(self, triples: list[dict[str, Any]]) -> str:
        """Render triples as context lines; empty on failure or without a retriever."""
        if not triples:
            return ""
        try:
            if hasattr(self.kg_retriever, "format_triples") and self.kg_retriever:
                return str(self.kg_retriever.format_triples(triples))
            if self.kg_retriever:
                return str(self.kg_retriever.kg_store.triples_to_text(triples))
        except Exception:
            logger.warning(
                "Triple formatting failed; dropping %d triples from the context.",
                len(triples),
                exc_info=True,
            )
            return ""
        return ""

    def _grade(self, state: RAGState) -> dict:
        """Judge whether the retrieved evidence matches the question.

        Counts the evidence units (nodes, triples, subgraph and path triples,
        and the text context as one unit) that contain a salient term of the
        question as a whole word. Relevant when at least one matches and,
        unless only text was retrieved, at least two match or 30 % of the
        units do.

        Returns:
            ``{"relevance": "relevant" | "not_relevant"}``.
        """
        nodes_count = int(state.get("retrieved_nodes_count", 0) or 0)
        triples_count = len(state.get("kg_triples", []) or [])
        subgraph_count = int(state.get("retrieved_subgraph_count", 0) or 0)
        shortest_path_count = int(state.get("retrieved_shortest_path_count", 0) or 0)
        text_context = str(state.get("text_context", "") or "")
        has_text_evidence = bool(text_context.strip())

        # Stronger semantic gating: ensure retrieved KG items actually match the
        # salient terms in the query/context instead of accepting any hit.
        kg_evidence_units = (
            nodes_count + triples_count + subgraph_count + shortest_path_count
        )
        evidence_units = kg_evidence_units + (1 if has_text_evidence else 0)
        if evidence_units == 0:
            return {"relevance": "not_relevant"}

        query = state.get("rewritten_question") or state.get("question", "")
        salient = set(self._extract_salient_terms_from_text(query))
        if not salient:
            salient = set(self._extract_salient_terms(query=query, context=""))

        matched = 0

        for triple in state.get("kg_triples", []) or []:
            hay = f"{triple.get('subject', '')} {triple.get('predicate', '')} {triple.get('object', '')}".lower()
            if any(_term_matches(term, hay) for term in salient):
                matched += 1

        for node in state.get("retrieved_nodes", []) or state.get("nodes", []) or []:
            text = str(node.get("text", "")).lower()
            if any(_term_matches(term, text) for term in salient):
                matched += 1

        for item in (
            state.get("retrieved_subgraph", []) or state.get("subgraph", []) or []
        ):
            hay = f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}".lower()
            if any(_term_matches(term, hay) for term in salient):
                matched += 1

        for item in (
            state.get("retrieved_shortest_path", [])
            or state.get("shortest_path", [])
            or []
        ):
            hay = f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}".lower()
            if any(_term_matches(term, hay) for term in salient):
                matched += 1

        if has_text_evidence:
            context_lower = text_context.lower()
            if any(_term_matches(term, context_lower) for term in salient):
                matched += 1

        # Determine relevance: require at least one semantic match, and either
        # multiple matches or a reasonable match ratio to accept as relevant.
        match_ratio = matched / max(1, evidence_units)
        if kg_evidence_units == 0 and has_text_evidence:
            is_relevant = matched >= 1
        else:
            is_relevant = matched >= 1 and (matched >= 2 or match_ratio >= 0.30)

        logger.debug(
            "Grading retrieval: evidence_units=%d matched=%d match_ratio=%.2f salient=%s",
            evidence_units,
            matched,
            match_ratio,
            list(salient)[:8],
        )

        return {"relevance": "relevant" if is_relevant else "not_relevant"}

    def _prepend_source_definition(
        self,
        answer: str,
        question: str,
        evidence: Sequence[EvidenceItem],
        language: str,
    ) -> tuple[str, str | None]:
        """Open a definitional answer with the source's literal definition.

        Readers want the definition first and the explanation after. The model
        explains well but paraphrases the definition, so the quotation is built
        here, from the retrieved passage with the highest definitional score:
        verbatim by construction, tagged with the reference it came from, and
        in the source's own language.

        Args:
            answer: The answer after the citation and quote gates.
            question: The user question.
            evidence: The index for this turn.
            language: ``"it"`` or ``"en"``, for the lead-in wording.

        Returns:
            ``(answer, ref_id)``. ``ref_id`` is the reference the quotation came
            from, so the caller can add it to the source list, or ``None`` when
            nothing was prepended.
        """
        if not self.config.prefer_verbatim_definitions:
            return answer, None
        term = questions.definitional_term(question)
        if not term:
            return answer, None

        best_item: EvidenceItem | None = None
        best_sentence = ""
        best_score = 0.0
        for item in evidence:
            if item.kind != "text":
                continue
            sentence = questions.definition_sentence(item.text, term)
            if not sentence:
                continue
            score = questions.definition_score(sentence, term)
            if score > best_score:
                best_item, best_sentence, best_score = item, sentence, score

        if best_item is None:
            logger.info("No verbatim definition of %r in the retrieved passages", term)
            return answer, None

        # The model sometimes gets there on its own; quoting it twice is worse
        # than not quoting it at all.
        normalized = " ".join(best_sentence.lower().split())[:80]
        if normalized and normalized in " ".join(answer.lower().split()):
            return answer, None

        lead_in = "Dalla fonte" if language == "it" else "From the source"
        quotation = f"**{lead_in}:** «{best_sentence}» [{best_item.ref_id}]"
        logger.info(
            "Verbatim definition of %r taken from %s (score %.1f)",
            term,
            best_item.ref_id,
            best_score,
        )
        return quotation + "\n\n" + answer.lstrip(), best_item.ref_id

    def _retrieval_channels_disabled(self) -> bool:
        """Whether the configuration turns every retrieval channel off.

        True only for a deliberately retrieval-free arm (the `no_retrieval`
        LLM-only baseline). It separates "retrieval ran and found nothing",
        which is an honest insufficiency, from "retrieval was never asked to
        run", where refusing measures nothing about the model.

        Returns:
            True when no KG channel and no text channel is enabled.
        """
        cfg = self.config
        return not (
            cfg.include_nodes
            or cfg.include_triples
            or cfg.include_neighbors
            or cfg.include_subgraph
            or cfg.include_shortest_path
            or cfg.use_text_retriever
        )

    def _generate(self, state: RAGState) -> dict:
        """Generate the answer and apply the citation and quote gates.

        With no evidence at all (outside the LLM-only baseline) a fixed
        insufficiency message is returned. A sparse context adds an instruction
        to answer from what is there; a refusal, or an ungrounded answer on a
        sparse context, is replaced by an evidence summary. With
        ``cite_evidence`` the tags are verified, quotes checked, a verbatim
        definition prepended when asked for, and a source list appended;
        otherwise a graph-verification block is appended.

        Returns:
            ``answer``, plus ``pre_retry_answer``, ``refusal_retry_applied``
            and, with citations, ``citation_report`` and ``quote_report``.
        """
        query = state.get("question", "")
        # Every fixed string below picks its language like the generated
        # answer does, with the transcript behind the question
        # (`LLMManager._answer_language`): a continuation such as "Non ho
        # capito niente" carries no marker, and the tie would go to English.
        transcript = str(state.get("transcript", "") or "")
        answer_language = LLMManager._answer_language(
            query, transcript, self.config.fallback_language
        )
        context = state.get("text_context", "")
        has_text_evidence = bool(str(context or "").strip())
        nodes_count = int(state.get("retrieved_nodes_count", 0) or 0)
        triples_count = len(state.get("kg_triples", []) or [])
        subgraph_count = int(state.get("retrieved_subgraph_count", 0) or 0)
        shortest_path_count = int(state.get("retrieved_shortest_path_count", 0) or 0)

        kg_evidence_units = (
            nodes_count + triples_count + subgraph_count + shortest_path_count
        )
        evidence_units = kg_evidence_units + (1 if has_text_evidence else 0)

        llm_only_baseline = evidence_units == 0 and self._retrieval_channels_disabled()

        if evidence_units == 0 and not llm_only_baseline:
            logger.warning(
                "Generation with zero evidence: retrieval mode=%s returned no nodes, "
                "triples, subgraph, shortest_path or text context for query=%r",
                state.get("chosen_retrieval_mode", "HYBRID"),
                str(query)[:200],
            )
            if answer_language == "it":
                return {
                    "answer": (
                        "Il contesto disponibile non è sufficiente per dare una risposta fondata. "
                        "Prova a riformulare la domanda o a renderla più specifica."
                    )
                }
            return {
                "answer": (
                    "The provided context is insufficient to generate a grounded response. "
                    "Please provide additional context or a more specific question."
                )
            }

        # The sparse-context nudge tells the model to work from the available
        # context; on the LLM-only arm there is none, so it would be an
        # instruction to answer from nothing.
        sparse_context = (
            not llm_only_baseline
            and evidence_units <= 2
            and len(str(context or "").strip()) < 1600
        )
        effective_query = query
        if sparse_context:
            # Match the instruction language to the question language: a fixed
            # Italian instruction on an English question pushes the model into
            # mixed-language answers.
            if answer_language == "it":
                effective_query = (
                    query
                    + "\n\nIstruzione: rispondi direttamente usando solo il contesto disponibile. "
                    + "Se il contesto e limitato, fornisci comunque la migliore risposta possibile e aggiungi una breve sezione 'Limiti e affidabilità'."
                )
            else:
                effective_query = (
                    query
                    + "\n\nInstruction: answer directly using only the available context. "
                    + "If the context is limited, still provide the best possible answer and add a short 'Limits and confidence' section."
                )

        # The LLM-only baseline must be asked the question the way a bare LLM
        # would be asked it. Under the default grounding rule ("use ONLY the
        # provided context") an empty context turns the whole arm into a refusal
        # generator, which measures the prompt rather than the model and makes
        # every retrieval arm look better than it is.
        generation_config = self.config
        if llm_only_baseline:
            generation_config = dataclasses.replace(
                self.config, allow_parametric_fallback=True
            )

        if self.llm:
            result = self.llm.generate(
                query=effective_query,
                context=context,
                config=generation_config,
                transcript=str(state.get("transcript", "") or ""),
                on_token=self._on_token,
            )
            answer = result.get("answer", "")
            # Carried to the artifacts so the abstention metric can be computed
            # on the pre-retry answer.
            retry_fields = {
                "pre_retry_answer": str(result.get("pre_retry_answer", "") or ""),
                "refusal_retry_applied": bool(result.get("refusal_retry_applied")),
            }
            logger.info(
                "LLM returned (first 500 chars): %s | sparse_context=%s | evidence_units=%d",
                answer[:500],
                sparse_context,
                evidence_units,
            )
            if evidence_units > 0 and self._should_replace_with_fallback(
                answer=answer,
                query=query,
                context=context,
                triples=state.get("kg_triples", []) or [],
                sparse_context=sparse_context,
            ):
                logger.info(
                    "FALLBACK TRIGGERED: replacing LLM answer with evidence-based fallback"
                )
                answer = self._build_sparse_fallback_answer(
                    query=query,
                    context=context,
                    triples=state.get("kg_triples", []) or [],
                    language=answer_language,
                )
            evidence_items = evidence_from_dicts(
                state.get("evidence_index", []) or []
            )
            if self.config.cite_evidence and evidence_items:
                # The citation gate replaces the verification block: the source
                # list is derived from what the model actually cited, not from
                # the top retrieved triples.
                language = answer_language
                visible = state.get("visible_evidence_refs")
                report = verify_citations(
                    answer=answer,
                    evidence=evidence_items,
                    policy=self.config.citation_policy,
                    language=language,
                    visible_refs=visible if visible else None,
                )
                answer = report.answer
                quote_report = None
                if self.config.verify_quoted_passages:
                    # The definitional prompt asks the model to open with the
                    # source's own words. A fabricated quote can carry a valid [S2], so
                    # the citation gate cannot see it: the quoted string itself
                    # is matched against the passages the model was shown.
                    quote_report = verify_quotes(answer=answer, evidence=evidence_items)
                    answer = quote_report.answer
                # The source's own definition, extracted here rather than asked
                # of the model: asked to copy an English passage into an Italian
                # answer, the model translates it, and a translated quotation is
                # not a quotation.
                answer, definition_ref = self._prepend_source_definition(
                    answer=answer,
                    question=query,
                    evidence=evidence_items,
                    language=language,
                )
                cited_refs = list(report.cited_refs)
                if definition_ref and definition_ref not in cited_refs:
                    # The quotation is a citation: without this the passage it
                    # came from can be missing from the source list under it.
                    cited_refs.insert(0, definition_ref)
                if self.config.citation_display == "label":
                    # Reader-facing rendering: ids for the gate, document and
                    # page for the person reading the answer. Grouping the
                    # source list by document also keeps a heavily cited
                    # answer's list from being cut short.
                    answer = render_display_citations(answer, evidence_items)
                    references = render_grouped_reference_list(
                        evidence=evidence_items,
                        cited_refs=cited_refs,
                        language=language,
                    )
                else:
                    references = render_reference_list(
                        evidence=evidence_items,
                        cited_refs=cited_refs,
                        language=language,
                    )
                if references:
                    answer = answer.rstrip() + "\n\n" + references
                logger.info(
                    "Citation gate: %d tags, %d distinct valid, %d phantom",
                    report.total_citations,
                    len(report.cited_refs),
                    len(report.phantom_refs),
                )
                generated = {
                    "answer": answer,
                    "citation_report": report.as_dict(),
                    **retry_fields,
                }
                if quote_report is not None and quote_report.total_quotes:
                    generated["quote_report"] = quote_report.as_dict()
                return generated

            # A graph-verification block under an answer produced with no graph
            # at all is noise, and on the LLM-only arm it would land inside the
            # very text the answer-channel scorer reads.
            if not llm_only_baseline:
                verification_section = self._build_verification_section(
                    triples=state.get("kg_triples", []) or [],
                    nodes=state.get("retrieved_nodes", []) or [],
                    language=answer_language,
                )
                if verification_section:
                    answer = answer.rstrip() + "\n\n" + verification_section

            return {"answer": answer, **retry_fields}

        return {"answer": "LLM not available."}

    def _should_replace_with_fallback(
        self,
        answer: str,
        query: str,
        context: str,
        triples: list[dict[str, object]],
        sparse_context: bool,
    ) -> bool:
        """Whether to replace the answer with the evidence-based fallback.

        Args:
            answer: Generated answer.
            query: The question.
            context: Retrieved context.
            triples: Retrieved triples.
            sparse_context: Whether the context was sparse.

        Returns:
            True for a refusal or empty answer, or for an answer on a sparse
            context that mentions no salient term of the question, context or
            triples.
        """
        # A genuine refusal / empty answer is always replaced with the evidence
        # block; this is the only unconditional trigger.
        if looks_like_refusal(answer):
            return True

        # Otherwise only intervene when the context was sparse AND the answer is
        # ungrounded. A well-formed answer that references a salient query/context
        # term or a retrieved triple is kept as-is. Generic words such as
        # "context" or "information" are deliberately not treated as signs of an
        # ungrounded answer: they occur in perfectly good answers.
        if not sparse_context:
            return False

        answer_lower = answer.lower()
        salient_terms = self._extract_salient_terms(query=query, context=context)
        triple_terms = self._extract_salient_terms_from_triples(triples)

        # Nothing to judge groundedness against: trust the model's answer.
        if not salient_terms and not triple_terms:
            return False

        if any(term in answer_lower for term in salient_terms):
            return False
        if triple_terms and any(term in answer_lower for term in triple_terms):
            return False

        return True

    @staticmethod
    def _build_sparse_fallback_answer(
        query: str,
        context: str,
        triples: list[dict[str, object]],
        language: str = "en",
    ) -> str:
        """Fixed partial answer listing the relevant evidence found.

        Args:
            query: The question.
            context: Retrieved context.
            triples: Retrieved triples.
            language: ``"it"`` or anything else, which is answered in English.

        Returns:
            A short answer with a limits note and the relevant triples (or
            context lines), or a "too sparse" message when there is nothing
            to list.
        """
        triple_summaries = KGRAGAgent._triple_summaries(triples, query=query)
        highlights = KGRAGAgent._extract_context_highlights(
            query=query, context=context
        )
        is_it = language == "it"

        if triple_summaries or highlights:
            evidence_block = "\n".join(
                f"- {line}" for line in (triple_summaries or highlights)
            )
            if is_it:
                return (
                    "Dal contesto disponibile emergono i seguenti elementi rilevanti. "
                    "La risposta e quindi parziale, ma contiene le evidenze trovate nel grafo.\n\n"
                    "Limiti e fiducia:\n"
                    "Il contesto e limitato, quindi non posso inferire l'intero perimetro tematico con alta fiducia.\n\n"
                    "Evidenze rilevanti:\n"
                    f"{evidence_block}"
                )
            return (
                "The available context surfaces the following relevant elements. "
                "The answer is therefore partial, but it reports the evidence found in the graph.\n\n"
                "Limits and confidence:\n"
                "The context is limited, so the full thematic scope cannot be inferred with high confidence.\n\n"
                "Relevant evidence:\n"
                f"{evidence_block}"
            )

        if is_it:
            return (
                "Il contesto disponibile e troppo scarno per costruire una risposta affidabile. "
                "Serve un recupero piu specifico o piu evidenza dal grafo."
            )
        return (
            "The available context is too sparse to build a reliable answer. "
            "A more specific retrieval or more graph evidence is needed."
        )

    @staticmethod
    def _build_verification_section(
        triples: list[dict[str, object]],
        nodes: list[dict[str, object]],
        limit: int = 4,
        language: str = "it",
    ) -> str:
        """Render the graph-verification block in the answer's language.

        Args:
            triples: Retrieved triples to show.
            nodes: Retrieved nodes, used when no triple is renderable.
            limit: Maximum lines to render.
            language: ``"it"`` or ``"en"``; anything else is treated as English.

        Returns:
            The rendered block, or a language-matched "no evidence" line.
        """
        lines: list[str] = []
        seen: set[str] = set()

        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            predicate = str(triple.get("predicate", "")).strip()
            obj = str(triple.get("object", "")).strip()
            if not (subject or predicate or obj):
                continue

            parts = [f"({subject}, {predicate}, {obj})"]

            subject_id = str(triple.get("subject_id", "")).strip()
            object_id = str(triple.get("object_id", "")).strip()
            if subject_id or object_id:
                id_bits = ", ".join(
                    bit
                    for bit in (
                        f"s={subject_id}" if subject_id else "",
                        f"o={object_id}" if object_id else "",
                    )
                    if bit
                )
                if id_bits:
                    parts.append(f"[{id_bits}]")

            rel_props = triple.get("relationship_properties", {})
            if isinstance(rel_props, dict):
                source_doc = str(
                    rel_props.get("source_doc", "") or rel_props.get("source", "") or ""
                ).strip()
                page_range = str(rel_props.get("page_range", "")).strip()
                provenance_bits = [bit for bit in (source_doc, page_range) if bit]
                if provenance_bits:
                    parts.append(f"<{' | '.join(provenance_bits)}>")

            line = " ".join(parts)
            if line in seen:
                continue
            seen.add(line)
            lines.append(f"- {line}")
            if len(lines) >= limit:
                break

        if not lines:
            for node in nodes:
                text = str(node.get("text", "")).strip()
                node_id = str(node.get("node_id", "")).strip()
                labels = node.get("labels", [])
                label_text = (
                    ", ".join(str(label) for label in labels)
                    if isinstance(labels, list)
                    else ""
                )
                if not text:
                    continue
                detail = f"({text})"
                if label_text:
                    detail += f" [{label_text}]"
                if node_id:
                    detail += f" [id={node_id}]"
                if detail in seen:
                    continue
                seen.add(detail)
                lines.append(f"- {detail}")
                if len(lines) >= limit:
                    break

        heading = "Verifica nel grafo:" if language == "it" else "Graph verification:"

        if not lines:
            if language == "it":
                return (
                    f"{heading}\n"
                    "- Nessuna evidenza strutturata recuperata da mostrare in "
                    "modo affidabile."
                )
            return (
                f"{heading}\n"
                "- No structured evidence was retrieved that can be shown "
                "reliably."
            )

        return f"{heading}\n" + "\n".join(lines)

    @staticmethod
    def _extract_context_highlights(
        query: str, context: str, limit: int = 4
    ) -> list[str]:
        """Context lines containing a salient term, else the first lines.

        Returns:
            At most ``limit`` distinct lines.
        """
        tokens = set(KGRAGAgent._extract_salient_terms(query=query, context=context))

        highlights: list[str] = []
        seen: set[str] = set()

        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            lowered = line.lower()
            if not any(token in lowered for token in tokens):
                continue

            if line in seen:
                continue

            seen.add(line)
            highlights.append(line)
            if len(highlights) >= limit:
                break

        if highlights:
            return highlights

        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            highlights.append(line)
            if len(highlights) >= limit:
                break

        return highlights

    @staticmethod
    def _triple_summaries(
        triples: list[dict[str, object]], query: str, limit: int = 5
    ) -> list[str]:
        """Triples rendered as ``(s, p, o)``, those matching the question first.

        Returns:
            Up to ``limit`` triples containing a salient term of the question,
            or the first ``limit`` triples when none does.
        """
        if not triples:
            return []

        query_terms = KGRAGAgent._extract_salient_terms_from_text(query)
        focus_terms = {term for term in query_terms if term}

        matched: list[str] = []
        fallback: list[str] = []

        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            predicate = str(triple.get("predicate", "")).strip()
            obj = str(triple.get("object", "")).strip()
            if not (subject or predicate or obj):
                continue

            summary = f"({subject}, {predicate}, {obj})"
            fallback.append(summary)

            haystack = f"{subject} {predicate} {obj}".lower()
            if any(term in haystack for term in focus_terms):
                matched.append(summary)
                if len(matched) >= limit:
                    break

        if matched:
            return matched

        return fallback[:limit]

    @staticmethod
    def _extract_salient_terms_from_triples(
        triples: list[dict[str, object]],
    ) -> list[str]:
        """Up to 16 lower-cased acronyms found in the triples' fields."""
        terms: list[str] = []
        seen: set[str] = set()

        for triple in triples:
            for field in ("subject", "predicate", "object"):
                raw_value = str(triple.get(field, "")).strip()
                if not raw_value:
                    continue
                for token in re.findall(r"\b[A-Z][A-Z0-9/&.-]{1,}\b", raw_value):
                    lowered = token.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    terms.append(lowered)

        return terms[:16]

    @staticmethod
    def _extract_salient_terms_from_text(text: str) -> list[str]:
        """Salient terms for relevance grading, most discriminative first.

        Three tiers, in order: ALL-CAPS acronyms, capitalised proper nouns, then
        lowercase content words. Acronyms alone would reduce a question to its
        one acronym, and find nothing in a question that has none.

        Args:
            text: Question or context to mine.

        Returns:
            Up to 16 lowercase terms, deduplicated, ordered by tier.
        """
        terms: list[str] = []
        seen: set[str] = set()

        def add(value: str, min_len: int) -> None:
            """Append a lower-cased term unless short, a stopword or repeated."""
            lowered = value.strip().lower()
            if len(lowered) < min_len or lowered in seen or lowered in _STOPWORDS:
                return
            seen.add(lowered)
            terms.append(lowered)

        # Tier 1: acronyms (CEFF, SDG, MATTM).
        for token in re.findall(r"\b[A-Z][A-Z0-9/&.-]{1,}\b", text):
            add(token, 2)
        # Tier 2: capitalised proper nouns (Farm to Fork, Piemonte).
        for token in re.findall(r"\b[A-Z][A-Za-z0-9/&.-]{2,}\b", text):
            add(token, 3)
        # Tier 3: lowercase content words, the only tier that survives a
        # question with no capitalisation at all.
        for token in re.findall(
            r"\b[\wÀ-ÖØ-öø-ÿ'-]{%d,}\b" % _MIN_CONTENT_TERM_LEN,
            text,
            flags=re.UNICODE,
        ):
            add(token, _MIN_CONTENT_TERM_LEN)

        return terms[:16]

    @staticmethod
    def _extract_salient_terms(query: str, context: str) -> list[str]:
        """Salient terms of the question, then of the context.

        Acronyms, capitalised words and other words of at least three
        characters, minus stopwords.

        Returns:
            Up to 12 lower-cased distinct terms.
        """
        terms: list[str] = []
        seen: set[str] = set()

        def add_term(value: str) -> None:
            """Append a lower-cased term unless short, a stopword or repeated."""
            normalized = value.strip().lower()
            if len(normalized) < 3:
                return
            if normalized in seen:
                return
            if normalized in _STOPWORDS:
                return
            seen.add(normalized)
            terms.append(normalized)

        # Acronyms and proper nouns first.
        for token in re.findall(r"\b[A-Z][A-Z0-9/&.-]{1,}\b", query):
            add_term(token)
        for token in re.findall(r"\b[A-Z][A-Za-z0-9/&.-]{2,}\b", query):
            add_term(token)

        for token in re.findall(r"\b[\wÀ-ÖØ-öø-ÿ'/-]{3,}\b", query, flags=re.UNICODE):
            add_term(token)

        for line in context.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for token in re.findall(r"\b[A-Z][A-Z0-9/&.-]{1,}\b", stripped):
                add_term(token)
            for token in re.findall(r"\b[A-Z][A-Za-z0-9/&.-]{2,}\b", stripped):
                add_term(token)
            for token in re.findall(
                r"\b[\wÀ-ÖØ-öø-ÿ'/-]{3,}\b", stripped, flags=re.UNICODE
            ):
                add_term(token)

        return terms[:12]

    def _rewrite_with_memory(
        self, question: str, memory: ConversationMemory
    ) -> str:
        """Make an elliptical follow-up self-contained, for retrieval only.

        Args:
            question: The question as typed.
            memory: Conversation memory supplying the seed entities and the
                previous question.

        Returns:
            The rewritten question, or ``question`` when there is no LLM, no
            seed entity, the call fails, or the reply is implausible.
        """
        if self.llm is None:
            return question

        entities = memory.seed_entities()
        if not entities:
            logger.debug(
                "Follow-up detected but no seed entities available; "
                "keeping the question as typed."
            )
            return question

        prompt = PromptLibrary.followup_rewrite_prompt(self.config)
        rendered = prompt.invoke(
            {
                "question": question,
                "entities": ", ".join(entities),
                "previous_question": memory.last_question or "(none)",
            }
        )
        try:
            model = self.llm.load_llm()
            output = self.llm._invoke_with_retry(model, rendered)
        except Exception as exc:  # noqa: BLE001 - a failed rewrite must not lose the turn
            logger.warning("Follow-up rewrite failed (%s); keeping the question.", exc)
            return question

        raw = str(output.content if hasattr(output, "content") else output)
        rewritten = _plausible_rewrite(raw, question)

        if rewritten != question:
            logger.info("Follow-up rewritten for retrieval: %r -> %r", question, rewritten)
        return rewritten

    def invoke(
        self,
        question: str,
        memory: ConversationMemory | None = None,
        on_token: Callable[[str | None], None] | None = None,
    ) -> dict:
        """Answer one question, optionally in the context of a conversation.

        Args:
            question: The question as typed by the user.
            memory: Intra-session memory. With `None` — the default, and what
                every CLI, gold and experiment run uses — no rewrite,
                transcript or quoted-source preference is applied.
            on_token: Called with each piece of the answer as the model writes
                it, and with `None` when a retry discards what was already
                sent. Generation is almost the whole wait, so streaming lets a
                reader watch the answer fill instead of waiting. The text it
                emits is the model's own: the citation gate and the source list
                are applied after, and the caller is expected to replace what
                it streamed with the final answer.

        Returns:
            The final graph state, plus `latency_ms`, `stage_timings_ms` and,
            when memory is active, the original question, the question sent to
            retrieval, the entities that resolved it and the follow-up flag.

        Raises:
            Exception: Whatever the graph raised; the failed turn is recorded
                in ``memory`` first.
        """
        # Reset per invocation, on this thread only: `_timed` adds to it as
        # each node runs and `output` carries the result away.
        self._stage_clock.timings = {}
        start = time.perf_counter()
        self._on_token = on_token
        initial_state = {
            "question": question,
            "run_id": str(uuid.uuid4()),
            "rewrite_count": 0,
        }

        # Memory steers retrieval only: `_retrieve` and `_grade` read
        # `rewritten_question`, `_generate` reads `question`. The reader's
        # literal wording keeps driving the answer and its language.
        follow_up = False
        retrieval_question = question
        seed_entities: list[str] = []
        if memory is not None:
            seed_entities = memory.seed_entities()
            # Condense whenever the conversation has started, and let the
            # rewrite prompt decide: it is told to repeat the question
            # unchanged when it already stands on its own. Surface heuristics
            # misjudge it both ways ("Spiegameli meglio" looks fresh,
            # "e <anything>" looks like a continuation).
            follow_up = memory.has_context()
            # Read by `_scope_gate` (through `_gate_question`), so a question
            # that only makes sense against the previous turn is judged on its
            # rewritten form.
            initial_state["follow_up"] = follow_up
            # Read by `_generate`, so a question that quotes an earlier answer
            # is answered instead of being denied. Empty on the first turn,
            # which adds nothing to that turn's prompt.
            transcript = memory.transcript()
            if transcript:
                initial_state["transcript"] = transcript
            # When the question repeats a sentence the assistant wrote, the
            # document that backed that sentence is better provenance for the
            # follow-up than the words of the question. Empty unless something
            # is actually quoted, so no other turn changes.
            quoted_sources = memory.sources_for_quote(question)
            if quoted_sources:
                initial_state["quoted_sources"] = quoted_sources
                logger.info(
                    "Question quotes an earlier answer; preferring %s",
                    ", ".join(quoted_sources),
                )
            if follow_up:
                retrieval_question = self._rewrite_with_memory(question, memory)
                if retrieval_question != question:
                    initial_state["rewritten_question"] = retrieval_question

        try:
            output = self.graph.invoke(
                initial_state, config={"recursion_limit": self.config.recursion_limit}
            )
        except GraphRecursionError:
            logger.warning(
                "Graph recursion limit reached (limit=%d) for question: %s",
                self.config.recursion_limit,
                question,
            )
            # Same rule as the messages in `_generate`: a mute continuation
            # takes the conversation's language, not the tie's default.
            if (
                LLMManager._answer_language(
                    question,
                    str(initial_state.get("transcript", "") or ""),
                    self.config.fallback_language,
                )
                == "it"
            ):
                output = {
                    "answer": (
                        "Il processo ha raggiunto il limite di ricorsione dell'agente prima di convergere. "
                        "Prova con una domanda piu specifica o aumenta --recursion-limit."
                    )
                }
            else:
                output = {
                    "answer": (
                        "The agent hit its recursion limit before converging. "
                        "Try a more specific question or raise --recursion-limit."
                    )
                }
        except Exception:
            # The turn happened even though it produced nothing. `observe` runs
            # below, after a successful invoke, so without this a raised first
            # turn would leave `has_context()` false, and the next follow-up —
            # asked precisely because the first attempt failed — would be
            # treated as a fresh question. Recorded, then re-raised: the caller
            # still needs to know, and the demo's graph failover depends on
            # catching it.
            if memory is not None:
                memory.observe_failure(question)
            raise
        latency_ms = (time.perf_counter() - start) * 1000.0
        output["latency_ms"] = latency_ms
        output["stage_timings_ms"] = dict(
            getattr(self._stage_clock, "timings", None) or {}
        )

        if memory is not None:
            output["original_question"] = question
            output["retrieval_question"] = retrieval_question
            output["memory_entities"] = seed_entities
            output["follow_up"] = follow_up
            memory.observe(
                question=question,
                answer=str(output.get("answer", "") or ""),
                nodes=output.get("retrieved_nodes", []) or [],
                triples=[
                    *(output.get("kg_triples", []) or []),
                    *(output.get("retrieved_subgraph", []) or []),
                ],
            )
        return output
