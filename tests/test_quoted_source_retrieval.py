"""A follow-up that quotes the assistant is retrieved against its own source.

A reader repeats a sentence the assistant wrote and asks about it. The
sentence is backed by a cited passage (REPORT MATTM p. 70 in the case below),
but retrieval on the words of the question comes back with unrelated
documents. The claim's own citation is better provenance for a question about
that claim than the question's phrasing.

Nothing fires unless a sentence is actually quoted — a small fraction of the
recorded turns — so every other turn retrieves exactly as it would without
the feature.
"""

from __future__ import annotations

from typing import Any

from graphrag.agent.core import KGRAGAgent
from graphrag.agent.memory import ConversationMemory
from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever

# A real turn, from a recorded session.
ANSWER = (
    "Il progetto ha analizzato le grafiche di molti packaging alimentari "
    "presenti sul mercato [REPORT MATTM, p. 70]. La ricerca resta aperta ad "
    "altri materiali."
)
FOLLOW_UP = "Hai scritto grafiche di molti packaging alimentari presenti sul mercato, quali?"


class _Chunk:
    """A text chunk with a `doc#page=N` source."""

    def __init__(self, source: str, content: str = "testo") -> None:
        self.source = source
        self.content = content


def _memory_after_the_answer() -> ConversationMemory:
    """Memory after the turn whose answer the follow-up quotes."""
    memory = ConversationMemory()
    memory.observe(question="Parlami del packaging", answer=ANSWER)
    return memory


# --- what the memory can tell -------------------------------------------


def test_a_quoted_sentence_names_the_document_that_backed_it():
    memory = _memory_after_the_answer()

    assert memory.sources_for_quote(FOLLOW_UP) == ["REPORT MATTM, p. 70"]


def test_a_question_that_quotes_nothing_names_nothing():
    memory = _memory_after_the_answer()

    assert memory.sources_for_quote("E il micelio, centra qualcosa?") == []


def test_a_short_overlap_is_not_a_quotation():
    memory = _memory_after_the_answer()

    # Four words in common is a shared turn of phrase, not a citation to follow.
    assert memory.sources_for_quote("Parlami del packaging alimentari") == []


def test_an_uncited_sentence_pins_nothing_rather_than_guessing():
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="Le 3C introducono una dimensione qualitativa e relazionale del sistema.",
    )

    assert memory.sources_for_quote(
        "Hai scritto: introducono una dimensione qualitativa e relazionale del sistema"
    ) == []


def test_an_unverified_reference_is_never_followed():
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer=(
            "Il consorzio raccoglie duemila tonnellate l'anno "
            "[riferimento non verificato]."
        ),
    )

    assert memory.sources_for_quote(
        "Hai scritto: raccoglie duemila tonnellate l'anno, da dove viene?"
    ) == []


def test_a_bare_evidence_id_is_not_a_document():
    memory = ConversationMemory()
    memory.observe(question="q", answer="La paglia di riso serve come substrato [S3].")

    assert memory.sources_for_quote(
        "Hai scritto: la paglia di riso serve come substrato, perché?"
    ) == []


def test_the_most_recent_turn_wins_when_two_answers_are_quoted():
    memory = ConversationMemory()
    memory.observe(question="q1", answer="Il packaging alimentare pesa molto [Doc A, p. 1].")
    memory.observe(question="q2", answer="Il packaging alimentare pesa molto [Doc B, p. 2].")

    assert memory.sources_for_quote(
        "Hai scritto che il packaging alimentare pesa molto, quanto?"
    ) == ["Doc B, p. 2", "Doc A, p. 1"]


# --- what the retriever does with it -------------------------------------


def test_the_quoted_document_is_floated_to_the_top():
    chunks = [
        _Chunk("artifacts/corpus/Notpla brief.pdf#page=2#chunk=1"),
        _Chunk("artifacts/corpus/REPORT MATTM_Definitivo.pdf#page=70#chunk=3"),
        _Chunk("artifacts/corpus/Geca 45.pdf#page=1#chunk=0"),
    ]

    promoted = KGRetriever._promote_documents(chunks, ["REPORT MATTM, p. 70"])

    assert [c.source for c in promoted][0].endswith("REPORT MATTM_Definitivo.pdf#page=70#chunk=3")
    # A promotion, not a filter: the rest survives, in its original order.
    assert len(promoted) == 3
    assert [c.source for c in promoted[1:]] == [chunks[0].source, chunks[2].source]


def test_no_preference_leaves_the_ranking_exactly_as_it_was():
    chunks = [_Chunk("a.pdf#page=1"), _Chunk("b.pdf#page=2")]

    assert KGRetriever._promote_documents(chunks, []) == chunks
    assert KGRetriever._promote_documents(chunks, ["", "  "]) == chunks


def test_a_label_only_matches_the_document_it_was_made_from():
    chunks = [_Chunk("artifacts/corpus/REPORT MATTM_Definitivo.pdf#page=70")]

    assert KGRetriever._promote_documents(chunks, ["Notpla brief, p. 2"]) == chunks


# --- the two ends joined --------------------------------------------------


class _RecordingRetriever:
    """Stands in for KGRetriever and records what the agent asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def retrieve(self, query: str, prefer_documents: Any = ()) -> dict[str, Any]:
        self.calls.append((query, tuple(prefer_documents)))
        return {"context_text": "contesto", "nodes": [], "triples": []}


def _state(question: str, quoted: list[str] | None = None) -> dict[str, Any]:
    """Retrieve-node state, with quoted sources when given."""
    state: dict[str, Any] = {"question": question, "chosen_retrieval_mode": "TEXT"}
    if quoted is not None:
        state["quoted_sources"] = quoted
    return state


def test_the_retrieve_node_passes_the_preference_through():
    retriever = _RecordingRetriever()
    agent = KGRAGAgent(config=AgentConfig(), kg_retriever=retriever, llm=None)

    agent._retrieve(_state(FOLLOW_UP, ["REPORT MATTM, p. 70"]))

    assert retriever.calls
    assert all(call[1] == ("REPORT MATTM, p. 70",) for call in retriever.calls)


def test_a_turn_without_a_quote_asks_for_no_preference():
    retriever = _RecordingRetriever()
    agent = KGRAGAgent(config=AgentConfig(), kg_retriever=retriever, llm=None)

    agent._retrieve(_state("Parlami delle 3C"))

    assert retriever.calls
    assert all(call[1] == () for call in retriever.calls)


def test_the_cache_does_not_serve_an_unpreferred_turn_to_a_preferred_one():
    retriever = _RecordingRetriever()
    agent = KGRAGAgent(config=AgentConfig(), kg_retriever=retriever, llm=None)

    agent._retrieve(_state(FOLLOW_UP))
    calls_after_first = len(retriever.calls)
    agent._retrieve(_state(FOLLOW_UP, ["REPORT MATTM, p. 70"]))

    # Same question text: without the preference in the key the second turn
    # would have been served the first turn's ranking.
    assert len(retriever.calls) > calls_after_first
    assert retriever.calls[-1][1] == ("REPORT MATTM, p. 70",)


class _RecordingGraphAgent(KGRAGAgent):
    """Agent whose graph records the initial state instead of running."""

    def _build_graph(self):  # type: ignore[override]
        agent = self

        class _Graph:
            def invoke(self, state: dict, config: dict | None = None) -> dict:
                agent.seen_state = dict(state)
                return {"answer": "risposta", "retrieved_nodes": [], "kg_triples": []}

        return _Graph()


def test_invoke_seeds_the_state_from_the_conversation():
    agent = _RecordingGraphAgent(config=AgentConfig(), kg_retriever=None, llm=None)
    memory = _memory_after_the_answer()

    agent.invoke(FOLLOW_UP, memory=memory)

    assert agent.seen_state.get("quoted_sources") == ["REPORT MATTM, p. 70"]


def test_invoke_leaves_the_key_absent_when_nothing_is_quoted():
    agent = _RecordingGraphAgent(config=AgentConfig(), kg_retriever=None, llm=None)
    memory = _memory_after_the_answer()

    agent.invoke("Parlami invece del micelio", memory=memory)

    # Absent, not empty: a first-turn state must be byte-identical to one built
    # without this feature.
    assert "quoted_sources" not in agent.seen_state


def test_a_first_turn_never_carries_a_preference():
    agent = _RecordingGraphAgent(config=AgentConfig(), kg_retriever=None, llm=None)

    agent.invoke(FOLLOW_UP, memory=ConversationMemory())

    assert "quoted_sources" not in agent.seen_state


# --- following the citation: the passage must arrive even when ranking misses ---


class _FakePipeline:
    """A text index that ranks, and can also be asked for a document directly."""

    def __init__(self, ranked: list[_Chunk], indexed: list[_Chunk] | None = None) -> None:
        self.ranked = ranked
        self.indexed = indexed if indexed is not None else list(ranked)
        self.calls: list[int] = []
        self.lookups: list[tuple[str, str]] = []

    def retrieve(self, query: str, top_k: int = 5, **_kw: Any) -> list[_Chunk]:
        self.calls.append(top_k)
        return self.ranked[:top_k]

    def chunks_from(self, document_label: str, page: str = "") -> list[_Chunk]:
        self.lookups.append((document_label, page))
        from graphrag.agent.evidence import parse_chunk_source, short_doc_label

        out = []
        for chunk in self.indexed:
            document, chunk_page = parse_chunk_source(chunk.source)
            if short_doc_label(document).strip().lower() != document_label.strip().lower():
                continue
            out.append((chunk_page == page.strip(), chunk))
        out.sort(key=lambda pair: not pair[0])
        return [chunk for _, chunk in out]


def _retriever_with(pipeline: _FakePipeline) -> KGRetriever:
    """A `KGRetriever` over `pipeline`, with only the text channel on."""
    config = AgentConfig(
        use_text_retriever=True,
        text_retriever_top_k=3,
        text_retriever_max_per_doc=2,
        text_retriever_fetch_k=20,
        vector_retrieval=False,
        include_nodes=False,
        include_triples=False,
    )
    retriever = KGRetriever.__new__(KGRetriever)
    retriever.config = config
    retriever.text_pipeline = pipeline
    return retriever


def test_the_cited_passage_arrives_even_when_the_ranking_never_reaches_it():
    """A deeper pool is not enough: the cited passage is looked up, not ranked.

    A larger pool of the same ranking does not contain the cited document: the
    question is phrased in the reader's words, not the source's. The citation
    names a document and a page, so the passage is a lookup.
    """
    ranked = [
        _Chunk("corpus/MR 51.pdf#page=4"),
        _Chunk("corpus/Materia 45.pdf#page=4"),
        _Chunk("corpus/MR37.pdf#page=6"),
    ]
    cited = _Chunk("corpus/REPORT MATTM_Definitivo.pdf#page=70", "il passo citato")
    pipeline = _FakePipeline(ranked, indexed=ranked + [cited])

    out = _retriever_with(pipeline)._retrieve_text_chunks(
        "una domanda che non nomina il documento", ["REPORT MATTM, p. 70"]
    )

    sources = [c.source for c in out]
    assert "REPORT MATTM" in sources[0], sources
    assert pipeline.lookups == [("REPORT MATTM", "p. 70")]


def test_the_cited_page_is_preferred_over_other_pages_of_the_same_document():
    ranked = [_Chunk("corpus/MR37.pdf#page=6")]
    indexed = ranked + [
        _Chunk("corpus/REPORT MATTM_Definitivo.pdf#page=12"),
        _Chunk("corpus/REPORT MATTM_Definitivo.pdf#page=70"),
    ]
    pipeline = _FakePipeline(ranked, indexed=indexed)

    out = _retriever_with(pipeline)._retrieve_text_chunks(
        "domanda", ["REPORT MATTM, p. 70"]
    )

    assert out[0].source.endswith("#page=70")


def test_no_lookup_when_the_document_is_already_in_the_ranking():
    ranked = [
        _Chunk("corpus/REPORT MATTM_Definitivo.pdf#page=70"),
        _Chunk("corpus/REPORT MATTM_Definitivo.pdf#page=194"),
        _Chunk("corpus/MR37.pdf#page=6"),
    ]
    pipeline = _FakePipeline(ranked)

    out = _retriever_with(pipeline)._retrieve_text_chunks(
        "domanda", ["REPORT MATTM, p. 70"]
    )

    assert [c.source for c in out] == [c.source for c in ranked]
    assert pipeline.lookups == [], "looked the document up while it was already there"


def test_other_documents_survive_the_promotion():
    """The per-document cap must hold: one PDF may not take the whole context."""
    ranked = [_Chunk("corpus/MR37.pdf#page=6"), _Chunk("corpus/Materia 45.pdf#page=4")]
    indexed = ranked + [
        _Chunk(f"corpus/REPORT MATTM_Definitivo.pdf#page={page}")
        for page in (70, 194, 206, 265, 279)
    ]
    pipeline = _FakePipeline(ranked, indexed=indexed)

    out = _retriever_with(pipeline)._retrieve_text_chunks(
        "domanda", ["REPORT MATTM, p. 70"]
    )

    mattm = [c for c in out if "REPORT MATTM" in c.source]
    others = [c for c in out if "REPORT MATTM" not in c.source]
    assert len(mattm) <= 2, [c.source for c in out]
    assert others, "the promotion swallowed every other source"


def test_a_citation_to_a_document_that_is_not_indexed_changes_nothing():
    ranked = [_Chunk("corpus/MR37.pdf#page=6"), _Chunk("corpus/Materia 45.pdf#page=4")]
    pipeline = _FakePipeline(ranked)

    out = _retriever_with(pipeline)._retrieve_text_chunks(
        "domanda", ["Un documento che non esiste, p. 1"]
    )

    assert [c.source for c in out] == [c.source for c in ranked]


def test_a_failing_lookup_leaves_retrieval_working():
    class _Boom(_FakePipeline):
        def chunks_from(self, document_label: str, page: str = ""):
            raise RuntimeError("index unavailable")

    ranked = [_Chunk("corpus/MR37.pdf#page=6"), _Chunk("corpus/Materia 45.pdf#page=4")]
    pipeline = _Boom(ranked)

    out = _retriever_with(pipeline)._retrieve_text_chunks(
        "domanda", ["REPORT MATTM, p. 70"]
    )

    # The preference is a bonus. Losing it must not lose the answer.
    assert [c.source for c in out] == [c.source for c in ranked]


def test_a_pipeline_without_the_lookup_still_works():
    """An older pipeline object must not break the turn."""

    class _Legacy:
        def retrieve(self, query: str, top_k: int = 5, **_kw: Any):
            return [_Chunk("corpus/MR37.pdf#page=6")]

    retriever = KGRetriever.__new__(KGRetriever)
    retriever.config = AgentConfig(
        use_text_retriever=True, text_retriever_top_k=3, text_retriever_max_per_doc=2
    )
    retriever.text_pipeline = _Legacy()

    out = retriever._retrieve_text_chunks("domanda", ["REPORT MATTM, p. 70"])

    assert len(out) == 1
