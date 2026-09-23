"""Agent and knowledge-graph configuration."""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

logger = logging.getLogger("graphrag")


class OUTPUT_TONE(enum.Enum):
    """Register the answer is written in."""

    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"
    FORMAL = "formal"


class OUTPUT_COMPLEXITY(enum.Enum):
    """Level of detail the answer is written at."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class AgentConfig:
    """Settings of one agent run: question anchors, retrieval channels and
    limits, ranking, prompts, and the optional gates and answer rules.

    Most feature flags default to off: each one changes the rendered prompt or
    the retrieved evidence, so a measurement run opts in explicitly and the
    library defaults stay the control arm. Named combinations live in
    :mod:`graphrag.profiles`.
    """

    query: str | None = None
    entity: str | None = None
    entity_a: str | None = None
    entity_b: str | None = None
    hops: int = 1
    max_depth: int = 6
    nodes_limit: int = 10
    triples_limit: int = 20
    neighbors_limit: int = 25
    subgraph_limit: int = 200
    labels: tuple[str, ...] = ()
    relationship_types: tuple[str, ...] = ()
    include_nodes: bool = True
    include_triples: bool = True
    include_neighbors: bool = True
    include_subgraph: bool = True
    include_shortest_path: bool = True
    answer_prompt: str = ""
    rewrite_prompt: str = ""
    kg_reasoning_prompt: str = ""
    decomposition_prompt: str = ""
    reflection_prompt: str = ""
    adaptive_router_prompt: str = ""
    llm_warmup: bool = False
    enable_decomposition_step: bool = False
    enable_adaptive_routing_step: bool = False
    enable_cache: bool = True
    cache_maxsize: int = 128
    recursion_limit: int = 50
    # Context budget. 6000 fits a 32k window with prompt and answer; a much
    # smaller budget makes head/tail compression cut the mid-context evidence
    # of multi-channel retrievals.
    max_content_tokens: int = 6000
    token_estimator_ratio: float = 0.25  # tokens-per-char (~4 chars/token)
    tone: OUTPUT_TONE = OUTPUT_TONE.TECHNICAL
    complexity: OUTPUT_COMPLEXITY = OUTPUT_COMPLEXITY.MEDIUM
    target_audience: str = "domain_expert"
    use_structured_response: bool = False
    rank_triples: bool = True
    # When decomposition produces multiple retrieval queries, merged results
    # keep arrival order; enable to re-rank them globally by score.
    rerank_merged_results: bool = False
    # The answer prompt asks for a 'Limits and confidence' section only when
    # context is sparse; enable to request it on every answer.
    always_include_limits: bool = False
    # Render retrieved evidence as numbered blocks carrying document and page,
    # ask for [S1]/[T1] tags on specific claims, and check every tag against
    # the evidence index after generation.
    cite_evidence: bool = False
    # What to do with a reference tag the model invented: "mark" flags it in
    # place, "strip" deletes it. Marking is the default because deleting leaves
    # an unsupported claim looking like ordinary prose.
    citation_policy: str = "mark"
    evidence_max_text_items: int = 12
    evidence_max_triple_items: int = 30
    # How verified references are shown to the reader. "id" keeps [S1]/[T3],
    # which is what experiment artifacts and the citation metrics parse;
    # "label" rewrites them as "[SEeD for Change, p. 3]" after the gate has run,
    # because a reader cannot check "S3" against anything.
    citation_display: str = "id"
    # Turn the language detected on the question into an explicit constraint
    # of the answer prompt, written in that language, and retry once when the
    # answer comes back in the other language.
    enforce_language: bool = False
    # The answer language when neither the question nor the conversation has a
    # word that tells: "scotta", "vinacce e raspi" and "python" score zero on
    # both sides. English keeps experiment runs as they are; a deployment whose
    # readers write in Italian sets "it".
    fallback_language: str = "en"
    # Triple ranking weights; they should sum to 1. Triples carry no per-edge
    # confidence, so its weight is 0.0 and lexical and mention scores share
    # the rest; the field lets a confidence signal be weighted without code
    # changes.
    ranker_weight_lexical: float = 0.70
    ranker_weight_mention: float = 0.30
    ranker_weight_confidence: float = 0.0
    ranker_system_link_penalty: float = 0.5
    adaptive_hops: bool = True
    min_subgraph_triples: int = 10
    max_hops: int = 4
    include_triple_metadata: bool = True
    # For definitional questions, rank the chunk carrying the verbatim
    # definition first and ask for an answer that quotes before it
    # paraphrases.
    prefer_verbatim_definitions: bool = False
    # Weight of the definitional signal when reordering already-retrieved
    # chunks. It reorders, it never fetches: the worst case is the order the
    # retriever would have produced anyway.
    definition_boost_weight: float = 1.0
    # Checks that every «...» passage occurs in the retrieved text, and drops
    # the guillemets when it does not. Independent of
    # `prefer_verbatim_definitions` because a model can quote unprompted, and a
    # fabricated quote carrying a valid [S2] is the one failure the citation
    # gate cannot see.
    verify_quoted_passages: bool = True
    use_text_retriever: bool = False
    text_retriever_top_k: int = 5
    # Source diversification in the text channel. MMR trades a little query
    # similarity for coverage; the per-document cap is what stops one PDF from
    # filling the context, since two pages of the same document can be far
    # apart in embedding space and still both be selected.
    text_retriever_mmr: bool = False
    text_retriever_mmr_lambda: float = 0.7
    # 0 disables the cap. Enumerative questions get twice this budget: their
    # answer is usually one list on contiguous pages of a single document, and
    # capping that document truncates the list.
    text_retriever_max_per_doc: int = 0
    # Candidate pool the cap and the definitional boost choose from. 0 means
    # ``4 * text_retriever_top_k``.
    text_retriever_fetch_k: int = 0
    text_retriever_backend: str = "tfidf"  # "tfidf" | "dense"
    dense_embedding_model: str = "intfloat/multilingual-e5-base"
    dense_query_prefix: str = "query: "
    dense_passage_prefix: str = "passage: "
    dense_normalize: bool = True
    dense_device: str = "auto"  # "auto" | "cpu" | "cuda"
    vector_index_dir: str = "artifacts/vector_index"
    # The full-text query is a flat OR of every term the question yields, all
    # weighted alike, so a generic token ("framework") outvotes the specific
    # phrase by matching more nodes. Enabling this drops query tokens whose
    # node-name document frequency exceeds `lexical_df_max_ratio` and boosts
    # the surviving terms by rarity.
    lexical_specificity: bool = False
    # A token in more than this share of node names carries no discriminative
    # power.
    lexical_df_max_ratio: float = 0.01
    # Multi-word candidates are the reliable anchors, single tokens the risky
    # ones: weight the phrase query so it survives alongside common tokens.
    lexical_phrase_boost: float = 4.0
    # Ceiling on the rarity boost given to a surviving single token, so one
    # hapax cannot monopolise the result set.
    lexical_max_token_boost: float = 3.0
    lexical_df_cache_path: str = "artifacts/kg_token_df.json"
    # Anchor the neighbour, subgraph and shortest-path channels on node names
    # the index returned rather than on the first search term, which is a raw
    # question word ("valuable", "implementation") that often matches no node.
    # It changes which subgraph those three channels expand.
    seed_from_retrieved: bool = False
    # Add a vector channel to the lexical one. The graph is largely Italian and
    # the questions often English, and many entities exist only under an
    # Italian surface form that no lexical query can reach; a multilingual
    # encoder puts both languages in one space. It is added to the lexical
    # channel, never replaces it: exact surface matches are still the most
    # precise signal. Requires scripts/kg/kg_vector_index.py and a running
    # embedding endpoint, and degrades to lexical-only if either is missing.
    vector_retrieval: bool = False
    vector_index: str = "node_embedding"
    # Nodes pulled from the vector channel per query. Kept near nodes_limit so
    # the two channels contribute comparably instead of one drowning the other.
    vector_nodes_limit: int = 10
    vector_triples_limit: int = 10
    # Nearest nodes expanded into triples. Small on purpose: most of the graph
    # is leaves, so expanding many weak seeds adds edges, not answers.
    vector_seed_limit: int = 5
    # Cosine floor. e5 scores short names in a narrow high band, so a hard
    # threshold mostly removes the tail; ranking does the real work.
    vector_min_score: float = 0.0
    # The answer prompt's "use ONLY the provided context" suppresses the
    # model's own knowledge even when retrieval missed. Enabling this allows a
    # fallback to parametric knowledge, but only marked as such, so
    # groundedness stays measurable.
    allow_parametric_fallback: bool = False
    # Use the earlier closing line of the answer prompt, which permitted a
    # declaration of insufficiency only when the context was empty or carried
    # no factual evidence. Only for reproducing the thesis campaigns E1-E8,
    # which ran with that wording, next to the current one in the same server
    # session.
    legacy_insufficiency_wording: bool = False
    # Out-of-domain gate, run once before retrieval. Without it the agent has
    # no path to abstain: the dense retriever has no score floor, so `_grade`
    # always sees evidence, and `grade_condition` sends every question to
    # `generate` after three rewrites.
    #
    # The gate is an LLM call, not a similarity threshold, because no threshold
    # separates the two: in-domain and out-of-domain top-1 cosines overlap, and
    # e5 compresses everything into a narrow band (see
    # scripts/domain_gate/calibrate_domain_gate.py). It adds one short call and
    # a terminal state.
    enable_domain_gate: bool = False
    # Answer greetings, pings and questions about the assistant itself
    # ("ciao", "chi sei?", "prova, sistema operativo?") before retrieval, with
    # a fixed introduction plus `example_questions`. They have nothing to
    # retrieve. Off in measurement runs, which must reach retrieval for every
    # question in their set.
    answer_meta_questions: bool = False
    # The questions offered on a refusal and in that introduction. The engine
    # never invents them: the corpus grows and its owner decides what is worth
    # asking, so `product/config.py` (DEMO_EXAMPLE_QUESTIONS) supplies them.
    example_questions: tuple[str, ...] = ()
    # What the gate lets through. It errs towards accepting: a false refusal
    # stonewalls a legitimate question, while a false accept still reaches the
    # answer path and is marked ungrounded.
    domain_scope: str = ""
    # Predicates that carry no answerable content, such as RELATED_TO and the
    # bibliographic ones. Dropping them frees context budget for triples that
    # can support a claim. An empty tuple keeps every predicate.
    drop_predicates: tuple[str, ...] = ()
    # Check that the anchor matches a node before expanding neighbours, the
    # subgraph and the shortest path. Those three channels each scan the graph
    # when the anchor matches nothing, which is slow and returns no evidence.
    # It changes latency, never the evidence, since a seed that matches no node
    # cannot produce any.
    verify_anchor_exists: bool = True
    # How many anchors the subgraph channel expands from, each with a share of
    # the triple budget. Anchoring on retrieved nodes makes every seed accurate
    # but the 2-hop neighbourhood of a single seed narrow; a few anchors restore
    # breadth without reverting to question-word seeds.
    subgraph_seed_count: int = 1
    # Ask for an answer to exactly what was asked, leaving out related material
    # the evidence happens to carry. A richer context otherwise yields a more
    # discursive answer that names entities belonging to other questions.
    focused_answer: bool = False

    def __post_init__(self) -> None:
        """Warn when triple ranking is on and its weights do not sum to 1."""
        if self.rank_triples:
            weight_sum = (
                self.ranker_weight_lexical
                + self.ranker_weight_mention
                + self.ranker_weight_confidence
            )
            if abs(weight_sum - 1.0) > 0.01:
                logger.warning(
                    "Ranker weights sum to %.3f instead of 1.0 "
                    "(lexical=%.2f mention=%.2f confidence=%.2f): triple scores "
                    "will not be comparable across configurations",
                    weight_sum,
                    self.ranker_weight_lexical,
                    self.ranker_weight_mention,
                    self.ranker_weight_confidence,
                )


@dataclass(slots=True)
class KGConfig:
    """Neo4j connection settings of the retrieval side.

    Attributes:
        url: Bolt or Neo4j URI.
        username: User name.
        password: Password.
        database: Database name, or ``None`` for the server default.
        node_name_properties: Node properties that hold a node's name, in
            order: all are compared when matching, the first set one is
            displayed.
        default_limit: Row limit used when a query does not set one.
    """

    url: str
    username: str
    password: str
    database: str | None = None
    node_name_properties: tuple[str, ...] = (
        "name",
        "title",
        "label",
        "id",
        "uuid",
        "entity",
    )
    default_limit: int = 50


def build_kg_config_from_env(
    url_env: str = "NEO4J_URL",
    username_env: str = "NEO4J_USERNAME",
    password_env: str = "NEO4J_PASSWORD",
    database_env: str = "NEO4J_DATABASE",
) -> KGConfig:
    """Build a ``KGConfig`` from environment variables.

    Args:
        url_env: Variable holding the URI.
        username_env: Variable holding the user name.
        password_env: Variable holding the password.
        database_env: Variable holding the database name (optional).

    Returns:
        The configuration.

    Raises:
        ValueError: If the URI, user name or password variable is unset.
    """
    url = os.getenv(url_env)
    username = os.getenv(username_env)
    password = os.getenv(password_env)
    database = os.getenv(database_env)

    missing = [
        key
        for key, value in (
            (url_env, url),
            (username_env, username),
            (password_env, password),
        )
        if not value
    ]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {missing_csv}")

    return KGConfig(
        url=url,
        username=username,
        password=password,
        database=database,
    )
