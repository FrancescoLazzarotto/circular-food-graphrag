"""Shared configuration for the two interactive demos.

``product/app.py`` (Streamlit) and ``product/console.py`` (console) are the
same product, so the agent they run is configured here, once: an improvement
reaches both demos at once and the two cannot answer the same question
differently.

Every setting is an environment variable (``DEMO_*``) with the demo's default
value.

Nothing here is imported by the CLI, the experiment runners or the evaluation
scripts: experiment configuration stays in ``graphrag.config`` and
``graphrag.strategies``. ``src/graphrag`` is the engine, ``product/`` is how it
is presented.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from graphrag.config import (
    AgentConfig,
    KGConfig,
    OUTPUT_COMPLEXITY,
    build_kg_config_from_env,
)
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.strategies import apply_strategy

logger = logging.getLogger("graphrag")

ROOT = Path(__file__).resolve().parents[1]


def _flag(name: str, default: str = "1") -> bool:
    """Whether environment variable ``name`` (default ``default``) equals ``"1"``."""
    return os.environ.get(name, default).strip() == "1"


STRATEGY = os.environ.get("DEMO_STRATEGY", "hybrid")
MAX_CONTEXT_TOKENS = int(os.environ.get("DEMO_MAX_CONTEXT_TOKENS", "6000"))
# A detailed answer with figures, names and per-claim references needs room;
# a small cap only fits a generic summary.
MAX_NEW_TOKENS = int(os.environ.get("DEMO_MAX_NEW_TOKENS", "2048"))
# HIGH drops the "1-2 short paragraphs" instruction and adds the specificity
# rule. The answer language is pinned to the question language.
COMPLEXITY = OUTPUT_COMPLEXITY(os.environ.get("DEMO_COMPLEXITY", "high"))
ENFORCE_LANGUAGE = _flag("DEMO_ENFORCE_LANGUAGE")
# Show the full model answer (including 'Verifica nel grafo'); ask the prompt
# for a 'Limits and confidence' section on every answer, not only sparse ones.
SHOW_FULL_ANSWER = _flag("DEMO_SHOW_FULL_ANSWER")
ALWAYS_LIMITS = _flag("DEMO_ALWAYS_LIMITS")
# Numbered evidence in the context, [S1]/[T1] tags on specific claims, and a
# source list built from what the model actually cited, instead of the
# 'Verifica nel grafo' block listing the top triples regardless of use.
CITE_EVIDENCE = _flag("DEMO_CITE_EVIDENCE")
CITATION_POLICY = os.environ.get("DEMO_CITATION_POLICY", "mark")
# "label" shows "[SEeD for Change, p. 3]" instead of "[S1]": ids mean nothing
# to a reader, a document and a page can be checked.
CITATION_DISPLAY = os.environ.get("DEMO_CITATION_DISPLAY", "label")
# Intra-session conversational memory. A follow-up ("mi indichi le strategie
# nel settore vino") often takes its subject from the previous answer; without
# memory the question reaches retrieval isolated. Steers retrieval only —
# never a source of facts. Demo-only: every other entry point passes no memory.
MEMORY = _flag("DEMO_MEMORY")
# On a definitional question the chunk carrying the verbatim definition is
# ranked first and the answer opens with it between guillemets, instead of
# describing the term only through graph triples.
VERBATIM_DEFINITIONS = _flag("DEMO_VERBATIM_DEFINITIONS")
# MMR plus a per-document cap, so one PDF does not fill the whole context. The
# larger top_k pays for the cap: without it, diversification buys breadth by
# giving up depth on the document that actually answers.
TEXT_TOP_K = int(os.environ.get("DEMO_TEXT_TOP_K", "8"))
TEXT_MMR = _flag("DEMO_TEXT_MMR")
TEXT_MMR_LAMBDA = float(os.environ.get("DEMO_TEXT_MMR_LAMBDA", "0.7"))
TEXT_MAX_PER_DOC = int(os.environ.get("DEMO_TEXT_MAX_PER_DOC", "2"))
TEXT_RETRIEVER_BACKEND = os.environ.get("DEMO_TEXT_RETRIEVER_BACKEND", "dense")
# Named once so the pipeline that gets built (`build_text_pipeline`) and the
# config that gets recorded (`build_agent_config`) cannot drift apart.
DENSE_EMBEDDING_MODEL = os.environ.get(
    "DEMO_DENSE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
)
# Two layers over the same failure, because it has two causes that look alike.
# An out-of-domain question is refused outright by the gate (no retrieval, no
# answer). An in-domain question whose retrieval came back weak is answered,
# with everything the evidence does not support marked '(not in the retrieved
# evidence)'. A single hard gate for both would stonewall legitimate
# questions.
DOMAIN_GATE = _flag("DEMO_DOMAIN_GATE")
# The third layer, before either of those two: a greeting or a question about
# the assistant ("ciao", "chi sei?", "prova, sistema operativo?") is answered
# with an introduction and the example questions, not sent to retrieval. Both
# other layers assume a subject to look up; these have none.
META_REPLY = _flag("DEMO_META_REPLY")
PARAMETRIC_FALLBACK = _flag("DEMO_PARAMETRIC_FALLBACK")
# The cross-lingual half of retrieval: the graph is largely Italian, the
# questions arrive in both languages. Needs the :NodeVec carriers built by
# scripts/kg/kg_vector_index.py.
VECTOR_RETRIEVAL = _flag("DEMO_VECTOR_RETRIEVAL")
# ...and what to do when that encoder is unreachable. The engine raises by
# default, which is right for an experiment: a run that silently changes
# retrieval method halfway is worse than a run that stops. Here it would make
# every question fail, including those the graph can still answer lexically
# and those answered from the text channel, which does not use the encoder.
# Degrading is only acceptable because the UI says so on the affected answer;
# without that caption this line would trade a loud failure for a quiet loss
# of quality. `setdefault`, so an operator can still export 0.
os.environ.setdefault("GRAPHRAG_VECTOR_ALLOW_DEGRADED", "1")
# Stage0 runs feeding the text index, most authoritative first. Explicit on
# purpose: the newest run can be a partial repair run, and older runs in the
# same artifacts folder can hold a different corpus that must stay out.
TEXT_STAGE0_RUNS = os.environ.get(
    "DEMO_TEXT_STAGE0_RUNS",
    "run_fix2docs_20260710,run_full_circular_20260707",
)
# ---------------------------------------------------------------------- #
# presentation
# ---------------------------------------------------------------------- #
# The name and the line under it. Settings rather than literals in the page,
# because naming the product is the owner's decision and it must be changeable
# without editing the interface.
PRODUCT_NAME = os.environ.get("DEMO_PRODUCT_NAME", "Assistente AI - CEFF")
PRODUCT_TAGLINE = os.environ.get(
    "DEMO_PRODUCT_TAGLINE",
    "Risponde sull'economia circolare del cibo citando i documenti da cui prende "
    "ogni affermazione.",
)
# The tagline is a setting, so it does not go through the interface dictionary;
# the English one is a setting too, or the language switch would leave an
# Italian sentence under an English page.
PRODUCT_TAGLINE_EN = os.environ.get(
    "DEMO_PRODUCT_TAGLINE_EN",
    "Answers on the circular economy of food, citing the documents every claim "
    "is taken from.",
)
PRODUCT_ICON = os.environ.get("DEMO_PRODUCT_ICON", "\U0001F33E")
# How the citations the engine renders into the prose are set on screen.
# "dim" puts them in small grey italics between parentheses and shortens the
# document to DEMO_CITATION_DOC_CHARS; "plain" leaves the engine's own
# "[Document, p. 12]" untouched. Presentation only — the stored answer, and so
# everything copied or exported, keeps the full label either way.
CITATION_STYLE = os.environ.get("DEMO_CITATION_STYLE", "dim")
CITATION_DOC_CHARS = int(os.environ.get("DEMO_CITATION_DOC_CHARS", "60"))
# Interface language. Independent of the answer language, which the engine pins
# to the language of the question.
UI_LANGUAGE = os.environ.get("DEMO_UI_LANGUAGE", "it")
# The answer language when the question gives no clue: a bare term such as
# "scotta" or "vinacce e raspi" has no word that marks either language, and the
# engine's own fallback is English. The readers of this demo write in Italian.
FALLBACK_LANGUAGE = os.environ.get("DEMO_FALLBACK_LANGUAGE", UI_LANGUAGE)
# Serving detail on screen: the model id, the strategy and the graph URL. Off by
# default — the graph label names the hosted instance, and a reader of the page
# is not the audience for a connection string.
DEBUG = _flag("DEMO_DEBUG", "0")
# Offered when a question is refused as out of domain, so the refusal points
# somewhere instead of ending the session. Configuration, not a literal in the
# page: the corpus grows, and the examples have to be able to grow with it
# without a code change.
EXAMPLE_QUESTIONS = tuple(
    q.strip()
    for q in os.environ.get(
        "DEMO_EXAMPLE_QUESTIONS",
        "Che cos'è l'economia circolare applicata al cibo?"
        "|Quali sottoprodotti agroalimentari possono essere valorizzati, e come?"
        "|Che cosa dicono i documenti sul recupero degli scarti in una filiera?",
    ).split("|")
    if q.strip()
)

ENV_FILE = os.environ.get("DEMO_ENV_FILE", str(ROOT / "kg_pipeline" / ".env"))
LOG_DIR = Path(os.environ.get("DEMO_LOG_DIR", str(ROOT / "artifacts" / "demo_sessions")))
# Comma-separated vLLM endpoints offered in the model selector; each is probed
# at startup and skipped when unreachable, so a stopped server just disappears
# from the list instead of breaking the demo.
VLLM_ENDPOINTS = os.environ.get(
    "DEMO_VLLM_ENDPOINTS",
    "http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8003/v1",
)


# ---------------------------------------------------------------------- #
# graph connection
# ---------------------------------------------------------------------- #


def _probe_kg(config: KGConfig) -> KnowledgeGraphManager:
    """Open a connection and prove it answers, or raise."""
    manager = KnowledgeGraphManager(config)
    # Straight through the driver, not through run_query: that one retries
    # three times with backoff, which is right mid-session and wrong here,
    # where the point is to find out quickly whether to use the other graph.
    manager.graph.query("RETURN 1 AS ok")
    return manager


def build_kg_manager() -> tuple[KnowledgeGraphManager, str]:
    """Connect to the primary graph, fall back to the secondary one.

    The primary graph is an Aura Free instance, which suspends itself after
    three idle days and then resolves to nothing at all; the same graph is also
    mirrored locally. An unreachable primary moves to the fallback
    (``DEMO_NEO4J_FALLBACK_URL`` and its siblings) instead of failing the
    demo.

    Returns:
        The connected manager and a label naming which graph answered.

    Raises:
        RuntimeError: Neither graph could be reached.
    """
    # Idempotent and never overrides an exported variable, so a caller that
    # already loaded the file (or set NEO4J_URL inline) keeps its choice.
    load_dotenv(ENV_FILE, override=False)

    primary_error: Exception | None = None
    try:
        primary = build_kg_config_from_env()
        return _probe_kg(primary), f"primario ({primary.url})"
    except Exception as exc:  # noqa: BLE001 - any failure means "try the other one"
        primary_error = exc
        logger.warning("Primary graph unreachable (%s); trying the fallback.", exc)

    fallback_url = os.environ.get("DEMO_NEO4J_FALLBACK_URL", "").strip()
    if not fallback_url:
        raise RuntimeError(
            f"Grafo primario non raggiungibile ({primary_error}) e nessun "
            "fallback configurato: imposta DEMO_NEO4J_FALLBACK_URL / "
            "_USERNAME / _PASSWORD / _DATABASE."
        )
    fallback = build_kg_config_from_env(
        url_env="DEMO_NEO4J_FALLBACK_URL",
        username_env="DEMO_NEO4J_FALLBACK_USERNAME",
        password_env="DEMO_NEO4J_FALLBACK_PASSWORD",
        database_env="DEMO_NEO4J_FALLBACK_DATABASE",
    )
    # An unset database is not "no database": the driver then reads NEO4J_DATABASE
    # itself, so the fallback would inherit the hosted database name and fail
    # with DatabaseNotFound against a local instance that only has "neo4j".
    if not fallback.database:
        fallback.database = "neo4j"
    try:
        manager = _probe_kg(fallback)
    except Exception as exc:  # noqa: BLE001 - report both failures, not the last
        raise RuntimeError(
            f"Nessun grafo raggiungibile. Primario: {primary_error}. "
            f"Fallback ({fallback_url}): {exc}."
        ) from exc
    logger.warning("Using the fallback graph at %s.", fallback.url)
    return manager, f"fallback ({fallback.url})"


# ---------------------------------------------------------------------- #
# model selection
# ---------------------------------------------------------------------- #


def probe_vllm_endpoints(timeout_sec: float = 3.0) -> dict[str, tuple[str, str]]:
    """Map "model (:port)" -> (base_url, model_id) for every endpoint that answers.

    Falls back to VLLM_BASE_URL/VLLM_MODEL_NAME when no endpoint answers, so the
    demo keeps working in single-server setups without the selector env var.

    Args:
        timeout_sec: Timeout of each ``/models`` probe.

    Returns:
        Selector label -> ``(base_url, model_id)``; empty when nothing answers
        and the fallback variables are unset.
    """
    options: dict[str, tuple[str, str]] = {}
    for base_url in (u.strip().rstrip("/") for u in VLLM_ENDPOINTS.split(",") if u.strip()):
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=timeout_sec) as resp:
                model_id = json.load(resp)["data"][0]["id"]
        except (urllib.error.URLError, OSError, KeyError, IndexError, json.JSONDecodeError):
            continue
        port = urllib.parse.urlparse(base_url).port or "?"
        options[f"{model_id.split('/')[-1]} (:{port})"] = (base_url, model_id)
    if not options:
        model_id = os.environ.get("VLLM_MODEL_NAME", "")
        base_url = os.environ.get("VLLM_BASE_URL", "")
        if model_id and base_url:
            options[model_id.split("/")[-1]] = (base_url, model_id)
    return options


# ---------------------------------------------------------------------- #
# agent
# ---------------------------------------------------------------------- #


def build_text_pipeline(backend: str = TEXT_RETRIEVER_BACKEND) -> object | None:
    """Index the corpus from ``TEXT_STAGE0_RUNS``, reusing the CLI's builder.

    Args:
        backend: ``"tfidf"`` or ``"dense"``.

    Returns:
        The indexed text pipeline, or ``None`` when there is nothing to index.
    """
    import argparse

    from graphrag import cli as graphrag_cli

    ns = argparse.Namespace(
        text_retriever_backend=backend,
        dense_embedding_model=DENSE_EMBEDDING_MODEL,
        vector_index_dir=str(ROOT / "artifacts" / "vector_index"),
        text_docs_dir="",
        text_stage0_runs=TEXT_STAGE0_RUNS,
    )
    return graphrag_cli._build_text_pipeline(ns)


def build_agent_config(strategy: str = STRATEGY) -> AgentConfig:
    """Build the demo's agent configuration.

    Args:
        strategy: Retrieval-strategy preset applied on top of the settings.

    Returns:
        The configuration.
    """
    base = AgentConfig(
        max_content_tokens=MAX_CONTEXT_TOKENS,
        always_include_limits=ALWAYS_LIMITS,
        cite_evidence=CITE_EVIDENCE,
        citation_policy=CITATION_POLICY,
        citation_display=CITATION_DISPLAY,
        complexity=COMPLEXITY,
        enforce_language=ENFORCE_LANGUAGE,
        fallback_language=FALLBACK_LANGUAGE,
        prefer_verbatim_definitions=VERBATIM_DEFINITIONS,
        text_retriever_top_k=TEXT_TOP_K,
        text_retriever_mmr=TEXT_MMR,
        text_retriever_mmr_lambda=TEXT_MMR_LAMBDA,
        text_retriever_max_per_doc=TEXT_MAX_PER_DOC,
        enable_domain_gate=DOMAIN_GATE,
        answer_meta_questions=META_REPLY,
        example_questions=EXAMPLE_QUESTIONS,
        allow_parametric_fallback=PARAMETRIC_FALLBACK,
        vector_retrieval=VECTOR_RETRIEVAL,
        # Copied so the recorded config names the text retriever that
        # `build_text_pipeline` actually builds, not the library default.
        text_retriever_backend=TEXT_RETRIEVER_BACKEND,
        dense_embedding_model=DENSE_EMBEDDING_MODEL,
        vector_index_dir=str(ROOT / "artifacts" / "vector_index"),
    )
    return apply_strategy(base, strategy)


def build_demo_agent(
    base_url: str,
    model_id: str,
    strategy: str = STRATEGY,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[object, str]:
    """Build the agent both demos run, and say which graph it is talking to.

    Args:
        base_url: vLLM endpoint.
        model_id: Served model name.
        strategy: Retrieval-strategy preset.
        max_new_tokens: Generation budget per answer.

    Returns:
        The ``KGRAGAgent`` and the label of the graph it connected to.

    Raises:
        RuntimeError: If no graph can be reached.
    """
    from graphrag.agent.core import KGRAGAgent

    kg_manager, graph_label = build_kg_manager()
    config = build_agent_config(strategy)

    text_pipeline = build_text_pipeline() if config.use_text_retriever else None
    retriever = KGRetriever(
        kg_store=kg_manager, config=config, text_pipeline=text_pipeline
    )
    llm = LLMManager(
        model_id=model_id,
        warmup=False,
        max_new_tokens=max_new_tokens,
        use_vllm=True,
        vllm_base_url=base_url,
    )
    return KGRAGAgent(config=config, kg_retriever=retriever, llm=llm), graph_label


# ---------------------------------------------------------------------- #
# corpus manifest
# ---------------------------------------------------------------------- #


TITLE_OVERRIDES_FILE = Path(
    os.environ.get("DEMO_TITLE_OVERRIDES", str(ROOT / "product" / "corpus_titles.json"))
)

# Markdown the title extractor carried over from the page it read it off.
_TITLE_NOISE = re.compile(r"[*_#`]+|\[[†*]\]")
# A title that is really a masthead, a keyword list or a caption. These reach
# the page as the source of a claim, where they are worse than a filename.
_TITLE_REJECT = re.compile(
    r"^(key ?words?\b|hanno contribuito|scientific board|©)", re.IGNORECASE
)


def _clean_title(raw: object) -> str:
    """Strip the page's own markup off a recorded title, or reject it.

    Returns:
        The title, or an empty string when there is nothing usable — too short,
        too long to be a title, or one of the recurring non-titles.
    """
    text = " ".join(_TITLE_NOISE.sub("", str(raw or "")).split())
    if not 10 <= len(text) <= 200 or _TITLE_REJECT.match(text):
        return ""
    return text


def document_titles() -> dict[str, str]:
    """What each document is called, by filename.

    A citation naming "REPORT MATTM_Definitivo.pdf" names the file someone
    happened to save; the reader wants the work. The pipeline already records a
    title per document, and for most of the corpus it is the right one — where
    it picked up a masthead instead, `corpus_titles.json` overrides it. A
    document with neither keeps its filename, so the corpus can grow without
    anyone editing anything.

    Returns:
        ``{filename: title}``, holding only the documents that have one.
    """
    titles: dict[str, str] = {}
    artifacts = ROOT / "kg_pipeline" / "artifacts"
    for run in (r.strip() for r in TEXT_STAGE0_RUNS.split(",") if r.strip()):
        manifest = artifacts / run / "stage0_documents.json"
        if not manifest.exists():
            continue
        try:
            docs = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # noqa: BLE001 - a bad manifest is not fatal
            logger.warning("Corpus manifest %s unreadable: %s", manifest, exc)
            continue
        if not isinstance(docs, list):
            continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            filename = str(doc.get("filename", "") or "").strip()
            title = _clean_title(doc.get("title"))
            if filename and title and filename not in titles:
                titles[filename] = title

    try:
        overrides = json.loads(TITLE_OVERRIDES_FILE.read_text(encoding="utf-8"))
        titles.update(
            {
                str(name): str(title).strip()
                for name, title in (overrides.get("titles") or {}).items()
                if str(title).strip()
            }
        )
    except (OSError, ValueError) as exc:  # noqa: BLE001 - the manifest still stands
        logger.warning("Title overrides %s unreadable: %s", TITLE_OVERRIDES_FILE, exc)

    return titles


def corpus_manifest() -> dict[str, object]:
    """What the collection currently holds, read from the indexed stage0 runs.

    The interface has to be able to say how much it has read without a number
    typed into a sentence: the corpus grows, and a hard-coded count silently
    becomes a lie. The source of truth is the same manifest the text channel is
    built from (`TEXT_STAGE0_RUNS`), so the page can never claim documents the
    retriever cannot reach.

    The manifest also carries ``publication_year``, and it is deliberately not
    returned: the extracted years are not reliable enough to show a reader a
    date range.

    Returns:
        ``documents`` (filenames, most authoritative run first, no repeats) and
        their ``count``. Both are empty when no manifest can be read — the
        caller then says nothing about the corpus rather than guessing.
    """
    documents: list[str] = []
    artifacts = ROOT / "kg_pipeline" / "artifacts"

    for run in (r.strip() for r in TEXT_STAGE0_RUNS.split(",") if r.strip()):
        manifest = artifacts / run / "stage0_documents.json"
        if not manifest.exists():
            continue
        try:
            docs = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # noqa: BLE001 - a bad manifest is not fatal
            logger.warning("Corpus manifest %s unreadable: %s", manifest, exc)
            continue
        if not isinstance(docs, list):
            continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            filename = str(doc.get("filename", "") or "").strip()
            # Runs are listed most authoritative first, exactly as the text
            # pipeline reads them, so a reprocessed document is not counted twice.
            if not filename or filename in documents:
                continue
            documents.append(filename)

    return {"documents": documents, "count": len(documents)}
