"""Command line: answer one question, or run a batch experiment over strategies."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import enum
import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from graphrag.agent.core import KGRAGAgent
from graphrag.config import (
    AgentConfig,
    DEFAULT_MODEL_ID,
    OUTPUT_COMPLEXITY,
    build_kg_config_from_env,
)
from graphrag.experiments import ExperimentRunner, Question
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.profiles import PROFILES
from graphrag.strategies import apply_strategy
from graphrag.text_rag.factory import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_CHUNK_CHARS,
    make_text_pipeline,
)
from graphrag.text_rag.pipeline import StandardTextRAGPipeline

logger = logging.getLogger("graphrag.cli")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser; every flag's help text lives here."""
    parser = argparse.ArgumentParser(description="Run GraphRAG demo pipeline")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=None,
        help=(
            "Start from a named configuration profile (graphrag.profiles). "
            "Any flag given explicitly still wins over the profile."
        ),
    )
    parser.add_argument(
        "--question",
        default="Quali sono gli obiettivi della strategia Farm to Fork?",
    )
    parser.add_argument(
        "--entity",
        default="",
        help="Optional entity seed for graph traversal (leave empty for auto-seeding)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--llm", action="store_true", help="Enable LLM generation")
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use a vLLM OpenAI-compatible endpoint instead of local Hugging Face loading",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible API",
    )
    parser.add_argument(
        "--llm-warmup", action="store_true", help="Preload model at startup"
    )
    parser.add_argument(
        "--enable-decomposition-step",
        action="store_true",
        help="Enable LLM decomposition step before retrieval",
    )
    parser.add_argument(
        "--enable-adaptive-routing-step",
        action="store_true",
        help="Enable LLM adaptive routing step before retrieval",
    )
    parser.add_argument(
        "--cite-evidence",
        action="store_true",
        help=(
            "Number retrieved evidence, ask the model for [S1]/[T1] tags on "
            "specific claims, and verify every tag against the index"
        ),
    )
    parser.add_argument(
        "--citation-policy",
        choices=("mark", "strip"),
        default="mark",
        help="What to do with a reference tag absent from the evidence index",
    )
    parser.add_argument(
        "--citation-display",
        choices=("id", "label"),
        default="id",
        help=(
            "How verified references are shown: 'id' keeps [S1]/[T3], 'label' "
            "rewrites them as [document, p. N] after the citation gate"
        ),
    )
    parser.add_argument(
        "--prefer-verbatim-definitions",
        action="store_true",
        help=(
            "On a definitional question, rank the passage that defines the term "
            "first and open the answer with it, quoted between guillemets"
        ),
    )
    parser.add_argument(
        "--vector-retrieval",
        action="store_true",
        help=(
            "Add the multilingual vector channel beside the lexical one, so an "
            "English question can reach an Italian node name. Needs "
            "scripts/kg/kg_vector_index.py and a running embedding endpoint"
        ),
    )
    parser.add_argument(
        "--vector-index",
        default="node_embedding",
        help="Neo4j vector index name",
    )
    parser.add_argument(
        "--vector-nodes-limit",
        type=int,
        default=10,
        help="Nodes taken from the vector channel per query",
    )
    parser.add_argument(
        "--vector-triples-limit",
        type=int,
        default=10,
        help="Triples taken from the vector channel per query",
    )
    parser.add_argument(
        "--seed-from-retrieved",
        action="store_true",
        help=(
            "Anchor the neighbour, subgraph and shortest-path channels on node "
            "names retrieval returned instead of on raw question words"
        ),
    )
    parser.add_argument(
        "--allow-parametric-fallback",
        action="store_true",
        help=(
            "Let the answer use the model's own knowledge where the retrieved "
            "evidence does not cover the question, marked '(not in the "
            "retrieved evidence)' so grounded and ungrounded stay separable"
        ),
    )
    parser.add_argument(
        "--focused-answer",
        action="store_true",
        help=(
            "Ask the model to name only what answers the question, not every "
            "related concept the retrieved evidence happens to carry"
        ),
    )
    parser.add_argument(
        "--enable-domain-gate",
        action="store_true",
        help=(
            "Classify the question against the collection's scope before "
            "retrieving, and refuse out-of-domain questions instead of "
            "answering them from an unrelated context"
        ),
    )
    parser.add_argument(
        "--legacy-insufficiency-wording",
        action="store_true",
        help=(
            "Restore the pre-repair closing line of the answer prompt, which "
            "allowed a declaration of insufficiency only for an empty context; "
            "for reproducing the thesis campaigns E1-E8"
        ),
    )
    parser.add_argument(
        "--subgraph-seed-count",
        type=int,
        default=1,
        help="How many anchors the subgraph channel expands from (1 = best only)",
    )
    parser.add_argument(
        "--subgraph-limit",
        type=int,
        default=200,
        help=(
            "Triples the subgraph channel may pull per anchor set before ranking. "
            "The cap is applied in Cypher, so it truncates the neighbourhood in "
            "graph order and the ranker never sees what was cut. On a sparse "
            "graph the cap never binds; on a dense one it decides the answer"
        ),
    )
    parser.add_argument(
        "--evidence-max-triple-items",
        type=int,
        default=30,
        help="Cap on numbered triple evidence blocks placed in the context",
    )
    parser.add_argument(
        "--drop-predicates",
        default="",
        help=(
            "Comma-separated predicates to remove from retrieved triples, e.g. "
            "RELATED_TO,PUBLISHED,AUTHORED_BY"
        ),
    )
    parser.add_argument(
        "--text-retriever-mmr",
        action="store_true",
        help=(
            "Select text chunks with Maximal Marginal Relevance instead of pure "
            "top-k, trading some query similarity for source coverage"
        ),
    )
    parser.add_argument(
        "--text-retriever-mmr-lambda",
        type=float,
        default=0.7,
        help="MMR relevance/diversity balance: 1.0 is pure similarity",
    )
    parser.add_argument(
        "--text-retriever-max-per-doc",
        type=int,
        default=0,
        help=(
            "Cap on chunks contributed by one document (0 disables it); "
            "enumerative questions get twice this budget"
        ),
    )
    parser.add_argument(
        "--complexity",
        choices=("low", "medium", "high"),
        default="medium",
        help=(
            "Answer depth. 'high' drops the '1-2 short paragraphs' cap and adds "
            "the specificity rule (figures, names, article numbers)"
        ),
    )
    parser.add_argument(
        "--enforce-language",
        action="store_true",
        help=(
            "Constrain the answer to the language detected on the question, "
            "written in that language, and retry once on a mismatch"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per response",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=6000,
        help="Maximum tokens for compressed context before generation",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="Maximum LangGraph recursion steps before aborting",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.92,
        help="Fraction of each GPU memory reserved for model placement (0,1]",
    )
    parser.add_argument(
        "--allow-large-model-fp16-fallback",
        action="store_true",
        help="Allow fp16 fallback for large models when 4-bit quantized loading fails",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run batch experiments and persist outputs",
    )
    parser.add_argument(
        "--questions-file",
        help="Questions to run. .txt: one question per line (optionally "
             "'Q01<TAB>question' to carry the gold id); .json: the gold's "
             "{'queries': [{'query_id','query'}]} shape; .jsonl: one such object "
             "per line; .csv: query_id + query columns. Declaring ids lets the "
             "evaluator join results to the gold by query_id instead of by text.",
    )
    parser.add_argument(
        "--strategies", default="default", help="Comma-separated strategy presets"
    )
    parser.add_argument("--runs-per-strategy", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument(
        "--text-docs-dir",
        default="",
        help="Directory of documents (PDF/txt/md) to index for text_only standard RAG. "
             "If omitted, auto-discovers from the latest KG pipeline stage0 artifacts.",
    )
    parser.add_argument(
        "--text-stage0-runs",
        default=os.environ.get("GRAPHRAG_TEXT_STAGE0_RUNS", ""),
        help="Comma-separated kg_pipeline/artifacts run names feeding the text index, "
             "most authoritative first (later runs cannot shadow earlier ones on the "
             "same filename). Defaults to every run, newest first.",
    )
    parser.add_argument(
        "--text-retriever-backend",
        default="tfidf",
        choices=("tfidf", "dense"),
        help="Retrieval backend for standard RAG: 'tfidf' (lexical, default) or 'dense' (cosine/FAISS).",
    )
    parser.add_argument(
        "--dense-embedding-model",
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model ID for dense retrieval (ignored for tfidf).",
    )
    parser.add_argument(
        "--vector-index-dir",
        default="artifacts/vector_index",
        help="Directory for persisted FAISS index cache (ignored for tfidf).",
    )
    return parser


def _build_llm_manager(args: argparse.Namespace, warmup: bool) -> LLMManager | None:
    """Create the LLM backend from the CLI flags.

    Args:
        args: Parsed CLI namespace.
        warmup: Load the model immediately.

    Returns:
        The manager, or ``None`` without ``--llm``.
    """
    if not args.llm:
        return None

    return LLMManager(
        model_id=args.model_id,
        warmup=warmup,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_fraction=args.gpu_memory_fraction,
        allow_large_model_fp16_fallback=args.allow_large_model_fp16_fallback,
        use_vllm=args.vllm,
        vllm_base_url=args.vllm_base_url,
    )


# AgentConfig fields whose CLI flag carries a different name. The parser writes
# `--max-context-tokens` into `max_context_tokens`, which `_build_base_config`
# then assigns to `max_content_tokens` -- content, not context. Everything else
# maps onto itself.
_PROFILE_FIELD_TO_ARG = {"max_content_tokens": "max_context_tokens"}


def _profile_defaults(
    parser: argparse.ArgumentParser,
    profile: str,
    known_dests: set[str],
) -> dict[str, object]:
    """Translate a profile into parser defaults, keyed by argparse dest.

    Exits through ``parser.error`` if the profile carries a field with no
    corresponding flag: silently dropping it would hand the caller a
    configuration that is not the profile they asked for.

    Args:
        parser: The CLI parser.
        profile: Profile name from :data:`graphrag.profiles.PROFILES`.
        known_dests: Every argparse dest the parser defines.

    Returns:
        Defaults to install with ``parser.set_defaults``; enum values are
        given as their string value.
    """
    defaults: dict[str, object] = {}
    unreachable: list[str] = []
    for field, value in PROFILES[profile].items():
        dest = _PROFILE_FIELD_TO_ARG.get(field, field)
        if dest not in known_dests:
            unreachable.append(field)
            continue
        defaults[dest] = value.value if isinstance(value, enum.Enum) else value
    if unreachable:
        parser.error(
            f"profile '{profile}' sets {', '.join(sorted(unreachable))}, which "
            "the CLI has no flag for; use it through graphrag.profiles instead"
        )
    return defaults


def _parse_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    """Parse the command line, letting ``--profile`` supply the defaults.

    Precedence is argparse's own. The profile is installed with
    ``set_defaults`` and the same argv is parsed a second time, so a flag the
    caller typed beats the profile without this function having to work out
    which flags were typed -- a distinction argparse does not otherwise expose.

    Args:
        parser: The parser from :func:`_build_arg_parser`.
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed namespace. Without ``--profile`` it is exactly what a single
        ``parse_args`` would have returned.
    """
    args = parser.parse_args(argv)
    profile = getattr(args, "profile", None)
    if not profile:
        return args
    parser.set_defaults(**_profile_defaults(parser, profile, set(vars(args))))
    return parser.parse_args(argv)


def _build_base_config(args: argparse.Namespace) -> AgentConfig:
    """Build the agent configuration the strategies are applied to.

    Every retrieval channel is on; strategies switch channels off.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The base configuration.
    """
    return AgentConfig(
        query=args.question,
        entity=args.entity,
        include_nodes=True,
        include_triples=True,
        include_neighbors=True,
        include_subgraph=True,
        include_shortest_path=True,
        llm_warmup=args.llm_warmup,
        enable_decomposition_step=args.enable_decomposition_step,
        enable_adaptive_routing_step=args.enable_adaptive_routing_step,
        recursion_limit=args.recursion_limit,
        max_content_tokens=args.max_context_tokens,
        cite_evidence=getattr(args, "cite_evidence", False),
        citation_policy=getattr(args, "citation_policy", "mark"),
        citation_display=getattr(args, "citation_display", "id"),
        complexity=OUTPUT_COMPLEXITY(getattr(args, "complexity", "medium")),
        enforce_language=getattr(args, "enforce_language", False),
        prefer_verbatim_definitions=getattr(args, "prefer_verbatim_definitions", False),
        text_retriever_mmr=getattr(args, "text_retriever_mmr", False),
        text_retriever_mmr_lambda=getattr(args, "text_retriever_mmr_lambda", 0.7),
        text_retriever_max_per_doc=getattr(args, "text_retriever_max_per_doc", 0),
        vector_retrieval=getattr(args, "vector_retrieval", False),
        vector_index=getattr(args, "vector_index", "node_embedding"),
        vector_nodes_limit=getattr(args, "vector_nodes_limit", 10),
        vector_triples_limit=getattr(args, "vector_triples_limit", 10),
        seed_from_retrieved=getattr(args, "seed_from_retrieved", False),
        allow_parametric_fallback=getattr(args, "allow_parametric_fallback", False),
        enable_domain_gate=getattr(args, "enable_domain_gate", False),
        legacy_insufficiency_wording=getattr(
            args, "legacy_insufficiency_wording", False
        ),
        focused_answer=getattr(args, "focused_answer", False),
        subgraph_seed_count=getattr(args, "subgraph_seed_count", 1),
        subgraph_limit=getattr(args, "subgraph_limit", 200),
        evidence_max_triple_items=getattr(args, "evidence_max_triple_items", 30),
        drop_predicates=tuple(
            p.strip()
            for p in getattr(args, "drop_predicates", "").split(",")
            if p.strip()
        ),
        # Copied so `config.json` records the text backend that actually ran,
        # not the library default.
        text_retriever_backend=getattr(args, "text_retriever_backend", "tfidf"),
        dense_embedding_model=getattr(
            args, "dense_embedding_model", "intfloat/multilingual-e5-base"
        ),
        vector_index_dir=getattr(args, "vector_index_dir", "artifacts/vector_index"),
    )


def _build_text_pipeline(args: argparse.Namespace) -> StandardTextRAGPipeline | None:
    """Build and index the text pipeline used by text-channel strategies.

    With ``--text-docs-dir`` the directory's files are indexed; otherwise the
    pages of the KG pipeline's ``stage0_documents.json`` are, from the runs
    chosen by :func:`_resolve_stage0_runs`, the first run winning on a
    repeated file name.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The indexed pipeline, or ``None`` when there is nothing to index.
    """
    logger = logging.getLogger("graphrag.cli")
    backend = getattr(args, "text_retriever_backend", "tfidf")
    pipeline = make_text_pipeline(
        backend=backend,
        embedding_model=getattr(args, "dense_embedding_model", "intfloat/multilingual-e5-base"),
        vector_index_dir=getattr(args, "vector_index_dir", "artifacts/vector_index"),
    )

    docs_dir = (args.text_docs_dir or "").strip()
    if docs_dir:
        target = Path(docs_dir)
        if not target.exists():
            logger.warning("--text-docs-dir %s not found; text retrieval disabled", docs_dir)
            return None
        n = pipeline.index_directory(target)
        logger.info("Text pipeline: indexed %d chunks from %s", n, docs_dir)
        return pipeline

    # Build from KG stage0 artifacts. Which run(s) to read matters: the newest
    # run can be a partial repair run holding a few documents, which would
    # leave the text channel blind to the rest of the corpus. See
    # `_resolve_stage0_runs`.
    kg_artifacts = Path("kg_pipeline/artifacts")
    if not kg_artifacts.exists():
        logger.warning("No text documents found; text_only strategy will have empty context")
        return None

    run_dirs = _resolve_stage0_runs(args, kg_artifacts)
    if not run_dirs:
        logger.warning("No text documents found; text_only strategy will have empty context")
        return None

    import json as _json
    from graphrag.text_rag.manager import TextChunk

    chunks: list[TextChunk] = []
    seen_files: set[str] = set()
    indexed_runs: list[str] = []

    for run_dir in run_dirs:
        stage0 = run_dir / "stage0_documents.json"
        if not stage0.exists():
            continue
        try:
            docs = _json.loads(stage0.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load stage0 from %s: %s", run_dir, exc)
            continue
        if not isinstance(docs, list):
            continue

        added = 0
        for doc_idx, doc in enumerate(docs, start=1):
            filename = str(doc.get("filename", f"doc_{doc_idx}"))
            # Runs are listed most authoritative first: a document reprocessed
            # in a later repair run must not be shadowed by its older version.
            if filename in seen_files:
                continue
            doc_chunks = _stage0_document_chunks(doc, doc_idx, filename, TextChunk)
            if not doc_chunks:
                continue
            seen_files.add(filename)
            chunks.extend(doc_chunks)
            added += 1

        if added:
            indexed_runs.append(f"{run_dir.name} (+{added} docs)")

    if not chunks:
        logger.warning("No text documents found; text_only strategy will have empty context")
        return None

    pipeline.retriever.add_chunks(chunks)
    logger.info(
        "Text pipeline: indexed %d chunks from %d documents — %s",
        len(chunks),
        len(seen_files),
        ", ".join(indexed_runs),
    )
    return pipeline


def _resolve_stage0_runs(args: argparse.Namespace, kg_artifacts: Path) -> list[Path]:
    """Pick which stage0 run directories feed the text index.

    Args:
        args: Parsed CLI namespace; ``--text-stage0-runs`` wins when given.
        kg_artifacts: The ``kg_pipeline/artifacts`` directory.

    Returns:
        Run directories, most authoritative first. An explicit selection is
        taken in the given order, skipping names that do not exist; otherwise
        only the most recently modified ``run_*`` directory, with a warning
        naming the others.
    """
    logger = logging.getLogger("graphrag.cli")
    requested = str(getattr(args, "text_stage0_runs", "") or "").strip()
    if requested:
        selected: list[Path] = []
        for name in (part.strip() for part in requested.split(",")):
            if not name:
                continue
            candidate = kg_artifacts / name
            if candidate.is_dir():
                selected.append(candidate)
            else:
                logger.warning("--text-stage0-runs: %s not found, skipped", name)
        return selected

    available = sorted(
        [p for p in kg_artifacts.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not available:
        return []

    # Only the newest run by default. Unioning every run would mix corpora:
    # the artifacts can hold runs over different document sets, and a domain
    # the graph no longer covers would leak into the text channel. Callers who
    # want a union say so.
    if len(available) > 1:
        logger.warning(
            "Text index built from %s only. Other runs are available (%s); pass "
            "--text-stage0-runs to combine them explicitly.",
            available[0].name,
            ", ".join(p.name for p in available[1:6]),
        )
    return available[:1]


def _stage0_document_chunks(
    doc: dict, doc_idx: int, filename: str, chunk_cls: type
) -> list:
    """Chunk one stage0 document, keeping page provenance when available.

    ``page_chunks`` carries ``{page_number, text}`` per page, so chunks can be
    tagged ``<file>#page=N#chunk=M`` — the format the citation layer parses into
    a readable source label. Without it the answer can only name the document,
    never the page. Pages, or the whole text when there are none, are cut
    into overlapping windows of the default chunk size.

    Args:
        doc: One entry of ``stage0_documents.json``.
        doc_idx: 1-based index, used to build unique chunk ids.
        filename: Document file name, used as the source root.
        chunk_cls: The ``TextChunk`` class (imported lazily by the caller).

    Returns:
        The document's chunks, empty when it carries no usable text.
    """
    step = DEFAULT_CHUNK_SIZE - DEFAULT_CHUNK_OVERLAP

    def windows(text: str) -> list[str]:
        """Cut ``text`` into overlapping windows, dropping the too-short ones."""
        out: list[str] = []
        for start in range(0, len(text), step):
            fragment = text[start : start + DEFAULT_CHUNK_SIZE].strip()
            if len(fragment) >= DEFAULT_MIN_CHUNK_CHARS:
                out.append(fragment)
        return out

    chunks: list = []
    page_chunks = doc.get("page_chunks")
    if isinstance(page_chunks, list) and page_chunks:
        for page in page_chunks:
            if not isinstance(page, dict):
                continue
            page_text = str(page.get("text", "") or "").strip()
            if not page_text:
                continue
            page_number = page.get("page_number", "")
            for c_idx, fragment in enumerate(windows(page_text), start=1):
                chunks.append(
                    chunk_cls(
                        chunk_id=f"d{doc_idx:04d}-p{page_number}-c{c_idx:04d}",
                        content=fragment,
                        source=f"{filename}#page={page_number}#chunk={c_idx}",
                    )
                )
        if chunks:
            return chunks

    text = str(doc.get("markdown_text", "") or "").strip()
    if not text:
        return []
    return [
        chunk_cls(
            chunk_id=f"d{doc_idx:04d}-c{c_idx:04d}",
            content=fragment,
            source=filename,
        )
        for c_idx, fragment in enumerate(windows(text), start=1)
    ]


def _question_from_obj(obj: dict, where: str) -> Question:
    """Build a Question from a JSON/JSONL/CSV entry.

    Args:
        obj: Mapping with a question field and an optional id field.
        where: Human-readable location, for error messages.

    Returns:
        The parsed Question.

    Raises:
        ValueError: If no question text field is present.
    """
    text = str(
        obj.get("query") or obj.get("question") or obj.get("text") or ""
    ).strip()
    if not text:
        raise ValueError(f"{where}: entry has no 'query'/'question'/'text' field")
    query_id = str(obj.get("query_id") or obj.get("id") or "").strip()
    return Question(text=text, query_id=query_id)


def _questions_from_text(path: Path) -> list[Question]:
    """Parse the plain-text format: one question per line.

    A line may optionally carry its gold id as ``Q01<TAB>question text``. For a
    line without a TAB the whole line is the question and the run emits no id
    for it.

    Args:
        path: Questions file.

    Returns:
        The questions in file order; blank lines are skipped.
    """
    questions: list[Question] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        query_id = ""
        text = line
        if "\t" in line:
            head, _, tail = line.partition("\t")
            if head.strip() and tail.strip():
                query_id, text = head.strip(), tail.strip()
            else:
                text = line.replace("\t", " ").strip()
        questions.append(Question(text=text, query_id=query_id))
    return questions


def _questions_from_jsonl(path: Path) -> list[Question]:
    """Parse a JSONL questions file: one ``{query_id, query}`` object per line.

    Args:
        path: Questions file.

    Returns:
        The questions in file order.

    Raises:
        ValueError: If a line is not a JSON object with a question field.
    """
    questions: list[Question] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        questions.append(_question_from_obj(obj, f"{path}:{lineno}"))
    return questions


def _questions_from_json(path: Path) -> list[Question]:
    """Parse a JSON questions file.

    Accepts the gold's own shape (``{"queries": [{"query_id", "query"}, ...]}``),
    so a gold file can be handed straight to --questions-file and the run is
    guaranteed to emit ids that join to it. A bare list of objects or of plain
    strings also works.

    Args:
        path: Questions file.

    Returns:
        The questions in file order.

    Raises:
        ValueError: If the JSON is invalid or has an unexpected shape.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if isinstance(payload, dict):
        raw_entries = payload.get("queries")
        if raw_entries is None:
            raise ValueError(f"{path}: JSON object has no 'queries' array")
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise ValueError(f"{path}: expected a JSON object with 'queries' or a list")

    if not isinstance(raw_entries, list):
        raise ValueError(f"{path}: 'queries' must be a list")

    questions: list[Question] = []
    for idx, entry in enumerate(raw_entries):
        if isinstance(entry, str):
            if entry.strip():
                questions.append(Question(text=entry.strip()))
        elif isinstance(entry, dict):
            questions.append(_question_from_obj(entry, f"{path}[{idx}]"))
        else:
            raise ValueError(f"{path}[{idx}]: expected an object or a string")
    return questions


def _questions_from_csv(path: Path) -> list[Question]:
    """Parse a CSV questions file with a query/question column and optional query_id.

    Args:
        path: Questions file.

    Returns:
        The questions in file order; empty rows are skipped.

    Raises:
        ValueError: If a row has no question field.
    """
    questions: list[Question] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for lineno, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            questions.append(_question_from_obj(dict(row), f"{path}:{lineno}"))
    return questions


def _load_questions(args: argparse.Namespace) -> list[Question]:
    """Load the questions to run, with their gold ids when the file declares them.

    Supported --questions-file formats, picked by suffix:
      * ``.txt`` / anything else: one question per line, optionally
        ``Q01<TAB>question text``;
      * ``.json``: the gold's ``{"queries": [...]}`` shape, or a bare list;
      * ``.jsonl``: one ``{"query_id", "query"}`` object per line;
      * ``.csv``: a ``query_id`` column plus ``query`` or ``question``.

    Args:
        args: Parsed CLI namespace; without ``--questions-file`` the single
            ``--question`` is used.

    Returns:
        The questions in file order.

    Raises:
        FileNotFoundError: If the questions file does not exist.
        ValueError: If the file is empty, malformed, or repeats a query_id.
    """
    logger = logging.getLogger("graphrag.cli")

    if not args.questions_file:
        return [Question(text=args.question)]

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    suffix = questions_path.suffix.lower()
    if suffix == ".json":
        questions = _questions_from_json(questions_path)
    elif suffix == ".jsonl":
        questions = _questions_from_jsonl(questions_path)
    elif suffix == ".csv":
        questions = _questions_from_csv(questions_path)
    else:
        questions = _questions_from_text(questions_path)

    if not questions:
        raise ValueError(f"Questions file is empty: {questions_path}")

    ids = [q.query_id for q in questions if q.query_id]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        # Duplicated ids would make the evaluator's join ambiguous, and it joins
        # on exactly this field.
        raise ValueError(
            f"{questions_path}: duplicate query_id(s) {duplicates}"
        )

    if not ids:
        logger.warning(
            "Questions file %s declares no query_id: results.jsonl will carry an "
            "empty query_id and the evaluator must fall back to joining on "
            "question TEXT. Use a gold .json/.jsonl/.csv, or 'Q01<TAB>question' "
            "lines, to join by id.",
            questions_path,
        )
    elif len(ids) < len(questions):
        logger.warning(
            "Questions file %s declares query_id for only %d/%d questions; the "
            "rest will fall back to a text join at evaluation time.",
            questions_path,
            len(ids),
            len(questions),
        )
    else:
        logger.info(
            "Loaded %d questions with query_id from %s", len(questions), questions_path
        )
    return questions


def _run_experiments(
    args: argparse.Namespace, kg_manager: KnowledgeGraphManager
) -> None:
    """Run every question under every strategy and write the run directory.

    Writes ``results.jsonl``, ``results.csv``, ``summary.txt``,
    ``summary.json`` and ``config.json`` under
    ``<output-dir>/<timestamp>_<tag>/``.

    Args:
        args: Parsed CLI namespace.
        kg_manager: Connected graph manager.

    Raises:
        ValueError: If ``--runs-per-strategy`` or ``--strategies`` is invalid,
            or the questions cannot be loaded.
    """
    if args.runs_per_strategy < 1:
        raise ValueError("--runs-per-strategy must be >= 1")

    questions = _load_questions(args)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if not strategies:
        raise ValueError("--strategies must include at least one strategy")

    base_config = _build_base_config(args)
    llm_manager = _build_llm_manager(args=args, warmup=args.llm_warmup)
    runner = ExperimentRunner(questions=questions)

    # Build the text pipeline if ANY selected strategy resolves to a config that
    # uses raw-text retrieval (text_only, hybrid, ...). Deriving this from the
    # resolved config instead of a hardcoded name list keeps new text-using
    # strategies working automatically.
    needs_text = any(
        apply_strategy(base_config, s).use_text_retriever for s in strategies
    )
    text_pipeline = _build_text_pipeline(args) if needs_text else None

    strategy_configs: dict[str, dict] = {}
    for strategy in strategies:
        for run_index in range(1, args.runs_per_strategy + 1):
            config = apply_strategy(base_config, strategy)
            if strategy not in strategy_configs:
                strategy_configs[strategy] = dataclasses.asdict(config)
            retriever = KGRetriever(
                kg_store=kg_manager,
                config=config,
                text_pipeline=text_pipeline if config.use_text_retriever else None,
            )
            agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
            runner.run_agent(
                agent=agent,
                label=strategy,
                run_metadata={
                    "run_index": run_index,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url
                    if args.llm and args.vllm
                    else "",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction
                    if args.llm
                    else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                    "enable_decomposition_step": args.enable_decomposition_step,
                    "enable_adaptive_routing_step": args.enable_adaptive_routing_step,
                },
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.experiment_tag.strip() or "batch"
    output_dir = Path(args.output_dir) / f"{timestamp}_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # From here every log line names the directory its results land in.
    set_run_id(output_dir.name)

    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path = output_dir / "summary.json"
    config_json_path = output_dir / "config.json"

    # Which graph the run actually read. The Neo4j target comes from the
    # environment, never from the CLI, so without this the artifacts could not
    # say whether a run used staging or the hosted graph. The password is
    # deliberately not recorded.
    config_json_path.write_text(
        json.dumps(
            {
                "cli_args": {k: v for k, v in vars(args).items()},
                "graph_target": {
                    "neo4j_url": os.getenv("NEO4J_URL", ""),
                    "neo4j_database": os.getenv("NEO4J_DATABASE", ""),
                    "embed_base_url": os.getenv("GRAPHRAG_EMBED_BASE_URL", ""),
                    "embed_model": os.getenv("GRAPHRAG_EMBED_MODEL", ""),
                },
                "strategy_configs": strategy_configs,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    runner.export_jsonl(str(jsonl_path))
    runner.export_csv(str(csv_path))
    summary_text = runner.summary()
    summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "tag": tag,
                "questions_count": len(questions),
                # Lets a reader tell at a glance whether this run can be joined
                # to the gold by id or only by question text.
                "questions_with_query_id": sum(1 for q in questions if q.query_id),
                "strategies": strategies,
                "runs_per_strategy": args.runs_per_strategy,
                "llm": {
                    "enabled": args.llm,
                    "model_id": args.model_id if args.llm else "none",
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url
                    if args.llm and args.vllm
                    else "",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction
                    if args.llm
                    else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                },
                "agent_pipeline": {
                    "enable_decomposition_step": args.enable_decomposition_step,
                    "enable_adaptive_routing_step": args.enable_adaptive_routing_step,
                },
                "stats": runner.summary_stats(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("Experiment completed. Output directory: %s", output_dir)
    for path in (jsonl_path, csv_path, summary_txt_path, summary_json_path, config_json_path):
        logger.info("Output file: %s", path)
    logger.info("Summary:\n%s", summary_text)


# The run the process is currently working on. A campaign writes its results
# into `<timestamp>_<tag>/`, and that directory name is the only identifier the
# run has — so it is the one worth stamping on every log line. Held in a dict
# rather than a plain global so `set_run_id` can rebind it after the handlers
# are already attached: logging is configured before the output directory
# exists, and the two cannot be reordered without moving the .env load.
_RUN_ID: dict[str, str] = {"value": "-"}

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(run_id)s | %(name)s | %(message)s"


class _RunIdFilter(logging.Filter):
    """Stamps the current run id on every record that passes a handler.

    Attached to handlers rather than loggers on purpose: a record from a
    third-party logger would otherwise reach the formatter without the
    attribute and raise while trying to format it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Set ``record.run_id`` and let the record through."""
        record.run_id = _RUN_ID["value"]
        return True


def set_run_id(run_id: str) -> None:
    """Name the run whose lines follow.

    Two campaigns in one nohup file, or one campaign whose arms run in
    parallel, interleave their lines; the run id lets a warning be attributed
    to the run whose results it explains.

    Args:
        run_id: Usually the output directory's name. Empty restores ``-``.
    """
    _RUN_ID["value"] = str(run_id).strip() or "-"


def _attach_run_id(handler: logging.Handler) -> None:
    """Give ``handler`` the run-id filter, once."""
    if not any(isinstance(f, _RunIdFilter) for f in handler.filters):
        handler.addFilter(_RunIdFilter())


def _configure_logging() -> None:
    """Give a campaign log a timestamp and, on request, a file of its own.

    Every line carries a timestamp and the run id, so a long campaign log
    shows how long a turn took, when a retry happened, and which of several
    arms in one file a warning belongs to.

    GRAPHRAG_LOG_FILE adds a file handler beside the console one. It is an
    environment variable rather than a flag because every campaign flag is part
    of the experiment's identity, recorded in config.json and compared across
    arms — where the log is written is not.
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    for handler in logging.getLogger().handlers:
        _attach_run_id(handler)

    log_file = os.getenv("GRAPHRAG_LOG_FILE", "").strip()
    if not log_file:
        return

    root = logging.getLogger()
    if any(getattr(h, "_graphrag_file_handler", False) for h in root.handlers):
        return
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    _attach_run_id(handler)
    handler._graphrag_file_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    logger.info("Logging to %s", path)


def main() -> None:
    """Entry point: validate the flags, then answer one question or run ``--experiment``."""
    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Load local .env if present while preserving variables already set by system/scheduler.
    load_dotenv(override=False)

    if args.llm_warmup and not args.llm:
        parser.error("--llm-warmup requires --llm")
    if args.vllm and not args.llm:
        parser.error("--vllm requires --llm")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be >= 1")
    if args.recursion_limit < 1:
        parser.error("--recursion-limit must be >= 1")
    if args.gpu_memory_fraction <= 0 or args.gpu_memory_fraction > 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")

    _configure_logging()

    kg_config = build_kg_config_from_env()
    kg_manager = KnowledgeGraphManager(kg_config)

    logger.info("Graph Schema: %s", kg_manager.refresh_schema())

    if args.experiment:
        _run_experiments(args=args, kg_manager=kg_manager)
        return

    config = _build_base_config(args)
    # Single-question mode applies the first --strategies entry, so `hybrid`
    # gets its text channel. "default" is a no-op deepcopy.
    single_strategy = str(args.strategies or "default").split(",")[0].strip() or "default"
    config = apply_strategy(config, single_strategy)

    text_pipeline = _build_text_pipeline(args) if config.use_text_retriever else None
    retriever = KGRetriever(kg_store=kg_manager, config=config, text_pipeline=text_pipeline)
    llm_manager = _build_llm_manager(args=args, warmup=False)

    agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
    result = agent.invoke(args.question)

    print("\nAgent answer:")
    print(result.get("answer", ""))
    print("\nLatency (ms):", f"{result.get('latency_ms', 0.0):.2f}")


if __name__ == "__main__":
    main()
