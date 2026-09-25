"""Run a question set through every standard-RAG and GraphRAG strategy.

Each strategy answers every question, ``--runs-per-strategy`` times. The run
directory ``<output-dir>/<timestamp>_<tag>`` receives ``results.jsonl``,
``results.csv``, ``summary.txt``, ``summary.json``, ``config.json`` and, with
resource monitoring on, the resource samples and summary. Outputs are written
even when the run fails part-way.

Strategies are chosen with ``--standard-strategies`` and ``--graph-strategies``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig, DEFAULT_MODEL_ID, build_kg_config_from_env
from graphrag.experiments import ExperimentRunner
from graphrag.experiments.resource_monitor import ResourceMonitor
from graphrag.experiments.standard_presets import (
    STANDARD_STRATEGIES_DEFAULT,
    STANDARD_STRATEGIES_SMOKE,
    STANDARD_STRATEGY_PRESETS,
)
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.strategies import STRATEGY_PRESETS, apply_strategy
from graphrag.text_rag.agent import StandardRAGAgent
from graphrag.text_rag.factory import make_text_pipeline
from graphrag.text_rag.pipeline import StandardTextRAGPipeline

_GRAPH_STRATEGIES_DEFAULT = STRATEGY_PRESETS
_GRAPH_STRATEGIES_SMOKE = ("default", "text_plus_triples")

_DEFAULT_MAX_NEW_TOKENS = 256
_DEFAULT_GPU_MEMORY_FRACTION = 0.92
_PRODUCTION_FAST_LARGE_MODEL_MAX_NEW_TOKENS = 160
_PERFORMANCE_PROFILE_CHOICES = ("auto", "default", "production_fast")


# Standard RAG baseline presets live in graphrag.experiments.standard_presets
# (shared module); aliased here to keep existing references readable.
_STANDARD_STRATEGY_PRESETS = STANDARD_STRATEGY_PRESETS
_STANDARD_STRATEGIES_DEFAULT = STANDARD_STRATEGIES_DEFAULT
_STANDARD_STRATEGIES_SMOKE = STANDARD_STRATEGIES_SMOKE


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run multi-question, multi-strategy matrix over Standard RAG and GraphRAG",
    )

    parser.add_argument(
        "--question", default="Quali sono gli obiettivi della strategia Farm to Fork?"
    )
    parser.add_argument("--questions-file", default="", help="One question per line")
    parser.add_argument(
        "--entity",
        default="",
        help="Optional entity seed for graph traversal (leave empty for auto-seeding)",
    )

    parser.add_argument(
        "--documents",
        nargs="+",
        default=["docs", "README.md"],
        help="Files/folders used by standard text RAG",
    )
    parser.add_argument(
        "--doc-patterns",
        default="*.pdf,*.txt,*.md,*.markdown",
        help="Comma-separated file patterns for directory discovery",
    )

    parser.add_argument(
        "--standard-strategies", default=",".join(_STANDARD_STRATEGIES_DEFAULT)
    )
    parser.add_argument(
        "--graph-strategies", default=",".join(_GRAPH_STRATEGIES_DEFAULT)
    )

    parser.add_argument("--runs-per-strategy", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--experiment-tag", default="")

    parser.add_argument(
        "--monitor-resources",
        dest="monitor_resources",
        action="store_true",
        help="Enable CPU/RAM/GPU sampling and save telemetry files",
    )
    parser.add_argument(
        "--no-monitor-resources",
        dest="monitor_resources",
        action="store_false",
        help="Disable resource telemetry sampling",
    )
    parser.set_defaults(monitor_resources=True)
    parser.add_argument(
        "--resource-sample-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds for resource telemetry",
    )

    parser.add_argument("--llm", action="store_true")
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
    parser.add_argument("--llm-warmup", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--performance-profile",
        default="auto",
        choices=_PERFORMANCE_PROFILE_CHOICES,
        help=(
            "LLM performance profile. In auto mode, production_fast is applied by default for large models (>=30B)."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--allow-large-model-fp16-fallback", action="store_true")
    parser.add_argument("--enable-decomposition-step", action="store_true")
    parser.add_argument("--enable-adaptive-routing-step", action="store_true")

    parser.add_argument(
        "--dense-embedding-model",
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model ID for dense retrieval (multilingual-e5-base by default)",
    )
    parser.add_argument(
        "--vector-index-dir",
        default="artifacts/vector_index",
        help="Directory for persisted FAISS index cache (used by dense strategies)",
    )
    parser.add_argument(
        "--dense-device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for dense embedding model: auto, cpu, or cuda",
    )

    parser.add_argument("--skip-standard", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")

    parser.add_argument("--smoke", action="store_true", help="Run a smaller test first")
    parser.add_argument("--smoke-questions", type=int, default=2)
    parser.add_argument(
        "--smoke-standard-strategies", default=",".join(_STANDARD_STRATEGIES_SMOKE)
    )
    parser.add_argument(
        "--smoke-graph-strategies", default=",".join(_GRAPH_STRATEGIES_SMOKE)
    )

    return parser


def _parse_csv(raw: str) -> list[str]:
    """The non-empty items of a comma-separated value."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_performance_profile(args: argparse.Namespace) -> tuple[str, int, float]:
    """Resolve the performance profile and the generation settings it implies.

    ``auto`` applies ``production_fast`` to large models (>=30B) when the LLM
    is on, ``default`` otherwise; ``production_fast`` lowers the token budget
    unless ``--max-new-tokens`` was given.

    Args:
        args: Parsed command line.

    Returns:
        The applied profile, the token budget and the GPU memory fraction.
    """
    requested = (args.performance_profile or "auto").strip().lower()
    if requested not in _PERFORMANCE_PROFILE_CHOICES:
        requested = "auto"

    applied = requested
    if requested == "auto":
        if args.llm and LLMManager._is_large_model(args.model_id):
            applied = "production_fast"
        else:
            applied = "default"

    max_new_tokens = (
        int(args.max_new_tokens)
        if args.max_new_tokens is not None
        else _DEFAULT_MAX_NEW_TOKENS
    )
    gpu_memory_fraction = (
        float(args.gpu_memory_fraction)
        if args.gpu_memory_fraction is not None
        else _DEFAULT_GPU_MEMORY_FRACTION
    )

    if applied == "production_fast":
        if args.max_new_tokens is None:
            max_new_tokens = _PRODUCTION_FAST_LARGE_MODEL_MAX_NEW_TOKENS
        if args.gpu_memory_fraction is None:
            gpu_memory_fraction = _DEFAULT_GPU_MEMORY_FRACTION

    return applied, max_new_tokens, gpu_memory_fraction


def _format_optional_float(value: float | None) -> str:
    """`value` to two decimals, or ``n/a``."""
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _resource_summary_lines(resource_summary: dict[str, Any] | None) -> list[str]:
    """The resource monitor's peaks as summary lines; empty without a summary."""
    if not resource_summary:
        return []

    lines = [
        "Resource Monitor:",
        f"- duration_sec={_format_optional_float(resource_summary.get('monitoring_duration_sec'))}",
        f"- peak_process_rss_mb={_format_optional_float(resource_summary.get('peak_process_rss_mb'))}",
        f"- peak_system_ram_percent={_format_optional_float(resource_summary.get('peak_system_ram_percent'))}",
        f"- peak_system_cpu_percent={_format_optional_float(resource_summary.get('peak_system_cpu_percent'))}",
    ]

    gpus = resource_summary.get("gpus", [])
    if isinstance(gpus, list) and gpus:
        for gpu in gpus:
            if not isinstance(gpu, dict):
                continue
            gpu_index = gpu.get("index", "?")
            gpu_name = gpu.get("name", "unknown")
            lines.append(
                f"- gpu[{gpu_index}] {gpu_name}: peak_mem_mb={_format_optional_float(gpu.get('peak_memory_used_mb'))}, "
                f"peak_util_gpu_percent={_format_optional_float(gpu.get('peak_utilization_gpu_percent'))}"
            )
    else:
        lines.append("- gpu: no samples detected")

    return lines


def _load_questions_from_json(path: Path) -> list[str]:
    """Questions from a JSON list, or an object with a ``questions`` list.

    Entries may be strings or objects with a ``question`` field;
    case-insensitive repeats are dropped.

    Raises:
        ValueError: If the payload has another shape or holds no question.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        raw_questions = payload.get("questions", [])
    else:
        raw_questions = payload

    if not isinstance(raw_questions, list):
        raise ValueError(
            f"Unsupported JSON question payload in {path}. Expected a list or an object with 'questions'."
        )

    questions: list[str] = []
    seen: set[str] = set()
    for item in raw_questions:
        if isinstance(item, dict):
            value = str(item.get("question", "")).strip()
        else:
            value = str(item).strip()

        if not value:
            continue

        key = value.lower()
        if key in seen:
            continue

        seen.add(key)
        questions.append(value)

    if not questions:
        raise ValueError(f"Questions JSON is empty: {path}")

    return questions


def _load_questions(args: argparse.Namespace) -> list[str]:
    """The questions: ``--question``, or the lines or JSON of ``--questions-file``.

    Raises:
        FileNotFoundError: If the questions file does not exist.
        ValueError: If it holds no question.
    """
    if not args.questions_file:
        return [args.question]

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    if questions_path.suffix.lower() == ".json":
        questions = _load_questions_from_json(questions_path)
    else:
        questions = [
            line.strip()
            for line in questions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if not questions:
        raise ValueError(f"Questions file is empty: {questions_path}")
    return questions


def _base_graph_config(args: argparse.Namespace, default_question: str) -> AgentConfig:
    """The GraphRAG config every graph strategy is applied on top of."""
    return AgentConfig(
        query=default_question,
        entity=args.entity,
        include_nodes=True,
        include_triples=True,
        include_neighbors=True,
        include_subgraph=True,
        include_shortest_path=True,
        llm_warmup=args.llm_warmup,
        enable_decomposition_step=args.enable_decomposition_step,
        enable_adaptive_routing_step=args.enable_adaptive_routing_step,
    )


def _resolve_standard_labels(raw_labels: list[str]) -> list[str]:
    """Check that every standard strategy name is a known preset.

    Raises:
        ValueError: If any name is unknown.
    """
    invalid = [label for label in raw_labels if label not in _STANDARD_STRATEGY_PRESETS]
    if invalid:
        allowed = ", ".join(sorted(_STANDARD_STRATEGY_PRESETS.keys()))
        invalid_csv = ", ".join(invalid)
        raise ValueError(
            f"Unknown standard strategies: {invalid_csv}. Allowed: {allowed}"
        )
    return raw_labels


def _run_standard_matrix(
    runner: ExperimentRunner,
    questions: list[str],
    labels: list[str],
    runs_per_strategy: int,
    llm_manager: LLMManager | None,
    args: argparse.Namespace,
) -> None:
    """Run every standard-RAG strategy over the questions.

    Each strategy indexes ``--documents`` with its own chunking and backend,
    then answers the question set `runs_per_strategy` times.

    Args:
        runner: Collects the results.
        questions: The questions.
        labels: Standard strategy names.
        runs_per_strategy: Repetitions of each strategy.
        llm_manager: The generator, or ``None`` for retrieval only.
        args: Parsed command line.
    """
    if not labels:
        return

    discovery_patterns = _parse_csv(args.doc_patterns)

    for label in labels:
        preset = _STANDARD_STRATEGY_PRESETS[label]
        effective_embedding_model = preset.embedding_model or args.dense_embedding_model
        pipeline = make_text_pipeline(
            backend=preset.backend,
            chunk_size=preset.chunk_size,
            chunk_overlap=preset.chunk_overlap,
            min_chunk_chars=preset.min_chunk_chars,
            embedding_model=effective_embedding_model,
            vector_index_dir=args.vector_index_dir,
            device=args.dense_device,
        )
        indexed_chunks = pipeline.index_paths(
            args.documents, discovery_patterns=discovery_patterns
        )

        agent_config = AgentConfig(
            query=questions[0],
            llm_warmup=args.llm_warmup,
        )

        for run_index in range(1, runs_per_strategy + 1):
            print(
                f"[standard_rag] strategy={label} run={run_index}/{runs_per_strategy} "
                f"questions={len(questions)} start"
            )
            started_at = time.perf_counter()
            agent = StandardRAGAgent(
                pipeline=pipeline,
                config=agent_config,
                llm=llm_manager,
                top_k=preset.top_k,
                include_sources=preset.include_sources,
            )
            runner.run_agent(
                agent=agent,
                label=label,
                run_metadata={
                    "framework": "standard_rag",
                    "run_index": run_index,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url
                    if args.llm and args.vllm
                    else "",
                    "performance_profile": str(
                        getattr(args, "performance_profile_applied", "default")
                    ),
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction
                    if args.llm
                    else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                    "indexed_chunks": indexed_chunks,
                    "top_k": preset.top_k,
                    "chunk_size": preset.chunk_size,
                    "chunk_overlap": preset.chunk_overlap,
                    "backend": preset.backend,
                    "embedding_model": effective_embedding_model if preset.backend == "dense" else "",
                    "documents": [
                        str(Path(item).expanduser()) for item in args.documents
                    ],
                },
            )
            elapsed_sec = time.perf_counter() - started_at
            print(
                f"[standard_rag] strategy={label} run={run_index}/{runs_per_strategy} "
                f"done elapsed_sec={elapsed_sec:.2f}"
            )


def _run_graph_matrix(
    runner: ExperimentRunner,
    questions: list[str],
    labels: list[str],
    runs_per_strategy: int,
    llm_manager: LLMManager | None,
    args: argparse.Namespace,
) -> None:
    """Run every GraphRAG strategy over the questions.

    A shared text pipeline is built once when any strategy uses the text
    channel; without it a hybrid strategy would silently collapse into the
    graph-only one.

    Args:
        runner: Collects the results.
        questions: The questions.
        labels: Graph strategy names.
        runs_per_strategy: Repetitions of each strategy.
        llm_manager: The generator, or ``None`` for retrieval only.
        args: Parsed command line.
    """
    if not labels:
        return

    kg_manager = KnowledgeGraphManager(build_kg_config_from_env())
    print("Graph Schema:", kg_manager.refresh_schema())

    base_config = _base_graph_config(args=args, default_question=questions[0])

    # Build a shared text pipeline once if any graph strategy resolves to a config
    # that uses raw-text retrieval (e.g. the hybrid strategy). Without this the
    # text channel is silently skipped and hybrid would collapse into default.
    text_pipeline = None
    if any(apply_strategy(base_config, label).use_text_retriever for label in labels):
        text_pipeline = make_text_pipeline(
            backend=base_config.text_retriever_backend,
            embedding_model=args.dense_embedding_model,
            vector_index_dir=args.vector_index_dir,
            device=args.dense_device,
        )
        indexed_chunks = text_pipeline.index_paths(
            args.documents, discovery_patterns=_parse_csv(args.doc_patterns)
        )
        print(
            f"[graph_rag] text pipeline ({base_config.text_retriever_backend}) "
            f"indexed {indexed_chunks} chunks for hybrid strategies"
        )

    for label in labels:
        for run_index in range(1, runs_per_strategy + 1):
            print(
                f"[graph_rag] strategy={label} run={run_index}/{runs_per_strategy} "
                f"questions={len(questions)} start"
            )
            started_at = time.perf_counter()
            config = apply_strategy(base_config, label)
            retriever = KGRetriever(
                kg_store=kg_manager,
                config=config,
                text_pipeline=text_pipeline if config.use_text_retriever else None,
            )
            agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
            runner.run_agent(
                agent=agent,
                label=label,
                run_metadata={
                    "framework": "graph_rag",
                    "run_index": run_index,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url
                    if args.llm and args.vllm
                    else "",
                    "performance_profile": str(
                        getattr(args, "performance_profile_applied", "default")
                    ),
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction
                    if args.llm
                    else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                    "enable_decomposition_step": args.enable_decomposition_step,
                    "enable_adaptive_routing_step": args.enable_adaptive_routing_step,
                },
            )
            elapsed_sec = time.perf_counter() - started_at
            print(
                f"[graph_rag] strategy={label} run={run_index}/{runs_per_strategy} "
                f"done elapsed_sec={elapsed_sec:.2f}"
            )


def main() -> None:
    """Run the matrix and write the run directory; re-raise a run failure at the end."""
    parser = _build_parser()
    args = parser.parse_args()

    applied_profile, resolved_max_new_tokens, resolved_gpu_memory_fraction = (
        _resolve_performance_profile(args)
    )
    args.performance_profile_applied = applied_profile
    args.max_new_tokens = resolved_max_new_tokens
    args.gpu_memory_fraction = resolved_gpu_memory_fraction

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("graphrag").info(
        "Performance profile requested=%s applied=%s (max_new_tokens=%s, gpu_memory_fraction=%.2f)",
        args.performance_profile,
        args.performance_profile_applied,
        args.max_new_tokens,
        args.gpu_memory_fraction,
    )

    if args.llm_warmup and not args.llm:
        parser.error("--llm-warmup requires --llm")
    if args.vllm and not args.llm:
        parser.error("--vllm requires --llm")
    if args.runs_per_strategy < 1:
        parser.error("--runs-per-strategy must be >= 1")
    if args.resource_sample_interval <= 0:
        parser.error("--resource-sample-interval must be > 0")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.gpu_memory_fraction <= 0 or args.gpu_memory_fraction > 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")
    if args.skip_standard and args.skip_graph:
        parser.error("Cannot skip both standard and graph matrices")

    load_dotenv(override=False)

    questions = _load_questions(args)
    if args.smoke:
        questions = questions[: max(1, int(args.smoke_questions))]

    if not questions:
        parser.error("No questions available after filtering")

    standard_labels = (
        []
        if args.skip_standard
        else _resolve_standard_labels(
            _parse_csv(
                args.smoke_standard_strategies
                if args.smoke
                else args.standard_strategies
            )
        )
    )
    graph_labels = (
        []
        if args.skip_graph
        else _parse_csv(
            args.smoke_graph_strategies if args.smoke else args.graph_strategies
        )
    )

    llm_manager = (
        LLMManager(
            model_id=args.model_id,
            warmup=args.llm_warmup,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_fraction=args.gpu_memory_fraction,
            allow_large_model_fp16_fallback=args.allow_large_model_fp16_fallback,
            use_vllm=args.vllm,
            vllm_base_url=args.vllm_base_url,
        )
        if args.llm
        else None
    )
    runner = ExperimentRunner(questions=questions)

    resource_monitor: ResourceMonitor | None = None
    resource_summary: dict[str, Any] | None = None
    run_error: Exception | None = None

    if args.monitor_resources:
        resource_monitor = ResourceMonitor(
            sample_interval_sec=args.resource_sample_interval, include_gpu=True
        )
        resource_monitor.start()

    try:
        _run_standard_matrix(
            runner=runner,
            questions=questions,
            labels=standard_labels,
            runs_per_strategy=args.runs_per_strategy,
            llm_manager=llm_manager,
            args=args,
        )
        _run_graph_matrix(
            runner=runner,
            questions=questions,
            labels=graph_labels,
            runs_per_strategy=args.runs_per_strategy,
            llm_manager=llm_manager,
            args=args,
        )
    except Exception as exc:
        run_error = exc
        logging.exception(
            "Matrix execution failed. Partial outputs will still be exported."
        )
    finally:
        if resource_monitor is not None:
            resource_summary = resource_monitor.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_tag = "retrieval_matrix_smoke" if args.smoke else "retrieval_matrix"
    tag = args.experiment_tag.strip() or default_tag

    output_dir = Path(args.output_dir) / f"{timestamp}_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path = output_dir / "summary.json"
    config_json_path = output_dir / "config.json"
    resource_samples_path = output_dir / "resource_samples.jsonl"
    resource_summary_path = output_dir / "resource_summary.json"

    base_config = _base_graph_config(args, questions[0])
    config_json_path.write_text(
        json.dumps(
            {
                "cli_args": dict(vars(args)),
                "graph_strategy_configs": {
                    label: dataclasses.asdict(apply_strategy(base_config, label))
                    for label in graph_labels
                },
                "standard_strategy_presets": {
                    label: dataclasses.asdict(_STANDARD_STRATEGY_PRESETS[label])
                    for label in standard_labels
                },
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
    resource_lines = _resource_summary_lines(resource_summary)
    if resource_lines:
        summary_text = (
            summary_text + ("\n\n" if summary_text else "") + "\n".join(resource_lines)
        )

    summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")

    summary_json_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "tag": tag,
                "mode": "smoke" if args.smoke else "full",
                "questions_count": len(questions),
                "runs_per_strategy": args.runs_per_strategy,
                "standard_strategies": standard_labels,
                "graph_strategies": graph_labels,
                "documents": [str(Path(item).expanduser()) for item in args.documents],
                "llm_enabled": args.llm,
                "model_id": args.model_id if args.llm else "none",
                "vllm_enabled": args.vllm,
                "vllm_base_url": args.vllm_base_url if args.llm and args.vllm else "",
                "performance_profile_requested": args.performance_profile,
                "performance_profile_applied": args.performance_profile_applied,
                "max_new_tokens": args.max_new_tokens if args.llm else 0,
                "gpu_memory_fraction": args.gpu_memory_fraction if args.llm else 0.0,
                "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                "enable_decomposition_step": args.enable_decomposition_step,
                "enable_adaptive_routing_step": args.enable_adaptive_routing_step,
                "stats": runner.summary_stats(),
                "status": "failed" if run_error else "completed",
                "error": str(run_error) if run_error else "",
                "resource_monitor": {
                    "enabled": args.monitor_resources,
                    "sample_interval_sec": args.resource_sample_interval,
                    "summary": resource_summary,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if resource_monitor is not None:
        resource_monitor.export_samples_jsonl(str(resource_samples_path))
        resource_monitor.export_summary_json(
            str(resource_summary_path),
            extra={
                "timestamp": timestamp,
                "tag": tag,
                "status": "failed" if run_error else "completed",
                "model_id": args.model_id if args.llm else "none",
                "llm_enabled": args.llm,
                "vllm_enabled": args.vllm,
                "vllm_base_url": args.vllm_base_url if args.llm and args.vllm else "",
                "performance_profile_requested": args.performance_profile,
                "performance_profile_applied": args.performance_profile_applied,
                "max_new_tokens": args.max_new_tokens if args.llm else 0,
                "gpu_memory_fraction": args.gpu_memory_fraction if args.llm else 0.0,
                "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
            },
        )

    if run_error:
        print("\nCombined retrieval matrix failed. Partial outputs exported.")
    else:
        print("\nCombined retrieval matrix completed.")
    print("Output directory:", output_dir)
    print("-", jsonl_path)
    print("-", csv_path)
    print("-", summary_txt_path)
    print("-", summary_json_path)
    if resource_monitor is not None:
        print("-", resource_samples_path)
        print("-", resource_summary_path)
    print("\nSummary:")
    print(summary_text)

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
