"""Optional RAGAS evaluation, with a local or vLLM-served judge."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from evalkit.models import EvalRow

logger = logging.getLogger("graphrag")

DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]


def _coerce_score(value: Any) -> float | None:
    """`value` as a finite float, or ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_ragas_sample(row: EvalRow) -> dict[str, Any]:
    """A row as the question/answer/ground_truth/contexts sample RAGAS reads."""
    contexts = [str(c).strip() for c in row.contexts if str(c).strip()]
    return {
        "question": row.question,
        "answer": row.answer,
        "ground_truth": row.ground_truth,
        "contexts": contexts,
    }


def _resolve_metrics(metric_names: list[str]) -> tuple[list[Any], list[str]]:
    """Map metric names to RAGAS metric objects, across RAGAS versions.

    Returns:
        The metric objects, and the names that could not be resolved.
    """
    import ragas.metrics as ragas_metrics  # type: ignore

    candidates: dict[str, list[str]] = {
        "faithfulness": ["faithfulness", "Faithfulness"],
        "answer_relevancy": ["answer_relevancy", "answer_relevance", "AnswerRelevancy", "ResponseRelevancy"],
        "answer_correctness": ["answer_correctness", "AnswerCorrectness"],
        "context_precision": ["context_precision", "ContextPrecision"],
        "context_recall": ["context_recall", "ContextRecall"],
    }

    resolved: list[Any] = []
    missing: list[str] = []

    for name in metric_names:
        found = None
        for candidate in candidates.get(name, [name]):
            if hasattr(ragas_metrics, candidate):
                found = getattr(ragas_metrics, candidate)
                break

        if found is None:
            missing.append(name)
            continue

        if isinstance(found, type):
            try:
                resolved.append(found())
            except TypeError:
                missing.append(name)
        else:
            resolved.append(found)

    return resolved, missing


def _build_embeddings(embed_model: str) -> Any:
    """Wrap a local HuggingFace embedding model for RAGAS.

    Embeddings stay local regardless of the judge backend — only the judge LLM
    is swapped between the transformers and vLLM paths.
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embed_model))


def _build_transformers_llm(judge_model: str, max_new_tokens: int) -> Any:
    """Load a local HuggingFace causal LM as the RAGAS judge (in-process)."""
    from langchain_huggingface import HuggingFacePipeline  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    base_model = AutoModelForCausalLM.from_pretrained(judge_model, device_map="auto")
    generation = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return LangchainLLMWrapper(HuggingFacePipeline(pipeline=generation))


def _build_vllm_llm(
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> Any:
    """Use a model served by a vLLM OpenAI-compatible endpoint as the RAGAS judge.

    No weights are loaded in-process: requests go to *base_url* (e.g. the 32B
    already warm on GPU0), so a large judge is available without a second load.
    """
    from langchain_openai import ChatOpenAI  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore

    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key or "EMPTY",  # vLLM ignores the key but the client requires one
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=180,
        max_retries=3,
    )
    return LangchainLLMWrapper(llm)


def _build_llm_and_embeddings(
    judge_model: str,
    embed_model: str,
    judge_backend: str = "transformers",
    vllm_base_url: str = "",
    vllm_model: str = "",
    vllm_api_key: str = "",
    max_new_tokens: int = 192,
    temperature: float = 0.0,
) -> tuple[Any | None, Any | None]:
    """The judge LLM and embeddings for the chosen backend.

    Returns:
        ``(llm, embeddings)``; ``(None, None)`` without a judge model on the
        transformers backend, which leaves RAGAS on its own defaults.
    """
    if judge_backend == "vllm":
        ragas_llm = _build_vllm_llm(
            model=vllm_model,
            base_url=vllm_base_url,
            api_key=vllm_api_key,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return ragas_llm, _build_embeddings(embed_model)

    # transformers backend (default)
    if not judge_model:
        # Fall back to RAGAS defaults (OpenAI via OPENAI_API_KEY).
        return (None, None)

    return _build_transformers_llm(judge_model, max_new_tokens), _build_embeddings(embed_model)


def run_ragas(
    rows: list[EvalRow],
    metric_names: list[str] | None = None,
    judge_model: str = "",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    judge_backend: str = "transformers",
    vllm_base_url: str = "",
    vllm_model: str = "",
    vllm_api_key: str = "",
    max_new_tokens: int = 192,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run RAGAS metrics on a list of EvalRow.

    Args:
        rows: EvalRows to evaluate; rows with skip_reason are skipped.
        metric_names: RAGAS metric names to compute.
        judge_model: Local HuggingFace model ID for the judge (transformers
            backend; empty → RAGAS default OpenAI via OPENAI_API_KEY).
        embed_model: Embedding model ID for RAGAS (always local HuggingFace).
        judge_backend: "transformers" (in-process HF load) or "vllm"
            (OpenAI-compatible endpoint, no weights loaded in-process).
        vllm_base_url: vLLM endpoint, e.g. http://localhost:8000/v1 (vllm backend).
        vllm_model: Model id served by the vLLM endpoint (vllm backend).
        vllm_api_key: API key for the endpoint; vLLM ignores it but the client
            requires a non-empty value (defaults to VLLM_API_KEY/OPENAI_API_KEY).
        max_new_tokens: Max tokens the judge may generate per call.
        temperature: Judge sampling temperature (0 = deterministic).

    Returns:
        Dict with rows_evaluated, rows_skipped, metrics (mean/std/count), row_scores.
    """
    if metric_names is None:
        metric_names = DEFAULT_METRICS

    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
    except Exception as exc:
        raise ImportError(
            "RAGAS dependencies missing. Install with:\n"
            "  pip install -r evaluation/requirements.txt"
        ) from exc

    metric_objects, missing = _resolve_metrics(metric_names)
    if missing:
        logger.warning("Skipped unsupported RAGAS metrics: %s", ", ".join(missing))
    if not metric_objects:
        raise ValueError("No valid RAGAS metrics resolved.")

    judge_model = judge_model.strip()
    judge_backend = (judge_backend or "transformers").strip().lower()
    if judge_backend not in {"transformers", "vllm"}:
        raise ValueError(f"Unknown judge_backend {judge_backend!r} (use 'transformers' or 'vllm').")

    if judge_backend == "vllm":
        vllm_base_url = vllm_base_url.strip() or os.getenv("VLLM_BASE_URL", "")
        vllm_model = vllm_model.strip() or os.getenv("VLLM_MODEL_NAME", "")
        vllm_api_key = vllm_api_key.strip() or os.getenv("VLLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        if not vllm_base_url or not vllm_model:
            raise ValueError(
                "vllm backend requires --vllm-base-url and --vllm-model "
                "(or VLLM_BASE_URL / VLLM_MODEL_NAME in the environment)."
            )
    elif not judge_model and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Provide --judge-model for local evaluation or set OPENAI_API_KEY for OpenAI."
        )

    ragas_llm, ragas_embeddings = _build_llm_and_embeddings(
        judge_model,
        embed_model,
        judge_backend=judge_backend,
        vllm_base_url=vllm_base_url,
        vllm_model=vllm_model,
        vllm_api_key=vllm_api_key,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    scored_rows: list[dict[str, Any]] = []
    skipped = 0

    for row in rows:
        if row.skip_reason:
            skipped += 1
            continue

        sample = _to_ragas_sample(row)
        try:
            dataset = Dataset.from_list([sample])
            result = evaluate(
                dataset=dataset,
                metrics=metric_objects,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
        except Exception as exc:
            logger.warning("RAGAS failed for question %r: %s", row.question[:60], exc)
            skipped += 1
            continue

        # Extract scores
        scores: dict[str, float] = {}
        if hasattr(result, "to_pandas"):
            frame = result.to_pandas()
            if not frame.empty:
                row_data = frame.iloc[0].to_dict()
                for key, value in row_data.items():
                    if key in {"question", "answer", "ground_truth", "contexts"}:
                        continue
                    s = _coerce_score(value)
                    if s is not None:
                        scores[str(key)] = s
        elif isinstance(result, dict):
            for key, value in result.get("scores", {}).items():
                s = _coerce_score(value)
                if s is not None:
                    scores[str(key)] = s

        if not scores:
            logger.warning("RAGAS returned empty scores for %r; skipping", row.question[:60])
            skipped += 1
            continue

        entry = {
            "run_dir": row.run_dir,
            "model_id": row.model_id,
            "framework": row.framework,
            "strategy": row.strategy,
            "question": row.question,
            "question_type": row.question_type,
        }
        entry.update(scores)
        scored_rows.append(entry)

    metric_cols = sorted(
        {
            key
            for row_data in scored_rows
            for key in row_data
            if key not in {"run_dir", "model_id", "framework", "strategy", "question", "question_type"}
        }
    )

    summaries: dict[str, Any] = {}
    for col in metric_cols:
        values = [float(r[col]) for r in scored_rows if _coerce_score(r.get(col)) is not None]
        if values:
            summaries[col] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }

    return {
        "rows_evaluated": len(scored_rows),
        "rows_skipped": skipped,
        "metrics": summaries,
        "row_scores": scored_rows,
    }
