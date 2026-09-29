"""Configuration of an evaluation run and of its LLM judge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_CI = 0.95
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_REGRESSION_THRESHOLD = 0.05


@dataclass
class JudgeConfig:
    """How the LLM judge is run.

    Attributes:
        backend: ``local_hf``, ``vllm`` or ``api``.
        model_id: Judge model.
        vllm_base_url: vLLM endpoint.
        vllm_api_key: Key for the vLLM endpoint.
        api_provider: ``anthropic`` or ``openai`` for the API backend.
        rubrics: Rubrics scored on every row.
        max_new_tokens: Judge output budget.
        cache_size: Verdicts kept in memory.
    """

    backend: str = "vllm"  # "local_hf" | "vllm" | "api"
    model_id: str = ""
    vllm_base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    vllm_api_key: str = field(
        default_factory=lambda: os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    )
    api_provider: str = "anthropic"  # "anthropic" | "openai"
    # factual_correctness does not fold in coverage, so completeness must be asked
    # for explicitly or a default run silently drops one of the gold's judge_dimensions.
    # `abstention` is not listed: it is applied automatically to distractor rows only.
    rubrics: list[str] = field(
        default_factory=lambda: [
            "factual_correctness",
            "completeness",
            "groundedness",
            "relevance",
        ]
    )
    max_new_tokens: int = 256
    cache_size: int = 256


@dataclass
class EvalConfig:
    """Settings of one evaluation run.

    Attributes:
        gold_dir: Directory of the gold sets.
        baselines_path: Baseline metrics for the regression check.
        k: Cut-off of the ranked retrieval metrics; ``None`` for all.
        bertscore: Also compute BERTScore.
        ragas_enabled: Also run RAGAS.
        ragas_metrics: RAGAS metrics to compute.
        n_bootstrap: Bootstrap resamples.
        ci: Confidence level of the bootstrap interval.
        seed: Bootstrap seed.
        regression_threshold: Change that counts as a regression.
        judge: Judge settings.
        extra: Free-form options.
    """

    gold_dir: Path = field(default_factory=lambda: Path("evaluation/gold"))
    baselines_path: Path = field(
        default_factory=lambda: Path("evaluation/baselines/baseline_metrics.json")
    )

    # Retrieval metrics
    k: int | None = None
    bertscore: bool = False

    # RAGAS (optional)
    ragas_enabled: bool = False
    ragas_metrics: list[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "context_precision",
            "context_recall",
        ]
    )

    # Bootstrap
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N
    ci: float = DEFAULT_BOOTSTRAP_CI
    seed: int = DEFAULT_BOOTSTRAP_SEED

    # Regression
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD

    # Judge
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    extra: dict[str, Any] = field(default_factory=dict)
