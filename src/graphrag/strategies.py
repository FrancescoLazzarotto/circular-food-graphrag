"""Retrieval-strategy presets shared by the CLI and the experiment drivers.

``graphrag.cli`` and ``scripts/runners/run_retrieval_matrix.py`` both import
these presets, so this module is the single place they are defined; keep
:data:`STRATEGY_PRESETS` and :func:`apply_strategy` in sync.

A preset only toggles the retrieval channels (the ``include_*`` flags, the text
retriever, hops). Limits and ranking options come from the base
``AgentConfig``; the resolved per-strategy config is serialised to
``config.json`` in each experiment output directory.
"""

from __future__ import annotations

import copy

from graphrag.config import AgentConfig

STRATEGY_PRESETS: tuple[str, ...] = (
    "default",
    "hybrid",
    "text_only",
    "no_retrieval",
    "text_plus_triples",
    "neighbors_focus",
    "subgraph_2hop",
    "shortest_path",
)


def apply_strategy(base: AgentConfig, label: str) -> AgentConfig:
    """Return a deep copy of ``base`` configured for the named retrieval strategy.

    Args:
        base: Baseline agent configuration to clone.
        label: One of :data:`STRATEGY_PRESETS`.

    Returns:
        A new ``AgentConfig`` with the strategy's ``include_*`` flags applied.

    Raises:
        ValueError: If ``label`` is not a known strategy.
    """
    config = copy.deepcopy(base)

    if label == "default":
        return config

    if label == "hybrid":
        # GraphRAG default (all KG channels) augmented with raw-text retrieval:
        # the graph supplies relations, the text supplies the fine-grained facts
        # that triples abstract away. The caller must pass a text pipeline to the
        # KGRetriever, otherwise the text channel is silently skipped.
        config.use_text_retriever = True
        return config

    if label == "text_only":
        config.include_nodes = False
        config.include_triples = False
        config.include_neighbors = False
        config.include_subgraph = False
        config.include_shortest_path = False
        config.use_text_retriever = True
        return config

    if label == "no_retrieval":
        config.include_nodes = False
        config.include_triples = False
        config.include_neighbors = False
        config.include_subgraph = False
        config.include_shortest_path = False
        config.use_text_retriever = False
        return config

    if label == "text_plus_triples":
        config.include_neighbors = False
        config.include_subgraph = False
        config.include_shortest_path = False
        return config

    if label == "neighbors_focus":
        config.include_nodes = False
        config.include_subgraph = False
        config.include_shortest_path = False
        return config

    if label == "subgraph_2hop":
        config.hops = max(2, int(config.hops))
        config.include_nodes = False
        config.include_neighbors = False
        config.include_shortest_path = False
        return config

    if label == "shortest_path":
        config.include_nodes = False
        config.include_neighbors = False
        config.include_subgraph = False
        return config

    allowed = ",".join(STRATEGY_PRESETS)
    raise ValueError(f"Unknown strategy '{label}'. Allowed: {allowed}")
