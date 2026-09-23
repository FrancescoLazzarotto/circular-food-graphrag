"""Text-only RAG agent used as the baseline to the graph agent."""

from __future__ import annotations

import time
import uuid

from graphrag.config import AgentConfig
from graphrag.llm.manager import LLMManager
from graphrag.text_rag.pipeline import StandardTextRAGPipeline


class StandardRAGAgent:
    """Simple text-only RAG agent built on top of StandardTextRAGPipeline."""

    def __init__(
        self,
        pipeline: StandardTextRAGPipeline,
        config: AgentConfig | None = None,
        llm: LLMManager | None = None,
        top_k: int = 4,
        include_sources: bool = True,
    ) -> None:
        """Create the agent, warming the LLM up when the config asks for it.

        Args:
            pipeline: Indexed text pipeline to retrieve from.
            config: Agent configuration; defaults to ``AgentConfig()``.
            llm: LLM used to answer; ``None`` returns a context preview instead.
            top_k: Chunks retrieved per question.
            include_sources: Prefix each chunk with its source in the context.

        Raises:
            ValueError: If ``top_k`` is not positive.
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.pipeline = pipeline
        self.config = config or AgentConfig()
        self.llm = llm
        self.top_k = top_k
        self.include_sources = include_sources

        if self.llm is not None and self.config.llm_warmup:
            self.llm.warmup()

    def invoke(self, question: str) -> dict:
        """Retrieve context for ``question`` and answer it.

        Args:
            question: The question.

        Returns:
            A result dict with the same keys as the graph agent's (``answer``,
            ``text_context``, counts, ``latency_ms``, ...); the graph fields
            are empty.
        """
        start = time.perf_counter()

        retrieved = self.pipeline.retrieve(query=question, top_k=self.top_k)
        if self.include_sources:
            parts = [
                f"Source: {c.source}\n{c.content}" if c.source else c.content
                for c in retrieved
            ]
        else:
            parts = [c.content for c in retrieved]
        context = "\n\n---\n\n".join(parts)

        if self.llm is not None:
            generated = self.llm.generate(
                query=question, context=context, config=self.config
            )
            answer = generated.get("answer", "")
        else:
            if context:
                answer = f"Retrieved context preview:\n{context[:800]}"
            else:
                answer = "No context retrieved."

        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "run_id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "text_context": context,
            "retrieved_text_chunks_count": len(retrieved),
            "kg_triples": [],
            "retrieved_neighbors_count": 0,
            "retrieved_subgraph_count": 0,
            "retrieved_shortest_path_count": 0,
            "sub_questions": [question],
            "latency_ms": latency_ms,
        }
