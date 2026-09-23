"""Run question sets against agents and export per-question results."""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from graphrag.llm.refusal import is_insufficient
from graphrag.types import triple_key

logger = logging.getLogger("graphrag")


class SupportsInvoke(Protocol):
    """Anything that answers a question and returns its final state as a dict."""

    def invoke(self, question: str) -> dict[str, Any]:
        """Answer ``question`` and return the final state."""
        ...


@dataclass(frozen=True)
class Question:
    """One benchmark question, optionally carrying its gold identifier.

    Attributes:
        text: The question as sent to the agent.
        query_id: Gold query id (``Q01``…``Q30``) when the questions file
            declares one, else "". Emitting it into results.jsonl is what lets
            the evaluator join runs to the gold by id instead of by question
            text.
    """

    text: str
    query_id: str = ""


@dataclass
class ExperimentResult:
    """One question answered by one strategy, as written to results.jsonl.

    The inline comments below document the fields whose meaning is not obvious
    from their name.
    """

    strategy: str
    question: str
    answer: str
    latency_ms: float
    # Empty for questions files that declare no ids (one question per line).
    query_id: str = ""
    kg_triples_used: int = 0
    kg_neighbors_used: int = 0
    kg_subgraph_triples_used: int = 0
    kg_shortest_path_triples_used: int = 0
    sub_questions: int = 0
    insufficient_answer: bool = False
    # The answer before the refusal-rescue retry, and whether it fired. An
    # abstention measured on the final answer measures post-retry behaviour;
    # `insufficient_answer_pre_retry` is the abstention signal to report.
    pre_retry_answer: str = ""
    refusal_retry_applied: bool = False
    insufficient_answer_pre_retry: bool = False
    contexts: list[str] = field(default_factory=list)
    retrieved_triples: list[dict[str, Any]] = field(default_factory=list)
    retrieved_entities: list[dict[str, Any] | str] = field(default_factory=list)
    # Provenance of retrieved text chunks ({"source", "chunk_id"}); lets the
    # provenance analysis attribute the text channel to its origin documents.
    retrieved_text_sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Runs a list of questions against one or more agents and exports artifacts."""

    def __init__(self, questions: list[str] | list[Question]) -> None:
        """Initialise the runner.

        Args:
            questions: Either plain question strings (no gold ids) or
                ``Question`` objects carrying ``query_id``.
        """
        self.questions: list[Question] = [
            q if isinstance(q, Question) else Question(text=str(q)) for q in questions
        ]
        self.results: list[ExperimentResult] = []

    def run_agent(
        self,
        agent: SupportsInvoke,
        label: str,
        run_metadata: dict[str, Any] | None = None,
    ) -> list[ExperimentResult]:
        """Ask every question to ``agent`` and record the results.

        Args:
            agent: The agent to run.
            label: Strategy name recorded on each result.
            run_metadata: Keys added to each result's ``metadata``, after the
                agent's ``run_id``.

        Returns:
            This agent's results, also appended to :attr:`results`.
        """
        batch: list[ExperimentResult] = []
        total = len(self.questions)
        for idx, item in enumerate(self.questions, start=1):
            question = item.text
            state = agent.invoke(question)
            metadata: dict[str, Any] = {"run_id": state.get("run_id", "")}
            if run_metadata:
                metadata.update(run_metadata)

            contexts = self._extract_contexts(state)
            retrieved_triples = self._extract_retrieved_triples(state)
            retrieved_entities = self._extract_retrieved_entities(
                state=state,
                triples=retrieved_triples,
            )
            retrieved_text_sources = self._extract_text_sources(state)

            answer = state.get("answer", "")
            pre_retry_answer = str(state.get("pre_retry_answer", "") or "")
            result = ExperimentResult(
                strategy=label,
                question=question,
                answer=answer,
                latency_ms=float(state.get("latency_ms", 0.0)),
                query_id=item.query_id,
                kg_triples_used=len(state.get("kg_triples", []))
                if isinstance(state.get("kg_triples", []), list)
                else 0,
                kg_neighbors_used=int(state.get("retrieved_neighbors_count", 0) or 0),
                kg_subgraph_triples_used=int(
                    state.get("retrieved_subgraph_count", 0) or 0
                ),
                kg_shortest_path_triples_used=int(
                    state.get("retrieved_shortest_path_count", 0) or 0
                ),
                sub_questions=len(state.get("sub_questions", []))
                if isinstance(state.get("sub_questions", []), list)
                else 0,
                insufficient_answer=self._is_insufficient(answer),
                pre_retry_answer=pre_retry_answer,
                refusal_retry_applied=bool(state.get("refusal_retry_applied")),
                insufficient_answer_pre_retry=self._is_insufficient(
                    pre_retry_answer or answer
                ),
                contexts=contexts,
                retrieved_triples=retrieved_triples,
                retrieved_entities=retrieved_entities,
                retrieved_text_sources=retrieved_text_sources,
                metadata=metadata,
            )
            batch.append(result)
            # Through the logger, not print: this is the only line that says
            # which arm and which question the surrounding INFO lines belong
            # to, so it needs a timestamp and must reach the log file.
            logger.info(
                "[%s] q%d/%d latency_ms=%.0f insufficient=%s kg_triples=%d",
                label,
                idx,
                total,
                result.latency_ms,
                result.insufficient_answer,
                result.kg_triples_used,
            )

        self.results.extend(batch)
        return batch

    @staticmethod
    def _extract_contexts(state: dict[str, Any]) -> list[str]:
        """Collect the distinct non-empty context strings of a final state."""
        contexts: list[str] = []
        seen: set[str] = set()

        # `kg_context` and `merged_context` are declared in RAGState but no node
        # writes them; they are read so states produced by older revisions
        # still yield their context.
        for key in ("text_context", "kg_context", "merged_context"):
            value = str(state.get(key, "") or "").strip()
            if not value:
                continue
            normalized = " ".join(value.split()).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            contexts.append(value)

        return contexts

    @staticmethod
    def _triple_key(triple: dict[str, Any]) -> tuple[str, str, str]:
        """De-duplication key of a triple; see :func:`graphrag.types.triple_key`."""
        return triple_key(triple)

    def _extract_retrieved_triples(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect the distinct triples of every graph channel.

        Args:
            state: Final agent state.

        Returns:
            ``{"subject", "predicate", "object", "source_doc"}`` records from
            ``kg_triples``, ``retrieved_subgraph`` and
            ``retrieved_shortest_path``, de-duplicated.
        """
        triples: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for key in ("kg_triples", "retrieved_subgraph", "retrieved_shortest_path"):
            value = state.get(key, [])
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                rel_props = item.get("relationship_properties", {})
                if not isinstance(rel_props, dict):
                    rel_props = {}
                triple = {
                    "subject": str(item.get("subject", "")),
                    "predicate": str(item.get("predicate", "")),
                    "object": str(item.get("object", "")),
                    # Origin document of this edge (rel prop), kept so provenance
                    # analysis can tell gold-doc facts from cross-document ones.
                    "source_doc": str(rel_props.get("source_doc", "")),
                }
                key_tuple = self._triple_key(item)
                if key_tuple in seen:
                    continue
                seen.add(key_tuple)
                triples.append(triple)

        return triples

    @staticmethod
    def _extract_retrieved_entities(
        state: dict[str, Any],
        triples: list[dict[str, Any]],
    ) -> list[dict[str, Any] | str]:
        """Collect the distinct entities retrieved in a turn.

        Args:
            state: Final agent state.
            triples: Triples from :meth:`_extract_retrieved_triples`.

        Returns:
            Retrieved nodes as ``{"id", "name", "labels"}`` dicts, followed by
            the subject and object names of the triples not already listed.
        """
        entities: list[dict[str, Any] | str] = []
        seen: set[str] = set()

        nodes = state.get("retrieved_nodes", []) or state.get("nodes", []) or []
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("node_id", "")).strip()
                name = str(node.get("text", "")).strip()
                labels = node.get("labels", [])
                label_list = (
                    [str(label) for label in labels if str(label).strip()]
                    if isinstance(labels, list)
                    else []
                )

                key = f"id:{node_id}" if node_id else f"name:{name.lower()}"
                if key in seen:
                    continue
                seen.add(key)

                item: dict[str, Any] = {}
                if node_id:
                    item["id"] = node_id
                if name:
                    item["name"] = name
                if label_list:
                    item["labels"] = label_list
                if item:
                    entities.append(item)

        for triple in triples:
            for role in ("subject", "object"):
                value = str(triple.get(role, "")).strip()
                if not value:
                    continue
                key = f"name:{value.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                entities.append(value)

        return entities

    @staticmethod
    def _extract_text_sources(state: dict[str, Any]) -> list[dict[str, str]]:
        """Collect the provenance of retrieved text chunks from agent state.

        Each entry is ``{"source": "<path>#page=N#chunk=M", "chunk_id": "..."}``
        as emitted by the text retriever. Only the provenance tag is kept here;
        the chunk text itself already lives in ``contexts``.

        Args:
            state: Final agent state.

        Returns:
            Text-chunk provenance records; empty when no text channel ran.
        """
        raw = state.get("retrieved_text_sources", [])
        if not isinstance(raw, list):
            return []
        sources: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            if not source:
                continue
            sources.append(
                {"source": source, "chunk_id": str(item.get("chunk_id", ""))}
            )
        return sources

    @staticmethod
    def _is_insufficient(answer: str) -> bool:
        """Whether the answer abstains; see :func:`graphrag.llm.refusal.is_insufficient`."""
        return is_insufficient(answer)

    def compare(self) -> dict[str, list[ExperimentResult]]:
        """Group the results by strategy, in run order."""
        grouped: dict[str, list[ExperimentResult]] = {}
        for result in self.results:
            grouped.setdefault(result.strategy, []).append(result)
        return grouped

    def export_jsonl(self, path: str) -> None:
        """Write one JSON line per result to ``path``."""
        with open(path, "w", encoding="utf-8") as output_file:
            for result in self.results:
                output_file.write(
                    json.dumps(dataclasses.asdict(result), ensure_ascii=False) + "\n"
                )

    def export_csv(self, path: str) -> None:
        """Write the results to ``path`` as CSV, list fields as JSON strings."""
        with open(path, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "query_id",
                    "strategy",
                    "question",
                    "answer",
                    "latency_ms",
                    "kg_triples_used",
                    "kg_neighbors_used",
                    "kg_subgraph_triples_used",
                    "kg_shortest_path_triples_used",
                    "sub_questions",
                    "contexts_json",
                    "retrieved_triples_json",
                    "retrieved_entities_json",
                    "insufficient_answer",
                    "retrieved_text_sources_json",
                    "metadata_json",
                ]
            )
            for result in self.results:
                writer.writerow(
                    [
                        result.query_id,
                        result.strategy,
                        result.question,
                        result.answer,
                        f"{result.latency_ms:.6f}",
                        result.kg_triples_used,
                        result.kg_neighbors_used,
                        result.kg_subgraph_triples_used,
                        result.kg_shortest_path_triples_used,
                        result.sub_questions,
                        json.dumps(result.contexts, ensure_ascii=False),
                        json.dumps(result.retrieved_triples, ensure_ascii=False),
                        json.dumps(result.retrieved_entities, ensure_ascii=False),
                        result.insufficient_answer,
                        json.dumps(
                            result.retrieved_text_sources, ensure_ascii=False
                        ),
                        json.dumps(result.metadata, ensure_ascii=False, sort_keys=True),
                    ]
                )

    def summary_stats(self) -> dict[str, dict[str, float | int]]:
        """Per-strategy averages and insufficiency counts.

        Returns:
            Mapping from strategy to run count, average latency, average
            channel sizes and sub-questions, and insufficient count and rate.
        """
        grouped = self.compare()
        summary: dict[str, dict[str, float | int]] = {}
        for strategy, results in grouped.items():
            if not results:
                continue
            count = len(results)
            insufficient_count = sum(1 for item in results if item.insufficient_answer)
            summary[strategy] = {
                "runs": count,
                "avg_latency_ms": sum(item.latency_ms for item in results) / count,
                "avg_kg_triples_used": sum(item.kg_triples_used for item in results)
                / count,
                "avg_kg_neighbors_used": sum(item.kg_neighbors_used for item in results)
                / count,
                "avg_kg_subgraph_triples_used": sum(
                    item.kg_subgraph_triples_used for item in results
                )
                / count,
                "avg_kg_shortest_path_triples_used": sum(
                    item.kg_shortest_path_triples_used for item in results
                )
                / count,
                "avg_sub_questions": sum(item.sub_questions for item in results)
                / count,
                "insufficient_count": insufficient_count,
                "insufficient_rate": insufficient_count / count,
            }
        return summary

    def summary(self) -> str:
        """Render :meth:`summary_stats` as one line per strategy."""
        lines: list[str] = []
        for strategy, stats in self.summary_stats().items():
            lines.append(
                f"{strategy}: runs={int(stats['runs'])}, "
                f"avg_latency_ms={float(stats['avg_latency_ms']):.2f}, "
                f"avg_kg_triples_used={float(stats['avg_kg_triples_used']):.2f}, "
                f"avg_kg_neighbors_used={float(stats['avg_kg_neighbors_used']):.2f}, "
                f"avg_kg_subgraph_triples_used={float(stats['avg_kg_subgraph_triples_used']):.2f}, "
                f"avg_kg_shortest_path_triples_used={float(stats['avg_kg_shortest_path_triples_used']):.2f}, "
                f"avg_sub_questions={float(stats['avg_sub_questions']):.2f}, "
                f"insufficient_count={int(stats['insufficient_count'])}, "
                f"insufficient_rate={float(stats['insufficient_rate']):.1%}"
            )
        return "\n".join(lines)
