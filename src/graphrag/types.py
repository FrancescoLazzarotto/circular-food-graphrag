"""Shared types: the LangGraph agent state and the retrieved graph records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict


class Triple(TypedDict):
    """A subject-predicate-object fact as shown to the model.

    Attributes:
        subject: Subject name.
        predicate: Relationship type.
        object: Object name.
    """

    subject: str
    predicate: str
    object: str


class ProvenanceRecord(TypedDict, total=False):
    """The evidence a claim of the answer rests on.

    Attributes:
        claim: The supported claim.
        source_type: Whether the evidence is a text chunk or a graph triple.
        source_id: Identifier of the evidence.
        content: Text of the evidence.
    """

    claim: str
    source_type: Literal["text_chunk", "kg_triple"]
    source_id: str
    content: str


class RAGState(TypedDict, total=False):
    """LangGraph channel schema of the agent.

    A key that a node returns but that is not declared here is silently
    dropped by LangGraph, so every value passed between nodes, or read from
    the final state, must be declared.
    """

    question: str
    run_id: str
    sub_questions: list[str]
    rewritten_question: str
    rewrite_count: int
    # The conversation so far, as plain text with reference tags stripped. Read
    # by the generate node.
    transcript: str
    # Documents cited by the sentences the current question quotes back from an
    # earlier answer. Read by the retrieve node, which floats those documents to
    # the top of the text channel; empty on a turn that quotes nothing.
    quoted_sources: list[str]
    text_context: str
    kg_triples: list[Triple]
    # Retrieved evidence lists, carried from the retrieve node to the final
    # state. The experiment runner serialises them for provenance and answer
    # analysis; the *_count fields below mirror their lengths.
    retrieved_nodes: list[dict[str, Any]]
    retrieved_neighbors: list[dict[str, Any]]
    retrieved_subgraph: list[dict[str, Any]]
    retrieved_shortest_path: list[dict[str, Any]]
    retrieved_text_sources: list[dict[str, Any]]
    retrieved_nodes_count: int
    retrieved_neighbors_count: int
    retrieved_subgraph_count: int
    retrieved_shortest_path_count: int
    kg_context: str
    merged_context: str
    chosen_retrieval_mode: str
    relevance: Literal["relevant", "not_relevant"]
    confidence: float
    confidence_retries: int
    # Numbered, citable evidence for this turn. Serialised as plain dicts so
    # the state stays JSON-dumpable for the experiment runner; rebuild the
    # dataclasses with graphrag.agent.evidence.evidence_from_dicts.
    evidence_index: list[dict[str, Any]]
    # Reference ids that survived context compression, i.e. the blocks the model
    # was actually shown. The citation gate validates against these.
    visible_evidence_refs: list[str]
    citation_report: dict[str, Any]
    quote_report: dict[str, Any]
    answer: str
    # The answer before the refusal-rescue retry, and whether that retry fired.
    # Abstention is measured on the pre-retry text.
    pre_retry_answer: str
    refusal_retry_applied: bool
    # Domain gate. `in_domain` is False only when the gate ran and rejected the
    # question; `out_of_scope` marks the answer as the fixed refusal, so callers
    # can tell an abstention from a generated answer without parsing prose.
    # `follow_up` exempts a question that continues an already-admitted topic.
    in_domain: bool
    out_of_scope: bool
    # The refusal is an introduction, not a refusal: the question was about the
    # assistant itself. Lets the surfaces title it as such instead of telling
    # someone who typed "ciao" that they are out of scope.
    meta_question: bool
    # Which language that introduction is written in. Carried in the state
    # rather than re-detected in the terminal node: two words are not enough
    # for `_detect_query_language`, but the pattern that matched knows.
    meta_language: str
    follow_up: bool
    provenance: list[ProvenanceRecord]
    reflection_passed: bool
    reflection_feedback: str
    strategy: str
    latency_ms: float
    node_timings: dict[str, float]


class KGNode(TypedDict, total=False):
    """A node retrieved from the graph.

    Attributes:
        node_id: Neo4j element id.
        labels: Node labels.
        properties: Node properties, without the embedding vector.
        text: Display name: the first set name property, else the element id.
    """

    node_id: str
    labels: list[str]
    properties: dict[str, Any]
    text: str


class KGTriple(TypedDict, total=False):
    """A triple retrieved from the graph, with its endpoints' ids and data.

    Attributes:
        subject_id: Element id of the subject node.
        subject: Subject name.
        predicate: The relationship's ``predicate`` property, else its type.
        object_id: Element id of the object node.
        object: Object name.
        subject_labels: Labels of the subject node.
        object_labels: Labels of the object node.
        subject_properties: Properties of the subject node.
        object_properties: Properties of the object node.
        relationship_properties: Properties of the relationship.
    """

    subject_id: str
    subject: str
    predicate: str
    object_id: str
    object: str
    subject_labels: list[str]
    object_labels: list[str]
    subject_properties: dict[str, Any]
    object_properties: dict[str, Any]
    relationship_properties: dict[str, Any]

def triple_key(triple: Mapping[str, Any]) -> tuple[str, str, str]:
    """Identity of a triple for de-duplication.

    Element ids when both endpoints carry one, surface forms otherwise: the
    same fact retrieved by two queries must collapse onto one entry, and two
    nodes that happen to share a name must not. The agent, the retriever and
    the experiment runner all de-duplicate with this function, so the notion
    of "same triple" is identical across a turn.

    Args:
        triple: A retrieved triple, with ``subject``/``predicate``/``object``
            and optionally ``subject_id``/``object_id``.

    Returns:
        The key, id-based when both ids are present.
    """
    subject_id = str(triple.get("subject_id", "")).strip()
    object_id = str(triple.get("object_id", "")).strip()
    predicate = str(triple.get("predicate", "")).strip().lower()

    if subject_id and object_id:
        return (f"id:{subject_id}", predicate, f"id:{object_id}")

    subject = str(triple.get("subject", "")).strip().lower()
    obj = str(triple.get("object", "")).strip().lower()
    return (subject, predicate, obj)
