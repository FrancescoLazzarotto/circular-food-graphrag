"""Pydantic records exchanged between pipeline stages, and the triple JSON schema."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SEED_ONTOLOGY_LABELS = [
    "Organization",
    "Person",
    "Place",
    "Product",
    "Material",
    "Process",
    "Method",
    "Project",
    "Indicator",
    "DataValue",
    "Policy",
    "Document",
    "Event",
    "Concept",
]

_PREDICATE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SectionRecord(BaseModel):
    """A document section delimited by its heading and the next one.

    Attributes:
        title: Heading text, without Markdown emphasis.
        level: Heading level, 1 to 6.
        start_page: 1-based page where the section starts.
        end_page: 1-based page where the section ends.
        start_offset: Character offset in ``start_page`` where the section
            body begins, just after its heading line.
        end_offset: Character offset in ``end_page`` where the next heading
            begins, or ``None`` when the section runs to the end of the page.
            The defaults of both offsets select whole pages.
    """

    model_config = ConfigDict(extra="forbid")
    title: str
    level: int = Field(ge=1, le=6)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    start_offset: int = Field(default=0, ge=0)
    end_offset: int | None = Field(default=None, ge=0)


class PageChunkRecord(BaseModel):
    """The Markdown text of one PDF page.

    Attributes:
        page_number: 1-based page number.
        text: Page content rendered as Markdown.
    """

    model_config = ConfigDict(extra="forbid")
    page_number: int = Field(ge=1)
    text: str


class DocumentRecord(BaseModel):
    """A parsed source document (stage 0 output).

    Attributes:
        doc_id: Identifier derived from the file name.
        filename: PDF file name.
        page_count: Number of pages.
        markdown_text: Full document text, pages joined by blank lines.
        sections: Sections detected from Markdown headings.
        page_chunks: Per-page Markdown text.
        title: Document title, when one could be detected.
        publication_year: Publication year, when one could be detected.
    """

    model_config = ConfigDict(extra="forbid")
    doc_id: str
    filename: str
    page_count: int = Field(ge=1)
    markdown_text: str
    sections: list[SectionRecord] = Field(default_factory=list)
    page_chunks: list[PageChunkRecord] = Field(default_factory=list)
    title: str | None = None
    publication_year: int | None = None


class ChunkRecord(BaseModel):
    """A text window sent to NER and extraction (stage 1 output).

    Attributes:
        doc_id: Identifier of the source document.
        filename: Source PDF file name.
        chunk_id: Unique identifier, ``<doc_id>_chunk_<index>``.
        page_range: Pages covered, formatted ``"start-end"``.
        section_title: Title of the section the chunk belongs to.
        chunk_index: 1-based position of the chunk within its document.
        text: Chunk text.
    """

    model_config = ConfigDict(extra="forbid")
    doc_id: str
    filename: str
    chunk_id: str
    page_range: str
    section_title: str
    chunk_index: int = Field(ge=1)
    text: str


class NEREntityCandidate(BaseModel):
    """An entity span proposed by GLiNER (stage 2 output).

    Attributes:
        text_span: Surface text of the entity.
        entity_label: Ontology label assigned by the model.
        start_char: Start offset in the chunk text.
        end_char: End offset in the chunk text.
        confidence_score: Model score, clamped to ``[0, 1]``.
    """

    model_config = ConfigDict(extra="forbid")
    text_span: str
    entity_label: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    confidence_score: float = Field(ge=0.0, le=1.0)


class KGTriple(BaseModel):
    """A subject-predicate-object triple with node and relationship properties.

    Validation strips entity names, defaults empty label lists to
    ``["Concept"]``, upper-cases the predicate and requires it to be
    ``SCREAMING_SNAKE_CASE``, and fills ``name``, ``source_doc`` and
    ``extraction_method`` when missing.

    Attributes:
        subject: Subject entity name.
        predicate: Relationship type.
        object: Object entity name.
        subject_labels: Node labels of the subject.
        object_labels: Node labels of the object.
        subject_properties: Properties of the subject node.
        object_properties: Properties of the object node.
        relationship_properties: Properties of the relationship, including
            provenance.
        properties: Additional key/value pairs from the sentence context.
    """

    model_config = ConfigDict(extra="forbid")
    subject: str
    predicate: str
    object: str
    subject_labels: list[str]
    object_labels: list[str]
    subject_properties: dict[str, Any]
    object_properties: dict[str, Any]
    relationship_properties: dict[str, Any]
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subject", "object")
    @classmethod
    def _trim_entity_names(cls, value: str) -> str:
        """Strip an entity name and reject it when empty."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("subject/object cannot be empty")
        return cleaned

    @field_validator("subject_labels", "object_labels")
    @classmethod
    def _normalize_labels(cls, labels: list[str]) -> list[str]:
        """Strip labels, drop blank ones, and default to ``["Concept"]``."""
        cleaned = [label.strip() for label in labels if str(label).strip()]
        if not cleaned:
            return ["Concept"]
        return cleaned

    @field_validator("predicate")
    @classmethod
    def _validate_predicate(cls, value: str) -> str:
        """Upper-case the predicate and require ``SCREAMING_SNAKE_CASE``."""
        cleaned = value.strip().upper()
        if not _PREDICATE_RE.fullmatch(cleaned):
            raise ValueError("predicate must be SCREAMING_SNAKE_CASE")
        return cleaned

    @model_validator(mode="after")
    def _normalize_properties(self) -> "KGTriple":
        """Fill node names and relationship provenance when missing."""
        if "name" not in self.subject_properties:
            self.subject_properties["name"] = self.subject
        if "name" not in self.object_properties:
            self.object_properties["name"] = self.object
        if "source_doc" not in self.relationship_properties:
            self.relationship_properties["source_doc"] = ""
        if "extraction_method" not in self.relationship_properties:
            self.relationship_properties["extraction_method"] = "llm"
        return self

    def as_dict(self) -> dict[str, Any]:
        """Return the triple as a plain dict with all nine fields."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "subject_labels": self.subject_labels,
            "object_labels": self.object_labels,
            "subject_properties": self.subject_properties,
            "object_properties": self.object_properties,
            "relationship_properties": self.relationship_properties,
            "properties": self.properties,
        }


class CanonicalEntityRecord(BaseModel):
    """A resolved entity and the aliases merged into it (stage 4 output).

    Attributes:
        canonical_name: Name used for the entity node.
        aliases: Every surface form merged into the entity.
        labels: Node labels of the entity.
        merged_properties: Union of the aliases' properties; the first value
            seen for a key wins.
        alias_sources: Mapping from alias to the documents it appears in.
    """

    model_config = ConfigDict(extra="forbid")
    canonical_name: str
    aliases: list[str]
    labels: list[str]
    merged_properties: dict[str, Any]
    alias_sources: dict[str, list[str]]


def kg_triple_array_schema() -> dict[str, Any]:
    """JSON schema of an array of ``KGTriple`` objects, for structured output.

    Returns:
        A JSON Schema dict passed as ``response_format`` to the LLM server.
    """
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "subject",
                "predicate",
                "object",
                "subject_labels",
                "object_labels",
                "subject_properties",
                "object_properties",
                "relationship_properties",
            ],
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "subject_labels": {"type": "array", "items": {"type": "string"}},
                "object_labels": {"type": "array", "items": {"type": "string"}},
                "subject_properties": {"type": "object"},
                "object_properties": {"type": "object"},
                "relationship_properties": {"type": "object"},
                "properties": {"type": "object"},
            },
        },
    }
