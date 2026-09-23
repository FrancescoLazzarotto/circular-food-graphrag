"""Prompt for stage 3 triple extraction."""

from __future__ import annotations

import json
from pathlib import Path

from kg_pipeline.models.types import ChunkRecord


# Fallback when no relation_vocab is passed: read the canonical vocab file so
# the prompt can never drift from what validation enforces.
_VOCAB_PATH = Path(__file__).resolve().parents[1] / "relation_vocab_circular_v1_draft.json"
_DEFAULT_RELATION_VOCAB: list[str] = json.loads(
    _VOCAB_PATH.read_text(encoding="utf-8")
)


def build_extraction_prompt(
    chunk: ChunkRecord,
    candidate_entities: list[dict],
    allowed_labels: list[str],
    relation_vocab: list[str] | None = None,
) -> str:
    """Build the extraction prompt for one chunk.

    The prompt carries the extraction rules, the allowed labels and relation
    vocabulary, four few-shot triples, the chunk metadata, the GLiNER
    candidates and the chunk text.

    Args:
        chunk: Chunk to extract from.
        candidate_entities: GLiNER candidates for the chunk, as dicts.
        allowed_labels: Ontology labels the model may use.
        relation_vocab: Allowed predicates. When ``None`` or empty, the
            vocabulary in ``relation_vocab_circular_v1_draft.json`` is used.

    Returns:
        The prompt text, stripped.
    """
    canonical_vocab = relation_vocab or _DEFAULT_RELATION_VOCAB
    canonical_vocab = [
        str(item).strip().upper() for item in canonical_vocab if str(item).strip()
    ]
    few_shot = [
        # Qualitative fact, English source: a project implementing a process.
        {
            "subject": "RePoPP",
            "predicate": "IMPLEMENTS",
            "object": "surplus food recovery",
            "subject_labels": ["Project"],
            "object_labels": ["Process"],
            "subject_properties": {"name": "RePoPP"},
            "object_properties": {"name": "surplus food recovery"},
            "relationship_properties": {
                "source_doc": "example_paper.pdf",
                "extraction_method": "llm",
                "confidence": 0.9,
            },
        },
        # Italian source text: entity names stay in Italian, labels and
        # predicate stay in English.
        {
            "subject": "economia circolare",
            "predicate": "REDUCES",
            "object": "spreco alimentare",
            "subject_labels": ["Concept"],
            "object_labels": ["Process"],
            "subject_properties": {"name": "economia circolare"},
            "object_properties": {"name": "spreco alimentare"},
            "relationship_properties": {
                "source_doc": "esempio_rapporto.pdf",
                "extraction_method": "llm",
                "confidence": 0.85,
            },
        },
        # Quantitative fact: year-scoped subject node plus numeric DataValue
        # with value/unit/year both on the node and on the relationship.
        {
            "subject": "food waste per capita Italy 2022",
            "predicate": "HAS_VALUE",
            "object": "67 kg per year",
            "subject_labels": ["Indicator"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "food waste per capita Italy 2022"},
            "object_properties": {
                "name": "67 kg per year",
                "value": 67,
                "unit": "kg/year",
                "year": 2022,
            },
            "relationship_properties": {
                "source_doc": "example_paper.pdf",
                "extraction_method": "llm",
                "value": 67,
                "unit": "kg/year",
                "year": 2022,
                "confidence": 0.9,
            },
        },
        # Collaboration between two organizations.
        {
            "subject": "Slow Food",
            "predicate": "COLLABORATES_WITH",
            "object": "Università di Scienze Gastronomiche",
            "subject_labels": ["Organization"],
            "object_labels": ["Organization"],
            "subject_properties": {"name": "Slow Food"},
            "object_properties": {"name": "Università di Scienze Gastronomiche"},
            "relationship_properties": {
                "source_doc": "esempio_rapporto.pdf",
                "extraction_method": "llm",
                "confidence": 0.9,
            },
        },
    ]

    prompt = f"""
You are an information extraction system for documents about circular economy and food systems
(academic papers, books, reports, magazines). The text may be in ENGLISH or ITALIAN.

Task:
1) Validate and correct candidate entities from GLiNER.
2) Extract all semantic and quantitative relationships from the chunk.
3) Add entities that GLiNER missed when necessary.
4) Return only a JSON array of KGTriple dictionaries.

Language rules:
- Keep entity names (subject/object) in the ORIGINAL language of the text - do not translate.
- Labels and predicates are ALWAYS in English, from the allowed lists below.

CRITICAL Validation rules (DO NOT VIOLATE):
- SUBJECT and OBJECT must NEVER be empty strings - always have a value (1+ chars).
- SUBJECT and OBJECT must be stripped of leading/trailing whitespace.
- PREDICATE must be ALL UPPERCASE letters, numbers, and underscores (e.g., HAS_VALUE, REDUCES).
- PREDICATE must start with a letter (not a number).
- PREDICATE must be 1-4 words in SCREAMING_SNAKE_CASE format (max 50 chars total).
- If a relationship cannot be fully extracted, SKIP IT - do not emit triples with empty fields.
- If you cannot find a valid OBJECT, SKIP the relationship - do not use empty string.

Additional guidelines:
- Allowed labels: {json.dumps(allowed_labels)}
- Canonical relation vocabulary (use ONLY these predicate types):
{json.dumps(canonical_vocab, ensure_ascii=False, indent=2)}
- You may introduce a new label only if none of the allowed labels fits.
- Do not use meta labels like Email, PageRange, SectionTitle, Chunk, Grant, Identifier.
- Document label is only for whole documents or cited publications (books, papers, reports),
  not section headings.
- Avoid generic predicates like RELATED_TO and do not emit MENTIONED_IN (added later by the system).
- Skip editorial boilerplate: author affiliations, journal mastheads, page footers,
  citation lists, DOIs, copyright notices. Extract facts, not publishing metadata.
- Output must be pure JSON array only. No prose.
- Include numeric relationship attributes in relationship_properties when present
    (for example value, unit, year).
- If possible, add relationship_properties.confidence as a float between 0 and 1.
- relationship_properties must always include source_doc and extraction_method.
- extraction_method must be "llm".

Connect the entities of this passage to each other:
- after extracting the explicit relations, read back the list of entities you
  produced and add the relations between TWO ENTITIES YOU ALREADY NAMED that
  the passage asserts and that you left out;
- prefer reusing an entity you already named over coining a new one with
  different words for the same thing;
- do not invent relations the passage does not assert: two entities merely
  standing near each other in the text are not related.

Predicate guidance:
- Circular-economy actions: use REDUCES, REUSES, RECYCLES, GENERATES for waste/material flows
  (e.g. (compostaggio, RECYCLES, scarti organici)).
- Use IMPLEMENTS for projects/initiatives put into practice by actors; FUNDED_BY for financing.
- Use PUBLISHED for publications (Organization PUBLISHED Document) and AUTHORED_BY for
  authorship (Document AUTHORED_BY Person).
- Use RELATED_TO only if no other canonical predicate fits.

Temporal and quantitative facts:
- Use HAS_VALUE ONLY for numeric, measurable quantities: numbers, percentages, counts, physical
  units (kg, t, %, ha, million tonnes, euro, etc.). Do NOT use HAS_VALUE for qualitative
  statements, goals, or trends without a concrete number.
- When the text states a NUMERIC value for a specific year or period, create a DISTINCT
  subject node that includes the year (see few-shot example 3), with subject_labels=["Indicator"],
  object_labels=["DataValue"], predicate="HAS_VALUE".
- Always set object_properties.value (numeric), object_properties.unit, object_properties.year
  (when stated) on DataValue nodes; mirror them in relationship_properties.
- Skip HAS_VALUE if you cannot populate a numeric value field.

PROPERTY RULES - the following must NEVER become relationships.
Encode them as properties of the subject node instead:
- Titles, full names, acronym expansions -> add property "full_name" or "title"
    to the subject node
- Role descriptors (e.g. "founder", "coordinator") -> add property "role"
- Membership of a geographic area -> add property "region"

Output schema note: each KGTriple object may include an optional top-level
"properties" dictionary for additional key/value pairs extracted from the
sentence context. Example triple schema:
{{
    "subject": "...",
    "subject_labels": ["..."],
    "predicate": "...",
    "object": "...",
    "object_labels": ["..."],
    "properties": {{}},
    "subject_properties": {{...}},
    "object_properties": {{...}},
    "relationship_properties": {{...}}
}}

Chunk metadata:
- filename: {chunk.filename}
- section_title: {chunk.section_title}
- chunk_id: {chunk.chunk_id}
- page_range: {chunk.page_range}

Few-shot examples (these are CORRECT format - follow them):
{json.dumps(few_shot, ensure_ascii=False, indent=2)}

Candidate entities from GLiNER:
{json.dumps(candidate_entities, ensure_ascii=False, indent=2)}

Chunk markdown:
{chunk.text}

Remember: Do not use empty strings for subject or object. If you cannot identify a complete relationship, skip it.
"""
    return prompt.strip()
