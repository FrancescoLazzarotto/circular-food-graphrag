#!/usr/bin/env python3
"""Run a deeper, local-only pipeline test on a single synthetic document.

This script:
- creates a single ChunkRecord with sample text
- provides a mocked LLM response (no external LLM calls)
- runs stage 3 extraction (using mocked _llm_call_async)
- runs entity resolution (cross-label merging enabled, logs to file)
- performs local quality checks equivalent to the Neo4j queries and
  writes a local `kg_quality_report_local.json` (no Neo4j ingestion)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from pprint import pprint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kg_pipeline.models.types import ChunkRecord
from kg_pipeline.stages import llm_extraction, resolution


def build_mock_response():
    """The canned extraction reply: a JSON array of raw triples."""
    # Triples that exercise the shapes the pipeline has to survive: a subject
    # spelled two ways under two labels, off-vocabulary predicates, an empty
    # object, a data value with properties.
    triples = [
        {
            "subject": "risk analysis",
            "predicate": "ENSURES_HIGH_LEVEL_OF_PROTECTION",
            "object": "consumer safety",
            "subject_labels": ["Method"],
            "object_labels": ["Concept"],
            "subject_properties": {"name": "risk analysis"},
            "object_properties": {"name": "consumer safety"},
            "relationship_properties": {
                "source_doc": "test.pdf",
                "extraction_method": "llm",
            },
        },
        {
            "subject": "Risk Analysis",
            "predicate": "ANALYZES",
            "object": "contaminant levels",
            "subject_labels": ["Concept"],
            "object_labels": ["Indicator"],
            "subject_properties": {"name": "Risk Analysis"},
            "object_properties": {"name": "contaminant levels"},
            "relationship_properties": {
                "source_doc": "test.pdf",
                "extraction_method": "llm",
            },
        },
        {
            "subject": "ACME Corp",
            "predicate": "MAJOR_GLOBAL_TRADER",
            "object": "",
            "subject_labels": ["Organization"],
            "object_labels": [""],
            "subject_properties": {
                "name": "ACME Corp",
                "full_name": "ACME Corporation",
            },
            "object_properties": {"name": ""},
            "relationship_properties": {
                "source_doc": "test.pdf",
                "extraction_method": "llm",
            },
        },
        {
            "subject": "wheat",
            "predicate": "HAS_MAX_LEVEL",
            "object": "5 μg/kg",
            "subject_labels": ["Commodity"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "wheat"},
            "object_properties": {
                "name": "5 μg/kg",
                "value": 5,
                "unit": "μg/kg",
                "contaminant": "Aflatoxin B1",
            },
            "relationship_properties": {
                "source_doc": "test.pdf",
                "extraction_method": "llm",
            },
        },
        {
            "subject": "wheat",
            "predicate": "CONTRIBUTES_TO",
            "object": "food security",
            "subject_labels": ["Commodity"],
            "object_labels": ["Concept"],
            "subject_properties": {"name": "wheat"},
            "object_properties": {"name": "food security"},
            "relationship_properties": {
                "source_doc": "test.pdf",
                "extraction_method": "llm",
            },
        },
    ]
    return json.dumps(triples, ensure_ascii=False)


def local_quality_checks(triples, allowed_preds):
    """Offline version of the post-ingestion quality checks.

    Args:
        triples: Resolved triples.
        allowed_preds: The relation vocabulary.

    Returns:
        Off-vocabulary predicates, names carried under more than one label, and
        up to 20 nodes with at most one neighbour.
    """
    report = {}
    # 1. Predicates out of vocabulary
    out = {}
    for t in triples:
        p = t.predicate
        if p not in allowed_preds:
            out[p] = out.get(p, 0) + 1
    report["predicates_out_of_vocab"] = [
        {"outOfVocab": k, "n": v} for k, v in sorted(out.items(), key=lambda x: -x[1])
    ]

    # 2. Nodes duplicated by name
    name_map = {}
    for t in triples:
        for role, name, labels in [
            ("subject", t.subject, t.subject_labels),
            ("object", t.object, t.object_labels),
        ]:
            if not name:
                continue
            name_map.setdefault(name, set()).update(labels or ["Concept"])
    dup = [
        {"name": n, "labels": list(ls), "count": 1}
        for n, ls in name_map.items()
        if len(ls) > 1
    ]
    report["duplicate_nodes_by_name"] = dup

    # 3. Sparsely connected nodes
    adj = {}
    for t in triples:
        adj.setdefault(t.subject, set()).add(t.object)
        adj.setdefault(t.object, set()).add(t.subject)
    sparse = [
        {"name": n, "degree": len(nei)} for n, nei in adj.items() if len(nei) <= 1
    ]
    report["sparsely_connected_nodes"] = sparse[:20]

    return report


def main():
    """Extract and resolve the synthetic chunk; outputs go to ``artifacts/tmp``."""
    out_dir = Path("artifacts/tmp").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk = ChunkRecord.model_validate(
        {
            "doc_id": "d-test",
            "filename": "test.pdf",
            "chunk_id": "c-test-1",
            "page_range": "1-2",
            "section_title": "Methods",
            "chunk_index": 1,
            "text": "Risk analysis ensures high level of protection. ACME Corp is a major global trader. Wheat has max level 5 μg/kg of Aflatoxin B1.",
        }
    )

    # monkeypatch async llm call to return mocked response
    orig_llm_call_async = llm_extraction._llm_call_async

    async def _mock_llm_call_async(*args, **kwargs):
        """Return the canned reply instead of calling a model."""
        return build_mock_response()

    llm_extraction._llm_call_async = _mock_llm_call_async

    try:
        triples, acr = llm_extraction.extract_triples(
            chunks=[chunk],
            ner_map={chunk.chunk_id: []},
            allowed_labels=[
                "Region",
                "Commodity",
                "Indicator",
                "DataValue",
                "Policy",
                "Organization",
                "Event",
                "Concept",
                "TimePeriod",
                "Document",
                "Dataset",
                "Method",
            ],
            base_url="",
            model_name="mock",
            api_key="",
            max_retries_per_chunk=1,
            temperature=0.0,
            seed=42,
            use_structured_output=False,
            failed_chunks_path=out_dir / "failed_chunks.jsonl",
            new_label_log_path=out_dir / "new_labels.log",
            checkpoint_every=0,
        )

        save_path = out_dir / "test_triples_raw.json"
        with save_path.open("w", encoding="utf-8") as fh:
            json.dump([t.as_dict() for t in triples], fh, ensure_ascii=False, indent=2)

        # resolve entities with cross-label merging and log to file
        registry_log = out_dir / "resolution_crosslabel.log"
        resolved, registry = resolution.resolve_entities(
            triples=triples,
            acronym_map=acr,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.92,
            context_jaccard_floor=0.1,
            base_url=None,
            api_key=None,
            model_name=None,
            crosslabel_log_path=registry_log,
        )

        resolved_path = out_dir / "test_triples_resolved.json"
        with resolved_path.open("w", encoding="utf-8") as fh:
            json.dump([t.as_dict() for t in resolved], fh, ensure_ascii=False, indent=2)

        # perform local quality checks (no Neo4j)
        allowed_preds = {
            "GOVERNS",
            "ESTABLISHES",
            "ESTABLISHED_BY",
            "HAS_COMPONENT",
            "BASED_ON",
            "AFFECTS",
            "CONTRIBUTES_TO",
            "APPLIES_TO",
            "DEFINED_AS",
            "INCLUDES",
            "IS_TYPE_OF",
            "HAS_MAXIMUM_LEVEL",
            "PUBLISHED",
            "WORKED_WITH",
            "EXCHANGES_INFO_WITH",
            "TAKE_INTO_ACCOUNT",
            "ENSURES",
            "SHOULD_BE_MANAGED_BY",
            "AIMS_TO_ACHIEVE",
            "NEEDED_FOR",
            "CONTAINS_DATA",
            "COMPLIES_WITH",
            "ANALYZES",
        }

        report = local_quality_checks(resolved, allowed_preds)
        report_path = out_dir / "kg_quality_report_local.json"
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

        print("Smoke run completed. Outputs:")
        print(" raw triples:", save_path)
        print(" resolved triples:", resolved_path)
        print(" registry log:", registry_log)
        print(" local report:", report_path)
        pprint(report)

    finally:
        llm_extraction._llm_call_async = orig_llm_call_async


if __name__ == "__main__":
    main()
