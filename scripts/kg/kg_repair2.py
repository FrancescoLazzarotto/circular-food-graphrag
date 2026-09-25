#!/usr/bin/env python3
"""KG repair pass 2: isolated nodes, relationship types, geographic concepts.

Normally run through ``scripts/kg/kg_postprocess.py``. Its LLM prompts
describe a food-security (FAO/EU) graph.

Steps:
  1. Isolated nodes: merge by name or DETACH DELETE
  2. Deterministic rel-type consolidation (no LLM)
  3. Geographic Concept → Region relabelling / merge
  4. RELATED_TO reclassification (drop DataValue↔DataValue first, then LLM)
  5. Second-round normalization of residual rare rel types via LLM
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from kg_pipeline.utils import neo4j_env
from openai import OpenAI
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_guard import require_confirmation  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg_pipeline.relations import CANONICAL_RELATION_TYPES
from kg_pipeline.utils.validation import parse_json_array

load_dotenv(ROOT / "kg_pipeline" / ".env")

# Resolved once, without raising: the guard in main() is what reports a
# missing setting, and importing this module must not fail for it.
_TARGET = neo4j_env.resolve_target(require=False)
NEO4J_URI = _TARGET.uri
NEO4J_USER = _TARGET.user
NEO4J_PASSWORD = _TARGET.password
NEO4J_DATABASE = _TARGET.database
VLLM_BASE_URL  = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL     = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct-AWQ")
VLLM_API_KEY   = os.getenv("VLLM_API_KEY", "EMPTY")

BATCH_RELATED_TO = 50
BATCH_RESIDUAL   = 100

# One canonical vocabulary for every pass that renames a relationship type: a
# private copy drifts from the others unnoticed.
CANONICAL_VOCAB: list[str] = CANONICAL_RELATION_TYPES
CANONICAL_SET: set[str] = set(CANONICAL_VOCAB)

# UN macro-regions and well-known geographic names to check for Concept→Region relabelling
GEO_REGION_NAMES: list[str] = [
    "Africa", "Asia", "Europe", "North America", "South America", "Oceania",
    "Antarctica", "Americas", "The Americas",
    "Eastern Africa", "Middle Africa", "Northern Africa", "Southern Africa",
    "Western Africa", "Sub-Saharan Africa", "Horn of Africa",
    "Eastern Asia", "South-Eastern Asia", "Southeast Asia", "Southern Asia",
    "Central Asia", "Western Asia", "Middle East",
    "Eastern Europe", "Northern Europe", "Southern Europe", "Western Europe",
    "Central Europe",
    "Latin America and the Caribbean", "Caribbean", "Central America",
    "Northern America", "North America",
    "Melanesia", "Micronesia", "Polynesia", "Australia and New Zealand",
    "Near East", "Far East", "Caucasus",
]

console = Console(force_terminal=True, highlight=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
    force=True,
)
for _h in logging.getLogger().handlers:
    _h.flush = lambda self=_h: (self.stream.flush(), None)[1]  # type: ignore[method-assign]
logger = logging.getLogger("kg_repair2")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunked(lst: list, n: int):
    """Consecutive slices of `lst`, `n` at a time."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _extract_first_json_array(text: str) -> str:
    """The first balanced JSON array in `text`, or ``""``."""
    start = text.find("[")
    if start < 0:
        return ""
    depth, in_string, escaped = 0, False, False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def _llm_json_array(client: OpenAI, prompt: str, max_tokens: int = 4096) -> list[dict[str, Any]]:
    """Ask the model and parse the JSON array of its reply.

    Falls back to the first balanced array in the reply when the whole reply
    does not parse.

    Raises:
        Exception: The parser's error, when no array parses.
    """
    response = client.chat.completions.create(
        model=VLLM_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or "[]"
    try:
        return parse_json_array(content)
    except Exception:
        candidate = _extract_first_json_array(content)
        if candidate:
            return parse_json_array(candidate)
        raise


def _count_rel(session, rel_type: str) -> int:
    """Number of relationships of type `rel_type`."""
    safe = rel_type.replace("`", "")
    r = session.run(f"MATCH ()-[r:`{safe}`]->() RETURN count(r) AS c").single()
    return int(r["c"]) if r else 0


def _rename_rel(session, old: str, new: str) -> int:
    """Rename rel type via apoc.refactor.rename.type; return count of renamed edges."""
    count = _count_rel(session, old)
    if count == 0:
        return 0
    session.run("CALL apoc.refactor.rename.type($old, $new)", old=old, new=new).consume()
    return count


def _invert_rel(session, old: str, new: str) -> int:
    """Invert direction of old rel type and rename it to new."""
    count = _count_rel(session, old)
    if count == 0:
        return 0
    old_safe = old.replace("`", "")
    new_safe = new.replace("`", "")
    session.run(
        f"MATCH (a)-[r:`{old_safe}`]->(b) "
        "WITH a, b, r, properties(r) AS props "
        f"MERGE (b)-[r2:`{new_safe}`]->(a) "
        "SET r2 += props "
        "DELETE r"
    ).consume()
    return count


# ── Step 1: Isolated nodes ────────────────────────────────────────────────────

def step_1_isolated_nodes(session) -> dict[str, Any]:
    """Merge each isolated named node into a connected namesake, or delete it.

    A node with ``source_documents`` and no namesake is kept.

    Returns:
        Counts, up to 20 samples, and the errors.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "merged": 0,
        "deleted": 0,
        "errors": [],
        "samples": [],
    }

    # As in neo4j_postprocess._fix_isolated_nodes: :NodeVec carriers are
    # isolated by design and carry the vector index, and having no `name` they
    # would reach the delete branch and take the index with them.
    rows = session.run(
        "MATCH (n) WHERE NOT (n)--() AND NOT n:NodeVec AND n.name IS NOT NULL "
        "RETURN id(n) AS id, n.name AS name, "
        "       coalesce(n.source_documents, []) AS src_docs, "
        "       labels(n) AS labels"
    ).data()
    report["candidates"] = len(rows)

    if not rows:
        return report

    for row in rows:
        node_id  = int(row["id"])
        name     = str(row.get("name") or "").strip()
        src_docs = row.get("src_docs") or []

        # Try to find a connected node with the same name (toLower+trim)
        match_row = None
        if name:
            match_row = session.run(
                "MATCH (m) "
                "WHERE id(m) <> $id "
                "  AND m.name IS NOT NULL "
                "  AND toLower(trim(m.name)) = toLower(trim($name)) "
                "MATCH (m)-[r]-() "
                "WITH m, count(r) AS degree "
                "ORDER BY degree DESC, id(m) "
                "LIMIT 1 "
                "RETURN id(m) AS id, m.name AS name, degree",
                id=node_id,
                name=name,
            ).single()

        has_source_docs = bool(src_docs)

        if match_row:
            # Merge isolated node into the connected one
            primary_id = int(match_row["id"])
            report["merged"] += 1
            if len(report["samples"]) < 20:
                report["samples"].append({
                    "action": "merge",
                    "from": name,
                    "into": match_row.get("name", ""),
                })
            try:
                session.run(
                    "MATCH (n) WHERE id(n) IN $ids "
                    "WITH n ORDER BY CASE id(n) WHEN $primary THEN 0 ELSE 1 END, id(n) "
                    "WITH collect(n) AS nodes "
                    "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                    "YIELD node RETURN id(node) AS merged_id",
                    ids=[primary_id, node_id],
                    primary=primary_id,
                ).consume()
            except Exception as exc:
                report["errors"].append(f"merge failed for node {node_id} '{name}': {exc}")
        elif has_source_docs:
            # Has source doc info — keep it but log; don't delete
            report["merged"] += 0
            if len(report["samples"]) < 20:
                report["samples"].append({
                    "action": "kept_has_source_docs",
                    "name": name,
                    "src_docs": src_docs[:3],
                })
            logger.info("Isolated node %d '%s' has source_documents — keeping", node_id, name)
        else:
            # No connections, no match, no source_documents → delete
            report["deleted"] += 1
            if len(report["samples"]) < 20:
                report["samples"].append({"action": "deleted", "name": name})
            try:
                session.run("MATCH (n) WHERE id(n) = $id DETACH DELETE n", id=node_id).consume()
            except Exception as exc:
                report["errors"].append(f"delete failed for node {node_id} '{name}': {exc}")

    return report


# ── Step 2: Deterministic rel-type consolidation ──────────────────────────────

def step_2_rel_consolidation(session) -> dict[str, Any]:
    """Apply the fixed relationship-type renames and direction inversions.

    Returns:
        The edges affected per rename and inversion, and the errors.
    """
    report: dict[str, Any] = {"renames": [], "inversions": [], "errors": []}

    # Simple renames: old → new
    renames = [
        ("PUBLISHES",          "PUBLISHED"),
        ("PUBLISHED_DOCUMENT", "PUBLISHED"),
        ("PUBLISHED_WITH",     "PUBLISHED"),
        ("PUBLISHED_IN",       "PUBLISHED"),
        ("HAS_MAX_LEVEL",      "HAS_MAXIMUM_LEVEL"),
        # DEFINED_AS is the canonical type that holds the definition edges.
        ("DEFINITION",         "DEFINED_AS"),
        ("CONTAINS_REGION",    "INCLUDES"),
    ]

    for old, new in renames:
        try:
            cnt = _rename_rel(session, old, new)
            entry = {"from": old, "to": new, "edges_affected": cnt}
            report["renames"].append(entry)
            status = f"→ {cnt} archi" if cnt else "non trovato"
            console.print(f"  rename {old} → {new}: {status}")
        except Exception as exc:
            msg = f"rename {old} → {new} failed: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    # Direction inversions: (a)-[:old]->(b) becomes (b)-[:new]->(a)
    inversions = [
        ("INCLUDED_IN", "INCLUDES"),
    ]

    for old, new in inversions:
        try:
            cnt = _invert_rel(session, old, new)
            entry = {"from": old, "to": new, "direction": "inverted", "edges_affected": cnt}
            report["inversions"].append(entry)
            status = f"→ {cnt} archi invertiti" if cnt else "non trovato"
            console.print(f"  invert {old} → {new} (dir. invertita): {status}")
        except Exception as exc:
            msg = f"invert {old} → {new} failed: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    return report


# ── Step 3: Geographic Concepts ───────────────────────────────────────────────

def _relabel_concept_to_region(session, node_id: int, name: str) -> None:
    """Relabel a Concept node as Region."""
    session.run(
        "MATCH (n:Concept) WHERE id(n) = $id REMOVE n:Concept SET n:Region",
        id=node_id,
    ).consume()


def _merge_concept_into_region(session, region_id: int, concept_id: int) -> None:
    """Merge a Concept node into a Region node, which survives."""
    session.run(
        "MATCH (n) WHERE id(n) IN $ids "
        "WITH n ORDER BY CASE id(n) WHEN $primary THEN 0 ELSE 1 END, id(n) "
        "WITH collect(n) AS nodes "
        "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
        "YIELD node RETURN id(node) AS merged_id",
        ids=[region_id, concept_id],
        primary=region_id,
    ).consume()


def step_3_geographic_concepts(session) -> dict[str, Any]:
    """Turn Concept nodes named after a known region into Region nodes.

    A Concept is merged into an existing Region of the same name, or
    relabelled when there is none.

    Returns:
        Counts, per-node details, and the errors.
    """
    report: dict[str, Any] = {
        "checked": 0,
        "relabeled": 0,
        "merged": 0,
        "not_found": 0,
        "errors": [],
        "details": [],
    }

    for geo_name in GEO_REGION_NAMES:
        norm = geo_name.strip().lower()

        # Is there a Concept node with this name?
        concept_rows = session.run(
            "MATCH (n:Concept) WHERE toLower(trim(n.name)) = $norm RETURN id(n) AS id, n.name AS name",
            norm=norm,
        ).data()

        if not concept_rows:
            continue

        report["checked"] += len(concept_rows)

        for crow in concept_rows:
            concept_id = int(crow["id"])
            cname = str(crow.get("name", ""))

            # Is there already a Region node with the same name?
            region_row = session.run(
                "MATCH (r:Region) WHERE toLower(trim(r.name)) = $norm RETURN id(r) AS id, r.name AS name "
                "ORDER BY id(r) LIMIT 1",
                norm=norm,
            ).single()

            detail: dict[str, Any] = {"name": cname, "concept_id": concept_id}

            if region_row:
                region_id = int(region_row["id"])
                detail["action"] = "merged_into_region"
                detail["region_id"] = region_id
                try:
                    _merge_concept_into_region(session, region_id, concept_id)
                    report["merged"] += 1
                except Exception as exc:
                    msg = f"merge concept '{cname}' into Region failed: {exc}"
                    report["errors"].append(msg)
                    logger.warning(msg)
            else:
                detail["action"] = "relabeled_to_region"
                try:
                    _relabel_concept_to_region(session, concept_id, cname)
                    report["relabeled"] += 1
                except Exception as exc:
                    msg = f"relabel concept '{cname}' to Region failed: {exc}"
                    report["errors"].append(msg)
                    logger.warning(msg)

            report["details"].append(detail)

    return report


# ── Step 4: RELATED_TO reclassification ──────────────────────────────────────

def _fetch_related_to_context(session, rel_ids: list[int]) -> list[dict[str, Any]]:
    """Each RELATED_TO relationship with its endpoints and their neighbours.

    Up to three neighbouring relationships per endpoint go into the prompt as
    context.
    """
    if not rel_ids:
        return []
    query = (
        "UNWIND $ids AS rid "
        "MATCH (s)-[r:RELATED_TO]->(t) WHERE id(r) = rid "
        "CALL { "
        "  WITH s, rid "
        "  MATCH (s)-[rs]-(sn) WHERE id(rs) <> rid "
        "  RETURN collect({type: type(rs), neighbor: coalesce(sn.name,''), "
        "                  labels: labels(sn)})[0..3] AS s_rels "
        "} "
        "CALL { "
        "  WITH t, rid "
        "  MATCH (t)-[rt]-(tn) WHERE id(rt) <> rid "
        "  RETURN collect({type: type(rt), neighbor: coalesce(tn.name,''), "
        "                  labels: labels(tn)})[0..3] AS t_rels "
        "} "
        "RETURN id(r) AS id, "
        "  {labels: labels(s), name: coalesce(s.name,'')} AS source, "
        "  {labels: labels(t), name: coalesce(t.name,'')} AS target, "
        "  s_rels AS source_context, t_rels AS target_context"
    )
    return session.run(query, ids=rel_ids).data()


def _reclass_prompt(vocab: list[str], items: list[dict[str, Any]]) -> str:
    """The prompt asking for the most specific predicate of each item."""
    return (
        "You reclassify Neo4j RELATED_TO relationships to the most specific predicate.\n"
        "Context: knowledge graph about food security (FAO/EU domain).\n"
        "Rules:\n"
        "- Use only the allowed list below.\n"
        "- Prefer the most specific relation; use RELATED_TO only if nothing fits.\n"
        "- Each item has source node (labels+name), target node, and up to 3 neighbouring "
        "relationships per endpoint for context.\n"
        'Return JSON array: [{"id": <int>, "type": "<PREDICATE>"}]\n\n'
        "Allowed predicates:\n"
        f"{json.dumps(vocab, indent=2)}\n\n"
        "Items to reclassify:\n"
        f"{json.dumps(items, indent=2)}"
    )


def step_4_reclassify_related_to(session, client: OpenAI) -> dict[str, Any]:
    """Delete DataValue-DataValue RELATED_TO edges, then retype the rest with the LLM.

    Returns:
        Totals, the count per new type, and the errors.
    """
    report: dict[str, Any] = {
        "datavalue_pairs_deleted": 0,
        "remaining_total": 0,
        "reclassified": 0,
        "kept_related_to": 0,
        "type_counts": {},
        "batches": 0,
        "errors": [],
    }

    # 4a. Delete all (DataValue)-[:RELATED_TO]->(DataValue) edges — no semantic value
    r = session.run(
        "MATCH (s:DataValue)-[rel:RELATED_TO]->(t:DataValue) "
        "DELETE rel RETURN count(rel) AS c"
    ).single()
    deleted_dv = int(r["c"]) if r else 0
    report["datavalue_pairs_deleted"] = deleted_dv
    console.print(f"  DataValue↔DataValue RELATED_TO eliminati: {deleted_dv}")

    # 4b. Fetch remaining RELATED_TO ids
    rows = session.run("MATCH ()-[r:RELATED_TO]->() RETURN id(r) AS id").data()
    rel_ids = [int(r["id"]) for r in rows]
    report["remaining_total"] = len(rel_ids)

    if not rel_ids:
        return report

    vocab_set = set(CANONICAL_VOCAB)
    total_batches = (len(rel_ids) + BATCH_RELATED_TO - 1) // BATCH_RELATED_TO

    for batch_idx, batch_ids in enumerate(_chunked(rel_ids, BATCH_RELATED_TO), start=1):
        report["batches"] += 1
        batch_set = set(batch_ids)

        context_rows = _fetch_related_to_context(session, batch_ids)
        prompt = _reclass_prompt(CANONICAL_VOCAB, context_rows)

        try:
            llm_rows = _llm_json_array(client, prompt)
        except Exception as exc:
            msg = f"Batch {batch_idx}/{total_batches}: LLM call failed: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)
            continue

        updates: list[dict[str, Any]] = []
        kept = 0

        for row in llm_rows:
            try:
                rid = int(row.get("id", -1))
            except (TypeError, ValueError):
                continue
            if rid not in batch_set:
                continue
            rel_type = str(row.get("type", "RELATED_TO")).strip().upper()
            if rel_type not in vocab_set:
                rel_type = "RELATED_TO"
            if rel_type == "RELATED_TO":
                kept += 1
                continue
            updates.append({"id": rid, "type": rel_type})
            report["type_counts"][rel_type] = report["type_counts"].get(rel_type, 0) + 1

        report["kept_related_to"] += kept

        if updates:
            try:
                result = session.run(
                    "UNWIND $updates AS item "
                    "MATCH ()-[r]->() WHERE id(r) = item.id "
                    "CALL apoc.refactor.setType(r, item.type) YIELD output "
                    "RETURN count(output) AS updated",
                    updates=updates,
                ).single()
                updated = int(result["updated"]) if result else 0
                report["reclassified"] += updated
            except Exception as exc:
                msg = f"Batch {batch_idx}/{total_batches}: apoc.refactor.setType failed: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)

        logger.info(
            "RELATED_TO batch %d/%d — processed %d, reclassified %d, kept %d",
            batch_idx, total_batches, len(batch_ids), len(updates), kept,
        )

    return report


# ── Step 5: Residual rare rel types via LLM ──────────────────────────────────

def _residual_mapping_prompt(vocab: list[str], items: list[dict[str, Any]]) -> str:
    """The prompt mapping non-canonical types onto the vocabulary."""
    return (
        "You map non-standard Neo4j relationship types to a fixed canonical vocabulary.\n"
        "Context: knowledge graph about food security (FAO/EU domain).\n"
        "Rules:\n"
        "- Use only the canonical list below.\n"
        "- Prefer the most specific relation; use RELATED_TO only if nothing clearly fits.\n"
        "- Each item has: 'type' (source rel type), 'count' (number of edges), "
        "and 'patterns' (label pairs for endpoints).\n"
        'Return JSON array: [{"source": "<original_type>", "target": "<CANONICAL_TYPE>"}]\n\n'
        "Canonical vocabulary:\n"
        f"{json.dumps(vocab, indent=2)}\n\n"
        "Items to map:\n"
        f"{json.dumps(items, indent=2)}"
    )


def step_5_residual_normalization(session, client: OpenAI) -> dict[str, Any]:
    """Map every non-canonical type with fewer than 10 edges onto the vocabulary.

    Types the LLM does not map become RELATED_TO.

    Returns:
        Counts, the mapping applied, and the errors.
    """
    report: dict[str, Any] = {
        "rare_types_found": 0,
        "already_canonical": 0,
        "to_map": 0,
        "renamed": 0,
        "total_edges_affected": 0,
        "mapping": [],
        "batches": 0,
        "errors": [],
    }

    # Fetch all rel types with < 10 edges
    rows = session.run(
        "MATCH ()-[r]->() "
        "WITH type(r) AS rtype, count(r) AS cnt "
        "WHERE cnt < 10 "
        "WITH rtype, cnt "
        "MATCH (s)-[r2]->(t) WHERE type(r2) = rtype "
        "WITH rtype, cnt, "
        "     collect(DISTINCT {source_labels: labels(s), target_labels: labels(t)})[0..3] AS patterns "
        "RETURN rtype AS type, cnt AS count, patterns "
        "ORDER BY cnt DESC"
    ).data()

    report["rare_types_found"] = len(rows)

    # Filter out already-canonical types
    pending: list[dict[str, Any]] = []
    for row in rows:
        rtype = str(row.get("type", "")).strip().upper()
        if rtype in CANONICAL_SET:
            report["already_canonical"] += 1
            continue
        pending.append({
            "type": rtype,
            "count": int(row.get("count", 0)),
            "patterns": row.get("patterns", []),
        })

    report["to_map"] = len(pending)

    if not pending:
        return report

    total_batches = (len(pending) + BATCH_RESIDUAL - 1) // BATCH_RESIDUAL
    mapping: dict[str, str] = {}

    for batch_idx, batch in enumerate(_chunked(pending, BATCH_RESIDUAL), start=1):
        report["batches"] += 1
        prompt = _residual_mapping_prompt(CANONICAL_VOCAB, batch)

        try:
            llm_rows = _llm_json_array(client, prompt)
        except Exception as exc:
            msg = f"Residual batch {batch_idx}/{total_batches}: LLM call failed: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)
            continue

        for row in llm_rows:
            source = str(row.get("source", "")).strip().upper()
            target = str(row.get("target", "")).strip().upper()
            if not source:
                continue
            if target not in CANONICAL_SET:
                target = "RELATED_TO"
            mapping[source] = target

        logger.info(
            "Residual batch %d/%d — %d types mapped",
            batch_idx, total_batches, len(llm_rows),
        )

    # Fallback: any pending type not returned by LLM → RELATED_TO
    for item in pending:
        if item["type"] not in mapping:
            mapping[item["type"]] = "RELATED_TO"

    # Apply mapping
    for source, target in mapping.items():
        count = next((it["count"] for it in pending if it["type"] == source), 0)
        entry: dict[str, Any] = {
            "from": source,
            "to": target,
            "edges": count,
            "applied": 0,
        }

        if source == target:
            report["mapping"].append(entry)
            continue

        try:
            session.run(
                "CALL apoc.refactor.rename.type($old, $new)",
                old=source,
                new=target,
            ).consume()
            entry["applied"] = count
            report["renamed"] += 1
            report["total_edges_affected"] += count
        except Exception as exc:
            msg = f"Residual rename {source} → {target} failed: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

        report["mapping"].append(entry)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Confirm the target, then run the five steps in one session."""
    require_confirmation(
        title="KG Repair 2",
        what_it_does="""merge near-duplicate entities
        drop nodes left without relationships
        rewrite malformed names in place""",
        uri=NEO4J_URI,
        database=NEO4J_DATABASE,
    )
    console.rule("[bold]KG Repair 2 — Neo4j Aura[/bold]")

    if not NEO4J_URI or not NEO4J_PASSWORD:
        console.print("[red]ERROR: NEO4J credentials missing — check kg_pipeline/.env[/red]")
        sys.exit(1)

    console.print(f"URI   : [cyan]{NEO4J_URI}[/cyan]")
    console.print(f"DB    : [cyan]{NEO4J_DATABASE or '<default>'}[/cyan]")
    console.print(f"vLLM  : [cyan]{VLLM_BASE_URL}[/cyan]  model: [cyan]{VLLM_MODEL}[/cyan]")
    console.print()

    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, timeout=300.0)
    session_kwargs = _TARGET.session_kwargs()

    with neo4j_env.connect(_TARGET) as driver:
        with driver.session(**session_kwargs) as session:

            # ── Step 1
            console.rule("[yellow]Step 1: Nodi isolati[/yellow]")
            r1 = step_1_isolated_nodes(session)
            console.print(f"  Trovati  : {r1['candidates']} nodi isolati")
            console.print(f"  Fusi     : {r1['merged']}")
            console.print(f"  Eliminati: {r1['deleted']}")
            if r1["errors"]:
                for err in r1["errors"][:5]:
                    console.print(f"  [red]{err}[/red]")
            if r1["samples"]:
                console.print("  Campioni:")
                for s in r1["samples"][:8]:
                    console.print(f"    {s}")

            # ── Step 2
            console.rule("[yellow]Step 2: Consolidamento deterministico rel types[/yellow]")
            r2 = step_2_rel_consolidation(session)
            total_r2 = sum(x["edges_affected"] for x in r2["renames"] + r2["inversions"])
            console.print(f"  Rinominati (tipi): {len(r2['renames'])}  |  Invertiti: {len(r2['inversions'])}")
            console.print(f"  Archi totali modificati: {total_r2}")
            if r2["errors"]:
                for err in r2["errors"][:5]:
                    console.print(f"  [red]{err}[/red]")

            # ── Step 3
            console.rule("[yellow]Step 3: Concept geografici → Region[/yellow]")
            r3 = step_3_geographic_concepts(session)
            console.print(f"  Trovati  : {r3['checked']} nodi Concept geografici")
            console.print(f"  Relabeled: {r3['relabeled']}")
            console.print(f"  Fusi     : {r3['merged']}")
            if r3["errors"]:
                for err in r3["errors"][:5]:
                    console.print(f"  [red]{err}[/red]")
            if r3["details"]:
                console.print("  Dettagli:")
                for d in r3["details"][:10]:
                    console.print(f"    {d}")

            # ── Step 4
            console.rule("[yellow]Step 4: RELATED_TO reclassification[/yellow]")
            r4 = step_4_reclassify_related_to(session, client)
            console.print(f"  DataValue↔DataValue eliminati : {r4['datavalue_pairs_deleted']}")
            console.print(f"  RELATED_TO residui            : {r4['remaining_total']}")
            console.print(f"  Riclassificati                : {r4['reclassified']}")
            console.print(f"  Mantenuti come RELATED_TO     : {r4['kept_related_to']}")
            console.print(f"  Batch                         : {r4['batches']}")
            console.print(f"  Errori                        : {len(r4['errors'])}")
            if r4["type_counts"]:
                console.print("  Distribuzione nuovi tipi:")
                for t, c in sorted(r4["type_counts"].items(), key=lambda x: -x[1]):
                    console.print(f"    {t}: {c}")
            for err in r4["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 5
            console.rule("[yellow]Step 5: Normalizzazione residui rari via LLM[/yellow]")
            r5 = step_5_residual_normalization(session, client)
            console.print(f"  Tipi rari trovati   : {r5['rare_types_found']}")
            console.print(f"  Già canonici        : {r5['already_canonical']}")
            console.print(f"  Da mappare via LLM  : {r5['to_map']}")
            console.print(f"  Tipi rinominati     : {r5['renamed']}")
            console.print(f"  Archi totali mossi  : {r5['total_edges_affected']}")
            console.print(f"  Batch               : {r5['batches']}")
            if r5["mapping"]:
                console.print("  Mappatura applicata (prime 20):")
                for m in r5["mapping"][:20]:
                    if m["from"] != m["to"]:
                        console.print(f"    {m['from']} → {m['to']}  ({m['edges']} archi)")
            if r5["errors"]:
                for err in r5["errors"][:5]:
                    console.print(f"  [red]{err}[/red]")

    console.rule("[green]KG Repair 2 completato[/green]")


if __name__ == "__main__":
    main()
