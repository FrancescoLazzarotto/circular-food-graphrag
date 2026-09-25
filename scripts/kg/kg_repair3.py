#!/usr/bin/env python3
"""KG repair pass 3: RELATED_TO by pattern, PUBLISHED, micro-types, residuals.

Normally run through ``scripts/kg/kg_postprocess.py``. Its LLM prompts
describe a food-security (FAO/EU) graph.

Steps:
  1. Deterministic RELATED_TO reclassification by endpoint pattern (no LLM),
     then LLM reclassification for residual Concept↔Concept / Org→Concept
  2. Unify PUBLISHED_BY → PUBLISHED (invert direction, merge properties)
  3. Deterministic micro-type consolidation
  4. Fix "High-food-budget countries" Commodity → Region
  5. Third-round residual normalization via LLM (< 5 edges, non-canonical);
     no RELATED_TO fallback — unmappable edges are deleted
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

console = Console(force_terminal=True, highlight=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
    force=True,
)
for _h in logging.getLogger().handlers:
    _h.flush = lambda self=_h: (self.stream.flush(), None)[1]  # type: ignore[method-assign]
logger = logging.getLogger("kg_repair3")


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
    """Rename relationship type `old` to `new`; returns how many were renamed."""
    count = _count_rel(session, old)
    if count == 0:
        return 0
    session.run("CALL apoc.refactor.rename.type($old, $new)", old=old, new=new).consume()
    return count


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


# ── Step 1: RELATED_TO reclassification ──────────────────────────────────────

# Pattern → canonical type; None means delete the edge
DETERMINISTIC_RELATED_TO: dict[tuple[str, str], str | None] = {
    ("Region",       "DataValue"):   "MEASURES",
    ("Organization", "Organization"): "WORKED_WITH",
    ("DataValue",    "DataValue"):   None,          # delete
    ("Indicator",    "DataValue"):   "MEASURES",
    ("Region",       "Concept"):     "AFFECTS",
}


def step_1_reclassify_related_to(session, client: OpenAI) -> dict[str, Any]:
    """Retype RELATED_TO by endpoint pattern, then the rest with the LLM.

    Returns:
        Counts per pattern, LLM totals, and the errors.
    """
    report: dict[str, Any] = {
        "deterministic": {},
        "deleted_deterministic": 0,
        "llm_reclassified": 0,
        "llm_kept_related_to": 0,
        "llm_batches": 0,
        "errors": [],
    }

    # 1a. Deterministic patterns
    for (src_label, tgt_label), new_type in DETERMINISTIC_RELATED_TO.items():
        try:
            if new_type is None:
                r = session.run(
                    f"MATCH (s:{src_label})-[rel:RELATED_TO]->(t:{tgt_label}) "
                    "DELETE rel RETURN count(rel) AS c"
                ).single()
                cnt = int(r["c"]) if r else 0
                report["deleted_deterministic"] += cnt
                console.print(f"  DELETE ({src_label})-[RELATED_TO]->({tgt_label}): {cnt} archi eliminati")
            else:
                r = session.run(
                    f"MATCH (s:{src_label})-[rel:RELATED_TO]->(t:{tgt_label}) "
                    "WITH s, t, rel, properties(rel) AS props "
                    f"CALL apoc.refactor.setType(rel, $new_type) YIELD output "
                    "RETURN count(output) AS c",
                    new_type=new_type,
                ).single()
                cnt = int(r["c"]) if r else 0
                report["deterministic"][f"({src_label})->({tgt_label})"] = {
                    "to": new_type,
                    "count": cnt,
                }
                console.print(
                    f"  ({src_label})-[RELATED_TO]->({tgt_label}) → {new_type}: {cnt} archi"
                )
        except Exception as exc:
            msg = f"deterministic remap ({src_label})→({tgt_label}): {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    # 1b. LLM reclassification for remaining RELATED_TO
    rows = session.run("MATCH ()-[r:RELATED_TO]->() RETURN id(r) AS id").data()
    rel_ids = [int(r["id"]) for r in rows]
    console.print(f"  RELATED_TO residui dopo deterministic: {len(rel_ids)}")

    if not rel_ids:
        return report

    vocab_set = CANONICAL_SET
    total_batches = (len(rel_ids) + BATCH_RELATED_TO - 1) // BATCH_RELATED_TO

    for batch_idx, batch_ids in enumerate(_chunked(rel_ids, BATCH_RELATED_TO), start=1):
        report["llm_batches"] += 1
        batch_set = set(batch_ids)
        context_rows = _fetch_related_to_context(session, batch_ids)
        prompt = (
            "You reclassify Neo4j RELATED_TO relationships to the most specific predicate.\n"
            "Context: knowledge graph about food security (FAO/EU domain).\n"
            "Rules:\n"
            "- Use only the allowed list below.\n"
            "- Prefer the most specific relation; use RELATED_TO only if nothing fits.\n"
            "- Each item has source node (labels+name), target node, and up to 3 neighbouring "
            "relationships per endpoint for context.\n"
            'Return JSON array: [{"id": <int>, "type": "<PREDICATE>"}]\n\n'
            "Allowed predicates:\n"
            f"{json.dumps(CANONICAL_VOCAB, indent=2)}\n\n"
            "Items to reclassify:\n"
            f"{json.dumps(context_rows, indent=2)}"
        )
        try:
            llm_rows = _llm_json_array(client, prompt)
        except Exception as exc:
            msg = f"LLM batch {batch_idx}/{total_batches} failed: {exc}"
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

        report["llm_kept_related_to"] += kept

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
                report["llm_reclassified"] += updated
            except Exception as exc:
                msg = f"apoc.refactor.setType batch {batch_idx}/{total_batches}: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)

        logger.info(
            "RELATED_TO LLM batch %d/%d — processed %d, reclassified %d, kept %d",
            batch_idx, total_batches, len(batch_ids), len(updates), kept,
        )

    return report


# ── Step 2: Unify PUBLISHED_BY → PUBLISHED ────────────────────────────────────

def step_2_unify_published(session) -> dict[str, Any]:
    """Invert Document-[PUBLISHED_BY]->Organization into Organization-[PUBLISHED].

    Any other PUBLISHED_BY left is renamed to PUBLISHED in place.

    Returns:
        Counts and errors.
    """
    report: dict[str, Any] = {"inverted": 0, "errors": []}

    # Fetch all (Document)-[PUBLISHED_BY]->(Organization) edges
    rows = session.run(
        "MATCH (doc:Document)-[r:PUBLISHED_BY]->(org:Organization) "
        "RETURN id(r) AS rid, id(doc) AS doc_id, id(org) AS org_id, properties(r) AS props"
    ).data()
    console.print(f"  (Document)-[PUBLISHED_BY]->(Organization) trovati: {len(rows)}")

    for row in rows:
        doc_id = int(row["doc_id"])
        org_id = int(row["org_id"])
        rid    = int(row["rid"])
        props  = row.get("props") or {}
        try:
            session.run(
                "MATCH (doc) WHERE id(doc) = $doc_id "
                "MATCH (org) WHERE id(org) = $org_id "
                "MERGE (org)-[r2:PUBLISHED]->(doc) "
                "SET r2 += $props "
                "WITH r2 "
                "MATCH ()-[old]->() WHERE id(old) = $rid "
                "DELETE old",
                doc_id=doc_id,
                org_id=org_id,
                rid=rid,
                props=props,
            ).consume()
            report["inverted"] += 1
        except Exception as exc:
            msg = f"invert PUBLISHED_BY edge {rid}: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    # Also handle any remaining non-Document/Organization PUBLISHED_BY edges
    # by renaming them to PUBLISHED (direction stays, semantics differ but better than noise)
    remaining = _count_rel(session, "PUBLISHED_BY")
    if remaining > 0:
        console.print(f"  PUBLISHED_BY residui (non-Doc→Org): {remaining} → rinominati PUBLISHED")
        try:
            session.run(
                "CALL apoc.refactor.rename.type('PUBLISHED_BY', 'PUBLISHED')"
            ).consume()
            report["remaining_renamed"] = remaining
        except Exception as exc:
            msg = f"rename residual PUBLISHED_BY: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    return report


# ── Step 3: Deterministic micro-type consolidation ────────────────────────────

def step_3_micro_consolidation(session) -> dict[str, Any]:
    """Apply the fixed micro-type renames and the ASSESSED_IN inversion.

    Returns:
        The edges affected per rename and inversion, and the errors.
    """
    report: dict[str, Any] = {"renames": [], "inversions": [], "errors": []}

    # Simple renames
    renames = [
        # HAS_PERCENTAGE only matters to an extraction run that produces it; it
        # goes to HAS_VALUE because percentages already live there, as values
        # with `unit = '%'`.
        ("HAS_PERCENTAGE", "HAS_VALUE"),
        ("AUTHORED",       "PUBLISHED"),
        ("TARGETS",        "AFFECTS"),
        ("ENSURES",        "REQUIRES"),
    ]
    for old, new in renames:
        try:
            cnt = _rename_rel(session, old, new)
            report["renames"].append({"from": old, "to": new, "count": cnt})
            status = f"{cnt} archi" if cnt else "non trovato"
            console.print(f"  rename {old} → {new}: {status}")
        except Exception as exc:
            msg = f"rename {old} → {new}: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    # ASSESSED_IN: (X)-[ASSESSED_IN]->(Y) should become (Y)-[ANALYZES]->(X)
    try:
        r = session.run("MATCH ()-[r:ASSESSED_IN]->() RETURN count(r) AS c").single()
        cnt = int(r["c"]) if r else 0
        if cnt > 0:
            session.run(
                "MATCH (a)-[r:ASSESSED_IN]->(b) "
                "WITH a, b, properties(r) AS props "
                "MERGE (b)-[r2:ANALYZES]->(a) "
                "SET r2 += props "
                "DELETE r"
            ).consume()
            report["inversions"].append({"from": "ASSESSED_IN", "to": "ANALYZES", "count": cnt})
            console.print(f"  invert ASSESSED_IN → ANALYZES: {cnt} archi")
        else:
            console.print("  ASSESSED_IN: non trovato")
    except Exception as exc:
        msg = f"invert ASSESSED_IN → ANALYZES: {exc}"
        report["errors"].append(msg)
        logger.warning(msg)

    return report


# ── Step 4: Fix "High-food-budget countries" label ───────────────────────────

def step_4_fix_commodity_label(session) -> dict[str, Any]:
    """Make "High-food-budget countries" a Region instead of a Commodity.

    Returns:
        The action taken and the errors.
    """
    report: dict[str, Any] = {"action": None, "errors": []}
    norm = "high-food-budget countries"

    commodity_rows = session.run(
        "MATCH (n:Commodity) WHERE toLower(trim(n.name)) = $norm RETURN id(n) AS id, n.name AS name",
        norm=norm,
    ).data()

    if not commodity_rows:
        console.print("  'High-food-budget countries' Commodity: non trovato")
        report["action"] = "not_found"
        return report

    for crow in commodity_rows:
        concept_id = int(crow["id"])
        cname = str(crow.get("name", ""))

        region_row = session.run(
            "MATCH (r:Region) WHERE toLower(trim(r.name)) = $norm RETURN id(r) AS id LIMIT 1",
            norm=norm,
        ).single()

        if region_row:
            region_id = int(region_row["id"])
            try:
                session.run(
                    "MATCH (n) WHERE id(n) IN $ids "
                    "WITH n ORDER BY CASE id(n) WHEN $primary THEN 0 ELSE 1 END, id(n) "
                    "WITH collect(n) AS nodes "
                    "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                    "YIELD node RETURN id(node) AS merged_id",
                    ids=[region_id, concept_id],
                    primary=region_id,
                ).consume()
                report["action"] = "merged_into_existing_region"
                console.print(f"  '{cname}' fuso nel nodo Region esistente (id={region_id})")
            except Exception as exc:
                msg = f"merge '{cname}' into Region: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)
        else:
            try:
                session.run(
                    "MATCH (n:Commodity) WHERE id(n) = $id REMOVE n:Commodity SET n:Region",
                    id=concept_id,
                ).consume()
                report["action"] = "relabeled_to_region"
                console.print(f"  '{cname}' relabeled Commodity → Region")
            except Exception as exc:
                msg = f"relabel '{cname}' to Region: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)

    return report


# ── Step 5: Third-round residual normalization via LLM ───────────────────────

def _residual_prompt_no_fallback(vocab: list[str], items: list[dict[str, Any]]) -> str:
    """The prompt mapping rare types onto the vocabulary, or to deletion."""
    return (
        "You map non-standard Neo4j relationship types to a fixed canonical vocabulary.\n"
        "Context: knowledge graph about food security (FAO/EU domain).\n"
        "Rules:\n"
        "- Use only the canonical list below.\n"
        "- Prefer the most specific relation.\n"
        "- If the source type is not clearly mappable to any canonical type, return null as target "
        "(the edges will be deleted — do NOT use RELATED_TO as a catch-all).\n"
        "- Each item has: 'type' (source rel type), 'count' (number of edges), "
        "and 'patterns' (label pairs for endpoints).\n"
        'Return JSON array: [{"source": "<original_type>", "target": "<CANONICAL_TYPE_or_null>"}]\n\n'
        "Canonical vocabulary:\n"
        f"{json.dumps(vocab, indent=2)}\n\n"
        "Items to map:\n"
        f"{json.dumps(items, indent=2)}"
    )


def step_5_residual_normalization(session, client: OpenAI) -> dict[str, Any]:
    """Map every non-canonical type with fewer than 5 edges, or delete its edges.

    Types the LLM does not map are deleted, not turned into RELATED_TO.

    Returns:
        Counts, the mapping applied, and the errors.
    """
    report: dict[str, Any] = {
        "rare_types_found": 0,
        "already_canonical": 0,
        "to_map": 0,
        "renamed": 0,
        "deleted_types": 0,
        "total_edges_renamed": 0,
        "total_edges_deleted": 0,
        "mapping": [],
        "batches": 0,
        "errors": [],
    }

    rows = session.run(
        "MATCH ()-[r]->() "
        "WITH type(r) AS rtype, count(r) AS cnt "
        "WHERE cnt < 5 "
        "WITH rtype, cnt "
        "MATCH (s)-[r2]->(t) WHERE type(r2) = rtype "
        "WITH rtype, cnt, "
        "     collect(DISTINCT {source_labels: labels(s), target_labels: labels(t)})[0..3] AS patterns "
        "RETURN rtype AS type, cnt AS count, patterns "
        "ORDER BY cnt DESC"
    ).data()

    report["rare_types_found"] = len(rows)

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
    mapping: dict[str, str | None] = {}

    for batch_idx, batch in enumerate(_chunked(pending, BATCH_RESIDUAL), start=1):
        report["batches"] += 1
        prompt = _residual_prompt_no_fallback(CANONICAL_VOCAB, batch)
        try:
            llm_rows = _llm_json_array(client, prompt)
        except Exception as exc:
            msg = f"Residual batch {batch_idx}/{total_batches}: LLM failed: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)
            continue

        for row in llm_rows:
            source = str(row.get("source", "")).strip().upper()
            target_raw = row.get("target")
            if not source:
                continue
            if target_raw is None or str(target_raw).strip().lower() in ("null", "none", ""):
                mapping[source] = None
            else:
                target = str(target_raw).strip().upper()
                mapping[source] = target if target in CANONICAL_SET else None

        logger.info(
            "Residual batch %d/%d — %d types mapped", batch_idx, total_batches, len(llm_rows)
        )

    # Any pending type not returned by LLM → delete (no RELATED_TO fallback)
    for item in pending:
        if item["type"] not in mapping:
            mapping[item["type"]] = None

    # Apply: rename or delete
    for source, target in mapping.items():
        count = next((it["count"] for it in pending if it["type"] == source), 0)
        entry: dict[str, Any] = {"from": source, "to": target, "edges": count, "applied": 0}

        if target is not None and source == target:
            report["mapping"].append(entry)
            continue

        if target is None:
            # Delete all edges of this type
            try:
                safe = source.replace("`", "")
                r = session.run(
                    f"MATCH ()-[r:`{safe}`]->() DELETE r RETURN count(r) AS c"
                ).single()
                deleted = int(r["c"]) if r else 0
                entry["applied"] = deleted
                report["deleted_types"] += 1
                report["total_edges_deleted"] += deleted
                console.print(f"  DELETE {source}: {deleted} archi eliminati")
            except Exception as exc:
                msg = f"delete {source}: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)
        else:
            try:
                session.run(
                    "CALL apoc.refactor.rename.type($old, $new)", old=source, new=target
                ).consume()
                entry["applied"] = count
                report["renamed"] += 1
                report["total_edges_renamed"] += count
                console.print(f"  rename {source} → {target}: {count} archi")
            except Exception as exc:
                msg = f"rename {source} → {target}: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)

        report["mapping"].append(entry)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Confirm the target, then run the five steps in one session."""
    require_confirmation(
        title="KG Repair 3",
        what_it_does="""reverse relationships stored the wrong way round
        reclassify generic relationships into specific types
        delete relationships that survive neither check""",
        uri=NEO4J_URI,
        database=NEO4J_DATABASE,
    )
    console.rule("[bold]KG Repair 3 — Neo4j Aura[/bold]")

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
            console.rule("[yellow]Step 1: RELATED_TO reclassification[/yellow]")
            r1 = step_1_reclassify_related_to(session, client)
            console.print(f"  Eliminati (deterministic):    {r1['deleted_deterministic']}")
            console.print(f"  LLM reclassificati           : {r1['llm_reclassified']}")
            console.print(f"  LLM mantenuti RELATED_TO     : {r1['llm_kept_related_to']}")
            console.print(f"  Batch LLM                    : {r1['llm_batches']}")
            if r1["deterministic"]:
                console.print("  Mapping deterministici:")
                for pat, info in r1["deterministic"].items():
                    console.print(f"    {pat} → {info['to']}: {info['count']}")
            for err in r1["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 2
            console.rule("[yellow]Step 2: Unifica PUBLISHED_BY → PUBLISHED[/yellow]")
            r2 = step_2_unify_published(session)
            console.print(f"  Archi invertiti (Doc→Org → Org→Doc): {r2['inverted']}")
            if r2.get("remaining_renamed"):
                console.print(f"  Residui rinominati: {r2['remaining_renamed']}")
            for err in r2["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 3
            console.rule("[yellow]Step 3: Micro-consolidamento deterministico[/yellow]")
            r3 = step_3_micro_consolidation(session)
            total_r3 = sum(x["count"] for x in r3["renames"] + r3["inversions"])
            console.print(f"  Rinominati (tipi): {len(r3['renames'])}  |  Invertiti: {len(r3['inversions'])}")
            console.print(f"  Archi totali modificati: {total_r3}")
            for err in r3["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 4
            console.rule("[yellow]Step 4: Fix 'High-food-budget countries' Commodity → Region[/yellow]")
            r4 = step_4_fix_commodity_label(session)
            console.print(f"  Azione: {r4['action']}")
            for err in r4["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 5
            console.rule("[yellow]Step 5: Normalizzazione residui rari via LLM (no fallback)[/yellow]")
            r5 = step_5_residual_normalization(session, client)
            console.print(f"  Tipi rari trovati        : {r5['rare_types_found']}")
            console.print(f"  Già canonici             : {r5['already_canonical']}")
            console.print(f"  Da mappare via LLM       : {r5['to_map']}")
            console.print(f"  Tipi rinominati          : {r5['renamed']}")
            console.print(f"  Archi rinominati         : {r5['total_edges_renamed']}")
            console.print(f"  Tipi eliminati           : {r5['deleted_types']}")
            console.print(f"  Archi eliminati          : {r5['total_edges_deleted']}")
            console.print(f"  Batch                    : {r5['batches']}")
            if r5["mapping"]:
                console.print("  Mappatura applicata (prime 20):")
                for m in r5["mapping"][:20]:
                    arrow = f"→ {m['to']}" if m["to"] else "→ [ELIMINATO]"
                    console.print(f"    {m['from']} {arrow}  ({m['edges']} archi)")
            for err in r5["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

    console.rule("[green]KG Repair 3 completato[/green]")


if __name__ == "__main__":
    main()
