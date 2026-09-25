#!/usr/bin/env python3
"""KG repair pass 4: PUBLISHED endpoints, FULL_NAME, micro-types, residuals.

Normally run through ``scripts/kg/kg_postprocess.py``. Its LLM prompts
describe a food-security (FAO/EU) graph.

Steps:
  1. Fix PUBLISHED edges with semantically wrong endpoints
       (Concept)-[PUBLISHED]->(Document): LLM → ANALYZES/CONTRIBUTES_TO/BASED_ON/RELATED_TO
       (Organization)-[PUBLISHED]->(Concept): LLM → ANALYZES/CONTRIBUTES_TO/AFFECTS/WORKED_WITH
  2. Turn FULL_NAME from a relationship into a property of the source node;
     delete the edge and the target if it is left orphaned
  3. Deterministic micro-type consolidation
       IMPACTS → AFFECTS
       AFFECTED_BY → reverse direction, AFFECTS
       INCREASED_BY → AFFECTS
       ASSESSED_IN → reverse direction, ANALYZES
       DEFINED_IN → DEFINED_AS
       ASSESSMENT_RESULT → HAS_VALUE
  4. (Concept)-[RELATED_TO]->(Concept): reclassify via LLM (batches of 50,
     restricted vocabulary)
  5. Final round on residual non-canonical micro-types with < 5 edges:
       clear endpoint pattern → apply it; otherwise DELETE (no RELATED_TO
       fallback)
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

BATCH_PUBLISHED   = 50
BATCH_RELATED_TO  = 50
BATCH_RESIDUAL    = 100

# One canonical vocabulary for every pass that renames a relationship type: a
# private copy drifts from the others unnoticed.
CANONICAL_VOCAB: list[str] = CANONICAL_RELATION_TYPES
CANONICAL_SET: set[str] = set(CANONICAL_VOCAB)

# Restricted vocabulary for step 4 (Concept→Concept RELATED_TO)
CONCEPT_RELATED_TO_VOCAB: list[str] = [
    "AFFECTS", "INCLUDES", "HAS_COMPONENT", "CONTRIBUTES_TO", "IS_TYPE_OF",
    "BASED_ON", "REQUIRES", "DEFINED_AS", "ANALYZES", "CAUSES", "NEEDED_FOR",
    "RELATED_TO",
]
CONCEPT_RELATED_TO_SET: set[str] = set(CONCEPT_RELATED_TO_VOCAB)

console = Console(force_terminal=True, highlight=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
    force=True,
)
for _h in logging.getLogger().handlers:
    _h.flush = lambda self=_h: (self.stream.flush(), None)[1]  # type: ignore[method-assign]
logger = logging.getLogger("kg_repair4")


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


def _fetch_edge_context(
    session,
    rel_type: str,
    rel_ids: list[int],
    neighbor_limit: int = 3,
) -> list[dict[str, Any]]:
    """Fetch label+name for src/tgt + up to neighbor_limit neighbouring rels per endpoint."""
    if not rel_ids:
        return []
    safe = rel_type.replace("`", "")
    query = (
        "UNWIND $ids AS rid "
        f"MATCH (s)-[r:`{safe}`]->(t) WHERE id(r) = rid "
        "CALL { "
        "  WITH s, rid "
        "  MATCH (s)-[rs]-(sn) WHERE id(rs) <> rid "
        f"  RETURN collect({{type: type(rs), neighbor: coalesce(sn.name,''), "
        f"                   labels: labels(sn)}})[0..{neighbor_limit}] AS s_rels "
        "} "
        "CALL { "
        "  WITH t, rid "
        "  MATCH (t)-[rt]-(tn) WHERE id(rt) <> rid "
        f"  RETURN collect({{type: type(rt), neighbor: coalesce(tn.name,''), "
        f"                   labels: labels(tn)}})[0..{neighbor_limit}] AS t_rels "
        "} "
        "RETURN id(r) AS id, "
        "  {labels: labels(s), name: coalesce(s.name,'')} AS source, "
        "  {labels: labels(t), name: coalesce(t.name,'')} AS target, "
        "  s_rels AS source_context, t_rels AS target_context"
    )
    return session.run(query, ids=rel_ids).data()


def _apply_reclassification(
    session,
    batch_ids: list[int],
    llm_rows: list[dict[str, Any]],
    allowed_set: set[str],
    keep_fallback: str | None,
    report: dict[str, Any],
    batch_label: str,
) -> None:
    """Apply LLM reclassification results; updates report in-place."""
    batch_set = set(batch_ids)
    updates: list[dict[str, Any]] = []
    kept = 0
    for row in llm_rows:
        try:
            rid = int(row.get("id", -1))
        except (TypeError, ValueError):
            continue
        if rid not in batch_set:
            continue
        rel_type = str(row.get("type", keep_fallback or "")).strip().upper()
        if rel_type not in allowed_set:
            rel_type = keep_fallback or ""
        if not rel_type or rel_type == keep_fallback:
            kept += 1
            continue
        updates.append({"id": rid, "type": rel_type})

    report.setdefault("kept_fallback", 0)
    report["kept_fallback"] += kept

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
            report.setdefault("reclassified", 0)
            report["reclassified"] += updated
        except Exception as exc:
            msg = f"apoc.refactor.setType in {batch_label}: {exc}"
            report.setdefault("errors", []).append(msg)
            logger.warning(msg)

    logger.info(
        "%s — processed %d, reclassified %d, kept %d",
        batch_label, len(batch_ids), len(updates), kept,
    )


# ── Step 1: Fix PUBLISHED with wrong endpoints ────────────────────────────────

def step_1_fix_published(session, client: OpenAI) -> dict[str, Any]:
    """Retype PUBLISHED edges whose endpoints cannot publish, with the LLM.

    Concept→Document edges may fall back to RELATED_TO; Organization→Concept
    edges keep PUBLISHED when the LLM returns nothing valid, for step 5.

    Returns:
        Counts and errors per pattern.
    """
    report: dict[str, Any] = {
        "concept_doc": {"total": 0, "reclassified": 0, "kept_fallback": 0, "errors": []},
        "org_concept": {"total": 0, "reclassified": 0, "kept_fallback": 0, "errors": []},
    }

    # ── 1a. (Concept)-[PUBLISHED]->(Document)
    vocab_1a = {"ANALYZES", "CONTRIBUTES_TO", "BASED_ON", "RELATED_TO"}
    rows_1a = session.run(
        "MATCH (s:Concept)-[r:PUBLISHED]->(t:Document) RETURN id(r) AS id"
    ).data()
    rel_ids_1a = [int(r["id"]) for r in rows_1a]
    report["concept_doc"]["total"] = len(rel_ids_1a)
    console.print(f"  (Concept)-[PUBLISHED]->(Document): {len(rel_ids_1a)} archi")

    total_batches_1a = (len(rel_ids_1a) + BATCH_PUBLISHED - 1) // BATCH_PUBLISHED
    for batch_idx, batch_ids in enumerate(_chunked(rel_ids_1a, BATCH_PUBLISHED), start=1):
        ctx = _fetch_edge_context(session, "PUBLISHED", batch_ids, neighbor_limit=3)
        prompt = (
            "Reclassify Neo4j PUBLISHED relationships between Concept→Document nodes.\n"
            "Context: knowledge graph about food security (FAO/EU domain).\n"
            "Rules:\n"
            "- Choose from the allowed list below.\n"
            "- Use RELATED_TO only if nothing else fits.\n"
            "- Each item has source node (labels+name), target node, and up to 3 neighbouring "
            "relationships per endpoint.\n"
            'Return JSON array: [{"id": <int>, "type": "<PREDICATE>"}]\n\n'
            "Allowed predicates: ANALYZES, CONTRIBUTES_TO, BASED_ON, RELATED_TO\n\n"
            "Items to reclassify:\n"
            f"{json.dumps(ctx, indent=2)}"
        )
        try:
            llm_rows = _llm_json_array(client, prompt)
        except Exception as exc:
            msg = f"LLM batch {batch_idx}/{total_batches_1a} (Concept→Doc): {exc}"
            report["concept_doc"]["errors"].append(msg)
            logger.warning(msg)
            continue
        _apply_reclassification(
            session, batch_ids, llm_rows, vocab_1a, "RELATED_TO",
            report["concept_doc"], f"1a batch {batch_idx}/{total_batches_1a}",
        )

    # ── 1b. (Organization)-[PUBLISHED]->(Concept)
    vocab_1b = {"ANALYZES", "CONTRIBUTES_TO", "AFFECTS", "WORKED_WITH"}
    rows_1b = session.run(
        "MATCH (s:Organization)-[r:PUBLISHED]->(t:Concept) RETURN id(r) AS id"
    ).data()
    rel_ids_1b = [int(r["id"]) for r in rows_1b]
    report["org_concept"]["total"] = len(rel_ids_1b)
    console.print(f"  (Organization)-[PUBLISHED]->(Concept): {len(rel_ids_1b)} archi")

    total_batches_1b = (len(rel_ids_1b) + BATCH_PUBLISHED - 1) // BATCH_PUBLISHED
    for batch_idx, batch_ids in enumerate(_chunked(rel_ids_1b, BATCH_PUBLISHED), start=1):
        ctx = _fetch_edge_context(session, "PUBLISHED", batch_ids, neighbor_limit=3)
        prompt = (
            "Reclassify Neo4j PUBLISHED relationships between Organization→Concept nodes.\n"
            "Context: knowledge graph about food security (FAO/EU domain).\n"
            "Rules:\n"
            "- Choose from the allowed list below.\n"
            "- Do NOT use RELATED_TO or PUBLISHED — choose one of the four allowed types.\n"
            "- Each item has source node (labels+name), target node, and up to 3 neighbouring "
            "relationships per endpoint.\n"
            'Return JSON array: [{"id": <int>, "type": "<PREDICATE>"}]\n\n'
            "Allowed predicates: ANALYZES, CONTRIBUTES_TO, AFFECTS, WORKED_WITH\n\n"
            "Items to reclassify:\n"
            f"{json.dumps(ctx, indent=2)}"
        )
        try:
            llm_rows = _llm_json_array(client, prompt)
        except Exception as exc:
            msg = f"LLM batch {batch_idx}/{total_batches_1b} (Org→Concept): {exc}"
            report["org_concept"]["errors"].append(msg)
            logger.warning(msg)
            continue
        # For Org→Concept there is no acceptable fallback: keep the old PUBLISHED only if
        # the LLM returns nothing valid (it will be caught by step 5 otherwise).
        _apply_reclassification(
            session, batch_ids, llm_rows, vocab_1b, None,
            report["org_concept"], f"1b batch {batch_idx}/{total_batches_1b}",
        )

    return report


# ── Step 2: FULL_NAME relationship → property ─────────────────────────────────

def step_2_full_name_to_property(session) -> dict[str, Any]:
    """Move FULL_NAME edges onto the source node as a ``full_name`` property.

    The edge is deleted, and so is its target when nothing else links to it.

    Returns:
        Counts and errors.
    """
    report: dict[str, Any] = {
        "converted": 0,
        "target_nodes_deleted": 0,
        "errors": [],
    }

    rows = session.run(
        "MATCH (a)-[r:FULL_NAME]->(b) "
        "RETURN id(r) AS rid, id(a) AS aid, id(b) AS bid, "
        "       coalesce(b.name, '') AS bname"
    ).data()
    console.print(f"  (A)-[FULL_NAME]->(B) trovati: {len(rows)}")

    for row in rows:
        rid  = int(row["rid"])
        aid  = int(row["aid"])
        bid  = int(row["bid"])
        bname = str(row.get("bname", ""))
        try:
            # Set property, delete the edge
            session.run(
                "MATCH (a) WHERE id(a) = $aid SET a.full_name = $bname "
                "WITH a "
                "MATCH ()-[r]->() WHERE id(r) = $rid DELETE r",
                aid=aid, bname=bname, rid=rid,
            ).consume()
            report["converted"] += 1

            # If B has no remaining relationships, delete it
            remaining = session.run(
                "MATCH (b) WHERE id(b) = $bid "
                "RETURN size([(b)-[]-() | 1]) AS deg",
                bid=bid,
            ).single()
            if remaining and int(remaining["deg"]) == 0:
                session.run(
                    "MATCH (b) WHERE id(b) = $bid DETACH DELETE b", bid=bid
                ).consume()
                report["target_nodes_deleted"] += 1

        except Exception as exc:
            msg = f"FULL_NAME edge {rid}: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    return report


# ── Step 3: Deterministic micro-type consolidation ────────────────────────────

def step_3_micro_consolidation(session) -> dict[str, Any]:
    """Apply the fixed micro-type renames and direction inversions.

    Returns:
        The edges affected per rename and inversion, and the errors.
    """
    report: dict[str, Any] = {"renames": [], "inversions": [], "errors": []}

    # Simple renames (old → new)
    renames = [
        ("IMPACTS",           "AFFECTS"),
        ("INCREASED_BY",      "AFFECTS"),
        ("DEFINED_IN",        "DEFINED_AS"),
        ("ASSESSMENT_RESULT", "HAS_VALUE"),
    ]
    for old, new in renames:
        safe_old = old.replace("`", "")
        try:
            count = _count_rel(session, old)
            if count == 0:
                console.print(f"  rename {old} → {new}: non trovato")
                report["renames"].append({"from": old, "to": new, "count": 0})
                continue
            session.run(
                "CALL apoc.refactor.rename.type($old, $new)", old=old, new=new
            ).consume()
            report["renames"].append({"from": old, "to": new, "count": count})
            console.print(f"  rename {old} → {new}: {count} archi")
        except Exception as exc:
            msg = f"rename {old} → {new}: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    # Inversions: (A)-[OLD]->(B) becomes (B)-[NEW]->(A)
    inversions = [
        ("AFFECTED_BY", "AFFECTS"),
        ("ASSESSED_IN",  "ANALYZES"),
    ]
    for old, new in inversions:
        try:
            r = session.run(
                f"MATCH ()-[r:`{old}`]->() RETURN count(r) AS c"
            ).single()
            count = int(r["c"]) if r else 0
            if count == 0:
                console.print(f"  invert {old} → {new}: non trovato")
                report["inversions"].append({"from": old, "to": new, "count": 0})
                continue
            session.run(
                f"MATCH (a)-[r:`{old}`]->(b) "
                "WITH a, b, properties(r) AS props "
                f"MERGE (b)-[r2:`{new}`]->(a) "
                "SET r2 += props "
                "DELETE r"
            ).consume()
            report["inversions"].append({"from": old, "to": new, "count": count})
            console.print(f"  invert {old} → {new}: {count} archi")
        except Exception as exc:
            msg = f"invert {old} → {new}: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)

    return report


# ── Step 4: (Concept)-[RELATED_TO]->(Concept) via LLM ────────────────────────

def step_4_concept_related_to(session, client: OpenAI) -> dict[str, Any]:
    """Retype Concept-[RELATED_TO]->Concept edges with the LLM, restricted vocabulary.

    Returns:
        Totals and errors.
    """
    report: dict[str, Any] = {
        "total": 0,
        "reclassified": 0,
        "kept_related_to": 0,
        "batches": 0,
        "errors": [],
    }

    rows = session.run(
        "MATCH (s:Concept)-[r:RELATED_TO]->(t:Concept) RETURN id(r) AS id"
    ).data()
    rel_ids = [int(r["id"]) for r in rows]
    report["total"] = len(rel_ids)
    console.print(f"  (Concept)-[RELATED_TO]->(Concept): {len(rel_ids)} archi")

    if not rel_ids:
        return report

    total_batches = (len(rel_ids) + BATCH_RELATED_TO - 1) // BATCH_RELATED_TO

    for batch_idx, batch_ids in enumerate(_chunked(rel_ids, BATCH_RELATED_TO), start=1):
        report["batches"] += 1
        batch_set = set(batch_ids)
        ctx = _fetch_edge_context(session, "RELATED_TO", batch_ids, neighbor_limit=5)

        prompt = (
            "You reclassify Neo4j RELATED_TO relationships between Concept nodes.\n"
            "Context: knowledge graph about food security (FAO/EU domain).\n"
            "Rules:\n"
            "- Use only the allowed list below.\n"
            "- Prefer the most specific relation; use RELATED_TO only if nothing clearly fits.\n"
            "- Each item has source node (labels+name), target node, and up to 5 neighbouring "
            "relationships per endpoint for context.\n"
            'Return JSON array: [{"id": <int>, "type": "<PREDICATE>"}]\n\n'
            "Allowed predicates:\n"
            f"{json.dumps(CONCEPT_RELATED_TO_VOCAB, indent=2)}\n\n"
            "Items to reclassify:\n"
            f"{json.dumps(ctx, indent=2)}"
        )
        try:
            llm_rows = _llm_json_array(client, prompt)
        except Exception as exc:
            msg = f"LLM batch {batch_idx}/{total_batches}: {exc}"
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
            if rel_type not in CONCEPT_RELATED_TO_SET:
                rel_type = "RELATED_TO"
            if rel_type == "RELATED_TO":
                kept += 1
                continue
            updates.append({"id": rid, "type": rel_type})

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
                msg = f"apoc.refactor.setType batch {batch_idx}/{total_batches}: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)

        logger.info(
            "RELATED_TO LLM batch %d/%d — processed %d, reclassified %d, kept %d",
            batch_idx, total_batches, len(batch_ids), len(updates), kept,
        )

    return report


# ── Step 5: Final round on residual micro-types ──────────────────────────────

# Deterministic endpoint-based fallback for rare types
# key = (src_label, tgt_label), value = canonical type or None (delete)
_PATTERN_MAP: dict[tuple[str, str], str | None] = {
    ("Region",       "DataValue"):    "MEASURES",
    ("Indicator",    "DataValue"):    "MEASURES",
    ("DataValue",    "DataValue"):    None,
    ("Organization", "Organization"): "WORKED_WITH",
    ("Organization", "Document"):     "PUBLISHED",
    ("Document",     "Organization"): "PUBLISHED",
    ("Region",       "Concept"):      "AFFECTS",
    ("Concept",      "Region"):       "AFFECTS",
    ("Indicator",    "Concept"):      "ANALYZES",
    ("Organization", "Region"):       "AFFECTS",
}


def _pattern_canonical(src_labels: list[str], tgt_labels: list[str]) -> str | None | bool:
    """Return canonical type, None (delete), or False (no clear mapping)."""
    for sl in src_labels:
        for tl in tgt_labels:
            if (sl, tl) in _PATTERN_MAP:
                return _PATTERN_MAP[(sl, tl)]
    return False  # no match


def step_5_residual_micro_types(session) -> dict[str, Any]:
    """Resolve every non-canonical type with fewer than 5 edges by endpoint pattern.

    A clear pattern renames the type (or deletes it, for DataValue pairs);
    without one, its edges are deleted.

    Returns:
        Counts, the mapping applied, and the errors.
    """
    report: dict[str, Any] = {
        "rare_types_found": 0,
        "already_canonical": 0,
        "deterministic_renames": 0,
        "deterministic_deletes": 0,
        "unmappable_deleted": 0,
        "total_edges_renamed": 0,
        "total_edges_deleted": 0,
        "mapping": [],
        "errors": [],
    }

    rows = session.run(
        "MATCH ()-[r]->() "
        "WITH type(r) AS rtype, count(r) AS cnt "
        "WHERE cnt < 5 "
        "WITH rtype, cnt "
        "MATCH (s)-[r2]->(t) WHERE type(r2) = rtype "
        "WITH rtype, cnt, "
        "     collect(DISTINCT {source_labels: labels(s), target_labels: labels(t)})[0..5] AS patterns "
        "RETURN rtype AS type, cnt AS count, patterns "
        "ORDER BY cnt DESC"
    ).data()

    report["rare_types_found"] = len(rows)

    for row in rows:
        rtype    = str(row.get("type", "")).strip().upper()
        count    = int(row.get("count", 0))
        patterns = row.get("patterns", [])

        if rtype in CANONICAL_SET:
            report["already_canonical"] += 1
            continue

        # Try deterministic pattern match from the first available pattern
        canonical: str | None | bool = False
        for pat in patterns:
            src_labels = pat.get("source_labels") or []
            tgt_labels = pat.get("target_labels") or []
            result = _pattern_canonical(src_labels, tgt_labels)
            if result is not False:
                canonical = result
                break

        entry: dict[str, Any] = {"from": rtype, "edges": count, "applied": 0, "action": ""}

        if canonical is False:
            # No clear mapping → delete
            try:
                safe = rtype.replace("`", "")
                r = session.run(
                    f"MATCH ()-[r:`{safe}`]->() DELETE r RETURN count(r) AS c"
                ).single()
                deleted = int(r["c"]) if r else 0
                entry["action"] = "deleted_unmappable"
                entry["applied"] = deleted
                report["unmappable_deleted"] += 1
                report["total_edges_deleted"] += deleted
                console.print(f"  DELETE (unmappable) {rtype}: {deleted} archi")
            except Exception as exc:
                msg = f"delete unmappable {rtype}: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)

        elif canonical is None:
            # Explicit delete from pattern
            try:
                safe = rtype.replace("`", "")
                r = session.run(
                    f"MATCH ()-[r:`{safe}`]->() DELETE r RETURN count(r) AS c"
                ).single()
                deleted = int(r["c"]) if r else 0
                entry["action"] = "deleted_by_pattern"
                entry["applied"] = deleted
                report["deterministic_deletes"] += 1
                report["total_edges_deleted"] += deleted
                console.print(f"  DELETE (pattern) {rtype}: {deleted} archi")
            except Exception as exc:
                msg = f"delete by-pattern {rtype}: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)

        else:
            # Rename to canonical
            try:
                session.run(
                    "CALL apoc.refactor.rename.type($old, $new)",
                    old=rtype, new=canonical,
                ).consume()
                entry["action"] = f"renamed→{canonical}"
                entry["applied"] = count
                report["deterministic_renames"] += 1
                report["total_edges_renamed"] += count
                console.print(f"  rename (pattern) {rtype} → {canonical}: {count} archi")
            except Exception as exc:
                msg = f"rename {rtype} → {canonical}: {exc}"
                report["errors"].append(msg)
                logger.warning(msg)

        report["mapping"].append(entry)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Confirm the target, then run the five steps in one session."""
    require_confirmation(
        title="KG Repair 4",
        what_it_does="""reclassify RELATED_TO into typed relationships
        collapse relationship types that mean the same thing
        remove relationships duplicated between the same pair""",
        uri=NEO4J_URI,
        database=NEO4J_DATABASE,
    )
    console.rule("[bold]KG Repair 4 — Neo4j Aura[/bold]")

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
            console.rule("[yellow]Step 1: Fix PUBLISHED con endpoint errati[/yellow]")
            r1 = step_1_fix_published(session, client)
            for sub, label in [("concept_doc", "(Concept)→(Document)"), ("org_concept", "(Organization)→(Concept)")]:
                d = r1[sub]
                console.print(f"  {label}: totale={d['total']}, reclassificati={d.get('reclassified',0)}, mantenuti={d.get('kept_fallback',0)}")
                for err in d["errors"][:5]:
                    console.print(f"    [red]{err}[/red]")

            # ── Step 2
            console.rule("[yellow]Step 2: FULL_NAME relazione → proprietà[/yellow]")
            r2 = step_2_full_name_to_property(session)
            console.print(f"  Archi convertiti          : {r2['converted']}")
            console.print(f"  Nodi target eliminati      : {r2['target_nodes_deleted']}")
            for err in r2["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 3
            console.rule("[yellow]Step 3: Consolidamento deterministico micro-tipi[/yellow]")
            r3 = step_3_micro_consolidation(session)
            total_r3 = sum(x["count"] for x in r3["renames"] + r3["inversions"])
            console.print(f"  Rinominati (tipi): {len(r3['renames'])}  |  Invertiti: {len(r3['inversions'])}")
            console.print(f"  Archi totali modificati  : {total_r3}")
            for err in r3["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 4
            console.rule("[yellow]Step 4: (Concept)-[RELATED_TO]->(Concept) via LLM[/yellow]")
            r4 = step_4_concept_related_to(session, client)
            console.print(f"  Totale                   : {r4['total']}")
            console.print(f"  LLM reclassificati       : {r4['reclassified']}")
            console.print(f"  Mantenuti RELATED_TO     : {r4['kept_related_to']}")
            console.print(f"  Batch                    : {r4['batches']}")
            for err in r4["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 5
            console.rule("[yellow]Step 5: Round finale micro-tipi residui[/yellow]")
            r5 = step_5_residual_micro_types(session)
            console.print(f"  Tipi rari trovati        : {r5['rare_types_found']}")
            console.print(f"  Già canonici             : {r5['already_canonical']}")
            console.print(f"  Rinominati (pattern)     : {r5['deterministic_renames']}  ({r5['total_edges_renamed']} archi)")
            console.print(f"  Eliminati (pattern)      : {r5['deterministic_deletes']}  ({r5['total_edges_deleted']} archi)")
            console.print(f"  Eliminati (unmappable)   : {r5['unmappable_deleted']}")
            if r5["mapping"]:
                console.print("  Dettaglio (prime 20):")
                for m in r5["mapping"][:20]:
                    console.print(f"    {m['from']} → {m['action']}  ({m['edges']} archi, applied={m['applied']})")
            for err in r5["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

    console.rule("[green]KG Repair 4 completato[/green]")


if __name__ == "__main__":
    main()
