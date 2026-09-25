#!/usr/bin/env python3
"""KG repair pass 1: hub artefacts, PUBLISHED_BY direction, RELATED_TO, properties.

Normally run through ``scripts/kg/kg_postprocess.py``. Requires neo4j,
openai, python-dotenv and rich. Its LLM prompts describe a food-security
(FAO/EU) graph.

Steps executed in sequence:
  1. Hub node artefact cleanup (pure Cypher + APOC mergeNodes)
  2. PUBLISHED_BY direction fix (pure Cypher)
  3. RELATED_TO reclassification via vLLM (batches of 50)
  4. Node property re-enrichment via vLLM (batches of 30)
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

from kg_pipeline.utils.validation import parse_json_array

# ── Credentials ───────────────────────────────────────────────────────────────
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

RELATED_TO_VOCAB: list[str] = [
    "AFFECTS", "INCLUDES", "HAS_COMPONENT", "CONTRIBUTES_TO", "IS_TYPE_OF",
    "BASED_ON", "ANALYZES", "PUBLISHES", "ESTABLISHES", "REQUIRES", "GOVERNS",
    "USES", "PRODUCES", "MEASURES", "LOCATED_IN", "WORKED_WITH", "DEFINED_AS",
    "CONTAINS_DATA", "NEEDED_FOR", "RELATED_TO",
]

BATCH_RELTYPE = 50
BATCH_ENRICH  = 30

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("kg_repair")


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _extract_first_json_array(text: str) -> str:
    """Scan text for the first balanced [...] array and return it as a string."""
    start = text.find("[")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
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
                return text[start:idx + 1]
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


def _chunked(lst: list, n: int):
    """Consecutive slices of `lst`, `n` at a time."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _node_exists(session, label: str, name: str) -> bool:
    """True when a node with this label and name exists."""
    r = session.run(
        f"MATCH (n:`{label}` {{name: $name}}) RETURN count(n) > 0 AS exists",
        name=name,
    ).single()
    return bool(r["exists"]) if r else False


# ── Step 1: Hub node artefact cleanup ────────────────────────────────────────

def step_1_hub_cleanup(session) -> dict[str, Any]:
    """Delete or fix the known hub artefacts.

    ``n.a.%`` indicators and ``true`` data values are deleted; Concept nodes
    named Africa, Asia or World become (or merge into) Region nodes; "The
    Authority" becomes (or merges into) "Authority".

    Returns:
        What happened to each artefact.
    """
    report: dict[str, Any] = {}

    # 1a. Indicator {name: "n.a.%"} → DETACH DELETE
    r = session.run("MATCH (n:Indicator {name: 'n.a.%'}) RETURN count(n) AS c").single()
    cnt = int(r["c"]) if r else 0
    if cnt > 0:
        session.run("MATCH (n:Indicator {name: 'n.a.%'}) DETACH DELETE n").consume()
    report["indicator_na_deleted"] = cnt

    # 1b. DataValue {name: "true"} → DETACH DELETE
    r = session.run("MATCH (n:DataValue {name: 'true'}) RETURN count(n) AS c").single()
    cnt = int(r["c"]) if r else 0
    if cnt > 0:
        session.run("MATCH (n:DataValue {name: 'true'}) DETACH DELETE n").consume()
    report["datavalue_true_deleted"] = cnt

    # 1c. Concept "Africa" → merge with Region or relabel
    has_region_africa = _node_exists(session, "Region", "Africa")
    has_concept_africa = _node_exists(session, "Concept", "Africa")
    if has_concept_africa:
        if has_region_africa:
            session.run(
                "MATCH (keep:Region {name:'Africa'}), (drop:Concept {name:'Africa'}) "
                "CALL apoc.refactor.mergeNodes([keep, drop], {properties:'discard', mergeRels:true}) "
                "YIELD node RETURN id(node)"
            ).consume()
            report["concept_africa"] = "merged_into_region"
        else:
            session.run(
                "MATCH (n:Concept {name:'Africa'}) REMOVE n:Concept SET n:Region"
            ).consume()
            report["concept_africa"] = "relabeled_to_region"
    else:
        report["concept_africa"] = "not_found"

    # 1d. Concept "Asia" → same logic
    has_region_asia = _node_exists(session, "Region", "Asia")
    has_concept_asia = _node_exists(session, "Concept", "Asia")
    if has_concept_asia:
        if has_region_asia:
            session.run(
                "MATCH (keep:Region {name:'Asia'}), (drop:Concept {name:'Asia'}) "
                "CALL apoc.refactor.mergeNodes([keep, drop], {properties:'discard', mergeRels:true}) "
                "YIELD node RETURN id(node)"
            ).consume()
            report["concept_asia"] = "merged_into_region"
        else:
            session.run(
                "MATCH (n:Concept {name:'Asia'}) REMOVE n:Concept SET n:Region"
            ).consume()
            report["concept_asia"] = "relabeled_to_region"
    else:
        report["concept_asia"] = "not_found"

    # 1e. Concept "world" or "World" → relabel to Region, normalize name to "World"
    r = session.run(
        "MATCH (n:Concept) WHERE toLower(n.name) = 'world' RETURN count(n) AS c"
    ).single()
    cnt = int(r["c"]) if r else 0
    if cnt > 0:
        session.run(
            "MATCH (n:Concept) WHERE toLower(n.name) = 'world' "
            "REMOVE n:Concept SET n:Region, n.name = 'World'"
        ).consume()
    report["concept_world_relabeled"] = cnt

    # 1f. Organization "The Authority" → merge with "Authority" or rename
    has_auth = _node_exists(session, "Organization", "Authority")
    has_the_auth = _node_exists(session, "Organization", "The Authority")
    if has_the_auth:
        if has_auth:
            session.run(
                "MATCH (keep:Organization {name:'Authority'}), "
                "      (drop:Organization {name:'The Authority'}) "
                "CALL apoc.refactor.mergeNodes([keep, drop], {properties:'discard', mergeRels:true}) "
                "YIELD node RETURN id(node)"
            ).consume()
            report["org_the_authority"] = "merged_into_authority"
        else:
            session.run(
                "MATCH (n:Organization {name:'The Authority'}) SET n.name = 'Authority'"
            ).consume()
            report["org_the_authority"] = "renamed_to_authority"
    else:
        report["org_the_authority"] = "not_found"

    return report


# ── Step 2: PUBLISHED_BY direction fix ───────────────────────────────────────

def step_2_published_by(session) -> dict[str, Any]:
    """Turn Organization-[PUBLISHED_BY]->Document into Document->Organization.

    Returns:
        How many were found the wrong way round and how many were fixed.
    """
    r = session.run(
        "MATCH (org:Organization)-[r:PUBLISHED_BY]->(doc:Document) RETURN count(r) AS c"
    ).single()
    cnt = int(r["c"]) if r else 0
    if cnt == 0:
        return {"wrong_direction_found": 0, "fixed": 0}

    result = session.run(
        "MATCH (org:Organization)-[r:PUBLISHED_BY]->(doc:Document) "
        "WITH org, doc, properties(r) AS props, r "
        "MERGE (doc)-[r2:PUBLISHED_BY]->(org) "
        "SET r2 += props "
        "DELETE r "
        "RETURN count(r2) AS fixed"
    ).single()
    return {
        "wrong_direction_found": cnt,
        "fixed": int(result["fixed"]) if result else 0,
    }


# ── Step 3: RELATED_TO reclassification ──────────────────────────────────────

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
        "  RETURN collect({type: type(rs), neighbor: coalesce(sn.name, ''), "
        "                  labels: labels(sn)})[0..3] AS s_rels "
        "} "
        "CALL { "
        "  WITH t, rid "
        "  MATCH (t)-[rt]-(tn) WHERE id(rt) <> rid "
        "  RETURN collect({type: type(rt), neighbor: coalesce(tn.name, ''), "
        "                  labels: labels(tn)})[0..3] AS t_rels "
        "} "
        "RETURN id(r) AS id, "
        "  {labels: labels(s), name: coalesce(s.name, '')} AS source, "
        "  {labels: labels(t), name: coalesce(t.name, '')} AS target, "
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
        "- Each item includes source node (labels+name), target node, and up to 3 neighbouring "
        "relationships per endpoint for context.\n"
        'Return JSON array of objects: [{"id": <int>, "type": "<PREDICATE>"}]\n\n'
        "Allowed predicates:\n"
        f"{json.dumps(vocab, indent=2)}\n\n"
        "Items to reclassify:\n"
        f"{json.dumps(items, indent=2)}"
    )


def step_3_reclassify_related_to(session, client: OpenAI) -> dict[str, Any]:
    """Retype the RELATED_TO relationships with the LLM, in batches.

    Returns:
        Totals, the count per new type, and the errors.
    """
    report: dict[str, Any] = {
        "total": 0,
        "reclassified": 0,
        "kept_related_to": 0,
        "type_counts": {},
        "batches": 0,
        "errors": [],
    }

    rows = session.run("MATCH ()-[r:RELATED_TO]->() RETURN id(r) AS id").data()
    rel_ids = [int(r["id"]) for r in rows]
    report["total"] = len(rel_ids)
    if not rel_ids:
        return report

    vocab_set = set(RELATED_TO_VOCAB)
    total_batches = (len(rel_ids) + BATCH_RELTYPE - 1) // BATCH_RELTYPE

    for batch_idx, batch_ids in enumerate(_chunked(rel_ids, BATCH_RELTYPE), start=1):
        report["batches"] += 1
        batch_set = set(batch_ids)

        context_rows = _fetch_related_to_context(session, batch_ids)
        prompt = _reclass_prompt(RELATED_TO_VOCAB, context_rows)

        try:
            llm_rows = _llm_json_array(client, prompt)
        except Exception as exc:
            msg = f"Batch {batch_idx}/{total_batches}: LLM call failed: {exc}"
            report["errors"].append(msg)
            logger.warning(msg)
            continue

        updates: list[dict[str, Any]] = []
        skipped = 0
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
                skipped += 1
                continue
            updates.append({"id": rid, "type": rel_type})
            report["type_counts"][rel_type] = report["type_counts"].get(rel_type, 0) + 1

        report["kept_related_to"] += skipped

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
            "Batch %d/%d: %d rels processed → %d to reclassify, %d kept as RELATED_TO",
            batch_idx, total_batches, len(batch_ids), len(updates), skipped,
        )

    return report


# ── Step 4: Property enrichment ───────────────────────────────────────────────

def _org_enrich_prompt(nodes: list[dict[str, Any]]) -> str:
    """The prompt filling the missing Organization properties."""
    return (
        "You enrich Organization nodes in a knowledge graph about food security (FAO/EU domain).\n"
        "For each node, fill in only the properties that are null.\n"
        "Rules:\n"
        "- description: one sentence describing what the organization does (max 20 words)\n"
        "- country: ISO 3166-1 alpha-2 code (e.g. 'IT', 'US', 'FR'), or 'international' "
        "for intergovernmental or multinational organizations\n"
        "- organization_type: exactly one of 'IGO', 'NGO', 'government', 'private'\n"
        "- Omit a key if it is already filled (not null) in the input.\n"
        'Return JSON array: [{"id": <int>, "<prop>": <value>, ...}]\n\n'
        "Nodes:\n"
        f"{json.dumps(nodes, indent=2)}"
    )


def _region_enrich_prompt(nodes: list[dict[str, Any]]) -> str:
    """The prompt filling the missing Region properties."""
    return (
        "You enrich Region nodes in a knowledge graph about food security (FAO/EU domain).\n"
        "For each node, fill in only the properties that are null.\n"
        "Rules:\n"
        "- description: one sentence describing the region (max 20 words)\n"
        "- region_type: exactly one of 'continent', 'subregion', 'country', 'city'\n"
        "- Omit a key if it is already filled (not null) in the input.\n"
        'Return JSON array: [{"id": <int>, "<prop>": <value>, ...}]\n\n'
        "Nodes:\n"
        f"{json.dumps(nodes, indent=2)}"
    )


def _apply_node_updates(session, llm_rows: list[dict[str, Any]], valid_ids: set[int]) -> int:
    """Build per-node prop dicts from LLM output and apply with UNWIND+SET."""
    updates: list[dict[str, Any]] = []
    for row in llm_rows:
        try:
            node_id = int(row.get("id", -1))
        except (TypeError, ValueError):
            continue
        if node_id not in valid_ids:
            continue
        props = {
            k: v
            for k, v in row.items()
            if k != "id" and v is not None and str(v).strip()
        }
        if props:
            updates.append({"id": node_id, "props": props})

    if not updates:
        return 0

    result = session.run(
        "UNWIND $updates AS u "
        "MATCH (n) WHERE id(n) = u.id "
        "SET n += u.props "
        "RETURN count(n) AS updated",
        updates=updates,
    ).single()
    return int(result["updated"]) if result else 0


def step_4_enrich_properties(session, client: OpenAI) -> dict[str, Any]:
    """Fill missing node properties with the LLM and fold MENTIONED_IN edges.

    Organizations get description, country and type; regions get
    description and type; MENTIONED_IN edges become a ``source_documents``
    property and are deleted.

    Returns:
        Update counts and errors.
    """
    report: dict[str, Any] = {
        "orgs_updated": 0,
        "orgs_errors": [],
        "regions_updated": 0,
        "regions_errors": [],
        "source_docs_nodes_updated": 0,
        "source_docs_arcs_deleted": 0,
    }

    # 4a. Organization: fill missing description / country / organization_type
    org_rows = session.run(
        "MATCH (n:Organization) "
        "WHERE n.description IS NULL OR n.country IS NULL OR n.organization_type IS NULL "
        "RETURN id(n) AS id, n.name AS name, "
        "       n.description AS description, n.country AS country, "
        "       n.organization_type AS organization_type"
    ).data()

    total_org_batches = (len(org_rows) + BATCH_ENRICH - 1) // BATCH_ENRICH
    for batch_idx, batch in enumerate(_chunked(org_rows, BATCH_ENRICH), start=1):
        valid_ids = {int(r["id"]) for r in batch}
        try:
            llm_rows = _llm_json_array(client, _org_enrich_prompt(batch))
        except Exception as exc:
            msg = f"Org batch {batch_idx}/{total_org_batches}: LLM failed: {exc}"
            report["orgs_errors"].append(msg)
            logger.warning(msg)
            continue
        try:
            cnt = _apply_node_updates(session, llm_rows, valid_ids)
            report["orgs_updated"] += cnt
        except Exception as exc:
            msg = f"Org batch {batch_idx}/{total_org_batches}: SET failed: {exc}"
            report["orgs_errors"].append(msg)
            logger.warning(msg)
        logger.info("Org batch %d/%d: batch_size=%d", batch_idx, total_org_batches, len(batch))

    # 4b. Region: fill missing description / region_type
    region_rows = session.run(
        "MATCH (n:Region) "
        "WHERE n.description IS NULL OR n.region_type IS NULL "
        "RETURN id(n) AS id, n.name AS name, "
        "       n.description AS description, n.region_type AS region_type"
    ).data()

    total_region_batches = (len(region_rows) + BATCH_ENRICH - 1) // BATCH_ENRICH
    for batch_idx, batch in enumerate(_chunked(region_rows, BATCH_ENRICH), start=1):
        valid_ids = {int(r["id"]) for r in batch}
        try:
            llm_rows = _llm_json_array(client, _region_enrich_prompt(batch))
        except Exception as exc:
            msg = f"Region batch {batch_idx}/{total_region_batches}: LLM failed: {exc}"
            report["regions_errors"].append(msg)
            logger.warning(msg)
            continue
        try:
            cnt = _apply_node_updates(session, llm_rows, valid_ids)
            report["regions_updated"] += cnt
        except Exception as exc:
            msg = f"Region batch {batch_idx}/{total_region_batches}: SET failed: {exc}"
            report["regions_errors"].append(msg)
            logger.warning(msg)
        logger.info("Region batch %d/%d: batch_size=%d", batch_idx, total_region_batches, len(batch))

    # 4c. source_documents from MENTIONED_IN arcs
    r = session.run("MATCH ()-[r:MENTIONED_IN]->() RETURN count(r) AS c").single()
    mentioned_count = int(r["c"]) if r else 0

    if mentioned_count == 0:
        logger.info("No MENTIONED_IN arcs found — skipped")
    else:
        result = session.run(
            "MATCH (n)-[:MENTIONED_IN]->(doc:Document) "
            "WITH n, collect(doc.name) AS doc_names "
            "SET n.source_documents = doc_names "
            "RETURN count(n) AS updated"
        ).single()
        report["source_docs_nodes_updated"] = int(result["updated"]) if result else 0
        session.run("MATCH ()-[r:MENTIONED_IN]->() DELETE r").consume()
        report["source_docs_arcs_deleted"] = mentioned_count
        logger.info(
            "Converted %d MENTIONED_IN arcs to source_documents property on %d nodes",
            mentioned_count, report["source_docs_nodes_updated"],
        )

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Confirm the target, then run the four steps in one session."""
    require_confirmation(
        title="KG Repair 1",
        what_it_does="""delete junk nodes and orphan entities
        normalise entity names and labels
        merge duplicates found by exact name""",
        uri=NEO4J_URI,
        database=NEO4J_DATABASE,
    )
    console.rule("[bold]KG Repair — Neo4j Aura[/bold]")

    if not NEO4J_URI or not NEO4J_PASSWORD:
        console.print("[red]ERROR: NEO4J credentials missing. Check kg_pipeline/.env[/red]")
        sys.exit(1)

    console.print(f"URI    : [cyan]{NEO4J_URI}[/cyan]")
    console.print(f"DB     : [cyan]{NEO4J_DATABASE or '<default>'}[/cyan]")
    console.print(f"vLLM   : [cyan]{VLLM_BASE_URL}[/cyan]")
    console.print(f"Model  : [cyan]{VLLM_MODEL}[/cyan]")
    console.print()

    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, timeout=180.0)
    session_kwargs = _TARGET.session_kwargs()

    with neo4j_env.connect(_TARGET) as driver:
        with driver.session(**session_kwargs) as session:

            # ── Step 1
            console.rule("[yellow]Step 1: Hub node artefact cleanup[/yellow]")
            r1 = step_1_hub_cleanup(session)
            for k, v in r1.items():
                console.print(f"  {k}: {v}")

            # ── Step 2
            console.rule("[yellow]Step 2: PUBLISHED_BY direction fix[/yellow]")
            r2 = step_2_published_by(session)
            console.print(f"  Wrong-direction edges found : {r2['wrong_direction_found']}")
            console.print(f"  Edges fixed                : {r2['fixed']}")

            # ── Step 3
            console.rule("[yellow]Step 3: RELATED_TO reclassification[/yellow]")
            r3 = step_3_reclassify_related_to(session, client)
            console.print(f"  Total RELATED_TO      : {r3['total']}")
            console.print(f"  Reclassified          : {r3['reclassified']}")
            console.print(f"  Kept as RELATED_TO    : {r3['kept_related_to']}")
            console.print(f"  Batches               : {r3['batches']}")
            console.print(f"  Errors                : {len(r3['errors'])}")
            if r3["type_counts"]:
                console.print("  New type distribution:")
                for t, c in sorted(r3["type_counts"].items(), key=lambda x: -x[1]):
                    console.print(f"    {t}: {c}")
            for err in r3["errors"][:5]:
                console.print(f"  [red]{err}[/red]")

            # ── Step 4
            console.rule("[yellow]Step 4: Property enrichment[/yellow]")
            r4 = step_4_enrich_properties(session, client)
            console.print(f"  Organizations enriched      : {r4['orgs_updated']}")
            console.print(f"  Organization errors         : {len(r4['orgs_errors'])}")
            console.print(f"  Regions enriched            : {r4['regions_updated']}")
            console.print(f"  Region errors               : {len(r4['regions_errors'])}")
            console.print(f"  MENTIONED_IN arcs deleted   : {r4['source_docs_arcs_deleted']}")
            console.print(f"  Nodes with source_documents : {r4['source_docs_nodes_updated']}")
            for err in r4["orgs_errors"][:3] + r4["regions_errors"][:3]:
                console.print(f"  [red]{err}[/red]")

    console.rule("[green]KG Repair complete[/green]")


if __name__ == "__main__":
    main()
