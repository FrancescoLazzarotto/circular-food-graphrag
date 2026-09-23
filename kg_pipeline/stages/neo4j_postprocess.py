"""Post-processing passes that clean an ingested graph in place.

The default run maps relationship types to the canonical vocabulary, applies
a fixed set of type rewrites and LLM reclassifications, merges duplicate
nodes, relabels ``Concept`` nodes, enriches node properties and creates
constraints. ``--fix`` runs a single task instead. Every step supports
``--dry-run`` and reports what it did, or would do, in a JSON report printed
to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml
from kg_pipeline.utils import neo4j_env
from openai import OpenAI
from dotenv import load_dotenv

# Allow direct execution from the repo root without `python -m`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg_pipeline.stages.neo4j_ingestion import _resolve_neo4j_env
from kg_pipeline.relations import CANONICAL_RELATION_TYPES
from kg_pipeline.utils.validation import parse_json_array

LOGGER = logging.getLogger("kg_pipeline.neo4j_postprocess")


# Default vocabulary when no --relation-vocab is given; shared with the
# kg_repair passes.
_CANONICAL_RELATION_TYPES = CANONICAL_RELATION_TYPES

# Types rewritten as their inverse, with the edge direction flipped
# (--rewrite-inverses).
_INVERSE_RELATION_REWRITES = [
    {"from": "ESTABLISHED_BY", "to": "ESTABLISHES"},
    {"from": "USED_BY", "to": "USES"},
    {"from": "CAUSED_BY", "to": "CAUSES"},
    {"from": "REQUIRED_BY", "to": "REQUIRES"},
    {"from": "REGULATES", "to": "REGULATED_BY"},
]

# Properties the enrichment step asks the LLM to fill, per label, with the
# description given to the model. Replaced by --property-schema.
_DEFAULT_PROPERTY_SCHEMA: dict[str, dict[str, str]] = {
    "Organization": {
        "description": "Short description of the organization",
        "organization_type": "Type such as company, agency, NGO, ministry",
        "country": "Primary country or region",
    },
    "Region": {
        "description": "Short description of the region",
        "region_type": "Type such as country, continent, basin",
        "country": "Country if applicable",
    },
    "Event": {
        "description": "Short description of the event",
        "date": "ISO date or year if known",
        "location": "Location or region",
    },
    "Indicator": {
        "description": "Short definition of the indicator",
        "unit": "Unit of measure if applicable",
        "category": "Category such as climate, nutrition, economy",
    },
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REL_CLEAN = re.compile(r"[^A-Z0-9_]+")
_RELATED_TO_BATCH_SIZE = 50
_RELATION_RECLASS_BATCH_SIZE = 50

# Types an anomalous HAS_COMPONENT edge may be reclassified to.
_AURA_RECLASS_TYPES = [
    "AFFECTS",
    "CONTRIBUTES_TO",
    "INCLUDES",
    "PART_OF",
    "ANALYZES",
    "RELATED_TO",
]

# Region nodes that are table-header artifacts, merged into a namesake or
# deleted.
_AURA_REGION_GARBAGE_NAMES = [
    "REGIONS/SUBREGIONS/COUNTRIES/TERRITORIES",
    "ASIA*",
]

# Inverse rewrites and renames applied by the default run.
_AURA_INVERSE_REWRITES = [
    {"from": "REGULATES", "to": "REGULATED_BY"},
    {"from": "USED_BY", "to": "USES"},
    {"from": "ESTABLISHED_BY", "to": "ESTABLISHES"},
]

_AURA_RENAME_REWRITES = [
    {"from": "INFLUENCES", "to": "AFFECTS"},
]

# Narrow types absorbed into a broader one (--fix micro-types, aura-issues).
_MICRO_RELATION_REWRITES = [
    {"from": "COMPOSED_OF", "to": "INCLUDES"},
    {"from": "USES_METHOD", "to": "USES"},
    {"from": "TARGETS", "to": "GOVERNS"},
    {"from": "MANAGES", "to": "GOVERNS"},
    {"from": "DEPENDS_ON", "to": "REQUIRES"},
    {"from": "PART_OF", "to": "HAS_COMPONENT"},
    {"from": "ASSOCIATED_WITH", "to": "RELATED_TO"},
    {"from": "OCCURS_IN", "to": "LOCATED_IN"},
    {"from": "LEADS_TO", "to": "AFFECTS"},
    {"from": "IMPACTS", "to": "AFFECTS"},
]

# Renames and inverse rewrites of verbose types (--fix cleanup-pass3).
_VERBOSE_RELATION_RENAMES = [
    {"from": "SHOULD_BE_MANAGED_BY", "to": "GOVERNED_BY"},
    {"from": "TAKE_INTO_ACCOUNT", "to": "BASED_ON"},
    {"from": "INDICATES", "to": "MEASURES"},
    {"from": "USED_BY", "to": "USES"},
]

_VERBOSE_RELATION_INVERSES = [
    {"from": "DRIVEN_BY", "to": "AFFECTS"},
]

# Tokens ignored when comparing relationship type names word by word.
_RELTYPE_STOP_TOKENS = {
    "A",
    "AN",
    "AND",
    "BY",
    "FOR",
    "FROM",
    "HAS",
    "IN",
    "IS",
    "OF",
    "ON",
    "OR",
    "THE",
    "TO",
    "WITH",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    """Yield slices of ``items`` of at most ``size``; non-positive means one slice."""
    if size <= 0:
        size = len(items) or 1
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _normalize_name(value: str) -> str:
    """Normalise a node name for duplicate detection.

    Args:
        value: Node name.

    Returns:
        The name lower-cased, without a leading English article, with every
        run of non-alphanumerics turned into a single space.
    """
    cleaned = " ".join(value.strip().split()).lower()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    cleaned = _NON_ALNUM.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _to_title_case(value: str) -> str:
    """Strip ``value`` and convert it to title case."""
    cleaned = value.strip()
    if not cleaned:
        return ""
    return cleaned.title()


def _normalize_rel_type(value: str) -> str:
    """Upper-case a relationship type and reduce it to ``[A-Z0-9_]``.

    Args:
        value: Relationship type.

    Returns:
        The type with every run of other characters replaced by one ``_`` and
        outer underscores stripped. Safe to interpolate in backticks.
    """
    cleaned = _REL_CLEAN.sub("_", value.strip().upper())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _sanitize_label(value: str) -> str:
    """Remove backticks from a label so it can be quoted in Cypher."""
    return value.replace("`", "").strip()


def _cypher_string_literal(value: str) -> str:
    """Quote ``value`` as a single-quoted Cypher string literal."""
    return "'" + value.replace("'", "''") + "'"


def _build_llm_client(base_url: str, api_key: str) -> OpenAI:
    """Build an OpenAI-compatible client for the vLLM server.

    Args:
        base_url: Server base URL.
        api_key: API key; empty becomes ``"EMPTY"``.

    Returns:
        A client whose timeout is ``VLLM_HTTP_TIMEOUT`` seconds (default 900).
    """
    http_timeout = float(os.getenv("VLLM_HTTP_TIMEOUT", "900"))
    return OpenAI(
        base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY", timeout=http_timeout
    )


def _setup_logging(log_file: Path | None) -> None:
    """Configure INFO logging to stderr and, optionally, to a file.

    Args:
        log_file: File to append log records to; its directory is created.
    """
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    if log_file is not None:
        LOGGER.info("Logging to %s", log_file)


def _confirm_db_changes(
    uri: str, database: str | None, dry_run: bool, assume_yes: bool
) -> None:
    """Ask the operator to type ``YES`` before the database is modified.

    Args:
        uri: Neo4j URI, shown in the prompt.
        database: Database name, shown in the prompt.
        dry_run: Skip the prompt, since nothing is written.
        assume_yes: Skip the prompt (``--yes``).

    Raises:
        SystemExit: If the answer is not ``YES``.
    """
    if dry_run or assume_yes:
        return
    db_name = database or "<default>"
    prompt = (
        "This will modify the Neo4j database "
        f"'{db_name}' at {uri}. Type YES to continue: "
    )
    if input(prompt).strip() != "YES":
        raise SystemExit("Aborted by user.")


def _resolve_llm_env() -> tuple[str, str, str]:
    """Read the vLLM endpoint from the environment.

    Returns:
        ``(base_url, model_name, api_key)`` from ``VLLM_BASE_URL`` (default
        ``http://localhost:8000/v1``), ``VLLM_MODEL_NAME`` and
        ``VLLM_API_KEY`` or ``OPENAI_API_KEY``.

    Raises:
        ValueError: If ``VLLM_MODEL_NAME`` is not set.
    """
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip()
    model_name = os.getenv("VLLM_MODEL_NAME", "").strip()
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not model_name:
        raise ValueError("Missing VLLM_MODEL_NAME for LLM mapping")
    return base_url, model_name, api_key


def _extract_first_json_array(text: str) -> str:
    """Return the first bracket-balanced ``[...]`` span of ``text``.

    Brackets inside JSON strings are ignored.

    Args:
        text: Model output that may wrap the array in prose.

    Returns:
        The array text, or ``""`` when no balanced array is found.
    """
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
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return ""


def _llm_json_array(
    client: OpenAI, model_name: str, prompt: str
) -> list[dict[str, Any]]:
    """Send a prompt at temperature 0 and parse the JSON array it returns.

    Args:
        client: OpenAI-compatible client.
        model_name: Served model name.
        prompt: Prompt text.

    Returns:
        The parsed array; when the whole output does not parse, the first
        balanced array in it is parsed instead.

    Raises:
        openai.OpenAIError: If the request fails.
        ValueError: If no JSON array can be parsed from the output.
    """
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
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


def _has_apoc(session) -> bool:
    """Whether the APOC plugin is installed on the server."""
    try:
        session.run("RETURN apoc.version() AS version").single()
        return True
    except Exception:
        return False


def _fetch_relation_types(session, max_patterns: int) -> list[dict[str, Any]]:
    """List relationship types with their counts and endpoint label patterns.

    Args:
        session: Neo4j session.
        max_patterns: Most frequent (source labels, target labels) patterns
            kept per type.

    Returns:
        Rows with ``type``, ``count`` and ``patterns``, most frequent first.
    """
    query = (
        "MATCH (a)-[r]->(b) "
        "WITH type(r) AS type, labels(a) AS source_labels, labels(b) AS target_labels, count(*) AS count "
        "ORDER BY count DESC "
        "WITH type, collect({source_labels: source_labels, target_labels: target_labels, count: count}) AS patterns, sum(count) AS total "
        "RETURN type, total AS count, patterns[0..$max_patterns] AS patterns "
        "ORDER BY count DESC"
    )
    return session.run(query, max_patterns=max_patterns).data()


def _relation_mapping_prompt(canonical: list[str], items: list[dict[str, Any]]) -> str:
    """Build the prompt that maps relationship types to the canonical list.

    Args:
        canonical: Canonical relationship types.
        items: Relationship types with counts and endpoint label patterns.

    Returns:
        The prompt; the model answers ``[{"source", "target"}, ...]``.
    """
    return (
        "You map Neo4j relationship types to a fixed canonical list.\n"
        "Rules:\n"
        "- Use only the canonical list.\n"
        "- Preserve direction and semantics when possible.\n"
        "- Prefer the most specific relation; avoid RELATED_TO unless nothing fits.\n"
        "- Use endpoint label patterns if provided.\n"
        'Return JSON array of objects: {"source": str, "target": str}.\n\n'
        "Canonical list:\n"
        f"{json.dumps(canonical, indent=2)}\n\n"
        "Items:\n"
        f"{json.dumps(items, indent=2)}"
    )


def _related_to_refinement_prompt(
    canonical: list[str], items: list[dict[str, Any]]
) -> str:
    """Build the prompt that retypes ``RELATED_TO`` edges.

    Args:
        canonical: Canonical relationship types.
        items: Edges with their endpoints and neighbouring relationships.

    Returns:
        The prompt; the model answers ``[{"id", "type"}, ...]``.
    """
    return (
        "You refine RELATED_TO relationships to a more specific predicate.\n"
        "Rules:\n"
        "- Use only the canonical list.\n"
        "- Keep direction and semantics.\n"
        "- Prefer the most specific relation; use RELATED_TO only if nothing fits.\n"
        'Return JSON array of objects: {"id": int, "type": str}.\n\n'
        "Canonical list:\n"
        f"{json.dumps(canonical, indent=2)}\n\n"
        "Items:\n"
        f"{json.dumps(items, indent=2)}"
    )


def _relation_reclass_prompt(allowed: list[str], items: list[dict[str, Any]]) -> str:
    """Build the prompt that reclassifies edges to an allowed list of types.

    Args:
        allowed: Allowed relationship types.
        items: Edges with their current type, endpoints and neighbours.

    Returns:
        The prompt; the model answers ``[{"id", "type"}, ...]``.
    """
    return (
        "You reclassify Neo4j relationships to a fixed allowed list.\n"
        "Rules:\n"
        "- Use only the allowed list.\n"
        "- Keep direction and semantics.\n"
        "- Use RELATED_TO only if nothing fits.\n"
        'Return JSON array of objects: {"id": int, "type": str}.\n\n'
        "Allowed list:\n"
        f"{json.dumps(allowed, indent=2)}\n\n"
        "Items:\n"
        f"{json.dumps(items, indent=2)}"
    )


def _classify_concepts_prompt(labels: list[str], nodes: list[dict[str, Any]]) -> str:
    """Build the prompt that assigns a label to ``Concept`` nodes.

    Args:
        labels: Allowed labels.
        nodes: Nodes with their name and neighbouring relationships.

    Returns:
        The prompt; the model answers ``[{"id", "label"}, ...]``.
    """
    return (
        "You assign a single label to each Concept node.\n"
        "Rules:\n"
        "- Use only the allowed labels list.\n"
        "- If unsure, return Concept.\n"
        'Return JSON array of objects: {"id": int, "label": str}.\n\n'
        "Allowed labels:\n"
        f"{json.dumps(labels, indent=2)}\n\n"
        "Nodes:\n"
        f"{json.dumps(nodes, indent=2)}"
    )


def _enrichment_prompt(
    schema: dict[str, dict[str, str]], nodes: list[dict[str, Any]]
) -> str:
    """Build the prompt that fills missing node properties.

    Args:
        schema: Properties per label, with their descriptions.
        nodes: Nodes with their label, name, missing properties and
            neighbouring relationships.

    Returns:
        The prompt; the model answers ``[{"id", "properties"}, ...]``.
    """
    return (
        "You enrich node properties for a knowledge graph.\n"
        "Rules:\n"
        "- Use only the properties defined in the schema per label.\n"
        "- Return only properties that are missing for the node.\n"
        "- If nothing to add, return an empty properties object.\n"
        'Return JSON array of objects: {"id": int, "properties": {..}}.\n\n'
        "Schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Nodes:\n"
        f"{json.dumps(nodes, indent=2)}"
    )


def _fallback_relation_target(source: str, canonical_set: set[str]) -> str:
    """Map a type to the canonical set without the LLM.

    Args:
        source: Relationship type.
        canonical_set: Normalised canonical types.

    Returns:
        The normalised type if canonical, else the type with an ``S``,
        ``ES``, ``ED`` or ``ING`` suffix removed if that is canonical, else
        ``RELATED_TO``.
    """
    normalized = _normalize_rel_type(source)
    if normalized in canonical_set:
        return normalized

    for suffix in ("S", "ES", "ED", "ING"):
        if normalized.endswith(suffix):
            candidate = normalized[: -len(suffix)]
            if candidate in canonical_set:
                return candidate

    return "RELATED_TO"


def _reltype_tokens(value: str) -> set[str]:
    """Split a relationship type into its significant words.

    Args:
        value: Relationship type.

    Returns:
        The ``_``-separated words, without stop words and single letters.
    """
    normalized = _normalize_rel_type(value)
    tokens = [tok for tok in normalized.split("_") if tok]
    return {
        tok
        for tok in tokens
        if tok not in _RELTYPE_STOP_TOKENS and (len(tok) > 1 or tok.isdigit())
    }


def _deterministic_relation_target(
    source: str,
    canonical_set: set[str],
    canonical_tokens: dict[str, set[str]],
) -> str:
    """Map a type to the canonical type sharing the most words with it.

    The best match maximises word overlap, then Jaccard similarity. It is
    accepted when it shares at least two words, or has a Jaccard similarity of
    at least 0.5, or shares one word with a type of at most two words at a
    similarity of at least 0.34.

    Args:
        source: Relationship type.
        canonical_set: Normalised canonical types.
        canonical_tokens: Significant words of each canonical type.

    Returns:
        The normalised type if canonical, else the accepted match, else
        ``RELATED_TO``.
    """
    normalized = _normalize_rel_type(source)
    if normalized in canonical_set:
        return normalized

    source_tokens = _reltype_tokens(normalized)
    if not source_tokens:
        return "RELATED_TO"

    best_target = "RELATED_TO"
    best_overlap = 0
    best_jaccard = 0.0

    for target, target_tokens in canonical_tokens.items():
        if target == "RELATED_TO" or not target_tokens:
            continue
        overlap = len(source_tokens & target_tokens)
        if overlap <= 0:
            continue
        union = len(source_tokens | target_tokens)
        jaccard = overlap / union if union else 0.0
        if overlap > best_overlap or (overlap == best_overlap and jaccard > best_jaccard):
            best_target = target
            best_overlap = overlap
            best_jaccard = jaccard

    if best_overlap >= 2:
        return best_target
    if best_jaccard >= 0.5:
        return best_target
    if best_overlap >= 1 and len(source_tokens) <= 2 and best_jaccard >= 0.34:
        return best_target
    return "RELATED_TO"


def _compact_relation_types_deterministic(
    session,
    canonical: list[str],
    dry_run: bool,
    apoc_available: bool,
    rare_threshold: int,
) -> dict[str, Any]:
    """Rename rare non-canonical relationship types without the LLM.

    Each non-canonical type with at most ``rare_threshold`` edges is renamed to
    :func:`_deterministic_relation_target`, which may be ``RELATED_TO``. More
    frequent non-canonical types are left alone and listed in the report.

    Args:
        session: Neo4j session.
        canonical: Canonical relationship types.
        dry_run: Report without writing.
        apoc_available: Whether APOC is installed; required unless dry run.
        rare_threshold: Maximum edge count of a type to be compacted.

    Returns:
        A report with type and edge counts, renamed samples, the skipped
        frequent types and ``errors``.
    """
    rows = session.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"
    ).data()
    canonical_set = {_normalize_rel_type(item) for item in canonical}
    canonical_tokens = {
        _normalize_rel_type(item): _reltype_tokens(item) for item in canonical
    }

    report: dict[str, Any] = {
        "total_relation_types": len(rows),
        "total_edges": sum(int(row.get("count", 0)) for row in rows),
        "rare_threshold": int(rare_threshold),
        "renamed_types": 0,
        "renamed_edges": 0,
        "collapsed_to_related_to_edges": 0,
        "already_canonical_types": 0,
        "skipped_high_freq_noncanonical": [],
        "renamed_samples": [],
        "errors": [],
    }

    if not dry_run and not apoc_available:
        report["errors"].append(
            "APOC unavailable, cannot rename relationship types for compaction"
        )
        return report

    for row in rows:
        source = str(row.get("type") or "").strip()
        count = int(row.get("count", 0))
        if not source:
            continue

        normalized = _normalize_rel_type(source)
        if normalized in canonical_set:
            report["already_canonical_types"] += 1
            continue

        if count > rare_threshold:
            if len(report["skipped_high_freq_noncanonical"]) < 200:
                report["skipped_high_freq_noncanonical"].append(
                    {"source": source, "count": count}
                )
            continue

        target = _deterministic_relation_target(
            source=source,
            canonical_set=canonical_set,
            canonical_tokens=canonical_tokens,
        )
        if not target or source == target:
            continue

        if not dry_run:
            try:
                session.run(
                    "CALL apoc.refactor.rename.type($old, $new)",
                    old=source,
                    new=target,
                ).consume()
            except Exception as exc:
                report["errors"].append(
                    f"compaction rename failed for {source} -> {target}: {exc}"
                )
                continue

        report["renamed_types"] += 1
        report["renamed_edges"] += count
        if target == "RELATED_TO":
            report["collapsed_to_related_to_edges"] += count
        if len(report["renamed_samples"]) < 250:
            report["renamed_samples"].append(
                {
                    "source": source,
                    "target": target,
                    "count": count,
                }
            )

    return report


# Pairs or node ids per query in the batched passes. Big enough to keep the
# number of round-trips small, small enough that one failed query does not
# lose the pass.
_BRIDGE_BATCH_SIZE = 1000


def _bridge_duplicate_name_groups(
    session,
    dry_run: bool,
    max_edges_per_group: int,
) -> dict[str, Any]:
    """Connect nodes that share a normalised name with ``RELATED_TO`` edges.

    In each duplicate-name group the node with the highest degree (then the
    lowest id) is the anchor; an edge anchor -> other is created for every
    other node not already connected to it, up to ``max_edges_per_group``
    per group. When the existence check fails for a batch, its pairs are
    treated as connected, so no duplicate edge is created.

    Args:
        session: Neo4j session.
        dry_run: Report without writing.
        max_edges_per_group: Edge cap per group; 0 or less means no cap.

    Returns:
        A report with group, pair and edge counts, samples and ``errors``.
    """
    groups = _find_duplicate_groups(session)
    report: dict[str, Any] = {
        "groups_considered": len(groups),
        "candidate_pairs": 0,
        "edges_created": 0,
        "skipped_already_connected": 0,
        "skipped_group_limit": 0,
        "samples": [],
        "errors": [],
    }

    # Every (anchor, other) pair the groups propose, group by group.
    pairs: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for group in groups:
        nodes = sorted(
            group["nodes"], key=lambda row: (-int(row.get("degree", 0)), int(row["id"]))
        )
        if len(nodes) < 2:
            continue
        anchor = nodes[0]
        anchor_id = int(anchor["id"])
        for node in nodes[1:]:
            other_id = int(node["id"])
            if other_id == anchor_id:
                continue
            pairs.append((anchor_id, other_id, group, node))

    # Existence is checked in chunks: one query per pair would cost a
    # round-trip per duplicate name, one query for all pairs would lose the
    # whole pass on a single failure.
    connected: set[tuple[int, int]] = set()
    for offset in range(0, len(pairs), _BRIDGE_BATCH_SIZE):
        chunk = pairs[offset : offset + _BRIDGE_BATCH_SIZE]
        payload = [{"a": a, "b": b} for a, b, _, _ in chunk]
        try:
            for row in session.run(
                "UNWIND $pairs AS pair "
                "MATCH (a)-[r]-(b) "
                "WHERE id(a) = pair.a AND id(b) = pair.b "
                "RETURN pair.a AS a, pair.b AS b, count(r) AS c",
                pairs=payload,
            ):
                if int(row["c"]) > 0:
                    connected.add((int(row["a"]), int(row["b"])))
        except Exception as exc:
            report["errors"].append(
                f"bridge existence check failed for {len(chunk)} pairs "
                f"at offset {offset}: {exc}"
            )
            # Unknown is not "unconnected": creating an edge that already exists
            # is the one outcome this pass must not produce.
            connected.update((a, b) for a, b, _, _ in chunk)

    # The per-group cap counts created edges only, so it is applied once the
    # existence answers are in.
    to_create: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    # Keyed by group identity, not by the normalized name: the cap is per
    # group, and nothing here guarantees the two are the same thing.
    local_edges: dict[int, int] = {}
    for anchor_id, other_id, group, node in pairs:
        key = id(group)
        if max_edges_per_group > 0 and local_edges.get(key, 0) >= max_edges_per_group:
            report["skipped_group_limit"] += 1
            continue
        if (anchor_id, other_id) in connected:
            report["skipped_already_connected"] += 1
            continue
        report["candidate_pairs"] += 1
        to_create.append((anchor_id, other_id, group, node))
        local_edges[key] = local_edges.get(key, 0) + 1

    created: set[tuple[int, int]] = set()
    if dry_run:
        created = {(a, b) for a, b, _, _ in to_create}
    else:
        for offset in range(0, len(to_create), _BRIDGE_BATCH_SIZE):
            chunk = to_create[offset : offset + _BRIDGE_BATCH_SIZE]
            payload = [{"a": a, "b": b} for a, b, _, _ in chunk]
            try:
                session.run(
                    "UNWIND $pairs AS pair "
                    "MATCH (a), (b) "
                    "WHERE id(a) = pair.a AND id(b) = pair.b "
                    "MERGE (a)-[:RELATED_TO]->(b)",
                    pairs=payload,
                ).consume()
            except Exception as exc:
                report["errors"].append(
                    f"bridge create failed for {len(chunk)} pairs "
                    f"at offset {offset}: {exc}"
                )
                continue
            created.update((a, b) for a, b, _, _ in chunk)

    for anchor_id, other_id, group, node in to_create:
        if (anchor_id, other_id) not in created:
            continue
        report["edges_created"] += 1
        if len(report["samples"]) < 100:
            anchor_name = next(
                (
                    str(row.get("name") or "")
                    for row in group["nodes"]
                    if int(row["id"]) == anchor_id
                ),
                "",
            )
            report["samples"].append(
                {
                    "normalized": group["normalized"],
                    "from": anchor_name,
                    "to": str(node.get("name") or ""),
                }
            )

    return report


def _run_semantic_compaction(
    session,
    relation_vocab: list[str],
    dry_run: bool,
    apoc_available: bool,
    rare_threshold: int,
    bridge_max_edges_per_group: int,
) -> dict[str, Any]:
    """Run ``--fix compact-semantic``: rare-type compaction, then name bridging.

    Args:
        session: Neo4j session.
        relation_vocab: Canonical relationship types.
        dry_run: Report without writing.
        apoc_available: Whether APOC is installed.
        rare_threshold: Maximum edge count of a type to be compacted.
        bridge_max_edges_per_group: Edge cap per duplicate-name group.

    Returns:
        ``{"relation_type_compaction": ..., "duplicate_name_bridging": ...}``.
    """
    report: dict[str, Any] = {}

    compaction = _compact_relation_types_deterministic(
        session=session,
        canonical=relation_vocab,
        dry_run=dry_run,
        apoc_available=apoc_available,
        rare_threshold=rare_threshold,
    )
    report["relation_type_compaction"] = compaction

    bridge = _bridge_duplicate_name_groups(
        session=session,
        dry_run=dry_run,
        max_edges_per_group=bridge_max_edges_per_group,
    )
    report["duplicate_name_bridging"] = bridge

    return report


def _labels_compatible(primary: list[str], secondary: list[str], mode: str) -> bool:
    """Whether two nodes' labels allow merging them.

    Args:
        primary: Labels of the node kept.
        secondary: Labels of the node merged into it.
        mode: ``"any"`` (always), ``"exact"`` (same label set) or any other
            value (at least one shared label).

    Returns:
        True when the nodes may be merged.
    """
    if mode == "any":
        return True
    primary_set = set(primary or [])
    secondary_set = set(secondary or [])
    if mode == "exact":
        return primary_set == secondary_set
    return bool(primary_set & secondary_set)


def _load_relation_vocab(path: str) -> list[str]:
    """Load the relation vocabulary, always including ``RELATED_TO``.

    Args:
        path: JSON array file; empty uses ``CANONICAL_RELATION_TYPES``.

    Returns:
        The types, stripped and upper-cased, with ``RELATED_TO`` prepended
        when missing.

    Raises:
        ValueError: If the file is not a JSON array.
    """
    if not path:
        vocab = list(_CANONICAL_RELATION_TYPES)
        if "RELATED_TO" not in vocab:
            vocab.insert(0, "RELATED_TO")
        return vocab
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("relation vocab must be a JSON array of strings")
    vocab = [str(item).strip().upper() for item in payload if str(item).strip()]
    if "RELATED_TO" not in vocab:
        vocab.insert(0, "RELATED_TO")
    return vocab


def _load_property_schema(path: str) -> dict[str, dict[str, str]]:
    """Load the property schema used by the enrichment step.

    Args:
        path: JSON object file mapping label to ``{property: description}``;
            empty uses ``_DEFAULT_PROPERTY_SCHEMA``.

    Returns:
        The schema, with non-object entries skipped and values as strings.

    Raises:
        ValueError: If the file is not a JSON object.
    """
    if not path:
        return dict(_DEFAULT_PROPERTY_SCHEMA)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("property schema must be a JSON object")
    output: dict[str, dict[str, str]] = {}
    for label, props in payload.items():
        if not isinstance(props, dict):
            continue
        output[str(label)] = {str(k): str(v) for k, v in props.items()}
    return output


def _fetch_node_context(session, ids: list[int]) -> list[dict[str, Any]]:
    """Fetch nodes with up to six of their relationships, for LLM prompts.

    Args:
        session: Neo4j session.
        ids: Internal node ids.

    Returns:
        Rows with ``id``, ``name``, ``labels`` and ``rels`` (type, neighbour
        name and labels).
    """
    if not ids:
        return []
    query = (
        "MATCH (n) WHERE id(n) IN $ids "
        "CALL { "
        "  WITH n "
        "  MATCH (n)-[r]-(m) "
        "  RETURN collect({type: type(r), neighbor: coalesce(m.name, ''), labels: labels(m)})[0..6] AS rels "
        "} "
        "RETURN id(n) AS id, n.name AS name, labels(n) AS labels, rels"
    )
    return session.run(query, ids=ids).data()


def _fetch_related_to_ids(session) -> list[int]:
    """Return the internal ids of every ``RELATED_TO`` edge, ascending."""
    rows = session.run(
        "MATCH ()-[r:RELATED_TO]->() RETURN id(r) AS id ORDER BY id(r)"
    ).data()
    return [int(row["id"]) for row in rows]


def _fetch_related_to_context(session, rel_ids: list[int]) -> list[dict[str, Any]]:
    """Fetch ``RELATED_TO`` edges with their endpoints and nearby edges.

    Args:
        session: Neo4j session.
        rel_ids: Internal relationship ids.

    Returns:
        Rows with ``id``, ``source``, ``target`` and up to three other
        relationships of each endpoint.
    """
    if not rel_ids:
        return []
    query = (
        "UNWIND $ids AS rid "
        "MATCH (s)-[r:RELATED_TO]->(t) "
        "WHERE id(r) = rid "
        "CALL { "
        "  WITH s, rid "
        "  MATCH (s)-[rs]-(sn) "
        "  WHERE id(rs) <> rid "
        "  RETURN collect({type: type(rs), neighbor: coalesce(sn.name, ''), labels: labels(sn)})[0..3] AS s_rels "
        "} "
        "CALL { "
        "  WITH t, rid "
        "  MATCH (t)-[rt]-(tn) "
        "  WHERE id(rt) <> rid "
        "  RETURN collect({type: type(rt), neighbor: coalesce(tn.name, ''), labels: labels(tn)})[0..3] AS t_rels "
        "} "
        "RETURN id(r) AS id, "
        "  {labels: labels(s), name: coalesce(s.name, '')} AS source, "
        "  {labels: labels(t), name: coalesce(t.name, '')} AS target, "
        "  s_rels AS source_context, t_rels AS target_context"
    )
    return session.run(query, ids=rel_ids).data()


def _fetch_relation_context(
    session, rel_ids: list[int], rel_type: str
) -> list[dict[str, Any]]:
    """Fetch edges of one type with their endpoints and nearby edges.

    Args:
        session: Neo4j session.
        rel_ids: Internal relationship ids.
        rel_type: Relationship type of those edges.

    Returns:
        Rows with ``id``, ``current_type``, ``source``, ``target`` and up to
        three other relationships of each endpoint.
    """
    if not rel_ids:
        return []
    safe_type = _normalize_rel_type(rel_type)
    query = (
        "UNWIND $ids AS rid "
        f"MATCH (s)-[r:`{safe_type}`]->(t) "
        "WHERE id(r) = rid "
        "CALL { "
        "  WITH s, rid "
        "  MATCH (s)-[rs]-(sn) "
        "  WHERE id(rs) <> rid "
        "  RETURN collect({type: type(rs), neighbor: coalesce(sn.name, ''), labels: labels(sn)})[0..3] AS s_rels "
        "} "
        "CALL { "
        "  WITH t, rid "
        "  MATCH (t)-[rt]-(tn) "
        "  WHERE id(rt) <> rid "
        "  RETURN collect({type: type(rt), neighbor: coalesce(tn.name, ''), labels: labels(tn)})[0..3] AS t_rels "
        "} "
        "RETURN id(r) AS id, type(r) AS current_type, "
        "  {labels: labels(s), name: coalesce(s.name, '')} AS source, "
        "  {labels: labels(t), name: coalesce(t.name, '')} AS target, "
        "  s_rels AS source_context, t_rels AS target_context"
    )
    return session.run(query, ids=rel_ids).data()


def _fetch_has_component_anomaly_ids(session) -> list[int]:
    """Return ids of implausible ``HAS_COMPONENT`` edges.

    These are Organization -> Concept and Concept -> Region edges.
    """
    rows = session.run(
        "MATCH (s:Organization)-[r:HAS_COMPONENT]->(t:Concept) RETURN id(r) AS id "
        "UNION "
        "MATCH (s:Concept)-[r:HAS_COMPONENT]->(t:Region) RETURN id(r) AS id"
    ).data()
    return [int(row["id"]) for row in rows]


def _coerce_value(value: Any) -> Any:
    """Convert a value to a primitive, a list of primitives, or ``None``.

    Args:
        value: Any value.

    Returns:
        Primitives unchanged, sequences and sets as lists with ``None``
        elements dropped, anything else as ``str(value)``.
    """
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            coerced = _coerce_value(item)
            if coerced is not None:
                out.append(coerced)
        return out
    return str(value)


def _sanitize_props(props: dict[str, Any]) -> dict[str, Any]:
    """Coerce property values and drop ``None`` and blank strings.

    Args:
        props: Properties returned by the LLM.

    Returns:
        A new map with string keys and storable values.
    """
    sanitized: dict[str, Any] = {}
    for key, value in props.items():
        coerced = _coerce_value(value)
        if coerced is None:
            continue
        if isinstance(coerced, str) and not coerced.strip():
            continue
        sanitized[str(key)] = coerced
    return sanitized


def _apply_relation_mapping(
    session,
    relation_items: list[dict[str, Any]],
    canonical: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Rename every relationship type to a canonical one (default step 1).

    Canonical types map to themselves. The others are mapped by the LLM in
    batches; an answer outside the canonical list, or no answer, falls back to
    :func:`_fallback_relation_target`. Renames use APOC.

    Args:
        session: Neo4j session.
        relation_items: Types with counts and patterns, from
            :func:`_fetch_relation_types`.
        canonical: Canonical relationship types.
        client: OpenAI-compatible client.
        model_name: Served model name.
        dry_run: Report without writing.
        batch_size: Types per LLM request.

    Returns:
        A report with ``renamed``, ``skipped`` and ``errors``.
    """
    report: dict[str, Any] = {
        "total_relation_types": len(relation_items),
        "renamed": [],
        "skipped": [],
        "errors": [],
    }

    canonical_set = {item.upper() for item in canonical}
    mapping: dict[str, str] = {}
    pending_items: list[dict[str, Any]] = []

    for item in relation_items:
        source = str(item.get("type", "")).strip()
        if not source:
            continue
        normalized = _normalize_rel_type(source)
        if normalized in canonical_set:
            mapping[source] = normalized
        else:
            pending_items.append(item)

    for batch in _chunked(pending_items, batch_size):
        prompt = _relation_mapping_prompt(canonical, batch)
        try:
            rows = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"llm mapping failed: {exc}")
            rows = []

        for row in rows:
            source = str(row.get("source", "")).strip()
            target = str(row.get("target", "")).strip().upper()
            if not source:
                continue
            if target not in canonical_set:
                target = ""
            if not target:
                target = _fallback_relation_target(source, canonical_set)
            mapping[source] = target

    for item in relation_items:
        source = item["type"]
        if source in mapping:
            continue
        mapping[source] = _fallback_relation_target(source, canonical_set)

    for source, target in mapping.items():
        if source == target:
            report["skipped"].append({"source": source, "target": target})
            continue

        source_safe = source.replace("`", "")
        count_query = f"MATCH ()-[r:`{source_safe}`]->() RETURN count(r) AS c"
        try:
            count = session.run(count_query).single()["c"]
        except Exception as exc:
            report["errors"].append(f"count failed for {source}: {exc}")
            count = 0

        if not dry_run:
            try:
                session.run(
                    "CALL apoc.refactor.rename.type($old, $new)", old=source, new=target
                )
            except Exception as exc:
                report["errors"].append(
                    f"rename failed for {source} -> {target}: {exc}"
                )
                continue

        report["renamed"].append(
            {"source": source, "target": target, "count": int(count)}
        )

    return report


def _rewrite_inverse_relationships(
    session,
    rewrites: list[dict[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    """Replace each ``(a)-[from]->(b)`` with ``(b)-[to]->(a)``.

    Properties are copied, and an existing ``to`` edge between the same nodes
    is reused (MERGE).

    Args:
        session: Neo4j session.
        rewrites: ``{"from": type, "to": inverse type}`` entries.
        dry_run: Report counts without writing.

    Returns:
        A report with per-pair ``count`` and ``rewritten``, and ``errors``.
    """
    report: dict[str, Any] = {"pairs": [], "errors": []}

    for item in rewrites:
        source = _normalize_rel_type(str(item.get("from", "")))
        target = _normalize_rel_type(str(item.get("to", "")))
        if not source or not target or source == target:
            continue

        count_query = f"MATCH ()-[r:`{source}`]->() RETURN count(r) AS c"
        try:
            count = session.run(count_query).single()["c"]
        except Exception as exc:
            report["errors"].append(f"inverse count failed for {source}: {exc}")
            continue

        if dry_run or int(count) == 0:
            report["pairs"].append(
                {
                    "source": source,
                    "target": target,
                    "count": int(count),
                    "rewritten": 0,
                }
            )
            continue

        query = (
            f"MATCH (a)-[r:`{source}`]->(b) "
            "WITH a, b, r, properties(r) AS props "
            f"MERGE (b)-[r2:`{target}`]->(a) "
            "SET r2 += props "
            "DELETE r "
            "RETURN count(r2) AS rewritten"
        )
        try:
            rewritten = session.run(query).single()["rewritten"]
        except Exception as exc:
            report["errors"].append(
                f"inverse rewrite failed for {source} -> {target}: {exc}"
            )
            continue

        report["pairs"].append(
            {
                "source": source,
                "target": target,
                "count": int(count),
                "rewritten": int(rewritten),
            }
        )

    return report


def _rename_relation_types(
    session,
    rewrites: list[dict[str, str]],
    dry_run: bool,
    apoc_available: bool,
) -> dict[str, Any]:
    """Rename relationship types, keeping edge direction.

    Uses APOC when available, otherwise copies each edge into a MERGEd edge of
    the new type and deletes the old one.

    Args:
        session: Neo4j session.
        rewrites: ``{"from": type, "to": new type}`` entries.
        dry_run: Report counts without writing.
        apoc_available: Whether APOC is installed.

    Returns:
        A report with per-pair ``count`` and ``updated``, and ``errors``.
    """
    report: dict[str, Any] = {"pairs": [], "errors": []}

    for item in rewrites:
        source = _normalize_rel_type(str(item.get("from", "")))
        target = _normalize_rel_type(str(item.get("to", "")))
        if not source or not target or source == target:
            continue

        try:
            count = _count_relationships(session, source)
        except Exception as exc:
            report["errors"].append(f"rename count failed for {source}: {exc}")
            continue

        updated = 0
        if not dry_run and count > 0:
            try:
                if apoc_available:
                    session.run(
                        "CALL apoc.refactor.rename.type($old, $new)",
                        old=source,
                        new=target,
                    ).consume()
                    updated = count
                else:
                    query = (
                        f"MATCH (a)-[r:`{source}`]->(b) "
                        "WITH a, b, r, properties(r) AS props "
                        f"MERGE (a)-[r2:`{target}`]->(b) "
                        "SET r2 += props "
                        "DELETE r "
                        "RETURN count(r2) AS updated"
                    )
                    updated = int(session.run(query).single()["updated"])
            except Exception as exc:
                report["errors"].append(
                    f"rename failed for {source} -> {target}: {exc}"
                )
                continue

        report["pairs"].append(
            {"source": source, "target": target, "count": count, "updated": updated}
        )

    return report


def _invert_published_direction(session, dry_run: bool) -> dict[str, Any]:
    """Turn ``Document -[PUBLISHED]-> Organization`` edges around.

    Args:
        session: Neo4j session.
        dry_run: Report the count without writing.

    Returns:
        A report with ``count``, ``rewritten`` and ``errors``.
    """
    report: dict[str, Any] = {"count": 0, "rewritten": 0, "errors": []}
    try:
        count = session.run(
            "MATCH (d:Document)-[r:PUBLISHED]->(o:Organization) RETURN count(r) AS c"
        ).single()["c"]
        report["count"] = int(count)
    except Exception as exc:
        report["errors"].append(f"published count failed: {exc}")
        return report

    if dry_run or report["count"] == 0:
        return report

    query = (
        "MATCH (d:Document)-[r:PUBLISHED]->(o:Organization) "
        "WITH d, o, r, properties(r) AS props "
        "MERGE (o)-[r2:PUBLISHED]->(d) "
        "SET r2 += props "
        "DELETE r "
        "RETURN count(r2) AS rewritten"
    )
    try:
        report["rewritten"] = int(session.run(query).single()["rewritten"])
    except Exception as exc:
        report["errors"].append(f"published inversion failed: {exc}")

    return report


def _cleanup_named_region_nodes(
    session, names: list[str], dry_run: bool
) -> dict[str, Any]:
    """Remove ``Region`` nodes with the given exact names.

    When another Region has the same name (case-insensitive, trimmed), the
    node's relationships are moved to it before the node is deleted;
    otherwise the node is deleted with its relationships. Requires APOC.

    Args:
        session: Neo4j session.
        names: Exact node names to remove.
        dry_run: Report without writing.

    Returns:
        A report with totals, samples, ``errors`` and a ``by_name``
        breakdown.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "matched": 0,
        "deleted_nodes": 0,
        "rewired_relationships": 0,
        "deleted_relationships": 0,
        "errors": [],
        "samples": [],
        "by_name": [],
    }

    cleaned_names = [str(name).strip() for name in names if str(name).strip()]
    if not cleaned_names:
        return report

    for name in cleaned_names:
        name_report = {
            "name": name,
            "found": False,
            "candidates": 0,
            "matched": 0,
            "deleted_nodes": 0,
            "rewired_relationships": 0,
            "deleted_relationships": 0,
            "errors": [],
            "samples": [],
        }
        literal = _cypher_string_literal(name)
        rows = session.run(
            "MATCH (n:Region) "
            f"WHERE n.name = {literal} "
            "CALL { "
            "  WITH n "
            "  OPTIONAL MATCH (m:Region) "
            "  WHERE id(m) <> id(n) "
            "    AND toLower(trim(m.name)) = toLower(trim(n.name)) "
            "  WITH m "
            "  ORDER BY id(m) "
            "  LIMIT 1 "
            "  RETURN id(m) AS match_id, m.name AS match_name "
            "} "
            "RETURN id(n) AS id, n.name AS name, match_id, match_name"
        ).data()

        name_report["candidates"] = len(rows)
        name_report["found"] = bool(rows)
        report["candidates"] += len(rows)

        for row in rows:
            bad_id = int(row["id"])
            match_id = row.get("match_id")
            match_id = int(match_id) if match_id is not None else None

            rel_count = 0
            try:
                rel_count = int(
                    session.run(
                        "MATCH (n) WHERE id(n) = $id MATCH (n)-[r]-() RETURN count(r) AS c",
                        id=bad_id,
                    ).single()["c"]
                )
            except Exception as exc:
                error_msg = f"region rel count failed for {bad_id}: {exc}"
                report["errors"].append(error_msg)
                name_report["errors"].append(error_msg)

            if match_id is not None:
                report["matched"] += 1
                report["deleted_nodes"] += 1
                name_report["matched"] += 1
                name_report["deleted_nodes"] += 1
                if len(report["samples"]) < 20:
                    report["samples"].append(
                        {"from": row.get("name", ""), "to": row.get("match_name", "")}
                    )
                if len(name_report["samples"]) < 20:
                    name_report["samples"].append(
                        {"from": row.get("name", ""), "to": row.get("match_name", "")}
                    )

                if dry_run:
                    report["rewired_relationships"] += rel_count
                    name_report["rewired_relationships"] += rel_count
                    continue

                query = (
                    "MATCH (bad:Region) WHERE id(bad) = $bad_id "
                    "MATCH (match:Region) WHERE id(match) = $match_id "
                    "CALL { "
                    "  WITH bad, match "
                    "  MATCH (bad)-[r]->(n) "
                    "  WHERE id(n) <> id(match) "
                    "  WITH match, n, r, type(r) AS rel_type, properties(r) AS props "
                    "  CALL apoc.create.relationship(match, rel_type, props, n) YIELD rel "
                    "  DELETE r "
                    "  RETURN count(rel) AS out_count "
                    "} "
                    "CALL { "
                    "  WITH bad, match "
                    "  MATCH (n)-[r]->(bad) "
                    "  WHERE id(n) <> id(match) "
                    "  WITH match, n, r, type(r) AS rel_type, properties(r) AS props "
                    "  CALL apoc.create.relationship(n, rel_type, props, match) YIELD rel "
                    "  DELETE r "
                    "  RETURN count(rel) AS in_count "
                    "} "
                    "DETACH DELETE bad "
                    "RETURN out_count + in_count AS rewired"
                )
                try:
                    rewired = int(
                        session.run(query, bad_id=bad_id, match_id=match_id).single()[
                            "rewired"
                        ]
                    )
                    report["rewired_relationships"] += rewired
                    name_report["rewired_relationships"] += rewired
                    deleted_rels = max(0, rel_count - rewired)
                    report["deleted_relationships"] += deleted_rels
                    name_report["deleted_relationships"] += deleted_rels
                except Exception as exc:
                    error_msg = f"region rewire failed for {bad_id}: {exc}"
                    report["errors"].append(error_msg)
                    name_report["errors"].append(error_msg)
                continue

            report["deleted_nodes"] += 1
            report["deleted_relationships"] += rel_count
            name_report["deleted_nodes"] += 1
            name_report["deleted_relationships"] += rel_count
            if len(report["samples"]) < 20:
                report["samples"].append({"from": row.get("name", ""), "to": None})
            if len(name_report["samples"]) < 20:
                name_report["samples"].append({"from": row.get("name", ""), "to": None})
            if dry_run:
                continue

            try:
                session.run(
                    "MATCH (n:Region) WHERE id(n) = $id DETACH DELETE n", id=bad_id
                ).consume()
            except Exception as exc:
                error_msg = f"region delete failed for {bad_id}: {exc}"
                report["errors"].append(error_msg)
                name_report["errors"].append(error_msg)

        report["by_name"].append(name_report)

    return report


def _isolated_delete_cap() -> int:
    """Return how many isolated nodes one run may delete before refusing.

    Returns:
        ``KG_ISOLATED_DELETE_MAX`` when it is a positive integer, else 500.
    """
    raw = os.getenv("KG_ISOLATED_DELETE_MAX", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 500
    return value if value > 0 else 500


def _cleanup_isolated_nodes(
    session,
    dry_run: bool,
    apoc_available: bool,
) -> dict[str, Any]:
    """Merge or delete named nodes that have no relationship.

    An isolated node is merged into the connected node with the same name
    (case-insensitive, trimmed) and the highest degree; without such a
    namesake it is deleted. ``:NodeVec`` nodes and nodes without a name are
    never touched. When there are more candidates than
    :func:`_isolated_delete_cap`, the run records an error and continues as a
    dry run.

    Args:
        session: Neo4j session.
        dry_run: Report without writing.
        apoc_available: Whether APOC is installed; merges require it.

    Returns:
        A report with candidate, merge and deletion counts, samples and
        ``errors``.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "matched": 0,
        "deleted_nodes": 0,
        "edges_modified": 0,
        "skipped": 0,
        "errors": [],
        "samples": [],
    }

    # :NodeVec carriers are isolated by design: one per entity, joined by the
    # `of` property rather than by an edge, and they hold the vector index.
    # They also have no `name`, so the merge branch below cannot match them and
    # they would fall through to DETACH DELETE. Both conditions are
    # load-bearing: the label keeps the vector index alive, the name check
    # keeps this pass to nodes it can reason about.
    rows = session.run(
        "MATCH (n) WHERE NOT (n)--() AND NOT n:NodeVec AND n.name IS NOT NULL "
        "RETURN id(n) AS id, n.name AS name"
    ).data()
    report["candidates"] = len(rows)
    if not rows:
        return report

    # Far more candidates than usual means the graph changed shape or a guard
    # above stopped working; either deserves a human before a DETACH DELETE.
    #
    # The refusal turns the run into a dry run rather than returning early, so
    # the report still lists exactly what would have been deleted. The error
    # makes `main()` exit non-zero.
    cap = _isolated_delete_cap()
    if not dry_run and len(rows) > cap:
        report["errors"].append(
            f"refusing to delete {len(rows)} isolated nodes: over the safety cap "
            f"of {cap}. Nothing was written and this run became a dry run. Check "
            "the sample below, then raise KG_ISOLATED_DELETE_MAX if it is right."
        )
        LOGGER.error(
            "Isolated-node cleanup refused: %d candidates over the cap of %d. "
            "Ran as a dry run instead; nothing was deleted.",
            len(rows),
            cap,
        )
        dry_run = True

    # Every candidate target is a node with at least one relationship, so one
    # query brings back all of them and the pick happens in memory: highest
    # degree first, lowest id to break ties.
    best_by_name: dict[str, dict[str, Any]] = {}
    for row in session.run(
        "MATCH (m)-[r]-() "
        "WHERE m.name IS NOT NULL AND trim(m.name) <> '' "
        "WITH m, count(r) AS degree "
        "RETURN id(m) AS id, m.name AS name, degree"
    ):
        normalized = str(row["name"]).strip().lower()
        current = best_by_name.get(normalized)
        candidate = {
            "id": int(row["id"]),
            "name": row["name"],
            "degree": int(row["degree"]),
        }
        if current is None or (candidate["degree"], -candidate["id"]) > (
            current["degree"],
            -current["id"],
        ):
            best_by_name[normalized] = candidate

    to_delete: list[int] = []
    for row in rows:
        node_id = int(row["id"])
        name = str(row.get("name") or "")
        normalized = name.strip().lower()
        match_row = best_by_name.get(normalized) if normalized else None

        if match_row:
            report["matched"] += 1
            if len(report["samples"]) < 20:
                report["samples"].append(
                    {"from": name, "to": match_row.get("name", "")}
                )
            if dry_run:
                continue
            if not apoc_available:
                report["skipped"] += 1
                report["errors"].append("APOC unavailable, cannot merge isolated nodes")
                continue
            # One query per merge on purpose: apoc.refactor.mergeNodes
            # rewrites relationships and invalidates ids, so batching several
            # merges in one statement risks a later row operating on a node the
            # earlier row already dissolved. Merges are the minority case —
            # most isolated nodes have no namesake and are deleted below, in
            # one query.
            try:
                session.run(
                    "MATCH (n) WHERE id(n) IN $ids "
                    "WITH n ORDER BY CASE id(n) WHEN $primary THEN 0 ELSE 1 END, id(n) "
                    "WITH collect(n) AS nodes "
                    "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                    "YIELD node RETURN id(node) AS id",
                    ids=[int(match_row["id"]), node_id],
                    primary=int(match_row["id"]),
                ).consume()
            except Exception as exc:
                report["errors"].append(
                    f"isolated node merge failed for {node_id}: {exc}"
                )
            continue

        report["deleted_nodes"] += 1
        if len(report["samples"]) < 20:
            report["samples"].append({"from": name, "to": None})
        if dry_run:
            continue
        to_delete.append(node_id)

    for offset in range(0, len(to_delete), _BRIDGE_BATCH_SIZE):
        chunk = to_delete[offset : offset + _BRIDGE_BATCH_SIZE]
        try:
            session.run(
                "MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", ids=chunk
            ).consume()
        except Exception as exc:
            report["errors"].append(
                f"isolated node delete failed for {len(chunk)} nodes "
                f"at offset {offset}: {exc}"
            )

    return report


def _find_duplicate_groups(session) -> list[dict[str, Any]]:
    """Group named nodes by :func:`_normalize_name`.

    Args:
        session: Neo4j session.

    Returns:
        Groups of two or more nodes, each ``{"normalized", "nodes"}`` with
        node rows holding ``id``, ``name``, ``labels`` and ``degree``.
    """
    rows = session.run(
        "MATCH (n) "
        "WHERE n.name IS NOT NULL AND trim(n.name) <> '' "
        "OPTIONAL MATCH (n)-[r]-() "
        "WITH n, count(r) AS degree "
        "RETURN id(n) AS id, n.name AS name, labels(n) AS labels, degree"
    ).data()

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        norm = _normalize_name(str(row.get("name", "")))
        if not norm:
            continue
        groups.setdefault(norm, []).append(row)

    return [
        {"normalized": norm, "nodes": items}
        for norm, items in groups.items()
        if len(items) > 1
    ]


def _merge_duplicate_groups(
    session,
    groups: list[dict[str, Any]],
    dry_run: bool,
    label_mode: str,
) -> dict[str, Any]:
    """Merge each duplicate-name group into its highest-degree node.

    Nodes whose labels are incompatible with the kept node (see
    :func:`_labels_compatible`) are left out. Merges use APOC, keeping the
    kept node's properties.

    Args:
        session: Neo4j session.
        groups: Groups from :func:`_find_duplicate_groups`.
        dry_run: Report without writing.
        label_mode: Label compatibility mode.

    Returns:
        A report with group and merge counts, samples and ``errors``.
    """
    report: dict[str, Any] = {
        "groups": 0,
        "merged_nodes": 0,
        "skipped_incompatible": 0,
        "errors": [],
        "samples": [],
    }

    for group in groups:
        nodes = group["nodes"]
        nodes_sorted = sorted(
            nodes, key=lambda row: (-int(row.get("degree", 0)), int(row["id"]))
        )
        primary = nodes_sorted[0]
        primary_labels = primary.get("labels", []) or []
        compatible = []
        for row in nodes_sorted[1:]:
            if _labels_compatible(primary_labels, row.get("labels", []), label_mode):
                compatible.append(row)
            else:
                report["skipped_incompatible"] += 1

        secondary_ids = [int(row["id"]) for row in compatible]
        report["groups"] += 1
        report["merged_nodes"] += len(secondary_ids)

        if len(report["samples"]) < 25:
            report["samples"].append(
                {
                    "normalized": group["normalized"],
                    "primary": primary["name"],
                    "secondary": [row["name"] for row in compatible],
                }
            )

        if dry_run or not secondary_ids:
            continue

        try:
            session.run(
                "MATCH (n) WHERE id(n) IN $ids "
                "WITH n ORDER BY CASE id(n) WHEN $primary THEN 0 ELSE 1 END, id(n) "
                "WITH collect(n) AS nodes "
                "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                "YIELD node RETURN id(node) AS id",
                ids=[int(primary["id"])] + secondary_ids,
                primary=int(primary["id"]),
            ).consume()
        except Exception as exc:
            report["errors"].append(f"merge failed for {group['normalized']}: {exc}")

    return report


def _normalize_all_caps_concepts(
    session,
    dry_run: bool,
    apoc_available: bool,
) -> dict[str, Any]:
    """Convert all-caps ``Concept`` names to title case.

    When a ``Concept`` with the title-cased name already exists, the all-caps
    node is merged into it (highest degree first); otherwise it is renamed.

    Args:
        session: Neo4j session.
        dry_run: Report without writing.
        apoc_available: Whether APOC is installed; merges require it.

    Returns:
        A report with merge and rename counts, samples and ``errors``.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "merged_nodes": 0,
        "renamed_nodes": 0,
        "edges_modified": 0,
        "skipped": 0,
        "errors": [],
        "samples": [],
    }

    rows = session.run(
        "MATCH (n:Concept) "
        "WHERE n.name IS NOT NULL AND trim(n.name) <> '' "
        "  AND n.name = toUpper(n.name) "
        "RETURN id(n) AS id, n.name AS name"
    ).data()
    report["candidates"] = len(rows)

    for row in rows:
        node_id = int(row["id"])
        name = str(row.get("name") or "")
        normalized = _to_title_case(name)
        if not normalized or normalized == name:
            continue

        match_row = session.run(
            "MATCH (m:Concept) "
            "WHERE id(m) <> $id AND m.name = $name "
            "OPTIONAL MATCH (m)-[r]-() "
            "WITH m, count(r) AS degree "
            "ORDER BY degree DESC, id(m) "
            "LIMIT 1 "
            "RETURN id(m) AS id, m.name AS name, degree",
            id=node_id,
            name=normalized,
        ).single()

        if match_row:
            report["merged_nodes"] += 1
            try:
                rel_count = int(
                    session.run(
                        "MATCH (n) WHERE id(n) = $id MATCH (n)-[r]-() RETURN count(r) AS c",
                        id=node_id,
                    ).single()["c"]
                )
            except Exception:
                rel_count = 0
            report["edges_modified"] += rel_count
            if len(report["samples"]) < 20:
                report["samples"].append({"from": name, "to": normalized})
            if dry_run:
                continue
            if not apoc_available:
                report["skipped"] += 1
                report["errors"].append("APOC unavailable, cannot merge Concept nodes")
                continue
            try:
                session.run(
                    "MATCH (n) WHERE id(n) IN $ids "
                    "WITH n ORDER BY CASE id(n) WHEN $primary THEN 0 ELSE 1 END, id(n) "
                    "WITH collect(n) AS nodes "
                    "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                    "YIELD node RETURN id(node) AS id",
                    ids=[int(match_row["id"]), node_id],
                    primary=int(match_row["id"]),
                ).consume()
            except Exception as exc:
                report["errors"].append(f"concept merge failed for {node_id}: {exc}")
            continue

        report["renamed_nodes"] += 1
        if len(report["samples"]) < 20:
            report["samples"].append({"from": name, "to": normalized})
        if dry_run:
            continue
        try:
            session.run(
                "MATCH (n:Concept) WHERE id(n) = $id SET n.name = $name",
                id=node_id,
                name=normalized,
            ).consume()
        except Exception as exc:
            report["errors"].append(f"concept rename failed for {node_id}: {exc}")

    return report


def _classify_concepts(
    session,
    labels: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
    label_mode: str,
) -> dict[str, Any]:
    """Ask the LLM for a more specific label for nodes labelled only Concept.

    Answers outside ``labels`` count as ``Concept`` and are not applied.

    Args:
        session: Neo4j session.
        labels: Allowed labels.
        client: OpenAI-compatible client.
        model_name: Served model name.
        dry_run: Report without writing.
        batch_size: Nodes per LLM request.
        label_mode: ``"add"`` keeps ``Concept`` next to the new label; any
            other value replaces it.

    Returns:
        A report with candidate and relabel counts, counts per label and
        ``errors``.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "relabeled": 0,
        "label_counts": {},
        "errors": [],
    }

    rows = session.run(
        "MATCH (n) WHERE 'Concept' IN labels(n) AND size(labels(n)) = 1 "
        "RETURN id(n) AS id, n.name AS name"
    ).data()

    concept_ids = [int(row["id"]) for row in rows]
    report["candidates"] = len(concept_ids)

    for batch in _chunked(concept_ids, batch_size):
        context_rows = _fetch_node_context(session, batch)
        nodes_payload = [
            {
                "id": row["id"],
                "name": row.get("name", ""),
                "context": row.get("rels", []),
            }
            for row in context_rows
        ]

        prompt = _classify_concepts_prompt(labels, nodes_payload)
        try:
            mapped = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"concept classification failed: {exc}")
            continue

        for item in mapped:
            node_id = int(item.get("id", -1))
            label = str(item.get("label", "Concept")).strip()
            if label not in labels:
                label = "Concept"

            if label == "Concept":
                continue

            report["label_counts"][label] = report["label_counts"].get(label, 0) + 1
            report["relabeled"] += 1

            if dry_run:
                continue

            safe_label = _sanitize_label(label)
            if label_mode == "add":
                session.run(
                    f"MATCH (n) WHERE id(n) = $id SET n:`{safe_label}`",
                    id=node_id,
                ).consume()
            else:
                session.run(
                    f"MATCH (n) WHERE id(n) = $id SET n:`{safe_label}` REMOVE n:Concept",
                    id=node_id,
                ).consume()

    return report


def _enrich_properties(
    session,
    schema: dict[str, dict[str, str]],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Ask the LLM to fill schema properties that nodes are missing.

    Only properties that are in the node's schema and missing on the node are
    written; ``None`` and blank values are dropped.

    Args:
        session: Neo4j session.
        schema: Properties per label, with their descriptions.
        client: OpenAI-compatible client.
        model_name: Served model name.
        dry_run: Report without writing.
        batch_size: Nodes per LLM request.

    Returns:
        A report with candidate and update counts, counts per property and
        ``errors``.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "updated_nodes": 0,
        "updated_props": {},
        "errors": [],
    }

    candidates: list[dict[str, Any]] = []
    candidate_lookup: dict[int, dict[str, Any]] = {}
    for label, props in schema.items():
        safe_label = _sanitize_label(label)
        rows = session.run(
            f"MATCH (n:`{safe_label}`) RETURN id(n) AS id, n.name AS name, properties(n) AS props"
        ).data()

        for row in rows:
            existing = row.get("props", {}) or {}
            missing = [
                key
                for key in props.keys()
                if key not in existing or existing.get(key) in (None, "")
            ]
            if not missing:
                continue
            candidate = {
                "id": int(row["id"]),
                "label": label,
                "name": row.get("name", ""),
                "missing": missing,
            }
            candidates.append(candidate)
            candidate_lookup[int(row["id"])] = candidate

    report["candidates"] = len(candidates)

    for batch in _chunked(candidates, batch_size):
        ids = [item["id"] for item in batch]
        context_rows = {row["id"]: row for row in _fetch_node_context(session, ids)}
        payload = []
        for item in batch:
            context = context_rows.get(item["id"], {})
            payload.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "name": item["name"],
                    "missing": item["missing"],
                    "context": context.get("rels", []),
                }
            )

        prompt = _enrichment_prompt(schema, payload)
        try:
            rows = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"property enrichment failed: {exc}")
            continue

        for row in rows:
            node_id = int(row.get("id", -1))
            props = row.get("properties") or {}
            if not isinstance(props, dict):
                continue
            candidate = candidate_lookup.get(node_id)
            if not candidate:
                continue
            allowed_keys = set(schema.get(candidate["label"], {}).keys())
            allowed_keys &= set(candidate.get("missing", []))

            filtered_props = {
                key: value for key, value in props.items() if key in allowed_keys
            }
            clean_props = _sanitize_props(filtered_props)
            if not clean_props:
                continue

            report["updated_nodes"] += 1
            for key in clean_props.keys():
                report["updated_props"][key] = report["updated_props"].get(key, 0) + 1

            if dry_run:
                continue

            session.run(
                "MATCH (n) WHERE id(n) = $id SET n += $props",
                id=node_id,
                props=clean_props,
            ).consume()

    return report


def _refine_related_to_relationships(
    session,
    canonical: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Ask the LLM for a more specific type for every ``RELATED_TO`` edge.

    Answers for ids outside the batch, or with a type outside ``canonical``,
    are ignored. Retyping uses APOC.

    Args:
        session: Neo4j session.
        canonical: Allowed relationship types.
        client: OpenAI-compatible client.
        model_name: Served model name.
        dry_run: Report without writing.

    Returns:
        A report with totals, counts per type and ``errors``.
    """
    report: dict[str, Any] = {
        "total_related_to": 0,
        "updated": 0,
        "skipped": 0,
        "type_counts": {},
        "batches": 0,
        "errors": [],
    }

    rel_ids = _fetch_related_to_ids(session)
    report["total_related_to"] = len(rel_ids)
    if not rel_ids:
        return report

    canonical_set = {item.upper() for item in canonical}

    for batch_ids in _chunked(rel_ids, _RELATED_TO_BATCH_SIZE):
        report["batches"] += 1
        batch_set = {int(item) for item in batch_ids}
        context_rows = _fetch_related_to_context(session, batch_ids)
        prompt = _related_to_refinement_prompt(canonical, context_rows)
        try:
            rows = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"RELATED_TO refinement failed: {exc}")
            continue

        updates: list[dict[str, Any]] = []
        for row in rows:
            try:
                rel_id = int(row.get("id", -1))
            except (TypeError, ValueError):
                rel_id = -1
            # An id the batch never asked about is a hallucination, and the
            # update query below matches on id alone: it would retype an
            # unrelated edge elsewhere in the graph.
            if rel_id < 0 or rel_id not in batch_set:
                report["skipped"] += 1
                continue

            raw_type = str(row.get("type", "")).strip()
            normalized = _normalize_rel_type(raw_type)
            if normalized not in canonical_set:
                normalized = "RELATED_TO"

            report["type_counts"][normalized] = (
                report["type_counts"].get(normalized, 0) + 1
            )
            if normalized == "RELATED_TO":
                report["skipped"] += 1
                continue

            updates.append({"id": rel_id, "type": normalized})

        if not updates or dry_run:
            report["updated"] += 0
            continue

        try:
            result = session.run(
                "UNWIND $updates AS item "
                "MATCH ()-[r]->() WHERE id(r) = item.id "
                "CALL apoc.refactor.setType(r, item.type) YIELD output "
                "RETURN count(output) AS updated",
                updates=updates,
            ).single()
            report["updated"] += int(result["updated"])
        except Exception as exc:
            report["errors"].append(f"RELATED_TO update failed: {exc}")

    return report


def _reclassify_relationships(
    session,
    rel_ids: list[int],
    rel_type: str,
    allowed: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
    skip_when_type: str | None = None,
    batch_label: str = "Reclass",
) -> dict[str, Any]:
    """Ask the LLM to retype a set of edges within an allowed list.

    Answers for ids outside the batch are ignored, ids left unanswered count
    as skipped, and a type outside ``allowed`` becomes ``RELATED_TO``.
    Retyping uses APOC.

    Args:
        session: Neo4j session.
        rel_ids: Internal ids of the edges to retype.
        rel_type: Current type of those edges.
        allowed: Allowed relationship types.
        client: OpenAI-compatible client.
        model_name: Served model name.
        dry_run: Report without writing.
        batch_size: Edges per LLM request.
        skip_when_type: Answers of this type are not applied.
        batch_label: Prefix of the progress log lines.

    Returns:
        A report with totals, counts per type and ``errors``.
    """
    report: dict[str, Any] = {
        "total_candidates": len(rel_ids),
        "updated": 0,
        "skipped": 0,
        "type_counts": {},
        "batches": 0,
        "errors": [],
    }

    if not rel_ids:
        return report

    allowed_set = {item.upper() for item in allowed}
    normalized_skip = _normalize_rel_type(skip_when_type) if skip_when_type else None

    total_batches = (len(rel_ids) + batch_size - 1) // batch_size

    for batch_index, batch_ids in enumerate(_chunked(rel_ids, batch_size), start=1):
        report["batches"] += 1
        LOGGER.info(
            "%s batch %d/%d: ids=%d",
            batch_label,
            batch_index,
            total_batches,
            len(batch_ids),
        )
        batch_set = {int(item) for item in batch_ids}
        context_rows = _fetch_relation_context(session, batch_ids, rel_type)
        prompt = _relation_reclass_prompt(allowed, context_rows)
        try:
            rows = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"relation reclass failed: {exc}")
            LOGGER.warning(
                "%s batch %d/%d failed: %s",
                batch_label,
                batch_index,
                total_batches,
                exc,
            )
            continue

        updates: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        batch_skipped = 0

        for row in rows:
            try:
                rel_id = int(row.get("id", -1))
            except (TypeError, ValueError):
                rel_id = -1
            if rel_id < 0 or rel_id not in batch_set:
                continue

            seen_ids.add(rel_id)
            raw_type = str(row.get("type", "")).strip()
            normalized = _normalize_rel_type(raw_type)
            if normalized not in allowed_set:
                normalized = "RELATED_TO"

            report["type_counts"][normalized] = (
                report["type_counts"].get(normalized, 0) + 1
            )
            if normalized_skip and normalized == normalized_skip:
                report["skipped"] += 1
                batch_skipped += 1
                continue

            updates.append({"id": rel_id, "type": normalized})

        missing = len(batch_set - seen_ids)
        if missing:
            report["skipped"] += missing
            batch_skipped += missing

        if not updates:
            LOGGER.info(
                "%s batch %d/%d: updates=0 skipped=%d",
                batch_label,
                batch_index,
                total_batches,
                batch_skipped,
            )
            continue

        if dry_run:
            LOGGER.info(
                "%s batch %d/%d: dry_run updates=%d skipped=%d",
                batch_label,
                batch_index,
                total_batches,
                len(updates),
                batch_skipped,
            )
            continue

        try:
            result = session.run(
                "UNWIND $updates AS item "
                "MATCH ()-[r]->() WHERE id(r) = item.id "
                "CALL apoc.refactor.setType(r, item.type) YIELD output "
                "RETURN count(output) AS updated",
                updates=updates,
            ).single()
            updated_count = int(result["updated"])
            report["updated"] += updated_count
            LOGGER.info(
                "%s batch %d/%d: updated=%d skipped=%d",
                batch_label,
                batch_index,
                total_batches,
                updated_count,
                batch_skipped,
            )
        except Exception as exc:
            report["errors"].append(f"relation reclass update failed: {exc}")
            LOGGER.warning(
                "%s batch %d/%d update failed: %s",
                batch_label,
                batch_index,
                total_batches,
                exc,
            )

    return report


def _reclassify_has_component_anomalies(
    session,
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Reclassify implausible ``HAS_COMPONENT`` edges within ``_AURA_RECLASS_TYPES``.

    Args:
        session: Neo4j session.
        client: OpenAI-compatible client.
        model_name: Served model name.
        dry_run: Report without writing.
        batch_size: Edges per LLM request.

    Returns:
        The :func:`_reclassify_relationships` report.
    """
    rel_ids = _fetch_has_component_anomaly_ids(session)
    return _reclassify_relationships(
        session=session,
        rel_ids=rel_ids,
        rel_type="HAS_COMPONENT",
        allowed=_AURA_RECLASS_TYPES,
        client=client,
        model_name=model_name,
        dry_run=dry_run,
        batch_size=batch_size,
        skip_when_type=None,
    )


def _reclassify_related_to_second_pass(
    session,
    client: OpenAI,
    model_name: str,
    allowed: list[str],
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Reclassify every ``RELATED_TO`` edge, keeping those still judged related.

    Args:
        session: Neo4j session.
        client: OpenAI-compatible client.
        model_name: Served model name.
        allowed: Allowed relationship types.
        dry_run: Report without writing.
        batch_size: Edges per LLM request.

    Returns:
        The :func:`_reclassify_relationships` report.
    """
    rel_ids = _fetch_related_to_ids(session)
    return _reclassify_relationships(
        session=session,
        rel_ids=rel_ids,
        rel_type="RELATED_TO",
        allowed=allowed,
        client=client,
        model_name=model_name,
        dry_run=dry_run,
        batch_size=batch_size,
        skip_when_type="RELATED_TO",
    )


def _find_region_artifacts(session) -> list[dict[str, Any]]:
    """Find ``Region`` nodes whose name looks like a table-header artifact.

    A name is suspect when it contains ``/`` or ``*``, or is all caps with
    more than three words.

    Args:
        session: Neo4j session.

    Returns:
        Rows with ``id``, ``name`` and, when another Region has the same name
        case-insensitively, its ``match_id`` and ``match_name``.
    """
    query = (
        "MATCH (n:Region) "
        "WHERE n.name IS NOT NULL AND trim(n.name) <> '' "
        "  AND ( "
        "    n.name CONTAINS '/' "
        "    OR n.name CONTAINS '*' "
        "    OR ( "
        "      n.name = toUpper(n.name) "
        "      AND size([w IN split(trim(n.name), ' ') WHERE w <> '']) > 3 "
        "    ) "
        "  ) "
        "CALL { "
        "  WITH n "
        "  MATCH (m:Region) "
        "  WHERE id(m) <> id(n) "
        "    AND toLower(trim(m.name)) = toLower(trim(n.name)) "
        "  RETURN id(m) AS match_id, m.name AS match_name "
        "  ORDER BY id(m) "
        "  LIMIT 1 "
        "} "
        "RETURN id(n) AS id, n.name AS name, match_id, match_name"
    )
    return session.run(query).data()


def _cleanup_region_artifacts(session, dry_run: bool) -> dict[str, Any]:
    """Merge artifact ``Region`` nodes into a namesake, or delete them.

    Args:
        session: Neo4j session.
        dry_run: Report without writing.

    Returns:
        A report with merge and deletion counts, samples and ``errors``.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "matched": 0,
        "deleted_nodes": 0,
        "rewired_relationships": 0,
        "deleted_relationships": 0,
        "errors": [],
        "samples": [],
    }

    artifacts = _find_region_artifacts(session)
    report["candidates"] = len(artifacts)

    for row in artifacts:
        bad_id = int(row["id"])
        match_id = row.get("match_id")
        match_id = int(match_id) if match_id is not None else None

        rel_count = 0
        try:
            rel_count = int(
                session.run(
                    "MATCH (n) WHERE id(n) = $id MATCH (n)-[r]-() RETURN count(r) AS c",
                    id=bad_id,
                ).single()["c"]
            )
        except Exception as exc:
            report["errors"].append(f"region rel count failed for {bad_id}: {exc}")

        if match_id is not None:
            report["matched"] += 1
            report["rewired_relationships"] += rel_count
            if len(report["samples"]) < 20:
                report["samples"].append(
                    {"from": row.get("name", ""), "to": row.get("match_name", "")}
                )
            if dry_run:
                continue
            try:
                session.run(
                    "MATCH (match:Region) WHERE id(match) = $match_id "
                    "MATCH (bad:Region) WHERE id(bad) = $bad_id "
                    "WITH [match, bad] AS nodes "
                    "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                    "YIELD node RETURN id(node) AS id",
                    match_id=match_id,
                    bad_id=bad_id,
                ).consume()
            except Exception as exc:
                report["errors"].append(f"region merge failed for {bad_id}: {exc}")
            continue

        report["deleted_nodes"] += 1
        report["deleted_relationships"] += rel_count
        if len(report["samples"]) < 20:
            report["samples"].append({"from": row.get("name", ""), "to": None})
        if dry_run:
            continue

        try:
            session.run(
                "MATCH (n:Region) WHERE id(n) = $id DETACH DELETE n", id=bad_id
            ).consume()
        except Exception as exc:
            report["errors"].append(f"region delete failed for {bad_id}: {exc}")

    return report


def _absorb_micro_relation_types(
    session,
    rewrites: list[dict[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    """Rename relationship types with APOC, keeping edge direction.

    Args:
        session: Neo4j session.
        rewrites: ``{"from": type, "to": new type}`` entries.
        dry_run: Report counts without writing.

    Returns:
        A report with per-pair ``count`` and ``updated``, and ``errors``.
    """
    report: dict[str, Any] = {"pairs": [], "errors": []}

    for item in rewrites:
        source = _normalize_rel_type(str(item.get("from", "")))
        target = _normalize_rel_type(str(item.get("to", "")))
        if not source or not target or source == target:
            continue

        count_query = f"MATCH ()-[r:`{source}`]->() RETURN count(r) AS c"
        try:
            count = int(session.run(count_query).single()["c"])
        except Exception as exc:
            report["errors"].append(f"micro type count failed for {source}: {exc}")
            continue

        updated = 0
        if not dry_run and count > 0:
            try:
                session.run(
                    "CALL apoc.refactor.rename.type($old, $new)", old=source, new=target
                ).consume()
                updated = count
            except Exception as exc:
                report["errors"].append(
                    f"micro type rename failed for {source} -> {target}: {exc}"
                )

        report["pairs"].append(
            {"source": source, "target": target, "count": count, "updated": updated}
        )

    return report


def _count_relationships(session, rel_type: str) -> int:
    """Count the edges of one relationship type."""
    safe_type = _normalize_rel_type(rel_type)
    query = f"MATCH ()-[r:`{safe_type}`]->() RETURN count(r) AS c"
    return int(session.run(query).single()["c"])


def _cleanup_mentioned_in(
    session,
    mode: str,
    dry_run: bool,
    prop_name: str,
    count_prop: str,
    mentions_prop: str,
) -> dict[str, Any]:
    """Delete ``MENTIONED_IN`` edges, optionally keeping them as properties.

    In ``"convert"`` mode each entity first receives the names of the
    documents it is mentioned in, their count and its mention count. In a dry
    run the conversion query still runs; the edges are not deleted.

    Args:
        session: Neo4j session.
        mode: ``"convert"`` or ``"delete"``.
        dry_run: Keep the edges.
        prop_name: Property holding the document names.
        count_prop: Property holding the number of documents.
        mentions_prop: Property holding the number of mentions.

    Returns:
        A report with edge and node counts and ``errors``.
    """
    report: dict[str, Any] = {
        "mode": mode,
        "relationships": 0,
        "nodes_updated": 0,
        "edges_deleted": 0,
        "errors": [],
    }

    try:
        report["relationships"] = _count_relationships(session, "MENTIONED_IN")
    except Exception as exc:
        report["errors"].append(f"count failed: {exc}")
        return report

    if report["relationships"] == 0:
        return report

    if mode == "convert":
        query = (
            "MATCH (e)-[r:MENTIONED_IN]->(d:Document) "
            "WITH e, collect(DISTINCT d.name) AS docs, count(r) AS mentions "
            "SET e[$prop_name] = docs, "
            "    e[$count_prop] = size(docs), "
            "    e[$mentions_prop] = mentions "
            "RETURN count(e) AS nodes_updated"
        )
        if dry_run:
            try:
                result = session.run(
                    query,
                    prop_name=prop_name,
                    count_prop=count_prop,
                    mentions_prop=mentions_prop,
                ).single()
                report["nodes_updated"] = int(result["nodes_updated"])
            except Exception as exc:
                report["errors"].append(f"convert dry-run failed: {exc}")
        else:
            try:
                result = session.run(
                    query,
                    prop_name=prop_name,
                    count_prop=count_prop,
                    mentions_prop=mentions_prop,
                ).single()
                report["nodes_updated"] = int(result["nodes_updated"])
            except Exception as exc:
                report["errors"].append(f"convert failed: {exc}")

    if dry_run:
        return report

    try:
        deleted = session.run(
            "MATCH ()-[r:MENTIONED_IN]->() DELETE r RETURN count(r) AS c"
        ).single()["c"]
        report["edges_deleted"] = int(deleted)
    except Exception as exc:
        report["errors"].append(f"delete failed: {exc}")

    return report


def _run_aura_issues(
    session,
    relation_vocab: list[str],
    dry_run: bool,
    batch_size: int,
    apoc_available: bool,
) -> dict[str, Any]:
    """Run ``--fix aura-issues``.

    1. Remove the ``Region`` nodes named in ``_AURA_REGION_GARBAGE_NAMES``.
    2. Reclassify ``RELATED_TO`` edges with the LLM, when there are more
       than 50.
    3. Absorb the types in ``_MICRO_RELATION_REWRITES``.

    Without APOC each step only counts its candidates and records an error.

    Args:
        session: Neo4j session.
        relation_vocab: Allowed relationship types.
        dry_run: Report without writing.
        batch_size: Edges per LLM request.
        apoc_available: Whether APOC is installed.

    Returns:
        One report entry per step, each with ``found``, the number of edges
        modified and the step's ``details``.

    Raises:
        ValueError: If the LLM is needed and ``VLLM_MODEL_NAME`` is not set.
    """
    report: dict[str, Any] = {}

    if apoc_available:
        garbage_report = _cleanup_named_region_nodes(
            session=session,
            names=_AURA_REGION_GARBAGE_NAMES,
            dry_run=dry_run,
        )
    else:
        total_candidates = 0
        for name in _AURA_REGION_GARBAGE_NAMES:
            literal = _cypher_string_literal(str(name).strip())
            count = session.run(
                f"MATCH (n:Region) WHERE n.name = {literal} RETURN count(n) AS c"
            ).single()["c"]
            total_candidates += int(count)
        garbage_report = {
            "candidates": total_candidates,
            "matched": 0,
            "deleted_nodes": 0,
            "rewired_relationships": 0,
            "deleted_relationships": 0,
            "errors": ["APOC unavailable, cannot rewire/delete Region garbage nodes"],
            "samples": [],
            "by_name": [],
        }

    garbage_found = int(garbage_report.get("candidates", 0)) > 0
    garbage_edges = int(garbage_report.get("rewired_relationships", 0)) + int(
        garbage_report.get("deleted_relationships", 0)
    )
    report["garbage_nodes"] = {
        "found": garbage_found,
        "deleted_nodes": int(garbage_report.get("deleted_nodes", 0)),
        "edges_modified": garbage_edges,
        "details": garbage_report,
    }
    LOGGER.info(
        "Issue 1 garbage nodes: %s (deleted_nodes=%d edges_modified=%d)",
        "TROVATO" if garbage_found else "NON TROVATO",
        int(garbage_report.get("deleted_nodes", 0)),
        garbage_edges,
    )

    related_total = _count_relationships(session, "RELATED_TO")
    if related_total > 50:
        if not apoc_available:
            related_report = {
                "total_candidates": related_total,
                "updated": 0,
                "skipped": related_total,
                "errors": ["APOC unavailable, cannot reclassify RELATED_TO"],
            }
        else:
            base_url, model_name, api_key = _resolve_llm_env()
            client = _build_llm_client(base_url=base_url, api_key=api_key)
            related_report = _reclassify_related_to_second_pass(
                session=session,
                client=client,
                model_name=model_name,
                allowed=relation_vocab,
                dry_run=dry_run,
                batch_size=batch_size,
            )
        report["related_to_reclass"] = {
            "found": True,
            "total_related_to": related_total,
            "edges_modified": int(related_report.get("updated", 0)),
            "details": related_report,
        }
        LOGGER.info(
            "Issue 2 RELATED_TO reclass: TROVATO (total=%d updated=%d)",
            related_total,
            int(related_report.get("updated", 0)),
        )
    else:
        report["related_to_reclass"] = {
            "found": False,
            "total_related_to": related_total,
            "edges_modified": 0,
            "details": {
                "total_candidates": related_total,
                "updated": 0,
                "skipped": related_total,
            },
        }
        LOGGER.info(
            "Issue 2 RELATED_TO reclass: NON TROVATO (total=%d <= 50)",
            related_total,
        )

    if apoc_available:
        micro_report = _absorb_micro_relation_types(
            session=session,
            rewrites=_MICRO_RELATION_REWRITES,
            dry_run=dry_run,
        )
    else:
        pairs = []
        for item in _MICRO_RELATION_REWRITES:
            source = _normalize_rel_type(str(item.get("from", "")))
            target = _normalize_rel_type(str(item.get("to", "")))
            if not source or not target or source == target:
                continue
            count = _count_relationships(session, source)
            pairs.append(
                {"source": source, "target": target, "count": count, "updated": 0}
            )
        micro_report = {
            "pairs": pairs,
            "errors": ["APOC unavailable, cannot rename relationship types"],
        }

    micro_pairs = micro_report.get("pairs", [])
    micro_found = any(int(pair.get("count", 0)) > 0 for pair in micro_pairs)
    micro_updated = sum(int(pair.get("updated", 0)) for pair in micro_pairs)
    report["micro_type_consolidation"] = {
        "found": micro_found,
        "edges_modified": micro_updated,
        "details": micro_report,
    }
    LOGGER.info(
        "Issue 3 micro-type consolidation: %s (edges_modified=%d)",
        "TROVATO" if micro_found else "NON TROVATO",
        micro_updated,
    )

    return report


def _run_verbose_relation_cleanup(
    session,
    dry_run: bool,
    apoc_available: bool,
) -> dict[str, Any]:
    """Apply ``_VERBOSE_RELATION_RENAMES`` and ``_VERBOSE_RELATION_INVERSES``.

    Args:
        session: Neo4j session.
        dry_run: Report without writing.
        apoc_available: Whether APOC is installed.

    Returns:
        ``{"rename": ..., "invert": ...}`` reports.
    """
    rename_report = _rename_relation_types(
        session=session,
        rewrites=_VERBOSE_RELATION_RENAMES,
        dry_run=dry_run,
        apoc_available=apoc_available,
    )
    invert_report = _rewrite_inverse_relationships(
        session=session,
        rewrites=_VERBOSE_RELATION_INVERSES,
        dry_run=dry_run,
    )
    return {"rename": rename_report, "invert": invert_report}


def _run_cleanup_pass3(
    session,
    relation_vocab: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    apoc_available: bool,
) -> dict[str, Any]:
    """Run ``--fix cleanup-pass3``.

    1. Merge or delete isolated nodes.
    2. Rename and invert verbose relationship types.
    3. Reclassify every ``RELATED_TO`` edge with the LLM.
    4. Title-case all-caps ``Concept`` names.

    Args:
        session: Neo4j session.
        relation_vocab: Allowed relationship types.
        client: OpenAI-compatible client.
        model_name: Served model name.
        dry_run: Report without writing.
        apoc_available: Whether APOC is installed.

    Returns:
        One report entry per step, each with ``found``, the number of nodes
        or edges modified and the step's ``details``.
    """
    report: dict[str, Any] = {}

    isolated_report = _cleanup_isolated_nodes(
        session=session,
        dry_run=dry_run,
        apoc_available=apoc_available,
    )
    isolated_found = int(isolated_report.get("candidates", 0)) > 0
    isolated_nodes = int(isolated_report.get("matched", 0)) + int(
        isolated_report.get("deleted_nodes", 0)
    )
    isolated_edges = int(isolated_report.get("edges_modified", 0))
    report["isolated_nodes"] = {
        "found": isolated_found,
        "nodes_modified": isolated_nodes,
        "edges_modified": isolated_edges,
        "details": isolated_report,
    }
    LOGGER.info(
        "Step 1 isolated nodes: %s (nodes_modified=%d edges_modified=%d)",
        "TROVATO" if isolated_found else "NON TROVATO",
        isolated_nodes,
        isolated_edges,
    )

    verbose_report = _run_verbose_relation_cleanup(
        session=session,
        dry_run=dry_run,
        apoc_available=apoc_available,
    )
    rename_pairs = verbose_report.get("rename", {}).get("pairs", [])
    invert_pairs = verbose_report.get("invert", {}).get("pairs", [])
    verbose_found = any(
        int(pair.get("count", 0)) > 0 for pair in rename_pairs + invert_pairs
    )
    verbose_edges = sum(int(pair.get("updated", 0)) for pair in rename_pairs) + sum(
        int(pair.get("rewritten", 0)) for pair in invert_pairs
    )
    report["verbose_relation_cleanup"] = {
        "found": verbose_found,
        "edges_modified": verbose_edges,
        "details": verbose_report,
    }
    LOGGER.info(
        "Step 2 verbose relation cleanup: %s (edges_modified=%d)",
        "TROVATO" if verbose_found else "NON TROVATO",
        verbose_edges,
    )

    related_total = _count_relationships(session, "RELATED_TO")
    if related_total > 0:
        related_report = _reclassify_relationships(
            session=session,
            rel_ids=_fetch_related_to_ids(session),
            rel_type="RELATED_TO",
            allowed=relation_vocab,
            client=client,
            model_name=model_name,
            dry_run=dry_run,
            batch_size=_RELATION_RECLASS_BATCH_SIZE,
            skip_when_type="RELATED_TO",
            batch_label="RELATED_TO pass3",
        )
        related_edges = int(related_report.get("updated", 0))
        report["related_to_pass3"] = {
            "found": True,
            "total_related_to": related_total,
            "edges_modified": related_edges,
            "details": related_report,
        }
        LOGGER.info(
            "Step 3 RELATED_TO pass3: TROVATO (total=%d updated=%d)",
            related_total,
            related_edges,
        )
    else:
        report["related_to_pass3"] = {
            "found": False,
            "total_related_to": related_total,
            "edges_modified": 0,
            "details": {
                "total_candidates": related_total,
                "updated": 0,
                "skipped": related_total,
            },
        }
        LOGGER.info(
            "Step 3 RELATED_TO pass3: NON TROVATO (total=%d)",
            related_total,
        )

    caps_report = _normalize_all_caps_concepts(
        session=session,
        dry_run=dry_run,
        apoc_available=apoc_available,
    )
    caps_found = int(caps_report.get("candidates", 0)) > 0
    caps_nodes = int(caps_report.get("merged_nodes", 0)) + int(
        caps_report.get("renamed_nodes", 0)
    )
    caps_edges = int(caps_report.get("edges_modified", 0))
    report["concept_caps_normalization"] = {
        "found": caps_found,
        "nodes_modified": caps_nodes,
        "edges_modified": caps_edges,
        "details": caps_report,
    }
    LOGGER.info(
        "Step 4 Concept all-caps normalization: %s (nodes_modified=%d edges_modified=%d)",
        "TROVATO" if caps_found else "NON TROVATO",
        caps_nodes,
        caps_edges,
    )

    return report


def _apply_constraints(
    session,
    all_labels: list[str],
    unique_labels: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Create name constraints for the ontology labels.

    Every label gets a ``name IS NOT NULL`` constraint and every label in
    ``unique_labels`` a ``name IS UNIQUE`` constraint, both ``IF NOT EXISTS``.

    Args:
        session: Neo4j session.
        all_labels: Labels that require a name.
        unique_labels: Labels whose names must be unique.
        dry_run: List the statements under ``skipped`` without running them.

    Returns:
        A report with the ``created`` and ``skipped`` statements and
        ``errors``.
    """
    report: dict[str, Any] = {"created": [], "skipped": [], "errors": []}

    for label in all_labels:
        safe_label = _sanitize_label(label)
        name = f"exists_{safe_label.lower()}_name"
        query = f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:`{safe_label}`) REQUIRE n.name IS NOT NULL"
        if dry_run:
            report["skipped"].append(query)
            continue
        try:
            session.run(query).consume()
            report["created"].append(query)
        except Exception as exc:
            report["errors"].append(f"constraint failed {name}: {exc}")

    for label in unique_labels:
        safe_label = _sanitize_label(label)
        name = f"uniq_{safe_label.lower()}_name"
        query = f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:`{safe_label}`) REQUIRE n.name IS UNIQUE"
        if dry_run:
            report["skipped"].append(query)
            continue
        try:
            session.run(query).consume()
            report["created"].append(query)
        except Exception as exc:
            report["errors"].append(f"constraint failed {name}: {exc}")

    return report


def main() -> None:
    """Parse the command line and run the default passes or a single ``--fix``.

    The JSON report is printed to stdout. The process exits with status 1
    when any step recorded an error.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kg_pipeline/config.yaml")
    parser.add_argument("--env-file", default="kg_pipeline/.env")
    parser.add_argument("--database", default="")
    parser.add_argument("--relation-vocab", default="")
    parser.add_argument("--property-schema", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--relation-batch-size", type=int, default=120)
    parser.add_argument("--concept-batch-size", type=int, default=50)
    parser.add_argument("--enrich-batch-size", type=int, default=30)
    parser.add_argument("--reltype-patterns", type=int, default=6)
    parser.add_argument(
        "--fix",
        choices=[
            "related-to",
            "region-artifacts",
            "micro-types",
            "aura-issues",
            "cleanup-pass3",
            "mentioned-in",
            "compact-semantic",
        ],
        default="",
        help="Run a single cleanup task and skip the default pipeline",
    )
    parser.add_argument(
        "--mentioned-in-mode",
        choices=["delete", "convert"],
        default="delete",
        help="How to handle MENTIONED_IN relationships when --fix mentioned-in",
    )
    parser.add_argument(
        "--mentioned-in-prop",
        default="source_documents",
        help="Node property name for document sources when converting",
    )
    parser.add_argument(
        "--mentioned-in-count-prop",
        default="source_documents_count",
        help="Node property name for document count when converting",
    )
    parser.add_argument(
        "--mentioned-in-mentions-prop",
        default="source_mentions",
        help="Node property name for mention count when converting",
    )
    parser.add_argument(
        "--dedup-label-mode",
        choices=["overlap", "exact", "any"],
        default="overlap",
        help="How strict label matching must be to merge duplicates",
    )
    parser.add_argument(
        "--concept-label-mode",
        choices=["replace", "add"],
        default="replace",
        help="Replace Concept label or add alongside it",
    )
    parser.add_argument(
        "--rewrite-inverses",
        action="store_true",
        help="Rewrite inverse relation pairs to a single direction",
    )
    parser.add_argument(
        "--rare-reltype-threshold",
        type=int,
        default=2,
        help="Compact non-canonical relationship types with count <= threshold",
    )
    parser.add_argument(
        "--duplicate-bridge-max-per-group",
        type=int,
        default=3,
        help="Maximum RELATED_TO bridges created per duplicate-name group",
    )
    args = parser.parse_args()

    log_path = None
    if args.log_file.strip():
        log_path = Path(args.log_file).expanduser()
    else:
        log_dir = Path("kg_pipeline") / "logs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"neo4j_postprocess_{timestamp}.log"
    _setup_logging(log_path)

    # override=False: an exported variable must win over the env file, which
    # points at the demo's live graph. This pass merges, relabels and deletes,
    # so a target set in the environment (e.g. a local staging instance) must
    # not be silently replaced. The env file still supplies everything the
    # operator did not set.
    load_dotenv(args.env_file, override=False)

    config = _load_yaml(Path(args.config))
    allowed_labels = [
        str(label) for label in config.get("ontology", {}).get("labels", [])
    ]
    if "Concept" not in allowed_labels:
        allowed_labels.append("Concept")

    non_concept_labels = [label for label in allowed_labels if label != "Concept"] + [
        "Concept"
    ]

    relation_vocab_path = args.relation_vocab.strip() or str(
        config.get("llm", {}).get("relation_vocab_path", "")
    ).strip()
    relation_vocab = _load_relation_vocab(relation_vocab_path)
    property_schema = _load_property_schema(args.property_schema)

    fix_mode = args.fix.strip()
    needs_llm = not fix_mode or fix_mode in {
        "related-to",
        "aura-issues",
        "cleanup-pass3",
    }

    base_url = ""
    model_name = ""
    api_key = ""
    client: OpenAI | None = None
    if needs_llm:
        base_url, model_name, api_key = _resolve_llm_env()
        client = _build_llm_client(base_url=base_url, api_key=api_key)

    uri, user, password, env_db = _resolve_neo4j_env()
    database = args.database.strip() or env_db

    _confirm_db_changes(
        uri=uri, database=database, dry_run=args.dry_run, assume_yes=args.yes
    )

    report: dict[str, Any] = {"dry_run": bool(args.dry_run)}

    LOGGER.info("Starting Neo4j postprocess dry_run=%s", args.dry_run)
    LOGGER.info("Target database=%s", database or "<default>")

    with neo4j_env.connect(
        neo4j_env.Neo4jTarget(uri, user, password, None)
    ) as driver:
        with driver.session(database=database) as session:
            apoc_available = _has_apoc(session)
            report["apoc_available"] = apoc_available
            LOGGER.info("APOC available=%s", apoc_available)

            if fix_mode:
                if fix_mode == "aura-issues":
                    report["aura_issues"] = _run_aura_issues(
                        session=session,
                        relation_vocab=relation_vocab,
                        dry_run=args.dry_run,
                        batch_size=_RELATION_RECLASS_BATCH_SIZE,
                        apoc_available=apoc_available,
                    )
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                    return

                if fix_mode == "mentioned-in":
                    report["mentioned_in_cleanup"] = _cleanup_mentioned_in(
                        session=session,
                        mode=args.mentioned_in_mode,
                        dry_run=args.dry_run,
                        prop_name=args.mentioned_in_prop,
                        count_prop=args.mentioned_in_count_prop,
                        mentions_prop=args.mentioned_in_mentions_prop,
                    )
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                    return

                if fix_mode == "compact-semantic":
                    LOGGER.info(
                        "Fix: semantic compaction (rare_reltype_threshold=%d bridge_max_edges_per_group=%d)",
                        int(args.rare_reltype_threshold),
                        int(args.duplicate_bridge_max_per_group),
                    )
                    report["semantic_compaction"] = _run_semantic_compaction(
                        session=session,
                        relation_vocab=relation_vocab,
                        dry_run=args.dry_run,
                        apoc_available=apoc_available,
                        rare_threshold=max(int(args.rare_reltype_threshold), 0),
                        bridge_max_edges_per_group=max(
                            int(args.duplicate_bridge_max_per_group), 0
                        ),
                    )
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                    return

                if not apoc_available:
                    report["error"] = "APOC unavailable, cleanup tasks require APOC"
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                    return

                if fix_mode == "cleanup-pass3":
                    LOGGER.info("Fix: cleanup pass3 tasks")
                    if client is None:
                        raise RuntimeError("LLM client required for cleanup pass3")
                    report["cleanup_pass3"] = _run_cleanup_pass3(
                        session=session,
                        relation_vocab=relation_vocab,
                        client=client,
                        model_name=model_name,
                        dry_run=args.dry_run,
                        apoc_available=apoc_available,
                    )
                elif fix_mode == "related-to":
                    LOGGER.info("Fix: refine RELATED_TO relationships")
                    if client is None:
                        raise RuntimeError(
                            "LLM client required for RELATED_TO refinement"
                        )
                    report["related_to_refinement"] = _refine_related_to_relationships(
                        session=session,
                        canonical=relation_vocab,
                        client=client,
                        model_name=model_name,
                        dry_run=args.dry_run,
                    )
                elif fix_mode == "region-artifacts":
                    LOGGER.info("Fix: cleanup Region header artifacts")
                    report["region_artifact_cleanup"] = _cleanup_region_artifacts(
                        session=session,
                        dry_run=args.dry_run,
                    )
                elif fix_mode == "micro-types":
                    LOGGER.info("Fix: absorb micro relationship types")
                    report["micro_type_absorption"] = _absorb_micro_relation_types(
                        session=session,
                        rewrites=_MICRO_RELATION_REWRITES,
                        dry_run=args.dry_run,
                    )

                print(json.dumps(report, ensure_ascii=False, indent=2))
                return

            relation_items = _fetch_relation_types(
                session, max_patterns=args.reltype_patterns
            )
            if not apoc_available:
                report["step1_relation_mapping"] = {
                    "error": "APOC unavailable, cannot rename relationship types",
                    "total_relation_types": len(relation_items),
                }
            else:
                LOGGER.info("Step 1: mapping %d relation types", len(relation_items))
                report["step1_relation_mapping"] = _apply_relation_mapping(
                    session=session,
                    relation_items=relation_items,
                    canonical=relation_vocab,
                    client=client,
                    model_name=model_name,
                    dry_run=args.dry_run,
                    batch_size=args.relation_batch_size,
                )
                step1 = report["step1_relation_mapping"]
                LOGGER.info(
                    "Step 1 done: renamed=%d skipped=%d errors=%d",
                    len(step1.get("renamed", [])),
                    len(step1.get("skipped", [])),
                    len(step1.get("errors", [])),
                )

            if args.rewrite_inverses:
                LOGGER.info("Step 1b: rewrite inverse relations")
                report["step1b_rewrite_inverses"] = _rewrite_inverse_relationships(
                    session=session,
                    rewrites=_INVERSE_RELATION_REWRITES,
                    dry_run=args.dry_run,
                )
                step1b = report["step1b_rewrite_inverses"]
                LOGGER.info(
                    "Step 1b done: pairs=%d errors=%d",
                    len(step1b.get("pairs", [])),
                    len(step1b.get("errors", [])),
                )

            aura_report: dict[str, Any] = {}

            LOGGER.info(
                "Aura cleanup step 1: invert PUBLISHED direction for Organization -> Document"
            )
            aura_report["step1_published_direction_fix"] = _invert_published_direction(
                session=session,
                dry_run=args.dry_run,
            )
            step_a1 = aura_report["step1_published_direction_fix"]
            LOGGER.info(
                "Aura step 1 done: count=%d rewritten=%d errors=%d",
                int(step_a1.get("count", 0)),
                int(step_a1.get("rewritten", 0)),
                len(step_a1.get("errors", [])),
            )

            if not apoc_available:
                aura_report["step2_region_garbage_cleanup"] = {
                    "error": "APOC unavailable, cannot rewire/delete Region garbage nodes"
                }
            else:
                LOGGER.info(
                    "Aura cleanup step 2: redirect/delete named Region garbage nodes"
                )
                aura_report["step2_region_garbage_cleanup"] = (
                    _cleanup_named_region_nodes(
                        session=session,
                        names=_AURA_REGION_GARBAGE_NAMES,
                        dry_run=args.dry_run,
                    )
                )
                step_a2 = aura_report["step2_region_garbage_cleanup"]
                LOGGER.info(
                    "Aura step 2 done: candidates=%d matched=%d deleted_nodes=%d rewired=%d deleted_rels=%d errors=%d",
                    int(step_a2.get("candidates", 0)),
                    int(step_a2.get("matched", 0)),
                    int(step_a2.get("deleted_nodes", 0)),
                    int(step_a2.get("rewired_relationships", 0)),
                    int(step_a2.get("deleted_relationships", 0)),
                    len(step_a2.get("errors", [])),
                )

            if not apoc_available:
                aura_report["step3_inverse_pairs"] = {
                    "error": "APOC unavailable, cannot rewrite inverse relationships"
                }
            else:
                LOGGER.info(
                    "Aura cleanup step 3: rewrite inverse pairs and rename INFLUENCES"
                )
                inverse_report = _rewrite_inverse_relationships(
                    session=session,
                    rewrites=_AURA_INVERSE_REWRITES,
                    dry_run=args.dry_run,
                )
                rename_report = _absorb_micro_relation_types(
                    session=session,
                    rewrites=_AURA_RENAME_REWRITES,
                    dry_run=args.dry_run,
                )
                aura_report["step3_inverse_pairs"] = {
                    "inverse": inverse_report,
                    "rename": rename_report,
                }
                LOGGER.info(
                    "Aura step 3 done: inverse_pairs=%d inverse_errors=%d rename_pairs=%d rename_errors=%d",
                    len(inverse_report.get("pairs", [])),
                    len(inverse_report.get("errors", [])),
                    len(rename_report.get("pairs", [])),
                    len(rename_report.get("errors", [])),
                )

            if client is None:
                raise RuntimeError(
                    "LLM client required for Aura relationship reclassification"
                )

            if not apoc_available:
                aura_report["step4_has_component_reclass"] = {
                    "error": "APOC unavailable, cannot reclassify HAS_COMPONENT relationships"
                }
            else:
                LOGGER.info(
                    "Aura cleanup step 4: reclassify anomalous HAS_COMPONENT relationships"
                )
                aura_report["step4_has_component_reclass"] = (
                    _reclassify_has_component_anomalies(
                        session=session,
                        client=client,
                        model_name=model_name,
                        dry_run=args.dry_run,
                        batch_size=_RELATION_RECLASS_BATCH_SIZE,
                    )
                )
                step_a4 = aura_report["step4_has_component_reclass"]
                LOGGER.info(
                    "Aura step 4 done: candidates=%d updated=%d skipped=%d errors=%d",
                    int(step_a4.get("total_candidates", 0)),
                    int(step_a4.get("updated", 0)),
                    int(step_a4.get("skipped", 0)),
                    len(step_a4.get("errors", [])),
                )

            if not apoc_available:
                aura_report["step5_related_to_reclass"] = {
                    "error": "APOC unavailable, cannot reclassify RELATED_TO relationships"
                }
            else:
                LOGGER.info(
                    "Aura cleanup step 5: second-pass reclassify RELATED_TO relationships"
                )
                aura_report["step5_related_to_reclass"] = (
                    _reclassify_related_to_second_pass(
                        session=session,
                        client=client,
                        model_name=model_name,
                        allowed=relation_vocab,
                        dry_run=args.dry_run,
                        batch_size=_RELATION_RECLASS_BATCH_SIZE,
                    )
                )
                step_a5 = aura_report["step5_related_to_reclass"]
                LOGGER.info(
                    "Aura step 5 done: candidates=%d updated=%d skipped=%d errors=%d",
                    int(step_a5.get("total_candidates", 0)),
                    int(step_a5.get("updated", 0)),
                    int(step_a5.get("skipped", 0)),
                    len(step_a5.get("errors", [])),
                )

            report["aura_cleanup"] = aura_report

            duplicate_groups = _find_duplicate_groups(session)
            if not apoc_available:
                report["step2_dedup"] = {
                    "error": "APOC unavailable, cannot merge duplicate nodes",
                    "candidate_groups": len(duplicate_groups),
                }
            else:
                LOGGER.info(
                    "Step 2: merging %d duplicate groups", len(duplicate_groups)
                )
                report["step2_dedup"] = _merge_duplicate_groups(
                    session=session,
                    groups=duplicate_groups,
                    dry_run=args.dry_run,
                    label_mode=args.dedup_label_mode,
                )
                step2 = report["step2_dedup"]
                LOGGER.info(
                    "Step 2 done: groups=%d merged_nodes=%d skipped_incompatible=%d errors=%d",
                    int(step2.get("groups", 0)),
                    int(step2.get("merged_nodes", 0)),
                    int(step2.get("skipped_incompatible", 0)),
                    len(step2.get("errors", [])),
                )

            LOGGER.info("Step 3: relabel Concept nodes")
            report["step3_relabel_concept"] = _classify_concepts(
                session=session,
                labels=non_concept_labels,
                client=client,
                model_name=model_name,
                dry_run=args.dry_run,
                batch_size=args.concept_batch_size,
                label_mode=args.concept_label_mode,
            )
            step3 = report["step3_relabel_concept"]
            LOGGER.info(
                "Step 3 done: candidates=%d relabeled=%d errors=%d",
                int(step3.get("candidates", 0)),
                int(step3.get("relabeled", 0)),
                len(step3.get("errors", [])),
            )

            LOGGER.info("Step 4: enrich node properties")
            report["step4_enrich_properties"] = _enrich_properties(
                session=session,
                schema=property_schema,
                client=client,
                model_name=model_name,
                dry_run=args.dry_run,
                batch_size=args.enrich_batch_size,
            )
            step4 = report["step4_enrich_properties"]
            LOGGER.info(
                "Step 4 done: candidates=%d updated_nodes=%d errors=%d",
                int(step4.get("candidates", 0)),
                int(step4.get("updated_nodes", 0)),
                len(step4.get("errors", [])),
            )

            unique_labels = [
                label
                for label in allowed_labels
                if label
                in {
                    "Organization",
                    "Region",
                    "Event",
                    "Indicator",
                    "Dataset",
                    "Method",
                    "Policy",
                    "Commodity",
                }
            ]
            LOGGER.info("Step 5: applying constraints")
            report["step5_constraints"] = _apply_constraints(
                session=session,
                all_labels=allowed_labels,
                unique_labels=unique_labels,
                dry_run=args.dry_run,
            )
            step5 = report["step5_constraints"]
            LOGGER.info(
                "Step 5 done: created=%d skipped=%d errors=%d",
                len(step5.get("created", [])),
                len(step5.get("skipped", [])),
                len(step5.get("errors", [])),
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Every step collects its failures into `errors` instead of raising, so
    # the exit status is derived from the report: a run whose steps all failed
    # must not look clean to a caller that only checks the status.
    failures = _collect_errors(report)
    if failures:
        LOGGER.error(
            "%d step(s) reported errors; see the report above. First: %s",
            len(failures),
            failures[0],
        )
        sys.exit(1)


def _collect_errors(report: object) -> list[str]:
    """Collect every ``errors`` entry anywhere in a nested report.

    Args:
        report: Report made of dicts, lists and tuples.

    Returns:
        The error messages, as strings, in traversal order.
    """
    found: list[str] = []
    if isinstance(report, dict):
        for key, value in report.items():
            if key == "errors" and isinstance(value, (list, tuple)):
                found.extend(str(item) for item in value)
            else:
                found.extend(_collect_errors(value))
    elif isinstance(report, (list, tuple)):
        for item in report:
            found.extend(_collect_errors(item))
    return found


if __name__ == "__main__":
    main()
