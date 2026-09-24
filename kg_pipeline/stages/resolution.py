"""Stage 4: resolve entity mentions to canonical entities."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from kg_pipeline.models.types import CanonicalEntityRecord, KGTriple
from datetime import datetime
from kg_pipeline.utils.acronym_map import expand_acronym
from kg_pipeline.utils.validation import parse_json_array


def _parse_llm_json_array(content: str) -> list:
    """Parse a JSON array from LLM output, tolerating fences and prose.

    Args:
        content: Raw model output.

    Returns:
        The parsed array. When the whole output does not parse, the text
        between the first ``[`` and the last ``]`` is parsed instead.

    Raises:
        ValueError: If no JSON array can be recovered.
    """
    try:
        return parse_json_array(content)
    except Exception:
        start = content.find("[")
        end = content.rfind("]")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, list):
            raise ValueError("LLM output is not a JSON array")
        return parsed


LOGGER = logging.getLogger("kg_pipeline")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

# Most-specific-first for the circular-food ontology; Concept is the fallback
# and must stay last.
_LABEL_PRECEDENCE = [
    "Person",
    "Organization",
    "Place",
    "Event",
    "Project",
    "Policy",
    "Document",
    "Indicator",
    "Method",
    "Product",
    "Material",
    "Process",
    "DataValue",
    "Concept",
]


def _majority_label(counts: Counter[str]) -> str:
    """The label most mentions of an entity carry; precedence breaks ties."""
    if not counts:
        return "Concept"
    top = max(counts.values())
    tied = [label for label, n in counts.items() if n == top]
    ranked = sorted(
        tied,
        key=lambda label: (
            _LABEL_PRECEDENCE.index(label) if label in _LABEL_PRECEDENCE else len(_LABEL_PRECEDENCE),
            label,
        ),
    )
    return ranked[0]


def _numbers(name: str) -> tuple[str, ...]:
    """The numbers written in ``name``, decimal comma read as a point."""
    return tuple(sorted(n.replace(",", ".") for n in _NUMBER.findall(name)))


def _drop_number_mismatches(
    approved: set[tuple[int, int]],
    mentions: list[dict[str, Any]],
    groups: list[list[int]],
) -> set[tuple[int, int]]:
    """Refuse approved pairs whose names carry different numbers.

    "3.3 ± 1.3" and "3.3 ± 1.0", "9 generation groups" and "8 generation
    groups" embed close and read alike to the model, but a measurement is its
    number. A name with no number can still merge with one that has a number
    ("17 SDGs" and "SDGs"); two names that both carry numbers merge only if the
    numbers are the same.

    Args:
        approved: Approved group pairs.
        mentions: Mention records.
        groups: Groups of mention indices.

    Returns:
        The approved pairs that survive.
    """

    def numbers_of(g: int) -> tuple[str, ...]:
        if not 0 <= g < len(groups) or not groups[g]:
            return ()
        return _numbers(str(mentions[groups[g][0]]["name"]))

    kept: set[tuple[int, int]] = set()
    for i, j in approved:
        a, b = numbers_of(i), numbers_of(j)
        if a and b and a != b:
            continue
        kept.add((i, j))
    if len(kept) < len(approved):
        LOGGER.info(
            "Refused %d approved merge pairs whose names carry different numbers",
            len(approved) - len(kept),
        )
    return kept


def _centre_clusters(
    groups: list[list[int]], approved: set[tuple[int, int]]
) -> dict[int, list[int]]:
    """Merge each group into at most one centre it was directly approved with.

    Approvals are not transitive. Each pair is judged on its own, so a chain of
    individually plausible pairs — "food" ~ "leftover food" ~ "food scraps" ~
    "food waste" — links things no single judgement would merge, and closing
    over the chains turns a few common concepts into hubs that swallow
    everything near them. Here the largest unassigned group becomes a centre,
    takes every unassigned group approved with it directly, and nothing
    reaches a centre through a third group.

    Args:
        groups: Groups of mention indices; a group's size orders the centres.
        approved: Approved group pairs; pairs outside the group range are
            logged and skipped.

    Returns:
        Mapping from centre group to the groups merged into it, centre first.
    """
    neighbours: dict[int, set[int]] = defaultdict(set)
    for i, j in approved:
        if 0 <= i < len(groups) and 0 <= j < len(groups):
            neighbours[i].add(j)
            neighbours[j].add(i)
        else:
            LOGGER.warning(
                "Approved merge pair (%d, %d) outside valid group range "
                "[0, %d) — skipped",
                i,
                j,
                len(groups),
            )

    def by_size(g: int) -> tuple[int, int]:
        return (-len(groups[g]), g)

    assigned: set[int] = set()
    clusters: dict[int, list[int]] = {}
    for centre in sorted(range(len(groups)), key=by_size):
        if centre in assigned:
            continue
        assigned.add(centre)
        members = [centre]
        for other in sorted(neighbours[centre], key=by_size):
            if other not in assigned:
                assigned.add(other)
                members.append(other)
        clusters[centre] = members
    return clusters


def _norm(text: str) -> str:
    """Lower-case ``text`` and drop every non-alphanumeric character."""
    return _NON_ALNUM.sub("", text.lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets; 1.0 when both are empty."""
    if not a and not b:
        return 1.0
    denom = len(a | b)
    if denom == 0:
        return 0.0
    return len(a & b) / float(denom)


def _load_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented UTF-8 JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_mentions(triples: list[KGTriple]) -> list[dict[str, Any]]:
    """Turn every subject and object of every triple into a mention record.

    Args:
        triples: Raw triples.

    Returns:
        Two mentions per triple (subject, then object), each a dict with
        ``name``, ``label`` (first label, or ``Concept``), ``doc`` (source
        document), ``properties`` and ``predicates`` (the triple's predicate).
    """
    mentions: list[dict[str, Any]] = []

    for triple in triples:
        src_doc = str(triple.relationship_properties.get("source_doc", "")).strip()
        predicate = triple.predicate

        mentions.append(
            {
                "name": triple.subject,
                "label": (
                    triple.subject_labels[0] if triple.subject_labels else "Concept"
                ),
                "doc": src_doc,
                "properties": dict(triple.subject_properties),
                "predicates": {predicate},
            }
        )
        mentions.append(
            {
                "name": triple.object,
                "label": (
                    triple.object_labels[0] if triple.object_labels else "Concept"
                ),
                "doc": src_doc,
                "properties": dict(triple.object_properties),
                "predicates": {predicate},
            }
        )

    return mentions


def _initial_groups(
    mentions: list[dict[str, Any]],
    acronym_map: dict[str, str],
    context_jaccard_floor: float,
) -> list[list[int]]:
    """Group mentions that share a label and a normalised name.

    Names are acronym-expanded and normalised with :func:`_norm`. Within one
    key, mentions are split greedily into clusters: a mention joins the first
    cluster whose combined predicates have a Jaccard similarity of at least
    ``context_jaccard_floor`` with its own.

    Args:
        mentions: Mention records from :func:`_build_mentions`.
        acronym_map: Mapping from acronym to long form.
        context_jaccard_floor: Minimum predicate overlap to share a group.

    Returns:
        Groups of mention indices.
    """
    groups_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)

    for idx, mention in enumerate(mentions):
        expanded = expand_acronym(mention["name"], acronym_map)
        key = (mention["label"], _norm(expanded))
        groups_by_key[key].append(idx)

    groups: list[list[int]] = []
    for _, idxs in groups_by_key.items():
        if len(idxs) == 1:
            groups.append(idxs)
            continue

        local_clusters: list[list[int]] = []
        for idx in idxs:
            placed = False
            for cluster in local_clusters:
                c_predicates = set()
                for cidx in cluster:
                    c_predicates |= set(mentions[cidx]["predicates"])
                if (
                    _jaccard(set(mentions[idx]["predicates"]), c_predicates)
                    >= context_jaccard_floor
                ):
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                local_clusters.append([idx])

        groups.extend(local_clusters)

    return groups


def _embedding_candidates(
    mentions: list[dict[str, Any]],
    groups: list[list[int]],
    embedding_model: str,
    threshold: float,
) -> list[tuple[int, int]]:
    """Propose group pairs whose representative names embed close together.

    Each group is represented by its longest name. Names are embedded with
    ``embedding_model`` on ``KG_EMBED_DEVICE`` (when set) and compared by
    cosine similarity.

    Args:
        mentions: Mention records.
        groups: Groups of mention indices.
        embedding_model: SentenceTransformer model name or path.
        threshold: Similarity a same-label pair must exceed. A pair across
            labels must exceed ``max(threshold, 0.92)``.

    Returns:
        Candidate ``(i, j)`` group index pairs with ``i < j``.
    """
    if len(groups) < 2:
        return []

    canonical_names: list[str] = []
    canonical_labels: list[str] = []

    for idxs in groups:
        names = sorted(
            {mentions[i]["name"] for i in idxs}, key=lambda x: (-len(x), x.lower())
        )
        canonical_names.append(names[0] if names else "")
        canonical_labels.append(mentions[idxs[0]]["label"])

    model = SentenceTransformer(
        embedding_model, device=os.environ.get("KG_EMBED_DEVICE") or None
    )
    emb = model.encode(canonical_names, normalize_embeddings=True)
    sims = np.matmul(emb, emb.T)

    # Cross-label pairs are kept as candidates (bilingual corpus: the same
    # entity is often typed differently in the two languages, e.g. "food
    # waste"[Product] vs "spreco alimentare"[Process]); the LLM confirmation
    # step receives both labels and decides. They need a stricter similarity
    # floor than same-label pairs, otherwise candidates explode.
    cross_label_threshold = max(threshold, 0.92)
    candidates: list[tuple[int, int]] = []
    n = len(canonical_names)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sims[i, j])
            if canonical_labels[i] == canonical_labels[j]:
                if sim > threshold:
                    candidates.append((i, j))
            elif sim > cross_label_threshold:
                candidates.append((i, j))
    return candidates


_CONFIRM_BATCH_SIZE = 40
_DEFAULT_CONCURRENT_REQUESTS = 8


def _confirm_prompt(doc: str, pairs: list[dict[str, Any]]) -> str:
    """Build the merge-confirmation prompt for one document's batch of pairs.

    Args:
        doc: Document the pairs are judged in, or ``"global"``.
        pairs: Pair payloads with group indices, names, labels and documents.

    Returns:
        The prompt text.
    """
    return f"""
You are resolving cross-document entities for a knowledge graph about circular economy
and food systems. The corpus is bilingual: the same real-world entity may appear with an
ENGLISH name in one document and an ITALIAN name in another (e.g. "circular economy" and
"economia circolare", "food waste" and "spreco alimentare"). Such cross-language pairs
SHOULD be merged when they denote the same entity or concept.

Document scope: {doc}

For each pair, decide if they refer to the same real-world entity.
Be conservative: if uncertain, return merge=false. Translation equivalence alone is
sufficient only when the meaning is clearly the same.
When left_label and right_label differ, merge only if the two names clearly denote the
same thing despite the typing difference; a concept and a publication named after it
(e.g. "sustainability" vs the journal "Sustainability") are NOT the same entity.
Do NOT merge a thing with an action or goal about that thing: "food waste" vs
"food waste reduction", "circular economy" vs "promoting the circular economy",
"CO2 emissions" vs "reducing CO2 emissions" are DIFFERENT entities.
Return only JSON array with objects:
{{"left_group": int, "right_group": int, "merge": true or false}}

Pairs:
{json.dumps(pairs, ensure_ascii=False, indent=2)}
""".strip()


def _approved_from_response(content: str, group_count: int) -> set[tuple[int, int]]:
    """Read the model's merge verdicts.

    Args:
        content: Raw model output.
        group_count: Number of groups; indices outside ``[0, group_count)``
            are logged and dropped.

    Returns:
        Approved pairs, each as a sorted ``(i, j)`` tuple.

    Raises:
        ValueError: If the output holds no JSON array.
        KeyError: If an approved item lacks a group index.
    """
    approved: set[tuple[int, int]] = set()
    for item in _parse_llm_json_array(content):
        if not isinstance(item, dict) or not bool(item.get("merge", False)):
            continue
        left_group = int(item["left_group"])
        right_group = int(item["right_group"])
        if not (0 <= left_group < group_count and 0 <= right_group < group_count):
            LOGGER.warning(
                "LLM merge pair (%d, %d) outside valid group range [0, %d) — skipped",
                left_group,
                right_group,
                group_count,
            )
            continue
        approved.add(tuple(sorted((left_group, right_group))))
    return approved


async def _confirm_batch_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    doc: str,
    pairs: list[dict[str, Any]],
    model_name: str,
    group_count: int,
) -> set[tuple[int, int]]:
    """Ask the model to confirm one batch of candidate pairs.

    Args:
        client: Async OpenAI-compatible client.
        semaphore: Limits concurrent requests.
        doc: Document the pairs are judged in.
        pairs: Pair payloads.
        model_name: Served model name.
        group_count: Number of groups, for index validation.

    Returns:
        The approved pairs; empty when the request or the parse fails.
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=[{"role": "user", "content": _confirm_prompt(doc, pairs)}],
            )
            return _approved_from_response(
                response.choices[0].message.content or "[]", group_count
            )
        except Exception as exc:  # noqa: BLE001 - one bad batch must not lose the stage
            LOGGER.warning(
                "LLM merge confirmation failed for doc=%s (%d pairs skipped): %s",
                doc,
                len(pairs),
                exc,
            )
            return set()


async def _confirm_batches_async(
    doc_batches: list[tuple[str, list[dict[str, Any]]]],
    base_url: str,
    api_key: str,
    http_timeout: float,
    concurrent_requests: int,
    model_name: str,
    group_count: int,
) -> list[set[tuple[int, int]]]:
    """Confirm every batch concurrently.

    Args:
        doc_batches: ``(doc, pairs)`` batches.
        base_url: vLLM server base URL.
        api_key: API key; empty becomes ``"EMPTY"``.
        http_timeout: Request timeout in seconds.
        concurrent_requests: Maximum requests in flight.
        model_name: Served model name.
        group_count: Number of groups, for index validation.

    Returns:
        The approved pairs of each batch, in completion order.
    """
    async with AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "EMPTY",
        timeout=http_timeout,
    ) as client:
        semaphore = asyncio.Semaphore(concurrent_requests)
        coros = [
            _confirm_batch_async(
                client=client,
                semaphore=semaphore,
                doc=doc,
                pairs=pairs,
                model_name=model_name,
                group_count=group_count,
            )
            for doc, pairs in doc_batches
        ]
        results: list[set[tuple[int, int]]] = []
        with tqdm(
            total=len(coros), desc="Stage 4 LLM Merge Confirm", unit="batch"
        ) as progress:
            for coro in asyncio.as_completed(coros):
                results.append(await coro)
                progress.update(1)
        return results


def _confirm_candidates_with_llm(
    base_url: str,
    api_key: str,
    model_name: str,
    mentions: list[dict[str, Any]],
    groups: list[list[int]],
    candidates: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Ask the LLM which candidate pairs denote the same entity.

    Each pair is judged once, in the batch of the first document (in sorted
    order) that either group appears in, or ``"global"`` when neither has a
    document. Batches hold up to ``_CONFIRM_BATCH_SIZE`` pairs and run with
    ``GRAPHRAG_LLM_CONCURRENT_REQUESTS`` concurrent requests (default 8).

    Args:
        base_url: vLLM server base URL.
        api_key: API key.
        model_name: Served model name.
        mentions: Mention records.
        groups: Groups of mention indices.
        candidates: Candidate group pairs.

    Returns:
        The approved pairs, as sorted ``(i, j)`` tuples.
    """
    if not candidates:
        return set()

    approved: set[tuple[int, int]] = set()

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, j in candidates:
        left_docs = sorted(
            {mentions[idx]["doc"] for idx in groups[i] if mentions[idx]["doc"]}
        )
        right_docs = sorted(
            {mentions[idx]["doc"] for idx in groups[j] if mentions[idx]["doc"]}
        )
        docs = sorted(set(left_docs + right_docs)) or ["global"]

        left_name = sorted(
            {mentions[idx]["name"] for idx in groups[i]},
            key=lambda x: (-len(x), x.lower()),
        )[0]
        right_name = sorted(
            {mentions[idx]["name"] for idx in groups[j]},
            key=lambda x: (-len(x), x.lower()),
        )[0]

        pair_payload = {
            "left_group": i,
            "right_group": j,
            "left_name": left_name,
            "right_name": right_name,
            "left_label": mentions[groups[i][0]]["label"],
            "right_label": mentions[groups[j][0]]["label"],
            "left_docs": left_docs,
            "right_docs": right_docs,
        }

        # One vote per pair, cast in the first document that carries it.
        # Judging a pair once per document it touches would let a single
        # merge:true among several verdicts merge the entities, against the
        # prompt's "if uncertain, return merge=false".
        by_doc[docs[0]].append(pair_payload)

    doc_batches: list[tuple[str, list[dict[str, Any]]]] = []
    for doc, pairs in by_doc.items():
        for start in range(0, len(pairs), _CONFIRM_BATCH_SIZE):
            doc_batches.append((doc, pairs[start : start + _CONFIRM_BATCH_SIZE]))

    # Batches are confirmed concurrently, with the same knob as stage 3
    # (GRAPHRAG_LLM_CONCURRENT_REQUESTS). Approvals land in a set, so the
    # completion order cannot change what is merged.
    try:
        concurrent_requests = int(
            os.getenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", str(_DEFAULT_CONCURRENT_REQUESTS))
        )
    except ValueError:
        concurrent_requests = _DEFAULT_CONCURRENT_REQUESTS
    concurrent_requests = max(1, concurrent_requests)
    http_timeout = float(os.getenv("VLLM_HTTP_TIMEOUT", "900"))

    for batch_approved in asyncio.run(
        _confirm_batches_async(
            doc_batches=doc_batches,
            base_url=base_url,
            api_key=api_key,
            http_timeout=http_timeout,
            concurrent_requests=concurrent_requests,
            model_name=model_name,
            group_count=len(groups),
        )
    ):
        approved.update(batch_approved)

    return approved


def _group_fingerprint(
    mentions: list[dict[str, Any]], groups: list[list[int]]
) -> str:
    """Stable hash of the group construction the merge indices refer to.

    Args:
        mentions: Mention records, in the order the groups index into.
        groups: Mention-index groups from :func:`_initial_groups`.

    Returns:
        A hex digest that changes whenever the grouping changes.
    """
    hasher = hashlib.sha256()
    hasher.update(str(len(groups)).encode("utf-8"))
    for group in groups:
        # The first mention's identity is enough to detect re-indexing without
        # hashing the whole corpus.
        head = mentions[group[0]] if group else {}
        hasher.update(
            f"|{len(group)}:{head.get('label', '')}:{head.get('name', '')}".encode(
                "utf-8"
            )
        )
    return hasher.hexdigest()[:16]


def _cached_pairs_if_current(
    payload: Any, fingerprint: str, path: Path | None
) -> set[tuple[int, int]] | None:
    """Return the cached merge pairs, or None when the cache cannot be trusted.

    Args:
        payload: Parsed cache file content, or None when absent.
        fingerprint: Fingerprint of the current group construction.
        path: Cache path, for logging.

    Returns:
        The approved pairs, or None to force LLM re-confirmation.
    """
    if payload is None:
        return None
    if isinstance(payload, list):
        LOGGER.warning(
            "Merge cache %s predates fingerprinting: its group indices cannot be "
            "verified against the current grouping. Ignoring it and "
            "re-confirming with the LLM.",
            path,
        )
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Merge cache %s is malformed; ignoring it.", path)
        return None
    if payload.get("group_fingerprint") != fingerprint:
        LOGGER.warning(
            "Merge cache %s was built for a different group construction "
            "(fingerprint %s, now %s): its indices point at other entities. "
            "Ignoring it and re-confirming with the LLM.",
            path,
            payload.get("group_fingerprint"),
            fingerprint,
        )
        return None
    return {(int(a), int(b)) for a, b in payload.get("pairs", [])}


def _pick_canonical_name(aliases: list[str], alias_documents: dict[str, set[str]]) -> str:
    """Choose the name of a merged entity among its aliases.

    The ranking matches the one ``kg_densify`` uses for its inventory. A name
    of up to five words is a term, a longer one is a phrase that mentions the
    term, so terms come first. Among terms, the one used by more documents is
    the more established. At equal standing a multi-word name beats a single
    word, so ``European Union`` wins over ``EU``: an acronym the graph never
    expands is a dead end for the reader. Remaining ties go to the shorter
    string, then alphabetically, so the choice is deterministic.

    Args:
        aliases: Surface forms of the entity; must not be empty.
        alias_documents: Mapping from alias to the documents that use it.

    Returns:
        The chosen canonical name.
    """

    def rank(alias: str) -> tuple[int, int, int, int, str]:
        """Sort key implementing the ranking above; lower is better."""
        words = len(alias.split())
        return (
            1 if words > 5 else 0,
            -len(alias_documents.get(alias, ())),
            1 if words == 1 else 0,
            len(alias),
            alias.lower(),
        )

    return sorted(aliases, key=rank)[0]


def resolve_entities(
    triples: list[KGTriple],
    acronym_map: dict[str, str],
    embedding_model: str,
    similarity_threshold: float,
    context_jaccard_floor: float,
    base_url: str | None,
    api_key: str | None,
    model_name: str | None,
    crosslabel_log_path: Path | None = None,
    merge_cache_path: Path | None = None,
) -> tuple[list[KGTriple], dict[str, CanonicalEntityRecord]]:
    """Merge entity mentions into canonical entities and rewrite the triples.

    1. Subjects and objects become mentions, grouped by label and normalised,
       acronym-expanded name, and split by predicate overlap.
    2. Group pairs with similar name embeddings become merge candidates.
    3. Candidates are confirmed by the LLM, or the approvals are loaded from
       ``merge_cache_path`` when it was built for the same grouping. Without a
       usable cache or an LLM endpoint, no candidate is merged.
    4. Approved pairs merge each group into one centre it was approved with
       directly (:func:`_centre_clusters`), never through a chain; each merged
       group becomes a registry entry named by :func:`_pick_canonical_name`.
    5. Entries whose names differ only in case are merged, keeping the most
       specific label.
    6. Triple subjects and objects are renamed to their canonical names and
       take the registry's properties and labels.

    Args:
        triples: Raw triples; modified in place.
        acronym_map: Mapping from acronym to long form.
        embedding_model: SentenceTransformer model for candidate generation.
        similarity_threshold: Embedding similarity a same-label pair must
            exceed to become a candidate.
        context_jaccard_floor: Minimum predicate overlap within a group.
        base_url: vLLM server base URL, or ``None`` to skip LLM confirmation.
        api_key: API key.
        model_name: Served model name, or ``None`` to skip LLM confirmation.
        crosslabel_log_path: JSONL log of case-variant merges, if wanted.
        merge_cache_path: Cache of LLM-approved pairs; read when present and
            written after an LLM confirmation.

    Returns:
        ``(resolved_triples, registry)``, the registry keyed by canonical name.
    """
    mentions = _build_mentions(triples)
    groups = _initial_groups(
        mentions, acronym_map, context_jaccard_floor=context_jaccard_floor
    )

    candidates = _embedding_candidates(
        mentions=mentions,
        groups=groups,
        embedding_model=embedding_model,
        threshold=similarity_threshold,
    )

    # Cached pairs are bare group indices, meaningful only for the group
    # construction that produced them. The fingerprint detects a cache built
    # for a different grouping, whose indices would point at other entities.
    group_fingerprint = _group_fingerprint(mentions, groups)

    approved: set[tuple[int, int]] = set()
    cached_payload: Any = None
    if merge_cache_path is not None and merge_cache_path.exists():
        cached_payload = json.loads(merge_cache_path.read_text(encoding="utf-8"))

    cached_pairs = _cached_pairs_if_current(
        cached_payload, group_fingerprint, merge_cache_path
    )
    if cached_pairs is not None:
        approved = cached_pairs
        LOGGER.info(
            "Loaded %d approved merge pairs from %s — skipping LLM confirmation",
            len(approved),
            merge_cache_path,
        )
    elif base_url and model_name:
        approved = _confirm_candidates_with_llm(
            base_url=base_url,
            api_key=(api_key or "EMPTY"),
            model_name=model_name,
            mentions=mentions,
            groups=groups,
            candidates=candidates,
        )
        if merge_cache_path is not None:
            merge_cache_path.write_text(
                json.dumps(
                    {
                        "group_fingerprint": group_fingerprint,
                        "n_groups": len(groups),
                        "pairs": sorted(approved),
                    }
                ),
                encoding="utf-8",
            )
            LOGGER.info(
                "Saved %d approved merge pairs to %s",
                len(approved),
                merge_cache_path,
            )

    approved = _drop_number_mismatches(approved, mentions, groups)
    merged_group_map = _centre_clusters(groups, approved)

    # Build initial registry (do not finalize alias -> canonical mapping yet)
    registry: dict[str, CanonicalEntityRecord] = {}
    # How often each label was given to the entity's mentions. The node takes
    # the most frequent one: the union made one noisy mention enough to type
    # "food waste" as a Product as well, and those labels reach the answer.
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for group_idxs in merged_group_map.values():
        mention_indices: list[int] = []
        for gidx in group_idxs:
            mention_indices.extend(groups[gidx])

        aliases = sorted(
            {mentions[midx]["name"] for midx in mention_indices},
            key=lambda x: (x.lower(), len(x)),
        )
        alias_documents: dict[str, set[str]] = defaultdict(set)
        for midx in mention_indices:
            doc = mentions[midx]["doc"]
            if doc:
                alias_documents[mentions[midx]["name"]].add(doc)
        canonical_name = _pick_canonical_name(aliases, alias_documents)
        labels = sorted({mentions[midx]["label"] for midx in mention_indices})
        label_counts[canonical_name].update(
            mentions[midx]["label"] for midx in mention_indices
        )
        merged_props: dict[str, Any] = {"name": canonical_name}
        alias_sources: dict[str, list[str]] = defaultdict(list)

        for midx in mention_indices:
            alias = mentions[midx]["name"]
            source_doc = mentions[midx]["doc"]
            if source_doc and source_doc not in alias_sources[alias]:
                alias_sources[alias].append(source_doc)
            for key, value in mentions[midx]["properties"].items():
                merged_props.setdefault(key, value)

        # Accumulate, never overwrite. Two merged groups can reach the same
        # canonical name: `_initial_groups` splits one (label, name) key by
        # predicate overlap, so the same surface name can yield several
        # groups. Overwriting would drop the earlier group's aliases, and their
        # triples would keep unresolved names.
        existing = registry.get(canonical_name)
        if existing is None:
            registry[canonical_name] = CanonicalEntityRecord(
                canonical_name=canonical_name,
                aliases=aliases,
                labels=labels,
                merged_properties=merged_props,
                alias_sources=dict(alias_sources),
            )
            continue

        for alias in aliases:
            if alias not in existing.aliases:
                existing.aliases.append(alias)
        existing.aliases.sort(key=lambda x: (x.lower(), len(x)))
        for label in labels:
            if label not in existing.labels:
                existing.labels.append(label)
        existing.labels.sort()
        for alias, docs in alias_sources.items():
            target = existing.alias_sources.setdefault(alias, [])
            for doc in docs:
                if doc not in target:
                    target.append(doc)
        for key, value in merged_props.items():
            existing.merged_properties.setdefault(key, value)

    def _cross_label_merge_registry(
        registry: dict[str, CanonicalEntityRecord],
        log_path: Path | None = None,
    ) -> tuple[dict[str, CanonicalEntityRecord], dict[str, str]]:
        """Merge registry entries whose canonical names differ only in case.

        The merged entry takes the label most of the entries' mentions carry
        (:func:`_majority_label`). The keeper is the longest name among the
        entries that carry that label, or among all entries when none does; it
        receives the others' aliases, alias sources and missing properties.

        Args:
            registry: Registry to merge; modified in place.
            log_path: JSONL file to append one record per merge to, if any.

        Returns:
            ``(registry, alias_to_canonical)``, the second mapping every alias
            to the canonical name of its entry.
        """
        norm_map: dict[str, list[str]] = defaultdict(list)
        for cname in list(registry.keys()):
            norm = cname.strip().lower()
            if not norm:
                LOGGER.warning(
                    "Registry entry with empty normalized name skipped: %r", cname
                )
                continue
            norm_map[norm].append(cname)

        log_lines: list[str] = []
        for norm, cnames in norm_map.items():
            if len(cnames) < 2:
                continue
            # The merged entry takes the label most of its mentions carry;
            # case-variant duplicates with a single shared label are merged as
            # well (same normalized name must map to one canonical entry).
            merged_counts: Counter[str] = Counter()
            for cname in cnames:
                merged_counts.update(label_counts.get(cname, Counter()))
            chosen_label = _majority_label(merged_counts)

            # choose keeper record (prefer record that already contains chosen_label)
            keeper: str | None = None
            candidates = [c for c in cnames if chosen_label in registry[c].labels]
            if candidates:
                keeper = sorted(candidates, key=lambda x: (-len(x), x.lower()))[0]
            else:
                keeper = sorted(cnames, key=lambda x: (-len(x), x.lower()))[0]

            removed = []
            for other in cnames:
                if other == keeper:
                    continue
                for a in registry[other].aliases:
                    if a not in registry[keeper].aliases:
                        registry[keeper].aliases.append(a)
                for a, srcs in registry[other].alias_sources.items():
                    registry[keeper].alias_sources.setdefault(a, [])
                    for s in srcs:
                        if s not in registry[keeper].alias_sources[a]:
                            registry[keeper].alias_sources[a].append(s)
                # merge properties (keep existing keys on keeper)
                for k, v in registry[other].merged_properties.items():
                    registry[keeper].merged_properties.setdefault(k, v)
                removed.append({"canonical": other, "labels": registry[other].labels})
                try:
                    del registry[other]
                except KeyError:
                    pass

            registry[keeper].labels = [chosen_label]
            for cname in cnames:
                label_counts.pop(cname, None)
            label_counts[keeper] = merged_counts

            ts = datetime.utcnow().isoformat()
            entry = {
                "timestamp": ts,
                "normalized_name": norm,
                "keeper": keeper,
                "removed": removed,
                "chosen_label": chosen_label,
            }
            log_lines.append(json.dumps(entry, ensure_ascii=False))

        if log_path:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    for line in log_lines:
                        fh.write(line + "\n")
            except OSError as exc:
                LOGGER.warning(
                    "Could not write cross-label merge log %s: %s", log_path, exc
                )

        alias_to_canonical: dict[str, str] = {}
        for cname, rec in registry.items():
            for a in rec.aliases:
                alias_to_canonical[a] = cname

        return registry, alias_to_canonical

    registry, alias_to_canonical = _cross_label_merge_registry(
        registry, log_path=Path(crosslabel_log_path) if crosslabel_log_path else None
    )
    for cname, record in registry.items():
        record.labels = [_majority_label(label_counts.get(cname, Counter(record.labels)))]

    resolved_triples: list[KGTriple] = []
    for triple in triples:
        triple.subject = alias_to_canonical.get(triple.subject, triple.subject)
        triple.object = alias_to_canonical.get(triple.object, triple.object)

        if triple.subject in registry:
            triple.subject_properties = dict(registry[triple.subject].merged_properties)
            if not triple.subject_labels:
                triple.subject_labels = list(registry[triple.subject].labels) or [
                    "Concept"
                ]
            else:
                triple.subject_labels = list(registry[triple.subject].labels)

        if triple.object in registry:
            triple.object_properties = dict(registry[triple.object].merged_properties)
            if not triple.object_labels:
                triple.object_labels = list(registry[triple.object].labels) or [
                    "Concept"
                ]
            else:
                triple.object_labels = list(registry[triple.object].labels)

        resolved_triples.append(triple)

    return resolved_triples, registry


def save_registry(path: Path, registry: dict[str, CanonicalEntityRecord]) -> None:
    """Write the registry to a JSON file, warning about case-variant names.

    Args:
        path: Output file; parent directories are created.
        registry: Canonical entities keyed by canonical name.
    """
    seen_norms: dict[str, str] = {}
    for key in registry:
        norm = key.strip().lower()
        if norm in seen_norms:
            LOGGER.warning(
                "Registry contains case-variant duplicate canonical names: %r and %r",
                seen_norms[norm],
                key,
            )
        else:
            seen_norms[norm] = key
    payload = {key: value.model_dump() for key, value in registry.items()}
    _save_json(path, payload)


def load_registry(path: Path) -> dict[str, CanonicalEntityRecord]:
    """Read a registry written by :func:`save_registry`.

    Args:
        path: JSON file to read.

    Returns:
        Mapping from canonical name to validated record.
    """
    payload = _load_json(path)
    return {
        key: CanonicalEntityRecord.model_validate(value)
        for key, value in payload.items()
    }


def save_triples(path: Path, triples: list[KGTriple]) -> None:
    """Write triples to a JSON file.

    Args:
        path: Output file; parent directories are created.
        triples: Triples to write.
    """
    payload = [triple.as_dict() for triple in triples]
    _save_json(path, payload)


def load_triples(path: Path) -> list[KGTriple]:
    """Read triples written by :func:`save_triples`.

    Args:
        path: JSON file to read.

    Returns:
        The validated triples.
    """
    payload = _load_json(path)
    return [KGTriple.model_validate(item) for item in payload]


def _cli() -> None:
    """Run stage 4 standalone from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples-json", required=True)
    parser.add_argument("--acronyms-json", required=True)
    parser.add_argument("--output-triples-json", required=True)
    parser.add_argument("--output-registry-json", required=True)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--context-jaccard-floor", type=float, default=0.15)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--crosslabel-log", default="")
    args = parser.parse_args()

    triples = load_triples(Path(args.triples_json))
    acronym_map = _load_json(Path(args.acronyms_json))

    resolved, registry = resolve_entities(
        triples=triples,
        acronym_map=acronym_map,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        context_jaccard_floor=args.context_jaccard_floor,
        base_url=args.base_url or None,
        api_key=args.api_key or None,
        model_name=args.model_name or None,
        crosslabel_log_path=Path(args.crosslabel_log)
        if args.crosslabel_log
        else Path(args.output_registry_json).resolve().parent
        / "resolution_crosslabel.log",
    )

    save_triples(Path(args.output_triples_json), resolved)
    save_registry(Path(args.output_registry_json), registry)


if __name__ == "__main__":
    _cli()
