"""Gold triples: candidates from Neo4j for human review, then the reviewed set."""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger("graphrag")

# Candidate review CSV columns. `keep` is filled by the human reviewer:
# any of TRUTHY_KEEP marks the triple as gold, anything else drops it.
CANDIDATE_FIELDS = [
    "question_id",
    "question",
    "rank",
    "subject",
    "predicate",
    "object",
    "matched_entities",
    "source",
    "score",
    "keep",
]

TRUTHY_KEEP = {"1", "x", "y", "yes", "si", "sì", "true", "keep", "ok"}

# Minimal IT+EN stopword list for lexical scoring — enough to stop function
# words from dominating overlap on short triple surfaces.
_STOPWORDS = {
    "a", "ai", "al", "alla", "alle", "and", "are", "che", "chi", "come", "con",
    "cosa", "da", "dal", "dei", "del", "della", "delle", "di", "do", "does",
    "e", "ed", "for", "from", "gli", "how", "i", "il", "in", "is", "it", "la",
    "le", "lo", "nel", "nella", "non", "of", "on", "per", "quali", "quale",
    "sono", "su", "sul", "sulla", "the", "to", "un", "una", "uno", "what",
    "which", "who", "why", "with",
}


# ─── Normalisation and matching ──────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", normalized)


def _tokenize(text: str) -> set[str]:
    """Normalized content-word tokens (stopwords removed)."""
    words = re.findall(r"\w+", _normalize(text))
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _humanize_predicate(predicate: str) -> str:
    """SCREAMING_SNAKE_CASE → lowercase words for lexical scoring."""
    return predicate.replace("_", " ").lower()


def match_entities_to_nodes(
    entities: list[str],
    node_names: list[str],
) -> dict[str, list[str]]:
    """Map gold expected_entities to canonical KG node names.

    Match order per entity: exact normalized equality, then substring
    containment in either direction (guarded to >= 4 chars to avoid
    matches on trivially short strings).

    Args:
        entities: Surface forms from the gold ``expected_entities`` field.
        node_names: All canonical node names in the KG.

    Returns:
        Mapping of entity surface form → matched canonical names (may be empty).
    """
    normalized_nodes = [(name, _normalize(name)) for name in node_names]
    matches: dict[str, list[str]] = {}

    for entity in entities:
        entity_norm = _normalize(str(entity))
        if not entity_norm:
            matches[str(entity)] = []
            continue

        exact = [name for name, norm in normalized_nodes if norm == entity_norm]
        if exact:
            matches[str(entity)] = exact
            continue

        if len(entity_norm) >= 4:
            partial = [
                name
                for name, norm in normalized_nodes
                if (entity_norm in norm) or (len(norm) >= 4 and norm in entity_norm)
            ]
        else:
            partial = []
        matches[str(entity)] = partial

    return matches


def score_triple(
    triple: dict[str, str],
    reference_tokens: set[str],
    matched_names_norm: set[str],
) -> float:
    """Score a candidate triple against question+answer text and matched entities.

    Combines lexical overlap of the triple surface with the reference tokens
    (60%) and anchoring of subject/object on matched gold entities (20% each).

    Args:
        triple: Dict with subject/predicate/object.
        reference_tokens: Content tokens of question + canonical answer.
        matched_names_norm: Normalized canonical names matched from
            expected_entities.

    Returns:
        Score in [0, 1].
    """
    surface = " ".join(
        [
            str(triple.get("subject", "")),
            _humanize_predicate(str(triple.get("predicate", ""))),
            str(triple.get("object", "")),
        ]
    )
    tokens = _tokenize(surface)
    lexical = len(tokens & reference_tokens) / len(tokens) if tokens else 0.0

    subject_hit = 1.0 if _normalize(str(triple.get("subject", ""))) in matched_names_norm else 0.0
    object_hit = 1.0 if _normalize(str(triple.get("object", ""))) in matched_names_norm else 0.0

    return 0.6 * lexical + 0.2 * subject_hit + 0.2 * object_hit


# ─── Neo4j fetch helpers ─────────────────────────────────────────────────────

def _fetch_node_names(session: Any) -> list[str]:
    """Name of every named node in the graph."""
    result = session.run("MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name")
    return [record["name"] for record in result]


def _fetch_one_hop_triples(session: Any, names: list[str]) -> list[dict[str, str]]:
    """All triples where subject or object is one of *names* (single query)."""
    if not names:
        return []
    result = session.run(
        """
        MATCH (s)-[r]->(o)
        WHERE s.name IN $names OR o.name IN $names
        RETURN DISTINCT s.name AS subject, type(r) AS predicate, o.name AS object
        """,
        names=names,
    )
    return [dict(record) for record in result]


def _fetch_bridge_triples(
    session: Any,
    pairs: list[tuple[str, str]],
    limit: int = 500,
) -> list[dict[str, str]]:
    """Triples on undirected paths (length <= 2) between matched entity pairs."""
    if not pairs:
        return []
    result = session.run(
        """
        UNWIND $pairs AS pair
        MATCH (a {name: pair[0]}), (b {name: pair[1]})
        MATCH p = (a)-[*..2]-(b)
        UNWIND relationships(p) AS r
        RETURN DISTINCT startNode(r).name AS subject,
               type(r) AS predicate,
               endNode(r).name AS object
        LIMIT $limit
        """,
        pairs=[list(pair) for pair in pairs],
        limit=limit,
    )
    return [dict(record) for record in result]


# ─── Extract mode ────────────────────────────────────────────────────────────

def extract_candidates(
    gold_path: Path,
    out_path: Path,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "neo4j",
    max_per_question: int = 30,
    bridge: bool = True,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Extract candidate gold triples from the KG for each gold question.

    For every gold row: match ``expected_entities`` to canonical node names,
    pull 1-hop triples around the matches (plus, optionally, triples bridging
    matched entity pairs within 2 hops), rank by lexical+anchoring score, and
    write a review CSV with an empty ``keep`` column for human validation.

    Args:
        gold_path: Gold CSV (needs question, expected_entities; question_id
            and canonical_answer used when present).
        out_path: Candidate review CSV to write.
        neo4j_url: Bolt URL.
        neo4j_user: Username.
        neo4j_password: Password.
        database: Database name.
        max_per_question: Rank cutoff per question.
        bridge: Also fetch bridging triples between matched entity pairs.
        min_score: Drop candidates scoring below this threshold.

    Returns:
        Summary dict (questions, matched entities, candidates written).
    """
    try:
        from kg_pipeline.utils import neo4j_env  # type: ignore
    except ImportError as exc:
        raise ImportError("Install neo4j: pip install neo4j") from exc

    from evalkit.io.gold_loader import extract_gold_annotations, normalize_question, pick_ground_truth

    with gold_path.open("r", encoding="utf-8", newline="") as fh:
        gold_rows = list(csv.DictReader(fh))
    if not gold_rows:
        raise ValueError(f"No rows in gold file: {gold_path}")

    driver = neo4j_env.connect(
        neo4j_env.resolve_target(
            uri=neo4j_url, user=neo4j_user, password=neo4j_password
        )
    )
    candidate_rows: list[dict[str, Any]] = []
    n_matched_entities = 0
    n_total_entities = 0

    try:
        with driver.session(database=database) as session:
            node_names = _fetch_node_names(session)
            logger.info("Fetched %d node names from KG", len(node_names))

            for row in gold_rows:
                question = (row.get("question") or "").strip()
                if not question:
                    continue
                question_id = (row.get("question_id") or "").strip() or normalize_question(question)
                expected_entities, _, _ = extract_gold_annotations(row)
                entities = [str(e) for e in expected_entities if str(e).strip()]
                n_total_entities += len(entities)

                entity_matches = match_entities_to_nodes(entities, node_names)
                matched_names = sorted({name for names in entity_matches.values() for name in names})
                n_matched_entities += sum(1 for names in entity_matches.values() if names)
                unmatched = [e for e, names in entity_matches.items() if not names]
                if unmatched:
                    logger.info("%s: unmatched entities: %s", question_id, ", ".join(unmatched))
                if not matched_names:
                    logger.warning("%s: no entities matched in KG — no candidates", question_id)
                    continue

                triples = _fetch_one_hop_triples(session, matched_names)
                sources = {_triple_key(t): "one_hop" for t in triples}

                if bridge and len(matched_names) > 1:
                    pairs = [
                        (a, b)
                        for i, a in enumerate(matched_names)
                        for b in matched_names[i + 1:]
                    ]
                    for t in _fetch_bridge_triples(session, pairs):
                        key = _triple_key(t)
                        if key in sources:
                            sources[key] = "one_hop+bridge"
                        else:
                            sources[key] = "bridge"
                            triples.append(t)

                reference_tokens = _tokenize(f"{question} {pick_ground_truth(row)}")
                matched_norm = {_normalize(name) for name in matched_names}

                scored = sorted(
                    (
                        (score_triple(t, reference_tokens, matched_norm), t)
                        for t in triples
                    ),
                    key=lambda pair: pair[0],
                    reverse=True,
                )

                rank = 0
                for score, triple in scored:
                    if score < min_score:
                        continue
                    rank += 1
                    if rank > max_per_question:
                        break
                    candidate_rows.append(
                        {
                            "question_id": question_id,
                            "question": question,
                            "rank": rank,
                            "subject": triple["subject"],
                            "predicate": triple["predicate"],
                            "object": triple["object"],
                            "matched_entities": "|".join(matched_names),
                            "source": sources[_triple_key(triple)],
                            "score": round(score, 4),
                            "keep": "",
                        }
                    )
    finally:
        driver.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(candidate_rows)

    summary = {
        "questions": len(gold_rows),
        "entities_total": n_total_entities,
        "entities_matched": n_matched_entities,
        "candidates_written": len(candidate_rows),
        "out": str(out_path),
    }
    logger.info(
        "gold-triples extract: %d candidates for %d questions (%d/%d entities matched) → %s",
        len(candidate_rows), len(gold_rows), n_matched_entities, n_total_entities, out_path,
    )
    return summary


def _triple_key(triple: dict[str, str]) -> str:
    """A triple as ``subject|predicate|object``."""
    return "|".join(
        [
            str(triple.get("subject", "")),
            str(triple.get("predicate", "")),
            str(triple.get("object", "")),
        ]
    )


# ─── Apply mode ──────────────────────────────────────────────────────────────

def apply_review(
    gold_path: Path,
    candidates_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Write reviewed candidate triples into the gold CSV's gold_triples column.

    Reads the reviewed candidates CSV, keeps rows whose ``keep`` value is in
    TRUTHY_KEEP, groups them by question, and writes a copy of the gold CSV
    with ``gold_triples`` populated as a JSON array of
    ``{"subject", "predicate", "object"}`` objects.

    Args:
        gold_path: Original gold CSV.
        candidates_path: Reviewed candidates CSV (from extract mode).
        out_path: New gold CSV to write (must differ from gold_path).

    Returns:
        Summary dict (questions updated, triples kept).
    """
    from evalkit.io.gold_loader import normalize_question

    if out_path.resolve() == gold_path.resolve():
        raise ValueError("--out must differ from --gold (refusing to overwrite the source gold file)")

    with candidates_path.open("r", encoding="utf-8", newline="") as fh:
        candidates = list(csv.DictReader(fh))

    kept_by_qid: dict[str, list[dict[str, str]]] = {}
    for cand in candidates:
        keep = (cand.get("keep") or "").strip().lower()
        if keep not in TRUTHY_KEEP:
            continue
        qid = (cand.get("question_id") or "").strip()
        if not qid:
            qid = normalize_question(cand.get("question") or "")
        kept_by_qid.setdefault(qid, []).append(
            {
                "subject": (cand.get("subject") or "").strip(),
                "predicate": (cand.get("predicate") or "").strip(),
                "object": (cand.get("object") or "").strip(),
            }
        )

    with gold_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        gold_rows = list(reader)
    if "gold_triples" not in fieldnames:
        fieldnames.append("gold_triples")

    n_updated = 0
    n_triples = 0
    for row in gold_rows:
        qid = (row.get("question_id") or "").strip() or normalize_question(row.get("question") or "")
        kept = kept_by_qid.get(qid)
        if kept is None:
            continue
        # Dedupe while preserving reviewer order.
        seen: set[str] = set()
        unique = []
        for triple in kept:
            key = _triple_key(triple)
            if key in seen:
                continue
            seen.add(key)
            unique.append(triple)
        row["gold_triples"] = json.dumps(unique, ensure_ascii=False)
        n_updated += 1
        n_triples += len(unique)

    orphan_qids = set(kept_by_qid) - {
        (r.get("question_id") or "").strip() or normalize_question(r.get("question") or "")
        for r in gold_rows
    }
    if orphan_qids:
        logger.warning(
            "Kept candidates for %d question_ids not present in gold: %s",
            len(orphan_qids), ", ".join(sorted(orphan_qids)),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gold_rows)

    summary = {
        "questions_updated": n_updated,
        "triples_kept": n_triples,
        "out": str(out_path),
    }
    logger.info(
        "gold-triples apply: %d triples over %d questions → %s",
        n_triples, n_updated, out_path,
    )
    return summary
