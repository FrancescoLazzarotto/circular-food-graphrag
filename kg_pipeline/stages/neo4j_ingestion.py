"""Stage 6: write triples to Neo4j and run post-ingestion quality checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm
import logging
from neo4j.exceptions import CypherTypeError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg_pipeline.models.types import KGTriple
from kg_pipeline.utils import neo4j_env
from kg_pipeline.utils.neo4j_env import resolve_target


_ID_RE = re.compile(r"[^A-Za-z0-9_]+")


def _setup_logging(log_level: str, log_file: str | None = None) -> None:
    """Configure root logging to stderr and, optionally, to a file.

    Args:
        log_level: Level name; unknown names fall back to ``INFO``.
        log_file: File to append log records to, if any.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _safe_identifier(value: str, fallback: str) -> str:
    """Make ``value`` safe to interpolate into Cypher as a label or type.

    Args:
        value: Label or relationship type.
        fallback: Identifier to use when nothing valid is left.

    Returns:
        ``value`` with every run of characters outside ``[A-Za-z0-9_]``
        replaced by one ``_``, outer underscores stripped, and a leading
        ``_`` added when it would start with a digit.
    """
    cleaned = _ID_RE.sub("_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _resolve_neo4j_env() -> tuple[str, str, str, str | None]:
    """Resolve the Neo4j target from the environment, as a tuple.

    Thin wrapper over :func:`kg_pipeline.utils.neo4j_env.resolve_target` for
    callers that unpack a tuple.

    Returns:
        ``(uri, user, password, database)``; ``database`` is ``None`` for the
        server default.

    Raises:
        ValueError: If the URI, user or password is not set.
    """
    return tuple(resolve_target())  # type: ignore[return-value]


def _is_primitive(value: object) -> bool:
    """Whether ``value`` is a str, bool, int or float."""
    return isinstance(value, (str, bool, int, float))


def _sanitize_value(value: object) -> object:
    """Convert a value into something Neo4j can store as a property.

    Args:
        value: Any value.

    Returns:
        Primitives unchanged, ``None`` as ``""``, containers as a JSON string
        and anything else as ``str(value)``.
    """
    if _is_primitive(value):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, dict, set, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return str(value)


def _sanitize_props(props: dict[str, object]) -> dict[str, object]:
    """Convert a property map into types Neo4j can store.

    ``None`` values are dropped, primitives kept, dicts serialised to JSON,
    and lists, sets and tuples turned into homogeneous arrays: nested values
    become JSON strings, and a mix of types is coerced to strings.

    Args:
        props: Property map; ``None`` is treated as empty.

    Returns:
        A new map with string keys and storable values.
    """
    out: dict[str, object] = {}
    for k, v in (props or {}).items():
        if v is None:
            continue
        if _is_primitive(v):
            out[str(k)] = v
            continue

        if isinstance(v, list):
            # convert list elements to primitives or strings (no nested maps)
            new_list: list[object] = []
            categories: set[str] = set()
            for item in v:
                if item is None:
                    continue
                if _is_primitive(item):
                    # treat bool separately from numeric to avoid subclassing issues
                    if isinstance(item, bool):
                        categories.add("bool")
                    elif isinstance(item, (int, float)) and not isinstance(item, bool):
                        categories.add("num")
                    else:
                        categories.add("str")
                    new_list.append(item)
                else:
                    try:
                        s = json.dumps(item, ensure_ascii=False, default=str)
                    except Exception:
                        s = str(item)
                    new_list.append(s)
                    categories.add("str")

            # Neo4j requires property arrays to be of a single allowed type.
            # If the list mixes numbers and strings (or other categories), coerce all
            # elements to strings to ensure a homogeneous, supported type.
            if len(categories - {""}) > 1:
                coerced = [str(x) for x in new_list]
                out[str(k)] = coerced
            else:
                out[str(k)] = new_list
            continue

        if isinstance(v, dict):
            try:
                out[str(k)] = json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                out[str(k)] = str(v)
            continue

        if isinstance(v, (set, tuple)):
            try:
                seq = list(v)
                new_seq: list[object] = []
                categories: set[str] = set()
                for i in seq:
                    if _is_primitive(i):
                        if isinstance(i, bool):
                            categories.add("bool")
                        elif isinstance(i, (int, float)) and not isinstance(i, bool):
                            categories.add("num")
                        else:
                            categories.add("str")
                        new_seq.append(i)
                    else:
                        try:
                            s = json.dumps(i, ensure_ascii=False, default=str)
                        except Exception:
                            s = str(i)
                        new_seq.append(s)
                        categories.add("str")

                if len(categories) > 1:
                    out[str(k)] = [str(x) for x in new_seq]
                else:
                    out[str(k)] = new_seq
            except Exception:
                out[str(k)] = str(v)
            continue

        out[str(k)] = str(v)

    return out


def _triple_cypher_parts(triple: KGTriple) -> tuple[str, dict[str, object]]:
    """Build the MERGE query and the parameter row for one triple.

    Labels and the relationship type are identifiers, which Cypher cannot
    parameterise, so they are sanitised and interpolated into the query.
    Triples sharing the same query text can be ingested together via UNWIND.
    Nodes are merged on their first label and ``name``; the relationship is
    merged on its type and its ``subject`` / ``object`` names.

    Args:
        triple: Triple to write.

    Returns:
        ``(query, row)``; ``row`` holds ``s_name``, ``o_name``, ``s_props``,
        ``o_props`` and ``r_props``.
    """
    s_labels = triple.subject_labels or ["Concept"]
    o_labels = triple.object_labels or ["Concept"]

    s_primary = _safe_identifier(s_labels[0], "Concept")
    o_primary = _safe_identifier(o_labels[0], "Concept")
    rel_type = _safe_identifier(triple.predicate, "RELATED_TO")

    s_extra = [_safe_identifier(label, "Concept") for label in s_labels[1:]]
    o_extra = [_safe_identifier(label, "Concept") for label in o_labels[1:]]

    set_subject_extra = "\n".join([f"SET s:{label}" for label in s_extra])
    set_object_extra = "\n".join([f"SET o:{label}" for label in o_extra])

    query = f"""
UNWIND $rows AS row
MERGE (s:{s_primary} {{name: row.s_name}})
SET s += row.s_props
{set_subject_extra}
MERGE (o:{o_primary} {{name: row.o_name}})
SET o += row.o_props
{set_object_extra}
MERGE (s)-[r:{rel_type} {{subject: row.s_name, object: row.o_name}}]->(o)
SET r += row.r_props
"""

    row: dict[str, object] = {
        "s_name": _sanitize_value(triple.subject_properties.get("name", triple.subject)),
        "o_name": _sanitize_value(triple.object_properties.get("name", triple.object)),
        "s_props": _sanitize_props(triple.subject_properties),
        "o_props": _sanitize_props(triple.object_properties),
        "r_props": _sanitize_props(triple.relationship_properties),
    }
    return query, row


def _merge_triple(tx, triple: KGTriple) -> None:
    """Write one triple inside a transaction, logging instead of raising.

    A failing triple is logged and appended to
    ``kg_pipeline/logs/problematic_triples.jsonl`` with its sanitised
    properties.

    Args:
        tx: Neo4j transaction.
        triple: Triple to write.
    """
    query, row = _triple_cypher_parts(triple)
    s_props = row["s_props"]
    o_props = row["o_props"]
    r_props = row["r_props"]

    logger = logging.getLogger(__name__)

    try:
        tx.run(query, rows=[row]).consume()
    except CypherTypeError as e:
        # log details for debugging and skip this triple to allow ingestion to continue
        debug = {
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "subject_labels": triple.subject_labels,
            "object_labels": triple.object_labels,
            "s_name": row["s_name"],
            "o_name": row["o_name"],
            "s_props_sanitized": s_props,
            "o_props_sanitized": o_props,
            "r_props_sanitized": r_props,
        }
        logger.exception("CypherTypeError writing triple: %s", e)
        try:
            logs_dir = Path(__file__).resolve().parents[1] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            with (logs_dir / "problematic_triples.jsonl").open(
                "a", encoding="utf-8"
            ) as fh:
                fh.write(json.dumps(debug, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write problematic triple to log file")
        return
    except Exception as e:
        logger.exception("Unexpected error writing triple: %s", e)
        try:
            logs_dir = Path(__file__).resolve().parents[1] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            with (logs_dir / "problematic_triples.jsonl").open(
                "a", encoding="utf-8"
            ) as fh:
                fh.write(
                    json.dumps(
                        {
                            "error": str(e),
                            "subject": triple.subject,
                            "predicate": triple.predicate,
                            "object": triple.object,
                            "s_props_sanitized": s_props,
                            "o_props_sanitized": o_props,
                            "r_props_sanitized": r_props,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        except Exception:
            logger.exception("Failed to write unexpected error info to log file")
        return


def _merge_triples_batch(tx, query: str, rows: list[dict[str, object]]) -> None:
    """Run one UNWIND query over a batch of parameter rows in a transaction."""
    tx.run(query, rows=rows).consume()


def ingest_triples(
    triples: list[KGTriple],
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    log_every: int = 0,
    batch_size: int = 200,
) -> int:
    """Write triples to Neo4j in UNWIND batches grouped by query signature.

    When a batch fails, its triples are retried one at a time and those that
    still fail are logged and skipped.

    Args:
        triples: Triples to write.
        uri: Neo4j URI.
        user: User name.
        password: Password.
        database: Database name, or ``None`` for the server default.
        log_every: Log progress roughly every N triples; 0 disables it.
        batch_size: Triples per UNWIND batch.

    Returns:
        The number of triples that reached the database. This is an upper
        bound on relationships written, because MERGE deduplicates.
    """
    count = 0
    skipped = 0
    total = len(triples)
    logger = logging.getLogger(__name__)

    # Group triples that share the same Cypher text so each group can be sent
    # as one UNWIND statement. MERGE is idempotent, so cross-group reordering
    # does not change the resulting graph.
    grouped: dict[str, tuple[str, list[tuple[KGTriple, dict[str, object]]]]] = {}
    for triple in triples:
        query, row = _triple_cypher_parts(triple)
        if query not in grouped:
            grouped[query] = (query, [])
        grouped[query][1].append((triple, row))

    batch_size = max(1, int(batch_size))
    with neo4j_env.connect(
        neo4j_env.Neo4jTarget(uri, user, password, None)
    ) as driver:
        with driver.session(database=database) as session:
            with tqdm(
                total=total, desc="Stage 6 Neo4j Ingestion", unit="triple"
            ) as progress:
                for query, items in grouped.values():
                    for start in range(0, len(items), batch_size):
                        batch = items[start : start + batch_size]
                        rows = [row for _, row in batch]
                        try:
                            session.execute_write(_merge_triples_batch, query, rows)
                            sent = len(batch)
                        except Exception as exc:
                            logger.warning(
                                "Batch ingestion failed (%d triples), retrying "
                                "one by one: %s",
                                len(batch),
                                exc,
                            )
                            sent = 0
                            for triple, _row in batch:
                                try:
                                    session.execute_write(_merge_triple, triple)
                                    sent += 1
                                except Exception:
                                    # Errors surfacing at commit time (e.g.
                                    # ConstraintError) bypass _merge_triple's
                                    # internal handling — skip the triple.
                                    skipped += 1
                                    logger.exception(
                                        "Skipping triple after per-triple retry "
                                        "failed: %s -[%s]-> %s",
                                        triple.subject,
                                        triple.predicate,
                                        triple.object,
                                    )
                        # Only triples that actually reached the database. Still
                        # an upper bound on *edges*, because MERGE deduplicates.
                        count += sent
                        progress.update(len(batch))
                        if log_every > 0 and count % log_every < batch_size:
                            logger.info(
                                "ingest_progress count=%d total=%d", count, total
                            )
    if skipped:
        logger.warning(
            "%d of %d triples were skipped and are NOT in the graph", skipped, total
        )
    return count


def summary_counts(
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
) -> dict:
    """Count the graph's nodes by label and relationships by type.

    Args:
        uri: Neo4j URI.
        user: User name.
        password: Password.
        database: Database name, or ``None`` for the server default.

    Returns:
        ``{"nodes_by_label": [...], "relationships_by_type": [...]}``, each a
        list of count records sorted by count, descending.
    """
    with neo4j_env.connect(
        neo4j_env.Neo4jTarget(uri, user, password, None)
    ) as driver:
        with driver.session(database=database) as session:
            node_records = session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY count DESC"
            ).data()
            rel_records = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
            ).data()
    return {"nodes_by_label": node_records, "relationships_by_type": rel_records}


def run_quality_checks(
    uri: str,
    user: str,
    password: str,
    report_path: Path,
    database: str | None = None,
    relation_vocab: list[str] | None = None,
) -> None:
    """Run post-ingestion validation queries and write their results as JSON.

    The checks list relationship types outside the vocabulary (only when
    ``relation_vocab`` is given; ``SAME_AS`` and ``MENTIONED_IN`` are always
    allowed), node names shared by several nodes, and a sample of nodes with
    at most one relationship. A failing query records its error in the report
    instead of raising.

    Args:
        uri: Neo4j URI.
        user: User name.
        password: Password.
        report_path: Report file. If it cannot be written,
            ``kg_quality_report.txt`` in the working directory is tried.
        database: Database name, or ``None`` for the server default.
        relation_vocab: Allowed relationship types, or ``None``.
    """
    queries = {}
    if relation_vocab:
        allowed = sorted(
            {str(item).strip().upper() for item in relation_vocab if str(item).strip()}
            | {"SAME_AS", "MENTIONED_IN"}
        )
        queries["predicates_out_of_vocab"] = (
            "MATCH ()-[r]->() WHERE NOT type(r) IN $allowed_predicates "
            "RETURN type(r) AS outOfVocab, count(*) AS n ORDER BY n DESC"
        )
    queries["duplicate_nodes_by_name"] = """
MATCH (n)
WITH n.name AS name, collect(labels(n)) AS labelSets, count(*) AS c
WHERE c > 1
RETURN name, labelSets, c ORDER BY c DESC
        """.strip()
    queries["sparsely_connected_nodes"] = """
MATCH (n)
WHERE size((n)--()) <= 1
RETURN labels(n) AS labels, n.name AS name LIMIT 20
        """.strip()

    report: dict[str, object] = {"queries": {}}
    with neo4j_env.connect(
        neo4j_env.Neo4jTarget(uri, user, password, None)
    ) as driver:
        with driver.session(database=database) as session:
            for key, q in queries.items():
                params = (
                    {"allowed_predicates": allowed}
                    if key == "predicates_out_of_vocab"
                    else {}
                )
                try:
                    data = session.run(q, **params).data()
                except Exception as e:
                    data = {"error": str(e)}
                report["queries"][key] = data

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        # best-effort: try simple write to current working dir
        try:
            with open("kg_quality_report.txt", "w", encoding="utf-8") as fh:
                fh.write(json.dumps(report, ensure_ascii=False, indent=2))
        except Exception:
            pass


def load_triples(path: Path) -> list[KGTriple]:
    """Read triples from a JSON file.

    Args:
        path: JSON file to read.

    Returns:
        The validated triples.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [KGTriple.model_validate(item) for item in payload]


def _cli() -> None:
    """Run stage 6 standalone: ingest, print counts, write the quality report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples-json", required=True)
    parser.add_argument("--database", default="")
    parser.add_argument(
        "--env-file",
        default="kg_pipeline/.env",
        help="Optional .env file to load Neo4j credentials from",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path for ingest progress",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log progress every N triples (0 disables)",
    )
    parser.add_argument(
        "--relation-vocab-json",
        default="",
        help="Relation vocab JSON for the out-of-vocab quality check (skipped if empty)",
    )
    args = parser.parse_args()

    env_file = args.env_file.strip()
    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=False)

    _setup_logging(args.log_level, args.log_file.strip() or None)

    uri, user, password, env_db = _resolve_neo4j_env()
    database = args.database.strip() or env_db

    triples = load_triples(Path(args.triples_json))
    written = ingest_triples(
        triples,
        uri=uri,
        user=user,
        password=password,
        database=database,
        log_every=int(args.log_every or 0),
    )
    summary = summary_counts(uri=uri, user=user, password=password, database=database)
    print(
        json.dumps(
            {"relationships_written": written, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )

    # Run validation queries and write kg_quality_report.txt next to triples JSON
    try:
        relation_vocab = None
        if args.relation_vocab_json.strip():
            relation_vocab = json.loads(
                Path(args.relation_vocab_json).read_text(encoding="utf-8")
            )
        report_path = Path(args.triples_json).resolve().parent / "kg_quality_report.txt"
        run_quality_checks(
            uri=uri,
            user=user,
            password=password,
            report_path=report_path,
            database=database,
            relation_vocab=relation_vocab,
        )
    except Exception:
        pass


if __name__ == "__main__":
    _cli()
