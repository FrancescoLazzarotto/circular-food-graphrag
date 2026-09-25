"""Embed every KG node name and build the Neo4j vector index.

Why this exists: the graph was extracted from a bilingual corpus and most
concepts ended up under their Italian surface form, while the gold questions are
English. Lexical retrieval (one full-text index over ``name`` and
``search_text``) never takes an English query to ``polifenoli``, ``Ciclicità``
or ``letame``, and many of the expected entities exist in the graph *only*
under an Italian form.

A multilingual sentence encoder puts the Italian node name and the English
question in the same space, so the bridge costs one embedding per node and one
per query. Embeddings are written to separate ``:NodeVec`` carrier nodes and
served by a native Neo4j vector index (available on this instance: 5.27-aura
enterprise).

The encoder is loaded through ``transformers`` directly (mean pooling +
L2 normalisation, the reference recipe for the e5 family) rather than
``sentence-transformers``, which is not installed in the ``graphllm``
environment and whose install would pull a different torch.

Usage::

    python scripts/kg/kg_vector_index.py                # embed + index
    python scripts/kg/kg_vector_index.py --probe        # check an existing index
    python scripts/kg/kg_vector_index.py --drop         # remove index and vectors
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

from graphrag.config import build_kg_config_from_env  # noqa: E402
from graphrag.embeddings import (  # noqa: E402
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    available,
    encode,
    model_id,
)
from graphrag.kg.manager import KnowledgeGraphManager  # noqa: E402

DEFAULT_INDEX = "node_embedding"
DEFAULT_PROPERTY = "embedding"
DEFAULT_LABEL = "NodeVec"

logger = logging.getLogger("kg_vector_index")


def fetch_nodes(store: KnowledgeGraphManager, context_chars: int = 0) -> list[dict]:
    """Node ids and the text to embed.

    ``context_chars > 0`` appends the node's own ``search_text`` (and aliases)
    to the name. Names alone are two or three words, and the encoder scores
    every short name in a narrow band, so the right node is retrieved but not
    ranked first. Extra text spreads the scores and is what makes the ranking
    usable.

    Args:
        store: The graph.
        context_chars: Characters of aliases and search text appended.

    Returns:
        One row per named node, with ``embed_text`` added.
    """
    rows = store.run_query(
        "MATCH (n) WHERE n.name IS NOT NULL RETURN elementId(n) AS node_id, "
        "toString(n.name) AS name, coalesce(n.search_text, '') AS search_text, "
        "coalesce(n.aliases, []) AS aliases, labels(n) AS labels"
    )
    for row in rows:
        text = row["name"]
        if context_chars:
            extras = [a for a in (row.get("aliases") or []) if a and a != row["name"]]
            extra = " ".join([*extras, str(row.get("search_text") or "")]).strip()
            if extra:
                text = f"{text}. {extra[:context_chars]}"
        row["embed_text"] = text
    return rows


def write_embeddings(
    store: KnowledgeGraphManager,
    rows: Sequence[dict],
    vectors: Sequence[Sequence[float]],
    prop: str,
    label: str,
    batch_size: int = 500,
) -> None:
    """Store each vector on its own node, keyed by the entity's elementId.

    Not on the entity itself: a 768-float array in the entity's property chain
    would make every ``properties(n)`` ship 10 KB of JSON and every name-based
    scan walk past the vectors, and a label shared by every node would flatten
    the schema view into ``(:Embeddable)-[...]->(:Embeddable)``. On separate
    nodes the entities stay untouched. The link is the entity's elementId,
    which is stable for a static graph; rebuild the index after any reload.

    Args:
        store: The graph.
        rows: Rows from `fetch_nodes`.
        vectors: One vector per row.
        prop: Vector property on the carrier.
        label: Carrier label.
        batch_size: Carriers written per query.
    """
    payload = [
        {"node_id": row["node_id"], "vec": list(vec)}
        for row, vec in zip(rows, vectors, strict=True)
    ]
    for start in range(0, len(payload), batch_size):
        chunk = payload[start : start + batch_size]
        store.run_query(
            f"UNWIND $rows AS row MERGE (v:{label} {{of: row.node_id}}) "
            f"WITH v, row CALL db.create.setNodeVectorProperty(v, '{prop}', row.vec)",
            {"rows": chunk},
        )
        logger.info("scritti %d/%d", min(start + batch_size, len(payload)), len(payload))


def create_index(
    store: KnowledgeGraphManager, index: str, label: str, prop: str, dimensions: int
) -> None:
    """Create the cosine vector index over the carriers, if it is missing."""
    store.run_query(
        f"CREATE VECTOR INDEX {index} IF NOT EXISTS FOR (n:{label}) ON (n.{prop}) "
        "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
        "`vector.similarity_function`: 'cosine'}}",
        {"dim": dimensions},
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Embed and index the nodes, or probe or drop the index.
    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="default: $GRAPHRAG_EMBED_MODEL")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--property", default=DEFAULT_PROPERTY)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--limit", type=int, default=0, help="0 = all nodes")
    parser.add_argument(
        "--context-chars",
        type=int,
        default=300,
        help="chars of search_text/aliases appended to the name (0 = name only)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--write-batch", type=int, default=500)
    parser.add_argument("--labels", default="", help="comma-separated label whitelist")
    parser.add_argument("--probe", action="store_true", help="query an existing index")
    parser.add_argument("--drop", action="store_true", help="drop index and vectors")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(REPO / ".env", override=False)
    store = KnowledgeGraphManager(build_kg_config_from_env())

    if not args.drop and not available():
        raise SystemExit(
            "embedding endpoint non raggiungibile. Avvia:\n"
            "  CUDA_VISIBLE_DEVICES=1 vllm serve intfloat/multilingual-e5-base "
            "--runner pooling --port 8002 --gpu-memory-utilization 0.12 "
            "--max-model-len 512"
        )

    if args.drop:
        store.run_query(f"DROP INDEX {args.index} IF EXISTS")
        store.run_query(f"MATCH (v:{args.label}) DETACH DELETE v")
        # Also clear vectors stored on the entities themselves, the older layout,
        # so a rebuild cannot leave stale vectors behind.
        store.run_query(
            f"MATCH (n) WHERE n.{args.property} IS NOT NULL REMOVE n.{args.property}"
        )
        store.run_query("MATCH (n:Embeddable) REMOVE n:Embeddable")
        print("index, nodi vettore e residui rimossi")
        return 0

    if args.probe:
        for probe in (
            "polyphenol",
            "rice straw",
            "whey and how it is produced",
            "the three C's of the Circular Economy for Food",
        ):
            vec = encode([probe], QUERY_PREFIX, model=args.model)[0]
            rows = store.run_query(
                f"CALL db.index.vector.queryNodes('{args.index}', 5, $vec) "
                "YIELD node AS v, score "
                "MATCH (n) WHERE elementId(n) = v.of "
                "RETURN n.name AS name, score",
                {"vec": vec},
            )
            hits = ", ".join(f"{r['name']}({r['score']:.3f})" for r in rows)
            print(f"  {probe:48} -> {hits}")
        return 0

    rows = fetch_nodes(store, context_chars=args.context_chars)
    if args.labels:
        wanted = {label.strip() for label in args.labels.split(",") if label.strip()}
        allowed = {
            r["node_id"]
            for r in store.run_query(
                "MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) "
                "RETURN elementId(n) AS node_id",
                {"labels": sorted(wanted)},
            )
        }
        rows = [r for r in rows if r["node_id"] in allowed]
    if args.limit:
        rows = rows[: args.limit]
    print(f"nodi da embeddare: {len(rows)}")

    print(f"encoder {args.model or model_id()}")

    started = time.perf_counter()
    vectors = encode(
        [r["embed_text"] for r in rows],
        PASSAGE_PREFIX,
        model=args.model,
        batch_size=args.batch_size,
    )
    print(f"embedding calcolati in {time.perf_counter() - started:.1f}s, dim={len(vectors[0])}")

    write_embeddings(
        store, rows, vectors, args.property, args.label, args.write_batch
    )
    create_index(store, args.index, args.label, args.property, len(vectors[0]))
    print(f"indice {args.index} creato ({len(vectors[0])} dim, cosine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
