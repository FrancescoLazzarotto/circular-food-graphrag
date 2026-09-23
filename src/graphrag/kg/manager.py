"""Neo4j query layer: node, triple, neighbour, subgraph and path lookups.

Every query goes through :meth:`KnowledgeGraphManager.run_query`, which
retries transient failures and fails fast for a few seconds once the graph is
known to be unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain_community.graphs import Neo4jGraph

try:  # pragma: no cover - depends on runtime dependency details
    from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

    _RETRYABLE_NEO4J_EXCEPTIONS: tuple[type[BaseException], ...] = (
        SessionExpired,
        ServiceUnavailable,
        TransientError,
    )
    # A narrower family than the retryable one, on purpose: TransientError
    # means the server answered and asked to be asked again (a deadlock, a
    # leader switch), so the graph is reachable and the breaker below must not
    # trip on it. These two mean nobody answered.
    _OUTAGE_NEO4J_EXCEPTIONS: tuple[type[BaseException], ...] = (
        SessionExpired,
        ServiceUnavailable,
    )
except (
    Exception
):  # pragma: no cover - fallback if neo4j exception classes are unavailable
    _RETRYABLE_NEO4J_EXCEPTIONS = ()
    _OUTAGE_NEO4J_EXCEPTIONS = ()

from graphrag.config import KGConfig
from graphrag.types import KGNode, KGTriple


def _env_float(name: str, default: float) -> float:
    """A positive float from the environment, or the default if it is unusable."""
    try:
        value = float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default

logger = logging.getLogger("graphrag")

# Property holding the multilingual node embedding (scripts/kg/kg_vector_index.py).
VECTOR_PROPERTY = os.getenv("GRAPHRAG_VECTOR_PROPERTY", "embedding")


def NODE_PROPS(expr: str) -> str:
    """Cypher for a node's properties without its embedding vector.

    ``properties(n)`` would ship the whole vector — roughly 10 KB of JSON on
    every node and every triple endpoint — which slows retrieval by an order of
    magnitude and would leak the raw vector into the assembled context. Needs
    APOC; without it the alternative is a fixed property whitelist.

    Args:
        expr: Cypher expression evaluating to a node.

    Returns:
        A Cypher expression yielding the node's properties map, minus the vector.
    """
    return f"apoc.map.removeKey(properties({expr}), '{VECTOR_PROPERTY}')"


class KnowledgeGraphManager:
    """High-level helper for import and query operations on Neo4j."""

    def __init__(self, config: KGConfig, graph: Neo4jGraph | None = None) -> None:
        """Connect to the graph.

        Retries are configured by ``GRAPHRAG_NEO4J_QUERY_RETRIES`` (default 3)
        and ``GRAPHRAG_NEO4J_QUERY_RETRY_BACKOFF_SEC`` (default 1.0); the
        full-text index name by ``GRAPHRAG_FULLTEXT_INDEX`` (default
        ``node_search``).

        Args:
            config: Connection settings and name properties.
            graph: Existing ``Neo4jGraph`` to use instead of building one.
        """
        self.config = config
        self.graph = graph or self._build_graph()

        retry_attempts_raw = os.getenv("GRAPHRAG_NEO4J_QUERY_RETRIES", "3").strip()
        retry_backoff_raw = os.getenv(
            "GRAPHRAG_NEO4J_QUERY_RETRY_BACKOFF_SEC", "1.0"
        ).strip()

        try:
            self.query_retry_attempts = max(1, int(retry_attempts_raw))
        except ValueError:
            self.query_retry_attempts = 3

        try:
            self.query_retry_backoff_sec = max(0.0, float(retry_backoff_raw))
        except ValueError:
            self.query_retry_backoff_sec = 1.0

        self.fulltext_index = os.getenv(
            "GRAPHRAG_FULLTEXT_INDEX", "node_search"
        ).strip()
        # None = not probed yet; False = index missing, use the CONTAINS scan.
        self._fulltext_available: bool | None = None
        # A disabled index is disabled until this monotonic deadline, not for
        # the life of the process: see _fulltext_retry_delay_sec.
        self._fulltext_retry_at: float = 0.0
        self._fulltext_failures: int = 0
        self._token_df: dict[str, int] | None = None
        self._token_df_total: int = 0
        # Queries that lost the vector channel because the index could not be
        # queried. Counted, not just logged, so the demo can say so on the
        # affected answer the way it does for an unreachable encoder.
        self._vector_skips: int = 0
        # Circuit breaker on an unreachable graph: monotonic deadline until
        # which queries fail immediately, and the failure to re-raise.
        self._outage_until: float = 0.0
        self._outage_error: BaseException | None = None

    # ------------------------------------------------------------------ #
    # term specificity
    # ------------------------------------------------------------------ #

    _DF_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

    def token_document_frequency(
        self, cache_path: str | Path | None = None
    ) -> tuple[dict[str, int], int]:
        """How many node names contain each token, plus the node total.

        The full-text query is a flat OR of the query's terms, so a token like
        "framework" that occurs in hundreds of node names outvotes the specific
        phrase simply by matching more nodes. Knowing each token's document
        frequency is what lets the retriever demote it (see
        ``AgentConfig.lexical_specificity``).

        Computed once per process from all node names and cached on disk, keyed
        by the node count so a changed graph invalidates it.

        Args:
            cache_path: JSON cache location. ``None`` disables the disk cache.

        Returns:
            ``(token -> number of node names containing it, node count)``.
        """
        if self._token_df is not None:
            return self._token_df, self._token_df_total

        total = 0
        rows = self.run_query(
            "MATCH (n) WHERE n.name IS NOT NULL RETURN count(n) AS total"
        )
        if rows:
            total = int(rows[0].get("total") or 0)

        path = Path(cache_path) if cache_path else None
        if path and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                # The property list is part of the key: a cache built from
                # `n.name` alone counts different things than one built from all
                # of node_name_properties, at the same node count.
                same_properties = list(cached.get("properties") or []) == list(
                    self.config.node_name_properties
                )
                if int(cached.get("node_count", -1)) == total and same_properties:
                    self._token_df = {
                        str(k): int(v) for k, v in cached.get("token_df", {}).items()
                    }
                    self._token_df_total = total
                    return self._token_df, total
                logger.info(
                    "token DF cache stale (%s nodes / properties %s cached, %s / %s "
                    "now) — recomputing",
                    cached.get("node_count"),
                    cached.get("properties"),
                    total,
                    list(self.config.node_name_properties),
                )
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("unreadable token DF cache %s (%s) — recomputing", path, exc)

        # The frequency must be measured over the same properties the match
        # clause compares: built from `n.name` alone, a token common in titles
        # but absent from names would get no demotion at all.
        text_expr = " + ' ' + ".join(
            f"coalesce(toString(n.{prop}), '')"
            for prop in self.config.node_name_properties
        )
        counts: dict[str, int] = {}
        for row in self.run_query(
            f"MATCH (n) WHERE n.name IS NOT NULL "
            f"RETURN toLower({text_expr}) AS name"
        ):
            name = row.get("name") or ""
            # A token repeated inside one name still counts once: this is a
            # document frequency, not a term frequency.
            for token in set(self._DF_TOKEN_RE.findall(name)):
                counts[token] = counts.get(token, 0) + 1

        self._token_df = counts
        self._token_df_total = total
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "node_count": total,
                            "properties": list(self.config.node_name_properties),
                            "token_df": counts,
                        }
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("could not write token DF cache %s (%s)", path, exc)
        return counts, total

    def _build_graph(self) -> Neo4jGraph:
        """Build a ``Neo4jGraph`` with bounded query, retry and connection timeouts.

        Each timeout can be overridden by its ``GRAPHRAG_NEO4J_*`` variable.
        """
        return Neo4jGraph(
            url=self.config.url,
            username=self.config.username,
            password=self.config.password,
            database=self.config.database,
            timeout=_env_float("GRAPHRAG_NEO4J_QUERY_TIMEOUT_SEC", 45.0),
            driver_config={
                # Almost every retrieval query finishes in a fraction of a
                # second; the slowest, an unindexed CONTAINS scan, takes tens of
                # seconds. The 45 s query timeout leaves room for that one and
                # still bounds a runaway query.
                #
                # The retry window is the setting that matters. At the driver
                # default of 30 s, every query of a retrieval burns the window
                # on an unreachable graph and the retry loop here multiplies
                # it, turning a failover into minutes of dead air. The graph
                # either answers in a fraction of a second or is not there, so
                # 8 s still allows two driver retries.
                "max_transaction_retry_time": _env_float(
                    "GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", 8.0
                ),
                "connection_timeout": _env_float(
                    "GRAPHRAG_NEO4J_CONNECTION_TIMEOUT_SEC", 5.0
                ),
                "connection_acquisition_timeout": _env_float(
                    "GRAPHRAG_NEO4J_ACQUISITION_TIMEOUT_SEC", 10.0
                ),
            },
        )

    def _reconnect(self) -> None:
        """Replace the graph connection, closing the previous driver.

        Closing matters: on a flaky link the retry loop reconnects on every
        attempt, and each abandoned driver would keep its connection pool
        alive.
        """
        previous = getattr(self, "graph", None)
        self.graph = self._build_graph()
        if previous is not None:
            try:
                driver = getattr(previous, "_driver", None)
                if driver is not None:
                    driver.close()
            except Exception as exc:  # noqa: BLE001 - closing must not mask the retry
                logger.debug("could not close the previous Neo4j driver: %s", exc)

    @staticmethod
    def _is_retryable_query_error(exc: BaseException) -> bool:
        """Whether ``exc`` is a transient Neo4j or network failure worth retrying."""
        if _RETRYABLE_NEO4J_EXCEPTIONS and isinstance(exc, _RETRYABLE_NEO4J_EXCEPTIONS):
            return True

        text = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "sessionexpired",
            "serviceunavailable",
            "transienterror",
            "defunct connection",
            "connection reset",
            "connection aborted",
            "connection was closed",
            "failed to read from defunct connection",
            "failed to read",
            "network",
            "timed out",
        )
        return any(marker in text for marker in markers)

    @classmethod
    def from_env(
        cls,
        url_env: str = "NEO4J_URL",
        username_env: str = "NEO4J_USERNAME",
        password_env: str = "NEO4J_PASSWORD",
        database_env: str = "NEO4J_DATABASE",
    ) -> "KnowledgeGraphManager":
        """Build a manager from environment variables.

        Args:
            url_env: Variable holding the URI.
            username_env: Variable holding the user name.
            password_env: Variable holding the password.
            database_env: Variable holding the database name (optional).

        Returns:
            The connected manager.

        Raises:
            KeyError: If the URI, user name or password variable is unset.
        """
        return cls(
            KGConfig(
                url=os.environ[url_env],
                username=os.environ[username_env],
                password=os.environ[password_env],
                database=os.environ.get(database_env),
            )
        )

    @property
    def schema(self) -> str:
        """The graph schema as last fetched by ``Neo4jGraph``."""
        return getattr(self.graph, "schema", "")

    def refresh_schema(self) -> str:
        """Re-read the schema from the database and return it."""
        self.graph.refresh_schema()
        return self.schema

    # How long one unreachable-graph verdict stands for the queries that follow
    # it. A retrieval issues several queries, and without this each one would
    # rediscover the outage from scratch: driver retry window, then this
    # loop's own attempts, each with a reconnect. Short on purpose: it only has
    # to cover the rest of the retrieval in progress, after which the demo has
    # rebuilt onto the other graph. Anything longer would keep failing
    # questions the graph could already answer again.
    _OUTAGE_MEMORY_SEC = 5.0

    @staticmethod
    def _is_outage(exc: BaseException) -> bool:
        """Whether nobody answered, as opposed to answering with a refusal.

        Deliberately narrower than `_is_retryable_query_error`: a TransientError
        is the server asking to be asked again, so the graph is reachable and
        the breaker must not trip.
        """
        if _OUTAGE_NEO4J_EXCEPTIONS and isinstance(exc, _OUTAGE_NEO4J_EXCEPTIONS):
            return True
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            marker in text
            for marker in (
                "serviceunavailable",
                "sessionexpired",
                "could not connect",
                "couldn't connect",
                "unable to retrieve routing information",
                "cannot resolve address",
                "couldn't resolve address",
            )
        )

    def run_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a Cypher query, retrying transient failures with a reconnect.

        After an outage (nobody answered), later queries fail immediately with
        the same error for ``_OUTAGE_MEMORY_SEC``.

        Args:
            cypher: Query text.
            params: Query parameters.

        Returns:
            The result rows.

        Raises:
            Exception: The driver's error, when it is not retryable, the
                retries are exhausted, or an outage is being remembered.
        """
        payload = params or {}
        max_attempts = max(1, self.query_retry_attempts)

        if self._outage_error is not None:
            if time.monotonic() < self._outage_until:
                # Already established, moments ago, that nobody is there.
                raise self._outage_error
            self._outage_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                rows = self.graph.query(cypher, payload)
            except Exception as exc:
                retryable = self._is_retryable_query_error(exc)
                if not retryable or attempt >= max_attempts:
                    if self._is_outage(exc):
                        self._outage_until = time.monotonic() + self._OUTAGE_MEMORY_SEC
                        self._outage_error = exc
                        logger.warning(
                            "Neo4j unreachable; failing the next %.0f s of queries "
                            "immediately instead of rediscovering it each time: %s",
                            self._OUTAGE_MEMORY_SEC,
                            exc,
                        )
                    raise

                backoff_sec = self.query_retry_backoff_sec * attempt
                logger.warning(
                    "Neo4j transient query failure. retry=%d/%d backoff_sec=%.2f error=%s",
                    attempt,
                    max_attempts,
                    backoff_sec,
                    exc,
                )

                try:
                    self._reconnect()
                except Exception as reconnect_exc:  # pragma: no cover - depends on runtime network state
                    logger.warning("Neo4j reconnect attempt failed: %s", reconnect_exc)

                if backoff_sec > 0:
                    time.sleep(backoff_sec)
            else:
                self._outage_error = None
                return rows

        raise RuntimeError("unreachable: retry loop either returns or raises")

    def clear(self) -> None:
        """Delete every node and relationship in the database."""
        self.run_query("MATCH (n) DETACH DELETE n")

    def import_triples(
        self,
        triples: Sequence[dict[str, Any]],
        subject_label: str = "Entity",
        object_label: str = "Entity",
        relationship_type: str = "RELATED_TO",
    ) -> int:
        """MERGE triples into the graph under one label pair and one type.

        The triple's own predicate is stored in the relationship's
        ``predicate`` property.

        Args:
            triples: Dicts with ``subject``, ``predicate``, ``object`` and
                optional ``relationship_properties``.
            subject_label: Label of subject nodes.
            object_label: Label of object nodes.
            relationship_type: Relationship type.

        Returns:
            The number of relationships written.
        """
        if not triples:
            return 0

        subject_label = self._safe_identifier(subject_label)
        object_label = self._safe_identifier(object_label)
        relationship_type = self._safe_identifier(relationship_type)

        cypher = f"""
        UNWIND $triples AS triple
        MERGE (s:{subject_label} {{name: triple.subject}})
        MERGE (o:{object_label} {{name: triple.object}})
        MERGE (s)-[r:{relationship_type}]->(o)
        SET r.predicate = triple.predicate
        SET r += coalesce(triple.relationship_properties, {{}})
        RETURN count(r) AS relationships_written
        """
        rows = self.run_query(cypher, {"triples": list(triples)})
        return int(rows[0].get("relationships_written", 0)) if rows else 0

    def extract_nodes(
        self,
        text: str | None = None,
        labels: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[KGNode]:
        """Find nodes whose name contains ``text`` (CONTAINS scan).

        Args:
            text: Substring to match against the name properties.
            labels: Optional label whitelist.
            limit: Maximum nodes; defaults to ``config.default_limit``.

        Returns:
            The matching nodes.
        """
        limit = limit or self.config.default_limit
        where_clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}

        if labels:
            where_clauses.append("any(label IN labels(n) WHERE label IN $labels)")
            params["labels"] = list(labels)

        if text:
            where_clauses.append(self._node_text_match_clause("n", "text"))
            params["text"] = text

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cypher = f"""
        MATCH (n)
        {where_sql}
        RETURN DISTINCT
            elementId(n) AS node_id,
            labels(n) AS labels,
            {NODE_PROPS('n')} AS properties,
            {self._coalesce_name_expr("n")} AS text
        LIMIT $limit
        """
        return [self._row_to_node(row) for row in self.run_query(cypher, params)]

    def extract_triples(
        self,
        text: str | None = None,
        labels: Sequence[str] | None = None,
        relationship_types: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[KGTriple]:
        """Find triples with an endpoint whose name contains ``text`` (CONTAINS scan).

        Args:
            text: Substring to match against either endpoint's name.
            labels: Optional label whitelist for either endpoint.
            relationship_types: Optional relationship-type whitelist.
            limit: Maximum triples; defaults to ``config.default_limit``.

        Returns:
            The matching triples.
        """
        limit = limit or self.config.default_limit
        where_clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}

        if labels:
            where_clauses.append(
                "(any(label IN labels(s) WHERE label IN $labels) OR any(label IN labels(o) WHERE label IN $labels))"
            )
            params["labels"] = list(labels)

        if relationship_types:
            where_clauses.append("type(r) IN $relationship_types")
            params["relationship_types"] = list(relationship_types)

        if text:
            where_clauses.append(
                f"({self._node_text_match_clause('s', 'text')} OR {self._node_text_match_clause('o', 'text')})"
            )
            params["text"] = text

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cypher = f"""
        MATCH (s)-[r]->(o)
        {where_sql}
        RETURN DISTINCT
            elementId(s) AS subject_id,
            {self._coalesce_name_expr("s")} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(o) AS object_id,
            {self._coalesce_name_expr("o")} AS object,
            labels(s) AS subject_labels,
            labels(o) AS object_labels,
            {NODE_PROPS('s')} AS subject_properties,
            {NODE_PROPS('o')} AS object_properties,
            properties(r) AS relationship_properties
        LIMIT $limit
        """
        return [self._row_to_triple(row) for row in self.run_query(cypher, params)]

    _LUCENE_SPECIAL_RE = re.compile(r'(&&|\|\||[+\-!(){}\[\]^"~*?:\\/])')
    # Must match "the index does not exist" and nothing else. A generic "not
    # found", or the procedure name, would also match Neo4j's Lucene parse
    # error, which names the procedure it was invoking, so one malformed query
    # would disable full-text search and drop retrieval onto the CONTAINS
    # scan.
    _FULLTEXT_MISSING_MARKERS = (
        "no such fulltext schema index",
        "there is no such fulltext schema index",
        "no such index",
        "unable to find index",
    )

    @classmethod
    def _lucene_query(
        cls, terms: Sequence[str], boosts: Mapping[str, float] | None = None
    ) -> str:
        """Build a Lucene OR-query from search terms (phrases quoted).

        Args:
            terms: Search terms.
            boosts: Optional per-term weight. Without it every term counts the
                same, which lets a generic single token match more nodes than a
                specific phrase and take over the result set.

        Returns:
            The query, special characters escaped; empty when no term is left.
        """
        parts: list[str] = []
        for term in terms:
            cleaned = str(term or "").strip()
            if not cleaned:
                continue
            escaped = cls._LUCENE_SPECIAL_RE.sub(r"\\\1", cleaned)
            part = f'"{escaped}"' if " " in escaped else escaped
            weight = (boosts or {}).get(term)
            if weight is not None and abs(weight - 1.0) > 1e-6:
                part = f"{part}^{weight:g}"
            parts.append(part)
        return " OR ".join(parts)

    # ------------------------------------------------------------------ #
    # vector channel
    # ------------------------------------------------------------------ #

    def vector_search_nodes(
        self,
        vector: Sequence[float],
        limit: int,
        index: str = "node_embedding",
        labels: Sequence[str] | None = None,
        min_score: float = 0.0,
    ) -> list[KGNode]:
        """Nodes nearest to ``vector`` in the multilingual embedding space.

        This is the cross-lingual half of retrieval: the full-text index only
        matches surface forms, so an English question never reaches a node named
        ``polifenoli`` or ``Ciclicità``. Requires ``scripts/kg/kg_vector_index.py``
        to have been run.

        Args:
            vector: Query embedding, same encoder as the index.
            limit: Maximum nodes returned, nearest first.
            index: Vector index name.
            labels: Optional label whitelist applied after the lookup.
            min_score: Drop matches below this cosine score.

        Returns:
            Nodes ordered by decreasing similarity; empty when the index is
            missing, so the caller keeps whatever the lexical channel found.
        """
        if limit <= 0 or not vector:
            return []
        params: dict[str, Any] = {
            "index": index,
            "limit": limit,
            "vec": list(vector),
            "min_score": min_score,
        }
        label_filter = ""
        if labels:
            label_filter = "AND any(label IN labels(node) WHERE label IN $labels)"
            params["labels"] = list(labels)
        # The index holds :NodeVec carriers, one per entity, so the entity's
        # own properties and labels carry no vector; `of` is the entity's
        # elementId.
        cypher = f"""
        CALL db.index.vector.queryNodes($index, $limit, $vec)
        YIELD node AS carrier, score
        MATCH (node) WHERE elementId(node) = carrier.of
        WITH node, score WHERE score >= $min_score {label_filter}
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            {NODE_PROPS('node')} AS properties,
            {self._coalesce_name_expr("node")} AS text
        """
        try:
            rows = self.run_query(cypher, params)
        except Exception as exc:  # noqa: BLE001 - narrowed by marker check
            if self._handle_vector_error(exc, index):
                return []
            raise
        return [self._row_to_node(row) for row in rows]

    def vector_search_triples(
        self,
        vector: Sequence[float],
        limit: int,
        seed_limit: int = 10,
        index: str = "node_embedding",
        labels: Sequence[str] | None = None,
        relationship_types: Sequence[str] | None = None,
    ) -> list[KGTriple]:
        """Triples touching the nodes nearest to ``vector``.

        Mirrors :meth:`fulltext_search_triples` so the two channels return the
        same shape and can be merged without special cases.

        Args:
            vector: Query embedding.
            limit: Maximum triples returned.
            seed_limit: How many nearest nodes to expand around.
            index: Vector index name.
            labels: Optional label whitelist for either endpoint.
            relationship_types: Optional relationship-type whitelist.

        Returns:
            Triples ordered by the seed node's similarity.
        """
        if limit <= 0 or not vector:
            return []
        params: dict[str, Any] = {
            "index": index,
            "seed_limit": seed_limit,
            "vec": list(vector),
            "limit": limit,
        }
        filters: list[str] = []
        if labels:
            filters.append(
                "(any(label IN labels(startNode(r)) WHERE label IN $labels) "
                "OR any(label IN labels(endNode(r)) WHERE label IN $labels))"
            )
            params["labels"] = list(labels)
        if relationship_types:
            filters.append("type(r) IN $relationship_types")
            params["relationship_types"] = list(relationship_types)
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        cypher = f"""
        CALL db.index.vector.queryNodes($index, $seed_limit, $vec)
        YIELD node AS carrier, score
        MATCH (node) WHERE elementId(node) = carrier.of
        MATCH (node)-[r]-()
        WITH r, max(score) AS seed_score
        {where_sql}
        RETURN
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr("startNode(r)")} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr("endNode(r)")} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            {NODE_PROPS('startNode(r)')} AS subject_properties,
            {NODE_PROPS('endNode(r)')} AS object_properties,
            properties(r) AS relationship_properties
        ORDER BY seed_score DESC
        LIMIT $limit
        """
        try:
            rows = self.run_query(cypher, params)
        except Exception as exc:  # noqa: BLE001 - narrowed by marker check
            if self._handle_vector_error(exc, index):
                return []
            raise
        return [self._row_to_triple(row) for row in rows]

    # Neo4j 5 says exactly this when the index does not exist:
    #   Failed to invoke procedure `db.index.vector.queryNodes`: Caused by:
    #   java.lang.IllegalArgumentException: There is no such vector schema
    #   index: <name>
    # Nothing broader: every failure of this procedure names the procedure,
    # and "not found" also matches ProcedureNotFound or DatabaseNotFound. A
    # broader marker would report unrelated faults (a dimension mismatch from
    # a different encoder, for instance) as a missing index, with advice that
    # cannot fix them.
    _VECTOR_MISSING_MARKERS = ("there is no such vector schema index",)

    def _handle_vector_error(self, exc: Exception, index: str) -> bool:
        """Skip the vector channel for this query, and say why.

        Always returns True: the lexical and text channels can still answer,
        and killing the question over one unavailable channel is the wrong
        trade for an interactive product. What changes with the cause is the
        log — a missing index is an operator task with a known fix, anything
        else is a fault that needs reading before it is guessed at.

        Args:
            exc: The query error.
            index: Vector index name, for the log.

        Returns:
            True.
        """
        self._vector_skips += 1
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(marker in text for marker in self._VECTOR_MISSING_MARKERS):
            logger.warning(
                "Vector index %r does not exist — the cross-lingual channel is "
                "skipped for this query. Run scripts/kg/kg_vector_index.py to build it.",
                index,
            )
        else:
            logger.error(
                "Vector index %r could not be queried, and not because it is "
                "missing — the cross-lingual channel is skipped for this query "
                "and answers are lexical-only until this is fixed: %s",
                index,
                exc,
            )
        return True

    @property
    def vector_skips(self) -> int:
        """How many queries lost the vector channel at the index."""
        return self._vector_skips

    # Backoff schedule for a disabled index, in seconds. Disabling it for the
    # life of the process would downgrade every later question — in the
    # Streamlit demo, every question of every connected user — to the slower
    # and worse CONTAINS scan after one transient failure. Retrying on a fixed
    # short interval is the opposite mistake: when the index is genuinely
    # missing, each probe pays a full failed query. So: retry soon at first,
    # then back off to once a quarter of an hour.
    _FULLTEXT_RETRY_BACKOFF_SEC = (30.0, 120.0, 300.0, 900.0)

    def _fulltext_retry_delay_sec(self) -> float:
        """Current backoff delay, from the number of consecutive failures."""
        index = min(self._fulltext_failures, len(self._FULLTEXT_RETRY_BACKOFF_SEC)) - 1
        return self._FULLTEXT_RETRY_BACKOFF_SEC[max(index, 0)]

    def _fulltext_ready(self) -> bool:
        """Whether to use the index, re-enabling it once its backoff expired."""
        if self._fulltext_available is not False:
            return True
        if time.monotonic() < self._fulltext_retry_at:
            return False
        logger.info(
            "Re-probing full-text index %r after %.0f s disabled.",
            self.fulltext_index,
            self._fulltext_retry_delay_sec(),
        )
        self._fulltext_available = None
        return True

    def _handle_fulltext_error(self, exc: Exception) -> bool:
        """Return True (and disable full-text) when the index is unavailable."""
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(marker in text for marker in self._FULLTEXT_MISSING_MARKERS):
            self._fulltext_failures += 1
            delay_sec = self._fulltext_retry_delay_sec()
            logger.warning(
                "Full-text index %r unavailable (%s) — falling back to the "
                "CONTAINS scan, retrying in %.0f s. Run "
                "scripts/kg/kg_search_index.py to create the index.",
                self.fulltext_index,
                exc,
                delay_sec,
            )
            self._fulltext_available = False
            self._fulltext_retry_at = time.monotonic() + delay_sec
            return True
        return False

    def fulltext_search_nodes(
        self,
        terms: Sequence[str],
        labels: Sequence[str] | None = None,
        limit: int | None = None,
        boosts: Mapping[str, float] | None = None,
    ) -> list[KGNode] | None:
        """Match nodes for all ``terms`` in one indexed query.

        Args:
            terms: Search terms; combined into a single Lucene OR-query.
            labels: Optional label whitelist applied after the index lookup.
            limit: Maximum nodes returned (best score first).
            boosts: Optional per-term Lucene weights.

        Returns:
            Nodes ordered by Lucene score, or ``None`` when the full-text index
            is unavailable and the caller must fall back to the CONTAINS scan.
        """
        if not self._fulltext_ready():
            return None
        lucene = self._lucene_query(terms, boosts)
        if not lucene:
            return []
        limit = limit or self.config.default_limit
        params: dict[str, Any] = {
            "index": self.fulltext_index,
            "q": lucene,
            "limit": limit,
        }
        label_filter = ""
        if labels:
            label_filter = "WHERE any(label IN labels(node) WHERE label IN $labels)"
            params["labels"] = list(labels)
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index, $q, {{limit: $limit}})
        YIELD node, score
        {label_filter}
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            {NODE_PROPS('node')} AS properties,
            {self._coalesce_name_expr("node")} AS text
        """
        try:
            rows = self.run_query(cypher, params)
        except Exception as exc:  # noqa: BLE001 - narrowed by marker check
            if self._handle_fulltext_error(exc):
                return None
            raise
        self._fulltext_available = True
        self._fulltext_failures = 0
        return [self._row_to_node(row) for row in rows]

    def fulltext_search_triples(
        self,
        terms: Sequence[str],
        labels: Sequence[str] | None = None,
        relationship_types: Sequence[str] | None = None,
        limit: int | None = None,
        boosts: Mapping[str, float] | None = None,
    ) -> list[KGTriple] | None:
        """Match triples around index-matched nodes in one query.

        Args:
            terms: Search terms; combined into a single Lucene OR-query.
            labels: Optional label whitelist for either endpoint.
            relationship_types: Optional relationship-type whitelist.
            limit: Maximum triples returned (best seed score first).
            boosts: Optional per-term Lucene weights.

        Returns:
            Triples ordered by the matched endpoint's Lucene score, or ``None``
            when the full-text index is unavailable.
        """
        if not self._fulltext_ready():
            return None
        lucene = self._lucene_query(terms, boosts)
        if not lucene:
            return []
        limit = limit or self.config.default_limit
        params: dict[str, Any] = {
            "index": self.fulltext_index,
            "q": lucene,
            "limit": limit,
        }
        filters: list[str] = []
        if labels:
            filters.append(
                "(any(label IN labels(startNode(r)) WHERE label IN $labels) "
                "OR any(label IN labels(endNode(r)) WHERE label IN $labels))"
            )
            params["labels"] = list(labels)
        if relationship_types:
            filters.append("type(r) IN $relationship_types")
            params["relationship_types"] = list(relationship_types)
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index, $q, {{limit: $limit}})
        YIELD node, score
        MATCH (node)-[r]-()
        WITH r, max(score) AS seed_score
        {where_sql}
        RETURN
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr("startNode(r)")} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr("endNode(r)")} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            {NODE_PROPS('startNode(r)')} AS subject_properties,
            {NODE_PROPS('endNode(r)')} AS object_properties,
            properties(r) AS relationship_properties
        ORDER BY seed_score DESC
        LIMIT $limit
        """
        try:
            rows = self.run_query(cypher, params)
        except Exception as exc:  # noqa: BLE001 - narrowed by marker check
            if self._handle_fulltext_error(exc):
                return None
            raise
        self._fulltext_available = True
        self._fulltext_failures = 0
        return [self._row_to_triple(row) for row in rows]

    def extract_subgraph(
        self,
        entity: str,
        hops: int = 1,
        limit: int = 200,
        relationship_types: Sequence[str] | None = None,
    ) -> list[KGTriple]:
        """Triples within ``hops`` of the node matching ``entity``.

        The seed is matched exactly (by elementId when ``entity`` is one); a
        name that matches nothing exactly is retried by containment.

        Args:
            entity: Seed node name or elementId.
            hops: Path length, at least 1.
            limit: Maximum triples, applied in Cypher in graph order.
            relationship_types: Optional relationship-type whitelist.

        Returns:
            The distinct relationships of the neighbourhood, as triples.
        """
        hops = max(1, int(hops))
        params: dict[str, Any] = {"entity": entity, "limit": limit}
        seed_by_id = self.is_element_id(entity)

        rel_filter = "true"
        if relationship_types:
            rel_filter = "type(r) IN $relationship_types"
            params["relationship_types"] = list(relationship_types)

        cypher_exact = f"""
        MATCH (seed)
        WHERE {self._node_text_match_clause("seed", "entity", exact=True, id_only=seed_by_id)}
        MATCH p = (seed)-[*1..{hops}]-(other)
        UNWIND relationships(p) AS r
        WITH DISTINCT r
        WHERE {rel_filter}
        RETURN DISTINCT
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr("startNode(r)")} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr("endNode(r)")} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            {NODE_PROPS('startNode(r)')} AS subject_properties,
            {NODE_PROPS('endNode(r)')} AS object_properties,
            properties(r) AS relationship_properties
        LIMIT $limit
        """

        rows = self.run_query(cypher_exact, params)
        # Same rule as get_shortest_path: broadening is for names. For an
        # elementId the containment fallback would ask which node *names*
        # contain an id — never any — at the price of a scan of every node over
        # six lowercased properties, so it is skipped.
        if not rows and not seed_by_id:
            # Fall back to a looser text match when exact matching returns nothing.
            cypher_fallback = f"""
            MATCH (seed)
            WHERE {self._node_text_match_clause("seed", "entity", exact=False)}
            MATCH p = (seed)-[*1..{hops}]-(other)
            UNWIND relationships(p) AS r
            WITH DISTINCT r
            WHERE {rel_filter}
            RETURN DISTINCT
                elementId(startNode(r)) AS subject_id,
                {self._coalesce_name_expr("startNode(r)")} AS subject,
                coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
                elementId(endNode(r)) AS object_id,
                {self._coalesce_name_expr("endNode(r)")} AS object,
                labels(startNode(r)) AS subject_labels,
                labels(endNode(r)) AS object_labels,
                {NODE_PROPS('startNode(r)')} AS subject_properties,
                {NODE_PROPS('endNode(r)')} AS object_properties,
                properties(r) AS relationship_properties
            LIMIT $limit
            """
            rows = self.run_query(cypher_fallback, params)

        return [self._row_to_triple(row) for row in rows]

    def entity_exists(self, entity: str) -> bool:
        """Whether any node's name matches ``entity`` exactly or by containment.

        The neighbour, subgraph and shortest-path queries all start from a
        relationship scan filtered by a string comparison on the seed. When the
        seed matches nothing — which is what happens when the anchor is a raw
        question word like "valuable" — each of those scans still walks the whole
        graph, taking seconds, before returning zero rows. One existence check
        up front, trying the elementId, then an exact name, then containment,
        avoids that.

        Args:
            entity: Candidate anchor, a node name or an elementId.

        Returns:
            True when at least one node matches.
        """
        cleaned = str(entity or "").strip()
        if not cleaned:
            return False
        rows = self.run_query(
            "MATCH (n) WHERE elementId(n) = $entity RETURN 1 AS ok LIMIT 1",
            {"entity": cleaned},
        )
        if rows:
            return True
        cypher = (
            "MATCH (n) WHERE "
            f"{self._node_text_match_clause('n', 'entity', exact=True)} "
            "RETURN 1 AS ok LIMIT 1"
        )
        if self.run_query(cypher, {"entity": cleaned}):
            return True
        cypher = (
            "MATCH (n) WHERE "
            f"{self._node_text_match_clause('n', 'entity', exact=False)} "
            "RETURN 1 AS ok LIMIT 1"
        )
        return bool(self.run_query(cypher, {"entity": cleaned}))

    def get_neighbors(
        self,
        entity: str,
        limit: int = 25,
        relationship_types: Sequence[str] | None = None,
    ) -> list[KGNode]:
        """Nodes directly connected to the node matching ``entity``.

        Matched like :meth:`extract_subgraph`: exactly first, then by
        containment for a name.

        Args:
            entity: Seed node name or elementId.
            limit: Maximum neighbours.
            relationship_types: Optional relationship-type whitelist.

        Returns:
            The distinct neighbours.
        """
        params: dict[str, Any] = {"entity": entity, "limit": limit}
        seed_by_id = self.is_element_id(entity)
        rel_clause = ""
        if relationship_types:
            rel_clause = "AND type(r) IN $relationship_types"
            params["relationship_types"] = list(relationship_types)
        cypher_exact = f"""
        MATCH (seed)-[r]-(neighbor)
        WHERE {self._node_text_match_clause("seed", "entity", exact=True, id_only=seed_by_id)}
        {rel_clause}
        RETURN DISTINCT
            elementId(neighbor) AS node_id,
            labels(neighbor) AS labels,
            {NODE_PROPS('neighbor')} AS properties,
            {self._coalesce_name_expr("neighbor")} AS text
        LIMIT $limit
        """
        rows = self.run_query(cypher_exact, params)
        # Same rule as get_shortest_path: broadening is for names. For an
        # elementId the containment fallback would ask which node *names*
        # contain an id — never any — at the price of a scan of every node over
        # six lowercased properties, so it is skipped.
        if not rows and not seed_by_id:
            cypher_fallback = f"""
            MATCH (seed)-[r]-(neighbor)
            WHERE {self._node_text_match_clause("seed", "entity", exact=False)}
            {rel_clause}
            RETURN DISTINCT
                elementId(neighbor) AS node_id,
                labels(neighbor) AS labels,
                {NODE_PROPS('neighbor')} AS properties,
                {self._coalesce_name_expr("neighbor")} AS text
            LIMIT $limit
            """
            rows = self.run_query(cypher_fallback, params)

        return [self._row_to_node(row) for row in rows]

    def get_entity_types(self, entity: str) -> list[str]:
        """Labels of a node whose name equals ``entity``; empty if none."""
        cypher = f"""
        MATCH (n)
        WHERE {self._node_text_match_clause("n", "entity", exact=True)}
        RETURN DISTINCT labels(n) AS labels
        LIMIT 1
        """
        rows = self.run_query(cypher, {"entity": entity})
        if not rows:
            return []
        return list(rows[0].get("labels", []))

    def get_shortest_path(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 6,
    ) -> list[KGTriple]:
        """Triples on shortest paths between the nodes matching two anchors.

        Up to 8 candidates for ``entity_a`` (shortest names first) are paired
        with up to 2 candidates for ``entity_b`` each. Anchors are matched
        exactly, then by containment unless both are elementIds. A failing
        query is logged and yields no triples.

        Args:
            entity_a: First anchor, a name or elementId.
            entity_b: Second anchor, a name or elementId.
            max_depth: Maximum path length.

        Returns:
            The distinct relationships on the paths found, as triples.
        """
        # max_depth is interpolated into a variable-length pattern, so force it
        # to a safe positive integer (defence-in-depth; mirrors extract_subgraph).
        max_depth = max(1, int(max_depth))
        a_by_id = self.is_element_id(entity_a)
        b_by_id = self.is_element_id(entity_b)
        # The pair budget is spent per `a`, not globally: a flat limit on the
        # (a, b) product lets the surviving pairs share the same one or two b
        # nodes, so most a candidates get no partner. Two partners each keeps
        # the budget while guaranteeing every a is tried.
        cypher_exact = f"""
        MATCH (a)
        WHERE {self._node_text_match_clause("a", "entity_a", exact=True, id_only=a_by_id)}
        WITH DISTINCT a
        ORDER BY size({self._coalesce_name_expr("a")}) ASC
        LIMIT 8
        MATCH (b)
        WHERE {self._node_text_match_clause("b", "entity_b", exact=True, id_only=b_by_id)}
        WITH DISTINCT a, b
        ORDER BY size({self._coalesce_name_expr("b")}) ASC
        WITH a, collect(b)[0..2] AS partners
        UNWIND partners AS b
        MATCH p = shortestPath((a)-[*1..{max_depth}]-(b))
        UNWIND relationships(p) AS r
        RETURN DISTINCT
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr("startNode(r)")} AS subject,
                        coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr("endNode(r)")} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            {NODE_PROPS('startNode(r)')} AS subject_properties,
            {NODE_PROPS('endNode(r)')} AS object_properties,
            properties(r) AS relationship_properties
        """
        try:
            rows = self.run_query(cypher_exact, {"entity_a": entity_a, "entity_b": entity_b})
        except Exception as exc:
            # If the shortestPath call fails (e.g. same-node cartesian product),
            # return an empty result instead of propagating the DB error — but
            # log it, so "no path" stays distinguishable from "query failed".
            logger.warning(
                "shortestPath exact query failed for (%r, %r): %s",
                entity_a,
                entity_b,
                exc,
            )
            return []

        # Broadening is for names. An elementId either identifies a node or it
        # does not, and `_graph_anchors` hands this method ids on purpose (see
        # its docstring: the name path compares six lowercased properties on
        # every node, which is a scan). Running the CONTAINS fallback on an id
        # asks which node *names* contain a UUID — never any — and the only
        # nodes it can add are those whose own id contains the anchor's as a
        # substring, which "4:<uuid>:1" inside "4:<uuid>:12" makes an artifact
        # of id formatting rather than a relationship. On ids the fallback
        # returns nothing and dominates retrieval time.
        if not rows and not (a_by_id and b_by_id):
            # With a name on at least one side there is something to broaden.
            # The CONTAINS side can match hundreds of nodes on a generic term
            # ("food"): unbounded, the a×b cartesian product of shortestPath
            # calls takes tens of seconds and floods the context with deep-path
            # noise. Shortest names first ≈ most canonical.
            cypher_fallback = f"""
            MATCH (a)
            WHERE {self._node_text_match_clause("a", "entity_a", exact=False, id_only=a_by_id)}
            WITH DISTINCT a
            ORDER BY size({self._coalesce_name_expr("a")}) ASC
            LIMIT 8
            MATCH (b)
            WHERE {self._node_text_match_clause("b", "entity_b", exact=False, id_only=b_by_id)}
            WITH DISTINCT a, b
            ORDER BY size({self._coalesce_name_expr("b")}) ASC
            WITH a, collect(b)[0..2] AS partners
            UNWIND partners AS b
            MATCH p = shortestPath((a)-[*1..{max_depth}]-(b))
            UNWIND relationships(p) AS r
            RETURN DISTINCT
                elementId(startNode(r)) AS subject_id,
                {self._coalesce_name_expr("startNode(r)")} AS subject,
                            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
                elementId(endNode(r)) AS object_id,
                {self._coalesce_name_expr("endNode(r)")} AS object,
                labels(startNode(r)) AS subject_labels,
                labels(endNode(r)) AS object_labels,
                {NODE_PROPS('startNode(r)')} AS subject_properties,
                {NODE_PROPS('endNode(r)')} AS object_properties,
                properties(r) AS relationship_properties
            """
            try:
                rows = self.run_query(cypher_fallback, {"entity_a": entity_a, "entity_b": entity_b})
            except Exception as exc:
                logger.warning(
                    "shortestPath fallback query failed for (%r, %r): %s",
                    entity_a,
                    entity_b,
                    exc,
                )
                return []

        return [self._row_to_triple(row) for row in rows]

    def triples_to_text(self, triples: Sequence[KGTriple]) -> str:
        """Render triples as ``(subject, predicate, object)`` lines."""
        return "\n".join(
            f"({triple.get('subject', '')}, {triple.get('predicate', '')}, {triple.get('object', '')})"
            for triple in triples
        )

    def nodes_to_text(self, nodes: Sequence[KGNode]) -> str:
        """Render node names one per line, skipping nameless nodes."""
        return "\n".join(node.get("text", "") for node in nodes if node.get("text"))

    def get_subgraph_context(self, entity: str, hops: int = 1, limit: int = 200) -> str:
        """Render the subgraph around ``entity`` as triple lines."""
        return self.triples_to_text(
            self.extract_subgraph(entity=entity, hops=hops, limit=limit)
        )

    @staticmethod
    def _safe_identifier(value: str) -> str:
        """Upper-case ``value`` reduced to ``[0-9A-Z_]``, safe as a Cypher identifier.

        Returns ``RELATED_TO`` when nothing is left, and prefixes ``_`` when the
        result would start with a digit.
        """
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            return "RELATED_TO"
        if cleaned[0].isdigit():
            cleaned = f"_{cleaned}"
        return cleaned.upper()

    _ELEMENT_ID_RE = re.compile(r"^\d+:[0-9a-f-]{36}:\d+$", re.IGNORECASE)

    @classmethod
    def is_element_id(cls, value: str) -> bool:
        """Whether ``value`` is a Neo4j elementId rather than a node name."""
        return bool(cls._ELEMENT_ID_RE.match(str(value or "").strip()))

    def _node_text_match_clause(
        self, alias: str, param_name: str, exact: bool = False, id_only: bool = False
    ) -> str:
        """Cypher predicate matching a node by elementId or by any name property.

        Args:
            alias: Node variable.
            param_name: Query parameter holding the value.
            exact: Compare with ``=`` instead of ``CONTAINS``.
            id_only: Emit only the elementId comparison. The name comparisons
                call ``toLower`` on six properties, which forces a scan of every
                candidate; when the caller already knows the value is an
                elementId, dropping them lets the lookup be direct.

        Returns:
            A parenthesised Cypher boolean expression.
        """
        if id_only:
            return f"(elementId({alias}) = ${param_name})"
        operator = "=" if exact else "CONTAINS"
        properties_expr = f"properties({alias})"
        comparisons = [
            f"toLower(coalesce(toString({properties_expr}['{prop}']), '')) {operator} toLower(${param_name})"
            for prop in self.config.node_name_properties
        ]
        comparisons.append(f"elementId({alias}) {operator} ${param_name}")
        return "(" + " OR ".join(comparisons) + ")"

    def _coalesce_name_expr(self, alias: str) -> str:
        """Cypher for a node's display name: the first set name property, else its elementId."""
        properties_expr = f"properties({alias})"
        props = ", ".join(
            f"toString({properties_expr}['{prop}'])"
            for prop in self.config.node_name_properties
        )
        return f"coalesce({props}, elementId({alias}))"

    @staticmethod
    def _row_to_node(row: dict[str, Any]) -> KGNode:
        """Convert a query row to a ``KGNode``."""
        return {
            "node_id": str(row.get("node_id", "")),
            "labels": list(row.get("labels", [])),
            "properties": dict(row.get("properties", {})),
            "text": str(row.get("text", "")),
        }

    @staticmethod
    def _row_to_triple(row: dict[str, Any]) -> KGTriple:
        """Convert a query row to a ``KGTriple``."""
        return {
            "subject_id": str(row.get("subject_id", "")),
            "subject": str(row.get("subject", "")),
            "predicate": str(row.get("predicate", "")),
            "object_id": str(row.get("object_id", "")),
            "object": str(row.get("object", "")),
            "subject_labels": list(row.get("subject_labels", [])),
            "object_labels": list(row.get("object_labels", [])),
            "subject_properties": dict(row.get("subject_properties", {})),
            "object_properties": dict(row.get("object_properties", {})),
            "relationship_properties": dict(row.get("relationship_properties", {})),
        }
