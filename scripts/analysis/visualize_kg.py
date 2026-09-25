"""Render a sample of the Neo4j graph as an interactive HTML page (pyvis)."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from kg_pipeline.utils import neo4j_env

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    load_dotenv = None


def _resolve_neo4j_env() -> tuple[str, str, str, str | None]:
    """Neo4j connection settings from the environment.

    Returns:
        ``(uri, user, password, database)``.

    Raises:
        ValueError: If the URI, the user or the password is missing.
    """
    uri = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DB") or None

    missing: list[str] = []
    if not uri:
        missing.append("NEO4J_URI or NEO4J_URL")
    if not user:
        missing.append("NEO4J_USER or NEO4J_USERNAME")
    if not password:
        missing.append("NEO4J_PASSWORD")

    if missing:
        raise ValueError(
            "Missing Neo4j env vars: "
            + ", ".join(missing)
            + ". Provide them in shell or in an env file (example: kg_pipeline/.env)."
        )

    return uri, user, password, database


def _node_label(labels: list[str], name: str, element_id: str) -> str:
    """Display name of a node: its name, else label and id tail, else the id."""
    if name and str(name).strip():
        return str(name)
    if labels:
        return f"{labels[0]}:{element_id[-6:]}"
    return element_id


def _tooltip(title: str, labels: list[str], props: dict[str, Any]) -> str:
    """HTML tooltip with a node's title, labels and properties."""
    safe_props = json.dumps(props or {}, ensure_ascii=False, indent=2)
    return (
        f"<b>{title}</b><br>"
        f"labels: {', '.join(labels or [])}<br>"
        f"<pre style='white-space: pre-wrap; margin: 6px 0 0'>{safe_props}</pre>"
    )


def _fetch_subgraph(
    session, limit: int, custom_query: str | None
) -> list[dict[str, Any]]:
    """Fetch up to `limit` relationships with both endpoints.

    Args:
        session: Open Neo4j session.
        limit: Maximum number of relationships.
        custom_query: Cypher returning the same columns, used instead of the
            default query.

    Returns:
        One dict per relationship.
    """
    query = custom_query or (
        "MATCH (s)-[r]->(t) "
        "WITH s, r, t LIMIT $limit "
        "RETURN "
        "  elementId(s) AS s_id, coalesce(s.name, '') AS s_name, labels(s) AS s_labels, properties(s) AS s_props, "
        "  elementId(t) AS t_id, coalesce(t.name, '') AS t_name, labels(t) AS t_labels, properties(t) AS t_props, "
        "  type(r) AS r_type, properties(r) AS r_props"
    )
    return session.run(query, limit=limit).data()


def _build_pyvis(
    rows: list[dict[str, Any]], out_html: Path, directed: bool = True
) -> tuple[int, int]:
    """Write the relationships as an interactive pyvis page.

    Args:
        rows: As returned by `_fetch_subgraph`.
        out_html: Output file.
        directed: Draw the edges as arrows.

    Returns:
        The number of nodes and edges drawn.

    Raises:
        RuntimeError: If pyvis is not installed.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency pyvis. Install with: pip install pyvis"
        ) from exc

    net = Network(
        height="880px",
        width="100%",
        directed=directed,
        bgcolor="#faf7ef",
        font_color="#1f2a2e",
        select_menu=True,
        filter_menu=True,
        cdn_resources="in_line",
    )

    label_colors: dict[str, str] = {}
    palette = [
        "#0a7f5a",
        "#eb5e28",
        "#1d3557",
        "#ffb703",
        "#6d597a",
        "#3a86ff",
        "#8ac926",
        "#bc4749",
    ]

    def color_for(label: str) -> str:
        """A palette colour per label, assigned in order of first use."""
        if label not in label_colors:
            label_colors[label] = palette[len(label_colors) % len(palette)]
        return label_colors[label]

    nodes_seen: set[str] = set()
    edge_count = 0

    for row in rows:
        s_id = str(row["s_id"])
        t_id = str(row["t_id"])

        s_labels = [str(x) for x in row.get("s_labels", [])]
        t_labels = [str(x) for x in row.get("t_labels", [])]

        s_name = _node_label(s_labels, str(row.get("s_name", "")), s_id)
        t_name = _node_label(t_labels, str(row.get("t_name", "")), t_id)

        if s_id not in nodes_seen:
            top_label = s_labels[0] if s_labels else "Node"
            net.add_node(
                s_id,
                label=s_name,
                title=_tooltip(s_name, s_labels, row.get("s_props") or {}),
                color=color_for(top_label),
                group=top_label,
            )
            nodes_seen.add(s_id)

        if t_id not in nodes_seen:
            top_label = t_labels[0] if t_labels else "Node"
            net.add_node(
                t_id,
                label=t_name,
                title=_tooltip(t_name, t_labels, row.get("t_props") or {}),
                color=color_for(top_label),
                group=top_label,
            )
            nodes_seen.add(t_id)

        rel_type = str(row.get("r_type", "RELATED_TO"))
        rel_props = row.get("r_props") or {}
        rel_title = f"<b>{rel_type}</b><br><pre style='white-space: pre-wrap; margin: 6px 0 0'>{json.dumps(rel_props, ensure_ascii=False, indent=2)}</pre>"

        net.add_edge(s_id, t_id, label=rel_type, title=rel_title, arrows="to")
        edge_count += 1

    net.barnes_hut(
        gravity=-30000, central_gravity=0.18, spring_length=170, spring_strength=0.02
    )
    net.set_edge_smooth("dynamic")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_html), notebook=False)

    return len(nodes_seen), edge_count


def _print_stats(rows: list[dict[str, Any]]) -> None:
    """Print the most frequent relationship types and node labels."""
    rel_counter: dict[str, int] = defaultdict(int)
    label_counter: dict[str, int] = defaultdict(int)

    for row in rows:
        rel_counter[str(row.get("r_type", "RELATED_TO"))] += 1
        for lbl in row.get("s_labels", []) or []:
            label_counter[str(lbl)] += 1
        for lbl in row.get("t_labels", []) or []:
            label_counter[str(lbl)] += 1

    top_rels = sorted(rel_counter.items(), key=lambda x: x[1], reverse=True)[:12]
    top_labels = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)[:12]

    print("Top relationship types:")
    for rel, c in top_rels:
        print(f"  - {rel}: {c}")

    print("Top node labels (by endpoint frequency):")
    for lbl, c in top_labels:
        print(f"  - {lbl}: {c}")


def main() -> None:
    """Fetch, render and summarise a sample of the graph."""
    parser = argparse.ArgumentParser(
        description="Visualize a Neo4j KG as interactive HTML."
    )
    parser.add_argument(
        "--out",
        default="artifacts/visualization/kg_graph.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--limit", type=int, default=600, help="Max relationships to fetch"
    )
    parser.add_argument(
        "--query",
        default="",
        help="Custom Cypher query returning s_id,s_name,s_labels,s_props,t_id,t_name,t_labels,t_props,r_type,r_props",
    )
    parser.add_argument("--database", default="", help="Override DB name")
    parser.add_argument(
        "--env-file",
        default="kg_pipeline/.env",
        help="Path to .env file with Neo4j credentials",
    )
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser()
    if env_file.exists() and load_dotenv is not None:
        load_dotenv(env_file, override=False)
        print(f"Loaded environment from: {env_file}")
    elif env_file.exists() and load_dotenv is None:
        print("Warning: python-dotenv not installed; .env file was not loaded.")
        print("Install it with: pip install python-dotenv")

    target = neo4j_env.resolve_target(database=args.database)
    uri, database = target.uri, target.database

    with neo4j_env.connect(target) as driver:
        with driver.session(database=database) as session:
            rows = _fetch_subgraph(
                session,
                limit=max(1, args.limit),
                custom_query=args.query.strip() or None,
            )

    if not rows:
        print("No rows returned by query. Nothing to visualize.")
        return

    out_html = Path(args.out).expanduser()
    node_count, edge_count = _build_pyvis(rows=rows, out_html=out_html, directed=True)

    print(f"Rendered graph: {node_count} nodes, {edge_count} relationships")
    print(f"Saved HTML: {out_html}")
    _print_stats(rows)


if __name__ == "__main__":
    main()
