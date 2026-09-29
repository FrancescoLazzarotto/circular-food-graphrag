"""Draw one real neighbourhood of the knowledge graph, with its provenance.

The thesis reports the graph as a degree distribution and a component count, and
never shows one. This renders the edges around a single node exactly as the graph
holds them: the predicate on the arrow, and the document and page range each
assertion was extracted from under the entity it points at.

Nothing is filtered. Every edge on the anchor is drawn, duplicate node names
included, so the figure carries the graph's defects as well as its content.

Usage::

    NEO4J_URL=bolt://localhost:7689 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=... \\
    conda run -n graphllm python evaluation/scripts/plot_kg_subgraph.py \\
        --anchor "i residui alimentari" \\
        --out-dir /srv/projects/graphllm/experiments/thesis_v6/figures
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

from kg_pipeline.utils import neo4j_env

QUERY = """
MATCH (n {name: $anchor})-[r]-(m)
RETURN m.name          AS neighbour,
       labels(m)       AS labels,
       type(r)         AS predicate,
       startNode(r).name = $anchor AS outgoing,
       r.source_doc    AS document,
       r.page_range    AS pages
ORDER BY document, predicate
"""


def fetch(anchor: str) -> list[dict]:
    """Every edge on `anchor`, with its source document and pages."""
    driver = neo4j_env.connect()
    try:
        with driver.session() as session:
            return [dict(record) for record in session.run(QUERY, anchor=anchor)]
    finally:
        driver.close()


def doc_tags(edges: list[dict]) -> dict[str, str]:
    """One short tag per source document, numbered in first-appearance order."""
    tags: dict[str, str] = {}
    for edge in edges:
        document = edge["document"] or "unrecorded"
        if document not in tags:
            tags[document] = f"D{len(tags) + 1}"
    return tags


def pages(value: str | None) -> str:
    """``11-11`` reads as one page, ``11-12`` as a range."""
    if not value:
        return "no page recorded"
    first, _, last = str(value).partition("-")
    return f"p. {first}" if not last or first == last else f"pp. {first}\u2013{last}"


def draw(anchor: str, edges: list[dict], tags: dict[str, str], out_dir: Path, stem: str) -> list[Path]:
    """Draw the anchor, its edges and their sources; returns the files written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(11, 7.4))
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    box = dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor="#444444", linewidth=0.9)
    anchor_box = dict(boxstyle="round,pad=0.42", facecolor="#e8e8e8", edgecolor="#222222", linewidth=1.3)

    ax.text(0, 0, textwrap.fill(anchor, 18), ha="center", va="center", bbox=anchor_box, zorder=5)

    n = len(edges)
    for i, edge in enumerate(edges):
        angle = math.pi / 2 + 2 * math.pi * i / n
        radius_x, radius_y = (1.42, 1.02) if i % 2 == 0 else (1.20, 0.86)
        x, y = radius_x * math.cos(angle), radius_y * math.sin(angle)

        label = textwrap.fill(edge["neighbour"], 22)
        rows = label.count("\n") + 1
        ax.text(x, y, label, ha="center", va="center", fontsize=8, bbox=box, zorder=5)
        source = f"{tags[edge['document'] or 'unrecorded']}, {pages(edge['pages'])}"
        # Offset in points, so the line sits under the box whatever the axis scale is.
        ax.annotate(source, (x, y), textcoords="offset points",
                    xytext=(0, -(10 + (rows - 1) * 5.5)), ha="center", va="top",
                    fontsize=7.5, style="italic", color="#555555", zorder=5)

        # The arrow runs the way the assertion does, and stops short of both boxes.
        start, end = ((0, 0), (x, y)) if edge["outgoing"] else ((x, y), (0, 0))
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                     color="#666666", linewidth=0.9,
                                     shrinkA=42, shrinkB=42, zorder=2))
        mx, my = 0.53 * x, 0.53 * y
        ax.text(mx, my, edge["predicate"], fontsize=7.5, ha="center", va="center",
                color="#222222", zorder=4,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none"))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, draw the neighbourhood and its JSON sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="kg_subgraph")
    args = parser.parse_args(argv)

    edges = fetch(args.anchor)
    if not edges:
        parser.error(f"no edges on {args.anchor!r}")
    tags = doc_tags(edges)
    written = draw(args.anchor, edges, tags, args.out_dir, args.stem)

    sidecar = args.out_dir / f"{args.stem}.json"
    sidecar.write_text(json.dumps({
        "anchor": args.anchor,
        "graph": neo4j_env.resolve_target(require=False).uri,
        "documents": {tag: document for document, tag in tags.items()},
        "edges": edges,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    for path in written + [sidecar]:
        print(f"wrote {path}")
    print("\ndocument tags for the caption:")
    for document, tag in tags.items():
        print(f"  {tag}  {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
