"""What one hop and two hops actually reach on the knowledge graph.

Four of the eight strategies expand from an anchor by traversal, and the results
chapter explains their failure by the shape of the graph. That explanation rests on
one measurement: how many distinct nodes a walk of one hop, and of two, reaches from
an ordinary node. This computes it for every domain node, over the graph the
campaigns query, and draws the two distributions against each other.

The measure ignores the retrieval strategy's own limits: it is the reach of the
graph, not the reach of a configured channel, so the caps of the agent
configuration sit on top of what this shows rather than inside it.

Usage::

    NEO4J_URL=bolt://localhost:7689 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=... \\
    conda run -n graphllm python evaluation/scripts/plot_hop_reach.py \\
        --out-dir /srv/projects/graphllm/experiments/thesis_v6/figures
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from kg_pipeline.utils import neo4j_env

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_thesis_figures import BLUE, GREY, GRID, INK, INK_SOFT, bare, save, style  # noqa: E402

# Carrier nodes hold the embeddings and are not part of the domain graph.
NODES = "MATCH (n) WHERE NOT n:NodeVec RETURN elementId(n) AS id"
EDGES = ("MATCH (a)-[e]->(b) WHERE NOT a:NodeVec AND NOT b:NodeVec "
         "RETURN elementId(a) AS a, elementId(b) AS b")


def reach() -> tuple[list[int], list[int]]:
    """Distinct nodes within one hop and within two hops, for every domain node."""
    driver = neo4j_env.connect()
    try:
        with driver.session() as session:
            ids = [record["id"] for record in session.run(NODES)]
            edges = [(record["a"], record["b"]) for record in session.run(EDGES)]
    finally:
        driver.close()

    # Traversal in the retriever ignores direction, so the reach measured here does too.
    adjacency: dict[str, set[str]] = collections.defaultdict(set)
    for a, b in edges:
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)

    one_hop, two_hop = [], []
    for node in ids:
        near = adjacency[node]
        far = set().union(*(adjacency[n] for n in near)) if near else set()
        one_hop.append(len(near))
        two_hop.append(len((far | near) - {node}))
    return one_hop, two_hop


def ecdf(values: list[int]) -> tuple[list[float], list[float]]:
    """Empirical CDF points of `values`, zero moved onto the log axis."""
    counts = collections.Counter(values)
    total = len(values)
    xs, ys, seen = [], [], 0
    for value in sorted(counts):
        seen += counts[value]
        xs.append(max(value, 0.85))   # the log axis has no room for zero
        ys.append(seen / total)
    return xs, ys


def draw(one_hop: list[int], two_hop: list[int], out_dir: Path, stem: str,
         cap: int) -> list[Path]:
    """Plot both reach distributions as ECDFs, with the evidence cap marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    series = [
        ("one hop", one_hop, GREY, (0, (5, 2)), 1.7),
        ("two hops", two_hop, BLUE, "solid", 2.0),
    ]
    for label, values, colour, dashes, width in series:
        xs, ys = ecdf(values)
        ax.step(xs, ys, where="post", color=colour, lw=width, linestyle=dashes,
                solid_capstyle="round", zorder=4, label=label)
        median = statistics.median(values)
        ax.plot(max(median, 0.85), sum(1 for v in values if v <= median) / len(values),
                marker="o", markersize=9.5, color=colour, markeredgecolor="white",
                markeredgewidth=1.2, zorder=6)
        ax.annotate(f"median {median:.0f}", (max(median, 0.85), 0.5),
                    textcoords="offset points", xytext=(10, -16), fontsize=11, color=INK)

    above = sum(1 for v in two_hop if v > cap) / len(two_hop)
    ax.axvline(cap, color=INK_SOFT, lw=0.9, alpha=0.6, zorder=2)
    ax.annotate(f"{cap}-item evidence cap\n{above:.1%} of nodes reach past it at two hops",
                (cap, 0.16), textcoords="offset points", xytext=(-9, 0), ha="right",
                va="center", fontsize=10.5, color=INK_SOFT)

    ax.set_xscale("log")
    ax.set_xlim(0.8, 4200)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 2, 5, 10, 25, 100, 200, 1000, 3000])
    ax.set_xticklabels(["1", "2", "5", "10", "25", "100", "200", "1,000", "3,000"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("distinct nodes reached from one node")
    ax.set_ylabel("share of the 14,520 domain nodes")
    bare(ax, "y")

    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                       frameon=False, handlelength=2.6, columnspacing=2.4)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, measure reach, draw the figure and its JSON sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="hop_reach")
    parser.add_argument("--cap", type=int, default=200,
                        help="the subgraph evidence cap of the reported configuration")
    args = parser.parse_args(argv)

    one_hop, two_hop = reach()
    written = draw(one_hop, two_hop, args.out_dir, args.stem, args.cap)

    def summary(values: list[int]) -> dict[str, float]:
        """Mean, quartiles, 90th percentile and maximum of `values`."""
        ordered = sorted(values)
        n = len(ordered)
        return {
            "mean": sum(values) / n,
            "q25": ordered[n // 4], "median": statistics.median(values),
            "q75": ordered[3 * n // 4], "p90": ordered[int(0.90 * n)],
            "max": ordered[-1],
        }

    sidecar = args.out_dir / f"{args.stem}.json"
    sidecar.write_text(json.dumps({
        "graph": neo4j_env.resolve_target(require=False).uri,
        "nodes": len(one_hop),
        "one_hop": summary(one_hop),
        "two_hops": summary(two_hop),
        "isolated": sum(1 for v in one_hop if v == 0),
        "one_neighbour": sum(1 for v in one_hop if v == 1),
        "second_hop_adds_nothing": sum(1 for a, b in zip(one_hop, two_hop) if a == b),
        "above_cap": sum(1 for v in two_hop if v > args.cap),
        "cap": args.cap,
    }, indent=1), encoding="utf-8")

    for path in written + [sidecar]:
        print(f"wrote {path}")
    print(json.loads(sidecar.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
