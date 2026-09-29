"""Draw the campaign figures of the thesis from the scorer's own JSON.

A figure drawn by hand can drift from the table it plots with nothing to catch
it. This reads the same ``gold_score.json`` files that ``build_results_tables.py``
reads, writes the PDF and the PNG the document includes, and writes a JSON sidecar
carrying the values and the run directory each one came from.

Three figures, selected with ``--figure``:

``main_campaign``
    Concept F1 per generator and strategy, the plot of the main table.
``hard_subset``
    The same strategies twice: over every expected concept, and over the ones the
    generator misses without any context. Two panels, two scales, never one axis.
``citations``
    Citation coverage under the passage index against the passage index plus the
    graph, one row per generator.

Strategy identity is carried by marker shape, not by colour alone. The palette
puts ``hybrid`` at ``#1f77b4`` and ``no_retrieval`` at ``#d62728``, whose relative
luminances are 0.39 and 0.36: a greyscale print loses the two apart. Shape survives
the print, and it also survives an overlap, as on Qwen3-30B where two strategies sit
0.001 apart.

Usage::

    conda run -n graphllm python evaluation/scripts/plot_thesis_figures.py \\
        --figure all \\
        --campaign-root /srv/projects/graphllm/experiments/exp_results_fixed \\
        --out-dir /srv/projects/graphllm/experiments/thesis_v6/figures \\
        --hard-subset-json artifacts/evaluation/hard_subset_matrix.json \\
        --citations-json artifacts/evaluation/citation_coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_results_tables import MODELS, load, value  # noqa: E402

# ---------------------------------------------------------------------------
# House style. Serif to match the body text of the thesis; ink for type, hue for
# marks; one hairline grid on the value axis and no frame around the plot.
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
INK_SOFT = "#555555"
GRID = "#dcdcdc"
GREY = "#8c8c8c"          # the de-emphasised series: context rather than identity
BLUE = "#1f77b4"          # hybrid
ORANGE = "#ff7f0e"        # text_only
RED = "#d62728"           # no_retrieval

# The three strategies the figures name carry a shape each; the rest of the graph
# family shares one grey diamond.
HIGHLIGHT = {
    "hybrid": ("hybrid", "o", BLUE),
    "text_only": ("text_only", "s", ORANGE),
    "no_retrieval": ("no_retrieval", "^", RED),
}
GRAPH_ONLY = ["default", "subgraph_2hop", "shortest_path", "neighbors_focus", "text_plus_triples"]
# The runs call the flat-lookup preset ``text_plus_triples``, where "text" means
# node text rather than passages; the thesis names it for what it retrieves.
THESIS_NAME = {s: s for s in GRAPH_ONLY + list(HIGHLIGHT)}
THESIS_NAME["text_plus_triples"] = "nodes_plus_triples"


def style(plt) -> None:
    """Apply the house style to matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def bare(ax, grid_axis: str = "x") -> None:
    """Hairline grid on the value axis, no frame, no tick marks."""
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)


def marker(ax, x, y, spec, size=10.0, zorder=5):
    """One data point, with a surface ring so overlaps stay readable."""
    _, shape, colour = spec
    return ax.plot(x, y, marker=shape, markersize=size, color=colour, linestyle="none",
                   markeredgecolor="white", markeredgewidth=1.1, zorder=zorder)


def save(fig, out_dir: Path, stem: str) -> list[Path]:
    """Save `fig` as PDF and PNG under `out_dir`; returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Figure: the main campaign
# ---------------------------------------------------------------------------
def collect_campaign(reports: dict[str, dict]) -> dict[str, dict[str, float]]:
    """slug -> strategy -> concept-level micro F1 on the answer channel."""
    scores: dict[str, dict[str, float]] = {}
    for slug, _ in MODELS:
        if slug not in reports:
            continue
        row = {}
        for strategy in GRAPH_ONLY + list(HIGHLIGHT):
            f1 = value(reports[slug], "answer", strategy, "concept_micro", "f1")
            if f1 is not None:
                row[strategy] = f1
        scores[slug] = row
    return scores


def separate(points: list[tuple[str, float]], gap: float = 0.004,
             offset: float = 0.17) -> dict[str, float]:
    """Vertical offsets that keep two near-equal markers on a row both visible.

    Hybrid and the closed-book control sit 0.001 apart on Qwen3-30B, close enough
    that the later marker hides the earlier one. The row axis carries no value,
    so a small offset costs nothing and shows both.
    """
    placed: dict[str, float] = {}
    taken: list[float] = []
    for name, x in sorted(points, key=lambda item: item[1]):
        collides = any(abs(x - other) < gap for other in taken)
        placed[name] = offset if collides and len(taken) % 2 == 1 else (-offset if collides else 0.0)
        taken.append(x)
    return placed


def draw_campaign(scores: dict[str, dict[str, float]], out_dir: Path, stem: str) -> list[Path]:
    """Concept F1 per generator, one row each, named strategies highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    present = [(slug, name) for slug, name in MODELS if slug in scores]
    fig, ax = plt.subplots(figsize=(9.4, 4.5))

    for i, (slug, _) in enumerate(present):
        row = scores[slug]
        y = len(present) - 1 - i
        ax.plot([min(row.values()), max(row.values())], [y, y],
                color=GREY, lw=1.6, alpha=0.55, zorder=1, solid_capstyle="round")
        dodge = separate([(s, row[s]) for s in HIGHLIGHT if s in row])
        for strategy in GRAPH_ONLY:
            if strategy in row:
                marker(ax, row[strategy], y, ("", "D", GREY), size=7.0, zorder=2)
        for z, strategy in enumerate(reversed(list(HIGHLIGHT)), start=3):
            if strategy in row:
                marker(ax, row[strategy], y + dodge.get(strategy, 0.0), HIGHLIGHT[strategy],
                       zorder=z)

    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([name for _, name in reversed(present)])
    ax.set_ylim(-0.6, len(present) - 0.4)
    ax.set_xlabel("concept F1")
    bare(ax)

    handles = [plt.Line2D([], [], marker=shape, color=colour, linestyle="none",
                          markersize=10.0, markeredgecolor="white", label=label)
               for label, shape, colour in HIGHLIGHT.values()]
    handles.append(plt.Line2D([], [], marker="D", color=GREY, linestyle="none",
                              markersize=7.0, markeredgecolor="white",
                              label=f"{len(GRAPH_ONLY)} graph-only strategies"))
    legend = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                       ncol=len(handles), frameon=False, handletextpad=0.3, columnspacing=1.8)
    for text in legend.get_texts()[:len(HIGHLIGHT)]:
        text.set_family("monospace")

    fig.tight_layout()
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Figure: the hard subset
# ---------------------------------------------------------------------------
def draw_hard_subset(data: dict, campaign: dict[str, dict[str, float]],
                     out_dir: Path, stem: str) -> tuple[list[Path], dict]:
    """Two panels: every expected concept, then only the ones retrieval must supply.

    The panels measure different things over different denominators, so they get an
    axis each rather than one shared scale. What the figure is for is the width of
    each panel's spread, and that survives the split.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    strategies = ["hybrid", "text_only", "default", "subgraph_2hop",
                  "neighbors_focus", "text_plus_triples", "shortest_path"]
    aggregate = {s: sum(campaign[m][s] for m in campaign) / len(campaign) for s in strategies}
    pooled_hard = sum(v["hard"] for v in data.values())
    hard = {s: sum(v["hits_hard"].get(s, 0) for v in data.values()) / pooled_hard
            for s in strategies}
    closed_book = sum(campaign[m]["no_retrieval"] for m in campaign) / len(campaign)

    order = sorted(strategies, key=lambda s: aggregate[s])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)

    # The panel labels stay short: the caption carries what each one measures.
    panels = [
        (axes[0], aggregate, "concept F1, all 88 concepts", 0.02),
        (axes[1], hard, f"recall, the {pooled_hard} hard concepts", 0.05),
    ]
    for ax, values, label, pad in panels:
        lo, hi = min(values.values()), max(values.values())
        for i, strategy in enumerate(order):
            spec = HIGHLIGHT.get(strategy, ("", "D", GREY))
            size = 10.0 if strategy in HIGHLIGHT else 7.5
            ax.plot([lo - pad / 2, values[strategy]], [i, i], color=GRID, lw=1.0, zorder=1)
            marker(ax, values[strategy], i, spec, size=size)
            # Only the ends of the panel carry a printed value; the axis carries the rest.
            if i in (0, len(order) - 1):
                ax.annotate(f"{values[strategy]:.3f}", (values[strategy], i),
                            textcoords="offset points", xytext=(11, 0), va="center",
                            fontsize=10.5, color=INK)
        # The spread is the point of the figure, so it gets drawn rather than described.
        y = len(order) - 0.55
        ax.annotate("", (lo, y), (hi, y),
                    arrowprops=dict(arrowstyle="<->", color=INK_SOFT, lw=0.9))
        ax.annotate(f"spread {hi - lo:.3f}", ((lo + hi) / 2, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=10.5, color=INK)
        ax.set_xlim(lo - pad, hi + pad * 2.8)
        ax.set_xlabel(label, fontsize=11)
        bare(ax)

    axes[0].axvline(closed_book, color=RED, lw=0.9, alpha=0.6, zorder=1)
    axes[0].annotate(f"closed-book {closed_book:.3f}", (closed_book, -0.75),
                     textcoords="offset points", xytext=(6, 0), ha="left", va="center",
                     fontsize=10, color=RED)
    axes[0].set_yticks(range(len(order)))
    # Strategy names are identifiers, and the thesis sets identifiers in typewriter.
    axes[0].set_yticklabels([THESIS_NAME[s] for s in order], family="monospace", fontsize=11)
    axes[0].set_ylim(-1.0, len(order) - 0.05)

    fig.tight_layout(w_pad=3.2)
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return written, {"aggregate": aggregate, "hard_recall": hard,
                     "pooled_hard_slots": pooled_hard, "closed_book_mean": closed_book}


# ---------------------------------------------------------------------------
# Figure: citation coverage
# ---------------------------------------------------------------------------
def draw_citations(coverage: dict[str, dict[str, float]], out_dir: Path,
                   stem: str) -> tuple[list[Path], dict]:
    """One row per generator: coverage under text retrieval, then with the graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    present = [(slug, name) for slug, name in MODELS if slug in coverage]
    pooled = {s: sum(coverage[slug][s] for slug, _ in present) / len(present)
              for s in ("text_only", "hybrid")}

    rows = [(name, coverage[slug]) for slug, name in present]
    rows.append(("all six pooled", pooled))

    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    positions: list[tuple[float, str]] = []
    for i, (name, row) in enumerate(rows):
        y = len(rows) - 1 - i - (0.4 if name == "all six pooled" else 0.0)
        ax.annotate("", (row["hybrid"], y), (row["text_only"], y),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5,
                                    shrinkA=7, shrinkB=9), zorder=2)
        marker(ax, row["text_only"], y, HIGHLIGHT["text_only"])
        marker(ax, row["hybrid"], y, HIGHLIGHT["hybrid"])
        ax.annotate(f"+{row['hybrid'] - row['text_only']:.3f}",
                    ((row["text_only"] + row["hybrid"]) / 2, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10.5, color=INK_SOFT)
        positions.append((y, name))

    ax.axhline(0.3, color=GRID, lw=0.8, zorder=1)
    ax.set_yticks([y for y, _ in positions])
    ax.set_yticklabels([name for _, name in positions], fontsize=12)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(-1.05, len(rows) - 0.45)
    ax.set_xlim(0.355, 1.035)
    ax.set_xlabel("share of answers carrying at least one citation")
    bare(ax)

    handles = [plt.Line2D([], [], marker=HIGHLIGHT[s][1], color=HIGHLIGHT[s][2],
                          linestyle="none", markersize=10.0, markeredgecolor="white",
                          label=HIGHLIGHT[s][0]) for s in ("text_only", "hybrid")]
    legend = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                       ncol=2, frameon=False, handletextpad=0.3, columnspacing=2.0)
    for text in legend.get_texts():
        text.set_family("monospace")

    fig.tight_layout()
    written = save(fig, out_dir, stem)
    plt.close(fig)
    return written, {"per_generator": coverage, "pooled": pooled}


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse arguments and draw the selected figures, each with a JSON sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", default="all",
                        choices=["all", "main_campaign", "hard_subset", "citations"])
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hard-subset-json", type=Path,
                        help="{slug: {hard, hits_hard, ...}}, from collect_hard_subset.py")
    parser.add_argument("--citations-json", type=Path,
                        help="{slug: {strategy: coverage}}, from collect_citation_coverage.py")
    args = parser.parse_args(argv)

    reports = load(args.campaign_root)
    if not reports:
        parser.error(f"no gold_score.json under {args.campaign_root}")
    missing = [slug for slug, _ in MODELS if slug not in reports]
    if missing:
        print(f"WARNING: no score for {', '.join(missing)}; rows omitted")
    campaign = collect_campaign(reports)
    provenance = {
        "campaign_root": str(args.campaign_root),
        "runs": {slug: reports[slug]["run_dir"] for slug in campaign},
        "gold_sha256": {slug: reports[slug].get("gold_sha256") for slug in campaign},
    }
    written: list[Path] = []

    if args.figure in ("all", "main_campaign"):
        written += draw_campaign(campaign, args.out_dir, "main_campaign")
        path = args.out_dir / "main_campaign.json"
        path.write_text(json.dumps({**provenance, "channel": "answer",
                                    "level": "concept_micro", "field": "f1",
                                    "scores": campaign}, indent=1), encoding="utf-8")
        written.append(path)

    if args.figure in ("all", "hard_subset"):
        if not args.hard_subset_json:
            parser.error("--hard-subset-json is required for the hard-subset figure")
        data = json.loads(args.hard_subset_json.read_text())
        paths, summary = draw_hard_subset(data, campaign, args.out_dir, "hard_subset")
        written += paths
        path = args.out_dir / "hard_subset.json"
        path.write_text(json.dumps({**provenance, **summary, "per_generator": data}, indent=1),
                        encoding="utf-8")
        written.append(path)

    if args.figure in ("all", "citations"):
        if not args.citations_json:
            parser.error("--citations-json is required for the citation figure")
        coverage = json.loads(args.citations_json.read_text())
        paths, summary = draw_citations(coverage, args.out_dir, "citation_coverage")
        written += paths
        path = args.out_dir / "citation_coverage.json"
        path.write_text(json.dumps({**provenance, **summary}, indent=1), encoding="utf-8")
        written.append(path)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
