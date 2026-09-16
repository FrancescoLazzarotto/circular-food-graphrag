#!/usr/bin/env python3
"""
KG Evaluator for Neo4j Aura
============================
Run with:   python kg_evaluator.py
Requires:   pip install neo4j rich python-dotenv

Output: artifacts/kg_reports/kg_report_10.json  (upload to Claude for evaluation)
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from kg_pipeline.utils import neo4j_env
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv


# ── Load .env from pipeline (kg_pipeline/.env) if present
ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "kg_pipeline" / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


# ── Configuration (loaded from environment after load_dotenv)
# The defaults here used to be the literal placeholders
# `neo4j+s://<id>.databases.neo4j.io` and `<password>`, so an unconfigured run
# spent its timeout dialling a hostname that does not exist. Missing settings
# are now missing settings.
_TARGET = neo4j_env.resolve_target(require=False)
NEO4J_URI = _TARGET.uri
NEO4J_USER = _TARGET.user
NEO4J_PASSWORD = _TARGET.password
NEO4J_DATABASE = _TARGET.database or ""
SAMPLE_LIMIT = int(os.getenv("KG_EVALUATOR_SAMPLE_LIMIT", "2000"))
RELTYPE_ENDPOINT_MAX_TYPES = int(os.getenv("KG_EVALUATOR_RELTYPE_MAX_TYPES", "250"))

console = Console()


def run(session, query, **params):
    return session.run(query, **params).data()


# ── 1. Basic metrics ──────────────────────────────────────────────────────

def basic_counts(session):
    n = run(session, "MATCH (n) RETURN count(n) AS c")[0]["c"]
    r = run(session, "MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    labels = run(session, "CALL db.labels() YIELD label RETURN collect(label) AS l")[0]["l"]
    reltypes = run(session, "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS t")[0]["t"]
    return {
        "node_count": n,
        "edge_count": r,
        "label_count": len(labels),
        "reltype_count": len(reltypes),
        "labels": labels,
        "relationship_types": reltypes,
    }


# ── 2. Label distribution ───────────────────────────────────────────────

def label_distribution(session, labels):
    dist = {}
    for lbl in labels:
        c = run(session, f"MATCH (n:`{lbl}`) RETURN count(n) AS c")[0]["c"]
        dist[lbl] = c
    return dist


# ── 3. Relationship type distribution ───────────────────────────────────

def reltype_distribution(session, reltypes):
    dist = {}
    for rt in reltypes:
        c = run(session, f"MATCH ()-[r:`{rt}`]->() RETURN count(r) AS c")[0]["c"]
        dist[rt] = c
    return dist


# ── 4. Graph density ─────────────────────────────────────────────────────

def graph_density(n, e):
    """Direct density: e / (n*(n-1))"""
    if n < 2:
        return 0.0
    return round(e / (n * (n - 1)), 6)


# ── 5. Degree distribution ───────────────────────────────────────────────

def degree_stats(session, limit=SAMPLE_LIMIT):
    # Aura/modern Neo4j disallow size((n)--()) inside functions; use COUNT aggregation per node instead
    rows = run(session,
        f"MATCH (n) WITH n LIMIT {limit} "
        f"OPTIONAL MATCH (n)-[r]-() "
        f"WITH n, count(r) AS degree "
        f"RETURN degree")
    degrees = [r["degree"] for r in rows]
    if not degrees:
        return {}
    degrees.sort()
    n = len(degrees)
    mean = sum(degrees) / n
    median = degrees[n // 2]
    variance = sum((d - mean) ** 2 for d in degrees) / n
    std = math.sqrt(variance)
    non_zero = [d for d in degrees if d > 0]
    alpha = None
    if non_zero:
        alpha = round(1 + len(non_zero) / sum(math.log(d) for d in non_zero), 3)
    return {
        "sample_size": n,
        "min": degrees[0],
        "max": degrees[-1],
        "mean": round(mean, 2),
        "median": median,
        "std": round(std, 2),
        "power_law_alpha_estimate": alpha,
        "degree_distribution_top10": dict(Counter(degrees).most_common(10)),
    }


# ── 6. Isolated (orphan) nodes ─────────────────────────────────────────────────

def isolated_nodes(session):
    c = run(session, "MATCH (n) WHERE NOT (n)--() RETURN count(n) AS c")[0]["c"]
    return c


# ── 7. Hub nodes (top-k by degree) ────────────────────────────────────────────

def hub_nodes(session, k=10):
    # Use OPTIONAL MATCH + COUNT to compute degree per node (compatible with Aura)
    rows = run(session,
        f"MATCH (n) "
        f"OPTIONAL MATCH (n)-[r]-() "
        f"WITH n, count(r) AS degree "
        f"ORDER BY degree DESC LIMIT {k} "
        f"RETURN labels(n) AS labels, properties(n) AS props, degree")
    return [{"labels": r["labels"], "degree": r["degree"],
             "id_hint": _id_hint(r["props"])} for r in rows]


def _id_hint(props):
    for key in ("name", "title", "id", "uri", "label", "identifier"):
        if key in props:
            v = props[key]
            return str(v)[:80] if v else None
    if props:
        first_key = next(iter(props))
        return f"{first_key}={str(props[first_key])[:60]}"
    return None


# ── 8. Property coverage ─────────────────────────────────────────────

def property_coverage(session, labels, limit=SAMPLE_LIMIT):
    coverage = {}
    for lbl in labels:
        rows = run(session,
            f"MATCH (n:`{lbl}`) WITH n LIMIT {limit} "
            "RETURN keys(n) AS ks, count(n) AS c")
        total = sum(r["c"] for r in rows)
        if total == 0:
            coverage[lbl] = {}
            continue
        prop_counts = Counter()
        for r in rows:
            for k in r["ks"]:
                prop_counts[k] += r["c"]
        coverage[lbl] = {
            "total_sampled": total,
            "properties": {k: round(v / total, 3) for k, v in prop_counts.items()},
        }
    return coverage


# ── 9. Connected components ────────────────────────────────────────────────────

def connected_components(session):
    """Approximation via WCC (requires GDS or manual fallback)."""
    try:
        rows = run(session,
            "CALL gds.wcc.stats({nodeQuery: 'MATCH (n) RETURN id(n) AS id', "
            "relationshipQuery: 'MATCH (a)--(b) RETURN id(a) AS source, id(b) AS target'}) "
            "YIELD componentCount, componentDistribution "
            "RETURN componentCount, componentDistribution")
        if rows:
            r = rows[0]
            return {
                "method": "GDS WCC",
                "component_count": r["componentCount"],
                "distribution": r.get("componentDistribution", {}),
            }
    except Exception:
        pass
    try:
        # Approximate the largest component by starting from the highest-degree node
        # to avoid an expensive expansion from thousands of seeds.
        rows = run(session,
            "MATCH (n) "
            "OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS degree "
            "ORDER BY degree DESC LIMIT 1 "
            "CALL apoc.path.subgraphNodes(n, {maxLevel: 99}) YIELD node "
            "RETURN count(DISTINCT node) AS size")
        if rows:
            return {"method": "APOC (partial)", "largest_component_approx": rows[0]["size"]}
    except Exception:
        pass
    return {"method": "unavailable (no GDS/APOC)", "note": "install GDS or APOC on Aura for this metric"}


# ── 10. Most frequent triple patterns ───────────────────────────────────────

def triple_patterns(session, limit=20):
    rows = run(session,
        f"MATCH (a)-[r]->(b) "
        "WITH labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS tgt "
        f"RETURN src, rel, tgt, count(*) AS freq "
        f"ORDER BY freq DESC LIMIT {limit}")
    return [{"pattern": f"({r['src']})-[{r['rel']}]->({r['tgt']})", "count": r["freq"]}
            for r in rows]


# ── 11. Node-to-edge ratio (graph sparsity) ──────────────────────────────────

def sparsity_class(n, e):
    if n == 0:
        return "empty"
    ratio = e / n
    if ratio < 1:
        return "very sparse (<1)"
    elif ratio < 3:
        return "sparse (1-3)"
    elif ratio < 10:
        return "moderate (3-10)"
    else:
        return "dense (>10)"


# ── 12. Relationship type endpoint consistency ─────────────────────────────────

def reltype_endpoint_consistency(session, reltypes, limit=500):
    consistency = {}
    selected = reltypes[:RELTYPE_ENDPOINT_MAX_TYPES]
    for rt in selected:
        rows = run(session,
            f"MATCH (a)-[r:`{rt}`]->(b) WITH a, b LIMIT {limit} "
            "RETURN labels(a)[0] AS src, labels(b)[0] AS tgt, count(*) AS c")
        patterns = {f"({r['src']})->({r['tgt']})": r["c"] for r in rows}
        consistency[rt] = {
            "distinct_patterns": len(patterns),
            "patterns": patterns,
        }

    if len(reltypes) > len(selected):
        consistency["__truncated__"] = {
            "evaluated_types": len(selected),
            "total_types": len(reltypes),
            "limit_per_type": limit,
        }

    return consistency


# ── 13. Aggregate quality metrics ───────────────────────────────────────

def quality_scores(report):
    n = report["basic"]["node_count"]
    e = report["basic"]["edge_count"]
    isolated = report["isolated_node_count"]
    deg = report.get("degree_stats", {})

    structural = round(1 - (isolated / n if n else 0), 3)
    connectivity = round(min((e / n) / 5, 1.0), 3) if n else 0
    richness = round(
        min(report["basic"]["label_count"] / 10, 1.0) * 0.5 +
        min(report["basic"]["reltype_count"] / 20, 1.0) * 0.5, 3)

    cons_scores = []
    for rt, data in report.get("reltype_endpoint_consistency", {}).items():
        if not isinstance(data, dict) or "distinct_patterns" not in data:
            continue
        p = data["distinct_patterns"]
        cons_scores.append(1 / p if p > 0 else 0)
    consistency = round(sum(cons_scores) / len(cons_scores), 3) if cons_scores else None

    return {
        "structural_completeness": structural,
        "connectivity": connectivity,
        "semantic_richness": richness,
        "reltype_consistency": consistency,
        "note": "Scores 0-1. Higher values = better quality.",
    }


def evaluate(uri, user, password, database: str = ""):
    driver = neo4j_env.connect(
        neo4j_env.Neo4jTarget(uri, user, password, database or None)
    )
    report = {}

    steps = [
        ("Basic counts",              "basic"),
        ("Label distribution",        "label_distribution"),
        ("Rel. type distribution",    "reltype_distribution"),
        ("Degree statistics",         "degree_stats"),
        ("Isolated nodes",            "isolated_node_count"),
        ("Hub nodes",                 "hub_nodes"),
        ("Property coverage",         "property_coverage"),
        ("Connected components",      "connected_components"),
        ("Triple patterns",           "triple_patterns"),
        ("Rel. type consistency",     "reltype_endpoint_consistency"),
    ]

    session_kwargs = {}
    if database:
        session_kwargs["database"] = database

    with driver.session(**session_kwargs) as session:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:

            task = progress.add_task("Starting...", total=len(steps))

            progress.update(task, description=steps[0][0])
            basic = basic_counts(session)
            report["basic"] = basic
            progress.advance(task)

            progress.update(task, description=steps[1][0])
            report["label_distribution"] = label_distribution(session, basic["labels"])
            progress.advance(task)

            progress.update(task, description=steps[2][0])
            report["reltype_distribution"] = reltype_distribution(session, basic["relationship_types"])
            progress.advance(task)

            progress.update(task, description=steps[3][0])
            report["degree_stats"] = degree_stats(session)
            progress.advance(task)

            progress.update(task, description=steps[4][0])
            report["isolated_node_count"] = isolated_nodes(session)
            progress.advance(task)

            progress.update(task, description=steps[5][0])
            report["hub_nodes"] = hub_nodes(session)
            progress.advance(task)

            progress.update(task, description=steps[6][0])
            report["property_coverage"] = property_coverage(session, basic["labels"])
            progress.advance(task)

            progress.update(task, description=steps[7][0])
            report["connected_components"] = connected_components(session)
            progress.advance(task)

            progress.update(task, description=steps[8][0])
            report["triple_patterns"] = triple_patterns(session)
            progress.advance(task)

            progress.update(task, description=steps[9][0])
            report["reltype_endpoint_consistency"] = reltype_endpoint_consistency(
                session, basic["relationship_types"])
            progress.advance(task)

    driver.close()

    n, e = basic["node_count"], basic["edge_count"]
    report["graph_density"] = graph_density(n, e)
    report["sparsity_class"] = sparsity_class(n, e)
    report["edge_node_ratio"] = round(e / n, 3) if n else 0
    report["quality_scores"] = quality_scores(report)

    return report


if __name__ == "__main__":
    console.rule("[bold]KG Evaluator — Neo4j Aura[/bold]")
    console.print(f"URI: [cyan]{NEO4J_URI}[/cyan]")
    console.print(f"Database: [cyan]{NEO4J_DATABASE or '<default>'}[/cyan]")

    report = evaluate(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DATABASE)

    out_dir = Path("artifacts") / "kg_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_override = os.getenv("KG_EVALUATOR_OUT", "").strip()
    if out_override:
        out_path = Path(out_override)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"kg_report_{timestamp}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    console.rule("[green]Report saved[/green]")
    console.print(f"File: [bold]{out_path}[/bold]")
    console.print()
    console.print(
        "[yellow]→ Upload the report file to Claude for complete evaluation.[/yellow]"
    )

    b = report["basic"]
    qs = report["quality_scores"]
    console.print()
    console.print(f"  Nodes: {b['node_count']:,}  |  Edges: {b['edge_count']:,}  |  "
                  f"Labels: {b['label_count']}  |  Rel. types: {b['reltype_count']}")
    console.print(f"  Structural completeness : {qs['structural_completeness']}")
    console.print(f"  Connectivity            : {qs['connectivity']}")
    console.print(f"  Semantic richness       : {qs['semantic_richness']}")
    console.print(f"  Rel. type consistency   : {qs['reltype_consistency']}")

