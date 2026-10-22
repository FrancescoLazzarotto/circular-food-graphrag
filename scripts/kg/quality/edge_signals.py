#!/usr/bin/env python3
"""One row per edge with the schema signals that cost nothing to compute.

Reads a nightly JSON backup of the graph (never the live database) and writes,
for every edge, what the schema says about it:

- ``type`` and, for the fallback ``RELATED_TO``, the original ``predicate``
  when the pipeline kept it;
- ``in_vocab``: the type is in the extraction vocabulary;
- the labels of its two ends and whether either is the default ``Concept``;
- ``class_name_object``: the object node is named like a schema label
  ("Person", "Organization"), a class written as if it were an entity;
- ``violations``: domain and range rules the edge breaks, separated by ``;``;
- ``family``: a coarse group of relation types, for strata;
- the passage (``chunk_id``), document and pages, and the number of edges
  extracted from the same passage;
- the extractor's own ``confidence`` and ``mention_count``, as controls.

``edge_key`` is stable across restores (Neo4j ids are not): a hash of
document, passage, subject, type and object.

The domain and range rules are those the extraction prompt states
(``kg_pipeline/prompts/extraction_prompt.py``). ``--rules`` adds more from a
TSV with columns ``rule``, ``type``, ``subject_labels``, ``object_labels``,
``source`` (labels separated by ``|``, empty for any).

Usage:
    python scripts/kg/quality/edge_signals.py --out artifacts/kg_eval_2026-10/signals/<backup>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKUP_ROOT = Path(os.environ.get("ONTOLOGY_BACKUP_ROOT", "/mnt/storage/flazzarotto/graph_backups/daily"))
VOCAB = ROOT / "kg_pipeline" / "relation_vocab_circular_v1_draft.json"
FALLBACK = "RELATED_TO"
DEFAULT_LABEL = "Concept"

COLUMNS = [
    "edge_key", "backup_id", "chunk_id", "source_doc", "page_range", "chunk_edges",
    "type", "predicate", "family", "in_vocab", "fallback", "fallback_predicate",
    "subject", "subject_label", "object", "object_label", "concept_end",
    "class_name_object", "violations", "confidence", "mention_count",
]

# Coarse groups of the extraction vocabulary, so that a stratum is not one
# rare type. A type outside every group falls in "other".
FAMILIES = {
    "fallback": {"RELATED_TO"},
    "structure": {"PART_OF", "HAS_COMPONENT", "INCLUDES", "IS_TYPE_OF", "HAS_MEMBER", "LOCATED_IN"},
    "definition": {"DEFINED_AS", "BASED_ON"},
    "quantity": {"HAS_VALUE", "MEASURES"},
    "material_flow": {"PRODUCES", "USES", "REQUIRES", "GENERATES", "REDUCES", "REUSES", "RECYCLES"},
    "effect": {"AFFECTS", "CAUSES", "CONTRIBUTES_TO", "HAS_RESULT", "HAS_GOAL"},
    "action": {"PROMOTES", "SUPPORTS", "IMPLEMENTS", "INVOLVES", "COLLABORATES_WITH", "ORGANIZES",
               "ANALYZES", "APPLIES_TO", "REGULATED_BY", "FUNDED_BY"},
    "bibliographic": {"PUBLISHED", "AUTHORED_BY"},
}


@dataclass(frozen=True)
class Rule:
    """A domain and range constraint on one edge type.

    Attributes:
        rule: Identifier written in the ``violations`` column.
        type: Edge type it applies to.
        subject_labels: Labels the subject may have; empty means any.
        object_labels: Labels the object may have; empty means any.
        source: Where the constraint is stated.
    """

    rule: str
    type: str
    subject_labels: frozenset[str]
    object_labels: frozenset[str]
    source: str

    def broken_by(self, edge_type: str, subject_label: str, object_label: str) -> bool:
        """Whether an edge of these labels breaks the rule."""
        if edge_type != self.type:
            return False
        return bool(
            (self.subject_labels and subject_label not in self.subject_labels)
            or (self.object_labels and object_label not in self.object_labels)
        )


PROMPT_RULES = [
    Rule("published_org_doc", "PUBLISHED", frozenset({"Organization"}), frozenset({"Document"}),
         "extraction_prompt.py: Organization PUBLISHED Document"),
    Rule("authored_doc_person", "AUTHORED_BY", frozenset({"Document"}), frozenset({"Person"}),
         "extraction_prompt.py: Document AUTHORED_BY Person"),
    Rule("has_value_datavalue", "HAS_VALUE", frozenset(), frozenset({"DataValue"}),
         "extraction_prompt.py: object_labels=[\"DataValue\"], predicate=\"HAS_VALUE\""),
]


def latest_backup(root: Path = BACKUP_ROOT) -> Path:
    """The newest backup folder under ``root`` that holds ``edges.json``.

    Args:
        root: Folder of the nightly backups.

    Returns:
        Path of the backup.

    Raises:
        FileNotFoundError: No backup there.
    """
    found = sorted(p for p in root.iterdir() if (p / "edges.json").is_file())
    if not found:
        raise FileNotFoundError(f"no edges.json under {root}")
    return found[-1]


def read_rules(path: Path) -> list[Rule]:
    """Rules from a TSV with ``rule``, ``type``, ``subject_labels``, ``object_labels``, ``source``.

    Args:
        path: The TSV.

    Returns:
        The rules.
    """
    def labels(cell: str) -> frozenset[str]:
        return frozenset(x.strip() for x in (cell or "").split("|") if x.strip())

    with path.open(encoding="utf-8", newline="") as handle:
        return [
            Rule(row["rule"], row["type"], labels(row["subject_labels"]), labels(row["object_labels"]), row.get("source", ""))
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def edge_key(props: dict, edge_type: str) -> str:
    """Identifier of an edge that survives a restore.

    Args:
        props: Edge properties from the backup.
        edge_type: Edge type.

    Returns:
        The first 16 hex digits of a SHA-256.
    """
    parts = [props.get("source_doc", ""), props.get("chunk_id", ""), props.get("subject", ""), edge_type, props.get("object", "")]
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def signals(nodes: list[dict], edges: list[dict], vocab: set[str], rules: list[Rule]) -> list[dict]:
    """Compute the signal row of every edge.

    Args:
        nodes: ``nodes.json`` of a backup.
        edges: ``edges.json`` of the same backup.
        vocab: Edge types of the extraction vocabulary.
        rules: Domain and range rules.

    Returns:
        One dict per edge, keys as in ``COLUMNS``.
    """
    by_id = {n["id"]: n for n in nodes}
    class_names = {label.lower() for n in nodes for label in n["labels"]}
    per_chunk = Counter(e["props"].get("chunk_id", "") for e in edges)
    family_of = {t: f for f, types in FAMILIES.items() for t in types}
    rows = []
    for e in edges:
        props, edge_type = e["props"], e["type"]
        subject, obj = by_id[e["src"]], by_id[e["dst"]]
        s_label, o_label = subject["labels"][0], obj["labels"][0]
        ends = [side for side, label in (("subject", s_label), ("object", o_label)) if label == DEFAULT_LABEL]
        rows.append(
            {
                "edge_key": edge_key(props, edge_type),
                "backup_id": e["id"],
                "chunk_id": props.get("chunk_id", ""),
                "source_doc": props.get("source_doc", ""),
                "page_range": props.get("page_range", ""),
                "chunk_edges": per_chunk[props.get("chunk_id", "")],
                "type": edge_type,
                "predicate": props.get("predicate", "") if edge_type == FALLBACK else "",
                "family": family_of.get(edge_type, "other"),
                "in_vocab": int(edge_type in vocab),
                "fallback": int(edge_type == FALLBACK),
                "fallback_predicate": int(edge_type == FALLBACK and bool(props.get("predicate"))),
                "subject": subject["props"].get("name", ""),
                "subject_label": s_label,
                "object": obj["props"].get("name", ""),
                "object_label": o_label,
                "concept_end": "both" if len(ends) == 2 else (ends[0] if ends else "none"),
                "class_name_object": int(str(obj["props"].get("name", "")).strip().lower() in class_names),
                "violations": ";".join(r.rule for r in rules if r.broken_by(edge_type, s_label, o_label)),
                "confidence": props.get("confidence", ""),
                "mention_count": props.get("mention_count", ""),
            }
        )
    return rows


def summary(rows: list[dict], rules: list[Rule], backup: Path) -> dict:
    """Counts of every signal, with the number of edges as denominator.

    Args:
        rows: Output of :func:`signals`.
        rules: The rules applied.
        backup: The backup read.

    Returns:
        A JSON-serialisable summary.
    """
    exposed = {r.type for r in rules}
    return {
        "backup": backup.name,
        "edges": len(rows),
        "chunks": len({r["chunk_id"] for r in rows}),
        "documents": len({r["source_doc"] for r in rows}),
        "duplicate_edge_keys": len(rows) - len({r["edge_key"] for r in rows}),
        "fallback": sum(r["fallback"] for r in rows),
        "fallback_with_predicate": sum(r["fallback_predicate"] for r in rows),
        "out_of_vocab": sum(1 - r["in_vocab"] for r in rows),
        "concept_end": dict(Counter(r["concept_end"] for r in rows)),
        "class_name_object": sum(r["class_name_object"] for r in rows),
        # Edges at least one rule could fail on: the denominator of the violation rate.
        "exposed_to_rules": sum(r["type"] in exposed for r in rows),
        "violations": {rule.rule: sum(rule.rule in r["violations"].split(";") for r in rows) for rule in rules},
        "edges_with_violation": sum(bool(r["violations"]) for r in rows),
        "family": dict(Counter(r["family"] for r in rows).most_common()),
        "rules": [{"rule": r.rule, "type": r.type, "subject_labels": sorted(r.subject_labels),
                   "object_labels": sorted(r.object_labels), "source": r.source} for r in rules],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backup", type=Path, help="backup folder (default: the newest)")
    parser.add_argument("--rules", type=Path, help="extra domain and range rules (TSV)")
    parser.add_argument("--out", type=Path, required=True, help="folder for edge_signals.tsv and summary.json")
    args = parser.parse_args()

    backup = args.backup or latest_backup()
    nodes = json.loads((backup / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((backup / "edges.json").read_text(encoding="utf-8"))
    vocab = set(json.loads(VOCAB.read_text(encoding="utf-8")))
    rules = PROMPT_RULES + (read_rules(args.rules) if args.rules else [])
    rows = signals(nodes, edges, vocab, rules)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "edge_signals.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = summary(rows, rules, backup)
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rules"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
