"""The relationship vocabulary is one list, and every pass renames into it.

`neo4j_postprocess` and the `kg_repair*` scripts share the canonical
vocabulary. Private copies drift: a copy carrying `HAS_DEFINITION` renames
`DEFINITION` into a type no other pass considers canonical and which has no
instances in the graph.

Equality between copies is not what these tests check, since there is only
one list and comparing it to itself would pass forever. They check the two
things that can still go wrong: someone reintroducing a private copy, and a
rename whose target is not in the vocabulary.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kg_pipeline.relations import CANONICAL_RELATION_SET

ROOT = Path(__file__).resolve().parents[1]

VOCABULARY_CONSUMERS = [
    ROOT / "kg_pipeline" / "stages" / "neo4j_postprocess.py",
    ROOT / "scripts" / "kg" / "kg_repair2.py",
    ROOT / "scripts" / "kg" / "kg_repair3.py",
    ROOT / "scripts" / "kg" / "kg_repair4.py",
]


def _assignments(path: Path) -> list[tuple[str, ast.expr]]:
    """Every module-level ``name = value``, annotated or not."""
    out: list[tuple[str, ast.expr]] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out.append((node.target.id, node.value))
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out.append((node.targets[0].id, node.value))
    return out


@pytest.mark.parametrize("path", VOCABULARY_CONSUMERS, ids=lambda p: p.name)
def test_vocabulary_is_not_copied(path: Path) -> None:
    """No consumer may spell the vocabulary out again."""
    for name, value in _assignments(path):
        if "CANONICAL" not in name.upper():
            continue
        assert not isinstance(value, (ast.List, ast.Tuple, ast.Set)), (
            f"{path.name} assigns {name} a literal collection. The canonical "
            "vocabulary lives in kg_pipeline.relations; import it instead, or "
            "the copies drift apart the way HAS_DEFINITION did."
        )


def _rename_targets(path: Path) -> list[tuple[str, str, int]]:
    """Every ``(old, new)`` pair in a list whose name mentions a rename."""
    found: list[tuple[str, str, int]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if not name or "rename" not in name.lower():
            continue
        if not isinstance(node.value, ast.List):
            continue
        for element in node.value.elts:
            if isinstance(element, ast.Tuple) and len(element.elts) == 2:
                try:
                    old, new = (ast.literal_eval(e) for e in element.elts)
                except ValueError:
                    continue
                if isinstance(old, str) and isinstance(new, str):
                    found.append((old, new, element.lineno))
    return found


@pytest.mark.parametrize("path", VOCABULARY_CONSUMERS, ids=lambda p: p.name)
def test_every_rename_targets_a_canonical_type(path: Path) -> None:
    """A pass that renames into a non-canonical type creates an orphan type."""
    offenders = [
        f"{path.name}:{line} renames {old} -> {new}"
        for old, new, line in _rename_targets(path)
        if new not in CANONICAL_RELATION_SET
    ]
    assert not offenders, "rename targets outside the vocabulary: " + "; ".join(offenders)
