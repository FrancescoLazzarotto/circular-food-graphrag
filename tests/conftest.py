"""Shared pytest setup: the repository's packages importable from any directory."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable `graphrag` package importable even without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
# `kg_pipeline` is a top-level package in the repo root and `evalkit` lives under
# evaluation/. Without these entries both are importable only when pytest runs
# from the repo root, which puts the cwd on sys.path.
for _path in (_ROOT / "src", _ROOT, _ROOT / "evaluation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
