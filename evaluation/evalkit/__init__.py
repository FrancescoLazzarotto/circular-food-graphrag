"""Evaluation toolkit: gold loading, run joining, metrics, judges and reports."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# When evalkit is imported as evaluation.evalkit (e.g. python -m evaluation.evalkit.cli)
# the parent evaluation/ directory must be on sys.path so that intra-package
# imports like `from evalkit.models import ...` resolve correctly.
_EVAL_DIR = _Path(__file__).resolve().parent.parent  # …/evaluation/
if str(_EVAL_DIR) not in _sys.path:
    _sys.path.insert(0, str(_EVAL_DIR))
