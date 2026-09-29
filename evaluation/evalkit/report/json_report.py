"""JSON serialisation of a report."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from evalkit.models import ReportModel


def _serialise(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialise(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json_report(report: ReportModel, output_path: Path) -> None:
    """Serialise a ReportModel to a JSON file.

    Args:
        report: ReportModel to serialise.
        output_path: Destination .json path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialise(report)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
