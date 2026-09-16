"""Every `DEMO_*` variable the product reads must be documented.

`docs/configuration.md` is where an operator looks to find out what they can
change. It omitted eleven variables, and they were not the harmless ones: the
domain gate, intra-session memory, the vector channel, the parametric fallback
— the switches that change what the demo answers. A variable nobody documented
is a behaviour nobody can turn off deliberately.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PRODUCT = _ROOT / "product"
_DOC = _ROOT / "docs" / "configuration.md"
_VAR = re.compile(r"DEMO_[A-Z0-9_]+")


def _read_by_the_demo() -> set[str]:
    """Every DEMO_* the demo reads, from the whole package.

    Not `config.py` alone: `DEMO_LOG_DIR_RUNTIME` is read by `app.py`, which
    puts the operational log somewhere the session transcripts are not, and
    scoping this to one file made documenting it look like an invention.
    """
    return {
        nome
        for sorgente in sorted(_PRODUCT.rglob("*.py"))
        for nome in _VAR.findall(sorgente.read_text(encoding="utf-8"))
    }


def test_every_demo_variable_is_documented() -> None:
    letti = _read_by_the_demo()
    documentati = set(_VAR.findall(_DOC.read_text(encoding="utf-8")))

    mancanti = sorted(letti - documentati)

    assert not mancanti, (
        f"{len(mancanti)} DEMO_* variables are read under product/ and "
        f"absent from docs/configuration.md: {mancanti}"
    )


def test_the_documentation_invents_no_variable() -> None:
    """The other direction: a documented switch that does nothing misleads too."""
    letti = _read_by_the_demo()
    documentati = set(_VAR.findall(_DOC.read_text(encoding="utf-8")))

    inventate = sorted(documentati - letti)

    assert not inventate, (
        f"documented but not read under product/: {inventate}"
    )
