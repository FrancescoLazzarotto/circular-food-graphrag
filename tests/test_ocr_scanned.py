"""Which pages the OCR copy recognises: those without text a reader could read.

A page with no text layer is the scanned case. A page whose text layer is made
of control characters — a font without a Unicode map — is the other one: stage 0
turns it into U+FFFD, and the passages it makes are unreadable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "corpus" / "ocr_scanned.py"
_spec = importlib.util.spec_from_file_location("ocr_scanned", _PATH)
ocr_scanned = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ocr_scanned", ocr_scanned)
_spec.loader.exec_module(ocr_scanned)

READABLE = (
    "L'ecomuseo esalta i concetti di unicità e di irripetibilità di un luogo, "
    "aspetti che costituiscono la struttura originale dell'identità."
)


def test_a_page_with_ordinary_text_is_kept():
    assert ocr_scanned._has_text(READABLE)


def test_a_page_with_less_than_a_line_has_no_text():
    assert not ocr_scanned._has_text("12")


def test_a_layer_of_control_characters_has_no_text():
    """The bytes of a real page whose font has no Unicode map."""
    layer = "\x14\x08\x05\x04\x05\x06\x17\x08\x18\x19\x05\n\x02\x13\x02\n+\x06)\x0e\n" * 6
    assert not ocr_scanned._has_text(layer)


def test_a_layer_of_replacement_characters_has_no_text():
    assert not ocr_scanned._has_text("�" * 120)


def test_a_few_odd_characters_do_not_condemn_a_page():
    """Readable pages carry a fraction of a percent of them, a ligature or a bullet."""
    assert ocr_scanned._has_text(READABLE + "�\x0c" + READABLE)
