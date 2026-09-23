"""Acronym detection and expansion for entity surface forms."""

from __future__ import annotations

import re


# The letter class includes Latin-1 accented characters so Italian long forms
# ("Università di Scienze Gastronomiche (UNISG)") are captured in full.
_LONG_SHORT_RE = re.compile(
    r"\b([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ '\-/]{3,}?)\s*\(([A-Z]{2,10})\)"
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_surface(text: str) -> str:
    """Lower-case ``text`` and drop every non-alphanumeric character."""
    return _NON_ALNUM_RE.sub("", text.lower())


def update_acronym_map(acronym_map: dict[str, str], text: str) -> None:
    """Add every ``Long Form (ACRONYM)`` definition found in ``text``.

    Later definitions of the same acronym overwrite earlier ones.

    Args:
        acronym_map: Mapping from upper-case acronym to long form, updated in
            place.
        text: Text to scan for definitions.
    """
    for long_form, short_form in _LONG_SHORT_RE.findall(text):
        long_clean = " ".join(long_form.split()).strip()
        short_clean = short_form.strip().upper()
        if len(long_clean) > 2 and len(short_clean) > 1:
            acronym_map[short_clean] = long_clean


def expand_acronym(surface: str, acronym_map: dict[str, str]) -> str:
    """Replace a known acronym with its long form.

    Args:
        surface: Entity surface form.
        acronym_map: Mapping from upper-case acronym to long form.

    Returns:
        The long form when the stripped ``surface`` is a known acronym
        (case-insensitive), otherwise the stripped ``surface``.
    """
    stripped = surface.strip()
    if not stripped:
        return stripped
    if stripped.upper() in acronym_map:
        return acronym_map[stripped.upper()]
    return stripped
