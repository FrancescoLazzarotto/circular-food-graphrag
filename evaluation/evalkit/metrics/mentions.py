"""Gazetteer mention extraction from running text, for the answer channel.

Deterministic post-hoc extractor: finds gold surface forms (normalised_label +
alt_labels) in generated text via whole-word matching on the shared
``match_key`` folding. It closes the structural asymmetry where graph
pipelines report ``retrieved_entities`` but text-RAG reports none, so the
baseline scores 0 by construction: the same gazetteer and the same
normalisation are applied to every pipeline's ANSWER text, symmetrically.

There is no model and no threshold, so the only pre-registered choices are the
gazetteer source (the gold's own surface forms) and the folding
(``evalkit.normalisation.match_key``).

Known, accepted limitation: a gazetteer can only see gold vocabulary. Answer
text naming entities outside the gold (correct or hallucinated) is invisible
here, so answer-channel precision is measured against the gold vocabulary
only, not against an open vocabulary. Recall is the meaningful direction. An
open-vocabulary extractor (e.g. GLiNER) would need a threshold pre-registered
before any gold run; it is deliberately out of this module.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Iterable, Sequence

from ..models import EvalRow, GoldEntity, GoldQuery
from ..normalisation import fold_accents, match_key

__all__ = ["Gazetteer", "answer_channel_row"]


def _compile_form(form_key: str) -> re.Pattern[str]:
    """Compile a whole-word pattern for one already-folded surface form."""
    tokens = [re.escape(tok) for tok in form_key.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)")


class Gazetteer:
    """All gold surface forms, compiled once, searched in folded text."""

    def __init__(self, forms: Iterable[str]) -> None:
        """Compile a whole-word pattern for every distinct folded form."""
        keys = sorted({match_key(f) for f in forms if match_key(f)})
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            (key, _compile_form(key)) for key in keys
        ]

    @classmethod
    def from_entities(cls, entities: Sequence[GoldEntity]) -> "Gazetteer":
        """A gazetteer over the surface forms of `entities`."""
        return cls(form for entity in entities for form in entity.surface_forms)

    @classmethod
    def from_gold(cls, queries: Sequence[GoldQuery]) -> "Gazetteer":
        """Build over ALL queries' entities, not just one query's.

        A per-query gazetteer can never produce a spurious hit (it only knows
        the expected forms), which degenerates answer-channel precision to a
        constant 1.0. The global gazetteer at least exposes cross-query
        leakage: an answer naming concepts expected elsewhere in the gold.
        """
        return cls.from_entities(
            [e for q in queries for e in q.expected_entities]
        )

    def extract(self, text: str) -> list[str]:
        """Return every gazetteer form found in ``text``, whole-word, folded.

        Args:
            text: Running text (typically a generated answer).

        Returns:
            Matched form keys, deduplicated, in gazetteer order.
        """
        if not text:
            return []
        folded = fold_accents(text.lower())
        return [key for key, pattern in self._patterns if pattern.search(folded)]


def answer_channel_row(row: EvalRow, gazetteer: Gazetteer) -> EvalRow:
    """Clone ``row`` with ``retrieved_entities`` replaced by answer mentions.

    The clone feeds the unchanged ``score_row`` so both channels — what the
    retriever surfaced vs what the answer actually says — go through the exact
    same scoring path, resolver included.

    Args:
        row: The original evaluation row (left untouched).
        gazetteer: A gazetteer built over the whole gold.

    Returns:
        A copy of the row whose entity list is the answer-text mentions.
    """
    return dataclasses.replace(
        row, retrieved_entities=list(gazetteer.extract(row.answer))
    )
