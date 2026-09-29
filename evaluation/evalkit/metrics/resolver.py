"""The shared label -> URI resolver behind grounding-level scoring."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.normalisation import match_key

# The shared resolver: one deterministic `resolve(label) -> URI | None`,
# applied SYMMETRICALLY to every pipeline and fixed before any run. Nothing
# here may be tuned after seeing a result.
#
# Offline by construction: the only inputs are the frozen gold and a local
# snapshot of AGROVOC/ChEBI labels. No network call at build or resolve time, so
# an upstream vocabulary update cannot silently move the published numbers.

logger = logging.getLogger("graphrag")

DEFAULT_VOCAB_PATH = Path(__file__).resolve().parents[2] / "gold" / "scripts" / "vocab_labels.json"

SOURCE_GOLD = "gold"
SOURCE_VOCABULARY = "vocabulary"

# Precedence when the same surface form is claimed by two different URIs across
# sources. Lower wins. The gold outranks the external snapshot by the
# pre-registered decision recorded in the gold's
# `_meta.alt_labels_provenance.known_terminology_conflicts`: AGROVOC gives
# 'Pula di riso' as the Italian prefLabel of rice husks, while the corpus this
# gold is built on uses 'pula' for bran and 'lolla' for husks. Corpus usage is
# authoritative for a benchmark scored on that corpus. This is a declared rule,
# not a per-case choice: it is applied uniformly and every suppression it causes
# is logged and exposed on `Resolver.conflicts`.
_SOURCE_PRECEDENCE = {SOURCE_GOLD: 0, SOURCE_VOCABULARY: 1}

# --- Singular/plural lexicon --------------------------------------------------
#
# Folding is EXPLICIT: an entry exists for a form or it is not folded. Blind
# trailing -s stripping is not allowed, and this corpus shows exactly why:
#   'lees'   (wine lees, c_28445) -> a blind strip yields 'lee', a different word
#   'biogas' (c_9262)             -> a blind strip yields 'bioga', a non-word
#   'dregs', 'bran', 'pomace', 'whey', 'sludge' -> mass nouns / plurale tantum
# Every entry maps PLURAL -> SINGULAR and is applied token-wise to both the index
# and the query, so the two meet on the same canonical key regardless of which
# number each side used.

# Pairs whose two members are BOTH attested in the gold's own alt_labels or in
# the vocabulary snapshot. Provenance is machine-checked by
# test_from_gold_lexicon_pairs_are_attested_in_the_data.
_NUMBER_LEXICON_FROM_GOLD: dict[str, str] = {
    # Italian
    "acidi": "acido",
    "acque": "acqua",
    "anaerobici": "anaerobico",
    "biomasse": "biomassa",
    "bioplastiche": "bioplastica",
    "bottiglie": "bottiglia",
    "campi": "campo",
    "chimici": "chimico",
    "digestori": "digestore",
    "fecce": "feccia",
    "fenolici": "fenolico",
    "fertilizzanti": "fertilizzante",
    "lieviti": "lievito",
    "minerali": "minerale",
    "naturali": "naturale",
    "polifenoli": "polifenolo",
    "raspi": "raspo",
    "sorgenti": "sorgente",
    "vinacce": "vinaccia",
    # English
    "acids": "acid",
    "biomasses": "biomass",
    "bioplastics": "bioplastic",
    "bottles": "bottle",
    "digesters": "digester",
    "fillers": "filler",
    "firms": "firm",
    "flavonoids": "flavonoid",
    "goals": "goal",
    "husks": "husk",
    "polyphenols": "polyphenol",
    "residues": "residue",
    "sdgs": "sdg",
    "stalks": "stalk",
    "sterols": "sterol",
    "straws": "straw",
    "yeasts": "yeast",
}

# Curated additions: real number variants of head nouns of gold concepts whose
# counterpart happens not to occur in the gold. They exist to catch a pipeline
# emitting the other number ('husk'/'husks' is the typical case); they are
# pipeline-independent, added before any run, and can only
# ever map a form onto a concept the gold already declares.
_NUMBER_LEXICON_CURATED: dict[str, str] = {
    # Italian
    "aziende": "azienda",
    "cariche": "carica",
    "effluenti": "effluente",
    "fanghi": "fango",
    "flavonoidi": "flavonoide",
    "imprese": "impresa",
    "reflui": "refluo",
    "steroli": "sterolo",
    # English
    "cycles": "cycle",
    "fertilisers": "fertiliser",
    "fertilizers": "fertilizer",
    "hulls": "hull",
    "springs": "spring",
    "stems": "stem",
}

NUMBER_LEXICON: dict[str, str] = {**_NUMBER_LEXICON_FROM_GOLD, **_NUMBER_LEXICON_CURATED}


class ResolverError(Exception):
    """Base class for resolver failures."""


class AmbiguousLabelError(ResolverError):
    """A surface form resolves to more than one URI.

    Raised instead of picking a winner: an arbitrary choice would be an
    undeclared scoring rule, and the resolver has to stay auditable. Ambiguity
    is only ever reported for forms claimed by two URIs within the SAME source;
    a gold-vs-snapshot disagreement is settled by the declared precedence rule
    and surfaces on `Resolver.conflicts` instead.

    Attributes:
        label: The label as passed to `resolve`.
        key: The comparison key the label normalised to.
        uris: The conflicting URIs, sorted.
    """

    def __init__(self, label: str, key: str, uris: Sequence[str]) -> None:
        """Record the ambiguous label and its URIs."""
        self.label = label
        self.key = key
        self.uris = tuple(uris)
        super().__init__(
            f"label {label!r} (key {key!r}) is ambiguous across {len(self.uris)} URIs: "
            f"{', '.join(self.uris)}"
        )


@dataclass(frozen=True)
class LabelRecord:
    """One (surface form, URI) claim from one source.

    Attributes:
        form: Raw surface form, normalised by the resolver at build time.
        uri: The URI the form is claimed to denote.
        source: `SOURCE_GOLD` or `SOURCE_VOCABULARY`; decides precedence.
    """

    form: str
    uri: str
    source: str


@dataclass(frozen=True)
class LabelConflict:
    """A surface form claimed by different URIs in different sources.

    Attributes:
        key: The comparison key in conflict.
        kept_uri: URI retained, from the highest-precedence source.
        dropped_uris: URIs suppressed, from lower-precedence sources.
        kept_source: Source of `kept_uri`.
    """

    key: str
    kept_uri: str
    dropped_uris: tuple[str, ...]
    kept_source: str


def fold_number(key: str) -> str:
    """Fold a comparison key to singular via the explicit lexicon.

    Applied token-wise to both indexed forms and queries. Tokens absent from the
    lexicon are left untouched — there is no morphological rule and no blind -s
    strip.

    Args:
        key: A key already produced by `normalisation.match_key`.

    Returns:
        The key with every known plural token replaced by its singular.
    """
    return " ".join(NUMBER_LEXICON.get(token, token) for token in key.split())


def _index_records(
    records: Iterable[LabelRecord], fold: bool
) -> tuple[dict[str, str], dict[str, tuple[str, ...]], list[LabelConflict]]:
    """Build one lookup index, applying source precedence and ambiguity rules.

    Args:
        records: The label claims to index.
        fold: Whether to key by the number-folded form.

    Returns:
        Tuple of (key -> URI, key -> conflicting URIs, cross-source conflicts).
    """
    claims: dict[str, dict[int, set[str]]] = {}
    sources: dict[tuple[str, int], str] = {}
    for record in records:
        key = match_key(record.form)
        if not key:
            continue
        if fold:
            key = fold_number(key)
        rank = _SOURCE_PRECEDENCE[record.source]
        claims.setdefault(key, {}).setdefault(rank, set()).add(record.uri)
        sources[(key, rank)] = record.source

    index: dict[str, str] = {}
    ambiguous: dict[str, tuple[str, ...]] = {}
    conflicts: list[LabelConflict] = []
    for key, tiers in claims.items():
        best = min(tiers)
        winners = tiers[best]
        overridden = {uri for rank, uris in tiers.items() if rank > best for uri in uris} - winners
        if overridden and len(winners) == 1:
            conflicts.append(
                LabelConflict(
                    key=key,
                    kept_uri=next(iter(winners)),
                    dropped_uris=tuple(sorted(overridden)),
                    kept_source=sources[(key, best)],
                )
            )
        if len(winners) > 1:
            ambiguous[key] = tuple(sorted(winners))
        else:
            index[key] = next(iter(winners))
    return index, ambiguous, conflicts


class Resolver:
    """The shared label -> URI resolver.

    Built once from the frozen gold plus a local vocabulary snapshot, then reused
    for every row of every pipeline. Lookup is three-tiered and strictly ordered:

    1. identity, when the input already is a URI this resolver knows — the
       ontology-grounded pipeline is a no-op pass-through;
    2. exact match on the comparison key, against gold `normalised_label` +
       `alt_labels` and against vocabulary `skos:prefLabel` + `skos:altLabel`
       (using altLabels is correct vocabulary use, not generosity);
    3. match after singular/plural folding through the explicit lexicon.

    There is no fourth tier: no fuzzy match, no edit distance.
    """

    def __init__(self, records: Iterable[LabelRecord]) -> None:
        """Build the indices from label claims.

        Args:
            records: Every (form, URI, source) claim to index.
        """
        materialised = list(records)
        self._exact, self._exact_ambiguous, exact_conflicts = _index_records(
            materialised, fold=False
        )
        self._folded, self._folded_ambiguous, folded_conflicts = _index_records(
            materialised, fold=True
        )
        self._known_uris = frozenset(record.uri for record in materialised)

        deduped: dict[tuple[str, str, tuple[str, ...]], LabelConflict] = {}
        for conflict in (*exact_conflicts, *folded_conflicts):
            deduped.setdefault((conflict.key, conflict.kept_uri, conflict.dropped_uris), conflict)
        self._conflicts = tuple(deduped.values())

        for conflict in self._conflicts:
            logger.warning(
                "resolver: form %r claimed by %s; keeping %s (source=%s), suppressing %s",
                conflict.key,
                "multiple sources",
                conflict.kept_uri,
                conflict.kept_source,
                ", ".join(conflict.dropped_uris),
            )
        for key, uris in sorted(self._exact_ambiguous.items()):
            logger.warning(
                "resolver: form %r is ambiguous within its source (%s); resolve() will raise",
                key,
                ", ".join(uris),
            )
        logger.info(
            "resolver: %d exact forms, %d folded forms, %d URIs, %d conflicts, %d ambiguous",
            len(self._exact),
            len(self._folded),
            len(self._known_uris),
            len(self._conflicts),
            len(self._exact_ambiguous),
        )

    @classmethod
    def from_gold(
        cls, gold_path: str | Path, vocab_path: str | Path | None = None
    ) -> "Resolver":
        """Build the resolver from the gold and the local vocabulary snapshot.

        The `urn:ceff:` benchmark-local concepts have no external vocabulary and
        are resolved from the gold's own alt_labels; the external concepts
        additionally pick up their AGROVOC/ChEBI pref/altLabels from the snapshot.

        Args:
            gold_path: Path to the frozen gold JSON.
            vocab_path: Path to the label snapshot; defaults to
                `DEFAULT_VOCAB_PATH`. No network is ever consulted.

        Returns:
            A ready-to-use resolver.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        gold_file = Path(gold_path)
        vocab_file = Path(DEFAULT_VOCAB_PATH if vocab_path is None else vocab_path)
        gold: dict[str, Any] = json.loads(gold_file.read_text(encoding="utf-8"))
        vocab: dict[str, Any] = json.loads(vocab_file.read_text(encoding="utf-8"))

        records: list[LabelRecord] = []
        for query in gold.get("queries", []):
            for entity in query.get("expected_entities", []):
                uri = entity.get("uri")
                if not uri:
                    continue
                forms = [entity.get("normalised_label", ""), *entity.get("alt_labels", [])]
                records.extend(
                    LabelRecord(form=form, uri=uri, source=SOURCE_GOLD) for form in forms if form
                )
        for uri, entry in vocab.items():
            forms = [*entry.get("pref", []), *entry.get("alt", [])]
            records.extend(
                LabelRecord(form=form, uri=uri, source=SOURCE_VOCABULARY) for form in forms if form
            )

        logger.info(
            "resolver: loaded %d label claims from %s and %s",
            len(records),
            gold_file.name,
            vocab_file.name,
        )
        return cls(records)

    def resolve(self, label: str) -> str | None:
        """Resolve a surface form to its canonical URI.

        Args:
            label: A label from the gold or from any pipeline's output.

        Returns:
            The canonical URI, or None when the form is unknown. A None is a
            real, measurable property (the entity is unanchorable), not an error:
            it still counts at concept-level, never at grounding-level.

        Raises:
            AmbiguousLabelError: If the form is claimed by two URIs within one
                source. Never resolved arbitrarily.
        """
        if not label or not label.strip():
            return None
        if label.strip() in self._known_uris:
            return label.strip()

        key = match_key(label)
        if not key:
            return None
        if key in self._exact_ambiguous:
            raise AmbiguousLabelError(label, key, self._exact_ambiguous[key])
        exact = self._exact.get(key)
        if exact is not None:
            return exact

        folded = fold_number(key)
        if folded in self._folded_ambiguous:
            raise AmbiguousLabelError(label, folded, self._folded_ambiguous[folded])
        return self._folded.get(folded)

    @property
    def conflicts(self) -> tuple[LabelConflict, ...]:
        """Cross-source disagreements settled by precedence, for audit."""
        return self._conflicts

    @property
    def ambiguous_forms(self) -> dict[str, tuple[str, ...]]:
        """Forms that raise `AmbiguousLabelError`, mapped to their URIs."""
        return dict(self._exact_ambiguous)

    @property
    def known_uris(self) -> frozenset[str]:
        """Every URI this resolver can return."""
        return self._known_uris

    def __len__(self) -> int:
        """Number of distinct exact-match surface forms indexed."""
        return len(self._exact)
