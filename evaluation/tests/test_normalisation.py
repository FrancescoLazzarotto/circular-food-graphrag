"""Tests for the shared normalisation contract.

These functions are the base of every concept-level comparison and are consumed by
the resolver, the entity scorer and the gold loader, so a regression here silently
corrupts every downstream number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalkit.models import GoldEntity, GoldQuery
from evalkit.normalisation import fold_accents, match_key, normalise


class TestNormalise:
    def test_lowercases_and_collapses_whitespace(self) -> None:
        assert normalise("  Grape   Pomace  ") == "grape pomace"

    def test_strips_surrounding_punctuation(self) -> None:
        assert normalise('"whey".') == "whey"
        assert normalise("— co-evolution —") == "co-evolution"

    def test_keeps_internal_hyphen(self) -> None:
        """'co-evolution' must not become 'coevolution': they are distinct forms."""
        assert normalise("Co-evolution") == "co-evolution"

    def test_keeps_parenthetical_gloss(self) -> None:
        """Keep the ')' that closes a parenthetical gloss, as in Q25's label.

        A trailing ')' that closes a parenthetical is part of the term. Stripping it
        yields 'potassium bitartrate (cream of tartar' — unmatchable, and it silently
        breaks the curated-lexicon lookup keyed on the full label.
        """
        assert (
            normalise("potassium bitartrate (cream of tartar)")
            == "potassium bitartrate (cream of tartar)"
        )

    def test_empty_and_punctuation_only(self) -> None:
        assert normalise("") == ""
        assert normalise("   ") == ""
        assert normalise("...") == ""

    def test_is_idempotent(self) -> None:
        for raw in ["Whey", "  PM10 ", '"Ciclicità".', "potassium bitartrate (cream of tartar)"]:
            assert normalise(normalise(raw)) == normalise(raw)

    def test_does_not_strip_plural_s(self) -> None:
        """Plural folding goes through a lexicon, never a blind -s strip."""
        assert normalise("Rice husks") == "rice husks"


class TestFoldAccents:
    def test_folds_italian_accents(self) -> None:
        assert fold_accents("ciclicità") == "ciclicita"
        assert fold_accents("rinnovabilità") == "rinnovabilita"
        assert fold_accents("solidarietà") == "solidarieta"

    def test_leaves_unaccented_untouched(self) -> None:
        assert fold_accents("grape pomace") == "grape pomace"


class TestMatchKey:
    def test_combines_normalise_and_fold(self) -> None:
        assert match_key("  Ciclicità. ") == "ciclicita"

    def test_accented_and_folded_forms_share_a_key(self) -> None:
        """The reason folding exists: both spellings reach us from real pipelines."""
        assert match_key("Ciclicità") == match_key("ciclicita")

    def test_distinct_concepts_keep_distinct_keys(self) -> None:
        """Folding must not collapse concepts the gold keeps apart."""
        assert match_key("cooperation") != match_key("collaboration")
        assert match_key("symbiosis") != match_key("industrial symbiosis")
        assert match_key("rice husks") != match_key("rice bran")


class TestGoldEntity:
    def _entity(self, **kw: object) -> GoldEntity:
        base = dict(
            label="Grape pomace",
            normalised_label="grape pomace",
            alt_labels=("vinacce", "vinaccia", "grape marc"),
            uri="http://aims.fao.org/aos/agrovoc/c_909cff15",
            mapping_status="exact",
        )
        base.update(kw)
        return GoldEntity(**base)  # type: ignore[arg-type]

    def test_surface_forms_include_label_and_alts(self) -> None:
        forms = self._entity().surface_forms
        assert forms == {"grape pomace", "vinacce", "vinaccia", "grape marc"}

    def test_surface_forms_are_match_keys(self) -> None:
        entity = self._entity(
            normalised_label="cyclicality", alt_labels=("Ciclicità",), uri="urn:ceff:Cyclicality"
        )
        assert "ciclicita" in entity.surface_forms

    def test_surface_forms_drop_empty(self) -> None:
        entity = self._entity(alt_labels=("", "   ", "vinacce"))
        assert entity.surface_forms == {"grape pomace", "vinacce"}

    def test_exact_counts_at_grounding_level(self) -> None:
        assert self._entity(mapping_status="exact").counts_at_grounding_level is True

    def test_local_extension_excluded_from_grounding_level(self) -> None:
        """Grounding level scores only entities with a real vocabulary URI."""
        local = self._entity(mapping_status="benchmark_local_extension", uri="urn:ceff:Capital")
        assert local.counts_at_grounding_level is False

    def test_is_hashable(self) -> None:
        assert len({self._entity(), self._entity()}) == 1


class TestGoldQuery:
    def test_grounding_entities_filters_to_exact(self) -> None:
        exact = GoldEntity("Whey", "whey", ("siero",), "http://x/c_8376", "exact")
        local = GoldEntity("Scotta", "scotta", (), "urn:ceff:Scotta", "benchmark_local_extension")
        query = GoldQuery("Q11", "domain_specialist", "q", "a", (exact, local))
        assert query.grounding_entities == (exact,)

    def test_distractor_defaults_to_false(self) -> None:
        assert GoldQuery("Q01", "factual_simple", "q", "a", ()).distractor_expected is False


@pytest.mark.parametrize(
    ("gold_form", "pipeline_form"),
    [
        ("grape pomace", "Vinacce"),
        ("rice husks", "lolla di riso"),
        ("co-evolution", "Coevoluzione"),
        ("buttermilk", "latticello"),
        ("cyclicality", "Ciclicità"),
    ],
)
def test_cross_lingual_forms_match_via_alt_labels(gold_form: str, pipeline_form: str) -> None:
    """The central case: gold is English, the KG holds Italian labels.

    Runs against the real gold, so it also guards the alt_labels themselves: without
    them a correct Italian retrieval would be scored as a miss, which is exactly the
    failure mode alt_labels exist to prevent.
    """
    from evalkit.io.gold_loader import load_gold_json

    gold_path = Path(__file__).resolve().parents[2] / "evaluation" / "gold" / "gold.json"
    entities = [
        entity
        for query in load_gold_json(gold_path).values()
        for entity in query.expected_entities
        if entity.normalised_label == gold_form
    ]
    assert entities, f"{gold_form!r} not present in gold.json"
    assert match_key(pipeline_form) in entities[0].surface_forms
