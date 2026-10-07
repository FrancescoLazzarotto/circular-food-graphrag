"""The proposed document cards keep only what the model's reply actually says.

A card is shown to curators as a proposal and, once checked, names the
document in every citation, so a reply that is malformed, padded with
reasoning or carrying an impossible year must not turn into a confident card.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "corpus" / "document_cards.py"
_spec = importlib.util.spec_from_file_location("document_cards", _PATH)
document_cards = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("document_cards", document_cards)
_spec.loader.exec_module(document_cards)


def test_a_reply_with_reasoning_and_a_fence_is_read():
    raw = (
        "<think>the cover says...</think>```json\n"
        '{"title": " Design  Sistemico ", "authors": ["Luigi Bistagnino"], "year": "2011"}\n```'
    )

    assert document_cards.parse_card(raw) == {
        "title": "Design Sistemico",
        "authors": ["Luigi Bistagnino"],
        "year": 2011,
    }


def test_what_the_pages_do_not_show_stays_empty():
    card = document_cards.parse_card('{"title": null, "authors": [], "year": null}')

    assert card == {"title": None, "authors": [], "year": None}


def test_a_string_that_says_nothing_is_shown_is_not_a_value():
    card = document_cards.parse_card('{"title": "None", "authors": "null", "year": null}')

    assert card == {"title": None, "authors": [], "year": None}


def test_an_impossible_year_and_bad_authors_are_dropped():
    card = document_cards.parse_card('{"title": "X", "authors": "FAO", "year": 20211}')

    assert card == {"title": "X", "authors": ["FAO"], "year": None}


def test_a_reply_without_json_is_an_error():
    with pytest.raises(ValueError):
        document_cards.parse_card("I could not find a title.")


class _FakeCompletions:
    """Answers each call with the next reply and records the temperatures asked for."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.temperatures: list[float] = []

    async def create(self, **kwargs):
        self.temperatures.append(kwargs["temperature"])
        message = SimpleNamespace(content=self.replies[len(self.temperatures) - 1])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_client(replies: list[str]):
    completions = _FakeCompletions(replies)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_a_malformed_reply_is_asked_again_with_some_randomness():
    client, completions = _fake_client(['{\n"  "title": "X"}', '{"title": "X", "authors": [], "year": 2020}'])

    card, note = asyncio.run(document_cards.propose(client, "m", "f.pdf", "text"))

    assert card == {"title": "X", "authors": [], "year": 2020}
    assert note == ""
    # A second greedy attempt would return the same malformed reply.
    assert completions.temperatures[0] == 0.0 and completions.temperatures[1] > 0.0


def test_a_document_whose_every_reply_is_malformed_has_no_card():
    client, completions = _fake_client(["no json here"] * 3)

    card, note = asyncio.run(document_cards.propose(client, "m", "f.pdf", "text"))

    assert card is None
    assert note.startswith("risposta illeggibile")
    assert len(completions.temperatures) == 3


def test_the_outputs_hold_proposals_next_to_what_stage_zero_found(tmp_path):
    results = [
        {
            "id_documento": "design",
            "filename": "DESIGN SISTEMICO 2┬░ Edizione.pdf",
            "percorso": "Systems Thinking/DESIGN SISTEMICO 2° Edizione.pdf",
            "tema": "Systems Thinking",
            "titolo_estratto": "Slow Food® Editore srl",
            "anno_estratto": 2011,
            "card": {"title": "Design Sistemico", "authors": ["L. Bistagnino"], "year": 2011},
            "nota": "",
        },
        {
            "id_documento": "scan",
            "filename": "scan.pdf",
            "percorso": "scan.pdf",
            "tema": "",
            "titolo_estratto": "",
            "anno_estratto": "",
            "card": None,
            "nota": "nessun testo nelle prime pagine",
        },
    ]

    document_cards.write_outputs(results, tmp_path)

    with (tmp_path / "schede_proposte.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert rows[0]["titolo_estratto"] == "Slow Food® Editore srl"
    assert rows[0]["titolo_proposto"] == "Design Sistemico"
    assert rows[1]["nota"] == "nessun testo nelle prime pagine"
    catalog = json.loads((tmp_path / "catalogo_proposto.json").read_text(encoding="utf-8"))
    # Keyed by the file name on disk, as product/config.py looks titles up.
    assert catalog["titles"] == {"DESIGN SISTEMICO 2┬░ Edizione.pdf": "Design Sistemico"}
    assert catalog["authors"] == {"DESIGN SISTEMICO 2┬░ Edizione.pdf": ["L. Bistagnino"]}
