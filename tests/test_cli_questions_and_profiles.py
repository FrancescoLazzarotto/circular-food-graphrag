"""What a campaign is actually asked, and what it is actually run with.

Two decisions in `graphrag.cli` that are silent when they go wrong. The first
is which questions a run answers and what ids they carry: the evaluator joins
results to the gold set on `query_id`, so a duplicated or missing id turns a
scored campaign into an ambiguous one. The second is the precedence between a
profile and a typed flag — get it backwards and the run is not the arm its
`config.json` says it is.

No model and no graph: these read files and namespaces.
"""

from __future__ import annotations

import argparse
import json
import logging

import pytest

from graphrag import cli
from graphrag.profiles import PROFILES


def _args(**over):
    """A namespace with the question arguments, overridable by keyword."""
    base = {"question": "una domanda", "questions_file": ""}
    base.update(over)
    return argparse.Namespace(**base)


# --- one question, no file -------------------------------------------------


def test_without_a_file_the_single_question_is_the_run():
    questions = cli._load_questions(_args(question="cos'e' la scotta?"))

    assert len(questions) == 1
    assert questions[0].text == "cos'e' la scotta?"
    assert questions[0].query_id == ""


def test_a_file_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="Questions file not found"):
        cli._load_questions(_args(questions_file=str(tmp_path / "assente.txt")))


def test_an_empty_file_is_an_error_not_an_empty_campaign(tmp_path):
    path = tmp_path / "vuoto.txt"
    path.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="is empty"):
        cli._load_questions(_args(questions_file=str(path)))


# --- the four formats ------------------------------------------------------


def test_a_plain_text_file_is_one_question_per_line(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("prima domanda\n\nseconda domanda\n", encoding="utf-8")

    questions = cli._load_questions(_args(questions_file=str(path)))

    assert [q.text for q in questions] == ["prima domanda", "seconda domanda"]
    assert all(q.query_id == "" for q in questions)


def test_a_tab_gives_a_plain_text_line_its_gold_id(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("Q01\tcos'e' la scotta?\n", encoding="utf-8")

    questions = cli._load_questions(_args(questions_file=str(path)))

    assert (questions[0].query_id, questions[0].text) == ("Q01", "cos'e' la scotta?")


def test_a_line_with_an_empty_half_keeps_its_legacy_meaning(tmp_path):
    # Lines without a usable id keep meaning "the whole line is the question".
    path = tmp_path / "q.txt"
    path.write_text("\tcos'e' la scotta?\n", encoding="utf-8")

    questions = cli._load_questions(_args(questions_file=str(path)))

    assert questions[0].query_id == ""
    assert "scotta" in questions[0].text


def test_the_gold_file_can_be_handed_over_as_it_is(tmp_path):
    # So a run against the gold set is guaranteed to emit ids that join to it.
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps({"queries": [{"query_id": "Q01", "query": "what is whey?"}]}),
        encoding="utf-8",
    )

    questions = cli._load_questions(_args(questions_file=str(path)))

    assert (questions[0].query_id, questions[0].text) == ("Q01", "what is whey?")


def test_a_bare_json_list_also_works(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"id": "Q01", "question": "una"}]), encoding="utf-8")

    questions = cli._load_questions(_args(questions_file=str(path)))

    assert (questions[0].query_id, questions[0].text) == ("Q01", "una")


def test_a_jsonl_file_is_one_object_per_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"query_id": "Q01", "query": "una"}\n\n{"query_id": "Q02", "query": "due"}\n',
        encoding="utf-8",
    )

    questions = cli._load_questions(_args(questions_file=str(path)))

    assert [q.query_id for q in questions] == ["Q01", "Q02"]


def test_a_broken_jsonl_line_names_its_line_number(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"query": "una"}\nnon json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"q\.jsonl:2"):
        cli._load_questions(_args(questions_file=str(path)))


def test_a_csv_file_joins_on_its_query_id_column(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text("query_id,query\nQ01,una\nQ02,due\n", encoding="utf-8")

    questions = cli._load_questions(_args(questions_file=str(path)))

    assert [(q.query_id, q.text) for q in questions] == [("Q01", "una"), ("Q02", "due")]


def test_an_entry_with_no_question_text_is_refused(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"query_id": "Q01"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no 'query'/'question'/'text'"):
        cli._load_questions(_args(questions_file=str(path)))


@pytest.mark.parametrize("field", ["query", "question", "text"])
def test_any_of_the_three_text_fields_is_accepted(field):
    question = cli._question_from_obj({field: "una domanda"}, "dove")

    assert question.text == "una domanda"


@pytest.mark.parametrize("field", ["query_id", "id"])
def test_either_id_field_is_accepted(field):
    question = cli._question_from_obj({"query": "x", field: "Q01"}, "dove")

    assert question.query_id == "Q01"


# --- the ids the evaluator joins on ----------------------------------------


def test_a_repeated_id_is_refused_before_the_campaign_runs(tmp_path):
    # Duplicated ids make the evaluator's join ambiguous, and it joins on
    # exactly this field — a campaign that ran is expensive to discard.
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"query_id": "Q01", "query": "una"}\n{"query_id": "Q01", "query": "due"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate query_id\(s\) \['Q01'\]"):
        cli._load_questions(_args(questions_file=str(path)))


def test_a_file_with_no_ids_at_all_warns_about_the_text_join(tmp_path, caplog):
    path = tmp_path / "q.txt"
    path.write_text("una domanda\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        cli._load_questions(_args(questions_file=str(path)))

    assert "empty query_id" in caplog.text


def test_a_file_with_some_ids_warns_about_the_rest(tmp_path, caplog):
    path = tmp_path / "q.txt"
    path.write_text("Q01\tuna\naltra senza id\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        cli._load_questions(_args(questions_file=str(path)))

    assert "only 1/2" in caplog.text


def test_a_fully_identified_file_says_so_and_does_not_warn(tmp_path, caplog):
    path = tmp_path / "q.jsonl"
    path.write_text('{"query_id": "Q01", "query": "una"}\n', encoding="utf-8")

    with caplog.at_level(logging.INFO):
        cli._load_questions(_args(questions_file=str(path)))

    assert "Loaded 1 questions with query_id" in caplog.text
    assert "empty query_id" not in caplog.text


# --- profile against typed flag --------------------------------------------


def test_without_a_profile_the_parse_is_an_ordinary_one():
    parser = cli._build_arg_parser()

    args = cli._parse_args(parser, ["--question", "una"])

    assert args.question == "una"
    assert not getattr(args, "profile", None)


def _cli_usable_profiles() -> list[str]:
    """The profiles every field of which the CLI has a flag for."""
    parser = cli._build_arg_parser()
    known = set(vars(parser.parse_args(["--question", "x"])))
    return [
        name
        for name, fields in PROFILES.items()
        if all(cli._PROFILE_FIELD_TO_ARG.get(f, f) in known for f in fields)
    ]


@pytest.mark.parametrize("profile", sorted(_cli_usable_profiles()))
def test_a_profile_the_cli_can_express_reaches_the_namespace(profile):
    parser = cli._build_arg_parser()

    args = cli._parse_args(parser, ["--question", "una", "--profile", profile])

    assert args.profile == profile


def test_the_demo_profile_is_refused_rather_than_half_applied():
    # It sets three fields the CLI has no flag for. Dropping them silently
    # would hand the caller a configuration that is not the profile they asked
    # for, so the parser refuses and says which fields and where to use it.
    parser = cli._build_arg_parser()

    with pytest.raises(SystemExit):
        cli._parse_args(parser, ["--question", "una", "--profile", "demo"])


def test_the_refusal_names_the_fields_and_the_way_round_it(capsys):
    parser = cli._build_arg_parser()

    with pytest.raises(SystemExit):
        cli._parse_args(parser, ["--question", "una", "--profile", "demo"])

    message = capsys.readouterr().err
    assert "always_include_limits" in message
    assert "graphrag.profiles" in message


def test_a_typed_flag_beats_the_profile_that_would_have_set_it():
    # Precedence is argparse's own: the profile is installed with set_defaults
    # and the same argv parsed again, so a flag the caller typed wins without
    # anyone working out which flags were typed.
    parser = cli._build_arg_parser()
    profile = sorted(_cli_usable_profiles())[0]

    args = cli._parse_args(
        parser, ["--question", "una", "--profile", profile, "--max-new-tokens", "77"]
    )

    assert args.max_new_tokens == 77


def test_a_profile_field_not_typed_is_applied():
    parser = cli._build_arg_parser()
    profile = "thesis_campaign"
    expected = PROFILES[profile]

    args = cli._parse_args(parser, ["--question", "una", "--profile", profile])

    field = next(iter(expected))
    dest = cli._PROFILE_FIELD_TO_ARG.get(field, field)
    value = expected[field]
    assert getattr(args, dest) == (value.value if hasattr(value, "value") else value)


def test_a_profile_field_the_cli_cannot_express_is_refused(monkeypatch):
    parser = cli._build_arg_parser()
    monkeypatch.setitem(PROFILES, "_inventato", {"campo_inesistente": 1})

    with pytest.raises(SystemExit):
        cli._profile_defaults(parser, "_inventato", {"question"})


def test_an_enum_in_a_profile_reaches_the_namespace_as_its_value(monkeypatch):
    import enum

    class _Tone(enum.Enum):
        NEUTRAL = "neutral"

    parser = cli._build_arg_parser()
    monkeypatch.setitem(PROFILES, "_inventato", {"tone": _Tone.NEUTRAL})

    defaults = cli._profile_defaults(parser, "_inventato", {"tone"})

    assert defaults["tone"] == "neutral"
