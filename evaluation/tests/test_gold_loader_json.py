"""Tests for the JSON gold loader and the run↔gold join by query_id.

The join is the most delicate point of the evaluation pipeline: a join that
fails quietly produces wrong numbers with nothing to flag them. These tests
therefore assert not only that the happy path works, but that every degraded
path is *loud*.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from evalkit.io.dataset import build_dataset, rows_from_csv, rows_to_csv
from evalkit.io.gold_loader import (
    is_json_gold,
    load_gold,
    load_gold_json,
)
from evalkit.models import GoldEntity, GoldQuery

REPO_ROOT = Path(__file__).resolve().parents[2]

# The gold sets live under evaluation/gold/. Skip (never silently pass) if
# the file is absent.
_GOLD_CANDIDATES = (
    REPO_ROOT / "evaluation" / "gold" / "gold_circular_v1.json",
    REPO_ROOT / "evaluation" / "gold" / "gold.json",
)


def _find_real_gold() -> Path | None:
    """The first gold file present, or None."""
    return next((p for p in _GOLD_CANDIDATES if p.is_file()), None)


REAL_GOLD = _find_real_gold()
requires_real_gold = pytest.mark.skipif(
    REAL_GOLD is None,
    reason=f"gold not found in any of: {[str(p) for p in _GOLD_CANDIDATES]}",
)


# ─── fixtures ────────────────────────────────────────────────────────────────

def _entity(
    label: str = "whey",
    uri: str | None = "http://aims.fao.org/aos/agrovoc/c_8376",
    mapping_status: str = "exact",
    alt_labels: list[str] | None = None,
) -> dict:
    """A raw gold entity dict."""
    return {
        "label": label,
        "normalised_label": label.lower(),
        "alt_labels": alt_labels if alt_labels is not None else ["siero"],
        "uri": uri,
        "vocabulary": "AGROVOC",
        "mapping_status": mapping_status,
        "aligned_to": None,
        "note": "informational prose, never scored",
    }


def _query(
    query_id: str = "Q01",
    query: str = "What is whey?",
    distractor: bool = False,
    entities: list[dict] | None = None,
) -> dict:
    """A raw gold query dict with one entity by default."""
    return {
        "query_id": query_id,
        "query_type": "distractor" if distractor else "factual_simple",
        "query": query,
        "expected_answer": f"Answer to {query_id}.",
        "expected_entities": entities if entities is not None else [_entity()],
        "expected_relations": [],
        "source_verified": [{"doc": "d", "page": "1"}],
        "scoring": {"distractor_expected": distractor},
    }


def _write_gold(path: Path, queries: list[dict], meta: dict | None = None) -> Path:
    """Write a JSON gold of `queries` and return its path."""
    payload = {"_meta": meta or {"n_queries": len(queries)}, "queries": queries}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_run(run_dir: Path, rows: list[dict]) -> Path:
    """Write `rows` as a run's results.jsonl and return the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return run_dir


def _row(question: str, query_id: str | None = "Q01", strategy: str = "default") -> dict:
    """A raw run result row, with a query_id unless it is None."""
    row = {
        "strategy": strategy,
        "question": question,
        "answer": "some answer",
        "latency_ms": 10.0,
        "contexts": ["ctx"],
        "retrieved_triples": [],
        "retrieved_entities": ["whey"],
        "metadata": {"model_id": "m", "framework": "graphrag", "run_index": 1},
    }
    if query_id is not None:
        row["query_id"] = query_id
    return row


# ─── loader: happy path on the real gold ─────────────────────────────────────

@requires_real_gold
def test_load_gold_json_real_gold_keys_by_query_id():
    gold = load_gold_json(REAL_GOLD)

    assert len(gold) == 30
    assert list(gold)[:3] == ["Q01", "Q02", "Q03"]
    assert all(qid == q.query_id for qid, q in gold.items())
    assert all(isinstance(q, GoldQuery) for q in gold.values())


@requires_real_gold
def test_load_gold_json_real_gold_entity_counts_match_protocol():
    gold = load_gold_json(REAL_GOLD)

    total = sum(len(q.expected_entities) for q in gold.values())
    grounding = sum(len(q.grounding_entities) for q in gold.values())
    distractors = [q.query_id for q in gold.values() if q.distractor_expected]

    # Counts of the frozen gold.
    assert total == 88
    assert grounding == 23
    assert distractors == ["Q14", "Q15", "Q29", "Q30"]


@requires_real_gold
def test_load_gold_json_real_gold_parses_entities_as_objects():
    gold = load_gold_json(REAL_GOLD)
    capital = gold["Q01"].expected_entities[0]

    assert isinstance(capital, GoldEntity)
    assert capital.normalised_label == "capital"
    assert "capitale" in capital.alt_labels
    assert capital.uri == "urn:ceff:Capital"
    assert capital.mapping_status == "benchmark_local_extension"
    assert capital.counts_at_grounding_level is False


def test_load_gold_json_parses_scoring_and_alt_labels(tmp_path):
    path = _write_gold(tmp_path / "g.json", [_query(), _query("Q02", distractor=True)])
    gold = load_gold_json(path)

    assert gold["Q01"].distractor_expected is False
    assert gold["Q02"].distractor_expected is True
    entity = gold["Q01"].expected_entities[0]
    assert entity.alt_labels == ("siero",)
    assert entity.counts_at_grounding_level is True
    assert entity.vocabulary == "AGROVOC"


def test_load_gold_json_accepts_bare_list(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps([_query()]), encoding="utf-8")

    assert list(load_gold_json(path)) == ["Q01"]


def test_is_json_gold_discriminates_csv_from_json(tmp_path):
    assert is_json_gold(Path("gold.json")) is True
    assert is_json_gold(Path("gold_circular_v1_silver.csv")) is False


# ─── loader: failure modes must raise or warn, never pass silently ───────────

def test_load_gold_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gold_json(tmp_path / "nope.json")


def test_load_gold_json_invalid_json_raises(tmp_path):
    path = tmp_path / "g.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_gold_json(path)


def test_load_gold_json_duplicate_query_id_raises(tmp_path):
    # A duplicate id would silently drop a query from the benchmark.
    path = _write_gold(tmp_path / "g.json", [_query("Q01"), _query("Q01", query="other")])

    with pytest.raises(ValueError, match="Duplicate query_id"):
        load_gold_json(path)


def test_load_gold_json_missing_query_id_raises(tmp_path):
    raw = _query()
    del raw["query_id"]
    path = _write_gold(tmp_path / "g.json", [raw])

    with pytest.raises(ValueError, match="without 'query_id'"):
        load_gold_json(path)


def test_load_gold_json_no_queries_raises(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"_meta": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="no 'queries'"):
        load_gold_json(path)


def test_load_gold_json_warns_on_unknown_mapping_status(tmp_path, caplog):
    path = _write_gold(
        tmp_path / "g.json", [_query(entities=[_entity(mapping_status="exacte")])]
    )

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        gold = load_gold_json(path)

    assert "unknown mapping_status" in caplog.text
    # A typo must not quietly become a grounding-level entity.
    assert gold["Q01"].grounding_entities == ()


def test_load_gold_json_warns_on_exact_without_uri(tmp_path, caplog):
    path = _write_gold(
        tmp_path / "g.json", [_query(entities=[_entity(uri=None, mapping_status="exact")])]
    )

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        load_gold_json(path)

    assert "no URI" in caplog.text


def test_load_gold_json_warns_when_meta_count_disagrees(tmp_path, caplog):
    path = _write_gold(tmp_path / "g.json", [_query()], meta={"n_queries": 30})

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        load_gold_json(path)

    assert "n_queries=30" in caplog.text


# ─── join: query_id is primary ───────────────────────────────────────────────

def test_build_dataset_joins_by_query_id(tmp_path, caplog):
    gold = _write_gold(tmp_path / "g.json", [_query("Q01"), _query("Q02", query="What is scotta?")])
    run = _write_run(
        tmp_path / "run",
        [_row("What is whey?", "Q01"), _row("What is scotta?", "Q02")],
    )

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        rows = build_dataset([run], gold_path=gold)

    assert [r.question_id for r in rows] == ["Q01", "Q02"]
    assert [r.gold_query.query_id for r in rows] == ["Q01", "Q02"]
    assert [r.ground_truth for r in rows] == ["Answer to Q01.", "Answer to Q02."]
    assert all(r.skip_reason == "" for r in rows)
    # A clean join must be silent.
    assert "FALLBACK" not in caplog.text
    assert "NOT CLEAN" not in caplog.text


def test_build_dataset_join_ignores_question_text_when_id_matches(tmp_path):
    """The id wins: a reworded question still joins to its gold query."""
    gold = _write_gold(tmp_path / "g.json", [_query("Q01", query="What is whey?")])
    run = _write_run(tmp_path / "run", [_row("Totally reworded question?!", "Q01")])

    rows = build_dataset([run], gold_path=gold)

    assert rows[0].gold_query.query_id == "Q01"
    assert rows[0].ground_truth == "Answer to Q01."


def test_build_dataset_populates_expected_entities_as_gold_entities(tmp_path):
    gold = _write_gold(tmp_path / "g.json", [_query("Q01")])
    run = _write_run(tmp_path / "run", [_row("What is whey?", "Q01")])

    rows = build_dataset([run], gold_path=gold)

    assert all(isinstance(e, GoldEntity) for e in rows[0].expected_entities)
    assert rows[0].expected_entities[0].normalised_label == "whey"


def test_build_dataset_marks_distractor_rows(tmp_path):
    gold = _write_gold(tmp_path / "g.json", [_query("Q14", distractor=True)])
    run = _write_run(tmp_path / "run", [_row("What is whey?", "Q14")])

    rows = build_dataset([run], gold_path=gold)

    assert rows[0].is_distractor is True


def test_build_dataset_does_not_put_prose_relations_into_gold_triples(tmp_path):
    """expected_relations are prose: they must not reach triple metrics."""
    raw = _query("Q01")
    raw["expected_relations"] = ["IndustrialSymbiosis operationalises CoEvolution."]
    gold = _write_gold(tmp_path / "g.json", [raw])
    run = _write_run(tmp_path / "run", [_row("What is whey?", "Q01")])

    rows = build_dataset([run], gold_path=gold)

    assert rows[0].gold_triples == []
    assert rows[0].gold_query.expected_relations == (
        "IndustrialSymbiosis operationalises CoEvolution.",
    )


# ─── join: the fallback must be loud ─────────────────────────────────────────

def test_build_dataset_text_fallback_warns_loudly(tmp_path, caplog):
    gold = _write_gold(tmp_path / "g.json", [_query("Q01", query="What is whey?")])
    # A legacy run: no query_id field at all.
    run = _write_run(tmp_path / "run", [_row("What is whey?", query_id=None)])

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        rows = build_dataset([run], gold_path=gold)

    # It still joins — but never quietly.
    assert rows[0].gold_query.query_id == "Q01"
    assert "GOLD JOIN FALLBACK" in caplog.text
    assert "GOLD JOIN NOT CLEAN" in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_build_dataset_reports_fallback_total(tmp_path, caplog):
    queries = [_query(f"Q{i:02d}", query=f"Question {i}?") for i in range(1, 4)]
    gold = _write_gold(tmp_path / "g.json", queries)
    # Same 3 questions under 2 strategies = 6 rows, none carrying an id.
    rows_raw = [
        _row(f"Question {i}?", query_id=None, strategy=s)
        for s in ("default", "text_only")
        for i in range(1, 4)
    ]
    run = _write_run(tmp_path / "run", rows_raw)

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        build_dataset([run], gold_path=gold)

    assert "GOLD JOIN FALLBACK TOTAL: 6 rows over 3 distinct questions" in caplog.text


def test_build_dataset_warns_on_query_id_absent_from_gold(tmp_path, caplog):
    gold = _write_gold(tmp_path / "g.json", [_query("Q01", query="What is whey?")])
    run = _write_run(tmp_path / "run", [_row("What is whey?", "Q99")])

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        rows = build_dataset([run], gold_path=gold)

    assert "query_ids absent from the gold" in caplog.text
    assert "'Q99'" in caplog.text
    # It fell back on text, so it still joined — loudly.
    assert rows[0].gold_query.query_id == "Q01"
    assert "GOLD JOIN FALLBACK" in caplog.text


def test_build_dataset_warns_with_missing_gold_ids_when_coverage_incomplete(tmp_path, caplog):
    queries = [_query(f"Q{i:02d}", query=f"Question {i}?") for i in range(1, 4)]
    gold = _write_gold(tmp_path / "g.json", queries)
    run = _write_run(tmp_path / "run", [_row("Question 1?", "Q01")])

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        build_dataset([run], gold_path=gold)

    assert "GOLD COVERAGE" in caplog.text
    assert "['Q02', 'Q03']" in caplog.text


def test_build_dataset_unmatched_row_warns_and_has_no_gold(tmp_path, caplog):
    gold = _write_gold(tmp_path / "g.json", [_query("Q01", query="What is whey?")])
    run = _write_run(tmp_path / "run", [_row("An unrelated question?", query_id=None)])

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        rows = build_dataset([run], gold_path=gold)

    assert rows[0].gold_query is None
    assert rows[0].has_gold is False
    assert rows[0].skip_reason == "no_gold"
    assert "matched NO gold query" in caplog.text


def test_build_dataset_excludes_ambiguous_questions_from_text_fallback(tmp_path, caplog):
    # Two gold queries with the same question text: the fallback must refuse to guess.
    gold = _write_gold(
        tmp_path / "g.json",
        [_query("Q01", query="Same text?"), _query("Q02", query="Same text?")],
    )
    run = _write_run(tmp_path / "run", [_row("Same text?", query_id=None)])

    with caplog.at_level(logging.WARNING, logger="graphrag"):
        rows = build_dataset([run], gold_path=gold)

    assert "share the same normalised question text" in caplog.text
    assert rows[0].gold_query is None


# ─── the CSV gold path must keep working (silver dev set) ────────────────────

def test_load_gold_csv_still_works(tmp_path):
    path = tmp_path / "silver.csv"
    path.write_text(
        "question,ground_truth\nChi ha diretto The Matrix?,Le sorelle Wachowski\n",
        encoding="utf-8",
    )

    gold = load_gold(path)

    assert gold["chi ha diretto the matrix"]["ground_truth"] == "Le sorelle Wachowski"


def test_build_dataset_with_csv_gold_joins_on_text(tmp_path):
    gold_csv = tmp_path / "silver.csv"
    gold_csv.write_text(
        "question,ground_truth,question_id\nWhat is whey?,Whey is a by-product.,S1\n",
        encoding="utf-8",
    )
    run = _write_run(tmp_path / "run", [_row("What is whey?", query_id=None)])

    rows = build_dataset([run], gold_path=gold_csv)

    assert rows[0].ground_truth == "Whey is a by-product."
    assert rows[0].question_id == "S1"
    assert rows[0].gold_query is None  # CSV gold carries no GoldQuery


# ─── CSV round-trip must not lose the gold query ─────────────────────────────

def test_rows_csv_roundtrip_preserves_gold_query_and_entities(tmp_path):
    gold = _write_gold(
        tmp_path / "g.json", [_query("Q14", distractor=True), _query("Q01")]
    )
    run = _write_run(
        tmp_path / "run",
        [_row("What is whey?", "Q14"), _row("What is whey?", "Q01")],
    )
    rows = build_dataset([run], gold_path=gold)

    out = tmp_path / "eval.csv"
    rows_to_csv(rows, out)
    reloaded = rows_from_csv(out)

    # Without gold_query_json, is_distractor would silently degrade to False here.
    assert [r.is_distractor for r in reloaded] == [True, False]
    assert [r.gold_query.query_id for r in reloaded] == ["Q14", "Q01"]
    assert all(isinstance(e, GoldEntity) for r in reloaded for e in r.expected_entities)
    assert reloaded[0].expected_entities[0].uri == "http://aims.fao.org/aos/agrovoc/c_8376"


def test_rows_csv_roundtrip_of_legacy_string_entities_is_unchanged(tmp_path):
    """Legacy rows (entities as plain strings) must survive the round-trip as-is."""
    gold_csv = tmp_path / "silver.csv"
    gold_csv.write_text(
        'question,ground_truth,expected_entities_json\n'
        '"What is whey?",Answer,"[""whey"", ""scotta""]"\n',
        encoding="utf-8",
    )
    run = _write_run(tmp_path / "run", [_row("What is whey?", query_id=None)])
    rows = build_dataset([run], gold_path=gold_csv)

    out = tmp_path / "eval.csv"
    rows_to_csv(rows, out)
    reloaded = rows_from_csv(out)

    assert reloaded[0].expected_entities == ["whey", "scotta"]
    assert reloaded[0].gold_query is None
