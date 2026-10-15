# Circular Food Frameworks Ontology

A small, curated layer (the 3C, the R strategies with their versions, the waste
and food-waste hierarchies, supply chains, cases), kept separate from the
extracted graph and linked to it by name. Exchange format: SKOS, plus two
properties of our own (position, version). The sheets in `sources/` are the
source of truth; everything else is regenerated from them.

The ontology is never loaded into Neo4j. The application reads
`build/ontology.json` in memory through `graphrag.ontology`.

## Layout

| folder | contents |
|---|---|
| `competency_questions/` | what the ontology must be able to answer, with the expected answer |
| `sources/` | curated sheets: concepts, definitions, frameworks, members, version mappings, cases |
| `external/` | mappings to AGROVOC and EuroVoc (identifiers only) |
| `rules/` | hand-written rules (priority, broader chains, case classification) and `shapes.ttl` (SHACL) |
| `drafts/` | model drafts (`run_a.json`, `run_b.json`) and `pending.tsv`: what was not accepted, with the reason |
| `review/` | review sheets, following `sheet_template.tsv` |
| `build/` | generated files (`ontology.ttl`, `ontology.json`, `report.md`); never edited by hand |
| `scripts/` | drafting, acceptance, checks and build |
| `tests/` | unused: the competency questions run as tests in the repository suite, `tests/test_ontology.py` |

## Shared values

- `status`: `proposed`, `auto_accepted`, `approved`, `corrected`, `rejected`.
  `auto_accepted` means "taken from the source, not yet reviewed": the
  application uses it and says so. `proposed` rows are modelling choices
  waiting for the expert and are used only to look names up.
- `decision` (review sheets): `ok`, `fix`, `reject`.
- `type` in `version_mappings.tsv`: `same`, `merged`, `new`, `renamed`.
- `match_type` in `external/mappings.tsv`: `exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch`.
- Every definition, member and case carries document and page; quotes are
  copied verbatim from the corpus text. Sheets without a quote column
  (`frameworks`, `version_mappings`) keep the quote in `notes`, between « ».
- `framework_members.position`: `R0`…`R9` and `R+` for the R strategies,
  `1`…`3` for the 3C (order of presentation, not priority). A position read
  from the order of a list rather than written in the text says so in `notes`.
- `version_mappings`: `from` and `to` are `<framework_id>#<position>`; a `new`
  row has an empty `from`.
- Framework ids start with the id of the concept that groups their versions
  (`r_errando` → `r`, `3c_fassio_2021` → `3c`).

## Workflow

The machine builds and accepts what it can verify; the expert approves at the
end, in a single sheet, when available. Nothing waits for that step.

0. Competency questions → `competency_questions/`, written from the session
   logs, the gold set and the sources; they end up in the final sheet.
1. Candidate concepts and definitions from the corpus and the graph → `drafts/`.
2. External mappings → `external/`.
3. Structure drafted by the model twice, with different instructions
   (`scripts/draft_frameworks.py`) → `drafts/`.
4. Automatic acceptance (`scripts/accept.py`): an item enters `sources/` as
   `auto_accepted` only if its quote is found verbatim on the stated page, the
   source is a department publication or an official report, both drafts
   agree, and the checks pass. Everything else stays in `drafts/pending.tsv`
   with the reason.
5. Build and checks (`scripts/build.py`): Python checks, SHACL and Skosify,
   then `build/`. The build fails if a check fails.
6. Sample: the developer reads 50 `auto_accepted` items against the source →
   `review/`.
7. Expert approval, at the end, in one sheet of 30-50 rows → `review/`,
   decisions copied into `sources/` as `approved`, `corrected` or `rejected`.
   Those rows are never overwritten by a later acceptance run.
8. Tests against the competency questions (`tests/test_ontology.py` in the
   repository test suite).

## Running

The scripts use a separate virtualenv; `graphllm` needs nothing new.

```bash
# once
/mnt/storage/flazzarotto/.conda/envs/graphllm/bin/python -m venv /mnt/storage/flazzarotto/venvs/ontology
/mnt/storage/flazzarotto/venvs/ontology/bin/pip install rdflib pyshacl skosify

PY=/mnt/storage/flazzarotto/venvs/ontology/bin/python
$PY ontology/scripts/draft_frameworks.py   # needs the generator on :8001
$PY ontology/scripts/accept.py
$PY ontology/scripts/build.py
$PY ontology/scripts/sample.py             # 50 auto_accepted items to read
```

The corpus text (`stage0_documents.json`) and the graph backups are read from
their default locations; `ONTOLOGY_STAGE0` and `ONTOLOGY_BACKUP_ROOT` override
them. Links to the extracted graph are resolved by normalised name and alias
against the most recent backup, never by node id.
