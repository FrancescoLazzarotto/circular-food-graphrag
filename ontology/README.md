# Circular Food Frameworks Ontology

A small, curated layer (the 3C, the R strategies with their versions, the waste
and food-waste hierarchies, supply chains, cases), kept separate from the
extracted graph and linked to it. Exchange format: SKOS, plus two properties of
our own (position, version). The sheets in `sources/` are the source of truth;
everything else is regenerated from them.

## Layout

| folder | contents |
|---|---|
| `competency_questions/` | what the ontology must be able to answer, with the expected answer |
| `sources/` | curated sheets: concepts, definitions, frameworks, members, version mappings, cases |
| `external/` | mappings to AGROVOC and EuroVoc (identifiers only) |
| `rules/` | hand-written rules (priority, broader chains, case classification) |
| `drafts/` | model proposals, before review |
| `review/` | expert review sheets, one per session, following `sheet_template.tsv` |
| `build/` | generated files (SKOS, graph load); never edited by hand |
| `scripts/` | conversion, checks, loading |
| `tests/` | competency questions as tests |

## Shared values

- `status`: `proposed`, `approved`, `corrected`, `rejected`.
- `decision` (review sheets): `ok`, `fix`, `reject`.
- `type` in `version_mappings.tsv`: `same`, `merged`, `new`, `renamed`.
- `match_type` in `external/mappings.tsv`: `exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch`.
- Every definition, member and case carries document and page; quotes are
  copied verbatim from the corpus text.

## Workflow

0. Competency questions → `competency_questions/`, approved by the expert.
1. Candidate concepts and definitions from the corpus and the graph → `drafts/`.
2. External mappings → `external/`.
3. Structure drafted by the model, each item with its quote → `drafts/`.
4. Automatic checks.
5. Expert review in small batches → `review/`, outcomes copied into `sources/`.
6. Build and load → `build/`.
7. Test against the competency questions → `tests/`.
