"""Draw a sample of ``auto_accepted`` items for reading against the source.

The sample is spread over the sheets in proportion to their size, with a
fixed seed so the same sheets give the same sample. It is written in the
format of ``review/sheet_template.tsv``; ``decision`` and ``expert_note`` are
left for the reader.

Usage::

    /mnt/storage/flazzarotto/venvs/ontology/bin/python ontology/scripts/sample.py --size 50
"""

from __future__ import annotations

import argparse
import datetime as dt
import random

from checks import QUOTE_IN_NOTES, row_key
from common import REVIEW_DIR, SHEET_COLUMNS, read_sheet, write_tsv

TEMPLATE = ["row", "sheet", "id", "proposal", "quote", "document", "page", "reason", "decision", "expert_note"]


def describe(sheet: str, row: dict) -> tuple[str, str, str, str]:
    """Proposal, quote, document and page of a row, as a reader needs them.

    Args:
        sheet: Sheet name.
        row: The row.

    Returns:
        ``(proposal, quote, document, page)``.
    """
    notes_quote = QUOTE_IN_NOTES.search(row.get("notes", ""))
    if sheet == "frameworks":
        return (
            f"{row['name']} — versione {row['version']}; autore {row['author'] or '?'} {row['year']}; deriva da {row['derived_from'] or '-'}",
            notes_quote.group(1) if notes_quote else "",
            row["document"],
            row["pages"],
        )
    if sheet == "framework_members":
        return (
            f"{row['framework_id']} {row['position']} = {row['label_it']} / {row['label_en'] or '-'}; famiglia {row['family'] or '-'}; {row['notes']}",
            row["quote"],
            row["document"],
            row["page"],
        )
    if sheet == "definitions":
        return (f"definizione di {row['concept_id']} ({row['language']})", row["quoted_definition"], row["document"], row["page"])
    if sheet == "version_mappings":
        return (f"{row['from'] or '(nessuna origine)'} → {row['to']}: {row['type']}", notes_quote.group(1) if notes_quote else "", row["document"], row["page"])
    return (
        f"concetto {row['label_it']} / {row['label_en']}; più generale {row['broader'] or '-'}; sinonimi {row['synonyms_it'] or '-'}; {row['notes']}",
        "",
        "",
        "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    pool = [
        (sheet, row)
        for sheet in SHEET_COLUMNS
        for row in read_sheet(sheet)
        if row.get("status") == "auto_accepted"
    ]
    rng = random.Random(args.seed)
    by_sheet: dict[str, list] = {}
    for sheet, row in pool:
        by_sheet.setdefault(sheet, []).append(row)
    size = min(args.size, len(pool))
    quota = {s: round(size * len(rows) / len(pool)) for s, rows in by_sheet.items()}
    while sum(quota.values()) > size:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < size:
        quota[max(by_sheet, key=lambda s: len(by_sheet[s]) - quota[s])] += 1

    rows = []
    for sheet, members in by_sheet.items():
        for row in rng.sample(members, quota[sheet]):
            proposal, quote, document, page = describe(sheet, row)
            rows.append(
                {
                    "sheet": sheet,
                    "id": row_key(sheet, row),
                    "proposal": proposal,
                    "quote": quote,
                    "document": document,
                    "page": page,
                    "reason": "campione auto_accepted",
                }
            )
    for number, row in enumerate(rows, 1):
        row["row"] = number
    out = REVIEW_DIR / f"sample_auto_accepted_{dt.date.today().isoformat()}.tsv"
    write_tsv(out, TEMPLATE, rows)
    print(f"{len(rows)} voci su {len(pool)} auto_accepted → {out}")


if __name__ == "__main__":
    main()
