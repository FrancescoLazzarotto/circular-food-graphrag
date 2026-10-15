"""Structural checks on the sheets, shared by ``accept.py`` and ``build.py``.

A problem is reported against the row that causes it, so acceptance can hold
that row back and the build report can point at it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from common import MAPPING_TYPES, STATUSES, normalize, quote_on_page, split_list

# Rows the application may use; ``rejected`` rows stay in the sheet as record.
LIVE = {"proposed", "auto_accepted", "approved", "corrected"}
QUOTE_IN_NOTES = re.compile(r"«([^»]+)»")


@dataclass(frozen=True)
class Problem:
    """One failed check.

    Attributes:
        sheet: Sheet name.
        key: Row key within the sheet.
        message: What is wrong.
    """

    sheet: str
    key: str
    message: str


def member_key(row: dict) -> str:
    """Key of a framework member, as used in ``version_mappings.tsv``.

    Args:
        row: A ``framework_members`` row.

    Returns:
        ``<framework_id>#<position>``.
    """
    return f"{row['framework_id']}#{row['position']}"


def row_key(sheet: str, row: dict) -> str:
    """Identity of a row within its sheet.

    Args:
        sheet: Sheet name.
        row: The row.

    Returns:
        A string key, unique per sheet when the sheet is consistent.
    """
    if sheet == "framework_members":
        return member_key(row)
    if sheet == "definitions":
        return f"{row['concept_id']}|{row['language']}|{row['document']}|{row['page']}"
    if sheet == "version_mappings":
        return f"{row['from']}>{row['to']}"
    return row["id"]


def check_all(sheets: dict[str, list[dict]], verify_quotes: bool = True) -> list[Problem]:
    """Run every structural check.

    Args:
        sheets: Sheet name → rows.
        verify_quotes: Also re-check quotes against the corpus text.

    Returns:
        All problems found; empty when the sheets are consistent.
    """
    problems: list[Problem] = []

    def add(sheet: str, row: dict, message: str) -> None:
        problems.append(Problem(sheet, row_key(sheet, row), message))

    for sheet, rows in sheets.items():
        seen: set[str] = set()
        for row in rows:
            key = row_key(sheet, row)
            if key in seen:
                add(sheet, row, "riga duplicata")
            seen.add(key)
            if row.get("status") not in STATUSES:
                add(sheet, row, f"stato non valido: {row.get('status')!r}")

    live = {
        sheet: [r for r in rows if r.get("status") in LIVE]
        for sheet, rows in sheets.items()
    }
    frameworks = {r["id"]: r for r in live.get("frameworks", [])}
    concepts = {r["id"]: r for r in live.get("concepts", [])}
    members = {member_key(r): r for r in live.get("framework_members", [])}

    for row in live.get("frameworks", []):
        if not row.get("name"):
            add("frameworks", row, "nome mancante")
        if not row.get("document") or not row.get("pages"):
            add("frameworks", row, "fonte mancante")
        for parent in split_list(row.get("derived_from", "")):
            if parent not in frameworks:
                add("frameworks", row, f"derived_from sconosciuto: {parent}")

    positions: dict[str, set[str]] = {}
    for row in live.get("framework_members", []):
        if row["framework_id"] not in frameworks:
            add("framework_members", row, "framework sconosciuto o non attivo")
        if not row.get("position"):
            add("framework_members", row, "posizione mancante")
        taken = positions.setdefault(row["framework_id"], set())
        if row["position"] in taken:
            add("framework_members", row, "posizione ripetuta nella stessa versione")
        taken.add(row["position"])
        if not (row.get("label_it") or row.get("label_en")):
            add("framework_members", row, "etichetta mancante")
        if not row.get("document") or not row.get("page") or not row.get("quote"):
            add("framework_members", row, "fonte o citazione mancante")
        elif verify_quotes and not quote_on_page(row["quote"], row["document"], row["page"]):
            add("framework_members", row, "citazione non trovata alla lettera nella pagina")
        if row.get("concept_id") and row["concept_id"] not in concepts:
            add("framework_members", row, f"concetto sconosciuto: {row['concept_id']}")

    labels: dict[tuple[str, str], str] = {}
    for row in live.get("concepts", []):
        if not (row.get("label_it") or row.get("label_en")):
            add("concepts", row, "etichetta mancante")
        for lang in ("it", "en"):
            label = normalize(row.get(f"label_{lang}", ""))
            if not label:
                continue
            other = labels.get((lang, label))
            if other:
                add("concepts", row, f"etichetta {lang} uguale a quella di {other}")
            labels[(lang, label)] = row["id"]
        for field in ("broader", "related"):
            for target in split_list(row.get(field, "")):
                if target not in concepts:
                    add("concepts", row, f"{field} sconosciuto: {target}")
    for concept_id in _broader_cycles(concepts):
        add("concepts", concepts[concept_id], "ciclo nella catena 'più generale di'")

    for row in live.get("definitions", []):
        if row["concept_id"] not in concepts:
            add("definitions", row, "concetto sconosciuto")
        if row.get("language") not in {"it", "en"}:
            add("definitions", row, "lingua non valida")
        if not row.get("quoted_definition") or not row.get("document") or not row.get("page"):
            add("definitions", row, "definizione senza fonte")
        elif verify_quotes and not quote_on_page(
            row["quoted_definition"], row["document"], row["page"]
        ):
            add("definitions", row, "citazione non trovata alla lettera nella pagina")

    for row in live.get("version_mappings", []):
        if row.get("type") not in MAPPING_TYPES:
            add("version_mappings", row, f"tipo non valido: {row.get('type')!r}")
        if row.get("type") == "new":
            if row.get("from"):
                add("version_mappings", row, "una voce nuova non ha origine")
        elif row.get("from") not in members:
            add("version_mappings", row, f"voce di partenza sconosciuta: {row.get('from')}")
        if row.get("to") not in members:
            add("version_mappings", row, f"voce di arrivo sconosciuta: {row.get('to')}")
        quote = QUOTE_IN_NOTES.search(row.get("notes", ""))
        if not row.get("document") or not row.get("page") or not quote:
            add("version_mappings", row, "corrispondenza senza fonte o citazione")
        elif verify_quotes and not quote_on_page(quote.group(1), row["document"], row["page"]):
            add("version_mappings", row, "citazione non trovata alla lettera nella pagina")

    for row in live.get("cases", []):
        if row.get("framework_id") not in frameworks:
            add("cases", row, f"framework sconosciuto: {row.get('framework_id')}")
        for code in split_list(row.get("strategies", "")):
            if f"{row.get('framework_id')}#{code}" not in members:
                add("cases", row, f"strategia sconosciuta nella versione: {code}")
        quotes = QUOTE_IN_NOTES.findall(row.get("notes", ""))
        if not row.get("document") or not row.get("page") or not quotes:
            add("cases", row, "caso senza fonte o citazione")
        elif verify_quotes and not all(quote_on_page(q, row["document"], row["page"]) for q in quotes):
            add("cases", row, "citazione non trovata alla lettera nella pagina")

    problems.extend(mapping_problems(live.get("version_mappings", [])))
    return problems


def mapping_problems(mappings: list[dict]) -> list[Problem]:
    """Correspondences that contradict each other.

    Two rules. A merge is one claim with two origins: an origin alone is a
    problem. And where a version is mapped to another both directly and
    through an intermediate version, the two routes must reach the same
    members; when they do not, every correspondence on both routes is
    reported, since nothing says which one is wrong.

    Args:
        mappings: Live ``version_mappings`` rows.

    Returns:
        Problems, one per offending row.
    """
    problems: list[Problem] = []
    merged: dict[str, list[dict]] = {}
    edges: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in mappings:
        if row.get("type") == "merged":
            merged.setdefault(row["to"], []).append(row)
        if row.get("type") == "new" or not row.get("from"):
            continue
        route = (row["from"].split("#")[0], row["to"].split("#")[0])
        edges.setdefault(route, {}).setdefault(row["from"], []).append(row)
    for rows in merged.values():
        if len(rows) < 2:
            problems.append(Problem("version_mappings", row_key("version_mappings", rows[0]), "unione con una sola origine"))

    flagged: set[str] = set()
    for (a, c), direct_edges in edges.items():
        for (first, b), first_edges in edges.items():
            if first != a or b == c or (b, c) not in edges:
                continue
            second_edges = edges[(b, c)]
            for source, direct in direct_edges.items():
                hops = [(e1, e2) for e1 in first_edges.get(source, []) for e2 in second_edges.get(e1["to"], [])]
                if not hops:
                    continue
                if {e["to"] for e in direct} == {e2["to"] for _, e2 in hops}:
                    continue
                for row in direct + [e for hop in hops for e in hop]:
                    key = row_key("version_mappings", row)
                    if key not in flagged:
                        flagged.add(key)
                        problems.append(Problem("version_mappings", key, f"in contraddizione con il percorso {a} → {b} → {c}"))
    return problems


def _broader_cycles(concepts: dict[str, dict]) -> list[str]:
    in_cycle: list[str] = []
    for start in concepts:
        stack = list(split_list(concepts[start].get("broader", "")))
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == start:
                in_cycle.append(start)
                break
            if current in seen or current not in concepts:
                continue
            seen.add(current)
            stack.extend(split_list(concepts[current].get("broader", "")))
    return in_cycle
