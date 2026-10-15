"""Move drafted items into ``sources/`` when they pass all four criteria.

An item becomes ``auto_accepted`` only if

1. its quote occurs verbatim on the stated page (the normalisation of the
   answer-time quote gate) and names the item;
2. its source is a department publication or an official report;
3. both drafting runs agree on it;
4. the sheets stay consistent with it (``checks.py``).

The two runs must agree on the content (position, labels, family, type); the
quote kept is the first of the two that passes every quote check. Everything
else is written to ``drafts/pending.tsv`` with the reason. Rows already
``approved``, ``corrected`` or ``rejected`` by the expert are kept as they are
and win over a new draft with the same key.

Usage::

    /mnt/storage/flazzarotto/venvs/ontology/bin/python ontology/scripts/accept.py
"""

from __future__ import annotations

import json
import re

from checks import check_all, member_key, row_key
from common import (
    DRAFTS_DIR,
    SHEET_COLUMNS,
    contains_term,
    graph_nodes_for,
    normalize,
    page_text,
    quote_on_page,
    read_sheet,
    split_list,
    write_sheet,
    write_tsv,
)
from draft_frameworks import C3_TASK, CASE_TASK, LIST_TASKS, MAPPING_TASKS, R_TASKS, SOURCE_KINDS

AUTO = "auto_accepted"
EXPERT_STATUSES = {"approved", "corrected", "rejected"}
# Oldest first: a concept shared by several versions takes its labels from
# the version the others derive from.
R_ORDER = ["r_potting_2017", "r_errando", "r_nodes_2024"]
DERIVED_FROM = {
    "r_potting_2017": "",
    "r_errando": "r_potting_2017",
    "r_nodes_2024": "r_potting_2017; r_errando",
}

# Lookup entries that group the versions of a framework; their ids are the
# prefix of the framework ids. They are a modelling choice, not a reading of
# a source, so they stay ``proposed`` until the expert approves them.
FAMILY_CONCEPTS = [
    {
        "id": "r",
        "label_it": "Strategie R",
        "label_en": "R strategies",
        "synonyms_it": "10R; 9R; 11R; paradigma delle 10R; paradigma delle 9R; paradigma R; le R",
        "synonyms_en": "9Rs; 10Rs; R-strategies",
        "notes": "raggruppa le versioni r_*",
    },
    {
        "id": "3c",
        "label_it": "3C della Circular Economy for Food",
        "label_en": "3 C's of the Circular Economy for Food",
        "synonyms_it": "3C; tre C; le 3C; 3C della CEFF; 3C dell'economia circolare per il cibo",
        "synonyms_en": "3Cs; three C's; 3 C's",
        "notes": "raggruppa le versioni 3c_*",
    },
    {
        "id": "wh",
        "label_it": "Gerarchia dei rifiuti",
        "label_en": "Waste hierarchy",
        "synonyms_it": "gerarchia di gestione dei rifiuti; criteri di priorità nella gestione dei rifiuti",
        "synonyms_en": "waste management hierarchy",
        "notes": "raggruppa le versioni wh_*",
    },
]


class Ledger:
    """Accepted rows per sheet and the pending items with their reason."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {name: [] for name in SHEET_COLUMNS}
        self.pending: list[dict] = []

    def accept(self, sheet: str, row: dict) -> None:
        self.rows[sheet].append({**row, "status": AUTO})

    def hold(self, sheet: str, key: str, proposal: dict, reason: str) -> None:
        self.pending.append(
            {
                "sheet": sheet,
                "key": key,
                "proposal": json.dumps(proposal, ensure_ascii=False),
                "reason": reason,
            }
        )


def _same(a, b) -> bool:
    return normalize(str(a or "")) == normalize(str(b or ""))


def _slug(text: str) -> str:
    return normalize(text).replace(" ", "_")


def _label(text: str) -> str:
    # Table headings are in capitals ("R4 = REFILL"); the label is the word.
    return text.capitalize() if text.isupper() else text


def _locate(quote: str, document: str, page: int) -> str | None:
    """Page field under which the quote is verbatim, or None.

    A quote that starts on the stated page and runs onto the next one is
    found on the two pages together.
    """
    if quote_on_page(quote, document, page):
        return str(page)
    head = " ".join(normalize(quote).split()[:6])
    if head and head in normalize(page_text(document, page)) and quote_on_page(quote, document, f"{page}-{page + 1}"):
        return f"{page}-{page + 1}"
    return None


def _quote_problem(quote: str, document: str, page, terms: list[str]) -> str:
    if _locate(quote, document, int(page)) is None:
        return f"citazione non trovata alla lettera in {document} p.{page}"
    named = [t for t in terms if t]
    if named and not any(contains_term(quote, t) for t in named):
        return f"la citazione non nomina {' / '.join(named)}"
    return ""


def _pick_quote(candidates: list[dict], document: str, quote_field: str, page_field: str, terms_of) -> tuple[dict | None, str]:
    """First candidate whose quote passes; otherwise the reason of the first."""
    reasons = []
    for candidate in candidates:
        reason = _quote_problem(candidate[quote_field], document, candidate[page_field], terms_of(candidate))
        if not reason:
            return candidate, ""
        reasons.append(reason)
    return None, reasons[0] if reasons else "nessuna citazione"


def _source_problem(document: str) -> str:
    return "" if document in SOURCE_KINDS else f"fonte non del dipartimento né ufficiale: {document}"


def _position_written(quote: str, position: str) -> bool:
    if position == "R+":
        return bool(re.search(r"R\s?\+", quote))
    return contains_term(quote, position)


def _order_problem(quote: str, members: list[dict], index: int) -> str:
    """A position read from the order of a list needs that order in the quote."""
    text = f" {normalize(quote)} "

    def where(member: dict) -> int:
        return text.find(f" {normalize(member['label_it'])} ")

    here = where(members[index])
    before = members[index - 1] if index > 0 else None
    after = members[index + 1] if index + 1 < len(members) else None
    if before is not None and where(before) >= 0:
        return "" if where(before) < here else "posizione dedotta ma l'ordine nella citazione non torna"
    if after is not None and where(after) >= 0:
        return "" if here < where(after) else "posizione dedotta ma l'ordine nella citazione non torna"
    return "posizione dedotta ma l'ordine non è nella citazione"


def _label_en_problem(label_en: str, document: str, pages: list[int]) -> str:
    if not label_en or any(contains_term(page_text(document, p), label_en) for p in pages):
        return ""
    return f"etichetta inglese {label_en!r} non presente nelle pagine"


def _agreed_name(head_a: dict, head_b: dict) -> str:
    """The shorter of the two names when one contains the other, else empty.

    The runs copy the same span with or without a footnote mark or an
    article ("Le 11R" / "11R"); that is agreement, a different name is not.
    """
    a, b = head_a["name"], head_b["name"]
    shorter, longer = sorted((a, b), key=lambda n: len(normalize(n)))
    return shorter if contains_term(longer, shorter) else ""


def _framework_head(head_a: dict, head_b: dict, document: str) -> str:
    reason = _source_problem(document)
    name = _agreed_name(head_a, head_b)
    if not reason and not name:
        reason = f"nome discorde: {head_a['name']!r} / {head_b['name']!r}"
    if not reason and not contains_term(head_a["name_quote"], name):
        reason = f"il nome {name!r} non è nella citazione"
    return reason or _quote_problem(head_a["name_quote"], document, head_a["name_page"], [])


def accept_versions(
    ledger: Ledger, drafts_a: dict, drafts_b: dict, tasks: dict, derived_from: dict[str, str]
) -> dict[str, dict[str, dict]]:
    """Accept framework versions drafted as ordered lists, with their members.

    Args:
        ledger: Where accepted and held items go.
        drafts_a: First run's drafts, keyed by framework id.
        drafts_b: Second run's drafts, keyed by framework id.
        tasks: The drafting tasks, keyed by framework id.
        derived_from: Framework id → the versions it derives from.

    Returns:
        Framework id → position → accepted member row.
    """
    accepted: dict[str, dict[str, dict]] = {}
    for framework_id, task in tasks.items():
        draft_a, draft_b = drafts_a[framework_id], drafts_b[framework_id]
        document = task["document"]
        head_a, head_b = draft_a["framework"], draft_b["framework"]
        framework_row = {
            "id": framework_id,
            "name": _agreed_name(head_a, head_b) or head_a["name"],
            "version": task["version"],
            "derived_from": derived_from.get(framework_id, ""),
            "author": head_a["author"] if _same(head_a["author"], head_b["author"]) else "",
            "year": head_a["year"] if head_a["year"] == head_b["year"] else "",
            "document": document,
            "pages": ", ".join(str(p) for p in task["pages"]),
            "notes": f"nome: p.{head_a['name_page']} «{head_a['name_quote']}»",
        }
        reason = _framework_head(head_a, head_b, document)
        if reason:
            ledger.hold("frameworks", framework_id, framework_row, reason)
            for member in draft_a["members"]:
                ledger.hold("framework_members", f"{framework_id}#{member['position']}", member, "framework in sospeso")
            continue
        ledger.accept("frameworks", framework_row)

        members_a = draft_a["members"]
        by_b = {m["position"]: m for m in draft_b["members"]}
        accepted[framework_id] = {}
        for index, member in enumerate(members_a):
            key = f"{framework_id}#{member['position']}"
            other = by_b.get(member["position"])
            if other is None:
                ledger.hold("framework_members", key, member, "presente in una sola esecuzione")
                continue
            # Agreement is checked field by field: a field the runs disagree
            # on is left empty and held, the rest of the item still counts.
            fields = ("label_it", "label_en", "family")
            agreed = {f: member[f] if _same(member[f], other[f]) else "" for f in fields}
            split = [f for f in fields if not _same(member[f], other[f])]
            for field in split:
                ledger.hold(
                    "framework_members", f"{key}|{field}",
                    {field: [member[field], other[field]]}, "campo discorde fra le esecuzioni, lasciato vuoto",
                )
            row = {
                "framework_id": framework_id,
                "position": member["position"],
                "label_it": _label(agreed["label_it"]),
                "label_en": _label(agreed["label_en"]),
                "family": agreed["family"],
                "document": document,
            }
            if not (agreed["label_it"] or agreed["label_en"]):
                ledger.hold("framework_members", key, member, "nessuna etichetta concorde fra le esecuzioni")
                continue
            reason = _label_en_problem(agreed["label_en"], document, task["pages"])
            chosen, problem = None, reason
            if not reason:
                # A quote must name the item and show its position, either
                # written or as the order of the list it quotes.
                for candidate in (member, other):
                    problem = _quote_problem(candidate["quote"], document, candidate["page"], [agreed["label_it"], agreed["label_en"]])
                    written = _position_written(candidate["quote"], member["position"])
                    if not problem and not written:
                        problem = _order_problem(candidate["quote"], members_a, index)
                    if not problem:
                        chosen = candidate
                        break
                    reason = reason or problem
            if chosen is None:
                ledger.hold("framework_members", key, {**row, **member}, reason or problem)
                continue
            written = _position_written(chosen["quote"], member["position"])
            row.update(
                {
                    "page": _locate(chosen["quote"], document, chosen["page"]),
                    "quote": chosen["quote"],
                    "notes": "; ".join(
                        filter(None, ["" if written else "posizione dall'ordine dell'elenco",
                                      f"discordi, lasciati vuoti: {', '.join(split)}" if split else ""])
                    ),
                }
            )
            accepted[framework_id][member["position"]] = row
        for unmatched in sorted(set(by_b) - {m["position"] for m in members_a}):
            ledger.hold("framework_members", f"{framework_id}#{unmatched}", by_b[unmatched], "presente in una sola esecuzione")
        _accept_r_definitions(ledger, framework_id, document, draft_a, draft_b, accepted[framework_id])
    return accepted


def _accept_r_definitions(ledger, framework_id, document, draft_a, draft_b, members) -> None:
    by_b = {d["position"]: d for d in draft_b["definitions"]}
    for definition in draft_a["definitions"]:
        position = definition["position"]
        key = f"{framework_id}#{position}|it"
        member = members.get(position)
        other = by_b.get(position)
        proposal = {**definition, "document": document}
        if member is None:
            ledger.hold("definitions", key, proposal, "voce del framework in sospeso")
            continue
        if other is None:
            ledger.hold("definitions", key, proposal, "presente in una sola esecuzione")
            continue
        if not (normalize(definition["quoted_definition"]) in normalize(other["quoted_definition"])
                or normalize(other["quoted_definition"]) in normalize(definition["quoted_definition"])):
            ledger.hold("definitions", key, proposal, "le due esecuzioni citano frasi diverse")
            continue
        chosen, reason = _pick_quote(
            [definition, other], document, "quoted_definition", "page",
            lambda _: [member["label_it"], member["label_en"]],
        )
        if chosen is None:
            ledger.hold("definitions", key, proposal, reason)
            continue
        member.setdefault("_definitions", []).append(
            {
                "quoted_definition": chosen["quoted_definition"],
                "language": "it",
                "document": document,
                "page": _locate(chosen["quoted_definition"], document, chosen["page"]),
            }
        )
    for position in sorted(set(by_b) - {d["position"] for d in draft_a["definitions"]}):
        ledger.hold("definitions", f"{framework_id}#{position}|it", by_b[position], "presente in una sola esecuzione")


def accept_mappings(ledger: Ledger, run_a: dict, run_b: dict, members: dict) -> list[dict]:
    """Accept version mappings whose endpoints are accepted.

    Returns:
        Accepted mapping rows.
    """
    accepted = []
    for task, draft_a, draft_b in zip(MAPPING_TASKS, run_a["mappings"], run_b["mappings"]):
        document = task["document"]

        def ref(framework_id: str, position: str) -> str:
            return f"{framework_id}#{position}" if position else ""

        def pair(row: dict) -> tuple[str, str]:
            return row["from_position"], row["to_position"]

        rows_b = {pair(m): m for m in draft_b["rows"]}
        for mapping in draft_a["rows"]:
            row = {
                "from": ref(task["from"], mapping["from_position"]),
                "to": ref(task["to"], mapping["to_position"]),
                "type": mapping["type"],
                "document": document,
            }
            key = f"{row['from']}>{row['to']}"
            other = rows_b.get(pair(mapping))
            reason = _source_problem(document)
            if not reason and mapping["inconsistent"]:
                reason = f"numero di origini incoerente con il tipo {mapping['type']}"
            if not reason and other is None:
                reason = "presente in una sola esecuzione"
            if not reason and other["type"] != mapping["type"]:
                reason = f"tipo discorde: {mapping['type']} / {other['type']}"
            if not reason and other["inconsistent"]:
                reason = "l'altra esecuzione ha un numero di origini incoerente"
            chosen = None
            if not reason:
                chosen, reason = _pick_quote([mapping, other], document, "quote", "page", lambda _: [])
            if not reason:
                ends = [(task["to"], mapping["to_position"])]
                if mapping["from_position"]:
                    ends.append((task["from"], mapping["from_position"]))
                missing = [f"{f}#{p}" for f, p in ends if p not in members.get(f, {})]
                if missing:
                    reason = f"voce collegata in sospeso: {', '.join(missing)}"
            if reason:
                ledger.hold("version_mappings", key, {**row, "page": mapping["page"], "quote": mapping["quote"]}, reason)
                continue
            row["page"] = _locate(chosen["quote"], document, chosen["page"])
            row["notes"] = f"«{chosen['quote']}»; rispetto a {task['from']}"
            # A "new" row has no origin, so the same member can be new with
            # respect to several versions: one row, every comparison noted.
            twin = next((r for r in accepted if r["type"] == "new" and r["to"] == row["to"] and not row["from"]), None)
            if twin is not None:
                twin["notes"] += f"; anche rispetto a {task['from']}"
                continue
            accepted.append(row)
        for missing_pair in sorted(set(rows_b) - {pair(m) for m in draft_a["rows"]}):
            ledger.hold(
                "version_mappings",
                f"{ref(task['from'], missing_pair[0])}>{ref(task['to'], missing_pair[1])}",
                rows_b[missing_pair],
                "presente in una sola esecuzione",
            )
    return accepted


def r_concepts(members: dict, mappings: list[dict]) -> list[dict]:
    """Group accepted R members into shared concepts.

    Members linked by a ``same`` or ``renamed`` correspondence are one
    strategy; a ``merged`` target is a strategy of its own, related to the
    ones it unites.

    Returns:
        Concept rows; members get their ``concept_id`` set in place.
    """
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for framework_id in R_ORDER:
        for position in members.get(framework_id, {}):
            find(f"{framework_id}#{position}")
    for mapping in mappings:
        if mapping["type"] in {"same", "renamed"}:
            a, b = find(mapping["from"]), find(mapping["to"])
            if a != b:
                parent[max(a, b, key=_order_key)] = min(a, b, key=_order_key)

    groups: dict[str, list[dict]] = {}
    for framework_id in R_ORDER:
        for position, row in members.get(framework_id, {}).items():
            groups.setdefault(find(f"{framework_id}#{position}"), []).append(row)

    concepts, taken = [], set()
    for root in sorted(groups, key=_order_key):
        group = groups[root]
        concept_id = "r_" + _slug(group[0]["label_it"] or group[0]["label_en"])
        while concept_id in taken:
            concept_id += "_bis"
        taken.add(concept_id)
        labels_it = _distinct(m["label_it"] for m in group if m["label_it"])
        labels_en = _distinct(m["label_en"] for m in group if m["label_en"])
        for row in group:
            row["concept_id"] = concept_id
        concepts.append(
            {
                "id": concept_id,
                "label_it": labels_it[0] if labels_it else "",
                "label_en": labels_en[0] if labels_en else "",
                "synonyms_it": "; ".join(labels_it[1:]),
                "synonyms_en": "; ".join(labels_en[1:]),
                "broader": "r",
                "related": "",
                "notes": "voci: " + ", ".join(member_key(m) for m in group),
            }
        )
    by_member = {member_key(m): m["concept_id"] for f in members.values() for m in f.values()}
    rows = {c["id"]: c for c in concepts}
    for mapping in mappings:
        if mapping["type"] == "merged":
            target = rows[by_member[mapping["to"]]]
            related = set(split_list(target["related"])) | {by_member[mapping["from"]]}
            target["related"] = "; ".join(sorted(related))
    return concepts


def _order_key(member_ref: str) -> tuple[int, int]:
    framework_id, position = member_ref.split("#")
    return R_ORDER.index(framework_id), 99 if position == "R+" else int(position[1:])


def _distinct(values) -> list[str]:
    seen, result = set(), []
    for value in values:
        if normalize(value) not in seen:
            seen.add(normalize(value))
            result.append(value)
    return result


def accept_3c(ledger: Ledger, run_a: dict, run_b: dict) -> None:
    """Accept the 3C framework, its members, components and definitions."""
    task = C3_TASK
    doc_en, doc_it = task["document_en"], task["document_it"]
    draft_a, draft_b = run_a["3c"], run_b["3c"]
    head_a, head_b = draft_a["framework"], draft_b["framework"]
    framework_id = task["id"]

    name_doc = doc_en if quote_on_page(head_a["name_quote"], doc_en, head_a["name_page"]) else doc_it
    framework_row = {
        "id": framework_id,
        "name": _agreed_name(head_a, head_b) or head_a["name"],
        "version": task["version"],
        "derived_from": "",
        "author": head_a["author"] if _same(head_a["author"], head_b["author"]) else "",
        "year": head_a["year"] if head_a["year"] == head_b["year"] else "",
        "document": doc_en,
        "pages": ", ".join(str(p) for p in task["pages_en"]),
        "notes": f"nome: {name_doc} p.{head_a['name_page']} «{head_a['name_quote']}»; "
        f"testo italiano: {doc_it} pp. {', '.join(str(p) for p in task['pages_it'])}",
    }
    reason = _source_problem(doc_it) or _framework_head(head_a, head_b, name_doc)
    if reason:
        ledger.hold("frameworks", framework_id, framework_row, reason)
        return
    ledger.accept("frameworks", framework_row)

    def pick(items: list[dict]) -> tuple[dict | None, dict | None, str]:
        """Passing English and Italian quote among the two runs' items."""
        en, reason_en = _pick_quote(items, doc_en, "quote_en", "page_en", lambda c: [c["label_en"]])
        it, reason_it = _pick_quote(items, doc_it, "quote_it", "page_it", lambda c: [c["label_it"]])
        return en, it, reason_en or reason_it

    def agree(item: dict, other: dict | None) -> str:
        if other is None:
            return "presente in una sola esecuzione"
        if not (_same(item["label_en"], other["label_en"]) and _same(item["label_it"], other["label_it"])):
            return f"esecuzioni discordi: {item['label_en']}/{item['label_it']} vs {other['label_en']}/{other['label_it']}"
        return ""

    member_labels = [m["label_en"] for m in draft_a["members"]]

    def is_listing(quote: str) -> bool:
        # A quote that names all three C's introduces the framework; only a
        # sentence about one C defines it.
        return sum(contains_term(quote, label) for label in member_labels) == 3

    concept_of: dict[str, str] = {}
    by_b = {m["position"]: m for m in draft_b["members"]}
    for member in draft_a["members"]:
        key = f"{framework_id}#{member['position']}"
        other = by_b.get(member["position"])
        concept_id = "c3_" + _slug(member["label_en"])
        reason = agree(member, other)
        en = it = None
        if not reason:
            en, it, reason = pick([member, other])
        if reason:
            ledger.hold("framework_members", key, member, reason)
            continue
        ledger.accept(
            "framework_members",
            {
                "framework_id": framework_id,
                "position": str(member["position"]),
                "concept_id": concept_id,
                "label_it": member["label_it"],
                "label_en": member["label_en"],
                "family": "",
                "document": doc_en,
                "page": _locate(en["quote_en"], doc_en, en["page_en"]),
                "quote": en["quote_en"],
                "notes": "ordine di presentazione, non priorità; "
                f"it: {doc_it} p.{_locate(it['quote_it'], doc_it, it['page_it'])} «{it['quote_it']}»",
            },
        )
        concept_of[normalize(member["label_en"])] = concept_id
        ledger.accept(
            "concepts",
            {"id": concept_id, "label_it": member["label_it"], "label_en": member["label_en"], "broader": "3c", "notes": f"voce: {key}"},
        )
        _add_definitions(ledger, concept_id, [member, other], doc_en, doc_it, is_listing)

    subs_b = {normalize(s["label_en"]): s for s in draft_b["subconcepts"]}
    remaining = list(draft_a["subconcepts"])
    # Parents can be components themselves, so resolve in rounds.
    while remaining:
        progressed = False
        waiting = {normalize(s["label_en"]) for s in remaining}
        for sub in list(remaining):
            parent_key = normalize(sub["parent_label_en"])
            if parent_key not in concept_of and parent_key in waiting:
                continue
            remaining.remove(sub)
            progressed = True
            concept_id = "c3_" + _slug(sub["label_en"])
            other = subs_b.get(normalize(sub["label_en"]))
            parent = concept_of.get(parent_key)
            reason = "" if parent else f"concetto più generale in sospeso o sconosciuto: {sub['parent_label_en']}"
            reason = reason or agree(sub, other)
            if not reason and not _same(sub["parent_label_en"], other["parent_label_en"]):
                reason = f"più generale discorde: {sub['parent_label_en']} / {other['parent_label_en']}"
            if not reason:
                reason = pick([sub, other])[2]
            if reason:
                ledger.hold("concepts", concept_id, sub, reason)
                continue
            ledger.accept(
                "concepts",
                {"id": concept_id, "label_it": sub["label_it"], "label_en": sub["label_en"], "broader": parent, "notes": f"componente di {sub['parent_label_en']}"},
            )
            concept_of[normalize(sub["label_en"])] = concept_id
            _add_definitions(ledger, concept_id, [sub, other], doc_en, doc_it, lambda _: False)
        if not progressed:
            for sub in remaining:
                ledger.hold("concepts", "c3_" + _slug(sub["label_en"]), sub, "concetto più generale in sospeso")
            break
    for label in sorted(set(subs_b) - {normalize(s["label_en"]) for s in draft_a["subconcepts"]}):
        ledger.hold("concepts", "c3_" + label.replace(" ", "_"), subs_b[label], "presente in una sola esecuzione")


def _add_definitions(ledger: Ledger, concept_id: str, items: list[dict], doc_en: str, doc_it: str, is_listing) -> None:
    for lang, document in (("en", doc_en), ("it", doc_it)):
        candidates = [c for c in items if not is_listing(c[f"quote_{lang}"])]
        chosen, _ = _pick_quote(candidates, document, f"quote_{lang}", f"page_{lang}", lambda c: [c[f"label_{lang}"]])
        if chosen is None:
            continue
        ledger.accept(
            "definitions",
            {
                "concept_id": concept_id,
                "quoted_definition": chosen[f"quote_{lang}"],
                "language": lang,
                "document": document,
                "page": _locate(chosen[f"quote_{lang}"], document, chosen[f"page_{lang}"]),
            },
        )


def list_concepts(members: dict[str, dict[str, dict]]) -> list[dict]:
    """One concept per member of a single-version list, under its family.

    Args:
        members: Framework id → position → accepted member row; the rows get
            their ``concept_id`` set in place.

    Returns:
        Concept rows.
    """
    concepts = []
    for framework_id, rows in members.items():
        family = framework_id.split("_", 1)[0]
        for row in rows.values():
            row["concept_id"] = f"{family}_{_slug(row['label_it'] or row['label_en'])}"
            concepts.append(
                {
                    "id": row["concept_id"],
                    "label_it": row["label_it"],
                    "label_en": row["label_en"],
                    "broader": family,
                    "notes": f"voce: {member_key(row)}",
                }
            )
    return concepts


def accept_cases(ledger: Ledger, draft_a: dict, draft_b: dict, members: dict) -> None:
    """Accept the cases whose name and assigned strategies both runs agree on.

    The strategies are checked against the accepted members of the version
    the source uses; promoter and territory are kept only where the runs
    agree, like the fields of a framework member.
    """
    task = CASE_TASK
    document = task["document"]
    framework_id = task["framework_id"]
    by_name_b = {normalize(c["name"]): c for c in draft_b["cases"]}
    for case in draft_a["cases"]:
        key = "case_" + _slug(case["name"])[:60]
        other = by_name_b.get(normalize(case["name"]))
        reason = _source_problem(document)
        if not reason and other is None:
            reason = "presente in una sola esecuzione"
        if not reason and sorted(case["r_codes"]) != sorted(other["r_codes"]):
            reason = f"strategie discordi: {case['r_codes']} / {other['r_codes']}"
        if not reason and case["page"] != other["page"]:
            reason = f"pagina discorde: {case['page']} / {other['page']}"
        name_quote = code_quote = None
        if not reason:
            name_quote, reason = _pick_quote([case, other], document, "name_quote", "page", lambda c: [c["name"]])
        if not reason:
            code_quote, reason = _pick_quote([case, other], document, "r_quote", "page", lambda _: [])
        if not reason:
            missing = [code for code in case["r_codes"] if not _position_written(code_quote["r_quote"], code)]
            if missing:
                reason = f"la citazione non riporta {', '.join(missing)}"
        if not reason:
            unknown = [code for code in case["r_codes"] if code not in members.get(framework_id, {})]
            if unknown:
                reason = f"strategia non accettata in {framework_id}: {', '.join(unknown)}"
        row = {
            "id": key,
            "name": case["name"],
            "document": document,
            "page": case["page"],
            "territory": case["territory"] if other and _same(case["territory"], other["territory"]) else "",
            "actors": case["promoter"] if other and _same(case["promoter"], other["promoter"]) else "",
            "strategies": "; ".join(case["r_codes"]),
            "framework_id": framework_id,
        }
        if reason:
            ledger.hold("cases", key, {**row, **case}, reason)
            continue
        row["notes"] = f"«{name_quote['name_quote']}»; «{code_quote['r_quote']}»"
        ledger.accept("cases", row)
    for name in sorted(set(by_name_b) - {normalize(c["name"]) for c in draft_a["cases"]}):
        ledger.hold("cases", "case_" + name.replace(" ", "_")[:60], by_name_b[name], "presente in una sola esecuzione")


def merge_with_expert_rows(ledger: Ledger) -> dict[str, list[dict]]:
    """Combine new automatic rows with what the expert already decided.

    Returns:
        Sheet name → final rows.
    """
    final = {}
    for sheet in SHEET_COLUMNS:
        kept = [r for r in read_sheet(sheet) if r.get("status") in EXPERT_STATUSES | {"proposed"}]
        if sheet == "concepts":
            kept_ids = {r["id"] for r in kept}
            kept += [{**row, "status": "proposed"} for row in FAMILY_CONCEPTS if row["id"] not in kept_ids]
        taken = {row_key(sheet, r) for r in kept}
        final[sheet] = kept + [r for r in ledger.rows[sheet] if row_key(sheet, r) not in taken]
    return final


def main() -> None:
    run_a = json.loads((DRAFTS_DIR / "run_a.json").read_text(encoding="utf-8"))
    run_b = json.loads((DRAFTS_DIR / "run_b.json").read_text(encoding="utf-8"))
    ledger = Ledger()

    members = accept_versions(ledger, run_a["r"], run_b["r"], R_TASKS, DERIVED_FROM)
    mappings = accept_mappings(ledger, run_a, run_b, members)
    for concept in r_concepts(members, mappings):
        ledger.accept("concepts", concept)
    if "lists" in run_a and "lists" in run_b:
        listed = accept_versions(ledger, run_a["lists"], run_b["lists"], LIST_TASKS, {})
        for concept in list_concepts(listed):
            ledger.accept("concepts", concept)
        members.update(listed)
    if "cases" in run_a and "cases" in run_b:
        accept_cases(ledger, run_a["cases"], run_b["cases"], members)
    for framework_rows in members.values():
        for row in framework_rows.values():
            for definition in row.pop("_definitions", []):
                ledger.accept("definitions", {"concept_id": row["concept_id"], **definition})
            ledger.accept("framework_members", row)
    for mapping in mappings:
        ledger.accept("version_mappings", mapping)
    accept_3c(ledger, run_a, run_b)

    sheets = merge_with_expert_rows(ledger)
    for row in sheets["concepts"]:
        if not row.get("graph_nodes"):
            forms = [row.get("label_it", ""), row.get("label_en", "")]
            forms += split_list(row.get("synonyms_it", "")) + split_list(row.get("synonyms_en", ""))
            row["graph_nodes"] = "; ".join(graph_nodes_for([f for f in forms if f]))

    # Criterion 4: a row that breaks a check goes back to pending; what
    # depends on it fails the next round, until the sheets are clean.
    while True:
        problems = [p for p in check_all(sheets) if _is_auto(sheets, p)]
        if not problems:
            break
        for problem in problems:
            rows = sheets[problem.sheet]
            for row in [r for r in rows if row_key(problem.sheet, r) == problem.key and r["status"] == AUTO]:
                rows.remove(row)
                ledger.hold(problem.sheet, problem.key, row, f"controllo: {problem.message}")

    for sheet, rows in sheets.items():
        write_sheet(sheet, rows)
    write_tsv(DRAFTS_DIR / "pending.tsv", ["sheet", "key", "proposal", "reason"], ledger.pending)

    for sheet, rows in sheets.items():
        auto = sum(r["status"] == AUTO for r in rows)
        held = sum(p["sheet"] == sheet for p in ledger.pending)
        print(f"{sheet:18} auto_accepted={auto:3}  in sospeso={held:3}  altre={len(rows) - auto}")


def _is_auto(sheets: dict, problem) -> bool:
    return any(row_key(problem.sheet, r) == problem.key and r["status"] == AUTO for r in sheets[problem.sheet])


if __name__ == "__main__":
    main()
