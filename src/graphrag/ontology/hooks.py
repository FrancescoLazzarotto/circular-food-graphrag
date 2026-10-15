"""The two points where the agent uses the framework ontology.

Before retrieval, :func:`consult` recognises a question about a framework or
one of its concepts with word rules and turns what the ontology knows into
evidence items, labelled with document and page like retrieved passages.
After generation, :func:`check_answer` compares the lists in the answer with
the ontology, without any model call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from graphrag.llm.prompts import PromptLibrary

from .consult import USABLE, Ontology, _compact, confronta, normalize, verifica

# A single word names a strategy in too many unrelated questions ("ridurre gli
# sprechi"), so it counts only next to the framework's own name. Names of
# several words, the C's themselves and strategies a version introduced are
# specific enough on their own.
_MAX_NGRAM = 6
_STOPWORDS = {"le", "la", "il", "di", "del", "della", "delle", "et", "al", "and", "of", "the"}
_MAX_ITEMS = 5
_BULLET = re.compile(r"^\s*(?:[-*•–]|\d{1,2}[.)])\s+(.*)$")
_CODE = re.compile(r"^\[?(?:R\s?\d{1,2}|R\s?\+|\d{1,2})\]?\s*[:=.)–-]?\s*", re.IGNORECASE)


@dataclass
class Match:
    """What a question asks the ontology about.

    Attributes:
        frameworks: Framework versions named, oldest first.
        named_version: True when the question named one version rather than
            the framework as a whole.
        concepts: Concepts named.
        cases: Cases named.
    """

    frameworks: list[str] = field(default_factory=list)
    named_version: bool = False
    concepts: list[str] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.frameworks or self.concepts or self.cases)


def recognise(question: str, onto: Ontology) -> Match:
    """Find the frameworks and concepts a question names.

    Args:
        question: The question, self-contained when memory rewrote it.
        onto: The loaded ontology.

    Returns:
        The match; empty when the question is not about the ontology.
    """
    tokens = _compact(question).split()
    framework_hits: list[str] = []
    family_hits: list[str] = []
    concept_hits: list[str] = []
    case_hits: list[str] = []
    covered: set[int] = set()
    # Longest names first, and a word belongs to one name only: "capitale
    # naturale" is the component, not also the C called "capitale".
    for size in range(min(_MAX_NGRAM, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            span = set(range(start, start + size))
            if span & covered:
                continue
            hits = onto._hits(" ".join(tokens[start : start + size]))
            if hits:
                covered |= span
            for hit in hits:
                if hit["kind"] == "framework":
                    framework_hits.append(hit["id"])
                elif hit["kind"] == "case":
                    case_hits.append(hit["id"])
                elif hit["id"] in onto.families:
                    family_hits.append(hit["id"])
                else:
                    concept_hits.append(hit["id"])
    family_hits += _code_families(question, onto)

    named_framework = bool(framework_hits or family_hits)
    if named_framework:
        # An author or a year names a version only next to the framework's
        # name: "del 2024" alone says nothing about the R strategies.
        # The id carries the distinctive words (author, year, project); the
        # version label also has ordinary ones ("Mercato Circolare").
        words = set(tokens) - _STOPWORDS
        for fid in onto.frameworks:
            if words & set(fid.split("_")[1:]):
                framework_hits.append(fid)

    match = Match()
    if framework_hits:
        match.frameworks = sorted(set(framework_hits), key=onto._age)
        match.named_version = len(match.frameworks) == 1
    elif family_hits:
        match.frameworks = sorted(
            {fid for family in family_hits for fid in onto.families[family]["frameworks"]},
            key=onto._age,
        )
    match.concepts = [
        cid
        for cid in dict.fromkeys(concept_hits)
        if onto.concepts[cid]["status"] in USABLE
        and (named_framework or _specific(onto, cid))
    ]
    match.cases = [
        cid for cid in dict.fromkeys(case_hits) if onto.data["cases"][cid]["status"] in USABLE
    ]
    return match


def _code_families(question: str, onto: Ontology) -> list[str]:
    """Families named by a position code: "la R4", "la [R+]", "delle R"."""
    letters = {
        member["position"][0]: framework["family"]
        for framework in onto.frameworks.values()
        for member in framework["members"]
        if member["position"][:1].isalpha()
    }
    return [
        family
        for letter, family in letters.items()
        if re.search(rf"(?<!\w){re.escape(letter)}(?:\s?\d{{1,2}}|\s?\+)?(?!\w)", question)
    ]


def _specific(onto: Ontology, concept_id: str) -> bool:
    """Whether a concept named alone, without its framework, is about the ontology.

    Specific: a name of several words, a C of the 3C with a long name, a
    strategy a version introduced. A single short word ("capitale",
    "ridurre") is too common to mean the framework.
    """
    concept = onto.concepts[concept_id]
    labels = [concept["label_it"], concept["label_en"], *concept["synonyms_it"], *concept["synonyms_en"]]
    if any(len(normalize(label).split()) > 1 for label in labels if label):
        return True
    members = [onto.members[key] for key in concept["members"]]
    long_name = max((len(normalize(label)) for label in labels if label), default=0) > 8
    if long_name and any(onto.frameworks[m["framework_id"]]["family"] == "3c" for m in members):
        return True
    introduced = {m["to"] for m in onto.mappings if m["type"] == "new" and m["status"] in USABLE}
    return any(m["key"] in introduced for m in members)


def consult(question: str, onto: Ontology) -> tuple[Match, list[dict]]:
    """Evidence items for a question about the frameworks.

    Args:
        question: The question, self-contained when memory rewrote it.
        onto: The loaded ontology.

    Returns:
        The match and up to five items shaped like retrieved text chunks
        (``content``, ``source``, ``chunk_id``): one per framework version,
        one with the differences between versions, one per concept.
    """
    match = recognise(question, onto)
    if not match:
        return match, []
    # A named case comes first: the question is about it, and the framework
    # it names is the background.
    items: list[dict] = [_case_item(onto, case_id) for case_id in match.cases]
    for framework_id in match.frameworks:
        items.append(_framework_item(onto, framework_id, len(match.frameworks)))
    if len(match.frameworks) > 1:
        differences = _differences_item(onto, match.frameworks)
        if differences:
            items.append(differences)
    for concept_id in match.concepts:
        item = _concept_item(onto, concept_id)
        if item:
            items.append(item)
    return match, items[:_MAX_ITEMS]


def _source(onto: Ontology, document: str, pages: str) -> str:
    filename = onto.data.get("documents", {}).get(document) or document
    return f"{filename}#page={pages}" if pages else filename


def _status_note(statuses: set[str]) -> str:
    return PromptLibrary.ontology_reviewed_note(statuses <= {"approved", "corrected"})


def _framework_item(onto: Ontology, framework_id: str, versions: int) -> dict:
    framework = onto.frameworks[framework_id]
    members = [m for m in framework["members"] if m["status"] in USABLE]
    lines = [
        f"{framework['name']} — versione {framework['version']}"
        + (f", di {framework['author']}" if framework["author"] else "")
        + (f" ({framework['year']})" if framework["year"] else "")
        + "."
    ]
    for member in members:
        label = " / ".join(filter(None, (member["label_it"], member["label_en"])))
        family = f" [{member['family']}]" if member["family"] else ""
        lines.append(f"{member['position']} {label}{family}")
    statuses = {framework["status"], *(m["status"] for m in members)}
    lines.append(PromptLibrary.ontology_reading_note(versions) + _status_note(statuses))
    return {
        "content": "\n".join(lines),
        "source": _source(onto, framework["document"], _page_ranges(framework["pages"])),
        "chunk_id": f"ontology:{framework_id}",
    }


def _page_ranges(pages: str) -> str:
    """"6, 13, 15, 16, 19" → "6, 13, 15-16, 19": the label a reader sees."""
    try:
        numbers = sorted({int(p) for p in re.split(r"[,;]\s*", pages) if p.strip()})
    except ValueError:
        return pages
    runs: list[list[int]] = []
    for number in numbers:
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)


def _differences_item(onto: Ontology, framework_ids: list[str]) -> dict | None:
    lines, documents = [], []
    for older, newer in zip(framework_ids, framework_ids[1:]):
        result = confronta(older, newer, ontology=onto)
        changes = [c for c in result["changes"] if c["type"] != "same" or c.get("moved")]
        if not changes:
            continue
        lines.append(f"Da {onto.frameworks[older]['name']} a {onto.frameworks[newer]['name']}:")
        for change in changes:
            lines.append("- " + _describe(change))
            documents.extend(e["document"] for e in change["evidence"])
    if not lines:
        return None
    document = max(set(documents), key=documents.count) if documents else onto.frameworks[framework_ids[-1]]["document"]
    return {
        "content": "Differenze fra le versioni, dalle corrispondenze dell'ontologia:\n" + "\n".join(lines),
        "source": _source(onto, document, ""),
        "chunk_id": "ontology:differences",
    }


def _describe(change: dict) -> str:
    def name(member: dict) -> str:
        return f"{member['position']} {member['label_it'] or member['label_en']}"

    if change["type"] == "new":
        return f"nuova: {name(change['to'])}"
    if change["type"] == "merged":
        return f"unita: {name(change['from'])} confluisce in {name(change['to'])}"
    if change["type"] == "renamed":
        return f"rinominata: {name(change['from'])} diventa {name(change['to'])}"
    return f"spostata: {name(change['from'])} diventa {name(change['to'])}"


def _case_item(onto: Ontology, case_id: str) -> dict:
    case = onto.data["cases"][case_id]
    strategies = [onto.members[key] for key in case["strategies"] if key in onto.members]
    framework = onto.frameworks[case["framework_id"]]
    lines = [f"Caso: {case['name']}."]
    if case["actors"]:
        lines.append(f"Promotore: {case['actors']}")
    if case["territory"]:
        lines.append(f"Area: {case['territory']}")
    lines.append(
        f"Strategia R principale secondo {framework['name']}: "
        + ", ".join(f"{m['position']} {m['label_it'] or m['label_en']}" for m in strategies)
        + "."
    )
    return {
        "content": "\n".join(lines) + _status_note({case["status"], *(m["status"] for m in strategies)}),
        "source": _source(onto, case["document"], str(case["page"])),
        "chunk_id": f"ontology:{case_id}",
    }


def _concept_item(onto: Ontology, concept_id: str) -> dict | None:
    concept = onto.concepts[concept_id]
    definitions = [d for d in concept["definitions"] if d["status"] in USABLE]
    narrower = [onto.concepts[c] for c in concept["narrower"] if onto.concepts[c]["status"] in USABLE]
    broader = [onto.concepts[c] for c in concept["broader"] if c not in onto.families]
    members = [onto.members[key] for key in concept["members"] if onto.members[key]["status"] in USABLE]
    if not (definitions or narrower or members):
        return None
    label = " / ".join(filter(None, (concept["label_it"], concept["label_en"])))
    lines = [label + "."]
    if broader:
        lines.append("Fa parte di: " + ", ".join(b["label_it"] or b["label_en"] for b in broader) + ".")
    if narrower:
        lines.append("Comprende: " + ", ".join(n["label_it"] or n["label_en"] for n in narrower) + ".")
    if members:
        lines.append(
            "Compare in: "
            + "; ".join(
                f"{onto.frameworks[m['framework_id']]['name']} come {m['position']} {m['label_it'] or m['label_en']} (p. {m['page']})"
                for m in members
            )
            + "."
        )
    keys = {m["key"] for m in members}
    applied = [
        c for c in onto.data.get("cases", {}).values()
        if c["status"] in USABLE and keys & set(c["strategies"])
    ]
    if applied:
        lines.append("Casi che la applicano: " + "; ".join(f"{c['name']} (p. {c['page']})" for c in applied) + ".")
    for definition in definitions[:2]:
        lines.append(definition["quoted_definition"])
    document = definitions[0]["document"] if definitions else onto.members[concept["members"][0]]["document"] if concept["members"] else ""
    page = definitions[0]["page"] if definitions else ""
    return {
        "content": "\n".join(lines) + _status_note({concept["status"], *(d["status"] for d in definitions)}),
        "source": _source(onto, document, page),
        "chunk_id": f"ontology:{concept_id}",
    }


def list_items(answer: str) -> list[list[str]]:
    """The lists an answer contains, as item names.

    A list is a run of bullet or numbered lines, or a parenthesis or sentence
    of at least four comma-separated short items. The name is what comes
    before a colon, dash or parenthesis, without a leading code such as
    ``R4 =``.

    Args:
        answer: Generated answer.

    Returns:
        Each list's item names, in order.
    """
    text = re.sub(r"[*_`]", "", answer or "")
    lists: list[list[str]] = []
    run: list[str] = []
    for line in text.splitlines():
        bullet = _BULLET.match(line)
        if bullet:
            run.append(_item_name(bullet.group(1)))
            continue
        if len(run) >= 3:
            lists.append(run)
        run = []
    if len(run) >= 3:
        lists.append(run)
    for chunk in re.split(r"[.;:\n()]", text):
        parts = [p.strip() for p in re.split(r",|\be\b|\band\b", chunk) if p.strip()]
        # The first item carries the sentence that introduces the list ("le
        # strategie sono Rifiutare"): its last word is the item.
        if len(parts) >= 4 and len(parts[0].split()) > 3:
            parts[0] = parts[0].split()[-1]
        if len(parts) >= 4 and all(len(p.split()) <= 3 for p in parts):
            lists.append([_item_name(p) for p in parts])
    return [[item for item in found if item] for found in lists]


def _item_name(text: str) -> str:
    head = _CODE.sub("", text.strip())
    head = re.split(r"\s[–—-]\s|:|\(|\[", head, maxsplit=1)[0]
    return head.strip(" .,;")


def check_answer(answer: str, match: Match, onto: Ontology) -> dict:
    """Compare the framework lists in an answer with the ontology.

    Only lists that are mostly about the framework are judged: at least half
    of their items must be members of one of its versions, so an unrelated
    list in the same answer is left alone.

    Args:
        answer: Generated answer.
        match: What the question asked about.
        onto: The loaded ontology.

    Returns:
        ``{"checked", "ok", "framework", "missing", "extra"}``; ``checked``
        is False when the answer has no list about the framework.
    """
    report = {"checked": False, "ok": True, "framework": "", "missing": [], "extra": []}
    if not match.frameworks:
        return report
    target = match.frameworks[0] if match.named_version else None
    for items in list_items(answer):
        result = verifica(items, target or match.frameworks[0], ontology=onto)
        best = result["by_version"][result["best_match"]]
        if len(best["matched"]) * 2 < len(items) or len(best["matched"]) < 3:
            continue
        if target is None:
            # No version was named, so the list stands for the one it is
            # closest to.
            result = verifica(items, result["best_match"], ontology=onto)
        report.update(
            {
                "checked": True,
                "ok": report["ok"] and result["ok"],
                "framework": result["framework"],
                "missing": [m["label_it"] or m["label_en"] for m in result["missing"]],
                "extra": list(result["extra"]),
            }
        )
        if not result["ok"]:
            break
    return report


def check_note(report: dict, onto: Ontology, language: str) -> str:
    """The note to append when the check failed, else an empty string.

    Args:
        report: Output of :func:`check_answer`.
        onto: The loaded ontology.
        language: Answer language.

    Returns:
        The note.
    """
    if not report["checked"] or report["ok"]:
        return ""
    framework = onto.frameworks[report["framework"]]
    filename = onto.data.get("documents", {}).get(framework["document"]) or framework["document"]
    name = f"{framework['name']} ({filename}, p. {framework['pages']})"
    return PromptLibrary.ontology_check_note(language, name, report["missing"], report["extra"])
