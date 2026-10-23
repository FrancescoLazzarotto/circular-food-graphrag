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

from .consult import _ARTICLE, USABLE, Ontology, _compact, confronta, position_number, prima, verifica

# A single word names a strategy in too many unrelated questions ("ridurre gli
# sprechi"), so it counts only next to the framework's own name. Names of
# several words, the C's themselves and strategies a version introduced are
# specific enough on their own.
_MAX_NGRAM = 6
_STOPWORDS = {"le", "la", "il", "di", "del", "della", "delle", "et", "al", "and", "of", "the"}
_MAX_ITEMS = 5
_BULLET = re.compile(r"^\s*(?:[-*•–]|\d{1,2}[.)])\s+(.*)$")
# "Viene prima riusare o riciclare?", "meglio X o Y": a question about the order
# of two items, answered by the order rule of their family.
_PRIORITY = re.compile(r"\b(prima|dopo|meglio|preferibil\w*|priorit\w*|pi[uù] virtuos\w*|before|after|better|priority|rather)\b", re.IGNORECASE)
_CODE = re.compile(r"^\[?(?:R\s?\d{1,2}|R\s?\+|\d{1,2})\]?\s*[:=.)–-]?\s*", re.IGNORECASE)
# Square brackets hold citations ("[REPORT MATTM, p. 70]", "[T1-T12]") unless
# they hold a position ("[R+]"). Left in, a citation is cut at its full stop
# and its "p" becomes one more item of the list.
_CITATION = re.compile(r"\[(?!\s*R\s?(?:\d{1,2}|\+)\s*\])[^\[\]\n]*\]", re.IGNORECASE)
# A position after the name, "rifiutare (R0)", would cut the sentence at
# every parenthesis.
_TRAILING_CODE = re.compile(r"\s*\(\s*\[?R\s?(?:\d{1,2}|\+)\]?\s*\)", re.IGNORECASE)
# Members such as "preparazione per il riutilizzo" or "Rimettere a nuovo/Rifabbricare".
_MAX_ITEM_WORDS = 5
# "componenti come riusare, ricontenere e ridurre": the sentence itself says
# the list is a few examples, whatever share of the version it holds.
_EXAMPLES = re.compile(r"\b(?:come|tra cui|ad esempio|per esempio|such as|including)$", re.IGNORECASE)


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
            name = " ".join(tokens[start : start + size])
            hits = onto._hits(name)
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
                    # "la capitale" is one word of the concept, not two.
                    concept_hits.append((hit["id"], _ARTICLE.sub("", name)))
    family_hits += _code_families(question, onto)

    named_framework = bool(framework_hits or family_hits)
    if named_framework:
        # An author or a year names a version only next to the framework's
        # name: "del 2024" alone says nothing about the R strategies.
        # The id carries the distinctive words (author, year, project); the
        # version label also has ordinary ones ("Mercato Circolare").
        # Doubled letters are spelling, not a different word: "eRRRando" is
        # the version whose id says "errando".
        words = {_squeeze(w) for w in set(tokens) - _STOPWORDS}
        for fid in onto.frameworks:
            if words & {_squeeze(w) for w in fid.split("_")[1:]}:
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
    match.concepts = list(dict.fromkeys(
        cid
        for cid, name in concept_hits
        if onto.concepts[cid]["status"] in USABLE
        and (named_framework or _specific(onto, cid, name))
    ))
    match.cases = [
        cid for cid in dict.fromkeys(case_hits) if onto.data["cases"][cid]["status"] in USABLE
    ]
    return match


def _squeeze(word: str) -> str:
    """A word with every run of a repeated letter reduced to one letter."""
    return re.sub(r"(.)\1+", r"\1", word)


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


def _specific(onto: Ontology, concept_id: str, name: str) -> bool:
    """Whether a concept named alone, without its framework, is about the ontology.

    Specific: named with several words, a C of the 3C named with a long
    word, a strategy a version introduced. A single short word ("capitale",
    "ridurre", "recupero") is too common to mean the framework, even when the
    concept has a longer name too.

    Args:
        onto: The loaded ontology.
        concept_id: The concept matched.
        name: The words of the question that matched it.
    """
    if len(name.split()) > 1:
        return True
    concept = onto.concepts[concept_id]
    members = [onto.members[key] for key in concept["members"]]
    if len(name) > 8 and any(onto.frameworks[m["framework_id"]]["family"] == "3c" for m in members):
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
    if len(match.concepts) == 2 and _PRIORITY.search(question):
        order = _priority_item(onto, *match.concepts)
        if order:
            items.append(order)
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


def _priority_item(onto: Ontology, first_id: str, second_id: str) -> dict | None:
    """Which of two named items comes first, version by version, with the rule."""
    concepts = onto.concepts
    result = prima(concepts[first_id]["label_it"] or concepts[first_id]["label_en"],
                   concepts[second_id]["label_it"] or concepts[second_id]["label_en"], ontology=onto)
    if not result or not result["versions"] or not result["rule"]:
        return None
    lines = []
    for version in result["versions"]:
        ahead, behind = (version["a"], version["b"]) if version["first"] == "a" else (version["b"], version["a"])
        framework = onto.frameworks[version["framework"]]
        lines.append(
            f"{framework['name']} ({framework['version']}): {ahead['position']} {ahead['label_it'] or ahead['label_en']} "
            f"viene prima di {behind['position']} {behind['label_it'] or behind['label_en']}."
        )
    rule = result["rule"]
    lines.append(f"Regola: {rule['rule']}. «{rule['quote']}»" + _status_note({rule["status"]}))
    return {
        "content": "\n".join(lines),
        "source": _source(onto, rule["document"], rule["page"]),
        "chunk_id": f"ontology:{rule['id']}",
    }


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
    of at least three comma-separated short items. Citations in square
    brackets and positions in parentheses after a name are removed first.
    The name is what comes before a colon, dash or parenthesis, without a
    leading code such as ``R4 =``. A sentence list introduced by "come",
    "tra cui", "such as" and the like gives examples and is left out.

    Args:
        answer: Generated answer.

    Returns:
        Each list's item names, in order.
    """
    text = re.sub(r"[*_`]", "", answer or "")
    text = _TRAILING_CODE.sub("", _CITATION.sub("", text))
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
        names = [_item_name(p) for p in parts]
        # The first item carries the words that introduce the list ("le
        # strategie sono Rifiutare", "sono Capitale"): when it is longer than
        # every other item, only as many of its last words as the shortest
        # one has are the item.
        if len(names) >= 3:
            lengths = [len(n.split()) for n in names[1:]]
            words = names[0].split()
            if len(words) > max(lengths):
                keep = max(1, min(lengths))
                if _EXAMPLES.search(" ".join(words[:-keep])):
                    continue
                names[0] = " ".join(words[-keep:])
        if len(names) >= 3 and all(len(n.split()) <= _MAX_ITEM_WORDS for n in names):
            lists.append(names)
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

    A list made only of members of one family the answer names ("R Vita:
    riusare, ...") is checked against that family, not against the whole
    version. A list holding fewer than half of the version, or, in a version
    without families, an unbroken stretch of the scale ("R3-R7"), is a part
    named on purpose and is not judged.

    Returns:
        ``{"checked", "ok", "framework", "family", "missing", "extra"}``;
        ``checked`` is False when the answer has no list about the framework.
    """
    report = {"checked": False, "ok": True, "framework": "", "family": "", "missing": [], "extra": []}
    if not match.frameworks:
        return report
    target = match.frameworks[0] if match.named_version else None
    for items in list_items(answer):
        result = verifica(items, target or match.frameworks[0], ontology=onto)
        best = result["by_version"][result["best_match"]]
        if len(best["matched"]) * 2 < len(items) or len(best["matched"]) < 2:
            continue
        if target is None:
            # No version was named, so the list stands for the one it is
            # closest to.
            result = verifica(items, result["best_match"], ontology=onto)
        family = _single_family(onto, result)
        if family and f" {_compact(family)} " not in f" {_compact(answer)} ":
            # Members of one family the answer does not name are examples
            # ("riusare, riparare e ricondizionare"), not the family.
            continue
        if not family and (_is_stretch(onto, result) or _is_sample(onto, result)):
            continue
        missing = [m["label_it"] or m["label_en"] for m in result["missing"]]
        ok = result["ok"]
        if family:
            found = {m["position"] for m in result["matched"]}
            missing = [
                m["label_it"] or m["label_en"]
                for m in onto.frameworks[result["framework"]]["members"]
                if m["status"] in USABLE and m["family"] == family and m["position"] not in found
            ]
            ok = not missing and not result["extra"]
        report.update(
            {
                "checked": True,
                "ok": report["ok"] and ok,
                "framework": result["framework"],
                "family": family,
                "missing": missing,
                "extra": list(result["extra"]),
            }
        )
        if not ok:
            break
    return report


def _is_stretch(onto: Ontology, result: dict) -> bool:
    """Whether a partial list is consecutive positions of a version without families."""
    members = [m for m in onto.frameworks[result["framework"]]["members"] if m["status"] in USABLE]
    if result["extra"] or any(m["family"] for m in members) or len(result["matched"]) >= len(members):
        return False
    numbers = sorted(n for n in (position_number(m["position"]) for m in result["matched"]) if n is not None)
    return len(numbers) == len(result["matched"]) and numbers == list(range(numbers[0], numbers[0] + len(numbers)))


def _is_sample(onto: Ontology, result: dict) -> bool:
    """Whether a list names fewer than half of a version: examples, not the version."""
    members = [m for m in onto.frameworks[result["framework"]]["members"] if m["status"] in USABLE]
    return len(result["matched"]) * 2 < len(members)


def _single_family(onto: Ontology, result: dict) -> str:
    """The family a partial list is entirely drawn from, or an empty string."""
    members = [m for m in onto.frameworks[result["framework"]]["members"] if m["status"] in USABLE]
    families = {onto.members[f"{result['framework']}#{m['position']}"]["family"] for m in result["matched"]}
    every = {m["family"] for m in members}
    if len(families) != 1 or len(every) < 2 or len(result["matched"]) >= len(members):
        return ""
    return families.pop()


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
    if report.get("family"):
        name = f"«{report['family']}» — {name}"
    return PromptLibrary.ontology_check_note(language, name, report["missing"], report["extra"])
