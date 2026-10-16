"""In-memory consultation of ``ontology/build/ontology.json``.

All derived structure (ordered members, narrower and broader chains, the label
index) is computed by the build, so every call here is a few dictionary reads.
Versions of a framework are never merged into one "right" list: a question
that names the framework but not the version gets every version back.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "ontology" / "build" / "ontology.json"
SETTLED = {"approved", "corrected"}
USABLE = SETTLED | {"auto_accepted", "checked"}


class OntologyNotBuilt(FileNotFoundError):
    """``ontology.json`` is missing: run ``ontology/scripts/build.py``."""


def normalize(text: str) -> str:
    """Fold a label the way the build indexes it.

    Same folding as the answer-time quote gate
    (``graphrag.agent.evidence._normalize_for_quote_match``), repeated here so
    this module does not import the agent; a test keeps the two identical.

    Args:
        text: Label or query.

    Returns:
        Lower-case ASCII words separated by single spaces.
    """
    folded = unicodedata.normalize("NFKD", str(text or ""))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    folded = re.sub(r"[-­]\s*\n\s*", "", folded)
    return " ".join(re.sub(r"[^0-9a-zA-Z]+", " ", folded.lower()).split())


# "le 10 R", "the 3 C's": a leading article is not part of the name.
_ARTICLE = re.compile(r"^(?:il|lo|la|le|i|gli|l|the) ")


def _compact(text: str) -> str:
    # "10 R", "10R" and "10-R" are the same name for a reader.
    return re.sub(r"\b(\d+) (r|c|cs|rs)\b", r"\1\2", normalize(text))


class Ontology:
    """The framework ontology loaded in memory.

    Args:
        data: Content of ``ontology.json``.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.frameworks: dict[str, dict] = data["frameworks"]
        self.concepts: dict[str, dict] = data["concepts"]
        self.families: dict[str, dict] = data["families"]
        self.mappings: list[dict] = data["mappings"]
        self.labels: dict[str, list[dict]] = {}
        for key, hits in data["labels"].items():
            self.labels.setdefault(_compact(key), []).extend(hits)
        self.members = {
            m["key"]: m for f in self.frameworks.values() for m in f["members"]
        }

    # ------------------------------------------------------------------ lookup

    def _hits(self, name: str) -> list[dict]:
        key = _compact(name)
        return self.labels.get(key) or self.labels.get(_ARTICLE.sub("", key), [])

    def resolve_frameworks(self, name: str, version: str | None = None) -> list[str]:
        """Framework versions a name refers to.

        Args:
            name: Framework name as a user writes it ("10R", "le 3C", "eRRRando").
            version: Optional version hint (id, version label or a word of it).

        Returns:
            Matching framework ids, oldest first; empty if nothing matches.
        """
        found: list[str] = []
        for hit in self._hits(name):
            if hit["kind"] == "framework":
                found.append(hit["id"])
            elif hit["id"] in self.families:
                found.extend(self.families[hit["id"]]["frameworks"])
        if not found:
            query = set(_compact(name).split())
            for fid, framework in self.frameworks.items():
                distinctive = set(_compact(framework["version"]).split()) | set(fid.split("_")[1:])
                if query & distinctive - {"et", "al", "di", "la", "le"}:
                    found.append(fid)
        if version:
            wanted = _compact(version)
            found = [
                fid for fid in found
                if any(wanted in _compact(text) for text in (fid.replace("_", " "), self.frameworks[fid]["version"], self.frameworks[fid]["name"]))
            ]
        return sorted(set(found), key=self._age)

    def _age(self, framework_id: str) -> tuple:
        framework = self.frameworks[framework_id]
        return (len(framework["derived_from"]), str(framework.get("year") or ""), framework_id)

    def resolve_concept(self, name: str) -> str | None:
        """Concept id for a label, synonym or member label.

        Args:
            name: Any label.

        Returns:
            The concept id, or None.
        """
        for hit in self._hits(name):
            if hit["kind"] == "concept":
                return hit["id"]
        return None

    # ---------------------------------------------------------------- results

    def _framework_view(self, framework_id: str) -> dict:
        framework = self.frameworks[framework_id]
        members = [
            {k: m[k] for k in ("position", "label_it", "label_en", "family", "document", "page", "quote", "status", "concept_id")}
            | {"position_from_order": "posizione dall'ordine" in m.get("notes", "")}
            for m in framework["members"]
            if m["status"] in USABLE
        ]
        return {
            "id": framework_id,
            "name": framework["name"],
            "version": framework["version"],
            "author": framework["author"],
            "year": framework["year"],
            "derived_from": framework["derived_from"],
            "document": framework["document"],
            "pages": framework["pages"],
            "status": framework["status"],
            "members": members,
            "families": _families(members),
            "reviewed": framework["status"] in SETTLED and all(m["status"] in SETTLED for m in members),
        }

    def _concept_view(self, concept_id: str) -> dict:
        concept = self.concepts[concept_id]
        return {
            "id": concept_id,
            "label_it": concept["label_it"],
            "label_en": concept["label_en"],
            "status": concept["status"],
            "definitions": [d for d in concept["definitions"] if d["status"] in USABLE],
        }


def _families(members: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for member in members:
        if not member["family"]:
            continue
        if not groups or groups[-1]["family"] != member["family"]:
            groups.append({"family": member["family"], "positions": []})
        groups[-1]["positions"].append(member["position"])
    return groups


@lru_cache(maxsize=4)
def _load(path: str) -> Ontology:
    file = Path(path)
    if not file.is_file():
        raise OntologyNotBuilt(f"{file} non trovato: eseguire ontology/scripts/build.py")
    return Ontology(json.loads(file.read_text(encoding="utf-8")))


def load(path: str | Path | None = None) -> Ontology:
    """Load the ontology, once per path.

    Args:
        path: ``ontology.json``; defaults to ``GRAPHRAG_ONTOLOGY_JSON`` or the
            repository build.

    Returns:
        The loaded ontology.

    Raises:
        OntologyNotBuilt: The file does not exist.
    """
    return _load(str(path or os.environ.get("GRAPHRAG_ONTOLOGY_JSON") or DEFAULT_PATH))


def framework(nome: str, versione: str | None = None, *, ontology: Ontology | None = None) -> dict:
    """List a framework, in one version or in all of them.

    Args:
        nome: Framework name ("10R", "le 3C", "le 10 R di eRRRando").
        versione: Version hint; without it every matching version is returned.
        ontology: Loaded ontology; the default build if omitted.

    Returns:
        ``{"query", "versions": [...], "ambiguous"}``; each version lists its
        members in order with document, page, quote and status.
    """
    onto = ontology or load()
    ids = onto.resolve_frameworks(nome, versione)
    return {
        "query": {"nome": nome, "versione": versione},
        "versions": [onto._framework_view(fid) for fid in ids],
        "ambiguous": len(ids) > 1,
    }


def confronta(a: str, b: str, *, ontology: Ontology | None = None) -> dict:
    """Differences between two versions of a framework.

    Correspondences come from ``version_mappings.tsv``; where two versions are
    not mapped directly the chain through the versions in between is followed,
    and the strongest change on it is reported (merged over renamed over
    same). The comparison always runs from the older version to the one
    derived from it, whatever the order of the arguments.

    Args:
        a: One version (id or name).
        b: The other version (id or name).
        ontology: Loaded ontology; the default build if omitted.

    Returns:
        ``{"from", "to", "changes", "unchanged", "not_mapped"}``. Each change
        has ``type`` (``same`` when only the position moved, ``renamed``,
        ``merged``, ``new``), the members involved and the evidence (document,
        page, quote) of every correspondence on its chain. ``not_mapped`` lists
        the members whose correspondence is not in the ontology yet.
    """
    onto = ontology or load()
    older, newer = _single(onto, a), _single(onto, b)
    if _derives(onto, older, newer):
        older, newer = newer, older
    usable = [m for m in onto.mappings if m["status"] in USABLE]
    edges: dict[str, list[dict]] = {}
    for mapping in usable:
        if mapping["type"] != "new":
            edges.setdefault(mapping["from"], []).append(mapping)

    paths: list[tuple[str, list[dict]]] = []
    for member in onto.frameworks[older]["members"]:
        if member["status"] not in USABLE:
            continue
        frontier = [(member["key"], [])]
        while frontier:
            current, steps = frontier.pop()
            if current.startswith(f"{newer}#"):
                paths.append((member["key"], steps))
                continue
            if len(steps) < 4:
                frontier.extend((m["to"], steps + [m]) for m in edges.get(current, []))

    changes, unchanged = [], []
    for start, steps in paths:
        kinds = {step["type"] for step in steps}
        kind = "merged" if "merged" in kinds else "renamed" if "renamed" in kinds else "same"
        entry = {
            "type": kind,
            "from": _member(onto, start),
            "to": _member(onto, steps[-1]["to"]),
            "evidence": _evidence(steps),
        }
        entry["moved"] = entry["from"]["position"] != entry["to"]["position"]
        (unchanged if kind == "same" and not entry["moved"] else changes).append(entry)

    reached_old = {start for start, _ in paths}
    reached_new = {steps[-1]["to"] for _, steps in paths}
    # A member without a correspondence is "new" only when a source says it
    # was introduced; otherwise its correspondence is simply not known yet.
    not_mapped = []
    for member in onto.frameworks[newer]["members"]:
        if member["status"] in USABLE and member["key"] not in reached_new:
            origin = _origin(usable, member["key"])
            if origin and origin[0]["type"] == "new":
                changes.append({"type": "new", "to": _member(onto, member["key"]), "evidence": origin})
            else:
                not_mapped.append(_member(onto, member["key"]))
    for member in onto.frameworks[older]["members"]:
        if member["status"] in USABLE and member["key"] not in reached_old:
            not_mapped.append(_member(onto, member["key"]))
    return {
        "from": onto._framework_view(older) | {"members": None},
        "to": onto._framework_view(newer) | {"members": None},
        "changes": changes,
        "unchanged": unchanged,
        "not_mapped": not_mapped,
    }


def _single(onto: Ontology, name: str) -> str:
    if name in onto.frameworks:
        return name
    ids = onto.resolve_frameworks(name)
    if len(ids) != 1:
        raise ValueError(f"{name!r} indica {len(ids)} versioni: {', '.join(ids) or 'nessuna'}")
    return ids[0]


def _origin(mappings: list[dict], key: str) -> list[dict]:
    """Correspondences leading back from a member to where it was introduced."""
    chain, current = [], key
    while current:
        step = next((m for m in mappings if m["to"] == current), None)
        if step is None or step in chain:
            break
        chain.insert(0, step)
        current = step["from"]
    return _evidence(chain)


def _derives(onto: Ontology, child: str, parent: str) -> bool:
    seen, frontier = set(), list(onto.frameworks[child]["derived_from"])
    while frontier:
        current = frontier.pop()
        if current == parent:
            return True
        if current in seen or current not in onto.frameworks:
            continue
        seen.add(current)
        frontier.extend(onto.frameworks[current]["derived_from"])
    return False


def _member(onto: Ontology, key: str) -> dict:
    member = onto.members[key]
    return {
        "framework": member["framework_id"],
        **{k: member[k] for k in ("position", "label_it", "label_en", "document", "page", "status")},
    }


def _evidence(steps: list[dict]) -> list[dict]:
    return [
        {k: step[k] for k in ("type", "document", "page", "quote", "status")}
        for step in steps
    ]


def concetto(nome: str, *, ontology: Ontology | None = None) -> dict | None:
    """A concept with its definitions, place in the hierarchy and versions.

    Args:
        nome: Label or synonym, Italian or English.
        ontology: Loaded ontology; the default build if omitted.

    Returns:
        The concept, or None if the name is unknown. ``members`` lists where the
        concept appears in each framework version; ``versions`` is filled when
        the name is a framework family (e.g. "10R").
    """
    onto = ontology or load()
    concept_id = onto.resolve_concept(nome)
    if concept_id is None:
        return None
    concept = onto.concepts[concept_id]
    view = onto._concept_view(concept_id)
    view.update(
        {
            "synonyms_it": concept["synonyms_it"],
            "synonyms_en": concept["synonyms_en"],
            "broader": [onto._concept_view(c) | {"definitions": None} for c in concept["broader"]],
            "ancestors": [onto._concept_view(c)["label_it"] or onto._concept_view(c)["label_en"] for c in concept["ancestors"]],
            "narrower": [onto._concept_view(c) | {"definitions": None} for c in concept["narrower"] if onto.concepts[c]["status"] in USABLE],
            "related": [onto._concept_view(c) | {"definitions": None} for c in concept["related"]],
            "members": [_member(onto, key) for key in concept["members"] if onto.members[key]["status"] in USABLE],
            "graph_nodes": concept["graph_nodes"],
            "versions": onto.families.get(concept_id, {}).get("frameworks", []),
        }
    )
    return view


def prima(a: str, b: str, *, ontology: Ontology | None = None) -> dict | None:
    """Which of two items comes first in every version that holds both.

    The order rule of their family says what "first" means (for the R
    strategies, more circular); without one the answer is only the order of
    the list.

    Args:
        a: Label of the first item.
        b: Label of the second item.
        ontology: Loaded ontology; the default build if omitted.

    Returns:
        ``{"a", "b", "rule", "versions"}``, each version with both members and
        ``first`` (``"a"`` or ``"b"``); None if either name is unknown.
    """
    onto = ontology or load()
    ids = [onto.resolve_concept(a), onto.resolve_concept(b)]
    if None in ids:
        return None
    members = [
        {m.split("#")[0]: onto.members[m] for m in onto.concepts[cid]["members"] if onto.members[m]["status"] in USABLE}
        for cid in ids
    ]
    versions = []
    for framework_id in sorted(set(members[0]) & set(members[1]), key=onto._age):
        first, second = (position_number(members[i][framework_id]["position"]) for i in (0, 1))
        if first is None or second is None or first == second:
            continue
        versions.append(
            {
                "framework": framework_id,
                "a": _member(onto, members[0][framework_id]["key"]),
                "b": _member(onto, members[1][framework_id]["key"]),
                "first": "a" if first < second else "b",
            }
        )
    families = {onto.frameworks[v["framework"]]["family"] for v in versions}
    rules = [onto.families[f]["order"] for f in sorted(families) if "order" in onto.families.get(f, {})]
    return {
        "a": onto._concept_view(ids[0]),
        "b": onto._concept_view(ids[1]),
        "rule": rules[0] if len(rules) == 1 else None,
        "versions": versions,
    }


def piu_specifici(concetto_nome: str, gradini: int = 1, *, ontology: Ontology | None = None) -> list[dict]:
    """Narrower concepts, down to a number of steps.

    Args:
        concetto_nome: Label or synonym of the starting concept.
        gradini: How many steps down; capped at 2, beyond which the concepts
            stop being useful for widening a search.
        ontology: Loaded ontology; the default build if omitted.

    Returns:
        Narrower concepts with ``level`` (1 = direct), parent and definitions.
    """
    onto = ontology or load()
    start = onto.resolve_concept(concetto_nome)
    if start is None:
        return []
    result, frontier = [], [(start, 0)]
    while frontier:
        current, level = frontier.pop(0)
        if level >= min(int(gradini), 2):
            continue
        for child in onto.concepts[current]["narrower"]:
            if onto.concepts[child]["status"] not in USABLE:
                continue
            result.append(onto._concept_view(child) | {"level": level + 1, "parent": current})
            frontier.append((child, level + 1))
    return result


def casi(
    strategia: str | None = None,
    territorio: str | None = None,
    *,
    ontology: Ontology | None = None,
) -> list[dict]:
    """Cases of the corpus, with the strategies their authors assign.

    Args:
        strategia: A strategy (label, synonym or position code such as
            ``R4``); only cases applying it are returned.
        territorio: A word of the territory, e.g. ``Cuneo``.
        ontology: Loaded ontology; the default build if omitted.

    Returns:
        Cases with name, document, page, territory, promoter and the
        strategies as members (framework, position, label).
    """
    onto = ontology or load()
    wanted: set[str] | None = None
    if strategia:
        code = strategia.strip().upper().replace(" ", "")
        concept = onto.resolve_concept(strategia)
        wanted = {
            key for key, m in onto.members.items()
            if m["position"].upper() == code or (concept and m["concept_id"] == concept)
        }
    found = []
    for case in onto.data.get("cases", {}).values():
        if case["status"] not in USABLE:
            continue
        if wanted is not None and not wanted & set(case["strategies"]):
            continue
        if territorio and _compact(territorio) not in _compact(case["territory"]):
            continue
        found.append(_case_view(onto, case))
    return found


def _case_view(onto: Ontology, case: dict) -> dict:
    return {
        **{k: case[k] for k in ("id", "name", "document", "page", "territory", "actors", "status")},
        "strategies": [_member(onto, key) for key in case["strategies"] if key in onto.members],
    }


def verifica(elenco: list[str], framework_nome: str, versione: str | None = None, *, ontology: Ontology | None = None) -> dict:
    """Check a list (e.g. from a generated answer) against a framework version.

    Args:
        elenco: Items as written in the answer.
        framework_nome: Framework the answer claims to list.
        versione: Version the answer claims; without it the list is checked
            against every version and the closest is reported.
        ontology: Loaded ontology; the default build if omitted.

    Returns:
        ``{"framework", "ok", "matched", "missing", "extra", "unjudged",
        "complete", "order_ok", "best_match", "by_version"}``. ``framework``
        is the version checked against (the claimed one, or the best match
        when none is claimed). ``extra`` is filled only when that version has
        no gap in its positions; otherwise unmatched items are ``unjudged``.
    """
    onto = ontology or load()
    ids = onto.resolve_frameworks(framework_nome, versione)
    if not ids:
        raise ValueError(f"framework sconosciuto: {framework_nome!r}")
    family = onto.frameworks[ids[0]]["family"]
    candidates = [fid for fid, f in onto.frameworks.items() if f["family"] == family]
    results = {fid: _check_list(onto, elenco, fid) for fid in candidates}
    best = max(results, key=lambda fid: (results[fid]["score"], -candidates.index(fid)))
    claimed = ids[0] if len(ids) == 1 else best
    report = results[claimed]
    extra = report["extra"] if report["complete"] else []
    return {
        "framework": claimed,
        "ok": not report["missing"] and not extra and report["order_ok"],
        "matched": report["matched"],
        "missing": report["missing"],
        "extra": extra,
        # Items that match nothing in a version the ontology holds only in
        # part: they may be its missing members, so they are not called extra.
        "unjudged": [] if report["complete"] else report["extra"],
        "complete": report["complete"],
        "order_ok": report["order_ok"],
        "best_match": best,
        "by_version": {fid: {k: v for k, v in r.items() if k != "score"} for fid, r in results.items()},
    }


def _stem(text: str) -> str:
    # Verb and noun of the same strategy share their first six letters
    # (riciclare/riciclo, recuperare/recupero energetico); different strategies
    # of a family do not (riconvertire/ricondizionare).
    words = _compact(text).split()
    return words[0][:6] if words and len(words[0]) >= 6 else _compact(text)


def _complete(onto: Ontology, framework_id: str, members: list[dict]) -> bool:
    # Versions of a family number from the same start (R0 for the R
    # strategies): a version starting later is missing its first member.
    family = onto.frameworks[framework_id]["family"]
    starts = [
        position_number(m["position"])
        for fid, f in onto.frameworks.items()
        if f["family"] == family
        for m in f["members"]
        if m["status"] in USABLE and position_number(m["position"]) is not None
    ]
    numbered = sorted(position_number(m["position"]) for m in members if position_number(m["position"]) is not None)
    return bool(numbered) and numbered == list(range(min(starts), min(starts) + len(numbered)))


def position_number(position: str) -> int | None:
    """The number in a position code (``R4`` → 4, ``2`` → 2), None for ``R+``.

    Args:
        position: Position as written in the sheet.

    Returns:
        The number, or None.
    """
    digits = "".join(ch for ch in position if ch.isdigit())
    return int(digits) if digits else None


def _check_list(onto: Ontology, items: list[str], framework_id: str) -> dict:
    members = [m for m in onto.frameworks[framework_id]["members"] if m["status"] in USABLE]
    forms: dict[str, str] = {}
    stems: dict[str, str] = {}
    for member in members:
        concept = onto.concepts.get(member["concept_id"], {})
        for form in [member["label_it"], member["label_en"], concept.get("label_it", ""), concept.get("label_en", ""), *concept.get("synonyms_it", []), *concept.get("synonyms_en", [])]:
            if form:
                forms.setdefault(_compact(form), member["key"])
                stems.setdefault(_stem(form), member["key"])
    matched, extra, order = [], [], []
    for item in items:
        key = forms.get(_compact(item)) or stems.get(_stem(item))
        if key is None:
            extra.append(item)
            continue
        member = onto.members[key]
        matched.append({"item": item, **_member(onto, key)})
        order.append(members.index(member))
    found = {m["position"] for m in matched}
    missing = [_member(onto, m["key"]) for m in members if m["position"] not in found]
    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "order_ok": order == sorted(order),
        # With a gap in the positions the version is not fully in the
        # ontology yet, so an unmatched item may be the missing member.
        "complete": _complete(onto, framework_id, members),
        "score": len(matched) - len(missing) - len(extra),
    }
