"""Populate gold.json expected_entities[].alt_labels with IT+EN surface forms.

Sources, in precedence order:
  1. AGROVOC / ChEBI multilingual labels fetched from the vocabularies themselves
     (vocab_labels.json, produced by fetch_vocab_labels.py)
  2. A curated IT/EN lexicon for the benchmark-local (urn:ceff:) concepts, which
     have no external vocabulary to draw from.

Both sources are pipeline-independent: nothing is read from our Neo4j KG or from
any run's output, so alt_labels cannot be tuned to a pipeline's vocabulary.

Guarantees enforced before writing:
  - no alt_label equals its own normalised_label
  - no surface form maps to two different gold concepts (collision check)
  - ChEBI chemical-nomenclature synonyms are excluded (see _chebi_keep)
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

GOLD = Path(__file__).resolve().parents[1] / "gold.json"
VOCAB = Path(__file__).with_name("vocab_labels.json")

# Curated IT/EN surface forms, keyed by normalised_label.
# IT forms are what an Italian-source-doc pipeline would emit; EN forms cover
# morphology and common synonyms. Deliberately conservative: no near-synonyms
# that could denote a *different* gold concept.
CURATED: dict[str, list[str]] = {
    # --- CEFF framework concepts (3 C's and sub-concepts) ---
    "capital": ["capitale"],
    "cyclicality": ["ciclicità", "ciclicita", "cyclicity"],
    "co-evolution": ["coevoluzione", "co-evoluzione", "coevolution"],
    "natural capital": ["capitale naturale"],
    "cultural capital": ["capitale culturale"],
    "economic capital": ["capitale economico"],
    "relational capital": ["capitale relazionale"],
    "extension": ["estensione", "responsabilità estesa del produttore",
                  "extended producer responsibility"],
    "renewability": ["rinnovabilità", "rinnovabilita"],
    "metabolisation": ["metabolizzazione", "metabolization"],
    "short metabolisation cycle": ["ciclo corto", "ciclo di metabolizzazione corto",
                                   "short cycle", "short metabolization cycle"],
    "long metabolisation cycle": ["ciclo lungo", "ciclo di metabolizzazione lungo",
                                  "long cycle", "long metabolization cycle"],
    "cascading metabolisation cycle": ["ciclo a cascata", "cascading cycle",
                                       "cascade cycle", "cascading metabolization cycle"],
    "pure metabolisation cycle": ["ciclo puro", "pure cycle", "pure metabolization cycle"],
    "sustainable development goals": ["obiettivi di sviluppo sostenibile", "sdgs", "sdg",
                                      "sustainable development goal"],
    # --- co-evolution priorities ---
    "solidarity": ["solidarietà", "solidarieta"],
    "dialogue": ["dialogo", "dialog"],
    "cooperation": ["cooperazione"],
    "sharing": ["condivisione"],
    "symbiosis": ["simbiosi"],
    "industrial symbiosis": ["simbiosi industriale"],
    "collaboration": ["collaborazione"],
    # --- wine chain ---
    "grape stalk": ["raspi", "raspo", "raspi d'uva", "grape stalks", "grape stems"],
    "grape pomace": ["vinacce", "vinaccia", "grape marc", "vinacce d'uva"],
    "wine lees": ["fecce", "feccia", "fecce di vino", "lees", "wine dregs"],
    "phenolic acid": ["acidi fenolici", "acido fenolico", "phenolic acids"],
    "calcium tartrate": ["tartrato di calcio"],
    # normalised_label carries a parenthetical gloss no pipeline would emit verbatim;
    # both halves are listed as alts so either surface form matches. Flagged for the
    # gold author: normalised_label should probably be 'potassium bitartrate'.
    "potassium bitartrate (cream of tartar)": ["cremor tartaro", "cremortartaro",
                                               "bitartrato di potassio", "cream of tartar",
                                               "potassium bitartrate",
                                               "potassium hydrogen tartrate"],
    "yeast": ["lievito", "lieviti", "yeasts"],
    "pruning residue": ["residui di potatura", "sarmenti", "pruning residues", "vine shoots"],
    "soil fertilisation": ["fertilizzazione del suolo", "concimazione", "soil fertilization"],
    "winemaking": ["vinificazione", "wine production", "wine-making"],
    # --- dairy chain ---
    "whey": ["siero", "siero di latte", "milk whey"],
    "scotta": ["ricotta whey", "second whey", "siero di ricotta"],
    "ricotta processing": ["produzione di ricotta", "lavorazione della ricotta",
                           "ricotta production"],
    "milk coagulation": ["coagulazione del latte", "coagulazione", "milk clotting"],
    "curd separation": ["separazione della cagliata", "spurgo della cagliata",
                        "curd drainage"],
    "buttermilk": ["latticello"],
    "butter churning": ["burrificazione", "zangolatura", "churning"],
    "pasta-filata cheese production": ["produzione di formaggi a pasta filata",
                                       "pasta filata", "lavorazione della pasta filata"],
    "dairy sludge": ["fanghi", "fanghi caseari", "fanghi di lavorazione",
                     "sludge", "cheese sludge"],
    "wastewater treatment": ["depurazione", "trattamento delle acque reflue",
                             "depurazione delle acque"],
    "brine salting": ["salamoia", "salatura in salamoia", "salting"],
    "milk waste": ["spreco di latte", "latte sprecato", "wasted milk", "milk wastage"],
    # --- rice chain ---
    "rice husks": ["lolla", "lolla di riso", "rice husk", "rice hulls"],
    "rice bran": ["pula", "pula di riso", "crusca di riso"],
    "rice straw": ["paglia di riso", "rice straws"],
    "rice dehusking": ["sbramatura", "decorticazione del riso", "dehusking", "husking"],
    "rice": ["riso", "risone", "paddy"],
    "bioplastic": ["bioplastica", "bioplastiche", "bioplastics"],
    "natural filler": ["filler naturale", "filler naturali", "riempitivo naturale",
                       "cariche naturali", "natural fillers"],
    "pm10": ["polveri sottili", "particolato", "particulate matter", "pm 10"],
    "open-field burning": ["abbruciamento", "bruciatura in campo", "combustione in campo",
                           "open burning", "field burning", "abbruciamento in campo"],
    # --- biogas / Via del Campo case ---
    "animal manure": ["letame", "liquame", "reflui zootecnici", "effluenti zootecnici",
                      "manure", "livestock manure"],
    "crop biomass": ["biomassa vegetale", "biomasse agricole", "biomassa colturale",
                     "crop biomasses"],
    "biogas": ["bio-gas"],
    "anaerobic digester": ["digestore anaerobico", "digestori anaerobici",
                           "anaerobic digesters", "digestore"],
    "digestate": ["digestato"],
    "chemical fertiliser": ["fertilizzante chimico", "fertilizzanti chimici",
                            "concime chimico", "chemical fertilizer", "mineral fertiliser"],
    "field": ["campo", "campi", "terreno agricolo", "agricultural field"],
    # --- water chain ---
    "plastic bottle": ["bottiglia di plastica", "bottiglie di plastica", "plastic bottles",
                       "pet bottle", "bottiglie in pet"],
    "mineral water": ["acqua minerale", "acque minerali"],
    "mineral-water source": ["sorgente", "sorgenti", "fonte", "sorgente di acqua minerale",
                             "mineral water source", "spring"],
    "water-bottling firm": ["imbottigliatori", "aziende di imbottigliamento",
                            "imprese di imbottigliamento", "water bottling firms",
                            "bottling company"],
    # --- chemistry (ChEBI-backed, curated IT forms) ---
    "polyphenol": ["polifenoli", "polifenolo", "polyphenols"],
    "flavonoid": ["flavonoidi", "flavonoids"],
    "sterol": ["steroli", "sterols"],
    "cholesterol": ["colesterolo"],
}

# Surface forms that must not be attached to a specific concept, keyed by
# (normalised_label, surface form) so a block on one concept never strips the
# same form from another concept that legitimately owns it.
BLOCKLIST: dict[tuple[str, str], str] = {
    ("rice husks", "pula di riso"):
        "AGROVOC gives 'Pula di riso' as the IT prefLabel of rice husks, but the corpus "
        "and gold use 'pula' for rice BRAN. Blocked here so 'pula*' stays unambiguously "
        "bran; rice bran keeps it.",
    ("potassium bitartrate (cream of tartar)", "faecula"):
        "archaic Latin ChEBI synonym, one edit from 'feccia' (wine lees)",
    ("potassium bitartrate (cream of tartar)", "faecla"):
        "archaic Latin ChEBI synonym, one edit from 'feccia' (wine lees)",
    ("potassium bitartrate (cream of tartar)", "faccla"):
        "archaic Latin ChEBI synonym, one edit from 'feccia' (wine lees)",
    ("potassium bitartrate (cream of tartar)", "faccula"):
        "archaic Latin ChEBI synonym, one edit from 'feccia' (wine lees)",
}


def normalise(text: str) -> str:
    """Apply the gold normalisation: lowercase, strip whitespace/punctuation.

    Parentheses are left alone: a trailing ')' that closes a parenthetical gloss
    ('potassium bitartrate (cream of tartar)') is part of the term, not surrounding
    punctuation, and stripping it corrupts the label.
    """
    out = text.strip().lower()
    out = re.sub(r"^[\s\-—–\"'“”«»]+|[\s\-—–\"'“”«».,;:]+$", "", out)
    return re.sub(r"\s+", " ", out)


def _fold(text: str) -> str:
    """Accent-insensitive key, for detecting near-duplicate surface forms."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _chebi_keep(candidate: str, pref: str) -> bool:
    """Keep a ChEBI synonym only if it is a morphological variant of the pref label.

    ChEBI synonym lists are dominated by systematic chemical nomenclature
    ('2-aryl-1-benzopyran', 'Cholest-5-en-3beta-ol') that no natural-language
    answer would ever emit. Importing them would inflate the accepted set with
    noise, so only case/plural variants of the preferred label are taken.
    """
    cand, base = _fold(normalise(candidate)), _fold(normalise(pref))
    if re.search(r"\d", cand) or len(cand) > 30:
        return False
    return cand in {base, base + "s", base + "es"} or base in {cand, cand + "s"}


def collect(entity: dict, vocab: dict) -> tuple[list[str], list[str]]:
    """Return (alt_labels, provenance_notes) for one gold entity."""
    norm = normalise(entity["normalised_label"])
    candidates: dict[str, str] = {}  # normalised form -> source tag

    def offer(raw: str, source: str) -> None:
        """Add `raw` as a candidate unless it is the label itself or blocked."""
        form = normalise(raw)
        if not form or form == norm:
            return
        if (norm, form) in BLOCKLIST:
            return
        candidates.setdefault(form, source)

    for existing in entity.get("alt_labels") or []:
        offer(existing, "original")

    uri = entity.get("uri") or ""
    entry = vocab.get(uri)
    if entry:
        src = entry["source"]
        for label in entry.get("pref", []):
            if src == "ChEBI":
                continue
            offer(label, src)
        for label in entry.get("alt", []):
            if src == "ChEBI":
                pref = (entry.get("pref") or [entity["label"]])[0]
                if not _chebi_keep(label, pref):
                    continue
            offer(label, src)

    for label in CURATED.get(norm, []):
        offer(label, "curated")

    # Forms already in the gold first, then the rest alphabetically.
    ordered = sorted(candidates, key=lambda f: (candidates[f] != "original", f))
    return ordered, [f"{f} [{candidates[f]}]" for f in ordered]


def main(write: bool) -> int:
    """Fill alt_labels for every gold entity, refusing on any collision.

    Args:
        write: Write the gold back; otherwise a dry run.

    Returns:
        Process exit code; 1 when a surface form maps to two gold concepts.
    """
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    vocab = json.loads(VOCAB.read_text(encoding="utf-8")) if VOCAB.exists() else {}

    owners: dict[str, set[str]] = defaultdict(set)  # surface form -> normalised_labels
    per_entity: dict[str, list[str]] = {}
    filled = 0

    for query in gold["queries"]:
        for entity in query["expected_entities"]:
            norm = normalise(entity["normalised_label"])
            alts, _ = collect(entity, vocab)
            per_entity[norm] = alts
            owners[norm].add(norm)
            for form in alts:
                owners[form].add(norm)

    collisions = {f: sorted(o) for f, o in owners.items() if len(o) > 1}
    if collisions:
        print("COLLISIONS — a surface form maps to >1 gold concept:")
        for form, concepts in collisions.items():
            print(f"  {form!r} -> {concepts}")
        print("\nRefusing to write. Resolve in CURATED/BLOCKLIST first.")
        return 1

    for query in gold["queries"]:
        for entity in query["expected_entities"]:
            norm = normalise(entity["normalised_label"])
            alts = per_entity[norm]
            if alts and not entity.get("alt_labels"):
                filled += 1
            entity["alt_labels"] = alts

    total = sum(len(q["expected_entities"]) for q in gold["queries"])
    with_alts = sum(
        1 for q in gold["queries"] for e in q["expected_entities"] if e["alt_labels"]
    )
    n_alts = sum(len(e["alt_labels"]) for q in gold["queries"] for e in q["expected_entities"])

    print(f"entities with alt_labels: {with_alts}/{total} (newly filled: {filled})")
    print(f"total alt surface forms:  {n_alts}")
    print("no collisions detected")

    if write:
        gold["_meta"]["alt_labels_provenance"] = {
            "generated": "2026-07-16",
            "sources": [
                "AGROVOC skos:prefLabel/altLabel/hiddenLabel (IT+EN) fetched from "
                "agrovoc.fao.org for entities with an AGROVOC URI",
                "ChEBI labels via EBI OLS4, restricted to morphological variants of the "
                "preferred label (systematic chemical nomenclature excluded)",
                "curated IT/EN lexicon for urn:ceff: benchmark-local concepts, which have "
                "no external vocabulary",
            ],
            "independence": "No surface form was taken from the evaluated KG, from any "
                            "pipeline run, or from any system output. Built before the "
                            "evaluation runs, per protocol §6.",
            "invariants": [
                "no alt_label equals its own normalised_label",
                "no surface form maps to two different gold concepts",
            ],
            "known_terminology_conflicts": [
                "AGROVOC gives 'Pula di riso' as the Italian prefLabel of rice husks "
                "(c_24892) and 'Crusca di riso' for rice bran (c_77d35680). The corpus and "
                "this gold use 'lolla' for husks and 'pula' for bran. Gold/corpus usage is "
                "authoritative here; 'pula di riso' is blocked on husks to keep 'pula' "
                "unambiguous. Flagged for expert review.",
            ],
        }
        GOLD.write_text(json.dumps(gold, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwritten -> {GOLD}")
    else:
        print("\ndry-run (pass --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main(write="--write" in sys.argv))
