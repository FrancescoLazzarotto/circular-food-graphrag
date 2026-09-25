"""Build the Italian counterpart of the frozen reference set.

Only the ``query`` field changes. Expected entities, expected relations, the
reference answer and the scoring block are copied verbatim, so the two files
score against the same annotation and a difference between them is a property
of the question language rather than of the benchmark.

The reference answers stay in English on purpose. They feed the textual
similarity metrics, which are not comparable across languages and are therefore
not reported on this set; the concept and grounding channels, which match
surface forms carried in ``alt_labels``, are.

The translations below were produced for this experiment and are not part of the
curated benchmark. They carry their own version tag so no run can confuse them
with the frozen English set.
"""

from __future__ import annotations

import json
from pathlib import Path

IT_QUERIES: dict[str, str] = {
    "Q01": "Quali sono le tre C del framework Circular Economy for Food?",
    "Q02": (
        "Per quanti Obiettivi di Sviluppo Sostenibile il cibo funge da elemento "
        "di connessione?"
    ),
    "Q03": "Che percentuale degli scarti di vinificazione rappresentano i raspi?",
    "Q04": "Che cos'è il siero di latte e come viene prodotto?",
    "Q05": (
        "Quali sono i quattro cicli di attuazione della metabolizzazione e in "
        "che cosa differiscono?"
    ),
    "Q06": (
        "Come si lega la Ciclicità ai suoi sotto-concetti di estensione, "
        "metabolizzazione e rinnovabilità?"
    ),
    "Q07": (
        "Quali tipi di Capitale sono descritti come inscindibili e che cosa li lega?"
    ),
    "Q08": (
        "Ricostruisci il percorso di valorizzazione della lolla di riso: da quale "
        "processo proviene e che cosa può diventare?"
    ),
    "Q09": (
        "Come si collega il principio di co-evoluzione al concetto di simbiosi "
        "industriale nel sistema alimentare?"
    ),
    "Q10": (
        "Quali composti di valore si possono estrarre dalle vinacce e quali sono "
        "le loro applicazioni?"
    ),
    "Q11": "Che cos'è la scotta e in che cosa si differenzia dal siero di latte?",
    "Q12": (
        "Quali proprietà farmaceutiche sono state riscontrate nella pula di riso "
        "e quali composti ne sono responsabili?"
    ),
    "Q13": (
        "Che cos'è il latticello, da quale processo ha origine e in quale quantità?"
    ),
    "Q14": (
        "Qual è il volume annuo di produzione di vinacce nella regione Veneto?"
    ),
    "Q15": "Secondo il framework, quale delle tre C è la più importante?",
    "Q16": (
        "Nel caso del biogas di Via del Campo, quali due materiali in ingresso "
        "alimentano l'impianto e in quali proporzioni?"
    ),
    "Q17": (
        "Di quanto può ridurre l'uso di fertilizzanti chimici il digestato secondo "
        "il caso Via del Campo?"
    ),
    "Q18": (
        "Che percentuale delle bottiglie di acqua minerale in Italia è di plastica?"
    ),
    "Q19": (
        "In media, quanto latte viene sprecato a persona ogni settimana nelle "
        "famiglie italiane?"
    ),
    "Q20": (
        "Ricostruisci il ciclo circolare nel caso Via del Campo: dall'input in "
        "campo all'output in campo."
    ),
    "Q21": (
        "Quali cinque priorità abilitano la co-evoluzione e come danno "
        "complessivamente resilienza al paradigma circolare?"
    ),
    "Q22": (
        "In che modo la dispersione geografica delle imprese di imbottigliamento "
        "dell'acqua in Piemonte ostacola la simbiosi industriale?"
    ),
    "Q23": (
        "Collega il concetto di Estensione (responsabilità estesa del produttore) "
        "a un esempio concreto di riuso di sottoprodotti nella filiera del vino."
    ),
    "Q24": (
        "In che cosa il ciclo di metabolizzazione a cascata differisce dal ciclo "
        "corto, con un esempio della filiera alimentare?"
    ),
    "Q25": (
        "Che cosa sono le fecce di vino e quali sostanze di valore contengono?"
    ),
    "Q26": (
        "Che cosa sono i fanghi nella lavorazione lattiero-casearia e quali "
        "operazioni li generano?"
    ),
    "Q27": (
        "Perché la lolla di riso è preferita ad altri filler naturali per le "
        "bioplastiche?"
    ),
    "Q28": (
        "Che fine fa la paglia di riso e perché la bruciatura in campo aperto è "
        "soggetta a restrizioni?"
    ),
    "Q29": (
        "Qual è il prezzo di mercato esatto per tonnellata delle vinacce in "
        "Piemonte nel 2023?"
    ),
    "Q30": (
        "Quale delle sei filiere piemontesi produce più emissioni di CO2?"
    ),
}


def main() -> None:
    """Write ``gold_v3_it.json`` from ``gold_v3.json`` with the Italian queries."""
    root = Path(__file__).resolve().parents[2]
    gold_dir = root / "evaluation" / "gold"
    src = gold_dir / "gold_v3.json"
    dst = gold_dir / "gold_v3_it.json"

    data = json.loads(src.read_text(encoding="utf-8"))
    queries = data["queries"]

    missing = {q["query_id"] for q in queries} - set(IT_QUERIES)
    if missing:
        raise SystemExit(f"no Italian query for: {sorted(missing)}")

    for query in queries:
        query["query_en"] = query["query"]
        query["query"] = IT_QUERIES[query["query_id"]]

    meta = data.setdefault("_meta", {})
    meta["derived_from"] = src.name
    meta["language"] = (
        "queries in Italian, translated from the frozen English set for the "
        "cross-lingual experiment; expected answers and all annotation unchanged "
        "and still in English"
    )
    meta["translation_note"] = (
        "Translations were produced for the cross-lingual run and were not "
        "reviewed by the benchmark curator. Textual similarity metrics are not "
        "comparable against the English reference answers and are not reported."
    )

    dst.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {dst} with {len(queries)} Italian queries")


if __name__ == "__main__":
    main()
