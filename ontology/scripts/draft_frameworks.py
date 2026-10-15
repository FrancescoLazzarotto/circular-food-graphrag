"""Draft the framework sheets from the source pages with the served model.

Each framework is drafted twice, with two differently worded instructions, and
the output is constrained to a JSON schema that mirrors the sheets. The model
only reads and copies: positions, labels and quotes all have to come from the
pages it is given, and ``accept.py`` checks them against the corpus.

Usage::

    /mnt/storage/flazzarotto/venvs/ontology/bin/python ontology/scripts/draft_frameworks.py
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from common import DRAFTS_DIR, page_text

ENDPOINT = "http://localhost:8001/v1/chat/completions"
MODEL = "RedHatAI/Qwen3.8-27B-INT4"

# Department publications and official reports only; ``accept.py`` reads the
# kind from here to apply the source criterion.
SOURCE_KINDS = {
    "report_mattm_definitivo": "dipartimento: rapporto UNISG sull'economia circolare nel sistema agroalimentare piemontese",
    "it_circular_city_selfie": "dipartimento: pubblicazione UNISG con Mercato Circolare",
    "sustainability_16_08960_v2_it": "dipartimento: articolo Destefanis, Cela, Torri, Fassio (2024)",
    "f_fassio_diid_2021": "dipartimento: articolo di F. Fassio che introduce le 3C",
}

R_TASKS = {
    "r_potting_2017": {
        "version": "Potting et al. 2017",
        "document": "report_mattm_definitivo",
        "pages": [54, 55],
    },
    "r_errando": {
        "version": "eRRRando (Mercato Circolare)",
        "document": "it_circular_city_selfie",
        "pages": [13, 14],
    },
    "r_nodes_2024": {
        "version": "NODES 2024",
        "document": "sustainability_16_08960_v2_it",
        # The article's own positions are given only in the headings of the
        # case tables, so those pages are the source. Page 5 is left out: it
        # lists the 9R the article starts from, and with it in view the model
        # copies that list as if it were the article's version.
        "pages": [6, 13, 15, 16, 19, 20, 23, 24, 26, 28],
    },
}

C3_TASK = {
    "id": "3c_fassio_2021",
    "version": "Fassio 2021",
    "document_en": "f_fassio_diid_2021",
    "pages_en": [7, 8, 9],
    "document_it": "report_mattm_definitivo",
    "pages_it": [68, 69, 70],
}

MAPPING_TASKS = [
    {"from": "r_potting_2017", "to": "r_errando", "document": "it_circular_city_selfie", "pages": [13]},
    {"from": "r_errando", "to": "r_nodes_2024", "document": "sustainability_16_08960_v2_it", "pages": [6]},
    {"from": "r_potting_2017", "to": "r_nodes_2024", "document": "sustainability_16_08960_v2_it", "pages": [5, 6]},
]

R_POSITIONS = [f"R{i}" for i in range(10)] + ["R+"]

COMMON_R = (
    " Il nome del framework va copiato alla lettera com'è nel testo, e "
    "name_quote deve contenerlo. Se le pagine descrivono sia il modello di "
    "partenza sia quello adottato dagli autori, riporta quello adottato, con "
    "le loro modifiche. Come 'quote' di una voce preferisci la frase che "
    "elenca le strategie in ordine o che ne scrive il numero R: è quella da "
    "cui si legge la posizione."
)

INSTRUCTIONS = {
    "a": (
        "Sei un estrattore di dati. Leggi le pagine e riporta il framework di "
        "strategie R che descrivono, una voce per ogni strategia, nell'ordine "
        "del testo. Usa solo ciò che è scritto. Ogni campo 'quote' è una frase "
        "copiata alla lettera da una sola pagina, senza tagli né '...', e deve "
        "contenere il nome della voce. Se il numero R non è scritto, deducilo "
        "dall'ordine dell'elenco partendo da R0 e metti position_in_text=false. "
        "label_en solo se il testo dà il nome inglese, altrimenti stringa vuota. "
        "family è il gruppo in cui il testo mette la voce, altrimenti vuoto. "
        "Definizioni solo se la frase nomina la strategia." + COMMON_R
    ),
    "b": (
        "Compito di curatela di un tesauro. Per prima cosa individua nelle "
        "pagine il passaggio che elenca le strategie R; poi, per ciascuna, "
        "copia la frase esatta che la nomina (testo identico alla pagina, "
        "nessuna omissione) e solo dopo compila numero, nome italiano, nome "
        "inglese (vuoto se assente nel testo) e famiglia (vuota se il testo non "
        "raggruppa). Numero R: quello scritto; se manca, la posizione "
        "nell'elenco contando da R0, con position_in_text=false. Non aggiungere "
        "strategie che il testo non nomina. Una definizione va riportata solo "
        "se la frase contiene il nome della strategia." + COMMON_R
    ),
}

C3_INSTRUCTIONS = {
    "a": (
        "Sei un estrattore di dati. Le pagine EN e IT descrivono le stesse 3C. "
        "Riporta le tre C nell'ordine del testo (posizione 1, 2, 3) e i "
        "concetti più specifici che il testo elenca dentro ciascuna (anche i "
        "sottoconcetti di un sottoconcetto). Per ogni voce copia alla lettera, "
        "da una sola pagina, una frase inglese e una italiana che la nominano "
        "e la descrivono; nessun taglio né '...'. parent_label_en è l'etichetta "
        "inglese del concetto più generale. Non inventare voci. Il nome del "
        "framework va copiato alla lettera e name_quote deve contenerlo."
    ),
    "b": (
        "Curatela di un tesauro bilingue. Prima trova nel testo inglese e in "
        "quello italiano dove sono presentate le 3C; poi, per ogni C e per ogni "
        "concetto che il testo indica come sua componente, copia la frase "
        "esatta (identica alla pagina, senza omissioni) in ciascuna lingua e "
        "solo dopo compila etichette e pagina. Ordine e posizioni come nel "
        "testo. parent_label_en: etichetta inglese del concetto che contiene la "
        "voce. Solo voci scritte nel testo. Nome del framework: alla lettera "
        "come nel testo, contenuto in name_quote."
    ),
}

MAPPING_INSTRUCTIONS = {
    "a": (
        "Confronta due versioni dello stesso framework di strategie R. Per ogni "
        "strategia della versione di arrivo scrivi una riga con from_codes, "
        "le strategie di partenza da cui viene: una sola se type è same "
        "(stessa strategia, anche se il nome cambia forma, come riciclare e "
        "riciclo, o cambia posizione) o renamed (il testo le dà esplicitamente "
        "un nome nuovo); due se type è merged (il testo dice che sono state "
        "unite); nessuna se type è new (il testo la introduce). Per ogni riga "
        "copia alla lettera dalle pagine la frase che lo giustifica, senza tagli."
    ),
    "b": (
        "Allineamento fra due elenchi di strategie R. Leggi nelle pagine come "
        "gli autori descrivono le modifiche rispetto alla versione di partenza. "
        "Poi, per ogni strategia di arrivo, una riga: type=new e from_codes "
        "vuoto se il testo dice che è stata introdotta; type=merged e i due "
        "codici di partenza se il testo dice che due strategie sono state "
        "unite; type=renamed e un codice se il testo indica un nome diverso "
        "per la stessa strategia; altrimenti type=same e il codice della "
        "stessa strategia di partenza (anche se è in un'altra posizione o il "
        "nome cambia solo forma). Ogni riga porta la frase esatta della pagina "
        "che la sostiene, identica al testo."
    ),
}


def _string(**extra) -> dict:
    return {"type": "string", **extra}


def r_schema() -> dict:
    """JSON schema of an R framework draft.

    Returns:
        Schema with the framework header, members and definitions.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["framework", "members", "definitions"],
        "properties": {
            "framework": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "name_quote", "name_page", "author", "year"],
                "properties": {
                    "name": _string(),
                    "name_quote": _string(),
                    "name_page": {"type": "integer"},
                    "author": _string(),
                    "year": {"type": ["integer", "null"]},
                },
            },
            "members": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "r_code", "position_in_text", "label_it", "label_en",
                        "family", "page", "quote",
                    ],
                    "properties": {
                        "r_code": {"enum": R_POSITIONS},
                        "position_in_text": {"type": "boolean"},
                        "label_it": _string(),
                        "label_en": _string(),
                        "family": _string(),
                        "page": {"type": "integer"},
                        "quote": _string(),
                    },
                },
            },
            "definitions": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["r_code", "quoted_definition", "page"],
                    "properties": {
                        "r_code": {"enum": R_POSITIONS},
                        "quoted_definition": _string(),
                        "page": {"type": "integer"},
                    },
                },
            },
        },
    }


def c3_schema() -> dict:
    """JSON schema of the 3C draft (English and Italian side by side).

    Returns:
        Schema with the framework header, the three C's and their components.
    """
    bilingual = {
        "label_en": _string(),
        "label_it": _string(),
        "quote_en": _string(),
        "page_en": {"type": "integer"},
        "quote_it": _string(),
        "page_it": {"type": "integer"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["framework", "members", "subconcepts"],
        "properties": {
            "framework": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "name_quote", "name_page", "author", "year"],
                "properties": {
                    "name": _string(),
                    "name_quote": _string(),
                    "name_page": {"type": "integer"},
                    "author": _string(),
                    "year": {"type": ["integer", "null"]},
                },
            },
            "members": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["position", *bilingual],
                    "properties": {"position": {"enum": [1, 2, 3]}, **bilingual},
                },
            },
            "subconcepts": {
                "type": "array",
                "maxItems": 30,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["parent_label_en", *bilingual],
                    "properties": {"parent_label_en": _string(), **bilingual},
                },
            },
        },
    }


def mapping_schema(from_positions: list[str], to_positions: list[str]) -> dict:
    """JSON schema of a version-mapping draft: one row per target member.

    Args:
        from_positions: Positions of the starting version.
        to_positions: Positions of the target version.

    Returns:
        Schema of a list of mapping rows.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mappings"],
        "properties": {
            "mappings": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(to_positions),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["to_code", "type", "from_codes", "page", "quote"],
                    "properties": {
                        "to_code": {"enum": to_positions},
                        "type": {"enum": ["same", "renamed", "merged", "new"]},
                        "from_codes": {
                            "type": "array",
                            "maxItems": 2,
                            "items": {"enum": from_positions},
                        },
                        "page": {"type": "integer"},
                        "quote": _string(),
                    },
                },
            }
        },
    }


def pages_block(document: str, pages: list[int]) -> str:
    """Render source pages for the prompt, each with its number.

    Args:
        document: ``doc_id`` of the source.
        pages: Pages to include.

    Returns:
        Pages joined with explicit page markers.
    """
    return "\n\n".join(
        f"=== {document}, pagina {p} ===\n{page_text(document, p)}" for p in pages
    )


def call_model(system: str, user: str, schema: dict, max_tokens: int = 6000) -> dict:
    """Ask the generator for JSON constrained to a schema.

    Args:
        system: Instruction.
        user: Source pages and task.
        schema: JSON schema for ``structured_outputs``.
        max_tokens: Completion budget.

    Returns:
        The parsed JSON object, plus a ``_meta`` entry with timing and usage.
    """
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "structured_outputs": {"json": schema},
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read())
    choice = payload["choices"][0]
    content = choice["message"]["content"]
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        failed = DRAFTS_DIR / "failed_response.txt"
        failed.write_text(content, encoding="utf-8")
        raise RuntimeError(
            f"risposta non JSON (finish_reason={choice.get('finish_reason')}), salvata in {failed}"
        ) from None
    result["_meta"] = {
        "seconds": round(time.time() - started, 1),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
    }
    return result


def draft_r(variant: str, framework_id: str) -> dict:
    """Draft one R framework.

    Args:
        variant: Instruction variant, ``"a"`` or ``"b"``.
        framework_id: Key of :data:`R_TASKS`.

    Returns:
        The model's draft with the task attached.
    """
    task = R_TASKS[framework_id]
    user = (
        f"{pages_block(task['document'], task['pages'])}\n\n"
        "Estrai il framework di strategie R descritto in queste pagine."
    )
    draft = call_model(INSTRUCTIONS[variant], user, r_schema())
    for item in draft["members"] + draft["definitions"]:
        item["position"] = item.pop("r_code")
    draft["task"] = {"id": framework_id, **task}
    return draft


def draft_3c(variant: str) -> dict:
    """Draft the 3C framework from the English and Italian sources.

    Args:
        variant: Instruction variant, ``"a"`` or ``"b"``.

    Returns:
        The model's draft with the task attached.
    """
    task = C3_TASK
    user = (
        "TESTO INGLESE\n"
        f"{pages_block(task['document_en'], task['pages_en'])}\n\n"
        "TESTO ITALIANO\n"
        f"{pages_block(task['document_it'], task['pages_it'])}\n\n"
        "Estrai le 3C della Circular Economy for Food e le loro componenti."
    )
    draft = call_model(C3_INSTRUCTIONS[variant], user, c3_schema())
    draft["task"] = task
    return draft


def draft_mapping(variant: str, task: dict, drafts: dict) -> dict:
    """Draft the correspondences between two R versions.

    Args:
        variant: Instruction variant, ``"a"`` or ``"b"``.
        task: Entry of :data:`MAPPING_TASKS`.
        drafts: R drafts of the same variant, keyed by framework id.

    Returns:
        The model's draft with the task attached.
    """

    def listing(framework_id: str) -> str:
        return "\n".join(
            f"{m['position']} {m['label_it']}" for m in drafts[framework_id]["members"]
        )

    from_positions = [m["position"] for m in drafts[task["from"]]["members"]]
    to_positions = [m["position"] for m in drafts[task["to"]]["members"]]
    user = (
        f"{pages_block(task['document'], task['pages'])}\n\n"
        f"VERSIONE DI PARTENZA ({task['from']}):\n{listing(task['from'])}\n\n"
        f"VERSIONE DI ARRIVO ({task['to']}):\n{listing(task['to'])}\n\n"
        "Allinea le strategie della versione di arrivo a quelle di partenza."
    )
    draft = call_model(
        MAPPING_INSTRUCTIONS[variant], user, mapping_schema(from_positions, to_positions)
    )
    draft["rows"] = mapping_rows(draft["mappings"])
    draft["task"] = task
    return draft


def mapping_rows(drafted: list[dict]) -> list[dict]:
    """Expand one drafted row per target into one row per correspondence.

    A row whose number of origins does not fit its type is kept with
    ``"inconsistent": True``, so acceptance can hold it back with the reason.

    Args:
        drafted: Rows as returned by the model.

    Returns:
        Rows with ``from_position``, ``to_position``, ``type``, ``page``, ``quote``.
    """
    expected = {"new": 0, "same": 1, "renamed": 1, "merged": 2}
    rows = []
    for item in drafted:
        origins = list(dict.fromkeys(item["from_codes"]))
        inconsistent = len(origins) != expected[item["type"]]
        for origin in origins or [""]:
            rows.append(
                {
                    "from_position": origin,
                    "to_position": item["to_code"],
                    "type": item["type"],
                    "page": item["page"],
                    "quote": item["quote"],
                    "inconsistent": inconsistent,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default="a,b")
    args = parser.parse_args()
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    for variant in args.variants.split(","):
        r_drafts = {fid: draft_r(variant, fid) for fid in R_TASKS}
        run = {
            "variant": variant,
            "instructions": {
                "r": INSTRUCTIONS[variant],
                "3c": C3_INSTRUCTIONS[variant],
                "mappings": MAPPING_INSTRUCTIONS[variant],
            },
            "r": r_drafts,
            "3c": draft_3c(variant),
            "mappings": [draft_mapping(variant, t, r_drafts) for t in MAPPING_TASKS],
        }
        out = DRAFTS_DIR / f"run_{variant}.json"
        out.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
