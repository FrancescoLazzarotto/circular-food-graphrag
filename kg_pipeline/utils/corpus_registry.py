"""The corpus registry: one row per file of the corpus, edited by hand.

The registry is the list of documents stage 0 reads and the place where the
people who curate the corpus record what each file is: its theme, whether it
duplicates another file, whether it is excluded, whether its text comes from
OCR. It is a semicolon-separated CSV with a byte-order mark, the form a
spreadsheet opens correctly with a double click in an Italian locale. Column
names and yes/no values are in Italian because the curators are.

The registry, not the folder, decides which files are documents. The corpus
folder nests documents in theme folders, and a recursive scan alone would also
take duplicates and drafts; an explicit list with a reason per excluded file
is what lets curators keep those files where they are.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("kg_pipeline")

COLUMNS = (
    "id_documento",
    "percorso",
    "impronta",
    "tema",
    "sottotema",
    "lingua",
    "pagine",
    "pagine_senza_testo",
    "doppione_di",
    "genere",
    "livello",
    "priorita",
    "escluso",
    "ocr",
    "note",
)
LEVELS = (1, 2)

# What people type in a spreadsheet for yes and no, including the unaccented
# "si" and the "x" of a ticked column.
_YES = frozenset({"sì", "si", "s", "x", "yes", "y", "true", "vero", "1"})
_NO = frozenset({"", "no", "n", "false", "falso", "0"})
# The id becomes part of every chunk id and graph provenance record, so it is
# kept to characters that survive file names, Cypher and URLs unchanged.
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_HASH_BLOCK = 1 << 20


@dataclass
class RegistryRow:
    """One file of the corpus.

    Attributes:
        id_documento: Stable document identifier; never changes once assigned.
        percorso: Path relative to the corpus folder, as written on disk.
        impronta: SHA-256 of the file content.
        tema: First folder level of ``percorso``.
        sottotema: Remaining folder levels, joined with " / ".
        lingua: Main language, as proposed and then corrected by hand.
        pagine: Page count, for PDFs.
        pagine_senza_testo: Pages without a text layer, for PDFs.
        doppione_di: ``id_documento`` of the copy that is kept.
        genere: Kind of document, filled in by the curators.
        livello: 1 = searchable text only, 2 = also in the graph.
        priorita: Priority, filled in by the curators.
        escluso: Whether stage 0 skips the file.
        ocr: Whether stage 0 reads the OCR copy instead of the file.
        note: Free text.
    """

    id_documento: str
    percorso: str
    impronta: str = ""
    tema: str = ""
    sottotema: str = ""
    lingua: str = ""
    pagine: int | None = None
    pagine_senza_testo: int | None = None
    doppione_di: str = ""
    genere: str = ""
    livello: int = 1
    priorita: str = ""
    escluso: bool = False
    ocr: bool = False
    note: str = ""

    @property
    def filename(self) -> str:
        """File name without the folders."""
        return Path(self.percorso).name

    @property
    def is_pdf(self) -> bool:
        """Whether the file is a PDF, whatever the case of the suffix."""
        return Path(self.percorso).suffix.lower() == ".pdf"


def _parse_flag(value: str, column: str, line: int, errors: list[str]) -> bool:
    """Read a yes/no cell; an unreadable value is recorded in ``errors``."""
    key = unicodedata.normalize("NFC", value).strip().lower()
    if key in _YES:
        return True
    if key in _NO:
        return False
    errors.append(f"riga {line}, colonna {column}: {value!r} non è né sì né no")
    return False


def _parse_int(value: str, column: str, line: int, errors: list[str]) -> int | None:
    """Read an optional integer cell; an unreadable value is recorded in ``errors``."""
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        errors.append(f"riga {line}, colonna {column}: {value!r} non è un numero intero")
        return None


def _delimiter(header_line: str) -> str:
    """The delimiter of a registry file: a spreadsheet saving it may switch to commas."""
    return ";" if header_line.count(";") >= header_line.count(",") else ","


def load_registry(path: Path) -> list[RegistryRow]:
    """Read and validate a registry file.

    Args:
        path: The CSV file.

    Returns:
        The rows, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a column is missing or unknown, a cell cannot be read,
            or the rows are inconsistent (see :func:`validate`). The message
            lists every problem with its line number.
    """
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{path}: il registro è vuoto")
    reader = csv.DictReader(lines, delimiter=_delimiter(lines[0]))
    header = [name.strip() for name in (reader.fieldnames or [])]
    reader.fieldnames = header
    missing = [c for c in COLUMNS if c not in header]
    # An unknown column would be dropped the next time the file is written,
    # taking whatever a curator typed in it.
    unknown = [c for c in header if c and c not in COLUMNS]
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"colonne mancanti: {', '.join(missing)}")
        if unknown:
            problems.append(
                f"colonne non previste: {', '.join(unknown)} (il contenuto va spostato in 'note')"
            )
        raise ValueError(f"{path}: " + "; ".join(problems))

    rows: list[RegistryRow] = []
    errors: list[str] = []
    for line, record in enumerate(reader, start=2):
        cells = {k: (v or "").strip() for k, v in record.items() if k}
        if not any(cells.values()):
            continue
        livello = _parse_int(cells["livello"], "livello", line, errors)
        if livello is None:
            livello = 1
        elif livello not in LEVELS:
            errors.append(f"riga {line}, colonna livello: {livello} non è 1 né 2")
        rows.append(
            RegistryRow(
                id_documento=cells["id_documento"],
                percorso=cells["percorso"],
                impronta=cells["impronta"].lower(),
                tema=cells["tema"],
                sottotema=cells["sottotema"],
                lingua=cells["lingua"],
                pagine=_parse_int(cells["pagine"], "pagine", line, errors),
                pagine_senza_testo=_parse_int(
                    cells["pagine_senza_testo"], "pagine_senza_testo", line, errors
                ),
                doppione_di=cells["doppione_di"],
                genere=cells["genere"],
                livello=livello,
                priorita=cells["priorita"],
                escluso=_parse_flag(cells["escluso"], "escluso", line, errors),
                ocr=_parse_flag(cells["ocr"], "ocr", line, errors),
                note=cells["note"],
            )
        )
    errors.extend(validate(rows))
    if errors:
        raise ValueError(f"{path}: {len(errors)} problemi\n- " + "\n- ".join(errors))
    return rows


def validate(rows: list[RegistryRow]) -> list[str]:
    """Check the rows against each other.

    Args:
        rows: Registry rows.

    Returns:
        One message per problem; empty when the registry is consistent.
    """
    errors: list[str] = []
    ids: dict[str, str] = {}
    paths: dict[str, str] = {}
    included_names: dict[str, str] = {}
    for row in rows:
        if not _ID_RE.match(row.id_documento):
            errors.append(
                f"{row.percorso}: id_documento {row.id_documento!r} deve contenere solo "
                "lettere minuscole, cifre e '_'"
            )
        if row.id_documento in ids:
            errors.append(
                f"id_documento {row.id_documento!r} ripetuto: {ids[row.id_documento]} e {row.percorso}"
            )
        ids.setdefault(row.id_documento, row.percorso)
        if not row.percorso:
            errors.append(f"{row.id_documento}: percorso vuoto")
        key = unicodedata.normalize("NFC", row.percorso)
        if key in paths:
            errors.append(f"percorso ripetuto: {row.percorso}")
        paths.setdefault(key, row.id_documento)
        if row.escluso:
            continue
        if not row.is_pdf:
            errors.append(f"{row.percorso}: solo i PDF si leggono; convertirlo o segnarlo escluso")
        # Stage 0 keeps the bare file name, and later stages key documents on
        # it, so two included files may not share one.
        name = unicodedata.normalize("NFC", row.filename)
        if name in included_names:
            errors.append(
                f"stesso nome di file in due documenti inclusi: {included_names[name]} e {row.percorso}"
            )
        included_names.setdefault(name, row.percorso)
        if row.ocr and not row.impronta:
            errors.append(f"{row.percorso}: ocr = sì senza impronta, la copia OCR non si trova")
    for row in rows:
        if not row.doppione_di:
            continue
        if row.doppione_di == row.id_documento:
            errors.append(f"{row.percorso}: doppione di se stesso")
        elif row.doppione_di not in ids:
            errors.append(f"{row.percorso}: doppione_di {row.doppione_di!r} non è un id del registro")
    return errors


def _format(value: object) -> str:
    """Render a cell the way a curator expects to read and type it."""
    if isinstance(value, bool):
        return "sì" if value else "no"
    if value is None:
        return ""
    return str(value)


def save_registry(path: Path, rows: list[RegistryRow]) -> None:
    """Write the registry, sorted by path, replacing the file atomically.

    Args:
        path: The CSV file; parent directories are created.
        rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: unicodedata.normalize("NFC", r.percorso).casefold())
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in ordered:
            writer.writerow([_format(getattr(row, column)) for column in COLUMNS])
    tmp.replace(path)


def file_digest(path: Path) -> str:
    """SHA-256 of a file's content, as lowercase hex.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(corpus_dir: Path, relative: str) -> Path | None:
    """Find a registry path on disk, whichever Unicode form the name is in.

    macOS writes accented names decomposed and most other systems composed; a
    registry saved by a spreadsheet may hold the other form than the disk.

    Args:
        corpus_dir: Corpus folder.
        relative: ``percorso`` of a row.

    Returns:
        The existing path, or ``None``.
    """
    for form in (relative, unicodedata.normalize("NFD", relative), unicodedata.normalize("NFC", relative)):
        candidate = corpus_dir / form
        if candidate.is_file():
            return candidate
    return None


def ocr_copy(ocr_dir: Path, row: RegistryRow) -> Path:
    """Where the OCR copy of a row's file is kept.

    Keyed by the content of the original, so a renamed or moved file keeps its
    copy and a changed file does not reuse a stale one.
    """
    return ocr_dir / f"{row.impronta}.pdf"


def documents_to_ingest(
    rows: list[RegistryRow],
    corpus_dir: Path,
    ocr_dir: Path | None,
    single_doc: str | None = None,
) -> list[tuple[Path, RegistryRow]]:
    """The files stage 0 reads, one per included row, checked against the disk.

    Args:
        rows: Registry rows.
        corpus_dir: Corpus folder the paths are relative to.
        ocr_dir: Folder of the OCR copies, required when a row has ``ocr``.
        single_doc: Restrict to the row whose file name, path or id is this.

    Returns:
        ``(file to read, row)`` pairs, in registry order.

    Raises:
        FileNotFoundError: If ``single_doc`` matches no included row, or a file
            or an OCR copy is missing.
        ValueError: If a file's content no longer matches its ``impronta``.
    """
    included = [row for row in rows if not row.escluso]
    if single_doc:
        included = [
            row
            for row in included
            if single_doc in {row.filename, row.percorso, row.id_documento}
        ]
        if not included:
            raise FileNotFoundError(
                f"--single-doc {single_doc!r} is not an included document of the registry"
            )

    missing: list[str] = []
    changed: list[str] = []
    selected: list[tuple[Path, RegistryRow]] = []
    for row in included:
        original = resolve_path(corpus_dir, row.percorso)
        if original is None:
            missing.append(row.percorso)
            continue
        if row.impronta and file_digest(original) != row.impronta:
            changed.append(row.percorso)
            continue
        source = original
        if row.ocr:
            if ocr_dir is None:
                missing.append(f"{row.percorso} (ocr = sì, but no paths.ocr_dir is configured)")
                continue
            source = ocr_copy(ocr_dir, row)
            if not source.is_file():
                missing.append(f"{row.percorso} (OCR copy {source.name})")
                continue
        selected.append((source, row))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} registry file(s) not found under {corpus_dir}: "
            + "; ".join(missing)
            + ". Refresh the registry (scripts/corpus/build_registry.py) or run "
            "scripts/corpus/ocr_scanned.py for the OCR copies."
        )
    if changed:
        raise ValueError(
            f"{len(changed)} file(s) changed since the registry was written: "
            + "; ".join(changed)
            + ". Refresh it with scripts/corpus/build_registry.py."
        )
    return selected


def unregistered_files(rows: list[RegistryRow], corpus_dir: Path) -> list[str]:
    """Corpus files the registry does not list, as paths relative to the folder.

    Hidden files and office lock files are not documents and are left out.

    Args:
        rows: Registry rows.
        corpus_dir: Corpus folder.

    Returns:
        The relative paths, sorted.
    """
    listed = {unicodedata.normalize("NFC", row.percorso) for row in rows}
    found = []
    for path in corpus_dir.rglob("*"):
        relative = path.relative_to(corpus_dir)
        if not path.is_file() or is_ignored(relative):
            continue
        if unicodedata.normalize("NFC", relative.as_posix()) not in listed:
            found.append(relative.as_posix())
    return sorted(found)


def is_ignored(relative: Path) -> bool:
    """Whether a file is system clutter rather than a document.

    Hidden files and folders (``.DS_Store``) and the ``~$`` lock files office
    suites leave next to an open document.

    Args:
        relative: Path relative to the corpus folder, so that a dot in the
            folders above the corpus does not hide every file.
    """
    return any(part.startswith(".") for part in relative.parts) or relative.name.startswith("~$")
