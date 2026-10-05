#!/usr/bin/env python3
"""Create or refresh the corpus registry from the corpus folder.

Scans the folder recursively, hashes every file, reads every PDF once and
writes one registry row per file (see ``kg_pipeline/utils/corpus_registry.py``).

On an existing registry, the columns curators edit are left as they are:
``lingua``, ``doppione_di``, ``genere``, ``livello``, ``priorita``,
``escluso``, ``ocr``, ``note`` and the id. The columns that describe the file
(path, hash, theme folders, page counts) are recomputed. A file that moved is
recognised by its hash; a file that disappeared keeps its row, marked
excluded, so a curator's notes are not lost. Only new files get proposals:
an id, a language, duplicates, OCR, level.

Duplicates are found three ways: identical bytes, identical text, and
near-identical pages, which is what catches a print and an online edition or a
draft and its final version. The copy kept is the one already in the graph,
then the longest.

    python scripts/corpus/build_registry.py --corpus-dir "<corpus>" \
        --registry product/corpus_registry.csv --in-graph-dir "documents/test 1"
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
# The kg_pipeline package lives at the repo root and is not pip-installed.
sys.path.insert(0, str(ROOT))

from kg_pipeline.stages.ingestion import _doc_id_from_filename  # noqa: E402
from kg_pipeline.utils import corpus_registry  # noqa: E402
from kg_pipeline.utils.corpus_registry import RegistryRow  # noqa: E402

LOGGER = logging.getLogger("kg_pipeline")

# A page with less text than one line (a page number, a running header) is a
# page without a text layer.
_MIN_PAGE_CHARS = 80
# Pages with fewer distinct words than this carry too little to compare.
_MIN_PAGE_WORDS = 30
# Two pages are the same page when this share of their distinct words is
# common to both; reflowed text and a changed header stay above it.
_SAME_PAGE = 0.85
# A document is a copy of another when this share of its pages is found there.
_COPY_SHARE = 0.8
# Cheap first cut before comparing pages: the vocabulary of the copy must be
# almost entirely inside the other document's.
_VOCAB_SHARE = 0.8

_WORD_RE = re.compile(r"[^\W\d_]{3,}")
# Function words are short, so the language check counts words of any length.
_LANG_WORD_RE = re.compile(r"[^\W\d_]+")
_IT_WORDS = frozenset(
    "il lo la gli le di che è per non una un del della dei delle nel nella con sono "
    "come più anche questo questa alla al ai".split()
)
_EN_WORDS = frozenset(
    "the of and to in is that for are with as on by this be from or an it which at have not".split()
)


@dataclass
class Scan:
    """What a scan of one file found.

    Attributes:
        relative: Path relative to the corpus folder.
        digest: SHA-256 of the content.
        is_pdf: Whether the file is a PDF.
        pages: Page count (PDFs).
        empty_pages: Pages without a text layer (PDFs).
        chars: Characters of text (PDFs).
        language: Proposed language (PDFs).
        text_key: Hash of the normalised text, empty when there is none.
        page_words: Distinct words per page with enough words to compare.
        vocabulary: Distinct words of the whole document.
    """

    relative: str
    digest: str
    is_pdf: bool
    pages: int | None = None
    empty_pages: int | None = None
    chars: int = 0
    language: str = ""
    text_key: str = ""
    page_words: list[frozenset[str]] = field(default_factory=list)
    vocabulary: frozenset[str] = frozenset()


def guess_language(text: str) -> str:
    """Propose the main language of a text, for a curator to confirm.

    Args:
        text: Document text.

    Returns:
        ``"it"`` or ``"en"`` when one clearly dominates the common function
        words, ``"misto"`` when neither does, ``"nessuna"`` without text.
    """
    words = _LANG_WORD_RE.findall(text[:400_000].lower())
    if not words:
        return "nessuna"
    italian = sum(w in _IT_WORDS for w in words)
    english = sum(w in _EN_WORDS for w in words)
    if italian > 1.3 * english:
        return "it"
    if english > 1.3 * italian:
        return "en"
    return "misto"


def scan_file(corpus_dir: Path, path: Path) -> Scan:
    """Hash a file and, for a PDF, read what the registry and the duplicate check need."""
    relative = path.relative_to(corpus_dir).as_posix()
    scan = Scan(
        relative=relative,
        digest=corpus_registry.file_digest(path),
        is_pdf=path.suffix.lower() == ".pdf",
    )
    if not scan.is_pdf:
        return scan
    texts: list[str] = []
    with fitz.open(path) as doc:
        scan.pages = doc.page_count
        for page in doc:
            texts.append(page.get_text())
    scan.empty_pages = sum(len(t.strip()) < _MIN_PAGE_CHARS for t in texts)
    full = "\n".join(texts)
    scan.chars = len(full)
    scan.language = guess_language(full)
    words = [w.lower() for w in _WORD_RE.findall(full)]
    scan.text_key = hashlib.sha256(" ".join(words).encode()).hexdigest() if words else ""
    for text in texts:
        page = frozenset(w.lower() for w in _WORD_RE.findall(text))
        if len(page) >= _MIN_PAGE_WORDS:
            scan.page_words.append(page)
    scan.vocabulary = frozenset(words)
    return scan


def _share_found(pages: list[frozenset[str]], other: list[frozenset[str]]) -> float:
    """Share of ``pages`` that have a near-identical page in ``other``."""
    if not pages:
        return 0.0
    found = sum(
        any(len(p & q) / len(p | q) >= _SAME_PAGE for q in other) for p in pages
    )
    return found / len(pages)


def find_copies(scans: list[Scan], rank: dict[str, tuple]) -> dict[str, tuple[str, str]]:
    """Map each duplicate file to the file kept and the reason.

    Args:
        scans: Scans of the PDFs.
        rank: Sort key per relative path; the smallest key of a group is kept.

    Returns:
        ``{relative path of the copy: (relative path kept, reason)}``.
    """
    copy_of: dict[str, tuple[str, str]] = {}

    def record(copy: Scan, kept: Scan, reason: str) -> None:
        if copy.relative not in copy_of:
            copy_of[copy.relative] = (kept.relative, reason)

    for a, b in itertools.combinations(scans, 2):
        kept, copy = (a, b) if rank[a.relative] <= rank[b.relative] else (b, a)
        if a.digest == b.digest:
            record(copy, kept, "copia identica")
            continue
        if a.text_key and a.text_key == b.text_key:
            record(copy, kept, "stesso testo")
            continue
        if not copy.vocabulary or not kept.vocabulary:
            continue
        # Only the worse-ranked file can be marked a copy. When the better-ranked
        # one is the file contained (a paper already in the graph inside a
        # proceedings volume), both are real documents and neither is dropped.
        if len(copy.vocabulary & kept.vocabulary) / len(copy.vocabulary) < _VOCAB_SHARE:
            continue
        share = _share_found(copy.page_words, kept.page_words)
        if share >= _COPY_SHARE:
            record(copy, kept, f"{share:.0%} delle pagine si ritrovano")

    # A copy of a copy points at the file finally kept.
    for relative in list(copy_of):
        target, reason = copy_of[relative]
        seen = {relative}
        while target in copy_of and target not in seen:
            seen.add(target)
            target = copy_of[target][0]
        copy_of[relative] = (target, reason)
    return copy_of


def propose_id(filename: str, digest: str, taken: set[str]) -> str:
    """A new id: the file-name slug, with a hash suffix when the slug is taken."""
    base = _doc_id_from_filename(filename)
    candidate = base if base not in taken else f"{base}_{digest[:8]}"
    taken.add(candidate)
    return candidate


def readable(name: str) -> str:
    """A folder or file name as its author typed it.

    Zip archives made on macOS store UTF-8 names that some extractors decode
    as the old DOS Latin code page, which turns "è" into "e╠Ç" on disk. Re-encoding
    undoes it; a name not damaged this way does not survive the round trip and
    is returned unchanged.
    """
    try:
        repaired = name.encode("cp850").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name
    return unicodedata.normalize("NFC", repaired)


def themes(relative: str) -> tuple[str, str]:
    """``(tema, sottotema)`` from the folders of a relative path, in readable form."""
    folders = [readable(part) for part in Path(relative).parent.parts]
    if not folders:
        return "", ""
    return folders[0], " / ".join(folders[1:])


def build(
    corpus_dir: Path,
    existing: list[RegistryRow],
    in_graph: set[str],
    ocr_dir: Path | None,
) -> tuple[list[RegistryRow], list[str]]:
    """Merge a fresh scan of the corpus into the existing rows.

    Args:
        corpus_dir: Corpus folder.
        existing: Rows of the current registry (empty for a new one).
        in_graph: Hashes of the files the graph was built from.
        ocr_dir: Folder of the OCR copies; the language of a scanned file is
            read from its copy when there is one.

    Returns:
        ``(rows, report lines)``.
    """
    files = sorted(
        (p for p in corpus_dir.rglob("*") if p.is_file() and not corpus_registry.is_ignored(p.relative_to(corpus_dir))),
        key=lambda p: p.relative_to(corpus_dir).as_posix().casefold(),
    )
    scans = [scan_file(corpus_dir, p) for p in files]
    report = [f"file trovati: {len(scans)} ({sum(s.is_pdf for s in scans)} PDF)"]

    by_digest = {row.impronta: row for row in existing if row.impronta}
    by_path = {unicodedata.normalize("NFC", row.percorso): row for row in existing}
    taken = {row.id_documento for row in existing}
    matched: set[int] = set()
    rows: list[RegistryRow] = []
    new_scans: list[tuple[Scan, RegistryRow]] = []

    for scan in scans:
        key = unicodedata.normalize("NFC", scan.relative)
        row = by_path.get(key)
        if row is not None and id(row) in matched:
            row = None
        if row is None:
            candidate = by_digest.get(scan.digest)
            if candidate is not None and id(candidate) not in matched:
                row = candidate
        tema, sottotema = themes(scan.relative)
        if row is not None:
            matched.add(id(row))
            if row.percorso != scan.relative:
                report.append(f"spostato: {row.percorso} -> {scan.relative}")
            if row.impronta and row.impronta != scan.digest:
                report.append(f"contenuto cambiato: {scan.relative} (id {row.id_documento} invariato)")
            row.percorso, row.impronta = scan.relative, scan.digest
            row.tema, row.sottotema = tema, sottotema
            row.pagine, row.pagine_senza_testo = scan.pages, scan.empty_pages
            rows.append(row)
            continue
        row = RegistryRow(
            id_documento=propose_id(Path(scan.relative).name, scan.digest, taken),
            percorso=scan.relative,
            impronta=scan.digest,
            tema=tema,
            sottotema=sottotema,
            lingua=scan.language,
            pagine=scan.pages,
            pagine_senza_testo=scan.empty_pages,
            livello=2 if scan.digest in in_graph else 1,
        )
        if not scan.is_pdf:
            row.escluso = True
            row.note = f"formato {Path(scan.relative).suffix.lower()} non letto dalla pipeline: convertirlo in PDF"
        elif scan.pages and scan.empty_pages * 2 > scan.pages:
            row.ocr = True
            row.note = f"{scan.empty_pages} pagine su {scan.pages} senza testo: si legge la copia OCR"
            if ocr_dir is not None:
                copy = corpus_registry.ocr_copy(ocr_dir, row)
                if copy.is_file():
                    with fitz.open(copy) as doc:
                        row.lingua = guess_language("\n".join(page.get_text() for page in doc))
        rows.append(row)
        new_scans.append((scan, row))

    for row in existing:
        if id(row) not in matched:
            if not row.escluso:
                report.append(f"file non trovato, riga segnata esclusa: {row.percorso}")
            row.escluso = True
            if "file non trovato" not in row.note:
                row.note = (row.note + "; " if row.note else "") + "file non trovato nella cartella"
            rows.append(row)

    pdf_scans = [s for s in scans if s.is_pdf]
    row_of = {row.percorso: row for row in rows}
    rank = {
        s.relative: (
            -row_of[s.relative].livello,
            row_of[s.relative].escluso,
            -(s.pages or 0),
            -s.chars,
            s.relative.casefold(),
        )
        for s in pdf_scans
    }
    copies = find_copies(pdf_scans, rank)
    new_paths = {scan.relative for scan, _ in new_scans}
    for copy, (kept, reason) in sorted(copies.items()):
        row, kept_row = row_of[copy], row_of[kept]
        if copy in new_paths:
            row.doppione_di = kept_row.id_documento
            row.escluso = True
            note = f"doppione proposto di {readable(kept_row.percorso)} ({reason})"
            row.note = (row.note + "; " if row.note else "") + note
            report.append(f"doppione: {copy} -> {kept} ({reason})")
        elif row.doppione_di != kept_row.id_documento:
            report.append(f"possibile doppione non applicato (riga già esistente): {copy} -> {kept} ({reason})")

    problems = corpus_registry.validate(rows)
    if problems:
        raise ValueError("registro incoerente:\n- " + "\n- ".join(problems))
    return rows, report


def main() -> None:
    """Parse the command line, build the registry and print what changed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--registry", default=ROOT / "product" / "corpus_registry.csv", type=Path)
    parser.add_argument(
        "--in-graph-dir",
        type=Path,
        default=None,
        help="folder of the PDFs the graph was built from: their copies get livello 2",
    )
    parser.add_argument("--ocr-dir", type=Path, default=ROOT / "kg_pipeline" / "artifacts" / "corpus_ocr")
    parser.add_argument("--dry-run", action="store_true", help="print the report, do not write")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    existing = corpus_registry.load_registry(args.registry) if args.registry.exists() else []
    in_graph: set[str] = set()
    if args.in_graph_dir is not None:
        in_graph = {
            corpus_registry.file_digest(p)
            for p in args.in_graph_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        }
    rows, report = build(args.corpus_dir, existing, in_graph, args.ocr_dir)

    included = [r for r in rows if not r.escluso]
    report += [
        f"righe: {len(rows)}; incluse: {len(included)}; escluse: {len(rows) - len(included)}",
        f"doppioni: {sum(bool(r.doppione_di) for r in rows)}; da OCR: {sum(r.ocr for r in included)}; "
        f"livello 2: {sum(r.livello == 2 for r in rows)}",
        f"id unici: {len({r.id_documento for r in rows}) == len(rows)}",
    ]
    print("\n".join(report))
    if not args.dry_run:
        corpus_registry.save_registry(args.registry, rows)
        print(f"scritto {args.registry}")


if __name__ == "__main__":
    main()
