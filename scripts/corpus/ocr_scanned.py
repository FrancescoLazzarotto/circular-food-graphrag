#!/usr/bin/env python3
"""Make a searchable copy of every registry file marked ``ocr``.

A scanned PDF has pages that are only images, and stage 0 reads nothing from
them; some PDFs have a text layer made of control characters instead, which
reads as nothing just the same. This writes, for each such file, a copy whose
pages without readable text carry the text Tesseract recognises, laid over the
page where it was read; pages that already have readable text are copied
unchanged. The copy is named after the content hash of the
original (``<impronta>.pdf``), which is where stage 0 looks for it, and the
original in the corpus folder is never touched.

    python scripts/corpus/ocr_scanned.py --corpus-dir "<corpus>" \
        --registry product/corpus_registry.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
# The kg_pipeline package lives at the repo root and is not pip-installed.
sys.path.insert(0, str(ROOT))

from kg_pipeline.utils import corpus_registry  # noqa: E402

# Same rule as the registry's page count: less text than one line means the
# page has no text layer.
_MIN_PAGE_CHARS = 80
# A text layer can be there and still say nothing: a font without a Unicode map
# yields control characters instead of letters, which stage 0 reads as U+FFFD.
# On readable pages such characters are a fraction of a percent; on these pages
# they are most of the text, so the threshold sits well inside the gap.
_MAX_UNREADABLE_SHARE = 0.2


def _has_text(page_text: str) -> bool:
    """Whether a page's text layer carries text a reader could read."""
    text = page_text.strip()
    if len(text) < _MIN_PAGE_CHARS:
        return False
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return False
    unreadable = sum(
        c == "�" or unicodedata.category(c) in ("Cc", "Co", "Cn") for c in visible
    )
    return unreadable <= _MAX_UNREADABLE_SHARE * len(visible)


def _ocr_page(image: Path, out_base: Path, languages: str) -> Path:
    """Run Tesseract on one page image and return the one-page PDF it writes."""
    # One thread per Tesseract process: the pages already run in parallel.
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    subprocess.run(
        ["tesseract", str(image), str(out_base), "-l", languages, "pdf"],
        check=True,
        capture_output=True,
        env=env,
    )
    return out_base.with_suffix(".pdf")


def ocr_document(source: Path, target: Path, languages: str, dpi: int, jobs: int) -> tuple[int, int]:
    """Write the searchable copy of one PDF.

    Args:
        source: Scanned PDF.
        target: Copy to write; replaced atomically.
        languages: Tesseract languages, e.g. ``ita+eng``.
        dpi: Rendering resolution of the image pages.
        jobs: Pages recognised in parallel.

    Returns:
        ``(pages recognised, characters of text gained)``.
    """
    with tempfile.TemporaryDirectory(prefix="ocr_") as tmp_name:
        tmp = Path(tmp_name)
        with fitz.open(source) as doc:
            todo: list[tuple[int, Path]] = []
            for number, page in enumerate(doc):
                if _has_text(page.get_text()):
                    continue
                pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                # Tesseract sizes its PDF page from the image resolution, so
                # the recognised page keeps the original page size.
                pixmap.set_dpi(dpi, dpi)
                # Tesseract embeds the image it was given; JPEG keeps a scanned
                # page to a few hundred kilobytes instead of several megabytes.
                image = tmp / f"page_{number:05d}.jpg"
                pixmap.save(image, jpg_quality=75)
                todo.append((number, image))

            with ThreadPoolExecutor(max_workers=jobs) as pool:
                pdfs = dict(
                    zip(
                        (number for number, _ in todo),
                        pool.map(
                            lambda item: _ocr_page(item[1], item[1].with_suffix(""), languages),
                            todo,
                        ),
                    )
                )

            gained = 0
            with fitz.open() as out:
                for number in range(doc.page_count):
                    if number in pdfs:
                        with fitz.open(pdfs[number]) as page_pdf:
                            gained += len(page_pdf[0].get_text().strip())
                            out.insert_pdf(page_pdf)
                    else:
                        out.insert_pdf(doc, from_page=number, to_page=number)
                partial = tmp / "out.pdf"
                out.save(partial, garbage=3, deflate=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + ".tmp")
        shutil.copyfile(partial, staged)
        staged.replace(target)
    return len(todo), gained


def main() -> None:
    """Parse the command line and make the missing OCR copies."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--registry", default=ROOT / "product" / "corpus_registry.csv", type=Path)
    parser.add_argument("--ocr-dir", default=ROOT / "kg_pipeline" / "artifacts" / "corpus_ocr", type=Path)
    parser.add_argument("--languages", default="ita+eng")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="redo copies that already exist")
    args = parser.parse_args()

    if shutil.which("tesseract") is None:
        raise SystemExit("tesseract is not installed")
    rows = corpus_registry.load_registry(args.registry)
    for row in rows:
        if not row.ocr or row.escluso:
            continue
        target = corpus_registry.ocr_copy(args.ocr_dir, row)
        if target.is_file() and not args.force:
            print(f"già fatto: {row.percorso}")
            continue
        source = corpus_registry.resolve_path(args.corpus_dir, row.percorso)
        if source is None:
            raise SystemExit(f"file non trovato: {row.percorso}")
        pages, gained = ocr_document(source, target, args.languages, args.dpi, args.jobs)
        print(f"{row.id_documento}: {pages} pagine riconosciute, {gained} caratteri -> {target}")


if __name__ == "__main__":
    main()
