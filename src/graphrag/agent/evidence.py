"""Numbered evidence index, cited-context rendering and citation verification.

Three responsibilities, kept in one module because they share the
reference-id vocabulary:

1. :func:`build_evidence_index` turns retrieved text chunks and KG triples into
   a list of :class:`EvidenceItem` with stable ids (``S1``, ``S2``, ``T1``...).
2. :func:`render_cited_context` renders those items as the model-facing
   context, each one carrying its source document and page range, so the
   model has something to cite.
3. :func:`verify_citations` parses the reference tags the model produced and
   checks them against the index. A model can invent prose; it cannot invent a
   reference id it was never given, so unsupported tags are caught
   deterministically at zero cost.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence

logger = logging.getLogger("graphrag")

# Matches a citation tag, including grouped forms like "[S1, T2]". Lowercase is
# accepted because models emit it occasionally; ids are normalised on parse.
_CITATION_RE = re.compile(r"\[((?:[STst]\s?\d{1,3})(?:\s*[,;]\s*[STst]\s?\d{1,3})*)\]")
_REF_RE = re.compile(r"([STst])\s?(\d{1,3})")
# "<path>#page=N#chunk=M" — the tag StandardTextRAGPipeline attaches to chunks.
_CHUNK_SOURCE_RE = re.compile(r"^(?P<path>.*?)(?:#page=(?P<page>[^#]*))?(?:#chunk=(?P<chunk>.*))?$")
# Definitions are quoted between guillemets. Only guillemets are checked:
# straight and curly quotes carry emphasis, titles and scare quotes, and
# stripping those would rewrite answers that claim nothing about a source.
_QUOTE_RE = re.compile(r"«\s*([^»]{1,600}?)\s*»")

UNVERIFIED_MARK_IT = "[riferimento non verificato]"
UNVERIFIED_MARK_EN = "[unverified reference]"


@dataclass(slots=True)
class EvidenceItem:
    """One citable unit of retrieved evidence.

    Attributes:
        ref_id: Reference id, ``S<n>`` for passages and ``T<n>`` for triples.
        kind: ``"text"`` or ``"triple"``.
        text: Passage text, or the triple as ``(subject, predicate, object)``.
        source_doc: Basename of the source document.
        pages: Page label, e.g. ``p. 129``.
        chunk_id: Chunk id of a passage.
        metadata: ``key=value`` pairs of a triple (mention count, year, ...).
    """

    ref_id: str
    kind: str  # "text" | "triple"
    text: str
    source_doc: str = ""
    pages: str = ""
    chunk_id: str = ""
    metadata: str = ""

    def source_label(self) -> str:
        """Human-readable provenance, e.g. ``REPORT.pdf | p. 129``."""
        bits = [bit for bit in (self.source_doc, self.pages) if bit]
        return " | ".join(bits)

    def display_label(self) -> str:
        """Short in-text citation, e.g. ``REPORT MATTM, p. 129``.

        Used when reference tags are rendered for a reader instead of as
        ``[S1]``/``[T1]``: "S3" says nothing to the person reading the answer,
        the document name and page are what they can go and check.

        Returns:
            The label, or the reference id when there is no provenance at all.
        """
        doc = short_doc_label(self.source_doc)
        pages = self.pages.strip()
        if doc and pages:
            return f"{doc}, {pages}"
        return doc or pages or self.ref_id


@dataclass(slots=True)
class CitationReport:
    """Outcome of verifying the reference tags emitted by the model.

    Attributes:
        answer: The answer with invalid tags marked or removed.
        cited_refs: Valid ids kept, in order of first citation.
        phantom_refs: Ids cited that the model was never shown.
        total_citations: Ids found in the answer, repeats included.
    """

    answer: str
    cited_refs: list[str] = field(default_factory=list)
    phantom_refs: list[str] = field(default_factory=list)
    total_citations: int = 0

    @property
    def phantom_rate(self) -> float:
        """Share of phantom ids among all cited ids; 0.0 when none was cited."""
        if self.total_citations <= 0:
            return 0.0
        return len(self.phantom_refs) / float(self.total_citations)

    def as_dict(self) -> dict[str, Any]:
        """The report without the answer, for the agent state."""
        return {
            "cited_refs": list(self.cited_refs),
            "phantom_refs": list(self.phantom_refs),
            "total_citations": self.total_citations,
            "phantom_rate": round(self.phantom_rate, 4),
        }


@dataclass(slots=True)
class QuoteReport:
    """Outcome of verifying the verbatim passages quoted by the model.

    Attributes:
        answer: The answer with unverified quotes turned into plain prose.
        total_quotes: Quotes long enough to be checked.
        unverified_quotes: Quotes not found in the evidence.
    """

    answer: str
    total_quotes: int = 0
    unverified_quotes: list[str] = field(default_factory=list)

    @property
    def unverified_rate(self) -> float:
        """Share of unverified quotes; 0.0 when nothing was quoted."""
        if self.total_quotes <= 0:
            return 0.0
        return len(self.unverified_quotes) / float(self.total_quotes)

    def as_dict(self) -> dict[str, Any]:
        """The report without the answer, for the agent state."""
        return {
            "total_quotes": self.total_quotes,
            "unverified_quotes": list(self.unverified_quotes),
            "unverified_rate": round(self.unverified_rate, 4),
        }


def short_doc_label(document: str, max_chars: int = 34) -> str:
    """Shorten a document filename into something readable inside a sentence.

    Drops the extension and the noise that filenames accumulate ("(web)", a
    trailing language marker, version suffixes), then truncates on a word
    boundary. The full filename stays in the closing source list.

    Args:
        document: Document basename, e.g. ``F.Fassio ... (web) it.pdf``.
        max_chars: Longest label before truncation.

    Returns:
        The shortened label, possibly empty when ``document`` is empty.
    """
    name = Path(str(document or "").strip()).stem
    name = re.sub(r"\((?:web|pdf|online)\)", " ", name, flags=re.IGNORECASE)
    name = re.sub(r"[-_]?v\d+(?=\s|$)", " ", name)
    name = re.sub(r"\s{2,}", " ", name).strip(" -_,")
    # Filenames end in a pile of publication markers ("… ita web", "…_8-18-PB"):
    # they identify the file, not the document, and eat the label's budget.
    while True:
        stripped = re.sub(
            r"[\s_-]+(?:it|ita|en|eng|web|online|def|definitivo|pb)$",
            "",
            name,
            flags=re.IGNORECASE,
        )
        if stripped == name:
            break
        name = stripped
    name = re.sub(r"_+", " ", name).strip(" -,")
    name = re.sub(r"\s{2,}", " ", name)
    if len(name) <= max_chars:
        return name

    # "Circular Economy for Food, Fassio Tecco" -> the title, not the authors:
    # cutting at the comma reads better than cutting mid-name.
    head = name.split(",", 1)[0].strip()
    if 12 <= len(head) <= max_chars:
        return head

    cut = name[:max_chars].rsplit(" ", 1)[0].rstrip(" -,")
    return f"{cut or name[:max_chars]}…"


def parse_chunk_source(source: str) -> tuple[str, str]:
    """Split a chunk source tag into ``(document name, page label)``.

    Args:
        source: Tag in the form ``<path>#page=N#chunk=M`` (any part optional).

    Returns:
        The document basename and a ``p. N`` label, both possibly empty.
    """
    raw = str(source or "").strip()
    if not raw:
        return ("", "")

    match = _CHUNK_SOURCE_RE.match(raw)
    if not match:
        return (Path(raw).name, "")

    path = (match.group("path") or "").strip()
    page = (match.group("page") or "").strip()
    doc = Path(path).name if path else ""
    return (doc, f"p. {page}" if page else "")


def _triple_provenance(triple: dict[str, Any]) -> tuple[str, str, str]:
    """Return ``(source_doc, pages, metadata)`` for a KG triple."""
    props = triple.get("relationship_properties", {})
    if not isinstance(props, dict):
        return ("", "", "")

    source_doc = str(props.get("source_doc", "") or props.get("source", "") or "").strip()
    page_range = str(props.get("page_range", "") or "").strip()

    meta: list[str] = []
    for key in ("mention_count", "confidence", "year", "value", "unit"):
        value = props.get(key)
        if value in (None, "", 0):
            continue
        meta.append(f"{key}={value}")

    return (
        Path(source_doc).name if source_doc else "",
        f"p. {page_range}" if page_range else "",
        ", ".join(meta),
    )


def _triple_text(triple: dict[str, Any]) -> str:
    """Render a triple as ``(subject, predicate, object)``; empty if all blank."""
    subject = str(triple.get("subject", "")).strip()
    predicate = str(triple.get("predicate", "")).strip()
    obj = str(triple.get("object", "")).strip()
    if not (subject or predicate or obj):
        return ""
    return f"({subject}, {predicate}, {obj})"


def build_evidence_index(
    text_chunks: Sequence[dict[str, Any]] = (),
    triples: Sequence[dict[str, Any]] = (),
    max_text_items: int = 12,
    max_triple_items: int = 30,
) -> list[EvidenceItem]:
    """Assign stable reference ids to retrieved evidence.

    Text chunks become ``S1..Sn`` and triples ``T1..Tn``, in retrieval order —
    which is relevance order, so low numbers carry the strongest signal.
    Duplicates (same chunk id, same content, same triple) collapse onto a single
    id: the same fact retrieved by two sub-queries must not be citable twice.

    Args:
        text_chunks: Dicts with ``content`` and optionally ``source``/``chunk_id``.
        triples: KG triples, already merged and ranked by the caller.
        max_text_items: Cap on text evidence entries.
        max_triple_items: Cap on triple evidence entries.

    Returns:
        Evidence items in rendering order: text first, then triples.
    """
    items: list[EvidenceItem] = []

    seen_text: set[str] = set()
    for chunk in text_chunks:
        if len(seen_text) >= max_text_items:
            break
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content", "") or "").strip()
        if not content:
            continue
        chunk_id = str(chunk.get("chunk_id", "") or "").strip()
        key = chunk_id or " ".join(content.split()).lower()[:400]
        if key in seen_text:
            continue
        seen_text.add(key)

        source_doc, pages = parse_chunk_source(str(chunk.get("source", "") or ""))
        items.append(
            EvidenceItem(
                ref_id=f"S{len(seen_text)}",
                kind="text",
                text=content,
                source_doc=source_doc,
                pages=pages,
                chunk_id=chunk_id,
            )
        )

    seen_triples: set[tuple[str, str, str]] = set()
    for triple in triples:
        if len(seen_triples) >= max_triple_items:
            break
        if not isinstance(triple, dict):
            continue
        text = _triple_text(triple)
        if not text:
            continue
        key = (
            str(triple.get("subject", "")).strip().lower(),
            str(triple.get("predicate", "")).strip().lower(),
            str(triple.get("object", "")).strip().lower(),
        )
        if key in seen_triples:
            continue
        seen_triples.add(key)

        source_doc, pages, metadata = _triple_provenance(triple)
        items.append(
            EvidenceItem(
                ref_id=f"T{len(seen_triples)}",
                kind="triple",
                text=text,
                source_doc=source_doc,
                pages=pages,
                metadata=metadata,
            )
        )

    return items


def evidence_to_dicts(items: Sequence[EvidenceItem]) -> list[dict[str, Any]]:
    """Serialise evidence items so they can travel in the (JSON-dumpable) state."""
    return [
        {
            "ref_id": item.ref_id,
            "kind": item.kind,
            "text": item.text,
            "source_doc": item.source_doc,
            "pages": item.pages,
            "chunk_id": item.chunk_id,
            "metadata": item.metadata,
        }
        for item in items
    ]


def evidence_from_dicts(rows: Sequence[dict[str, Any]]) -> list[EvidenceItem]:
    """Rebuild evidence items from their serialised form."""
    items: list[EvidenceItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ref_id = str(row.get("ref_id", "") or "").strip()
        if not ref_id:
            continue
        items.append(
            EvidenceItem(
                ref_id=ref_id,
                kind=str(row.get("kind", "") or "text"),
                text=str(row.get("text", "") or ""),
                source_doc=str(row.get("source_doc", "") or ""),
                pages=str(row.get("pages", "") or ""),
                chunk_id=str(row.get("chunk_id", "") or ""),
                metadata=str(row.get("metadata", "") or ""),
            )
        )
    return items


def refs_present_in(text: str) -> set[str]:
    """Reference ids that appear as evidence-block headers in ``text``.

    Used to learn which blocks survived context compression, so the citation
    gate can validate against what the model was shown rather than against the
    full index.

    Args:
        text: Rendered context, possibly compressed.

    Returns:
        Normalised ref ids (``S1``, ``T12``) found in the text.
    """
    return {
        f"{kind.upper()}{number}"
        for kind, number in _REF_RE.findall(text or "")
    }


def render_cited_context(
    evidence: Sequence[EvidenceItem],
    entity_sections: Sequence[tuple[str, str]] = (),
) -> str:
    """Render the model-facing context with one numbered block per evidence item.

    The question is deliberately absent: the prompt carries it in its own
    ``question`` slot, and echoing it into the context would make the context
    non-empty even with nothing retrieved, disabling the emptiness checks and
    letting the relevance grader match the query against itself.

    Args:
        evidence: Items from :func:`build_evidence_index`.
        entity_sections: ``(title, body)`` pairs for non-citable context such as
            matched nodes and neighbours — names without provenance, which must
            not become reference ids.

    Returns:
        The full context string.
    """
    sections: list[str] = []

    text_items = [item for item in evidence if item.kind == "text"]
    triple_items = [item for item in evidence if item.kind == "triple"]

    if text_items:
        blocks: list[str] = []
        for item in text_items:
            header = f"[{item.ref_id}]"
            label = item.source_label()
            if label:
                header += f" <{label}>"
            blocks.append(f"{header}\n{item.text}")
        sections.append("Evidence — source passages:\n" + "\n\n".join(blocks))

    if triple_items:
        lines: list[str] = []
        for item in triple_items:
            line = f"[{item.ref_id}] {item.text}"
            label = item.source_label()
            if label:
                line += f" <{label}>"
            if item.metadata:
                line += f" [{item.metadata}]"
            lines.append(line)
        sections.append("Evidence — knowledge graph facts:\n" + "\n".join(lines))

    for title, body in entity_sections:
        value = str(body or "").strip()
        if value:
            sections.append(f"{title}\n{value}")

    return "\n\n".join(section for section in sections if section)


def _iter_citation_groups(answer: str) -> Iterable[tuple[int, int, list[str]]]:
    """Yield ``(start, end, refs)`` for each run of citation tags.

    Adjacent tags separated only by punctuation — ``[T1], [T4], [T6]`` or the
    range form ``[S1]-[S5]`` — are one citation on one claim, so they collapse
    into a single group. Treating them separately would let a model sidestep the
    per-tag cap simply by closing and reopening the brackets.

    Args:
        answer: Model output.

    Yields:
        The span of each group and its normalised ids, in order.
    """
    group_start: int | None = None
    group_end = 0
    refs: list[str] = []

    for match in _CITATION_RE.finditer(answer):
        found = [
            f"{kind.upper()}{int(number)}"
            for kind, number in _REF_RE.findall(match.group(1))
        ]
        if not found:
            continue

        gap = answer[group_end : match.start()] if group_start is not None else None
        if gap is not None and re.fullmatch(r"[\s,;:./\\|–—-]*", gap):
            refs.extend(found)
            group_end = match.end()
            continue

        if group_start is not None:
            yield (group_start, group_end, refs)
        group_start = match.start()
        group_end = match.end()
        refs = list(found)

    if group_start is not None:
        yield (group_start, group_end, refs)


def verify_citations(
    answer: str,
    evidence: Sequence[EvidenceItem],
    policy: str = "mark",
    language: str = "it",
    max_refs_per_tag: int = 2,
    visible_refs: Collection[str] | None = None,
) -> CitationReport:
    """Check every reference tag in ``answer`` against the evidence index.

    Args:
        answer: Raw model output.
        evidence: The index the model was given for this turn.
        policy: ``"mark"`` flags unsupported tags in place; ``"strip"`` deletes
            them. Marking is the default because a silently deleted citation
            leaves an unsupported claim looking supported.
        language: ``"it"`` or ``"en"``, for the marker wording.
        max_refs_per_tag: Ids kept per tag. Models happily emit ``[T4, T5, T6]``
            for one claim, which turns the answer and the source list into
            noise; the surplus is trimmed to the most relevant (first) ids.
        visible_refs: Reference ids that survived into the context the model was
            actually shown. Context compression drops the middle of the string,
            so the index can list blocks the model never saw, and a tag pointing
            at a dropped block must not pass the gate. ``None`` validates
            against the whole index.

    Returns:
        A report carrying the processed answer and the citation counts.
    """
    known = {item.ref_id for item in evidence}
    if visible_refs is not None:
        known &= set(visible_refs)
    cited: list[str] = []
    phantom: list[str] = []
    total = 0
    replacements: list[tuple[int, int, str]] = []
    keep_per_tag = max(1, int(max_refs_per_tag))

    for start, end, refs in _iter_citation_groups(answer):
        total += len(refs)
        # Dedup inside the group: "[T1], [T1]" is one citation, not two.
        unique_refs: list[str] = []
        for ref in refs:
            if ref not in unique_refs:
                unique_refs.append(ref)
        valid = [ref for ref in unique_refs if ref in known]
        invalid = [ref for ref in unique_refs if ref not in known]
        trimmed = valid[:keep_per_tag]

        for ref in trimmed:
            if ref not in cited:
                cited.append(ref)
        for ref in invalid:
            if ref not in phantom:
                phantom.append(ref)

        rewritten = f"[{', '.join(trimmed)}]" if trimmed else ""
        if not invalid and len(valid) == len(trimmed) and answer[start:end] == rewritten:
            continue

        if trimmed:
            # Keep the part of the citation that checks out, drop the rest.
            replacement = rewritten
        elif policy == "strip":
            replacement = ""
        else:
            mark = UNVERIFIED_MARK_IT if language == "it" else UNVERIFIED_MARK_EN
            replacement = mark
        replacements.append((start, end, replacement))

    processed = answer
    for start, end, replacement in reversed(replacements):
        processed = processed[:start] + replacement + processed[end:]

    if phantom:
        logger.warning(
            "Citation gate: %d/%d reference tags not in the evidence index (%s)",
            len(phantom),
            total,
            ", ".join(phantom[:10]),
        )

    return CitationReport(
        answer=re.sub(r"[ \t]{2,}", " ", processed).strip(),
        cited_refs=cited,
        phantom_refs=phantom,
        total_citations=total,
    )


def _normalize_for_quote_match(text: str) -> str:
    """Fold a passage down to what survives a copy out of a PDF.

    Case, accents, hyphenation at line breaks, curly quotes and every run of
    punctuation or whitespace are noise here: the model reflows the passage it
    quotes, and comparing raw strings would flag a faithful quote as invented.

    Args:
        text: Passage or quote.

    Returns:
        Lower-case ASCII words separated by single spaces.
    """
    folded = unicodedata.normalize("NFKD", str(text or ""))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    # Words split across a line break in the source PDF ("circo-\nlare").
    folded = re.sub(r"[-­]\s*\n\s*", "", folded)
    return " ".join(re.sub(r"[^0-9a-zA-Z]+", " ", folded.lower()).split())


def verify_quotes(
    answer: str,
    evidence: Sequence[EvidenceItem],
    min_chars: int = 25,
) -> QuoteReport:
    """Check that every «...» passage really occurs in the retrieved evidence.

    The definitional prompt asks the model to open with the author's own
    wording. That instruction is also an invitation to fabricate one when the
    corpus has no definition to give, and the citation gate cannot catch it: a
    made-up quote can carry a perfectly valid ``[S2]``. So the quoted string
    itself is checked against the text the model was actually shown.

    A quote that fails loses its guillemets and stays in the answer as ordinary
    prose. Deleting the sentence would remove content that may well be right;
    what must not survive is the *claim* that those are the source's words.

    Args:
        answer: The answer, after :func:`verify_citations`.
        evidence: The index for this turn.
        min_chars: Below this length (normalised) a quote is a term, not a
            passage — "«scotta»" is emphasis and has nothing to verify.

    Returns:
        A report carrying the processed answer and the quote counts.
    """
    haystack = " ".join(
        _normalize_for_quote_match(item.text)
        for item in evidence
        if item.kind == "text"
    )
    total = 0
    unverified: list[str] = []
    replacements: list[tuple[int, int, str]] = []

    for match in _QUOTE_RE.finditer(str(answer or "")):
        inner = match.group(1).strip()
        normalized = _normalize_for_quote_match(inner)
        if len(normalized) < max(1, int(min_chars)):
            continue
        total += 1
        # Models elide the middle of a long passage; each remaining fragment
        # still has to be verbatim.
        fragments = [
            fragment
            for fragment in (
                _normalize_for_quote_match(part)
                for part in re.split(r"\[?(?:\.\.\.|…)\]?", inner)
            )
            if len(fragment) >= 12
        ] or [normalized]
        if all(fragment in haystack for fragment in fragments):
            continue
        unverified.append(inner)
        replacements.append((match.start(), match.end(), inner))

    processed = str(answer or "")
    for start, end, replacement in reversed(replacements):
        processed = processed[:start] + replacement + processed[end:]

    if unverified:
        logger.warning(
            "Quote gate: %d/%d quoted passages not found in the evidence, "
            "guillemets removed (%s)",
            len(unverified),
            total,
            " | ".join(quote[:60] for quote in unverified[:3]),
        )

    return QuoteReport(
        answer=processed,
        total_quotes=total,
        unverified_quotes=unverified,
    )


def _reference_sort_key(item: EvidenceItem) -> tuple[int, int]:
    """Sort key for the closing source list: text passages first, then by id."""
    try:
        number = int(item.ref_id[1:])
    except (ValueError, IndexError):
        number = 0
    return (0 if item.kind == "text" else 1, number)


def render_reference_list(
    evidence: Sequence[EvidenceItem],
    cited_refs: Sequence[str],
    language: str = "it",
    max_items: int = 8,
) -> str:
    """Render the closing source list for the references actually used.

    An answer that cited nothing gets no source list. Falling back to the top
    evidence items would attribute an uncited answer, for instance to an
    out-of-domain question, to whatever the retriever happened to return, and
    the phantom-reference metric could not see it because the renderer, not
    the model, would have written those references.

    Args:
        evidence: The index for this turn.
        cited_refs: Reference ids surviving :func:`verify_citations`.
        language: ``"it"`` or ``"en"``.
        max_items: Longest list rendered; the remainder is summarised on one
            line. A twenty-entry list is a wall, not a set of sources.

    Returns:
        The formatted section, or an empty string when nothing was cited.
    """
    by_id = {item.ref_id: item for item in evidence}
    used = [by_id[ref] for ref in cited_refs if ref in by_id]
    if not used:
        return ""

    # Index order, text passages first: they carry document *and* page, which is
    # what a reader checks. Citation order could push them below the triples
    # and, on a long list, into the "(+N more)" line.
    used = sorted(used, key=_reference_sort_key)
    shown = used[: max(1, int(max_items))]
    hidden = len(used) - len(shown)

    title = "Fonti" if language == "it" else "Sources"
    lines: list[str] = []
    for item in shown:
        label = item.source_label() or (
            "fonte non disponibile" if language == "it" else "source unavailable"
        )
        if item.kind == "triple":
            lines.append(f"- [{item.ref_id}] {item.text} — {label}")
        else:
            lines.append(f"- [{item.ref_id}] {label}")

    if hidden > 0:
        lines.append(
            f"- (+{hidden} altre evidenze citate)"
            if language == "it"
            else f"- (+{hidden} further cited items)"
        )

    return f"{title}:\n" + "\n".join(lines)


def render_display_citations(answer: str, evidence: Sequence[EvidenceItem]) -> str:
    """Replace ``[S1]``/``[T3]`` tags with document-and-page labels.

    Runs *after* :func:`verify_citations`, so every id left in the text is known:
    the ids are what the deterministic gate needs, the label is what the reader
    needs. Ids sharing a label collapse into one, which is what happens when a
    passage and a triple come from the same page.

    Args:
        answer: The verified answer.
        evidence: The index for this turn.

    Returns:
        The answer with reader-facing citation labels.
    """
    by_id = {item.ref_id: item for item in evidence}

    def _replace(match: re.Match[str]) -> str:
        """Render one tag as labels; leave it unchanged if an id is unknown."""
        # Ids from the same document merge into one label with both pages:
        # "[SEeD for Change, p. 3; SEeD for Change, p. 3-4]" names the same
        # source twice, which is how a citation stops being readable.
        pages_by_doc: dict[str, list[str]] = {}
        for kind, number in _REF_RE.findall(match.group(1)):
            item = by_id.get(f"{kind.upper()}{int(number)}")
            if item is None:
                return match.group(0)
            doc = short_doc_label(item.source_doc) or item.ref_id
            pages = pages_by_doc.setdefault(doc, [])
            page = re.sub(r"^p\.\s*", "", item.pages.strip())
            if page and page not in pages:
                pages.append(page)

        if not pages_by_doc:
            return match.group(0)

        labels = [
            f"{doc}, p. {', '.join(pages)}" if pages else doc
            for doc, pages in pages_by_doc.items()
        ]
        return f"[{'; '.join(labels)}]"

    return _CITATION_RE.sub(_replace, str(answer or ""))


def render_grouped_reference_list(
    evidence: Sequence[EvidenceItem],
    cited_refs: Sequence[str],
    language: str = "it",
    max_triples_per_doc: int = 4,
) -> str:
    """Render the closing source list grouped by document.

    One entry per document instead of one per evidence item, so an answer
    citing many items is not cut at a cap that drops the tail — the part the
    reader is least likely to have already seen in the text.

    An answer that cited nothing gets no source list — see
    :func:`render_reference_list` for why there is no fallback.

    Args:
        evidence: The index for this turn.
        cited_refs: Reference ids surviving :func:`verify_citations`.
        language: ``"it"`` or ``"en"``.
        max_triples_per_doc: Graph facts spelled out per document; the rest are
            counted on the same line.

    Returns:
        The formatted section, or an empty string when nothing was cited.
    """
    by_id = {item.ref_id: item for item in evidence}
    used = [by_id[ref] for ref in cited_refs if ref in by_id]
    if not used:
        return ""

    italian = language == "it"
    grouped: dict[str, list[EvidenceItem]] = {}
    for item in sorted(used, key=_reference_sort_key):
        key = item.source_doc or ("documento non indicato" if italian else "unnamed document")
        grouped.setdefault(key, []).append(item)

    lines: list[str] = []
    for document, items in grouped.items():
        lines.append(f"- **{document}**")
        pages = [item.pages for item in items if item.kind == "text" and item.pages]
        if pages:
            unique_pages = list(dict.fromkeys(pages))
            label = "passaggi citati" if italian else "cited passages"
            lines.append(f"  - {label}: {', '.join(unique_pages)}")

        triples = [item for item in items if item.kind == "triple"]
        if triples:
            label = "fatti dal grafo" if italian else "graph facts"
            shown = triples[: max(1, int(max_triples_per_doc))]
            rendered = "; ".join(
                f"{item.text} ({item.pages})" if item.pages else item.text
                for item in shown
            )
            hidden = len(triples) - len(shown)
            if hidden > 0:
                rendered += (
                    f"; +{hidden} altri" if italian else f"; +{hidden} more"
                )
            lines.append(f"  - {label}: {rendered}")

    title = "Fonti" if italian else "Sources"
    return f"{title}:\n" + "\n".join(lines)
