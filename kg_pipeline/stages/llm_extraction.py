"""Stage 3: extract triples from each chunk with an LLM served by vLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import unicodedata
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from kg_pipeline.models.types import (
    ChunkRecord,
    KGTriple,
    NEREntityCandidate,
    kg_triple_array_schema,
)
from kg_pipeline.prompts.extraction_prompt import build_extraction_prompt
from kg_pipeline.utils.acronym_map import update_acronym_map
from kg_pipeline.utils.validation import (
    parse_json_array,
    validate_triples,
    write_failed_chunk,
)

LOGGER = logging.getLogger("kg_pipeline")


class _EmptyExtraction(ValueError):
    """The model returned a well-formed but empty triple array for a chunk."""


class _TruncatedExtraction(ValueError):
    """Generation stopped at the token cap, so the JSON array is cut short.

    A separate type because only a retry with a larger token budget can repair
    it, and because otherwise the truncated text would surface as a JSON
    decode error that looks like malformed model output.
    """


_GENERIC_SECTION_TITLES = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "annex",
    "appendix",
    "background",
    "bibliography",
    "conclusion",
    "contents",
    "discussion",
    "executive summary",
    "foreword",
    "introduction",
    "methods",
    "methodology",
    "preface",
    "references",
    "results",
    "summary",
    "table of contents",
    # Italian equivalents (bilingual corpus)
    "bibliografia",
    "conclusione",
    "conclusioni",
    "indice",
    "introduzione",
    "metodologia",
    "prefazione",
    "riassunto",
    "ringraziamenti",
    "risultati",
    "sommario",
}

_SECTION_PREFIX_RE = re.compile(
    r"^(annex|appendix|chapter|section|part|allegato|appendice|capitolo|sezione|parte)\s+\w+",
    re.IGNORECASE,
)

# Chunks whose section is pure front/back matter are skipped before extraction:
# citation lists, acknowledgements, tables of contents and editorial boilerplate
# yield publishing metadata, not domain facts.
#
# Matched anywhere in the section title, because these are unambiguous: no
# section about circular food is called "acknowledgements".
_SKIP_SECTION_RE = re.compile(
    r"(references|bibliograph|acknowledg|table of contents|list of (figures|tables|acronyms)"
    r"|copyright|colophon|editorial board|scientific (board|committee)"
    r"|conflicts? of interest|author contributions|data availability"
    r"|supplementary (material|data)"
    r"|bibliografia|sitografia|sommario|ringraziament|colofone|comitato scientifico"
    r"|indice delle|conflitto di interess|contributi degli autori)",
    re.IGNORECASE,
)

# Matched only against the *whole* title. These words are back matter when they
# are the heading and content when they are part of one: a substring rule would
# silently drop a section called "Fonti rinnovabili".
_SKIP_SECTION_EXACT = frozenset(
    {
        "abbreviations",
        "abbreviazioni",
        "acronimi",
        "acronyms",
        "credits",
        "crediti",
        "fonti",
        "funding",
        "funding information",
        "glossario",
        "glossary",
        "index",
        "indice",
        "note",
        "notes",
        "riferimenti",
        "sources",
    }
)

# Leading section numbering: "5.", "5.1", "A.", "IV -".
_SECTION_NUMBER_RE = re.compile(r"^[\s\-—–]*(?:[0-9]+(?:\.[0-9]+)*|[A-Z]|[IVXLC]+)[.)\-—–\s]+")


def _normalise_section_title(title: str) -> str:
    """The heading with its numbering and punctuation removed, lower-cased."""
    cleaned = _SECTION_NUMBER_RE.sub("", title.strip())
    cleaned = cleaned.strip(" \t:.-—–_*#|")
    return " ".join(cleaned.lower().split())


def _should_skip_chunk(chunk: ChunkRecord) -> bool:
    """Whether a chunk belongs to a front/back-matter section.

    Args:
        chunk: Chunk to check.

    Returns:
        True when its section title matches ``_SKIP_SECTION_RE`` anywhere, or
        equals an entry of ``_SKIP_SECTION_EXACT`` once numbering is removed.
    """
    title = chunk.section_title or ""
    if _SKIP_SECTION_RE.search(title):
        return True
    return _normalise_section_title(title) in _SKIP_SECTION_EXACT


def _build_client(base_url: str, api_key: str) -> OpenAI:
    """Build a synchronous OpenAI-compatible client for the vLLM server.

    Args:
        base_url: Server base URL.
        api_key: API key; empty becomes ``"EMPTY"``.

    Returns:
        A client whose timeout is ``VLLM_HTTP_TIMEOUT`` seconds (default 900).
    """
    http_client_timeout = float(os.getenv("VLLM_HTTP_TIMEOUT", "900"))
    return OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "EMPTY",
        timeout=http_client_timeout,
    )


def _load_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented UTF-8 JSON, atomically.

    The payload is written to a ``.tmp`` sibling and moved into place, so a
    crash mid-write never leaves a truncated file for a resume to load.

    Args:
        path: Output file; parent directories are created.
        payload: JSON-serialisable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_path, path)


def _log_new_label(new_label_path: Path, label: str) -> None:
    """Append an out-of-ontology label proposed by the model to a log file."""
    new_label_path.parent.mkdir(parents=True, exist_ok=True)
    with new_label_path.open("a", encoding="utf-8") as f:
        f.write(label.strip() + "\n")


def _normalize_title(value: str) -> str:
    """Lower-case ``value`` and collapse whitespace."""
    return " ".join(value.strip().lower().split())


def _looks_like_section_title(title: str, section_title: str) -> bool:
    """Whether an entity named as a document is really a section heading.

    Args:
        title: Entity title or name.
        section_title: Title of the chunk's section.

    Returns:
        True when ``title`` equals the chunk's section title, is a generic
        section name ("introduction", "risultati", ...), or starts with a
        section prefix ("Chapter 3", "Allegato A", ...).
    """
    title_norm = _normalize_title(title)
    if not title_norm:
        return False

    section_norm = _normalize_title(section_title)
    if section_norm in {"", "smalldoc", "full document"}:
        section_norm = ""

    if section_norm and title_norm == section_norm:
        return True
    if title_norm in _GENERIC_SECTION_TITLES:
        return True
    if _SECTION_PREFIX_RE.match(title_norm):
        return True
    return False


def _entity_title(entity_text: str, props: dict[str, Any]) -> str:
    """Return the entity's ``title`` property, else ``name``, else its text."""
    value = props.get("title") or props.get("name") or entity_text
    return str(value or "")


def _enforce_labels(
    triple: KGTriple,
    allowed_labels: set[str],
    new_label_log_path: Path,
    section_title: str,
) -> KGTriple:
    """Restrict a triple's subject and object labels to the ontology.

    Labels outside ``allowed_labels`` are dropped and logged, ``Document`` is
    dropped from entities that look like section headings, and an entity left
    without labels becomes ``Concept``.

    Args:
        triple: Triple to fix; modified in place.
        allowed_labels: Ontology labels.
        new_label_log_path: Log file for dropped labels.
        section_title: Title of the chunk's section.

    Returns:
        The same triple.
    """

    def normalize_labels(labels: list[str], entity_title: str) -> list[str]:
        """Apply the label rules above to one entity's labels."""
        cleaned = [label.strip() for label in labels if label.strip()]
        if not cleaned:
            cleaned = ["Concept"]
        output: list[str] = []
        for label in cleaned:
            if label not in allowed_labels:
                _log_new_label(new_label_log_path, label)
                continue
            output.append(label)

        if "Document" in output and _looks_like_section_title(
            entity_title, section_title
        ):
            output = [label for label in output if label != "Document"]

        return output or ["Concept"]

    triple.subject_labels = normalize_labels(
        triple.subject_labels,
        _entity_title(triple.subject, triple.subject_properties),
    )
    triple.object_labels = normalize_labels(
        triple.object_labels,
        _entity_title(triple.object, triple.object_properties),
    )
    return triple


# Cap generation length so a pathological chunk (e.g. a dense table page that
# makes the model emit an unbounded JSON array under structured output) cannot
# run until the HTTP timeout. A chunk's triples comfortably fit in this budget;
# override with KG_EXTRACTION_MAX_TOKENS if a corpus needs more.
_MAX_OUTPUT_TOKENS = int(os.getenv("KG_EXTRACTION_MAX_TOKENS", "4096"))

# Upper bound for the per-chunk budget when a truncated answer is retried with
# a doubled cap. A chunk that hits the cap returns a JSON array cut mid-value,
# and only a larger budget can repair it; the ceiling still stops a runaway
# generation.
_MAX_OUTPUT_TOKENS_CEILING = int(
    os.getenv("KG_EXTRACTION_MAX_TOKENS_CEILING", str(_MAX_OUTPUT_TOKENS * 4))
)

# Minimum temperature from the second attempt on. vLLM decodes greedily at
# temperature 0 and ignores the seed, so a retry that varies only the seed
# repeats the same request and the same failure. The first attempt keeps the
# configured temperature, so chunks that succeed at once stay deterministic.
_RETRY_TEMPERATURE = float(os.getenv("KG_EXTRACTION_RETRY_TEMPERATURE", "0.3"))

_DEFAULT_CONCURRENT_REQUESTS = 8

# How many concurrency windows a dispatch batch holds. Deep enough that a slow
# chunk has company while it finishes, shallow enough that a crash does not
# throw away much more than one checkpoint interval of work.
_BATCH_WINDOWS_IN_FLIGHT = 4


def _record_failed_chunk(
    *,
    failed_path: Path,
    chunk_metadata: dict[str, Any],
    attempt: int,
    error: str,
    raw_response: str,
) -> None:
    """Append one failure row to the failed-chunks log, never raising.

    Every caller is already handling a failure. An error while writing the
    row (a full disk, a read-only run directory) would otherwise escape the
    handler and abort the whole batch, so it is only logged.

    Args:
        failed_path: JSONL file to append to.
        chunk_metadata: The failing chunk, as a dict.
        attempt: 1-based attempt number (0 for per-item validation failures).
        error: Error message.
        raw_response: Model output that caused the failure.
    """
    try:
        write_failed_chunk(
            failed_path=failed_path,
            chunk_metadata=chunk_metadata,
            attempt=attempt,
            error=error,
            raw_response=raw_response,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately last-resort
        LOGGER.warning(
            "could not record the failure of chunk %s in %s: %s",
            chunk_metadata.get("chunk_id", "<unknown>"),
            failed_path,
            exc,
        )


async def _llm_call_async(
    client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    seed: int,
    use_structured_output: bool,
    semaphore: asyncio.Semaphore,
    max_tokens: int = _MAX_OUTPUT_TOKENS,
) -> tuple[str, str]:
    """Send one extraction request, bounded by the shared semaphore.

    Args:
        client: Async OpenAI-compatible client.
        model_name: Served model name.
        prompt: Extraction prompt.
        temperature: Sampling temperature.
        seed: Sampling seed.
        use_structured_output: Constrain the output to the ``KGTriple`` array
            JSON schema.
        semaphore: Limits concurrent requests.
        max_tokens: Generation cap.

    Returns:
        ``(text, finish_reason)``. ``finish_reason == "length"`` means the
        answer was cut at ``max_tokens``.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_structured_output:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "kg_triples",
                "schema": kg_triple_array_schema(),
            },
        }
    async with semaphore:
        response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    finish_reason = str(getattr(choice, "finish_reason", "") or "")
    return choice.message.content or "", finish_reason


async def _extract_chunk_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    chunk_idx: int,
    chunk: ChunkRecord,
    prompt: str,
    model_name: str,
    temperature: float,
    seed: int,
    use_structured_output: bool,
    max_retries: int,
    allowed_label_set: set[str],
    failed_chunks_path: Path,
    new_label_log_path: Path,
    allowed_predicates: list[str] | None,
) -> tuple[int, list[KGTriple], bool]:
    """Extract and clean the triples of one chunk, retrying on failure.

    Each retry uses ``seed + attempt`` and a temperature of at least
    ``_RETRY_TEMPERATURE``. A truncated answer is retried with a doubled
    token cap, up to ``_MAX_OUTPUT_TOKENS_CEILING``. A well-formed empty array
    is accepted after a second attempt. Accepted triples are filtered and
    normalised: triples whose proper names contain words absent from the
    chunk are dropped, labels are restricted to the ontology, and
    ``source_doc``, ``chunk_id`` and ``page_range`` are set from the chunk.
    Failed attempts are appended to ``failed_chunks_path``.

    Args:
        client: Async OpenAI-compatible client.
        semaphore: Limits concurrent requests.
        chunk_idx: Position of the chunk in the stage's chunk list.
        chunk: Chunk to extract from.
        prompt: Extraction prompt for the chunk.
        model_name: Served model name.
        temperature: Temperature of the first attempt.
        seed: Seed of the first attempt.
        use_structured_output: Constrain the output to the JSON schema.
        max_retries: Maximum number of attempts.
        allowed_label_set: Ontology labels.
        failed_chunks_path: JSONL log of failed attempts.
        new_label_log_path: Log of labels outside the ontology.
        allowed_predicates: Relation vocabulary, or ``None``.

    Returns:
        ``(chunk_idx, triples, success)``; ``success`` is False when no
        attempt produced a usable answer.
    """
    raw = ""
    max_tokens = _MAX_OUTPUT_TOKENS
    for attempt in range(1, max_retries + 1):
        try:
            # Make a retry differ from the call that failed: at temperature 0
            # vLLM decodes greedily and ignores the seed, so a retry also
            # raises the temperature.
            attempt_seed = seed if attempt == 1 else seed + attempt
            attempt_temperature = (
                temperature if attempt == 1 else max(temperature, _RETRY_TEMPERATURE)
            )
            raw, finish_reason = await _llm_call_async(
                client=client,
                model_name=model_name,
                prompt=prompt,
                temperature=attempt_temperature,
                seed=attempt_seed,
                use_structured_output=use_structured_output,
                semaphore=semaphore,
                max_tokens=max_tokens,
            )
            if finish_reason == "length":
                raise _TruncatedExtraction(
                    f"generation stopped at the {max_tokens}-token cap; "
                    "the triple array is cut mid-value"
                )
            parsed = parse_json_array(raw)
            validated = _validate_raw_triples(
                raw_items=parsed,
                chunk=chunk,
                failed_chunks_path=failed_chunks_path,
                raw_response=raw,
                allowed_predicates=allowed_predicates,
            )
            cleaned: list[KGTriple] = []
            for triple in validated:
                invented = [
                    (side, word)
                    for side, name, labels in (
                        ("subject", triple.subject, triple.subject_labels),
                        ("object", triple.object, triple.object_labels),
                    )
                    for word in _name_invented_words(name, labels, chunk.text)
                ]
                if invented:
                    LOGGER.warning(
                        "chunk %s: dropped %s --[%s]--> %s, name not in the source (%s)",
                        chunk.chunk_id,
                        triple.subject,
                        triple.predicate,
                        triple.object,
                        ", ".join(f"{side}:{word}" for side, word in invented),
                    )
                    continue
                triple = _enforce_labels(
                    triple, allowed_label_set, new_label_log_path, chunk.section_title
                )
                rel = dict(triple.relationship_properties)
                # Provenance is authoritative pipeline metadata: always
                # overwrite whatever the model put there (it copies chunk ids
                # or mistypes filenames).
                rel["source_doc"] = chunk.filename
                rel["chunk_id"] = chunk.chunk_id
                rel["page_range"] = chunk.page_range
                rel.setdefault("extraction_method", "llm")
                triple.relationship_properties = rel
                cleaned.append(triple)
            return chunk_idx, cleaned, True
        except _TruncatedExtraction as exc:
            # Sampling cannot repair this one and neither can another identical
            # call: the answer was too long for the budget. Raise the budget for
            # this chunk only, up to the ceiling, then give up loudly.
            if max_tokens >= _MAX_OUTPUT_TOKENS_CEILING or attempt >= max_retries:
                LOGGER.warning(
                    "chunk %s from %s lost: %s (ceiling %d reached); "
                    "raise KG_EXTRACTION_MAX_TOKENS_CEILING or chunk smaller",
                    chunk.chunk_id,
                    chunk.filename,
                    exc,
                    _MAX_OUTPUT_TOKENS_CEILING,
                )
                _record_failed_chunk(
                    failed_path=failed_chunks_path,
                    chunk_metadata=chunk.model_dump(),
                    attempt=attempt,
                    error=f"truncated: {exc}",
                    raw_response=raw,
                )
                return chunk_idx, [], False
            max_tokens = min(max_tokens * 2, _MAX_OUTPUT_TOKENS_CEILING)
            LOGGER.info(
                "chunk %s hit the %d-token cap; retrying with max_tokens=%d",
                chunk.chunk_id,
                max_tokens // 2,
                max_tokens,
            )
        except _EmptyExtraction:
            # A well-formed empty array is a valid answer: some chunks are a
            # figure caption or a column of numbers and carry no triple. Retry
            # once in case the model skipped the chunk, then accept it without
            # recording a failure.
            if attempt >= min(2, max_retries):
                LOGGER.debug(
                    "chunk %s yielded no triples after %d attempts; accepting",
                    chunk.chunk_id,
                    attempt,
                )
                return chunk_idx, [], True
        except Exception as exc:
            _record_failed_chunk(
                failed_path=failed_chunks_path,
                chunk_metadata=chunk.model_dump(),
                attempt=attempt,
                error=str(exc),
                raw_response=raw,
            )
    return chunk_idx, [], False


async def _run_batch_async(
    batch_tasks: list[tuple[int, ChunkRecord, str]],
    client: AsyncOpenAI,
    concurrent_requests: int,
    model_name: str,
    temperature: float,
    seed: int,
    use_structured_output: bool,
    max_retries: int,
    allowed_label_set: set[str],
    failed_chunks_path: Path,
    new_label_log_path: Path,
    allowed_predicates: list[str] | None,
) -> list[tuple[int, list[KGTriple], bool]]:
    """Extract a batch of chunks concurrently.

    A chunk whose coroutine raises is recorded as failed instead of aborting
    the batch; a cancellation is re-raised.

    Args:
        batch_tasks: ``(chunk_idx, chunk, prompt)`` for each chunk.
        client: Async OpenAI-compatible client.
        concurrent_requests: Maximum requests in flight.
        model_name: Served model name.
        temperature: Temperature of the first attempt.
        seed: Seed of the first attempt.
        use_structured_output: Constrain the output to the JSON schema.
        max_retries: Maximum attempts per chunk.
        allowed_label_set: Ontology labels.
        failed_chunks_path: JSONL log of failed attempts.
        new_label_log_path: Log of labels outside the ontology.
        allowed_predicates: Relation vocabulary, or ``None``.

    Returns:
        ``(chunk_idx, triples, success)`` for each chunk, in input order.

    Raises:
        asyncio.CancelledError: If a chunk's coroutine was cancelled.
    """
    semaphore = asyncio.Semaphore(concurrent_requests)
    coros = [
        _extract_chunk_async(
            client=client,
            semaphore=semaphore,
            chunk_idx=idx,
            chunk=ch,
            prompt=pr,
            model_name=model_name,
            temperature=temperature,
            seed=seed,
            use_structured_output=use_structured_output,
            max_retries=max_retries,
            allowed_label_set=allowed_label_set,
            failed_chunks_path=failed_chunks_path,
            new_label_log_path=new_label_log_path,
            allowed_predicates=allowed_predicates,
        )
        for idx, ch, pr in batch_tasks
    ]
    # `return_exceptions=True` confines an unexpected exception to the chunk
    # that raised it. Without it the first exception propagates out of stage 3
    # and the results of the other chunks in the batch are never collected.
    settled = await asyncio.gather(*coros, return_exceptions=True)

    results: list[tuple[int, list[KGTriple], bool]] = []
    for (chunk_idx, chunk, _prompt), outcome in zip(batch_tasks, settled):
        if isinstance(outcome, asyncio.CancelledError):
            # A real cancellation is a shutdown, not a chunk that failed: it
            # must not be filed as one, and it must not be swallowed.
            raise outcome
        if isinstance(outcome, BaseException):
            LOGGER.warning(
                "chunk %s from %s lost to an unhandled %s: %s",
                chunk.chunk_id,
                chunk.filename,
                type(outcome).__name__,
                outcome,
            )
            _record_failed_chunk(
                failed_path=failed_chunks_path,
                chunk_metadata=chunk.model_dump(),
                attempt=max_retries,
                error=f"unhandled {type(outcome).__name__}: {outcome}",
                raw_response="",
            )
            results.append((chunk_idx, [], False))
        else:
            results.append(outcome)
    return results


async def _extract_all_batches_async(
    *,
    chunks_remaining: list[ChunkRecord],
    start_chunk_idx: int,
    batch_size: int,
    base_url: str,
    api_key: str,
    http_timeout: float,
    concurrent_requests: int,
    model_name: str,
    temperature: float,
    seed: int,
    use_structured_output: bool,
    max_retries: int,
    allowed_label_set: set[str],
    ner_map: dict[str, list[NEREntityCandidate]],
    allowed_labels: list[str],
    relation_vocab: list[str] | None,
    all_triples: list[KGTriple],
    acronym_map: dict[str, str],
    checkpoint_every: int,
    checkpoint_path: Path,
    checkpoint_info_path: Path,
    total_chunks: int,
    failed_chunks_path: Path,
    new_label_log_path: Path,
) -> tuple[list[KGTriple], dict[str, str], list[str]]:
    """Extract every remaining chunk in batches, checkpointing as it goes.

    For each batch, prompts are built and the acronym map is updated from the
    chunk text and the NER candidates before extraction. A checkpoint is
    written every ``checkpoint_every`` chunks and once more at the end.

    Args:
        chunks_remaining: Chunks not yet extracted.
        start_chunk_idx: Index of ``chunks_remaining[0]`` in the full list.
        batch_size: Chunks dispatched per batch.
        base_url: vLLM server base URL.
        api_key: API key; empty becomes ``"EMPTY"``.
        http_timeout: Request timeout in seconds.
        concurrent_requests: Maximum requests in flight.
        model_name: Served model name.
        temperature: Temperature of the first attempt.
        seed: Seed of the first attempt.
        use_structured_output: Constrain the output to the JSON schema.
        max_retries: Maximum attempts per chunk.
        allowed_label_set: Ontology labels, as a set.
        ner_map: NER candidates by ``chunk_id``.
        allowed_labels: Ontology labels, as passed to the prompt.
        relation_vocab: Relation vocabulary, or ``None``.
        all_triples: Triples extracted so far; extended in place.
        acronym_map: Acronym map built so far; updated in place.
        checkpoint_every: Chunks between checkpoints; 0 disables them.
        checkpoint_path: Checkpoint file for the triples.
        checkpoint_info_path: Checkpoint file for progress and the acronym map.
        total_chunks: Number of chunks in the full list.
        failed_chunks_path: JSONL log of failed attempts.
        new_label_log_path: Log of labels outside the ontology.

    Returns:
        ``(all_triples, acronym_map, failed_chunk_ids)``.
    """
    _log = logging.getLogger("kg_pipeline")
    failed_chunk_ids: list[str] = []
    chunks_since_checkpoint = 0

    def _write_checkpoint(last_chunk_idx: int) -> None:
        """Save the triples and progress so far; a failure is only logged."""
        try:
            save_triples(checkpoint_path, all_triples)
            _save_json(
                checkpoint_info_path,
                {
                    "last_completed_chunk_idx": last_chunk_idx,
                    "total_chunks": total_chunks,
                    "triples_count": len(all_triples),
                    "acronym_map": acronym_map,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            _log.info(
                f"Checkpoint saved at chunk {last_chunk_idx + 1}/{total_chunks}: "
                f"{len(all_triples)} triples"
            )
        except Exception as e:
            _log.warning(f"Failed to save checkpoint: {e}")

    async with AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "EMPTY",
        timeout=http_timeout,
    ) as client:
        with tqdm(
            total=total_chunks,
            initial=start_chunk_idx,
            desc="Stage 3 LLM Extraction",
            unit="chunk",
        ) as progress:
            for batch_offset in range(0, len(chunks_remaining), batch_size):
                batch = chunks_remaining[batch_offset : batch_offset + batch_size]
                batch_abs_start = start_chunk_idx + batch_offset

                batch_tasks: list[tuple[int, ChunkRecord, str]] = []
                for i, chunk in enumerate(batch):
                    candidates = [entity.model_dump() for entity in ner_map.get(chunk.chunk_id, [])]
                    prompt = build_extraction_prompt(chunk, candidates, allowed_labels, relation_vocab=relation_vocab)
                    update_acronym_map(acronym_map, chunk.text)
                    for entity in candidates:
                        update_acronym_map(acronym_map, entity.get("text_span", ""))
                    batch_tasks.append((batch_abs_start + i, chunk, prompt))

                results = await _run_batch_async(
                    batch_tasks=batch_tasks,
                    client=client,
                    concurrent_requests=concurrent_requests,
                    model_name=model_name,
                    temperature=temperature,
                    seed=seed,
                    use_structured_output=use_structured_output,
                    max_retries=max_retries,
                    allowed_label_set=allowed_label_set,
                    failed_chunks_path=failed_chunks_path,
                    new_label_log_path=new_label_log_path,
                    allowed_predicates=relation_vocab,
                )

                chunk_id_by_idx = {idx: ch.chunk_id for idx, ch, _ in batch_tasks}
                for _chunk_idx, triples, success in results:
                    if success:
                        all_triples.extend(triples)
                    else:
                        failed_chunk_ids.append(
                            chunk_id_by_idx.get(_chunk_idx, str(_chunk_idx))
                        )

                progress.update(len(batch))
                last_chunk_idx = batch_abs_start + len(batch) - 1
                chunks_since_checkpoint += len(batch)

                if checkpoint_every > 0 and chunks_since_checkpoint >= checkpoint_every:
                    _write_checkpoint(last_chunk_idx)
                    chunks_since_checkpoint = 0

            # Checkpoint the tail that did not reach the threshold, so a rerun
            # does not redo work already done.
            if checkpoint_every > 0 and chunks_since_checkpoint > 0:
                _write_checkpoint(start_chunk_idx + len(chunks_remaining) - 1)

    return all_triples, acronym_map, failed_chunk_ids


# Labels whose name is copied from the text rather than composed by the model.
# `Indicator` and `DataValue` are deliberately out: the prompt *asks* for a
# composed name there ("food waste per capita Italy 2022"), so a token that is
# not in the chunk is expected.
_COPIED_NAME_LABELS = frozenset(
    {"Organization", "Person", "Place", "Project", "Document", "Product", "Event"}
)


def _fold_for_lookup(text: str) -> str:
    """Lower-case, strip accents and reduce ``text`` to space-separated words."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", folded)).strip()


def _name_invented_words(name: str, labels: list[str] | None, chunk_text: str) -> list[str]:
    """Find the words of a proper name that the chunk does not contain.

    Models sometimes translate fragments of a proper name copied from the
    source (``nel nome del pane`` becoming ``Nel name del pane``), although
    the prompt asks to keep names in the source language. A name is suspect
    when some of its words appear in the chunk and some do not: that is a
    copy the model edited, not a name it composed. Only labels whose names
    are copied from the text are checked, and only words of three or more
    characters, compared accent- and case-insensitively.

    Args:
        name: Entity name.
        labels: Entity labels.
        chunk_text: Source chunk text.

    Returns:
        The words missing from the chunk, or an empty list when the name is
        not checked, fully present, or entirely absent.
    """
    if not set(labels or []) & _COPIED_NAME_LABELS:
        return []
    words = [w for w in _fold_for_lookup(name).split() if len(w) >= 3]
    if not words:
        return []
    haystack = f" {_fold_for_lookup(chunk_text)} "
    missing = [w for w in words if f" {w} " not in haystack]
    return missing if missing and len(missing) < len(words) else []


def _validate_raw_triples(
    raw_items: list[dict[str, Any]],
    chunk: ChunkRecord,
    failed_chunks_path: Path,
    raw_response: str,
    allowed_predicates: list[str] | None,
) -> list[KGTriple]:
    """Validate the items of one response, dropping and logging invalid ones.

    Args:
        raw_items: Items parsed from the model output.
        chunk: Source chunk, recorded with each invalid item.
        failed_chunks_path: JSONL log of invalid items.
        raw_response: Unused; each invalid item is logged on its own.
        allowed_predicates: Relation vocabulary, or ``None``.

    Returns:
        The valid triples.

    Raises:
        _EmptyExtraction: If ``raw_items`` is empty, so the caller can tell
            "no triples here" apart from a failed call.
    """
    if not raw_items:
        raise _EmptyExtraction("LLM returned an empty items array for this chunk")

    valid_triples: list[KGTriple] = []

    for item in raw_items:
        try:
            valid_triples.extend(
                validate_triples([item], allowed_predicates=allowed_predicates)
            )
        except Exception as exc:
            _record_failed_chunk(
                failed_path=failed_chunks_path,
                chunk_metadata=chunk.model_dump(),
                attempt=0,
                error=str(exc),
                raw_response=json.dumps(item, ensure_ascii=False),
            )

    return valid_triples


def extract_triples(
    chunks: list[ChunkRecord],
    ner_map: dict[str, list[NEREntityCandidate]],
    allowed_labels: list[str],
    base_url: str,
    model_name: str,
    api_key: str,
    max_retries_per_chunk: int,
    temperature: float,
    seed: int,
    use_structured_output: bool,
    failed_chunks_path: Path,
    new_label_log_path: Path,
    relation_vocab: list[str] | None = None,
    checkpoint_every: int = 50,
    batch_size: int | None = None,
) -> tuple[list[KGTriple], dict[str, str]]:
    """Extract triples from every chunk, resuming from a checkpoint if present.

    Front/back-matter chunks are skipped. The checkpoint files
    ``stage3_checkpoint.json`` and ``stage3_checkpoint_info.json`` live next to
    ``failed_chunks_path``; when both exist, extraction resumes after the last
    completed chunk. A per-run verdict is written to ``stage3_summary.json`` in
    the same directory. Concurrency comes from
    ``GRAPHRAG_LLM_CONCURRENT_REQUESTS`` (default 8) and the request timeout
    from ``VLLM_HTTP_TIMEOUT`` (default 900 s).

    Args:
        chunks: Stage 1 chunks.
        ner_map: NER candidates by ``chunk_id``.
        allowed_labels: Ontology labels.
        base_url: vLLM server base URL.
        model_name: Served model name.
        api_key: API key.
        max_retries_per_chunk: Maximum attempts per chunk.
        temperature: Temperature of the first attempt.
        seed: Seed of the first attempt.
        use_structured_output: Constrain the output to the JSON schema.
        failed_chunks_path: JSONL log of failed attempts.
        new_label_log_path: Log of labels outside the ontology.
        relation_vocab: Relation vocabulary, or ``None`` for no check.
        checkpoint_every: Save a checkpoint at least every N chunks; 0
            disables checkpointing.
        batch_size: Chunks dispatched to the model at once. ``None`` or a
            non-positive value uses ``_BATCH_WINDOWS_IN_FLIGHT`` times the
            concurrency limit.

    Returns:
        ``(triples, acronym_map)``.
    """
    _log = logging.getLogger("kg_pipeline")
    allowed_label_set = set(allowed_labels)
    chunks_in = len(chunks)

    # Drop front/back-matter chunks before any indexing so checkpoint indices
    # stay aligned with the filtered list.
    skipped = [c for c in chunks if _should_skip_chunk(c)]
    if skipped:
        chunks = [c for c in chunks if not _should_skip_chunk(c)]
        _log.info(
            "Skipping %d front/back-matter chunks (sections: %s)",
            len(skipped),
            sorted({c.section_title for c in skipped})[:10],
        )

    all_triples: list[KGTriple] = []
    acronym_map: dict[str, str] = {}

    checkpoint_path = failed_chunks_path.parent / "stage3_checkpoint.json"
    checkpoint_info_path = failed_chunks_path.parent / "stage3_checkpoint_info.json"

    start_chunk_idx = 0
    if checkpoint_path.exists() and checkpoint_info_path.exists():
        try:
            all_triples = load_triples(checkpoint_path)
            checkpoint_info = _load_json(checkpoint_info_path)
            start_chunk_idx = checkpoint_info.get("last_completed_chunk_idx", 0) + 1
            acronym_map = checkpoint_info.get("acronym_map", {})
            # The triples file and the info file are written separately: a crash
            # between the two can leave triples from chunks past
            # last_completed_chunk_idx. Drop them so resume never duplicates.
            completed_chunk_ids = {c.chunk_id for c in chunks[:start_chunk_idx]}
            before_count = len(all_triples)
            all_triples = [
                t
                for t in all_triples
                if "chunk_id" not in t.relationship_properties
                or str(t.relationship_properties.get("chunk_id", ""))
                in completed_chunk_ids
            ]
            if len(all_triples) != before_count:
                _log.warning(
                    "Dropped %d checkpoint triples from chunks past the last "
                    "completed checkpoint (inconsistent crash recovery state)",
                    before_count - len(all_triples),
                )
            _log.info(
                f"Resuming from checkpoint: chunk {start_chunk_idx}/{len(chunks)}, "
                f"triples so far: {len(all_triples)}"
            )
        except Exception as e:
            _log.warning(f"Could not load checkpoint: {e}")
            all_triples = []
            acronym_map = {}
            start_chunk_idx = 0

    try:
        concurrent_requests = int(os.getenv("GRAPHRAG_LLM_CONCURRENT_REQUESTS", str(_DEFAULT_CONCURRENT_REQUESTS)))
    except ValueError:
        concurrent_requests = _DEFAULT_CONCURRENT_REQUESTS
    concurrent_requests = max(1, concurrent_requests)

    http_timeout = float(os.getenv("VLLM_HTTP_TIMEOUT", "900"))
    # A batch is the dispatch window: `_run_batch_async` holds it all in flight
    # behind a semaphore of `concurrent_requests`, and the next batch cannot
    # start until the slowest chunk of this one returns. Sized at the
    # concurrency limit, every straggler would idle the whole window, so the
    # default is a few windows deep. It is independent of the checkpoint
    # cadence.
    if batch_size is None or batch_size <= 0:
        batch_size = concurrent_requests * _BATCH_WINDOWS_IN_FLIGHT
    batch_size = max(1, int(batch_size))
    chunks_remaining = chunks[start_chunk_idx:]

    all_triples, acronym_map, failed_chunk_ids = asyncio.run(
        _extract_all_batches_async(
            chunks_remaining=chunks_remaining,
            start_chunk_idx=start_chunk_idx,
            batch_size=batch_size,
            base_url=base_url,
            api_key=api_key,
            http_timeout=http_timeout,
            concurrent_requests=concurrent_requests,
            model_name=model_name,
            temperature=temperature,
            seed=seed,
            use_structured_output=use_structured_output,
            max_retries=max_retries_per_chunk,
            allowed_label_set=allowed_label_set,
            ner_map=ner_map,
            allowed_labels=allowed_labels,
            relation_vocab=relation_vocab,
            all_triples=all_triples,
            acronym_map=acronym_map,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            checkpoint_info_path=checkpoint_info_path,
            total_chunks=len(chunks),
            failed_chunks_path=failed_chunks_path,
            new_label_log_path=new_label_log_path,
        )
    )

    if failed_chunk_ids:
        _log.warning(
            "%d of %d chunks produced no triples after %d attempts each "
            "(first: %s). Details in %s",
            len(failed_chunk_ids),
            len(chunks_remaining),
            max_retries_per_chunk,
            ", ".join(failed_chunk_ids[:5]),
            failed_chunks_path,
        )
    else:
        _log.info("Stage 3: every chunk extracted, no chunk lost")

    # `failed_chunks.jsonl` holds one row per failed *attempt*, including
    # attempts later retried successfully, so its line count does not measure
    # lost chunks. The per-chunk verdict is recorded here instead.
    _save_json(
        failed_chunks_path.parent / "stage3_summary.json",
        {
            "chunks_in": chunks_in,
            "chunks_skipped_front_back_matter": len(skipped),
            "chunks_eligible": len(chunks),
            "chunks_resumed_from_checkpoint": start_chunk_idx,
            "chunks_attempted": len(chunks_remaining),
            "chunks_failed": len(failed_chunk_ids),
            "failed_chunk_ids": failed_chunk_ids,
            "triples_extracted": len(all_triples),
            "max_retries_per_chunk": max_retries_per_chunk,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    return all_triples, acronym_map


def save_triples(path: Path, triples: list[KGTriple]) -> None:
    """Write triples to a JSON file atomically.

    Args:
        path: Output file; parent directories are created.
        triples: Triples to write.
    """
    payload = [triple.as_dict() for triple in triples]
    _save_json(path, payload)


def load_triples(path: Path) -> list[KGTriple]:
    """Read triples written by :func:`save_triples`.

    Args:
        path: JSON file to read.

    Returns:
        The validated triples.
    """
    payload = _load_json(path)
    return [KGTriple.model_validate(item) for item in payload]


def save_acronyms(path: Path, acronym_map: dict[str, str]) -> None:
    """Write the acronym map to a JSON file atomically.

    Args:
        path: Output file; parent directories are created.
        acronym_map: Mapping from acronym to long form.
    """
    _save_json(path, acronym_map)


def load_acronyms(path: Path) -> dict[str, str]:
    """Read the acronym map written by :func:`save_acronyms`.

    Args:
        path: JSON file to read.

    Returns:
        Mapping from acronym to long form.
    """
    return _load_json(path)


def _cli() -> None:
    """Run stage 3 standalone from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-json", required=True)
    parser.add_argument("--ner-json", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--relation-vocab-json", default="")
    parser.add_argument("--output-triples-json", required=True)
    parser.add_argument("--output-acronyms-json", required=True)
    parser.add_argument("--failed-chunks-jsonl", required=True)
    parser.add_argument("--new-label-log", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structured-output", action="store_true")
    args = parser.parse_args()

    chunks_payload = _load_json(Path(args.chunks_json))
    ner_payload = _load_json(Path(args.ner_json))
    labels = _load_json(Path(args.labels_json))
    relation_vocab = None
    if args.relation_vocab_json:
        relation_vocab = _load_json(Path(args.relation_vocab_json))

    chunks = [ChunkRecord.model_validate(item) for item in chunks_payload]
    ner_map = {
        chunk_id: [NEREntityCandidate.model_validate(e) for e in entities]
        for chunk_id, entities in ner_payload.items()
    }

    triples, acronym_map = extract_triples(
        chunks=chunks,
        ner_map=ner_map,
        allowed_labels=labels,
        relation_vocab=relation_vocab,
        base_url=args.base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        max_retries_per_chunk=args.max_retries,
        temperature=args.temperature,
        seed=args.seed,
        use_structured_output=args.structured_output,
        failed_chunks_path=Path(args.failed_chunks_jsonl),
        new_label_log_path=Path(args.new_label_log),
    )

    save_triples(Path(args.output_triples_json), triples)
    save_acronyms(Path(args.output_acronyms_json), acronym_map)


if __name__ == "__main__":
    _cli()
