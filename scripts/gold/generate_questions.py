"""Generate a test question suite from the documents of a KG pipeline run.

The served LLM writes questions of five types (fact-based, multi-hop,
comparative, aggregation, cross-document) from each document's chunks, and
optionally a ground-truth answer for each. Questions about metadata, placeholder
entities, near-duplicates and self-answering statements are dropped, and with
ground truth on, so is any question whose answer is a refusal or names none of
its expected entities.

Usage:
    python scripts/gold/generate_questions.py generate --output suite.json
    python scripts/gold/generate_questions.py stats --input suite.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure local package imports work when launched as `python scripts/gold/generate_questions.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_pipeline.models.types import ChunkRecord, DocumentRecord
from kg_pipeline.stages.chunking import load_chunks
from kg_pipeline.stages.ingestion import load_documents


LOGGER = logging.getLogger("generate_questions")

DEFAULT_OUTPUT = Path("artifacts/tmp/graphrag_test_suite.json")
DEFAULT_RUN_ROOT = Path("kg_pipeline/artifacts")
QUESTION_TYPES = ["fact_based", "multi_hop", "comparative", "aggregation", "cross_doc"]
NON_CROSS_TYPES = QUESTION_TYPES[:-1]
TARGET_COUNTS = {
    "fact_based": 7,
    "multi_hop": 5,
    "comparative": 4,
    "aggregation": 4,
    "cross_doc": 5,
}
QUESTION_NORMALIZE_RE = re.compile(r"\s+")
QUESTION_TOKEN_RE = re.compile(r"[^a-z0-9]+")
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
REFUSAL_PATTERNS = [
    re.compile(r"contesto[^\.\n]{0,120}(non\s+contiene|insufficiente)", re.IGNORECASE),
    re.compile(r"non\s+[eè]\s+possibile\s+(fornire\s+)?(una\s+)?rispost", re.IGNORECASE),
    re.compile(r"non\s+ho\s+informazioni\s+sufficienti", re.IGNORECASE),
    re.compile(r"(not\s+enough|insufficient)\s+context", re.IGNORECASE),
    re.compile(r"cannot\s+answer\s+from\s+the\s+provided\s+context", re.IGNORECASE),
    # LLM answers "the context does not provide/contain/mention..."
    re.compile(r"(the\s+)?(context|provided\s+context)\s+(does\s+not|doesn'?t)\s+(provide|contain|mention|include|specify)", re.IGNORECASE),
    re.compile(r"no\s+information\s+(is\s+)?(available|provided|found|given)\s+in\s+the\s+context", re.IGNORECASE),
    re.compile(r"(the\s+)?context\s+provided\s+does\s+not", re.IGNORECASE),
    re.compile(r"(the\s+)?(document|text|passage)\s+(does\s+not|doesn'?t)\s+(mention|contain|provide|specify|include)", re.IGNORECASE),
    re.compile(r"il\s+contesto\s+(fornito\s+)?(non\s+fornisce|non\s+contiene|non\s+menziona|non\s+specifica)", re.IGNORECASE),
    re.compile(r"non\s+(fornisce|contiene|menziona)\s+informazioni", re.IGNORECASE),
    re.compile(r"il\s+documento\s+non\s+(menziona|contiene|fornisce)", re.IGNORECASE),
]
QUESTION_METADATA_PATTERNS = [
    re.compile(r"\b(page|pages|page\s+range|issn|annex|table|figure)\b", re.IGNORECASE),
    re.compile(r"\bchapter\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bsection\s+\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bonline\s+issn\b", re.IGNORECASE),
    re.compile(r"\btitle\s+of\s+the\s+(report|document|publication)\b", re.IGNORECASE),
    re.compile(r"^what\s+is\s+the\s+title\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+the\s+(name\s+of\s+the\s+)?(document|publication)\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+chapter\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+figure\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+table\b", re.IGNORECASE),
    # Placeholder/generic questions (model hallucination artefacts)
    re.compile(r"\b(organization|company|indicator|entity|region|country)\s+[A-Z]\b", re.IGNORECASE),
    re.compile(r"\b(organization|company|indicator|entity|region|country)\s+[A-Z]\s+or\s+[A-Z]\b", re.IGNORECASE),
    re.compile(r"\btechcorp\b", re.IGNORECASE),
    re.compile(r"\binnovatech\b", re.IGNORECASE),
    # Off-domain economic/demographic data not relevant to food safety/systems docs
    re.compile(r"\bgdp\s+(growth|rate)\b", re.IGNORECASE),
    re.compile(r"\bunemployment\s+rate\b", re.IGNORECASE),
    re.compile(r"\btotal\s+(number\s+of\s+educational|population\s+of\s+the\s+countries)\b", re.IGNORECASE),
    # Meta-questions about retrieval/evaluation methodology
    re.compile(r"\b(graphrag|graph\s+rag|knowledge\s+graph\s+retrieval|rag\s+system|retrieval.augmented)\b", re.IGNORECASE),
    re.compile(r"\b(multi.hop\s+questions|comparative\s+questions|evaluation\s+metric)\b", re.IGNORECASE),
]
JSON_ONLY_SUFFIX = "Return only valid JSON. Do not use markdown, code fences, or backticks."

# JSON Schema for guided/structured generation (vLLM response_format).
# Ensures the model always returns a well-formed {"questions": [...]} envelope.
_QUESTION_ITEM_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question", "type", "source_doc", "source_docs", "expected_entities"],
    "properties": {
        "question": {"type": "string"},
        "type": {"type": "string", "enum": QUESTION_TYPES},
        "source_doc": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "source_docs": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "expected_entities": {"type": "array", "items": {"type": "string"}},
    },
}
_QUESTIONS_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": _QUESTION_ITEM_SCHEMA}},
}
_GROUND_TRUTH_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ground_truth"],
    "properties": {"ground_truth": {"type": "string"}},
}
QUESTION_LANGUAGES = {
    "en": "English",
    "it": "Italian",
}

# Markers that suggest an aggregation question (numeric/set operations).
# Anchored at word-start only; no trailing \b so "totale", "quante" etc. match.
_AGGREGATION_MARKERS_RE = re.compile(
    r"\b(total|sum|average|mean|count|how many|how much|percentage|rate|"
    r"totale|somma|media|quanti|quante|percentuale|tasso)",
    re.IGNORECASE,
)
# Markers for comparative questions.
# "compar" / "differ" are prefixes, so no trailing \b (matches "compare",
# "comparison", "difference", etc.).
_COMPARATIVE_MARKERS_RE = re.compile(
    r"\b(compar|differ|versus|vs\.?|more than|less than|higher|lower|"
    r"respect to|rispetto|confronto|maggiore|minore|superiore|inferiore)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VLLMConfig:
    """Where and how to reach the served model.

    Attributes:
        base_url: OpenAI-compatible base URL, without the trailing slash.
        model_name: Served model name.
        api_key: Bearer token; ``EMPTY`` for a local vLLM.
    """

    base_url: str
    model_name: str
    api_key: str


def _setup_logging(verbose: bool) -> None:
    """INFO logging when `verbose`, WARNING otherwise."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _normalize_text(value: str) -> str:
    """Accent-folded, lowercased, alphanumeric-only form used for deduplication."""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = QUESTION_TOKEN_RE.sub(" ", normalized.lower())
    return QUESTION_NORMALIZE_RE.sub(" ", normalized).strip()


def _looks_like_refusal(text: str) -> bool:
    """True when the answer is empty or says the context does not cover it."""
    candidate = text.strip()
    if not candidate:
        return True
    return any(pattern.search(candidate) for pattern in REFUSAL_PATTERNS)


def _is_metadata_question(text: str) -> bool:
    """True for empty, bibliographic, placeholder or off-domain questions."""
    candidate = text.strip()
    if not candidate:
        return True
    return any(pattern.search(candidate) for pattern in QUESTION_METADATA_PATTERNS)


def _iter_json_candidates(raw: str) -> list[str]:
    """Substrings of a model reply that may parse as JSON, most complete first.

    Args:
        raw: The reply.

    Returns:
        The whole reply, the fenced blocks, the outermost object or array, and
        everything from the first bracket, without repeats.
    """
    stripped = raw.strip()
    if not stripped:
        return []

    candidates: list[str] = [stripped]

    for match in JSON_FENCE_RE.findall(stripped):
        fragment = match.strip()
        if fragment:
            candidates.append(fragment)

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start >= 0 and end > start:
            candidates.append(stripped[start : end + 1].strip())

    starts = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if starts:
        candidates.append(stripped[min(starts) :].strip())

    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def _load_json(path: Path) -> Any:
    """Parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    """Write `payload` as indented JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_questions_txt(path: Path, questions: list[str]) -> None:
    """Write one question per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(question.strip() for question in questions if question.strip())
    path.write_text((body + "\n") if body else "", encoding="utf-8")


def _question_language_label(question_language: str) -> str:
    """English name of a question-language code; English by default."""
    code = str(question_language or "en").strip().lower()
    return QUESTION_LANGUAGES.get(code, QUESTION_LANGUAGES["en"])


def _suite_to_matrix_questions(payload: dict[str, Any], max_questions: int = 0) -> list[str]:
    """The suite's questions as plain lines for the retrieval-matrix runner.

    Args:
        payload: The generated suite.
        max_questions: Keep at most this many; 0 keeps all.

    Returns:
        The questions in order, near-identical wordings removed.
    """
    raw_items = payload.get("questions", [])
    if not isinstance(raw_items, list):
        return []

    lines: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            question = str(item.get("question", "")).strip()
        else:
            question = str(item).strip()

        if not question:
            continue

        normalized = _normalize_text(question)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        lines.append(question)

        if max_questions > 0 and len(lines) >= max_questions:
            break

    return lines


def _discover_latest_run_dir(run_root: Path) -> Path:
    """The most recently modified ``run_*`` directory under `run_root`.

    Raises:
        FileNotFoundError: If `run_root` is missing or holds no run.
    """
    if not run_root.exists() or not run_root.is_dir():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    candidates = [path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found under: {run_root}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_run_artifacts(run_dir: Path) -> tuple[list[DocumentRecord], list[ChunkRecord]]:
    """Load a run's stage 0 documents and stage 1 chunks.

    Raises:
        FileNotFoundError: If either artifact is missing.
    """
    docs_path = run_dir / "stage0_documents.json"
    chunks_path = run_dir / "stage1_chunks.json"
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing documents artifact: {docs_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks artifact: {chunks_path}")

    documents = load_documents(docs_path)
    chunks = load_chunks(chunks_path)
    return documents, chunks


def _resolve_docs_and_chunks(run_dir: Path, doc_filter: str | None) -> tuple[Path, list[DocumentRecord], list[ChunkRecord]]:
    """Load a run's documents and chunks, optionally restricted to one document.

    Args:
        run_dir: KG pipeline run directory.
        doc_filter: Filename or doc_id to keep; ``None`` keeps all.

    Returns:
        The run directory, the documents and their chunks.

    Raises:
        ValueError: If the filter matches nothing or nothing is left.
    """
    documents, chunks = _load_run_artifacts(run_dir)
    if doc_filter:
        filtered_docs = [doc for doc in documents if doc_filter in {doc.filename, doc.doc_id}]
        if not filtered_docs:
            available = ", ".join(sorted({doc.filename for doc in documents}))
            raise ValueError(f"Document '{doc_filter}' not found in {run_dir}. Available: {available}")
        filtered_ids = {doc.doc_id for doc in filtered_docs}
        documents = filtered_docs
        chunks = [chunk for chunk in chunks if chunk.doc_id in filtered_ids]
    if not documents:
        raise ValueError("No documents available after filtering")
    if not chunks:
        raise ValueError("No chunks available after filtering")
    return run_dir, documents, chunks


def _resolve_vllm_config() -> VLLMConfig:
    """Read the endpoint, model and key from the environment.

    Raises:
        ValueError: If ``VLLM_MODEL_NAME`` is not set.
    """
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip().rstrip("/")
    model_name = os.getenv("VLLM_MODEL_NAME", "").strip()
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")).strip()
    if not model_name:
        raise ValueError("Missing VLLM_MODEL_NAME environment variable")
    return VLLMConfig(base_url=base_url, model_name=model_name, api_key=api_key or "EMPTY")


def _chat_completion(
    vllm: VLLMConfig,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    guided_schema: dict | None = None,
) -> str:
    """One chat completion from the served model.

    Args:
        vllm: Endpoint settings.
        messages: Chat messages.
        temperature: Sampling temperature.
        guided_schema: JSON schema the reply must follow, if any.

    Returns:
        The reply text.

    Raises:
        RuntimeError: If the endpoint is unreachable or answers an HTTP error.
    """
    payload: dict = {
        "model": vllm.model_name,
        "messages": messages,
        "temperature": temperature,
    }
    if guided_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "questions_output", "schema": guided_schema},
        }
    request = urllib.request.Request(
        url=f"{vllm.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {vllm.api_key}",
        },
        method="POST",
    )
    timeout_sec = float(os.getenv("VLLM_HTTP_TIMEOUT", "900"))
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach vLLM endpoint at {vllm.base_url}: {exc}") from exc

    payload_obj = json.loads(body)
    return str(payload_obj["choices"][0]["message"].get("content", ""))


def _parse_json_once(raw: str) -> Any:
    """Parse the first JSON candidate of a reply that decodes.

    Raises:
        json.JSONDecodeError: If no candidate decodes.
    """
    last_error: json.JSONDecodeError | None = None
    for candidate in _iter_json_candidates(raw):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON content found in model response", raw, 0)


def _call_json_llm(
    vllm: VLLMConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    guided_schema: dict | None = None,
) -> Any:
    """Ask for JSON and repair an invalid reply up to twice.

    Args:
        vllm: Endpoint settings.
        system_prompt: System message.
        user_prompt: User message.
        temperature: Sampling temperature of the first call.
        guided_schema: JSON schema for the first call.

    Returns:
        The decoded JSON.

    Raises:
        RuntimeError: If no reply parses after the repairs.
    """
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = _chat_completion(vllm, base_messages, temperature=temperature, guided_schema=guided_schema)
    current_response = raw
    parse_error: Exception | None = None

    for attempt in range(3):
        try:
            return _parse_json_once(current_response)
        except (json.JSONDecodeError, TypeError) as exc:
            parse_error = exc
            if attempt == 2:
                break

            # Use non-zero temperature for repair attempts so repeated broken
            # responses don't just reproduce the same malformed output.
            repair_temperature = max(temperature, 0.3)
            repair_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was invalid JSON. Repair it and return only valid JSON.\n\n"
                        f"Original prompt:\n{user_prompt}\n\n"
                        f"Invalid answer:\n{current_response}"
                    ),
                },
            ]
            # Repair fallback: drop guided_schema so we can accept any valid
            # JSON the model produces rather than forcing the schema again.
            current_response = _chat_completion(vllm, repair_messages, temperature=repair_temperature)

    preview = current_response[:400].replace("\n", " ")
    raise RuntimeError(f"Failed to parse JSON response after retries: {parse_error}; preview={preview}")


def _section_join(chunks: list[ChunkRecord]) -> str:
    """Chunks as one text, each under a header with index, pages and section."""
    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"[Chunk {chunk.chunk_index} | pages {chunk.page_range} | {chunk.section_title}]\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    """`text` cut to `limit` characters, with an ellipsis when cut."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _sample_chunks_for_context(
    chunks: list[ChunkRecord],
    limit_chars: int,
    *,
    seed_offset: int = 0,
) -> list[ChunkRecord]:
    """Return a subset of chunks that fits within limit_chars.

    For small corpora (total text ≤ limit_chars) returns all chunks in order.
    For large corpora, stratifies chunks into equally-spaced buckets so that
    content from across the entire document is represented rather than only
    the opening sections.  seed_offset shifts the starting position within
    each bucket, enabling varied windows across refill rounds.
    """
    ordered = sorted(chunks, key=lambda c: c.chunk_index)
    full_text = _section_join(ordered)
    if len(full_text) <= limit_chars:
        return ordered

    # Estimate how many chunks fit in the budget.
    avg_chars = len(full_text) / max(len(ordered), 1)
    budget_chunks = max(1, int(limit_chars / avg_chars))

    if budget_chunks >= len(ordered):
        return ordered

    # Stratified sampling: pick one chunk from each equally-spaced stratum.
    selected: list[ChunkRecord] = []
    stratum_size = len(ordered) / budget_chunks
    for i in range(budget_chunks):
        pos = int((i + 0.5 + (seed_offset % budget_chunks) / budget_chunks) * stratum_size)
        pos = min(pos, len(ordered) - 1)
        selected.append(ordered[pos])

    return sorted(selected, key=lambda c: c.chunk_index)


def _build_doc_context(
    doc: DocumentRecord,
    chunks: list[ChunkRecord],
    limit_chars: int = 12000,
    *,
    seed_offset: int = 0,
) -> str:
    """A document's sampled chunks as context, at most `limit_chars` long."""
    sampled = _sample_chunks_for_context(chunks, limit_chars, seed_offset=seed_offset)
    return _truncate(_section_join(sampled), limit_chars)


def _build_doc_summary(doc: DocumentRecord, chunks: list[ChunkRecord], limit_chars: int = 2400) -> str:
    """A shorter context for one document, used in cross-document prompts."""
    context = _build_doc_context(doc, chunks, limit_chars=limit_chars * 2)
    return _truncate(context, limit_chars)


def _distributed_counts(total: int, buckets: int) -> list[int]:
    """Split `total` over `buckets` as evenly as possible."""
    if buckets <= 0:
        return []
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _split_type_counts(total: int, remaining: dict[str, int], allowed_types: list[str]) -> dict[str, int]:
    """Split `total` questions over the types, in proportion to what each lacks.

    Args:
        total: Questions to allocate.
        remaining: Questions still wanted per type.
        allowed_types: Types that may receive questions.

    Returns:
        Count per allowed type, never above what that type still needs.
    """
    if total <= 0:
        return {question_type: 0 for question_type in allowed_types}

    available_total = sum(max(0, remaining.get(question_type, 0)) for question_type in allowed_types)
    if available_total <= 0:
        return {question_type: 0 for question_type in allowed_types}

    provisional: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    assigned = 0
    for question_type in allowed_types:
        available = max(0, remaining.get(question_type, 0))
        exact = total * available / available_total
        count = min(available, int(exact))
        provisional[question_type] = count
        assigned += count
        fractions.append((exact - count, question_type))

    fractions.sort(reverse=True)
    for _, question_type in fractions:
        if assigned >= total:
            break
        if provisional[question_type] >= max(0, remaining.get(question_type, 0)):
            continue
        provisional[question_type] += 1
        assigned += 1

    for question_type in allowed_types:
        provisional.setdefault(question_type, 0)
    return provisional


def _question_generation_system_prompt(question_language: str) -> str:
    """System message for question generation."""
    language_label = _question_language_label(question_language)
    return (
        "You are an expert at creating factual question-answer pairs from domain documents. "
        "Generate only questions whose answers are explicitly present in the provided context. "
        "Questions must be about the CONTENT of the documents (regulations, organizations, data, policies), "
        "not about the evaluation method, retrieval systems, or document structure. "
        f"Write every question in {language_label}. "
        "You must follow the JSON schema exactly. "
        f"{JSON_ONLY_SUFFIX}"
    )


def _ground_truth_system_prompt(question_language: str) -> str:
    """System message for ground-truth generation."""
    language_label = _question_language_label(question_language)
    return (
        "You are an expert in Knowledge Graph and RAG evaluation. "
        "Write a factual and concise ground truth answer anchored only in the provided context. "
        f"Write every answer in {language_label}. "
        "Use at most 3 sentences. "
        f"{JSON_ONLY_SUFFIX}"
    )


def _format_question_prompt(
    *,
    context: str,
    question_counts: dict[str, int],
    source_label: str,
    question_language: str,
) -> str:
    """User message asking for per-document questions of the given counts."""
    language_label = _question_language_label(question_language)
    schema = {
        "questions": [
            {
                "question": "...",
                "type": "fact_based",
                "source_doc": source_label,
                "source_docs": None,
                "expected_entities": ["..."],
            }
        ]
    }
    return (
        "Create test questions for GraphRAG evaluation.\n"
        f"Requested counts: {json.dumps(question_counts, ensure_ascii=False)}\n"
        f"Required language: {language_label}\n"
        "Allowed question types: fact_based, multi_hop, comparative, aggregation.\n"
        "Rules:\n"
        "- Generate only questions that can be answered from the context.\n"
        "- Each question must include at least one concrete entity, indicator, organization, location, policy, or time period from the context.\n"
        "- Prefer answerable, evidence-grounded questions with explicit anchors (for example: entity + metric, or entity + year/time period).\n"
        "- Avoid broad or generic prompts with no concrete anchors (for example: 'main relationships', 'role in...', 'how important...').\n"
        "- Do NOT ask metadata/index questions about page numbers, chapter/annex labels, section numbering, ISSN, or document title lookup.\n"
        "- Do NOT ask 'What is the title of the document?' or similar bibliographic questions.\n"
        "- Do NOT generate questions about general world knowledge NOT present in this document (e.g., capital cities, common definitions found in any encyclopedia).\n"
        "- Do NOT generate questions about GraphRAG, Knowledge Graphs, RAG systems, or evaluation methodology — questions must be about the CONTENT of the documents.\n"
        "- Do NOT use placeholder names like 'Organization A', 'Company B', 'Indicator A' — use the actual names from the context.\n"
        "- Use concise, specific wording.\n"
        "- Vary wording to avoid near-duplicates.\n"
        f"- Write every question in {language_label}.\n"
        "- expected_entities MUST be NAMED ENTITIES from the document text: organization names, regulation names, specific indicators, country/region names, person names, or domain-specific technical terms.\n"
        "- Do NOT use common nouns (e.g., 'feed', 'production', 'transport', 'author') or bare numbers/years (e.g., '2023', '109') as the ONLY expected_entities.\n"
        "- Every expected_entity must appear verbatim or near-verbatim in the context.\n"
        "- source_doc must be the provided document label.\n"
        "- source_docs must be null for non-cross_doc items.\n"
        "- Return only a JSON object with key 'questions'.\n\n"
        f"Output schema example:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Context:\n{context}"
    )


def _format_cross_doc_prompt(
    contexts: dict[str, str],
    count: int,
    question_language: str,
) -> str:
    """User message asking for questions spanning several documents."""
    language_label = _question_language_label(question_language)
    schema = {
        "questions": [
            {
                "question": "...",
                "type": "cross_doc",
                "source_doc": None,
                "source_docs": ["doc1.txt", "doc2.md"],
                "expected_entities": ["..."],
            }
        ]
    }
    return (
        "Create cross-document GraphRAG evaluation questions.\n"
        f"Requested count: {count}\n"
        f"Required language: {language_label}\n"
        "Rules:\n"
        "- Every question must require information from more than one document.\n"
        "- Each question must include concrete entities or measurable anchors from the supplied summaries.\n"
        "- Do NOT ask metadata/index questions about page numbers, chapter/annex labels, section numbering, ISSN, or document title lookup.\n"
        "- Do NOT ask 'What is the title of the document?' or similar bibliographic questions.\n"
        "- Do NOT generate questions about general world knowledge not specific to these documents.\n"
        "- source_doc must be null.\n"
        "- source_docs must list all contributing documents.\n"
        f"- Write every question in {language_label}.\n"
        "- expected_entities MUST be NAMED ENTITIES (organization names, regulation names, specific indicators, country/region names, technical terms). Do NOT use bare numbers, years, or common nouns as the only entities.\n"
        "- Return only a JSON object with key 'questions'.\n\n"
        f"Output schema example:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Document summaries:\n"
        + "\n\n".join(f"### {name}\n{summary}" for name, summary in contexts.items())
    )


def _format_ground_truth_prompt(
    question: str,
    context: str,
    expected_entities: list[str],
    question_language: str,
) -> str:
    """User message asking for one question's ground-truth answer."""
    language_label = _question_language_label(question_language)
    return (
        "Write the ground truth answer for this GraphRAG question.\n"
        "Rules:\n"
        "- Answer only from the supplied context.\n"
        "- Be factual and concise.\n"
        f"- Write in {language_label}.\n"
        "- Use at most 3 sentences.\n"
        "- Do not add any external inference.\n"
        "- Return only JSON with key 'ground_truth'.\n\n"
        f"Question: {question}\n"
        f"Expected entities: {json.dumps(expected_entities, ensure_ascii=False)}\n\n"
        f"Context:\n{context}"
    )


def _tokenize_for_jaccard(text: str) -> frozenset[str]:
    """Return normalized token set for Jaccard similarity."""
    return frozenset(t for t in _normalize_text(text).split() if len(t) > 2)


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets; 1 when both are empty."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _deduplicate_near_questions(
    questions: list[dict[str, Any]],
    threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Remove near-duplicate questions using token Jaccard similarity.

    Keeps the first occurrence when similarity ≥ threshold.
    """
    kept: list[dict[str, Any]] = []
    kept_tokens: list[frozenset[str]] = []
    for item in questions:
        tokens = _tokenize_for_jaccard(str(item.get("question", "")))
        if any(_jaccard_similarity(tokens, t) >= threshold for t in kept_tokens):
            continue
        kept.append(item)
        kept_tokens.append(tokens)
    return kept


_INTERROGATIVE_RE = re.compile(
    r"^(what|who|when|where|which|how|why|does|did|is|are|was|were|can|could|"
    r"cosa|come|quando|dove|quale|quali|perché|chi)\b",
    re.IGNORECASE,
)

def _is_self_answering(question: str, expected_entities: list[str]) -> bool:
    """Return True if the question text embeds its own answer.

    Only flags declarative statements (no question mark, no interrogative
    word anywhere) where an expected entity appears verbatim in the text.
    Phrases like "According to X, what..." or "Compared to Y, how..." are
    real questions (contain interrogative words + end with ?) and are kept.
    """
    if not expected_entities:
        return False
    q_stripped = question.strip()
    # Any sentence ending with "?" is treated as a question.
    if q_stripped.endswith("?"):
        return False
    # Also pass if any interrogative word appears anywhere in the text.
    if _INTERROGATIVE_RE.search(q_stripped):
        return False
    # Pure declarative statement that contains an expected entity → self-answering.
    q_tokens = set(_normalize_text(q_stripped).split())
    for ent in expected_entities:
        ent_tokens = set(_normalize_text(str(ent)).split())
        if ent_tokens and ent_tokens.issubset(q_tokens):
            return True
    return False


_PURE_NUMBER_RE = re.compile(r"^\d+$")
_CHAPTER_NUM_RE = re.compile(r"^\d+(\.\d+)+$")
_GENERIC_WORDS = frozenset({
    "feed", "food", "land", "fish", "water", "milk", "meat", "rice", "crop",
    "farm", "data", "author", "document", "chapter", "section", "production",
    "manufacture", "transport", "distribution", "context", "report", "text",
    "figure", "table", "source", "result", "value", "level", "area", "rate",
})


def _is_quality_entity(entity: str) -> bool:
    """Return True if entity is a named entity useful for KG node seeding.

    Rejects: pure numbers/years, chapter-style decimals (5.1), single generic
    lowercase words. Accepts: proper nouns, multi-word phrases, regulation
    identifiers, technical terms with mixed case.
    """
    s = entity.strip()
    if not s:
        return False
    if _PURE_NUMBER_RE.fullmatch(s):
        return False
    if _CHAPTER_NUM_RE.fullmatch(s):
        return False
    tokens = s.split()
    if len(tokens) == 1:
        word = tokens[0]
        if word.lower() in _GENERIC_WORDS:
            return False
        # single short all-lowercase token with no digits → likely generic
        if word.islower() and len(word) < 6 and not any(c.isdigit() for c in word):
            return False
    return True


def _validate_question_type(
    question_type: str,
    question: str,
    expected_entities: list[str],
) -> str:
    """Heuristically validate and downgrade implausible question type labels.

    Does NOT discard questions — only adjusts the type when clear evidence
    contradicts the LLM-assigned label. Falls back to 'fact_based'.
    """
    if question_type == "multi_hop" and len(expected_entities) < 2:
        return "fact_based"
    if question_type == "comparative" and not _COMPARATIVE_MARKERS_RE.search(question):
        return "fact_based"
    if question_type == "aggregation" and not _AGGREGATION_MARKERS_RE.search(question):
        return "fact_based"
    return question_type


def _normalize_question_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Validate one generated question and bring it to the suite's shape.

    Args:
        item: One entry of the model's ``questions`` list.

    Returns:
        The cleaned item, with an implausible type downgraded, or ``None``
        when it is unusable: unknown type, metadata question, missing source,
        or no entity retrieval can seed on.
    """
    question = str(item.get("question", "")).strip()
    question_type = str(item.get("type", "")).strip()
    if not question or question_type not in QUESTION_TYPES:
        return None

    if _is_metadata_question(question):
        return None

    if question_type == "cross_doc":
        source_doc = None
        source_docs = item.get("source_docs")
        if not isinstance(source_docs, list):
            return None
        source_docs = [str(doc).strip() for doc in source_docs if str(doc).strip()]
        if not source_docs:
            return None
    else:
        source_doc = str(item.get("source_doc", "")).strip() or None
        if not source_doc:
            return None
        source_docs = None

    entities = item.get("expected_entities", [])
    if not isinstance(entities, list):
        entities = []
    expected_entities = [str(entity).strip() for entity in entities if str(entity).strip()]

    # Require at least one entity that KG retrieval can seed on.
    # Questions with only years, chapter numbers, or generic words will
    # consistently return zero triples, making them useless for KG evaluation.
    if expected_entities and not any(_is_quality_entity(e) for e in expected_entities):
        return None

    # Light heuristic type sanity checks: downgrade implausible type labels
    # rather than discarding the question entirely.
    question_type = _validate_question_type(question_type, question, expected_entities)

    return {
        "question": question,
        "type": question_type,
        "source_doc": source_doc,
        "source_docs": source_docs,
        "expected_entities": expected_entities,
    }


def _extract_question_items(payload: Any) -> list[dict[str, Any]]:
    """The valid, normalised questions of a model reply."""
    if isinstance(payload, dict):
        items = payload.get("questions", [])
    else:
        items = payload
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_item = _normalize_question_item(item)
        if normalized_item is not None:
            normalized.append(normalized_item)
    return normalized


def _generate_doc_questions(
    *,
    vllm: VLLMConfig,
    doc: DocumentRecord,
    doc_chunks: list[ChunkRecord],
    question_counts: dict[str, int],
    question_language: str,
    temperature: float = 0.0,
    seed_offset: int = 0,
    failure_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Ask for questions about one document.

    Args:
        vllm: Endpoint settings.
        doc: The document.
        doc_chunks: Its chunks.
        question_counts: Questions wanted per type.
        question_language: Language code of the questions.
        temperature: Sampling temperature.
        seed_offset: Shifts which chunks are sampled into the context.
        failure_log: Receives a record for every failed or empty call.

    Returns:
        The valid non-cross-document questions; empty on failure.
    """
    total_requested = sum(question_counts.values())
    if total_requested <= 0:
        return []

    context = _build_doc_context(doc, doc_chunks, seed_offset=seed_offset)
    prompt = _format_question_prompt(
        context=context,
        question_counts=question_counts,
        source_label=doc.filename,
        question_language=question_language,
    )
    try:
        payload = _call_json_llm(
            vllm,
            _question_generation_system_prompt(question_language),
            prompt,
            temperature=temperature,
            guided_schema=_QUESTIONS_OUTPUT_SCHEMA,
        )
    except Exception as exc:
        reason = str(exc)
        LOGGER.warning("Question generation failed for doc=%s: %s", doc.filename, reason)
        if failure_log is not None:
            failure_log.append({"doc": doc.filename, "reason": reason})
        return []
    items = [item for item in _extract_question_items(payload) if item["type"] in NON_CROSS_TYPES]
    if not items:
        LOGGER.warning("No valid questions produced for doc=%s", doc.filename)
        if failure_log is not None:
            failure_log.append({"doc": doc.filename, "reason": "no_valid_items"})
    for item in items:
        item["source_doc"] = doc.filename
        item["source_docs"] = None
    return items


def _generate_cross_doc_questions(
    *,
    vllm: VLLMConfig,
    docs: list[DocumentRecord],
    chunks_by_doc: dict[str, list[ChunkRecord]],
    count: int,
    question_language: str,
    temperature: float = 0.0,
    failure_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Ask for questions that need more than one document.

    Args:
        vllm: Endpoint settings.
        docs: The documents.
        chunks_by_doc: Chunks per doc_id.
        count: Questions wanted.
        question_language: Language code of the questions.
        temperature: Sampling temperature.
        failure_log: Receives a record for every failed or empty call.

    Returns:
        The valid cross-document questions; empty on failure or with fewer
        than two documents.
    """
    if count <= 0 or len(docs) < 2:
        return []

    summaries = {
        doc.filename: _build_doc_summary(doc, chunks_by_doc[doc.doc_id])
        for doc in docs
    }
    prompt = _format_cross_doc_prompt(summaries, count, question_language)
    try:
        payload = _call_json_llm(
            vllm,
            _question_generation_system_prompt(question_language),
            prompt,
            temperature=temperature,
            guided_schema=_QUESTIONS_OUTPUT_SCHEMA,
        )
    except Exception as exc:
        reason = str(exc)
        LOGGER.warning("Cross-doc question generation failed: %s", reason)
        if failure_log is not None:
            failure_log.append({"doc": "cross_doc", "reason": reason})
        return []
    items = [item for item in _extract_question_items(payload) if item["type"] == "cross_doc"]
    if not items:
        LOGGER.warning("No valid cross-doc questions produced")
        if failure_log is not None:
            failure_log.append({"doc": "cross_doc", "reason": "no_valid_items"})
    for item in items:
        item["source_doc"] = None
    return items


def _generate_ground_truths(
    *,
    vllm: VLLMConfig,
    questions: list[dict[str, Any]],
    docs: list[DocumentRecord],
    chunks_by_doc: dict[str, list[ChunkRecord]],
    summaries_by_doc: dict[str, str],
    question_language: str,
) -> None:
    """Write a ``ground_truth`` answer into every question, in place.

    A question whose source cannot be resolved, or whose generation fails,
    gets an empty answer.

    Args:
        vllm: Endpoint settings.
        questions: The questions to answer.
        docs: The documents.
        chunks_by_doc: Chunks per doc_id.
        summaries_by_doc: Cross-document summaries per filename.
        question_language: Language code of the answers.
    """
    doc_by_name = {doc.filename: doc for doc in docs}
    for item in questions:
        context = ""
        if item.get("type") == "cross_doc":
            source_docs = [
                str(name).strip()
                for name in (item.get("source_docs") or [])
                if str(name).strip() in summaries_by_doc
            ]
            if not source_docs:
                LOGGER.warning("Skipping GT generation for cross_doc question with invalid source_docs")
                item["ground_truth"] = ""
                continue
            item["source_docs"] = source_docs
            context = "\n\n".join(f"### {name}\n{summaries_by_doc[name]}" for name in source_docs)
        else:
            doc_name = str(item.get("source_doc", ""))
            doc = doc_by_name.get(doc_name)
            if doc is None:
                LOGGER.warning("Skipping GT generation for question with unknown source_doc=%s", doc_name)
                item["ground_truth"] = ""
                continue
            doc_chunks = chunks_by_doc.get(doc.doc_id)
            if not doc_chunks:
                LOGGER.warning("Skipping GT generation for source_doc=%s due to missing chunks", doc_name)
                item["ground_truth"] = ""
                continue
            context = _build_doc_context(doc, doc_chunks)

        if not context.strip():
            item["ground_truth"] = ""
            continue

        prompt = _format_ground_truth_prompt(
            question=str(item["question"]),
            context=context,
            expected_entities=list(item.get("expected_entities", [])),
            question_language=question_language,
        )
        try:
            payload = _call_json_llm(
                vllm,
                _ground_truth_system_prompt(question_language),
                prompt,
                guided_schema=_GROUND_TRUTH_OUTPUT_SCHEMA,
            )
        except Exception as exc:
            LOGGER.warning("Ground-truth generation failed for question '%s': %s", item.get("question", ""), exc)
            item["ground_truth"] = ""
            continue
        if isinstance(payload, dict):
            value = str(payload.get("ground_truth", "")).strip()
        else:
            value = ""
        item["ground_truth"] = value


def _build_question_set(
    *,
    vllm: VLLMConfig,
    docs: list[DocumentRecord],
    chunks: list[ChunkRecord],
    run_dir: Path,
    include_ground_truth: bool,
    verbose: bool,
    question_language: str,
) -> dict[str, Any]:
    """Generate, filter and refill the suite until the target counts are met.

    Args:
        vllm: Endpoint settings.
        docs: The documents.
        chunks: Their chunks.
        run_dir: Source run, recorded in the metadata.
        include_ground_truth: Generate answers and keep only grounded
            questions.
        verbose: Print every accepted question.
        question_language: Language code of questions and answers.

    Returns:
        The suite: ``metadata`` and the numbered ``questions``.
    """
    chunks_by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.doc_id].append(chunk)
    for chunk_list in chunks_by_doc.values():
        chunk_list.sort(key=lambda item: item.chunk_index)

    summaries_by_doc = {
        doc.filename: _build_doc_summary(doc, chunks_by_doc[doc.doc_id]) for doc in docs
    }

    remaining = dict(TARGET_COUNTS)
    remaining.pop("cross_doc", None)
    questions: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    generation_failures: list[dict[str, Any]] = []

    doc_totals = _distributed_counts(sum(remaining.values()), len(docs))
    for doc_index, doc in enumerate(docs):
        doc_target = doc_totals[doc_index]
        if doc_target <= 0:
            continue
        allocation = _split_type_counts(doc_target, remaining, NON_CROSS_TYPES)
        if sum(allocation.values()) <= 0:
            continue

        generated = _generate_doc_questions(
            vllm=vllm,
            doc=doc,
            doc_chunks=chunks_by_doc[doc.doc_id],
            question_counts=allocation,
            question_language=question_language,
            failure_log=generation_failures,
        )
        for item in generated:
            normalized = _normalize_text(item["question"])
            if normalized in seen_questions:
                continue
            question_type = item["type"]
            if remaining.get(question_type, 0) <= 0:
                continue
            seen_questions.add(normalized)
            remaining[question_type] -= 1
            questions.append(item)
            if verbose:
                print(f"[{question_type}] {item['question']}")

    refill_order = sorted(docs, key=lambda doc: len(chunks_by_doc[doc.doc_id]), reverse=True)
    while sum(remaining.values()) > 0:
        progress_made = False
        for doc in refill_order:
            if sum(remaining.values()) <= 0:
                break
            target = min(2, sum(remaining.values()))
            allocation = _split_type_counts(target, remaining, NON_CROSS_TYPES)
            if sum(allocation.values()) <= 0:
                continue
            # Use non-zero temperature so repeated refill calls don't produce
            # identical questions that the dedup filter would discard.
            generated = _generate_doc_questions(
                vllm=vllm,
                doc=doc,
                doc_chunks=chunks_by_doc[doc.doc_id],
                question_counts=allocation,
                question_language=question_language,
                temperature=0.7,
                failure_log=generation_failures,
            )
            for item in generated:
                normalized = _normalize_text(item["question"])
                if normalized in seen_questions:
                    continue
                question_type = item["type"]
                if remaining.get(question_type, 0) <= 0:
                    continue
                seen_questions.add(normalized)
                remaining[question_type] -= 1
                questions.append(item)
                progress_made = True
                if verbose:
                    print(f"[{question_type}] {item['question']}")
        if not progress_made:
            break

    if len(docs) >= 2 and TARGET_COUNTS["cross_doc"] > 0:
        generated = _generate_cross_doc_questions(
            vllm=vllm,
            docs=docs,
            chunks_by_doc=chunks_by_doc,
            count=TARGET_COUNTS["cross_doc"],
            question_language=question_language,
            failure_log=generation_failures,
        )
        for item in generated:
            normalized = _normalize_text(item["question"])
            if normalized in seen_questions:
                continue
            seen_questions.add(normalized)
            questions.append(item)
            if verbose:
                print(f"[cross_doc] {item['question']}")

    if include_ground_truth:
        _generate_ground_truths(
            vllm=vllm,
            questions=questions,
            docs=docs,
            chunks_by_doc=chunks_by_doc,
            summaries_by_doc=summaries_by_doc,
            question_language=question_language,
        )
    else:
        for item in questions:
            item["ground_truth"] = ""

    def _validate_grounding(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the grounded questions.

        A question stays when its sources exist and, with ground truth on, its
        answer is not a refusal and contains every word of some expected entity.
        """
        doc_by_name = {doc.filename: doc for doc in docs}
        kept: list[dict[str, Any]] = []
        removed_count = 0
        for item in items:
            question_type = str(item.get("type", "")).strip()
            if question_type == "cross_doc":
                source_docs = [
                    str(name).strip()
                    for name in (item.get("source_docs") or [])
                    if str(name).strip() in summaries_by_doc
                ]
                if not source_docs:
                    removed_count += 1
                    continue
                item["source_docs"] = source_docs
                item["source_doc"] = None
            else:
                source_doc = str(item.get("source_doc", "")).strip()
                if source_doc not in doc_by_name:
                    removed_count += 1
                    continue
                item["source_doc"] = source_doc
                item["source_docs"] = None

            if not include_ground_truth:
                kept.append(item)
                continue

            gt = str(item.get("ground_truth", "")).strip()
            expected = list(item.get("expected_entities", []) or [])

            if not gt or _looks_like_refusal(gt):
                removed_count += 1
                continue

            if expected:
                gt_tokens = set(QUESTION_TOKEN_RE.sub(" ", _normalize_text(gt)).split())
                grounded = False
                for ent in expected:
                    ent_norm = _normalize_text(str(ent))
                    if not ent_norm:
                        continue
                    ent_tokens = set(ent_norm.split())
                    # Require all tokens of the entity to appear in the ground
                    # truth (whole-word match), avoiding short-string false
                    # positives from substring search.
                    if ent_tokens and ent_tokens.issubset(gt_tokens):
                        grounded = True
                        break

                if not grounded:
                    removed_count += 1
                    continue

            kept.append(item)

        if removed_count and getattr(LOGGER, "warning", None):
            LOGGER.warning("Filtered out %d generated questions due to missing grounding", removed_count)
        return kept

    def _missing_targets(current: list[dict[str, Any]]) -> dict[str, int]:
        """Questions still missing per type."""
        counts = Counter(str(item.get("type", "")) for item in current)
        missing = {
            question_type: max(0, TARGET_COUNTS[question_type] - int(counts.get(question_type, 0)))
            for question_type in QUESTION_TYPES
        }
        if len(docs) < 2:
            missing["cross_doc"] = 0
        return missing

    questions = _validate_grounding(questions)

    max_refill_rounds = 8
    for round_idx in range(max_refill_rounds):
        missing = _missing_targets(questions)
        if sum(missing.values()) <= 0:
            break

        refill_batch: list[dict[str, Any]] = []
        non_cross_remaining = {qtype: missing[qtype] for qtype in NON_CROSS_TYPES}

        for doc in refill_order:
            pending = sum(non_cross_remaining.values())
            if pending <= 0:
                break
            allocation = _split_type_counts(min(2, pending), non_cross_remaining, NON_CROSS_TYPES)
            if sum(allocation.values()) <= 0:
                continue

            generated = _generate_doc_questions(
                vllm=vllm,
                doc=doc,
                doc_chunks=chunks_by_doc[doc.doc_id],
                question_counts=allocation,
                question_language=question_language,
                temperature=0.7,
                seed_offset=round_idx + 1,
                failure_log=generation_failures,
            )
            for item in generated:
                normalized = _normalize_text(str(item.get("question", "")))
                question_type = str(item.get("type", ""))
                if not normalized or normalized in seen_questions:
                    continue
                if non_cross_remaining.get(question_type, 0) <= 0:
                    continue
                seen_questions.add(normalized)
                non_cross_remaining[question_type] -= 1
                refill_batch.append(item)

        cross_remaining = int(missing.get("cross_doc", 0))
        if cross_remaining > 0 and len(docs) >= 2:
            generated_cross = _generate_cross_doc_questions(
                vllm=vllm,
                docs=docs,
                chunks_by_doc=chunks_by_doc,
                count=cross_remaining,
                question_language=question_language,
                temperature=0.7,
                failure_log=generation_failures,
            )
            for item in generated_cross:
                if cross_remaining <= 0:
                    break
                normalized = _normalize_text(str(item.get("question", "")))
                if not normalized or normalized in seen_questions:
                    continue
                seen_questions.add(normalized)
                cross_remaining -= 1
                refill_batch.append(item)

        if not refill_batch:
            break

        if include_ground_truth:
            _generate_ground_truths(
                vllm=vllm,
                questions=refill_batch,
                docs=docs,
                chunks_by_doc=chunks_by_doc,
                summaries_by_doc=summaries_by_doc,
                question_language=question_language,
            )
        else:
            for item in refill_batch:
                item["ground_truth"] = ""

        accepted = _validate_grounding(refill_batch)
        if accepted:
            questions.extend(accepted)

        if verbose:
            pending_after_round = sum(_missing_targets(questions).values())
            print(
                f"[refill] round={round_idx + 1} generated={len(refill_batch)} accepted={len(accepted)} pending={pending_after_round}"
            )

    missing_after_refill = _missing_targets(questions)
    if sum(missing_after_refill.values()) > 0:
        LOGGER.warning("Could not fill all target counts after strict grounding: %s", missing_after_refill)

    # Near-duplicate removal (token Jaccard ≥ 0.65) and anti-leakage filter.
    before_dedup = len(questions)
    questions = _deduplicate_near_questions(questions, threshold=0.65)
    leakage_filtered: list[dict[str, Any]] = []
    for item in questions:
        if _is_self_answering(
            str(item.get("question", "")),
            list(item.get("expected_entities", []) or []),
        ):
            LOGGER.warning("Dropped self-answering question: %r", item.get("question", ""))
        else:
            leakage_filtered.append(item)
    questions = leakage_filtered
    if len(questions) < before_dedup:
        LOGGER.warning(
            "Post-processing removed %d questions (near-dup or leakage). Final count: %d",
            before_dedup - len(questions),
            len(questions),
        )

    for index, item in enumerate(questions, start=1):
        item["id"] = f"q_{index:03d}"

    counts = Counter(item["type"] for item in questions)
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_questions": len(questions),
            "docs_used": [doc.filename for doc in docs],
            "doc_chunk_counts": {doc.filename: len(chunks_by_doc[doc.doc_id]) for doc in docs},
            "source_run_dir": str(run_dir.resolve()),
            "model": vllm.model_name,
            "question_language": str(question_language),
            "strict_grounding_enabled": bool(include_ground_truth),
            "target_counts": dict(TARGET_COUNTS),
            "missing_after_refill": missing_after_refill,
            "by_type": {question_type: counts.get(question_type, 0) for question_type in QUESTION_TYPES},
            "generation_failures": generation_failures,
        },
        "questions": questions,
    }


def _stats(path: Path) -> None:
    """Print the type counts, documents covered and provenance of a suite."""
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object")

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    questions = payload.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("questions must be a list")

    counts = Counter()
    docs = set()
    for item in questions:
        if not isinstance(item, dict):
            continue
        counts[str(item.get("type", ""))] += 1
        source_doc = item.get("source_doc")
        if isinstance(source_doc, str) and source_doc.strip():
            docs.add(source_doc.strip())
        source_docs = item.get("source_docs")
        if isinstance(source_docs, list):
            for doc in source_docs:
                if isinstance(doc, str) and doc.strip():
                    docs.add(doc.strip())

    print(f"total_questions={len(questions)}")
    for question_type in QUESTION_TYPES:
        print(f"{question_type}={counts.get(question_type, 0)}")
    print(f"docs_covered={len(docs)}")
    print(f"model={metadata.get('model', '')}")
    print(f"generated_at={metadata.get('generated_at', '')}")


def _build_parser() -> argparse.ArgumentParser:
    """The command-line parser, with ``generate`` and ``stats``."""
    parser = argparse.ArgumentParser(description="GraphRAG question suite generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a test suite JSON")
    generate.add_argument("--run-dir", default="", help="Override the KG run directory to use")
    generate.add_argument("--doc", default=None, help="Generate only for a specific document filename or doc_id")
    generate.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    generate.add_argument(
        "--question-language",
        choices=tuple(sorted(QUESTION_LANGUAGES.keys())),
        default="en",
        help="Language for generated questions and ground-truth answers",
    )
    generate.add_argument(
        "--matrix-output",
        default="",
        help="Optional output path for plain-text matrix questions (one per line)",
    )
    generate.add_argument(
        "--matrix-max-questions",
        type=int,
        default=0,
        help="Optional limit for matrix question export (0 keeps all)",
    )
    generate.add_argument("--no-ground-truth", action="store_true", help="Skip the ground truth generation step")
    generate.add_argument("--verbose", action="store_true", help="Print each question as it is generated")

    stats = subparsers.add_parser("stats", help="Print summary statistics for an existing suite")
    stats.add_argument("--input", required=True, help="Input test suite JSON")

    return parser


def main() -> int:
    """Run the chosen subcommand and return the exit status."""
    args = _build_parser().parse_args()
    _setup_logging(getattr(args, "verbose", False))

    if args.command == "stats":
        _stats(Path(args.input))
        return 0

    run_dir = Path(args.run_dir).expanduser() if args.run_dir else _discover_latest_run_dir(DEFAULT_RUN_ROOT)

    if int(args.matrix_max_questions) < 0:
        raise ValueError("--matrix-max-questions must be >= 0")

    _, documents, chunks = _resolve_docs_and_chunks(run_dir, args.doc)
    vllm = _resolve_vllm_config()
    suite = _build_question_set(
        vllm=vllm,
        docs=documents,
        chunks=chunks,
        run_dir=run_dir,
        include_ground_truth=not args.no_ground_truth,
        verbose=bool(args.verbose),
        question_language=str(args.question_language),
    )

    output_path = Path(args.output).expanduser()
    _save_json(output_path, suite)
    print(f"saved={output_path}")

    if args.matrix_output:
        matrix_questions = _suite_to_matrix_questions(
            suite,
            max_questions=int(args.matrix_max_questions),
        )
        matrix_path = Path(args.matrix_output).expanduser()
        _save_questions_txt(matrix_path, matrix_questions)
        print(f"matrix_saved={matrix_path}")
        print(f"matrix_questions={len(matrix_questions)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())