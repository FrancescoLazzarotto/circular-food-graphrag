"""Multilingual sentence encoder shared by the KG vector index and the retriever.

The graph was extracted from a bilingual corpus and many concepts carry only
their Italian surface form, while questions are often English; lexical
retrieval cannot cross that gap. Encoding node names and questions into one
multilingual space is the bridge, so index build time and query time must use
the *same* model, prefixes and pooling — a mismatch degrades similarity
silently instead of failing. Hence a single module, and a single transport.

That transport is an OpenAI-compatible ``/v1/embeddings`` endpoint, the same
pattern used for generation::

    CUDA_VISIBLE_DEVICES=1 vllm serve intfloat/multilingual-e5-base \\
        --runner pooling --port 8002 --gpu-memory-utilization 0.12 \\
        --max-model-len 512

The encoder does not run in-process because the ``graphllm`` environment
pairs torch 2.5.1 with torchvision 0.25.0, so importing any ``transformers``
text model fails with ``operator torchvision::nms does not exist``. Serving it
from the ``vllm-serve`` virtualenv keeps both environments untouched.

The e5 family is trained with ``query:`` / ``passage:`` prefixes; they are not
optional and are applied here so no caller can forget one.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence

import requests

logger = logging.getLogger("graphrag")

# The endpoint is local, so a failed request is almost always transient (server
# still warming, a queue rejection, a dropped connection). Retrying keeps a
# transient failure from silently removing the vector channel from one query.
_RETRY_ATTEMPTS = int(os.getenv("GRAPHRAG_EMBED_RETRIES", "3") or 3)
_RETRY_BACKOFF_SEC = float(os.getenv("GRAPHRAG_EMBED_RETRY_BACKOFF_SEC", "0.5") or 0.5)

# The e5 family stops at 512 tokens and the server answers 400 — not a truncated
# embedding — for anything longer. Truncating here removes that failure mode
# entirely: a slightly shortened vector is worth incomparably more than no vector
# at all, which is what the caller gets from a raised error. ~3.6 chars/token is
# conservative for mixed Italian/English text, and the prefix is counted too.
_MAX_INPUT_CHARS = int(os.getenv("GRAPHRAG_EMBED_MAX_CHARS", "1700") or 1700)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BASE_URL = "http://localhost:8002/v1"


class EmbeddingUnavailable(RuntimeError):
    """The embedding endpoint could not be reached or returned an error."""


def base_url() -> str:
    """Endpoint from ``GRAPHRAG_EMBED_BASE_URL``, else the local default."""
    return os.getenv("GRAPHRAG_EMBED_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def model_id() -> str:
    """Encoder id from ``GRAPHRAG_EMBED_MODEL``, else the e5 default."""
    return os.getenv("GRAPHRAG_EMBED_MODEL", DEFAULT_MODEL)


def _truncate(text: str) -> str:
    """Clip one prefixed input to the encoder's context window.

    Args:
        text: Text already carrying its ``query:``/``passage:`` prefix.

    Returns:
        The text, shortened at a word boundary when it exceeded the budget.
    """
    if len(text) <= _MAX_INPUT_CHARS:
        return text
    clipped = text[:_MAX_INPUT_CHARS]
    cut = clipped.rfind(" ")
    if cut > _MAX_INPUT_CHARS // 2:
        clipped = clipped[:cut]
    logger.warning(
        "embedding input truncated from %d to %d chars to fit the encoder window",
        len(text),
        len(clipped),
    )
    return clipped


def _post_batch(
    endpoint: str, name: str, batch: list[str], timeout: float
) -> dict:
    """POST one batch to the embeddings endpoint, retrying transient failures.

    Args:
        endpoint: Full ``/embeddings`` URL.
        name: Model id to request.
        batch: Prefixed texts.
        timeout: Per-request timeout in seconds.

    Returns:
        The decoded JSON payload.

    Raises:
        EmbeddingUnavailable: Every attempt failed. The message carries the
            server's own error body, so the failure can be diagnosed after the
            fact.
    """
    last_error = ""
    for attempt in range(1, max(1, _RETRY_ATTEMPTS) + 1):
        body = ""
        try:
            response = requests.post(
                endpoint,
                json={"model": name, "input": batch},
                timeout=timeout,
            )
            body = (response.text or "")[:500]
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = f"{exc}" + (f" | body: {body}" if body else "")
            if attempt < max(1, _RETRY_ATTEMPTS):
                logger.warning(
                    "embedding request failed (attempt %d/%d): %s — retrying",
                    attempt,
                    _RETRY_ATTEMPTS,
                    last_error,
                )
                time.sleep(_RETRY_BACKOFF_SEC * attempt)
    raise EmbeddingUnavailable(f"{endpoint}: {last_error}")


def encode(
    texts: Sequence[str],
    prefix: str,
    model: str | None = None,
    url: str | None = None,
    batch_size: int = 256,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Embed ``texts``, in input order.

    Args:
        texts: Strings to encode.
        prefix: :data:`QUERY_PREFIX` or :data:`PASSAGE_PREFIX`.
        model: Encoder id; must match the one the index was built with.
        url: Base URL of the OpenAI-compatible endpoint.
        batch_size: Texts per HTTP request.
        timeout: Per-request timeout in seconds.

    Returns:
        One vector per input.

    Raises:
        EmbeddingUnavailable: The endpoint is unreachable or replied with an
            error, so the caller can fall back to lexical retrieval instead of
            silently returning nothing.
    """
    if not texts:
        return []

    endpoint = f"{url or base_url()}/embeddings"
    name = model or model_id()
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [
            _truncate(f"{prefix}{text}")
            for text in texts[start : start + batch_size]
        ]
        payload = _post_batch(endpoint, name, batch, timeout)

        rows = payload.get("data") or []
        if len(rows) != len(batch):
            raise EmbeddingUnavailable(
                f"{endpoint}: asked for {len(batch)} embeddings, got {len(rows)}"
            )
        # The API is not required to preserve order, and silently mispairing a
        # vector with its node would poison the index in a way no later check
        # would catch.
        for row in sorted(rows, key=lambda item: item.get("index", 0)):
            out.append([float(value) for value in row["embedding"]])
    return out


def encode_query(
    text: str, model: str | None = None, url: str | None = None
) -> list[float]:
    """Embed one question with the query prefix.

    Args:
        text: The question.
        model: Encoder id; defaults to :func:`model_id`.
        url: Endpoint base URL; defaults to :func:`base_url`.

    Returns:
        The vector.

    Raises:
        EmbeddingUnavailable: The endpoint is unreachable or replied with an
            error.
    """
    vectors = encode([text], QUERY_PREFIX, model=model, url=url)
    return vectors[0] if vectors else []


def available(url: str | None = None, timeout: float = 5.0) -> bool:
    """Whether the endpoint answers, for a cheap pre-flight check.

    Args:
        url: Endpoint base URL; defaults to :func:`base_url`.
        timeout: Request timeout in seconds.

    Returns:
        True when ``GET /models`` returns 200.
    """
    try:
        response = requests.get(f"{url or base_url()}/models", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False
