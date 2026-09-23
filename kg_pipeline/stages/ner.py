"""Stage 2: propose entity candidates in each chunk with GLiNER."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from gliner import GLiNER
from tqdm import tqdm

from kg_pipeline.models.types import ChunkRecord, NEREntityCandidate

LOGGER = logging.getLogger("kg_pipeline")

# Chunks per forward pass, overridden by KG_NER_BATCH_SIZE. 16 keeps peak
# activation memory well inside an A40; 1 processes one chunk at a time.
_DEFAULT_NER_BATCH_SIZE = 16


def run_ner(
    chunks: list[ChunkRecord],
    model_name: str,
    labels: list[str],
    threshold: float,
    batch_size: int | None = None,
) -> dict[str, list[NEREntityCandidate]]:
    """Run GLiNER over every chunk.

    The model is moved to ``KG_NER_DEVICE`` when that variable is set. Labels
    are passed to the model lower-cased and mapped back to their configured
    spelling in the output.

    Args:
        chunks: Chunks to annotate.
        model_name: GLiNER model name or path.
        labels: Ontology labels to detect.
        threshold: Minimum span score.
        batch_size: Chunks per forward pass. ``None`` reads
            ``KG_NER_BATCH_SIZE``, falling back to ``_DEFAULT_NER_BATCH_SIZE``.

    Returns:
        Mapping from ``chunk_id`` to the candidates found in that chunk.
    """
    model = GLiNER.from_pretrained(model_name)
    device = os.environ.get("KG_NER_DEVICE", "").strip()
    if device:
        model = model.to(device)

    if batch_size is None:
        try:
            batch_size = int(os.environ.get("KG_NER_BATCH_SIZE", str(_DEFAULT_NER_BATCH_SIZE)))
        except ValueError:
            batch_size = _DEFAULT_NER_BATCH_SIZE
    batch_size = max(1, batch_size)

    label_map = {label.lower(): label for label in labels}
    model_labels = list(label_map.keys())

    output: dict[str, list[NEREntityCandidate]] = {}

    for batch in tqdm(
        _batched(chunks, batch_size),
        total=(len(chunks) + batch_size - 1) // batch_size,
        desc="Stage 2 GLiNER",
        unit="batch",
    ):
        per_chunk = _predict_batch(
            model, [chunk.text for chunk in batch], model_labels, threshold
        )
        for chunk, raw_entities in zip(batch, per_chunk):
            output[chunk.chunk_id] = [
                _to_candidate(item, label_map) for item in raw_entities
            ]

    return output


def _to_candidate(
    item: dict, label_map: dict[str, str]
) -> NEREntityCandidate:
    """Convert one GLiNER span into a candidate record.

    Args:
        item: Span dict returned by GLiNER.
        label_map: Mapping from lower-cased label to configured label.

    Returns:
        The candidate, with offsets clamped to be non-negative and the score
        clamped to ``[0, 1]``.
    """
    raw_label = str(item.get("label", "")).strip().lower()
    mapped_label = label_map.get(raw_label, item.get("label", "Concept"))

    start_char = int(item.get("start", item.get("start_char", 0)))
    end_char = int(item.get("end", item.get("end_char", 0)))
    score = float(item.get("score", item.get("confidence", 0.0)))

    return NEREntityCandidate(
        text_span=str(item.get("text", item.get("span", ""))).strip(),
        entity_label=str(mapped_label),
        start_char=max(0, start_char),
        end_char=max(0, end_char),
        confidence_score=max(0.0, min(1.0, score)),
    )


def _batched(items: list, size: int):
    """Yield consecutive slices of ``items`` of at most ``size`` elements."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _predict_batch(
    model: GLiNER,
    texts: list[str],
    model_labels: list[str],
    threshold: float,
) -> list[list[dict]]:
    """Run GLiNER on several chunks in one forward pass.

    Uses ``model.inference`` with the same defaults as ``predict_entities``
    (``flat_ner=True``, ``multi_label=False``), so the spans match per-chunk
    prediction. Falls back to one ``predict_entities`` call per chunk when the
    model has no ``inference`` method, when there is a single text, or when
    the batch call fails.

    Args:
        model: Loaded GLiNER model.
        texts: Chunk texts.
        model_labels: Lower-cased labels to detect.
        threshold: Minimum span score.

    Returns:
        One list of span dicts per input text, in input order.
    """
    inference = getattr(model, "inference", None)
    if callable(inference) and len(texts) > 1:
        try:
            return inference(
                texts,
                model_labels,
                flat_ner=True,
                threshold=threshold,
                multi_label=False,
                batch_size=len(texts),
            )
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the batch
            LOGGER.warning(
                "GLiNER batch inference failed (%s); falling back to one chunk "
                "at a time for these %d chunks.",
                exc,
                len(texts),
            )
    return [
        model.predict_entities(text, model_labels, threshold=threshold)
        for text in texts
    ]


def save_ner(path: Path, ner_map: dict[str, list[NEREntityCandidate]]) -> None:
    """Write NER candidates to a JSON file.

    Args:
        path: Output file; parent directories are created.
        ner_map: Mapping from ``chunk_id`` to candidates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        chunk_id: [entity.model_dump() for entity in entities]
        for chunk_id, entities in ner_map.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ner(path: Path) -> dict[str, list[NEREntityCandidate]]:
    """Read NER candidates written by :func:`save_ner`.

    Args:
        path: JSON file to read.

    Returns:
        Mapping from ``chunk_id`` to validated candidates.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        chunk_id: [NEREntityCandidate.model_validate(entity) for entity in entities]
        for chunk_id, entities in payload.items()
    }


def _cli() -> None:
    """Run stage 2 standalone from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    chunks_payload = json.loads(Path(args.chunks_json).read_text(encoding="utf-8"))
    chunks = [ChunkRecord.model_validate(item) for item in chunks_payload]
    labels = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))

    # Ensure recommended labels present for improved extraction
    for extra in ("TimePeriod", "DataValue"):
        if extra not in labels:
            labels.append(extra)

    ner_map = run_ner(chunks, args.model_name, labels, args.threshold)
    save_ner(Path(args.output_json), ner_map)


if __name__ == "__main__":
    _cli()
