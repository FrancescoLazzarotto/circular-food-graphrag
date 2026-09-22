from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

from kg_pipeline.models.types import (
    CanonicalEntityRecord,
    ChunkRecord,
    DocumentRecord,
    KGTriple,
    NEREntityCandidate,
)
from kg_pipeline.stages import (
    chunking,
    ingestion,
    linking,
    llm_extraction,
    neo4j_ingestion,
    ner,
    resolution,
)


LOGGER = logging.getLogger("kg_pipeline")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # PYTHONHASHSEED is read by CPython at interpreter startup only, so setting
    # it here does nothing — the pipeline claimed to seed it and did not. Warn
    # instead of pretending, and export it before launching if it matters. See
    # docs/code_audit_2026-08-15.md §3.8.
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        LOGGER.warning(
            "PYTHONHASHSEED is %r, not %r: it can only be set before the "
            "interpreter starts. Export PYTHONHASHSEED=%d before launching if "
            "set-iteration order must be reproducible.",
            os.environ.get("PYTHONHASHSEED"),
            str(seed),
            seed,
        )
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        LOGGER.warning(
            "torch not installed: GLiNER/SentenceTransformer outputs will not be seeded"
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_relation_vocab(config: dict[str, Any], config_path: Path) -> list[str] | None:
    rel_path = str(config.get("llm", {}).get("relation_vocab_path", "")).strip()
    if not rel_path:
        return None

    vocab_path = Path(rel_path)
    if not vocab_path.is_absolute():
        vocab_path = config_path.parent / vocab_path
    if not vocab_path.exists():
        raise FileNotFoundError(f"relation vocab not found: {vocab_path}")
    payload = _load_json(vocab_path)
    if not isinstance(payload, list):
        raise ValueError("relation vocab must be a JSON array")
    return [str(item).strip().upper() for item in payload if str(item).strip()]


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _write_run_metadata(
    run_dir: Path, config_path: Path, config: dict[str, Any], seed: int
) -> None:
    """Snapshot config, relation vocab and run metadata so a run is traceable."""
    config_snapshot = run_dir / "config.yaml"
    if config_path.resolve() != config_snapshot.resolve():
        shutil.copy2(config_path, config_snapshot)

    rel_path = str(config.get("llm", {}).get("relation_vocab_path", "")).strip()
    if rel_path:
        vocab_path = Path(rel_path)
        if not vocab_path.is_absolute():
            vocab_path = config_path.parent / vocab_path
        if vocab_path.exists():
            shutil.copy2(vocab_path, run_dir / vocab_path.name)

    # `started_at` used to be rewritten on every invocation, so a run whose
    # extraction finished on the 7th claimed to have started on the 9th — the
    # day someone resumed it to run a later stage. The first start is the one
    # that dates the artifacts; the rest are resumes, and get their own field.
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    previous = _load_json(run_dir / "run_metadata.json") if (run_dir / "run_metadata.json").exists() else None
    started_at = now
    invocations = 1
    if isinstance(previous, dict):
        started_at = str(previous.get("started_at") or now)
        invocations = int(previous.get("invocations") or 1) + 1

    metadata = {
        "seed": seed,
        "config_path": str(config_path.resolve()),
        "started_at": started_at,
        "last_run_at": now,
        "invocations": invocations,
        "git_commit": _git_commit_hash(),
        "gliner_model": config.get("gliner", {}).get("model_name"),
        "resolution_embedding_model": config.get("resolution", {}).get(
            "embedding_model"
        ),
        "vllm_model": os.getenv("VLLM_MODEL_NAME", ""),
        "vllm_base_url": os.getenv("VLLM_BASE_URL", ""),
    }
    _save_json(run_dir / "run_metadata.json", metadata)


def _log_versions(config: dict[str, Any]) -> None:
    pkg_names = [
        "pymupdf4llm",
        "gliner",
        "openai",
        "sentence-transformers",
        "neo4j",
        "pydantic",
        "tqdm",
    ]
    versions = {}
    for pkg in pkg_names:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"

    LOGGER.info("Dependency versions: %s", versions)
    LOGGER.info(
        "Configured vLLM model=%s base_url=%s",
        os.getenv("VLLM_MODEL_NAME", ""),
        os.getenv("VLLM_BASE_URL", ""),
    )


_FINGERPRINTS_FILE = "stage_fingerprints.json"


def _fingerprint(*parts: Any) -> str:
    """A short, stable digest of whatever a stage's output depends on."""
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _corpus_fingerprint(input_dir: Path, single_doc: str | None) -> str:
    """Which PDFs stage 0 would read, by name and size."""
    try:
        files = sorted(
            (path.relative_to(input_dir).as_posix(), path.stat().st_size)
            for path in ingestion.discover_pdfs(input_dir, warn=False)
        )
    except OSError:
        files = []
    return _fingerprint("stage0", str(input_dir), single_doc or "", files)


def _load_fingerprints(run_dir: Path) -> dict[str, str]:
    path = run_dir / _FINGERPRINTS_FILE
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _record_fingerprint(run_dir: Path, stage: str, value: str) -> None:
    recorded = _load_fingerprints(run_dir)
    recorded[stage] = value
    _save_json(run_dir / _FINGERPRINTS_FILE, recorded)


def _check_fingerprint(run_dir: Path, stage: str, expected: str, artifact: Path) -> None:
    """Refuse to reuse an artifact that was not produced from these inputs.

    Resume was keyed on the file existing, with nothing tying it to the settings
    and the upstream artifacts that produced it. Re-running stage 1 with a
    different window left stage 3 resuming against chunk indices that no longer
    meant the same thing, and the run reported success.

    A mismatch stops the run rather than silently redoing the work or silently
    skipping it. Stage 3 is seven hours: quietly repeating it is as surprising
    as quietly skipping it, and the operator is the one who knows which they
    want.

    Raises:
        SystemExit: if the artifact exists but was made from different inputs.
    """
    recorded = _load_fingerprints(run_dir).get(stage)
    if recorded is None:
        # An artifact from before this file existed. It cannot be checked, so it
        # is taken at face value once and stamped for next time.
        LOGGER.info(
            "%s has no recorded fingerprint; reusing it unverified and "
            "stamping it now",
            artifact.name,
        )
        _record_fingerprint(run_dir, stage, expected)
        return
    if recorded == expected:
        return
    raise SystemExit(
        f"{artifact.name} was produced from different inputs than this run asks "
        f"for (recorded {recorded}, current {expected}).\n"
        f"Reusing it would leave the later stages pointing at positions that no "
        f"longer mean the same thing.\n"
        f"Either delete {artifact.name} and every stage after it in {run_dir}, "
        f"or start a new --run-dir."
    )


def _stage_output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "documents": run_dir / "stage0_documents.json",
        "chunks": run_dir / "stage1_chunks.json",
        "ner": run_dir / "stage2_ner.json",
        "triples_raw": run_dir / "stage3_triples_raw.json",
        "acronyms": run_dir / "stage3_acronyms.json",
        "triples_resolved": run_dir / "stage4_triples_resolved.json",
        "registry": run_dir / "stage4_registry.json",
        "merge_cache": run_dir / "stage4_merge_approved.json",
        "triples_linked": run_dir / "stage5_triples_linked.json",
        "failed_chunks": run_dir / "failed_chunks.jsonl",
        "new_labels_log": run_dir / "new_labels.log",
        "neo4j_summary": run_dir / "stage6_neo4j_summary.json",
    }


# What each stage's output actually depends on. Deliberately field lists rather
# than whole config sections: `checkpoint_every` and `batch_size` change how the
# work is done, not what comes out, and putting them in would invalidate a
# seven-hour stage 3 for an operational knob.
_STAGE_INPUTS = {
    "chunks": lambda cfg: cfg.get("chunking", {}),
    "ner": lambda cfg: {
        "gliner": cfg.get("gliner", {}).get("model_name"),
        "threshold": cfg.get("gliner", {}).get("threshold"),
        "labels": cfg.get("ontology", {}).get("labels"),
    },
    "triples_raw": lambda cfg: {
        "labels": cfg.get("ontology", {}).get("labels"),
        "temperature": cfg.get("llm", {}).get("temperature"),
        "structured": cfg.get("llm", {}).get("use_structured_output"),
    },
    "triples_resolved": lambda cfg: cfg.get("resolution", {}),
    "triples_linked": lambda cfg: cfg.get("linking", {}),
}


def _load_or_run_documents(
    paths: dict[str, Path],
    config: dict[str, Any],
    single_doc: str | None,
    run_dir: Path,
) -> tuple[list[DocumentRecord], str]:
    stamp = _corpus_fingerprint(Path(config["paths"]["input_dir"]), single_doc)
    if paths["documents"].exists():
        _check_fingerprint(run_dir, "documents", stamp, paths["documents"])
        return ingestion.load_documents(paths["documents"]), stamp
    docs = ingestion.ingest_documents(
        input_dir=Path(config["paths"]["input_dir"]),
        single_doc=single_doc,
    )
    ingestion.save_documents(paths["documents"], docs)
    _record_fingerprint(run_dir, "documents", stamp)
    return docs, stamp


def _load_or_run_chunks(
    paths: dict[str, Path],
    config: dict[str, Any],
    docs: list[DocumentRecord],
    run_dir: Path,
    upstream: str,
) -> tuple[list[ChunkRecord], str]:
    stamp = _fingerprint(upstream, _STAGE_INPUTS["chunks"](config))
    if paths["chunks"].exists():
        _check_fingerprint(run_dir, "chunks", stamp, paths["chunks"])
        return chunking.load_chunks(paths["chunks"]), stamp
    chunks = chunking.chunk_documents(docs, config)
    chunking.save_chunks(paths["chunks"], chunks)
    _record_fingerprint(run_dir, "chunks", stamp)
    return chunks, stamp


def _load_or_run_ner(
    paths: dict[str, Path],
    config: dict[str, Any],
    chunks: list[ChunkRecord],
    run_dir: Path,
    upstream: str,
) -> tuple[dict[str, list[NEREntityCandidate]], str]:
    stamp = _fingerprint(upstream, _STAGE_INPUTS["ner"](config))
    if paths["ner"].exists():
        _check_fingerprint(run_dir, "ner", stamp, paths["ner"])
        return ner.load_ner(paths["ner"]), stamp
    ner_map = ner.run_ner(
        chunks=chunks,
        model_name=config["gliner"]["model_name"],
        labels=config["ontology"]["labels"],
        threshold=float(config["gliner"]["threshold"]),
        # Optional: falls back to KG_NER_BATCH_SIZE, then to the stage default.
        batch_size=config["gliner"].get("batch_size"),
    )
    ner.save_ner(paths["ner"], ner_map)
    _record_fingerprint(run_dir, "ner", stamp)
    return ner_map, stamp


def _load_or_run_raw_triples(
    paths: dict[str, Path],
    config: dict[str, Any],
    chunks: list[ChunkRecord],
    ner_map: dict[str, list[NEREntityCandidate]],
    seed: int,
    relation_vocab: list[str] | None,
    run_dir: Path,
    upstream: str,
) -> tuple[list[KGTriple], dict[str, str], str]:
    stamp = _fingerprint(
        upstream,
        _STAGE_INPUTS["triples_raw"](config),
        seed,
        relation_vocab,
        os.getenv("VLLM_MODEL_NAME", ""),
    )
    if paths["triples_raw"].exists() and paths["acronyms"].exists():
        _check_fingerprint(run_dir, "triples_raw", stamp, paths["triples_raw"])
        return (
            llm_extraction.load_triples(paths["triples_raw"]),
            llm_extraction.load_acronyms(paths["acronyms"]),
            stamp,
        )

    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("VLLM_MODEL_NAME", "")
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))

    triples, acronym_map = llm_extraction.extract_triples(
        chunks=chunks,
        ner_map=ner_map,
        allowed_labels=config["ontology"]["labels"],
        relation_vocab=relation_vocab,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        max_retries_per_chunk=int(config["llm"]["max_retries_per_chunk"]),
        temperature=float(config["llm"]["temperature"]),
        seed=seed,
        use_structured_output=bool(config["llm"]["use_structured_output"]),
        failed_chunks_path=paths["failed_chunks"],
        new_label_log_path=paths["new_labels_log"],
        checkpoint_every=int(config.get("llm", {}).get("checkpoint_every", 50)),
        batch_size=config.get("llm", {}).get("batch_size"),
    )

    llm_extraction.save_triples(paths["triples_raw"], triples)
    llm_extraction.save_acronyms(paths["acronyms"], acronym_map)
    _record_fingerprint(run_dir, "triples_raw", stamp)
    return triples, acronym_map, stamp


def _load_or_run_resolution(
    paths: dict[str, Path],
    config: dict[str, Any],
    triples: list[KGTriple],
    acronym_map: dict[str, str],
    run_dir: Path,
    upstream: str,
) -> tuple[list[KGTriple], dict[str, CanonicalEntityRecord], str]:
    stamp = _fingerprint(upstream, _STAGE_INPUTS["triples_resolved"](config))
    if paths["triples_resolved"].exists() and paths["registry"].exists():
        _check_fingerprint(run_dir, "triples_resolved", stamp, paths["triples_resolved"])
        return (
            resolution.load_triples(paths["triples_resolved"]),
            resolution.load_registry(paths["registry"]),
            stamp,
        )

    base_url = os.getenv("VLLM_BASE_URL", "")
    model_name = os.getenv("VLLM_MODEL_NAME", "")
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))

    resolved_triples, registry = resolution.resolve_entities(
        triples=triples,
        acronym_map=acronym_map,
        embedding_model=config["resolution"]["embedding_model"],
        similarity_threshold=float(config["resolution"]["similarity_threshold"]),
        context_jaccard_floor=float(config["resolution"]["context_jaccard_floor"]),
        base_url=base_url or None,
        api_key=api_key,
        model_name=model_name or None,
        merge_cache_path=paths["merge_cache"],
    )

    resolution.save_triples(paths["triples_resolved"], resolved_triples)
    resolution.save_registry(paths["registry"], registry)
    _record_fingerprint(run_dir, "triples_resolved", stamp)
    return resolved_triples, registry, stamp


def _load_or_run_linking(
    paths: dict[str, Path],
    resolved_triples: list[KGTriple],
    registry: dict[str, CanonicalEntityRecord],
    documents: list[DocumentRecord],
    config: dict[str, Any],
    run_dir: Path | None = None,
    upstream: str = "",
) -> list[KGTriple]:
    stamp = _fingerprint(upstream, _STAGE_INPUTS["triples_linked"](config))
    if paths["triples_linked"].exists():
        if run_dir is not None:
            _check_fingerprint(run_dir, "triples_linked", stamp, paths["triples_linked"])
        return linking.load_triples(paths["triples_linked"])

    include_mentioned_in = bool(
        config.get("linking", {}).get("include_mentioned_in", True)
    )
    linked = linking.add_cross_document_links(
        triples=resolved_triples,
        registry=registry,
        documents=documents,
        include_mentioned_in=include_mentioned_in,
    )
    linking.save_triples(paths["triples_linked"], linked)
    if run_dir is not None:
        _record_fingerprint(run_dir, "triples_linked", stamp)
    return linked


# The three passes that live outside this pipeline and that the production
# graph is made of. `kg_densify` alone accounts for 39.4 % of the edges, and
# the two index passes are what the full-text and dense retrieval channels run
# on. A rebuild that stops at stage 6 writes a graph with roughly 60 % of the
# edges, no `search_text` and no vectors — and used to say nothing about it.
_POST_PASSES = (
    (
        "densification",
        "scripts/kg/kg_densify.py",
        "39.4 % of the edges in the production graph; needs a served model",
    ),
    (
        "search_text + full-text index",
        "scripts/kg/kg_search_index.py",
        "without it every name lookup falls back to a full scan",
    ),
    (
        "vector index",
        "scripts/kg/kg_vector_index.py",
        "without it the dense channel is empty and `hybrid` loses its second leg",
    ),
)


def _run_post_passes(run: bool, config_path: Path | None = None,
                     env_file: Path | None = None) -> list[str]:
    """Report — and optionally run — the passes that follow stage 6.

    The two index passes are deterministic and take seconds, so `--run-post`
    runs them. Densification is hours of GPU and needs the operator to choose a
    model, so it is always reported as a command, never started behind their
    back.

    ⚠️ `kg_search_index` defaults its `--env-file` to `kg_pipeline/.env`, which
    points at the hosted graph the demo serves, and loads it with
    `override=True`. Running it without forwarding this run's own config and
    env file would rebuild the index on **production** while the pipeline wrote
    to staging. `kg_vector_index` reads the repository `.env` with
    `override=False`, so the environment this process already carries wins
    there.
    """
    done: list[str] = []
    for name, script, why in _POST_PASSES:
        if run and script != "scripts/kg/kg_densify.py":
            LOGGER.info("Running post pass: %s", name)
            command = [sys.executable, script]
            if script.endswith("kg_search_index.py"):
                if config_path:
                    command += ["--config", str(config_path)]
                if env_file:
                    command += ["--env-file", str(env_file)]
            result = subprocess.run(
                command, cwd=str(Path(__file__).resolve().parents[1])
            )
            if result.returncode != 0:
                raise RuntimeError(f"post pass {name} failed with {result.returncode}")
            done.append(name)
            continue
        LOGGER.warning(
            "NOT RUN: %s (%s). Run it with: python %s", name, why, script
        )
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    pipeline_dir = Path(__file__).resolve().parent
    parser.add_argument("--config", default=str(pipeline_dir / "config.yaml"))
    parser.add_argument("--env-file", default=str(pipeline_dir / ".env"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--single-doc", default=None)
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "ingestion",
            "chunking",
            "ner",
            "llm",
            "resolution",
            "linking",
            "neo4j",
        ],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-post",
        action="store_true",
        help="after stage 6, run the index passes that the graph needs (not densification)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = _load_yaml(config_path)
    load_dotenv(args.env_file, override=True)

    if args.run_dir.strip():
        run_dir = Path(args.run_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(config["paths"]["output_dir"]) / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )

    LOGGER.info("Logging to %s", log_path)

    if os.getenv("KG_PIPELINE_DEBUG_OPENAI", "") == "1":
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("openai._base_client").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        LOGGER.info(
            "Enabled DEBUG logging for openai/httpx/urllib3 (KG_PIPELINE_DEBUG_OPENAI=1)"
        )

    seed = int(config.get("seed", 42))
    _set_seed(seed)
    _write_run_metadata(run_dir, config_path, config, seed)
    _log_versions(config)
    paths = _stage_output_paths(run_dir)

    documents, fp = _load_or_run_documents(paths, config, args.single_doc, run_dir)
    if args.stage == "ingestion":
        LOGGER.info("Completed stage=ingestion docs=%d", len(documents))
        return

    chunks, fp = _load_or_run_chunks(paths, config, documents, run_dir, fp)
    if args.stage == "chunking":
        LOGGER.info("Completed stage=chunking chunks=%d", len(chunks))
        return

    ner_map, fp = _load_or_run_ner(paths, config, chunks, run_dir, fp)
    if args.stage == "ner":
        entity_count = sum(len(v) for v in ner_map.values())
        LOGGER.info("Completed stage=ner entities=%d", entity_count)
        return

    relation_vocab = _load_relation_vocab(config, config_path)
    raw_triples, acronym_map, fp = _load_or_run_raw_triples(
        paths,
        config,
        chunks,
        ner_map,
        seed=seed,
        relation_vocab=relation_vocab,
        run_dir=run_dir,
        upstream=fp,
    )
    if args.stage == "llm":
        LOGGER.info("Completed stage=llm triples=%d", len(raw_triples))
        return

    resolved_triples, registry, fp = _load_or_run_resolution(
        paths, config, raw_triples, acronym_map, run_dir=run_dir, upstream=fp
    )
    if args.stage == "resolution":
        LOGGER.info(
            "Completed stage=resolution triples=%d canonical_entities=%d",
            len(resolved_triples),
            len(registry),
        )
        return

    linked_triples = _load_or_run_linking(
        paths, resolved_triples, registry, documents, config, run_dir=run_dir, upstream=fp
    )
    if args.stage == "linking":
        LOGGER.info("Completed stage=linking triples=%d", len(linked_triples))
        return

    if args.dry_run:
        sample = [triple.as_dict() for triple in linked_triples[:5]]
        LOGGER.info("Dry-run enabled, skipping Neo4j ingestion.")
        LOGGER.info("Total triples after linking: %d", len(linked_triples))
        LOGGER.info(
            "Sample triples: %s", json.dumps(sample, ensure_ascii=False, indent=2)
        )
        return

    uri, user, password, env_db = neo4j_ingestion._resolve_neo4j_env()
    db = config.get("neo4j", {}).get("database") or env_db

    written = neo4j_ingestion.ingest_triples(
        triples=linked_triples,
        uri=uri,
        user=user,
        password=password,
        database=db,
    )
    summary = neo4j_ingestion.summary_counts(
        uri=uri, user=user, password=password, database=db
    )

    # `triples_sent` is what it has always been: triples that reached the
    # database. It is an upper bound on edges because MERGE deduplicates, so the
    # authoritative edge count is the one read back from Neo4j in `summary`.
    # It used to be called `relationships_written`, which it never was. See
    # docs/code_audit_2026-08-15.md §3.3.
    _save_json(
        paths["neo4j_summary"],
        {
            "triples_sent": written,
            "relationships_written": written,  # kept for existing readers
            "summary": summary,
        },
    )
    LOGGER.info("Neo4j ingestion complete, triples_sent=%d", written)

    ran = _run_post_passes(args.run_post, config_path, Path(args.env_file))
    _save_json(
        paths["neo4j_summary"],
        {
            "triples_sent": written,
            "relationships_written": written,
            "summary": summary,
            "post_passes_run": ran,
        },
    )


if __name__ == "__main__":
    main()
