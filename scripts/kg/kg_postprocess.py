"""
KG Post-process — unified entrypoint for the Neo4j repair passes
================================================================
Run with:  conda run -n graphllm python scripts/kg/kg_postprocess.py [--passes 1,2,3,4]

Each pass is implemented in its own module (kg_repair.py .. kg_repair4.py) and
operates on the live Neo4j graph. The passes are distinct, ordered
post-processing rounds — not versions of the same script — and are normally
applied in sequence after Stage 6 ingestion:

  1. kg_repair.py   — hub artefact cleanup, PUBLISHED_BY direction fix,
                      RELATED_TO reclassification (LLM), property enrichment (LLM)
  2. kg_repair2.py  — isolated nodes, deterministic rel-type consolidation,
                      geographic Concept→Region, residual normalization (LLM)
  3. kg_repair3.py  — pattern-based RELATED_TO reclassification, PUBLISHED
                      unification, micro-type consolidation, residual cleanup
  4. kg_repair4.py  — PUBLISHED endpoint fixes (LLM), FULL_NAME to property,
                      micro-type consolidation, final residual round
  5. kg_repair5.py  — SAME_AS cluster merge, PUBLISHED direction, inverse
                      micro-types (no LLM); opt-in, not in the default list

Requires NEO4J_* and VLLM_* env vars (each pass loads kg_pipeline/.env).
Use this entrypoint instead of running the kg_repair*.py modules directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

PASS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PASS_DIR.parents[1]
# The kg_repair*.py passes import the kg_pipeline package, which lives at the
# repo root and is not pip-installed. Put the root on sys.path so they load when
# this entrypoint is run directly (python scripts/kg/kg_postprocess.py).
sys.path.insert(0, str(REPO_ROOT))

PASS_MODULES = {
    1: "kg_repair.py",
    2: "kg_repair2.py",
    3: "kg_repair3.py",
    4: "kg_repair4.py",
    5: "kg_repair5.py",
}

logger = logging.getLogger("kg_pipeline")


def _load_pass(number: int):
    """Import the module of pass `number` from its file.

    Raises:
        ImportError: If the file cannot be loaded as a module.
    """
    module_path = PASS_DIR / PASS_MODULES[number]
    spec = importlib.util.spec_from_file_location(
        f"kg_postprocess_pass_{number}", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pass module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Run the selected passes in order."""
    parser = argparse.ArgumentParser(
        description="Run the Neo4j KG post-processing passes in sequence."
    )
    parser.add_argument(
        "--passes",
        default="1,2,3,4",
        help="Comma-separated pass numbers to run, in order (default: 1,2,3,4)",
    )
    args = parser.parse_args()

    try:
        selected = [int(item) for item in args.passes.split(",") if item.strip()]
    except ValueError:
        parser.error(f"--passes must be comma-separated integers, got {args.passes!r}")
    invalid = [n for n in selected if n not in PASS_MODULES]
    if invalid:
        parser.error(f"Unknown passes: {invalid}. Available: {sorted(PASS_MODULES)}")
    if not selected:
        parser.error("--passes selected no pass")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    for number in selected:
        logger.info("Running KG post-process pass %d (%s)", number, PASS_MODULES[number])
        module = _load_pass(number)
        module.main()

    logger.info("KG post-process complete: passes %s", selected)


if __name__ == "__main__":
    sys.exit(main())
