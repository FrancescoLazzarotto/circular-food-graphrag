"""GraphRAG over a Neo4j knowledge graph, with a text-only RAG baseline."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .config import (
    AgentConfig,
    KGConfig,
    OUTPUT_COMPLEXITY,
    OUTPUT_TONE,
    DEFAULT_MODEL_ID,
    build_kg_config_from_env,
)
from .agent.core import KGRAGAgent
from .kg.manager import KnowledgeGraphManager
from .kg.retriever import KGRetriever
from .llm.manager import LLMManager
from .text_rag.agent import StandardRAGAgent
from .text_rag.manager import TextChunk, TextRAGManager
from .text_rag.pipeline import RetrievedTextChunk, StandardTextRAGPipeline
from .types import KGNode, KGTriple, ProvenanceRecord, RAGState, Triple

# Read from the installed distribution metadata so the version has one source of
# truth (pyproject.toml). The fallback covers running straight from a source
# checkout that was never `pip install`ed, as the tests and the cluster sbatch
# scripts do.
try:
    __version__ = _pkg_version("graphrag-pipeline")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "AgentConfig",
    "KGConfig",
    "OUTPUT_COMPLEXITY",
    "OUTPUT_TONE",
    "DEFAULT_MODEL_ID",
    "build_kg_config_from_env",
    "KGRAGAgent",
    "KnowledgeGraphManager",
    "KGRetriever",
    "LLMManager",
    "StandardRAGAgent",
    "TextChunk",
    "TextRAGManager",
    "RetrievedTextChunk",
    "StandardTextRAGPipeline",
    "KGNode",
    "KGTriple",
    "ProvenanceRecord",
    "RAGState",
    "Triple",
]
