"""Add documents to an existing knowledge graph in lots, without rebuilding it.

A lot is a set of registry documents that enter the graph together. Each lot
has its own folder under ``kg_pipeline/artifacts/graph_lots/<lot>/`` holding
everything it produced and decided, and the ledger
``graph_lots/ledger.json`` records which lots were written to which graph, so
a document never enters one graph twice and a lot can be taken out again.

The steps, each resumable from the files of the lot folder:

1. ``prepare``: the lot's own registry (the lot's rows only), pipeline
   configuration and environment.
2. ``extract``: stages 0-3 of the pipeline on the lot's documents.
3. ``resolve``: stage 4 on the lot's triples, then the lot's entities matched
   against the nodes already in the graph.
4. ``write``: new nodes and edges written with the lot's name, existing nodes
   given only their new aliases.
5. ``remove``: the lot's edges, the nodes it created and the aliases it added
   taken out again.

Stages 0-3 are per document already: a chunk's prompt holds only that chunk,
so extracting a lot gives the triples a full rebuild would give for those
documents. What does not scale is stage 4 across the whole corpus and the
stage 6 ``MERGE``, which would overwrite the properties of an existing node:
this module replaces both for a lot.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from kg_pipeline.utils import corpus_registry
from kg_pipeline.utils.corpus_registry import RegistryRow

LOGGER = logging.getLogger("kg_pipeline")

ROOT = Path(__file__).resolve().parents[1]
LOTS_DIR = ROOT / "kg_pipeline" / "artifacts" / "graph_lots"
LEDGER = "ledger.json"
# Stages 0-3 need no graph; an address nothing listens on makes certain that a
# mistake cannot reach one.
_NO_GRAPH = {"NEO4J_URL": "bolt://127.0.0.1:1", "NEO4J_URI": "bolt://127.0.0.1:1"}
_LOT_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass
class LotFiles:
    """Where the files of one lot live.

    Attributes:
        dir: The lot folder; it is also the pipeline run directory.
    """

    dir: Path

    @property
    def registry(self) -> Path:
        return self.dir / "registro_lotto.csv"

    @property
    def config(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def extract_env(self) -> Path:
        return self.dir / "estrazione.env"

    @property
    def meta(self) -> Path:
        return self.dir / "lotto.json"


@dataclass
class Ledger:
    """Which lots exist and which graph each was written to.

    Attributes:
        path: The ledger file.
        lots: ``lot name -> record`` with ``documents``, ``graph`` and ``status``.
    """

    path: Path
    lots: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, lots_dir: Path) -> Ledger:
        """Read the ledger of ``lots_dir``; empty when there is none yet."""
        path = lots_dir / LEDGER
        if not path.exists():
            return cls(path=path)
        return cls(path=path, lots=json.loads(path.read_text(encoding="utf-8"))["lots"])

    def save(self) -> None:
        """Write the ledger atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"lots": self.lots}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def documents_in(self, graph: str, exclude: str | None = None) -> dict[str, str]:
        """``document id -> lot`` for every lot meant for ``graph`` and not removed.

        A lot counts from the moment it is prepared: two lots extracting the
        same document would both write it.

        Args:
            graph: Graph key (:func:`graph_key`).
            exclude: A lot not to count, usually the one being checked.

        Returns:
            The documents and the lot holding each.
        """
        out: dict[str, str] = {}
        for name, record in self.lots.items():
            if name == exclude or record.get("graph") != graph or record.get("status") == "rimosso":
                continue
            for doc in record.get("documents", []):
                out[doc] = name
        return out


def graph_key(uri: str, database: str | None = "neo4j") -> str:
    """One spelling per graph: the loopback names unified, the database appended."""
    text = uri.strip().rstrip("/").lower()
    for loopback in ("127.0.0.1", "[::1]"):
        text = text.replace(f"//{loopback}", "//localhost")
    return f"{text}/{database or 'neo4j'}"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lot_rows(rows: list[RegistryRow], doc_ids: list[str], in_graph: dict[str, str]) -> list[RegistryRow]:
    """The registry rows of a lot, checked.

    Args:
        rows: The corpus registry.
        doc_ids: Ids of the documents the lot should hold.
        in_graph: ``document id -> lot`` of the documents the target graph
            already holds through earlier lots.

    Returns:
        The rows of ``doc_ids``, in the order given.

    Raises:
        ValueError: If an id is not in the registry, is excluded, is marked as
            already in the graph (``livello`` 2), or is in an earlier lot.
    """
    by_id = {row.id_documento: row for row in rows}
    problems: list[str] = []
    selected: list[RegistryRow] = []
    for doc_id in doc_ids:
        row = by_id.get(doc_id)
        if row is None:
            problems.append(f"{doc_id}: non è nel registro")
        elif row.escluso:
            problems.append(f"{doc_id}: è escluso nel registro")
        elif row.livello == 2:
            problems.append(f"{doc_id}: il registro lo dà già nel grafo (livello 2)")
        elif doc_id in in_graph:
            problems.append(f"{doc_id}: è già nel grafo con il lotto {in_graph[doc_id]}")
        else:
            selected.append(row)
    if len(set(doc_ids)) != len(doc_ids):
        problems.append("lo stesso documento compare due volte")
    if problems:
        raise ValueError("lotto non valido:\n- " + "\n- ".join(problems))
    return selected


def lot_config(
    base: dict[str, Any],
    base_path: Path,
    corpus_dir: Path,
    files: LotFiles,
    ocr_dir: Path,
    stage0_cache: Path,
) -> dict[str, Any]:
    """The pipeline configuration of a lot: the base one, with the lot's paths.

    Args:
        base: The pipeline configuration the graph was built with.
        base_path: Its file, to resolve the relation vocabulary it names.
        corpus_dir: Corpus folder.
        files: The lot's files.
        ocr_dir: Folder of the OCR copies.
        stage0_cache: Stage 0 reading cache.

    Returns:
        A new configuration; ``base`` is left unchanged.
    """
    config = json.loads(json.dumps(base))
    config.setdefault("paths", {}).update(
        {
            "input_dir": str(corpus_dir.resolve()),
            "output_dir": str(files.dir.resolve()),
            "registry": str(files.registry.resolve()),
            "ocr_dir": str(ocr_dir.resolve()),
            "stage0_cache": str(stage0_cache.resolve()),
        }
    )
    vocab = str(config.get("llm", {}).get("relation_vocab_path", "")).strip()
    if vocab and not Path(vocab).is_absolute():
        # The lot's configuration lives in another folder than the base.
        config["llm"]["relation_vocab_path"] = str((base_path.parent / vocab).resolve())
    # Stages 0-3 never write; the graph to write to is named at the write step.
    config.setdefault("neo4j", {})["database"] = "non_usato_estrazione"
    return config


def prepare(
    name: str,
    doc_ids: list[str],
    registry: Path,
    corpus_dir: Path,
    graph: str,
    llm_base_url: str,
    llm_model: str,
    base_config: Path = ROOT / "kg_pipeline" / "config.yaml",
    ocr_dir: Path = ROOT / "kg_pipeline" / "artifacts" / "corpus_ocr",
    stage0_cache: Path | None = None,
    lots_dir: Path = LOTS_DIR,
) -> LotFiles:
    """Create the folder of a new lot with its registry, configuration and environment.

    Args:
        name: Lot name, a plain folder name.
        doc_ids: Registry ids of the lot's documents.
        registry: The corpus registry.
        corpus_dir: Corpus folder.
        graph: The graph the lot is meant for, as :func:`graph_key`; recorded
            in the ledger so the same document is not added to it twice.
        llm_base_url: OpenAI-compatible endpoint of the extraction model.
        llm_model: Served name of the extraction model; it should be the one
            the graph was built with.
        base_config: The pipeline configuration the graph was built with.
        ocr_dir: Folder of the OCR copies.
        stage0_cache: Stage 0 reading cache; the base configuration's when
            ``None``.
        lots_dir: Folder of the lots and of the ledger.

    Returns:
        The lot's files.

    Raises:
        ValueError: If the name is not a plain folder name, the lot exists
            already, or a document cannot enter the lot (see :func:`lot_rows`).
    """
    if not _LOT_NAME.fullmatch(name):
        raise ValueError(f"nome del lotto non valido: {name!r}")
    ledger = Ledger.load(lots_dir)
    if name in ledger.lots:
        raise ValueError(f"il lotto {name} esiste già")
    rows = lot_rows(corpus_registry.load_registry(registry), doc_ids, ledger.documents_in(graph))

    files = LotFiles(lots_dir / name)
    files.dir.mkdir(parents=True, exist_ok=False)
    corpus_registry.save_registry(files.registry, rows)
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cache = stage0_cache or ROOT / base["paths"]["stage0_cache"]
    config = lot_config(base, base_config, corpus_dir, files, ocr_dir, cache)
    files.config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    env = dict(_NO_GRAPH, VLLM_BASE_URL=llm_base_url, VLLM_MODEL_NAME=llm_model, VLLM_API_KEY="EMPTY")
    files.extract_env.write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
    meta = {
        "nome": name,
        "documenti": [row.id_documento for row in rows],
        "grafo": graph,
        "modello_estrazione": llm_model,
        "configurazione_di_base": str(base_config.resolve()),
        "creato": _now(),
    }
    files.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ledger.lots[name] = {
        "documents": meta["documenti"],
        "graph": graph,
        "status": "preparato",
        "created": meta["creato"],
    }
    ledger.save()
    return files


def extract_command(files: LotFiles, python: str) -> tuple[list[str], dict[str, str]]:
    """The command that runs stages 0-3 on a lot, and its environment.

    Stage 3 checkpoints every few dozen chunks, so the same command resumes an
    interrupted extraction.

    Args:
        files: The lot's files.
        python: Interpreter to run the pipeline with.

    Returns:
        ``(command, environment)``.
    """
    config = yaml.safe_load(files.config.read_text(encoding="utf-8"))
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONHASHSEED=str(config.get("seed", 42)))
    command = [
        python,
        "-m",
        "kg_pipeline.main",
        "--config",
        str(files.config),
        "--env-file",
        str(files.extract_env),
        "--run-dir",
        str(files.dir),
        "--stage",
        "llm",
    ]
    return command, env


# --------------------------------------------------------------------------- #
# matching the lot's entities against the graph
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GraphNode:
    """A named node of the graph, as the lot's entities are matched against it.

    Attributes:
        element_id: Neo4j element id.
        label: Its label.
        name: Its name.
        aliases: Its ``aliases`` property.
        degree: Its number of relationships.
    """

    element_id: str
    label: str
    name: str
    aliases: tuple[str, ...]
    degree: int


def read_graph_nodes(session: Any) -> list[GraphNode]:
    """Every named entity node of the graph; vector carriers are not entities."""
    rows = session.run(
        "MATCH (n) WHERE n.name IS NOT NULL AND NOT n:NodeVec "
        "RETURN elementId(n) AS id, [l IN labels(n) WHERE l <> 'NodeVec'][0] AS label, "
        "toString(n.name) AS name, coalesce(n.aliases, []) AS aliases, COUNT { (n)--() } AS degree"
    )
    return [
        GraphNode(r["id"], r["label"] or "Concept", r["name"], tuple(str(a) for a in r["aliases"]), int(r["degree"]))
        for r in rows
    ]


@dataclass(frozen=True)
class Match:
    """The graph node a lot entity is the same entity as.

    Attributes:
        node: The node.
        how: ``"nome"`` for an equal name, ``"giudice"`` for a pair the judge confirmed.
        similarity: Name similarity for a judged pair; 1.0 for an equal name.
    """

    node: GraphNode
    how: str
    similarity: float = 1.0


# A number node is its number: "50 %" of one document and "50 %" of another
# are different facts, and matching them would join the two documents through
# a node that means nothing.
_NEVER_MATCHED = frozenset({"DataValue"})


def _fold(text: str) -> str:
    """Lower case, accents removed, letters and digits only.

    Stage 4 drops every character outside a-z and 0-9, which turns "Perù"
    into "per"; between a lot and a graph, where nothing else checks the pair,
    the accents are folded instead of dropped.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in decomposed if not unicodedata.combining(c)))


def _rank(node: GraphNode, label: str, by_name: bool) -> tuple[int, int, int, str]:
    """Order of preference among nodes an entity's name equals."""
    return (node.label != label, not by_name, -node.degree, node.element_id)


def exact_matches(
    entries: dict[str, Any],
    nodes: list[GraphNode],
    acronym_map: dict[str, str],
) -> dict[str, Match]:
    """Lot entities whose name a graph node already carries, as stage 4 would merge them.

    Stage 4 merges two mentions when they share a label and the same name
    once acronyms are expanded and everything but letters and digits is
    dropped, and merges entities of any label whose names differ only in
    case. The same two rules are applied here between a lot entity (its name
    and aliases) and a node (its name and aliases), with accents folded
    rather than dropped (:func:`_fold`). The lot's acronym map is applied to both sides, as one
    run over the lot and the graph's documents would. Number nodes are never matched. Among
    several nodes, one with the entity's label wins, then one whose name (not
    an alias) matched, then the most connected.

    Args:
        entries: The lot's stage 4 registry, ``canonical name -> record``.
        nodes: The graph's nodes.
        acronym_map: The lot's acronym map.

    Returns:
        ``canonical name -> match`` for the entities a node already carries.
    """
    from kg_pipeline.utils.acronym_map import expand_acronym

    by_key: dict[tuple[str, str], list[tuple[GraphNode, bool]]] = {}
    by_lower: dict[str, list[tuple[GraphNode, bool]]] = {}
    for node in nodes:
        if node.label in _NEVER_MATCHED:
            continue
        for text, is_name in ((node.name, True), *((a, False) for a in node.aliases)):
            key = _fold(expand_acronym(text, acronym_map))
            if key:
                by_key.setdefault((node.label, key), []).append((node, is_name))
            # An alias is a name the graph already confirmed for its node, and
            # labels are noisy ("spreco alimentare" a Process, "food waste" a
            # Concept): a lot name equal to an alias is the same entity
            # whatever the labels.
            lower = text.strip().lower()
            if lower:
                by_lower.setdefault(lower, []).append((node, is_name))

    out: dict[str, Match] = {}
    for canonical, record in entries.items():
        label = (record.labels or ["Concept"])[0]
        if label in _NEVER_MATCHED:
            continue
        found: dict[str, tuple[GraphNode, bool]] = {}
        for text in {canonical, *record.aliases}:
            key = _fold(expand_acronym(text, acronym_map))
            for node, is_name in by_key.get((label, key), []) if key else []:
                found[node.element_id] = (node, is_name or found.get(node.element_id, (node, False))[1])
            for node, is_name in by_lower.get(text.strip().lower(), []):
                found[node.element_id] = (node, is_name or found.get(node.element_id, (node, False))[1])
        if found:
            node, _ = min(found.values(), key=lambda item: _rank(item[0], label, item[1]))
            out[canonical] = Match(node, "nome")
    return out


def _representative(canonical: str, aliases: list[str]) -> str:
    """The longest name of an entity, as stage 4 represents a group."""
    names = sorted({canonical, *aliases}, key=lambda x: (-len(x), x.lower()))
    return names[0]


class NameVectors:
    """Name embeddings of one model, kept on disk so each name is encoded once.

    Attributes:
        path: Folder of the cache.
        model_name: SentenceTransformer model.
    """

    def __init__(self, path: Path, model_name: str):
        self.path = path
        self.model_name = model_name
        self._model: Any = None
        self._names: list[str] = []
        self._index: dict[str, int] = {}
        self._vectors: Any = None
        stem = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
        self._names_file = path / f"nomi_{stem}.json"
        self._vectors_file = path / f"vettori_{stem}.npy"
        if self._names_file.exists() and self._vectors_file.exists():
            import numpy as np

            self._names = json.loads(self._names_file.read_text(encoding="utf-8"))
            self._vectors = np.load(self._vectors_file)
            self._index = {name: i for i, name in enumerate(self._names)}

    def encode(self, names: list[str]) -> Any:
        """Normalised embeddings of ``names``, encoding only those not cached."""
        import numpy as np

        missing = sorted({n for n in names if n not in self._index})
        if missing:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=os.environ.get("KG_EMBED_DEVICE") or None)
            fresh = np.asarray(self._model.encode(missing, normalize_embeddings=True, batch_size=256), dtype="float32")
            start = len(self._names)
            self._names.extend(missing)
            self._index.update({name: start + i for i, name in enumerate(missing)})
            self._vectors = fresh if self._vectors is None else np.vstack([self._vectors, fresh])
            self.path.mkdir(parents=True, exist_ok=True)
            np.save(self._vectors_file, self._vectors)
            self._names_file.write_text(json.dumps(self._names, ensure_ascii=False), encoding="utf-8")
        return self._vectors[[self._index[n] for n in names]]


def similar_candidates(
    entries: dict[str, Any],
    nodes: list[GraphNode],
    vectors: NameVectors,
    threshold: float,
    top_k: int = 10,
) -> list[tuple[str, GraphNode, float]]:
    """Graph nodes whose name embeds close to a lot entity's, with stage 4's thresholds.

    Each entity is represented by its longest name, as stage 4 represents a
    group, and compared with the nodes' names through an exact inner-product
    index over the normalised embeddings, top ``top_k`` only: comparing every
    pair would grow with the graph times the lot. A pair of the same label
    must exceed ``threshold``, a pair across labels ``max(threshold, 0.92)``,
    and two names that both carry numbers must carry the same ones. Number
    nodes are never candidates.

    Args:
        entries: Lot entities not matched by name, ``canonical name -> record``.
        nodes: The graph's nodes.
        vectors: Name embeddings of the stage 4 model.
        threshold: Stage 4's ``similarity_threshold``.
        top_k: Nearest nodes looked at per entity.

    Returns:
        ``(canonical name, node, similarity)`` candidates.
    """
    import faiss

    from kg_pipeline.stages.resolution import _numbers

    nodes = [node for node in nodes if node.label not in _NEVER_MATCHED]
    entries = {k: v for k, v in entries.items() if (v.labels or ["Concept"])[0] not in _NEVER_MATCHED}
    if not entries or not nodes:
        return []
    node_vectors = vectors.encode([node.name for node in nodes])
    index = faiss.IndexFlatIP(node_vectors.shape[1])
    index.add(node_vectors)
    canonicals = sorted(entries)
    reps = [_representative(c, entries[c].aliases) for c in canonicals]
    scores, ids = index.search(vectors.encode(reps), min(top_k, len(nodes)))
    cross_label = max(threshold, 0.92)
    out: list[tuple[str, GraphNode, float]] = []
    for canonical, rep, row_scores, row_ids in zip(canonicals, reps, scores, ids):
        label = (entries[canonical].labels or ["Concept"])[0]
        for score, idx in zip(row_scores, row_ids):
            if idx < 0:
                continue
            node = nodes[int(idx)]
            if score <= (threshold if node.label == label else cross_label):
                continue
            a, b = _numbers(rep), _numbers(node.name)
            if a and b and a != b:
                continue
            out.append((canonical, node, float(score)))
    return out


Judge = Any  # callable: list[tuple[str, str]] -> list[bool | None]


def judged_matches(
    candidates: list[tuple[str, GraphNode, float]],
    entries: dict[str, Any],
    judge: Judge,
    verdicts_path: Path,
) -> dict[str, Match]:
    """Candidates the strict judge confirms; each entity keeps its closest confirmed node.

    Verdicts are stored in ``verdicts_path`` keyed ``"entity name||node name"``
    and reused, so a repeated run asks only for pairs it has not judged; a
    pair without a verdict is asked again next time and never merged.

    Args:
        candidates: From :func:`similar_candidates`.
        entries: The lot's registry.
        judge: Takes name pairs, returns one verdict per pair (``None`` when
            none came back).
        verdicts_path: The verdict store.

    Returns:
        ``canonical name -> match``.
    """
    stored: dict[str, bool | None] = (
        json.loads(verdicts_path.read_text(encoding="utf-8")) if verdicts_path.exists() else {}
    )
    pairs = [(_representative(c, entries[c].aliases), node.name) for c, node, _ in candidates]
    todo = sorted({f"{a}||{b}" for a, b in pairs if stored.get(f"{a}||{b}") is None})
    if todo:
        answers = judge([tuple(key.split("||", 1)) for key in todo])
        for key, verdict in zip(todo, answers):
            stored[key] = verdict
        verdicts_path.write_text(json.dumps(stored, ensure_ascii=False, indent=1), encoding="utf-8")
    best: dict[str, Match] = {}
    for (canonical, node, score), (a, b) in zip(candidates, pairs):
        if stored.get(f"{a}||{b}") is True and score > best.get(canonical, Match(node, "", -1.0)).similarity:
            best[canonical] = Match(node, "giudice", score)
    return best


# --------------------------------------------------------------------------- #
# writing a lot into the graph, and taking it out again
# --------------------------------------------------------------------------- #

# A node of the lot is referenced as ("id", element id) when it is a node the
# graph already had, as ("new", label, name) when the lot creates it.
NodeRef = tuple


@dataclass
class WritePlan:
    """What a lot writes, decided before anything is written.

    Attributes:
        new_nodes: ``(label, name) -> properties`` of the nodes the lot creates.
        aliases: ``node ref -> aliases`` to add, to new and existing nodes alike.
        edges: ``(subject ref, type, object ref, properties)``.
        skipped_self_loops: Triples whose two ends became the same node.
    """

    new_nodes: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    aliases: dict[NodeRef, list[str]] = field(default_factory=dict)
    edges: list[tuple[NodeRef, str, NodeRef, dict[str, Any]]] = field(default_factory=list)
    skipped_self_loops: int = 0


def _node_props(props: dict[str, Any]) -> dict[str, Any]:
    """A node's own properties, without the name it is merged on."""
    return {k: v for k, v in props.items() if k not in {"name", "search_text", "aliases", "lotto"}}


def plan_writes(triples: list[Any], registry: dict[str, Any], matches: dict[str, Match]) -> WritePlan:
    """Turn the lot's linked triples into the nodes, aliases and edges to write.

    An entity matched to a graph node becomes that node, whose name and label
    stay; all its lot names become aliases of the node, since the entity now
    appears in documents of the graph and of the lot. Any other entity becomes
    a new node under its stage 4 name and label. ``SAME_AS`` triples, which
    stage 5 writes for an entity found in two documents of the lot, become
    aliases instead of alias nodes, the form the graph keeps them in.

    Args:
        triples: Stage 5 triples of the lot.
        registry: Stage 4 registry of the lot.
        matches: ``canonical name -> match``.

    Returns:
        The plan.
    """
    plan = WritePlan()

    def ref(name: str, labels: list[str], props: dict[str, Any]) -> NodeRef:
        if name in matches:
            return ("id", matches[name].node.element_id)
        label = (registry[name].labels if name in registry else labels or ["Concept"])[0]
        key = (label, name)
        if key not in plan.new_nodes:
            source = registry[name].merged_properties if name in registry else props
            plan.new_nodes[key] = _node_props(source)
        return ("new", label, name)

    def add_aliases(target: NodeRef, names: list[str], own_name: str) -> None:
        bucket = plan.aliases.setdefault(target, [])
        for alias in names:
            if alias and alias != own_name and alias not in bucket:
                bucket.append(alias)

    for canonical, match in matches.items():
        record = registry.get(canonical)
        names = [canonical, *(record.aliases if record else [])]
        add_aliases(("id", match.node.element_id), names, match.node.name)

    for triple in triples:
        if triple.predicate == "SAME_AS":
            target = ref(triple.object, triple.object_labels, triple.object_properties)
            own = matches[triple.object].node.name if triple.object in matches else triple.object
            add_aliases(target, [triple.subject], own)
            continue
        s = ref(triple.subject, triple.subject_labels, triple.subject_properties)
        o = ref(triple.object, triple.object_labels, triple.object_properties)
        if s == o:
            plan.skipped_self_loops += 1
            continue
        plan.edges.append((s, triple.predicate, o, dict(triple.relationship_properties)))
    plan.aliases = {k: v for k, v in plan.aliases.items() if v}
    return plan


def _safe_identifier(value: str) -> str:
    """A label or relationship type safe to put in a query: letters, digits, underscores."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"identificatore non valido: {value!r}")
    return value


def _scalar_props(props: dict[str, Any]) -> dict[str, Any]:
    """Properties Neo4j can store: scalars and lists of scalars; anything else as text."""
    out: dict[str, Any] = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
            out[key] = value
        else:
            out[key] = json.dumps(value, ensure_ascii=False)
    return out


def graph_counts(session: Any) -> dict[str, int]:
    """Nodes and edges of the graph, and how many of them belong to some lot."""
    row = session.run(
        "CALL () { MATCH (n) WHERE NOT n:NodeVec RETURN count(n) AS nodes, count(n.lotto) AS lot_nodes } "
        "CALL () { MATCH ()-[r]->() RETURN count(r) AS edges, count(r.lotto) AS lot_edges } "
        "RETURN nodes, lot_nodes, edges, lot_edges"
    ).single()
    return {k: int(row[k]) for k in ("nodes", "lot_nodes", "edges", "lot_edges")}


def check_matched_nodes(session: Any, expected: dict[str, tuple[str, str]]) -> list[str]:
    """Matched node ids that no longer point at the node they were matched with.

    Element ids are stable while a database runs, not across a restore: a
    write planned before the graph was reloaded would add aliases and edges to
    whatever node now has the id.

    Args:
        session: Neo4j session.
        expected: ``element id -> (label, name)`` recorded when matching.

    Returns:
        One line per id that is missing or now names another node.
    """
    rows = session.run(
        "UNWIND $ids AS id OPTIONAL MATCH (n) WHERE elementId(n) = id "
        "RETURN id, n.name AS name, labels(n) AS labels",
        ids=sorted(expected),
    )
    problems = []
    for r in rows:
        label, name = expected[r["id"]]
        if r["name"] != name or label not in (r["labels"] or []):
            problems.append(f"{r['id']}: atteso {label} {name!r}, trovato {r['labels']} {r['name']!r}")
    return problems


def _snapshot(session: Any, ids: list[str], record: dict[str, Any], log_path: Path) -> None:
    """Save the name and aliases of the nodes in ``ids`` not saved yet, before they change."""
    missing = [i for i in ids if i not in record["alias_prima"]]
    if not missing:
        return
    for r in session.run(
        "UNWIND $ids AS id MATCH (n) WHERE elementId(n) = id RETURN id, n.name AS name, n.aliases AS aliases",
        ids=missing,
    ):
        record["alias_prima"][r["id"]] = {"name": r["name"], "aliases": r["aliases"]}
    log_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


def write_lot(
    session: Any,
    lot: str,
    plan: WritePlan,
    log_path: Path,
    expected: dict[str, tuple[str, str]],
    batch: int = 500,
) -> dict[str, Any]:
    """Write a lot's plan: new nodes and edges tagged with the lot, aliases added.

    Nothing is written unless every matched node is still the node it was
    matched with (:func:`check_matched_nodes`). The name and aliases of every
    node that gets aliases are saved in ``log_path`` before they change, on
    every call, which is what :func:`remove_lot` restores; every write is a
    ``MERGE`` keyed on the lot, so a second call after an interruption does
    not duplicate what the first wrote. Nodes already in the graph keep their
    name, label and properties; their vector carrier is deleted so the vector
    index re-embeds them with the new aliases.

    Args:
        session: Neo4j session on the target database.
        lot: Lot name, written as ``lotto`` on what the lot creates.
        plan: From :func:`plan_writes`.
        log_path: The write record of the lot.
        expected: ``element id -> (label, name)`` of the matched nodes.
        batch: Rows per query.

    Returns:
        The write record.

    Raises:
        ValueError: If a matched node id no longer points at its node.
    """
    problems = check_matched_nodes(session, expected)
    if problems:
        raise ValueError(
            f"{len(problems)} nodi del grafo non sono più quelli trovati da resolve "
            "(il grafo è stato ricaricato?): rilanciare resolve.\n- " + "\n- ".join(problems[:10])
        )
    record: dict[str, Any] = (
        json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    )
    if "prima" not in record:
        record["prima"] = graph_counts(session)
        record["alias_prima"] = {}
    _snapshot(session, [r[1] for r in plan.aliases if r[0] == "id"], record, log_path)

    for label in sorted({label for label, _ in plan.new_nodes}):
        session.run(f"CREATE INDEX idx_{label.lower()}_name IF NOT EXISTS FOR (n:{_safe_identifier(label)}) ON (n.name)")

    resolved: dict[NodeRef, str] = {}
    by_label: dict[str, list[dict[str, Any]]] = {}
    for (label, name), props in plan.new_nodes.items():
        by_label.setdefault(label, []).append({"name": name, "props": _scalar_props(props)})
    found_existing: list[str] = []
    for label, rows in by_label.items():
        # A number node of the lot is merged within the lot only, so it never
        # becomes the graph's node of the same number (see _NEVER_MATCHED).
        key = "{name: row.name, lotto: $lot}" if label in _NEVER_MATCHED else "{name: row.name}"
        for start in range(0, len(rows), batch):
            result = session.run(
                f"UNWIND $rows AS row MERGE (n:{_safe_identifier(label)} {key}) "
                "ON CREATE SET n += row.props, n.lotto = $lot "
                "RETURN row.name AS name, elementId(n) AS id, coalesce(n.lotto = $lot, false) AS ours",
                rows=rows[start : start + batch],
                lot=lot,
            )
            for r in result:
                resolved[("new", label, r["name"])] = r["id"]
                if not r["ours"]:
                    found_existing.append(r["id"])
                    LOGGER.warning("%s %r esisteva già nel grafo: usato com'è", label, r["name"])
    # A node the graph already had, found by name here, may get aliases too.
    _snapshot(session, found_existing, record, log_path)

    def element(ref: NodeRef) -> str:
        return ref[1] if ref[0] == "id" else resolved[ref]

    alias_rows = [{"id": element(ref), "aliases": names} for ref, names in plan.aliases.items()]
    for start in range(0, len(alias_rows), batch):
        session.run(
            "UNWIND $rows AS row MATCH (n) WHERE elementId(n) = row.id "
            "SET n.aliases = coalesce(n.aliases, []) + "
            "[a IN row.aliases WHERE a <> n.name AND NOT a IN coalesce(n.aliases, [])]",
            rows=alias_rows[start : start + batch],
        )
    session.run("MATCH (v:NodeVec) WHERE v.of IN $ids DETACH DELETE v", ids=list(record["alias_prima"]))

    by_type: dict[str, list[dict[str, Any]]] = {}
    for s, rel_type, o, props in plan.edges:
        by_type.setdefault(rel_type, []).append(
            {"s": element(s), "o": element(o), "props": _scalar_props(props)}
        )
    written = 0
    for rel_type, rows in by_type.items():
        for start in range(0, len(rows), batch):
            written += session.run(
                "UNWIND $rows AS row MATCH (s) WHERE elementId(s) = row.s "
                "MATCH (o) WHERE elementId(o) = row.o "
                f"MERGE (s)-[r:{_safe_identifier(rel_type)} {{subject: s.name, object: o.name, lotto: $lot}}]->(o) "
                "ON CREATE SET r += row.props RETURN count(*) AS n",
                rows=rows[start : start + batch],
                lot=lot,
            ).single()["n"]
    if written != len(plan.edges):
        LOGGER.warning("archi del piano %d, scritti %d", len(plan.edges), written)

    record["dopo"] = graph_counts(session)
    record["nodi_nuovi"] = len(plan.new_nodes)
    record["nodi_con_alias_nuovi"] = len(plan.aliases)
    record["archi_del_piano"] = len(plan.edges)
    record["archi_scritti"] = written
    record["autoanelli_saltati"] = plan.skipped_self_loops
    record["scritto"] = _now()
    log_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def remove_lot(session: Any, lot: str, log_path: Path, batch: int = 5000) -> dict[str, Any]:
    """Take a lot out of the graph: its edges, the nodes it created, the aliases it added.

    Aliases are restored only on a node that still has the name recorded with
    them, so a reloaded graph whose ids now name other nodes is left alone and
    the mismatch reported. A node the lot created that a later lot has
    connected to stays, and is reported. The vector carriers of the nodes
    deleted or restored are deleted too, so the vector index re-embeds what is
    left. Lots on one graph come out in the reverse order they went in: a
    restore puts back the aliases as they were before this lot.

    Args:
        session: Neo4j session on the target database.
        lot: Lot name.
        log_path: The write record of the lot, with the aliases before the write.
        batch: Edges deleted per transaction.

    Returns:
        What was removed.
    """
    record = json.loads(log_path.read_text(encoding="utf-8"))
    edges = 0
    while True:
        deleted = session.run(
            "MATCH ()-[r]->() WHERE r.lotto = $lot WITH r LIMIT $batch DELETE r RETURN count(*) AS n",
            lot=lot,
            batch=batch,
        ).single()["n"]
        edges += deleted
        if deleted == 0:
            break
    before = [
        {"id": node_id, "name": saved["name"], "aliases": saved["aliases"]}
        for node_id, saved in record.get("alias_prima", {}).items()
    ]
    restored = session.run(
        "UNWIND $rows AS row MATCH (n) WHERE elementId(n) = row.id AND n.name = row.name "
        "SET n.aliases = row.aliases RETURN collect(row.id) AS ids",
        rows=before,
    ).single()["ids"]
    not_restored = sorted({row["id"] for row in before} - set(restored))
    kept = session.run(
        "MATCH (n) WHERE n.lotto = $lot AND COUNT { (n)--() } > 0 RETURN n.name AS name", lot=lot
    ).value()
    gone = session.run(
        "MATCH (n) WHERE n.lotto = $lot AND COUNT { (n)--() } = 0 "
        "WITH n, elementId(n) AS id DELETE n RETURN collect(id) AS ids",
        lot=lot,
    ).single()["ids"]
    session.run("MATCH (v:NodeVec) WHERE v.of IN $ids DETACH DELETE v", ids=gone + restored)
    return {
        "archi_tolti": edges,
        "nodi_tolti": len(gone),
        "alias_ripristinati": len(restored),
        "alias_non_ripristinati_nome_cambiato": not_restored,
        "nodi_tenuti_perche_collegati_da_altri_lotti": kept,
        "dopo": graph_counts(session),
        "rimosso": _now(),
    }


# --------------------------------------------------------------------------- #
# bilingual unions involving the lot
# --------------------------------------------------------------------------- #

# Nodes the unions never touch: people and documents are names, not concepts,
# and a number node is its number.
UNION_SKIP = ("Person", "Document", "DataValue", "NodeVec")


def propose_unions(
    session: Any,
    lot: str,
    encode: Any,
    judge: Judge,
    verdicts_path: Path,
    min_degree: int = 2,
    top_k: int = 5,
    min_cos: float = 0.86,
) -> list[dict[str, Any]]:
    """Unions of a lot node with a node of the same meaning, approved by the strict judge.

    Typically an Italian and an English name of one concept that the two
    resolution steps left apart. Only pairs with at least one node of the lot
    are proposed: the graph's own unions were read when it was built. The
    node outside the lot, when there is one, is always the centre, so the
    graph's nodes keep their name and label and the lot can still be taken
    out; between two lot nodes the more connected one is the centre.

    Args:
        session: Neo4j session.
        lot: Lot name.
        encode: Takes names, returns their vectors (the e5 encoder).
        judge: Takes name pairs, returns verdicts.
        verdicts_path: Verdict store keyed ``"name||name"``, reused on a later run.
        min_degree: Minimum relationships of a lot node to be considered.
        top_k: Nearest nodes looked at per lot node.
        min_cos: Minimum cosine similarity of a candidate pair.

    Returns:
        ``{"centro", "unito", "id_centro", "id_unito", "grado_centro", "grado_unito"}``
        records, one per union, for reading before they are applied.
    """
    import numpy as np

    from kg_pipeline.stages.resolution import _numbers

    rows = session.run(
        "MATCH (n) WHERE n.name IS NOT NULL AND NOT any(l IN labels(n) WHERE l IN $skip) "
        "RETURN elementId(n) AS id, toString(n.name) AS name, coalesce(n.lotto = $lot, false) AS ours, "
        "COUNT { (n)--() } AS deg",
        skip=list(UNION_SKIP),
        lot=lot,
    ).data()
    ours = [i for i, r in enumerate(rows) if r["ours"] and r["deg"] >= min_degree]
    if not ours:
        return []
    vectors = np.asarray(encode([r["name"] for r in rows]), dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    from kg_pipeline.stages.resolution import _norm

    pairs: set[tuple[int, int]] = set()
    for i in ours:
        sims = vectors @ vectors[i]
        sims[i] = -1
        for j in np.argsort(-sims)[:top_k]:
            j = int(j)
            if sims[j] < min_cos or _norm(rows[i]["name"]) == _norm(rows[j]["name"]):
                continue
            a, b = _numbers(rows[i]["name"]), _numbers(rows[j]["name"])
            if a and b and a != b:
                continue
            pairs.add(tuple(sorted((i, j))))
    candidates = sorted(pairs)
    stored: dict[str, bool | None] = (
        json.loads(verdicts_path.read_text(encoding="utf-8")) if verdicts_path.exists() else {}
    )
    keys = [f'{rows[i]["name"]}||{rows[j]["name"]}' for i, j in candidates]
    todo = sorted({key for key in keys if stored.get(key) is None})
    if todo:
        for key, verdict in zip(todo, judge([tuple(key.split("||", 1)) for key in todo])):
            stored[key] = verdict
        verdicts_path.write_text(json.dumps(stored, ensure_ascii=False, indent=1), encoding="utf-8")
    approved = [pair for pair, key in zip(candidates, keys) if stored.get(key) is True]

    def centre_first(i: int, j: int) -> tuple[int, int]:
        # A node outside the lot is the centre; otherwise the more connected one.
        key = lambda k: (rows[k]["ours"], -rows[k]["deg"], rows[k]["id"])  # noqa: E731
        return (i, j) if key(i) <= key(j) else (j, i)

    unions: list[dict[str, Any]] = []
    merged: set[int] = set()
    centres: set[int] = set()
    ordered = sorted(
        (centre_first(i, j) for i, j in approved),
        key=lambda p: (rows[p[0]]["ours"], -rows[p[0]]["deg"], rows[p[0]]["id"]),
    )
    for i, j in ordered:
        # Star unions, as stage 4: a merged node is never a centre, a centre
        # is never merged, and nothing is merged twice.
        if i in merged or j in merged or j in centres:
            continue
        centres.add(i)
        merged.add(j)
        unions.append(
            {
                "centro": rows[i]["name"],
                "unito": rows[j]["name"],
                "id_centro": rows[i]["id"],
                "id_unito": rows[j]["id"],
                "grado_centro": rows[i]["deg"],
                "grado_unito": rows[j]["deg"],
                "centro_nel_lotto": bool(rows[i]["ours"]),
            }
        )
    return unions


def apply_unions(
    session: Any,
    unions: list[dict[str, Any]],
    excluded: set[tuple[str, str]],
    write_log: Path,
) -> int:
    """Merge each union into its centre, except the excluded ones; return how many were applied.

    The centre keeps its name, label and properties; the merged node's name
    and aliases become aliases of the centre, and its relationships move to
    the centre with their ``lotto``. A centre outside the lot has its aliases
    saved first in the lot's write record, so taking the lot out restores
    them.

    Args:
        session: Neo4j session.
        unions: From :func:`propose_unions`.
        excluded: ``(unito, centro)`` name pairs rejected on reading.
        write_log: The lot's write record.

    Returns:
        The number of unions applied.
    """
    record = json.loads(write_log.read_text(encoding="utf-8"))
    todo = [u for u in unions if (u["unito"], u["centro"]) not in excluded]
    found = {
        (r["k"], r["o"]): (r["kn"], r["on"])
        for r in session.run(
            "UNWIND $rows AS row MATCH (k) WHERE elementId(k) = row.k MATCH (o) WHERE elementId(o) = row.o "
            "RETURN row.k AS k, row.o AS o, k.name AS kn, o.name AS on",
            rows=[{"k": u["id_centro"], "o": u["id_unito"]} for u in todo],
        )
    }
    valid = []
    for union in todo:
        if found.get((union["id_centro"], union["id_unito"])) != (union["centro"], union["unito"]):
            LOGGER.warning("unione saltata, i nodi non sono più quelli proposti: %s", union)
        else:
            valid.append(union)
    _snapshot(session, [u["id_centro"] for u in valid if not u["centro_nel_lotto"]], record, write_log)
    for union in valid:
        # One union per query: two unions on one centre must each see the
        # aliases the previous one gave it. mergeNodes adds to the centre
        # every property only the merged node has, its `lotto` included, and
        # merging relationships would fold a lot edge into one of the graph's:
        # the centre gets back exactly its own properties, and parallel edges
        # stay apart.
        session.run(
            "MATCH (k) WHERE elementId(k) = $k MATCH (o) WHERE elementId(o) = $o "
            "WITH k, o, properties(k) AS own, labels(k) AS keep, "
            "     [x IN coalesce(k.aliases, []) + [o.name] + coalesce(o.aliases, []) WHERE x <> k.name] AS al "
            "CALL apoc.refactor.mergeNodes([k, o], {properties: 'discard', mergeRels: false}) YIELD node "
            "SET node = own "
            "SET node.aliases = apoc.coll.toSet(al) "
            "WITH node, [l IN labels(node) WHERE NOT l IN keep] AS extra "
            "CALL apoc.create.removeLabels(node, extra) YIELD node AS n2 RETURN count(*)",
            k=union["id_centro"],
            o=union["id_unito"],
        )
    ids = [u["id_centro"] for u in valid] + [u["id_unito"] for u in valid]
    session.run("MATCH (v:NodeVec) WHERE v.of IN $ids DETACH DELETE v", ids=ids)
    session.run("MATCH (n)-[r]->(n) WHERE r.lotto IS NOT NULL DELETE r")
    return len(valid)
