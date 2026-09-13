"""
Abstract base for VAGEN trajectory → BaseGraph conversion.

Subclasses implement traj_to_transitions() and return VagenNodeData / VagenEdgeData
(or env-specific subclasses) to drive dedup and graph construction.

── Dedup: two phases ──────────────────────────────────────────────────────

Phase 1 — Node dedup (per-node, before edges):
    NodeData.unique_key()        → deterministic ID stored in the graph
    NodeData.bucket_key()        → coarse grouping key for fast lookup
    NodeData.is_similar_to(other)→ within-bucket tolerance check

  Flow:  bucket_key → scan bucket → similarity match → reuse unique_key
         or create new entry with unique_key()

  VagenNodeData defaults: bucket_key == unique_key == sha256(state)[:16],
                           is_similar_to=True  → one entry per bucket, O(1).
  Override in env-specific subclass for coarser grouping + real similarity.

Phase 2 — Edge dedup (after both endpoints are resolved):
    EdgeData.unique_key()        → dedup key (scoped to src/dst pair)
    EdgeData.bucket_key()        → coarse grouping for scan
    EdgeData.is_similar_to(other)→ pairwise check

  Edges are first constrained by the (from_uid, to_uid) pair, then
  deduped by bucket_key + is_similar_to within that pair.

  VagenEdgeData defaults: bucket_key == unique_key == repr(obs_str),
                           is_similar_to checks obs_str equality.

── Multiprocessing ────────────────────────────────────────────────────────
  config["num_workers"] = 4

Each worker builds a sub-graph via _build_sequential.  The main process
merges sub-graphs via _merge_graph (same dedup logic).

── Image handling ─────────────────────────────────────────────────────────
Set VagenNodeData.source_images to absolute paths.  They are copied to
graph_dir/images/{unique_key}_{idx}{suffix} (idempotent, concurrent-safe).
"""

import hashlib
import importlib
import json
import logging
import re
import shutil
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphrl.traj_to_sft.utils.base_graph import BaseGraph, EdgeData, NodeData

logger = logging.getLogger(__name__)


# ── Concrete data classes (Layer 1) ──────────────────────────────────────────

class VagenNodeData(NodeData):
    """
    Default VAGEN node data with sha256-based dedup.

    Fields:
        state         — raw state (any JSON-serializable); used for unique_key
        obs_str       — text representation shown to the LLM
        source_images — absolute paths to source images (copied into graph)
        image_paths   — relative paths after images are copied into graph dir
        extra         — arbitrary metadata dict
    """

    def __init__(
        self,
        state: Any,
        obs_str: Optional[str] = None,
        source_images: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state = state
        self.obs_str = obs_str if obs_str is not None else str(state)
        self.source_images = source_images or []
        self.image_paths = image_paths or []
        self.extra = extra or {}

    @staticmethod
    def state_to_id(state: Any) -> str:
        """Canonical key for a state: first 16 hex chars of sha256."""
        if isinstance(state, str):
            raw = state
        else:
            raw = json.dumps(state, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def unique_key(self) -> str:
        return self.state_to_id(self.state)

    def bucket_key(self) -> str:
        return self.unique_key()

    def is_similar_to(self, other: "NodeData") -> bool:
        return True


class VagenEdgeData(EdgeData):
    """
    Default VAGEN edge data with obs_str-based dedup.

    Fields:
        obs_str       — action text (e.g. "move_left", "turn_left | move_forward")
        image_paths   — relative paths (usually empty for edges)
        extra         — arbitrary metadata dict
    """

    def __init__(
        self,
        obs_str: str,
        image_paths: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.obs_str = obs_str
        self.image_paths = image_paths or []
        self.extra = extra or {}

    def unique_key(self) -> str:
        return repr(self.obs_str)

    def bucket_key(self) -> str:
        return self.unique_key()

    def is_similar_to(self, other: "EdgeData") -> bool:
        return isinstance(other, VagenEdgeData) and self.obs_str == other.obs_str


# ── module-level worker (must be importable by ProcessPoolExecutor) ───────────

def _chunk_worker(
    file_strs: List[str],
    rollout_dir_str: str,
    images_dir_str: str,
    builder_module: str,
    builder_class: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    mod = importlib.import_module(builder_module)
    cls = getattr(mod, builder_class)
    builder: "VagenGraphBuilder" = cls(config)
    graph = builder._build_sequential(
        files=[Path(f) for f in file_strs],
        rollout_dir=Path(rollout_dir_str),
        images_dir=Path(images_dir_str),
    )
    return graph.to_dict()


# ── VagenGraphBuilder ─────────────────────────────────────────────────

class VagenGraphBuilder:
    """Convert VAGEN rollout JSONLs → BaseGraph using a NetworkX backend.

    All current envs that build a graph use this implementation, so there is
    no abstract base — subclass this directly and override
    ``traj_to_transitions()``. Optionally override ``_parse_vagen_line()`` for
    non-ChatML templates.

    Dedup is driven by the NodeData/EdgeData objects returned from
    ``traj_to_transitions()``. Override ``unique_key`` / ``bucket_key`` /
    ``is_similar_to`` on your NodeData/EdgeData subclasses to customize.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._node_buckets: Dict[str, List[str]] = defaultdict(list)

    # ── MUST override ────────────────────────────────────────────────────────

    @abstractmethod
    def traj_to_transitions(
        self,
        messages: List[Dict[str, str]],
        rollout_dir: Path,
        step_idx: int,
        line_idx: int,
        episode_data: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[NodeData, EdgeData, NodeData]]:
        """Return list of (src_node, edge_data, dst_node) from one episode."""
        ...

    # ── internal: resolve ────────────────────────────────────────────────────

    def _resolve_node(self, node: NodeData, graph: BaseGraph) -> str:
        """Phase 1: bucket_key → scan → is_similar_to → unique_key."""
        bucket = node.bucket_key()
        for existing_uid in self._node_buckets.get(bucket, []):
            if existing_uid not in graph._g:
                continue
            ndata = graph._g.nodes[existing_uid]
            existing = self._make_node_data(ndata)
            if node.is_similar_to(existing):
                return existing_uid
        uid = node.unique_key()
        self._node_buckets[bucket].append(uid)
        return uid

    def _resolve_edge(
        self, from_uid: str, to_uid: str, edge: EdgeData, graph: BaseGraph,
    ) -> str:
        """Phase 2: constrained to (from_uid, to_uid), then bucket_key → scan → is_similar_to."""
        if graph._g.has_node(from_uid) and graph._g.has_node(to_uid):
            edge_dict = graph._g.get_edge_data(from_uid, to_uid)
            if edge_dict:
                # Group existing edges by bucket_key, then check similarity
                buckets: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
                for existing_eid, edata in edge_dict.items():
                    existing = self._make_edge_data(edata)
                    buckets[existing.bucket_key()].append((existing_eid, edata))

                target_bucket = edge.bucket_key()
                for existing_eid, edata in buckets.get(target_bucket, []):
                    existing = self._make_edge_data(edata)
                    if edge.is_similar_to(existing):
                        return existing_eid
        return edge.unique_key()

    # ── factory methods (override for env-specific subclasses) ───────────────

    def _make_node_data(self, ndata: Dict[str, Any]) -> NodeData:
        """Reconstruct a NodeData from NetworkX node attributes.

        Override in env-specific builders that use NodeData subclasses,
        so that dedup (bucket_key, is_similar_to) uses the correct class.
        """
        return VagenNodeData(
            state=ndata["state"],
            obs_str=ndata.get("obs_str"),
            image_paths=ndata.get("image_paths", []),
            extra=ndata.get("extra", {}),
        )

    def _make_edge_data(self, edata: Dict[str, Any]) -> EdgeData:
        """Reconstruct an EdgeData from NetworkX edge attributes.

        Override in env-specific builders that use EdgeData subclasses.
        """
        return VagenEdgeData(
            obs_str=edata["obs_str"],
            image_paths=edata.get("image_paths", []),
            extra=edata.get("extra", {}),
        )

    # ── internal: apply / merge ──────────────────────────────────────────────

    def _apply_transitions(
        self, graph: BaseGraph,
        transitions: List[Tuple[NodeData, EdgeData, NodeData]],
        images_dir: Path,
    ) -> None:
        for src, edge, dst in transitions:
            # Phase 1: resolve nodes
            src_uid = self._resolve_node(src, graph)
            src_imgs = self._copy_images(src, src_uid, images_dir)
            graph._upsert_node(
                src_uid, src.state,
                obs_str=src.obs_str, extra=src.extra, image_paths=src_imgs,
            )

            dst_uid = self._resolve_node(dst, graph)
            dst_imgs = self._copy_images(dst, dst_uid, images_dir)
            graph._upsert_node(
                dst_uid, dst.state,
                obs_str=dst.obs_str, extra=dst.extra, image_paths=dst_imgs,
            )

            # Phase 2: resolve edge (skip self-loops)
            if src_uid == dst_uid:
                continue
            eid = self._resolve_edge(src_uid, dst_uid, edge, graph)
            graph._add_edge(
                eid, src_uid, edge.obs_str, dst_uid,
                extra=edge.extra, image_paths=edge.image_paths,
            )

    def _merge_graph(self, target: BaseGraph, source: BaseGraph) -> None:
        """Merge source into target with full dedup."""
        key_map: Dict[str, str] = {}
        for old_key, ndata in source._g.nodes(data=True):
            node = self._make_node_data(ndata)
            resolved = self._resolve_node(node, target)
            key_map[old_key] = resolved
            target._upsert_node(
                resolved, ndata["state"],
                obs_str=ndata.get("obs_str"),
                extra=ndata.get("extra", {}),
                image_paths=ndata.get("image_paths", []),
            )
        for u, v, _eid, data in source._g.edges(data=True, keys=True):
            new_u, new_v = key_map.get(u, u), key_map.get(v, v)
            edge = self._make_edge_data(data)
            eid = self._resolve_edge(new_u, new_v, edge, target)
            target._add_edge(
                eid, new_u, edge.obs_str, new_v,
                extra=edge.extra, image_paths=edge.image_paths,
            )

    # ── internal: build ──────────────────────────────────────────────────────

    def convert_files(self, files: List[Path], rollout_dir: Path, graph_dir: Path) -> None:
        if not files:
            return
        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        images_dir = graph_dir / "images"
        images_dir.mkdir(exist_ok=True)
        num_workers = int(self.config.get("num_workers", 1))
        if num_workers > 1:
            graph = self._build_parallel(files, rollout_dir, images_dir, num_workers)
        else:
            graph = self._build_sequential(files, rollout_dir, images_dir)
        graph.save(graph_dir)
        logger.info(
            "[%s] %d file(s) → graph (%s): %d nodes, %d edges",
            self.__class__.__name__, len(files), graph_dir,
            graph.num_nodes, graph.num_edges,
        )

    def _build_sequential(self, files: List[Path], rollout_dir: Path, images_dir: Path) -> BaseGraph:
        graph = BaseGraph()
        for f in files:
            if not f.exists():
                logger.warning("[%s] Missing: %s", self.__class__.__name__, f)
                continue
            try:
                step_idx = int(f.stem) if f.stem.isdigit() else 0
                with open(f, encoding="utf-8") as fh:
                    for line_idx, raw in enumerate(fh):
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            logger.warning("%s:%d JSON error: %s", f, line_idx, exc)
                            continue
                        messages = self._parse_vagen_line(data)
                        transitions = self.traj_to_transitions(
                            messages,
                            rollout_dir,
                            step_idx,
                            line_idx,
                            episode_data=data,
                        )
                        self._apply_transitions(graph, transitions, images_dir)
            except Exception as exc:
                logger.warning(
                    "[%s] Error processing %s: %s",
                    self.__class__.__name__, f, exc, exc_info=True,
                )
        return graph

    def _build_parallel(
        self, files: List[Path], rollout_dir: Path,
        images_dir: Path, num_workers: int,
    ) -> BaseGraph:
        chunks = [list(files[i::num_workers]) for i in range(num_workers)]
        sub_dicts: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(
                    _chunk_worker, [str(f) for f in chunk],
                    str(rollout_dir), str(images_dir),
                    self.__class__.__module__, self.__class__.__name__,
                    self.config,
                )
                for chunk in chunks if chunk
            ]
            for fut in as_completed(futures):
                try:
                    sub_dicts.append(fut.result())
                except Exception as exc:
                    logger.warning("[%s] Worker failed: %s", self.__class__.__name__, exc)
        graph = BaseGraph()
        for d in sub_dicts:
            self._merge_graph(graph, BaseGraph.from_dict(d))
        return graph

    # ── internal: images ─────────────────────────────────────────────────────

    def _copy_images(self, node: NodeData, unique_key: str, images_dir: Path) -> List[str]:
        source_images = getattr(node, "source_images", [])
        if not source_images:
            return getattr(node, "image_paths", [])
        images_dir.mkdir(parents=True, exist_ok=True)
        stored: List[str] = []
        for idx, src_str in enumerate(source_images):
            src = Path(src_str)
            suffix = src.suffix or ".jpg"
            dst_name = f"{unique_key}_{idx}{suffix}"
            dst = images_dir / dst_name
            if not dst.exists():
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    logger.warning("Source image missing: %s", src)
                    continue
            stored.append(f"images/{dst_name}")
        return stored

    # ── internal: VAGEN parsing ──────────────────────────────────────────────

    def _parse_vagen_line(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Parse VAGEN JSONL into conversation turns.

        Older rollout writers preserved ChatML delimiters.  Newer tokenizer
        versions decode with special tokens removed, leaving newline-delimited
        role markers (``system\n...\nuser\n...``).  Accept both representations so
        TrajToSFT can consume rollouts produced by either stack.
        """
        full = (data.get("input", "") + data.get("output", "")).replace("<|endoftext|>", "")
        pattern = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
        messages = [
            {"role": role, "content": content.strip()}
            for role, content in pattern.findall(full)
        ]
        if messages:
            return messages

        role_pattern = re.compile(r"^(system|user|assistant)\r?\n", re.MULTILINE)
        markers = list(role_pattern.finditer(full))
        return [
            {
                "role": marker.group(1),
                "content": full[
                    marker.end():markers[index + 1].start()
                    if index + 1 < len(markers)
                    else len(full)
                ].strip(),
            }
            for index, marker in enumerate(markers)
        ]
