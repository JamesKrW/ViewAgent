"""No-graph (within-trajectory) variant of the ViewSuite IVP graph builder.

This removes the **cross-trajectory node merge** that defines view-graph
distillation. Node identity is scoped to the trajectory that produced it, so
identical poses seen in *different* rollouts never collapse into one shared
graph node. Each trajectory therefore stays an isolated chain, and the
(unchanged) path sampler can only draw paths *within a single trajectory* —
i.e. hindsight relabel-and-distill WITHOUT cross-trajectory graph composition.

What is kept identical to the graph builder:
  - transition extraction, edge (shorter-wins) policy, image quality filter,
    the refine / redundant-edge passes (these now operate per-trajectory only,
    since the graph is a disjoint union of per-trajectory sub-graphs),
  - the generators and their per-scene sampling budget: the sampling bucket
    still keys on the REAL scene id (``extra["scene_id"]``), so the number of
    samples per scene matches the graph run (then halved via config).

Only cross-trajectory pose dedup is disabled — exactly "remove the merge-node
operation" while "sampling still follows the original graph setup".
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from graphrl.traj_to_sft.utils.base_graph import NodeData, EdgeData
from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.interactive_view_planning_graph_builder import (
    InteractiveViewPlanningGraphBuilder,
    ViewSuiteNodeData,
)


class TrajScopedViewSuiteNodeData(ViewSuiteNodeData):
    """Node data whose identity is scoped to a single trajectory.

    Same pose in two different trajectories → two distinct nodes (no merge).
    ``extra["scene_id"]`` is left as the real scene id so downstream per-scene
    sampling is unaffected; only ``state["traj_uid"]`` gates dedup/merging.
    """

    def unique_key(self) -> str:
        p = self.state["pose"]
        pose_str = (
            f"{p['tx']:.4f}_{p['ty']:.4f}_{p['tz']:.4f}_"
            f"{p['rx']:.4f}_{p['ry']:.4f}_{p['rz']:.4f}"
        )
        traj_uid = self.state.get("traj_uid", "")
        raw = f"{self.state['scene_id']}|{traj_uid}|{pose_str}"
        pose_hash = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{self.state['scene_id']}_{traj_uid}_{pose_hash}"

    def bucket_key(self) -> str:
        # Restrict similarity comparison to nodes from the SAME trajectory,
        # so _resolve_node never merges poses across trajectories.
        return f"{self.state['scene_id']}|{self.state.get('traj_uid', '')}"

    def is_similar_to(self, other: "NodeData") -> bool:
        if not isinstance(other, TrajScopedViewSuiteNodeData):
            return False
        if self.state.get("traj_uid") != other.state.get("traj_uid"):
            return False
        return super().is_similar_to(other)


class NoGraphViewPlanningGraphBuilder(InteractiveViewPlanningGraphBuilder):
    """Graph builder with cross-trajectory node merging disabled."""

    def _make_node_data(self, ndata: Dict[str, Any]) -> NodeData:
        # state carries traj_uid (stamped at extraction), so re-created nodes
        # (e.g. cross-worker merge) keep their trajectory scope.
        return TrajScopedViewSuiteNodeData(
            state=ndata["state"],
            obs_str=ndata.get("obs_str"),
            image_paths=ndata.get("image_paths", []),
            extra=ndata.get("extra", {}),
        )

    @staticmethod
    def _to_scoped(node: NodeData, traj_uid: str) -> "TrajScopedViewSuiteNodeData":
        state = dict(node.state)
        state["traj_uid"] = traj_uid
        return TrajScopedViewSuiteNodeData(
            state=state,
            obs_str=node.obs_str,
            source_images=list(getattr(node, "source_images", []) or []),
            image_paths=list(getattr(node, "image_paths", []) or []),
            extra=dict(node.extra or {}),  # keeps real scene_id for sampling
        )

    def traj_to_transitions(
        self,
        messages: List[Dict[str, str]],
        rollout_dir: Path,
        step_idx: int,
        line_idx: int,
    ) -> List[Tuple[NodeData, EdgeData, NodeData]]:
        base = super().traj_to_transitions(messages, rollout_dir, step_idx, line_idx)
        traj_uid = f"{step_idx}_{line_idx}"  # unique per rollout line
        return [
            (self._to_scoped(src, traj_uid), edge, self._to_scoped(dst, traj_uid))
            for src, edge, dst in base
        ]
