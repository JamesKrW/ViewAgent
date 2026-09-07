"""Trajectory-isolated ViewSuite graph builder.

The graph object here is only a compatibility container for the existing SFT
generators.  Node identity includes a rollout id, so paths can never cross,
merge, or compose information between trajectories.  Sampling from this graph
is therefore equivalent to sampling hindsight goals directly on each rollout.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.interactive_view_planning_graph_builder import (
    InteractiveViewPlanningGraphBuilder,
    ViewSuiteNodeData,
)
from graphrl.traj_to_sft.utils.base_graph import EdgeData, NodeData


class TrajScopedViewSuiteNodeData(ViewSuiteNodeData):
    """A view whose identity and similarity bucket are local to one rollout."""

    def unique_key(self) -> str:
        pose = self.state["pose"]
        pose_str = (
            f"{pose['tx']:.4f}_{pose['ty']:.4f}_{pose['tz']:.4f}_"
            f"{pose['rx']:.4f}_{pose['ry']:.4f}_{pose['rz']:.4f}"
        )
        traj_uid = self.state.get("traj_uid", "")
        raw = f"{self.state['scene_id']}|{traj_uid}|{pose_str}"
        pose_hash = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{self.state['scene_id']}_{traj_uid}_{pose_hash}"

    def bucket_key(self) -> str:
        return f"{self.state['scene_id']}|{self.state.get('traj_uid', '')}"

    def is_similar_to(self, other: NodeData) -> bool:
        return (
            isinstance(other, TrajScopedViewSuiteNodeData)
            and self.state.get("traj_uid") == other.state.get("traj_uid")
            and super().is_similar_to(other)
        )


class NoGraphViewPlanningGraphBuilder(InteractiveViewPlanningGraphBuilder):
    """Keep every rollout as an isolated chain with no cross-rollout merge."""

    def _make_node_data(self, ndata: Dict[str, Any]) -> NodeData:
        return TrajScopedViewSuiteNodeData(
            state=ndata["state"],
            obs_str=ndata.get("obs_str"),
            image_paths=ndata.get("image_paths", []),
            extra=ndata.get("extra", {}),
        )

    @staticmethod
    def _to_scoped(
        node: NodeData, traj_uid: str
    ) -> TrajScopedViewSuiteNodeData:
        state = dict(node.state)
        state["traj_uid"] = traj_uid
        return TrajScopedViewSuiteNodeData(
            state=state,
            obs_str=node.obs_str,
            source_images=list(getattr(node, "source_images", []) or []),
            image_paths=list(getattr(node, "image_paths", []) or []),
            extra=dict(node.extra or {}),
        )

    def traj_to_transitions(
        self,
        messages: List[Dict[str, str]],
        rollout_dir: Path,
        step_idx: int,
        line_idx: int,
        episode_data: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[NodeData, EdgeData, NodeData]]:
        base = super().traj_to_transitions(
            messages,
            rollout_dir,
            step_idx,
            line_idx,
            episode_data=episode_data,
        )
        episode_data = episode_data or {}
        external_id = episode_data.get("episode_id") or episode_data.get(
            "conversation_id", ""
        )
        traj_uid = f"{step_idx}_{line_idx}_{external_id}"
        return [
            (self._to_scoped(src, traj_uid), edge, self._to_scoped(dst, traj_uid))
            for src, edge, dst in base
        ]
