"""Within-trajectory hindsight relabeling for the ViewSuite IVP ablation."""

from __future__ import annotations

from typing import Dict, Type

from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.traj_to_sft import (
    InteractiveViewPlanningTrajToSFT,
)
from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder

from .nograph_graph_builder import NoGraphViewPlanningGraphBuilder


class NoGraphInteractiveViewPlanningTrajToSFT(
    InteractiveViewPlanningTrajToSFT
):
    """Reuse the SFT generators while restricting every path to one rollout."""

    name = "TrajToSFT(viewsuite_interactive_view_planning_nograph)"

    def graph_builder_class(self) -> Type[VagenGraphBuilder]:
        return NoGraphViewPlanningGraphBuilder

    def generate_datasets(self, graph, images_dir):
        datasets = super().generate_datasets(graph, images_dir)
        expected: Dict[str, int] = self.config.get("expected_records", {}) or {}
        mismatches = {
            name: {"expected": int(count), "actual": len(datasets.get(name, ([], None))[0])}
            for name, count in expected.items()
            if len(datasets.get(name, ([], None))[0]) != int(count)
        }
        if mismatches:
            raise RuntimeError(
                "no-graph SFT budget mismatch; refusing an unmatched ablation: "
                f"{mismatches}"
            )
        return datasets
