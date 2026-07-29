"""No-graph (within-trajectory) TrajToSFT for ViewSuite Interactive View Planning.

Identical to :class:`InteractiveViewPlanningTrajToSFT` except it swaps in a graph
builder that disables cross-trajectory node merging
(:class:`NoGraphViewPlanningGraphBuilder`). The 7 generators, dataset formats,
and per-scene sampling logic are inherited unchanged — so this is a clean
"graph vs. no-graph" ablation: relabel-and-distill from each trajectory in
isolation, WITHOUT composing sub-paths across trajectories.

Pipeline.yaml::

    traj_to_sft:
      module: graphrl.envs.viewsuite.viewsuite_interactive_view_planning_nograph.NoGraphInteractiveViewPlanningTrajToSFT
      generators: [multi_turn_action_gen, view_difference, view_difference_mcq]
      # sample_per_scene halved vs. the graph run (see pipeline.yaml here)
"""
from __future__ import annotations

import logging
from typing import Type

from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder
from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.traj_to_sft import (
    InteractiveViewPlanningTrajToSFT,
)
from .nograph_graph_builder import NoGraphViewPlanningGraphBuilder

logger = logging.getLogger(__name__)


class NoGraphInteractiveViewPlanningTrajToSFT(InteractiveViewPlanningTrajToSFT):
    """View-graph distillation with cross-trajectory composition disabled."""

    name = "TrajToSFT(viewsuite_interactive_view_planning_nograph)"

    def graph_builder_class(self) -> Type[VagenGraphBuilder]:
        return NoGraphViewPlanningGraphBuilder
