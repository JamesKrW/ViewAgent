"""ViewSuite Interactive View Planning — NO-GRAPH ablation package.

Same distillation pipeline as ``viewsuite_interactive_view_planning`` but with
cross-trajectory node merging disabled (node identity scoped per trajectory).
Addressed by pipeline.yaml via the dotted path to
``NoGraphInteractiveViewPlanningTrajToSFT``.
"""

from .nograph_graph_builder import (  # noqa: F401
    NoGraphViewPlanningGraphBuilder,
    TrajScopedViewSuiteNodeData,
)
from .traj_to_sft import NoGraphInteractiveViewPlanningTrajToSFT  # noqa: F401

__all__ = [
    "NoGraphInteractiveViewPlanningTrajToSFT",
    "NoGraphViewPlanningGraphBuilder",
    "TrajScopedViewSuiteNodeData",
]
