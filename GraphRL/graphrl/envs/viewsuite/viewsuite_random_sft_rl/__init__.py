"""
ViewSuite Random-Action SFT → RL environment package.

Each iteration's TrajToSFT phase collects FRESH random-action trajectories
(via the VAGEN-SLIME evaluator) instead of using the just-finished RL
rollouts. The graph builder + dataset generators come from
``viewsuite_interactive_view_planning``; this env subclasses ``InteractiveViewPlanningTrajToSFT``
to override the data-source step.
"""

from .traj_to_sft import RandomActionTrajToSFT  # noqa: F401

__all__ = ["RandomActionTrajToSFT"]
