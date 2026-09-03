"""GraphRL: iterative RL → TrajToSFT → SFT pipeline.

Mono-backend by design — RL is VAGEN/verl, SFT is LLaMA-Factory.
The only user-extension point is TrajToSFT: subclass one of the four
TrajToSFT base classes (always single inheritance, pick the most specific
one that fits your use case) and point ``traj_to_sft.module`` in
pipeline.yaml at its dotted path::

    TrajToSFTModule              ← override run()
    TrajToSFTGraphBase           ← override graph_builder_class + generate_datasets
    TrajToSFTReasoningBase       ← override generate(); reasoning post-step added
    TrajToSFTGraphReasoningBase  ← graph + reasoning post-step
"""

from graphrl.state import ModuleOutput, ModuleState
from graphrl.vagen.vagen_wrapper import VagenWrapper
from graphrl.llama_factory.lf_wrapper import LFWrapper
from graphrl.traj_to_sft.traj_to_sft_base import (
    TrajToSFTModule,
    TrajToSFTPaths,
    load_traj_to_sft_class,
)
from graphrl.traj_to_sft.traj_to_sft_graph_base import TrajToSFTGraphBase
from graphrl.traj_to_sft.traj_to_sft_reasoning_base import TrajToSFTReasoningBase
from graphrl.traj_to_sft.traj_to_sft_graph_reasoning_base import (
    TrajToSFTGraphReasoningBase,
)

__all__ = [
    "VagenWrapper",
    "LFWrapper",
    "TrajToSFTModule",
    "TrajToSFTGraphBase",
    "TrajToSFTReasoningBase",
    "TrajToSFTGraphReasoningBase",
    "TrajToSFTPaths",
    "load_traj_to_sft_class",
    "ModuleOutput",
    "ModuleState",
]
