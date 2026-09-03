"""
Graph builder for Sokoban text environment.

Extends VagenGraphBuilder — only needs to implement traj_to_transitions().
Plain utility class; no registry. Instantiate it directly inside
``SokobanTextTrajToSFT.run()``.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder, VagenNodeData, VagenEdgeData
from graphrl.traj_to_sft.utils.base_graph import NodeData, EdgeData

logger = logging.getLogger(__name__)

# ── grid / action extraction ─────────────────────────────────────────────────

_INITIAL_OBS_RE = re.compile(
    r"\[Initial Observation\]:\s*\n(.*?)\nDecide your next action",
    re.DOTALL,
)
_AFTER_OBS_RE = re.compile(
    r"After that, the observation is:\s*\n(.*?)\nDecide your next action",
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _normalize_grid(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _extract_grid(content: str, initial: bool) -> Optional[str]:
    m = (_INITIAL_OBS_RE if initial else _AFTER_OBS_RE).search(content)
    return _normalize_grid(m.group(1)) if m else None


def _extract_action(content: str) -> Optional[str]:
    m = _ANSWER_RE.search(content)
    return m.group(1).strip() if m else None


# ── graph builder ─────────────────────────────────────────────────────────────

class SokobanTextGraphBuilder(VagenGraphBuilder):
    """
    Converts VAGEN Sokoban rollout files into a BaseGraph.

    For each episode, extracts (grid_state, action_sequence, next_grid_state)
    transitions from the conversation.  Both successful and failed episodes
    contribute — the graph captures all explored transitions.

    Config keys (inherited from VagenGraphBuilder):
        num_workers (int, default 1): parallel worker processes.
    """

    def traj_to_transitions(
        self,
        messages: List[dict],
        _rollout_dir: Path,
        _step_idx: int,
        _line_idx: int,
        episode_data: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[NodeData, EdgeData, NodeData]]:
        """
        Extract (src_grid, action, dst_grid) from one Sokoban episode.

        Processes the conversation sequentially to correctly handle turns where
        the agent failed to output a parseable action.  In that case the
        transition is skipped but the new observation still becomes the
        current state, so later valid transitions are not misaligned.

        Example with a missing action at step 1:
          user s0 → assistant ∅ → user s1 → assistant a1 → user s2
          ⇒ only (s1, a1, s2) is emitted  (not the broken (s0, ∅, s1))
        """
        del episode_data
        transitions: List[Tuple[NodeData, EdgeData, NodeData]] = []
        current_state: Optional[str] = None
        pending_action: Optional[str] = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                is_first = current_state is None
                grid = _extract_grid(content, initial=is_first)
                if grid is None and is_first:
                    grid = _extract_grid(content, initial=False)

                if grid is not None:
                    if current_state is not None and pending_action is not None:
                        transitions.append((
                            VagenNodeData(state=current_state, obs_str=current_state),
                            VagenEdgeData(obs_str=pending_action),
                            VagenNodeData(state=grid, obs_str=grid),
                        ))
                    # Advance state regardless — keeps later pairs aligned.
                    current_state = grid
                    pending_action = None

            elif role == "assistant":
                action = _extract_action(content)
                if action:
                    pending_action = action

        return transitions
