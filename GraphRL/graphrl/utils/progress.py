"""
Progress detection for pipeline resume.

Scans the experiment directory for completed iteration phases and determines
where to resume execution.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphrl.adaptive.model import (
    is_complete_checkpoint_manifest,
    is_complete_hf_model,
)
from graphrl.adaptive.rollouts import durable_rollout_prefix
from graphrl.state import ModuleOutput

logger = logging.getLogger(__name__)


def _rl_phase_is_complete(iter_dir: Path) -> bool:
    """Validate a SLIME phase commit while retaining legacy-dir readability."""

    rl_dir = iter_dir / "rl"
    model_dir = rl_dir / "rl_model"
    if not is_complete_hf_model(model_dir):
        return False

    marker_path = rl_dir / ".slime_rl_done.json"
    is_slime_layout = (
        marker_path.exists()
        or (rl_dir / "slime_launch.json").exists()
        or (rl_dir / "slime_checkpoints").exists()
    )
    if not is_slime_layout:
        # A pre-migration experiment has no SLIME metadata. Keep its historical
        # model-only completion rule so old directories remain resumable.
        return True
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        rollout_id = int(marker["rollout_id"])
        num_rollout = int(marker["num_rollout"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    checkpoint = rl_dir / "slime_checkpoints" / f"iter_{rollout_id:07d}"
    return (
        rollout_id == num_rollout - 1
        and is_complete_checkpoint_manifest(checkpoint)
        and durable_rollout_prefix(rl_dir / "rollout_data") >= num_rollout
    )


def detect_progress(
    experiment_dir: Path,
    num_iterations: int,
) -> Tuple[int, int, Optional[ModuleOutput]]:
    """
    Detect the latest completed phase for pipeline resume.

    Iterations are 0-indexed (0 .. num_iterations-1).

    Scans ``iter_XXX/`` directories in reverse order and checks for
    well-known completion markers at each phase:

      - **SFT complete**: ``iter_XXX/sft/sft_model`` is a complete HF model
      - **TrajToSFT complete**: ``iter_XXX/traj_to_sft/sft_data/.phase_done`` exists
      - **RL complete**: a legacy model directory, or a complete SLIME phase
        marker backed by the terminal checkpoint and rollout prefix

    Returns:
        (start_iteration_idx, start_phase_idx, last_output)
    """

    for iter_idx in range(num_iterations - 1, -1, -1):
        iter_dir = experiment_dir / f"iter_{iter_idx:03d}"
        if not iter_dir.exists():
            continue

        # Phase 3 complete: SFT model ready
        sft_model_dir = iter_dir / "sft" / "sft_model"
        if is_complete_hf_model(sft_model_dir):
            output = ModuleOutput(model_path=str(sft_model_dir))
            if iter_idx >= num_iterations - 1:
                logger.info(f"Pipeline already complete (all {num_iterations} iterations done)")
                return num_iterations, 0, output
            logger.info(f"Resuming after iteration {iter_idx} (SFT complete)")
            return iter_idx + 1, 0, output

        # Phase 2 complete: SFT data ready -> resume at SFT (phase index 2).
        # We require the ``.phase_done`` marker (written at the end of
        # TrajToSFTModule.launch()) rather than just ``dataset_info.json``
        # — the latter appears halfway through any reasoning post-step,
        # which would let an interrupted reasoning step look "done" and
        # silently skip the rest on resume.
        sft_data_dir = iter_dir / "traj_to_sft" / "sft_data"
        rl_model_dir = iter_dir / "rl" / "rl_model"
        if (
            (sft_data_dir / ".phase_done").is_file()
            and is_complete_hf_model(rl_model_dir)
        ):
            output = ModuleOutput(
                model_path=str(rl_model_dir),
                data_paths={"sft_data": str(sft_data_dir)},
            )
            logger.info(f"Resuming iteration {iter_idx} at SFT phase (TrajToSFT complete)")
            return iter_idx, 2, output

        # Phase 1 complete: RL model ready -> resume at TrajToSFT (phase index 1)
        if _rl_phase_is_complete(iter_dir):
            graph_dir = iter_dir / "traj_to_sft" / "graph"
            trajs_dir = iter_dir / "trajs"
            output = ModuleOutput(
                model_path=str(rl_model_dir),
                data_paths={
                    "graph": str(graph_dir) if graph_dir.exists() else "",
                    "trajs": str(trajs_dir) if trajs_dir.exists() else "",
                },
            )
            logger.info(f"Resuming iteration {iter_idx} at TrajToSFT phase (RL complete)")
            return iter_idx, 1, output

    logger.info("No previous progress detected, starting from scratch")
    return 0, 0, None
