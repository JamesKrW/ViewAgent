"""ReasoningEnv: drives a single SFT datapoint through one-shot generation
+ feedback-driven refinement.

Episode layout (seed = datapoint index):
  reset(seed)
    obs_str   = serialized original conversation + rules (all `<image>` placeholders preserved)
    images    = all datapoint images concatenated in original order
  step(reply)
    turn 1: parse + check reply; if all good → done, augmented dict stored in info.
    turn 2..max_turns: user message contains per-turn feedback; model re-outputs
      the full set of augmented turns. Same check runs again.

The number of episode turns used controls the refinement budget
(``max_turns=1`` → pure one-shot; ``max_turns=3`` → one-shot + up to 2 refines).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from view_suite.gym.gym_image_env import GymImageEnv

from .base import BaseChecker, BaseDataset, Datapoint, import_by_path


class ReasoningEnv(GymImageEnv):
    """env_config keys:
        sft_path (required): input SFT JSON.
        image_root, image_size: forwarded to the dataset.
        system_prompt_path, user_prompt_path (required): .md files.
        checker_cls (default the ObsActionChecker shipped with this package): checker FQCN.
        checker_kwargs (dict, optional): passed to the checker constructor.
        dataset_cls (default the ShareGPTDataset shipped with this package): dataset FQCN.
        dataset_kwargs (dict, optional): passed to the dataset constructor.
    """

    _PKG = "graphrl.traj_to_sft.self_reasoning"

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self._sys_text = Path(env_config["system_prompt_path"]).read_text(encoding="utf-8")
        self._rules_text = Path(env_config["user_prompt_path"]).read_text(encoding="utf-8")

        ds_cls = import_by_path(env_config.get("dataset_cls", f"{self._PKG}.dataset.ShareGPTDataset"))
        ds_kwargs = dict(env_config.get("dataset_kwargs", {}))
        ds_kwargs.setdefault("sft_path", env_config["sft_path"])
        ds_kwargs.setdefault("image_root", env_config.get("image_root"))
        ds_kwargs.setdefault("image_size", env_config.get("image_size"))
        self._dataset: BaseDataset = ds_cls(**ds_kwargs)

        ck_cls = import_by_path(env_config.get("checker_cls", f"{self._PKG}.checker.ObsActionChecker"))
        self._checker: BaseChecker = ck_cls(**(env_config.get("checker_kwargs") or {}))

        self._dp: Datapoint | None = None
        self._turn_no: int = 0
        self._augmented: List[str] | None = None

    async def close(self) -> None:
        self._dp = None

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": self._sys_text}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert seed is not None, "reset(seed) requires a seed"
        idx = seed % len(self._dataset)
        self._dp = self._dataset.get(idx)
        self._turn_no = 0
        self._augmented = None

        obs_str = self._serialize(self._dp) + "\n\n" + self._rules_text
        obs: Dict[str, Any] = {"obs_str": obs_str}
        if self._dp.images:
            obs["multi_modal_input"] = {"<image>": list(self._dp.images)}
        info = {
            "sft_idx": idx,
            "n_assistant_turns": len(self._dp.assistant_texts),
            "n_images": len(self._dp.images),
            "success": False,
        }
        return obs, info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self._dp is not None, "step() called before reset()"
        self._turn_no += 1

        results = self._checker.check(action_str, self._dp.assistant_texts)
        per_turn = [r.__dict__ for r in results]
        ok = self._checker.all_ok(results)

        if ok:
            self._augmented = [r.augmented for r in results]
            info = {
                "turn": self._turn_no,
                "per_turn": per_turn,
                "augmented": self._augmented,
                "sft_idx": self._dp.idx,
                "success": True,
            }
            return {"obs_str": "All turns valid. Done."}, 1.0, True, info

        feedback = self._checker.feedback(results)
        info = {
            "turn": self._turn_no,
            "per_turn": per_turn,
            "sft_idx": self._dp.idx,
            "success": False,
        }
        return {"obs_str": feedback}, 0.0, False, info

    # ------------------------------------------------------------------
    @staticmethod
    def _serialize(dp: Datapoint) -> str:
        parts: List[str] = ["=== ORIGINAL CONVERSATION ==="]
        u = a = 0
        for m in dp.messages:
            role = m["role"]
            text = m["content"]
            if role == "system":
                parts.append(f"[SYSTEM]\n{text}")
            elif role == "user":
                u += 1
                parts.append(f"[USER {u}]\n{text}")
            elif role == "assistant":
                a += 1
                parts.append(f"[ASSISTANT {a}]\n{text}")
        return "\n\n".join(parts)
