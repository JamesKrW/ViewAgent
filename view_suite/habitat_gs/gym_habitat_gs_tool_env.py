"""Tool-enabled camera env for Habitat-GS scenes.

Mirrors GymAi2thorToolEnv, with two deliberate differences:

  1. The camera is a ``HabitatGSViewManipulator``, so actions mean what they mean in
     Habitat: turning is ground-parallel, forward motion is horizontal regardless of
     pitch. The same manipulator drives data generation, so the ground truth in the
     JSONL and the transitions the agent experiences here are the same function.

  2. **No roll.** Habitat's default agent has no roll axis, so rotate_ccw / rotate_cw
     are absent from the vocabulary rather than present-and-ignored -- an action the
     prompt offers but the world does not implement is a silent scoring hazard.
"""
from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Tuple

from view_suite.envs.utils.action_space_prompt import build_action_space_instruction
from view_suite.envs.utils.parse_utils import FormatRegistry, ParsedAction, parse_actions
from view_suite.habitat_gs.gym_habitat_gs_render_env import GymHabitatGSRenderEnv
from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator


class GymHabitatGSToolEnv(GymHabitatGSRenderEnv):
    """Camera-control action vocabulary over a Habitat-GS scene."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.step_translation = float(env_config.get("step_translation", 0.5))
        self.step_rotation_deg = float(env_config.get("step_rotation_deg", 30.0))
        self.is_discrete = bool(env_config.get("is_discrete", True))
        self.is_snap_every_step = bool(env_config.get("is_snap_every_step", True))
        self.pitch_limit_deg = float(env_config.get("pitch_limit_deg", 60.0))
        self.action_only_mode = bool(env_config.get("action_only_mode", False))
        self.ground_plane_movement = bool(
            env_config.get("ground_plane_movement", False)
        )
        self.view_engine = HabitatGSViewManipulator(
            step_translation=self.step_translation,
            step_rotation_deg=self.step_rotation_deg,
            pitch_limit_deg=self.pitch_limit_deg,
            discrete=self.is_discrete,
            is_snap_every_step=self.is_snap_every_step,
            ground_plane_movement=self.ground_plane_movement,
        )

    # -------------------------
    # Tool prompt / action vocab
    # -------------------------
    @cached_property
    def _keymap(self) -> Dict[str, str]:
        return {
            "move_forward": "w",
            "move_backward": "s",
            "move_right": "d",
            "move_left": "a",
            "move_up": "y",
            "move_down": "h",
            "turn_left": "q",
            "turn_right": "e",
            "look_up": "r",
            "look_down": "f",
        }

    @cached_property
    def _action_only_allowed(self) -> List[str]:
        return [
            "move_forward", "move_backward", "move_right", "move_left",
            "move_up", "move_down",
            "turn_left", "turn_right", "look_up", "look_down",
            "answer",
        ]

    @cached_property
    def _action_full(self) -> List[str]:
        return self._action_only_allowed[:-1] + [
            "query_pose", "select_view", "get_view", "answer",
        ]

    @cached_property
    def action_description(self) -> Dict[str, str]:
        vertical_up = ("move along world +Y" if self.ground_plane_movement
                       else "move along sensor-local up")
        vertical_down = ("move along world -Y" if self.ground_plane_movement
                         else "move along sensor-local down")
        return {
            "move_forward":  "move forward along the yaw-only horizontal heading.",
            "move_backward": "move backward along the yaw-only horizontal heading.",
            "move_left":     "strafe left on the horizontal XZ plane.",
            "move_right":    "strafe right on the horizontal XZ plane.",
            "move_up":       f"{vertical_up}.",
            "move_down":     f"{vertical_down}.",
            "turn_left":     "yaw left about world Y.",
            "turn_right":    "yaw right about world Y.",
            "look_up":       f"pitch up about sensor-local X (clamped to "
                             f"+/-{self.pitch_limit_deg} degrees).",
            "look_down":     f"pitch down about sensor-local X (clamped to "
                             f"+/-{self.pitch_limit_deg} degrees).",
            "query_pose":    "return the 6-DoF pose of a named view in DEGREES; does "
                             "NOT change the camera.",
            "select_view":   "reset the camera to the named view and render an image.",
            "get_view":      "directly set the camera pose (c2w, Euler XYZ in "
                             "DEGREES) and render an image.",
            # The spelled-out signature and the positional-only sentence are load-bearing:
            # with a vaguer description Qwen2.5-VL emitted answer(tx=..., ty=...) with
            # keyword arguments, the parser rejected every one of them, and IVP scored
            # 0/288 with 287 episodes running out of turns. The model was answering; the
            # prompt had not told it how.
            "answer":        "submit tx, ty, tz in meters and rx, ry, rz in degrees. "
                             "All arguments must be positional plain numbers. "
                             "This action is terminal and no further actions can be "
                             "taken.",
        }

    @cached_property
    def _tool_instruction(self) -> str:
        actions = self._action_only_allowed if self.action_only_mode else self._action_full
        if self.ground_plane_movement:
            mode_description = (
                "Habitat body movement; forward/backward/strafe are yaw-only and "
                "horizontal, while up/down use world Y"
            )
        else:
            mode_description = (
                "native Habitat viewer movement; forward/backward/strafe are yaw-only "
                "and horizontal, while up/down follow the pitched sensor"
            )
        # Per-scene Habitat steps can be awkward floats.  The corpus stores them at
        # two-decimal precision, so quote that same value in the prompt.
        return build_action_space_instruction(
            is_discrete=self.is_discrete,
            snap_rotations=self.is_discrete and self.is_snap_every_step,
            step_translation=str(round(self.step_translation, 2)),
            step_rotation_deg=str(round(self.step_rotation_deg, 2)),
            mode_description=mode_description,
            coordinate_description=(
                "horizontal plane is XZ; world up is +Y; roll is unavailable"
            ),
            snap_description=(
                "after a pose is initialized/set and after every rotation, controller "
                "yaw and pitch are rounded to the nearest multiples of the rotation step."
            ),
            actions=actions,
            action_descriptions=self.action_description,
            action_only_mode=self.action_only_mode,
            motion_wildcards="move_*, turn_*, or look_* action",
        )

    @cached_property
    def _view_dict(self) -> Dict[str, Any]:
        return self._get_view_dict()

    # -------------------------
    # Action parse + execute
    # -------------------------
    def _parse_action_str(
        self, action_str: str, format: str = "free_think"
    ) -> Tuple[bool, List[ParsedAction]]:
        is_no_think = (format == "no_think")
        ft = FormatRegistry.parse(format, action_str)
        if not ft["ok"]:
            return False, []
        actions_ok, parsed_actions = parse_actions(ft["actions_blob"])
        if not actions_ok:
            return (True, []) if is_no_think else (False, [])
        return True, parsed_actions

    def _execute_action(self, action: "ParsedAction") -> Dict[str, Any]:
        if self.action_only_mode and action.name not in self._action_only_allowed:
            return {"success": False, "is_answer": False,
                    "result": f"action not allowed in action_only_mode: {action.name}",
                    "need_render": False}

        if action.name in self._keymap:
            try:
                self.view_engine.step(self._keymap[action.name])
                return {"success": True, "is_answer": False, "result": None,
                        "need_render": True}
            except Exception as e:
                return {"success": False, "is_answer": False, "result": str(e),
                        "need_render": False}

        match action.name:
            case "query_pose":
                view = self._view_dict.get(action.arg)
                if not view:
                    return {"success": False, "is_answer": False,
                            "result": f"view not found: {action.arg}", "need_render": False}
                return {"success": True, "is_answer": False,
                        "result": view.get("c2w_se3_deg"), "need_render": False}

            case "select_view":
                view = self._view_dict.get(action.arg)
                if not view:
                    return {"success": False, "is_answer": False,
                            "result": f"view not found: {action.arg}", "need_render": False}
                try:
                    self.view_engine.reset(view.get("c2w_extrinsic"))
                    return {"success": True, "is_answer": False, "result": None,
                            "need_render": True}
                except Exception as e:
                    return {"success": False, "is_answer": False, "result": str(e),
                            "need_render": False}

            case "get_view":
                try:
                    self.view_engine.set_se3(action.arg, degrees=True)
                    return {"success": True, "is_answer": False, "result": None,
                            "need_render": True}
                except Exception as e:
                    return {"success": False, "is_answer": False, "result": str(e),
                            "need_render": False}

            case "answer":
                return {"success": True, "is_answer": True, "result": action.arg,
                        "need_render": False}

            case _:
                return {"success": False, "is_answer": False,
                        "result": f"unknown action: {action.name}", "need_render": False}

    # -------------------------
    # Abstracts (filled by GymProxyTool)
    # -------------------------
    @abstractmethod
    def _get_view_dict(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    @abstractmethod
    async def system_prompt(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...

    @abstractmethod
    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        ...
