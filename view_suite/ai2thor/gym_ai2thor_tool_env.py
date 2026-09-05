"""AI2-THOR tool env.

Adapted from view_suite.scannet.gym_scannet_tool_env.GymScannetToolEnv; only
two things differ from the scannet version:
  1. Inherits from GymAi2thorRenderEnv (HTTP AI2-THOR service) instead of
     GymScannetRenderEnv (ScanNet mesh/3DGS service).
  2. It uses the same AI2-THOR ViewManipulator as data generation, including
     Unity's Y-up pose convention.

All action semantics, parsing, and descriptions are identical so that prompt
templates, action vocabulary, and proxy-task grading code can be reused
without changes.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from view_suite.ai2thor.gym_ai2thor_render_env import GymAi2thorRenderEnv
from view_suite.ai2thor.view_manipulator import ViewManipulator
from view_suite.envs.utils.parse_utils import ParsedAction, FormatRegistry, parse_actions


class GymAi2thorToolEnv(GymAi2thorRenderEnv):
    """Tool-enabled env for AI2-THOR scenes (proxy-task flavor).

    Same camera-control action vocabulary as scannet/GymScannetToolEnv:
      move_{forward,backward,left,right,up,down}, turn_{left,right},
      look_{up,down}, rotate_{ccw,cw}, query_pose, select_view, get_view,
      answer.
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.step_translation = float(env_config.get("step_translation", 0.5))
        self.step_rotation_deg = float(env_config.get("step_rotation_deg", 30.0))
        self.is_discrete = bool(env_config.get("is_discrete", True))
        self.is_snap_every_step = bool(env_config.get("is_snap_every_step", True))
        self.image_y_down = bool(env_config.get("image_y_down", True))
        self.action_only_mode = bool(env_config.get("action_only_mode", False))
        self.ground_plane_movement = bool(
            env_config.get("ground_plane_movement", False)
        )
        self.view_engine = ViewManipulator(
            init_pose=None,
            step_translation=self.step_translation,
            step_rotation_deg=self.step_rotation_deg,
            pitch_limit_deg=float(env_config.get("pitch_limit_deg", 89.0)),
            roll_enabled=True,
            is_discrete=self.is_discrete,
            is_snap_every_step=self.is_snap_every_step,
            ground_plane_movement=self.ground_plane_movement,
        )

    # -------------------------
    # Tool prompt / action vocab
    # -------------------------
    @cached_property
    def _tool_instruction(self) -> str:
        lines = [
            "SUPPORTED ACTIONS",
            "-----------------",
            "Arguments are inside parentheses.",
            "",
        ]
        actions = self._action_only_allowed if self.action_only_mode else self._action_full
        lines += [f"- {name} : {self.action_description[name]}" for name in actions]
        instruction = "\n".join(lines).strip()

        if not self.action_only_mode:
            instruction += (
                "\n\nACTION ORDER CONSTRAINTS\n"
                "------------------------\n"
                "- You MUST call exactly one of:\n"
                "    - select_view(view_name), or\n"
                "    - get_view(tx, ty, tz, rx, ry, rz)\n"
                "before performing ANY of the following actions:\n"
                "    move_*, turn_*, look_*, rotate_*.\n\n"
                "- Calling move / turn / look / rotate before a view is selected\n"
                "is INVALID and will result in failure.\n\n"
                "- query_pose(...) does NOT count as selecting a view.\n\n"
                "- The episode terminates immediately after calling answer(...).\n"
                "No further actions are allowed.\n"
            )
        else:
            instruction += (
                "- The episode terminates immediately after calling answer(...).\n"
                "No further actions are allowed.\n"
            )
        if self.is_discrete:
            instruction += (
                "\nDISCRETE MODE\n"
                "-------------\n"
                f"- translation step: {self.step_translation} meters\n"
                f"- rotation step: {self.step_rotation_deg} degrees\n"
            )
            if self.is_snap_every_step:
                instruction += (
                    "\n(Note: after every rotation, the Euler angles (rx, ry, rz) are "
                    "rounded to the nearest integer multiples of the rotation step along each axis.)\n"
                )
        if self.ground_plane_movement:
            instruction += (
                "\nGROUND-PLANE MOVEMENT\n"
                "---------------------\n"
                "- Forward/backward follow yaw only on the horizontal XZ plane; "
                "looking up/down does not change that direction.\n"
                "- Left/right strafe perpendicular to the heading on the horizontal "
                "XZ plane.\n"
                "- Up/down move only along world +Y/-Y.\n"
                "- Turn left/right is yaw about world Y; look up/down is local pitch.\n"
            )
        else:
            instruction += (
                "\nCAMERA-LOCAL MOVEMENT (legacy)\n"
                "------------------------------\n"
                "- Translation follows the camera axes. After looking up/down, a "
                "forward move can change both horizontal position and height.\n"
            )
        return instruction

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
            "rotate_ccw": "t",
            "rotate_cw": "g",
        }

    @cached_property
    def _action_only_allowed(self) -> list[str]:
        return [
            "move_forward", "move_backward", "move_right", "move_left",
            "move_up", "move_down",
            "turn_left", "turn_right", "look_up", "look_down",
            "answer",
        ]

    @cached_property
    def _action_full(self) -> list[str]:
        return [
            "move_forward", "move_backward", "move_right", "move_left",
            "move_up", "move_down",
            "turn_left", "turn_right", "look_up", "look_down",
            "rotate_ccw", "rotate_cw",
            "query_pose", "select_view", "get_view", "answer",
        ]

    @cached_property
    def action_description(self) -> Dict[str, str]:
        if self.ground_plane_movement:
            forward = "move forward along the yaw-only horizontal heading"
            backward = "move backward along the yaw-only horizontal heading"
            right = "strafe right on the horizontal XZ plane"
            left = "strafe left on the horizontal XZ plane"
            up = "move along world +Y"
            down = "move along world -Y"
        else:
            forward = "move along camera-local forward"
            backward = "move along camera-local backward"
            right = "move along camera-local right"
            left = "move along camera-local left"
            up = "move along camera-local up"
            down = "move along camera-local down"
        return {
            "move_forward":  f"{forward} by {self.step_translation} meters.",
            "move_backward": f"{backward} by {self.step_translation} meters.",
            "move_right":    f"{right} by {self.step_translation} meters.",
            "move_left":     f"{left} by {self.step_translation} meters.",
            "move_up":       f"{up} by {self.step_translation} meters.",
            "move_down":     f"{down} by {self.step_translation} meters.",
            "turn_left":     f"yaw left by {self.step_rotation_deg} degrees.",
            "turn_right":    f"yaw right by {self.step_rotation_deg} degrees.",
            "look_up":       f"pitch up by {self.step_rotation_deg} degrees.",
            "look_down":     f"pitch down by {self.step_rotation_deg} degrees.",
            "rotate_ccw":    f"roll counter clockwise by {self.step_rotation_deg} degrees.",
            "rotate_cw":     f"roll clockwise by {self.step_rotation_deg} degrees.",
            "query_pose":    "query_pose(view_name), return the 6-DoF pose of a named view in DEGREES; does NOT change the camera.",
            "select_view":   "select_view(view_name), reset the camera to the named view and render an image.",
            "get_view":      "get_view(tx, ty, tz, rx, ry, rz), directly set the camera pose (c2w, Euler XYZ in DEGREES) and render an image.",
            "answer":        "answer(tx, ty, tz, rx, ry, rz), where tx, ty, tz are translation in meters and rx, ry, rz are rotation in degrees. All arguments must be positional plain numbers. This action is terminal and no further actions can be taken.",
        }

    @cached_property
    def _view_dict(self) -> Dict[str, Any]:
        return self._get_view_dict()

    # -------------------------
    # Action parse + execute
    # -------------------------
    def _parse_action_str(self, action_str: str, format: str = "free_think") -> Tuple[bool, List[ParsedAction]]:
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
                return {"success": True, "is_answer": False, "result": None, "need_render": True}
            except Exception as e:
                return {"success": False, "is_answer": False, "result": str(e), "need_render": False}

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
                    return {"success": True, "is_answer": False, "result": None, "need_render": True}
                except Exception as e:
                    return {"success": False, "is_answer": False, "result": str(e), "need_render": False}

            case "get_view":
                try:
                    self.view_engine.set_se3(action.arg, degrees=True)
                    return {"success": True, "is_answer": False, "result": None, "need_render": True}
                except Exception as e:
                    return {"success": False, "is_answer": False, "result": str(e), "need_render": False}

            case "answer":
                return {"success": True, "is_answer": True, "result": action.arg, "need_render": False}

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
