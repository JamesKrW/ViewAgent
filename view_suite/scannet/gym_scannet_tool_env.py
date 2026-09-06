
from typing import Dict, Any
from typing import List
from abc import abstractmethod
from view_suite.scannet.gym_scannet_render_env import GymScannetRenderEnv
from view_suite.scannet.view_manipulator import ViewManipulator
from typing import Tuple, List
from dataclasses import dataclass
from typing import Optional
from functools import cached_property
from view_suite.envs.utils.action_space_prompt import build_action_space_instruction
from view_suite.envs.utils.parse_utils import ParsedAction, FormatRegistry, parse_actions





class GymScannetToolEnv(GymScannetRenderEnv):
    """
    Tool-enabled single-turn QA with exploration.

    This env enables the agent to explore the scannet scene and answer the question.

    The agent can explore the scene by issuing camera-control actions.
    Always respond using the free_think format:
    <think>...</think><action>action1|action2|...|</action>

    IMPORTANT CONVENTION:
    - All angles YOU input or see in the observations are in DEGREES.
    - The environment internally uses radians and extrinsic matrices, but you do not need to convert them.

    Supported actions (arguments are inside parentheses):
    - move_forward / move_backward: move along the configured forward basis.
    - move_right / move_left: strafe along the configured right basis.
    - move_up      : move up by a fixed step (meters).
    - move_down    : move down by a fixed step (meters).
    - turn_left    : yaw left by a fixed angle (degrees).
    - turn_right   : yaw right by a fixed angle (degrees).
    - look_up      : pitch up by a fixed angle (degrees).
    - look_down    : pitch down by a fixed angle (degrees).
    - rotate_ccw   : rotate counter-clockwise for your current view by a fixed angle (degrees).
    - rotate_cw    : rotate clockwise for your current view by a fixed angle (degrees).
    - query_pose(view_name) : return the 6-DoF pose of the named view in DEGREES; does NOT change the camera.
    - select_view(view_name): reset the camera to the named view and render an image.
    - get_view(tx,ty,tz,rx,ry,rz): directly set camera w2c pose with Euler XYZ in DEGREES and render.
    - answer(X) where X in {A,B,C,D}: submit your final answer and terminate the episode.
    """
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.step_translation = float(env_config.get("step_translation", 0.5))
        self.step_rotation_deg = float(env_config.get("step_rotation_deg", 30.0))
        self.is_discrete = bool(env_config.get("is_discrete", True))
        self.is_snap_every_step = bool(env_config.get("is_snap_every_step", True))
        self.image_y_down = bool(env_config.get("image_y_down", True))
        self.action_only_mode = bool(env_config.get("action_only_mode", False))
        self.allow_rotate = bool(env_config.get("allow_rotate", True))
        self.ground_plane_movement = bool(
            env_config.get("ground_plane_movement", False)
        )
        self.view_engine = ViewManipulator(
            step_translation=self.step_translation,
            step_rotation_deg=self.step_rotation_deg,
            world_up_axis="Z",
            is_discrete=self.is_discrete,
            is_snap_every_step=self.is_snap_every_step,
            image_y_down=self.image_y_down,
            ground_plane_movement=self.ground_plane_movement,
        )

    @cached_property
    def _tool_instruction(self) -> str:
        actions = self._action_only_allowed if self.action_only_mode else self._action_full
        if self.ground_plane_movement:
            mode_description = (
                "ground-aligned body movement; camera tilt changes the view but does "
                "not tilt translation directions"
            )
        else:
            mode_description = (
                "full camera-local translation; after looking up/down, forward can "
                "have both horizontal and vertical components"
            )
        return build_action_space_instruction(
            is_discrete=self.is_discrete,
            snap_rotations=self.is_discrete and self.is_snap_every_step,
            step_translation=str(self.step_translation),
            step_rotation_deg=str(self.step_rotation_deg),
            mode_description=mode_description,
            coordinate_description=(
                "horizontal plane is XY; world up is +Z; roll actions are "
                + ("enabled" if self.allow_rotate else "disabled")
            ),
            snap_description=(
                "after a pose is initialized/set and after every rotation, the c2w "
                "Euler XYZ angles are rounded to the nearest multiples of the "
                "rotation step."
            ),
            actions=actions,
            action_descriptions=self.action_description,
            action_only_mode=self.action_only_mode,
            motion_wildcards=(
                "move_*, turn_*, look_*, or rotate_* action"
                if self.allow_rotate
                else "move_*, turn_*, or look_* action"
            ),
        )


    @cached_property
    def _keymap(self)->Dict[str, str]:
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

    @property
    def _action_only_allowed(self) -> tuple[str, ...]:
        if self.allow_rotate:
            return (
                "move_forward",
                "move_backward",
                "move_right",
                "move_left",
                "move_up",
                "move_down",
                "turn_left",
                "turn_right",
                "look_up",
                "look_down",
                "rotate_cw",
                "rotate_ccw",
                "answer",
            )
        else:
            return (
                "move_forward",
                "move_backward",
                "move_right",
                "move_left",
                "move_up",
                "move_down",
                "turn_left",
                "turn_right",
                "look_up",
                "look_down",
                "answer",
            )

    @property
    def _action_full(self) -> tuple[str, ...]:
        actions = (
            "move_forward",
            "move_backward",
            "move_right",
            "move_left",
            "move_up",
            "move_down",
            "turn_left",
            "turn_right",
            "look_up",
            "look_down",
        )
        if self.allow_rotate:
            actions += ("rotate_ccw", "rotate_cw")
        return actions + (
            "query_pose",
            "select_view",
            "get_view",
            "answer",
        )

    @cached_property
    def action_description(self):
        if self.ground_plane_movement:
            forward = "move forward along the horizontal heading"
            backward = "move backward along the horizontal heading"
            right = "strafe right on the horizontal plane"
            left = "strafe left on the horizontal plane"
            up = "move along world +Z"
            down = "move along world -Z"
        else:
            forward = "move along camera-local forward"
            backward = "move along camera-local backward"
            right = "move along camera-local right"
            left = "move along camera-local left"
            up = "move along camera-local screen-up"
            down = "move along camera-local screen-down"
        return {
            "move_forward": f"{forward}.",
            "move_backward": f"{backward}.",
            "move_right": f"{right}.",
            "move_left": f"{left}.",
            "move_up": f"{up}.",
            "move_down": f"{down}.",
            "turn_left": "yaw left about world Z." if self.ground_plane_movement
                         else "yaw left about camera-local Y.",
            "turn_right": "yaw right about world Z." if self.ground_plane_movement
                          else "yaw right about camera-local Y.",
            "look_up": "pitch up about camera-local X.",
            "look_down": "pitch down about camera-local X.",
            "rotate_ccw": "roll counter-clockwise about the camera view axis.",
            "rotate_cw": "roll clockwise about the camera view axis.",
            "query_pose": "return the 6-DoF pose of a named view in DEGREES; does NOT change the camera.",
            "select_view": "reset the camera to the named view and render an image.",
            "get_view": "directly set the camera pose (c2w, Euler XYZ in DEGREES) and render an image.",
            "answer": "submit tx, ty, tz in meters and rx, ry, rz in degrees. All arguments must be positional plain numbers. This action is terminal and no further actions can be taken.",
        }
        
    
        
    @cached_property
    def _view_dict(self)->Dict[str, Any]:
        return self._get_view_dict()




    def _parse_action_str(self, action_str: str, format: str = "free_think") -> Tuple[bool, List[ParsedAction]]:
        """
        Returns:
        bool: True if the action string is valid, False otherwise
        List[ParsedAction]: The parsed actions

        Args:
        action_str: The action string to parse
        format: One of "free_think", "eval_mode", "no_think"
        """
        is_no_think = (format == "no_think")
        ft = FormatRegistry.parse(format, action_str)
        if not ft["ok"]:
            return False, []
        actions_ok, parsed_actions = parse_actions(ft["actions_blob"])
        if not actions_ok:
            return (True, []) if is_no_think else (False, [])
        return True, parsed_actions




    def _execute_action(self, action: "ParsedAction") -> Dict[str, Any]:
        """
        Returns:
        {
            "success": bool,
            "is_answer": bool,
            "result": Any,
            "need_render": bool,
        }
        """

        if self.action_only_mode and action.name not in self._action_only_allowed:
            return {
                "success": False,
                "is_answer": False,
                "result": f"action not allowed in action_only_mode: {action.name}",
                "need_render": False,
            }

        if not self.allow_rotate and action.name in {"rotate_ccw", "rotate_cw"}:
            return {
                "success": False,
                "is_answer": False,
                "result": f"action disabled by allow_rotate=false: {action.name}",
                "need_render": False,
            }

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
                    return {"success": False, "is_answer": False, "result": f"view not found: {action.arg}", "need_render": False}
                return {"success": True, "is_answer": False, "result": view.get("c2w_se3_deg"), "need_render": False}

            case "select_view":
                view = self._view_dict.get(action.arg)
                if not view:
                    return {"success": False, "is_answer": False, "result": f"view not found: {action.arg}", "need_render": False}
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
                return {"success": False, "is_answer": False, "result": f"unknown action: {action.name}", "need_render": False}

    @abstractmethod
    def _get_view_dict(self)->Dict[str, Any]:
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
