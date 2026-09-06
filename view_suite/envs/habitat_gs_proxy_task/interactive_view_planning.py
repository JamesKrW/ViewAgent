"""No-concat interactive view planning on Habitat-GS.

The task is to recover the 6-DoF camera pose of a target image. Every
observation is self-contained: it repeats the target image, a top-down reference
with its known pose, and every explored image/pose/action transition so far.

The navigation controls intentionally match Habitat-GS' interactive viewer:

    W/S             move forward/backward on the horizontal plane
    A/D             strafe left/right on the horizontal plane
    Z/X             move up/down along world Y
    arrow_left/right turn left/right (yaw)
    arrow_up/down    look up/down (pitch)

A turn is either a ``|``-separated navigation batch or exactly one
``submit_pose(tx,ty,tz,rx,ry,rz)`` action. Navigation and submission cannot be
mixed. Running this file directly starts a shell UI and saves all images plus a
JSON trajectory under ``--save-dir``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

# Support the exact invocation requested by the interactive demo:
#   python /path/to/view_suite/envs/.../interactive_view_planning.py
if __package__ in {None, ""}:
    _REPO_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT_FOR_SCRIPT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT_FOR_SCRIPT))

from view_suite.envs.habitat_gs_proxy_task.gym_proxy_tool import GymProxyTool
from view_suite.envs.utils.parse_utils import (
    FormatRegistry,
    ParsedAction,
    get_format_instruction,
    parse_actions,
)
from view_suite.envs.utils.scannet_utils import fmt_pose6_deg, parse_get_view_arg_deg


@dataclass(frozen=True)
class ExplorationFrame:
    """One rendered node in the trajectory returned to the model."""

    image: Image.Image
    pose: tuple[float, float, float, float, float, float]
    incoming_actions: tuple[str, ...] | None = None


class HabitatGSInteractiveViewPlanning(GymProxyTool):
    """Self-contained/no-concat IVP environment with viewer-style controls."""

    NAVIGATION_ACTIONS: tuple[str, ...] = (
        "w",
        "s",
        "a",
        "d",
        "z",
        "x",
        "arrow_left",
        "arrow_right",
        "arrow_up",
        "arrow_down",
    )
    SUBMIT_ACTION = "submit_pose"

    _ARROW_ALIASES: ClassVar[dict[str, str]] = {
        "←": "arrow_left",
        "⬅": "arrow_left",
        "→": "arrow_right",
        "➡": "arrow_right",
        "↑": "arrow_up",
        "⬆": "arrow_up",
        "↓": "arrow_down",
        "⬇": "arrow_down",
    }
    _ENGINE_ACTIONS: ClassVar[dict[str, str]] = {
        "w": "w",
        "s": "s",
        "a": "a",
        "d": "d",
        "arrow_left": "q",
        "arrow_right": "e",
        "arrow_up": "r",
        "arrow_down": "f",
    }

    def __init__(self, env_config: dict[str, Any]):
        # This task always starts at init_view and exposes only the compact action
        # space, regardless of the legacy GymProxyTool defaults.
        config = dict(env_config)
        config["action_only_mode"] = True
        config.setdefault("format", "eval_mode")
        config.setdefault("use_example_in_sys_prompt", False)
        super().__init__(config)

        self.max_actions_per_turn = int(config.get("max_actions_per_turn", 4))
        if self.max_actions_per_turn < 1:
            raise ValueError("max_actions_per_turn must be >= 1")

        self._episode_images: dict[str, Image.Image] = {}
        self._trajectory: list[ExplorationFrame] = []
        self._format_correct_turns = 0
        self._valid_action_turns = 0
        self._primitive_actions = 0
        self._submitted = False
        self._success = False
        self._terminal_reason: str | None = None

    @property
    def exploration_history(self) -> tuple[ExplorationFrame, ...]:
        """Read-only view of the frames included in the latest observation."""

        return tuple(self._trajectory)

    async def system_prompt(self) -> dict[str, Any]:
        format_instruction = get_format_instruction(
            self.format,
            action_example="w|arrow_left|d  OR  submit_pose(tx,ty,tz,rx,ry,rz)",
        )
        text = f"""
You are controlling a camera in a Habitat-GS scene.

TASK
The observation contains a TARGET VIEW whose camera pose is unknown. Infer and
submit that target camera pose as [tx, ty, tz, rx, ry, rz], where translation is
in meters and rotation is c2w Euler XYZ in degrees.

OBSERVATION
Every turn is self-contained. It contains:
1. the TARGET VIEW (pose hidden),
2. a TOP-DOWN REFERENCE and its camera pose,
3. the complete EXPLORED TRAJECTORY. Each explored image has its exact camera
   pose, and consecutive images are connected by the action batch that moved
   between them.

NAVIGATION ACTIONS
- w / s: move forward / backward on the horizontal plane.
- a / d: strafe left / right on the horizontal plane.
- z / x: move up / down along the world Y axis.
- arrow_left / arrow_right: turn left / right (yaw).
- arrow_up / arrow_down: look up / down (pitch).

TURN RULES
- A navigation turn contains 1 to {self.max_actions_per_turn} navigation actions,
  separated by |. The environment renders once after the whole batch.
- A submission turn contains exactly one action:
  submit_pose(tx,ty,tz,rx,ry,rz)
- Never mix navigation actions and submit_pose in the same turn.
- submit_pose is terminal, whether the estimate is correct or incorrect.
- You have at most {self.max_turns} turns.

OUTPUT FORMAT
{format_instruction}
""".strip()
        return {"obs_str": text}

    def _reset_ivp_runtime(self) -> None:
        self._episode_images.clear()
        self._trajectory.clear()
        self._format_correct_turns = 0
        self._valid_action_turns = 0
        self._primitive_actions = 0
        self._submitted = False
        self._success = False
        self._terminal_reason = None

    def _recover_loaded_images(self) -> dict[str, Image.Image]:
        """Map GymProxyTool's ordered image list back to dataset field names."""

        assert self.current_item is not None
        details = self.current_item.get("image_detail") or {}
        loaded_names = [
            name
            for name in ("target_view", "init_view", "top_down_view")
            if isinstance(details.get(name), dict) and details[name].get("path")
        ]
        if len(loaded_names) != len(self.images):
            raise ValueError(
                "Unexpected Habitat-GS datapoint: loaded image count does not "
                f"match image_detail ({len(self.images)} vs {len(loaded_names)})"
            )
        recovered = dict(zip(loaded_names, self.images))
        required = {"target_view", "init_view", "top_down_view"}
        missing = sorted(required - recovered.keys())
        if missing:
            raise ValueError(f"IVP datapoint is missing required images: {missing}")
        for name, image in recovered.items():
            if not isinstance(image, Image.Image):
                raise TypeError(f"Could not load {name} image")
        return recovered

    @staticmethod
    def _pose_tuple(
        pose: Sequence[float],
    ) -> tuple[float, float, float, float, float, float]:
        values = tuple(float(x) for x in pose)
        if len(values) != 6:
            raise ValueError(f"Expected a 6-DoF pose, got {len(values)} values")
        return values  # type: ignore[return-value]

    def _observation_payload(
        self, status: str | None = None
    ) -> tuple[str, list[Image.Image]]:
        if not self._episode_images or not self._trajectory:
            raise RuntimeError("Call reset() before building an IVP observation")

        topdown = self._named_views.get("top_down_view")
        if topdown is None:
            raise ValueError("IVP datapoint has no top_down_view pose")

        lines: list[str] = []
        images: list[Image.Image] = []
        lines.extend(
            [
                "TARGET VIEW (camera pose unknown)",
                "<image>",
                "",
                "TOP-DOWN REFERENCE",
                "<image>",
                "camera pose: " + fmt_pose6_deg(topdown["c2w_se3_deg"]),
                "",
                "EXPLORED TRAJECTORY",
            ]
        )
        images.extend(
            [
                self._episode_images["target_view"],
                self._episode_images["top_down_view"],
            ]
        )

        for index, frame in enumerate(self._trajectory):
            if index > 0:
                actions = " | ".join(frame.incoming_actions or ())
                lines.append(f"E{index - 1} --[{actions}]--> E{index}")
            label = " (initial view)" if index == 0 else ""
            lines.extend(
                [
                    f"E{index}{label}",
                    "<image>",
                    "camera pose: " + fmt_pose6_deg(frame.pose),
                ]
            )
            images.append(frame.image)

        pos_thr, ang_thr = self._current_thresholds()
        # Keep per-turn text after the stable target/reference/history prefix. In
        # no-concat rollouts, each new observation can then reuse the cached prefix
        # containing every frame that was already explored on the previous turn.
        if status:
            lines.extend(["", "LAST ACTION RESULT", status])
        lines.extend(
            [
                "",
                "EPISODE STATE",
                f"translation step: {self.step_translation:.4f} m",
                f"rotation step: {self.step_rotation_deg:.2f} degrees",
                f"success threshold: position <= {pos_thr:.4f} m, rotation <= {ang_thr:.2f} degrees",
                f"turns used: {self.current_step}/{self.max_turns}",
            ]
        )
        if self.episode_done:
            lines.append(f"episode finished: {self._terminal_reason or 'done'}")
        else:
            lines.append(f"next turn: {self.current_step + 1}/{self.max_turns}")
        return "\n".join(lines), images

    def _current_thresholds(self) -> tuple[float, float]:
        assert self.current_item is not None
        # _grade_answer_pose resolves the same fields. Keeping this small helper
        # local avoids exposing the hidden target pose while still telling the model
        # what precision is required.
        from view_suite.envs.scannet_proxy_task.utils.gym_proxy_tool_utils import (
            resolve_thresholds,
        )

        position, rotation = resolve_thresholds(
            self.current_item, self.tol_trans_l2_m, self.tol_rot_l2_deg
        )
        return float(position), float(rotation)

    def _full_observation(self, status: str | None = None) -> dict[str, Any]:
        text, images = self._observation_payload(status=status)
        return self._obs(text, images)

    def _metric_snapshot(self) -> dict[str, Any]:
        turns = self.current_step
        return {
            "success": bool(self._success),
            "format_compliance": (
                float(self._format_correct_turns / turns) if turns else 1.0
            ),
            "action_valid_rate": (
                float(self._valid_action_turns / turns) if turns else 1.0
            ),
            "primitive_actions": int(self._primitive_actions),
            "submitted": bool(self._submitted),
            "turns_used": int(turns),
        }

    def _with_metrics(self, info: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(info or {})
        metrics = self._metric_snapshot()
        merged.update(metrics)
        merged["metrics"] = dict(metrics)
        return merged

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self._reset_ivp_runtime()
        _, info = await super().reset(seed=seed)
        self._episode_images = self._recover_loaded_images()

        if not self._has_active_camera:
            raise RuntimeError("IVP reset did not activate the initial camera")
        initial_pose = self._pose_tuple(self.view_engine.get_se3(degrees=True))
        self._trajectory.append(
            ExplorationFrame(
                image=self._episode_images["init_view"],
                pose=initial_pose,
                incoming_actions=None,
            )
        )
        self.is_format_correct = True
        return self._full_observation(), self._with_metrics(info)

    @classmethod
    def _replace_arrow_glyphs(cls, text: str) -> str:
        for glyph, action in cls._ARROW_ALIASES.items():
            text = text.replace(glyph, action)
        return text

    def _parse_response(self, action_str: str) -> tuple[bool, list[ParsedAction], str]:
        normalized = self._replace_arrow_glyphs(action_str)
        try:
            formatted = FormatRegistry.parse(self.format, normalized)
        except ValueError as exc:
            return False, [], str(exc)
        if not formatted["ok"]:
            return False, [], f"response does not match format={self.format!r}"
        actions_ok, actions = parse_actions(formatted["actions_blob"])
        if not actions_ok or not actions:
            return False, [], "action block is empty or has invalid syntax"
        return True, actions, ""

    def _validate_batch(
        self, actions: Sequence[ParsedAction]
    ) -> tuple[str | None, tuple[float, ...] | None, str | None]:
        submissions = [
            action for action in actions if action.name == self.SUBMIT_ACTION
        ]
        if submissions:
            if len(actions) != 1:
                return None, None, "submit_pose must be the only action in its turn"
            arg = submissions[0].arg
            if not isinstance(arg, str):
                return None, None, "submit_pose requires 6 numeric arguments"
            pose = parse_get_view_arg_deg(arg)
            if (
                pose is None
                or len(pose) != 6
                or not all(math.isfinite(x) for x in pose)
            ):
                return (
                    None,
                    None,
                    "submit_pose requires 6 finite numbers: tx,ty,tz,rx,ry,rz",
                )
            return "submit", tuple(float(x) for x in pose), None

        if len(actions) > self.max_actions_per_turn:
            return (
                None,
                None,
                f"at most {self.max_actions_per_turn} navigation actions are allowed per turn",
            )
        for action in actions:
            if action.name not in self.NAVIGATION_ACTIONS:
                allowed = ", ".join((*self.NAVIGATION_ACTIONS, self.SUBMIT_ACTION))
                return None, None, f"unknown action {action.name!r}; allowed: {allowed}"
            if action.arg is not None:
                return (
                    None,
                    None,
                    f"navigation action {action.name!r} takes no arguments",
                )
        return "navigate", None, None

    def _execute_navigation_action(self, action: str) -> None:
        engine_action = self._ENGINE_ACTIONS.get(action)
        if engine_action is not None:
            self.view_engine.step(engine_action)
            return
        if action == "z":
            self.view_engine.pos[1] += float(self.step_translation)
            return
        if action == "x":
            self.view_engine.pos[1] -= float(self.step_translation)
            return
        raise ValueError(f"Unsupported navigation action: {action}")

    def _turn_limit_reached(self) -> bool:
        if self.current_step < self.max_turns:
            return False
        self.episode_done = True
        self._terminal_reason = "turn limit reached without submission"
        return True

    def _error_step(
        self, message: str, *, format_was_correct: bool
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.is_format_correct = False
        if format_was_correct:
            self._format_correct_turns += 1
        done = self._turn_limit_reached()
        status = f"ACTION ERROR: {message}"
        if done:
            status += "\nTURN LIMIT REACHED."
        info = self._with_metrics(
            {"success": False, "error": message, "truncated": done}
        )
        return self._full_observation(status), 0.0, done, info

    async def step(
        self, action_str: str
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.episode_done:
            info = self._with_metrics(
                {"success": self._success, "error": "episode_done", "truncated": False}
            )
            return self._full_observation("ACTION ERROR: episode_done"), 0.0, True, info
        if self.current_item is None:
            raise RuntimeError("Call reset() before step().")

        self.current_step += 1
        parse_ok, actions, parse_error = self._parse_response(action_str)
        if not parse_ok:
            return self._error_step(parse_error, format_was_correct=False)

        mode, submitted_pose, validation_error = self._validate_batch(actions)
        if validation_error is not None:
            return self._error_step(validation_error, format_was_correct=True)

        self._format_correct_turns += 1
        self._valid_action_turns += 1

        if mode == "submit":
            assert submitted_pose is not None
            success, metrics, _ = self._grade_answer_pose(
                submitted_pose, self.current_item
            )
            if not metrics:
                # This should only happen for a malformed datapoint because finite,
                # six-number submissions have already been validated.
                self._valid_action_turns -= 1
                return self._error_step(
                    "target pose is unavailable", format_was_correct=False
                )

            self._submitted = True
            self._success = bool(success)
            self.episode_done = True
            self._terminal_reason = "submission accepted"

            closeness = float(metrics.get("pose_closeness", 0.0) or 0.0)
            reward = self.give_answer_reward
            reward += self.answer_reward if success else 0.0
            reward += self.answer_close_reward * closeness
            if self.is_format_correct:
                reward += self.format_reward
            if self.single_action_reward:
                reward += self.single_action_reward

            answer_info = self._info_answer(success, metrics)
            answer_info["truncated"] = False
            status = (
                f"SUBMISSION RESULT: {'success' if success else 'incorrect'}\n"
                f"position error: {metrics['pos_err_m']:.4f} m\n"
                f"rotation error: {metrics['ang_err_deg']:.2f} degrees"
            )
            return (
                self._full_observation(status),
                float(reward),
                True,
                self._with_metrics(answer_info),
            )

        action_names = tuple(action.name for action in actions)
        start_state = self.view_engine.get_state()
        try:
            for action_name in action_names:
                self._execute_navigation_action(action_name)
            width, height = (
                self.image_size if self.image_size is not None else (512, 512)
            )
            rendered = await self._render_current(width=int(width), height=int(height))
        except Exception:
            # Batch semantics are atomic: a renderer failure must not leave the camera
            # in a partially-applied state that is absent from the returned history.
            self.view_engine.set_state(start_state)
            self._valid_action_turns -= 1
            raise

        pose = self._pose_tuple(self.view_engine.get_se3(degrees=True))
        self._trajectory.append(
            ExplorationFrame(
                image=rendered,
                pose=pose,
                incoming_actions=action_names,
            )
        )
        self._primitive_actions += len(action_names)

        reward = self.per_turn_format_reward
        if len(action_names) == 1 and self.single_action_reward:
            reward += self.single_action_reward

        done = self._turn_limit_reached()
        status = f"NAVIGATION OK: {' | '.join(action_names)}"
        if done:
            status += "\nTURN LIMIT REACHED."
        info = self._with_metrics({"success": False, "truncated": done})
        return self._full_observation(status), float(reward), done, info


# ---------------------------------------------------------------------------
# Direct shell demo
# ---------------------------------------------------------------------------

_SHELL_ALIASES = {
    "left": "arrow_left",
    "right": "arrow_right",
    "up": "arrow_up",
    "down": "arrow_down",
    "arrowleft": "arrow_left",
    "arrowright": "arrow_right",
    "arrowup": "arrow_up",
    "arrowdown": "arrow_down",
    **HabitatGSInteractiveViewPlanning._ARROW_ALIASES,
}

_TERMINAL_ARROW_KEYS = {
    "\x1b[D": "arrow_left",
    "\x1b[C": "arrow_right",
    "\x1b[A": "arrow_up",
    "\x1b[B": "arrow_down",
}


def _wrap_action_blob(blob: str, format_name: str) -> str:
    if format_name == "free_think":
        return f"<think>interactive shell</think><action>{blob}</action>"
    return f"<action>{blob}</action>"


def _shell_command_to_response(
    command: str, format_name: str
) -> tuple[str | None, str]:
    """Translate one viewer key (or one submission) to the model-facing format."""

    command = command.strip()
    if not command:
        return None, "empty command"

    if "<action>" in command.lower():
        return None, "raw <action> XML is disabled in interactive viewer mode"

    submit_match = re.fullmatch(
        r"(?:submit|submit_pose)\s*(?:\(([^()]*)\)|\s+(.+))",
        command,
        flags=re.IGNORECASE,
    )
    if submit_match:
        numbers = submit_match.group(1) or submit_match.group(2) or ""
        numbers = ",".join(
            part for part in re.split(r"[\s,]+", numbers.strip()) if part
        )
        return _wrap_action_blob(f"submit_pose({numbers})", format_name), ""

    raw = command.lower().strip()
    for glyph, name in HabitatGSInteractiveViewPlanning._ARROW_ALIASES.items():
        raw = raw.replace(glyph, f" {name} ")

    raw_tokens = [token for token in re.split(r"[|,\s]+", raw) if token]
    tokens: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[wasdzx]+", token):
            tokens.extend(token)
        else:
            tokens.append(token)

    normalized: list[str] = []
    allowed = set(HabitatGSInteractiveViewPlanning.NAVIGATION_ACTIONS)
    for token in tokens:
        token = _SHELL_ALIASES.get(token, token)
        if token not in allowed:
            return None, f"unknown shell action: {token!r}"
        normalized.append(token)
    if len(normalized) != 1:
        return None, "interactive viewer mode accepts exactly one navigation key"
    return _wrap_action_blob(normalized[0], format_name), ""


def _read_viewer_key() -> str:
    """Read one key immediately on a TTY; use one line when stdin is piped."""

    if not sys.stdin.isatty():
        return input("\nivp key> ").strip()

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    print("\nivp key> ", end="", flush=True)
    try:
        tty.setraw(fd)
        first = os.read(fd, 1)
        if first == b"\x03":
            raise KeyboardInterrupt

        sequence = first
        if first == b"\x1b":
            for _ in range(2):
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    break
                sequence += os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)

    sequence_text = sequence.decode("utf-8", errors="replace")
    if sequence_text in _TERMINAL_ARROW_KEYS:
        key = _TERMINAL_ARROW_KEYS[sequence_text]
    elif sequence_text in {"\r", "\n"} or sequence_text.lower() == "t":
        key = "submit"
    elif sequence_text.lower() == "p":
        key = "pose"
    elif sequence_text.lower() == "q":
        key = "quit"
    elif sequence_text.lower() == "h" or sequence_text == "?":
        key = "help"
    else:
        key = sequence_text.lower()

    print(key)
    return key


def _save_demo_state(
    env: HabitatGSInteractiveViewPlanning,
    run_dir: Path,
    turn_records: Sequence[dict[str, Any]],
) -> list[Path]:
    """Save newly available images and atomically refresh trajectory.json."""

    run_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    references = {
        "target.png": env._episode_images.get("target_view"),
        "top_down.png": env._episode_images.get("top_down_view"),
    }
    for filename, image in references.items():
        path = run_dir / filename
        if image is not None and not path.exists():
            image.save(path)
            saved.append(path)

    frames_json: list[dict[str, Any]] = []
    for index, frame in enumerate(env.exploration_history):
        filename = f"exploration_{index:03d}.png"
        path = run_dir / filename
        if not path.exists():
            frame.image.save(path)
            saved.append(path)
        frames_json.append(
            {
                "index": index,
                "image": filename,
                "incoming_actions": list(frame.incoming_actions or ()),
                "pose_c2w_euler_xyz_deg": list(frame.pose),
            }
        )

    item = env.current_item or {}
    document = {
        "scene_id": item.get("scene_id"),
        "sample_id": item.get("sample_id"),
        "jsonl_index": env.current_index,
        "translation_step_m": env.step_translation,
        "rotation_step_deg": env.step_rotation_deg,
        "frames": frames_json,
        "turns": list(turn_records),
    }
    trajectory_path = run_dir / "trajectory.json"
    temporary_path = run_dir / "trajectory.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(trajectory_path)
    return saved


def _demo_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Interactively inspect the no-concat Habitat-GS IVP environment."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=repo_root / "data/viewagent15k_habitat_gs/interactive_view_planning_test.jsonl",
    )
    parser.add_argument("--client-url", default="")
    parser.add_argument(
        "--client-url-file",
        type=Path,
        default=repo_root / "client_url_habitat_gs.txt",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=repo_root / "runs/habitat_gs_ivp_interactive",
        help="Base directory; each invocation creates a timestamped run subfolder.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-actions-per-turn", type=int, default=4)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(512, 512),
    )
    parser.add_argument(
        "--format",
        choices=("eval_mode", "no_think", "free_think"),
        default="eval_mode",
    )
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help=(
            "Accept a self-signed HTTPS certificate from the render service. "
            "Use only for a service you trust."
        ),
    )
    parser.add_argument("--show-system-prompt", action="store_true")
    return parser


def _print_shell_help() -> None:
    print(
        "Viewer controls (one key = one env turn = one render):\n"
        "  W / S         move forward / backward\n"
        "  A / D         strafe left / right\n"
        "  Z / X         move world-up / world-down\n"
        "  Arrow Left/Right   turn left / right\n"
        "  Arrow Up/Down      look up / down\n"
        "  Enter or T    submit a camera pose (then type six numbers)\n"
        "  P             print the current exact camera pose\n"
        "  H or ?        show this help\n"
        "  Q             quit\n"
        "Interactive mode deliberately rejects multi-key batches such as 'wwd' "
        "or 'w|left'."
    )


async def _run_interactive_demo(args: argparse.Namespace) -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = args.save_dir.expanduser().resolve() / f"seed_{args.seed}_{timestamp}"
    config: dict[str, Any] = {
        "jsonl_path": str(args.jsonl.expanduser().resolve()),
        "image_size": tuple(args.image_size),
        "render_size": tuple(args.image_size),
        "format": args.format,
        "max_turns": args.max_turns,
        "max_actions_per_turn": args.max_actions_per_turn,
        "action_only_mode": True,
        "use_example_in_sys_prompt": False,
    }
    if args.client_url:
        config["client_url"] = args.client_url
    else:
        config["client_url_file_path"] = str(
            args.client_url_file.expanduser().resolve()
        )

    if args.insecure_tls:
        os.environ["RENDER_TLS_NO_VERIFY"] = "1"

    env: HabitatGSInteractiveViewPlanning | None = None
    turn_records: list[dict[str, Any]] = []
    try:
        env = HabitatGSInteractiveViewPlanning(config)
        if args.show_system_prompt:
            system = await env.system_prompt()
            print("=" * 80)
            print(system["obs_str"])

        obs, info = await env.reset(seed=args.seed)
        saved = _save_demo_state(env, run_dir, turn_records)
        print("=" * 80)
        print(
            f"Habitat-GS IVP | scene={info.get('scene_id')} sample={info.get('sample_id')}"
        )
        print(f"Images and trajectory: {run_dir}")
        print(f"Saved: {', '.join(path.name for path in saved)}")
        print("=" * 80)
        print(obs["obs_str"])
        print()
        _print_shell_help()

        done = False
        while not done:
            try:
                command = _read_viewer_key()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            lowered = command.lower()
            if lowered in {"quit", "exit", "q"}:
                break
            if lowered in {"help", "?"}:
                _print_shell_help()
                continue
            if lowered == "pose":
                print(fmt_pose6_deg(env.view_engine.get_se3(degrees=True)))
                continue
            if lowered in {"submit", "t"}:
                try:
                    pose_text = input(
                        "submit pose [tx ty tz rx ry rz] (blank cancels)> "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not pose_text:
                    print("Submission cancelled.")
                    continue
                command = f"submit {pose_text}"

            response, error = _shell_command_to_response(command, args.format)
            if response is None:
                print(f"Input error: {error}")
                continue

            obs, reward, done, info = await env.step(response)
            record = {
                "turn": env.current_step,
                "shell_input": command,
                "model_response": response,
                "reward": reward,
                "done": done,
                "success": bool(info.get("success", False)),
                "error": info.get("error"),
            }
            turn_records.append(record)
            saved = _save_demo_state(env, run_dir, turn_records)

            print("-" * 80)
            print(obs["obs_str"])
            print(
                f"reward={reward:.3f} done={done} success={info.get('success', False)}"
            )
            if saved:
                print(f"Saved: {', '.join(path.name for path in saved)}")
            print(f"Trajectory: {run_dir / 'trajectory.json'}")

        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports infrastructure errors.
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            print(
                "The render endpoint uses a self-signed certificate. Re-run with "
                "--insecure-tls if you trust that endpoint.",
                file=sys.stderr,
            )
        print(
            "Check that the Habitat-GS render service is running and that "
            "--client-url/--client-url-file is correct.",
            file=sys.stderr,
        )
        return 1
    finally:
        if env is not None:
            await env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _demo_parser().parse_args(argv)
    return asyncio.run(_run_interactive_demo(args))


if __name__ == "__main__":
    raise SystemExit(main())
