from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from view_suite.envs.habitat_gs_proxy_task.interactive_view_planning import (
    HabitatGSInteractiveViewPlanning,
    _shell_command_to_response,
)
from view_suite.habitat_gs.pose_utils import intrinsics_from_fov
from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator


class HabitatGSInteractiveViewPlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.jsonl_path = self._write_datapoint(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _camera_pose(position, yaw=0.0, pitch=0.0):
        camera = HabitatGSViewManipulator(
            position=position,
            yaw_deg=yaw,
            pitch_deg=pitch,
            step_translation=0.5,
            step_rotation_deg=30.0,
        )
        return camera.get_c2w().tolist()

    @classmethod
    def _write_datapoint(cls, root: Path) -> Path:
        colors = {
            "target_view": (220, 20, 20),
            "init_view": (20, 220, 20),
            "top_down_view": (20, 20, 220),
        }
        poses = {
            "target_view": cls._camera_pose((1.0, 1.5, -1.0), yaw=30.0),
            "init_view": cls._camera_pose((0.0, 1.5, 0.0)),
            "top_down_view": cls._camera_pose((0.0, 8.0, 0.0), pitch=-60.0),
        }
        intrinsics = intrinsics_from_fov(32, 32, 90.0).tolist()
        details = {}
        for name, color in colors.items():
            filename = f"{name}.png"
            Image.new("RGB", (32, 32), color).save(root / filename)
            details[name] = {
                "path": filename,
                "c2w_extrinsics": poses[name],
                "c2w_intrinsics": intrinsics,
            }
        row = {
            "scene_id": "test_scene",
            "sample_id": "test_sample",
            "image_detail": details,
            "meta": {
                "step_translation_m": 0.5,
                "step_rotation_deg": 30.0,
                "gt_action_seq_letters": ["w"],
            },
        }
        path = root / "ivp.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def _make_env(self, **overrides):
        config = {
            "jsonl_path": str(self.jsonl_path),
            "client_url": "http://127.0.0.1:1",
            "format": "eval_mode",
            "image_size": (32, 32),
            "max_turns": 3,
            "max_actions_per_turn": 4,
        }
        config.update(overrides)
        return HabitatGSInteractiveViewPlanning(config)

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    def test_concat_reset_has_references_once_and_hides_target_pose(self):
        env = self._make_env()
        self.assertFalse(env.ground_plane_movement)
        self.assertEqual(env.observation_mode, "concat")

        async def scenario():
            try:
                obs, info = await env.reset(seed=0)
                system = (await env.system_prompt())["obs_str"]
                images = obs["multi_modal_input"]["<image>"]
                self.assertIn("ACTION SPACE\n------------", system)
                self.assertNotIn("LEGACY_V1", system)
                self.assertIn("Rotation snapping: enabled", system)
                self.assertIn("move along sensor-local up", system)
                self.assertIn("- move_forward:", system)
                self.assertIn("- answer(tx, ty, tz, rx, ry, rz):", system)
                self.assertNotIn("- w:", system)
                self.assertNotIn("submit_pose", system)
                self.assertNotIn("NAVIGATION ACTIONS", system)
                self.assertEqual(obs["obs_str"].count("<image>"), len(images))
                self.assertEqual(len(images), 3)
                self.assertIn("TARGET VIEW (camera pose unknown)", obs["obs_str"])
                self.assertIn("TOP-DOWN REFERENCE", obs["obs_str"])
                self.assertIn("CURRENT VIEW (initial)", obs["obs_str"])
                self.assertNotIn("EXPLORED TRAJECTORY", obs["obs_str"])
                self.assertIn(
                    "Every later observation contains only the latest CURRENT VIEW",
                    system,
                )
                self.assertNotIn("SUBMISSION RESULT", obs["obs_str"])
                self.assertEqual(info["metrics"]["turns_used"], 0)
                self.assertEqual(
                    info["rollout_metadata"],
                    {
                        "scene_id": "test_scene",
                        "sample_id": "test_sample",
                        "jsonl_idx": 0,
                    },
                )
            finally:
                await env.close()

        self._run(scenario())

    def test_navigation_batch_renders_once_and_appends_one_frame(self):
        env = self._make_env(
            ground_plane_movement=True,
            is_snap_every_step=False,
        )

        async def scenario():
            calls = []

            async def fake_render(width, height):
                calls.append((width, height))
                return Image.new("RGB", (width, height), (123, 45, 67))

            try:
                await env.reset(seed=0)
                system = (await env.system_prompt())["obs_str"]
                self.assertIn("ACTION SPACE\n------------", system)
                self.assertNotIn("GROUND_PLANE_V1", system)
                self.assertIn("Rotation snapping: disabled", system)
                self.assertIn("move along world +Y", system)
                y_before = float(env.view_engine.pos[1])
                env._render_current = fake_render
                obs, _, done, info = await env.step("<action>arrow_up|z|w</action>")
                self.assertFalse(done)
                self.assertEqual(calls, [(32, 32)])
                self.assertAlmostEqual(env.view_engine.pos[1], y_before + 0.5)
                self.assertEqual(len(env.exploration_history), 2)
                self.assertEqual(
                    env.exploration_history[-1].incoming_actions,
                    ("look_up", "move_up", "move_forward"),
                )
                self.assertEqual(obs["obs_str"].count("<image>"), 1)
                self.assertEqual(len(obs["multi_modal_input"]["<image>"]), 1)
                self.assertEqual(info["primitive_actions"], 3)
                self.assertIn("CURRENT VIEW", obs["obs_str"])
                self.assertNotIn("TARGET VIEW", obs["obs_str"])
                self.assertNotIn("TOP-DOWN REFERENCE", obs["obs_str"])
                self.assertNotIn("EXPLORED TRAJECTORY", obs["obs_str"])
                self.assertIn("LAST ACTION RESULT", obs["obs_str"])
            finally:
                await env.close()

        self._run(scenario())

    def test_no_concat_observation_repeats_complete_trajectory(self):
        env = self._make_env(observation_mode="no_concat")

        async def scenario():
            async def fake_render(width, height):
                return Image.new("RGB", (width, height), (123, 45, 67))

            try:
                reset_obs, _ = await env.reset(seed=0)
                system = (await env.system_prompt())["obs_str"]
                self.assertIn("Every turn is self-contained", system)
                self.assertIn("EXPLORED TRAJECTORY", reset_obs["obs_str"])
                self.assertIn("E0 (initial view)", reset_obs["obs_str"])

                env._render_current = fake_render
                obs, _, done, _ = await env.step(
                    "<action>move_forward</action>"
                )
                self.assertFalse(done)
                self.assertEqual(obs["obs_str"].count("<image>"), 4)
                self.assertEqual(len(obs["multi_modal_input"]["<image>"]), 4)
                self.assertIn("TARGET VIEW", obs["obs_str"])
                self.assertIn("TOP-DOWN REFERENCE", obs["obs_str"])
                self.assertIn("E0 --[move_forward]--> E1", obs["obs_str"])
            finally:
                await env.close()

        self._run(scenario())

    def test_observation_mode_is_validated(self):
        with self.assertRaisesRegex(ValueError, "observation_mode"):
            self._make_env(observation_mode="invalid")

    def test_mixed_submit_is_rejected_without_moving(self):
        env = self._make_env()

        async def scenario():
            try:
                await env.reset(seed=0)
                pose_before = env.view_engine.get_pose().copy()
                _, reward, done, info = await env.step(
                    "<action>move_forward|answer(0,0,0,0,0,0)</action>"
                )
                self.assertFalse(done)
                self.assertEqual(reward, 0.0)
                np.testing.assert_allclose(env.view_engine.get_pose(), pose_before)
                self.assertEqual(len(env.exploration_history), 1)
                self.assertIn("only action", info["error"])
                self.assertEqual(info["action_valid_rate"], 0.0)
            finally:
                await env.close()

        self._run(scenario())

    def test_exact_submit_is_terminal(self):
        env = self._make_env()

        async def scenario():
            try:
                await env.reset(seed=0)
                target = env.target_view["c2w_se3_deg"]
                blob = ",".join(str(float(value)) for value in target)
                obs, _, done, info = await env.step(
                    f"<action>answer({blob})</action>"
                )
                self.assertTrue(done)
                self.assertTrue(info["success"])
                self.assertTrue(info["submitted"])
                self.assertIn("SUBMISSION RESULT: success", obs["obs_str"])
            finally:
                await env.close()

        self._run(scenario())

    def test_shell_syntax(self):
        response, error = _shell_command_to_response("←", "eval_mode")
        self.assertEqual(error, "")
        self.assertEqual(response, "<action>turn_left</action>")

        response, error = _shell_command_to_response("ww←z", "eval_mode")
        self.assertIsNone(response)
        self.assertIn("exactly one", error)

        response, error = _shell_command_to_response(
            "submit 1 2 3 -10 20 30", "eval_mode"
        )
        self.assertEqual(error, "")
        self.assertEqual(
            response,
            "<action>answer(1,2,3,-10,20,30)</action>",
        )


if __name__ == "__main__":
    unittest.main()
