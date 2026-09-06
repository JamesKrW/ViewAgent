from __future__ import annotations

import pytest

from view_suite.ai2thor.gym_ai2thor_tool_env import GymAi2thorToolEnv
from view_suite.envs.utils.parse_utils import ParsedAction
from view_suite.habitat_gs.gym_habitat_gs_tool_env import GymHabitatGSToolEnv
from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator
from view_suite.scannet.gym_scannet_tool_env import GymScannetToolEnv


class _ConcreteMixin:
    def _get_view_dict(self):
        return {}

    async def close(self):
        return None

    async def system_prompt(self):
        return {}

    async def reset(self, seed):
        return {}, {}

    async def step(self, action_str):
        return {}, 0.0, False, {}


class _ScanNetToolEnv(_ConcreteMixin, GymScannetToolEnv):
    pass


class _Ai2ThorToolEnv(_ConcreteMixin, GymAi2thorToolEnv):
    pass


class _HabitatGSToolEnv(_ConcreteMixin, GymHabitatGSToolEnv):
    pass


def _bare_env(cls, **overrides):
    values = {
        "step_translation": 0.5,
        "step_rotation_deg": 30.0,
        "is_discrete": True,
        "is_snap_every_step": True,
        "image_y_down": True,
        "pitch_limit_deg": 60.0,
        "action_only_mode": True,
        "allow_rotate": False,
        "ground_plane_movement": False,
    }
    values.update(overrides)
    env = cls.__new__(cls)
    for name, value in values.items():
        setattr(env, name, value)
    return env


@pytest.mark.parametrize(
    ("cls", "mode_detail", "coordinates", "snap_detail"),
    [
        (
            _ScanNetToolEnv,
            "full camera-local translation",
            "horizontal plane is XY; world up is +Z",
            "c2w Euler XYZ angles",
        ),
        (
            _Ai2ThorToolEnv,
            "full camera-local translation",
            "horizontal plane is XZ; world up is +Y",
            "controller yaw, pitch, and roll",
        ),
        (
            _HabitatGSToolEnv,
            "native Habitat viewer movement",
            "horizontal plane is XZ; world up is +Y",
            "controller yaw and pitch",
        ),
    ],
)
def test_legacy_prompt_is_one_mode_aware_action_space(
    cls, mode_detail, coordinates, snap_detail
):
    prompt = _bare_env(cls)._tool_instruction

    assert prompt.startswith("ACTION SPACE\n------------")
    assert "LEGACY_V1" not in prompt
    assert "DISCRETE, SNAPPED" not in prompt
    assert mode_detail in prompt
    assert coordinates in prompt
    assert snap_detail in prompt
    assert prompt.count("0.5 meters") == 1
    assert prompt.count("30.0 degrees") == 1
    assert prompt.count("This action is terminal") == 1
    assert "SUPPORTED ACTIONS" not in prompt
    assert "DISCRETE MODE" not in prompt
    assert "GROUND-PLANE MOVEMENT" not in prompt
    assert "taken.- The episode" not in prompt


@pytest.mark.parametrize(
    ("cls", "movement_detail", "vertical_detail"),
    [
        (_ScanNetToolEnv, "horizontal heading", "move along world +Z"),
        (_Ai2ThorToolEnv, "yaw-only horizontal heading", "move along world +Y"),
        (_HabitatGSToolEnv, "yaw-only horizontal heading", "move along world +Y"),
    ],
)
def test_ground_plane_prompt_explains_each_environment_axes(
    cls, movement_detail, vertical_detail
):
    prompt = _bare_env(cls, ground_plane_movement=True)._tool_instruction

    assert prompt.startswith("ACTION SPACE\n------------")
    assert "GROUND_PLANE_V1" not in prompt
    assert movement_detail in prompt
    assert vertical_detail in prompt


@pytest.mark.parametrize("cls", [_ScanNetToolEnv, _Ai2ThorToolEnv, _HabitatGSToolEnv])
def test_prompt_reports_disabled_snap_knob(cls):
    prompt = _bare_env(cls, is_snap_every_step=False)._tool_instruction

    assert "Rotation snapping: disabled" in prompt
    assert "`is_snap_every_step=false`" in prompt


@pytest.mark.parametrize("cls", [_ScanNetToolEnv, _Ai2ThorToolEnv, _HabitatGSToolEnv])
def test_non_discrete_mode_disables_snapping_even_if_requested(cls):
    prompt = _bare_env(
        cls, is_discrete=False, is_snap_every_step=True
    )._tool_instruction

    assert "`is_discrete=false`" in prompt


@pytest.mark.parametrize("cls", [_ScanNetToolEnv, _Ai2ThorToolEnv])
def test_allow_rotate_controls_roll_action_vocabulary(cls):
    without_roll = _bare_env(cls, allow_rotate=False, action_only_mode=False)
    with_roll = _bare_env(cls, allow_rotate=True, action_only_mode=False)

    assert "- rotate_ccw:" not in without_roll._tool_instruction
    assert "- rotate_cw:" not in without_roll._tool_instruction
    assert "roll actions are disabled" in without_roll._tool_instruction
    assert "- rotate_ccw:" in with_roll._tool_instruction
    assert "- rotate_cw:" in with_roll._tool_instruction
    assert "roll actions are enabled" in with_roll._tool_instruction


@pytest.mark.parametrize("cls", [_ScanNetToolEnv, _Ai2ThorToolEnv])
def test_allow_rotate_false_rejects_roll_even_outside_action_only_mode(cls):
    env = _bare_env(cls, allow_rotate=False, action_only_mode=False)

    result = env._execute_action(ParsedAction("rotate_cw", None))

    assert result["success"] is False
    assert "allow_rotate=false" in result["result"]


def test_habitat_snap_knob_controls_actual_camera_quantization():
    snapped = HabitatGSViewManipulator(
        yaw_deg=17.0,
        pitch_deg=11.0,
        discrete=True,
        is_snap_every_step=True,
    )
    unsnapped = HabitatGSViewManipulator(
        yaw_deg=17.0,
        pitch_deg=11.0,
        discrete=True,
        is_snap_every_step=False,
    )

    assert (snapped.yaw, snapped.pitch) == (30.0, 0.0)
    assert (unsnapped.yaw, unsnapped.pitch) == (17.0, 11.0)
    unsnapped.turn_left()
    assert unsnapped.yaw == 47.0
