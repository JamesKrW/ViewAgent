from __future__ import annotations

import numpy as np

from view_suite.ai2thor.pose_utils import unity_pose_to_c2w
from view_suite.ai2thor.view_manipulator import ViewManipulator as Ai2ThorManipulator
from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator
from view_suite.scannet.view_manipulator import ViewManipulator as ScanNetManipulator


def _scannet_level_pose() -> np.ndarray:
    """OpenCV camera: +X right, +Y down, +Z forward; world +Z is up."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    return c2w


def test_scannet_ground_forward_ignores_pitch_and_vertical_is_world_z():
    ground = ScanNetManipulator(
        step_translation=0.5,
        step_rotation_deg=30.0,
        world_up_axis="Z",
        ground_plane_movement=True,
    )
    legacy = ScanNetManipulator(
        step_translation=0.5,
        step_rotation_deg=30.0,
        world_up_axis="Z",
        ground_plane_movement=False,
    )
    for vm in (ground, legacy):
        vm.reset(_scannet_level_pose())
        vm.step("r")

    ground.step("w")
    legacy.step("w")
    assert np.isclose(ground.get_pose()[2, 3], 0.0)
    assert not np.isclose(legacy.get_pose()[2, 3], 0.0)
    assert np.isclose(np.linalg.norm(ground.get_pose()[:3, 3]), 0.5)

    before = ground.get_pose()[:3, 3].copy()
    ground.step("y")
    np.testing.assert_allclose(ground.get_pose()[:3, 3] - before, [0.0, 0.0, 0.5])

    # A body yaw rotates about world-up, so it preserves the existing pitch.
    forward_up_before = float(ground.get_pose()[:3, 2] @ np.array([0.0, 0.0, 1.0]))
    ground.step("q")
    forward_up_after = float(ground.get_pose()[:3, 2] @ np.array([0.0, 0.0, 1.0]))
    assert np.isclose(forward_up_after, forward_up_before)


def test_ai2thor_runtime_api_matches_generator_and_ground_forward_is_horizontal():
    init_pose = {
        "position": {"x": 1.0, "y": 1.2, "z": -2.0},
        "rotation": {"x": -30.0, "y": 60.0, "z": 0.0},
    }
    generator_vm = Ai2ThorManipulator(
        init_pose=init_pose,
        step_translation=0.5,
        step_rotation_deg=30.0,
        is_discrete=True,
        ground_plane_movement=False,
    )
    runtime_vm = Ai2ThorManipulator(
        step_translation=0.5,
        step_rotation_deg=30.0,
        is_discrete=True,
        ground_plane_movement=False,
    )
    runtime_vm.reset(unity_pose_to_c2w(init_pose))
    for action in ("w", "e", "f", "d"):
        generator_vm.step(action)
        runtime_vm.step(action)
    np.testing.assert_allclose(runtime_vm.get_pose(), generator_vm.get_pose(), atol=1e-10)

    ground = Ai2ThorManipulator(
        init_pose=init_pose,
        step_translation=0.5,
        step_rotation_deg=30.0,
        ground_plane_movement=True,
    )
    y_before = ground.pos[1]
    ground.step("w")
    assert np.isclose(ground.pos[1], y_before)
    ground.step("y")
    assert np.isclose(ground.pos[1], y_before + 0.5)


def test_habitat_vertical_knob_and_horizontal_body_motion():
    ground = HabitatGSViewManipulator(
        position=(0.0, 1.5, 0.0),
        pitch_deg=30.0,
        step_translation=0.5,
        ground_plane_movement=True,
    )
    legacy = HabitatGSViewManipulator(
        position=(0.0, 1.5, 0.0),
        pitch_deg=30.0,
        step_translation=0.5,
        ground_plane_movement=False,
    )
    for vm in (ground, legacy):
        y_before = vm.pos[1]
        vm.step("w")
        assert np.isclose(vm.pos[1], y_before)

    before = ground.pos.copy()
    ground.step("y")
    np.testing.assert_allclose(ground.pos - before, [0.0, 0.5, 0.0])

    before = legacy.pos.copy()
    legacy.step("y")
    assert not np.allclose(legacy.pos - before, [0.0, 0.5, 0.0])
