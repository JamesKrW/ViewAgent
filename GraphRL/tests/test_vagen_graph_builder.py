from __future__ import annotations

from pathlib import Path

from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.interactive_view_planning_graph_builder import (
    InteractiveViewPlanningGraphBuilder,
)
from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder


class _TestBuilder(VagenGraphBuilder):
    def traj_to_transitions(
        self,
        messages,
        rollout_dir: Path,
        step_idx: int,
        line_idx: int,
        episode_data=None,
    ):
        return []


def test_parse_vagen_line_with_chatml_tokens():
    builder = _TestBuilder({})
    data = {
        "input": (
            "<|im_start|>system\nrules<|im_end|>\n"
            "<|im_start|>user\nYou're in the scene FloorPlan1.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "output": "move<|im_end|>",
    }

    assert builder._parse_vagen_line(data) == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "You're in the scene FloorPlan1."},
        {"role": "assistant", "content": "move"},
    ]


def test_parse_vagen_line_without_special_tokens():
    builder = _TestBuilder({})
    data = {
        "input": (
            "system\nrules\n"
            "user\nYou're in the scene FloorPlan227.\n"
            "assistant\n"
        ),
        "output": "move\nuser\nnew observation\nassistant\nanswer",
    }

    assert builder._parse_vagen_line(data) == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "You're in the scene FloorPlan227."},
        {"role": "assistant", "content": "move"},
        {"role": "user", "content": "new observation"},
        {"role": "assistant", "content": "answer"},
    ]


def test_viewsuite_reset_uses_initial_image_in_prompt_order(tmp_path):
    image_dir = tmp_path / "image_1" / "images_0"
    image_dir.mkdir(parents=True)
    for index in range(4):
        (image_dir / f"{index}.png").write_bytes(b"png")

    messages = [
        {
            "role": "user",
            "content": (
                "You're in the scene FloorPlan1.\n"
                "target view <image>, initial view <image>, top-down view <image>.\n"
                "Initial view camera: [tx=0, ty=1, tz=2, rx=0, ry=0, rz=0]"
            ),
        },
        {"role": "assistant", "content": "<action>move_forward</action>"},
        {
            "role": "user",
            "content": (
                "Current camera: [tx=0, ty=1, tz=2.5, rx=0, ry=0, rz=0]\n"
                "<image>"
            ),
        },
    ]

    transitions = InteractiveViewPlanningGraphBuilder({}).traj_to_transitions(
        messages, tmp_path, 1, 0,
    )

    assert len(transitions) == 1
    assert transitions[0][0].source_images == [str(image_dir / "1.png")]
    assert transitions[0][2].source_images == [str(image_dir / "3.png")]


def test_habitat_gs_uses_structural_scene_id_when_prompt_hides_it(tmp_path):
    image_dir = tmp_path / "image_1" / "images_0"
    image_dir.mkdir(parents=True)
    for index in range(7):
        (image_dir / f"{index}.png").write_bytes(b"png")

    messages = [
        {
            "role": "user",
            "content": (
                "TARGET VIEW (camera pose unknown)\n<image>\n"
                "TOP-DOWN REFERENCE\n<image>\n"
                "EXPLORED TRAJECTORY\nE0 (initial view)\n<image>\n"
                "camera pose: [tx=0, ty=1, tz=2, rx=0, ry=0, rz=0]"
            ),
        },
        {"role": "assistant", "content": "<action>move_forward</action>"},
        {
            "role": "user",
            "content": (
                "TARGET VIEW (camera pose unknown)\n<image>\n"
                "TOP-DOWN REFERENCE\n<image>\n"
                "EXPLORED TRAJECTORY\nE0 (initial view)\n<image>\n"
                "camera pose: [tx=0, ty=1, tz=2, rx=0, ry=0, rz=0]\n"
                "E0 --[move_forward]--> E1\nE1\n<image>\n"
                "camera pose: [tx=0, ty=1, tz=2.5, rx=0, ry=0, rz=0]"
            ),
        },
    ]

    transitions = InteractiveViewPlanningGraphBuilder({}).traj_to_transitions(
        messages,
        tmp_path,
        1,
        0,
        episode_data={"rollout_metadata": {"scene_id": "habitat_scene"}},
    )

    assert len(transitions) == 1
    assert transitions[0][0].state["scene_id"] == "habitat_scene"
    assert transitions[0][2].state["scene_id"] == "habitat_scene"
