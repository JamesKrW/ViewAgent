from pathlib import Path

from graphrl.envs.viewsuite.viewsuite_interactive_view_planning_nograph.nograph_graph_builder import (
    NoGraphViewPlanningGraphBuilder,
)


def _messages():
    return [
        {
            "role": "user",
            "content": (
                "You're in the scene scene0001_00.\n"
                "CURRENT VIEW (initial) <image>\n"
                "camera pose: [tx=0, ty=0, tz=0, rx=0, ry=0, rz=0]"
            ),
        },
        {"role": "assistant", "content": "<action>move_forward</action>"},
        {
            "role": "user",
            "content": (
                "CURRENT VIEW <image>\n"
                "camera pose: [tx=0, ty=0.5, tz=0, rx=0, ry=0, rz=0]"
            ),
        },
    ]


def test_identical_poses_in_different_rollouts_never_merge(tmp_path: Path):
    builder = NoGraphViewPlanningGraphBuilder({})
    first = builder.traj_to_transitions(
        _messages(), tmp_path, 3, 4, {"scene_id": "scene0001_00"}
    )[0]
    second = builder.traj_to_transitions(
        _messages(), tmp_path, 3, 5, {"scene_id": "scene0001_00"}
    )[0]

    assert first[0].unique_key() != second[0].unique_key()
    assert first[0].bucket_key() != second[0].bucket_key()
    assert not first[0].is_similar_to(second[0])
