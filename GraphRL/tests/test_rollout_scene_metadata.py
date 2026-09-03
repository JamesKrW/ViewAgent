from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.interactive_view_planning_graph_builder import (
    InteractiveViewPlanningGraphBuilder,
)
from graphrl.slime.rollout import _episode_metadata, _stage_episode


def _node(*, role=None, content="", generated=False, root=False, call_id=0):
    return SimpleNamespace(
        role=role,
        message={"role": role, "content": content},
        is_generated=generated,
        is_root=root,
        gen=SimpleNamespace(call_id=call_id),
        span=SimpleNamespace(frames=[]),
        parent=None,
        children=[],
    )


def test_rollout_record_persists_structured_scene_identity(tmp_path):
    root = _node(root=True)
    system = _node(role="system", content="system")
    user = _node(
        role="user",
        content="camera pose: [tx=0, ty=0, tz=0, rx=0, ry=0, rz=0]",
    )
    assistant = _node(
        role="assistant", content="<action>w</action>", generated=True, call_id=1
    )
    root.children = [system]
    system.parent, system.children = root, [user]
    user.parent, user.children = system, [assistant]
    assistant.parent = user

    context = SimpleNamespace(
        run=SimpleNamespace(
            extra={
                "legacy_rollout_dir": str(tmp_path),
                "record_rollout_images": False,
            }
        ),
        frames=SimpleNamespace(entries={}),
    )
    record = SimpleNamespace(root=root, total_reward=0.25)
    sample = SimpleNamespace(index=2, group_index=3)
    spec = SimpleNamespace(seed=7, env_name="HabitatGSInteractiveViewPlanning")

    key = _stage_episode(
        context,
        record,
        sample,
        spec,
        0,
        {"success": 0.0},
        {"scene_id": "interior_0007_840137", "sample_id": "sample_007"},
    )

    data = json.loads(
        (tmp_path / ".staging" / "step_1" / key / "record.json").read_text()
    )
    assert data["scene_id"] == "interior_0007_840137"
    assert data["sample_id"] == "sample_007"


def test_episode_identity_survives_delegate_close():
    env = SimpleNamespace(
        reset_info={
            "scene_id": "interior_0007_840137",
            "sample_id": "sample_007",
            "success": False,
        }
    )

    assert _episode_metadata(env) == {
        "scene_id": "interior_0007_840137",
        "sample_id": "sample_007",
    }


def test_graph_builder_uses_record_scene_id_and_normalizes_compact_actions(tmp_path):
    rollout_dir = tmp_path / "rollout_data"
    rollout_dir.mkdir()
    source = rollout_dir / "1.jsonl"
    chatml = "".join(
        [
            "<|im_start|>user\nTARGET VIEW <image>\n"
            "TOP-DOWN REFERENCE <image>\n"
            "camera pose: [tx=99, ty=99, tz=99, rx=90, ry=0, rz=0]\n"
            "EXPLORED TRAJECTORY\nE0 <image>\n"
            "camera pose: [tx=0, ty=0, tz=0, rx=0, ry=0, rz=0]"
            "<|im_end|>\n",
            "<|im_start|>assistant\n"
            "<action>w|arrow_left|d</action><|im_end|>\n",
            "<|im_start|>user\nTARGET VIEW <image>\n"
            "TOP-DOWN REFERENCE <image>\n"
            "camera pose: [tx=99, ty=99, tz=99, rx=90, ry=0, rz=0]\n"
            "EXPLORED TRAJECTORY\nE0 <image>\n"
            "camera pose: [tx=0, ty=0, tz=0, rx=0, ry=0, rz=0]\n"
            "E1 <image>\n"
            "camera pose: [tx=1, ty=0, tz=1, rx=0, ry=30, rz=0]"
            "<|im_end|>\n",
        ]
    )
    source.write_text(
        json.dumps(
            {
                "input": chatml,
                "output": "",
                "scene_id": "interior_0007_840137",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    images_dir = tmp_path / "graph" / "images"
    images_dir.mkdir(parents=True)
    builder = InteractiveViewPlanningGraphBuilder({})
    graph = builder._build_sequential([source], rollout_dir, images_dir)

    assert graph.num_nodes == 2
    assert graph.num_edges == 1
    [(src, dst, edge)] = graph._g.edges(data=True)
    assert graph._g.nodes[src]["state"]["scene_id"] == "interior_0007_840137"
    assert graph._g.nodes[dst]["state"]["scene_id"] == "interior_0007_840137"
    assert graph._g.nodes[src]["state"]["pose"]["tx"] == 0.0
    assert graph._g.nodes[dst]["state"]["pose"]["tx"] == 1.0
    assert edge["obs_str"] == "move_forward | turn_left | move_right"
