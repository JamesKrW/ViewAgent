"""Episode dump: write the tree, read it back, frames land once.

Replaces ``tests/test_transcript.py``. What that package recorded, the tree already holds
-- and the tree is the copy training is built from, so keeping a second one meant two
records of the same episode that could disagree.

    python tests/test_dump.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vagen_agent.rollout.dump import dump_episode, read_episode, scan  # noqa: E402
from vagen_agent.rollout.frames import FrameTable  # noqa: E402
from vagen_agent.rollout.trajectory import Generated, Record, Span  # noqa: E402


class Run:
    transcript_dir = None
    transcript_group = "step0"
    transcript_images = True
    harness = "compact"
    algorithm = "default_gae"


class Spec:
    env_name, source_name, seed = "Sokoban", "sokoban_train", 42


def _image(colour):
    from PIL import Image

    return Image.new("RGB", (8, 8), colour)


def build(frames: FrameTable) -> Record:
    """A compaction-shaped episode with a frame reused across both conversations."""
    r = Record(episode_id="ep-1")
    key = frames.intern(_image((10, 20, 30)))
    again = frames.intern(_image((10, 20, 30)))       # same pixels, must be the same entry
    assert key == again, "frame table did not dedup identical pixels"

    s = r.add_context_node(r.root, {"role": "system", "content": "S"}, Span(ids=[1]))
    u = r.add_context_node(s, {"role": "user", "content": "U0"}, Span(ids=[2], frames=[key]))
    a1 = r.add_generated(u, {"role": "assistant", "content": "a1"},
                         Generated(1, [10], [-0.5], "stop"), Span(ids=[10]))
    r.rewards[1] = [1.0]
    u2 = r.add_context_node(s, {"role": "user", "content": "summary+obs"},
                            Span(ids=[3], frames=[key]))
    r.add_generated(u2, {"role": "assistant", "content": "a2"},
                    Generated(2, [11], [-0.5], "stop"), Span(ids=[11]))
    r.rewards[2] = [2.0]
    r.turns, r.truncated = 2, True
    a1.gen.info = {"format_correct": True}
    return r


def main() -> int:
    problems = []
    frames = FrameTable()
    record = build(frames)

    with tempfile.TemporaryDirectory() as tmp:
        Run.transcript_dir = tmp
        path = dump_episode(record, Run, frames, Spec)
        back = read_episode(path)

        header, nodes = back["header"], back["nodes"]
        if header["episode_id"] != "ep-1":
            problems.append(f"episode_id {header['episode_id']}")
        if abs(header["total_reward"] - 3.0) > 1e-9:
            problems.append(f"total_reward {header['total_reward']} != 3.0")
        if header["harness"] != "compact" or header["seed"] != 42:
            problems.append("header lost the run/spec facts")
        if len(nodes) != 5:
            problems.append(f"{len(nodes)} nodes, expected 5")

        generated = [n for n in nodes if n.get("generated")]
        if len(generated) != 2:
            problems.append(f"{len(generated)} generated nodes, expected 2")
        if not any(n.get("info") == {"format_correct": True} for n in generated):
            problems.append("per-turn env info did not survive the dump")

        # The tree is reconstructable: every node's parent is a node we also wrote.
        ids = {n["id"] for n in nodes}
        for n in nodes:
            if n["parent"] is not None and n["parent"] not in ids and n["parent"] != 0:
                problems.append(f"node {n['id']} points at a parent that was not written")

        # A frame used by two conversations is stored once, under its content hash.
        written = list((Path(tmp) / "step0" / "frames").glob("*.png"))
        if len(written) != 1:
            problems.append(f"{len(written)} frame files, expected 1 (deduped)")
        used = {k for n in nodes for k in n["frames"]}
        if {p.stem for p in written} != used:
            problems.append("frame files do not match the keys the nodes reference")

        if len(scan(tmp)) != 1:
            problems.append(f"scan found {len(scan(tmp))} transcripts, expected 1")

    if problems:
        print("FAIL  dump")
        for problem in problems:
            print(f"        - {problem}")
        return 1
    print("PASS  dump             header, 5 nodes, per-turn info, 1 deduped frame")
    print("\n1 passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
