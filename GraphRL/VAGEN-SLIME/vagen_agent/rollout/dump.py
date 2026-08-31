"""One JSONL per episode: the tree, as it was.

This replaces ``transcript/`` (five files, ~1340 lines). Everything that package recorded,
the tree already holds -- which messages entered the model's view, which range a
compaction replaced, which conversation is which training row -- and the tree is the copy
training is actually built from. Two records of the same thing, only one of which feeds
the loss, is exactly the arrangement this repo is elsewhere careful to avoid.

What was genuinely lost and is worth knowing: streaming (a crashed process leaves no
partial record), and refusing an unrecognised event on read-back (that rule protected
cross-version replay, which does not arise when the writer and the reader are the same
code).

Frames come from the frame table, which is already content-addressed and already
deduplicated -- ``FileImageStore``'s job, without a second implementation of it.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _node_record(node, index: dict) -> dict:
    out = {
        "id": index[id(node)],
        "parent": index.get(id(node.parent)),
        "role": node.role,
        "content": _content(node.message),
        "n_ids": len(node.span.ids),
        "frames": list(node.span.frames),
    }
    if node.is_generated:
        gen = node.gen
        out |= {"generated": True, "call_id": gen.call_id,
                "n_output": len(gen.output_ids), "finish_reason": gen.finish_reason,
                "info": gen.info}
    return out


def _content(message):
    if not message:
        return None
    content = message.get("content", "")
    if isinstance(content, list):
        return [p.get("text", f"<{p.get('type')}>") if isinstance(p, dict) else str(p)
                for p in content]
    return content


def dump_episode(record, run, frames, spec=None) -> Path | None:
    """Write the episode's tree, and its frames when the run asked for them."""
    root = Path(run.transcript_dir) / (run.transcript_group or "")
    path = root / f"{record.episode_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes, stack = [], [record.root]
    while stack:
        node = stack.pop(0)
        nodes.append(node)
        stack.extend(node.children)
    index = {id(n): i for i, n in enumerate(nodes)}

    header = {
        "episode_id": record.episode_id,
        "env_name": getattr(spec, "env_name", None),
        "source_name": getattr(spec, "source_name", None),
        "seed": getattr(spec, "seed", None),
        "harness": run.harness,
        "algorithm": run.algorithm,
        "turns": record.turns,
        "total_reward": record.total_reward,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "aborted": record.aborted,
        "rewards": {str(k): v for k, v in record.rewards.items()},
    }
    with path.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for node in nodes:
            if node.is_root:
                continue
            fh.write(json.dumps(_node_record(node, index)) + "\n")

    if run.transcript_images:
        frame_dir = root / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for node in nodes:
            for key in node.span.frames:
                target = frame_dir / f"{key}.png"
                if not target.exists():
                    payload = frames.entries[key].b64.split(",", 1)[-1]
                    target.write_bytes(base64.b64decode(payload))
    return path


def read_episode(path) -> dict:
    """Read one back: ``{"header": ..., "nodes": [...]}``."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {"header": lines[0], "nodes": lines[1:]}


def scan(directory) -> list:
    return sorted(str(p) for p in Path(directory).rglob("*.jsonl"))


__all__ = ["dump_episode", "read_episode", "scan"]
