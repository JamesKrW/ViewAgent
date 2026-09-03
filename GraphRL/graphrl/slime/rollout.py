"""ViewAgent's VAGEN-SLIME rollout seam and legacy graph-data exporter."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from vagen_agent.rollout.runner import _build_episode, _get_context, _run_episode
from vagen_agent.rollout.trajectory import assemble, dropped_sample, episode_status
from view_suite.envs.slime_adapter import register_viewsuite_envs


register_viewsuite_envs()


def _rollout_id() -> int:
    from slime.rollout import sample_hooks

    value = sample_hooks._current_rollout_id  # shared in the RolloutManager process
    if value is None:
        raise RuntimeError("SLIME did not expose the current rollout id")
    return int(value)


def _content(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    raw = message.get("content", "")
    if isinstance(raw, str):
        return raw
    output: list[str] = []
    for part in raw or []:
        if isinstance(part, dict) and part.get("type") == "image":
            output.append("<image>")
        elif isinstance(part, dict):
            output.append(str(part.get("text", "")))
        else:
            output.append(str(part))
    return "".join(output)


def _chatml(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def _generated_nodes(record) -> list[Any]:
    nodes, queue = [], [record.root]
    while queue:
        node = queue.pop(0)
        if node.is_generated:
            nodes.append(node)
        queue.extend(node.children)
    return sorted(nodes, key=lambda node: node.gen.call_id)


def _nearest_user(node) -> Any | None:
    current = node.parent
    while current is not None and not current.is_root:
        if current.role == "user" and not current.is_generated:
            return current
        current = current.parent
    return None


def _system_node(record) -> Any | None:
    queue = [record.root]
    while queue:
        node = queue.pop(0)
        if node.role == "system" and not node.is_generated:
            return node
        queue.extend(node.children)
    return None


def _capture_images_enabled(ctx) -> bool:
    enabled = bool(ctx.run.extra.get("record_rollout_images", True))
    experiment = os.environ.get("GRAPHRL_ADAPTIVE_EXPERIMENT_DIR")
    if not enabled or not experiment:
        return enabled
    try:
        from graphrl.adaptive.state import STATE_FILENAME

        state = json.loads((Path(experiment) / STATE_FILENAME).read_text(encoding="utf-8"))
        return not bool(state.get("latched"))
    except (OSError, ValueError):
        return enabled


def _episode_metadata(env: Any) -> dict[str, Any]:
    """Return stable episode identity retained by the VAGEN env adapter.

    ``ViewSuiteSlimeEnv.close()`` releases its legacy delegate, so fields such
    as ``delegate.current_item`` are no longer reachable when export runs.  The
    reset ``info`` is deliberately retained by ``BaseVagenEnv`` and contains
    the same identity without extending the environment lifetime.
    """

    metadata: dict[str, Any] = {}
    reset_info = getattr(env, "reset_info", None)
    if isinstance(reset_info, dict):
        for name in ("scene_id", "sample_id"):
            value = reset_info.get(name)
            if value is not None:
                metadata[name] = value
    return metadata


def _stage_episode(
    ctx,
    record,
    sample,
    spec,
    rollout_id: int,
    metrics: dict[str, float],
    episode_metadata: dict[str, Any] | None = None,
) -> str | None:
    root_value = ctx.run.extra.get("legacy_rollout_dir")
    if not root_value:
        return None
    identity = json.dumps(
        {
            "rollout_id": rollout_id,
            "sample_index": getattr(sample, "index", None),
            "group_index": getattr(sample, "group_index", None),
            "seed": spec.seed,
            "env": spec.env_name,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    episode_dir = Path(root_value) / ".staging" / f"step_{rollout_id + 1}" / key
    episode_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[tuple[str, str, list[str]]] = []
    system = _system_node(record)
    if system is not None:
        pieces.append(("system", _content(system.message), list(system.span.frames)))
    for generated in _generated_nodes(record):
        user = _nearest_user(generated)
        if user is not None:
            pieces.append(("user", _content(user.message), list(user.span.frames)))
        pieces.append(("assistant", _content(generated.message), []))

    chatml = "".join(_chatml(role, content) for role, content, _ in pieces)
    capture_images = _capture_images_enabled(ctx)
    image_names: list[str] = []
    image_index = 0
    if capture_images:
        for _role, _text, frame_keys in pieces:
            for frame_key in frame_keys:
                image_path = episode_dir / f"{image_index}.png"
                payload = ctx.frames.entries[frame_key].b64.split(",", 1)[-1]
                image_path.write_bytes(base64.b64decode(payload))
                image_names.append(image_path.name)
                image_index += 1

    record_path = episode_dir / "record.json"
    temporary = episode_dir / ".record.json.tmp"
    episode_metadata = dict(episode_metadata or {})
    temporary.write_text(
        json.dumps(
            {
                "input": chatml,
                "output": "",
                "score": record.total_reward,
                "step": rollout_id + 1,
                # Termination only means that the task ended; an unsuccessful
                # submit is terminal too.  Use the environment's graded metric.
                "traj_success": bool(
                    metrics.get("traj_success", metrics.get("success", 0.0))
                ),
                "env_name": spec.env_name,
                "seed": spec.seed,
                # The Habitat-GS IVP prompt intentionally omits the scene name.
                # Keep identity as structured rollout data instead of requiring
                # downstream graph builders to scrape a particular prompt format.
                **episode_metadata,
                "images": image_names,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, record_path)
    return key


async def generate(args, sample, sampling_params, evaluation: bool = False):
    """Run one VAGEN-SLIME episode and stage GraphRL-compatible rollout data."""

    ctx = _get_context(args)
    spec, record, client, env, harness = _build_episode(args, ctx, sample)
    client.sampling_params.update(sampling_params or {})
    try:
        await _run_episode(harness, client, env, record)
    finally:
        await env.close()

    episode_metadata = _episode_metadata(env)

    if ctx.run.transcript_dir:
        from vagen_agent.rollout.dump import dump_episode

        dump_episode(record, ctx.run, ctx.frames, spec)

    metrics = dict(getattr(env, "last_metrics", {}) or {})
    metrics.setdefault("success", float(getattr(env, "success", False)))
    metrics.setdefault("traj_success", metrics["success"])
    samples = assemble(
        record,
        sample,
        algorithm=ctx.run.algorithm,
        frames=ctx.frames,
        status=episode_status(aborted=record.aborted, truncated=record.truncated),
        metadata_extra={
            "source_name": spec.source_name,
            "env_name": spec.env_name,
            "metrics": metrics,
            **episode_metadata,
        },
    )
    if not samples:
        samples = [dropped_sample(sample, "the model never spoke in any conversation")]
    if not evaluation:
        key = _stage_episode(
            ctx,
            record,
            sample,
            spec,
            _rollout_id(),
            metrics,
            episode_metadata,
        )
        for output in samples:
            output.metadata = dict(getattr(output, "metadata", None) or {})
            output.metadata["graphrl_rollout_key"] = key
    return samples


def finalize_rollout_step(args: Any, rollout_id: int, samples: Any) -> Path | None:
    """Publish one complete legacy JSONL only after every selected episode exists."""

    ctx = _get_context(args)
    root_value = ctx.run.extra.get("legacy_rollout_dir")
    if not root_value:
        return None
    root = Path(root_value)
    step = int(rollout_id) + 1
    staging = root / ".staging" / f"step_{step}"
    flat = []
    stack = list(samples) if isinstance(samples, (list, tuple)) else [samples]
    while stack:
        value = stack.pop(0)
        if isinstance(value, (list, tuple)):
            stack[:0] = list(value)
        else:
            flat.append(value)
    keys = sorted(
        {
            str((getattr(sample, "metadata", None) or {}).get("graphrl_rollout_key"))
            for sample in flat
            if (getattr(sample, "metadata", None) or {}).get("graphrl_rollout_key")
        }
    )
    if not keys:
        return None

    root.mkdir(parents=True, exist_ok=True)
    image_root = root / f"image_{step}"
    temporary_images = root / f".image_{step}.tmp"
    if temporary_images.exists():
        shutil.rmtree(temporary_images)
    lines = []
    wrote_images = False
    for line_index, key in enumerate(keys):
        source = staging / key
        record_path = source / "record.json"
        if not record_path.is_file():
            raise RuntimeError(f"rollout staging record is incomplete: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        lines.append(json.dumps({name: value for name, value in record.items() if name != "images"}, ensure_ascii=False))
        filenames = list(record.get("images") or [])
        if filenames:
            target_images = temporary_images / f"images_{line_index}"
            target_images.mkdir(parents=True)
        for image_index, filename in enumerate(filenames):
            source_image = source / filename
            if not source_image.is_file():
                raise RuntimeError(f"rollout image is incomplete: {source_image}")
            shutil.copy2(source_image, target_images / f"{image_index}.png")
            wrote_images = True

    temporary_jsonl = root / f".{step}.jsonl.tmp"
    temporary_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_jsonl = root / f"{step}.jsonl"
    if image_root.exists():
        shutil.rmtree(image_root)
    if wrote_images:
        os.replace(temporary_images, image_root)
    else:
        shutil.rmtree(temporary_images, ignore_errors=True)
    os.replace(temporary_jsonl, final_jsonl)
    (root / f"{step}.complete").write_text("complete\n", encoding="utf-8")
    shutil.rmtree(staging, ignore_errors=True)
    return final_jsonl


__all__ = ["finalize_rollout_step", "generate"]
