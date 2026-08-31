"""
Convert evaluator dumps into the GraphRL rollout layout that
``InteractiveViewPlanningGraphBuilder`` expects.

Input (eval_random dumps)::

    <dump_dir>/tag_<tag_id>/<rollout_id>/
        messages.json     # ChatML messages (image parts shadowed to <data_url>)
        images/turn_01_01.png, turn_01_02.png, turn_02_01.png, ...
        metrics.json

Output (VAGEN rollout layout)::

    <rollout_dir>/0.jsonl                          # one line per rollout
    <rollout_dir>/image_0/images_<line_idx>/<global_img_idx>.png

Each JSONL line is ``{"input": "<ChatML>", "output": ""}`` where the ChatML
string is built from ``messages.json`` using ``<|im_start|>role\\n...<|im_end|>``
markers.  ``<image>`` tokens are already present in user message text coming
from the ScannetTool env, so we just concatenate the text parts verbatim and
append one ``<image>`` for every ``image_url`` part (defensive: adapters that
emit image_url parts still round-trip correctly).

Images are renumbered by their cumulative index across user messages, which
matches how ``traj_to_transitions`` reads them back.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_IMAGE_TOKEN = "<image>"


def _render_content(content: Any) -> str:
    """Flatten shadowed ChatML content back to a plain text string.

    - If ``content`` is already a string, return it unchanged.
    - If it's a list of parts, concatenate text parts verbatim.  Each
      ``image_url`` part becomes an extra ``<image>`` token (the random_nav
      adapter never emits image_url parts, but openai-style adapters do).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    out: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            out.append(str(part.get("text", "")))
        elif kind == "image_url":
            out.append(_IMAGE_TOKEN)
    return "".join(out)


def _messages_to_chatml(messages: List[Dict[str, Any]]) -> str:
    """Serialize dumped messages into a single ChatML string."""
    pieces: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _render_content(msg.get("content"))
        pieces.append(f"<|im_start|>{role}\n{text}<|im_end|>")
    return "\n".join(pieces) + "\n"


def _count_image_tokens(text: str) -> int:
    return text.count(_IMAGE_TOKEN)


def _copy_rollout_images(
    rollout_src: Path,
    messages: List[Dict[str, Any]],
    images_dst_dir: Path,
) -> int:
    """Copy ``turn_XX_YY.png`` files to ``<global_img_idx>.png`` layout.

    Returns the number of images actually copied (or missing placeholders
    skipped silently).
    """
    src_img_dir = rollout_src / "images"
    images_dst_dir.mkdir(parents=True, exist_ok=True)

    global_img_idx = 0
    user_turn_idx = 0  # 1-based turn index used in ``turn_{t:02d}_...``

    for msg in messages:
        if msg.get("role") != "user":
            continue
        user_turn_idx += 1
        text = _render_content(msg.get("content"))
        n_imgs = _count_image_tokens(text)
        for i in range(n_imgs):
            # Source filename follows ``enumerate(..., start=1)`` on both axes.
            src_name = f"turn_{user_turn_idx:02d}_{i + 1:02d}.png"
            src = src_img_dir / src_name
            dst = images_dst_dir / f"{global_img_idx}.png"
            if src.is_file():
                if not dst.exists():
                    shutil.copy2(src, dst)
            else:
                # It's possible the env didn't attach images to this token
                # (e.g. select_view returned nothing yet).  Skip silently:
                # the graph builder only dedups nodes that have a real image.
                logger.debug("Missing source image: %s", src)
            global_img_idx += 1

    return global_img_idx


def _iter_rollout_dirs(dump_root: Path, tag_filter: str | None) -> List[Path]:
    """Yield every completed rollout directory under a dump root.

    ``tag_filter`` (if set) restricts to a specific ``tag_<name>`` subdir; all
    other tags are ignored.
    """
    rollouts: List[Path] = []
    if not dump_root.is_dir():
        return rollouts
    for tag_dir in sorted(dump_root.iterdir()):
        if not tag_dir.is_dir() or not tag_dir.name.startswith("tag_"):
            continue
        if tag_filter is not None and tag_dir.name != f"tag_{tag_filter}":
            continue
        for rid_dir in sorted(tag_dir.iterdir()):
            if not rid_dir.is_dir():
                continue
            if not (rid_dir / "messages.json").is_file():
                continue
            rollouts.append(rid_dir)
    return rollouts


def convert_dump_to_vagen_rollouts(
    dump_root: Path,
    output_rollout_dir: Path,
    *,
    tag_filter: str | None = None,
    step_idx: int = 0,
) -> Tuple[int, int]:
    """Convert a whole eval dump into one VAGEN-style rollout directory.

    Args:
        dump_root: ``experiment.dump_dir`` from the eval config (contains
            ``tag_<name>/<rid>/messages.json`` subtrees).
        output_rollout_dir: Destination passed to ``InteractiveViewPlanningGraphBuilder``
            as ``rollout_dir``.  Will contain ``{step_idx}.jsonl`` and
            ``image_{step_idx}/images_<line_idx>/<k>.png``.
        tag_filter: Optional tag_id (without the ``tag_`` prefix) to restrict.
        step_idx: Used for the JSONL filename and the ``image_<step_idx>``
            subdirectory.  Default ``0`` matches the single-file case.

    Returns:
        (num_rollouts_written, num_images_copied)
    """
    output_rollout_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_rollout_dir / f"{step_idx}.jsonl"
    image_root = output_rollout_dir / f"image_{step_idx}"
    image_root.mkdir(parents=True, exist_ok=True)

    rollouts = _iter_rollout_dirs(dump_root, tag_filter)
    total_images = 0
    n_written = 0

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for line_idx, rid_dir in enumerate(rollouts):
            try:
                with open(rid_dir / "messages.json", encoding="utf-8") as mf:
                    messages = json.load(mf)
            except Exception as exc:
                logger.warning("Failed to load %s/messages.json: %s", rid_dir, exc)
                continue

            if not isinstance(messages, list) or not messages:
                logger.warning("Empty/invalid messages.json in %s", rid_dir)
                continue

            chatml = _messages_to_chatml(messages)
            entry = {"input": chatml, "output": ""}
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

            images_dst = image_root / f"images_{line_idx}"
            n_imgs = _copy_rollout_images(rid_dir, messages, images_dst)
            total_images += n_imgs
            n_written += 1

    logger.info(
        "[converter] %s → %s: %d rollouts, %d images",
        dump_root, output_rollout_dir, n_written, total_images,
    )
    return n_written, total_images
