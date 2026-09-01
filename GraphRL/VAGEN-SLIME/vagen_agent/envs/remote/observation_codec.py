"""Lossless JSON structure plus binary images for remote observations.

The service keeps the observation dictionary intact in the JSON multipart part. PIL
images in the conventional ``multi_modal_input["<image>"]`` slot are replaced by
stable references and carried as binary multipart parts, avoiding base64 expansion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_REFERENCE_KEY = "$vagen_multipart_image"


def to_json_value(value: Any, *, path: str = "value") -> Any:
    """Convert common structured values without silently stringifying unknown types."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite floats")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
            result[key] = to_json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            to_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    # NumPy arrays/scalars and similar containers expose one of these without making
    # NumPy a dependency of the generic remote package.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_json_value(tolist(), path=path)
    item = getattr(value, "item", None)
    if callable(item):
        return to_json_value(item(), path=path)
    raise TypeError(
        f"{path} contains unsupported {type(value).__name__}; provide JSON-compatible "
        "values or PIL images in multi_modal_input['<image>']"
    )


def pack_observation(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], list[Image.Image] | None]:
    """Return a JSON-safe observation and its ordered binary image payloads."""

    if not isinstance(observation, dict):
        raise TypeError(
            "remote environment observation must be a dict, got "
            f"{type(observation).__name__}"
        )

    source = dict(observation)
    multimodal = source.pop("multi_modal_input", None)
    packed = to_json_value(source, path="observation")
    if multimodal is None:
        return packed, None
    if not isinstance(multimodal, Mapping):
        raise TypeError("observation.multi_modal_input must be a mapping")

    multimodal_source = dict(multimodal)
    raw_images = multimodal_source.pop("<image>", []) or []
    if not isinstance(raw_images, (list, tuple)):
        raise TypeError("observation.multi_modal_input['<image>'] must be a sequence")
    images = list(raw_images)
    if any(not isinstance(image, Image.Image) for image in images):
        raise TypeError(
            "observation.multi_modal_input['<image>'] must contain PIL images"
        )

    packed_multimodal = to_json_value(
        multimodal_source, path="observation.multi_modal_input"
    )
    if images:
        packed_multimodal["<image>"] = [
            {IMAGE_REFERENCE_KEY: index} for index in range(len(images))
        ]
    elif "<image>" in multimodal:
        packed_multimodal["<image>"] = []
    packed["multi_modal_input"] = packed_multimodal
    return packed, images or None


def unpack_observation(
    value: Any, images: list[Image.Image]
) -> dict[str, Any]:
    """Restore image references in an observation received from the service."""

    if not isinstance(value, dict):
        raise TypeError("remote observation payload must be an object")
    observation = dict(value)
    multimodal = observation.get("multi_modal_input")
    if multimodal is None:
        if images:
            raise ValueError("remote response included unreferenced image parts")
        return observation
    if not isinstance(multimodal, dict):
        raise TypeError("remote observation multi_modal_input must be an object")

    restored_multimodal = dict(multimodal)
    references = restored_multimodal.get("<image>")
    if references is None:
        if images:
            raise ValueError("remote response included unreferenced image parts")
        observation["multi_modal_input"] = restored_multimodal
        return observation
    if not isinstance(references, list):
        raise TypeError("remote observation image references must be a list")

    restored_images = []
    used_indices = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {IMAGE_REFERENCE_KEY}:
            raise TypeError("remote observation contains an invalid image reference")
        index = reference[IMAGE_REFERENCE_KEY]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("remote observation image index must be an integer")
        if index < 0 or index >= len(images):
            raise ValueError(f"remote observation image index {index} is out of range")
        used_indices.append(index)
        restored_images.append(images[index])
    if sorted(used_indices) != list(range(len(images))):
        raise ValueError("remote observation image references do not match image parts")
    restored_multimodal["<image>"] = restored_images
    observation["multi_modal_input"] = restored_multimodal
    return observation


__all__ = [
    "IMAGE_REFERENCE_KEY",
    "pack_observation",
    "to_json_value",
    "unpack_observation",
]
