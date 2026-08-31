"""Validate model-neutral observations and render them as chat messages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from PIL import Image


def _images(values: Any) -> list[Any]:
    result = []
    for image in values or []:
        if image is None:
            continue
        result.append(image.convert("RGB") if isinstance(image, Image.Image) else image)
    return result


def validate_observation(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{source} observation must be a dict")
    if "content" in value:
        return value
    if not isinstance(value.get("obs_str"), str):
        raise TypeError(f"{source} observation must contain string obs_str")
    multimodal = value.get("multi_modal_input", {}) or {}
    if not isinstance(multimodal, Mapping):
        raise TypeError(f"{source} multi_modal_input must be a mapping")
    unsupported = [key for key, item in multimodal.items() if key != "<image>" and item]
    if unsupported:
        raise NotImplementedError(
            f"{source} returned unsupported multimodal keys {unsupported}; "
            "only '<image>' is supported"
        )
    images = list(multimodal.get("<image>", []) or [])
    if value["obs_str"].count("<image>") != len(images):
        raise ValueError(
            f"{source} returned {len(images)} images but "
            f"{value['obs_str'].count('<image>')} '<image>' placeholders"
        )
    return value


def observation_to_message(observation: Any, *, role: str = "user") -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"role": role, "content": str(observation)}
    observation = validate_observation(observation, source="environment")
    if "content" in observation:
        return {**observation, "role": role}

    text = observation["obs_str"]
    multimodal = observation.get("multi_modal_input", {}) or {}
    segments = re.split(r"(<image>)", text)
    content = [
        {"type": "image"}
        if segment == "<image>"
        else {"type": "text", "text": segment}
        for segment in segments
        if segment
    ]
    message = {
        "role": role,
        "content": content,
        "images": _images(multimodal.get("<image>", [])),
    }
    reserved = {"obs_str", "multi_modal_input", "role", "content", "images"}
    message.update(
        {key: item for key, item in observation.items() if key not in reserved}
    )
    return message


__all__ = ["observation_to_message", "validate_observation"]
