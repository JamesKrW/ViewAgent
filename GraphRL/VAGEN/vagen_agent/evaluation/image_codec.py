"""Shared JPEG encoding for model requests and trajectory artifacts."""

from __future__ import annotations

import io
from typing import Any

IMAGE_FORMAT = "JPEG"
IMAGE_MIME = "image/jpeg"
IMAGE_SUFFIX = ".jpg"
JPEG_OPTIONS = {"quality": 90, "subsampling": 0}


def encode_jpeg(image: Any) -> bytes:
    """Encode a PIL-compatible image with the evaluation-wide JPEG policy."""

    buffer = io.BytesIO()
    rgb = image if getattr(image, "mode", None) == "RGB" else image.convert("RGB")
    rgb.save(buffer, format=IMAGE_FORMAT, **JPEG_OPTIONS)
    return buffer.getvalue()


__all__ = [
    "IMAGE_FORMAT",
    "IMAGE_MIME",
    "IMAGE_SUFFIX",
    "JPEG_OPTIONS",
    "encode_jpeg",
]
