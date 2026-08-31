"""Multipart transport for JSON metadata and image observations.

Adapted from VAGEN's ``vagen.envs_remote.multipart_codec``.  Images stay as
binary multipart parts instead of being base64-expanded inside JSON.
"""

from __future__ import annotations

import io
import json
import uuid
from typing import Any

from PIL import Image


def encode_multipart(
    data: dict[str, Any],
    images: list[Image.Image] | None = None,
    *,
    image_format: str = "PNG",
    image_mime: str = "image/png",
    image_options: dict[str, Any] | None = None,
    boundary_prefix: str = "gym_env_",
) -> tuple[str, bytes]:
    """Encode JSON plus optional PIL images as one multipart body."""

    boundary = f"{boundary_prefix}{uuid.uuid4().hex}"
    marker = boundary.encode("ascii")
    crlf = b"\r\n"
    body = bytearray()

    payload = json.dumps(data or {}, ensure_ascii=False).encode("utf-8")
    body += b"--" + marker + crlf
    body += b'Content-Disposition: form-data; name="data"' + crlf
    body += b"Content-Type: application/json; charset=utf-8" + crlf + crlf
    body += payload + crlf

    save_options = dict(image_options or {})
    if "format" in save_options:
        raise ValueError("image_options must not override image_format")
    suffix = (
        ".jpg"
        if image_format.upper() in {"JPEG", "JPG"}
        else f".{image_format.lower()}"
    )
    for index, image in enumerate(images or []):
        buffer = io.BytesIO()
        encoded_image = image
        if image_format.upper() in {"JPEG", "JPG"} and image.mode not in {"RGB", "L"}:
            encoded_image = image.convert("RGB")
        encoded_image.save(buffer, format=image_format, **save_options)
        body += b"--" + marker + crlf
        body += (
            f'Content-Disposition: form-data; name="images"; filename="{index}{suffix}"'
        ).encode() + crlf
        body += f"Content-Type: {image_mime}".encode("ascii") + crlf + crlf
        body += buffer.getvalue() + crlf

    body += b"--" + marker + b"--" + crlf
    return boundary, bytes(body)


def _extract_boundary(content_type: str) -> str:
    for part in (content_type or "").split(";"):
        key, separator, value = part.strip().partition("=")
        if separator and key.lower() == "boundary" and value.strip('"'):
            return value.strip('"')
    raise ValueError(f"missing multipart boundary: {content_type!r}")


def decode_multipart(
    content_type: str, body: bytes
) -> tuple[dict[str, Any], list[Image.Image]]:
    """Decode a body produced by :func:`encode_multipart`."""

    marker = ("--" + _extract_boundary(content_type)).encode("ascii")
    data: dict[str, Any] = {}
    images: list[Image.Image] = []
    for chunk in body.split(marker):
        chunk = chunk.strip()
        if not chunk or chunk == b"--":
            continue
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip()
        header_blob, separator, payload = chunk.partition(b"\r\n\r\n")
        if not separator:
            continue
        payload = payload.rstrip(b"\r\n")
        part_type = ""
        for line in header_blob.decode("utf-8", errors="ignore").split("\r\n"):
            key, colon, value = line.partition(":")
            if colon and key.strip().lower() == "content-type":
                part_type = value.strip().lower()
        if "application/json" in part_type:
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError("multipart JSON part must be an object")
            data = value
        elif part_type.startswith("image/"):
            image = Image.open(io.BytesIO(payload))
            image.load()
            images.append(image.convert("RGB"))
    return data, images


__all__ = ["decode_multipart", "encode_multipart"]
