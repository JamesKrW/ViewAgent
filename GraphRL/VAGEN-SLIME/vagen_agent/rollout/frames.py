"""Content-addressed frame table.

One entry per distinct picture, shared across every episode this process runs. A frame is
paid for once: encoded to base64 once (the engine re-processes the whole prompt on every
call, so the same frame goes over the wire on every turn of a concat episode) and run
through the processor once (a second pass would also be a second source of truth about how
it was tiled -- see ``models/_common/common.py::Rendered``).

Keyed by the content hash of the PNG bytes, not by object identity. Two renders of the
same Sokoban board must land on the same entry or nothing dedups. The hash deliberately
does **not** try to match sglang's ``pad_value`` hash, which is taken over the *server's*
processor output: the two numbers differ, but as long as both sides are deterministic they
partition frames the same way, and a partition is all we need.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Frame:
    key: str
    image: Any
    b64: str
    mm_train: dict[str, Any] | None = None


@dataclass
class FrameTable:
    entries: dict[str, Frame] = field(default_factory=dict)

    def key_of(self, image: Any) -> str:
        """The content hash of this picture, as the PNG bytes we would send."""
        buf = io.BytesIO()
        rgb = image if getattr(image, "mode", None) == "RGB" else image.convert("RGB")
        rgb.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()[:32]

    def intern(self, image: Any) -> str:
        """Return the key for this picture, encoding it only if it is new."""
        buf = io.BytesIO()
        rgb = image if getattr(image, "mode", None) == "RGB" else image.convert("RGB")
        rgb.save(buf, format="PNG")
        raw = buf.getvalue()
        key = hashlib.sha256(raw).hexdigest()[:32]
        if key not in self.entries:
            b64 = "data:image/png;base64," + base64.b64encode(raw).decode("utf-8")
            self.entries[key] = Frame(key=key, image=rgb, b64=b64)
        return key

    def intern_all(self, images) -> list[str]:
        return [self.intern(img) for img in images or ()]

    def b64_of(self, keys) -> list[str]:
        return [self.entries[k].b64 for k in keys]

    def images(self, keys) -> list[Any]:
        return [self.entries[k].image for k in keys]

    def merge_train_inputs(self, chunks: list[dict]) -> dict | None:
        """One concat per key along dim 0 -- the order the per-render pieces line up in.

        Non-tensor values are dropped rather than guessed at: there is no general way to
        combine them, and silently keeping the first turn's copy is wrong for anything
        that varies.
        """
        if not chunks:
            return None
        import torch

        by_key: dict[str, list] = {}
        for chunk in chunks:
            for key, value in (chunk or {}).items():
                if value is not None:
                    by_key.setdefault(key, []).append(value)
        merged = {}
        for key, values in by_key.items():
            if not all(hasattr(x, "dim") for x in values):
                continue
            trailing = {tuple(v.shape[1:]) for v in values}
            if len(trailing) > 1:
                # Every tensor here must describe the frames, so only the leading
                # dimension may differ. One that varies in a later dimension is aligned to
                # something else -- the token sequence, most likely -- and would either
                # fail inside slime's `get_batch` or, worse, concatenate into nonsense.
                raise ValueError(
                    f"multimodal input {key!r} cannot be concatenated: shapes "
                    f"{[tuple(v.shape) for v in values]} differ outside dimension 0, so it "
                    f"is not frame-aligned. Add it to the model adapter's token-aligned "
                    f"drop list.")
            merged[key] = torch.cat(values, dim=0)
        return merged or None


__all__ = ["Frame", "FrameTable"]
