"""Action value paired with the model response that produced it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvAction:
    value: Any
    response: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)


__all__ = ["EnvAction"]
