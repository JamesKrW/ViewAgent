"""Harness-facing model client routing.

Backends remain deliberately outside this module. A role may be backed by vLLM, SGLang,
or a hosted service as long as the object exposes the existing ``create``/``size``
surface. This is composition, not a second backend registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class MissingModelRole(ValueError):
    """A harness requested a model role that the run did not configure."""


class RoleClients:
    """Immutable collection of named harness clients.

    Attribute access is supported for concise custom harnesses (``clients.planner``),
    while ``for_role`` gives built-ins an explicit, inspectable path.
    """

    def __init__(self, clients: Mapping[str, Any]) -> None:
        values = {str(role).strip(): client for role, client in clients.items()}
        if not values or any(not role for role in values):
            raise ValueError("role clients require at least one non-empty role name")
        if any(client is None for client in values.values()):
            raise ValueError("role clients cannot contain None")
        self._clients = MappingProxyType(values)

    @property
    def roles(self) -> frozenset[str]:
        return frozenset(self._clients)

    def for_role(self, role: str) -> Any:
        try:
            return self._clients[role]
        except KeyError as exc:
            raise MissingModelRole(
                f"model role {role!r} is not configured; available roles: "
                f"{sorted(self._clients)}"
            ) from exc

    def __getattr__(self, role: str) -> Any:
        if role.startswith("_"):
            raise AttributeError(role)
        try:
            return self.for_role(role)
        except MissingModelRole as exc:
            raise AttributeError(role) from exc


def client_for_role(client: Any, role: str) -> Any:
    """Resolve one named client, retaining compatibility with single-model harnesses."""

    resolver = getattr(client, "for_role", None)
    if callable(resolver):
        return resolver(role)
    if role == "default":
        return client
    raise MissingModelRole(
        f"harness requires model role {role!r}, but the runner supplied one unlabelled "
        "client; configure a role-based model bundle"
    )


__all__ = ["MissingModelRole", "RoleClients", "client_for_role"]
