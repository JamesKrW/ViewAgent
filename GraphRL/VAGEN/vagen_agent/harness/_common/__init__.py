"""Shared harness contract and message helpers."""

from vagen_agent.harness._common.base_harness import (
    BaseHarness,
    Msg,
    assistant,
    obs_to_message,
    user,
)
from vagen_agent.harness._common.clients import (
    MissingModelRole,
    RoleClients,
    client_for_role,
)

__all__ = [
    "BaseHarness",
    "MissingModelRole",
    "Msg",
    "RoleClients",
    "assistant",
    "client_for_role",
    "obs_to_message",
    "user",
]
