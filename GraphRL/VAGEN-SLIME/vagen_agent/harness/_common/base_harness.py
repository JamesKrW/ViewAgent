"""The context-policy contract.

A harness decides what context each model call gets, and when the episode is over. That is
all it is now: one abstract method containing ordinary agent code.

It holds a messages list, calls ``client.create(messages)``, steps the environment with
what came back, and loops until the environment says stop. It never sees a token, a mask
or a Sample, and it never *files* a reward -- the archiving seam does that before the
value reaches it (``rollout/scoring.py``). It may read the reward and act on it.

What used to be here and is not any more: ``next_call`` / ``accept``, and the room
accounting (``note_room`` / ``note_usage`` / ``exhausted`` / ``pending_observation`` /
``continues_conversation`` / ``_left`` / ``_reserve`` / ``max_new_tokens``). Those existed
so a harness could make budget decisions without being allowed to hold the client. It
holds the client now, so they are local variables in ``run_episode``.

Nor does a harness count turns. ``BaseVagenEnv`` owns ``max_turns`` for the built-in
environment families and reports exhaustion as ``truncated``. The
backstop against a harness that generates without ever stepping is the client's
per-episode call ceiling, which a turn count could not catch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from vagen_agent.envs._common.base_env import BaseEnv
from vagen_agent.envs._common.utils import observation_to_message

Msg = dict


class BaseHarness(ABC):
    #: Environment capability required by this episode policy.
    environment_type: ClassVar[type[BaseEnv]] = BaseEnv

    #: Named model clients required by this harness. Existing harnesses consume the
    #: ordinary single ``default`` client; composite harnesses declare their roles so
    #: configuration can fail before an episode starts.
    model_roles: ClassVar[tuple[str, ...]] = ("default",)

    #: Role whose output is sent to the environment. Evaluation uses this declaration
    #: to select the action response limit instead of knowing harness-specific names.
    action_model_role: ClassVar[str] = "default"

    #: Slime currently constructs one trainable client/trajectory per rollout. A harness
    #: using more than that must opt out until the training runner can construct and
    #: attribute several policy streams explicitly. Standalone evaluation is unaffected.
    supports_training: ClassVar[bool] = True

    @abstractmethod
    async def run_episode(self, client, env) -> None:
        """Run one episode to completion.

        Stop when the environment says so (``terminated or truncated``). The return value
        is ignored: the training data is in the record, not in what this returns.
        """

    # ------------------------------------------------------------------ helpers
    # Plain functions, not template methods -- nothing here is called by the framework.

    @staticmethod
    def room(client, messages, window: int | None,
             response_limit: int | None = None) -> dict:
        """Bound one generation by both its per-turn cap and the context window.

        ``response_limit`` is the configured response budget for one model call;
        ``window - prompt_tokens`` is how much room that call actually has.  Both are
        upper bounds, so the smaller one wins.  Treating the per-turn budget as a floor
        lets a 512-token Sokoban turn expand to almost the entire 8192-token context.
        """
        limits: list[int] = []
        if response_limit is not None:
            response_limit = int(response_limit)
            if response_limit < 1:
                raise ValueError(f"response_limit must be positive, got {response_limit}")
            limits.append(response_limit)

        if window:
            prompt_tokens = client.size(messages)
            remaining = int(window) - prompt_tokens
            if remaining < 1:
                raise RuntimeError(
                    f"the prompt already uses {prompt_tokens} tokens, leaving no room in "
                    f"the {window}-token context window")
            limits.append(remaining)

        if not limits:
            return {}
        return {"max_new_tokens": min(limits)}


def assistant(text: str) -> Msg:
    return {"role": "assistant", "content": text}


def user(text: str) -> Msg:
    return {"role": "user", "content": text}


def obs_to_message(obs, *, role: str = "user") -> Msg:
    """Environments speak in observations; harnesses speak in messages."""
    return observation_to_message(obs, role=role)


__all__ = ["BaseHarness", "Msg", "assistant", "obs_to_message", "user"]
