"""The archiving seam: the one place that sees an env return value, a call id and a span.

It wraps the environment. The harness calls ``env.step(response)`` and gets the ordinary
gym five-tuple back, reward included -- it may read it, branch on it, rerank with it. What
it does *not* do is file it: normalising the reward to a vector and recording it against
``response.call_id`` happens here, as a side effect, before the harness sees anything.

That split is deliberate. Hiding the reward would make "the harness does not touch reward"
a structural fact, but it would also mean a non-standard env signature, and someone's
existing gym loop would stop working -- which is the thing this refactor exists to make
possible. What actually needs guaranteeing is narrower: the harness holds a value it may
read, not an obligation it can forget. Forgetting to register is not reachable, because
registering is not something it does.

Normalising here rather than in each env is what removes the old ``inspect.signature``
probe: an env never has to know how long its own response was.
"""

from __future__ import annotations

from vagen_agent.rollout.trajectory import Record


class RewardMisaligned(ValueError):
    """A reward vector does not describe the response it was returned for."""


def normalise(reward, n: int) -> list[float]:
    """A reward, as a vector of length ``n``.

    A scalar lands on the **last** token -- the outcome-reward convention, and not a
    degradation: if the environment only computed one number there are no tokens that
    individually earned it. Where the score sits is load-bearing, not cosmetic: at
    gamma = lambda = 1 the return at t is the suffix sum from t, so a score at the last
    position is seen by every token of the response and a score at position j is seen by
    none after j. That difference is the whole of ``default_gae`` vs ``token_level_gae``.
    """
    if isinstance(reward, (list, tuple)):
        vector = [float(v) for v in reward]
        if len(vector) != n:
            raise RewardMisaligned(
                f"the environment returned {len(vector)} rewards for a {n}-token "
                f"response. A vector must be aligned to response.token_ids, one value "
                f"per token.")
        return vector
    if n == 0:
        # An aborted generation produced nothing; there is nowhere to put the credit.
        return []
    return [0.0] * (n - 1) + [float(reward)]


class ScoringSeam:
    """``env``, with reward archiving in front of it."""

    def __init__(self, env, record: Record, *, seed=None):
        self.env = env
        self.record = record
        self.seed = seed

    def __getattr__(self, name):
        # Anything this wrapper has no opinion about belongs to the environment
        # underneath -- `success`, the renderer, whatever a caller reaches for.
        return getattr(self.env, name)

    async def reset(self, seed=None):
        return await self.env.reset(self.seed if seed is None else seed)

    async def system_prompt(self):
        return await self.env.system_prompt()

    async def close(self):
        await self.env.close()

    async def step(self, response):
        obs, reward, terminated, truncated, info = await self.env.step(response)

        self._archive(response, reward, info)
        self.record.terminated = bool(terminated)
        self.record.truncated = bool(truncated)

        return obs, reward, terminated, truncated, info

    def _archive(self, response, reward, info) -> None:
        """Attach one scalar/vector reward to the exact sampled token sequence."""

        vector = normalise(reward, len(response.token_ids))
        self.record.rewards[response.call_id] = vector
        self.record.turns += 1
        for node in _generated_nodes(self.record):
            if node.gen.call_id == response.call_id:
                node.gen.info = dict(info or {})
                break


def _generated_nodes(record: Record):
    stack = [record.root]
    while stack:
        node = stack.pop()
        if node.is_generated:
            yield node
        stack.extend(node.children)


__all__ = ["RewardMisaligned", "ScoringSeam", "normalise"]
