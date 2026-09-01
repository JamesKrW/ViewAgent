"""Concat until a budget is hit, then summarise and start again.

Closely related to CompactionRL (arXiv:2607.05378, Li et al. 2026), which trains task
execution and summary generation jointly under context compaction. Here too the summary is
written by the policy and carries gradient like any other turn -- it is not a free
preprocessing step. It is also the one model call in this repo that no environment step
follows, which is why the leaf filter is written as "did any response on this row get a
reward filed" and not "did this response get one": the summary rides on the same row as
the turns around it, and is trained with them.
"""

from __future__ import annotations

from vagen_agent.envs._common.base_compact_env import BaseCompactEnv
from vagen_agent.harness._common.base_harness import (
    BaseHarness,
    assistant,
    obs_to_message,
    user,
)


class CompactionMakesNoProgress(RuntimeError):
    """Conversation after conversation closed after a single turn.

    Not a slow run: the episode still finishes, every row is well-formed, and the only
    trace is a rollout that cost twice what it should and a summary per environment step.
    One turn per conversation is not compaction, it is no_concat at twice the price.
    Nothing downstream distinguishes it from compaction working.

    Raised on a repeat, not on the first one. A single conversation cut short by an
    unusually large observation is data; two in a row cannot be, because the second opened
    on a summary written under this budget -- if that still leaves no room, nothing later
    will.
    """

SUMMARY_REQUEST = "Summarise the conversation so far. Keep every fact needed to continue."
#: What the summary is wrapped in when it seeds the next conversation.
SUMMARY_PREFIX = "Summary so far: "


def _with_summary(summary: str, observation: dict) -> dict:
    """The summary and the observation as one user message.

    Not two messages in a row. What opens a new conversation is a single user turn -- here
    is the story so far, and here is where you are now -- so the exchange stays
    system / user / assistant. Chat templates are not obliged to handle two consecutive
    user messages the same way and an episode should not depend on which way they do.

    The blank line is part of the summary text rather than something the caller adds: a
    parts list is concatenated by the template with nothing between the parts, so
    separating them only in the string branch ran the summary straight into the
    observation with no boundary at all.
    """
    body = f"{summary}\n\n"
    content = observation.get("content")
    if isinstance(content, str):
        merged = f"{body}{content}"
    elif isinstance(content, list):
        merged = [{"type": "text", "text": body}, *content]
    else:
        merged = summary
    return {**observation, "role": "user", "content": merged}


class CompactHarness(BaseHarness):
    """Concat until the conversation is about to outgrow ``budget``, then ask the model to
    summarise and reseed from the summary.

    The trigger fires on "another turn would not fit" rather than "we are already over",
    against a turn cost **measured** as the episode goes. Charging the configured ceiling
    instead fires after the first turn of every conversation and compaction degenerates
    into no_concat with a summary attached: on Sokoban the ceiling is 512 and a real turn
    is about 80.

    The estimate is the *last* continuation's cost, not the largest. A maximum stops
    predicting and starts remembering -- one response at the ceiling, which is exactly
    what the configuration permits, would set the estimate for the rest of the episode and
    cut every later conversation off after a single turn. Being wrong low costs one turn
    of overshoot; being wrong high has no bound.
    """

    environment_type = BaseCompactEnv

    def __init__(self, budget: int | None = None, summary_budget: int | None = None,
                 window: int | None = None, response_limit: int | None = None):
        self.budget = budget
        self.summary_budget = summary_budget
        self.window, self.response_limit = window, response_limit

    async def run_episode(self, client, env):
        initial, _reset_info = await env.reset()
        observation = obs_to_message(initial)
        system = obs_to_message(await env.system_prompt(), role="system")
        messages = [system, observation]
        # The observation the environment just returned is held *out* of the conversation
        # until we have decided what to do with it. That is the whole reason compaction
        # can reseed cheaply: if it goes in first and we then compact, it is rendered
        # into the conversation being discarded *and* into the summary that replaces it,
        # so the episode pays for it twice.
        pending: dict | None = None
        # `used` is how large the current conversation has grown; `turn_cost` is what one
        # more turn of it is expected to cost. Both are per-conversation and both reset on
        # a reseed -- carried across, a single expensive turn keeps predicting for
        # conversations it was never part of, and it cannot be corrected because an
        # opening call is not a turn and may not inform the estimate.
        used = turn_cost = 0
        opening = True
        turns_here = short_streak = 0

        while True:
            if pending is not None:
                if not opening and self.budget and used + turn_cost >= self.budget:
                    short_streak = short_streak + 1 if turns_here <= 1 else 0
                    if short_streak >= 2:
                        raise CompactionMakesNoProgress(
                            f"{short_streak} conversations in a row closed after a single "
                            f"turn. This one had grown to {used} tokens with a turn costing "
                            f"about {turn_cost}, against compact_budget={self.budget} and "
                            f"compact_summary_budget={self.summary_budget}. Compacting buys "
                            f"no turns that way and every environment step costs two "
                            f"generations. Raise compact_budget, or lower "
                            f"compact_summary_budget / the per-turn response length.")
                    summary_messages = [*messages, user(SUMMARY_REQUEST)]
                    summary = await client.create(
                        summary_messages,
                        **self.room(client, summary_messages, self.window,
                                    self.summary_budget or self.response_limit))
                    messages = [system,
                                _with_summary(f"{SUMMARY_PREFIX}{summary.text}", pending)]
                    used = turn_cost = 0
                    opening, turns_here = True, 0
                else:
                    messages.append(pending)
                pending = None

            response = await client.create(
                messages, **self.room(client, messages, self.window,
                                      self.response_limit))
            # Native SGLang rollouts carry exact token ids. OpenAI-compatible APIs do
            # not, but do report completion_tokens in usage. Use that common accounting
            # surface so the same harness works for local engines and hosted models.
            completion_tokens = (
                response.usage.completion_tokens
                if response.usage.completion_tokens
                else len(response.token_ids)
            )
            grown = response.usage.prompt_tokens + completion_tokens
            if not opening:
                turn_cost = grown - used
            used, opening = grown, False

            turns_here += 1
            messages.append(assistant(response.text))
            obs, _reward, terminated, truncated, _info = await env.step(response)
            if terminated or truncated:
                return
            pending = obs_to_message(obs)
