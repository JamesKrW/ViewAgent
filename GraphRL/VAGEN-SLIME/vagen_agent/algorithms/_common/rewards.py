"""Turning what the environment paid into what the estimator reads.

The environment credits a reward to the turn that earned it (``tape.Conversation.add_reward``
writes it on that turn's last model token). What an estimator wants is not always that
shape, and under ``no_concat`` / ``compact`` the episode is spread over several rows. This
module is where the two are reconciled -- on the rollout side, in plain arithmetic, which
is what lets slime's own per-row GAE stand in for an episode-global estimator.

The derivation is in :func:`episode_reward_per_row`. It is short, and it is the reason this
port needs no changes to slime.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ scalar path
def episode_reward_per_row(episode_total: float, n_rows: int) -> list[float]:
    """The scalar every row of one episode should carry, for ``default_gae``.

    **The identity this port rests on.** At ``gamma = lam = 1`` GAE telescopes::

        delta_t = r_t + V(t+1) - V(t)
        A_t     = delta_t + A_{t+1}
        =>  A_t = sum_{s >= t} r_s  -  V(t)          and   returns_t = sum_{s >= t} r_s

    VAGEN's ``default_gae`` packs an episode's rows into one sequence, moves every reward
    the episode collected onto its **last** model token, and runs that recursion. So for
    every token ``j`` anywhere in the episode::

        A_j = R_total - V(j)            returns_j = R_total

    slime runs the same recursion per row, with the row's scalar reward placed at the row's
    last response position. Give each row ``R_total`` and it computes::

        A_t = R_total - V(t)            returns_t = R_total

    -- the same numbers, at every mask-1 position, with no cross-row recursion, no
    cross-rank gather and no patch to slime. Both quantities that differ between the two
    layouts (where the reward sits, where the recursion restarts) cancel exactly, and only
    at ``gamma = lam = 1``. ``RunConfig._check_against`` refuses anything else.

    Note this is **not** "split the reward across the rows". Each row gets the *whole*
    episode total. Dividing it by ``n_rows`` would be the intuitive thing and would give
    every token ``R_total / n`` where the reference gives ``R_total`` -- a rescaling that
    survives whitening as a *different* relative weighting between episodes of different
    length, i.e. a length bias with no reward-model basis.
    """
    if n_rows <= 0:
        return []
    return [float(episode_total)] * n_rows


# --------------------------------------------------------------- per-token path
def episode_reward_vectors(row_rewards: list[list[float]]) -> list[list[float]]:
    """Per-token reward vectors that make each row's suffix-sum the *episode's* suffix-sum.

    For ``token_level_gae``, where a turn's reward stays on the turn that earned it. At
    ``gamma = lam = 1`` the target is ``A_j = G_j - V(j)`` with ``G_j`` summed over the rest
    of the **episode**, but slime's per-row recursion only ever sums over the rest of the
    **row**. The fix is arithmetic and belongs here rather than in an estimator: add the
    total of all *later* rows onto the current row's last position. Then::

        row i's suffix sum at j  =  (its own rewards from j on) + (rows i+1..N total)
                                 =  the episode's suffix sum at j

    Args:
        row_rewards: one list per row, aligned to that row's response tokens.

    Returns:
        New vectors of the same shapes, with the future-rows total folded into each row's
        final position.

    Note the sum *across* rows is deliberately NOT preserved -- it grows, because every row
    has to see the whole future independently. What is preserved is the thing the estimator
    reads: each row's suffix sum equals the episode's suffix sum at the same token. Summing
    the folded rows against the original total is a tempting check and a wrong one.
    """
    vectors = [list(r) for r in row_rewards]
    running = 0.0
    # Walk backwards so `running` is the total of everything strictly after row i.
    for i in range(len(vectors) - 1, -1, -1):
        own = sum(vectors[i])
        if vectors[i]:
            vectors[i][-1] += running
        elif running:
            # A row with no response tokens cannot be credited, and dropping the tail
            # silently would lose reward the episode really earned.
            logger.warning("row %d has no response tokens; %.4f of later-row reward has "
                           "nowhere to land", i, running)
        running += own
    return vectors


# ------------------------------------------------------------- cross-row folding
def episode_scalar(record) -> float:
    """What this episode was worth, as one number.

    Computed from the raw per-call vectors, **before** any fold. Under
    ``token_level_gae`` a folded row sums to more than the episode total -- every row has
    to see the whole future independently -- so ``sum(folded)`` is not this. slime's
    metrics, pass@k and the zero-std check all read ``Sample.reward`` as "what this
    episode scored".
    """
    return float(sum(sum(v) for v in record.rewards.values()))


def fold_across_rows(rows: list[list[float]], algorithm: str) -> list[list[float]]:
    """Per-row score vectors, adjusted so slime's per-row GAE reproduces VAGEN's
    episode-global estimator.

    One signature, two policies, and the difference is only *where the future goes*:

    ``default_gae``      every row carries the episode total on its last position, which
                         at gamma = lam = 1 gives every token ``A = R_total - V(t)`` --
                         exactly what an episode-global recursion produces. Not
                         ``R_total / n``: dividing would rescale by episode length, a
                         length bias with no reward-model basis.
    ``token_level_gae``  each turn's reward stays where it was earned, and the total of
                         all *later* rows is added to this row's last position, so a
                         row-local suffix sum equals the episode-global one.
    """
    from vagen_agent.algorithms._common.spec import resolve_algorithm

    spec = resolve_algorithm(algorithm)
    if spec.reward_folding == "identity":
        # This algorithm brings its own advantage function, so it reads the rewards the
        # environment actually gave. Folding exists only to make slime's *native* per-row
        # GAE reproduce an episode-global estimator; applied here it would destroy the
        # per-turn structure the custom estimator was written to use, and silently -- the
        # run would train, on rewards nobody assigned.
        return [list(r) for r in rows]
    if spec.reward_folding == "token_suffix":
        return episode_reward_vectors(rows)
    total = float(sum(sum(r) for r in rows))
    return [([0.0] * (len(r) - 1) + [total]) if r else [] for r in rows]


# --------------------------------------------------- slime reward post-process
def post_process(args, samples):
    """``--custom-reward-post-process-path``: group-relative normalisation, by episode.

    Only needed for the GRPO-shaped estimators. slime's own version does::

        rewards.reshape(-1, args.n_samples_per_prompt)

    which assumes **one training sample per rollout**. Under ``no_concat`` an episode
    becomes a variable number of rows, so that reshape either raises or -- worse, when the
    counts happen to divide -- silently groups rows from different prompts together and
    normalises against the wrong baseline.

    The unit of normalisation is the **episode**, not the row. Every row of an episode
    carries the same ``R_total``, so counting them once each would weight an episode by how
    many turns it happened to take: a long episode would pull the group mean harder than a
    short one, and the policy would be rewarded for verbosity through the baseline rather
    than through the reward.

    Returns ``(raw_rewards, rewards)``: the first is what the metrics and pass@k read and
    stays untouched, the second is what the estimator consumes.
    """
    raw = [_scalar(s.get_reward_value(args)) for s in samples]

    # Per-token rewards, when the rollout produced them. `raw_reward` stays scalar -- it is
    # what pass@k and the reward histograms read -- while `rewards` becomes the vector the
    # ppo branch adds elementwise into the KL tensor. slime's
    # get_advantages_and_returns_batch is documented in exactly this shape
    # ("rewards_list: list[Tensor], each shape = [resp_len_i]"); the scalar path only
    # exists to construct it.
    vectors = [(getattr(s, "metadata", None) or {}).get("per_token_reward") for s in samples]
    if any(v is not None for v in vectors):
        if not all(v is not None for v in vectors):
            raise ValueError(
                "some samples carry a per-token reward and some do not. Mixing the two in "
                "one batch would give the scalar rows their reward at the last token and "
                "the others theirs where it was earned -- two different credit assignments "
                "under one estimator."
            )
        return raw, [list(v) for v in vectors]

    if args.advantage_estimator not in ("grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"):
        return raw, raw
    if not getattr(args, "rewards_normalization", True):
        return raw, raw

    # group -> episode -> that episode's reward (identical across its rows)
    per_group: dict[object, dict[object, float]] = defaultdict(dict)
    for sample, reward in zip(samples, raw, strict=True):
        per_group[sample.group_index][_episode_key(sample)] = reward

    use_std = getattr(args, "grpo_std_normalization", True)
    centred: dict[tuple, float] = {}
    for group, episodes in per_group.items():
        values = list(episodes.values())
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        for key, value in episodes.items():
            adv = value - mean
            if use_std:
                # A group whose episodes all scored the same carries no signal; dividing by
                # its zero std would produce NaNs rather than zeros.
                adv = adv / (std + 1e-6) if std > 0 else 0.0
            centred[(group, key)] = adv

    rewards = [centred[(s.group_index, _episode_key(s))] for s in samples]
    return raw, rewards


def _episode_key(sample):
    """What identifies an episode. ``rollout_id`` is set to the same value on every row of
    one episode by ``emit.row_to_sample``; ``index`` is slime's own fallback."""
    rid = getattr(sample, "rollout_id", None)
    return rid if rid is not None else getattr(sample, "index", None)


def _scalar(reward) -> float:
    """A row's reward as one number, whatever shape it was stored in."""
    if isinstance(reward, (list, tuple)):
        return float(sum(reward))
    return float(reward)


__all__ = ["episode_reward_per_row", "episode_reward_vectors", "episode_scalar",
           "fold_across_rows", "post_process"]
