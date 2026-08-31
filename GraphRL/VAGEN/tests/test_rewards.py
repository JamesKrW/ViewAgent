"""The per-token reward path, and the identity it rests on.

Two things are checked, both on CPU:

1. **The identity.** Under ``no_concat`` an episode is several rows, and slime's GAE runs
   per row. `episode_reward_vectors` folds each later row's total into the previous row's
   last position so that a row-local suffix sum equals the episode-global one. At
   gamma = lam = 1 GAE telescopes to ``A_t = (suffix sum) - V(t)``, so if the identity
   holds, running slime's own per-row GAE gives exactly the advantages an episode-global
   estimator would. That is the whole reason `token_level_gae` needs no custom estimator.

2. **The wiring.** The rollout puts the vector in ``metadata`` and keeps ``Sample.reward``
   scalar; `post_process` swaps the vector in for `rewards` while leaving `raw_reward`
   scalar. Getting this backwards breaks slime's metrics rather than the training, which
   is the harder failure to notice.

The slime side of the feature -- the ppo branch adding a vector elementwise instead of at
``[-1]`` -- is exercised indirectly here by using the same `vanilla_gae` it feeds.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.utils.ppo_utils import vanilla_gae  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

from vagen_agent.algorithms import episode_reward_vectors, post_process  # noqa: E402


def pad(rows, width):
    out = torch.zeros(len(rows), width)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype=torch.float32)
    return out


def test_identity():
    """Per-row GAE on the folded vectors == episode-global GAE on the concatenation."""
    torch.manual_seed(0)
    # Three rows of an episode, each with its own per-turn rewards at arbitrary positions.
    row_rewards = [[0.0, 0.02, 0.0], [0.0, 0.0, 0.02, 0.0], [0.0, 1.02]]
    lengths = [len(r) for r in row_rewards]
    values = [torch.randn(n).tolist() for n in lengths]

    # --- reference: one sequence, episode-global GAE -------------------------
    flat_r = [x for r in row_rewards for x in r]
    flat_v = [x for v in values for x in v]
    ref_adv, ref_ret = vanilla_gae(pad([flat_r], len(flat_r)), pad([flat_v], len(flat_v)),
                                   gamma=1.0, lambd=1.0)
    ref_adv, ref_ret = ref_adv[0], ref_ret[0]

    # --- what this port does: fold, then per-row GAE -------------------------
    folded = episode_reward_vectors(row_rewards)
    width = max(lengths)
    per_row_adv, per_row_ret = vanilla_gae(pad(folded, width), pad(values, width),
                                           gamma=1.0, lambd=1.0)
    got_adv = torch.cat([per_row_adv[i, : lengths[i]] for i in range(len(lengths))])
    got_ret = torch.cat([per_row_ret[i, : lengths[i]] for i in range(len(lengths))])

    assert torch.allclose(got_adv, ref_adv, atol=1e-5), f"\n{got_adv}\n{ref_adv}"
    assert torch.allclose(got_ret, ref_ret, atol=1e-5), f"\n{got_ret}\n{ref_ret}"
    # The invariant folding actually maintains: each row's suffix sum equals the episode's
    # suffix sum at the same token. (The sum across rows is NOT preserved and must not be
    # checked -- every row carries the whole future, so it grows.)
    k = 0
    for row in folded:
        for j in range(len(row)):
            assert abs(sum(row[j:]) - sum(flat_r[k + j:])) < 1e-9, (row, j, k)
        k += len(row)
    print(f"  identity holds over {len(flat_r)} tokens / {len(lengths)} rows "
          f"(max |diff| = {(got_adv - ref_adv).abs().max():.2e})")


def test_identity_breaks_without_folding():
    """The naive version -- each row keeps only its own rewards -- must NOT match.

    Without this the first test would pass for a trivial reason (e.g. all-zero rewards) and
    would keep passing if the folding were removed.
    """
    row_rewards = [[0.0, 0.02, 0.0], [0.0, 0.0, 0.02, 0.0], [0.0, 1.02]]
    lengths = [len(r) for r in row_rewards]
    values = [[0.0] * n for n in lengths]
    width = max(lengths)
    naive, _ = vanilla_gae(pad(row_rewards, width), pad(values, width), gamma=1.0, lambd=1.0)
    folded, _ = vanilla_gae(pad(episode_reward_vectors(row_rewards), width),
                            pad(values, width), gamma=1.0, lambd=1.0)
    assert not torch.allclose(naive, folded), "folding made no difference -- test is vacuous"
    print("  unfolded rewards give different advantages, as they must")


def test_post_process_swaps_the_vector_in():
    class Args:
        def __init__(self, **kw): self.__dict__.update(kw)
        def __getattr__(self, name): return None

    args = Args(advantage_estimator="ppo", reward_key=None)
    samples = []
    for i, vec in enumerate([[0.0, 0.02], [0.0, 1.02]]):
        s = Sample(index=i, group_index=0, rollout_id=0, tokens=[1, 2, 3], response_length=2,
                   loss_mask=[1, 1], rollout_log_probs=[0.0, 0.0], reward=1.04)
        s.metadata = {"per_token_reward": vec}
        samples.append(s)

    raw, rewards = post_process(args, samples)
    assert raw == [1.04, 1.04], raw                       # scalar, for metrics / pass@k
    assert rewards == [[0.0, 0.02], [0.0, 1.02]], rewards  # vector, for the ppo branch
    print("  raw_reward stays scalar; rewards carries the per-token vector")

    # A batch that mixes the two would silently apply two credit assignments.
    samples[1].metadata = {}
    try:
        post_process(args, samples)
    except ValueError as exc:
        print(f"  mixed batch refused: {str(exc)[:60]}...")
    else:
        raise AssertionError("a half-vector batch should be refused")


if __name__ == "__main__":
    for fn in (test_identity, test_identity_breaks_without_folding,
               test_post_process_swaps_the_vector_in):
        print(f"{fn.__name__}:")
        fn()
    print("\nOK: per-token reward path")
