"""Exercise the logging hooks for real, not just their signatures.

The first version of this test checked `inspect.signature` and passed while the module
imported `dict_add_prefix` from the wrong package -- because that import sits *inside* the
function, so nothing ran it until the first rollout, inside a Ray actor, after the engines
were up and the model was loaded. Call the functions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.utils import logging_utils
from slime.utils.types import Sample

from vagen_agent.metrics import episode_metrics, log_eval_rollout_data, log_rollout_data


def make(rid, success, turns, ntokens=40):
    s = Sample(index=rid, group_index=0, rollout_id=rid, prompt="p",
               tokens=list(range(ntokens)), response_length=10, loss_mask=[1]*10,
               rollout_log_probs=[0.0]*10, reward=float(success))
    s.metadata = {"metrics": {"success": float(success),
                               "action_valid_rate": 1.0 if success else 0.25},
                  "episode_turns": turns,
                  "episode_reward": float(success), "truncated": not success,
                  "terminated": bool(success), "source_name": "sokoban"}
    s.status = Sample.Status.COMPLETED
    return s


rows = [make(7, 1, 5) for _ in range(5)] + [make(8, 0, 1)]

m = episode_metrics(rows)
assert m["success"] == 0.5, m              # per episode, not per row (per-row would be .833)
assert m["action_valid_rate"] == 0.625, m
assert m["episodes"] == 2 and m["samples_per_episode"] == 3.0, m

# A stand-in for slime's args. Fields we care about are explicit; everything else falls
# back to None via __getattr__, which is the "unset" value for almost every optional slime
# flag. Enumerating the full surface instead would make this test a mirror of slime's
# argument parser -- it would break every time slime adds a flag, for no signal about the
# code under test.
class Args:
    def __init__(self, **kw): self.__dict__.update(kw)
    def __getattr__(self, name): return None


args = Args(
    rollout_batch_size=2, n_samples_per_prompt=1, num_rollout=10,
    global_batch_size=2, num_steps_per_rollout=1,
    advantage_estimator="ppo", rollout_global_dataset=True,
    use_wandb=False, use_tensorboard=False,
)

logged = []
original_log = logging_utils.log
logging_utils.log = lambda args, metrics, step_key: logged.append((metrics, step_key))
try:
    assert log_rollout_data(0, args, rows, {}, 1.0) is True
    assert log_eval_rollout_data(0, args, {"sokoban": {"samples": rows, "rewards": [1.0, 0.0]}}, {}) is True
finally:
    logging_utils.log = original_log

assert len(logged) == 2, logged
for metrics, step_key in logged:
    assert step_key == "rollout/step", (metrics, step_key)
    assert metrics["rollout/step"] == 0, metrics
    assert "eval/step" not in metrics, metrics
print("OK: episode dedup + both hooks execute end to end")
