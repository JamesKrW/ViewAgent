"""Things that are about *reporting* a run rather than producing it.

Nothing in here is on the rollout path: if a module here fails, the rollout must still
finish. See ``wandb_util`` for the one place that rule is enforced.
"""
