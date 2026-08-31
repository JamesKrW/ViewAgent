"""VAGEN on slime: the layer that replaces verl.

Four extension axes, one invariant rollout core. Change one axis without touching the
others:

    env        vagen_agent/envs/       env_registry.yaml + a BaseEnv subclass
    harness    vagen_agent/harness/    @register_harness (concat / no_concat / compact)
    model      vagen_agent/models/     @register_model   (how a family tokenizes)
    algorithm  vagen_agent/algorithms/ @register_algorithm (credit assignment)

Each extension axis keeps contracts under ``_common/``, concrete implementations in one
folder per type, and its registry/public facade in ``__init__.py``.
Standalone evaluation adds the same pattern under ``evaluation/backends/``.

``rollout.generate`` is the only entry point slime calls; it wires the axes together and
converts an episode into ``list[Sample]``. The inference call is not an axis: it is one
POST inside ``RolloutClient.create``. Token routing, frame storage, trajectory assembly,
reward alignment, and episode lifecycle are invariant rollout machinery rather than
customizable adapters.

What cannot be made orthogonal -- the yaml's ``algorithm:`` against slime's
``--advantage-estimator`` and ``--gamma``/``--lambd`` -- is checked when the config is
built, rather than failing silently as a plausible-looking curve.
"""
