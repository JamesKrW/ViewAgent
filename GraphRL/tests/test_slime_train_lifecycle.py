from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from graphrl.adaptive.state import DECISION_CONTINUE_RL


class _RemoteMethod:
    def __init__(self, events: list[str], name: str, result=None):
        self._events = events
        self._name = name
        self._result = result

    def remote(self, *args, **kwargs):
        self._events.append(self._name)
        return self._result


class _RolloutManager:
    def __init__(self, events: list[str]):
        self.onload_weights = _RemoteMethod(events, "rollout.onload_weights")
        self.onload_kv = _RemoteMethod(events, "rollout.onload_kv")
        self.generate = _RemoteMethod(events, "rollout.generate", "rollout-data")
        self.offload = _RemoteMethod(events, "rollout.offload")
        self.eval = _RemoteMethod(events, "rollout.eval")
        self.save = _RemoteMethod(events, "rollout.save")
        self.dispose = _RemoteMethod(events, "rollout.dispose")
        self.check_weights = _RemoteMethod(events, "rollout.check_weights")


class _Actor:
    def __init__(self, events: list[str]):
        self._events = events

    def create(self):
        self._events.append("actor.create")

    def update_weights(self):
        self._events.append("actor.update_weights")

    def async_train(self, *args, **kwargs):
        self._events.append("actor.async_train")
        return "actor-train"

    def clear_memory(self):
        self._events.append("actor.clear_memory")

    def save_model(self, *args, **kwargs):
        self._events.append("actor.save_model")


class _AdaptiveSchedule:
    round_index = 0

    def __init__(self):
        self.store = SimpleNamespace(load=lambda: {})

    def reconcile_resume(self, *args, **kwargs):
        return None

    def decision(self):
        return DECISION_CONTINUE_RL

    def commit(self, *args, **kwargs):
        raise AssertionError("this one-step lifecycle test must not commit")


def _args(*, adaptive: bool) -> SimpleNamespace:
    return SimpleNamespace(
        release_train=False,
        offload_rollout=True,
        offload_train=True,
        check_weight_update_equal=False,
        start_rollout_id=1 if adaptive else 0,
        num_rollout=2 if adaptive else 1,
        eval_interval=None,
        skip_eval_before_train=False,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=None,
        rollout_global_dataset=False,
        save="/unused",
    )


@pytest.mark.parametrize(
    ("module_name", "adaptive"),
    [
        ("graphrl.slime.train", False),
        ("graphrl.slime.train_adaptive", True),
    ],
)
def test_colocated_train_restores_rollout_storage_before_each_weight_update(
    monkeypatch, module_name: str, adaptive: bool
):
    module = importlib.import_module(module_name)
    events: list[str] = []
    rollout = _RolloutManager(events)
    actor = _Actor(events)

    monkeypatch.setattr(module, "configure_logger", lambda: None)
    monkeypatch.setattr(module, "init_tracking", lambda args: None)
    monkeypatch.setattr(module, "finish_tracking", lambda args: None)
    monkeypatch.setattr(module, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(module, "create_rollout_manager", lambda args, group: (rollout, None))
    monkeypatch.setattr(module, "create_training_models", lambda args, groups, manager: (actor, None))
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    if adaptive:
        monkeypatch.setattr(module, "_schedule", _AdaptiveSchedule)

    module.train(_args(adaptive=adaptive))

    assert events == [
        "rollout.onload_weights",
        "actor.update_weights",
        "rollout.onload_kv",
        "rollout.generate",
        "rollout.offload",
        "actor.async_train",
        "rollout.onload_weights",
        "actor.update_weights",
        "rollout.onload_kv",
        "rollout.dispose",
    ]
