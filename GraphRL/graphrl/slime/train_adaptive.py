"""Synchronous SLIME loop controlled by GraphRL's metric-driven state machine."""

from __future__ import annotations

import os
from pathlib import Path

import ray

from slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action

from graphrl.adaptive.state import DECISION_CONTINUE_RL
from graphrl.slime.schedule import SlimeAdaptiveSchedule


def _schedule() -> SlimeAdaptiveSchedule:
    root = os.environ.get("GRAPHRL_ADAPTIVE_EXPERIMENT_DIR")
    round_value = os.environ.get("GRAPHRL_ADAPTIVE_ROUND")
    if not root or round_value is None:
        raise RuntimeError(
            "adaptive SLIME requires GRAPHRL_ADAPTIVE_EXPERIMENT_DIR and "
            "GRAPHRL_ADAPTIVE_ROUND"
        )
    return SlimeAdaptiveSchedule(root, int(round_value))


def _required_checkpoint_paths(args, rollout_id: int) -> tuple[Path, ...]:
    checkpoint = Path(args.save) / f"iter_{int(rollout_id):07d}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"SLIME actor checkpoint is missing: {checkpoint}")
    paths: list[Path] = []
    if args.use_critic:
        critic = Path(args.save) / "critic" / checkpoint.name
        if not critic.is_dir():
            raise RuntimeError(f"SLIME critic checkpoint is missing: {critic}")
        paths.append(critic)
    if args.rollout_global_dataset:
        dataset_state = (
            Path(args.save)
            / "rollout"
            / f"global_dataset_state_dict_{int(rollout_id)}.pt"
        )
        if not dataset_state.is_file():
            raise RuntimeError(
                f"SLIME rollout dataset checkpoint is missing: {dataset_state}"
            )
        paths.append(dataset_state)
    return tuple(paths)


def train(args):
    configure_logger()
    schedule = _schedule()
    release_train = args.release_train
    groups = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, per_epoch = create_rollout_manager(args, groups["rollout"])
    actor, critic = create_training_models(args, groups, rollout_manager)
    schedule.reconcile_resume(args.save, args.start_rollout_id)

    # Match SLIME's native synchronous colocate lifecycle. The rollout manager
    # is initially offloaded by create_rollout_manager(); restore its weights,
    # publish the actor weights, and only then restore KV/CUDA-graph memory.
    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())
    actor.update_weights()
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if args.start_rollout_id == 0:
        ray.get(rollout_manager.eval.remote(rollout_id=-1))
        schedule.commit(
            checkpoint_root=args.save,
            rollout_id=-1,
            hf_model=args.hf_checkpoint,
            initial=True,
        )

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())
        if release_train:
            actor.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            values = critic.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor.async_train(rollout_id, rollout_data_ref, external_data=values))
            else:
                ray.get(values)
        else:
            ray.get(actor.async_train(rollout_id, rollout_data_ref))

        periodic = should_run_periodic_action(
            rollout_id, args.save_interval, per_epoch, args.num_rollout
        )
        evaluate = should_run_periodic_action(
            rollout_id, args.eval_interval, per_epoch, args.num_rollout
        )
        checkpoint_before_eval = release_train or periodic or evaluate
        if checkpoint_before_eval:
            if actor_trains:
                actor.save_model(rollout_id, force_sync=True)
            if args.use_critic:
                critic.save_model(rollout_id, force_sync=True)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        # With offload_train, SLIME actors offload themselves after train().
        # Otherwise clear the model that occupied the GPU this step.
        if not args.offload_train:
            if not args.use_critic or actor_trains:
                actor.clear_memory()
            else:
                critic.clear_memory()

        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if evaluate:
            ray.get(rollout_manager.eval.remote(rollout_id))

        state = schedule.store.load()
        current = dict((state.get("rounds") or {}).get(str(schedule.round_index)) or {})
        is_best = int(current.get("best_step", -1)) == rollout_id + 1 and not current.get("checkpoint")
        terminal = schedule.decision() != DECISION_CONTINUE_RL
        if (is_best or terminal) and not checkpoint_before_eval:
            if actor_trains:
                actor.save_model(rollout_id, force_sync=True)
            if args.use_critic:
                critic.save_model(rollout_id, force_sync=True)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
        if periodic or is_best or terminal:
            hf_model = Path(args.save_hf.format(rollout_id=rollout_id))
            schedule.commit(
                checkpoint_root=args.save,
                rollout_id=rollout_id,
                hf_model=hf_model,
                related_paths=_required_checkpoint_paths(args, rollout_id),
            )
        if terminal:
            break

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    train(parse_args())
