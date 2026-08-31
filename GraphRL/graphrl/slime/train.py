"""Synchronous SLIME driver with an unambiguous step-0 validation."""

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

from graphrl.adaptive.model import write_checkpoint_manifest


def _commit_checkpoint(args, rollout_id: int) -> None:
    checkpoint = Path(args.save) / f"iter_{int(rollout_id):07d}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"SLIME actor checkpoint is missing: {checkpoint}")
    related = []
    critic = Path(args.save) / "critic" / checkpoint.name
    dataset_state = (
        Path(args.save) / "rollout" / f"global_dataset_state_dict_{int(rollout_id)}.pt"
    )
    if args.use_critic:
        if not critic.is_dir():
            raise RuntimeError(f"SLIME critic checkpoint is missing: {critic}")
        related.append(critic)
    if args.rollout_global_dataset:
        if not dataset_state.is_file():
            raise RuntimeError(
                f"SLIME rollout dataset checkpoint is missing: {dataset_state}"
            )
        related.append(dataset_state)
    write_checkpoint_manifest(checkpoint, tuple(related))


def train(args):
    configure_logger()
    groups = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, per_epoch = create_rollout_manager(args, groups["rollout"])
    actor, critic = create_training_models(args, groups, rollout_manager)

    actor.update_weights()
    # -1 is reserved for the pre-training baseline. The logging hook maps it to
    # step 0; completed rollout r maps to step r+1.
    if args.start_rollout_id == 0 and args.eval_interval is not None and not args.skip_eval_before_train:
        ray.get(rollout_manager.eval.remote(rollout_id=-1))
    if args.num_rollout == 0 and args.eval_interval is not None:
        if args.start_rollout_id != 0 or args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id=-1))
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)
        return

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            values = critic.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor.async_train(rollout_id, rollout_data_ref, external_data=values))
            else:
                ray.get(values)
        else:
            ray.get(actor.async_train(rollout_id, rollout_data_ref))

        actor.clear_memory()
        actor.update_weights()
        if should_run_periodic_action(rollout_id, args.eval_interval, per_epoch, args.num_rollout):
            ray.get(rollout_manager.eval.remote(rollout_id))

        # Publish the checkpoint only after rollout export and same-step eval
        # have both finished.  A tracker is therefore never ahead of the
        # observable training/evaluation state after a crash.
        if should_run_periodic_action(
            rollout_id, args.save_interval, per_epoch, args.num_rollout
        ):
            if actor_trains:
                actor.save_model(rollout_id, force_sync=True)
            if args.use_critic:
                critic.save_model(rollout_id, force_sync=True)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            _commit_checkpoint(args, rollout_id)

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    train(parse_args())
