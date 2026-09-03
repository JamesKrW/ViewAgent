"""Translate GraphRL's pipeline RL block into an explicit SLIME launch spec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _at(value: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _first(*values: Any, default: Any = None) -> Any:
    return next((value for value in values if value is not None), default)


@dataclass(frozen=True)
class SlimeLaunchSpec:
    model_path: str
    output_dir: str
    train_envs: str
    eval_envs: str
    project_name: str
    experiment_name: str
    iteration: int
    num_rollout: int
    rollout_batch_size: int
    n_samples_per_prompt: int
    num_steps_per_rollout: int
    eval_interval: int
    save_interval: int
    num_gpus: int
    actor_gpus: int
    rollout_gpus: int
    colocate: bool
    tensor_parallel_size: int
    rollout_tensor_parallel_size: int
    rollout_max_context_len: int
    rollout_max_response_len: int
    max_tokens_per_gpu: int
    rollout_temperature: float
    rollout_top_p: float
    eval_temperature: float
    eval_top_p: float
    data_seed: int
    seed: int
    actor_lr: float
    critic_lr: float
    weight_decay: float
    kl_coef: float
    kl_loss_coef: float
    entropy_coef: float
    eps_clip: float
    gamma: float
    lambd: float
    algorithm: str
    harness: str
    model_adapter: str
    sglang_attention_backend: str
    sglang_mm_attention_backend: str
    sglang_mem_fraction_static: float
    megatron_attention_backend: str
    train_script: str
    generate_function_path: str
    rollout_log_function_path: str
    eval_rollout_log_function_path: str
    adaptive: bool
    resume: bool
    record_rollout_images: bool
    rollout_only: bool
    chat_template_kwargs: str
    megatron_model_type: str | None
    # Diagnostic escape hatch for runtimes where CUDA graph capture is known
    # to be unsupported. Normal GraphRL/SLIME runs leave CUDA graphs enabled.
    sglang_disable_cuda_graph: bool = False
    # Extra Megatron args for the actor role, passed straight through to
    # SLIME's per-role override seam. Colocate parks the whole optimizer on
    # GPU and the memory saver must re-back all of it every wake_up, so
    # `optimizer_cpu_offload` is the knob that makes 80GB hosts viable.
    megatron_actor_overrides: dict[str, Any] = field(default_factory=dict)
    # Long-tail control. SLIME only aborts still-flying samples once
    # `rollout_batch_size` of them have *finished*, so leaving
    # over_sampling_batch_size at the batch size gives it no slack and the step
    # waits on the slowest trajectory -- measured: 126/128 done in 10s at
    # 7431 tok/s, then 10+ min at one request and ~1 tok/s. Oversampling
    # creates the slack; partial_rollout recycles the aborted work instead of
    # discarding it.
    over_sampling_batch_size: int | None = None
    partial_rollout: bool = False
    mask_offpolicy_in_partial_rollout: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_launch_spec(
    config: dict[str, Any],
    *,
    model_path: str,
    output_dir: str | Path,
    adaptive: bool = False,
) -> SlimeLaunchSpec:
    """Build one stable, backend-native configuration.

    ``slime:`` is the preferred interface.  The old ``hydra_overrides`` values
    remain accepted so every committed example keeps an equivalent launch path
    while configs are migrated gradually.
    """

    legacy = dict(config.get("hydra_overrides") or {})
    direct = dict(config.get("slime") or {})
    data = dict(legacy.get("data") or {})
    trainer = dict(legacy.get("trainer") or {})
    algorithm_cfg = dict(legacy.get("algorithm") or {})
    actor_ref = dict(legacy.get("actor_rollout_ref") or {})
    actor = dict(actor_ref.get("actor") or {})
    rollout = dict(actor_ref.get("rollout") or {})
    critic = dict(legacy.get("critic") or {})

    train_envs = _first(direct.get("train_envs"), data.get("train_files"))
    eval_envs = _first(direct.get("eval_envs"), data.get("val_files"))
    if not train_envs or not eval_envs:
        raise ValueError(
            "RL requires slime.train_envs/slime.eval_envs (legacy "
            "data.train_files/data.val_files are also accepted)"
        )

    raw_algorithm = str(
        _first(direct.get("algorithm"), algorithm_cfg.get("adv_estimator"), default="gae")
    ).lower()
    algorithm = {
        "gae": "default_gae",
        "default_gae": "default_gae",
        "grpo": "trajectory_grpo",
        "trajectory_grpo": "trajectory_grpo",
        "token_level_gae": "token_level_gae",
    }.get(raw_algorithm, raw_algorithm)

    num_gpus = int(_first(direct.get("num_gpus"), trainer.get("n_gpus_per_node"), default=8))
    colocate = bool(direct.get("colocate", False))
    actor_gpus = int(
        _first(
            direct.get("actor_gpus"),
            default=num_gpus if colocate else max(1, num_gpus // 2),
        )
    )
    rollout_gpus = int(
        _first(
            direct.get("rollout_gpus"),
            default=actor_gpus if colocate else num_gpus - actor_gpus,
        )
    )
    requested_gpus = max(actor_gpus, rollout_gpus) if colocate else actor_gpus + rollout_gpus
    if actor_gpus < 1 or rollout_gpus < 1 or requested_gpus > num_gpus:
        placement = "colocated" if colocate else "disjoint"
        raise ValueError(
            f"SLIME needs positive {placement} actor/rollout GPU counts within num_gpus; "
            f"got actor_gpus={actor_gpus}, rollout_gpus={rollout_gpus}, num_gpus={num_gpus}"
        )

    response = int(_first(direct.get("rollout_max_response_len"), data.get("max_response_length"), default=4096))
    prompt = int(_first(direct.get("rollout_max_prompt_len"), data.get("max_prompt_length"), default=4096))
    output = Path(output_dir).expanduser().resolve()
    tracker = output / "slime_checkpoints" / "latest_checkpointed_iteration.txt"
    resume = bool(_first(direct.get("resume"), config.get("resume"), default=tracker.is_file()))
    train_script = str(
        "graphrl/slime/train_adaptive.py"
        if adaptive
        else _first(direct.get("train_script"), default="graphrl/slime/train.py")
    )

    return SlimeLaunchSpec(
        model_path=str(Path(model_path).expanduser()),
        output_dir=str(output),
        train_envs=str(Path(str(train_envs)).expanduser().resolve()),
        eval_envs=str(Path(str(eval_envs)).expanduser().resolve()),
        project_name=str(config.get("_project_name", "graphrl")),
        experiment_name=str(config.get("_experiment_name", "graphrl_pipeline")),
        iteration=int(config.get("_iter_num", 0)),
        num_rollout=int(
            _first(config.get("training_steps"), direct.get("num_rollout"), default=1)
        ),
        rollout_batch_size=int(_first(direct.get("rollout_batch_size"), data.get("train_batch_size"), default=128)),
        n_samples_per_prompt=int(_first(direct.get("n_samples_per_prompt"), rollout.get("n"), default=1)),
        num_steps_per_rollout=int(direct.get("num_steps_per_rollout", 1)),
        eval_interval=int(_first(direct.get("eval_interval"), trainer.get("test_freq"), default=20)),
        save_interval=int(_first(direct.get("save_interval"), trainer.get("save_freq"), default=20)),
        num_gpus=num_gpus,
        actor_gpus=actor_gpus,
        rollout_gpus=rollout_gpus,
        colocate=colocate,
        tensor_parallel_size=int(direct.get("tensor_parallel_size", 1)),
        rollout_tensor_parallel_size=int(_first(direct.get("rollout_tensor_parallel_size"), rollout.get("tensor_model_parallel_size"), default=1)),
        rollout_max_context_len=int(direct.get("rollout_max_context_len", prompt + response)),
        rollout_max_response_len=response,
        # The old max_num_batched_tokens belongs to the inference server and
        # is not a safe substitute for Megatron's per-GPU training-token cap.
        max_tokens_per_gpu=int(_first(direct.get("max_tokens_per_gpu"), default=16384)),
        rollout_temperature=float(_first(direct.get("rollout_temperature"), rollout.get("temperature"), default=0.7)),
        rollout_top_p=float(_first(direct.get("rollout_top_p"), rollout.get("top_p"), default=0.9)),
        eval_temperature=float(direct.get("eval_temperature", 0.0)),
        eval_top_p=float(direct.get("eval_top_p", 1.0)),
        data_seed=int(_first(direct.get("data_seed"), data.get("seed"), default=0)),
        seed=int(_first(direct.get("seed"), data.get("seed"), default=42)),
        actor_lr=float(_first(direct.get("actor_lr"), _at(actor, "optim", "lr"), default=1e-6)),
        critic_lr=float(_first(direct.get("critic_lr"), _at(critic, "optim", "lr"), default=1e-5)),
        weight_decay=float(direct.get("weight_decay", 0.1)),
        kl_coef=float(_first(direct.get("kl_coef"), _at(algorithm_cfg, "kl_ctrl", "kl_coef"), default=0.0)),
        kl_loss_coef=float(_first(direct.get("kl_loss_coef"), actor.get("kl_loss_coef"), default=0.0)),
        entropy_coef=float(_first(direct.get("entropy_coef"), actor.get("entropy_coeff"), default=0.0)),
        eps_clip=float(direct.get("eps_clip", 0.2)),
        gamma=float(_first(direct.get("gamma"), algorithm_cfg.get("gamma"), default=1.0)),
        lambd=float(_first(direct.get("lambd"), algorithm_cfg.get("lam"), default=1.0)),
        algorithm=algorithm,
        harness=str(direct.get("harness", "no_concat")),
        model_adapter=str(direct.get("model_adapter", "qwen")),
        sglang_attention_backend=str(direct.get("sglang_attention_backend", "triton")),
        sglang_mm_attention_backend=str(direct.get("sglang_mm_attention_backend", "sdpa")),
        sglang_mem_fraction_static=float(_first(direct.get("sglang_mem_fraction_static"), rollout.get("gpu_memory_utilization"), default=0.6)),
        megatron_attention_backend=str(direct.get("megatron_attention_backend", "flash")),
        train_script=train_script,
        generate_function_path=str(direct.get("generate_function_path", "graphrl.slime.rollout.generate")),
        rollout_log_function_path=str(direct.get("rollout_log_function_path", "graphrl.slime.metrics.log_rollout_data")),
        eval_rollout_log_function_path=str(direct.get("eval_rollout_log_function_path", "graphrl.slime.metrics.log_eval_rollout_data")),
        adaptive=bool(adaptive),
        resume=resume,
        record_rollout_images=bool(direct.get("record_rollout_images", True)),
        rollout_only=bool(direct.get("rollout_only", False)),
        chat_template_kwargs=str(direct.get("chat_template_kwargs", "")),
        megatron_model_type=(str(direct["megatron_model_type"]) if direct.get("megatron_model_type") else None),
        sglang_disable_cuda_graph=bool(direct.get("sglang_disable_cuda_graph", False)),
        megatron_actor_overrides=dict(direct.get("megatron_actor_overrides") or {}),
        over_sampling_batch_size=(
            int(direct["over_sampling_batch_size"])
            if direct.get("over_sampling_batch_size") is not None
            else None
        ),
        partial_rollout=bool(direct.get("partial_rollout", False)),
        mask_offpolicy_in_partial_rollout=bool(
            direct.get("mask_offpolicy_in_partial_rollout", False)
        ),
    )


__all__ = ["SlimeLaunchSpec", "build_launch_spec"]
