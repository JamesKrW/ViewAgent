"""Generic synchronous VAGEN-SLIME launcher for every GraphRL example."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

from vagen_agent.algorithms import resolve_algorithm
from vagen_agent.make_dataset import build_rows, write_jsonl
from view_suite.envs.slime_adapter import resolve_legacy_data_paths

from slime.utils.external_utils.command_utils import execute_train

from graphrl.slime.config import SlimeLaunchSpec


REPO_ROOT = Path(__file__).resolve().parents[3]
VAGEN_SLIME_ROOT = REPO_ROOT / "GraphRL" / "VAGEN-SLIME"


def _wandb_run_id(spec: SlimeLaunchSpec) -> str:
    """Stable per-RL-phase id so a checkpoint resume extends the same run."""

    identity = "\0".join(
        (
            spec.project_name,
            spec.experiment_name,
            str(spec.iteration),
            str(Path(spec.output_dir).resolve()),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _megatron_source_root() -> Path:
    """Locate the Megatron source produced by VAGEN-SLIME's build script."""

    explicit = os.environ.get("VAGEN_SLIME_MEGATRON_ROOT")
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        VAGEN_SLIME_ROOT / "build" / "Megatron-LM",
        REPO_ROOT.parent / "VAGEN-SLIME" / "build" / "Megatron-LM",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "megatron" / "training").is_dir():
            return candidate.resolve()
    expected = VAGEN_SLIME_ROOT / "build" / "Megatron-LM"
    raise RuntimeError(
        "Megatron-LM is not built for the vendored VAGEN-SLIME checkout. Run "
        f"`bash {VAGEN_SLIME_ROOT / 'scripts' / 'build_slime_env.sh'}` or set "
        f"VAGEN_SLIME_MEGATRON_ROOT; expected {expected}"
    )


def _model_type(model_path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = Path(model_path).name.lower().replace("-instruct", "").replace("-thinking", "")
    match = re.search(r"qwen(2\.5|3|3\.5)-vl-(\d+(?:\.\d+)?b(?:-a\d+b)?)", name)
    if match:
        candidate = f"qwen{match.group(1)}-{match.group(2)}"
        return _canonical_model_type(candidate)
    match = re.search(r"qwen(2\.5|3|3\.5)-(\d+(?:\.\d+)?b(?:-a\d+b)?)", name)
    if match:
        candidate = f"qwen{match.group(1)}-{match.group(2)}"
        return _canonical_model_type(candidate)
    raise ValueError(
        f"cannot infer a SLIME MODEL_ARGS file from {model_path!r}; set "
        "general_overrides.rl.slime.megatron_model_type"
    )


def _canonical_model_type(candidate: str) -> str:
    scripts = VAGEN_SLIME_ROOT / "slime" / "scripts" / "models"
    for path in scripts.glob("*.sh"):
        if path.stem.lower() == candidate.lower():
            return path.stem
    raise ValueError(f"VAGEN-SLIME has no MODEL_ARGS script matching {candidate!r}")


def _rope_theta(model_path: str) -> str | None:
    try:
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = (config.get("text_config") or config).get("rope_theta")
    return str(int(value)) if value else None


def _resolve_model_path(value: str) -> str:
    """Return a local HF directory, downloading a repo id only when necessary."""

    path = Path(value).expanduser()
    if path.is_dir():
        return str(path.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"{value!r} is not a local model directory and huggingface_hub is unavailable"
        ) from exc
    return str(Path(snapshot_download(repo_id=value)).resolve())


def _runtime_library_path() -> str:
    prefix = Path(os.environ.get("CONDA_PREFIX") or Path(sys.executable).resolve().parents[1])
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = (
        prefix / "lib" / version / "site-packages" / "nvidia" / "cudnn" / "lib",
        prefix / "lib",
    )
    values: list[str] = []
    for item in (*map(str, candidates), *os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)):
        if item and item not in values:
            values.append(item)
    return os.pathsep.join(values)


def _write_runtime_configs(spec: SlimeLaunchSpec) -> tuple[Path, Path, Path, Path, Path]:
    root = Path(spec.output_dir)
    config_dir = root / "slime_config"
    data_dir = root / "slime_data"
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_data = data_dir / "train.jsonl"
    eval_data = data_dir / "eval.jsonl"
    def rows(path: str) -> list[dict[str, Any]]:
        output = build_rows(path, base_seed=spec.data_seed)
        for row in output:
            metadata = row.get("metadata") or {}
            config = resolve_legacy_data_paths(dict(metadata.get("config") or {}))
            if raw_path := config.get("jsonl_path"):
                if not Path(str(raw_path)).is_file():
                    raise FileNotFoundError(
                        f"environment dataset referenced by {path} is missing: {raw_path}"
                    )
            metadata["config"] = config
            row["metadata"] = metadata
        return output

    write_jsonl(rows(spec.train_envs), str(train_data))
    write_jsonl(rows(spec.eval_envs), str(eval_data))

    harness = config_dir / "harness.yaml"
    harness.write_text(
        yaml.safe_dump(
            {
                "harness": spec.harness,
                "algorithm": spec.algorithm,
                "model_adapter": spec.model_adapter,
                "transcript_dir": str(root / "slime_transcripts"),
                "transcript_images": spec.record_rollout_images,
                "vagen_extra": {
                    "legacy_rollout_dir": str(root / "rollout_data"),
                    "record_rollout_images": spec.record_rollout_images,
                    "adaptive": spec.adaptive,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    eval_config = config_dir / "eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "defaults": {
                        "temperature": spec.eval_temperature,
                        "top_p": spec.eval_top_p,
                        "n_samples_per_eval_prompt": 1,
                        "metadata_key": "metadata",
                        "input_key": "prompt",
                        "label_key": "label",
                    },
                    "datasets": [{"name": "ae", "path": str(eval_data)}],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    checkpoint_dir = root / "slime_checkpoints"
    roles = config_dir / "megatron_roles.yaml"
    actor_overrides = {"attention_backend": spec.megatron_attention_backend}
    critic_overrides = {
        "attention_backend": spec.megatron_attention_backend,
        "lr": spec.critic_lr,
        "save": str(checkpoint_dir / "critic"),
    }
    if spec.resume:
        critic_overrides["load"] = str(checkpoint_dir / "critic")
    roles.write_text(
        yaml.safe_dump(
            {
                "megatron": [
                    {"name": "default", "role": "actor", "overrides": actor_overrides},
                    {"name": "default", "role": "critic", "overrides": critic_overrides},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return train_data, eval_data, harness, eval_config, roles


def build_train_args(spec: SlimeLaunchSpec) -> list[str]:
    train_data, _eval_data, harness, eval_config, roles = _write_runtime_configs(spec)
    root = Path(spec.output_dir)
    save_dir = root / "slime_checkpoints"
    hf_pattern = root / "hf_checkpoints" / "rollout_{rollout_id}"
    algorithm = resolve_algorithm(spec.algorithm)
    args = [
        "--hf-checkpoint", spec.model_path,
        "--load", str(save_dir if spec.resume else spec.model_path),
        "--save", str(save_dir),
        "--save-hf", str(hf_pattern),
        "--save-interval", str(spec.save_interval),
        "--prompt-data", str(train_data),
        "--input-key", "prompt",
        "--label-key", "label",
        "--metadata-key", "metadata",
        "--rollout-shuffle",
        "--custom-generate-function-path", spec.generate_function_path,
        "--custom-config-path", str(harness),
        "--custom-reward-post-process-path", "vagen_agent.algorithms.post_process",
        "--custom-rollout-log-function-path", spec.rollout_log_function_path,
        "--custom-eval-rollout-log-function-path", spec.eval_rollout_log_function_path,
        "--num-rollout", str(spec.num_rollout),
        "--rollout-batch-size", str(spec.rollout_batch_size),
        "--n-samples-per-prompt", str(spec.n_samples_per_prompt),
        "--num-steps-per-rollout", str(spec.num_steps_per_rollout),
        "--rollout-max-context-len", str(spec.rollout_max_context_len),
        "--rollout-max-response-len", str(spec.rollout_max_response_len),
        "--rollout-temperature", str(spec.rollout_temperature),
        "--rollout-top-p", str(spec.rollout_top_p),
        "--balance-data",
        "--eval-config", str(eval_config),
        "--eval-interval", str(spec.eval_interval),
        "--eval-temperature", str(spec.eval_temperature),
        "--eval-top-p", str(spec.eval_top_p),
        "--advantage-estimator", algorithm.slime_estimator,
        "--gamma", str(spec.gamma),
        "--lambd", str(spec.lambd),
        "--normalize-advantages",
        "--kl-coef", str(spec.kl_coef),
        "--kl-loss-coef", str(spec.kl_loss_coef),
        "--entropy-coef", str(spec.entropy_coef),
        "--eps-clip", str(spec.eps_clip),
        "--optimizer", "adam",
        "--lr", str(spec.actor_lr),
        "--lr-decay-style", "constant",
        "--weight-decay", str(spec.weight_decay),
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
        "--seed", str(spec.seed),
        "--train-backend", "megatron",
        "--tensor-model-parallel-size", str(spec.tensor_parallel_size),
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(spec.max_tokens_per_gpu),
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", spec.megatron_attention_backend,
        "--megatron-to-hf-mode", "bridge",
        "--rollout-num-gpus-per-engine", str(spec.rollout_tensor_parallel_size),
        "--sglang-mem-fraction-static", str(spec.sglang_mem_fraction_static),
        "--sglang-attention-backend", spec.sglang_attention_backend,
        "--sglang-prefill-attention-backend", spec.sglang_attention_backend,
        "--sglang-decode-attention-backend", spec.sglang_attention_backend,
        "--sglang-mm-attention-backend", spec.sglang_mm_attention_backend,
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(spec.actor_gpus),
        "--rollout-num-gpus", str(spec.rollout_gpus),
    ]
    if algorithm.needs_critic:
        args.extend(("--megatron-config-path", str(roles)))
    if algorithm.custom_advantage_path:
        args.extend(("--custom-advantage-function-path", algorithm.custom_advantage_path))
    if spec.chat_template_kwargs:
        args.extend(("--apply-chat-template-kwargs", spec.chat_template_kwargs))
    if spec.rollout_only:
        args.extend(("--debug-rollout-only", "--dump-details", str(root / "debug")))
    if os.environ.get("WANDB_MODE", "online") != "disabled":
        run_name = f"{spec.experiment_name}_rl_iter{spec.iteration:03d}"
        run_id = os.environ.get("WANDB_RUN_ID") or _wandb_run_id(spec)
        args.extend(
            (
                "--use-wandb",
                "--wandb-mode", os.environ.get("WANDB_MODE", "online"),
                "--wandb-project", os.environ.get("WANDB_PROJECT", spec.project_name),
                "--wandb-group", run_name,
                "--wandb-run-id", run_id,
                "--disable-wandb-random-suffix",
            )
        )
        if os.environ.get("WANDB_BASE_URL"):
            args.extend(("--wandb-host", os.environ["WANDB_BASE_URL"]))
        if os.environ.get("WANDB_ENTITY"):
            args.extend(("--wandb-team", os.environ["WANDB_ENTITY"]))
        if os.environ.get("WANDB_DIR"):
            args.extend(("--wandb-dir", os.environ["WANDB_DIR"]))
    return args


def launch(spec: SlimeLaunchSpec, *, dry_run: bool = False) -> None:
    model_type = _model_type(spec.model_path, spec.megatron_model_type)
    train_script = Path(spec.train_script)
    if not train_script.is_absolute():
        train_script = (REPO_ROOT / "GraphRL" / train_script).resolve()
    if dry_run:
        args = build_train_args(spec)
        print(json.dumps({"model_type": model_type, "train_script": str(train_script), "args": args}, indent=2))
        return
    spec = dataclasses.replace(spec, model_path=_resolve_model_path(spec.model_path))
    args = build_train_args(spec)
    if theta := _rope_theta(spec.model_path):
        os.environ["MODEL_ARGS_ROTARY_BASE"] = theta
    interpreter_bin = str(Path(sys.executable).resolve().parent)
    if os.environ.get("PATH", "").split(os.pathsep)[0] != interpreter_bin:
        os.environ["PATH"] = os.pathsep.join((interpreter_bin, os.environ.get("PATH", "")))
    megatron_root = _megatron_source_root()
    wandb_env: dict[str, str] = {}
    if os.environ.get("WANDB_MODE", "online") != "disabled":
        wandb_env = {
            "WANDB_RUN_ID": os.environ.get("WANDB_RUN_ID") or _wandb_run_id(spec),
            "WANDB_RESUME": os.environ.get("WANDB_RESUME", "allow"),
        }
    execute_train(
        train_args=shlex.join(args),
        num_gpus_per_node=spec.num_gpus,
        megatron_model_type=None if spec.rollout_only else model_type,
        train_script=str(train_script),
        extra_env_vars={
            "PYTHONPATH": os.pathsep.join(
                value for value in (
                    str(REPO_ROOT),
                    str(REPO_ROOT / "GraphRL"),
                    str(VAGEN_SLIME_ROOT),
                    str(VAGEN_SLIME_ROOT / "slime"),
                    str(megatron_root),
                    os.environ.get("PYTHONPATH", ""),
                ) if value
            ),
            "LD_LIBRARY_PATH": _runtime_library_path(),
            **wandb_env,
            **{
                name: os.environ[name]
                for name in (
                    "VIEWSUITE_ROOT", "RENDER_TLS_NO_VERIFY", "SSL_CERT_FILE",
                    "WANDB_BASE_URL", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_MODE", "WANDB_DIR",
                    "GRAPHRL_ADAPTIVE_EXPERIMENT_DIR", "GRAPHRL_ADAPTIVE_ROUND",
                )
                if os.environ.get(name)
            },
            "NVTE_FUSED_ATTN": "0",
            "NVTE_FLASH_ATTN": "1",
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="resolved SlimeLaunchSpec JSON")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    raw: dict[str, Any] = json.loads(Path(args.config).read_text(encoding="utf-8"))
    launch(SlimeLaunchSpec(**raw), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
