"""Sokoban -- no_concat -- PPO/GAE with a critic.

Built on slime's own launcher (``slime.utils.external_utils.command_utils``) rather than
hand-rolled bash, so process cleanup, ``ray start``, the runtime-env JSON, NVLink
detection, MODEL_ARGS sourcing, checkpoint conversion and the wandb naming all come from
slime and stay in step with it.

Everything that decides *what algorithm this is* lives in two places: ``no_concat_ppo.yaml``
(the four axes) and ``ppo_args`` below. ``ALIGNMENT.md`` maps each of those to the VAGEN
setting it reproduces, and ``README.md`` is the walkthrough.

    python examples/train/sokoban/run_sokoban.py                      # defaults
    python examples/train/sokoban/run_sokoban.py --num-rollout 60 --actor-gpus 1
    SLIME_SCRIPT_MODEL_NAME=Qwen3.5-4B python examples/train/sokoban/run_sokoban.py

Every field of ``Config`` is both a ``--flag`` and a ``SLIME_SCRIPT_<NAME>`` env var --
that is what ``dataclass_cli`` gives us, and it is slime's convention for its own scripts.
"""

# No `from __future__ import annotations` here: slime's dataclass_cli reads the first
# parameter's annotation with inspect.signature and asserts it is a dataclass, and the
# future import turns every annotation into a string, so the assert fires with no message.
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # VAGEN-SLIME
sys.path.insert(0, str(REPO))

from slime.utils.external_utils.command_utils import (  # noqa: E402
    execute_train,
    get_default_wandb_args,
)
from slime.utils.external_utils.typer_utils import dataclass_cli  # noqa: E402


@dataclass
class Config:
    # -- what to run -----------------------------------------------------------
    model_name: str = "Qwen3-VL-4B-Instruct"
    harness_config: str = str(HERE / "no_concat_ppo.yaml")
    envs_yaml: str = str(HERE / "train.yaml")
    eval_yaml: str = str(HERE / "val.yaml")
    eval_config: str = str(HERE / "eval_datasets.yaml")
    megatron_roles_config: str = str(HERE / "megatron_roles.yaml")
    generate_function_path: str = "vagen_agent.rollout.generate"

    # -- where things live. Generated artifacts stay under the repository root and
    #    are excluded from version control by .gitignore.
    models_dir: str = str(Path.home() / "models")
    # Optional exact HF checkpoint directory. Empty preserves the original
    # ``<models_dir>/<model_name>`` layout; an explicit path is useful for local
    # Hugging Face cache snapshots and mounted model stores.
    model_path: str = ""
    data_dir: str = str(REPO / "data")
    runs_dir: str = str(REPO / "runs")
    # Keep generated artifacts and default run names honest when this launcher is
    # reused by another environment. The Sokoban defaults are unchanged.
    dataset_prefix: str = "sokoban"
    experiment_prefix: str = "sokoban"

    # -- scale -----------------------------------------------------------------
    num_gpus: int = 2                       # total visible to ray
    actor_gpus: int = 1                     # rest go to sglang; train_async needs them disjoint
    tp: int = 1
    rollout_tp: int = 1
    # sglang's own attention backend, which is not slime's `--attention-backend` (that one
    # is Megatron's). Left to itself sglang picks FlashAttention v4 on Blackwell, and v4
    # needs `flash_attn.cute` + `cutlass`, which this environment does not have -- the
    # engine then dies with SIGQUIT during warm-up and the router 500s every request.
    #
    # All three of attention/prefill/decode have to be set: the prefill and decode backends
    # default to None and are auto-selected independently, so setting only the first leaves
    # them both on v4.
    sglang_attention_backend: str = "triton"
    max_tokens_per_gpu: int = 16384

    # -- schedule. Defaults are VAGEN's shipped Sokoban settings; see ALIGNMENT.md.
    # 0 means eval only: slime's train.py has a special case
    # (`if args.num_rollout == 0 and args.eval_interval is not None`) that evaluates once
    # and exits. That is the check to run when a new model or environment arrives -- it
    # exercises the whole rollout path (client, model adapter, harness, env, reward,
    # metrics) against the
    # held-out set without spending anything on training, and a bad rollout shows up as a
    # success rate rather than as a loss curve that looks plausible.
    num_rollout: int = 401
    rollout_batch_size: int = 128           # VAGEN data.train_batch_size
    n_samples_per_prompt: int = 1           # VAGEN rollout.n
    # Consume the complete 128-episode rollout in one optimizer update, matching the
    # shipped Geo3K recipe's one-update-per-rollout schedule. Row-splitting harnesses may
    # produce more than 128 Sample rows, but they still carry 128 unique rollout_ids and
    # therefore remain one training step.
    num_steps_per_rollout: int = 1
    eval_interval: int = 20                 # VAGEN trainer.test_freq
    save_interval: int = 200                # VAGEN trainer.save_freq
    #: Continue an interrupted run of the same `--exp-name` instead of starting over.
    #:
    #: Points `--load` at the save directory rather than the HF checkpoint. slime tells the
    #: two apart by looking for `latest_checkpointed_iteration.txt`
    #: (`megatron_utils/checkpoint.py:123`), reads the iteration -- which is the rollout id
    #: it was saved under -- and continues from the next one, restoring the optimizer, the
    #: LR schedule and the dataset position along with the weights.
    #:
    #: Only whole `--save-interval` boundaries exist to resume from, so this rewinds to the
    #: last one; wandb gets a fresh run, because the run id is minted per launch.
    resume: bool = False

    # -- rollout-only verification ---------------------------------------------
    # slime's own `--debug-rollout-only`: the placement group gets zero actor GPUs, the
    # Megatron argument validation is skipped, and `TrainRayActor` short-circuits. Only the
    # sglang engines come up, loading straight from `--hf-checkpoint` -- so a model needs
    # no torch_dist conversion to be checked, and a 4-GPU box can verify a rollout in
    # minutes. Implies --num-rollout 0 (eval once and exit) and turns on --dump-details,
    # which writes every Sample to <runs>/<exp>/dump/rollout_data/eval_0.pt for
    # tests/verify_rollout_dump.py to read.
    rollout_only: bool = False
    #: JSON, passed to slime's --apply-chat-template-kwargs and from there to the model
    #: adapter. `{"enable_thinking": false}` is the one that matters on Qwen3.5.
    chat_template_kwargs: str = ""
    #: Which kernel the *vision tower* uses. Empty leaves sglang's own choice, which is
    #: `fa4` on this hardware -- and `fa4` imports `flash_attn.cute`, which needs the
    #: `cutlass` module from `nvidia-cutlass-dsl-libs-cu12`. That wheel is absent here and
    #: PyPI no longer carries the 4.5.1 that matches the installed base package, so the
    #: engine's scheduler dies on the first frame with `No module named 'cutlass'`.
    #: `sdpa` is pure torch and always available; `triton_attn` and `fa3` are faster if
    #: they work in your environment. Text-only runs never build a vision tower, so this
    #: is inert for them.
    sglang_mm_attention_backend: str = "sdpa"
    #: The whole response-region budget used by VAGEN's Sokoban recipes. slime exposes
    #: this as its default generation cap rather than a separate padded response region;
    #: every action call is tightened to ``turn_tokens`` by the harness below. Keeping the
    #: two values separate is load-bearing for concat/compact, whose one training row can
    #: contain several 512-token generations.
    rollout_response_tokens: int = 4000
    #: Hard inference context. Kept wider than VAGEN's 1000+4000 padded training regions
    #: so image-token variation and a pending compact summary still have headroom.
    rollout_context_tokens: int = 8192
    #: Per-turn generation budget, overriding the env yaml. This is VAGEN's
    #: ``response_length_per_turn`` and is deliberately independent of the 4000-token
    #: response-region budget above.
    turn_tokens: int = 512

    exp_name: str = ""                      # defaults to sokoban_<harness>_<model>
    # `train_async.py`: generate(t+1) overlaps train(t), and it syncs the in-flight
    # generation before every weight update, so a multi-turn episode never spans two
    # policies. The fully-async path does not -- it aborts and re-queues the whole
    # trajectory, which for a 5-turn episode discards every step it had taken.
    train_script: str = "train_async.py"


def megatron_model_type_of(model_name: str, model_dir: str) -> str:
    """Which of slime's MODEL_ARGS specs describes this checkpoint's *language tower*.

    A VLM trains its language tower under a spec written for the text model of the same
    size -- slime's own geo3k VLM example does exactly this mapping -- because the vision
    tower is built by `AutoBridge.from_hf_pretrained` off the HF config, not by MODEL_ARGS.
    Qwen3-VL-4B-Instruct's language tower is layer-for-layer the qwen3-4B spec.
    """
    if "-VL-" not in model_name:
        # Text checkpoints have a spec of their own, named after them.
        return model_name.replace("Qwen", "qwen", 1)
    name = model_name.replace("-Instruct", "").replace("-Thinking", "")
    name = (
        name.replace("Qwen2.5-VL-", "qwen2.5-")
        .replace("Qwen3-VL-", "qwen3-")
        .replace("Qwen3.5-VL-", "qwen3.5-")
    )
    return name.replace("Qwen", "qwen", 1)


def eval_dataset_name(eval_yaml: str) -> str:
    """What to call the held-out set in wandb, read off the yaml that defines it.

    Not a literal in eval_datasets.yaml: that name becomes the panel prefix
    (`eval/<name>-success`), and a literal goes stale the moment --eval-yaml points
    somewhere else. It read `sokoban_text` against a vision dataset for a while, which is
    exactly as misleading as it sounds.
    """
    import yaml as _yaml

    try:
        envs = (_yaml.safe_load(Path(eval_yaml).read_text()) or {}).get("envs") or []
        first = envs[0]
        mode = (first.get("config") or {}).get("render_mode")
        name = str(first.get("name", "env")).lower()
        return f"{name}_{mode}" if mode else name
    except (OSError, ValueError, KeyError, IndexError, AttributeError):
        return Path(eval_yaml).stem


def _harness_name(harness_config: str) -> str:
    """Read the configured harness for stable, collision-free default run names."""

    import yaml as _yaml

    config = _yaml.safe_load(Path(harness_config).read_text()) or {}
    if not isinstance(config, dict):
        raise TypeError(f"harness config {harness_config} must contain a mapping")
    name = str(config.get("harness", "")).strip()
    if not name:
        raise ValueError(f"harness config {harness_config} does not declare `harness`")
    return name


def _default_experiment_name(
    model_name: str,
    harness_config: str,
    prefix: str = "sokoban",
) -> str:
    return f"{prefix}_{_harness_name(harness_config)}_{model_name}"


def role_config(cfg, save_dir: str) -> str:
    """``megatron_roles.yaml`` with the critic's checkpoint paths filled in.

    Both roles run ``megatron.save_checkpoint(rollout_id, ...)`` against their own
    ``args.save``, and slime hands them the same one -- so they write the same
    ``iter_<rollout_id>`` directory and the second one silently overwrites the first.
    Megatron says so ("Detected an existing checkpoint in ...") and carries on. What
    survives is the critic, whose ``output_layer`` is the [1, hidden] value head, so the
    run finishes with no trained policy on disk and a resume would load the critic's
    weights into the actor.

    slime's own PPO example never passes ``--save``, which is why this has not come up
    there. The role config is the fix: ``_apply_megatron_role_overrides`` setattrs whatever
    keys it is given, so the critic can be pointed at a subdirectory. Written out resolved
    rather than committed, because the source yaml cannot interpolate a run directory.

    ``load`` is overridden only when resuming. At a cold start the critic must read the
    pretrained checkpoint the actor reads -- slime then notices the ``output_layer`` shape
    does not match and reinitialises just the value head
    (``megatron_utils/model.py::_critic_output_layer_needs_reinit``).
    """
    import yaml as _yaml

    config = _yaml.safe_load(Path(cfg.megatron_roles_config).read_text())
    critic = next(e for e in config["megatron"] if e.get("role") == "critic")
    critic.setdefault("overrides", {})["save"] = f"{save_dir}/critic"
    if cfg.resume:
        critic["overrides"]["load"] = f"{save_dir}/critic"
    out = Path(save_dir) / "megatron_roles.resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_yaml.safe_dump(config, sort_keys=False))
    return str(out)


def rotary_base_of(model_dir: str) -> str | None:
    """`rope_theta` from the checkpoint, for the spec's ${MODEL_ARGS_ROTARY_BASE}.

    Read rather than hardcoded: the shared specs default to 1000000 and Qwen3-VL-4B wants
    5000000. Getting this wrong does not fail -- it trains a model whose positions mean
    something else than they did at rollout, and nothing reports it.
    """
    import json

    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    theta = (cfg.get("text_config") or cfg).get("rope_theta")
    return str(int(theta)) if theta else None


def dataset(cfg: Config, envs_yaml: str, out: str) -> str:
    """Build the jsonl if the yaml is newer. Seed expansion is deterministic in
    --base-seed, so this is cheap and keeps the data in step with the config."""
    out_path = Path(out)
    if not out_path.exists() or Path(envs_yaml).stat().st_mtime > out_path.stat().st_mtime:
        print(f"building {out} from {envs_yaml}")
        subprocess.run(
            [sys.executable, "-m", "vagen_agent.make_dataset",
             "--envs", envs_yaml, "--out", out, "--base-seed", "0",
             "--response-length-per-turn", str(cfg.turn_tokens)],
            cwd=REPO, check=True,
        )
    return out


def runtime_ld_library_path() -> str:
    """Library search path for local and Ray-worker processes.

    Newer PyTorch wheels keep cuDNN under ``site-packages/nvidia/cudnn/lib``
    instead of the conda prefix's top-level ``lib``.  Ray constructs a fresh
    worker environment, so propagating only ``$CONDA_PREFIX/lib`` makes
    Transformer Engine fail on ``libcudnn_graph.so.9`` even though it imports
    correctly in the activated launcher shell.
    """
    env_prefix = Path(os.environ.get("CONDA_PREFIX") or Path(sys.executable).resolve().parents[1])
    python_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = (
        env_prefix / "lib" / python_lib / "site-packages" / "nvidia" / "cudnn" / "lib",
        env_prefix / "lib",
    )
    inherited = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    paths: list[str] = []
    for path in (*map(str, candidates), *inherited):
        if path and path not in paths:
            paths.append(path)
    return os.pathsep.join(paths)


def main(cfg: Config) -> None:
    # ``execute_train`` launches the Ray CLI by name.  Keep it tied to the
    # interpreter running this launcher even when callers use an absolute
    # ``.../env/bin/python`` without first activating that conda environment.
    # Otherwise the model imports from one env while ``ray start`` either comes
    # from another env or is not found at all.
    interpreter_bin = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    if current_path.split(os.pathsep)[0] != interpreter_bin:
        os.environ["PATH"] = os.pathsep.join(
            part for part in (interpreter_bin, current_path) if part
        )

    # Absolute, before anything reads them. The defaults already are, but a relative
    # `--harness-config examples/...` is resolved by whoever opens it -- and that is a Ray
    # worker, whose working directory is wherever `ray start` happened, not this launcher's.
    # Under `SLIME_SCRIPT_EXTERNAL_RAY` those two are routinely different directories.
    for field in (
        "harness_config",
        "envs_yaml",
        "eval_yaml",
        "eval_config",
        "megatron_roles_config",
    ):
        setattr(cfg, field, str(Path(getattr(cfg, field)).resolve()))

    if not 0 < cfg.turn_tokens <= cfg.rollout_response_tokens <= cfg.rollout_context_tokens:
        raise SystemExit(
            "token budgets must satisfy 0 < turn_tokens <= rollout_response_tokens "
            "<= rollout_context_tokens; got "
            f"{cfg.turn_tokens}, {cfg.rollout_response_tokens}, "
            f"{cfg.rollout_context_tokens}"
        )

    if cfg.rollout_only:
        cfg.num_rollout = 0

    # Both drivers now support pre-training eval, but eval-only still exists in train.py
    # only. With `--num-rollout 0`, train_async.py has no loop iteration and exits without
    # evaluating. Pick the driver with the explicit eval-only path.
    if cfg.num_rollout == 0 and cfg.train_script != "train.py":
        print(f"eval-only requested: using train.py instead of {cfg.train_script} "
              f"(train_async.py has no eval-only path)")
        cfg.train_script = "train.py"

    model_dir = cfg.model_path or f"{cfg.models_dir}/{cfg.model_name}"
    # slime names its MODEL_ARGS files after the model, lowercased vendor prefix. None
    # under --rollout-only: `execute_train` then neither sources the file nor passes
    # ${MODEL_ARGS[@]}, so a family slime has no Megatron spec for -- every VLM, today --
    # can still have its rollout verified.
    megatron_model_type = (None if cfg.rollout_only
                           else megatron_model_type_of(cfg.model_name, model_dir))
    if (theta := rotary_base_of(model_dir)):
        # Consumed by ${MODEL_ARGS_ROTARY_BASE} inside slime's MODEL_ARGS script.
        os.environ["MODEL_ARGS_ROTARY_BASE"] = theta
    exp_name = cfg.exp_name or _default_experiment_name(
        cfg.model_name, cfg.harness_config, cfg.experiment_prefix
    )

    # No torch_dist conversion. Under `--megatron-to-hf-mode bridge` the model is built
    # by `AutoBridge.from_hf_pretrained`, which reads the HF checkpoint directly -- so
    # `--load` is the HF directory and there is nothing to convert. slime's own geo3k VLM
    # example does exactly this, and for a VLM it is not merely a shortcut: the converter
    # is driven by MODEL_ARGS, which describes only the language tower, so it has nothing
    # to say about the vision one.

    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    # Named after the yaml that generated it, so swapping --eval-yaml cannot silently
    # overwrite the set another run is reading.
    train_data = dataset(
        cfg,
        cfg.envs_yaml,
        f"{cfg.data_dir}/{cfg.dataset_prefix}_{Path(cfg.envs_yaml).stem}_t{cfg.turn_tokens}.jsonl",
    )
    eval_data = dataset(
        cfg,
        cfg.eval_yaml,
        f"{cfg.data_dir}/{cfg.dataset_prefix}_{Path(cfg.eval_yaml).stem}_t{cfg.turn_tokens}.jsonl",
    )

    save_dir = f"{cfg.runs_dir}/{exp_name}"
    if cfg.resume:
        # Both, because the critic writes its own tracker under critic/ (see role_config)
        # and resuming one role from a checkpoint the other never wrote is worse than not
        # resuming: it looks like a continuation right up until the loss curve.
        for role, path in (("actor", save_dir), ("critic", f"{save_dir}/critic")):
            if not Path(f"{path}/latest_checkpointed_iteration.txt").is_file():
                raise SystemExit(
                    f"--resume: no {role} checkpoint under {path}. The first one is "
                    f"written after rollout {cfg.save_interval - 1} "
                    f"(--save-interval {cfg.save_interval}).")

    ckpt_args = (
        f"--hf-checkpoint {model_dir} "
        # The bridge loads weights from here. No --ref-load: a reference model is only
        # built when `kl_coef != 0 or use_kl_loss`, and both are 0 in ppo_args below.
        + ("" if cfg.rollout_only else f"--load {save_dir if cfg.resume else model_dir} ")
        + f"--save {save_dir} "
        f"--save-interval {cfg.save_interval} "
    )

    verify_args = (
        f"--debug-rollout-only --dump-details {save_dir}/dump "
        if cfg.rollout_only else ""
    ) + (
        f"--apply-chat-template-kwargs '{cfg.chat_template_kwargs}' "
        if cfg.chat_template_kwargs else ""
    )

    rollout_args = (
        f"--prompt-data {train_data} "
        "--input-key prompt --label-key label --metadata-key metadata "
        "--rollout-shuffle "
        # The whole of the verl replacement.
        f"--custom-generate-function-path {cfg.generate_function_path} "
        f"--custom-config-path {cfg.harness_config} "
        # Group-relative normalisation over episodes, not rows. Inert under ppo; left on
        # so switching `algorithm:` in the yaml needs no edit here.
        "--custom-reward-post-process-path vagen_agent.algorithms.post_process "
        # Environment-level metrics. Without these the only success signal is
        # `1 - truncated`, exact for Sokoban by coincidence and wrong for any environment
        # that can fail early.
        "--custom-rollout-log-function-path vagen_agent.metrics.log_rollout_data "
        "--custom-eval-rollout-log-function-path vagen_agent.metrics.log_eval_rollout_data "
        # No --rm-type: the environment scores its own actions and the rollout sets
        # sample.reward, so slime's `if sample.reward is None: async_rm(...)` never fires.
        f"--num-rollout {cfg.num_rollout} "
        f"--rollout-batch-size {cfg.rollout_batch_size} "
        f"--n-samples-per-prompt {cfg.n_samples_per_prompt} "
        f"--num-steps-per-rollout {cfg.num_steps_per_rollout} "
        f"--rollout-max-context-len {cfg.rollout_context_tokens} "
        f"--rollout-max-response-len {cfg.rollout_response_tokens} "
        "--rollout-temperature 1.0 "
        "--balance-data "
        # Held-out eval, declared in slime's --eval-config yaml so a second environment is
        # a yaml entry rather than a launcher change. Under train.py this also runs before
        # step 0 (--skip-eval-before-train deliberately not passed): the untrained baseline
        # is what catches a "learning curve" that was really the eval set being easy.
        f"--eval-config {cfg.eval_config} "
        f"--eval-interval {cfg.eval_interval} "
    )

    ppo_args = (
        # "ppo" is load-bearing twice: it selects slime's GAE, and it is the literal string
        # slime tests to decide whether to build a critic (use_critic = estimator == "ppo").
        "--advantage-estimator ppo "
        # Do not change. The port reproduces VAGEN's episode-global default_gae by giving
        # every row of an episode the episode total and letting slime's per-row GAE run;
        # that identity is a telescoping property of gamma = lam = 1 and holds nowhere else.
        # RunConfig._check_against stops the run rather than training something else.
        "--gamma 1.0 --lambd 1.0 "
        # Every VAGEN estimator ends with masked_whiten; slime's default is off.
        "--normalize-advantages "
        "--kl-coef 0.00 --kl-loss-coef 0.00 --entropy-coef 0.00 "
        # Symmetric, verl's default, which VAGEN does not override. An asymmetric high side
        # is the DAPO recipe -- a different algorithm.
        "--eps-clip 0.2 "
        # VAGEN critic.optim.lr=1e-5 against actor 1e-6. slime has no --critic-lr; a
        # role-tagged Megatron config is the channel.
        f"--megatron-config-path {role_config(cfg, save_dir)} "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    backend_args = (
        "--train-backend megatron "
        f"--tensor-model-parallel-size {cfg.tp} "
        "--pipeline-model-parallel-size 1 "
        # Keep at 1: the reward/advantage path is not CP-aware.
        "--context-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {cfg.max_tokens_per_gpu} "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        # flash, not the default `auto`: auto picks TE's cuDNN fused attention, which on a
        # box with both cuda-12 and cuda-13 in the ldconfig cache dies with "Multiple
        # libcudart libraries found" -- inside the first forward pass, after the rollout has
        # already been paid for.
        "--attention-backend flash "
        # The bridge builds the model from the HF config, which is what gives a VLM its
        # vision tower -- MODEL_ARGS only describes the language tower.
        "--megatron-to-hf-mode bridge "
    )

    # slime prefixes every sglang ServerArgs flag with `--sglang-`
    # (`slime/backends/sglang_utils/arguments.py`), so anything sglang accepts is reachable.
    sglang_args = (
        f"--rollout-num-gpus-per-engine {cfg.rollout_tp} "
        "--sglang-mem-fraction-static 0.6 "
        + ("".join(f"--sglang-{n}attention-backend {cfg.sglang_attention_backend} "
                    for n in ("", "prefill-", "decode-"))
           if cfg.sglang_attention_backend else "")
        + (f"--sglang-mm-attention-backend {cfg.sglang_mm_attention_backend} "
           if cfg.sglang_mm_attention_backend else "")
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {cfg.actor_gpus} "
        f"--rollout-num-gpus {cfg.num_gpus - cfg.actor_gpus} "
    )

    execute_train(
        train_args=" ".join([ckpt_args, verify_args, rollout_args, ppo_args, optimizer_args,
                             backend_args, sglang_args, misc_args,
                             get_default_wandb_args(__file__, run_name_prefix=exp_name)]),
        num_gpus_per_node=cfg.num_gpus,
        megatron_model_type=megatron_model_type,
        train_script=f"{Path(os.environ.get('SLIME_DIR', REPO / 'slime'))}/{cfg.train_script}",
        extra_env_vars={
            # Read by eval_datasets.yaml through OmegaConf's ${oc.env:...}, so the eval config needs
            # no absolute paths and works from any working directory.
            "VAGEN_DATA_DIR": cfg.data_dir,
            # eval_datasets.yaml interpolates this, so --eval-yaml is the only place the held-out
            # set is named.
            "VAGEN_EVAL_DATA": eval_data,
            # Interpolated by eval_datasets.yaml, so the panel prefix follows --eval-yaml
            # instead of being frozen in the file.
            "VAGEN_EVAL_NAME": eval_dataset_name(cfg.eval_yaml),
            # execute_train hardcodes PYTHONPATH=/root/Megatron-LM/ (its docker layout);
            # ours has to carry this repo too, or the Ray workers cannot import vagen_agent.
            # Keep any site overlays supplied by the launcher as well.  AWS jobs install
            # the small Gym/Sokoban dependency set on FSx and expose it through the
            # parent PYTHONPATH; dropping that path here makes Ray workers unable to
            # import gym_sokoban even though the launcher process can import it.
            "PYTHONPATH": ":".join(
                p for p in (str(REPO), f"{REPO}/build/Megatron-LM",
                            os.environ.get("PYTHONPATH", "")) if p
            ),
            # Ray workers do not source the conda activation hook.  Carry both
            # libstdc++ from the env and cuDNN from the PyTorch wheel layout.
            "LD_LIBRARY_PATH": runtime_ld_library_path(),
            # Environment services may use a private CA or an explicitly trusted
            # self-signed certificate. Keep that narrow transport decision available
            # inside Ray workers; it affects only the render client.
            **{
                name: os.environ[name]
                for name in ("RENDER_TLS_NO_VERIFY", "SSL_CERT_FILE")
                if os.environ.get(name)
            },
            # W&B authenticates from ~/.netrc.  Propagate only non-secret settings:
            # runtime-env JSON is printed by Ray, so an API key must never be put here.
            **{
                name: os.environ[name]
                for name in (
                    "WANDB_BASE_URL", "WANDB_ENTITY", "WANDB_PROJECT",
                    "WANDB_MODE", "WANDB_DIR",
                )
                if os.environ.get(name)
            },
            # A VLM's vision tower is built by the bridge off the HF config and picks its
            # Transformer Engine backend on its own -- Megatron's --attention-backend does
            # not reach it. TE's cuDNN fused attention then refuses to run on a box that
            # carries both a cuda-12 and a cuda-13 libcudart in the ldconfig cache (this
            # one does: /usr/local/cuda and /usr/local/cuda-13), raising "Multiple
            # libcudart libraries found" inside the first forward pass, after the rollout
            # has already been paid for.
            #
            # These two are only consistent because *both* roles are on flash -- Megatron
            # asserts if they contradict the chosen backend, and a role config that omits
            # attention_backend falls back to `auto`, which expects fused. See
            # megatron_roles.yaml.
            "NVTE_FUSED_ATTN": "0",
            "NVTE_FLASH_ATTN": "1",
            # Meta's W&B is an internal instance, not wandb.ai. `~/.config/wandb/settings`
            # already points there, but a Ray worker's environment is built explicitly, so
            # pass it through rather than relying on the file being read.
            **({"WANDB_BASE_URL": os.environ["WANDB_BASE_URL"]}
               if os.environ.get("WANDB_BASE_URL") else {}),
        },
    )


if __name__ == "__main__":
    import typer

    typer.run(dataclass_cli(main))
