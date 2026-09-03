"""
VAGEN command builder.

Builds the command line for launching ``python3 -m vagen.training.main`` with
GraphRL's Hydra config directory as the config source.

Config architecture:
  - VAGEN's own ``vagen/configs/vagen_multiturn.yaml`` is the primary config, so
    the backend's settings are inherited rather than copied. It has to be the
    primary one: it sets ``hydra.searchpath``, and Hydra allows that only there.
  - ``graphrl/configs/vagen_configs/env_registry.yaml``: the ViewSuite
    environment registry, emitted as ``+env_registry.<Name>=<path>`` overrides
    that append to VAGEN's own registry. No submodule edit, no copied config.
  - ``graphrl/configs/vagen_configs/config.yaml``: GraphRL's own defaults, whose
    ``hydra_overrides`` block reaches VAGEN the same way any experiment's does.
  - ``config["hydra_overrides"]``: flattened into Hydra CLI args
  - Controller-managed paths (model, checkpoint, rollout) are always appended last

Porting notes (VAGEN 260814 and later):
  - The entrypoint is ``vagen.training.main``. The old ``vagen.main_ppo`` is gone.
  - ``vagen/configs/baseline_vllm.flags`` holds the flags that make a run work at
    all -- most importantly the two that select VAGEN's own agent loop. Without
    them verl silently runs its own loop and the job looks healthy while none of
    VAGEN's rollout code executes. The file is read here rather than duplicated so
    the two cannot drift; ``$V`` inside it is the VAGEN checkout.
  - verl is a checkout on ``PYTHONPATH``, not an installed package, and its Hydra
    config dir must be on ``hydra.searchpath`` or ``ppo_trainer`` will not resolve.
    See :func:`build_vagen_env`.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

# GraphRL package root (graphrl/) -- this file lives at graphrl/vagen/utils/
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# Default Hydra config directory for VAGEN
_DEFAULT_HYDRA_CONFIG_DIR = _PACKAGE_ROOT / "configs" / "vagen_configs"

# Repo root (ViewAgent/) -- graphrl/ lives at GraphRL/graphrl/
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
_DEFAULT_VAGEN_DIR = _REPO_ROOT / "GraphRL" / "VAGEN"


def resolve_vagen_dir(config: Dict[str, Any]) -> Path:
    """Absolute path to the VAGEN checkout that supplies the backend."""
    raw = config.get("vagen_dir")
    if not raw:
        return _DEFAULT_VAGEN_DIR
    path = Path(raw).expanduser()
    # Historical configs say ``vagen_dir: VAGEN`` and rely on the process CWD being
    # the repo root. Anchor a relative value to the repo instead, so the launcher
    # works from anywhere.
    return path if path.is_absolute() else (_REPO_ROOT / path)


def resolve_verl_dir(vagen_dir: Path) -> Path:
    """The verl checkout VAGEN trains through.

    Probed for a file rather than a directory: an uninitialised submodule leaves
    ``VAGEN/verl`` present but empty, and left unresolved that surfaces much later
    as a Hydra error that does not mention verl.
    """
    for candidate in (vagen_dir / "verl", vagen_dir.parent / "verl"):
        if (candidate / "verl" / "trainer" / "config" / "ppo_trainer.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"verl not found at {vagen_dir}/verl or {vagen_dir.parent}/verl. "
        f"Run: git submodule update --init --recursive"
    )


def build_vagen_env(config: Dict[str, Any]) -> Dict[str, str]:
    """Process environment for the VAGEN subprocess.

    verl has to come first on ``PYTHONPATH`` so this fork wins over any other copy
    (``conda_envs/slime`` ships an installed verl 0.6.1 that would otherwise shadow it).

    The repo root follows, because the environment classes named in
    ``vagen_configs/env_registry.yaml`` live in ViewAgent's own ``view_suite``
    package. VAGEN imports them by dotted path from inside its own checkout, so
    without this the registry resolves to ``ModuleNotFoundError: view_suite`` --
    which VAGEN's loader reports per-env as a warning, leaving a run that starts
    cleanly and then fails on the first unknown env name.
    """
    vagen_dir = resolve_vagen_dir(config)
    verl_dir = resolve_verl_dir(vagen_dir)
    parts = [str(verl_dir), str(vagen_dir), str(_REPO_ROOT)]
    inherited = os.environ.get("PYTHONPATH", "")
    if inherited:
        parts.extend(p for p in inherited.split(os.pathsep) if p)
    # dict.fromkeys de-duplicates while keeping first-wins order
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(parts)),
    }


def _baseline_flags(vagen_dir: Path) -> List[str]:
    """Read VAGEN's shared flags file, substituting ``$V`` for the checkout."""
    flags_path = vagen_dir / "vagen" / "configs" / "baseline_vllm.flags"
    if not flags_path.is_file():
        raise FileNotFoundError(
            f"VAGEN baseline flags not found at {flags_path}. Without them verl runs "
            f"its own agent loop and none of VAGEN's rollout code executes."
        )
    flags: List[str] = []
    for line in flags_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        flags.append(stripped.replace("$V", str(vagen_dir)))
    return flags


def _env_registry_args() -> List[str]:
    """ViewSuite env registrations, as Hydra append overrides.

    ``+`` because these keys do not exist in VAGEN's registry; a ``+`` override on
    a key that does exist is an error, which is why the YAML must not restate
    VAGEN's built-ins.
    """
    registry_path = _DEFAULT_HYDRA_CONFIG_DIR / "env_registry.yaml"
    if not registry_path.is_file():
        return []
    import yaml

    entries = (yaml.safe_load(registry_path.read_text()) or {}).get("env_registry") or {}
    return [f"+env_registry.{name}={path}" for name, path in entries.items()]


def _override_key(arg: str) -> str:
    """The dotted key of a Hydra CLI arg, ignoring any ``+``/``++`` prefix."""
    return arg.lstrip("+").split("=", 1)[0]


def build_vagen_command(
    config: Dict[str, Any],
    model_path: str,
    output_dir: Path,
) -> List[str]:
    """
    Build the full command list for launching VAGEN PPO training.

    The command uses GraphRL's own Hydra config directory (which inherits
    from VAGEN via searchpath) instead of pointing directly to VAGEN's configs.

    ``config["hydra_overrides"]`` is flattened into Hydra CLI args.
    Everything else in config is for GraphRL internal use.

    The controller always injects these paths (appended last, wins over config):
      - ``actor_rollout_ref.model.path``
      - ``critic.model.path``
      - ``trainer.default_local_dir``
      - ``trainer.rollout_data_dir``
    """
    config_name = config.get("hydra_config_name", "vagen_multiturn")

    vagen_dir = resolve_vagen_dir(config)
    verl_dir = resolve_verl_dir(vagen_dir)

    # VAGEN's own config dir, unless an experiment points somewhere else.
    hydra_config_dir = config.get("hydra_config_dir")
    if hydra_config_dir:
        config_dir = Path(hydra_config_dir).resolve()
    else:
        config_dir = vagen_dir / "vagen" / "configs"

    cmd = [
        # sys.executable, not "python3": the controller runs under the training env,
        # but "python3" resolves from PATH in the child, which is whatever conda env
        # happens to be active -- typically `base`, where hydra is not installed. That
        # surfaced as `ModuleNotFoundError: No module named 'hydra'` from a launcher
        # that had just imported hydra successfully itself.
        sys.executable or "python3",
        "-m",
        "vagen.training.main",
        f"--config-path={config_dir}",
        f"--config-name={config_name}",
    ]

    # Absolute searchpath to verl's config dir, which supplies ``ppo_trainer``. VAGEN's
    # own default for this is relative and only resolves when the CWD is its repo root;
    # every caller is expected to override it. Legal here only because the primary
    # config is VAGEN's own -- Hydra rejects a searchpath set from an included config.
    cmd.append(f"hydra.searchpath=[file://{verl_dir}/verl/trainer/config]")

    # Same story as the searchpath: VAGEN ships a relative default that only works from
    # its repo root.
    cmd.append(f"data.custom_cls.path={vagen_dir}/vagen/training/dataset.py")

    # ViewSuite environments, appended to VAGEN's registry.
    cmd.extend(_env_registry_args())

    # The flags that make a run work at all, including agent-loop selection.
    baseline = _baseline_flags(vagen_dir)

    # Flatten hydra_overrides into CLI args
    hydra_overrides = config.get("hydra_overrides", {})
    override_args = _flatten(hydra_overrides, prefix="") if hydra_overrides else []

    # Hydra takes the last value for a repeated key, so appending the experiment's
    # overrides after the baseline is already correct. Drop the shadowed baseline entry
    # anyway: leaving both makes the effective value invisible in the logged command,
    # and that is the command people read when a run does not do what the config says.
    overridden = {_override_key(a) for a in override_args}
    cmd.extend(f for f in baseline if _override_key(f) not in overridden)
    cmd.extend(override_args)

    # Auto-inject training_steps so users don't have to set it in two places
    training_steps = config.get("training_steps")
    if training_steps is not None:
        cmd.append(f"trainer.total_training_steps={training_steps}")

    # Always inject controller-managed paths (appended last = highest priority)
    # Use absolute paths so VAGEN (which runs from a different CWD) writes to the right location.
    ckpt_dir = str(output_dir.resolve() / "verl_checkpoints")
    rollout_dir = str(output_dir.resolve() / "rollout_data")
    cmd.extend([
        f"actor_rollout_ref.model.path={model_path}",
        f"critic.model.path={model_path}",
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.rollout_data_dir={rollout_dir}",
    ])

    # Inject WandB settings (always last so they override anything in hydra_overrides)
    project_name = config.get("_project_name", "graphrl")
    experiment_name = config.get("_experiment_name", "graphrl_pipeline")
    iter_num = config.get("_iter_num", 0)
    run_name = f"{experiment_name}_rl_iter{iter_num:03d}"
    cmd.extend([
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={run_name}",
        "trainer.logger=['console','wandb']",
    ])

    return cmd


# Keys that need Hydra's '+' prefix (new keys not in VAGEN's base config).
# Matches ViewSuite's convert_rl_rollout_to_sft convention.
_HYDRA_APPEND_KEY_PATTERNS = ["engine_kwargs", "eval_files"]


def _flatten(d: Union[Dict, Any], prefix: str) -> List[str]:
    """Recursively flatten a nested dict into Hydra CLI args."""
    if not isinstance(d, dict):
        # Add '+' prefix for keys that don't exist in VAGEN's base Hydra config
        if any(pattern in prefix for pattern in _HYDRA_APPEND_KEY_PATTERNS):
            return [f"+{prefix}={_format_value(d)}"]
        return [f"{prefix}={_format_value(d)}"]

    args: List[str] = []
    for key, value in d.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        args.extend(_flatten(value, new_prefix))
    return args


def _format_value(value: Any) -> str:
    """Format a Python value for Hydra CLI."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return str(value).replace(" ", "")
    return str(value)
