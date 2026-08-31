"""Habitat-GS IVP on the vendored synchronous VAGEN-SLIME backend.

The heavy launcher and checkpoint semantics stay in ``GraphRL/VAGEN``.  This
ViewAgent-owned file supplies only Habitat-GS paths/defaults and makes the local
``view_suite`` package visible to Ray workers.

First end-to-end check::

    python GraphRL/examples/viewsuite/habitat_gs_interactive_view_planning/slime/run_habitat_gs.py \
      --rollout-only \
      --eval-yaml GraphRL/examples/viewsuite/habitat_gs_interactive_view_planning/slime/verify.yaml
"""

# Keep annotations concrete: slime's dataclass_cli inspects the Config annotation.
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRAPHRL_ROOT = HERE.parents[3]
VIEWAGENT_ROOT = GRAPHRL_ROOT.parent
VAGEN_SLIME_ROOT = GRAPHRL_ROOT / "VAGEN"
sys.path.insert(0, str(VAGEN_SLIME_ROOT))
sys.path.insert(0, str(VAGEN_SLIME_ROOT / "slime"))

from slime.utils.external_utils.typer_utils import dataclass_cli

_BASE_LAUNCHER = runpy.run_path(
    str(VAGEN_SLIME_ROOT / "examples/train/sokoban/run_sokoban.py"),
    run_name="vagen_slime_shared_launcher",
)
BaseConfig = _BASE_LAUNCHER["Config"]
run_slime = _BASE_LAUNCHER["main"]


def _cached_qwen25_path() -> str:
    named = Path.home() / "models/Qwen2.5-VL-7B-Instruct"
    if (named / "config.json").is_file():
        return str(named)
    snapshots = Path.home() / (
        ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
    )
    for candidate in sorted(snapshots.glob("*"), reverse=True):
        if (candidate / "config.json").is_file():
            return str(candidate)
    return str(named)


@dataclass
class Config(BaseConfig):
    model_name: str = "Qwen2.5-VL-7B-Instruct"
    model_path: str = _cached_qwen25_path()
    harness_config: str = str(HERE / "no_concat_ppo.yaml")
    envs_yaml: str = str(HERE / "train.yaml")
    eval_yaml: str = str(HERE / "val.yaml")
    eval_config: str = str(HERE / "eval_datasets.yaml")
    megatron_roles_config: str = str(HERE / "megatron_roles.yaml")
    generate_function_path: str = "graphrl.slime.rollout.generate"
    dataset_prefix: str = "habitat_gs_ivp"
    experiment_prefix: str = "habitat_gs_ivp_slime"
    viewagent_root: str = str(VIEWAGENT_ROOT)
    data_dir: str = str(GRAPHRL_ROOT / "outputs/slime_data")
    runs_dir: str = str(GRAPHRL_ROOT / "exps/viewagent_slime")
    # The current internal renderer serves a self-signed HTTPS certificate. This maps
    # only to ViewAgent's render client switch and is propagated to Ray workers.
    insecure_render_tls: bool = True

    num_gpus: int = 8
    actor_gpus: int = 4
    rollout_batch_size: int = 128
    num_rollout: int = 801
    eval_interval: int = 20
    save_interval: int = 20
    max_tokens_per_gpu: int = 32768
    rollout_response_tokens: int = 512
    rollout_context_tokens: int = 16384
    turn_tokens: int = 512
    # Keep the first migration runs strictly synchronous: rollout -> train -> checkpoint
    # / eval. This removes async overlap as a variable while checkpoint and resume are
    # being validated.
    train_script: str = "train.py"


def main(cfg: Config) -> None:
    viewagent = Path(cfg.viewagent_root).resolve()
    if not (viewagent / "view_suite").is_dir():
        raise SystemExit(f"ViewAgent checkout not found at {viewagent}")
    if not Path(cfg.model_path).joinpath("config.json").is_file():
        raise SystemExit(
            f"Qwen checkpoint not found at {cfg.model_path}; pass --model-path explicitly"
        )

    os.environ["VIEWSUITE_ROOT"] = str(viewagent)
    if cfg.insecure_render_tls:
        os.environ["RENDER_TLS_NO_VERIFY"] = "1"
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(
        part
        for part in (
            str(viewagent),
            str(GRAPHRL_ROOT),
            str(VAGEN_SLIME_ROOT),
            str(VAGEN_SLIME_ROOT / "slime"),
            inherited,
        )
        if part
    )
    run_slime(cfg)


if __name__ == "__main__":
    import typer

    typer.run(dataclass_cli(main))
