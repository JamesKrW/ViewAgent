"""
GraphRL pipeline entry point + center controller.

Orchestrates the iterative pipeline ``RL → TrajToSFT → SFT`` per iteration.

Mono-backend by design:
  - RL  is always VAGEN-SLIME  (graphrl.slime.SlimeWrapper)
  - SFT is always LLaMA-Factory (graphrl.llama_factory.LFWrapper)
  - TrajToSFT is the only user-extension point — pipeline.yaml supplies a
    dotted path to a ``TrajToSFTModule`` subclass via ``traj_to_sft.module``.

Phase skipping: set ``iteration_overrides.iterN.<phase>: null`` (or
``{skip: true}``) to drop that phase for that iteration. ``merge_iteration_config``
returns ``None`` in those cases and the phase is not built.

Responsibilities of ``GraphRLController``:
  1. Auto path orchestration  -- generate iter_XXX/ directory trees
  2. Config merging           -- module defaults + general/iter overrides
  3. Resource preemption      -- kill the previous GPU module before starting next
  4. Non-blocking polling     -- launch() then poll is_done()
  5. Resume detection         -- skip already-completed phases
  6. End-of-iter HF upload (optional)

Usage::

    python -m graphrl.main --config-path configs --config-name pipeline \\
        experiment_dir=/path/to/exp \\
        initial_model_path=Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
from omegaconf import DictConfig, OmegaConf

from graphrl import (
    LFWrapper,
    ModuleOutput,
    ModuleState,
    TrajToSFTModule,
    TrajToSFTPaths,
    SlimeWrapper,
    load_traj_to_sft_class,
)
from graphrl.utils.config import (
    RL_DEFAULTS_PATH,
    SFT_DEFAULTS_PATH,
    load_backend_default_config,
    load_pipeline_config,
    merge_iteration_config,
)
from graphrl.utils.iter_cleanup import cleanup_iter, process_pending_deletes
from graphrl.utils.logging import setup_logging
from graphrl.utils.progress import detect_progress

logger = logging.getLogger(__name__)


# ── unified iter shape ────────────────────────────────────────────────────
#
# The framework guarantees that every successfully-running iter ends up with
# both ``iter_N/rl_model/`` and ``iter_N/sft_model/`` in place. When a phase
# is *skipped* (config null), the controller adds a ``MaterializePhase`` that
# symlinks the upstream model into the missing slot — so resume detection,
# downstream phases and end-of-iter cleanup all see a uniform layout.


class MaterializePhase:
    """A trivial phase: symlink ``source`` → ``dest``.

    Used by ``GraphRLController._build_phases`` when an iter skips RL or SFT
    but the unified shape requires the corresponding model dir to exist.
    Matches the same launch/is_done/kill/get_output interface as the real
    backend wrappers so the controller's main loop doesn't need to care.

    The symlink is always relative to ``source`` (no copy, no disk doubling).
    Cleanup is symlink-aware so the underlying source is never silently
    deleted out from under a still-needed alias.
    """

    def __init__(self, source: Path, dest: Path, name: str):
        self.source = Path(source)
        self.dest = Path(dest)
        self.name = name
        # Mimic the wrappers' interface so the controller can introspect.
        self.config: Dict[str, Any] = {}
        self.input_paths: Dict[str, str] = {"source": str(self.source)}
        self.output_paths: Dict[str, str] = {"model": str(self.dest)}
        self._state = ModuleState.IDLE

    @property
    def state(self) -> ModuleState:
        return self._state

    def launch(self) -> None:
        if self.dest.exists() or self.dest.is_symlink():
            # Idempotent: if already in place (from prior run), accept it.
            self._state = ModuleState.DONE
            return
        if not self.source.exists():
            raise FileNotFoundError(
                f"{self.name}: source {self.source} does not exist; "
                "cannot materialize the symlink."
            )
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(self.source, self.dest)
        logger.info("[%s] symlinked %s → %s", self.name, self.dest, self.source)
        self._state = ModuleState.DONE

    def is_done(self) -> bool:
        return self.dest.exists() or self.dest.is_symlink()

    def is_already_complete(self) -> bool:
        return self.is_done()

    def kill(self) -> None:
        self._state = ModuleState.TERMINATED

    def get_output(self) -> ModuleOutput:
        if self.is_done():
            return ModuleOutput(model_path=str(self.dest))
        return ModuleOutput()


# A "phase" is anything with launch / is_done / kill / get_output —
# i.e. one of the three concrete module classes plus MaterializePhase.
Phase = Union[SlimeWrapper, TrajToSFTModule, LFWrapper, MaterializePhase]


# ── framework defaults (used when pipeline.yaml omits these keys) ─────────
#
# Two delete triggers (file-system signals):
#   * ``delete_on_sft_model``      — fires when *this* iter's sft_model is real
#   * ``delete_on_next_rl_model``  — fires when the *next* iter's rl_model is real
#
# Defaults reclaim everything upstream of sft_model on the first trigger and
# drop sft_model itself on the second. Override at top level or per-iter; set
# to ``[]`` (or ``null``) to disable for that scope.
DEFAULT_UPLOAD_TO_HF = ["rl_model", "sft_model"]
DEFAULT_DELETE_ON_SFT_MODEL = ["rl_model", "rollout_data", "rl", "traj_to_sft"]
DEFAULT_DELETE_ON_NEXT_RL_MODEL = ["sft_model", "sft", "sft_data"]


class GraphRLController:
    """Center controller for the GraphRL iterative pipeline."""

    def __init__(self, config):
        if isinstance(config, (str, Path)):
            self.raw_config = load_pipeline_config(str(config))
        else:
            self.raw_config = config

        self.experiment_dir = Path(
            self.raw_config.get("experiment_dir", "experiments")
        ).expanduser().resolve()
        self.initial_model = self.raw_config.get("initial_model_path")

        # Default configs for the two fixed backends are loaded here so
        # pipeline.yaml does not need to declare ``module_defaults``.
        self.defaults: Dict[str, Any] = {
            "rl": load_backend_default_config(RL_DEFAULTS_PATH),
            "sft": load_backend_default_config(SFT_DEFAULTS_PATH),
            # TrajToSFT has no module-level defaults — entirely env-defined.
            "traj_to_sft": {},
        }
        self.general_overrides = self.raw_config.get("general_overrides", {})

        self.num_iterations = self.raw_config.get("iterations", 1)
        # iteration_overrides keys can be ints or "iter0"/"iter1" strings.
        raw_overrides = self.raw_config.get("iteration_overrides", {})
        self.iteration_overrides: Dict[int, dict] = {
            int(str(k)[4:]) if str(k).startswith("iter") else int(k): v
            for k, v in raw_overrides.items()
        } if raw_overrides else {}

        self._active_module: Optional[Phase] = None

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.experiment_dir)

        num_iterations = self.num_iterations
        logger.info(f"GraphRL Pipeline: {num_iterations} iteration(s) configured")
        logger.info(f"Experiment dir: {self.experiment_dir}")
        logger.info(f"Initial model: {self.initial_model}")

        start_iter, start_phase, last_output = detect_progress(
            self.experiment_dir, num_iterations
        )
        if start_iter >= num_iterations:
            logger.info("Pipeline already complete!")
            return

        current_model = last_output.model_path if last_output else self.initial_model
        current_data: Dict[str, str] = last_output.data_paths if last_output else {}

        for iter_idx in range(start_iter, num_iterations):
            logger.info("=" * 60)
            logger.info(f"ITERATION {iter_idx}/{num_iterations - 1}")
            logger.info("=" * 60)

            iter_dir = self.experiment_dir / f"iter_{iter_idx:03d}"
            iter_config = self.iteration_overrides.get(iter_idx, {})

            # Retain the established fixed-flow environment variables for
            # TrajToSFT extensions that use the round identity. Adaptive control
            # has a separate namespace and entry point in ``main_adaptive``.
            os.environ["GRAPHRL_EXPERIMENT_DIR"] = str(self.experiment_dir)
            os.environ["GRAPHRL_ITERATION"] = str(iter_idx)

            phases = self._build_phases(iter_idx, iter_dir, iter_config, current_model)
            phase_start = start_phase if iter_idx == start_iter else 0

            for phase_idx in range(phase_start, len(phases)):
                module = phases[phase_idx]

                # If a SFT phase has no input dataset, skip it (TrajToSFT may
                # have produced nothing; e.g. during warm-up iters).
                if isinstance(module, LFWrapper):
                    sft_data_dir = module.input_paths.get("sft_data")
                    if sft_data_dir and not (
                        Path(sft_data_dir) / "dataset_info.json"
                    ).exists():
                        logger.warning(
                            f"[{module.name}] No SFT training data found, "
                            f"skipping SFT training for this iteration"
                        )
                        continue

                if module.is_already_complete():
                    logger.info(f"[{module.name}] Already complete, skipping")
                    output = module.get_output()
                else:
                    output = self._run_module(module)

                if output.model_path:
                    current_model = output.model_path
                current_data.update(output.data_paths)

                logger.info(
                    f"[{module.name}] Done. model={output.model_path}, "
                    f"data={list(output.data_paths.keys())}"
                )

                # Retry deferred deletes after EACH phase finishes — not at
                # iter end. The "next iter's rl_model is real" trigger
                # becomes true the moment THIS iter's RL phase exits, so
                # firing here lets us drop iter_(N-1)'s sft_model + sft_data
                # *before* iter_N's TrajToSFT/SFT need disk for their own
                # outputs, instead of running with both iters' models on
                # disk simultaneously.
                process_pending_deletes(self.experiment_dir)

            # Resolve per-iter overrides (iter_overrides → top-level → default)
            upload_list = self._iter_value(
                iter_config, "upload_to_hf", DEFAULT_UPLOAD_TO_HF,
            ) or []
            on_sft = self._iter_value(
                iter_config, "delete_on_sft_model", DEFAULT_DELETE_ON_SFT_MODEL,
            ) or []
            on_next_rl = self._iter_value(
                iter_config, "delete_on_next_rl_model", DEFAULT_DELETE_ON_NEXT_RL_MODEL,
            ) or []

            self._maybe_upload_to_hf(iter_idx, iter_dir, upload_list)

            cleanup_iter(
                iter_num=iter_idx,
                experiment_dir=self.experiment_dir,
                delete_on_sft_model=on_sft,
                delete_on_next_rl_model=on_next_rl,
            )

        if self._active_module:
            self._active_module.kill()
            self._active_module = None

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Final model: {current_model}")
        logger.info("=" * 60)

    # ── phase execution ──────────────────────────────────────────────────

    def _run_module(self, module: Phase) -> ModuleOutput:
        needs_gpu = isinstance(module, (SlimeWrapper, LFWrapper))

        if needs_gpu and self._active_module:
            logger.info(
                f"Resource preemption: killing [{self._active_module.name}] "
                f"before launching [{module.name}]"
            )
            self._active_module.kill()
            self._active_module = None

        logger.info(f"[{module.name}] Launching...")
        module.launch()

        if needs_gpu:
            self._active_module = module
            self._poll_until_done(module)
        # TrajToSFT: launch() runs synchronously — no preemption needed
        # (the prior RL was already killed before reaching this branch),
        # and the subclass is free to use the GPU during ``run()``.

        output = module.get_output()

        logger.info(f"[{module.name}] Releasing resources...")
        module.kill()
        if self._active_module is module:
            self._active_module = None

        return output

    def _poll_until_done(self, module: Phase) -> None:
        poll_interval = module.config.get("poll_interval", 30)
        timeout = module.config.get("timeout", 259200)  # 72h default
        start = time.monotonic()

        while True:
            if module.is_done():
                logger.info(f"[{module.name}] Done")
                return
            if module.state == ModuleState.FAILED:
                raise RuntimeError(f"Module [{module.name}] failed")
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Module [{module.name}] timed out after {elapsed:.0f}s"
                )
            time.sleep(poll_interval)

    # ── phase construction ────────────────────────────────────────────────

    def _build_phases(
        self,
        iter_num: int,
        iter_dir: Path,
        iter_config: dict,
        current_model: str,
    ) -> List[Phase]:
        """Build the phase list for one iteration.

        Per-iteration directory layout (unified shape — every iter has these
        two model dirs after running, real or symlinked)::

            iter_XXX/
                rl/                  # RL working directory
                    rollout_data/    # compatibility JSONLs (TrajToSFT input)
                    slime_checkpoints/
                    rl_model/        # RL output, OR symlink → upstream model
                traj_to_sft/
                    sft_data/        # LLaMA-Factory dataset
                sft/                 # SFT working directory (logs, config)
                    sft_model/       # SFT output, OR symlink → rl_model

        When RL is skipped (``rl: null`` for this iter), the controller adds
        a ``MaterializePhase`` that symlinks the upstream model into
        ``iter_N/rl/rl_model``. iter_0 with RL skipped requires a local
        ``initial_model_path`` (or a pre-placed ``iter_0/rl/rl_model/``).

        When SFT is skipped, a ``MaterializePhase`` symlinks
        ``iter_N/rl/rl_model`` → ``iter_N/sft/sft_model`` so downstream iters
        can always start from ``iter_N/sft/sft_model``.
        """
        phases: List[Phase] = []

        project_name = self.raw_config.get("project_name", "graphrl")
        experiment_name = self.raw_config.get("experiment_name", "graphrl_pipeline")
        # Layout contract (v2): every phase artefact lives under iter_XXX/<phase>/
        rl_dir = iter_dir / "rl"
        traj_dir = iter_dir / "traj_to_sft"
        sft_dir = iter_dir / "sft"
        rl_model_path = rl_dir / "rl_model"
        sft_model_path = sft_dir / "sft_model"
        sft_data_path = traj_dir / "sft_data"

        # ── Phase 1: RL (or symlink-materialize the rl_model slot) ───────
        rl_cfg = merge_iteration_config(
            self.defaults.get("rl", {}),
            self.general_overrides.get("rl"),
            iter_config.get("rl", iter_config.get("rl_config", {})),
        )
        if rl_cfg is not None:
            rl_cfg["_iter_num"] = iter_num
            rl_cfg["_project_name"] = project_name
            rl_cfg["_experiment_name"] = experiment_name
            phases.append(SlimeWrapper(
                config=rl_cfg,
                input_paths={"model": current_model},
                output_paths={
                    "base_dir": str(rl_dir),
                    "model": str(rl_model_path),
                },
            ))
        else:
            phases.append(self._build_skip_rl_materialize(iter_num, rl_model_path))

        # ── Phase 2: TrajToSFT ───────────────────────────────────────────
        traj_cfg = merge_iteration_config(
            self.defaults.get("traj_to_sft", {}),
            self.general_overrides.get("traj_to_sft"),
            iter_config.get("traj_to_sft", iter_config.get("traj_to_sft_config", {})),
        )
        if traj_cfg is not None:
            module_spec = traj_cfg.get("module")
            if not module_spec:
                raise ValueError(
                    "traj_to_sft.module must be set to a dotted-path of a "
                    "TrajToSFTModule subclass (e.g. "
                    "'graphrl.envs.viewsuite.viewsuite_interactive_view_planning.InteractiveViewPlanningTrajToSFT')"
                )
            traj_cls = load_traj_to_sft_class(module_spec)
            paths = TrajToSFTPaths(
                base_dir=iter_dir,
                rollout_data=rl_dir / "rollout_data",
                rl_model=rl_model_path,
                sft_data=sft_data_path,
            )
            phases.append(traj_cls(config=traj_cfg, paths=paths))

        # ── Phase 3: SFT (or symlink-materialize the sft_model slot) ─────
        sft_cfg = merge_iteration_config(
            self.defaults.get("sft", {}),
            self.general_overrides.get("sft"),
            iter_config.get("sft", iter_config.get("sft_config", {})),
        )
        if sft_cfg is not None:
            sft_cfg["_iter_num"] = iter_num
            sft_cfg["_project_name"] = project_name
            sft_cfg["_experiment_name"] = experiment_name
            # ``input_model_path`` (optional, default = this iter's
            # rl/rl_model). Set in pipeline.yaml or per-iter override when
            # the SFT phase should fine-tune from a DIFFERENT model than
            # the one RL just produced — e.g. the post-reasoning-training
            # pipeline where iter_0's RL is used purely for trajectory
            # collection (rollouts → reasoning annotation), and the SFT
            # phase trains base Qwen on the annotated data so the
            # finetune isn't biased by the prior RL drift.
            sft_input_model_override = sft_cfg.get("input_model_path")
            sft_input_model = (
                str(sft_input_model_override) if sft_input_model_override
                else str(rl_model_path)
            )
            phases.append(LFWrapper(
                config=sft_cfg,
                input_paths={
                    "model": sft_input_model,
                    "sft_data": str(sft_data_path),
                },
                output_paths={
                    "base_dir": str(sft_dir),
                    "model": str(sft_model_path),
                },
            ))
        else:
            phases.append(MaterializePhase(
                source=rl_model_path,
                dest=sft_model_path,
                name="materialize-sft_model",
            ))

        return phases

    def _build_skip_rl_materialize(
        self, iter_num: int, rl_model_path: Path,
    ) -> "MaterializePhase":
        """Resolve the source for a skipped RL phase + return the materializer.

        For iter_0: source = ``initial_model_path`` (must be a local dir,
        unless ``rl_model_path`` is already pre-placed by the user).
        For iter_N (N > 0): source = ``iter_{N-1}/sft_model``.

        If ``rl_model_path`` already exists (e.g. pre-placed by run.sh's
        Stage 0b), we accept it as-is and skip source validation —
        MaterializePhase is idempotent and treats an existing dest as done.

        Otherwise, if the source can't be resolved as a local path (e.g.
        ``initial_model_path`` is a HuggingFace id like
        ``"Qwen/Qwen2.5-VL-7B-Instruct"``), raises ``ValueError`` — the user
        must pre-place ``iter_0/rl_model/`` themselves (e.g. via a
        ``download_base_model.sh`` step in run.sh).
        """
        # Pre-placed: accept whatever's there (real dir or symlink).
        if rl_model_path.exists() or rl_model_path.is_symlink():
            return MaterializePhase(
                source=rl_model_path, dest=rl_model_path,
                name=f"materialize-rl_model(iter{iter_num})",
            )

        if iter_num == 0:
            if not self.initial_model:
                raise ValueError(
                    "iter_0 has 'rl: null' but no initial_model_path is set. "
                    "Either provide a local initial_model_path or pre-place "
                    f"a model dir at {rl_model_path}."
                )
            src = Path(self.initial_model).expanduser()
            if not src.is_absolute() and not src.exists():
                # Probably a HF id like "Qwen/Qwen2.5-VL-7B-Instruct" — bail.
                raise ValueError(
                    f"iter_0 has 'rl: null' but initial_model_path "
                    f"{self.initial_model!r} does not resolve to a local "
                    "directory. Either point initial_model_path at a local "
                    "snapshot, or pre-place the base model at "
                    f"{rl_model_path} before running (e.g. in run.sh)."
                )
            if not src.is_dir():
                raise ValueError(
                    f"iter_0 has 'rl: null' but initial_model_path {src} is "
                    f"not a directory. Pre-place the base model at {rl_model_path}."
                )
            return MaterializePhase(
                source=src.resolve(), dest=rl_model_path,
                name="materialize-rl_model(iter0)",
            )
        prev_sft = self.experiment_dir / f"iter_{iter_num - 1:03d}" / "sft" / "sft_model"
        return MaterializePhase(
            source=prev_sft, dest=rl_model_path,
            name=f"materialize-rl_model(iter{iter_num})",
        )

    # ── per-iter override resolver ────────────────────────────────────────

    def _iter_value(self, iter_config: dict, key: str, default=None):
        """Per-iter value if set on the iter, else top-level pipeline.yaml value.

        Used for ``upload_to_hf`` / ``delete_on_sft_model`` /
        ``delete_on_next_rl_model`` so each iter can override the top-level
        list. To disable for an iter explicitly, set the key to ``[]`` in
        ``iteration_overrides.iterN``.
        """
        if key in iter_config:
            return iter_config[key]
        return self.raw_config.get(key, default)

    # ── HuggingFace upload (per-iter, optional) ──────────────────────────

    def _maybe_upload_to_hf(self, iter_num: int, iter_dir: Path, resources) -> None:
        if not resources:
            return
        from graphrl.utils.hf_uploader import upload_iter_resources
        try:
            upload_iter_resources(
                iter_dir=iter_dir,
                iter_num=iter_num,
                resources=list(resources),
                project_name=self.raw_config.get("project_name", "graphrl"),
                experiment_name=self.raw_config.get("experiment_name", "graphrl_pipeline"),
                repo_owner=self.raw_config.get("upload_to_hf_repo_owner"),
                model_repo=self.raw_config.get("upload_to_hf_model_repo"),
                data_repo=self.raw_config.get("upload_to_hf_data_repo"),
                visibility=self.raw_config.get("upload_to_hf_visibility", "public"),
                unified_repo=bool(self.raw_config.get("upload_to_hf_unified_repo", True)),
            )
        except Exception as exc:
            logger.warning(
                "[upload_to_hf] iteration %d upload failed: %s",
                iter_num, exc, exc_info=True,
            )


# ── Hydra entry point ─────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    GraphRLController(config_dict).run()


if __name__ == "__main__":
    main()
