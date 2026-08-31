"""Adaptive, metric-driven GraphRL controller.

This is a separate entry point from :mod:`graphrl.main`. Fixed-schedule scripts keep
their fixed iteration/step control flow; adaptive scripts opt into this state
machine by invoking ``python -m graphrl.main_adaptive``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from graphrl import LFWrapper, SlimeWrapper, TrajToSFTModule
from graphrl.adaptive.model import (
    CHECKPOINT_MANIFEST,
    is_complete_checkpoint_manifest,
    is_complete_hf_model,
    is_complete_snapshot,
)
from graphrl.adaptive.rollouts import (
    durable_rollout_prefix,
    required_rollout_prefix,
)
from graphrl.adaptive.state import (
    CONFIG_FILENAME,
    DECISION_STOP_RUN,
    DECISION_SWITCH_TO_SFT,
    PHASE_FINALIZE_ROUND,
    PHASE_FINISHED,
    PHASE_RL,
    PHASE_SFT,
    PHASE_TRAJ_TO_SFT,
    AdaptiveStateStore,
    normalize_adaptive_config,
)
from graphrl.adaptive.viewsuite_traj_to_sft import ATOMIZE_STATE_FILENAME
from graphrl.llama_factory.adaptive_lf_wrapper import AdaptiveLFWrapper
from graphrl.main import (
    DEFAULT_DELETE_ON_NEXT_RL_MODEL,
    DEFAULT_DELETE_ON_SFT_MODEL,
    DEFAULT_UPLOAD_TO_HF,
    GraphRLController,
    Phase,
)
from graphrl.utils.iter_cleanup import cleanup_iter, process_pending_deletes
from graphrl.utils.logging import setup_logging
from graphrl.slime.adaptive_wrapper import AdaptiveSlimeWrapper

logger = logging.getLogger(__name__)

_SFT_DONE = ".adaptive_sft_done.json"


class AdaptiveGraphRLController(GraphRLController):
    """Run RL/SFT rounds until metric decisions or total RL budget stop them."""

    def __init__(self, config):
        super().__init__(config)
        raw_schedule = self.raw_config.get("adaptive_schedule")
        if not raw_schedule:
            raise ValueError("main_adaptive requires a top-level adaptive_schedule block")
        self.adaptive_config = normalize_adaptive_config(raw_schedule)
        self.store = AdaptiveStateStore(self.experiment_dir, self.adaptive_config)
        raw_round_overrides = self.raw_config.get("adaptive_round_overrides") or {}
        self.adaptive_round_overrides = {
            int(str(key)[4:]) if str(key).startswith("iter") else int(key): value
            for key, value in raw_round_overrides.items()
        }

    def run(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.experiment_dir)
        self._write_schedule_config()
        self._recover_or_guard_state()
        self.store.initialize()

        logger.info("Adaptive GraphRL Pipeline")
        logger.info("Experiment dir: %s", self.experiment_dir)
        logger.info("Initial model: %s", self.initial_model)
        logger.info(
            "Control: metric=%s, eval_every=%d, min_delta=%s, "
            "min_relative_delta=%s, sft_patience=%d, "
            "latch_threshold=%s, finish_patience=%d, total_rl_steps=%d",
            self.adaptive_config["metric"],
            self.adaptive_config["eval_every_steps"],
            self.adaptive_config["min_delta"],
            self.adaptive_config["min_relative_delta"],
            self.adaptive_config["sft_patience"],
            self.adaptive_config["latch_threshold"],
            self.adaptive_config["finish_patience"],
            self.adaptive_config["total_rl_steps"],
        )

        final_model = None
        try:
            while True:
                state = self.store.load()
                round_index = int(state["round"])
                phase = state["phase"]

                if phase == PHASE_FINISHED or (
                    state.get("decision") == DECISION_STOP_RUN
                    and state.get("decision_ready")
                ):
                    reason = state.get("stop_reason") or "adaptive_stop"
                    self.store.finish(reason)
                    final_model = self._global_best_model()
                    break

                if int(state.get("total_rl_steps", 0)) >= self.adaptive_config["total_rl_steps"]:
                    self.store.finish("total_rl_steps")
                    final_model = self._global_best_model()
                    break

                if phase == PHASE_FINALIZE_ROUND:
                    logger.info("=" * 60)
                    logger.info("ADAPTIVE ROUND %d — phase=%s", round_index, phase)
                    logger.info("=" * 60)
                    iter_dir = self.experiment_dir / f"iter_{round_index:03d}"
                    iter_config = copy.deepcopy(
                        self.adaptive_round_overrides.get(round_index, {})
                    )
                    self._finish_round(round_index, iter_dir, iter_config)
                    self.store.begin_next_round(round_index)
                    self._garbage_collect_best_snapshots()
                    continue

                logger.info("=" * 60)
                logger.info("ADAPTIVE ROUND %d — phase=%s", round_index, phase)
                logger.info("=" * 60)

                iter_dir = self.experiment_dir / f"iter_{round_index:03d}"
                upstream_model = self._phase_input_model(round_index, phase, state)
                iter_config = self._adaptive_iter_config(round_index, state)
                phases = self._build_adaptive_phases(
                    round_index, iter_dir, iter_config, upstream_model
                )

                os.environ.pop("GRAPHRL_EXPERIMENT_DIR", None)
                os.environ.pop("GRAPHRL_ITERATION", None)
                os.environ["GRAPHRL_ADAPTIVE_EXPERIMENT_DIR"] = str(self.experiment_dir)
                os.environ["GRAPHRL_ADAPTIVE_ROUND"] = str(round_index)

                if phase == PHASE_RL:
                    rl_module = self._one_phase(phases, AdaptiveSlimeWrapper)
                    output = self._run_or_resume(rl_module)
                    if not output.model_path:
                        raise RuntimeError("adaptive RL completed without a model output")
                    process_pending_deletes(self.experiment_dir)

                    state = self.store.load()
                    self._require_decision_round(state, round_index)
                    if not state.get("decision_ready"):
                        raise RuntimeError(
                            "adaptive RL returned before its terminal checkpoint "
                            "and state snapshot were committed"
                        )
                    if state["decision"] == DECISION_STOP_RUN:
                        self.store.finish(state.get("stop_reason") or "adaptive_stop")
                        final_model = self._global_best_model()
                        break
                    if state["decision"] != DECISION_SWITCH_TO_SFT:
                        raise RuntimeError(
                            "adaptive RL exited without SWITCH_TO_SFT or STOP_RUN; "
                            f"decision={state.get('decision')!r}"
                        )
                    self.store.set_phase(round_index, PHASE_TRAJ_TO_SFT)
                    continue

                if phase == PHASE_TRAJ_TO_SFT:
                    self._ensure_rl_output(phases)
                    traj_module = self._one_phase(phases, TrajToSFTModule)
                    marker = Path(traj_module.paths.sft_data) / ".phase_done"
                    if marker.is_file() and self._traj_output_is_complete(traj_module):
                        logger.info("[TrajToSFT] Completion marker found; resuming after phase")
                    else:
                        if marker.is_file():
                            logger.warning(
                                "[adaptive] invalid TrajToSFT output found; rebuilding phase"
                            )
                            self._clear_traj_output(traj_module)
                        self._require_complete_rollouts(traj_module, state, round_index)
                        # Any existing SFT checkpoint was trained against an
                        # older/incomplete dataset and must never be resumed
                        # after TrajToSFT is rebuilt.
                        self._clear_sft_output(phases)
                        self._run_module(traj_module)
                        if not self._traj_output_is_complete(traj_module):
                            raise RuntimeError(
                                "adaptive TrajToSFT completed without every configured "
                                "non-empty dataset"
                            )
                    process_pending_deletes(self.experiment_dir)
                    self.store.set_phase(round_index, PHASE_SFT)
                    continue

                if phase == PHASE_SFT:
                    self._ensure_rl_output(phases)
                    traj_module = self._one_phase(phases, TrajToSFTModule)
                    traj_marker = Path(traj_module.paths.sft_data) / ".phase_done"
                    if not traj_marker.is_file() or not self._traj_output_is_complete(
                        traj_module
                    ):
                        logger.warning(
                            "[adaptive] state says SFT but TrajToSFT output is "
                            "missing or incomplete; replaying TrajToSFT"
                        )
                        self._clear_traj_output(traj_module)
                        self._clear_sft_output(phases)
                        self.store.set_phase(round_index, PHASE_TRAJ_TO_SFT)
                        continue
                    sft_module = self._one_phase(phases, AdaptiveLFWrapper)
                    sft_marker = Path(sft_module.output_paths["base_dir"]) / _SFT_DONE
                    if sft_marker.is_file() and sft_module.has_complete_output():
                        logger.info("[SFT] Completion marker found; resuming after phase")
                    elif sft_module.has_complete_output() and sft_module.is_already_complete():
                        logger.info("[SFT] Complete output recovered; writing phase marker")
                        self._write_sft_marker(sft_marker, round_index)
                    else:
                        sft_data_dir = Path(sft_module.input_paths["sft_data"])
                        if not (sft_data_dir / "dataset_info.json").is_file():
                            raise RuntimeError(
                                f"adaptive SFT requested but dataset is missing: {sft_data_dir}"
                            )
                        output = self._run_module(sft_module)
                        if not output.model_path or not sft_module.has_complete_output():
                            raise RuntimeError("adaptive SFT completed without a valid model")
                        self._write_sft_marker(sft_marker, round_index)

                    self.store.set_phase(round_index, PHASE_FINALIZE_ROUND)
                    continue

                raise RuntimeError(f"unknown adaptive phase: {phase!r}")
        finally:
            if self._active_module:
                self._active_module.kill()
                self._active_module = None

        logger.info("=" * 60)
        logger.info("ADAPTIVE PIPELINE COMPLETE")
        logger.info("Final model: %s", final_model)
        logger.info("=" * 60)

    def _build_adaptive_phases(
        self, round_index: int, iter_dir: Path, iter_config: dict, current_model: str
    ) -> list[Phase]:
        phases = super()._build_phases(round_index, iter_dir, iter_config, current_model)
        adapted: list[Phase] = []
        for module in phases:
            if isinstance(module, SlimeWrapper):
                adapted.append(
                    AdaptiveSlimeWrapper(
                        config=module.config,
                        input_paths=module.input_paths,
                        output_paths=module.output_paths,
                    )
                )
            elif isinstance(module, LFWrapper):
                adapted.append(
                    AdaptiveLFWrapper(
                        config=module.config,
                        input_paths=module.input_paths,
                        output_paths=module.output_paths,
                    )
                )
            else:
                adapted.append(module)
        return adapted

    def _adaptive_iter_config(self, round_index: int, state: dict) -> dict:
        # Deliberately ignore the legacy pipeline's iteration_overrides: those
        # encode fixed-round behavior. Adaptive runs have their own optional
        # namespace for explicit round-specific experiments.
        iter_config = copy.deepcopy(self.adaptive_round_overrides.get(round_index, {}))
        rl_override = iter_config.get("rl", {})
        if rl_override is None or (
            isinstance(rl_override, dict) and rl_override.get("skip") is True
        ):
            raise ValueError("adaptive rounds cannot skip RL")
        rl_override = copy.deepcopy(rl_override or {})

        prior_steps = sum(
            int(steps)
            for key, steps in (state.get("steps_by_round") or {}).items()
            if int(key) != round_index
        )
        round_horizon = self.adaptive_config["total_rl_steps"] - prior_steps
        current_step = int((state.get("steps_by_round") or {}).get(str(round_index), 0))
        if round_horizon <= current_step:
            raise RuntimeError(
                f"no RL budget remains for round {round_index}: "
                f"current_step={current_step}, horizon={round_horizon}"
            )
        rl_override["training_steps"] = round_horizon

        slime = copy.deepcopy(rl_override.get("slime") or {})
        slime["eval_interval"] = self.adaptive_config["eval_every_steps"]
        slime["save_interval"] = self.adaptive_config["eval_every_steps"]
        slime["train_script"] = "graphrl/slime/train_adaptive.py"
        rl_override["slime"] = slime
        iter_config["rl"] = rl_override

        logger.info(
            "[adaptive] round %d local backend horizon=%d; prior completed "
            "RL steps=%d",
            round_index,
            round_horizon,
            prior_steps,
        )
        return iter_config

    def _run_or_resume(self, module):
        if module.is_already_complete():
            logger.info("[%s] Already complete, resuming after phase", module.name)
            return module.get_output()
        return self._run_module(module)

    def _ensure_rl_output(self, phases: list[Phase]) -> None:
        rl_module = self._one_phase(phases, AdaptiveSlimeWrapper)
        if not rl_module.is_already_complete():
            raise RuntimeError(
                "adaptive state advanced past RL, but its round-best model "
                "cannot be recovered"
            )

    @staticmethod
    def _one_phase(phases: list[Phase], phase_type):
        matches = [module for module in phases if isinstance(module, phase_type)]
        if len(matches) != 1:
            raise RuntimeError(
                f"adaptive pipeline expected exactly one {phase_type.__name__}; "
                f"found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _require_decision_round(state: dict, round_index: int) -> None:
        if int(state.get("decision_round", -1)) != round_index:
            raise RuntimeError(
                f"adaptive decision belongs to round {state.get('decision_round')}, "
                f"not current round {round_index}"
            )

    def _upstream_model(self, round_index: int) -> str:
        if round_index == 0:
            if not self.initial_model:
                raise ValueError("adaptive round 0 requires initial_model_path")
            return str(self.initial_model)
        previous = self.experiment_dir / f"iter_{round_index - 1:03d}" / "sft" / "sft_model"
        if not is_complete_hf_model(previous):
            raise RuntimeError(f"previous round SFT model is incomplete: {previous}")
        return str(previous)

    def _phase_input_model(self, round_index: int, phase: str, state: dict) -> str:
        current_rl = (
            self.experiment_dir / f"iter_{round_index:03d}" / "rl" / "rl_model"
        )
        terminal_rl = (
            int(state.get("decision_round", -1)) == round_index
            and state.get("decision") == DECISION_SWITCH_TO_SFT
            and state.get("decision_ready")
        )
        if (phase != PHASE_RL or terminal_rl) and is_complete_hf_model(current_rl):
            return str(current_rl)
        return self._upstream_model(round_index)

    def _finish_round(self, round_index: int, iter_dir: Path, iter_config: dict) -> None:
        upload_list = self._iter_value(iter_config, "upload_to_hf", DEFAULT_UPLOAD_TO_HF) or []
        self._maybe_upload_to_hf(round_index, iter_dir, upload_list)
        cleanup_iter(
            iter_num=round_index,
            experiment_dir=self.experiment_dir,
            delete_on_sft_model=(
                self._iter_value(iter_config, "delete_on_sft_model", DEFAULT_DELETE_ON_SFT_MODEL)
                or []
            ),
            delete_on_next_rl_model=(
                self._iter_value(
                    iter_config,
                    "delete_on_next_rl_model",
                    DEFAULT_DELETE_ON_NEXT_RL_MODEL,
                )
                or []
            ),
        )
        process_pending_deletes(self.experiment_dir)

    @staticmethod
    def _traj_output_is_complete(module: TrajToSFTModule) -> bool:
        info_path = Path(module.paths.sft_data) / "dataset_info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(info, dict) or not info:
            return False

        for entry in info.values():
            if not isinstance(entry, dict) or not entry.get("file_name"):
                return False
            data_path = info_path.parent / entry["file_name"]
            try:
                records = json.loads(data_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if not isinstance(records, list) or not records:
                return False

        atomize = (module.config.get("graph_builder") or {}).get("atomize") or {}
        if atomize.get("enabled"):
            atomize_state_path = (
                Path(module.paths.sft_data).parent
                / "graph"
                / ATOMIZE_STATE_FILENAME
            )
            try:
                atomize_state = json.loads(
                    atomize_state_path.read_text(encoding="utf-8")
                )
                multi_edges = int(atomize_state.get("multi_edges", 0))
                rendered = int(atomize_state.get("rendered", 0))
                dropped = int(atomize_state.get("dropped", 0))
                leftover = int(atomize_state.get("leftover_removed", 0))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return False
            if dropped or leftover or (multi_edges and not rendered):
                return False
        return True

    @staticmethod
    def _clear_traj_output(module: TrajToSFTModule) -> None:
        traj_root = Path(module.paths.sft_data).parent
        shutil.rmtree(traj_root, ignore_errors=True)

    def _clear_sft_output(self, phases: list[Phase]) -> None:
        module = self._one_phase(phases, AdaptiveLFWrapper)
        paths = {
            Path(module.output_paths["base_dir"]),
            Path(module.output_paths["model"]),
        }
        removed = False
        for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
                removed = True
            elif path.is_dir():
                shutil.rmtree(path)
                removed = True
        if removed:
            logger.warning(
                "[adaptive] removed stale SFT output because its TrajToSFT "
                "dataset is being rebuilt"
            )

    def _recover_or_guard_state(self) -> None:
        legacy_state = self.experiment_dir / "rl_schedule_state.json"
        if not self.store.path.exists() and legacy_state.exists():
            raise RuntimeError(
                f"refusing to start adaptive control in a legacy experiment: {legacy_state}"
            )
        root_state = self.store.load() if self.store.path.exists() else None
        current_round = None if root_state is None else int(root_state["round"])
        current_round_step = 0
        max_resume_step = None
        rollback_for_rollouts = False
        if root_state is not None and root_state.get("phase") in {
            PHASE_RL,
            PHASE_TRAJ_TO_SFT,
            PHASE_SFT,
        }:
            expected_step = int(
                (root_state.get("steps_by_round") or {}).get(
                    str(current_round), root_state.get("decision_step") or 0
                )
            )
            current_round_step = expected_step
            required_resume_step = required_rollout_prefix(
                root_state, expected_step
            )
            if required_resume_step:
                rollout_dir = (
                    self.experiment_dir
                    / f"iter_{current_round:03d}"
                    / "rl"
                    / "rollout_data"
                )
                max_resume_step = durable_rollout_prefix(rollout_dir)
            else:
                max_resume_step = 0
            rollback_for_rollouts = max_resume_step < required_resume_step
            if rollback_for_rollouts:
                logger.warning(
                    "[adaptive] controller/checkpoint step %d exceeds durable "
                    "rollout prefix %d; rolling back to a consistent checkpoint",
                    required_resume_step,
                    max_resume_step,
                )

        if (
            root_state is not None
            and root_state.get("phase") != PHASE_RL
            and not rollback_for_rollouts
        ):
            return

        if current_round is None:
            candidates = list(
                self.experiment_dir.glob(
                    "iter_*/rl/slime_checkpoints/iter_*/adaptive_schedule_state.json"
                )
            )
            candidates.extend(
                self.experiment_dir.glob(
                    "adaptive_best_snapshots/*/adaptive_schedule_state.json"
                )
            )
        else:
            candidates = list(
                self.experiment_dir.glob(
                    f"iter_{current_round:03d}/rl/slime_checkpoints/"
                    "iter_*/adaptive_schedule_state.json"
                )
            )
            candidates.extend(
                self.experiment_dir.glob(
                    f"adaptive_best_snapshots/round_{current_round:03d}_*/"
                    "adaptive_schedule_state.json"
                )
            )
        valid: list[tuple[tuple[int, int, int], Path]] = []
        for path in candidates:
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if state.get("config_fingerprint") != self.store.config_fingerprint:
                    continue
                if current_round is not None and int(state.get("round", -1)) != current_round:
                    continue
                if "adaptive_best_snapshots" in path.parts:
                    candidate_round = int(state.get("round", 0))
                    candidate_step = int(
                        (state.get("steps_by_round") or {}).get(
                            str(candidate_round), state.get("decision_step") or 0
                        )
                    )
                    # A best snapshot intentionally contains only HF model weights
                    # and controller metadata. Once RL has taken a train step it is
                    # not an exact resume point: optimizer, scheduler, dataloader and
                    # RNG state exist only in a regular SLIME checkpoint.
                    if state.get("phase") == PHASE_RL and candidate_step > 0:
                        continue
                    if not is_complete_snapshot(path.parent):
                        continue
                if "slime_checkpoints" in path.parts:
                    checkpoint = path.parent
                    checkpoint_step = int(checkpoint.name.rsplit("_", 1)[-1]) + 1
                    checkpoint_round = int(checkpoint.parents[2].name[5:])
                    checkpoint_rollouts = (
                        self.experiment_dir
                        / f"iter_{checkpoint_round:03d}"
                        / "rl"
                        / "rollout_data"
                    )
                    checkpoint_required_rollouts = required_rollout_prefix(
                        state, checkpoint_step
                    )
                    if checkpoint_required_rollouts:
                        checkpoint_rollout_prefix = durable_rollout_prefix(
                            checkpoint_rollouts
                        )
                        if checkpoint_required_rollouts > checkpoint_rollout_prefix:
                            continue
                    manifest = checkpoint / CHECKPOINT_MANIFEST
                    if not manifest.exists() or not is_complete_checkpoint_manifest(checkpoint):
                        continue
                key = (
                    int(state.get("total_rl_steps", 0)),
                    int(state.get("round", 0)),
                    int(state.get("decision_step") or 0),
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            valid.append((key, path.parent))
        if valid:
            _, checkpoint_dir = max(valid, key=lambda item: item[0])
            restored = self.store.restore_from(checkpoint_dir)
            recovered_round = int(restored.get("round", 0))
            recovered_step = int(
                (restored.get("steps_by_round") or {}).get(
                    str(recovered_round), restored.get("decision_step") or 0
                )
            )
            if "slime_checkpoints" in checkpoint_dir.parts:
                checkpoint_root = checkpoint_dir.parent
                rollout_id = int(checkpoint_dir.name.rsplit("_", 1)[-1])
                tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
                temporary = tracker.with_name(f".{tracker.name}.tmp")
                temporary.write_text(str(rollout_id), encoding="utf-8")
                os.replace(temporary, tracker)
            else:
                # A step-0 model snapshot is not a full optimizer/RNG resume
                # point. Ensure SLIME starts the round cold from that model.
                tracker = (
                    self.experiment_dir
                    / f"iter_{recovered_round:03d}"
                    / "rl"
                    / "slime_checkpoints"
                    / "latest_checkpointed_iteration.txt"
                )
                tracker.unlink(missing_ok=True)
                rollout_id = -1
            self._prune_slime_checkpoints_after(recovered_round, rollout_id)
            if root_state is None:
                logger.warning(
                    "[adaptive] recovered missing root state from %s", checkpoint_dir
                )
            elif restored != root_state:
                logger.warning(
                    "[adaptive] reconciled root state to committed checkpoint %s",
                    checkpoint_dir,
                )
            if root_state is None or restored != root_state or rollback_for_rollouts:
                self._clear_round_outputs_after_rl_rollback(recovered_round)
                self._prune_rollouts_after(recovered_round, recovered_step)
            return

        if root_state is None and any(self.experiment_dir.glob("iter_*")):
            raise RuntimeError(
                "adaptive state is missing but iteration directories already exist; "
                "use a new experiment directory or restore its adaptive state"
            )
        if rollback_for_rollouts:
            raise RuntimeError(
                "adaptive state is ahead of every checkpoint with a complete "
                "rollout prefix; no safe resume point exists"
            )
        if (
            root_state is not None
            and root_state.get("phase") == PHASE_RL
            and current_round_step > 0
        ):
            raise RuntimeError(
                "adaptive RL state is ahead of step 0, but no complete regular "
                "checkpoint with model, optimizer, scheduler, RNG and controller "
                "state exists; a best-model snapshot is model-only and cannot be "
                "used for exact resume"
            )

    def _write_schedule_config(self) -> None:
        path = self.experiment_dir / CONFIG_FILENAME
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(
            json.dumps(self.adaptive_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)

    @staticmethod
    def _require_complete_rollouts(
        module: TrajToSFTModule, state: dict, round_index: int
    ) -> None:
        expected = int(
            (state.get("steps_by_round") or {}).get(
                str(round_index), state.get("decision_step") or 0
            )
        )
        actual = durable_rollout_prefix(module.paths.rollout_data)
        if actual < expected:
            raise RuntimeError(
                "adaptive TrajToSFT requires a complete rollout prefix: "
                f"round={round_index}, expected_through={expected}, durable_through={actual}"
            )

    def _clear_round_outputs_after_rl_rollback(self, round_index: int) -> None:
        iter_dir = self.experiment_dir / f"iter_{round_index:03d}"
        for path in (
            iter_dir / "rl" / ".slime_rl_done.json",
            iter_dir / "rl" / "rl_model",
            iter_dir / "traj_to_sft",
            iter_dir / "sft",
        ):
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)

    def _prune_rollouts_after(self, round_index: int, step: int) -> None:
        rollout_dir = (
            self.experiment_dir
            / f"iter_{round_index:03d}"
            / "rl"
            / "rollout_data"
        )
        for path in rollout_dir.glob("*"):
            name = path.name
            suffix = name[6:] if name.startswith("image_") else path.stem
            try:
                rollout_step = int(suffix)
            except ValueError:
                continue
            if rollout_step <= step:
                continue
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)

    def _prune_slime_checkpoints_after(
        self, round_index: int, rollout_id: int
    ) -> None:
        rl_dir = self.experiment_dir / f"iter_{round_index:03d}" / "rl"
        checkpoint_root = rl_dir / "slime_checkpoints"
        for base in (checkpoint_root, checkpoint_root / "critic"):
            for path in base.glob("iter_*"):
                try:
                    candidate = int(path.name.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                if candidate > rollout_id:
                    shutil.rmtree(path, ignore_errors=True)
        for path in (checkpoint_root / "rollout").glob(
            "global_dataset_state_dict_*.pt"
        ):
            try:
                candidate = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if candidate > rollout_id:
                path.unlink(missing_ok=True)
        for path in (rl_dir / "hf_checkpoints").glob("rollout_*"):
            try:
                candidate = int(path.name.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if candidate > rollout_id:
                shutil.rmtree(path, ignore_errors=True)

    def _global_best_model(self) -> str | None:
        path = self.experiment_dir / "best_val_run" / "actor" / "huggingface"
        if not is_complete_hf_model(path):
            state = self.store.load()
            relative = (state.get("run_best") or {}).get("checkpoint")
            if relative:
                snapshot = self.experiment_dir / relative
                if is_complete_snapshot(snapshot):
                    link = self.experiment_dir / "best_val_run"
                    if link.exists() or link.is_symlink():
                        if link.is_dir() and not link.is_symlink():
                            shutil.rmtree(link)
                        else:
                            link.unlink()
                    os.symlink(os.path.relpath(snapshot, link.parent), link)
        if not is_complete_hf_model(path):
            raise RuntimeError(f"adaptive run has no complete global-best model: {path}")
        return str(path)

    @staticmethod
    def _write_sft_marker(path: Path, round_index: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(
            json.dumps({"round": round_index, "complete": True}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def _garbage_collect_best_snapshots(self) -> None:
        root = self.experiment_dir / "adaptive_best_snapshots"
        if not root.is_dir():
            return
        keep = set()
        for link in [self.experiment_dir / "best_val_run", *self.experiment_dir.glob("iter_*/rl/best_val")]:
            if link.is_symlink():
                try:
                    keep.add(link.resolve())
                except OSError:
                    pass
        for snapshot in root.iterdir():
            if snapshot.is_dir() and snapshot.resolve() not in keep:
                shutil.rmtree(snapshot, ignore_errors=True)


@hydra.main(config_path="configs", config_name="pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    config_dict: dict = OmegaConf.to_container(cfg, resolve=True)
    AdaptiveGraphRLController(config_dict).run()


if __name__ == "__main__":
    main()
