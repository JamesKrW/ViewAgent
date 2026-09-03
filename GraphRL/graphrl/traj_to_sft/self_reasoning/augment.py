"""High-level reasoning-augmentation orchestrator.

In-process equivalent of ``view_suite/self_reasoner/main.py`` — but instead
of being a CLI tool, this exposes :func:`augment_sft_json` for the
TrajToSFT phase to call directly.

For each call we:

  1. Build a VAGEN-SLIME evaluation config in memory (one env entry, one episode
     per record, seed = record index).
  2. Register :class:`ReasoningEnv` through the ViewSuite-to-SLIME adapter under a
     unique name so vagen can instantiate it.
  3. Retry unsuccessful prior results, then run the shared SLIME evaluator.
  4. Export its records to the historical ``tag_<tag_id>`` view consumed by
     the existing reasoning postprocessor.

The sglang server is the caller's responsibility — passing ``base_url``
to point at an already-running server. The :class:`SGLangServer` context
manager is provided for callers that want to launch one.
"""
from __future__ import annotations

import json
import logging
import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .env import ReasoningEnv
from .postprocess import build_sft, collect_augmented

logger = logging.getLogger(__name__)

_REGISTERED = False


def _ensure_registered(env_name: str, env_cls: type = None) -> None:
    """Register an env class with VAGEN's registry under ``env_name``.

    Defaults to :class:`ReasoningEnv` for backward-compat with the
    single-turn flow; pass ``env_cls`` to register a different class
    (e.g. :class:`MCQReasoningEnv`) under a distinct name. Reasoners
    that need a custom env should always pass their own ``env_cls`` —
    the registry is a process-global mapping, so re-registration is
    fine but the LAST class registered under a name wins.
    """
    global _REGISTERED
    from vagen.envs import register_env
    if env_cls is None:
        env_cls = ReasoningEnv
    # Registered as-is: ReasoningEnv and MCQReasoningEnv subclass GymImageEnv and
    # already speak VAGEN's GymBaseEnv contract. The wrapper this used to go
    # through existed only to reshape them for the SLIME backend.
    register_env(env_name, env_cls)
    _REGISTERED = True


# Public alias so reasoner subclasses (e.g. single_turn) can reuse the
# registration without poking at a private name.
ensure_registered = _ensure_registered


def run_vagen_eval_and_collect(
    *,
    sft_path: Path,
    image_root: Path,
    dump_dir: Path,
    tag_id: str,
    base_url: str,
    model_name: str,
    system_prompt_path: Path,
    user_prompt_path: Path,
    image_size: Optional[List[int]] = None,
    max_turns: int = 3,
    max_concurrent_jobs: int = 16,
    max_retries: int = 6,
    chat_config: Optional[Dict[str, Any]] = None,
    salvage_partial: bool = True,
    n_records: Optional[int] = None,
    resume: bool = True,
    env_name: str = "GraphRLReasoningEnv",
    checker_cls: Optional[str] = None,
    checker_kwargs: Optional[Dict[str, Any]] = None,
    dataset_cls: Optional[str] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    num_workers: int = 1,
    env_cls: Optional[type] = None,
    extra_env_config: Optional[Dict[str, Any]] = None,
):
    """Run vagen ``run_eval`` against ``sft_path`` and return the per-seed
    augmentation outcome (``seed → list[Optional[body]]``). The caller is
    responsible for turning that into a final SFT JSON — see
    :func:`augment_sft_json` for the default 1:1 path and
    :mod:`graphrl.traj_to_sft.self_reasoning.single_turn` for the
    explode-and-reassemble path.

    All arguments mirror :func:`augment_sft_json` except the output-shape
    args (``output_path``, ``keep_unaugmented``, suffixes) which live in
    the build-SFT step, not this one.

    ``dataset_cls`` / ``dataset_kwargs`` (optional) override the default
    :class:`ShareGPTDataset` that :class:`ReasoningEnv` uses. Single-turn
    needs this to expose its custom ``assistant_texts`` extraction.
    """
    sft_path = Path(sft_path)
    image_root = Path(image_root)
    dump_dir = Path(dump_dir)
    system_prompt_path = Path(system_prompt_path)
    user_prompt_path = Path(user_prompt_path)

    with open(sft_path, encoding="utf-8") as f:
        n_total = len(json.load(f))
    n_envs = int(n_records) if n_records is not None else n_total
    assert 0 < n_envs <= n_total, f"n_records={n_envs} must be in (0, {n_total}]"

    env_cfg: Dict[str, Any] = {
        "sft_path": str(sft_path),
        "image_root": str(image_root),
        "image_size": image_size,
        "system_prompt_path": str(system_prompt_path),
        "user_prompt_path": str(user_prompt_path),
    }
    if checker_cls is not None:
        env_cfg["checker_cls"] = checker_cls
    if checker_kwargs is not None:
        env_cfg["checker_kwargs"] = checker_kwargs
    if dataset_cls is not None:
        env_cfg["dataset_cls"] = dataset_cls
    if dataset_kwargs is not None:
        env_cfg["dataset_kwargs"] = dataset_kwargs
    if extra_env_config:
        env_cfg.update(extra_env_config)

    vagen_cfg: Dict[str, Any] = {
        "default_chat_config": chat_config or {"temperature": 0.0, "max_tokens": 2048},
        "envs": [
            {
                "name": env_name,
                "n_envs": n_envs,
                "tag_id": tag_id,
                "seed_list": list(range(n_envs)),
                "split": "train",
                "max_turns": int(max_turns),
                "concat_multi_turn": True,
                "config": env_cfg,
            }
        ],
        "experiment": {"dump_dir": str(dump_dir), "default_max_turns": int(max_turns)},
        "run": {
            "backend": "sglang",
            "resume": "skip_completed" if resume else "off",
            "live_summary": True,
            "max_concurrent_jobs": int(max_concurrent_jobs),
            "base_seed": 0,
            # ``num_workers > 1`` triggers VAGEN's opt-in multi-process
            # rollout. Workers each run their own asyncio loop against
            # the shared sglang base_url; per-worker concurrency stays
            # at ``max_concurrent_jobs``. See VAGEN's
            # ``run_eval._run_jobs_multiprocess`` for details.
            "num_workers": int(num_workers),
            # NORMAL_FINISH_REASONS is monkey-patched in the parent below;
            # threading the value here lets VAGEN propagate it into spawned
            # workers (otherwise child processes would default to
            # ``{"done", "max_turns"}`` and incorrectly mark max_turns as
            # complete on resume).
            "normal_finish_reasons": ["done"],
        },
        "backends": {
            "sglang": {
                "base_url": base_url,
                "api_key": "EMPTY",
                "model": model_name,
                "max_concurrency": int(max_concurrent_jobs),
                "max_retries": int(max_retries),
            }
        },
    }

    dump_dir.mkdir(parents=True, exist_ok=True)
    vagen_yaml = dump_dir / "_vagen_config.yaml"
    with open(vagen_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(vagen_cfg, f, sort_keys=False)

    _ensure_registered(env_name, env_cls=env_cls)

    # ★ Not ported to the VAGEN/verl backend. VAGEN's evaluation package exposes
    # `run_eval_parallel(...)`, whose signature and result layout differ from the
    # SLIME runner this was written against, so the call below is not a rename.
    # Nothing in the shipped pipelines reaches here -- self-reasoning is an opt-in
    # TrajToSFT surface and none of the three ViewSuite experiments enable it --
    # so this raises where the gap is instead of failing as a bare ImportError
    # several frames deeper.
    try:
        from vagen.evaluation.runner import run_evaluation
    except ImportError as exc:
        raise NotImplementedError(
            "self-reasoning augmentation is not available on the VAGEN/verl backend: "
            "vagen.evaluation.runner exposes run_eval_parallel, not run_evaluation. "
            "Port run_vagen_eval_and_collect() to that API to re-enable it."
        ) from exc
    from view_suite.evaluation.config import load_legacy_config
    from view_suite.evaluation.run_eval import _export_legacy_layout

    config = load_legacy_config(vagen_yaml)
    # Reasoning attempts that exhausted max_turns are intentionally retried.
    model_root = config.output_dir / config.experiment_id / config.models[0].name
    for result_path in model_root.glob("tag_*/seed_*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        if not result.get("success"):
            shutil.rmtree(result_path.parent, ignore_errors=True)
    asyncio.run(run_evaluation(config))
    _export_legacy_layout(config)

    return collect_augmented(
        dump_dir / f"tag_{tag_id}", salvage_partial=salvage_partial,
    )


def augment_sft_json(
    *,
    sft_path: Path,
    image_root: Path,
    output_path: Path,
    dump_dir: Path,
    tag_id: str,
    base_url: str,
    model_name: str,
    system_prompt_path: Path,
    user_prompt_path: Path,
    image_size: Optional[List[int]] = None,
    max_turns: int = 3,
    max_concurrent_jobs: int = 16,
    max_retries: int = 6,
    chat_config: Optional[Dict[str, Any]] = None,
    keep_unaugmented: bool = False,
    salvage_partial: bool = True,
    n_records: Optional[int] = None,
    resume: bool = True,
    env_name: str = "GraphRLReasoningEnv",
    checker_cls: Optional[str] = None,
    checker_kwargs: Optional[Dict[str, Any]] = None,
    augmented_system_prompt_suffix: Optional[str] = None,
    raw_system_prompt_suffix: Optional[str] = None,
    num_workers: int = 1,
) -> int:
    """Augment one SFT JSON in-process and write the result to ``output_path``.

    Returns the number of records written. ``output_path`` may equal
    ``sft_path`` to overwrite in place.

    Inputs:
      sft_path          path to the input ShareGPT SFT JSON.
      image_root        dir that the records' relative ``images: [...]`` paths resolve against.
      output_path       where to write the reasoning-augmented JSON.
      dump_dir          per-call vagen rollout dump dir; resume-safe across calls.
      tag_id            run_eval groups dumps under ``dump_dir/tag_<tag_id>``.
      base_url          OpenAI-compatible endpoint, e.g. http://127.0.0.1:30000/v1.
      model_name        ``model`` field passed to the chat API (HF id or local path).
      system_prompt_path / user_prompt_path
                        ``.md`` files baked into ReasoningEnv.
      image_size        optional [w, h] resize.
      max_turns         1 → pure one-shot; N → one-shot + (N-1) refines.
      max_concurrent_jobs / max_retries / chat_config / resume
                        run_eval / sglang knobs.
      keep_unaugmented  keep records whose rollout never produced a valid augmentation.
      salvage_partial   for records that hit max_turns without all turns passing,
                        write the per-turn augmented body where the checker passed
                        and put the original assistant content back for turns that
                        didn't. ``True`` is strongly recommended.
      n_records         truncate the dataset (smoke tests). ``None`` = all records.
      env_name          VAGEN registry name. Must match across the run.
      checker_cls / checker_kwargs
                        override the default ObsActionChecker via FQCN.
    """
    augmented = run_vagen_eval_and_collect(
        sft_path=sft_path,
        image_root=image_root,
        dump_dir=dump_dir,
        tag_id=tag_id,
        base_url=base_url,
        model_name=model_name,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        image_size=image_size,
        max_turns=max_turns,
        max_concurrent_jobs=max_concurrent_jobs,
        max_retries=max_retries,
        chat_config=chat_config,
        salvage_partial=salvage_partial,
        n_records=n_records,
        resume=resume,
        env_name=env_name,
        checker_cls=checker_cls,
        checker_kwargs=checker_kwargs,
        num_workers=num_workers,
    )
    n = build_sft(
        Path(sft_path), augmented, Path(output_path),
        keep_unaugmented=keep_unaugmented,
        augmented_system_prompt_suffix=augmented_system_prompt_suffix,
        raw_system_prompt_suffix=raw_system_prompt_suffix,
    )
    logger.info("augment_sft_json: %d augmented records → %s", n, output_path)
    return n
