"""Composable evaluation over VAGEN environments, harnesses, and OAI model APIs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vagen_agent.envs import build_env, get_env_cls
from vagen_agent.evaluation.backends import build_backend
from vagen_agent.evaluation.config import (
    EnvironmentSpec,
    EvaluationConfig,
    ModelSpec,
    load_config,
)
from vagen_agent.evaluation.recording import (
    EpisodeRecorder,
    EpisodeStore,
    EvaluationClient,
    RecordingEnv,
    atomic_json,
    json_safe,
)
from vagen_agent.harness import RoleClients, build_harness, resolve_harness

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationJob:
    model: ModelSpec
    environment: EnvironmentSpec
    seed: int


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root
        )
        dirty = bool(status.strip())
        fingerprint = None
        if dirty:
            digest = hashlib.sha256()
            digest.update(subprocess.check_output(
                ["git", "diff", "--binary", "HEAD", "--"], cwd=root
            ))
            for raw in sorted(
                subprocess.check_output(
                    ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
                ).split(b"\0")
            ):
                if not raw:
                    continue
                path = root / os.fsdecode(raw)
                digest.update(raw + b"\0")
                if path.is_file():
                    digest.update(path.read_bytes())
            fingerprint = digest.hexdigest()
        return {"commit": commit, "dirty": dirty, "dirty_fingerprint": fingerprint}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "dirty_fingerprint": None}


def _token_limit(
    model: ModelSpec, environment: EnvironmentSpec, *, role: str = "default"
) -> int | None:
    if environment.response_length_per_turn is not None:
        return environment.response_length_per_turn
    leaf = model.for_role(role)
    sampling = {**leaf.sampling, **environment.sampling}
    value = sampling.get(leaf.token_limit_field)
    return int(value) if value is not None else None


class EvaluationRunner:
    def __init__(
        self,
        config: EvaluationConfig,
        *,
        backend_factory=None,
        models: set[str] | None = None,
        tags: set[str] | None = None,
        seeds: set[int] | None = None,
    ) -> None:
        self.config = config
        self.root = config.output_dir / config.experiment_id
        self.store = EpisodeStore(self.root, record_images=config.record_images)
        factory = backend_factory or build_backend
        self.backends = {
            model.name: {
                role: factory(model.for_role(role))
                for role in model.available_roles
            }
            for model in config.models
        }
        self._episode_gate = asyncio.Semaphore(config.max_concurrent_episodes)
        self._summary_lock = asyncio.Lock()
        self._model_filter = models
        self._tag_filter = tags
        self._seed_filter = seeds

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "experiment_id": self.config.experiment_id,
            "config_sha256": self.config.fingerprint,
            "config_path": str(self.config.path),
            "config": self.config.public_config,
            "source": _git_state(Path(__file__).resolve().parents[2]),
        }
        path = self.root / "evaluation_manifest.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != manifest:
                raise ValueError(
                    f"evaluation manifest changed in {self.root}; use a new experiment.id"
                )
        else:
            atomic_json(path, manifest)

    def jobs(self) -> list[EvaluationJob]:
        return [
            EvaluationJob(model=model, environment=environment, seed=seed)
            for model in self.config.models
            for environment in self.config.environments
            for seed in environment.seeds
            if (self._model_filter is None or model.name in self._model_filter)
            and (self._tag_filter is None or environment.tag in self._tag_filter)
            and (self._seed_filter is None or seed in self._seed_filter)
        ]

    async def _run_job(self, job: EvaluationJob) -> dict[str, Any]:
        async with self._episode_gate:
            spec = job.environment
            recorder = EpisodeRecorder()
            started = time.time()
            adapted = None
            status = "completed"
            finish_reason = "harness_returned"
            error: dict[str, str] | None = None
            try:
                harness_cls = resolve_harness(spec.harness)
                harness = build_harness(
                    spec.harness,
                    window=spec.context_window,
                    response_limit=_token_limit(
                        job.model,
                        spec,
                        role=harness_cls.action_model_role,
                    ),
                    **spec.harness_config,
                )
                env_cls = get_env_cls(spec.name)
                adapted = build_env(
                    env_cls,
                    spec.config,
                    max_turns=spec.max_turns,
                    required_type=harness.environment_type,
                )
                environment = RecordingEnv(adapted, recorder, seed=job.seed)
                clients = {}
                for role in harness.model_roles:
                    leaf = job.model.for_role(role)
                    clients[role] = EvaluationClient(
                        self.backends[job.model.name][role],
                        recorder,
                        role=role,
                        sampling={**leaf.sampling, **spec.sampling},
                        thinking_token_budget=(
                            spec.thinking_token_budget
                            if spec.thinking_token_budget is not None
                            else leaf.thinking_token_budget
                        ),
                        max_calls=max(16, spec.max_turns * 4),
                    )
                client = (
                    clients["default"]
                    if harness.model_roles == ("default",)
                    else RoleClients(clients)
                )
                await harness.run_episode(client, environment)
                if recorder.truncated:
                    finish_reason = "max_turns"
                elif recorder.terminated:
                    finish_reason = "done"
            except Exception as exc:  # one failed episode must not cancel the matrix
                LOGGER.exception(
                    "evaluation failed model=%s tag=%s seed=%s",
                    job.model.name,
                    spec.tag,
                    job.seed,
                )
                status = "error"
                finish_reason = "error"
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "repr": repr(exc),
                }
            finally:
                if adapted is not None:
                    try:
                        await adapted.close()
                    except Exception as exc:  # noqa: BLE001 - cleanup failure is reported
                        if error is None:
                            status = "error"
                            finish_reason = "error"
                            error = {
                                "type": type(exc).__name__,
                                "message": f"environment close failed: {exc}",
                                "repr": repr(exc),
                            }

            env_metrics = dict(adapted.last_metrics) if adapted is not None else {}
            if adapted is not None:
                env_metrics.setdefault("success", float(adapted.success))
            format_compliance = env_metrics.get("format_compliance")
            result = {
                "schema_version": 1,
                "job_id": f"{job.model.name}/{spec.tag}/{job.seed}",
                "config_sha256": self.config.fingerprint,
                "experiment_id": self.config.experiment_id,
                "model": job.model.name,
                "served_model": job.model.served_models,
                "model_roles": sorted(job.model.available_roles),
                "environment": spec.name,
                "tag": spec.tag,
                "harness": spec.harness,
                "seed": job.seed,
                "status": status,
                "finish_reason": finish_reason,
                "error": error,
                "steps": len(recorder.rewards),
                "return": float(sum(recorder.rewards)),
                "success": bool(adapted.success) if adapted is not None else False,
                "metrics": env_metrics,
                "format_compliance": format_compliance,
                "model_calls": len(recorder.calls),
                "length_limited_calls": sum(
                    call.get("response", {}).get("finish_reason") == "length"
                    for call in recorder.calls
                ),
                "empty_final_content_calls": sum(
                    not call.get("response", {}).get("content")
                    for call in recorder.calls
                ),
                "started_at": started,
                "finished_at": time.time(),
            }
            self.store.write(result, recorder)
            await self._write_summaries()
            return result

    def _results(self) -> list[dict[str, Any]]:
        values = []
        for path in self.root.glob("*/tag_*/seed_*/result.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("config_sha256") == self.config.fingerprint:
                values.append(value)
        return values

    async def _write_summaries(self) -> dict[str, Any]:
        async with self._summary_lock:
            values = self._results()
            tasks: dict[str, dict[str, Any]] = {}
            for model in self.config.models:
                for spec in self.config.environments:
                    matching = [
                        result for result in values
                        if result.get("model") == model.name and result.get("tag") == spec.tag
                    ]
                    completed = [result for result in matching if result.get("status") == "completed"]
                    format_values = [
                        float(value["format_compliance"])
                        for value in completed if value.get("format_compliance") is not None
                    ]
                    metric_values: dict[str, list[float]] = {}
                    for value in completed:
                        for name, metric in (value.get("metrics") or {}).items():
                            if isinstance(metric, (int, float, bool)):
                                metric_values.setdefault(name, []).append(float(metric))
                    mean_metrics = {
                        name: sum(items) / len(items)
                        for name, items in metric_values.items()
                        if items
                    }
                    key = f"{model.name}/{spec.tag}"
                    tasks[key] = {
                        "model": model.name,
                        "environment": spec.name,
                        "tag": spec.tag,
                        "harness": spec.harness,
                        "jobs": len(spec.seeds),
                        "completed": len(completed),
                        "errors": sum(value.get("status") == "error" for value in matching),
                        "mean_return": (
                            sum(float(value.get("return", 0.0)) for value in completed)
                            / len(completed) if completed else None
                        ),
                        "success_rate": (
                            sum(bool(value.get("success")) for value in completed) / len(completed)
                            if completed else None
                        ),
                        "format_compliance": (
                            sum(format_values) / len(format_values) if format_values else None
                        ),
                        "metrics": mean_metrics,
                        "length_limited_calls": sum(
                            int(value.get("length_limited_calls", 0)) for value in completed
                        ),
                        "empty_final_content_calls": sum(
                            int(value.get("empty_final_content_calls", 0)) for value in completed
                        ),
                    }
            summary = {
                "experiment_id": self.config.experiment_id,
                "config_sha256": self.config.fingerprint,
                "scheduled": sum(len(spec.seeds) for spec in self.config.environments)
                * len(self.config.models),
                "recorded": len(values),
                "tasks": tasks,
                "updated_at": time.time(),
            }
            atomic_json(self.root / "summary.json", summary)
            return summary

    async def run(self) -> dict[str, Any]:
        try:
            self.prepare()
            selected = self.jobs()
            if not selected:
                raise ValueError("evaluation filters selected no jobs")
            pending = []
            for job in selected:
                if self.config.resume == "skip_completed" and self.store.is_completed(
                    job.model.name, job.environment.tag, job.seed, self.config.fingerprint
                ):
                    continue
                pending.append(job)
            LOGGER.info(
                "evaluation %s: %d selected, %d pending, concurrency=%d",
                self.config.experiment_id,
                len(selected),
                len(pending),
                self.config.max_concurrent_episodes,
            )
            await asyncio.gather(*(self._run_job(job) for job in pending))
            return await self._write_summaries()
        finally:
            await asyncio.gather(*(
                backend.close()
                for roles in self.backends.values()
                for backend in roles.values()
            ))


async def run_evaluation(
    config: EvaluationConfig,
    *,
    models: set[str] | None = None,
    tags: set[str] | None = None,
    seeds: set[int] | None = None,
) -> dict[str, Any]:
    return await EvaluationRunner(
        config, models=models, tags=tags, seeds=seeds
    ).run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", action="append", help="run only this model slug")
    parser.add_argument("--tag", action="append", help="run only this environment tag")
    parser.add_argument("--seed", action="append", type=int, help="run only this seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    summary = asyncio.run(run_evaluation(
        load_config(args.config),
        models=set(args.model) if args.model else None,
        tags=set(args.tag) if args.tag else None,
        seeds=set(args.seed) if args.seed else None,
    ))
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    if any(value.get("errors") for value in summary["tasks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["EvaluationJob", "EvaluationRunner", "main", "run_evaluation"]
