"""Configuration for the standalone, OpenAI-compatible evaluator.

The environment and harness names use VAGEN's existing registries.  A config may list
several models and repeat one environment with several harnesses; evaluation runs their
Cartesian product without introducing a second environment abstraction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vagen_agent.envs.specs import EnvSpec, _generate_seeds_for_spec
from vagen_agent.evaluation.backends import resolve_backend
from vagen_agent.harness import resolve_harness


def _expanded(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expanded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expanded(item) for item in value]
    return value


def _redacted(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            lowered = str(key).lower()
            out[key] = "<redacted>" if lowered in {
                "api_key", "authorization", "access_token", "secret"
            } else _redacted(item)
        return out
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def _slug(value: Any) -> str:
    text = "".join(c.lower() if c.isalnum() else "-" for c in str(value))
    slug = "-".join(part for part in text.split("-") if part)
    if not slug:
        raise ValueError(f"cannot make a filesystem-safe name from {value!r}")
    return slug


def _run_id(value: Any) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text):
        raise ValueError(
            "experiment.id must be one filesystem-safe component containing only "
            "letters, digits, '.', '_' and '-'"
        )
    return text


def _urls(item: dict[str, Any], *, name: str) -> tuple[str, ...]:
    if item.get("base_url") and item.get("base_urls"):
        raise ValueError(f"model {name!r} cannot set both base_url and base_urls")
    value = item.get("base_urls", item.get("base_url"))
    if isinstance(value, str):
        candidates = value.replace(";", "\n").splitlines()
    elif isinstance(value, list):
        candidates = value
    else:
        raise TypeError(f"model {name!r} requires base_url or base_urls")
    urls = tuple(str(candidate).strip().rstrip("/") for candidate in candidates
                 if str(candidate).strip())
    if not urls:
        raise ValueError(f"model {name!r} has no usable endpoint")
    if len(set(urls)) != len(urls):
        raise ValueError(f"model {name!r} has duplicate endpoints")
    return urls


@dataclass(frozen=True)
class ModelSpec:
    name: str
    served_model: str = ""
    base_urls: tuple[str, ...] = ()
    backend: str = "openai"
    api_key: str = "EMPTY"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 600.0
    max_concurrency: int = 1
    max_retries: int = 2
    min_backoff: float = 0.5
    max_backoff: float = 4.0
    token_limit_field: str = "max_tokens"
    thinking_token_budget: int | None = None
    thinking_budget_field: str | None = None
    thinking_answer_reserve: int = 512
    sampling: dict[str, Any] = field(default_factory=dict)
    #: A composite evaluation target. Leaf specs leave this empty and are exposed as the
    #: ``default`` role; a planner/actor system stores one ordinary OpenAI-compatible
    #: leaf spec per role. Backend selection remains identical for every leaf.
    roles: dict[str, ModelSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.roles:
            if self.served_model or self.base_urls:
                raise ValueError(
                    "a composite ModelSpec stores endpoints under roles, not at its root"
                )
            if any(spec.roles for spec in self.roles.values()):
                raise ValueError("nested model role bundles are not supported")
            return
        if not self.served_model.strip():
            raise ValueError(f"model {self.name!r} requires served_model")
        if not self.base_urls:
            raise ValueError(f"model {self.name!r} requires at least one base_url")

    @property
    def available_roles(self) -> frozenset[str]:
        return frozenset(self.roles) if self.roles else frozenset({"default"})

    def for_role(self, role: str) -> ModelSpec:
        if self.roles:
            try:
                return self.roles[role]
            except KeyError as exc:
                raise ValueError(
                    f"model bundle {self.name!r} has no role {role!r}; available roles: "
                    f"{sorted(self.roles)}"
                ) from exc
        if role == "default":
            return self
        raise ValueError(
            f"model {self.name!r} is single-role and cannot provide {role!r}"
        )

    @property
    def served_models(self) -> str | dict[str, str]:
        if self.roles:
            return {role: spec.served_model for role, spec in self.roles.items()}
        return self.served_model


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    tag: str
    seeds: tuple[int, ...]
    max_turns: int
    config: dict[str, Any]
    harness: str
    harness_config: dict[str, Any]
    response_length_per_turn: int | None
    context_window: int | None
    thinking_token_budget: int | None
    sampling: dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    path: Path
    experiment_id: str
    output_dir: Path
    resume: str
    max_concurrent_episodes: int
    record_images: bool
    models: tuple[ModelSpec, ...]
    environments: tuple[EnvironmentSpec, ...]
    fingerprint: str
    public_config: dict[str, Any]


def _leaf_model(
    value: dict[str, Any],
    *,
    name: str,
    default_sampling: dict[str, Any],
    location: str,
) -> ModelSpec:
    backend = str(value.get("backend", "openai")).lower()
    resolve_backend(backend)
    sampling = value.get("sampling") or {}
    headers = value.get("headers") or {}
    if not isinstance(sampling, dict) or not isinstance(headers, dict):
        raise TypeError(f"{location} sampling and headers must be mappings")
    token_limit_field = str(value.get("token_limit_field", "max_tokens"))
    if token_limit_field not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("token_limit_field must be max_tokens or max_completion_tokens")
    thinking_budget = value.get("thinking_token_budget")
    thinking_field = value.get("thinking_budget_field")
    thinking_reserve = int(value.get("thinking_answer_reserve", 512))
    if thinking_budget is not None and thinking_field is None:
        raise ValueError(
            f"{location} sets thinking_token_budget but not thinking_budget_field "
            "(for example 'custom_params.thinking_budget' for SGLang or "
            "'thinking_token_budget' for vLLM)"
        )
    if thinking_field is not None and any(
        not part or not part.replace("_", "").isalnum()
        for part in str(thinking_field).split(".")
    ):
        raise ValueError(f"{location}.thinking_budget_field is invalid")
    if thinking_budget is not None and int(thinking_budget) < 0:
        raise ValueError("thinking_token_budget must be non-negative")
    if thinking_reserve < 1:
        raise ValueError("thinking_answer_reserve must be positive")
    spec = ModelSpec(
        name=name,
        served_model=str(value.get("served_model", value.get("name", ""))),
        base_urls=_urls(value, name=name),
        backend=backend,
        api_key=str(value.get("api_key", "EMPTY")),
        headers={str(k): str(v) for k, v in headers.items()},
        timeout=float(value.get("timeout", 600.0)),
        max_concurrency=int(value.get("max_concurrency", 1)),
        max_retries=int(value.get("max_retries", 2)),
        min_backoff=float(value.get("min_backoff", 0.5)),
        max_backoff=float(value.get("max_backoff", 4.0)),
        token_limit_field=token_limit_field,
        thinking_token_budget=(int(thinking_budget) if thinking_budget is not None else None),
        thinking_budget_field=(str(thinking_field) if thinking_field is not None else None),
        thinking_answer_reserve=thinking_reserve,
        sampling={**default_sampling, **sampling},
    )
    if spec.max_concurrency < 1 or spec.max_retries < 0:
        raise ValueError("model concurrency must be positive and retries non-negative")
    return spec


def _role_value(shared: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    merged = {**shared, **value}
    for field_name in ("headers", "sampling"):
        common = shared.get(field_name) or {}
        specific = value.get(field_name) or {}
        if not isinstance(common, dict) or not isinstance(specific, dict):
            raise TypeError(f"composite model {field_name} values must be mappings")
        merged[field_name] = {**common, **specific}
    return merged


def load_config(path: str | Path) -> EvaluationConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error
        raise ImportError("evaluation configs require PyYAML") from exc

    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("evaluation config root must be a mapping")
    raw = _expanded(raw)
    unknown = set(raw) - {"experiment", "run", "defaults", "models", "envs"}
    if unknown:
        raise ValueError(f"unknown evaluation config keys: {sorted(unknown)}")

    experiment = raw.get("experiment") or {}
    run = raw.get("run") or {}
    defaults = raw.get("defaults") or {}
    model_values = raw.get("models")
    env_values = raw.get("envs")
    if not isinstance(experiment, dict) or not isinstance(run, dict) \
            or not isinstance(defaults, dict):
        raise TypeError("experiment, run, and defaults must be mappings")
    if not isinstance(model_values, list) or not model_values:
        raise TypeError("models must be a non-empty list")
    if not isinstance(env_values, list) or not env_values:
        raise TypeError("envs must be a non-empty list")

    experiment_id_value = str(experiment.get("id", "")).strip()
    if not experiment_id_value:
        raise ValueError("experiment.id is required")
    experiment_id = _run_id(experiment_id_value)
    output_dir = Path(str(experiment.get("output_dir", "runs/eval")))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()

    resume = str(run.get("resume", "skip_completed"))
    if resume not in {"skip_completed", "force_rerun"}:
        raise ValueError("run.resume must be skip_completed or force_rerun")
    max_concurrent = int(run.get("max_concurrent_episodes", 1))
    if max_concurrent < 1:
        raise ValueError("run.max_concurrent_episodes must be positive")

    default_sampling = defaults.get("sampling") or {}
    default_harness_config = defaults.get("harness_config") or {}
    if not isinstance(default_sampling, dict) or not isinstance(default_harness_config, dict):
        raise TypeError("defaults.sampling and defaults.harness_config must be mappings")

    models: list[ModelSpec] = []
    model_names: set[str] = set()
    for index, value in enumerate(model_values):
        if not isinstance(value, dict):
            raise TypeError(f"models[{index}] must be a mapping")
        backend = str(value.get("backend", "openai")).lower()
        resolve_backend(backend)
        name = _slug(value.get("name", value.get("served_model", "")))
        if name in model_names:
            raise ValueError(f"duplicate model name {name!r}")
        model_names.add(name)
        role_values = value.get("roles")
        if role_values is None:
            models.append(_leaf_model(
                value,
                name=name,
                default_sampling=default_sampling,
                location=f"models[{index}]",
            ))
            continue
        if not isinstance(role_values, dict) or not role_values:
            raise TypeError(f"models[{index}].roles must be a non-empty mapping")
        shared = {key: item for key, item in value.items() if key not in {"name", "roles"}}
        roles: dict[str, ModelSpec] = {}
        for raw_role, role_config in role_values.items():
            role = str(raw_role).strip()
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", role):
                raise ValueError(f"models[{index}] has invalid role name {raw_role!r}")
            if role in roles:
                raise ValueError(f"models[{index}] has duplicate role {role!r}")
            if not isinstance(role_config, dict):
                raise TypeError(f"models[{index}].roles.{role} must be a mapping")
            if "roles" in role_config:
                raise ValueError("nested model role bundles are not supported")
            roles[role] = _leaf_model(
                _role_value(shared, role_config),
                name=f"{name}-{_slug(role)}",
                default_sampling=default_sampling,
                location=f"models[{index}].roles.{role}",
            )
        models.append(ModelSpec(name=name, roles=roles))

    environments: list[EnvironmentSpec] = []
    tags: set[str] = set()
    for index, value in enumerate(env_values):
        if not isinstance(value, dict):
            raise TypeError(f"envs[{index}] must be a mapping")
        env_name = str(value.get("name", "")).strip()
        if not env_name:
            raise ValueError(f"envs[{index}].name is required")
        harness = str(value.get("harness", defaults.get("harness", "no_concat")))
        resolve_harness(harness)
        tag = _slug(value.get("tag", f"{env_name}-{harness}"))
        if tag in tags:
            raise ValueError(f"duplicate environment tag {tag!r}")
        tags.add(tag)
        count = int(value.get("n_envs", 1))
        max_turns = int(value.get("max_turns", defaults.get("max_turns", 1)))
        if count < 1 or max_turns < 1:
            raise ValueError(f"envs[{index}] n_envs and max_turns must be positive")
        env_config = value.get("config") or {}
        harness_config = value.get("harness_config") or {}
        sampling = value.get("sampling") or {}
        if not all(isinstance(item, dict) for item in (
            env_config, harness_config, sampling
        )):
            raise TypeError(
                f"envs[{index}] config, harness_config, and sampling must be mappings"
            )
        seed_spec = EnvSpec(
            name=env_name,
            n_envs=count,
            seed=value.get("seed", [0]),
            seed_list=value.get("seed_list"),
        )
        seeds = tuple(_generate_seeds_for_spec(seed_spec, int(run.get("base_seed", 0)), index))
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"envs[{index}] resolved duplicate seeds; episode resume requires unique seeds"
            )
        response_length = value.get(
            "response_length_per_turn", defaults.get("response_length_per_turn")
        )
        context_window = value.get("context_window", defaults.get("context_window"))
        if response_length is not None and int(response_length) < 1:
            raise ValueError(f"envs[{index}].response_length_per_turn must be positive")
        if context_window is not None and int(context_window) < 1:
            raise ValueError(f"envs[{index}].context_window must be positive")
        thinking_budget = value.get("thinking_token_budget")
        if thinking_budget is not None and int(thinking_budget) < 0:
            raise ValueError(f"envs[{index}].thinking_token_budget must be non-negative")
        environments.append(EnvironmentSpec(
            name=env_name,
            tag=tag,
            seeds=seeds,
            max_turns=max_turns,
            config=dict(env_config),
            harness=harness,
            harness_config={**default_harness_config, **harness_config},
            response_length_per_turn=(int(response_length) if response_length is not None else None),
            context_window=(int(context_window) if context_window is not None else None),
            thinking_token_budget=(int(thinking_budget) if thinking_budget is not None else None),
            sampling=dict(sampling),
        ))

    for model in models:
        for environment in environments:
            required = frozenset(resolve_harness(environment.harness).model_roles)
            missing = required - model.available_roles
            if missing:
                raise ValueError(
                    f"model {model.name!r} cannot run harness {environment.harness!r}: "
                    f"missing roles {sorted(missing)}; available roles are "
                    f"{sorted(model.available_roles)}"
                )

    public = _redacted(raw)
    fingerprint = hashlib.sha256(
        json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return EvaluationConfig(
        path=config_path,
        experiment_id=experiment_id,
        output_dir=output_dir,
        resume=resume,
        max_concurrent_episodes=max_concurrent,
        record_images=bool(run.get("record_images", True)),
        models=tuple(models),
        environments=tuple(environments),
        fingerprint=fingerprint,
        public_config=public,
    )


__all__ = ["EnvironmentSpec", "EvaluationConfig", "ModelSpec", "load_config"]
