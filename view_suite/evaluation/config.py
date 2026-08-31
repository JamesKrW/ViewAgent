"""Translate existing ViewSuite evaluation YAML into VAGEN-SLIME config."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from vagen_agent.envs.specs import EnvSpec, _generate_seeds_for_spec
from vagen_agent.evaluation.config import EnvironmentSpec, EvaluationConfig
from view_suite.envs.slime_adapter import resolve_legacy_data_paths

# Importing registers all project provider backends before the config is used.
from view_suite.evaluation import backends as _backends  # noqa: F401


def _slug(value: Any) -> str:
    text = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    value = "-".join(part for part in text.split("-") if part)
    return value or "model"


def _component(value: Any, *, field_name: str) -> str:
    """Preserve established tag spelling while rejecting unsafe path values."""

    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text):
        raise ValueError(f"{field_name} must be one filesystem-safe component: {text!r}")
    return text


def _redacted(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if str(key).lower() in {"api_key", "azure_api_key", "authorization", "secret"}
                else _redacted(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def _resolve_defaults(path: Path, cfg: DictConfig, seen: set[Path] | None = None) -> DictConfig:
    defaults = OmegaConf.select(cfg, "defaults", default=None)
    if not defaults:
        return cfg
    seen = set() if seen is None else seen
    absolute = path.resolve()
    if absolute in seen:
        raise ValueError(f"cyclic defaults reference: {absolute}")
    seen.add(absolute)
    merged = OmegaConf.create()
    for entry in defaults:
        if not isinstance(entry, str):
            raise TypeError(f"defaults entries must be paths, got {entry!r}")
        target = (path.parent / entry).with_suffix(".yaml") if not Path(entry).suffix else path.parent / entry
        target = target.resolve()
        if not target.is_file():
            raise FileNotFoundError(f"default config not found: {target}")
        merged = OmegaConf.merge(merged, _resolve_defaults(target, OmegaConf.load(target), seen))
    with open_dict(cfg):
        del cfg["defaults"]
    return OmegaConf.merge(merged, cfg)


def load_legacy_dict(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    cfg = _resolve_defaults(config_path, OmegaConf.load(config_path))
    for override in overrides or ():
        if "=" not in override:
            raise ValueError(f"evaluation override must be key=value: {override!r}")
        key, raw_value = override.split("=", 1)
        # Preserve the familiar Hydra CLI spelling, including list indices such
        # as ``envs.0.n_envs=1``. Merging one dot-list DictConfig into the root
        # cannot update a ListConfig; OmegaConf.update can.
        key = key.lstrip("+")
        parsed = OmegaConf.from_dotlist([f"value={raw_value}"])["value"]
        OmegaConf.update(cfg, key, parsed, merge=True, force_add=True)
    OmegaConf.resolve(cfg)
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("evaluation config root must be a mapping")
    return _resolve_paths(value, config_path.parent)


def _resolve_paths(value: Any, base: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: _resolve_paths(item, base, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base, key) for item in value]
    if isinstance(value, str) and ("path" in key.lower() or key.lower().endswith("_dir")):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if expanded and not re.match(r"^[a-z]+://", expanded) and not os.path.isabs(expanded):
            return str((base / expanded).resolve())
        return expanded
    return value


@dataclass(frozen=True)
class CompatModelSpec:
    """Runtime-compatible superset of VAGEN-SLIME's ``ModelSpec``."""

    name: str
    served_model: str
    base_urls: tuple[str, ...]
    backend: str
    api_key: str = ""
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
    options: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, Any] = field(default_factory=dict)

    @property
    def available_roles(self) -> frozenset[str]:
        return frozenset({"default"})

    def for_role(self, role: str) -> "CompatModelSpec":
        if role != "default":
            raise ValueError(f"single-role model cannot provide {role!r}")
        return self

    @property
    def served_models(self) -> str:
        return self.served_model


def _api_key(backend: str, cfg: dict[str, Any]) -> str:
    if cfg.get("api_key"):
        return str(cfg["api_key"])
    if cfg.get("azure_api_key"):
        return str(cfg["azure_api_key"])
    candidates = {
        "openai": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
        "together": ("TOGETHER_API_KEY",),
        "sglang": ("SGLANG_API_KEY",),
        "vllm": ("VLLM_API_KEY",),
        "azure": ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"),
        "azure_responses": ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"),
        "claude": ("ANTHROPIC_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }.get(backend, ())
    return next((os.environ[name] for name in candidates if os.environ.get(name)), "EMPTY")


def _base_urls(backend: str, cfg: dict[str, Any]) -> tuple[str, ...]:
    value = cfg.get("base_urls", cfg.get("base_url"))
    if value is None:
        value = {
            "openai": "https://api.openai.com/v1",
            "together": "https://api.together.xyz/v1",
            "sglang": "http://127.0.0.1:30000/v1",
            "vllm": "http://127.0.0.1:8000/v1",
            "claude": "https://api.anthropic.com",
            "gemini": "https://generativelanguage.googleapis.com",
            "random_response": "random://response",
            "random_navigation": "random://navigation",
            "azure": cfg.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "azure_responses": cfg.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        }.get(backend, "")
    values = value if isinstance(value, list) else str(value).replace(";", "\n").splitlines()
    return tuple(str(item).strip().rstrip("/") for item in values if str(item).strip())


def _sampling(raw: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(raw)
    extra = value.pop("extra_body", None)
    if isinstance(extra, dict):
        value.update(extra)
    return value


def load_legacy_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> EvaluationConfig:
    """Load the repository's established Hydra-style evaluation format."""

    config_path = Path(path).expanduser().resolve()
    raw = load_legacy_dict(config_path, overrides)
    run = dict(raw.get("run") or {})
    experiment = dict(raw.get("experiment") or {})
    backend = str(run.get("backend", "openai")).lower()
    backend_cfg = dict((raw.get("backends") or {}).get(backend) or {})
    model = str(backend_cfg.get("model") or backend_cfg.get("deployment") or "").strip()
    if not model:
        raise ValueError(f"backends.{backend}.model (or deployment) is required")

    defaults = _sampling(dict(raw.get("default_chat_config") or {}))
    base_urls = _base_urls(backend, backend_cfg)
    if not base_urls:
        raise ValueError(f"backends.{backend} requires an endpoint/base_url")
    token_field = "max_completion_tokens" if "max_completion_tokens" in defaults else "max_tokens"
    options = {
        "azure_endpoint": backend_cfg.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "deployment": backend_cfg.get("deployment") or model,
        "api_version": backend_cfg.get("azure_api_version") or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        ),
        "responses_url": backend_cfg.get("responses_url"),
    }
    model_spec = CompatModelSpec(
        name=_slug(model),
        served_model=model,
        base_urls=base_urls,
        backend=backend,
        api_key=_api_key(backend, backend_cfg),
        headers={str(key): str(value) for key, value in (backend_cfg.get("headers") or {}).items()},
        timeout=float(backend_cfg.get("timeout", 600.0)),
        max_concurrency=int(backend_cfg.get("max_concurrency", 1)),
        max_retries=int(backend_cfg.get("max_retries", 2)),
        min_backoff=float(backend_cfg.get("min_backoff", 0.5)),
        max_backoff=float(backend_cfg.get("max_backoff", 4.0)),
        token_limit_field=token_field,
        sampling=defaults,
        options=options,
    )

    base_seed = int(run.get("base_seed", run.get("start_seed", 0)))
    default_max_turns = int(experiment.get("default_max_turns", 1))
    environments = []
    seen_tags: set[str] = set()
    for index, item in enumerate(raw.get("envs") or []):
        value = dict(item)
        name = str(value["name"])
        tag = _component(value.get("tag_id", f"{name}-{index}"), field_name="tag_id")
        if tag in seen_tags:
            raise ValueError(f"duplicate environment tag after normalization: {tag}")
        seen_tags.add(tag)
        count = int(value.get("n_envs", 1))
        seed_spec = EnvSpec(
            name=name,
            n_envs=count,
            seed=value.get("seed", [0]),
            seed_list=value.get("seed_list"),
        )
        seeds = tuple(_generate_seeds_for_spec(seed_spec, base_seed, index))
        per_env = _sampling(dict(value.get("chat_config") or {}))
        effective_sampling = {**defaults, **per_env}
        response_limit = (
            effective_sampling.get("max_tokens")
            or effective_sampling.get("max_completion_tokens")
            or effective_sampling.get("max_output_tokens")
            or (effective_sampling.get("generation_config") or {}).get("max_output_tokens")
        )
        concat = bool(value.get("concat_multi_turn", True))
        environments.append(
            EnvironmentSpec(
                name=name,
                tag=tag,
                seeds=seeds,
                max_turns=int(value.get("max_turns") or default_max_turns),
                config=resolve_legacy_data_paths(dict(value.get("config") or {})),
                harness="concat" if concat else "no_concat",
                harness_config={},
                response_length_per_turn=(int(response_limit) if response_limit else None),
                context_window=None,
                thinking_token_budget=None,
                sampling=per_env,
            )
        )

    if not environments:
        raise ValueError("evaluation config contains no envs")
    dump_dir = Path(str(experiment.get("dump_dir", "rollouts"))).expanduser().resolve()
    public = _redacted(raw)
    fingerprint = hashlib.sha256(
        json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    resume = str(run.get("resume", "skip_completed"))
    return EvaluationConfig(
        path=config_path,
        experiment_id=dump_dir.name,
        output_dir=dump_dir.parent,
        resume="force_rerun" if resume in {"off", "overwrite", "force_rerun"} else "skip_completed",
        max_concurrent_episodes=int(run.get("max_concurrent_jobs", 4)),
        record_images=bool(run.get("record_images", True)),
        models=(model_spec,),
        environments=tuple(environments),
        fingerprint=fingerprint,
        public_config=public,
    )


__all__ = ["CompatModelSpec", "load_legacy_config", "load_legacy_dict"]
