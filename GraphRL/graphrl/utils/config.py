"""
Configuration loading + per-iteration override merging.

Three pieces:
  - :func:`load_pipeline_config`        — load the user's pipeline YAML
  - :func:`load_backend_default_config` — load a backend's bundled defaults
                                          (graphrl/configs/<rel_path>.yaml)
  - :func:`merge_iteration_config`      — merge defaults ⇐ general ⇐ iter
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Paths (under ``graphrl/configs/``) of the two backend default config files.
RL_DEFAULTS_PATH = "slime_configs/config"
SFT_DEFAULTS_PATH = "llamafactory_configs/config"

# graphrl/configs/ — resolved once. This file lives at graphrl/utils/config.py,
# so parents[1] is ``graphrl/``.
_PACKAGE_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """Load the pipeline YAML and resolve OmegaConf interpolations.

    The mono-backend controller auto-loads SLIME/LlamaFactory module defaults
    itself; pipeline.yaml does not declare ``module_defaults``. This loader
    just resolves interpolations and returns a plain dict.
    """
    config_path = Path(config_path).resolve()
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def load_backend_default_config(rel_path: str) -> Dict[str, Any]:
    """Load a default config bundled under ``graphrl/configs/<rel_path>``.

    ``rel_path`` may include or omit the ``.yaml`` suffix. Missing files
    yield an empty dict and a warning (so a stale ``module_defaults`` block
    doesn't crash the run).
    """
    candidate = _PACKAGE_CONFIGS / rel_path
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    if not candidate.exists():
        logger.warning("Default config not found: %s", candidate)
        return {}
    return OmegaConf.to_container(OmegaConf.load(candidate), resolve=True)


def merge_iteration_config(
    default_config: Dict[str, Any],
    general_override: Any = None,
    iteration_override: Any = None,
) -> Optional[Dict[str, Any]]:
    """Merge configs in order: defaults <- general_overrides <- iteration_overrides.

    Returns:
        ``None`` if *iteration_override* is explicitly ``None`` or has
        ``skip: true`` — the controller treats this as "skip this phase".
        Otherwise the merged dict (with the ``skip`` key stripped).
    """
    if iteration_override is None:
        return None

    if isinstance(iteration_override, dict) and iteration_override.get("skip") is True:
        return None

    base = OmegaConf.create(default_config or {})
    if general_override:
        base = OmegaConf.merge(base, OmegaConf.create(general_override))
    if iteration_override:
        base = OmegaConf.merge(base, OmegaConf.create(iteration_override))

    merged = OmegaConf.to_container(base, resolve=True)
    if isinstance(merged, dict):
        merged.pop("skip", None)
    return merged
