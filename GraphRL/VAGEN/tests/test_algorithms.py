"""Algorithm-axis registry and reward-folding contracts."""

import sys
from types import ModuleType

from vagen_agent.algorithms import (
    AlgorithmSpec,
    registered_algorithms,
    resolve_algorithm,
)
from vagen_agent.algorithms._common.rewards import fold_across_rows


def test_builtin_algorithm_directories_register_every_supported_type() -> None:
    assert registered_algorithms() == (
        "default_gae",
        "token_level_gae",
        "trajectory_grpo",
        "turn_level_gae",
    )


def test_algorithm_specs_select_reward_folding_without_name_branches() -> None:
    rows = [[0.0, 1.0], [0.0, 2.0]]

    assert resolve_algorithm("default_gae").reward_folding == "episode_total"
    assert fold_across_rows(rows, "default_gae") == [[0.0, 3.0], [0.0, 3.0]]

    assert resolve_algorithm("token_level_gae").reward_folding == "token_suffix"
    assert fold_across_rows(rows, "token_level_gae") == [[0.0, 3.0], [0.0, 2.0]]

def test_algorithm_can_be_resolved_from_an_external_import_path() -> None:
    module = ModuleType("test_external_algorithm")
    module.SPEC = AlgorithmSpec(slime_estimator="grpo")
    sys.modules[module.__name__] = module
    try:
        assert resolve_algorithm("test_external_algorithm:SPEC") is module.SPEC
    finally:
        sys.modules.pop(module.__name__, None)
