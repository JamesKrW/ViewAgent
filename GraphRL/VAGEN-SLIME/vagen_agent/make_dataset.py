"""VAGEN's ``envs:`` yaml -> a jsonl slime can read.

slime's dataset is a flat file of prompts; VAGEN's is a list of environment *families*, each
expanded into ``n_envs`` instances with generated seeds. This bridges the two, and the
bridge is the whole of "how do I add an environment": the environment spec rides in each
row's ``metadata``, slime carries that through to ``sample.metadata``, and
``vagen_agent.rollout`` builds the environment from it. No rollout code changes, and a
batch may legitimately mix families.

The seed expansion is **not** reimplemented here. It is imported from the vendored
``vagen/gym_agent_dataset.py``, verbatim from VAGEN, so a run on this port draws the same
seed sequence as the same yaml on verl -- which is the difference between comparing two
systems and comparing two datasets.

Usage::

    python -m vagen_agent.make_dataset \\
        --envs examples/train/sokoban/train_sokoban_vision.yaml \\
        --out  data/sokoban_train.jsonl \\
        --base-seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vagen_agent.envs.specs import _generate_seeds_for_spec, load_envspecs

logger = logging.getLogger(__name__)


def build_rows(envs_yaml: str, base_seed: int = 0,
               response_length_per_turn: int | None = None) -> list[dict]:
    """One row per environment instance, in spec order.

    ``response_length_per_turn`` overrides the yaml's value on every row. It is kept
    separate from slime's ``--rollout-max-response-len``: the former caps one environment
    action, while the latter corresponds to VAGEN's whole response-region budget and may
    contain several actions in concat/compact.
    """
    specs = load_envspecs(envs_yaml).specs
    if not specs:
        raise ValueError(f"{envs_yaml} declares no `envs:`")

    rows: list[dict] = []
    for spec_idx, spec in enumerate(specs):
        seeds = _generate_seeds_for_spec(spec, base_seed, spec_idx)
        for seed in seeds:
            rows.append({
                # A placeholder, and deliberately in **conversation form**.
                #
                # Under this rollout the prompt is not in the dataset at all: the
                # environment produces the system prompt and the first observation at
                # ``reset()``, because both depend on the seed. Nothing trains on this --
                # the rollout replaces ``sample.tokens`` wholesale -- so it is written to
                # be useful in a dump.
                #
                # It has to be a list rather than a string: slime's ``Dataset`` asserts
                # ``isinstance(prompt, list)`` whenever the checkpoint has a processor
                # (``utils/data.py:249``), which every VLM checkpoint does -- Qwen3.5
                # included. A bare string fails there at startup with
                # "prompt must be a list when processor is not None". Passing
                # ``--apply-chat-template`` would also satisfy it, but that flag means
                # something we do not want and would be a lie about this column.
                "prompt": [{"role": "user", "content": f"[{spec.name} seed={int(seed)}]"}],
                "label": "",
                "metadata": {
                    "env_name": spec.name,
                    "config": dict(getattr(spec, "config", None) or {}),
                    "seed": int(seed),
                    "max_turns": int(getattr(spec, "max_turns", 0) or 0) or None,
                    "response_length_per_turn": (
                        response_length_per_turn
                        or getattr(spec, "response_length_per_turn", None)),
                    "max_env_response_per_turn": getattr(spec, "max_env_response_per_turn", None),
                    # slime reads this into its `source_names` column, which survives the
                    # DP split -- so per-environment metrics stay separable in a mixed batch
                    # without adding anything to slime.
                    "source_name": getattr(spec, "data_source", None) or spec.name,
                },
            })
    # Drop the keys the spec did not set, rather than writing nulls: `EnvSpec.from_metadata`
    # falls back to the run config for anything absent, and an explicit null would override
    # that fallback with None.
    for row in rows:
        row["metadata"] = {k: v for k, v in row["metadata"].items() if v is not None}
    return rows


def write_jsonl(rows: list[dict], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", required=True, help="VAGEN envs yaml (train_*.yaml)")
    parser.add_argument("--out", required=True, help="output .jsonl")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="expands into a deterministic per-spec RNG seed; the same "
                             "value reproduces the same instance set")
    parser.add_argument("--response-length-per-turn", type=int, default=None,
                        help="override the yaml's per-turn generation budget on every row")
    args = parser.parse_args()

    rows = build_rows(args.envs, args.base_seed, args.response_length_per_turn)
    write_jsonl(rows, args.out)

    by_env: dict[str, int] = {}
    for row in rows:
        name = row["metadata"]["env_name"]
        by_env[name] = by_env.get(name, 0) + 1
    print(f"wrote {len(rows)} rows -> {args.out}")
    for name, count in sorted(by_env.items()):
        print(f"  {name:24s} {count}")


if __name__ == "__main__":
    main()
