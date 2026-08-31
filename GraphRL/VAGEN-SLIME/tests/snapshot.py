"""Structural snapshot of what a Sokoban episode produces, for old-vs-new diffing.

The refactor rewrites every layer between the harness and the Sample. What must not move
is the Sample itself, so this dumps exactly that -- tokens, mask, log-probs, reward
placement, row identity -- as canonical JSON, for all three harnesses, from a fixed seed
and a fixed fake backend.

    python tests/snapshot.py --out baseline.json      # before
    python tests/snapshot.py --out after.json         # after
    python tests/snapshot.py --diff baseline.json after.json

The driver below is rewritten as the internal API changes; the *format* is not. That is
the point: a diff of two snapshots is a diff of what training would see, not of how we
got there.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

ENV_CONFIG = {
    "render_mode": "text",
    "prompt_format": "free_think",
    "format_reward": 0.02,
    "min_solution_steps": [1, 5],
}
SEED = 42
MAX_TURNS = 4
ACTIONS = ["Left", "Right", "Up", "Down", "Left", "Right", "Up", "Down"]
ROOM = {"response_len": 8192, "floor": 256}

#: (label, harness, cfg). ``compact`` at 1200 never reaches its budget on this episode and
#: degenerates to concat -- which is worth snapshotting, but it exercises no branching. So
#: ``compact_tight`` forces at least two compactions: that is the only scenario in which
#: the tree actually has to fork, and the whole refactor turns on getting forks right.
SCENARIOS = (
    ("concat", "concat", {}),
    ("no_concat", "no_concat", {}),
    ("compact", "compact", {"budget": 1200, "summary_budget": 256}),
    ("compact_tight", "compact", {"budget": 650, "summary_budget": 128}),
)


def reply(n: int) -> str:
    action = ACTIONS[n % len(ACTIONS)]
    return (f"<think>a box and a target; I will step {action.lower()}.</think>\n"
            f"<answer>{action}</answer>")


# --------------------------------------------------------------------------- format
def rle(bits) -> list:
    """[1,1,0,0,0,1] -> [[1,2],[0,3],[1,1]]. A mask diff is then readable."""
    out = []
    for b in bits:
        if out and out[-1][0] == b:
            out[-1][1] += 1
        else:
            out.append([int(b), 1])
    return out


def sha(values) -> str:
    return hashlib.sha256(
        ",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values).encode()
    ).hexdigest()[:16]


def sample_record(sample) -> dict:
    """One Sample, as the fields training actually consumes."""
    scores = getattr(sample, "vagen_scores", None)      # per-token vector, when present
    return {
        "rollout_id": sample.rollout_id,
        "index": sample.index,
        "group_index": sample.group_index,
        "reward": round(float(sample.reward), 9) if not isinstance(sample.reward, list)
                  else [round(float(v), 9) for v in sample.reward],
        "status": sample.status.value if sample.status else None,
        "n_tokens": len(sample.tokens),
        "response_length": sample.response_length,
        "prompt_len": len(sample.tokens) - sample.response_length,
        "tokens": list(sample.tokens),
        "loss_mask_rle": rle(sample.loss_mask or []),
        "n_trainable": sum(sample.loss_mask or []),
        "logprobs_sha": sha(sample.rollout_log_probs or []),
        "round_number": (sample.metadata or {}).get("round_number"),
        "scores_nonzero": [[i, round(float(v), 9)] for i, v in enumerate(scores or [])
                           if v] or None,
        "remove_sample": bool(getattr(sample, "remove_sample", False)),
    }


# --------------------------------------------------------------------------- driver
def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True,
                                         trust_remote_code=True)


async def run_one(label: str, harness_name: str, cfg: dict, tokenizer) -> dict:
    """NEW-API driver: rollout client + message tree + assemble."""
    from vagen_agent.envs import build_env, get_env_cls
    from vagen_agent.harness import build_harness
    from vagen_agent.models import build_model_adapter
    from vagen_agent.rollout.client import RolloutClient
    from vagen_agent.rollout.scoring import ScoringSeam
    from vagen_agent.rollout.trajectory import Record, assemble

    class FakeClient(RolloutClient):
        async def _post(self, engine_ids, image_data, sampling):
            text = reply(self.n_calls)
            ids = tokenizer.encode(text, add_special_tokens=False)
            eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(eos, int) and eos >= 0:
                ids = ids + [eos]
            return {"text": text, "token_ids": ids, "logprobs": [-0.5] * len(ids),
                    "stop_reason": "stop",
                    "usage": {"prompt_tokens": len(engine_ids),
                              "completion_tokens": len(ids)}}

    model = build_model_adapter("qwen", tokenizer, None)
    record = Record(episode_id=f"snap-{label}")
    client = FakeClient(model, record, sampling_params={"max_new_tokens": 512})
    harness = build_harness(harness_name, **cfg)
    inner = build_env(
        get_env_cls("Sokoban"),
        dict(ENV_CONFIG),
        max_turns=MAX_TURNS,
        required_type=harness.environment_type,
    )
    env = ScoringSeam(inner, record, seed=SEED)

    try:
        await harness.run_episode(client, env)
    finally:
        await env.close()

    from vagen_agent.rollout.trajectory import episode_status

    class Src:
        index, group_index, prompt, apply_chat_template_kwargs, label = 7, 3, "", {}, None
        tokens, metadata = [], {}

    samples = assemble(record, Src, algorithm="default_gae",
                       status=episode_status(aborted=record.aborted,
                                             truncated=record.truncated))
    return {
        "harness": harness_name,
        "label": label,
        "n_calls": client.n_calls,
        "episode": {"turns": record.turns, "total_reward": round(record.total_reward, 9),
                    "terminated": record.terminated, "truncated": record.truncated},
        "rows": [sample_record(s) for s in samples],
    }


def build(out: Path) -> int:
    tokenizer = load_tokenizer()
    snap = {"model": MODEL, "seed": SEED, "max_turns": MAX_TURNS, "harnesses": {}}
    for label, name, cfg in SCENARIOS:
        snap["harnesses"][label] = asyncio.run(run_one(label, name, cfg, tokenizer))
        h = snap["harnesses"][label]
        print(f"  {label:14s} {len(h['rows'])} row(s)  calls={h['n_calls']}  "
              f"turns={h['episode']['turns']}  reward={h['episode']['total_reward']:+.4f}  "
              f"tokens={[r['n_tokens'] for r in h['rows']]}  "
              f"trainable={[r['n_trainable'] for r in h['rows']]}")
    out.write_text(json.dumps(snap, indent=1, sort_keys=True))
    print(f"\nwrote {out}")
    return 0


# ----------------------------------------------------------------------------- diff
def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        yield path, node          # compare lists whole; a token stream diff is one line
    else:
        yield path, node


def diff(a: Path, b: Path) -> int:
    left, right = json.loads(a.read_text()), json.loads(b.read_text())
    la, lb = dict(walk(left)), dict(walk(right))
    problems = []
    for key in sorted(set(la) | set(lb)):
        if key not in la:
            problems.append(f"  + {key}")
        elif key not in lb:
            problems.append(f"  - {key}")
        elif la[key] != lb[key]:
            va, vb = la[key], lb[key]
            if isinstance(va, list) and isinstance(vb, list):
                where = next((i for i in range(min(len(va), len(vb))) if va[i] != vb[i]),
                             min(len(va), len(vb)))
                problems.append(f"  ~ {key}: len {len(va)} -> {len(vb)}, first differs at "
                                f"{where}: {va[where:where+4]} -> {vb[where:where+4]}")
            else:
                problems.append(f"  ~ {key}: {va!r} -> {vb!r}")
    if problems:
        print(f"DIFFERS ({len(problems)}):")
        print("\n".join(problems[:60]))
        if len(problems) > 60:
            print(f"  ... and {len(problems) - 60} more")
        return 1
    print("IDENTICAL")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--diff", type=Path, nargs=2)
    args = p.parse_args()
    if args.diff:
        return diff(*args.diff)
    return build(args.out or Path("snapshot.json"))


if __name__ == "__main__":
    raise SystemExit(main())
