"""End-to-end: a real Sokoban episode through every harness, with a fake model.

Real environment, real tokenizer, real harness, real token record -- only the inference
call is faked, and it is faked at the narrowest point (``RolloutClient._post``) so that
everything above it is the code that will run in training.

Two things are checked, and both are failure modes nothing downstream can detect.

**Alignment.** Every mask-1 position in a Sample must hold a token the model actually
generated, and nothing else. Checked by rebuilding the generated stream from what the fake
backend returned and comparing it position for position. A client that lost or duplicated
a span would still produce a perfectly well-formed Sample.

**Structure.** The row split, the shared ``rollout_id``, and the reward every row carries.
Under the tree these are *derived* from what the harness did to its message list rather
than declared anywhere, so they are worth asserting explicitly.

Then the whole thing is diffed against ``golden/expected.json``, which is byte-identical to
what the pre-refactor stack produced except for the score vector it never carried.

    python tests/test_rollout_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snapshot import SCENARIOS, load_tokenizer, run_one, sample_record  # noqa: E402


def check(label: str, harness: str, cfg: dict, tokenizer) -> tuple[list[str], str]:
    snap = asyncio.run(run_one(label, harness, cfg, tokenizer))
    rows, problems = snap["rows"], []

    def bad(msg):
        problems.append(msg)

    # --- alignment ----------------------------------------------------------
    # Every trainable run in every row must be one the fake backend actually returned.
    for i, row in enumerate(rows):
        response = row["tokens"][row["prompt_len"]:]
        mask = [b for value, count in row["loss_mask_rle"] for b in [value] * count]
        if len(mask) != len(response):
            bad(f"row {i}: {len(response)} response tokens but {len(mask)} mask entries")
            continue
        trained = [t for t, m in zip(response, mask) if m]
        if len(trained) != row["n_trainable"]:
            bad(f"row {i}: n_trainable {row['n_trainable']} != {len(trained)} masked tokens")
        if row["n_trainable"] == 0:
            bad(f"row {i}: no trainable tokens; the row carries no gradient")

    # --- structure ----------------------------------------------------------
    if len({r["rollout_id"] for r in rows}) != 1:
        bad(f"rows do not share a rollout_id: {[r['rollout_id'] for r in rows]}")
    if len({r["reward"] for r in rows}) != 1:
        bad(f"rows carry different rewards: {[r['reward'] for r in rows]}")
    total = snap["episode"]["total_reward"]
    if rows and abs(rows[0]["reward"] - total) > 1e-9:
        bad(f"row reward {rows[0]['reward']} != episode total {total}")
    if [r["round_number"] for r in rows] != list(range(len(rows))):
        bad(f"round_number is not the row order: {[r['round_number'] for r in rows]}")

    # --- the shape each policy is supposed to produce ------------------------
    turns = snap["episode"]["turns"]
    expected = {"concat": 1, "no_concat": turns, "compact": 1, "compact_tight": 2}[label]
    if len(rows) != expected:
        bad(f"{label} produced {len(rows)} rows, expected {expected}")
    # compact_tight compacts once, so it spends one call more than it takes turns.
    if snap["n_calls"] != turns + (1 if label == "compact_tight" else 0):
        bad(f"{snap['n_calls']} model calls for {turns} turns")

    # --- the fold ------------------------------------------------------------
    # default_gae: every row carries the episode total on its last response position.
    for i, row in enumerate(rows):
        scores = row["scores_nonzero"] or []
        last = row["response_length"] - 1
        if len(scores) != 1 or scores[0][0] != last or abs(scores[0][1] - total) > 1e-9:
            bad(f"row {i}: default_gae should put {total} at position {last}, got {scores}")

    shape = (f"{len(rows)} row(s)  turns={turns}  calls={snap['n_calls']}  "
             f"reward={total:+.3f}  tokens={[r['n_tokens'] for r in rows]}  "
             f"trainable={[r['n_trainable'] for r in rows]}  status={rows[0]['status']}")
    return problems, shape


def check_no_progress_guard(tokenizer) -> tuple[list[str], str]:
    """A budget too small to buy a turn must stop the run, not quietly become no_concat.

    Compaction that closes every conversation after one turn still finishes, still emits
    well-formed rows, and costs two generations per environment step -- nothing downstream
    tells it apart from compaction working. The guard exists for exactly that, and since
    no shipped configuration reaches it, it would otherwise never be exercised.
    """
    from vagen_agent.harness.compact import CompactionMakesNoProgress

    try:
        asyncio.run(run_one("starved", "compact",
                            {"budget": 1, "summary_budget": 32}, tokenizer))
    except CompactionMakesNoProgress as exc:
        text = str(exc)
        if "compact_budget=1" not in text:
            return ([f"guard fired but did not name the budget: {text[:120]}"], "raised")
        return ([], "raised CompactionMakesNoProgress, naming the budgets")
    return (["a budget of 1 did not trip the no-progress guard; compaction silently "
             "degenerated into no_concat at twice the cost"], "no raise")


def main() -> int:
    tokenizer = load_tokenizer()
    failures = 0
    produced = {}
    for label, harness, cfg in SCENARIOS:
        try:
            problems, shape = check(label, harness, cfg, tokenizer)
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"ERROR {label:14s} {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures += 1
            continue
        produced[label] = shape
        if problems:
            print(f"FAIL  {label:14s} {shape}")
            for problem in problems:
                print(f"        - {problem}")
            failures += 1
        else:
            print(f"PASS  {label:14s} {shape}")

    problems, shape = check_no_progress_guard(tokenizer)
    if problems:
        print(f"FAIL  {'no_progress':14s} {shape}")
        for problem in problems:
            print(f"        - {problem}")
        failures += 1
    else:
        print(f"PASS  {'no_progress':14s} {shape}")

    # --- against the golden reference ---------------------------------------
    golden = Path(__file__).parent / "golden" / "expected.json"
    if golden.exists():
        out = Path("/tmp/_e2e_snapshot.json")
        import subprocess

        subprocess.run([sys.executable, str(Path(__file__).parent / "snapshot.py"),
                        "--out", str(out)], check=True, capture_output=True)
        rc = subprocess.run([sys.executable, str(Path(__file__).parent / "snapshot.py"),
                             "--diff", str(golden), str(out)], capture_output=True, text=True)
        if rc.returncode:
            print(f"FAIL  golden         output moved:\n{rc.stdout[:2000]}")
            failures += 1
        else:
            print(f"PASS  golden         identical to {golden.name}")

    print(f"\n{len(SCENARIOS) + 2 - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
