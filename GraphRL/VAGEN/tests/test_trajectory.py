"""The tree, on topologies no shipped harness produces.

concat is one chain, no_concat's branches are all cashed in, and compact's summary is
saved by riding on the same row as the turns around it. So the two rules that matter most
under a *branching* harness -- leaf filtering and shared-prefix deduplication -- are never
exercised by a real run, and would ship untested.

Built from synthetic records: no tokenizer, no environment, no sglang, no GPU. Tokens are
small integers, which is all the tree cares about.

    python tests/test_trajectory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vagen_agent.rollout.trajectory import (  # noqa: E402
    Generated,
    Record,
    Span,
    assemble,
)


class Src:
    index, group_index, prompt, label = 7, 3, "", None
    tokens, metadata = [], {}


def ctx(record, parent, role, text, ids):
    return record.add_context_node(parent, {"role": role, "content": text},
                                   Span(ids=list(ids), engine_ids=list(ids)))


def gen(record, parent, call_id, ids, *, reward=None):
    """Attach a generated node. ``reward=None`` means no env step was ever taken on it --
    which is exactly what the leaf filter keys off."""
    node = record.add_generated(
        parent, {"role": "assistant", "content": f"a{call_id}"},
        Generated(call_id=call_id, output_ids=list(ids), logprobs=[-0.5] * len(ids)),
        Span(ids=list(ids), engine_ids=list(ids)))
    if reward is not None:
        record.rewards[call_id] = [0.0] * (len(ids) - 1) + [float(reward)]
    return node


def rows_of(record):
    samples = assemble(record, Src, algorithm="default_gae")
    out = []
    for s in samples:
        response = s.tokens[len(s.tokens) - s.response_length:]
        out.append({
            "tokens": list(s.tokens),
            "trained": [t for t, m in zip(response, s.loss_mask) if m],
            "reward": float(s.reward),
        })
    return out


# --------------------------------------------------------------------------- cases
def case_linear():
    """concat: one chain, both responses trained, one row."""
    r = Record()
    s = ctx(r, r.root, "system", "S", [1])
    u = ctx(r, s, "user", "U0", [2])
    a1 = gen(r, u, 1, [10, 11], reward=0.5)
    u1 = ctx(r, a1, "user", "U1", [3])
    gen(r, u1, 2, [12, 13], reward=0.5)
    return r, [{"trained": [10, 11, 12, 13], "reward": 1.0}]


def case_siblings():
    """no_concat: one system prompt, T independent branches, T rows."""
    r = Record()
    s = ctx(r, r.root, "system", "S", [1])
    for i in range(3):
        u = ctx(r, s, "user", f"U{i}", [2 + i])
        gen(r, u, i + 1, [20 + i], reward=1.0)
    return r, [{"trained": [20], "reward": 3.0},
               {"trained": [21], "reward": 3.0},
               {"trained": [22], "reward": 3.0}]


def case_compaction():
    """compact: the summary has no env step of its own, but rides a row that does."""
    r = Record()
    s = ctx(r, r.root, "system", "S", [1])
    u = ctx(r, s, "user", "U0", [2])
    a1 = gen(r, u, 1, [10], reward=1.0)
    req = ctx(r, a1, "user", "summarise", [4])
    gen(r, req, 2, [11])                       # the summary: no reward filed
    u2 = ctx(r, s, "user", "summary+obs", [5])
    gen(r, u2, 3, [12], reward=2.0)
    # Row 0 keeps the summary: the filter asks whether *any* response on the row was
    # cashed in, and a1 was.
    return r, [{"trained": [10, 11], "reward": 3.0},
               {"trained": [12], "reward": 3.0}]


def case_best_of_n():
    """Three candidates from one state, one chosen. Only the chosen one becomes a row."""
    r = Record()
    s = ctx(r, r.root, "system", "S", [1])
    u = ctx(r, s, "user", "U0", [2])
    gen(r, u, 1, [10], reward=1.0)             # chosen
    gen(r, u, 2, [11])                         # rejected
    gen(r, u, 3, [12])                         # rejected
    return r, [{"trained": [10], "reward": 1.0}]


def case_retry():
    """The ordering trap: the abandoned response is inserted *first*.

    ``leaves()`` walks children in insertion order, so a filter applied after the fact
    would let the abandoned branch claim the shared prefix and then throw the row away --
    taking that prefix's gradient with it, and that part is real data.
    """
    r = Record()
    s = ctx(r, r.root, "system", "S", [1])
    u = ctx(r, s, "user", "U0", [2])
    a_bad = gen(r, u, 1, [90, 91])             # abandoned, no reward
    gen(r, u, 2, [10, 11], reward=1.0)         # the retry that was kept
    assert u.children[0] is a_bad, "the abandoned node must be visited first"
    return r, [{"trained": [10, 11], "reward": 1.0}]


def case_shared_generated_prefix():
    """Two branches below a *generated* node: it is trained exactly once."""
    r = Record()
    s = ctx(r, r.root, "system", "S", [1])
    u = ctx(r, s, "user", "U0", [2])
    a1 = gen(r, u, 1, [10], reward=1.0)
    u2 = ctx(r, a1, "user", "U1", [3])
    gen(r, u2, 2, [20], reward=1.0)
    u3 = ctx(r, a1, "user", "U2", [4])
    gen(r, u3, 3, [30], reward=1.0)
    # a1 is on both paths. The first row trains it; the second re-emits it as context.
    return r, [{"trained": [10, 20], "reward": 3.0},
               {"trained": [30], "reward": 3.0}]


CASES = [
    ("linear", case_linear),
    ("siblings", case_siblings),
    ("compaction", case_compaction),
    ("best_of_n", case_best_of_n),
    ("retry", case_retry),
    ("shared_prefix", case_shared_generated_prefix),
]


def main() -> int:
    failures = 0
    for name, build in CASES:
        record, expected = build()
        try:
            got = rows_of(record)
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"ERROR {name:16s} {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures += 1
            continue

        problems = []
        if len(got) != len(expected):
            problems.append(f"{len(got)} rows, expected {len(expected)}")
        else:
            for i, (g, e) in enumerate(zip(got, expected)):
                if g["trained"] != e["trained"]:
                    problems.append(f"row {i} trains {g['trained']}, expected {e['trained']}")
                if abs(g["reward"] - e["reward"]) > 1e-9:
                    problems.append(f"row {i} reward {g['reward']}, expected {e['reward']}")

        # Every response is trained at most once across the whole episode.
        seen: list[int] = []
        for g in got:
            seen += g["trained"]
        if len(seen) != len(set(seen)):
            dupes = {t for t in seen if seen.count(t) > 1}
            problems.append(f"tokens trained more than once: {sorted(dupes)}")

        if problems:
            print(f"FAIL  {name:16s} {[g['trained'] for g in got]}")
            for problem in problems:
                print(f"        - {problem}")
            failures += 1
        else:
            print(f"PASS  {name:16s} rows={[g['trained'] for g in got]}  "
                  f"reward={got[0]['reward'] if got else None}")

    print(f"\n{len(CASES) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
