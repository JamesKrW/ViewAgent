"""Path-pool enumeration and the proportional-sampling knobs.

The enumerator is checked against an independently written brute-force count over
200 random multigraphs, because everything downstream (the sampling fraction, the
graph-vs-no-graph rate comparison) is only as trustworthy as the denominator.

Run: PYTHONPATH=GraphRL python -m graphrl.envs.viewsuite.viewsuite_interactive_view_planning.tests.test_path_pool
"""
import itertools, random, sys
import networkx as nx

from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.utils.sft_generators.helpers import (
    _walk_paths_for_scene, _count_paths_by_length, _sample_paths_for_scene,
)

class FakeGraph:
    def __init__(self, g): self._g = g

def make(n_nodes, edges):
    g = nx.MultiDiGraph()
    for i in range(n_nodes):
        g.add_node(f"n{i}", obs_str=f"s{i}", extra={"scene_id": "sc"})
    for u, v, k in edges:
        g.add_edge(f"n{u}", f"n{v}", key=k, obs_str=f"a{u}{v}{k}")
    return FakeGraph(g)

def brute(graph, starts, min_len, max_len):
    """Independent count: simple-node walks, distinct by edge-key sequence."""
    total = {L: 0 for L in range(min_len, max_len + 1)}
    def rec(cur, visited, depth):
        if depth >= min_len: total[depth] += 1
        if depth >= max_len: return
        for u, v, k in graph._g.out_edges(cur, keys=True):
            if v in visited: continue
            rec(v, visited | {v}, depth + 1)
    for s in starts: rec(s, {s}, 0)
    return total

random.seed(0)
ok = True
for trial in range(200):
    n = random.randint(2, 6)
    edges = set()
    for _ in range(random.randint(1, 10)):
        u, v = random.randrange(n), random.randrange(n)
        if u != v: edges.add((u, v, random.randrange(2)))
    G = make(n, edges)
    starts = [f"n{i}" for i in range(n)]
    lo, hi = 1, random.randint(1, 4)
    got, _ = _count_paths_by_length(G, starts, lo, hi, cap=None)
    exp = brute(G, starts, lo, hi)
    if got != exp:
        print("MISMATCH", n, sorted(edges), lo, hi, got, exp); ok = False; break
print("1) 枚举计数 vs 独立暴力实现:", "一致 ✅" if ok else "不一致 ❌")

# 池子 <= target 时应返回全部,且不重复
G = make(4, {(0,1,0),(1,2,0),(2,3,0)})
starts = [f"n{i}" for i in range(4)]
pool, _ = _count_paths_by_length(G, starts, 1, 3, cap=None)
rng = random.Random(0)
got = _sample_paths_for_scene(G, starts, 1, 3, num_samples=999, rng=rng, telemetry=[])
keys = {tuple(s["action"] for s in p) for p in got}
print(f"2) 小池子:pool={sum(pool.values())} 返回={len(got)} 去重后={len(keys)}",
      "✅" if len(got) == sum(pool.values()) == len(keys) else "❌")

# sample_fraction:取池子的一半
tel = []
rng = random.Random(0)
got = _sample_paths_for_scene(G, starts, 1, 3, num_samples=0, rng=rng,
                              sample_fraction=0.5, telemetry=tel, scene_id="sc")
print(f"3) fraction=0.5:pool_total={tel[0]['pool_total']} target={tel[0]['target']} 实得={len(got)}",
      "✅" if tel[0]["target"] == round(0.5 * tel[0]["pool_total"]) else "❌")
print("   pool_by_length =", tel[0]["pool_by_length"])

# 原行为不变:不传新参数时仍是随机游走、且受 num_samples 限制
rng = random.Random(0)
got = _sample_paths_for_scene(G, starts, 1, 3, num_samples=2, rng=rng)
print(f"4) 兼容旧调用(num_samples=2):得到 {len(got)} 条", "✅" if len(got) == 2 else "❌")
