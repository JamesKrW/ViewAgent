"""The default call path must stay byte-identical to the pre-change sampler.

sample_fraction and the pool telemetry are opt-in, but they sit in the middle of the
function every generator calls, so a regression here would silently change what every
future run samples -- and quietly break comparability with every run already recorded.
This replays the original implementation against the current one on the same RNG seed.

Run: PYTHONPATH=GraphRL python -m graphrl.envs.viewsuite.\
viewsuite_interactive_view_planning.tests.test_sampler_unchanged
"""
import random, sys, networkx as nx
from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.utils.sft_generators.helpers import (
    _sample_paths_for_scene)

class FG:
    def __init__(s,g): s._g=g

def make(n, seed):
    rng=random.Random(seed); g=nx.MultiDiGraph()
    for i in range(n): g.add_node(f"n{i}", obs_str=f"s{i}", extra={"scene_id":"sc"})
    for _ in range(n*3):
        u,v=rng.randrange(n),rng.randrange(n)
        if u!=v: g.add_edge(f"n{u}",f"n{v}",key=str(rng.randrange(3)),obs_str=f"a{u}{v}")
    return FG(g)

# 复刻改动前的实现,逐字节对比
def legacy(graph, ids, min_len, max_len, num_samples, rng):
    if not ids or graph._g.number_of_edges()==0: return []
    seen=set(); paths=[]; max_attempts=num_samples*30; attempts=0
    while len(paths)<num_samples and attempts<max_attempts:
        attempts+=1
        cur=rng.choice(ids); target_len=rng.randint(min_len,max_len)
        steps=[]; ekey=[]; visited={cur}; ok=True
        for _ in range(target_len):
            out=list(graph._g.out_edges(cur,data=True,keys=True))
            if not out: ok=False; break
            unv=[e for e in out if e[1] not in visited]
            if not unv: break
            u,v,eid,data=rng.choice(unv)
            steps.append({"from_id":u,"from_state":graph._g.nodes[u]["obs_str"],
                          "action":data["obs_str"],"to_id":v,"to_state":graph._g.nodes[v]["obs_str"]})
            ekey.append(eid); visited.add(v); cur=v
        if not ok or len(steps)<min_len: continue
        k=tuple(ekey)
        if k in seen: continue
        seen.add(k); paths.append(steps)
    return paths

bad=0
for t in range(300):
    G=make(random.Random(t).randint(3,25), t)
    ids=[f"n{i}" for i in range(G._g.number_of_nodes())]
    lo=random.Random(t+7).randint(1,2); hi=lo+random.Random(t+9).randint(0,3)
    ns=random.Random(t+11).randint(1,25)
    a=_sample_paths_for_scene(G, ids, lo, hi, ns, random.Random(1234))
    b=legacy(G, ids, lo, hi, ns, random.Random(1234))
    if a!=b: bad+=1; print("DIFF at trial",t,len(a),len(b)); break
print(f"300 组随机图 × 相同种子:默认路径产出与旧实现{'完全一致 ✅' if bad==0 else '不一致 ❌'}")
