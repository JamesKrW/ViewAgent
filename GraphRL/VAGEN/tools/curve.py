import pathlib
import re
import sys

repo = pathlib.Path(__file__).resolve().parents[1]
txt = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   repo / "logs" / "runA.log").read_text(errors="replace")
rows={}
# metrics land as "<something> <step>: {...}" on several prefixes (rollout / train / perf)
for m in re.finditer(r"(\d+): (\{'(?:rollout|train|perf|env)/.*?\})\n", txt):
    step, blob = int(m.group(1)), m.group(2)
    try: d=eval(blob)
    except Exception: continue
    rows.setdefault(step, {}).update(d)
cols=[("reward","rollout/rewards"),("success","env/success"),
      ("turns","env/episode_turns"),("samp/ep","env/samples_per_episode"),
      ("len","rollout/response_len/mean"),("adv","rollout/advantages"),
      ("val","rollout/values"),("clipfrac","train/pg_clipfrac"),
      ("gnorm","train/grad_norm"),("ppo_kl","train/ppo_kl")]
print("step  " + "  ".join(f"{h:>8s}" for h,_ in cols))
for step in sorted(rows):
    d=rows[step]
    if "rollout/rewards" not in d: continue
    print(f"{step:4d}  " + "  ".join(
        (f"{d[k]:8.4f}" if isinstance(d.get(k),(int,float)) else f"{'-':>8s}") for _,k in cols))
