# Remote environment service

VAGEN-SLIME provides a small stateful HTTP boundary for environments that must run in
another process, Python environment, or host. The transport exposes exactly the direct
`BaseEnv` lifecycle used by the built-in Harnesses:

```text
connect/reset -> system_prompt -> step -> close
```

There is no separate Gymnasium adapter and no real-time `get_frame`/`submit`/`poll`
protocol in VAGEN. Projects that need another interaction protocol own both its abstract
environment base and its Harness.

The rollout side uses the registered `RemoteEnv` implementation:

```yaml
envs:
  - name: RemoteEnv
    n_envs: 8
    max_turns: 10
    harness: no_concat
    config:
      base_urls: [http://env01:8000, http://env02:8000]
      timeout: 120
      retries: 3
      task: my_task
```

All keys except transport settings are forwarded unchanged to the service's
`create_env()` implementation. The first `reset(seed=...)` creates a session and pins
the client to one server. `retries` applies only to connection establishment because
retrying `step` could execute an action twice.

On the service side, implement one factory method returning a `BaseEnv`:

```python
from vagen_agent.envs.remote import BaseGymHandler, GymService


class Handler(BaseGymHandler):
    async def create_env(self, env_config):
        return MyEnv(env_config)


app = GymService(Handler(max_sessions=8)).build()
```

Run one Uvicorn worker because sessions are held in process memory:

```bash
uvicorn my_service:app --host 0.0.0.0 --port 8000 --workers 1
```

Observations remain complete dictionaries. JSON-compatible structured fields such as
DOM state, screen size, and proprioception stay in the multipart metadata; PIL images in
`multi_modal_input["<image>"]` are binary parts. Responses default to JPEG quality 90
with 4:4:4 chroma sampling. Lossless environments can opt into PNG explicitly:

```python
app = GymService(
    Handler(), image_format="PNG", image_mime="image/png", image_options={}
).build()
```

The implementation is adapted from VAGEN's MIT-licensed `vagen.envs_remote` at revision
`72ac57b3a2171c3bc9c844f54e47d81ed16dc5bd`.
