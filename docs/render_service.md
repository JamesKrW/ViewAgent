# Running the ScanNet render service

ViewSuite tasks need a rendered view for every camera pose the agent proposes, so
training throughput is bounded by rendering, not by the model. The service exists so
that rendering can live on machines with a working GPU graphics stack while training
runs elsewhere.

## Backends

| backend | renderer | when to use |
|---|---|---|
| `open3d` | `MeshRenderer` (Filament) | the original path; single-GPU only |
| `habitat` | `HabitatRenderer` (Habitat-Sim) | **multi-GPU**; what you want on an 8-GPU box |
| `gsplat` | `GaussianSplatRenderer` | 3DGS scenes |

**Why `habitat` exists.** Open3D renders through Filament → EGL, and EGL enumerates
devices independently of CUDA, so `CUDA_VISIBLE_DEVICES` has no effect on which GPU it
draws on — every worker lands on EGL device 0. That is why only single-GPU boxes ever
worked. Habitat-Sim takes an explicit `gpu_device_id` (`eglQueryDevicesEXT` +
`eglGetPlatformDisplayEXT` internally), which does isolate: requesting device 6 grows
GPU 6's memory and no other's.

The trade-off: shading differs from Open3D's `defaultLit` PBR, so **models trained
against one backend do not transfer to the other**. Comparison renders are in
`render_compare/`.

## Setup

Habitat-Sim cannot share an environment with the training stack, so it gets its own —
and it needs the service dependencies too, not just habitat-sim:

```bash
conda create -y -n habitat python=3.9
conda install -y -n habitat habitat-sim headless -c conda-forge -c aihabitat
conda run -n habitat pip install uvicorn fastapi fire httpx python-multipart plyfile
```

The launcher picks that interpreter automatically for `BACKEND=habitat` and fails loudly
if it is missing, rather than starting a service that answers 200 with zero images.

Scene data goes at `data/scannet/scans/<scene_id>/<scene_id>_vh_clean.ply`
(`bash scripts/download_scannet.sh`).

## Running

```bash
export VIEWSUITE_ROOT=$PWD
# args: MAX_WORKERS GPU_IDS OMP_CAP PORT RESTART_INTERVAL BACKEND
bash scripts/scannet_http_service_loop.sh 160 0,1,2,3,4,5,6,7 1 8811 10800 habitat
```

The supervisor restarts the service periodically and reaps leaked worker groups. Workers
are `multiprocessing`-spawn children, so their cmdline is `multiprocessing.spawn`, not
`service.py` — killing only the parent orphans them and they keep their GPU contexts.
Reaping is by process group for that reason.

`MAX_WORKERS` is the number of resident scenes: the pool caches **one scene per worker
process** and pins each worker to a GPU round-robin, so 160 workers over 8 GPUs is 20
scenes per GPU. Sizing matters — a request for a scene that is not resident evicts
another and pays a reload (~2 s), whereas a resident scene answers in ~25 ms.

### TLS

```bash
--ssl_keyfile /path/key.pem --ssl_certfile /path/cert.pem
```

Worth doing whenever the client is on a different machine: some networks drop long-lived
plaintext connections between distant hosts, which truncates large multi-image responses
partway through. A self-signed pair is fine if the client skips verification.

Also bind `--host "::"` if any caller reaches you over IPv6; the default `0.0.0.0` is
IPv4-only.

## Cost, measured

8× B200, real ScanNet meshes, 640×480, 160 workers:

| | |
|---|---|
| VRAM per resident scene | ~322 MiB marginal; 433 MiB for the first scene in a fresh process |
| Scene load | 0.6–0.8 s, once |
| Render, fixed intrinsics | ~25 ms |
| Render, intrinsics varying per request | **~2 s** — see below |
| Throughput, warm | ~178 images/s across the box |

Per-scene VRAM tracks mesh size — 248 MiB for a 220 MB `.ply`, 401 MiB for 386 MB — so
do not extrapolate a single number across a corpus that ranges 42–580 MB.

**Keep intrinsics fixed if you can.** Habitat bakes resolution and FOV into the sensor at
construction, so changing either forces a full scene rebuild. Resolution changes are
unavoidable; FOV changes cannot be applied in place either
(`set_projection_params` is accepted and silently does nothing in habitat-sim 0.3.3).

## Load testing

```bash
python view_suite/scannet/tests/test_scannet_http_stress.py \
    --url=https://localhost:8811 --scene_folder_path=data/scannet \
    --num_scenes=32 --num_clients=64 --requests_per_client=5 --num_tasks_per_request=5
```

A request counts as successful only if it returned one image per task. That matters: the
service answers internal errors with **HTTP 200** plus `meta={"error": ...}` and no
images, so a harness that only checks for an exception will report 100% success against
a renderer that is producing nothing. Watch `Images/sec` — if it is 0.00, nothing is
being rendered regardless of the success rate.

## Gotchas

- **One current GL context per process.** Constructing a second `Simulator` steals the
  context from the first, so with several resident scenes only the last one can draw,
  and `close()` on a non-current simulator aborts the process. `HabitatRenderer` calls
  `acquire_gl_context()` before rendering and before closing.
- **ScanNet is Z-up and Habitat loads the mesh unrotated** — applying a Z-up→Y-up
  rotation renders pure black.
- **`override_scene_light_defaults=True` is required**, or the `.ply` loads on a flat
  vertex-colour path that ignores lights entirely.
- **Non-square intrinsics.** Habitat's projection is always square-pixel, but ScanNet's
  intrinsics are not (fx 462 / fy 617). The renderer compensates by rendering at height
  `H·fx/fy` and resampling; the principal point (cx/cy) is still ignored, which is
  harmless for centred intrinsics and wrong for a cropped K.
