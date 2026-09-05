# Ground-plane action space

`ground_plane_v1` is shared by the ScanNet, AI2-THOR, and Habitat-GS proxy
environments. Enable it with:

```yaml
ground_plane_movement: true
```

The exact world axes differ by corpus, but the agent-facing semantics do not:

- `move_forward` / `move_backward`: move one full translation step along the
  yaw-only heading projected onto the horizontal plane. Pitch and roll do not
  add a vertical component.
- `move_left` / `move_right`: strafe one full step on that horizontal plane,
  perpendicular to the heading.
- `move_up` / `move_down`: move along world-up/world-down (`Z` in ScanNet,
  `Y` in AI2-THOR and Habitat-GS).
- `turn_left` / `turn_right`: yaw about world-up.
- `look_up` / `look_down`: change camera pitch without changing the body
  heading.
- Roll is excluded from the unified navigation set.

The old default remains `ground_plane_movement: false` in the reusable
manipulators so legacy datasets remain replayable. New rows carry both
`"ground_plane_movement": true` and
`"action_space_version": "ground_plane_v1"`; proxy environments reject an
explicit mode mismatch instead of silently training against an unreachable
target.

To derive a ground-plane corpus from an accepted dataset while preserving its
scene split:

```bash
python -m view_suite.envs.utils.regenerate_ground_plane_data \
  --corpus scannet --src-root data/viewsuite_15k \
  --out-root data/viewagent15k_scannet_open3d_ground_plane

python -m view_suite.envs.utils.regenerate_ground_plane_data \
  --corpus ai2thor --src-root data/ai2thor \
  --out-root data/viewagent15k_ai2thor_ground_plane --render-backend local --gpu-id 0

python -m view_suite.envs.utils.regenerate_ground_plane_data \
  --corpus habitat_gs --src-root data/habitat_gs \
  --out-root data/viewagent15k_habitat_gs_ground_plane
```

The converter reuses unchanged source images through read-only symlinks and stores
only changed renders in `ground_plane_cache/`. It rewrites all three task JSONLs
and records counts and provenance in `dataset_manifest.json`.

The generated datasets in this workspace are:

| corpus | directory | train | dev/eval | test |
|---|---|---:|---:|---:|
| ScanNet | `data/viewagent15k_scannet_open3d_ground_plane` | 3,188 | 353 | 497 |
| AI2-THOR | `data/viewagent15k_ai2thor_ground_plane` | 1,235 | 255 | 268 |
| Habitat-GS | `data/viewagent15k_habitat_gs_ground_plane` | 1,183 | 204 | 234 |

The ScanNet conversion drops 148 samples containing roll (not part of the shared
action set) and 111 whose newly rendered option fails the image-quality gate.
AI2-THOR drops one newly blank sample. Habitat-GS reuses every target because its
sampled horizontal actions already had body-ground semantics.
