from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from view_suite.envs.ai2thor_proxy_task.data_gen.generate_data import (
    _render_topdown_mapview,
)
from view_suite.envs.habitat_gs_proxy_task.data_gen.regenerate_top_down import (
    CameraMetadata,
    _project_vertices,
    _read_ply_vertices,
)
from view_suite.envs.habitat_gs_proxy_task.data_gen.rerender_habitat_gs_top_down import (
    SH_C0,
    SH_C1,
    _select_gaussians,
    _sh_basis,
)


class RegenerateTopDownTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_colored_ply(self) -> Path:
        path = self.root / "scene.ply"
        vertices: list[tuple[float, float, float, int, int, int]] = []
        for z, color in ((0.0, (220, 40, 30)), (2.5, (20, 220, 30))):
            for y in np.linspace(-1.0, 1.0, 32):
                for x in np.linspace(-1.0, 1.0, 32):
                    vertices.append((float(x), float(y), z, *color))
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        with path.open("wb") as handle:
            handle.write(header)
            for vertex in vertices:
                handle.write(struct.pack("<fffBBB", *vertex))
        return path

    def test_reads_only_fixed_width_vertex_element(self) -> None:
        vertices = _read_ply_vertices(self._write_colored_ply())
        self.assertEqual(len(vertices), 2048)
        self.assertEqual(vertices.dtype.names, ("x", "y", "z", "red", "green", "blue"))
        self.assertAlmostEqual(float(vertices[0]["x"]), -1.0)

    def test_projection_removes_geometry_above_eye_level(self) -> None:
        vertices = _read_ply_vertices(self._write_colored_ply())
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.diag([1.0, -1.0, -1.0])
        c2w[:3, 3] = [0.0, 0.0, 5.0]
        camera = CameraMetadata(
            c2w=c2w,
            intrinsics=np.asarray(
                [[48.0, 0.0, 32.0], [0.0, 48.0, 32.0], [0.0, 0.0, 1.0]]
            ),
            eye_heights=np.asarray([1.5]),
            view_positions=np.empty((0, 3), dtype=np.float64),
            source="test",
        )
        output = self.root / "candidate.png"
        result = _project_vertices(
            vertices,
            camera,
            output,
            vertical_axis=2,
            gaussian=False,
            image_size=64,
            oversample=1,
        )

        self.assertEqual(result.retained_vertices, 1024)
        self.assertTrue(output.is_file())
        rgb = np.asarray(Image.open(output).convert("RGB"))
        foreground = rgb[np.any(rgb < 240, axis=2)]
        self.assertGreater(len(foreground), 0)
        self.assertGreater(
            float(foreground[:, 0].mean()), float(foreground[:, 1].mean())
        )

    def test_ai2thor_map_render_hides_and_restores_ceiling(self) -> None:
        class Event:
            def __init__(self, metadata, frames=()):
                self.metadata = metadata
                self.third_party_camera_frames = list(frames)

        class Controller:
            def __init__(self):
                self.calls = []
                self.last_event = Event({})

            def step(self, *, action, **kwargs):
                self.calls.append((action, kwargs))
                if action == "ToggleMapView":
                    self.last_event = Event({"lastActionSuccess": True})
                elif action == "GetMapViewCameraProperties":
                    self.last_event = Event(
                        {
                            "actionReturn": {
                                "position": {"x": 1.0, "y": 3.0, "z": 2.0},
                                "rotation": {"x": 90.0, "y": 0.0, "z": 0.0},
                                "orthographic": True,
                                "orthographicSize": 4.0,
                            }
                        }
                    )
                elif action == "AddThirdPartyCamera":
                    self.last_event = Event(
                        {"lastActionSuccess": True},
                        [np.full((8, 8, 3), 123, dtype=np.uint8)],
                    )
                return self.last_event

        controller = Controller()
        rgb, pose = _render_topdown_mapview(controller)
        self.assertEqual(
            [call[0] for call in controller.calls],
            [
                "ToggleMapView",
                "GetMapViewCameraProperties",
                "AddThirdPartyCamera",
                "ToggleMapView",
            ],
        )
        self.assertEqual(rgb.shape, (8, 8, 3))
        self.assertEqual(pose["position"]["y"], 3.0)

    def test_gaussian_filter_uses_robust_floor_instead_of_eye_minus_two(self) -> None:
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("opacity", "<f4"),
                ("scale_0", "<f4"),
                ("scale_1", "<f4"),
                ("scale_2", "<f4"),
            ]
        )
        vertices = np.zeros(402, dtype=dtype)
        vertices["x"] = np.linspace(-1.0, 1.0, len(vertices))
        vertices["z"] = np.linspace(-1.0, 1.0, len(vertices))
        vertices["y"] = np.concatenate(
            [np.asarray([-100.0]), np.full(200, -4.0), np.zeros(200), [4.0]]
        )
        for name in ("scale_0", "scale_1", "scale_2"):
            vertices[name] = -2.0
        camera = CameraMetadata(
            c2w=np.eye(4),
            intrinsics=np.eye(3),
            eye_heights=np.asarray([1.7]),
            view_positions=np.empty((0, 3)),
            source="test",
        )

        selected, cutoffs = _select_gaussians(
            vertices,
            camera,
            ceiling_margin=0.25,
            lower_percentile=0.5,
            opacity_min=-4.0,
            max_log_scale=0.5,
        )
        heights = vertices["y"][selected]
        self.assertAlmostEqual(float(heights.min()), -4.0)
        self.assertLessEqual(float(heights.max()), 1.95)
        self.assertLess(cutoffs["lower_cutoff"], -3.9)

    def test_sh_basis_matches_renderer_order_for_axis_direction(self) -> None:
        basis = _sh_basis(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), 1)
        np.testing.assert_allclose(
            basis,
            np.asarray([[SH_C0, 0.0, SH_C1, 0.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
