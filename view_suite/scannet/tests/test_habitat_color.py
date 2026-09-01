from __future__ import annotations

import unittest

import numpy as np

from view_suite.scannet.render.habitat_render import (
    _apply_color_lut,
    _is_top_down_camera,
    _make_color_lut,
)


class HabitatColorTest(unittest.TestCase):
    def test_baked_srgb_lut_lifts_midtones_without_changing_black_or_white(self):
        lut = _make_color_lut("srgb", (1.0, 1.07, 1.16))
        self.assertIsNotNone(lut)
        assert lut is not None
        image = np.asarray([[[0, 0, 0], [64, 64, 64], [255, 255, 255]]], np.uint8)

        transformed = _apply_color_lut(image, lut)

        np.testing.assert_array_equal(transformed[0, 0], [0, 0, 0])
        self.assertGreater(int(transformed[0, 1, 0]), 64)
        self.assertLess(int(transformed[0, 1, 0]), int(transformed[0, 1, 1]))
        self.assertLess(int(transformed[0, 1, 1]), int(transformed[0, 1, 2]))
        np.testing.assert_array_equal(transformed[0, 2], [255, 255, 255])

    def test_identity_transform_avoids_a_copy(self):
        self.assertIsNone(_make_color_lut("none", (1.0, 1.0, 1.0)))

    def test_invalid_transfer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_transfer"):
            _make_color_lut("aces", (1.0, 1.0, 1.0))

    def test_top_down_detection_uses_scannet_negative_z_forward(self):
        top_down = np.eye(4)
        top_down[:3, 2] = [0.0, 0.0, -1.0]
        perspective = np.eye(4)
        perspective[:3, 2] = [0.0, np.sqrt(0.75), -0.5]

        self.assertTrue(_is_top_down_camera(top_down))
        self.assertFalse(_is_top_down_camera(perspective))


if __name__ == "__main__":
    unittest.main()
