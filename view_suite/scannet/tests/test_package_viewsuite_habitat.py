from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from view_suite.envs.scannet_proxy_task.data_gen.package_viewsuite_habitat import (
    materialize_intermediate,
)


class PackageViewSuiteHabitatTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base = self.root / "base"
        self.intermediate = self.root / "intermediate"
        self.base.mkdir()
        self.intermediate.mkdir()

        base_image = self.base / "scene1/sample_000/view_0.png"
        intermediate_image = self.intermediate / "scene1/pose_0.png"
        base_image.parent.mkdir(parents=True)
        intermediate_image.parent.mkdir(parents=True)
        base_image.write_bytes(b"base-image")
        intermediate_image.write_bytes(b"intermediate-image")
        row = {
            "scene_id": "scene1",
            "image_path": ["scene1/sample_000/view_0.png"],
            "intermediate_image_path": ["scene1/pose_0.png"],
        }
        for split in ("train", "dev", "test"):
            (self.intermediate / f"path_to_view_{split}.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_intermediate_and_base_promotion_reports_keep_their_provenance(
        self,
    ) -> None:
        base_report = {"scope": "base", "jsonl_rows_updated": 60}
        intermediate_report = {"scope": "intermediate", "jsonl_rows_updated": 20}
        (self.base / "top_down_promotion.json").write_text(
            json.dumps(base_report), encoding="utf-8"
        )
        (self.intermediate / "top_down_promotion.json").write_text(
            json.dumps(intermediate_report), encoding="utf-8"
        )
        destination = self.root / "standalone"

        materialize_intermediate(
            task="path_to_view",
            base_root=self.base,
            intermediate_root=self.intermediate,
            destination=destination,
        )

        self.assertEqual(
            json.loads((destination / "top_down_promotion.json").read_text()),
            intermediate_report,
        )
        self.assertEqual(
            json.loads((destination / "top_down_base_promotion.json").read_text()),
            base_report,
        )

    def test_base_report_is_used_when_intermediate_report_is_absent(self) -> None:
        base_report = {"scope": "base"}
        (self.base / "top_down_promotion.json").write_text(
            json.dumps(base_report), encoding="utf-8"
        )
        destination = self.root / "standalone"

        materialize_intermediate(
            task="path_to_view",
            base_root=self.base,
            intermediate_root=self.intermediate,
            destination=destination,
        )

        self.assertEqual(
            json.loads((destination / "top_down_promotion.json").read_text()),
            base_report,
        )
        self.assertFalse((destination / "top_down_base_promotion.json").exists())


if __name__ == "__main__":
    unittest.main()
