from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from view_suite.envs.habitat_gs_proxy_task.data_gen.promote_top_down import (
    load_reviewed_candidates,
    promote_root,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PromoteTopDownTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = self.root / "dataset"
        self.review = self.root / "review"
        self.dataset.mkdir()
        self.review.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _fixtures(self) -> tuple[Path, Path, Path, Path]:
        original = self.dataset / "scene1/top_down.png"
        candidate = self.review / "round_2/scene1.png"
        original.parent.mkdir(parents=True)
        candidate.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), (200, 0, 0)).save(original)
        Image.new("RGB", (8, 8), (0, 200, 0)).save(candidate)
        old_pose = [[1.0, 0.0], [0.0, 1.0]]
        old_intrinsics = [[1.0, 0.0], [0.0, 1.0]]
        row = {
            "scene_id": "scene1",
            "image_detail": {
                "top_down_view": {
                    "path": "scene1/top_down.png",
                    "c2w_extrinsics": old_pose,
                    "c2w_intrinsics": old_intrinsics,
                }
            },
        }
        (self.dataset / "interactive_view_planning_train.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        sample = self.dataset / "scene1/sample_000/meta.json"
        sample.parent.mkdir(parents=True)
        sample.write_text(
            json.dumps(
                {
                    "top_down": {
                        "image": "../top_down.png",
                        "pose_c2w": old_pose,
                        "intrinsics": old_intrinsics,
                    }
                }
            ),
            encoding="utf-8",
        )
        new_pose = [[2.0, 0.0], [0.0, 2.0]]
        new_intrinsics = [[3.0, 0.0], [0.0, 3.0]]
        item = {
            "id": "test::scene1",
            "corpus": "test",
            "scene_id": "scene1",
            "image_path": str(candidate),
            "lineage": {
                "round": 2,
                "parent_sha256": _sha256(original),
                "parent_image_rel": "scene1/top_down.png",
                "method": "test",
                "camera_pose": new_pose,
                "camera_intrinsics": new_intrinsics,
            },
        }
        manifest = self.review / "round_2_manifest.json"
        manifest.write_text(json.dumps({"round": 2, "items": [item]}))
        labels = self.review / "round_2_labels.json"
        labels.write_text(
            json.dumps(
                {
                    "review_round": 2,
                    "items": [
                        {
                            **item,
                            "image_rel": "round_2/scene1.png",
                            "sha256": _sha256(candidate),
                            "verdict": "keep",
                        }
                    ],
                }
            )
        )
        return labels, manifest, original, candidate

    def test_apply_replaces_image_and_updates_metadata(self) -> None:
        labels, manifest, original, candidate = self._fixtures()
        candidates, rejected = load_reviewed_candidates(labels, manifest)
        report = promote_root(
            corpus="test",
            root=self.dataset,
            candidates=candidates,
            rejected=rejected,
            labels_path=labels,
            manifest_path=manifest,
            apply=True,
        )

        self.assertEqual(original.read_bytes(), candidate.read_bytes())
        row = json.loads(
            (self.dataset / "interactive_view_planning_train.jsonl").read_text()
        )
        self.assertEqual(
            row["image_detail"]["top_down_view"]["c2w_intrinsics"],
            [[3.0, 0.0], [0.0, 3.0]],
        )
        sidecar = json.loads((self.dataset / "scene1/sample_000/meta.json").read_text())
        self.assertEqual(sidecar["top_down"]["pose_c2w"], [[2.0, 0.0], [0.0, 2.0]])
        self.assertEqual(report["image_replaced"], 1)
        self.assertEqual(report["jsonl_rows_updated"], 1)
        self.assertEqual(report["sidecar_files_updated"], 1)
        self.assertTrue((self.dataset / "top_down_promotion.json").is_file())

    def test_dry_run_is_read_only_and_apply_is_idempotent(self) -> None:
        labels, manifest, original, _ = self._fixtures()
        original_bytes = original.read_bytes()
        candidates, rejected = load_reviewed_candidates(labels, manifest)
        dry_run = promote_root(
            corpus="test",
            root=self.dataset,
            candidates=candidates,
            rejected=rejected,
            labels_path=labels,
            manifest_path=manifest,
            apply=False,
        )
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(dry_run["image_replaced"], 1)
        promote_root(
            corpus="test",
            root=self.dataset,
            candidates=candidates,
            rejected=rejected,
            labels_path=labels,
            manifest_path=manifest,
            apply=True,
        )
        repeated = promote_root(
            corpus="test",
            root=self.dataset,
            candidates=candidates,
            rejected=rejected,
            labels_path=labels,
            manifest_path=manifest,
            apply=True,
        )
        self.assertEqual(repeated["image_replaced"], 0)
        self.assertEqual(repeated["image_already_current"], 1)
        self.assertEqual(repeated["jsonl_rows_updated"], 0)

    def test_rejects_diverged_destination_by_default(self) -> None:
        labels, manifest, original, _ = self._fixtures()
        Image.new("RGB", (8, 8), (0, 0, 200)).save(original)
        candidates, rejected = load_reviewed_candidates(labels, manifest)
        with self.assertRaisesRegex(ValueError, "destination changed since review"):
            promote_root(
                corpus="test",
                root=self.dataset,
                candidates=candidates,
                rejected=rejected,
                labels_path=labels,
                manifest_path=manifest,
                apply=False,
            )


if __name__ == "__main__":
    unittest.main()
