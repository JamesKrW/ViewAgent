from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from view_suite.envs.habitat_gs_proxy_task.data_gen.review_top_down import (
    ReviewStore,
    collect_corpora,
    collect_top_downs,
    load_candidate_manifest,
)


class ReviewTopDownTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _image(self, relative: str, color: tuple[int, int, int]) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color).save(path)
        return path

    def _jsonl(self, filename: str, rows: list[tuple[str, str]]) -> None:
        content = []
        for index, (scene_id, topdown_path) in enumerate(rows):
            content.append(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "sample_id": f"{scene_id}_{index}",
                        "image_detail": {
                            "top_down_view": {"path": topdown_path},
                        },
                    }
                )
            )
        (self.root / filename).write_text("\n".join(content) + "\n", encoding="utf-8")

    def test_collects_all_tasks_and_deduplicates_scene_content(self):
        scene_a = self._image("scene_a/top_down.png", (255, 0, 0))
        duplicate = self._image("scene_a/top_down_copy.png", (255, 0, 0))
        self._image("scene_b/top_down.png", (0, 255, 0))
        self._image("scene_c/top_down.png", (0, 0, 255))  # orphan

        self._jsonl(
            "path_to_view_train.jsonl",
            [("scene_a", str(scene_a.relative_to(self.root)))] * 2
            + [("scene_b", "scene_b/top_down.png")],
        )
        self._jsonl(
            "view_to_path_test.jsonl",
            [("scene_a", str(duplicate.relative_to(self.root)))],
        )
        self._jsonl(
            "interactive_view_planning.jsonl",
            [("scene_a", "scene_a/top_down.png")],
        )

        inventory = collect_top_downs(self.root, corpus="test")
        self.assertEqual(len(inventory.items), 3)
        self.assertEqual(
            {item.scene_id for item in inventory.items},
            {"scene_a", "scene_b", "scene_c"},
        )
        scene_a_item = next(
            item for item in inventory.items if item.scene_id == "scene_a"
        )
        self.assertEqual(scene_a_item.variant_count, 1)
        self.assertEqual(len(scene_a_item.equivalent_paths), 2)
        self.assertEqual(sum(source.rows for source in scene_a_item.sources), 4)
        self.assertEqual(len(inventory.jsonls_scanned), 3)

    def test_conflicting_images_become_explicit_variants(self):
        self._image("scene_a/top_down.png", (255, 0, 0))
        self._image("scene_a/other_top_down.png", (0, 255, 0))
        self._jsonl("path_to_view.jsonl", [("scene_a", "scene_a/top_down.png")])
        self._jsonl("view_to_path.jsonl", [("scene_a", "scene_a/other_top_down.png")])

        inventory = collect_top_downs(self.root, corpus="test", include_orphans=False)
        self.assertEqual(len(inventory.items), 2)
        self.assertEqual(
            {item.id for item in inventory.items},
            {"test::scene_a::v1", "test::scene_a::v2"},
        )
        self.assertTrue(all(item.variant_count == 2 for item in inventory.items))

    def test_labels_persist_and_are_invalidated_when_image_changes(self):
        image = self._image("scene_a/top_down.png", (255, 0, 0))
        self._jsonl(
            "interactive_view_planning.jsonl", [("scene_a", "scene_a/top_down.png")]
        )
        output = self.root / "labels.json"

        inventory = collect_top_downs(self.root, corpus="test")
        store = ReviewStore(inventory, output)
        stored = store.update("test::scene_a", "reject", "off-manifold smear")
        self.assertEqual(stored["verdict"], "reject")
        self.assertTrue(output.is_file())
        self.assertTrue(output.with_suffix(".csv").is_file())

        resumed = ReviewStore(collect_top_downs(self.root, corpus="test"), output)
        self.assertEqual(resumed.item_dict("test::scene_a")["verdict"], "reject")
        self.assertEqual(
            resumed.item_dict("test::scene_a")["note"], "off-manifold smear"
        )

        Image.new("RGB", (16, 16), (10, 20, 30)).save(image)
        regenerated = ReviewStore(collect_top_downs(self.root, corpus="test"), output)
        item = regenerated.item_dict("test::scene_a")
        self.assertIsNone(item["verdict"])
        self.assertEqual(item["stale_previous_verdict"], "reject")

    def test_same_scene_name_is_not_merged_across_corpora(self):
        roots = {"first": self.root / "first", "second": self.root / "second"}
        for index, root in enumerate(roots.values()):
            image = root / "shared_scene/top_down.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (index * 100, 20, 30)).save(image)
            row = {
                "scene_id": "shared_scene",
                "sample_id": f"sample_{index}",
                "image_detail": {
                    "top_down_view": {"path": "shared_scene/top_down.png"}
                },
            }
            (root / "interactive_view_planning.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )

        inventory = collect_corpora(roots)
        self.assertEqual(len(inventory.items), 2)
        self.assertEqual(
            {item.id for item in inventory.items},
            {"first::shared_scene", "second::shared_scene"},
        )

    def test_round_two_manifest_preserves_lineage(self):
        candidate = self._image("round_2/ai2thor/FloorPlan1.png", (1, 2, 3))
        manifest = self.root / "round_2_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "round": 2,
                    "items": [
                        {
                            "id": "ai2thor::FloorPlan1",
                            "corpus": "ai2thor",
                            "scene_id": "FloorPlan1",
                            "image_path": str(candidate.relative_to(self.root)),
                            "lineage": {
                                "parent_sha256": "old-hash",
                                "failure_reason": "blur",
                                "method": "regenerated-test",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        inventory = load_candidate_manifest(manifest, expected_round=2)
        self.assertEqual(len(inventory.items), 1)
        self.assertEqual(inventory.items[0].lineage["round"], 2)
        output = self.root / "round_2_labels.json"
        store = ReviewStore(
            inventory,
            output,
            review_round=2,
            source_manifest=manifest,
            parent_labels=self.root / "round_1_labels.json",
        )
        payload = store.payload()
        self.assertEqual(payload["review_round"], 2)
        self.assertEqual(payload["total_rounds"], 3)
        self.assertEqual(payload["items"][0]["lineage"]["parent_sha256"], "old-hash")

    def test_non_finite_lineage_is_persisted_as_standard_json_null(self):
        candidate = self._image("round_2/habitat_gs/scene01.png", (1, 2, 3))
        manifest = self.root / "round_2_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "round": 2,
                    "items": [
                        {
                            "id": "habitat_gs::scene01",
                            "corpus": "habitat_gs",
                            "scene_id": "scene01",
                            "image_path": str(candidate.relative_to(self.root)),
                            "lineage": {"tube_radius": float("nan")},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        inventory = load_candidate_manifest(manifest, expected_round=2)
        output = self.root / "round_2_labels.json"
        store = ReviewStore(inventory, output, review_round=2)
        store.update("habitat_gs::scene01", "keep", "")

        payload = json.loads(
            output.read_text(encoding="utf-8"),
            parse_constant=lambda value: self.fail(
                f"non-standard JSON constant persisted: {value}"
            ),
        )
        self.assertIsNone(payload["items"][0]["lineage"]["tube_radius"])


if __name__ == "__main__":
    unittest.main()
