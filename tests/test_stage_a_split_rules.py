import math
import unittest

from robotarm_sam.build_finetune_dataset import (
    compute_auto_val_scene_count,
    sample_indices_with_first_last,
    split_scenes,
)


class TestStageASplitRules(unittest.TestCase):
    def test_compute_auto_val_scene_count_bounds(self):
        self.assertEqual(compute_auto_val_scene_count(2, 0.2), 1)
        self.assertEqual(compute_auto_val_scene_count(3, 0.2), 1)
        self.assertEqual(compute_auto_val_scene_count(10, 0.2), 2)
        self.assertEqual(compute_auto_val_scene_count(10, 0.01), 1)
        self.assertEqual(compute_auto_val_scene_count(10, 1.0), 9)

    def test_compute_auto_val_scene_count_n_scene_lt_2(self):
        self.assertEqual(compute_auto_val_scene_count(0, 0.2), 0)
        self.assertEqual(compute_auto_val_scene_count(1, 0.2), 0)

    def test_split_scenes_auto_tail_pick_and_formula(self):
        scenes = [
            "task_a/scene1",
            "task_a/scene2",
            "task_b/scene01",
            "task_b/scene10",
            "task_b/scene2",
        ]

        plan = split_scenes(scenes, val_ratio=0.4, manual_val_scenes=None)
        self.assertEqual(plan.mode, "auto")
        self.assertEqual(plan.n_scene, 5)

        expected_val_count = min(max(1, math.ceil(0.4 * 5)), 4)
        self.assertEqual(plan.computed_val_scene_count, expected_val_count)
        self.assertEqual(len(plan.val_scenes), expected_val_count)

        # 核验尾部切分规则
        self.assertEqual(plan.train_scenes, plan.all_scenes[:-expected_val_count])
        self.assertEqual(plan.val_scenes, plan.all_scenes[-expected_val_count:])

    def test_split_scenes_auto_fails_when_n_scene_lt_2(self):
        scenes = ["task_only/scene1"]
        plan = split_scenes(scenes, val_ratio=0.2, manual_val_scenes=None)
        self.assertEqual(plan.mode, "auto")
        self.assertTrue(plan.auto_split_failed)
        self.assertEqual(plan.val_scenes, [])
        self.assertEqual(plan.train_scenes, ["task_only/scene1"])

    def test_split_scenes_manual_supports_scene_name_if_unique(self):
        scenes = [
            "task_a/scene1",
            "task_a/scene2",
            "task_b/scene3",
        ]
        plan = split_scenes(scenes, val_ratio=0.2, manual_val_scenes=["scene3"])
        self.assertEqual(plan.mode, "manual")
        self.assertEqual(plan.val_scenes, ["task_b/scene3"])
        self.assertNotIn("task_b/scene3", plan.train_scenes)

    def test_split_scenes_manual_requires_full_name_when_duplicated_scene(self):
        scenes = [
            "task_a/scene1",
            "task_b/scene1",
        ]
        with self.assertRaises(ValueError):
            split_scenes(scenes, val_ratio=0.2, manual_val_scenes=["scene1"])

    def test_sample_indices_keeps_first_last_and_stride(self):
        self.assertEqual(sample_indices_with_first_last(0, 30), [])
        self.assertEqual(sample_indices_with_first_last(1, 30), [0])
        self.assertEqual(sample_indices_with_first_last(2, 30), [0, 1])

        picked = sample_indices_with_first_last(65, 30)
        self.assertEqual(picked, [0, 30, 60, 64])

        with self.assertRaises(ValueError):
            sample_indices_with_first_last(10, 0)


if __name__ == "__main__":
    unittest.main()
