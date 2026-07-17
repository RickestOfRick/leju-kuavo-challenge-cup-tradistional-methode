#!/usr/bin/env python3

import math
import os
import sys
import unittest


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from task2_vision.layout import (
    INSTANCE_NAMING_YAW_PRIOR,
    OBJECT_NAMES,
    YAW_PRIORS_BY_MOTION_NAME,
    assign_motion_names,
    build_published_objects,
    joint_yaw_prior_assignment,
    joint_yaw_window_assignment,
    update_published_objects_base_frame,
    yaw_prior_regularized_score,
    yaw_prior_window_candidates,
)


class DynamicTruthLayoutTest(unittest.TestCase):
    @staticmethod
    def _yaw_quat_wxyz(yaw):
        half = 0.5 * float(yaw)
        return [math.cos(half), 0.0, 0.0, math.sin(half)]

    @staticmethod
    def _layout():
        layout = {}
        for index, name in enumerate(OBJECT_NAMES):
            layout[name] = {
                "pos": [0.30, -0.30 + 0.12 * index, 0.80],
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
        return layout

    def test_current_base_translation_updates_truth_without_renaming(self):
        initial = build_published_objects(
            self._layout(), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )
        updated = update_published_objects_base_frame(
            initial,
            [0.02, -0.01, 0.005],
            [1.0, 0.0, 0.0, 0.0],
        )

        self.assertEqual(
            [item["name"] for item in updated],
            [item["name"] for item in initial],
        )
        for before, after in zip(initial, updated):
            self.assertAlmostEqual(after["position"][0], before["position"][0] - 0.02)
            self.assertAlmostEqual(after["position"][1], before["position"][1] + 0.01)
            self.assertAlmostEqual(after["position"][2], before["position"][2] - 0.005)
            self.assertEqual(
                after["truth_reference"], "current_mujoco_qpos_base_link"
            )

    def test_current_body_pose_overrides_startup_layout_pose(self):
        initial = build_published_objects(
            self._layout(), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )
        target = initial[0]
        updated = update_published_objects_base_frame(
            initial,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            {
                target["ground_truth_name"]: (
                    [0.41, -0.22, 0.83],
                    [1.0, 0.0, 0.0, 0.0],
                )
            },
        )

        self.assertEqual(updated[0]["position"], [0.41, -0.22, 0.83])
        self.assertEqual(updated[0]["world_position"], [0.41, -0.22, 0.83])

    def test_motion_names_use_opposite_ab_and_c_numbering_rules(self):
        layout = self._layout()
        layout["part_type_a_1"]["pos"][1] = 0.20
        layout["part_type_a_2"]["pos"][1] = -0.20
        layout["part_type_b_1"]["pos"][1] = 0.25
        layout["part_type_b_2"]["pos"][1] = -0.25
        layout["part_type_c_1"]["pos"][1] = -0.30
        layout["part_type_c_2"]["pos"][1] = 0.30

        published = build_published_objects(
            layout, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )

        self.assertEqual([item["name"] for item in published], list(OBJECT_NAMES))
        self.assertEqual(
            {
                item["name"]: item["ground_truth_name"]
                for item in published
            },
            {
                "part_type_a_1": "part_type_a_2",
                "part_type_a_2": "part_type_a_1",
                "part_type_b_1": "part_type_b_2",
                "part_type_b_2": "part_type_b_1",
                "part_type_c_1": "part_type_c_2",
                "part_type_c_2": "part_type_c_1",
            },
        )

    def test_yaw_prior_names_ignore_position_and_common_frame_offset(self):
        common_offset = 0.37
        observations = []
        expected_sources = {}
        for index, motion_name in enumerate(reversed(OBJECT_NAMES)):
            class_name = motion_name.rsplit("_", 1)[0]
            source_name = "roi_{}".format(index)
            observations.append(
                {
                    "name": source_name,
                    "ground_truth_name": source_name,
                    "class_name": class_name,
                    "position": [0.3, -0.5 + 0.2 * index, 0.0],
                    "yaw": YAW_PRIORS_BY_MOTION_NAME[motion_name]
                    + common_offset,
                }
            )
            expected_sources[motion_name] = source_name

        assigned = assign_motion_names(observations, INSTANCE_NAMING_YAW_PRIOR)

        self.assertEqual([item["name"] for item in assigned], list(OBJECT_NAMES))
        self.assertEqual(
            {
                item["name"]: item["instance_naming_source_name"]
                for item in assigned
            },
            expected_sources,
        )
        for item in assigned:
            self.assertAlmostEqual(
                item["yaw_prior_common_offset"], common_offset
            )
            self.assertAlmostEqual(item["yaw_prior_match_error"], 0.0)

    def test_mode0_build_can_name_directly_from_truth_yaw(self):
        layout = self._layout()
        for name in OBJECT_NAMES:
            layout[name]["quat"] = self._yaw_quat_wxyz(
                YAW_PRIORS_BY_MOTION_NAME[name]
            )
        layout["part_type_a_1"]["pos"][1] = 0.20
        layout["part_type_a_2"]["pos"][1] = -0.20
        layout["part_type_b_1"]["pos"][1] = 0.25
        layout["part_type_b_2"]["pos"][1] = -0.25
        layout["part_type_c_1"]["pos"][1] = -0.30
        layout["part_type_c_2"]["pos"][1] = 0.30

        published = build_published_objects(
            layout,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            INSTANCE_NAMING_YAW_PRIOR,
        )

        self.assertEqual(
            {
                item["name"]: item["ground_truth_name"]
                for item in published
            },
            {name: name for name in OBJECT_NAMES},
        )

    def test_joint_prior_matching_cannot_assign_both_objects_to_same_yaw(self):
        fits = {
            "roi_left": {
                "yaw_prior_candidate_scores": {
                    "part_type_b_1": 1.0,
                    "part_type_b_2": 20.0,
                }
            },
            "roi_right": {
                "yaw_prior_candidate_scores": {
                    "part_type_b_1": 2.0,
                    "part_type_b_2": 3.0,
                }
            },
        }

        assignment = joint_yaw_prior_assignment(
            "part_type_b", ["roi_left", "roi_right"], fits
        )

        self.assertEqual(
            assignment,
            {
                "roi_left": "part_type_b_1",
                "roi_right": "part_type_b_2",
            },
        )

    def test_relative_mode_searches_both_prior_windows(self):
        candidates = yaw_prior_window_candidates(
            "part_type_b",
            frame_yaw_offset=0.0,
            half_width=math.radians(20.0),
            step=math.radians(10.0),
        )
        by_prior = {}
        for prior_name, yaw in candidates:
            by_prior.setdefault(prior_name, []).append(
                round(math.degrees(yaw), 6)
            )

        self.assertEqual(
            by_prior["part_type_b_1"],
            [-40.053523, -30.053523, -20.053523, -10.053523, -0.053523],
        )
        self.assertEqual(
            by_prior["part_type_b_2"],
            [21.252961, 31.252961, 41.252961, 51.252961, 61.252961],
        )

    def test_joint_window_assignment_prevents_same_branch_for_both_objects(self):
        fits = {
            "part_type_b_1": {
                "part_type_b_1": {
                    "accepted": True,
                    "yaw_selection_score": 1.0,
                },
                "part_type_b_2": {
                    "accepted": True,
                    "yaw_selection_score": 20.0,
                },
            },
            "part_type_b_2": {
                "part_type_b_1": {
                    "accepted": True,
                    "yaw_selection_score": 2.0,
                },
                "part_type_b_2": {
                    "accepted": True,
                    "yaw_selection_score": 3.0,
                },
            },
        }

        result = joint_yaw_window_assignment(
            "part_type_b", ["part_type_b_1", "part_type_b_2"], fits
        )

        self.assertEqual(result["selected_mode"], "direct")
        self.assertEqual(
            result["assignment"],
            {
                "part_type_b_1": "part_type_b_1",
                "part_type_b_2": "part_type_b_2",
            },
        )
        self.assertAlmostEqual(result["direct_cost"], 4.0)
        self.assertAlmostEqual(result["swapped_cost"], 22.0)
        self.assertAlmostEqual(result["margin"], 18.0)

    def test_joint_window_assignment_uses_only_complete_valid_pairing(self):
        fits = {
            "left": {
                "part_type_b_1": {
                    "accepted": False,
                    "yaw_selection_score": 1.0,
                },
                "part_type_b_2": {
                    "accepted": True,
                    "yaw_selection_score": 4.0,
                },
            },
            "right": {
                "part_type_b_1": {
                    "accepted": True,
                    "yaw_selection_score": 3.0,
                },
                "part_type_b_2": {
                    "accepted": True,
                    "yaw_selection_score": 2.0,
                },
            },
        }

        result = joint_yaw_window_assignment(
            "part_type_b", ["left", "right"], fits
        )

        self.assertEqual(result["selected_mode"], "swapped")
        self.assertEqual(
            result["assignment"],
            {"left": "part_type_b_2", "right": "part_type_b_1"},
        )
        self.assertIsNone(result["direct_cost"])
        self.assertIsNone(result["margin"])

    def test_yaw_prior_regularization_is_only_a_weak_tie_breaker(self):
        center = math.radians(41.252961)
        center_score = yaw_prior_regularized_score(
            10.0, center, center, math.radians(8.0), weight=0.1
        )
        boundary_score = yaw_prior_regularized_score(
            7.0,
            center - math.radians(18.0),
            center,
            math.radians(8.0),
            weight=0.1,
        )

        self.assertAlmostEqual(center_score["prior_penalty"], 0.0)
        self.assertAlmostEqual(boundary_score["prior_penalty"], 0.50625)
        self.assertLess(boundary_score["objective_score"], 10.0)


if __name__ == "__main__":
    unittest.main()
