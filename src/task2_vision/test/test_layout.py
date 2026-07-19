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
    joint_yaw_prior_assignment,
    joint_yaw_window_assignment,
    yaw_prior_regularized_score,
    yaw_prior_window_candidates,
)


class LayoutAssignmentTest(unittest.TestCase):
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
            self.assertAlmostEqual(item["yaw_prior_common_offset"], common_offset)
            self.assertAlmostEqual(item["yaw_prior_match_error"], 0.0)

    def test_joint_prior_matching_uses_two_different_branches(self):
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
            {"roi_left": "part_type_b_1", "roi_right": "part_type_b_2"},
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
            by_prior.setdefault(prior_name, []).append(round(math.degrees(yaw), 6))

        self.assertEqual(
            by_prior["part_type_b_1"],
            [-40.053523, -30.053523, -20.053523, -10.053523, -0.053523],
        )
        self.assertEqual(
            by_prior["part_type_b_2"],
            [21.252961, 31.252961, 41.252961, 51.252961, 61.252961],
        )

    def test_joint_window_assignment_uses_complete_pair_cost(self):
        fits = {
            "left": {
                "part_type_b_1": {"accepted": True, "yaw_selection_score": 1.0},
                "part_type_b_2": {"accepted": True, "yaw_selection_score": 20.0},
            },
            "right": {
                "part_type_b_1": {"accepted": True, "yaw_selection_score": 2.0},
                "part_type_b_2": {"accepted": True, "yaw_selection_score": 3.0},
            },
        }

        result = joint_yaw_window_assignment("part_type_b", ["left", "right"], fits)

        self.assertEqual(result["selected_mode"], "direct")
        self.assertEqual(
            result["assignment"],
            {"left": "part_type_b_1", "right": "part_type_b_2"},
        )
        self.assertAlmostEqual(result["direct_cost"], 4.0)
        self.assertAlmostEqual(result["swapped_cost"], 22.0)

    def test_zero_prior_weight_uses_only_cad_score(self):
        center = math.radians(41.252961)
        center_score = yaw_prior_regularized_score(
            10.0, center, center, math.radians(8.0), weight=0.0
        )
        boundary_score = yaw_prior_regularized_score(
            7.0,
            center - math.radians(18.0),
            center,
            math.radians(8.0),
            weight=0.0,
        )

        self.assertEqual(center_score["prior_penalty"], 0.0)
        self.assertEqual(boundary_score["prior_penalty"], 0.0)
        self.assertEqual(center_score["objective_score"], 10.0)
        self.assertEqual(boundary_score["objective_score"], 7.0)


if __name__ == "__main__":
    unittest.main()
