#!/usr/bin/env python3

import math
import os
import sys
import unittest

import numpy as np


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from task2_vision.pose_estimation import (
    angle_difference,
    bounded_yaw_grid,
    circular_median,
    orientation_xyzw_from_yaw,
    pca_yaw_hypotheses,
    rotation_from_yaw,
)


class PoseEstimationTest(unittest.TestCase):
    def test_pca_yaw_comes_only_from_observed_and_model_geometry(self):
        model = np.array(
            [
                [-0.04, -0.01, 0.0],
                [-0.04, 0.01, 0.0],
                [0.04, -0.01, 0.0],
                [0.04, 0.01, 0.0],
            ],
            dtype=np.float64,
        )
        expected = math.radians(37.0)
        observed = model.dot(rotation_from_yaw("part_type_a", expected).T)[:, :2]
        hypotheses, anisotropy = pca_yaw_hypotheses(
            "part_type_a", observed, model
        )
        self.assertGreater(anisotropy, 0.8)
        self.assertLess(
            min(abs(angle_difference(value, expected)) for value in hypotheses),
            1e-9,
        )

    def test_c_resting_orientation_is_sideways(self):
        yaw = math.radians(58.0)
        rotation = rotation_from_yaw("part_type_c", yaw)
        np.testing.assert_allclose(
            rotation,
            rotation_from_yaw("part_type_a", yaw).dot(
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
                )
            ),
            atol=1e-12,
        )
        quaternion = orientation_xyzw_from_yaw("part_type_c", yaw)
        self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0)

    def test_circular_median_crosses_pi_without_collapsing_to_zero(self):
        result = circular_median(
            [math.radians(179.0), math.radians(-179.0), math.radians(178.0)]
        )
        self.assertLess(abs(abs(math.degrees(result)) - 179.0), 1.1)

    def test_b_bounded_grid_never_contains_180_degree_branch(self):
        candidates = bounded_yaw_grid(
            math.radians(-60.0), math.radians(60.0), math.radians(10.0)
        )
        self.assertEqual(len(candidates), 13)
        self.assertAlmostEqual(math.degrees(candidates[0]), -60.0)
        self.assertAlmostEqual(math.degrees(candidates[-1]), 60.0)
        self.assertTrue(
            all(-60.0 <= math.degrees(value) <= 60.0 for value in candidates)
        )


if __name__ == "__main__":
    unittest.main()
