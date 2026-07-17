#!/usr/bin/env python3

import os
import sys
import unittest


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from task2_vision.rest_pose import body_origin_z_from_table


class RestPoseHeightTest(unittest.TestCase):
    def test_class_body_origins_follow_local_table_height(self):
        plane = {"coefficients": [0.01, -0.02, -0.10]}
        xy = [0.30, -0.20]
        table_z = -0.093

        self.assertAlmostEqual(
            body_origin_z_from_table("part_type_a", xy, plane),
            table_z - 0.0009 - 0.0272,
        )
        self.assertAlmostEqual(
            body_origin_z_from_table("part_type_b", xy, plane),
            table_z - 0.0009 - 0.00004,
        )
        self.assertAlmostEqual(
            body_origin_z_from_table("part_type_c", xy, plane),
            table_z - 0.0009,
        )


if __name__ == "__main__":
    unittest.main()
