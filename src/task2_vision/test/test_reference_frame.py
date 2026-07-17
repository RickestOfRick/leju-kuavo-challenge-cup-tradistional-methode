#!/usr/bin/env python3

import math
import os
import sys
import unittest

import numpy as np


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from task2_vision.reference_frame import (
    transform_point_between_base_payloads,
    transform_rotation_between_base_payloads,
)


class ReferenceFrameTest(unittest.TestCase):
    @staticmethod
    def _payload(position, yaw_deg=0.0):
        half = 0.5 * math.radians(float(yaw_deg))
        return {
            "world_base_position": list(position),
            "world_base_quat_wxyz": [
                math.cos(half),
                0.0,
                0.0,
                math.sin(half),
            ],
        }

    def test_translation_to_earlier_mode0_reference(self):
        current = self._payload([0.02, -0.01, 0.0])
        mode0_reference = self._payload([0.0, 0.0, 0.0])
        result = transform_point_between_base_payloads(
            [0.30, 0.20, -0.10], current, mode0_reference
        )
        np.testing.assert_allclose(result, [0.32, 0.19, -0.10], atol=1e-12)

    def test_rotation_is_not_treated_as_translation_only(self):
        current = self._payload([0.0, 0.0, 0.0], yaw_deg=90.0)
        mode0_reference = self._payload([0.0, 0.0, 0.0])
        result = transform_point_between_base_payloads(
            [1.0, 0.0, 0.0], current, mode0_reference
        )
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-12)

    def test_identity_snapshot_keeps_point(self):
        snapshot = self._payload([0.3, -0.2, 0.8], yaw_deg=-17.0)
        point = [0.31, 0.07, -0.12]
        result = transform_point_between_base_payloads(point, snapshot, snapshot)
        np.testing.assert_allclose(result, point, atol=1e-12)

    def test_orientation_follows_base_snapshot_rotation(self):
        current = self._payload([0.0, 0.0, 0.0], yaw_deg=30.0)
        mode0_reference = self._payload([0.0, 0.0, 0.0], yaw_deg=-10.0)
        result = transform_rotation_between_base_payloads(
            np.eye(3), current, mode0_reference
        )
        expected_yaw = math.radians(40.0)
        np.testing.assert_allclose(
            result,
            np.array(
                [
                    [math.cos(expected_yaw), -math.sin(expected_yaw), 0.0],
                    [math.sin(expected_yaw), math.cos(expected_yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
