#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from task2_vision.cad_depth_alignment import CadDepthAligner


class CadDepthAlignmentTest(unittest.TestCase):
    @staticmethod
    def _make_aligner():
        vertices = np.array(
            [
                [-0.03, -0.02, -0.015],
                [0.03, -0.02, -0.015],
                [0.03, 0.02, -0.015],
                [-0.03, 0.02, -0.015],
                [-0.03, -0.02, 0.015],
                [0.03, -0.02, 0.015],
                [0.03, 0.02, 0.015],
                [-0.03, 0.02, 0.015],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
            ],
            dtype=np.int32,
        )
        return CadDepthAligner(
            {"box": vertices[faces]},
            sample_count=10000,
            maximum_iterations=8,
            minimum_overlap_ratio=0.20,
        )

    @staticmethod
    def _render_observation(
        aligner,
        true_xy,
        body_z,
        orientation,
        intrinsics,
        camera_to_base,
        roi,
        plane,
    ):
        rendering = aligner._project_visible(
            "box",
            true_xy,
            body_z,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
            [-2.0, 2.0],
        )
        return rendering["rendered_depth"], rendering["rendered_mask"]

    def test_recovers_lateral_xy_from_rendered_depth(self):
        aligner = self._make_aligner()
        orientation = np.eye(3)
        camera_to_base = (np.eye(3), np.zeros(3))
        intrinsics = (400.0, 400.0, 320.0, 240.0)
        roi = [250, 170, 390, 310]
        plane = {"coefficients": [0.0, 0.0, 0.0]}
        true_xy = np.array([0.04, -0.03])
        rendered, observed_mask = self._render_observation(
            aligner,
            true_xy,
            1.0,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
        )

        result = aligner.refine_xy(
            "box",
            [0.022, -0.045],
            1.0,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
            [-2.0, 2.0],
            rendered,
            observed_mask,
        )

        self.assertIsNotNone(result)
        error = np.abs(np.asarray(result["xy"]) - true_xy)
        self.assertLessEqual(float(np.max(error)), 0.0011)
        self.assertLessEqual(result["iterations"], 8)
        self.assertGreater(result["iou"], 0.95)

    def test_recovers_base_x_along_camera_depth_axis(self):
        aligner = self._make_aligner()
        orientation = np.eye(3)
        camera_to_base = (
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ]
            ),
            np.zeros(3),
        )
        intrinsics = (400.0, 400.0, 320.0, 240.0)
        roi = [240, 160, 390, 320]
        plane = {"coefficients": [0.0, 0.0, -1.0]}
        true_xy = np.array([1.0, 0.03])
        rendered, observed_mask = self._render_observation(
            aligner,
            true_xy,
            0.0,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
        )

        result = aligner.refine_xy(
            "box",
            [0.982, 0.015],
            0.0,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
            [-2.0, 2.0],
            rendered,
            observed_mask,
        )

        self.assertIsNotNone(result)
        error = np.abs(np.asarray(result["xy"]) - true_xy)
        self.assertLessEqual(float(np.max(error)), 0.0006)

    def test_region_only_still_constrains_flat_object_xy(self):
        aligner = self._make_aligner()
        orientation = np.eye(3)
        camera_to_base = (np.eye(3), np.zeros(3))
        intrinsics = (400.0, 400.0, 320.0, 240.0)
        roi = [250, 170, 390, 310]
        plane = {"coefficients": [0.0, 0.0, 0.0]}
        true_xy = np.array([0.04, -0.03])
        _, observed_mask = self._render_observation(
            aligner,
            true_xy,
            1.0,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
        )
        no_depth = np.full(observed_mask.shape, np.nan, dtype=np.float64)

        result = aligner.refine_xy(
            "box",
            [0.025, -0.042],
            1.0,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
            [-2.0, 2.0],
            no_depth,
            observed_mask,
        )

        self.assertIsNotNone(result)
        error = np.abs(np.asarray(result["xy"]) - true_xy)
        self.assertLessEqual(float(np.max(error)), 0.0015)
        self.assertEqual(result["depth_inliers"], 0)

    def test_quality_rejection_keeps_fit_diagnostics(self):
        aligner = self._make_aligner()
        orientation = np.eye(3)
        camera_to_base = (np.eye(3), np.zeros(3))
        intrinsics = (400.0, 400.0, 320.0, 240.0)
        roi = [250, 170, 390, 310]
        plane = {"coefficients": [0.0, 0.0, 0.0]}
        true_xy = np.array([0.04, -0.03])
        rendered, observed_mask = self._render_observation(
            aligner,
            true_xy,
            1.0,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
        )

        result = aligner.refine_xy(
            "box",
            [0.022, -0.045],
            1.0,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            plane,
            [-2.0, 2.0],
            rendered,
            observed_mask,
            minimum_overlap_ratio=1.01,
        )

        self.assertIsNotNone(result)
        self.assertFalse(result["accepted"])
        self.assertEqual(
            result["rejection_reason"],
            "bidirectional_coverage_below_threshold",
        )
        self.assertIn("xy", result)
        self.assertIn("iou", result)


if __name__ == "__main__":
    unittest.main()
