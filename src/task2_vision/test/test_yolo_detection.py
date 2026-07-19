#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from task2_vision.yolo_detection import (
    build_task2_detections,
    has_complete_task2_detections,
    normalize_model_names,
    remap_fitted_detections_by_assignment,
    task2_detection_counts,
)


class Task2YoloDetectionTest(unittest.TestCase):
    @staticmethod
    def _complete_detection_and_fit_maps():
        detections = {}
        fits = {}
        for class_name in ("part_type_a", "part_type_b", "part_type_c"):
            for index in (1, 2):
                name = "{}_{}".format(class_name, index)
                detections[name] = {"name": name, "class_name": class_name}
                fits[name] = {
                    "xy": [0.3, 0.2 if index == 1 else -0.2],
                    "z": 0.0,
                    "yaw": 0.0,
                    "token": name,
                }
        return detections, fits

    def test_selects_two_highest_confidence_boxes_per_class(self):
        boxes = []
        scores = []
        classes = []
        for class_id in range(3):
            for index, score in enumerate((0.4, 0.9, 0.7)):
                x1 = 10.0 + class_id * 100.0 + index * 10.0
                boxes.append([x1, 20.0, x1 + 20.0, 60.0])
                scores.append(score)
                classes.append(class_id)
        detections = build_task2_detections(
            np.asarray(boxes),
            np.asarray(scores),
            np.asarray(classes),
            (100, 400),
            (200, 800),
        )
        self.assertTrue(has_complete_task2_detections(detections))
        self.assertEqual(
            task2_detection_counts(detections),
            {"part_type_a": 2, "part_type_b": 2, "part_type_c": 2},
        )
        self.assertEqual({item["confidence"] for item in detections}, {0.7, 0.9})
        self.assertEqual(detections[0]["name"], "part_type_a_1")
        self.assertEqual(detections[-1]["name"], "part_type_c_2")
        self.assertEqual(detections[0]["bbox_size"], [40.0, 80.0])

    def test_incomplete_class_count_is_reported(self):
        detections = build_task2_detections(
            [[0, 0, 10, 10]], [0.8], [0], (100, 100), (100, 100)
        )
        self.assertFalse(has_complete_task2_detections(detections))

    def test_model_class_names_are_strict(self):
        self.assertEqual(
            normalize_model_names(
                {0: "part_type_a", 1: "part_type_b", 2: "part_type_c"}
            )[2],
            "part_type_c",
        )
        with self.assertRaises(ValueError):
            normalize_model_names({0: "a", 1: "b", 2: "c"})

    def test_assignment_rekeys_detection_and_fit_together(self):
        detections, fits = self._complete_detection_and_fit_maps()
        remapped_detections, remapped_fits = remap_fitted_detections_by_assignment(
            detections,
            fits,
            {
                "part_type_b_1": "part_type_b_2",
                "part_type_b_2": "part_type_b_1",
            },
            "yaw_prior",
        )
        self.assertEqual(
            remapped_detections["part_type_b_1"]["detection_name"],
            "part_type_b_2",
        )
        self.assertEqual(remapped_fits["part_type_b_1"]["token"], "part_type_b_2")
        self.assertEqual(remapped_fits["part_type_b_2"]["token"], "part_type_b_1")

    def test_failed_a_does_not_block_b_and_c_assignment(self):
        detections, fits = self._complete_detection_and_fit_maps()
        fits["part_type_a_1"] = {"accepted": False, "token": "failed_a1"}
        fits["part_type_a_2"] = {"accepted": False, "token": "failed_a2"}
        _, remapped = remap_fitted_detections_by_assignment(
            detections,
            fits,
            {
                "part_type_b_1": "part_type_b_2",
                "part_type_b_2": "part_type_b_1",
                "part_type_c_1": "part_type_c_2",
                "part_type_c_2": "part_type_c_1",
            },
            "yaw_prior",
        )
        self.assertEqual(remapped["part_type_b_1"]["token"], "part_type_b_2")
        self.assertEqual(remapped["part_type_c_1"]["token"], "part_type_c_2")
        self.assertEqual(remapped["part_type_a_1"]["token"], "failed_a1")


if __name__ == "__main__":
    unittest.main()
