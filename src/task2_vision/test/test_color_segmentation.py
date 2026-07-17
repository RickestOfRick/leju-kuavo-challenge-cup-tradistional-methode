#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from task2_vision.color_segmentation import green_table_mask


class ColorSegmentationTest(unittest.TestCase):
    def test_strict_green_table_rejection_with_edge_dilation(self):
        image = np.full((5, 5, 3), [190, 190, 190], dtype=np.uint8)
        image[2, 2] = [26, 77, 26]
        image[0, 0] = [20, 20, 180]

        mask = green_table_mask(image, image.shape, [35, 40, 10], [90, 255, 255], 1)

        self.assertEqual(int(np.count_nonzero(mask)), 9)
        self.assertTrue(bool(mask[2, 2]))
        self.assertFalse(bool(mask[0, 0]))
        self.assertFalse(bool(mask[4, 4]))

    def test_resizes_color_mask_to_depth_resolution(self):
        image = np.full((4, 4, 3), [26, 77, 26], dtype=np.uint8)
        mask = green_table_mask(image, (2, 3), [35, 40, 10], [90, 255, 255], 0)
        self.assertEqual(mask.shape, (2, 3))
        self.assertTrue(bool(np.all(mask)))

    def test_dark_green_shadow_is_rejected_but_gray_and_black_parts_are_not(self):
        image = np.array(
            [
                [
                    [1, 4, 1],
                    [77, 97, 61],
                    [8, 8, 8],
                    [2, 2, 2],
                    [10, 10, 80],
                ]
            ],
            dtype=np.uint8,
        )

        mask = green_table_mask(
            image,
            image.shape,
            [35, 40, 10],
            [90, 255, 255],
            dilate_pixels=0,
            chroma_min=0.38,
            chroma_dominance_min=0.04,
            shadow_min_intensity_sum=3.0,
        )

        self.assertEqual(mask.tolist(), [[True, True, False, False, False]])


if __name__ == "__main__":
    unittest.main()
