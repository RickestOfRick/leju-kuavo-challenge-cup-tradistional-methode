"""Color masks shared by task 2 RGB-D fitting."""

import cv2
import numpy as np


def green_table_mask(
    color_bgr,
    output_shape,
    hsv_lower,
    hsv_upper,
    dilate_pixels=1,
    chroma_min=0.38,
    chroma_dominance_min=0.04,
    shadow_min_intensity_sum=3.0,
):
    color_bgr = np.asarray(color_bgr)
    if color_bgr.ndim != 3 or color_bgr.shape[2] < 3:
        raise ValueError("color image must be BGR with three channels")
    height, width = [int(value) for value in output_shape[:2]]
    if height <= 0 or width <= 0:
        raise ValueError("output shape must be non-empty")
    if color_bgr.shape[:2] != (height, width):
        color_bgr = cv2.resize(
            color_bgr, (width, height), interpolation=cv2.INTER_NEAREST
        )
    hsv = cv2.cvtColor(color_bgr[:, :, :3], cv2.COLOR_BGR2HSV)
    lower = np.asarray(hsv_lower, dtype=np.uint8).reshape(3)
    upper = np.asarray(hsv_upper, dtype=np.uint8).reshape(3)
    hsv_green = cv2.inRange(hsv, lower, upper) > 0

    bgr = color_bgr[:, :, :3].astype(np.float32)
    blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    intensity_sum = red + green + blue
    safe_sum = np.maximum(intensity_sum, 1.0)
    red_chroma = red / safe_sum
    green_chroma = green / safe_sum
    blue_chroma = blue / safe_sum
    green_dominance = green_chroma - np.maximum(red_chroma, blue_chroma)
    shadow_green = (
        (intensity_sum >= float(shadow_min_intensity_sum))
        & (green_chroma >= float(chroma_min))
        & (green_dominance >= float(chroma_dominance_min))
    )
    mask = hsv_green | shadow_green
    radius = max(0, int(dilate_pixels))
    if radius > 0:
        size = 2 * radius + 1
        mask = cv2.dilate(
            mask.astype(np.uint8), np.ones((size, size), dtype=np.uint8)
        ) > 0
    return mask
