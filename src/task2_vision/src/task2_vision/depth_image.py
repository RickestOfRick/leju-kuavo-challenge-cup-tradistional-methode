"""Shared Task2 compressed-depth decoding and metric conversion."""

import struct

import cv2
import numpy as np


DEFAULT_DEPTH_IMAGE_TOPIC = "/cam_h/depth/image_raw/compressedDepth"


def decode_compressed_depth_m(message):
    """Decode a ROS compressedDepth message to float32 meters."""
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError("empty compressed depth message")
    message_format = str(getattr(message, "format", ""))
    offsets = [12, 16, 0] if "compressedDepth" in message_format else [0, 12, 16]
    image = None
    for offset in offsets:
        if raw.size <= offset:
            continue
        image = cv2.imdecode(raw[offset:], cv2.IMREAD_UNCHANGED)
        if image is not None:
            break
    if image is None:
        raise ValueError("compressed depth PNG decode failed")
    if image.ndim == 3:
        image = image[:, :, 0]

    if "32FC1" in message_format and image.dtype == np.uint16:
        if len(message.data) < 12:
            raise ValueError("32FC1 compressedDepth header is incomplete")
        depth_quant_a, depth_quant_b = struct.unpack("<ff", message.data[4:12])
        raw_depth = image.astype(np.float32)
        depth = np.zeros(raw_depth.shape, dtype=np.float32)
        valid = raw_depth > 0.0
        denominator = raw_depth[valid] - float(depth_quant_b)
        denominator_valid = np.abs(denominator) > 1e-6
        values = np.zeros_like(denominator, dtype=np.float32)
        values[denominator_valid] = (
            float(depth_quant_a) / denominator[denominator_valid]
        )
        depth[valid] = values
        return depth
    if image.dtype == np.uint16:
        return image.astype(np.float32) * 0.001
    return image.astype(np.float32)


def depth_m_to_uint16_mm(depth_m):
    """Convert metric depth to lossless millimeter labels, with zero invalid."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(
        np.rint(depth[valid] * 1000.0), 1.0, 65535.0
    ).astype(np.uint16)
    return depth_mm
