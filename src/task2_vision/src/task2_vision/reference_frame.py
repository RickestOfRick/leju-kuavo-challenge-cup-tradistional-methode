"""Rigid transforms between two base_link snapshots."""

import math

import numpy as np


def _quaternion_wxyz_to_matrix(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("base quaternion norm is zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_point_between_base_payloads(point, source_payload, target_payload):
    """Transform a point from source base_link into target base_link."""
    source_position = np.asarray(
        source_payload.get("world_base_position", []), dtype=np.float64
    )
    target_position = np.asarray(
        target_payload.get("world_base_position", []), dtype=np.float64
    )
    source_quaternion = source_payload.get("world_base_quat_wxyz", [])
    target_quaternion = target_payload.get("world_base_quat_wxyz", [])
    if (
        source_position.shape != (3,)
        or target_position.shape != (3,)
        or len(source_quaternion) != 4
        or len(target_quaternion) != 4
    ):
        raise ValueError("base pose snapshot is incomplete")

    source_rotation = _quaternion_wxyz_to_matrix(source_quaternion)
    target_rotation = _quaternion_wxyz_to_matrix(target_quaternion)
    world_point = source_rotation.dot(np.asarray(point, dtype=np.float64))
    world_point += source_position
    return target_rotation.T.dot(world_point - target_position)


def transform_rotation_between_base_payloads(
    rotation, source_payload, target_payload
):
    """Transform an orientation from source base_link into target base_link."""
    source_quaternion = source_payload.get("world_base_quat_wxyz", [])
    target_quaternion = target_payload.get("world_base_quat_wxyz", [])
    if len(source_quaternion) != 4 or len(target_quaternion) != 4:
        raise ValueError("base pose snapshot is incomplete")
    source_rotation = _quaternion_wxyz_to_matrix(source_quaternion)
    target_rotation = _quaternion_wxyz_to_matrix(target_quaternion)
    object_rotation = np.asarray(rotation, dtype=np.float64)
    if object_rotation.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    return target_rotation.T.dot(source_rotation).dot(object_rotation)
