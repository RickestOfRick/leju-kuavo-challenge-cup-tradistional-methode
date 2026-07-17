"""Truth-independent task 2 resting-pose and yaw estimation helpers."""

import math

import numpy as np


REST_ROLL_BY_CLASS = {
    "part_type_a": 0.0,
    "part_type_b": 0.0,
    "part_type_c": math.pi / 2.0,
}


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference(angle, reference):
    return wrap_angle(float(angle) - float(reference))


def rotation_from_yaw(class_name, yaw):
    if class_name not in REST_ROLL_BY_CLASS:
        raise ValueError("unsupported task2 part class: {}".format(class_name))
    yaw = float(yaw)
    roll = REST_ROLL_BY_CLASS[class_name]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    rotation_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    return rotation_z.dot(rotation_x)


def yaw_from_rotation(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    return wrap_angle(math.atan2(rotation[1, 0], rotation[0, 0]))


def quaternion_xyzw_from_rotation(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("rotation produced a zero quaternion")
    return [float(value) for value in quaternion / norm]


def orientation_xyzw_from_yaw(class_name, yaw):
    return quaternion_xyzw_from_rotation(rotation_from_yaw(class_name, yaw))


def yaw_orientation_xyzw(yaw):
    half = 0.5 * float(yaw)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def principal_axis_angle(points_xy):
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("points_xy must have shape Nx2 with N >= 3")
    centered = points - np.mean(points, axis=0)
    covariance = centered.T.dot(centered) / float(max(1, points.shape[0] - 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    maximum = float(np.max(eigenvalues))
    minimum = float(np.min(eigenvalues))
    anisotropy = (maximum - minimum) / max(maximum + minimum, 1e-12)
    return wrap_angle(math.atan2(axis[1], axis[0])), float(anisotropy)


def pca_yaw_hypotheses(class_name, observed_xy, model_vertices):
    observed_angle, observed_anisotropy = principal_axis_angle(observed_xy)
    rest_vertices = np.asarray(model_vertices, dtype=np.float64).dot(
        rotation_from_yaw(class_name, 0.0).T
    )
    model_angle, model_anisotropy = principal_axis_angle(rest_vertices[:, :2])
    first = wrap_angle(observed_angle - model_angle)
    return (
        [first, wrap_angle(first + math.pi)],
        min(observed_anisotropy, model_anisotropy),
    )


def circular_median(angles):
    values = [wrap_angle(value) for value in angles]
    if not values:
        raise ValueError("angles must not be empty")
    anchor = values[-1]
    unwrapped = [anchor + angle_difference(value, anchor) for value in values]
    return wrap_angle(float(np.median(np.asarray(unwrapped, dtype=np.float64))))


def bounded_yaw_grid(minimum, maximum, step):
    minimum = float(minimum)
    maximum = float(maximum)
    step = float(step)
    if minimum >= maximum or step <= 0.0:
        raise ValueError("bounded yaw grid requires min < max and step > 0")
    count = int(math.floor((maximum - minimum) / step))
    values = [minimum + index * step for index in range(count + 1)]
    if values[-1] < maximum - 1e-9:
        values.append(maximum)
    return values
