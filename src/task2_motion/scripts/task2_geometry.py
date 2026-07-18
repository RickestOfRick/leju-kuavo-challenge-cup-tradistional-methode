#!/usr/bin/env python3
"""任务二的抓取点、姿态、作业手和交接目标计算。"""

import math

from task2_config import (
    CLASS_POSES,
    GRASP_OFFSETS_BASE,
    GRASP_YAW_CANDIDATES_DEG_BY_CLASS,
    HANDOFF_CONFIG,
    HANDOFF_MODE_BY_CLASS,
    HANDOFF_MODE_SOURCE_FK_DELAY,
    HANDOFF_MODE_VISION_RELOCALIZE,
    HANDOFF_SOURCE_FK_RELATIVE_CONFIG,
    HAND_EE_CALIBRATION,
    SORTING_OBJECT_ORDER,
    SORTING_USE_DYNAMIC_PRIORITY,
)


def _class_name(object_name):
    return str(object_name).rsplit("_", 1)[0]


def _sorting_bin_for_class(class_name):
    class_name = str(class_name)
    if class_name not in ("part_type_a", "part_type_b", "part_type_c"):
        raise ValueError("unsupported task2 part class: {}".format(class_name))
    return "sorting_bin_{}".format(class_name.rsplit("_", 1)[1])


def _quat_normalize(quat):
    values = [float(value) for value in quat]
    norm = math.sqrt(sum(value * value for value in values))
    if len(values) != 4 or norm <= 1e-12:
        raise ValueError("invalid xyzw quaternion")
    return [value / norm for value in values]


def rotate_vector(quat_xyzw, vector):
    qx, qy, qz, qw = _quat_normalize(quat_xyzw)
    x, y, z = [float(value) for value in vector]
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return [
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    ]


def _matrix_multiply(left, right):
    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _pose_matrix(pose_values, object_yaw, arm=None, yaw_offset=0.0):
    yaw_bias, pitch, _roll, manual_yaw, manual_pitch, manual_roll = pose_values
    if arm == "left":
        manual_yaw = -manual_yaw
    yaw = float(object_yaw) + float(yaw_bias) + float(yaw_offset)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    adaptive = [
        [cy * cp, -sy, cy * sp],
        [sy * cp, cy, sy * sp],
        [-sp, 0.0, cp],
    ]
    manual = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if abs(manual_yaw) > 0.01:
        c, s = math.cos(manual_yaw), math.sin(manual_yaw)
        manual = _matrix_multiply([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], manual)
    if abs(manual_pitch) > 0.01:
        c, s = math.cos(manual_pitch), math.sin(manual_pitch)
        manual = _matrix_multiply([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], manual)
    if abs(manual_roll) > 0.01:
        c, s = math.cos(manual_roll), math.sin(manual_roll)
        manual = _matrix_multiply([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], manual)
    return _matrix_multiply(adaptive, manual)


def _matrix_to_quaternion(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        w = math.sqrt(trace + 1.0) / 2.0
        x = (matrix[2][1] - matrix[1][2]) / (4.0 * w)
        y = (matrix[0][2] - matrix[2][0]) / (4.0 * w)
        z = (matrix[1][0] - matrix[0][1]) / (4.0 * w)
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        x = math.sqrt(max(0.0, 1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])) / 2.0
        y = (matrix[0][1] + matrix[1][0]) / (4.0 * x)
        z = (matrix[0][2] + matrix[2][0]) / (4.0 * x)
        w = (matrix[2][1] - matrix[1][2]) / (4.0 * x)
    elif matrix[1][1] > matrix[2][2]:
        y = math.sqrt(max(0.0, 1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])) / 2.0
        x = (matrix[0][1] + matrix[1][0]) / (4.0 * y)
        z = (matrix[1][2] + matrix[2][1]) / (4.0 * y)
        w = (matrix[0][2] - matrix[2][0]) / (4.0 * y)
    else:
        z = math.sqrt(max(0.0, 1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])) / 2.0
        x = (matrix[0][2] + matrix[2][0]) / (4.0 * z)
        y = (matrix[1][2] + matrix[2][1]) / (4.0 * z)
        w = (matrix[1][0] - matrix[0][1]) / (4.0 * z)
    return _quat_normalize([x, y, z, w])


def pose_quaternion(pose_values, object_yaw=0.0, arm=None, yaw_offset=0.0):
    return _matrix_to_quaternion(
        _pose_matrix(pose_values, object_yaw, arm=arm, yaw_offset=yaw_offset)
    )


def select_arm(observation):
    name = observation["name"]
    object_y = float(observation["position"][1])
    if name.startswith("part_type_a") and -0.2 <= object_y <= 0.1:
        return "right"
    if name.startswith("part_type_c") and -0.2 <= object_y <= 0.0:
        return "left"
    return "left" if object_y >= 0.0 else "right"


def _tool_forward_x_score(pose_values, object_yaw, arm, yaw_offset):
    matrix = _pose_matrix(pose_values, object_yaw, arm=arm, yaw_offset=yaw_offset)
    return float(-matrix[0][2])


def grasp_yaw_candidates(observation):
    class_name = str(observation["class_name"])
    try:
        candidates = [
            math.radians(float(value))
            for value in GRASP_YAW_CANDIDATES_DEG_BY_CLASS[class_name]
        ]
    except KeyError:
        raise ValueError("no grasp yaw candidates for {}".format(class_name))
    if len(candidates) != 2:
        raise ValueError("{} grasp yaw candidates must contain two angles".format(class_name))
    separation = abs(
        math.atan2(
            math.sin(candidates[0] - candidates[1]),
            math.cos(candidates[0] - candidates[1]),
        )
    )
    if abs(separation - math.pi) > 1e-6:
        raise ValueError("{} grasp yaw candidates must differ by 180 deg".format(class_name))
    return candidates


def alternate_grasp_yaw_offset(observation, current_offset):
    current = float(current_offset)
    candidates = grasp_yaw_candidates(observation)
    return max(
        candidates,
        key=lambda candidate: abs(
            math.atan2(
                math.sin(candidate - current),
                math.cos(candidate - current),
            )
        ),
    )


def flipped_yaw_offset(current_offset):
    """Return the equivalent wrist-yaw candidate 180 degrees away."""
    value = float(current_offset) + math.pi
    return math.atan2(math.sin(value), math.cos(value))


def _other_narrow_edge(observation, arm):
    name = observation["name"]
    object_y = float(observation["position"][1])
    orientation = observation["orientation"]
    if name.startswith("part_type_c") and arm == "right" and -0.3 <= object_y <= 0.0:
        axis = rotate_vector(orientation, [-1.0, 0.0, 1.0])
        norm = math.sqrt(sum(value * value for value in axis))
        return norm > 1e-9 and (-axis[1] / norm) >= math.cos(math.radians(20.0))

    if name.startswith("part_type_a") and -0.2 <= object_y <= 0.1:
        target = [1.0, 1.0]
    elif name.startswith("part_type_b") and 0.0 <= object_y <= 0.2:
        target = [1.0, -1.0]
    elif name.startswith("part_type_b") and -0.2 <= object_y < 0.0:
        target = [1.0, 1.0]
    else:
        return False
    axis = rotate_vector(orientation, [0.0, 1.0, 0.0])[:2]
    axis_norm = math.hypot(axis[0], axis[1])
    if axis_norm <= 1e-9:
        return True
    cosine = abs((axis[0] * target[0] + axis[1] * target[1]) / (axis_norm * math.sqrt(2.0)))
    return cosine < math.cos(math.radians(45.0))


def narrow_edge_offset(observation, arm, pose_values=None):
    if pose_values is None:
        pose_values = CLASS_POSES[observation["class_name"]]["grasp"]
    object_yaw = observation["yaw"]
    candidates = grasp_yaw_candidates(observation)
    selected = max(
        candidates,
        key=lambda value: _tool_forward_x_score(pose_values, object_yaw, arm, value),
    )
    if _other_narrow_edge(observation, arm):
        selected = alternate_grasp_yaw_offset(observation, selected)
    return selected


def _grasp_offset_base(object_name, arm):
    offset = [float(value) for value in GRASP_OFFSETS_BASE[object_name][arm]]
    if len(offset) != 3:
        raise ValueError(
            "{} {} base_link grasp offset must contain xyz".format(object_name, arm)
        )
    return offset


def apply_hand_calibration(position, arm):
    correction = HAND_EE_CALIBRATION[arm]
    return [float(position[index]) + correction[index] for index in range(3)]


def grasp_target(observation, arm):
    base_offset = _grasp_offset_base(observation["name"], arm)
    reference = [
        observation["position"][index] + base_offset[index] for index in range(3)
    ]
    return apply_hand_calibration(reference, arm)


def grasp_quaternion(observation, arm, yaw_offset=None):
    if yaw_offset is None:
        yaw_offset = narrow_edge_offset(observation, arm)
    return pose_quaternion(
        CLASS_POSES[observation["class_name"]]["grasp"],
        observation["yaw"],
        arm=arm,
        yaw_offset=yaw_offset,
    )


def lift_quaternion(observation, arm, yaw_offset=None):
    if yaw_offset is None:
        yaw_offset = narrow_edge_offset(observation, arm)
    return pose_quaternion(
        CLASS_POSES[observation["class_name"]]["lift"],
        observation["yaw"],
        arm=arm,
        yaw_offset=yaw_offset,
    )


def handoff_target_arm(job):
    if job["bin"] == "sorting_bin_a" and job["arm"] == "left":
        return "right"
    if job["bin"] == "sorting_bin_c" and job["arm"] == "right":
        return "left"
    return None


def _handoff_vector(config, field, class_name):
    try:
        values = [float(value) for value in config[field][class_name]]
    except KeyError:
        raise ValueError("handoff config has no {} for {}".format(field, class_name))
    if len(values) != 3:
        raise ValueError("handoff {} for {} must contain xyz".format(field, class_name))
    return values


def _vision_relocalize_handoff_values(job, key, class_name):
    config = HANDOFF_CONFIG[key]
    return {
        "receiver_position_mode": "vision_relocalize",
        "pre_place_position": _handoff_vector(
            config, "pre_place_position", class_name
        ),
        "place_position": _handoff_vector(config, "place_position", class_name),
        "place_quaternion": pose_quaternion(config["place_pose"][class_name]),
        "grasp_offset": _handoff_vector(config, "grasp_offset", class_name),
        "grasp_pose_values": list(config["grasp_pose"][class_name]),
        "lift_pose_values": list(config["lift_pose"][class_name]),
    }


def _source_fk_relative_handoff_values(job, key, class_name):
    config = HANDOFF_SOURCE_FK_RELATIVE_CONFIG[key]
    return {
        "receiver_position_mode": "source_fk_offset",
        "pre_place_position": apply_hand_calibration(
            _handoff_vector(config, "pre_place_position", class_name), job["arm"]
        ),
        "place_position": apply_hand_calibration(
            _handoff_vector(config, "place_position", class_name), job["arm"]
        ),
        "place_quaternion": pose_quaternion(config["place_pose"][class_name]),
        "pre_grasp_offset": _handoff_vector(
            config, "pre_grasp_position", class_name
        ),
        "grasp_offset": _handoff_vector(config, "grasp_position", class_name),
        "grasp_quaternion": pose_quaternion(config["grasp_pose"][class_name]),
    }


def handoff_values(job, target_arm):
    key = "left_to_right" if job["arm"] == "left" else "right_to_left"
    class_name = job["observation"]["class_name"]
    try:
        mode = str(HANDOFF_MODE_BY_CLASS[class_name])
    except KeyError:
        raise ValueError("no handoff mode for {}".format(class_name))
    if mode == HANDOFF_MODE_SOURCE_FK_DELAY:
        return _source_fk_relative_handoff_values(job, key, class_name)
    if mode == HANDOFF_MODE_VISION_RELOCALIZE:
        return _vision_relocalize_handoff_values(job, key, class_name)
    raise ValueError("unsupported handoff mode for {}: {}".format(class_name, mode))


def resolve_handoff_receiver_positions(targets, source_fk_position=None):
    """Resolve source-FK receiver offsets after the giver reaches place."""
    mode = str(targets.get("receiver_position_mode", ""))
    if mode != "source_fk_offset":
        raise ValueError("unknown handoff receiver position mode: {}".format(mode))
    if "grasp_position" in targets or "pre_grasp_position" in targets:
        raise ValueError("relative handoff targets contain absolute receiver positions")
    if source_fk_position is None:
        raise ValueError("relative handoff requires source-hand FK position")
    source = [float(value) for value in source_fk_position]
    if len(source) != 3:
        raise ValueError("source-hand FK position must contain xyz")
    pre_offset = [float(value) for value in targets["pre_grasp_offset"]]
    grasp_offset = [float(value) for value in targets["grasp_offset"]]
    if len(pre_offset) != 3 or len(grasp_offset) != 3:
        raise ValueError("relative handoff receiver offsets must contain xyz")

    resolved = dict(targets)
    resolved.pop("pre_grasp_offset")
    resolved.pop("grasp_offset")
    resolved["receiver_position_mode"] = "resolved_source_fk"
    resolved["source_fk_position"] = source
    resolved["pre_grasp_position"] = [
        source[index] + pre_offset[index] for index in range(3)
    ]
    resolved["grasp_position"] = [
        source[index] + grasp_offset[index] for index in range(3)
    ]
    return resolved


def resolve_relocalized_handoff_receiver(
    targets, observation, target_arm
):
    """Build the receiver target from a fresh full-angle object observation."""
    if str(targets.get("receiver_position_mode", "")) != "vision_relocalize":
        raise ValueError("handoff targets do not request vision relocalization")
    if str(observation.get("name", "")) == "":
        raise ValueError("relocalized observation has no object name")
    resolved = dict(targets)
    resolved["receiver_position_mode"] = "resolved_vision_relocalize"
    resolved["relocalized_observation"] = observation
    grasp_offset = [float(value) for value in targets["grasp_offset"]]
    if len(grasp_offset) != 3:
        raise ValueError("visual handoff grasp offset must contain xyz")
    visual_reference = [
        float(observation["position"][index]) + grasp_offset[index]
        for index in range(3)
    ]
    resolved["grasp_position"] = apply_hand_calibration(
        visual_reference, target_arm
    )
    grasp_pose_values = targets["grasp_pose_values"]
    lift_pose_values = targets["lift_pose_values"]
    yaw_offset = narrow_edge_offset(
        observation, target_arm, pose_values=grasp_pose_values
    )
    grasp_quaternion_value = pose_quaternion(
        grasp_pose_values,
        observation["yaw"],
        arm=target_arm,
        yaw_offset=yaw_offset,
    )
    lift_quaternion_value = pose_quaternion(
        lift_pose_values,
        observation["yaw"],
        arm=target_arm,
        yaw_offset=yaw_offset,
    )
    resolved["grasp_yaw_offset"] = yaw_offset
    resolved["grasp_quaternion"] = grasp_quaternion_value
    resolved["lift_quaternion"] = lift_quaternion_value
    resolved["pre_grasp_position"] = None
    return resolved


def build_jobs(observations):
    jobs = []
    order_index = {name: index for index, name in enumerate(SORTING_OBJECT_ORDER)}
    for name in SORTING_OBJECT_ORDER:
        observation = observations[name]
        arm = select_arm(observation)
        yaw_offset = narrow_edge_offset(observation, arm)
        jobs.append(
            {
                "object": name,
                "bin": _sorting_bin_for_class(observation["class_name"]),
                "arm": arm,
                "observation": observation,
                "grasp_position": grasp_target(observation, arm),
                "grasp_yaw_offset": yaw_offset,
                "grasp_quaternion": grasp_quaternion(
                    observation, arm, yaw_offset=yaw_offset
                ),
                "lift_quaternion": lift_quaternion(
                    observation, arm, yaw_offset=yaw_offset
                ),
            }
        )
    if not SORTING_USE_DYNAMIC_PRIORITY:
        return jobs
    return sorted(
        jobs,
        key=lambda job: (
            1 if handoff_target_arm(job) else 0,
            order_index[job["object"]],
        ),
    )
