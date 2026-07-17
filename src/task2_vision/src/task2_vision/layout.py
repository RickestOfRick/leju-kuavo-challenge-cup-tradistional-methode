"""读取场景二受保护布局，并转换为稳定的发布结构。"""

import copy
import itertools
import math
import os
import sys


SCENE_NAME = "scene2"
OBJECT_NAMES = (
    "part_type_a_1",
    "part_type_a_2",
    "part_type_b_1",
    "part_type_b_2",
    "part_type_c_1",
    "part_type_c_2",
)
CLASS_NAMES = ("part_type_a", "part_type_b", "part_type_c")
POSITION_SOURCE = "challenge_secret_layout_ground_truth"

INSTANCE_NAMING_RELATIVE_POSITION = "relative_position"
INSTANCE_NAMING_YAW_PRIOR = "yaw_prior"
INSTANCE_NAMING_METHODS = (
    INSTANCE_NAMING_RELATIVE_POSITION,
    INSTANCE_NAMING_YAW_PRIOR,
)
# Fixed scene2 world-yaw values recovered from challenge_secret over the full
# sampled seed range. Matching estimates and removes a common frame-yaw offset,
# so these remain valid after conversion to base_link.
YAW_PRIORS_BY_MOTION_NAME = {
    "part_type_a_1": 0.05,
    "part_type_a_2": 0.82,
    "part_type_b_1": -0.35,
    "part_type_b_2": 0.72,
    "part_type_c_1": math.pi / 2.0 + 0.22,
    "part_type_c_2": math.pi / 2.0 - 0.55,
}


def yaw_from_quat_wxyz(quat_wxyz):
    """从 MuJoCo 顺序的四元数 `[w, x, y, z]` 提取绕 Z 轴 yaw。"""
    if len(quat_wxyz) != 4:
        raise ValueError("quat_wxyz must contain 4 values")
    qw, qx, qy, qz = [float(value) for value in quat_wxyz]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise ValueError("quat_wxyz norm is zero")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def yaw_quat_xyzw(yaw):
    """将平面 yaw 转换为 ROS 顺序的四元数 `[x, y, z, w]`。"""
    half = 0.5 * float(yaw)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _quat_normalize_wxyz(quat_wxyz):
    values = [float(value) for value in quat_wxyz]
    if len(values) != 4:
        raise ValueError("quat_wxyz must contain 4 values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("quat_wxyz norm is zero")
    return [value / norm for value in values]


def _quat_inverse_wxyz(quat_wxyz):
    qw, qx, qy, qz = _quat_normalize_wxyz(quat_wxyz)
    return [qw, -qx, -qy, -qz]


def _quat_multiply_wxyz(left, right):
    lw, lx, ly, lz = _quat_normalize_wxyz(left)
    rw, rx, ry, rz = _quat_normalize_wxyz(right)
    return _quat_normalize_wxyz(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _rotate_vector_wxyz(quat_wxyz, vector):
    qw, qx, qy, qz = _quat_normalize_wxyz(quat_wxyz)
    x, y, z = [float(value) for value in vector]
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return [
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    ]


def object_class_name(object_name):
    """把实例名 `part_type_a_1` 转为类别名 `part_type_a`。"""
    return str(object_name).rsplit("_", 1)[0]


def normalize_instance_naming_method(method):
    value = str(method).strip().lower()
    if value not in INSTANCE_NAMING_METHODS:
        raise ValueError(
            "instance naming method must be one of {}, got {}".format(
                ", ".join(INSTANCE_NAMING_METHODS), method
            )
        )
    return value


def yaw_prior_candidates(class_name):
    if class_name not in CLASS_NAMES:
        raise ValueError("unsupported task2 class: {}".format(class_name))
    return [
        (motion_name, float(YAW_PRIORS_BY_MOTION_NAME[motion_name]))
        for motion_name in OBJECT_NAMES
        if object_class_name(motion_name) == class_name
    ]


def yaw_prior_window_candidates(
    class_name, frame_yaw_offset, half_width, step
):
    """Return both class-prior neighborhoods as wrapped yaw candidates."""
    half_width = abs(float(half_width))
    step = abs(float(step))
    if half_width <= 0.0 or step <= 0.0:
        raise ValueError("yaw prior window and step must be positive")

    offsets = []
    value = -half_width
    while value <= half_width + 1e-12:
        offsets.append(min(value, half_width))
        value += step
    if offsets[-1] < half_width - 1e-12:
        offsets.append(half_width)

    candidates = []
    for motion_name, prior_yaw in yaw_prior_candidates(class_name):
        for offset in offsets:
            yaw = _wrap_angle(prior_yaw + frame_yaw_offset + offset)
            if all(
                abs(_wrap_angle(yaw - existing[1])) > 1e-9
                for existing in candidates
            ):
                candidates.append((motion_name, yaw))
    return candidates


def joint_yaw_prior_assignment(class_name, object_names, fits):
    """Choose the minimum-cost one-to-one object/prior assignment."""
    priors = [name for name, _ in yaw_prior_candidates(class_name)]
    if len(object_names) != 2 or len(priors) != 2:
        raise ValueError("joint yaw-prior assignment requires two objects")

    def candidate_cost(object_name, prior_name):
        fit = fits.get(object_name)
        if fit is None:
            return float("inf")
        value = fit.get("yaw_prior_candidate_scores", {}).get(prior_name)
        if value is None:
            return float("inf")
        return float(value)

    first, second = object_names
    direct_cost = candidate_cost(first, priors[0]) + candidate_cost(
        second, priors[1]
    )
    swapped_cost = candidate_cost(first, priors[1]) + candidate_cost(
        second, priors[0]
    )
    if not math.isfinite(direct_cost) and not math.isfinite(swapped_cost):
        return None
    if direct_cost <= swapped_cost:
        return {first: priors[0], second: priors[1]}
    return {first: priors[1], second: priors[0]}


def joint_yaw_window_assignment(class_name, object_names, window_fits):
    """Assign two objects to different prior windows using current-frame costs."""
    priors = [name for name, _ in yaw_prior_candidates(class_name)]
    if len(object_names) != 2 or len(priors) != 2:
        raise ValueError("joint yaw-window assignment requires two objects")

    def candidate_cost(object_name, prior_name):
        fit = window_fits.get(object_name, {}).get(prior_name)
        if fit is None or not bool(fit.get("accepted", False)):
            return float("inf")
        value = fit.get("yaw_selection_score")
        if value is None or not math.isfinite(float(value)):
            return float("inf")
        return float(value)

    first, second = object_names
    assignments = {
        "direct": {first: priors[0], second: priors[1]},
        "swapped": {first: priors[1], second: priors[0]},
    }
    costs = {
        "direct": candidate_cost(first, priors[0])
        + candidate_cost(second, priors[1]),
        "swapped": candidate_cost(first, priors[1])
        + candidate_cost(second, priors[0]),
    }
    finite_modes = [mode for mode, cost in costs.items() if math.isfinite(cost)]
    if not finite_modes:
        return None
    selected_mode = min(finite_modes, key=lambda mode: (costs[mode], mode))
    other_mode = "swapped" if selected_mode == "direct" else "direct"
    margin = (
        abs(costs["direct"] - costs["swapped"])
        if math.isfinite(costs["direct"]) and math.isfinite(costs["swapped"])
        else None
    )
    return {
        "assignment": assignments[selected_mode],
        "selected_mode": selected_mode,
        "selected_cost": float(costs[selected_mode]),
        "alternative_cost": (
            float(costs[other_mode]) if math.isfinite(costs[other_mode]) else None
        ),
        "direct_cost": (
            float(costs["direct"]) if math.isfinite(costs["direct"]) else None
        ),
        "swapped_cost": (
            float(costs["swapped"]) if math.isfinite(costs["swapped"]) else None
        ),
        "margin": float(margin) if margin is not None else None,
    }


def _wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_prior_regularized_score(
    data_score, yaw, prior_yaw, sigma_rad, weight=1.0
):
    """Return a CAD score with a soft Gaussian yaw-prior penalty."""
    sigma_rad = abs(float(sigma_rad))
    if sigma_rad <= 0.0:
        raise ValueError("yaw prior sigma must be positive")
    data_score = float(data_score)
    deviation = abs(_wrap_angle(float(yaw) - float(prior_yaw)))
    penalty = max(0.0, float(weight)) * (deviation / sigma_rad) ** 2
    return {
        "data_score": data_score,
        "prior_penalty": float(penalty),
        "objective_score": float(data_score + penalty),
        "deviation_rad": float(deviation),
    }


def _yaw_prior_assignment(class_name, class_objects):
    priors = yaw_prior_candidates(class_name)
    if len(class_objects) != len(priors):
        raise ValueError(
            "{} requires {} objects for yaw matching, got {}".format(
                class_name, len(priors), len(class_objects)
            )
        )

    best = None
    for permutation in itertools.permutations(class_objects):
        residuals = [
            _wrap_angle(float(item["yaw"]) - prior_yaw)
            for item, (_, prior_yaw) in zip(permutation, priors)
        ]
        sin_sum = sum(math.sin(value) for value in residuals)
        cos_sum = sum(math.cos(value) for value in residuals)
        common_offset = math.atan2(sin_sum, cos_sum)
        errors = [_wrap_angle(value - common_offset) for value in residuals]
        score = sum(value * value for value in errors)
        tie_breaker = tuple(
            str(item.get("ground_truth_name", item.get("name", "")))
            for item in permutation
        )
        candidate = (score, tie_breaker, permutation, common_offset, errors)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2], best[3], best[4], best[0]


def assign_motion_names(objects, method=INSTANCE_NAMING_RELATIVE_POSITION):
    """Assign motion suffixes by relative Y or by fitted/truth yaw priors."""
    method = normalize_instance_naming_method(method)
    by_class = {class_name: [] for class_name in CLASS_NAMES}
    for item in objects:
        class_name = str(item.get("class_name", ""))
        if class_name not in by_class:
            raise ValueError("unsupported task2 class: {}".format(class_name))
        by_class[class_name].append(item)

    renamed = {}
    for class_name in CLASS_NAMES:
        class_objects = by_class[class_name]
        if len(class_objects) != 2:
            raise ValueError(
                "{} must contain exactly two objects, got {}".format(
                    class_name, len(class_objects)
                )
            )
        if method == INSTANCE_NAMING_RELATIVE_POSITION:
            # base_link Y is positive left: A/B right-to-left, C left-to-right.
            class_objects.sort(
                key=lambda item: float(item["position"][1]),
                reverse=class_name == "part_type_c",
            )
            assignments = [
                ("{}_{}".format(class_name, index), item, None, None)
                for index, item in enumerate(class_objects, start=1)
            ]
        else:
            if any("yaw" not in item for item in class_objects):
                raise ValueError("{} yaw-prior matching requires yaw".format(class_name))
            permutation, common_offset, errors, score = _yaw_prior_assignment(
                class_name, class_objects
            )
            assignments = [
                (motion_name, item, error, score)
                for (motion_name, _), item, error in zip(
                    yaw_prior_candidates(class_name), permutation, errors
                )
            ]

        for motion_name, item, yaw_error, yaw_score in assignments:
            previous_name = str(item.get("name", ""))
            item["instance_naming_method"] = method
            item["instance_naming_source_name"] = previous_name
            item["motion_name"] = motion_name
            item["name"] = motion_name
            if method == INSTANCE_NAMING_YAW_PRIOR:
                prior_yaw = float(YAW_PRIORS_BY_MOTION_NAME[motion_name])
                item["yaw_prior"] = prior_yaw
                item["yaw_prior_deg"] = math.degrees(prior_yaw)
                item["yaw_prior_common_offset"] = float(common_offset)
                item["yaw_prior_common_offset_deg"] = math.degrees(common_offset)
                item["yaw_prior_match_error"] = float(yaw_error)
                item["yaw_prior_match_error_deg"] = math.degrees(yaw_error)
                item["yaw_prior_match_score"] = float(yaw_score)
            renamed[motion_name] = item

    return [renamed[name] for name in OBJECT_NAMES]


def load_scene2_layout(simulator_path, seed):
    """调用与数采脚本相同的 `.so` 接口读取指定 seed 的布局。"""
    lib_dir = os.path.join(os.path.abspath(simulator_path), "lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    try:
        from challenge_secret import get_object_layout
    except ImportError as exc:
        raise RuntimeError("cannot import challenge_secret from {}: {}".format(lib_dir, exc))

    layout = get_object_layout(SCENE_NAME, int(seed))
    if not isinstance(layout, dict) or not layout:
        raise RuntimeError("challenge_secret returned an empty scene2 layout")

    missing = [name for name in OBJECT_NAMES if name not in layout]
    if missing:
        raise RuntimeError("scene2 layout missing objects: {}".format(", ".join(missing)))
    return layout


def build_published_objects(
    layout,
    world_base_position,
    world_base_quat_wxyz,
    instance_naming_method=INSTANCE_NAMING_RELATIVE_POSITION,
):
    """将 world 真值转换到 base_link，再按空间位置生成 motion 实例名。"""
    base_position = [float(value) for value in world_base_position]
    if len(base_position) != 3:
        raise ValueError("world_base_position must contain 3 values")
    world_to_base_quat = _quat_inverse_wxyz(world_base_quat_wxyz)
    objects = []
    for ground_truth_name in OBJECT_NAMES:
        item = layout[ground_truth_name]
        world_position = [float(value) for value in item["pos"]]
        world_quat_wxyz = [float(value) for value in item["quat"]]
        if len(world_position) != 3:
            raise ValueError(
                "{} position must contain 3 values".format(ground_truth_name)
            )
        if not all(
            math.isfinite(value) for value in world_position + world_quat_wxyz
        ):
            raise ValueError(
                "{} layout contains a non-finite value".format(ground_truth_name)
            )

        relative_world = [
            world_position[index] - base_position[index] for index in range(3)
        ]
        position = _rotate_vector_wxyz(world_to_base_quat, relative_world)
        base_quat_wxyz = _quat_multiply_wxyz(
            world_to_base_quat, world_quat_wxyz
        )
        yaw = yaw_from_quat_wxyz(base_quat_wxyz)
        objects.append(
            {
                "name": ground_truth_name,
                "motion_name": None,
                "ground_truth_name": ground_truth_name,
                "class_name": object_class_name(ground_truth_name),
                "position": position,
                "yaw": yaw,
                "yaw_deg": math.degrees(yaw),
                "yaw_valid": True,
                "orientation": [
                    base_quat_wxyz[1],
                    base_quat_wxyz[2],
                    base_quat_wxyz[3],
                    base_quat_wxyz[0],
                ],
                "yaw_orientation": yaw_quat_xyzw(yaw),
                "world_position": world_position,
                "world_quat_wxyz": world_quat_wxyz,
                "position_source": POSITION_SOURCE,
            }
        )
    return assign_motion_names(objects, instance_naming_method)


def update_published_objects_base_frame(
    objects,
    world_base_position,
    world_base_quat_wxyz,
    world_poses_by_name=None,
):
    """Keep motion names stable while updating truth in the current base frame."""
    base_position = [float(value) for value in world_base_position]
    if len(base_position) != 3:
        raise ValueError("world_base_position must contain 3 values")
    world_to_base_quat = _quat_inverse_wxyz(world_base_quat_wxyz)
    world_poses = world_poses_by_name or {}
    updated = []
    for source in objects:
        item = copy.deepcopy(source)
        ground_truth_name = str(item["ground_truth_name"])
        if ground_truth_name in world_poses:
            world_position, world_quat_wxyz = world_poses[ground_truth_name]
        else:
            world_position = item["world_position"]
            world_quat_wxyz = item["world_quat_wxyz"]
        world_position = [float(value) for value in world_position]
        world_quat_wxyz = [float(value) for value in world_quat_wxyz]

        relative_world = [
            world_position[index] - base_position[index] for index in range(3)
        ]
        position = _rotate_vector_wxyz(world_to_base_quat, relative_world)
        base_quat_wxyz = _quat_multiply_wxyz(
            world_to_base_quat, world_quat_wxyz
        )
        yaw = yaw_from_quat_wxyz(base_quat_wxyz)
        item.update(
            {
                "position": position,
                "yaw": yaw,
                "yaw_deg": math.degrees(yaw),
                "yaw_valid": True,
                "orientation": [
                    base_quat_wxyz[1],
                    base_quat_wxyz[2],
                    base_quat_wxyz[3],
                    base_quat_wxyz[0],
                ],
                "yaw_orientation": yaw_quat_xyzw(yaw),
                "world_position": world_position,
                "world_quat_wxyz": world_quat_wxyz,
                "truth_reference": "current_mujoco_qpos_base_link",
            }
        )
        updated.append(item)
    return updated
