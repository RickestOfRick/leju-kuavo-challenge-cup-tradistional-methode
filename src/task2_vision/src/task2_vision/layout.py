"""读取场景二受保护布局，并转换为稳定的发布结构。"""

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


def _assign_motion_names(objects):
    """按 base_link 左右位置生成 motion 使用的稳定实例编号。"""
    by_class = {class_name: [] for class_name in CLASS_NAMES}
    for item in objects:
        by_class[item["class_name"]].append(item)

    renamed = {}
    for class_name in CLASS_NAMES:
        class_objects = by_class[class_name]
        if len(class_objects) != 2:
            raise ValueError(
                "{} must contain exactly two objects, got {}".format(
                    class_name, len(class_objects)
                )
            )
        # base_link 的 Y 正方向为左：A/B 从右到左，C 从左到右。
        class_objects.sort(
            key=lambda item: float(item["position"][1]),
            reverse=class_name == "part_type_c",
        )
        for index, item in enumerate(class_objects, start=1):
            motion_name = "{}_{}".format(class_name, index)
            item["motion_name"] = motion_name
            item["name"] = motion_name
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


def build_published_objects(layout, world_base_position, world_base_quat_wxyz):
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
    return _assign_motion_names(objects)
