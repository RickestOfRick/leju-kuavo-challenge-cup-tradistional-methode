#!/usr/bin/env python3
"""
scene2_auto_labeler.py — 从 rosbag + 布局 YAML 自动生成 YOLO 训练标注

用法:
  # 单个 bag
  python3 tools/scene2_auto_labeler.py \
      --bag-dir bags/scene2/ \
      --layout-yaml src/challenge_cup_simulator/test/collect_scene2_dataset/generated_layouts/scene2_seed_0_parts.yaml \
      --output-dir datasets/scene2_yolo/train \
      --camera cam_h

  # 批量处理（自动匹配 seed）
  python3 tools/scene2_auto_labeler.py \
      --bag-dir bags/scene2/ \
      --layout-dir src/challenge_cup_simulator/test/collect_scene2_dataset/generated_layouts/ \
      --output-dir datasets/scene2_yolo/train \
      --camera cam_h

原理:
  ground truth 零件 3D 位姿 (layout YAML)
    + 碰撞盒尺寸 (scene2.yaml)
    → 8 角点世界坐标
    → 相机外参变换 → 相机坐标系
    → 内参投影 → 2D bbox
    → YOLO 归一化格式标注
"""

import argparse
import math
import os
import sys
import re
from pathlib import Path

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge

# ---------------------------------------------------------------------------
# 相机内参 (来自 src/challenge_cup_simulator/scripts/cam_info_pub.py)
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 720
CAMERA_CONFIG = {
    "cam_h": {
        "fovy": 85.0,
        "fx": (HEIGHT / 2.0) / math.tan(math.radians(85.0) / 2.0),  # ≈ 392.9
        "fy": (HEIGHT / 2.0) / math.tan(math.radians(85.0) / 2.0),
        "cx": WIDTH / 2.0,  # 640
        "cy": HEIGHT / 2.0,  # 360
        "color_topic": "/cam_h/color/image_raw/compressed",
    },
    "cam_l": {
        "fovy": 60.0,
        "fx": (HEIGHT / 2.0) / math.tan(math.radians(60.0) / 2.0),  # ≈ 623.5
        "fy": (HEIGHT / 2.0) / math.tan(math.radians(60.0) / 2.0),
        "cx": WIDTH / 2.0,
        "cy": HEIGHT / 2.0,
        "color_topic": "/cam_l/color/image_raw/compressed",
    },
    "cam_r": {
        "fovy": 60.0,
        "fx": (HEIGHT / 2.0) / math.tan(math.radians(60.0) / 2.0),
        "fy": (HEIGHT / 2.0) / math.tan(math.radians(60.0) / 2.0),
        "cx": WIDTH / 2.0,
        "cy": HEIGHT / 2.0,
        "color_topic": "/cam_r/color/image_raw/compressed",
    },
}

# ---------------------------------------------------------------------------
# 碰撞盒半边长 (来自 scene2.yaml)
# ---------------------------------------------------------------------------
COLLISION_HALF = {
    # 视觉包围盒（非物理碰撞体！物理碰撞体太紧，投影后 bbox 太小或角点落相机后方）
    # A类钢管紧固件：视觉约 3cm×5cm×2cm
    "part_type_a": [0.015, 0.025, 0.010],
    # B类T型接头：视觉约 5cm×6cm×5cm
    "part_type_b": [0.025, 0.030, 0.025],
    # C类螺丝刀：视觉约 16cm 长 × 3cm 直径
    "part_type_c": [0.080, 0.015, 0.015],
}

PART_TO_CLASS = {"part_type_a": 0, "part_type_b": 1, "part_type_c": 2}
CLASS_NAMES = ["part_a", "part_b", "part_c"]

# ---------------------------------------------------------------------------
# 默认头部相机外参 (世界 → 相机)
# 来自 MuJoCo head_camera: pos="0.121 0.0 0.0345" quat="0.612372 0.353553 -0.353553 -0.612372"
# 叠加 S52 站立高度 ~1.25m（head link 世界 z）
# ---------------------------------------------------------------------------
DEFAULT_HEAD_EXTRINSICS = {
    "world_pos": [0.12, 0.0, 1.28],
    "world_quat_xyzw": [0.354, -0.354, -0.612, 0.612],  # xyzw (scipy convention)
}


# ===================================================================
# 工具函数
# ===================================================================

def quat_xyzw_to_rotmat(q_xyzw):
    """四元数 [x,y,z,w] → 3×3 旋转矩阵"""
    x, y, z, w = [float(v) for v in q_xyzw]
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y],
    ])


def quat_wxyz_to_rotmat(q_wxyz):
    """四元数 [w,x,y,z] (MuJoCo/YAML 格式) → 3×3 旋转矩阵"""
    w, x, y, z = [float(v) for v in q_wxyz]
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y],
    ])


def build_extrinsics(world_pos, world_quat_xyzw):
    """构建世界→相机变换矩阵 T_cam_world (4×4)
    
    支持两种格式:
    - world_quat_xyzw 是 4 元素列表: 按四元数 [x,y,z,w] 处理
    - world_quat_xyzw 是 9 元素列表: 当作 3×3 旋转矩阵 (行主序), 直接使用
    """
    q = world_quat_xyzw
    if len(q) == 9:
        # 直接使用 3×3 旋转矩阵 (行: right, up, forward)
        R = np.array(q).reshape(3, 3)
    else:
        R = quat_xyzw_to_rotmat(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.array(world_pos)
    return T


def box_corners_world(world_pos, quat_wxyz, half_extents):
    """碰撞盒 8 角点 (世界坐标系)"""
    R = quat_wxyz_to_rotmat(quat_wxyz)
    offsets = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1],  [1, -1, 1],  [-1, 1, 1],  [1, 1, 1],
    ]) * half_extents
    return (R @ offsets.T).T + np.array(world_pos)


def project_points(points_world, T_cam_world, K):
    """世界坐标点 → 像素坐标"""
    # points_world: (N, 3)
    ones = np.ones((points_world.shape[0], 1))
    p_world_h = np.hstack([points_world, ones])        # (N, 4)
    p_cam = (T_cam_world @ p_world_h.T).T[:, :3]       # (N, 3)
    # 只保留 Z > 0 的点（在相机前方）
    mask = p_cam[:, 2] > 0.01
    if not mask.any():
        return None, None
    p_cam = p_cam[mask]
    uv = (K @ p_cam.T).T                                  # (N, 3)
    uv = uv[:, :2] / uv[:, 2:3]                           # 归一化
    return uv, mask


def bbox_from_corners(corners_world, T_cam_world, K, margin=5):
    """碰撞盒角点 → YOLO bbox [cx, cy, w, h] (归一化)"""
    uv, _mask = project_points(corners_world, T_cam_world, K)
    if uv is None or len(uv) < 2:
        return None

    x_min = max(0, float(uv[:, 0].min()) - margin)
    y_min = max(0, float(uv[:, 1].min()) - margin)
    x_max = min(WIDTH - 1, float(uv[:, 0].max()) + margin)
    y_max = min(HEIGHT - 1, float(uv[:, 1].max()) + margin)

    if x_max <= x_min or y_max <= y_min:
        return None

    return [
        (x_min + x_max) / 2 / WIDTH,
        (y_min + y_max) / 2 / HEIGHT,
        (x_max - x_min) / WIDTH,
        (y_max - y_min) / HEIGHT,
    ]


# ===================================================================
# 布局 YAML 读取
# ===================================================================

def parse_float_list(text):
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"expected YAML inline list, got: {text}")
    body = text[1:-1].strip()
    if not body:
        return []
    return [float(item.strip()) for item in body.split(",")]


def load_layout_yaml(yaml_path):
    """读取 generated_layouts YAML，返回 {part_name: {world_xyz, quat_wxyz, yaw_z_rad}}"""
    parts = {}
    current_name = None
    with open(yaml_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
                current_name = stripped[:-1]
                parts[current_name] = {}
                continue
            if current_name is None or not line.startswith("    ") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            value = value.strip()
            if key in ("world_xyz", "quat_wxyz"):
                parts[current_name][key] = parse_float_list(value)
            elif key in ("yaw_z_rad", "yaw_z_deg"):
                parts[current_name][key] = float(value)
    return parts


# ===================================================================
# 主逻辑
# ===================================================================

def label_one_bag(bag_path, layout, output_dir, camera, extrinsics, margin,
                  max_frames=None, skip_interval=1):
    """处理单个 bag: 提取图像 + 生成 YOLO 标注"""
    bridge = CvBridge()
    cam_cfg = CAMERA_CONFIG[camera]
    color_topic = cam_cfg["color_topic"]
    K = np.array([
        [cam_cfg["fx"], 0, cam_cfg["cx"]],
        [0, cam_cfg["fy"], cam_cfg["cy"]],
        [0, 0, 1],
    ])

    T_cam_world = build_extrinsics(
        extrinsics["world_pos"], extrinsics["world_quat_xyzw"]
    )

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")

    import rosbag
    bag = rosbag.Bag(bag_path, "r")

    frame_idx = 0
    saved = 0
    skipped_no_part = 0

    print(f"[labeler] processing {os.path.basename(bag_path)} (camera={camera})")
    print(f"[labeler]   layout parts: {list(layout.keys())}")
    print(f"[labeler]   output: {output_dir}")

    for topic, msg, t in bag.read_messages(topics=[color_topic]):
        if frame_idx % skip_interval != 0:
            frame_idx += 1
            continue
        if max_frames is not None and saved >= max_frames:
            break
        frame_idx += 1

        # 解码图像
        try:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            print(f"[labeler]   WARN: decode error at frame {frame_idx}: {exc}")
            continue

        # 为每个零件生成 bbox
        labels = []
        any_visible = False
        for part_name, part_data in layout.items():
            part_type = part_name.rsplit("_", 1)[0]  # "part_type_a_1" → "part_type_a"
            if part_type not in COLLISION_HALF:
                continue

            half = COLLISION_HALF[part_type]
            corners_w = box_corners_world(
                part_data["world_xyz"], part_data["quat_wxyz"], half
            )
            bbox = bbox_from_corners(corners_w, T_cam_world, K, margin)
            if bbox is None:
                continue

            any_visible = True
            cls_id = PART_TO_CLASS[part_type]
            labels.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

        if not any_visible:
            skipped_no_part += 1
            continue

        # 保存
        ts_ns = t.to_nsec()
        stem = f"{camera}_{ts_ns}"
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

        cv2.imwrite(img_path, cv_img)
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))
        saved += 1

    bag.close()
    print(f"[labeler]   done: {saved} frames saved, {skipped_no_part} skipped (no parts visible)")
    return saved


def resolve_layouts(bag_dir, layout_dir, layout_yaml):
    """解析布局来源优先级: layout_dir(自动匹配seed) > layout_yaml(单文件)
    
    Returns dict[str, str] mapping seed_str → yaml_path.
    """
    if layout_dir:
        pattern = re.compile(r"scene2_seed_(\d+)_parts\.yaml")
        layouts: dict[str, str] = {}
        for fname in os.listdir(layout_dir):
            m = pattern.match(fname)
            if m:
                layouts[str(m.group(1))] = os.path.join(layout_dir, fname)
        return layouts
    elif layout_yaml:
        return {"default": layout_yaml}
    return {}


def find_bags(bag_dir):
    """查找目录下所有 .bag 文件（排除 .bag.active）"""
    bags = []
    for fname in sorted(os.listdir(bag_dir)):
        if fname.endswith(".bag") and not fname.endswith(".bag.active"):
            bags.append(os.path.join(bag_dir, fname))
    return bags


def extract_seed_from_bag_name(bag_path):
    """从 bag 文件名提取 seed，如 scene2_seed_0_20250101.bag → 0"""
    m = re.search(r"seed_(\d+)", os.path.basename(bag_path))
    return int(m.group(1)) if m else None


def parse_extrinsics_yaml(path):
    """读取相机外参校准文件"""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {
        "world_pos": data["world_pos"],
        "world_quat_xyzw": data["world_quat_xyzw"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="scene2 auto-labeler: rosbag → YOLO training dataset"
    )
    parser.add_argument("--bag-dir", required=True, help="Directory of .bag files")
    parser.add_argument("--layout-dir", default=None,
                        help="generated_layouts/ directory (auto-match seed from bag name)")
    parser.add_argument("--layout-yaml", default=None,
                        help="Single layout YAML (used if --layout-dir not set)")
    parser.add_argument("--output-dir", required=True, help="Output dir (images/ + labels/)")
    parser.add_argument("--camera", default="cam_h", choices=["cam_h", "cam_l", "cam_r"])
    parser.add_argument("--extrinsics-yaml", default=None,
                        help="Camera extrinsics calibration YAML (world_pos + world_quat_xyzw)")
    parser.add_argument("--margin", type=int, default=5, help="Extra bbox margin in pixels")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames per bag (None = all)")
    parser.add_argument("--skip-interval", type=int, default=1,
                        help="Process every Nth frame (default 1 = all)")
    args = parser.parse_args()

    # 加载外参
    if args.extrinsics_yaml:
        extrinsics = parse_extrinsics_yaml(args.extrinsics_yaml)
        print(f"[labeler] using calibrated extrinsics from {args.extrinsics_yaml}")
    else:
        extrinsics = DEFAULT_HEAD_EXTRINSICS
        print("[labeler] using DEFAULT head camera extrinsics")
        print(f"[labeler]   pos={extrinsics['world_pos']}, quat_xyzw={extrinsics['world_quat_xyzw']}")

    # 加载布局
    all_layouts = resolve_layouts(args.bag_dir, args.layout_dir, args.layout_yaml)
    if not all_layouts:
        print("[labeler] ERROR: no layout YAML found. Use --layout-dir or --layout-yaml.")
        sys.exit(1)

    bags = find_bags(args.bag_dir)
    if not bags:
        print(f"[labeler] ERROR: no .bag files found in {args.bag_dir}")
        sys.exit(1)

    print(f"[labeler] found {len(bags)} bag(s), {len(all_layouts)} layout(s)")
    total_saved = 0

    for bag_path in bags:
        # 根据 seed 匹配布局
        seed = extract_seed_from_bag_name(bag_path)
        seed_key = str(seed) if seed is not None else None
        if seed_key is not None and seed_key in all_layouts:
            layout_path = all_layouts[seed_key]
        elif "default" in all_layouts:
            layout_path = all_layouts["default"]
        else:
            print(f"[labeler] WARN: cannot match layout for {os.path.basename(bag_path)}, skipping")
            continue

        layout = load_layout_yaml(layout_path)
        saved = label_one_bag(
            bag_path, layout, args.output_dir, args.camera,
            extrinsics, args.margin,
            max_frames=args.max_frames,
            skip_interval=args.skip_interval,
        )
        total_saved += saved

    print(f"[labeler] ALL DONE: {total_saved} total frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
