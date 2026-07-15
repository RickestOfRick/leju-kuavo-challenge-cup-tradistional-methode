#!/usr/bin/env python3
"""
2D→3D 投影精度调试工具：对比 YOLO 检测 + 深度投影 的估计位置与 ground truth。

用法（仿真运行中）:
  python3 debug_2d3d.py --seed 3 [--frames 20]

输出每帧每零件的逐轴误差，以及多帧统计（均值/标准差）。
"""

import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np
import rospy
import cv2
import tf2_ros
from scipy.spatial.transform import Rotation


# ── 路径设置 ──────────────────────────────────────────────────────────
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_TASK_SCRIPTS = os.path.join(
    _FILE_DIR, "..", "..", "..", "challenge_cup_task_template", "scripts"
)
_GRASP_IK_DIR = os.path.join(_FILE_DIR, "collect_scene2_dataset")

sys.path.insert(0, _GRASP_IK_DIR)

# 用 importlib 直接从文件路径加载，避免 collect_scene2_dataset/challenge_task.py 影子导入
_spec = importlib.util.spec_from_file_location("challenge_task",
    os.path.join(_TASK_SCRIPTS, "challenge_task.py"))
_challenge_task_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_challenge_task_mod)
ONNXDetector = _challenge_task_mod.ONNXDetector

_GRASP_ONNX_PATH = os.path.join(_TASK_SCRIPTS, "..", "weights", "yolov8n_augmented.onnx")


# ══════════════════════════════════════════════════════════════════════
# Ground truth 加载（复用 scene2_part_grasp_ik 的布局读取逻辑）
# ══════════════════════════════════════════════════════════════════════

def _layout_path(seed: int) -> str:
    return os.path.join(
        _GRASP_IK_DIR, "generated_layouts", f"scene2_seed_{seed}_parts.yaml"
    )


def load_ground_truth(seed: int) -> dict:
    """返回 {part_name: [x,y,z]} world frame 真值。"""
    path = _layout_path(seed)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"layout yaml not found: {path}")

    parts = {}
    current_name = None
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
                current_name = stripped[:-1]
                parts[current_name] = {}
            elif "world_xyz:" in stripped and current_name:
                val = stripped.split(":", 1)[1].strip()
                parts[current_name]["world_xyz"] = _parse_float_list(val)
    return {k: v["world_xyz"] for k, v in parts.items() if "world_xyz" in v}


def _parse_float_list(text: str):
    text = text.strip().strip("[").strip("]")
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",")]


# ══════════════════════════════════════════════════════════════════════
# 相机 / TF / 深度获取（精简自 challenge_task.py）
# ══════════════════════════════════════════════════════════════════════

TOPIC_TIMEOUT = 15.0
CLASS_NAMES = ["part_a", "part_b", "part_c"]
CLASS_TO_GT = {0: ["part_type_a_1", "part_type_a_2"],
               1: ["part_type_b_1", "part_type_b_2"],
               2: ["part_type_c_1", "part_type_c_2"]}


def get_camera_intrinsics():
    from sensor_msgs.msg import CameraInfo
    msg = rospy.wait_for_message("/cam_h/color/camera_info", CameraInfo, timeout=TOPIC_TIMEOUT)
    return np.array(msg.K).reshape(3, 3)


def get_head_rgb():
    from sensor_msgs.msg import CompressedImage
    msg = rospy.wait_for_message(
        "/cam_h/color/image_raw/compressed", CompressedImage, timeout=TOPIC_TIMEOUT
    )
    return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)


def get_head_depth():
    from sensor_msgs.msg import CompressedImage
    msg = rospy.wait_for_message(
        "/cam_h/depth/image_raw/compressedDepth", CompressedImage, timeout=TOPIC_TIMEOUT
    )
    buf = np.frombuffer(msg.data, np.uint8)
    # strip leading garbage before PNG sig
    png_sig = bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    sig_pos = bytes(buf[:64].tobytes()).find(png_sig)
    if sig_pos > 0:
        buf = buf[sig_pos:]
    for flag in (cv2.IMREAD_UNCHANGED, cv2.IMREAD_ANYDEPTH):
        depth = cv2.imdecode(buf, flag)
        if depth is not None and depth.size > 0:
            if depth.dtype == np.uint16:
                return depth.astype(np.float32) * 0.001
            if depth.ndim == 3:
                return depth[:, :, 0].astype(np.float32) * 0.001
            return depth.astype(np.float32) * 0.001
    raise RuntimeError("failed to decode depth")


def pixel_to_world(cx, cy, depth_img, K, tf_buffer):
    """2D→3D 核心函数 — 纯深度投影（无经验修正），返回 (P_cam, P_base, Z_median)。
    与 challenge_task._pixel_to_world 一致，但不依赖 rospy.xxx 日志。
    """
    px, py = int(round(cx)), int(round(cy))
    H, W = depth_img.shape[:2]

    r = 2
    y0, y1 = max(0, py - r), min(H, py + r + 1)
    x0, x1 = max(0, px - r), min(W, px + r + 1)
    patch = depth_img[y0:y1, x0:x1]
    valid = patch[(patch > 0.0) & (patch <= 3.0)]
    if len(valid) == 0:
        raise RuntimeError(f"no valid depth at ({px},{py})")
    Z = float(np.median(valid))

    if Z <= 0.0 or Z > 3.0:
        raise RuntimeError(f"invalid depth Z={Z:.3f} at ({px},{py})")

    X_cam = (cx - K[0, 2]) * Z / K[0, 0]
    Y_cam = (cy - K[1, 2]) * Z / K[1, 1]
    P_cam = np.array([X_cam, Y_cam, Z])

    _CAM_FRAMES = [
        "cam_h_color_optical_frame",
        "cam_h_color_frame",
        "cam_h_link",
        "Head Camera View",
    ]
    for cam_frame in _CAM_FRAMES:
        try:
            tf_s = tf_buffer.lookup_transform("base_link", cam_frame, rospy.Time(0))
        except Exception:
            continue
        t = tf_s.transform.translation
        rq = tf_s.transform.rotation
        R_mat = Rotation.from_quat([rq.x, rq.y, rq.z, rq.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = [t.x, t.y, t.z]
        P_base = (T @ np.append(P_cam, 1.0))[:3]
        return P_cam, P_base, Z
    raise RuntimeError("no camera TF found")


# ══════════════════════════════════════════════════════════════════════
# 匹配与统计
# ══════════════════════════════════════════════════════════════════════

def match_detection_to_gt(det_cls_id, det_world_xyz, gt_dict, used_gt):
    """按类别匹配，返回最近的未使用 GT 零件名和距离。"""
    candidates = CLASS_TO_GT.get(det_cls_id, [])
    best_name, best_dist = None, float("inf")
    for name in candidates:
        if name in used_gt or name not in gt_dict:
            continue
        gt_xyz = np.array(gt_dict[name])
        # 只用 XY 距离匹配（Z 误差可能大，不用 Z 做匹配）
        dist = np.linalg.norm(np.array(det_world_xyz[:2]) - gt_xyz[:2])
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, best_dist


def get_base_to_world(tf_buffer):
    """返回 base_link→world 的平移。"""
    try:
        w2b = tf_buffer.lookup_transform("world", "base_link", rospy.Time(0))
        return [w2b.transform.translation.x,
                w2b.transform.translation.y,
                w2b.transform.translation.z]
    except Exception:
        return [-0.65, 0.0, 0.9263857]


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="2D→3D 投影精度调试")
    parser.add_argument("--seed", type=int, required=True, help="仿真 seed")
    parser.add_argument("--frames", type=int, default=10, help="采集帧数")
    parser.add_argument("--onnx", default=_GRASP_ONNX_PATH, help="YOLO ONNX 路径")
    args = parser.parse_args()

    rospy.init_node("debug_2d3d", anonymous=True)
    rospy.loginfo("debug_2d3d: loading ground truth for seed=%d", args.seed)
    gt = load_ground_truth(args.seed)
    rospy.loginfo("debug_2d3d: ground truth loaded %d parts", len(gt))
    for name, xyz in sorted(gt.items()):
        rospy.loginfo("  GT %s: [%.4f, %.4f, %.4f]", name, *xyz)

    K = get_camera_intrinsics()
    rospy.loginfo("debug_2d3d: K fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                  K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    detector = ONNXDetector(os.path.abspath(args.onnx))
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)

    # 累积统计
    all_errors = []  # [{part_name: [dx,dy,dz]}, ...]

    for frame_idx in range(args.frames):
        if rospy.is_shutdown():
            break
        rospy.sleep(0.3)  # 等新帧

        try:
            bgr = get_head_rgb()
            depth = get_head_depth()
        except Exception as e:
            rospy.logwarn("debug_2d3d: frame %d capture failed: %s", frame_idx, e)
            continue

        base_world = get_base_to_world(tf_buffer)
        dets = detector.detect(bgr)

        if len(dets) < len(gt):
            rospy.logwarn("debug_2d3d: frame %d only %d detections (gt=%d), skip",
                          frame_idx, len(dets), len(gt))
            continue

        frame_errors = {}
        used_gt = set()

        for cls_id, cx, cy, conf in dets:
            try:
                P_cam, P_base, Z = pixel_to_world(cx, cy, depth, K, tf_buffer)
            except RuntimeError as e:
                rospy.logwarn("debug_2d3d: cls=%d pixel=(%.0f,%.0f) depth fail: %s",
                              cls_id, cx, cy, e)
                continue

            # base_link → world（纯平移 + 无经验修正）
            est_world = [
                float(P_base[0]) + base_world[0],
                float(P_base[1]) + base_world[1],
                float(P_base[2]) + base_world[2],
            ]

            gt_name, match_dist = match_detection_to_gt(cls_id, est_world, gt, used_gt)
            if gt_name is None:
                rospy.logwarn("debug_2d3d: cls=%d no unmatched GT left", cls_id)
                continue

            used_gt.add(gt_name)
            gt_xyz = np.array(gt[gt_name])
            err = np.array(est_world) - gt_xyz

            rospy.loginfo(
                "[2D3D_DEBUG] %s cls=%d pixel=(%.0f,%.0f) conf=%.2f "
                "est=[%.4f,%.4f,%.4f] gt=[%.4f,%.4f,%.4f] "
                "err_dx=%.3f dy=%.3f dz=%.3f dist=%.3f Z=%.3f match=%.3f",
                gt_name, cls_id, cx, cy, conf,
                est_world[0], est_world[1], est_world[2],
                gt_xyz[0], gt_xyz[1], gt_xyz[2],
                err[0], err[1], err[2],
                np.linalg.norm(err), Z, match_dist,
            )
            frame_errors[gt_name] = err.tolist()

        all_errors.append(frame_errors)

    # ── 统计输出 ──────────────────────────────────────────────────────
    if not all_errors:
        rospy.logerr("debug_2d3d: no valid frames collected")
        return

    # 按零件聚合
    part_stats = {}
    for fe in all_errors:
        for name, err in fe.items():
            part_stats.setdefault(name, []).append(err)

    print("\n" + "=" * 90)
    print("2D→3D 投影精度统计（纯深度投景，无经验修正）")
    print("=" * 90)
    print(f"{'零件':<16} {'帧数':>4} {'dX mean±std':>16} {'dY mean±std':>16} {'dZ mean±std':>16} {'dist mean':>10}")
    print("-" * 90)

    for name in sorted(part_stats.keys()):
        arr = np.array(part_stats[name])  # (N, 3)
        n = len(arr)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        dist = np.linalg.norm(arr, axis=1).mean()
        print(f"{name:<16} {n:>4} {mean[0]:>7.3f}±{std[0]:.3f} {mean[1]:>7.3f}±{std[1]:.3f} {mean[2]:>7.3f}±{std[2]:.3f} {dist:>7.3f}")

    # 全局平均
    all_arr = np.vstack([np.array(v) for v in part_stats.values()])
    print("-" * 90)
    print(f"{'GLOBAL':<16} {len(all_arr):>4} {all_arr[:,0].mean():>7.3f}±{all_arr[:,0].std():.3f} {all_arr[:,1].mean():>7.3f}±{all_arr[:,1].std():.3f} {all_arr[:,2].mean():>7.3f}±{all_arr[:,2].std():.3f} {np.linalg.norm(all_arr, axis=1).mean():>7.3f}")
    print("=" * 90)

    rospy.loginfo("debug_2d3d: done")


if __name__ == "__main__":
    main()
