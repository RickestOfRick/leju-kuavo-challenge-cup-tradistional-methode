#!/usr/bin/env python3
"""
2D→3D 投影精度调试工具 — 一键启动仿真 + 对比估计 vs 真值。

用法:
  python3 run_debug_2d3d.py --seed 3 --frames 15

流程:
  1. 启动指定 seed 的 Scene2 仿真
  2. 等待相机 / TF 就绪
  3. 连续采集 N 帧 → YOLO 检测 → 深度投影(无经验修正) → 与 layout YAML 真值对比
  4. 输出每个零件的 dX/dY/dZ 误差统计
  5. 自动关闭仿真
"""

import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 路径初始化（延迟到 main() 中执行，因为需要 --repo-root 参数）
# ══════════════════════════════════════════════════════════════════════

_g_initialized = False

def _init_paths(repo_root: str):
    """根据 --repo-root 初始化 import 路径，并加载 ONNXDetector + ChallengeSimLauncher。"""
    global _g_initialized, ONNXDetector, ChallengeSimLauncher, GRASP_ONNX_PATH, _GRASP_IK_DIR
    if _g_initialized:
        return
    _g_initialized = True

    task_scripts = os.path.join(repo_root, "src", "challenge_cup_task_template", "scripts")
    _grasp_ik    = os.path.join(repo_root, "src", "challenge_cup_simulator", "test", "collect_scene2_dataset")
    sim_utils    = os.path.join(repo_root, "src", "challenge_cup_simulator", "utils")

    # 用 importlib 直接从文件路径加载，避免 collect_scene2_dataset/challenge_task.py 同名冲突
    spec = importlib.util.spec_from_file_location(
        "challenge_task", os.path.join(task_scripts, "challenge_task.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ONNXDetector = mod.ONNXDetector

    sys.path.insert(0, _grasp_ik)
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher as _CSL
    ChallengeSimLauncher = _CSL

    GRASP_ONNX_PATH = os.path.join(task_scripts, "..", "weights", "yolov8n_augmented.onnx")
    _GRASP_IK_DIR = _grasp_ik


# ══════════════════════════════════════════════════════════════════════
# Ground truth 加载
# ══════════════════════════════════════════════════════════════════════

def _layout_path(seed: int) -> str:
    return os.path.join(
        _GRASP_IK_DIR, "generated_layouts", f"scene2_seed_{seed}_parts.yaml"
    )


def load_ground_truth(seed: int) -> dict:
    path = _layout_path(seed)
    if not os.path.isfile(path):
        # 列出可用 seed
        layout_dir = os.path.dirname(path)
        available = []
        for f in sorted(os.listdir(layout_dir)):
            if f.startswith("scene2_seed_") and f.endswith("_parts.yaml"):
                try:
                    available.append(int(f[len("scene2_seed_"):-len("_parts.yaml")]))
                except ValueError:
                    pass
        raise FileNotFoundError(
            f"seed={seed} 没有预生成的 layout YAML。\n"
            f"可用 seed: {available}\n"
            f"请使用: bash debug_2d3d.sh <可用seed>"
        )
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
# ROS 感知层
# ══════════════════════════════════════════════════════════════════════

import rospy
import cv2
import tf2_ros
from scipy.spatial.transform import Rotation

TOPIC_TIMEOUT = 30.0
CLASS_TO_GT = {
    0: ["part_type_a_1", "part_type_a_2"],
    1: ["part_type_b_1", "part_type_b_2"],
    2: ["part_type_c_1", "part_type_c_2"],
}


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


# ══════════════════════════════════════════════════════════════════════
# 2D → 3D 核心（纯深度投影，无经验修正）
# ══════════════════════════════════════════════════════════════════════

def pixel_to_world(cx, cy, depth_img, K, tf_buffer):
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
        "cam_h_color_optical_frame", "cam_h_color_frame",
        "cam_h_link", "Head Camera View",
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


def get_base_to_world(tf_buffer):
    try:
        w2b = tf_buffer.lookup_transform("world", "base_link", rospy.Time(0))
        return [w2b.transform.translation.x,
                w2b.transform.translation.y,
                w2b.transform.translation.z]
    except Exception:
        return [-0.65, 0.0, 0.9263857]


def match_detection_to_gt(det_cls_id, det_world_xyz, gt_dict, used_gt):
    candidates = CLASS_TO_GT.get(det_cls_id, [])
    best_name, best_dist = None, float("inf")
    for name in candidates:
        if name in used_gt or name not in gt_dict:
            continue
        gt_xyz = np.array(gt_dict[name])
        dist = np.linalg.norm(np.array(det_world_xyz[:2]) - gt_xyz[:2])
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, best_dist


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="2D→3D 一键调试")
    parser.add_argument("--seed", type=int, required=True, help="仿真 seed")
    parser.add_argument("--frames", type=int, default=15, help="采集帧数")
    parser.add_argument("--repo-root", required=True, help="项目根目录（由 shell 脚本传入）")
    parser.add_argument("--no-timer-gui", action="store_true", help="不弹出计时器窗口")
    args = parser.parse_args()

    # ── 0. 初始化路径（必须在任何业务逻辑之前） ──────────────────────
    _init_paths(args.repo_root)
    global ONNXDetector, ChallengeSimLauncher, GRASP_ONNX_PATH
    onnx_path = GRASP_ONNX_PATH

    # ── 1. 启动仿真 ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Scene2 2D→3D 调试  |  seed={args.seed}  |  frames={args.frames}")
    print(f"{'='*70}\n")

    launcher = ChallengeSimLauncher(
        scene="scene2",
        seed=args.seed,
        match_time_limit=None,
        timer_gui=not args.no_timer_gui,
    )
    print("[DEBUG] 启动仿真环境...")
    launcher.start(node_name="debug_2d3d", timeout=120)

    # ── 设置头部相机角度（看向桌面） ──────────────────────────────────
    print("[DEBUG] 设置头部相机角度...")
    from kuavo_msgs.msg import robotHeadMotionData
    head_pub = rospy.Publisher("/robot_head_motion_data", robotHeadMotionData, queue_size=10)
    # 等待 subscriber 连接
    _t0 = time.time()
    while head_pub.get_num_connections() == 0 and time.time() - _t0 < 10.0:
        rospy.sleep(0.2)
    msg = robotHeadMotionData()
    msg.joint_data = [0.0, 20.0]  # head_yaw=0, head_pitch=20° (低头看桌面)
    for _ in range(5):
        head_pub.publish(msg)
        rospy.sleep(0.1)
    rospy.sleep(1.0)
    print("[DEBUG] 头部就绪，等待相机稳定（3s）...")
    rospy.sleep(3.0)

    # ── 2. 加载真值 + 初始化感知 ──────────────────────────────────────
    print("[DEBUG] 加载真值...")
    gt = load_ground_truth(args.seed)
    print(f"[DEBUG]   ground truth: {len(gt)} 零件")
    for name, xyz in sorted(gt.items()):
        print(f"    GT {name}: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")

    K = get_camera_intrinsics()
    print(f"[DEBUG] 相机内参 fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    onnx_path = os.path.abspath(onnx_path)
    print(f"[DEBUG] 加载 YOLO: {onnx_path}")
    detector = ONNXDetector(onnx_path)

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(2.0)

    # ── 3. 采集循环 ──────────────────────────────────────────────────
    all_errors = []  # [{part_name: [dx,dy,dz]}, ...]
    frame_det_counts = []

    for fi in range(args.frames):
        if rospy.is_shutdown():
            break

        try:
            bgr = get_head_rgb()
            depth = get_head_depth()
        except Exception as e:
            print(f"  [frame {fi+1}] capture fail: {e}")
            continue

        base_world = get_base_to_world(tf_buffer)
        dets = detector.detect(bgr)
        frame_det_counts.append(len(dets))

        if len(dets) < len(gt):
            print(f"  [frame {fi+1}] only {len(dets)} detections (gt={len(gt)}), skip")
            continue

        frame_errors = {}
        used_gt = set()

        for cls_id, cx, cy, conf in dets:
            try:
                P_cam, P_base, Z = pixel_to_world(cx, cy, depth, K, tf_buffer)
            except RuntimeError as e:
                continue

            est_world = [
                float(P_base[0]) + base_world[0],
                float(P_base[1]) + base_world[1],
                float(P_base[2]) + base_world[2],
            ]

            gt_name, match_dist = match_detection_to_gt(cls_id, est_world, gt, used_gt)
            if gt_name is None:
                continue

            used_gt.add(gt_name)
            gt_xyz = np.array(gt[gt_name])
            err = np.array(est_world) - gt_xyz

            print(
                f"  [{gt_name:<16}] cls={cls_id} pixel=({int(cx):>4},{int(cy):>4}) "
                f"conf={conf:.2f} | "
                f"est=[{est_world[0]:.4f},{est_world[1]:.4f},{est_world[2]:.4f}] "
                f"gt=[{gt_xyz[0]:.4f},{gt_xyz[1]:.4f},{gt_xyz[2]:.4f}] | "
                f"err=[{err[0]:+.3f},{err[1]:+.3f},{err[2]:+.3f}] "
                f"dist={np.linalg.norm(err):.3f} Z={Z:.3f}"
            )
            frame_errors[gt_name] = err.tolist()

        all_errors.append(frame_errors)
        rospy.sleep(0.2)

    # ── 4. 统计输出 ──────────────────────────────────────────────────
    launcher.stop()
    rospy.sleep(1.0)

    if not all_errors:
        print("\n[ERROR] 无有效帧")
        return

    part_stats = {}
    for fe in all_errors:
        for name, err in fe.items():
            part_stats.setdefault(name, []).append(err)

    print(f"\n{'='*90}")
    print(f"2D→3D 投影精度统计（纯深度投影，无经验修正）| seed={args.seed}")
    print(f"{'='*90}")
    print(f"有效帧: {len(all_errors)}/{args.frames}  平均检测数: {np.mean(frame_det_counts):.1f}")
    print(f"{'='*90}")
    print(f"{'零件':<16} {'帧':>4} {'dX mean':>8} {'±std':>6} {'dY mean':>8} {'±std':>6} {'dZ mean':>8} {'±std':>6} {'Euclid':>8}")
    print(f"{'-'*90}")

    for name in sorted(part_stats.keys()):
        arr = np.array(part_stats[name])
        n = len(arr)
        print(
            f"{name:<16} {n:>4} "
            f"{arr[:,0].mean():>8.4f} {arr[:,0].std():>6.4f} "
            f"{arr[:,1].mean():>8.4f} {arr[:,1].std():>6.4f} "
            f"{arr[:,2].mean():>8.4f} {arr[:,2].std():>6.4f} "
            f"{np.linalg.norm(arr, axis=1).mean():>8.4f}"
        )

    all_arr = np.vstack([np.array(v) for v in part_stats.values()])
    print(f"{'-'*90}")
    print(
        f"{'GLOBAL':<16} {len(all_arr):>4} "
        f"{all_arr[:,0].mean():>8.4f} {all_arr[:,0].std():>6.4f} "
        f"{all_arr[:,1].mean():>8.4f} {all_arr[:,1].std():>6.4f} "
        f"{all_arr[:,2].mean():>8.4f} {all_arr[:,2].std():>6.4f} "
        f"{np.linalg.norm(all_arr, axis=1).mean():>8.4f}"
    )
    print(f"{'='*90}")

    # 分轴独立分析
    print(f"\n分轴误差分析:")
    for axis, label in [(0, "dX"), (1, "dY"), (2, "dZ")]:
        vals = all_arr[:, axis]
        print(f"  {label}: mean={vals.mean():+.4f} std={vals.std():.4f} "
              f"min={vals.min():+.4f} max={vals.max():+.4f} "
              f"range={vals.max()-vals.min():.4f}")

    print(f"\n[DEBUG] 完成。")

    # 建议
    print(f"\n{'─'*70}")
    print(f"解读指南:")
    print(f"  mean ≈ 系统偏差（可加常数修正）")
    print(f"  std  ≈ 随机噪声（需多点采样/滤波降低）")
    print(f"  range = 单点深度的波动范围")
    print(f"  Euclid = 3D 综合误差")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
