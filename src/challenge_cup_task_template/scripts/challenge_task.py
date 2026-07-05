#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple


SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
    },
}


ARM_JOINT_NAMES = ["arm_joint_{}".format(i) for i in range(1, 15)]
BASE_FRAME = "base_link"
LOG_PREFIX = "!!!!!"
# RGB-D/TF 有时给出接近地面世界系的高度；IK 本地系里桌面高度通常接近 0。
IK_WORLD_Z_THRESHOLD = 0.75
IK_WORLD_TO_LOCAL_Z_OFFSET = -0.923783
HAND_WORKSPACES = {
    "left": {
        "x": (0.18, 0.48),
        "y": (0.04, 0.36),
        "z": (-0.06, 0.60),
        "park": (0.28, 0.24, 0.55),
    },
    "right": {
        "x": (0.18, 0.48),
        "y": (-0.36, -0.04),
        "z": (-0.06, 0.60),
        "park": (0.28, -0.24, 0.55),
    },
}
BASE_APPROACH_TARGET_X = 0.42
BASE_APPROACH_MAX_DISTANCE = 0.85
BASE_APPROACH_MIN_DISTANCE = 0.04
BASE_APPROACH_SPEED = 0.16
BASE_APPROACH_SETTLE = 0.25
HEAD_LOOK_TARGET = [0.0, 20.0]
HEAD_SETTLE_TIME = 0.8


def _log_action(rospy, action):
    rospy.loginfo("%s现在执行：%s", LOG_PREFIX, action)


def _log_success(rospy, action):
    rospy.loginfo("%s执行成功：%s", LOG_PREFIX, action)


def _log_failure(rospy, action):
    rospy.logwarn("%s执行失败：%s", LOG_PREFIX, action)


def _log_error(rospy, action):
    rospy.logerr("%s执行失败：%s", LOG_PREFIX, action)


@dataclass
class ObjectEstimate:
    """Runtime estimate produced only from public sensor data."""

    label: str
    kind: str
    confidence: float
    x: float
    y: float
    z: float
    yaw: float = 0.0
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_frame: str = ""
    source: str = ""

    def distance_xy(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


class SensorCache:
    """Small ROS cache for RGB-D frames, camera info and robot feedback."""

    CAMERA_TOPICS = {
        "head": {
            "color": "/cam_h/color/image_raw/compressed",
            "depth": "/cam_h/depth/image_raw/compressedDepth",
            "info": "/cam_h/color/camera_info",
        },
        "left": {
            "color": "/cam_l/color/image_raw/compressed",
            "depth": "/cam_l/depth/image_rect_raw/compressedDepth",
            "info": "/cam_l/color/camera_info",
        },
        "right": {
            "color": "/cam_r/color/image_raw/compressed",
            "depth": "/cam_r/depth/image_rect_raw/compressedDepth",
            "info": "/cam_r/color/camera_info",
        },
    }

    def __init__(self, rospy, sensor_msgs, kuavo_msgs, tf2_ros):
        self.rospy = rospy
        self._lock = threading.RLock()
        self.cameras = {
            name: {"color": None, "depth": None, "info": None}
            for name in self.CAMERA_TOPICS
        }
        self.claw_state = None
        self.sensor_data = None
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        compressed = sensor_msgs.CompressedImage
        camera_info = sensor_msgs.CameraInfo
        for name, topics in self.CAMERA_TOPICS.items():
            rospy.Subscriber(topics["color"], compressed,
                             self._make_camera_cb(name, "color"),
                             queue_size=1, buff_size=2 ** 24)
            rospy.Subscriber(topics["depth"], compressed,
                             self._make_camera_cb(name, "depth"),
                             queue_size=1, buff_size=2 ** 24)
            rospy.Subscriber(topics["info"], camera_info,
                             self._make_camera_cb(name, "info"),
                             queue_size=1)

        rospy.Subscriber("/leju_claw_state", kuavo_msgs.lejuClawState,
                         self._claw_cb, queue_size=1)
        rospy.Subscriber("/sensors_data_raw", kuavo_msgs.sensorsData,
                         self._sensor_cb, queue_size=1)

    def _make_camera_cb(self, name, slot):
        def cb(msg):
            with self._lock:
                self.cameras[name][slot] = msg
        return cb

    def _claw_cb(self, msg):
        with self._lock:
            self.claw_state = msg

    def _sensor_cb(self, msg):
        with self._lock:
            self.sensor_data = msg

    def get_camera(self, name):
        with self._lock:
            data = self.cameras.get(name, {})
            return data.get("color"), data.get("depth"), data.get("info")

    def get_claw_state(self):
        with self._lock:
            return self.claw_state

    def wait_for_sensor_data(self, timeout=8.0):
        deadline = time.time() + timeout
        while not self.rospy.is_shutdown() and time.time() < deadline:
            with self._lock:
                if self.sensor_data is not None:
                    return True
            self.rospy.sleep(0.05)
        return False

    def wait_for_head_camera(self, timeout=8.0):
        deadline = time.time() + timeout
        while not self.rospy.is_shutdown() and time.time() < deadline:
            color, depth, info = self.get_camera("head")
            if color is not None and depth is not None and info is not None:
                return True
            self.rospy.sleep(0.05)
        return False

    def transform_point(self, point_xyz, source_frame, target_frame=BASE_FRAME):
        if not source_frame or source_frame == target_frame:
            return point_xyz
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                self.rospy.Time(0),
                self.rospy.Duration(0.15),
            )
        except Exception as exc:
            self.rospy.logwarn_throttle(
                2.0, "%sTF转换失败 %s -> %s: %s",
                LOG_PREFIX, source_frame, target_frame, exc)
            return None

        t = trans.transform.translation
        q = trans.transform.rotation
        rotated = self._rotate_by_quaternion(
            point_xyz, (q.x, q.y, q.z, q.w))
        return (rotated[0] + t.x, rotated[1] + t.y, rotated[2] + t.z)

    @staticmethod
    def _rotate_by_quaternion(v, q):
        x, y, z, w = q
        vx, vy, vz = v
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        )


class Perception:
    """Rule-based RGB-D recognizer for the three Challenge Cup scenes."""

    def __init__(self, rospy, cache, cv2, np):
        self.rospy = rospy
        self.cache = cache
        self.cv2 = cv2
        self.np = np

    def scan_scene(self, scene, passes=6):
        action = "识别{}".format(SCENE_CONFIGS.get(scene, {}).get("title", "场景"))
        _log_action(self.rospy, action)
        estimates = []
        for _ in range(max(1, passes)):
            estimates.extend(self._scan_once(scene, "head"))
            self.rospy.sleep(0.06)
        merged = self._merge_estimates(estimates)
        self.rospy.loginfo("%s识别到 %d 个候选目标", LOG_PREFIX, len(merged))
        for est in merged:
            self.rospy.loginfo(
                "%s  %s/%s conf=%.2f pos=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f)",
                LOG_PREFIX, est.kind, est.label, est.confidence,
                est.x, est.y, est.z, est.size[0], est.size[1], est.size[2])
        if merged:
            _log_success(self.rospy, action)
        else:
            _log_failure(self.rospy, action)
        return merged

    def refine_target(self, target, side):
        camera = "left" if side == "left" else "right"
        estimates = self._scan_once(None, camera, wanted_kind=target.kind)
        if not estimates:
            return None
        estimates.sort(key=lambda est: est.distance_xy(target))
        best = estimates[0]
        if best.distance_xy(target) > 0.20:
            return None
        best.confidence = min(1.0, best.confidence + 0.08)
        return best

    def _scan_once(self, scene, camera_name, wanted_kind=None):
        frame = self._get_decoded_frame(camera_name)
        if frame is None:
            return []
        bgr, depth_m, info, frame_id = frame
        hsv = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2HSV)

        if wanted_kind == "parcel":
            return self._detect_parcels(bgr, hsv, depth_m, info, frame_id)
        if wanted_kind in ("part_a", "part_b", "part_c"):
            return self._detect_parts(bgr, hsv, depth_m, info, frame_id, wanted_kind)
        if wanted_kind == "tray":
            return self._detect_trays(bgr, hsv, depth_m, info, frame_id)

        if scene == "scene1":
            return self._scan_scene1(bgr, hsv, depth_m, info, frame_id)
        if scene == "scene2":
            return self._scan_scene2(bgr, hsv, depth_m, info, frame_id)
        if scene == "scene3":
            return self._scan_scene3(bgr, hsv, depth_m, info, frame_id)
        return []

    def _scan_scene1(self, bgr, hsv, depth_m, info, frame_id):
        estimates = []
        green = self._detect_colored_regions(
            hsv, depth_m, info, frame_id, "green_region", "green",
            [((32, 35, 45), (105, 255, 255))], min_area=350, max_area=220000)
        estimates.extend(self._classify_green_regions(green, include_bins=True))
        estimates.extend(self._detect_parcels(bgr, hsv, depth_m, info, frame_id))
        return estimates

    def _scan_scene2(self, bgr, hsv, depth_m, info, frame_id):
        estimates = []
        bins = [
            ("bin_a", "bin_a", [((92, 50, 40), (132, 255, 255))]),
            ("bin_b", "bin_b", [((5, 70, 45), (28, 255, 255))]),
            ("bin_c", "bin_c", [((128, 45, 35), (166, 255, 255))]),
        ]
        for kind, label, ranges in bins:
            found = self._detect_colored_regions(
                hsv, depth_m, info, frame_id, kind, label,
                ranges, min_area=500, max_area=140000)
            estimates.extend(self._pick_largest_per_kind(found, kind))
        estimates.extend(self._detect_parts(bgr, hsv, depth_m, info, frame_id))
        return estimates

    def _scan_scene3(self, bgr, hsv, depth_m, info, frame_id):
        estimates = []
        green = self._detect_colored_regions(
            hsv, depth_m, info, frame_id, "green_region", "green",
            [((38, 65, 85), (95, 255, 255))], min_area=900, max_area=180000)
        estimates.extend(self._classify_green_regions(green, include_bins=True))
        for est in estimates:
            if est.kind == "drop_box":
                est.kind = "out_box"
                est.label = "out_box"
        estimates.extend(self._detect_trays(bgr, hsv, depth_m, info, frame_id))
        return estimates

    def _get_decoded_frame(self, camera_name):
        color_msg, depth_msg, info = self.cache.get_camera(camera_name)
        if color_msg is None or depth_msg is None or info is None:
            self.rospy.logwarn_throttle(
                2.0, "%s等待 %s RGB-D 数据...", LOG_PREFIX, camera_name)
            return None
        bgr = self._decode_color(color_msg)
        depth_m = self._decode_depth(depth_msg)
        if bgr is None or depth_m is None:
            return None
        if depth_m.shape[:2] != bgr.shape[:2]:
            depth_m = self.cv2.resize(depth_m, (bgr.shape[1], bgr.shape[0]),
                                      interpolation=self.cv2.INTER_NEAREST)
        frame_id = info.header.frame_id or color_msg.header.frame_id
        return bgr, depth_m, info, frame_id

    def _decode_color(self, msg):
        data = self.np.frombuffer(bytes(msg.data), dtype=self.np.uint8)
        img = self.cv2.imdecode(data, self.cv2.IMREAD_COLOR)
        if img is None:
            self.rospy.logwarn_throttle(2.0, "%sRGB图像解码失败", LOG_PREFIX)
            pass
        return img

    def _decode_depth(self, msg):
        raw = bytes(msg.data)
        png_magic = b"\x89PNG"
        offset = raw.find(png_magic)
        if offset < 0:
            offset = 0
        data = self.np.frombuffer(raw[offset:], dtype=self.np.uint8)
        depth = self.cv2.imdecode(data, self.cv2.IMREAD_UNCHANGED)
        if depth is None:
            self.rospy.logwarn_throttle(2.0, "%s深度图像解码失败", LOG_PREFIX)
            return None
        if depth.dtype == self.np.uint16:
            depth = depth.astype(self.np.float32) / 1000.0
        else:
            depth = depth.astype(self.np.float32)
        depth[~self.np.isfinite(depth)] = 0.0
        return depth

    def _detect_parcels(self, bgr, hsv, depth_m, info, frame_id):
        ranges = [
            ((0, 18, 40), (42, 255, 255)),
            ((75, 8, 45), (140, 230, 255)),
            ((0, 0, 70), (45, 95, 230)),
        ]
        found = self._detect_colored_regions(
            hsv, depth_m, info, frame_id, "parcel", "parcel",
            ranges, min_area=90, max_area=42000)
        parcels = []
        for est in found:
            long_side = max(est.size[0], est.size[1])
            short_side = min(est.size[0], est.size[1])
            if 0.010 <= short_side <= 0.20 and 0.025 <= long_side <= 0.30:
                est.confidence = min(0.95, est.confidence + 0.08)
                parcels.append(est)
        return parcels

    def _detect_parts(self, bgr, hsv, depth_m, info, frame_id, wanted_kind=None):
        estimates = []
        if wanted_kind in (None, "part_c"):
            red_ranges = [((0, 70, 40), (12, 255, 255)),
                          ((170, 70, 40), (180, 255, 255))]
            estimates.extend(self._detect_colored_regions(
                hsv, depth_m, info, frame_id, "part_c", "screwdriver",
                red_ranges, min_area=120, max_area=18000))

        if wanted_kind in (None, "part_a"):
            metal_mask = self.cv2.inRange(hsv, (0, 0, 105), (180, 75, 255))
            estimates.extend(self._estimates_from_mask(
                metal_mask, depth_m, info, frame_id,
                "part_a", "metal_part", min_area=90, max_area=15000))

        if wanted_kind in (None, "part_b"):
            dark_mask = self.cv2.inRange(hsv, (0, 0, 0), (180, 125, 90))
            estimates.extend(self._estimates_from_mask(
                dark_mask, depth_m, info, frame_id,
                "part_b", "dark_t_part", min_area=90, max_area=15000))

        filtered = []
        for est in estimates:
            long_side = max(est.size[0], est.size[1])
            short_side = min(est.size[0], est.size[1])
            if est.kind == "part_c":
                if long_side >= 0.035:
                    est.confidence = min(0.98, est.confidence + 0.12)
                    filtered.append(est)
            elif 0.008 <= short_side <= 0.10 and 0.018 <= long_side <= 0.18:
                filtered.append(est)
        return filtered

    def _detect_trays(self, bgr, hsv, depth_m, info, frame_id):
        gray_mask = self.cv2.inRange(hsv, (0, 0, 70), (180, 85, 230))
        gray = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY)
        edges = self.cv2.Canny(gray, 60, 160)
        edges = self.cv2.dilate(edges, self.np.ones((3, 3), self.np.uint8))
        mask = self.cv2.bitwise_and(gray_mask, edges)
        estimates = self._estimates_from_mask(
            mask, depth_m, info, frame_id, "tray", "smt_tray",
            min_area=120, max_area=30000)
        trays = []
        for est in estimates:
            long_side = max(est.size[0], est.size[1])
            short_side = min(est.size[0], est.size[1])
            if 0.04 <= long_side <= 0.35 and 0.015 <= short_side <= 0.20:
                est.confidence = min(0.95, est.confidence + 0.05)
                trays.append(est)
        return trays

    def _detect_colored_regions(self, hsv, depth_m, info, frame_id,
                                kind, label, ranges, min_area, max_area):
        mask = self.np.zeros(hsv.shape[:2], dtype=self.np.uint8)
        for low, high in ranges:
            mask = self.cv2.bitwise_or(
                mask, self.cv2.inRange(hsv, low, high))
        return self._estimates_from_mask(
            mask, depth_m, info, frame_id, kind, label, min_area, max_area)

    def _estimates_from_mask(self, mask, depth_m, info, frame_id,
                             kind, label, min_area, max_area):
        kernel = self.np.ones((5, 5), self.np.uint8)
        clean = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
        clean = self.cv2.morphologyEx(clean, self.cv2.MORPH_CLOSE, kernel)
        contours, _ = self.cv2.findContours(
            clean, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)

        estimates = []
        for contour in contours:
            area = self.cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            estimate = self._contour_to_estimate(
                contour, depth_m, info, frame_id, kind, label, area)
            if estimate is not None:
                estimates.append(estimate)
        estimates.sort(key=lambda est: est.confidence, reverse=True)
        return estimates

    def _contour_to_estimate(self, contour, depth_m, info, frame_id,
                             kind, label, area):
        moments = self.cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            return None
        u = int(moments["m10"] / moments["m00"])
        v = int(moments["m01"] / moments["m00"])
        depth = self._median_depth_near(depth_m, contour, u, v)
        if depth is None or depth <= 0.05 or depth > 5.0:
            return None

        point_camera = self._pixel_to_camera(u, v, depth, info)
        if point_camera is None:
            return None
        point_base = self.cache.transform_point(point_camera, frame_id, BASE_FRAME)
        if point_base is None:
            point_base = point_camera
            source_frame = frame_id
        else:
            source_frame = BASE_FRAME

        rect = self.cv2.minAreaRect(contour)
        (w_px, h_px) = rect[1]
        fx = float(info.K[0]) if info.K[0] else 1.0
        fy = float(info.K[4]) if info.K[4] else fx
        sx = abs(w_px * depth / fx)
        sy = abs(h_px * depth / fy)
        yaw = math.radians(rect[2])
        conf = max(0.10, min(0.90, 0.22 + math.log1p(area) / 11.0))
        if source_frame == BASE_FRAME:
            conf = min(0.98, conf + 0.05)
        return ObjectEstimate(
            label=label,
            kind=kind,
            confidence=conf,
            x=float(point_base[0]),
            y=float(point_base[1]),
            z=float(point_base[2]),
            yaw=yaw,
            size=(float(sx), float(sy), 0.04),
            source_frame=source_frame,
            source=label,
        )

    def _median_depth_near(self, depth_m, contour, u, v):
        mask = self.np.zeros(depth_m.shape[:2], dtype=self.np.uint8)
        self.cv2.drawContours(mask, [contour], -1, 255, -1)
        values = depth_m[mask > 0]
        values = values[(values > 0.05) & (values < 5.0)]
        if values.size < 12:
            y0 = max(0, v - 5)
            y1 = min(depth_m.shape[0], v + 6)
            x0 = max(0, u - 5)
            x1 = min(depth_m.shape[1], u + 6)
            values = depth_m[y0:y1, x0:x1].reshape(-1)
            values = values[(values > 0.05) & (values < 5.0)]
        if values.size == 0:
            return None
        return float(self.np.median(values))

    @staticmethod
    def _pixel_to_camera(u, v, z, info):
        fx = float(info.K[0])
        fy = float(info.K[4])
        cx = float(info.K[2])
        cy = float(info.K[5])
        if fx == 0.0 or fy == 0.0:
            return None
        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        return (x, y, z)

    def _classify_green_regions(self, green_regions, include_bins):
        if not green_regions:
            return []
        regions = sorted(green_regions, key=lambda est: est.size[0] * est.size[1],
                         reverse=True)
        classified = []
        if include_bins:
            box = regions[0]
            box.kind = "drop_box"
            box.label = "drop_box"
            box.confidence = min(0.98, box.confidence + 0.10)
            classified.append(box)
            for est in regions[1:5]:
                if est.distance_xy(box) < 0.10:
                    continue
                est.kind = "weighing_area"
                est.label = "weighing_area"
                est.confidence = min(0.94, est.confidence + 0.06)
                classified.append(est)
                break
        else:
            area = regions[0]
            area.kind = "weighing_area"
            area.label = "weighing_area"
            classified.append(area)
        return classified

    @staticmethod
    def _pick_largest_per_kind(estimates, kind):
        if not estimates:
            return []
        estimates.sort(key=lambda est: est.size[0] * est.size[1], reverse=True)
        best = estimates[0]
        best.kind = kind
        best.label = kind
        best.confidence = min(0.97, best.confidence + 0.10)
        return [best]

    def _merge_estimates(self, estimates):
        groups = []
        for est in sorted(estimates, key=lambda item: item.confidence, reverse=True):
            group = None
            for candidate in groups:
                if candidate[0].kind == est.kind and candidate[0].distance_xy(est) < 0.08:
                    group = candidate
                    break
            if group is None:
                groups.append([est])
            else:
                group.append(est)

        merged = []
        for group in groups:
            weight_sum = sum(max(0.01, est.confidence) for est in group)
            x = sum(est.x * est.confidence for est in group) / weight_sum
            y = sum(est.y * est.confidence for est in group) / weight_sum
            z = sum(est.z * est.confidence for est in group) / weight_sum
            best = max(group, key=lambda est: est.confidence)
            merged.append(ObjectEstimate(
                label=best.label,
                kind=best.kind,
                confidence=min(0.99, sum(est.confidence for est in group) / len(group)),
                x=x,
                y=y,
                z=z,
                yaw=best.yaw,
                size=best.size,
                source_frame=best.source_frame,
                source=best.source,
            ))
        merged.sort(key=lambda est: est.confidence, reverse=True)
        return merged


class RobotControl:
    """ROS control facade used by the task executor."""

    def __init__(self, rospy, geometry_msgs, sensor_msgs, kuavo_msgs):
        self.rospy = rospy
        self.geometry_msgs = geometry_msgs
        self.sensor_msgs = sensor_msgs
        self.kuavo_msgs = kuavo_msgs
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", geometry_msgs.Twist, queue_size=10)
        self.arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", sensor_msgs.JointState, queue_size=10)
        self.arm_target_pub = rospy.Publisher("/kuavo_arm_target_poses", kuavo_msgs.armTargetPoses, queue_size=10)
        self.head_pub = rospy.Publisher("/robot_head_motion_data", kuavo_msgs.robotHeadMotionData, queue_size=10)
        self.claw_srv = self._connect_service("/control_robot_leju_claw", kuavo_msgs.controlLejuClaw, 5.0)
        self.arm_mode_srv = self._connect_service("/arm_traj_change_mode", kuavo_msgs.changeArmCtrlMode, 3.0)
        self.ik_srv = self._connect_service("/ik/two_arm_hand_pose_cmd_srv", kuavo_msgs.twoArmHandPoseCmdSrv, 6.0)
        self.last_claw = {"left": 5.0, "right": 5.0}
        self.last_arm_deg = [0.0] * 14

    def _connect_service(self, name, srv_type, timeout):
        try:
            self.rospy.wait_for_service(name, timeout=timeout)
            return self.rospy.ServiceProxy(name, srv_type)
        except Exception as exc:
            self.rospy.logwarn("%s服务暂不可用 %s: %s", LOG_PREFIX, name, exc)
            return None

    @staticmethod
    def _clamp(value, low, high):
        return max(float(low), min(float(high), float(value)))

    def _clamp_end_effector_target(self, side, x, y, z):
        workspace = HAND_WORKSPACES[side]
        safe_x = self._clamp(x, *workspace["x"])
        safe_y = self._clamp(y, *workspace["y"])
        safe_z = self._clamp(z, *workspace["z"])
        changed = (
            abs(safe_x - float(x)) > 1e-6
            or abs(safe_y - float(y)) > 1e-6
            or abs(safe_z - float(z)) > 1e-6
        )
        return safe_x, safe_y, safe_z, changed

    def prepare(self):
        action = "准备机器人"
        _log_action(self.rospy, action)
        head_ok = self.set_head_target(HEAD_LOOK_TARGET, settle=HEAD_SETTLE_TIME)
        mode_ok = self.set_arm_control_mode(2)
        left_ok = self.open_claw("left")
        right_ok = self.open_claw("right")
        arm_ok = self.publish_arm_degrees([0.0] * 14, duration=2.0)
        ok = head_ok and mode_ok and left_ok and right_ok and arm_ok
        if ok:
            _log_success(self.rospy, action)
        else:
            _log_failure(self.rospy, action)
        return ok

    def set_arm_control_mode(self, mode):
        action = "切换手臂控制模式"
        _log_action(self.rospy, action)
        if self.arm_mode_srv is None:
            _log_failure(self.rospy, action)
            return False
        req = self.kuavo_msgs.changeArmCtrlModeRequest()
        req.control_mode = int(mode)
        try:
            res = self.arm_mode_srv(req)
        except Exception as exc:
            self.rospy.logwarn("%s切换手臂控制模式失败: %s", LOG_PREFIX, exc)
            _log_failure(self.rospy, action)
            return False
        if not getattr(res, "result", False):
            actual_mode = getattr(res, "mode", None)
            if actual_mode != int(mode):
                self.rospy.logwarn(
                    "%s手臂控制模式未切换: mode=%s message=%s",
                    LOG_PREFIX, actual_mode, getattr(res, "message", ""))
                _log_failure(self.rospy, action)
                return False
        _log_success(self.rospy, action)
        return True

    def set_head_target(self, joint_data, settle=0.0):
        action = "调整头部视角"
        _log_action(self.rospy, action)
        msg = self.kuavo_msgs.robotHeadMotionData()
        msg.joint_data = [float(value) for value in joint_data]
        for _ in range(5):
            if self.rospy.is_shutdown():
                _log_failure(self.rospy, action)
                return False
            self.head_pub.publish(msg)
            self.rospy.sleep(0.1)
        if settle > 0.0:
            self.rospy.sleep(float(settle))
        _log_success(self.rospy, action)
        return True

    def open_claw(self, side):
        return self.set_claw(side, 5.0)

    def close_claw(self, side):
        return self.set_claw(side, 88.0)

    def set_claw(self, side, percent):
        side_name = {"left": "左夹爪", "right": "右夹爪"}.get(side, "双夹爪")
        action = "{}{}".format("关闭" if float(percent) >= 50.0 else "打开", side_name)
        _log_action(self.rospy, action)
        if self.claw_srv is None:
            self.rospy.logwarn("%s夹爪服务不可用，跳过夹爪命令", LOG_PREFIX)
            _log_failure(self.rospy, action)
            return False
        percent = max(0.0, min(100.0, float(percent)))
        if side == "left":
            left, right = percent, self.last_claw["right"]
        elif side == "right":
            left, right = self.last_claw["left"], percent
        else:
            left = right = percent
        req = self.kuavo_msgs.controlLejuClawRequest()
        req.data.name = ["left_claw", "right_claw"]
        req.data.position = [left, right]
        req.data.velocity = [55.0, 55.0]
        req.data.effort = [1.2, 1.2]
        try:
            res = self.claw_srv(req)
        except Exception as exc:
            self.rospy.logwarn("%s夹爪命令失败: %s", LOG_PREFIX, exc)
            _log_failure(self.rospy, action)
            return False
        if not getattr(res, "success", False):
            self.rospy.logwarn(
                "%s夹爪命令被拒绝: %s", LOG_PREFIX, getattr(res, "message", ""))
            _log_failure(self.rospy, action)
            return False
        self.last_claw["left"] = left
        self.last_claw["right"] = right
        _log_success(self.rospy, action)
        return True

    def publish_arm_degrees(self, arm_deg, duration=2.0):
        action = "发布手臂关节目标"
        _log_action(self.rospy, action)
        if len(arm_deg) != 14:
            self.rospy.logwarn("%s手臂目标长度错误: %d", LOG_PREFIX, len(arm_deg))
            _log_failure(self.rospy, action)
            return False
        msg = self.kuavo_msgs.armTargetPoses()
        msg.times = [float(max(0.2, duration))]
        msg.values = [float(value) for value in arm_deg]
        msg.frame = 0
        self.arm_target_pub.publish(msg)

        joint = self.sensor_msgs.JointState()
        joint.header.stamp = self.rospy.Time.now()
        joint.name = ARM_JOINT_NAMES
        joint.position = [float(value) for value in arm_deg]
        self.arm_traj_pub.publish(joint)
        self.last_arm_deg = list(arm_deg)
        self.rospy.sleep(max(0.2, duration))
        _log_success(self.rospy, action)
        return True

    def move_end_effector(self, side, x, y, z, yaw=0.0, duration=2.5):
        side_name = "左手" if side == "left" else "右手"
        action = "移动{}末端".format(side_name)
        _log_action(self.rospy, action)
        if self.ik_srv is None:
            self.rospy.logwarn("%sIK服务不可用，无法移动末端", LOG_PREFIX)
            _log_failure(self.rospy, action)
            return False

        safe_x, safe_y, safe_z, clamped = self._clamp_end_effector_target(side, x, y, z)
        if clamped:
            self.rospy.logwarn(
                "%s%s目标超出保守工作空间，已从 [%.3f, %.3f, %.3f] 限制到 [%.3f, %.3f, %.3f]",
                LOG_PREFIX, side_name, float(x), float(y), float(z), safe_x, safe_y, safe_z)

        req = self.kuavo_msgs.twoArmHandPoseCmd()
        req.frame = 2
        req.use_custom_ik_param = True
        req.joint_angles_as_q0 = True
        req.ik_param = self._default_ik_param()

        left_pose = req.hand_poses.left_pose
        right_pose = req.hand_poses.right_pose
        q0_rad = [math.radians(value) for value in self.last_arm_deg]
        self._fill_pose(left_pose, *HAND_WORKSPACES["left"]["park"],
                        yaw=yaw, joint_angles=q0_rad[:7])
        self._fill_pose(right_pose, *HAND_WORKSPACES["right"]["park"],
                        yaw=yaw, joint_angles=q0_rad[7:14])
        if side == "left":
            self._fill_pose(left_pose, safe_x, safe_y, safe_z, yaw,
                            joint_angles=q0_rad[:7])
        else:
            self._fill_pose(right_pose, safe_x, safe_y, safe_z, yaw,
                            joint_angles=q0_rad[7:14])

        try:
            res = self.ik_srv(req)
        except Exception as exc:
            self.rospy.logwarn("%sIK调用失败: %s", LOG_PREFIX, exc)
            _log_failure(self.rospy, action)
            return False
        if not getattr(res, "success", False):
            reason = getattr(res, "error_reason", "")
            if reason:
                self.rospy.logwarn("%sIK求解失败原因: %s", LOG_PREFIX, reason)
            _log_failure(self.rospy, action)
            return False

        q_arm = list(getattr(res, "q_arm", []))
        if len(q_arm) != 14:
            self.rospy.logwarn("%sIK返回关节数异常: %d", LOG_PREFIX, len(q_arm))
            _log_failure(self.rospy, action)
            return False
        arm_deg = [math.degrees(value) for value in q_arm]
        if self.publish_arm_degrees(arm_deg, duration=duration):
            _log_success(self.rospy, action)
            return True
        _log_failure(self.rospy, action)
        return False

    def _default_ik_param(self):
        params = self.kuavo_msgs.ikSolveParam()
        params.major_optimality_tol = 1e-3
        params.major_feasibility_tol = 1e-3
        params.minor_feasibility_tol = 1e-3
        params.major_iterations_limit = 120
        params.oritation_constraint_tol = 2e-3
        params.pos_constraint_tol = 2e-3
        params.pos_cost_weight = 0.0
        return params

    @staticmethod
    def _fill_pose(pose, x, y, z, yaw, joint_angles=None):
        pose.pos_xyz = [float(x), float(y), float(z)]
        pose.quat_xyzw = [0.0, 1.0, 0.0, 0.0]
        pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        if joint_angles is None:
            pose.joint_angles = [0.0] * 7
        else:
            pose.joint_angles = [float(value) for value in joint_angles]

    def stop_base(self):
        twist = self.geometry_msgs.Twist()
        self.cmd_vel_pub.publish(twist)

    def drive_timed(self, vx=0.0, vy=0.0, wz=0.0, duration=0.0):
        twist = self.geometry_msgs.Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.angular.z = float(wz)
        rate = self.rospy.Rate(10)
        end_time = time.time() + max(0.0, duration)
        while not self.rospy.is_shutdown() and time.time() < end_time:
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        self.stop_base()


class TaskExecutor:
    """Pick-place state machine shared by all scenes."""

    def __init__(self, rospy, perception, control, cache):
        self.rospy = rospy
        self.perception = perception
        self.control = control
        self.cache = cache
        self.base_forward_offset = 0.0

    def execute_pick_place(self, target, place, max_attempts=2):
        preferred_side = self._choose_side(target, place)
        sides = [preferred_side, self._other_side(preferred_side)]
        for attempt in range(1, max_attempts + 1):
            for side in sides:
                attempt_target = target
                refine_action = "复核目标"
                _log_action(self.rospy, refine_action)
                refined = self.perception.refine_target(target, side)
                if refined is not None:
                    self.rospy.loginfo("%s腕部相机复核成功: %s", LOG_PREFIX, refined.kind)
                    _log_success(self.rospy, refine_action)
                    attempt_target = refined
                else:
                    self.rospy.loginfo("%s腕部复核不可用，使用头部相机估计", LOG_PREFIX)
                    _log_failure(self.rospy, refine_action)

                if self._run_pick_place_attempt(attempt_target, place, side):
                    return True
                self.rospy.logwarn("%s目标 %s 第 %d 次执行失败", LOG_PREFIX, target.kind, attempt)
                side_name = "左手" if side == "left" else "右手"
                _log_failure(self.rospy, "第{}次{}抓取放置".format(attempt, side_name))
                self.control.open_claw(side)
                self.control.publish_arm_degrees([0.0] * 14, duration=1.4)
        return False

    def _run_pick_place_attempt(self, target, place, side):
        target = self._estimate_in_current_base(target)
        place = self._estimate_in_current_base(place)
        action = "抓取{}并放置到{}".format(target.kind, place.kind)
        grasp_z = self._safe_z(target.z + 0.035)
        pre_z = self._safe_z(grasp_z + 0.16)
        lift_z = self._safe_z(grasp_z + 0.24)
        place_z = self._safe_z(place.z + 0.14)
        pre_place_z = self._safe_z(place_z + 0.12)

        self.rospy.loginfo("%s抓取 %s -> %s, side=%s", LOG_PREFIX, target.kind, place.kind, side)
        _log_action(self.rospy, action)
        if not self.control.open_claw(side):
            _log_failure(self.rospy, action)
            return False
        if not self._approach_fixed_target(target, "抓取点", [target, place]):
            _log_failure(self.rospy, action)
            return False
        steps = [
            (target.x, target.y, pre_z, target.yaw, 2.2),
            (target.x, target.y, grasp_z, target.yaw, 1.4),
        ]
        for x, y, z, yaw, duration in steps:
            if not self.control.move_end_effector(side, x, y, z, yaw, duration):
                _log_failure(self.rospy, action)
                return False
        if not self.control.close_claw(side):
            _log_failure(self.rospy, action)
            return False
        self.rospy.sleep(0.7)
        if not self._claw_has_feedback(side):
            self.rospy.logwarn("%s未确认夹爪到位，继续执行保守抬升", LOG_PREFIX)
            _log_failure(self.rospy, "确认夹爪到位")

        steps = [
            (target.x, target.y, lift_z, target.yaw, 1.4),
        ]
        for x, y, z, yaw, duration in steps:
            if not self.control.move_end_effector(side, x, y, z, yaw, duration):
                _log_failure(self.rospy, action)
                return False
        if not self._approach_fixed_target(place, "放置点", [place]):
            _log_failure(self.rospy, action)
            return False
        steps = [
            (place.x, place.y, pre_place_z, place.yaw, 2.4),
            (place.x, place.y, place_z, place.yaw, 1.2),
        ]
        for x, y, z, yaw, duration in steps:
            if not self.control.move_end_effector(side, x, y, z, yaw, duration):
                _log_failure(self.rospy, action)
                return False
        if not self.control.open_claw(side):
            _log_failure(self.rospy, action)
            return False
        self.rospy.sleep(0.5)
        if self.control.move_end_effector(
            side, place.x, place.y, pre_place_z + 0.04, place.yaw, 1.2
        ):
            _log_success(self.rospy, action)
            return True
        _log_failure(self.rospy, action)
        return False

    def _approach_fixed_target(self, primary, label, shifted_estimates):
        x_max = HAND_WORKSPACES["left"]["x"][1]
        if primary.x <= x_max:
            return True

        distance = min(
            BASE_APPROACH_MAX_DISTANCE,
            max(0.0, float(primary.x) - BASE_APPROACH_TARGET_X),
        )
        if distance < BASE_APPROACH_MIN_DISTANCE:
            return True

        action = "底盘前移接近{}".format(label)
        _log_action(self.rospy, action)
        self.control.drive_timed(
            vx=BASE_APPROACH_SPEED,
            duration=distance / BASE_APPROACH_SPEED,
        )
        self.base_forward_offset += distance
        for estimate in shifted_estimates:
            estimate.x = float(estimate.x) - distance
        self.rospy.sleep(BASE_APPROACH_SETTLE)

        if primary.x <= x_max:
            _log_success(self.rospy, action)
            return True
        self.rospy.logwarn(
            "%s%s仍超出手臂前向工作空间: x=%.3f, limit=%.3f",
            LOG_PREFIX, label, float(primary.x), x_max)
        _log_failure(self.rospy, action)
        return False

    def _estimate_in_current_base(self, estimate):
        adjusted = replace(estimate)
        adjusted.x = float(adjusted.x) - self.base_forward_offset
        return adjusted

    def _choose_side(self, target, place=None):
        left_cost = self._workspace_cost("left", target, place)
        right_cost = self._workspace_cost("right", target, place)
        if left_cost < right_cost:
            return "left"
        if right_cost < left_cost:
            return "right"
        y_sum = float(target.y) + (float(place.y) if place is not None else 0.0)
        return "left" if y_sum >= 0.0 else "right"

    @staticmethod
    def _other_side(side):
        return "right" if side == "left" else "left"

    @staticmethod
    def _workspace_cost(side, *estimates):
        workspace = HAND_WORKSPACES[side]
        cost = 0.0
        for estimate in estimates:
            if estimate is None:
                continue
            for axis, value in (
                ("x", estimate.x),
                ("y", estimate.y),
                ("z", TaskExecutor._safe_z(estimate.z)),
            ):
                low, high = workspace[axis]
                value = float(value)
                if value < low:
                    cost += (low - value) ** 2
                elif value > high:
                    cost += (value - high) ** 2
        return cost

    def _claw_has_feedback(self, side):
        state_msg = self.cache.get_claw_state()
        if state_msg is None or not getattr(state_msg, "state", None):
            return False
        idx = 0 if side == "left" else 1
        if len(state_msg.state) <= idx:
            return False
        return state_msg.state[idx] >= 2

    @staticmethod
    def _safe_z(z):
        z = float(z)
        if z > IK_WORLD_Z_THRESHOLD:
            z += IK_WORLD_TO_LOCAL_Z_OFFSET
        z_min, z_max = HAND_WORKSPACES["left"]["z"]
        return max(z_min, min(z_max, z))


class BaseSceneTask:
    def __init__(self, rospy, perception, executor):
        self.rospy = rospy
        self.perception = perception
        self.executor = executor

    @staticmethod
    def _best(estimates, kind):
        candidates = [est for est in estimates if est.kind == kind]
        if not candidates:
            return None
        candidates.sort(key=lambda est: est.confidence, reverse=True)
        return candidates[0]

    @staticmethod
    def _all(estimates, kind, limit=None):
        candidates = [est for est in estimates if est.kind == kind]
        candidates.sort(key=lambda est: (-est.confidence, abs(est.y), est.x))
        return candidates if limit is None else candidates[:limit]

    def _rescan(self, scene, retries=8):
        for _ in range(retries):
            estimates = self.perception.scan_scene(scene)
            if estimates:
                return estimates
            self.rospy.sleep(1.0)
        return []


class Scene1Task(BaseSceneTask):
    def run(self):
        estimates = self._rescan("scene1")
        drop_box = self._best(estimates, "drop_box")
        weighing_area = self._best(estimates, "weighing_area")
        parcels = self._all(estimates, "parcel", limit=4)
        if drop_box is None or not parcels:
            self.rospy.logerr("%sscene1 识别失败：缺少分拣箱或包裹", LOG_PREFIX)
            _log_error(self.rospy, "识别scene1目标")
            return False

        for parcel in parcels:
            if self.rospy.is_shutdown():
                _log_failure(self.rospy, "执行scene1任务")
                return False
            if weighing_area is not None:
                if not self.executor.execute_pick_place(parcel, weighing_area):
                    continue
                self.rospy.sleep(0.8)
                staged = ObjectEstimate(
                    label="parcel_on_scale",
                    kind="parcel",
                    confidence=0.70,
                    x=weighing_area.x,
                    y=weighing_area.y,
                    z=weighing_area.z + 0.035,
                    yaw=weighing_area.yaw,
                    size=parcel.size,
                    source_frame=BASE_FRAME,
                    source="staged",
                )
                self.executor.execute_pick_place(staged, drop_box)
            else:
                self.executor.execute_pick_place(parcel, drop_box)
        return True


class Scene2Task(BaseSceneTask):
    def run(self):
        estimates = self._rescan("scene2")
        bins = {
            "part_a": self._best(estimates, "bin_a"),
            "part_b": self._best(estimates, "bin_b"),
            "part_c": self._best(estimates, "bin_c"),
        }
        if any(target is None for target in bins.values()):
            self.rospy.logerr("%sscene2 识别失败：缺少一个或多个收纳盒", LOG_PREFIX)
            _log_error(self.rospy, "识别scene2收纳盒")
            return False

        targets = []
        for kind in ("part_a", "part_b", "part_c"):
            targets.extend(self._all(estimates, kind, limit=2))
        if not targets:
            self.rospy.logerr("%sscene2 识别失败：未找到零件", LOG_PREFIX)
            _log_error(self.rospy, "识别scene2零件")
            return False

        for target in targets:
            if self.rospy.is_shutdown():
                _log_failure(self.rospy, "执行scene2任务")
                return False
            place = bins.get(target.kind)
            if place is not None:
                self.executor.execute_pick_place(target, place)
        return True


class Scene3Task(BaseSceneTask):
    def run(self):
        estimates = self._rescan("scene3")
        out_box = self._best(estimates, "out_box") or self._best(estimates, "drop_box")
        trays = self._all(estimates, "tray", limit=5)
        if out_box is None or not trays:
            self.rospy.logerr("%sscene3 识别失败：缺少出库箱或料盘", LOG_PREFIX)
            _log_error(self.rospy, "识别scene3目标")
            return False

        for tray in trays:
            if self.rospy.is_shutdown():
                _log_failure(self.rospy, "执行scene3任务")
                return False
            self.executor.execute_pick_place(tray, out_box, max_attempts=2)
        return True


def _load_launcher():
    # 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
    # 从那里导入，确保完整性校验无法被绕过。
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy
    import geometry_msgs.msg as geometry_msgs
    import sensor_msgs.msg as sensor_msgs
    import tf2_ros
    import kuavo_msgs.msg as kuavo_msg_types
    import kuavo_msgs.srv as kuavo_srv_types

    try:
        import cv2
        import numpy as np
    except Exception as exc:
        rospy.logerr("%s缺少视觉依赖 OpenCV/NumPy，无法执行识别: %s", LOG_PREFIX, exc)
        _log_error(rospy, "加载视觉依赖")
        return

    class KuavoTypes:
        sensorsData = kuavo_msg_types.sensorsData
        lejuClawState = kuavo_msg_types.lejuClawState
        robotHeadMotionData = kuavo_msg_types.robotHeadMotionData
        armTargetPoses = kuavo_msg_types.armTargetPoses
        ikSolveParam = kuavo_msg_types.ikSolveParam
        twoArmHandPoseCmd = kuavo_msg_types.twoArmHandPoseCmd
        controlLejuClaw = kuavo_srv_types.controlLejuClaw
        controlLejuClawRequest = kuavo_srv_types.controlLejuClawRequest
        changeArmCtrlMode = kuavo_srv_types.changeArmCtrlMode
        changeArmCtrlModeRequest = kuavo_srv_types.changeArmCtrlModeRequest
        twoArmHandPoseCmdSrv = kuavo_srv_types.twoArmHandPoseCmdSrv

    rospy.loginfo("%s=== %s任务启动 ===", LOG_PREFIX, config["title"])
    _log_action(rospy, config["title"])

    kuavo_types = KuavoTypes()
    cache = SensorCache(rospy, sensor_msgs, kuavo_types, tf2_ros)
    perception = Perception(rospy, cache, cv2, np)
    control = RobotControl(rospy, geometry_msgs, sensor_msgs, kuavo_types)
    executor = TaskExecutor(rospy, perception, control, cache)

    camera_action = "等待头部RGB-D就绪"
    _log_action(rospy, camera_action)
    if cache.wait_for_head_camera(timeout=8.0):
        _log_success(rospy, camera_action)
    else:
        rospy.logwarn("%s头部RGB-D尚未就绪，继续尝试任务流程", LOG_PREFIX)
        _log_failure(rospy, camera_action)
    control.prepare()
    sensor_action = "等待传感器数据稳定"
    _log_action(rospy, sensor_action)
    if cache.wait_for_sensor_data(timeout=8.0):
        rospy.sleep(1.0)
        _log_success(rospy, sensor_action)
    else:
        rospy.logwarn("%s未等到 /sensors_data_raw 更新，继续尝试任务流程", LOG_PREFIX)
        _log_failure(rospy, sensor_action)
    rospy.loginfo("%s场景实例已初始化。", LOG_PREFIX)
    _log_success(rospy, "初始化场景实例")

    if scene == "scene1":
        ok = Scene1Task(rospy, perception, executor).run()
    elif scene == "scene2":
        ok = Scene2Task(rospy, perception, executor).run()
    else:
        ok = Scene3Task(rospy, perception, executor).run()

    control.stop_base()
    if ok:
        rospy.loginfo("%s=== %s任务流程完成 ===", LOG_PREFIX, config["title"])
        _log_success(rospy, config["title"])
    else:
        rospy.logwarn("%s=== %s任务流程未完成 ===", LOG_PREFIX, config["title"])
        _log_failure(rospy, config["title"])


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认按 scene 自动设置")
    parser.add_argument("--timeout", type=int, default=120,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
    )


if __name__ == "__main__":
    main()
