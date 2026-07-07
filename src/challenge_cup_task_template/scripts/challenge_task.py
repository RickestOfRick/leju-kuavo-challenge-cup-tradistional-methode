#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import glob
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 全局配置与常量区
# ---------------------------------------------------------------------------

BASE_FRAME = "base_link"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE1_YOLO_MODEL = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "models", "scene1_objects.pt")
)
DEFAULT_SCENE3_YOLO_MODEL = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "models", "yolo", "scene3_smt_tray.pt")
)
SCENE1_YOLO_CLASS_NAME = "parcel"
SCENE1_YOLO_TARGET_BIN = "drop_box"
SCENE1_YOLO_CLASS_NAMES = ("parcel", "weighing_area", "drop_box")
SCENE1_YOLO_CLASS_ALIASES = {
    "parcel": "parcel",
    "package": "parcel",
    "package_box": "parcel",
    "delivery_package": "parcel",
    "cardboard_box": "parcel",
    "weighing_area": "weighing_area",
    "weigh_area": "weighing_area",
    "scale_area": "weighing_area",
    "drop_box": "drop_box",
    "sorting_box": "drop_box",
    "destination_box": "drop_box",
}
SCENE1_YOLO_TARGET_BINS = {
    "parcel": "drop_box",
    "weighing_area": None,
    "drop_box": None,
}
SCENE1_YOLO_DEFAULT_CONF = 0.45
SCENE1_YOLO_DEFAULT_IMGSZ = 640
SCENE1_YOLO_CENTER_DEPTH_FRACTION = 0.45
SCENE3_YOLO_CLASS_NAME = "smt_tray"
SCENE3_YOLO_TARGET_BIN = "outbound_box"
SCENE3_YOLO_WORLD_PROMPTS = (
    "smt tray",
    "tray",
    "electronics tray",
    "component tray",
    "plastic tray",
    "",
)
SCENE3_YOLO_CLASS_ALIASES = {
    "tray": SCENE3_YOLO_CLASS_NAME,
    "charging_tray": SCENE3_YOLO_CLASS_NAME,
    "smt_tray": SCENE3_YOLO_CLASS_NAME,
    "smt tray": SCENE3_YOLO_CLASS_NAME,
    "smttray": SCENE3_YOLO_CLASS_NAME,
    "electronics_tray": SCENE3_YOLO_CLASS_NAME,
    "electronics tray": SCENE3_YOLO_CLASS_NAME,
    "component_tray": SCENE3_YOLO_CLASS_NAME,
    "component tray": SCENE3_YOLO_CLASS_NAME,
    "plastic_tray": SCENE3_YOLO_CLASS_NAME,
    "plastic tray": SCENE3_YOLO_CLASS_NAME,
}
SCENE3_SHELF_SPLIT_Z_M = 0.95
SCENE3_APPROACH_TARGET_X_M = 0.72
SCENE3_APPROACH_MAX_DETECT_X_M = 2.60
SCENE3_APPROACH_SPEED_MPS = 0.16
SCENE3_APPROACH_MIN_DURATION_SEC = 1.0
SCENE3_APPROACH_MAX_DURATION_SEC = 5.0
SCENE3_APPROACH_RETRY = 4

ARM_JOINT_NAMES = [
    "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
    "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
    "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
    "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
]

HOME_ARM_DEG = [
    20, 0, 0, -30, 0, 0, 0,
    20, 0, 0, -30, 0, 0, 0,
]

WORK_ARM_DEG = [
    30, 10, 10, -110, -55, 0, 0,
    30, -10, -10, -110, 55, 0, 0,
]

SIDE_LIFT_ARM_DEG = [
    20, 65, 0, -40, 0, 0, 0,
    20, -65, 0, -40, 0, 0, 0,
]

ARM_MODE_EXTERNAL_CONTROL = 2
ARM_MODE_AUTO_SWING = 1
ARM_TRAJ_HZ = 50.0
CLAW_OPEN_PERCENT = 5.0
CLAW_CLOSE_PERCENT = 88.0

SAFE_PRE_GRASP_DZ_M = 0.14
SAFE_LIFT_DZ_M = 0.18
SAFE_PLACE_DZ_M = 0.12
SAFE_MIN_Z_BASE_M = -0.14
SAFE_MAX_Z_BASE_M = 0.85
SCENE_SAFE_MAX_Z_BASE_M = {
    "scene3": 1.30,
}
REACH_X_MIN_M = 0.12
REACH_X_MAX_M = 0.88
TARGET_WAIT_SEC = 8.0
SCENE2_STABLE_TARGET_COUNT = 6
SCENE2_STABLE_FRAMES = 2
SCENE2_STABLE_POS_EPS_M = 0.03
SCENE2_DETECT_RETRY = 4

TARGET_BOUNDS_BASE = {
    "default": {
        "x_min": -0.20,
        "x_max": 1.10,
        "z_min": -0.25,
        "z_max": 0.90,
    },
    "scene3": {
        "x_min": -0.20,
        "x_max": 0.95,
        "z_min": -0.25,
        "z_max": 1.35,
    },
}

CAMERA_SPECS = {
    "head": {
        "label": "头部相机",
        "color": "/cam_h/color/image_raw/compressed",
        "depth": "/cam_h/depth/image_raw/compressedDepth",
        "info": "/cam_h/color/camera_info",
    },
    "left_wrist": {
        "label": "左腕相机",
        "color": "/cam_l/color/image_raw/compressed",
        "depth": "/cam_l/depth/image_rect_raw/compressedDepth",
        "info": "/cam_l/color/camera_info",
    },
    "right_wrist": {
        "label": "右腕相机",
        "color": "/cam_r/color/image_raw/compressed",
        "depth": "/cam_r/depth/image_rect_raw/compressedDepth",
        "info": "/cam_r/color/camera_info",
    },
}

# HSV 规则阈值使用公开材质颜色做基线。阈值保守放宽，实际筛选还会叠加面积、
# 长宽比、深度有效性和去重。
TARGET_RULES = {
    "scene1": [
        {
            "class_name": "parcel",
            "target_bin": "drop_box",
            "hsv_ranges": [
                ((5, 35, 80), (35, 255, 255)),      # brown/yellow/orange parcel
                ((88, 15, 115), (128, 120, 255)),   # pale blue parcel
            ],
            "min_area_px": 120,
            "max_area_px": 35000,
            "min_aspect": 0.25,
            "max_aspect": 4.2,
            "min_extent": 0.22,
        },
    ],
    "scene2": [
        {
            "class_name": "part_type_a",
            "target_bin": "sorting_bin_a",
            "hsv_ranges": [((0, 0, 115), (180, 65, 255))],
            "min_area_px": 70,
            "max_area_px": 18000,
            "min_aspect": 0.18,
            "max_aspect": 5.0,
            "min_extent": 0.12,
        },
        {
            "class_name": "part_type_b",
            "target_bin": "sorting_bin_b",
            "hsv_ranges": [((0, 0, 0), (180, 90, 85))],
            "min_area_px": 70,
            "max_area_px": 18000,
            "min_aspect": 0.18,
            "max_aspect": 5.0,
            "min_extent": 0.12,
        },
        {
            "class_name": "part_type_c",
            "target_bin": "sorting_bin_c",
            "hsv_ranges": [
                ((0, 80, 60), (12, 255, 255)),
                ((170, 80, 60), (179, 255, 255)),
            ],
            "min_area_px": 45,
            "max_area_px": 14000,
            "min_aspect": 0.12,
            "max_aspect": 8.0,
            "min_extent": 0.08,
        },
    ],
}

SCENE2_YOLO_CLASS_ALIASES = {
    "pipe_fitting": "part_type_a",
    "steel_pipe_fasteners": "part_type_a",
    "part_type_a": "part_type_a",
    "pipe_clamp": "part_type_b",
    "t_junction": "part_type_b",
    "part_type_b": "part_type_b",
    "screwdriver": "part_type_c",
    "part_type_c": "part_type_c",
}

SCENE2_YOLO_MODEL_PATTERNS = [
    "models/yolo/best.pt",
    "models/yolo/yolov8n_seg_scene2_demo.pt",
    "src/challenge_cup_task_template/models/yolo/best.pt",
    "src/challenge_cup_task_template/models/yolo/yolov8n_seg_scene2_demo.pt",
    "runs/**/weights/best.pt",
]

SCENE2_GRASP_OFFSETS_BASE = {
    # base_link 下的经验偏置。视觉给出的是实例可见区域中心，抓取点需要略微移到
    # 稳定受力位置；数值保持保守，后续按实测 seed 微调。
    "part_type_a": (0.00, 0.00, -0.015),
    "part_type_b": (0.00, 0.00, -0.010),
    "part_type_c": (0.015, 0.00, -0.012),
}

SCENE2_NOMINAL_PARTS_BASE = [
    # 仅作为视觉完全失败时的保守退路。正式策略仍以相机/深度/TF 检测为主。
    ("part_type_a", (0.29, -0.35, -0.12)),
    ("part_type_a", (0.29, -0.25, -0.12)),
    ("part_type_b", (0.29, -0.07, -0.10)),
    ("part_type_b", (0.29, 0.03, -0.10)),
    ("part_type_c", (0.31, 0.19, -0.10)),
    ("part_type_c", (0.31, 0.33, -0.10)),
]

BIN_RULES = {
    "scene1": [
        {
            "class_name": "weighing_area",
            "hsv_ranges": [((45, 60, 60), (90, 255, 255))],
            "min_area_px": 400,
            "max_area_px": 50000,
            "min_aspect": 0.35,
            "max_aspect": 3.0,
            "min_extent": 0.25,
        },
        {
            "class_name": "drop_box",
            "hsv_ranges": [((45, 45, 45), (95, 255, 255))],
            "min_area_px": 500,
            "max_area_px": 90000,
            "min_aspect": 0.35,
            "max_aspect": 4.0,
            "min_extent": 0.20,
        },
    ],
    "scene2": [
        {
            "class_name": "sorting_bin_a",
            "hsv_ranges": [((95, 80, 55), (130, 255, 255))],
            "min_area_px": 400,
            "max_area_px": 90000,
            "min_aspect": 0.35,
            "max_aspect": 4.0,
            "min_extent": 0.20,
        },
        {
            "class_name": "sorting_bin_b",
            "hsv_ranges": [((6, 80, 70), (28, 255, 255))],
            "min_area_px": 400,
            "max_area_px": 90000,
            "min_aspect": 0.35,
            "max_aspect": 4.0,
            "min_extent": 0.20,
        },
        {
            "class_name": "sorting_bin_c",
            "hsv_ranges": [((130, 45, 50), (165, 255, 255))],
            "min_area_px": 400,
            "max_area_px": 90000,
            "min_aspect": 0.35,
            "max_aspect": 4.0,
            "min_extent": 0.20,
        },
    ],
}

# 放置目标坐标系为 base_link，来源于公开场景几何与样例控制脚本的
# 人工标定量级；场景三出库区直接使用该预设点。
PLACE_TARGETS_BASE = {
    "scene1": {
        "weighing_area": (0.42, -0.56, 0.12),
        "drop_box": (0.62, 0.30, 0.20),
    },
    "scene2": {
        "sorting_bin_a": (0.56, -0.43, 0.18),
        "sorting_bin_b": (0.56, 0.00, 0.18),
        "sorting_bin_c": (0.56, 0.43, 0.18),
    },
    "scene3": {
        "outbound_box": (0.58, 0.00, 0.22),
    },
}

SCENE_TARGET_LIMIT = {
    "scene1": 4,
    "scene2": 6,
    "scene3": 2,
}

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


# ---------------------------------------------------------------------------
# 调试输出工具区
# ---------------------------------------------------------------------------


def log_stage(scene: str, stage: str, message: str, *args: Any) -> None:
    import rospy

    rospy.loginfo("[%s][%s] " + message, scene, stage, *args)


def log_warn(scene: str, stage: str, message: str, *args: Any) -> None:
    import rospy

    rospy.logwarn("[%s][%s] " + message, scene, stage, *args)


def log_error(scene: str, stage: str, message: str, *args: Any) -> None:
    import rospy

    rospy.logerr("[%s][%s] " + message, scene, stage, *args)


def _fmt_xyz(xyz: Optional[Sequence[float]]) -> str:
    if xyz is None:
        return "None"
    return "({:.3f},{:.3f},{:.3f})m".format(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def log_target_list(scene: str, stage: str, targets: Sequence["PerceptionTarget"],
                    max_items: int = 8) -> None:
    import rospy

    rospy.loginfo("[%s][%s] targets=%d", scene, stage, len(targets))
    for i, target in enumerate(list(targets)[:max_items]):
        rospy.loginfo(
            "[%s][%s][#%02d] class=%s conf=%.2f cam=%s uv=%s base=%s bin=%s area=%.0f",
            scene,
            stage,
            i + 1,
            target.class_name,
            target.confidence,
            target.camera,
            target.center_uv,
            _fmt_xyz(target.position_base),
            target.target_bin,
            target.area_px,
        )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def distance3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def quat_down(yaw_rad: float = 0.0) -> List[float]:
    # 末端近似朝下；yaw 用于在桌面平面内调整夹爪方向。
    half_yaw = yaw_rad * 0.5
    half_pitch = -math.pi * 0.25
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    return [sy * sp, cy * sp, sy * cp, cy * cp]


# ---------------------------------------------------------------------------
# 轻量数据结构区
# ---------------------------------------------------------------------------


@dataclass
class PerceptionTarget:
    class_name: str
    confidence: float
    center_uv: Optional[Tuple[int, int]]
    position_base: Optional[Tuple[float, float, float]]
    position_camera: Optional[Tuple[float, float, float]]
    camera: str
    area_px: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    yaw_rad: Optional[float] = None
    target_bin: Optional[str] = None
    timestamp: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    ok: bool
    stage: str
    reason: str = ""
    target: Optional[PerceptionTarget] = None
    duration_sec: float = 0.0
    retry_count: int = 0


# ---------------------------------------------------------------------------
# 底层 ROS/硬件适配层
# ---------------------------------------------------------------------------


class RobotBase:
    def __init__(self, cmd_vel_pub: Any, arm_traj_pub: Any, scene: str):
        import rospy
        from kuavo_msgs.msg import lejuClawCommand, lejuClawState, sensorsData

        self.rospy = rospy
        self.scene = scene
        self.cmd_vel_pub = cmd_vel_pub
        self.arm_traj_pub = arm_traj_pub
        self.claw_cmd_pub = rospy.Publisher("/leju_claw_command", lejuClawCommand, queue_size=10)
        self.current_arm_rad = [0.0] * 14
        self.last_arm_deg = list(HOME_ARM_DEG)
        self.claw_state = None
        self._sensor_seen = False
        self._arm_mode: Optional[int] = None

        rospy.Subscriber("/sensors_data_raw", sensorsData, self._sensor_cb, queue_size=1)
        rospy.Subscriber("/leju_claw_state", lejuClawState, self._claw_state_cb, queue_size=1)

    def _sensor_cb(self, msg: Any) -> None:
        joint_q = list(getattr(getattr(msg, "joint_data", None), "joint_q", []))
        if len(joint_q) >= 26:
            self.current_arm_rad = [float(v) for v in joint_q[12:26]]
            self._sensor_seen = True

    def _claw_state_cb(self, msg: Any) -> None:
        self.claw_state = msg

    def wait_for_joint_state(self, timeout_sec: float = 4.0) -> bool:
        start = time.time()
        rate = self.rospy.Rate(20)
        while not self.rospy.is_shutdown() and time.time() - start < timeout_sec:
            if self._sensor_seen:
                return True
            self.rospy.loginfo_throttle(1.0, "[%s][初始化] 等待 /sensors_data_raw", self.scene)
            rate.sleep()
        log_warn(self.scene, "初始化", "未等到手臂关节状态，后续使用上一条关节目标作为 IK 初值")
        return False

    def current_arm_deg(self) -> List[float]:
        if not self._sensor_seen:
            return list(self.last_arm_deg)
        return [math.degrees(float(v)) for v in self.current_arm_rad]

    def prepare(self) -> None:
        log_stage(self.scene, "初始化", "等待传感器、打开夹爪、准备任务姿态")
        self.wait_for_joint_state(timeout_sec=3.0)
        if self.scene == "scene3":
            self.set_arm_mode(ARM_MODE_AUTO_SWING, warn_only=True)
            self.open_claw("both")
            log_stage(self.scene, "初始化", "场景三导航阶段保持自动摆臂，抓取前再切换外部控制")
            return
        self.set_arm_mode(ARM_MODE_EXTERNAL_CONTROL)
        self.open_claw("both")
        self.move_arm_deg(SIDE_LIFT_ARM_DEG, duration=1.3)
        self.move_arm_deg(WORK_ARM_DEG, duration=1.5)

    def finish(self) -> None:
        log_stage(self.scene, "结束", "停止底盘并回到安全手臂姿态")
        self.stop_base(duration=0.4)
        self.move_arm_deg(SIDE_LIFT_ARM_DEG, duration=1.0)
        self.move_arm_deg(HOME_ARM_DEG, duration=1.2)
        self.open_claw("both")
        self.set_arm_mode(ARM_MODE_AUTO_SWING, warn_only=True)

    def set_arm_mode(self, mode: int, warn_only: bool = True) -> bool:
        from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

        if self._arm_mode == int(mode):
            return True
        service_names = ["/arm_traj_change_mode", "/humanoid_change_arm_ctrl_mode"]
        for service_name in service_names:
            try:
                self.rospy.wait_for_service(service_name, timeout=1.5)
                proxy = self.rospy.ServiceProxy(service_name, changeArmCtrlMode)
                req = changeArmCtrlModeRequest()
                req.control_mode = int(mode)
                res = proxy(req)
                if hasattr(res, "result") and not res.result:
                    raise RuntimeError(getattr(res, "message", "service returned false"))
                self._arm_mode = int(mode)
                log_stage(self.scene, "手臂模式", "%s -> mode=%d", service_name, mode)
                return True
            except Exception as exc:
                last_exc = exc
                continue
        if warn_only:
            log_warn(self.scene, "手臂模式", "切换模式失败，继续尝试直接发轨迹: %s", last_exc)
            return False
        raise RuntimeError("set_arm_mode failed: {}".format(last_exc))

    def move_arm_deg(self, target_deg: Sequence[float], duration: float = 1.0) -> None:
        from sensor_msgs.msg import JointState

        self.set_arm_mode(ARM_MODE_EXTERNAL_CONTROL)
        target = [float(v) for v in target_deg]
        if len(target) != 14:
            raise ValueError("arm target expects 14 joints, got {}".format(len(target)))

        start = self.current_arm_deg()
        if len(start) != 14:
            start = list(self.last_arm_deg)
        steps = max(2, int(max(duration, 0.1) * ARM_TRAJ_HZ))
        rate = self.rospy.Rate(ARM_TRAJ_HZ)
        msg = JointState()
        msg.name = list(ARM_JOINT_NAMES)
        msg.velocity = [0.0] * 14
        msg.effort = [0.0] * 14
        for i in range(steps + 1):
            if self.rospy.is_shutdown():
                break
            alpha = smoothstep(float(i) / float(steps))
            msg.header.stamp = self.rospy.Time.now()
            msg.position = [start[j] + (target[j] - start[j]) * alpha for j in range(14)]
            self.arm_traj_pub.publish(msg)
            rate.sleep()
        self.last_arm_deg = target

    def send_single_arm_deg(self, arm: str, active_deg: Sequence[float],
                            duration: float = 1.0) -> None:
        target = list(self.current_arm_deg())
        if len(target) != 14:
            target = list(self.last_arm_deg)
        values = [float(v) for v in active_deg]
        if len(values) != 7:
            raise ValueError("single arm target expects 7 joints")
        if arm == "left":
            target[:7] = values
        elif arm == "right":
            target[7:] = values
        else:
            raise ValueError("unknown arm: {}".format(arm))
        self.move_arm_deg(target, duration=duration)

    def stop_base(self, duration: float = 0.3) -> None:
        from geometry_msgs.msg import Twist

        msg = Twist()
        end = self.rospy.Time.now() + self.rospy.Duration(float(duration))
        rate = self.rospy.Rate(20)
        while not self.rospy.is_shutdown() and self.rospy.Time.now() < end:
            self.cmd_vel_pub.publish(msg)
            rate.sleep()

    def drive_for(self, linear_x: float = 0.0, linear_y: float = 0.0,
                  angular_z: float = 0.0, duration: float = 1.0) -> None:
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import JointState

        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)
        arm_hold_msg = None
        if self._arm_mode == ARM_MODE_EXTERNAL_CONTROL:
            arm_hold_msg = JointState()
            arm_hold_msg.name = list(ARM_JOINT_NAMES)
            arm_hold_msg.velocity = [0.0] * 14
            arm_hold_msg.effort = [0.0] * 14
            arm_hold_msg.position = list(self.last_arm_deg)
        end = self.rospy.Time.now() + self.rospy.Duration(float(duration))
        rate = self.rospy.Rate(20)
        while not self.rospy.is_shutdown() and self.rospy.Time.now() < end:
            self.cmd_vel_pub.publish(msg)
            if arm_hold_msg is not None:
                arm_hold_msg.header.stamp = self.rospy.Time.now()
                self.arm_traj_pub.publish(arm_hold_msg)
            rate.sleep()
        self.stop_base(duration=0.25)

    def open_claw(self, arm: str) -> bool:
        return self.set_claw(arm, CLAW_OPEN_PERCENT, settle_sec=0.45)

    def close_claw(self, arm: str, percent: float = CLAW_CLOSE_PERCENT) -> bool:
        return self.set_claw(arm, percent, settle_sec=0.65)

    def set_claw(self, arm: str, percent: float, settle_sec: float = 0.2) -> bool:
        from kuavo_msgs.msg import endEffectorData, lejuClawCommand
        from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

        names, positions = self._claw_targets(arm, percent)
        data = endEffectorData()
        data.name = names
        data.position = positions
        data.velocity = [70.0] * len(names)
        data.effort = [1.5] * len(names)

        ok = False
        try:
            self.rospy.wait_for_service("/control_robot_leju_claw", timeout=1.5)
            req = controlLejuClawRequest()
            req.data = data
            res = self.rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)(req)
            ok = bool(getattr(res, "success", True))
            if not ok:
                log_warn(self.scene, "夹爪", "服务拒绝命令: %s", getattr(res, "message", ""))
        except Exception as exc:
            log_warn(self.scene, "夹爪", "服务调用失败，改用话题兜底: %s", exc)

        if not ok:
            msg = lejuClawCommand()
            msg.header.stamp = self.rospy.Time.now()
            msg.data = data
            self.claw_cmd_pub.publish(msg)
            ok = True

        self.rospy.sleep(float(settle_sec))
        return ok

    def _claw_targets(self, arm: str, percent: float) -> Tuple[List[str], List[float]]:
        percent = clamp(percent, 0.0, 100.0)
        if arm == "left":
            return ["left_claw"], [percent]
        if arm == "right":
            return ["right_claw"], [percent]
        if arm == "both":
            return ["left_claw", "right_claw"], [percent, percent]
        raise ValueError("unknown arm: {}".format(arm))

    def solve_ik(self, arm: str, target_xyz_base: Sequence[float],
                 quat_xyzw: Optional[Sequence[float]] = None,
                 constraint_mode: int = 2) -> List[float]:
        from kuavo_msgs.msg import ikSolveParam, twoArmHandPoseCmd
        from kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv

        current = list(self.current_arm_rad)
        if len(current) != 14 or not self._sensor_seen:
            current = [math.radians(v) for v in self.last_arm_deg]

        self.rospy.wait_for_service("/ik/fk_srv", timeout=2.0)
        fk = self.rospy.ServiceProxy("/ik/fk_srv", fkSrv)(current).hand_poses

        req = twoArmHandPoseCmd()
        req.use_custom_ik_param = True
        req.joint_angles_as_q0 = True
        req.frame = 0
        req.ik_param = ikSolveParam()
        req.ik_param.major_optimality_tol = 1e-3
        req.ik_param.major_feasibility_tol = 1e-3
        req.ik_param.minor_feasibility_tol = 1e-3
        req.ik_param.major_iterations_limit = 90
        req.ik_param.oritation_constraint_tol = 2e-2
        req.ik_param.pos_constraint_tol = 2e-3
        req.ik_param.pos_cost_weight = 1.0
        req.ik_param.constraint_mode = int(constraint_mode)

        req.hand_poses.left_pose.joint_angles = list(current[:7])
        req.hand_poses.right_pose.joint_angles = list(current[7:])
        req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.left_pose.pos_xyz = list(fk.left_pose.pos_xyz)
        req.hand_poses.right_pose.pos_xyz = list(fk.right_pose.pos_xyz)
        req.hand_poses.left_pose.quat_xyzw = list(fk.left_pose.quat_xyzw)
        req.hand_poses.right_pose.quat_xyzw = list(fk.right_pose.quat_xyzw)

        yaw = 0.0 if arm == "left" else math.pi
        quat = list(quat_xyzw) if quat_xyzw is not None else quat_down(yaw)
        if arm == "left":
            req.hand_poses.left_pose.pos_xyz = [float(v) for v in target_xyz_base]
            req.hand_poses.left_pose.quat_xyzw = quat
        elif arm == "right":
            req.hand_poses.right_pose.pos_xyz = [float(v) for v in target_xyz_base]
            req.hand_poses.right_pose.quat_xyzw = quat
        else:
            raise ValueError("unknown arm: {}".format(arm))

        self.rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=2.0)
        res = self.rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(req)
        if not getattr(res, "success", False):
            raise RuntimeError(getattr(res, "error_reason", "IK service returned false"))

        q_arm = list(getattr(res, "q_arm", []))
        if len(q_arm) < 14:
            q_arm = list(res.hand_poses.left_pose.joint_angles) + list(res.hand_poses.right_pose.joint_angles)
        if len(q_arm) < 14:
            raise RuntimeError("IK response did not contain 14 arm joints")
        return [math.degrees(float(v)) for v in q_arm[:14]]

    def move_hand_ik(self, arm: str, target_xyz_base: Sequence[float],
                     yaw_rad: float = 0.0, duration: float = 1.2,
                     label: str = "ik") -> bool:
        z_max = SCENE_SAFE_MAX_Z_BASE_M.get(self.scene, SAFE_MAX_Z_BASE_M)
        target = [
            clamp(float(target_xyz_base[0]), -0.05, REACH_X_MAX_M),
            float(target_xyz_base[1]),
            clamp(float(target_xyz_base[2]), SAFE_MIN_Z_BASE_M, z_max),
        ]
        quat = quat_down(yaw_rad if arm == "left" else yaw_rad + math.pi)
        try:
            q_deg = self.solve_ik(arm, target, quat_xyzw=quat, constraint_mode=2)
            self.move_arm_deg(q_deg, duration=duration)
            return True
        except Exception as exc:
            log_warn(self.scene, "IK", "%s 失败，使用关节空间兜底: %s target=%s", label, exc, _fmt_xyz(target))
            self.move_arm_deg(self._fallback_pose(arm, target), duration=duration)
            return False

    def _fallback_pose(self, arm: str, target_xyz_base: Sequence[float]) -> List[float]:
        pose = list(self.current_arm_deg())
        if len(pose) != 14:
            pose = list(self.last_arm_deg)
        x = clamp(float(target_xyz_base[0]), 0.18, 0.75)
        y = float(target_xyz_base[1])
        z = float(target_xyz_base[2])
        reach = clamp((0.38 - x) * 90.0 + (0.18 - z) * 70.0, -35.0, 35.0)
        side = clamp(y * 85.0, -35.0, 35.0)
        if arm == "left":
            pose[:7] = [30 + reach, 18 - side, 8, -95, -45, 0, 0]
        else:
            pose[7:] = [30 + reach, -18 - side, -8, -95, 45, 0, 0]
        return pose


# ---------------------------------------------------------------------------
# 感知层：规则检测、深度投影和 TF 变换
# ---------------------------------------------------------------------------


class RuleBasedPerception:
    def __init__(self, scene: str, base_frame: str = BASE_FRAME):
        import rospy
        import tf2_ros
        from sensor_msgs.msg import CameraInfo, CompressedImage

        self.rospy = rospy
        self.scene = scene
        self.base_frame = base_frame
        self.cv2 = None
        self.np = None
        self._cv_import_failed = False
        self._yolo_model = None
        self._yolo_import_failed = False
        self._yolo_model_path = None
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cameras: Dict[str, Dict[str, Any]] = {}

        for name, spec in CAMERA_SPECS.items():
            state = {"color_msg": None, "depth_msg": None, "info_msg": None}
            self.cameras[name] = state
            rospy.Subscriber(spec["color"], CompressedImage,
                             self._make_store_cb(name, "color_msg"), queue_size=1)
            rospy.Subscriber(spec["depth"], CompressedImage,
                             self._make_store_cb(name, "depth_msg"), queue_size=1)
            rospy.Subscriber(spec["info"], CameraInfo,
                             self._make_store_cb(name, "info_msg"), queue_size=1)

    def _make_store_cb(self, camera_name: str, key: str):
        def cb(msg: Any) -> None:
            self.cameras[camera_name][key] = msg
        return cb

    def _ensure_cv(self) -> bool:
        if self.cv2 is not None and self.np is not None:
            return True
        if self._cv_import_failed:
            return False
        try:
            import cv2
            import numpy as np

            self.cv2 = cv2
            self.np = np
            return True
        except Exception as exc:
            self._cv_import_failed = True
            log_error(self.scene, "感知", "无法导入 cv2/numpy，规则识别不可用: %s", exc)
            return False

    def wait_ready(self, timeout_sec: float = TARGET_WAIT_SEC) -> bool:
        start = time.time()
        rate = self.rospy.Rate(20)
        while not self.rospy.is_shutdown() and time.time() - start < timeout_sec:
            ready = []
            for name, state in self.cameras.items():
                if (state["color_msg"] is not None
                        and state["depth_msg"] is not None
                        and state["info_msg"] is not None):
                    ready.append(name)
            if ready:
                log_stage(self.scene, "等待传感器", "可用相机=%s", ",".join(ready))
                return True
            self.rospy.loginfo_throttle(1.0, "[%s][等待传感器] 等待相机图像、深度与 CameraInfo", self.scene)
            rate.sleep()
        log_warn(self.scene, "等待传感器", "相机数据超时")
        return False

    def detect_targets(self, class_names: Optional[Sequence[str]] = None,
                       cameras: Optional[Sequence[str]] = None,
                       max_targets: Optional[int] = None) -> List[PerceptionTarget]:
        if self.scene == "scene2":
            yolo_targets = self._detect_scene2_with_yolo(class_names=class_names, max_targets=max_targets)
            if yolo_targets:
                return yolo_targets

        rules = list(TARGET_RULES.get(self.scene, []))
        if class_names is not None:
            wanted = set(class_names)
            rules = [rule for rule in rules if rule["class_name"] in wanted]
        return self._detect_with_rules(
            rules=rules,
            cameras=cameras or self._default_cameras_for_targets(),
            max_targets=max_targets,
            purpose="检测",
        )

    def detect_bins(self, bin_names: Optional[Sequence[str]] = None) -> List[PerceptionTarget]:
        rules = list(BIN_RULES.get(self.scene, []))
        if bin_names is not None:
            wanted = set(bin_names)
            rules = [rule for rule in rules if rule["class_name"] in wanted]
        return self._detect_with_rules(
            rules=rules,
            cameras=["head"],
            max_targets=None,
            purpose="目标区",
        )

    def _default_cameras_for_targets(self) -> List[str]:
        if self.scene == "scene3":
            return ["head", "left_wrist", "right_wrist"]
        return ["head", "left_wrist", "right_wrist"]

    def _find_yolo_model_path(self) -> Optional[str]:
        if self._yolo_model_path:
            return self._yolo_model_path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_dir = os.path.abspath(os.path.join(script_dir, ".."))
        repo_root = os.path.abspath(os.path.join(pkg_dir, "..", ".."))
        bases = [pkg_dir, repo_root, os.getcwd()]
        for pattern in SCENE2_YOLO_MODEL_PATTERNS:
            for base in bases:
                matches = glob.glob(os.path.join(base, pattern), recursive=True)
                for path in sorted(matches):
                    if os.path.isfile(path):
                        self._yolo_model_path = path
                        return path
        return None

    def _ensure_yolo(self) -> bool:
        if self._yolo_model is not None:
            return True
        if self._yolo_import_failed:
            return False
        model_path = self._find_yolo_model_path()
        if not model_path:
            self._yolo_import_failed = True
            log_warn(self.scene, "YOLO", "未找到 Scene2 YOLO 权重，使用 HSV 规则检测兜底")
            return False
        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(model_path)
            log_stage(self.scene, "YOLO", "加载模型 %s", model_path)
            return True
        except Exception as exc:
            self._yolo_import_failed = True
            log_warn(self.scene, "YOLO", "加载失败，使用 HSV 规则检测兜底: %s", exc)
            return False

    def _detect_scene2_with_yolo(self, class_names: Optional[Sequence[str]],
                                 max_targets: Optional[int]) -> List[PerceptionTarget]:
        if not self._ensure_cv() or not self._ensure_yolo():
            return []
        state = self.cameras.get("head")
        if not state:
            return []
        color_msg = state.get("color_msg")
        info_msg = state.get("info_msg")
        if color_msg is None or info_msg is None:
            return []
        color_bgr = self._decode_color(color_msg)
        if color_bgr is None:
            return []
        depth_m = self._decode_depth(state.get("depth_msg"))
        wanted = set(class_names or ["part_type_a", "part_type_b", "part_type_c"])
        try:
            results = self._yolo_model.predict(color_bgr, verbose=False, conf=0.35, imgsz=640)
        except Exception as exc:
            log_warn(self.scene, "YOLO", "推理失败，使用 HSV 规则检测兜底: %s", exc)
            return []

        targets: List[PerceptionTarget] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            if boxes is None:
                continue
            box_data = getattr(boxes, "data", None)
            if box_data is None:
                continue
            try:
                box_rows = box_data.detach().cpu().numpy()
            except Exception:
                box_rows = box_data.cpu().numpy()
            mask_rows = []
            if masks is not None and getattr(masks, "data", None) is not None:
                try:
                    mask_rows = masks.data.detach().cpu().numpy()
                except Exception:
                    mask_rows = masks.data.cpu().numpy()
            for index, row in enumerate(box_rows):
                if len(row) < 6:
                    continue
                cls_idx = int(row[5])
                raw_name = str(names.get(cls_idx, cls_idx))
                class_name = SCENE2_YOLO_CLASS_ALIASES.get(raw_name, raw_name)
                if class_name not in wanted:
                    continue
                conf = float(row[4])
                x0, y0, x1, y1 = [int(round(float(v))) for v in row[:4]]
                x0 = int(clamp(x0, 0, color_bgr.shape[1] - 1))
                y0 = int(clamp(y0, 0, color_bgr.shape[0] - 1))
                x1 = int(clamp(x1, x0 + 1, color_bgr.shape[1]))
                y1 = int(clamp(y1, y0 + 1, color_bgr.shape[0]))
                mask = self._mask_from_yolo(mask_rows[index], color_bgr.shape[:2]) if index < len(mask_rows) else None
                target = self._yolo_detection_to_target(
                    color_msg=color_msg,
                    info_msg=info_msg,
                    depth_m=depth_m,
                    mask=mask,
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    class_name=class_name,
                    confidence=conf,
                )
                if target is not None:
                    targets.append(target)

        targets = self._dedupe_targets(targets)
        targets.sort(key=lambda t: t.confidence, reverse=True)
        if max_targets is not None:
            targets = targets[:int(max_targets)]
        if targets:
            log_stage(
                self.scene,
                "YOLO",
                "检测结果 %s",
                " ".join("{}={}".format(name, sum(1 for t in targets if t.class_name == name))
                         for name in ["part_type_a", "part_type_b", "part_type_c"]),
            )
        return targets

    def _mask_from_yolo(self, mask_data: Any, shape_hw: Tuple[int, int]) -> Any:
        mask = (mask_data > 0.5).astype(self.np.uint8) * 255
        if mask.shape[:2] != shape_hw:
            mask = self.cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=self.cv2.INTER_NEAREST)
        return mask

    def _yolo_detection_to_target(self, color_msg: Any, info_msg: Any, depth_m: Optional[Any],
                                  mask: Optional[Any], bbox: Tuple[int, int, int, int],
                                  class_name: str, confidence: float) -> Optional[PerceptionTarget]:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        if mask is None:
            mask = self.np.zeros((max(y + h + 1, 1), max(x + w + 1, 1)), dtype=self.np.uint8)
            mask[y:y + h, x:x + w] = 255
        moments = self.cv2.moments(mask[y:y + h, x:x + w])
        if abs(moments["m00"]) > 1e-6:
            u = int(x + moments["m10"] / moments["m00"])
            v = int(y + moments["m01"] / moments["m00"])
        else:
            u = int(x + w * 0.5)
            v = int(y + h * 0.5)
        depth = self._median_depth_for_contour(depth_m, mask, x, y, w, h)
        pos_camera = None
        pos_base = None
        if depth is not None:
            pos_camera = self._pixel_to_camera(u, v, depth, info_msg)
            pos_base = self._transform_camera_to_base(pos_camera, color_msg.header.frame_id, color_msg.header.stamp)
        contours = self._find_contours(mask[y:y + h, x:x + w])
        yaw_rad = 0.0
        if contours:
            yaw_rad = self._contour_yaw(max(contours, key=self.cv2.contourArea))
        area = float(self.cv2.countNonZero(mask[y:y + h, x:x + w]))
        return PerceptionTarget(
            class_name=class_name,
            confidence=float(confidence),
            center_uv=(u, v),
            position_base=pos_base,
            position_camera=pos_camera,
            camera="head",
            area_px=area,
            bbox=(int(x), int(y), int(w), int(h)),
            yaw_rad=yaw_rad,
            target_bin=Scene2Controller.PART_TO_BIN.get(class_name),
            timestamp=time.time(),
            debug={
                "source": "yolov8_seg",
                "depth_m": depth,
                "image_size": (int(mask.shape[1]), int(mask.shape[0])),
                "frame_id": color_msg.header.frame_id,
            },
        )

    def _detect_with_rules(self, rules: Sequence[Dict[str, Any]], cameras: Sequence[str],
                           max_targets: Optional[int], purpose: str) -> List[PerceptionTarget]:
        if not rules or not self._ensure_cv():
            return []

        targets: List[PerceptionTarget] = []
        for camera in cameras:
            state = self.cameras.get(camera)
            if not state:
                continue
            color_msg = state.get("color_msg")
            info_msg = state.get("info_msg")
            if color_msg is None or info_msg is None:
                self.rospy.loginfo_throttle(
                    1.0,
                    "[%s][%s] %s 未就绪 color=%s info=%s",
                    self.scene,
                    purpose,
                    camera,
                    color_msg is not None,
                    info_msg is not None,
                )
                continue

            color = self._decode_color(color_msg)
            if color is None:
                continue
            depth = self._decode_depth(state.get("depth_msg"))
            if depth is None:
                self.rospy.loginfo_throttle(
                    1.0,
                    "[%s][%s] %s 深度图未就绪或解码失败，候选将缺少 base_link 坐标",
                    self.scene,
                    purpose,
                    camera,
                )
            targets.extend(self._detect_in_image(camera, color_msg, info_msg, color, depth, rules, purpose))

        targets = self._dedupe_targets(targets)
        targets.sort(key=lambda t: t.confidence, reverse=True)
        if max_targets is not None:
            targets = targets[:int(max_targets)]
        debug_summary = " ".join(
            "{}={}".format(rule["class_name"], sum(1 for t in targets if t.class_name == rule["class_name"]))
            for rule in rules
        )
        log_stage(self.scene, purpose, "规则结果 %s", debug_summary or "empty")
        return targets

    def _decode_color(self, msg: Any) -> Optional[Any]:
        try:
            data = self.np.frombuffer(msg.data, dtype=self.np.uint8)
            return self.cv2.imdecode(data, self.cv2.IMREAD_COLOR)
        except Exception as exc:
            log_warn(self.scene, "感知", "RGB 解码失败: %s", exc)
            return None

    def _decode_depth(self, msg: Any) -> Optional[Any]:
        if msg is None:
            return None
        try:
            format_text = str(getattr(msg, "format", "") or "")
            raw_bytes = bytes(msg.data)
            raw = self.np.frombuffer(raw_bytes, dtype=self.np.uint8)
            header = None
            if len(raw_bytes) >= 12:
                try:
                    header = struct.unpack("iff", raw_bytes[:12])
                except struct.error:
                    header = None
            offsets = [12, 16, 0]
            for offset in offsets:
                if len(raw) <= offset:
                    continue
                decoded = self.cv2.imdecode(raw[offset:], self.cv2.IMREAD_UNCHANGED)
                if decoded is None:
                    continue
                depth = decoded.astype(self.np.float32)
                if "16UC1" in format_text or decoded.dtype == self.np.uint16:
                    depth *= 0.001
                elif (offset in (12, 16)
                        and header is not None
                        and int(header[0]) == 0
                        and abs(float(header[1])) > 1e-6):
                    depth_quant_a = float(header[1])
                    depth_quant_b = float(header[2])
                    valid = depth > 0.0
                    converted = self.np.zeros_like(depth, dtype=self.np.float32)
                    converted[valid] = depth_quant_a / (depth[valid] - depth_quant_b)
                    depth = converted
                elif self.np.nanmax(depth) > 20.0:
                    depth *= 0.001
                return depth
        except Exception as exc:
            log_warn(self.scene, "感知", "深度解码失败: %s", exc)
        return None

    def _detect_in_image(self, camera: str, color_msg: Any, info_msg: Any,
                         color_bgr: Any, depth_m: Optional[Any],
                         rules: Sequence[Dict[str, Any]], purpose: str) -> List[PerceptionTarget]:
        hsv = self.cv2.cvtColor(color_bgr, self.cv2.COLOR_BGR2HSV)
        height, width = color_bgr.shape[:2]
        found: List[PerceptionTarget] = []

        for rule in rules:
            mask = self._mask_for_rule(hsv, rule)
            mask = self._apply_workspace_roi(mask, purpose)
            contours = self._find_contours(mask)
            raw_count = len(contours)
            kept_count = 0
            for contour in contours:
                candidate = self._contour_to_target(
                    camera=camera,
                    color_msg=color_msg,
                    info_msg=info_msg,
                    depth_m=depth_m,
                    mask=mask,
                    contour=contour,
                    rule=rule,
                    image_width=width,
                    image_height=height,
                )
                if candidate is not None:
                    kept_count += 1
                    found.append(candidate)
            self.rospy.loginfo_throttle(
                1.0,
                "[%s][%s] cam=%s class=%s raw_contours=%d kept=%d",
                self.scene,
                purpose,
                camera,
                rule["class_name"],
                raw_count,
                kept_count,
            )
        return found

    def _mask_for_rule(self, hsv: Any, rule: Dict[str, Any]) -> Any:
        mask = self.np.zeros(hsv.shape[:2], dtype=self.np.uint8)
        for lower, upper in rule.get("hsv_ranges", []):
            lo = self.np.array(lower, dtype=self.np.uint8)
            hi = self.np.array(upper, dtype=self.np.uint8)
            mask = self.cv2.bitwise_or(mask, self.cv2.inRange(hsv, lo, hi))
        kernel = self.np.ones((5, 5), dtype=self.np.uint8)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)
        return mask

    def _apply_workspace_roi(self, mask: Any, purpose: str) -> Any:
        h, w = mask.shape[:2]
        roi = self.np.zeros_like(mask)
        top = int(h * 0.12)
        bottom = int(h * 0.96)
        left = int(w * 0.08)
        right = int(w * 0.92)
        roi[top:bottom, left:right] = 255
        return self.cv2.bitwise_and(mask, roi)

    def _find_contours(self, mask: Any) -> List[Any]:
        result = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        contours = result[0] if len(result) == 2 else result[1]
        return list(contours)

    def _contour_to_target(self, camera: str, color_msg: Any, info_msg: Any,
                           depth_m: Optional[Any], mask: Any, contour: Any,
                           rule: Dict[str, Any], image_width: int,
                           image_height: int) -> Optional[PerceptionTarget]:
        area = float(self.cv2.contourArea(contour))
        if area < float(rule.get("min_area_px", 0)) or area > float(rule.get("max_area_px", 1e9)):
            return None
        x, y, w, h = self.cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None
        aspect = float(w) / float(h)
        if aspect < float(rule.get("min_aspect", 0.0)) or aspect > float(rule.get("max_aspect", 999.0)):
            return None
        extent = area / float(max(1, w * h))
        if extent < float(rule.get("min_extent", 0.0)):
            return None

        moments = self.cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            u = int(moments["m10"] / moments["m00"])
            v = int(moments["m01"] / moments["m00"])
        else:
            u = int(x + w * 0.5)
            v = int(y + h * 0.5)

        depth = self._median_depth_for_contour(depth_m, mask, x, y, w, h)
        pos_camera = None
        pos_base = None
        if depth is not None:
            pos_camera = self._pixel_to_camera(u, v, depth, info_msg)
            pos_base = self._transform_camera_to_base(pos_camera, color_msg.header.frame_id, color_msg.header.stamp)

        yaw_rad = self._contour_yaw(contour)
        confidence = self._confidence(area, image_width, image_height, depth, extent)
        return PerceptionTarget(
            class_name=rule["class_name"],
            confidence=confidence,
            center_uv=(u, v),
            position_base=pos_base,
            position_camera=pos_camera,
            camera=camera,
            area_px=area,
            bbox=(int(x), int(y), int(w), int(h)),
            yaw_rad=yaw_rad,
            target_bin=rule.get("target_bin"),
            timestamp=time.time(),
            debug={
                "aspect": aspect,
                "extent": extent,
                "depth_m": depth,
                "image_size": (image_width, image_height),
                "frame_id": color_msg.header.frame_id,
            },
        )

    def _median_depth_for_contour(self, depth_m: Optional[Any], mask: Any,
                                  x: int, y: int, w: int, h: int) -> Optional[float]:
        if depth_m is None:
            return None
        dh, dw = depth_m.shape[:2]
        if dh <= 0 or dw <= 0:
            return None
        sx = float(dw) / float(mask.shape[1])
        sy = float(dh) / float(mask.shape[0])
        dx0 = clamp(int(x * sx), 0, dw - 1)
        dy0 = clamp(int(y * sy), 0, dh - 1)
        dx1 = clamp(int((x + w) * sx), 0, dw)
        dy1 = clamp(int((y + h) * sy), 0, dh)
        if dx1 <= dx0 or dy1 <= dy0:
            return None

        roi_depth = depth_m[int(dy0):int(dy1), int(dx0):int(dx1)]
        roi_mask = mask[y:y + h, x:x + w]
        if roi_mask.shape[:2] != roi_depth.shape[:2]:
            roi_mask = self.cv2.resize(roi_mask, (roi_depth.shape[1], roi_depth.shape[0]),
                                       interpolation=self.cv2.INTER_NEAREST)
        valid = roi_depth[(roi_mask > 0) & self.np.isfinite(roi_depth) & (roi_depth > 0.05) & (roi_depth < 5.0)]
        if valid.size < 5:
            valid = roi_depth[self.np.isfinite(roi_depth) & (roi_depth > 0.05) & (roi_depth < 5.0)]
        if valid.size < 5:
            return None
        return float(self.np.median(valid))

    def _pixel_to_camera(self, u: int, v: int, depth: float, info_msg: Any) -> Tuple[float, float, float]:
        k = list(info_msg.K)
        fx = float(k[0]) if abs(float(k[0])) > 1e-6 else 1.0
        fy = float(k[4]) if abs(float(k[4])) > 1e-6 else fx
        cx = float(k[2])
        cy = float(k[5])
        z = float(depth)
        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        return (x, y, z)

    def _transform_camera_to_base(self, point_camera: Tuple[float, float, float],
                                  source_frame: str, stamp: Any) -> Optional[Tuple[float, float, float]]:
        if not source_frame:
            return None
        try:
            from geometry_msgs.msg import PointStamped
            import tf2_geometry_msgs  # noqa: F401

            ps = PointStamped()
            ps.header.frame_id = source_frame
            ps.header.stamp = stamp
            ps.point.x, ps.point.y, ps.point.z = point_camera
            out = self.tf_buffer.transform(ps, self.base_frame, self.rospy.Duration(0.15))
            return (float(out.point.x), float(out.point.y), float(out.point.z))
        except Exception as first_exc:
            try:
                import tf.transformations as tft

                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    source_frame,
                    self.rospy.Time(0),
                    self.rospy.Duration(0.15),
                )
                tr = transform.transform.translation
                q = transform.transform.rotation
                mat = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
                mat[0, 3] = tr.x
                mat[1, 3] = tr.y
                mat[2, 3] = tr.z
                point = self.np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
                out = mat.dot(point)
                return (float(out[0]), float(out[1]), float(out[2]))
            except Exception as exc:
                last_exc = exc
            self.rospy.logwarn_throttle(
                1.0,
                "[%s][TF] %s -> %s 失败: %s / %s",
                self.scene,
                source_frame,
                self.base_frame,
                first_exc,
                last_exc,
            )
            return None

    def _contour_yaw(self, contour: Any) -> float:
        rect = self.cv2.minAreaRect(contour)
        (w, h) = rect[1]
        angle = float(rect[2])
        if w < h:
            angle += 90.0
        return math.radians(angle)

    def _confidence(self, area: float, width: int, height: int,
                    depth: Optional[float], extent: float) -> float:
        area_score = clamp(area / float(width * height) * 80.0, 0.0, 0.55)
        depth_score = 0.30 if depth is not None else 0.10
        extent_score = clamp(extent, 0.0, 0.25)
        return clamp(area_score + depth_score + extent_score, 0.05, 0.99)

    def _dedupe_targets(self, targets: Sequence[PerceptionTarget]) -> List[PerceptionTarget]:
        kept: List[PerceptionTarget] = []
        for target in sorted(targets, key=lambda t: t.confidence, reverse=True):
            duplicate = False
            for old in kept:
                if target.class_name != old.class_name:
                    continue
                if target.position_base is not None and old.position_base is not None:
                    if distance3(target.position_base, old.position_base) < 0.055:
                        duplicate = True
                        break
                elif target.camera == old.camera and target.center_uv and old.center_uv:
                    du = target.center_uv[0] - old.center_uv[0]
                    dv = target.center_uv[1] - old.center_uv[1]
                    if math.sqrt(du * du + dv * dv) < 45.0:
                        duplicate = True
                        break
            if not duplicate:
                kept.append(target)
        return kept


class YoloScene1Perception(RuleBasedPerception):
    def __init__(self, scene: str, base_frame: str = BASE_FRAME,
                 model_path: str = DEFAULT_SCENE1_YOLO_MODEL,
                 device: str = "auto",
                 conf: float = SCENE1_YOLO_DEFAULT_CONF,
                 imgsz: int = SCENE1_YOLO_DEFAULT_IMGSZ):
        super().__init__(scene=scene, base_frame=base_frame)
        self.yolo_model_path = os.path.abspath(os.path.expanduser(model_path))
        self.yolo_device = device
        self.yolo_conf = float(conf)
        self.yolo_imgsz = int(imgsz)
        self.yolo_class_name = SCENE1_YOLO_CLASS_NAME
        self.yolo_target_bin = SCENE1_YOLO_TARGET_BIN
        self.yolo_class_names = tuple(SCENE1_YOLO_CLASS_NAMES)
        self.yolo_class_aliases = dict(SCENE1_YOLO_CLASS_ALIASES)
        self.yolo_target_bins = dict(SCENE1_YOLO_TARGET_BINS)
        self.yolo_center_depth_fraction = SCENE1_YOLO_CENTER_DEPTH_FRACTION
        self._yolo_model = None
        self._yolo_unavailable_reason: Optional[str] = None
        self._yolo_unavailable_reported = False
        self._yolo_ready_logged = False

    def detect_targets(self, class_names: Optional[Sequence[str]] = None,
                       cameras: Optional[Sequence[str]] = None,
                       max_targets: Optional[int] = None) -> List[PerceptionTarget]:
        wanted = self._wanted_yolo_classes(class_names, allowed=self.yolo_class_names)
        return self._detect_yolo_classes(
            class_names=wanted,
            cameras=cameras or self._default_cameras_for_targets(),
            max_targets=max_targets,
            purpose="检测",
        )

    def detect_bins(self, bin_names: Optional[Sequence[str]] = None) -> List[PerceptionTarget]:
        wanted = self._wanted_yolo_classes(bin_names, allowed=("weighing_area", "drop_box"))
        return self._detect_yolo_classes(
            class_names=wanted,
            cameras=["head"],
            max_targets=None,
            purpose="目标区",
        )

    def _detect_yolo_classes(self, class_names: Sequence[str], cameras: Sequence[str],
                             max_targets: Optional[int], purpose: str) -> List[PerceptionTarget]:
        if not class_names:
            return []
        if not self._ensure_yolo():
            reason = self._yolo_unavailable_reason or "unknown"
            self._report_yolo_unavailable("error", "YOLO 不可用: %s", reason)
            return []

        try:
            return self._detect_with_yolo(
                cameras=cameras,
                max_targets=max_targets,
                class_names=class_names,
                purpose=purpose,
            )
        except Exception as exc:
            log_error(self.scene, "YOLO", "推理失败: %s", exc)
            return []

    def _ensure_yolo(self) -> bool:
        if self._yolo_model is not None:
            return True
        if self._yolo_unavailable_reason is not None:
            return False
        if not self._ensure_cv():
            self._yolo_unavailable_reason = "cv2/numpy import failed"
            return False
        if not os.path.isfile(self.yolo_model_path):
            self._yolo_unavailable_reason = "model not found: {}".format(self.yolo_model_path)
            return False

        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(self.yolo_model_path)
            if not self._yolo_ready_logged:
                log_stage(
                    self.scene,
                    "YOLO",
                    "已加载模型 %s device=%s conf=%.2f imgsz=%d",
                    self.yolo_model_path,
                    self.yolo_device,
                    self.yolo_conf,
                    self.yolo_imgsz,
                )
                self._yolo_ready_logged = True
            return True
        except Exception as exc:
            self._yolo_unavailable_reason = "ultralytics/model load failed: {}".format(exc)
            self._yolo_model = None
            return False

    def _report_yolo_unavailable(self, level: str, message: str, *args: Any) -> None:
        if self._yolo_unavailable_reported:
            return
        self._yolo_unavailable_reported = True
        if level == "error":
            log_error(self.scene, "YOLO", message, *args)
        else:
            log_warn(self.scene, "YOLO", message, *args)

    def _detect_with_yolo(self, cameras: Sequence[str], max_targets: Optional[int],
                          class_names: Sequence[str], purpose: str) -> List[PerceptionTarget]:
        if not self._ensure_cv():
            return []

        targets: List[PerceptionTarget] = []
        for camera in cameras:
            state = self.cameras.get(camera)
            if not state:
                continue
            color_msg = state.get("color_msg")
            info_msg = state.get("info_msg")
            if color_msg is None or info_msg is None:
                self.rospy.loginfo_throttle(
                    1.0,
                    "[%s][YOLO] %s 未就绪 color=%s info=%s",
                    self.scene,
                    camera,
                    color_msg is not None,
                    info_msg is not None,
                )
                continue

            color = self._decode_color(color_msg)
            if color is None:
                continue
            depth = self._decode_depth(state.get("depth_msg"))
            if depth is None:
                self.rospy.loginfo_throttle(
                    1.0,
                    "[%s][YOLO] %s 深度图未就绪或解码失败，候选将缺少 base_link 坐标",
                    self.scene,
                    camera,
                )

            targets.extend(self._detect_yolo_in_image(
                camera,
                color_msg,
                info_msg,
                color,
                depth,
                class_names=class_names,
            ))

        targets = self._dedupe_targets(targets)
        targets.sort(key=lambda t: t.confidence, reverse=True)
        if max_targets is not None:
            targets = targets[:int(max_targets)]
        debug_summary = " ".join(
            "{}={}".format(name, sum(1 for target in targets if target.class_name == name))
            for name in class_names
        )
        log_stage(self.scene, purpose, "YOLO结果 %s", debug_summary or "empty")
        return targets

    def _detect_yolo_in_image(self, camera: str, color_msg: Any, info_msg: Any,
                              color_bgr: Any, depth_m: Optional[Any],
                              class_names: Sequence[str]) -> List[PerceptionTarget]:
        height, width = color_bgr.shape[:2]
        kwargs = {
            "source": color_bgr,
            "conf": self.yolo_conf,
            "imgsz": self.yolo_imgsz,
            "verbose": False,
        }
        if self.yolo_device != "auto":
            kwargs["device"] = self.yolo_device

        results = self._yolo_model.predict(**kwargs)
        found: List[PerceptionTarget] = []
        raw_count = 0
        kept_count = 0
        for result in results:
            names = self._normalized_yolo_names(
                getattr(result, "names", None) or getattr(self._yolo_model, "names", None)
            )
            boxes = getattr(result, "boxes", None)
            box_list = list(boxes) if boxes is not None else []
            mask_rows = self._yolo_mask_rows(getattr(result, "masks", None), color_bgr.shape[:2])
            raw_count += len(box_list)
            for index, box in enumerate(box_list):
                mask = mask_rows[index] if index < len(mask_rows) else None
                candidate = self._yolo_box_to_target(
                    camera=camera,
                    color_msg=color_msg,
                    info_msg=info_msg,
                    depth_m=depth_m,
                    box=box,
                    mask=mask,
                    names=names,
                    class_names=class_names,
                    image_width=width,
                    image_height=height,
                )
                if candidate is not None:
                    kept_count += 1
                    found.append(candidate)

        self.rospy.loginfo_throttle(
            1.0,
            "[%s][YOLO] cam=%s raw_boxes=%d kept=%d",
            self.scene,
            camera,
            raw_count,
            kept_count,
        )
        return found

    def _yolo_box_to_target(self, camera: str, color_msg: Any, info_msg: Any,
                            depth_m: Optional[Any], box: Any,
                            mask: Optional[Any],
                            names: Dict[int, str], class_names: Sequence[str], image_width: int,
                            image_height: int) -> Optional[PerceptionTarget]:
        xyxy = self._as_float_list(getattr(box, "xyxy", []))
        if len(xyxy) < 4:
            return None
        x0 = int(clamp(min(xyxy[0], xyxy[2]), 0, image_width - 1))
        y0 = int(clamp(min(xyxy[1], xyxy[3]), 0, image_height - 1))
        x1 = int(clamp(max(xyxy[0], xyxy[2]), 0, image_width))
        y1 = int(clamp(max(xyxy[1], xyxy[3]), 0, image_height))
        if x1 <= x0 or y1 <= y0:
            return None

        class_id = int(round(self._as_float(getattr(box, "cls", 0.0))))
        class_name = self._target_class_name_from_yolo(names, class_id, class_names)
        if class_name is None:
            return None

        w = x1 - x0
        h = y1 - y0
        center_uv = self._yolo_target_center(mask, x0, y0, w, h)
        u, v = center_uv
        if mask is not None:
            depth = self._median_depth_for_contour(depth_m, mask, x0, y0, w, h)
        else:
            depth = self._median_depth_for_bbox_center(depth_m, image_width, image_height, x0, y0, w, h)
        pos_camera = None
        pos_base = None
        if depth is not None:
            pos_camera = self._pixel_to_camera(u, v, depth, info_msg)
            pos_base = self._transform_camera_to_base(pos_camera, color_msg.header.frame_id, color_msg.header.stamp)

        confidence = clamp(self._as_float(getattr(box, "conf", 0.0)), 0.0, 0.99)
        raw_name = names.get(class_id, str(class_id))
        area_px = float(w * h)
        yaw_rad = None
        source = "yolo"
        if mask is not None:
            source = "yolo_seg"
            roi_mask = mask[y0:y1, x0:x1]
            area_px = float(self.cv2.countNonZero(roi_mask))
            contours = self._find_contours(roi_mask)
            if contours:
                yaw_rad = self._contour_yaw(max(contours, key=self.cv2.contourArea))
        return PerceptionTarget(
            class_name=class_name,
            confidence=confidence,
            center_uv=(u, v),
            position_base=pos_base,
            position_camera=pos_camera,
            camera=camera,
            area_px=area_px,
            bbox=(x0, y0, w, h),
            yaw_rad=yaw_rad,
            target_bin=self.yolo_target_bins.get(class_name),
            timestamp=time.time(),
            debug={
                "source": source,
                "raw_class": raw_name,
                "class_id": class_id,
                "depth_m": depth,
                "image_size": (image_width, image_height),
                "frame_id": color_msg.header.frame_id,
            },
        )

    def _wanted_yolo_classes(self, class_names: Optional[Sequence[str]],
                             allowed: Sequence[str]) -> List[str]:
        allowed_set = set(allowed)
        if class_names is None:
            return [name for name in self.yolo_class_names if name in allowed_set]
        wanted = []
        for name in class_names:
            normalized = self._normalize_yolo_name(name)
            canonical = self.yolo_class_aliases.get(normalized, normalized)
            if canonical in allowed_set and canonical not in wanted:
                wanted.append(canonical)
        return wanted

    def _target_class_name_from_yolo(self, names: Dict[int, str],
                                     class_id: int,
                                     class_names: Sequence[str]) -> Optional[str]:
        wanted = set(class_names)
        aliases = {
            self._normalize_yolo_name(alias): canonical
            for alias, canonical in self.yolo_class_aliases.items()
        }
        target_class_ids: Dict[int, str] = {}
        for item_id, name in names.items():
            normalized = self._normalize_yolo_name(name)
            canonical = aliases.get(normalized)
            if canonical is None:
                for class_name in self.yolo_class_names:
                    if class_name in normalized:
                        canonical = class_name
                        break
            if canonical in wanted:
                target_class_ids[item_id] = canonical
        if target_class_ids:
            return target_class_ids.get(class_id)
        if len(names) == 1 and len(wanted) == 1:
            return next(iter(wanted))
        if not names and class_id == 0 and len(wanted) == 1:
            return next(iter(wanted))
        return None

    def _yolo_mask_rows(self, masks: Any, shape_hw: Tuple[int, int]) -> List[Any]:
        if masks is None or getattr(masks, "data", None) is None:
            return []
        mask_data = masks.data
        try:
            rows = mask_data.detach().cpu().numpy()
        except Exception:
            try:
                rows = mask_data.cpu().numpy()
            except Exception:
                return []
        return [self._mask_from_yolo(row, shape_hw) for row in rows]

    def _yolo_target_center(self, mask: Optional[Any], x: int, y: int,
                            w: int, h: int) -> Tuple[int, int]:
        if mask is not None:
            moments = self.cv2.moments(mask[y:y + h, x:x + w])
            if abs(moments["m00"]) > 1e-6:
                return (
                    int(x + moments["m10"] / moments["m00"]),
                    int(y + moments["m01"] / moments["m00"]),
                )
        return (int(x + w * 0.5), int(y + h * 0.5))

    def _normalize_yolo_name(self, name: Any) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    def _median_depth_for_bbox_center(self, depth_m: Optional[Any], image_width: int,
                                      image_height: int, x: int, y: int,
                                      w: int, h: int) -> Optional[float]:
        if depth_m is None:
            return None
        dh, dw = depth_m.shape[:2]
        if dh <= 0 or dw <= 0:
            return None

        frac = clamp(self.yolo_center_depth_fraction, 0.10, 1.0)
        sx = float(dw) / float(image_width)
        sy = float(dh) / float(image_height)
        cx0 = x + w * (0.5 - frac * 0.5)
        cy0 = y + h * (0.5 - frac * 0.5)
        cx1 = x + w * (0.5 + frac * 0.5)
        cy1 = y + h * (0.5 + frac * 0.5)

        depth = self._median_depth_for_scaled_roi(depth_m, cx0, cy0, cx1, cy1, sx, sy)
        if depth is not None:
            return depth
        return self._median_depth_for_scaled_roi(depth_m, x, y, x + w, y + h, sx, sy)

    def _median_depth_for_scaled_roi(self, depth_m: Any, x0: float, y0: float,
                                     x1: float, y1: float, sx: float,
                                     sy: float) -> Optional[float]:
        dh, dw = depth_m.shape[:2]
        dx0 = max(0, min(dw - 1, int(math.floor(x0 * sx))))
        dy0 = max(0, min(dh - 1, int(math.floor(y0 * sy))))
        dx1 = max(dx0 + 1, min(dw, int(math.ceil(x1 * sx))))
        dy1 = max(dy0 + 1, min(dh, int(math.ceil(y1 * sy))))
        roi_depth = depth_m[dy0:dy1, dx0:dx1]
        if roi_depth.size <= 0:
            return None
        valid = roi_depth[
            self.np.isfinite(roi_depth)
            & (roi_depth > 0.05)
            & (roi_depth < 5.0)
        ]
        if valid.size < 5:
            return None
        return float(self.np.median(valid))

    def _normalized_yolo_names(self, names: Any) -> Dict[int, str]:
        if isinstance(names, dict):
            normalized: Dict[int, str] = {}
            for key, value in names.items():
                try:
                    normalized[int(key)] = str(value)
                except Exception:
                    continue
            return normalized
        if isinstance(names, (list, tuple)):
            return {idx: str(value) for idx, value in enumerate(names)}
        return {}

    def _as_float_list(self, value: Any) -> List[float]:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            if hasattr(value, "tolist"):
                value = value.tolist()
            while isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            if isinstance(value, (list, tuple)):
                return [float(item) for item in value]
            return [float(value)]
        except Exception:
            return []

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        values = self._as_float_list(value)
        if not values:
            return float(default)
        return float(values[0])


class YoloScene3Perception(YoloScene1Perception):
    def __init__(self, scene: str, base_frame: str = BASE_FRAME,
                 model_path: str = DEFAULT_SCENE3_YOLO_MODEL,
                 device: str = "auto",
                 conf: float = SCENE1_YOLO_DEFAULT_CONF,
                 imgsz: int = SCENE1_YOLO_DEFAULT_IMGSZ):
        super().__init__(
            scene=scene,
            base_frame=base_frame,
            model_path=model_path,
            device=device,
            conf=conf,
            imgsz=imgsz,
        )
        self.yolo_class_name = SCENE3_YOLO_CLASS_NAME
        self.yolo_target_bin = SCENE3_YOLO_TARGET_BIN
        self.yolo_class_names = (SCENE3_YOLO_CLASS_NAME,)
        self.yolo_class_aliases = dict(SCENE3_YOLO_CLASS_ALIASES)
        self.yolo_target_bins = {SCENE3_YOLO_CLASS_NAME: SCENE3_YOLO_TARGET_BIN}
        self.yolo_world_prompts = tuple(SCENE3_YOLO_WORLD_PROMPTS)

    def _ensure_yolo(self) -> bool:
        if self._yolo_model is not None:
            return True
        if self._yolo_unavailable_reason is not None:
            return False
        if not self._ensure_cv():
            self._yolo_unavailable_reason = "cv2/numpy import failed"
            return False
        if (not os.path.isfile(self.yolo_model_path)
                and not self._is_ultralytics_world_model_name(self.yolo_model_path)):
            self._yolo_unavailable_reason = (
                "model not found: {}; generate scene3_smt_tray.pt with YOLOWorld "
                "or pass yolov8s-worldv2.pt as --yolo-model"
            ).format(self.yolo_model_path)
            return False

        load_errors: List[str] = []
        if self._try_load_scene3_yolo_world(load_errors):
            return True

        self._yolo_unavailable_reason = "; ".join(load_errors) or "ultralytics/model load failed"
        self._yolo_model = None
        return False

    def _try_load_scene3_yolo_world(self, load_errors: List[str]) -> bool:
        try:
            from ultralytics import YOLOWorld

            model = YOLOWorld(self.yolo_model_path)
            prompts_applied = self._apply_scene3_world_prompts(model)
            self._yolo_model = model
            self._log_scene3_yolo_ready("YOLOWorld", prompts_applied=prompts_applied)
            return True
        except Exception as exc:
            load_errors.append("YOLOWorld load failed: {}".format(exc))
            self._yolo_model = None
            return False

    def _apply_scene3_world_prompts(self, model: Any) -> bool:
        if not self._scene3_model_needs_prompts() or not hasattr(model, "set_classes"):
            return False
        model.set_classes(list(self.yolo_world_prompts))
        return True

    def _scene3_model_needs_prompts(self) -> bool:
        basename = os.path.basename(self.yolo_model_path).lower()
        return "world" in basename

    def _is_ultralytics_world_model_name(self, model_path: str) -> bool:
        basename = os.path.basename(model_path)
        return (
            basename == model_path
            and basename.lower().endswith(".pt")
            and "world" in basename.lower()
        )

    def _log_scene3_yolo_ready(self, backend: str, prompts_applied: bool) -> None:
        if self._yolo_ready_logged:
            return
        prompt_summary = ",".join(
            prompt or "<background>" for prompt in self.yolo_world_prompts
        )
        log_stage(
            self.scene,
            "YOLO",
            "已加载%s模型 %s device=%s conf=%.2f imgsz=%d prompts_applied=%s prompts=%s",
            backend,
            self.yolo_model_path,
            self.yolo_device,
            self.yolo_conf,
            self.yolo_imgsz,
            prompts_applied,
            prompt_summary,
        )
        self._yolo_ready_logged = True


# ---------------------------------------------------------------------------
# 导航层
# ---------------------------------------------------------------------------


class Navigation:
    def __init__(self, robot: RobotBase):
        self.robot = robot
        self.scene = robot.scene

    def drive_for(self, linear_x: float = 0.0, linear_y: float = 0.0,
                  angular_z: float = 0.0, duration: float = 1.0,
                  label: str = "drive") -> None:
        log_stage(
            self.scene,
            "导航",
            "%s vx=%.2f vy=%.2f wz=%.2f duration=%.1fs",
            label,
            linear_x,
            linear_y,
            angular_z,
            duration,
        )
        self.robot.drive_for(linear_x=linear_x, linear_y=linear_y,
                             angular_z=angular_z, duration=duration)

    def fine_align_to_image_target(self, target: PerceptionTarget) -> None:
        if not target.center_uv:
            return
        image_size = target.debug.get("image_size") if target.debug else None
        if not image_size:
            return
        width = float(image_size[0])
        error = (float(target.center_uv[0]) - width * 0.5) / max(width * 0.5, 1.0)
        if abs(error) < 0.10:
            return
        angular = -clamp(error * 0.18, -0.12, 0.12)
        self.drive_for(angular_z=angular, duration=0.35, label="fine_align")


# ---------------------------------------------------------------------------
# 操作/技能层
# ---------------------------------------------------------------------------


class Skills:
    def __init__(self, scene: str, robot: RobotBase, navigation: Navigation):
        self.scene = scene
        self.robot = robot
        self.navigation = navigation

    def choose_arm(self, target: PerceptionTarget) -> str:
        if target.position_base is None:
            return "right"
        return "left" if float(target.position_base[1]) >= 0.05 else "right"

    def pick(self, target: PerceptionTarget, arm: Optional[str] = None) -> ActionResult:
        start = time.time()
        if target.position_base is None:
            reason = "target has no base_link position"
            log_warn(self.scene, "抓取", "%s class=%s", reason, target.class_name)
            return ActionResult(False, "pick", reason, target, time.time() - start)

        arm = arm or self.choose_arm(target)
        xyz = self._reachable_xyz(self._grasp_xyz(target))
        yaw = float(target.yaw_rad or 0.0)
        pre = (xyz[0], xyz[1], xyz[2] + SAFE_PRE_GRASP_DZ_M)
        grasp = (xyz[0], xyz[1], xyz[2])
        lift = (xyz[0], xyz[1], xyz[2] + SAFE_LIFT_DZ_M)

        log_stage(
            self.scene,
            "抓取",
            "class=%s arm=%s pre=%s grasp=%s yaw=%.1fdeg",
            target.class_name,
            arm,
            _fmt_xyz(pre),
            _fmt_xyz(grasp),
            math.degrees(yaw),
        )
        self.navigation.fine_align_to_image_target(target)
        self.robot.open_claw(arm)
        self.robot.move_hand_ik(arm, pre, yaw_rad=yaw, duration=1.2, label="pre_grasp")
        ik_ok = self.robot.move_hand_ik(arm, grasp, yaw_rad=yaw, duration=1.0, label="grasp")
        self.robot.close_claw(arm, percent=self._close_percent(target.class_name))
        self.robot.move_hand_ik(arm, lift, yaw_rad=yaw, duration=1.0, label="lift")
        reason = "" if ik_ok else "IK fallback used"
        return ActionResult(True, "pick", reason, target, time.time() - start)

    def place(self, arm: str, place_xyz_base: Sequence[float],
              label: str = "place") -> ActionResult:
        start = time.time()
        place = self._reachable_xyz(place_xyz_base)
        pre = (place[0], place[1], place[2] + SAFE_PLACE_DZ_M)
        retreat = (place[0], place[1], place[2] + SAFE_LIFT_DZ_M)
        log_stage(self.scene, "放置", "arm=%s label=%s place=%s", arm, label, _fmt_xyz(place))
        self.robot.move_hand_ik(arm, pre, duration=1.1, label=label + "_pre")
        self.robot.move_hand_ik(arm, place, duration=0.9, label=label)
        self.robot.open_claw(arm)
        self.robot.move_hand_ik(arm, retreat, duration=0.9, label=label + "_retreat")
        return ActionResult(True, "place", "", None, time.time() - start)

    def pick_and_place(self, target: PerceptionTarget, place_xyz_base: Sequence[float],
                       arm: Optional[str] = None, label: str = "pick_place") -> ActionResult:
        arm = arm or self.choose_arm(target)
        pick_result = self.pick(target, arm=arm)
        if not pick_result.ok:
            return pick_result
        place_result = self.place(arm, place_xyz_base, label=label)
        return ActionResult(
            place_result.ok,
            "pick_place",
            place_result.reason,
            target,
            pick_result.duration_sec + place_result.duration_sec,
        )

    def _reachable_xyz(self, xyz: Sequence[float]) -> Tuple[float, float, float]:
        z_max = SCENE_SAFE_MAX_Z_BASE_M.get(self.scene, SAFE_MAX_Z_BASE_M)
        return (
            clamp(float(xyz[0]), -0.05, REACH_X_MAX_M),
            float(xyz[1]),
            clamp(float(xyz[2]), SAFE_MIN_Z_BASE_M, z_max),
        )

    def _grasp_xyz(self, target: PerceptionTarget) -> Tuple[float, float, float]:
        assert target.position_base is not None
        xyz = list(target.position_base)
        if self.scene == "scene2":
            offset = SCENE2_GRASP_OFFSETS_BASE.get(target.class_name, (0.0, 0.0, 0.0))
            xyz = [float(xyz[i]) + float(offset[i]) for i in range(3)]
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    def _close_percent(self, class_name: str) -> float:
        if class_name == "smt_tray":
            return 72.0
        if class_name == "part_type_c":
            return 82.0
        return CLAW_CLOSE_PERCENT


# ---------------------------------------------------------------------------
# 场景任务控制层
# ---------------------------------------------------------------------------


class BaseSceneController:
    def __init__(self, scene: str, robot: RobotBase, perception: RuleBasedPerception,
                 navigation: Navigation, skills: Skills):
        self.scene = scene
        self.robot = robot
        self.perception = perception
        self.navigation = navigation
        self.skills = skills

    def run(self) -> ActionResult:
        raise NotImplementedError

    def wait_for_perception(self) -> bool:
        log_stage(self.scene, "等待传感器", "等待相机与 TF")
        return self.perception.wait_ready(timeout_sec=TARGET_WAIT_SEC)

    def detect_targets(self, class_names: Optional[Sequence[str]] = None,
                       max_targets: Optional[int] = None) -> List[PerceptionTarget]:
        raw_targets = self.perception.detect_targets(class_names=class_names, max_targets=None)
        targets: List[PerceptionTarget] = []
        rejected: List[Tuple[PerceptionTarget, str]] = []
        for target in raw_targets:
            reason = self._target_reject_reason(target)
            if reason is None:
                targets.append(target)
            else:
                rejected.append((target, reason))

        self._log_rejected_targets(rejected)
        if max_targets is not None:
            targets = targets[:int(max_targets)]
        log_target_list(self.scene, "检测", targets)
        return targets

    def resolve_place(self, name: str) -> Tuple[float, float, float]:
        detected = self.perception.detect_bins(bin_names=[name])
        detected = [item for item in detected if item.position_base is not None]
        if detected:
            best = sorted(detected, key=lambda item: item.confidence, reverse=True)[0]
            xyz = best.position_base
            assert xyz is not None
            place = (float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.08)
            log_stage(self.scene, "目标区", "%s 使用感知检测位置 %s", name, _fmt_xyz(place))
            return place

        place = PLACE_TARGETS_BASE[self.scene][name]
        log_warn(self.scene, "目标区", "%s 未检测到，使用预设兜底 %s", name, _fmt_xyz(place))
        return place

    def make_target_at(self, class_name: str, xyz_base: Sequence[float],
                       target_bin: Optional[str] = None) -> PerceptionTarget:
        return PerceptionTarget(
            class_name=class_name,
            confidence=0.15,
            center_uv=None,
            position_base=(float(xyz_base[0]), float(xyz_base[1]), float(xyz_base[2])),
            position_camera=None,
            camera="fallback",
            area_px=0.0,
            target_bin=target_bin,
            timestamp=time.time(),
            debug={"source": "nominal_place_regrasp"},
        )

    def _target_is_usable(self, target: PerceptionTarget) -> bool:
        return self._target_reject_reason(target) is None

    def _target_reject_reason(self, target: PerceptionTarget) -> Optional[str]:
        if target.position_base is None:
            depth = None
            frame = None
            if target.debug:
                depth = target.debug.get("depth_m")
                frame = target.debug.get("frame_id")
            return "no base_link position depth={} frame={}".format(depth, frame)
        x, _y, z = target.position_base
        bounds = TARGET_BOUNDS_BASE.get(self.scene, TARGET_BOUNDS_BASE["default"])
        if not (bounds["x_min"] <= x <= bounds["x_max"]):
            return "x out of range {:.3f} not in [{:.2f},{:.2f}]".format(
                float(x),
                bounds["x_min"],
                bounds["x_max"],
            )
        if not (bounds["z_min"] <= z <= bounds["z_max"]):
            return "z out of range {:.3f} not in [{:.2f},{:.2f}]".format(
                float(z),
                bounds["z_min"],
                bounds["z_max"],
            )
        return None

    def _log_rejected_targets(self, rejected: Sequence[Tuple[PerceptionTarget, str]]) -> None:
        for target, reason in list(rejected)[:4]:
            log_warn(
                self.scene,
                "检测过滤",
                "丢弃 class=%s cam=%s uv=%s base=%s reason=%s",
                target.class_name,
                target.camera,
                target.center_uv,
                _fmt_xyz(target.position_base),
                reason,
            )
        if len(rejected) > 4:
            log_warn(self.scene, "检测过滤", "另有 %d 个候选被过滤", len(rejected) - 4)


class Scene1Controller(BaseSceneController):
    def run(self) -> ActionResult:
        start = time.time()
        self.wait_for_perception()
        weigh_xyz = self.resolve_scene1_yolo_place("weighing_area")
        drop_xyz = self.resolve_scene1_yolo_place("drop_box")
        if weigh_xyz is None or drop_xyz is None:
            reason = "scene1 yolo place target missing"
            log_error(self.scene, "失败", reason)
            return ActionResult(False, "scene1", reason, duration_sec=time.time() - start)

        targets = self.detect_targets(class_names=["parcel"], max_targets=SCENE_TARGET_LIMIT["scene1"])
        if not targets:
            reason = "no parcel detected"
            log_error(self.scene, "失败", reason)
            return ActionResult(False, "scene1", reason, duration_sec=time.time() - start)

        targets.sort(key=lambda t: (t.position_base[1], t.position_base[0]) if t.position_base else (0.0, 0.0))
        for index, parcel in enumerate(targets, start=1):
            log_stage(self.scene, "流程", "快递 #%02d 抓取 -> 称重 -> 入箱", index)
            arm = self.skills.choose_arm(parcel)
            result = self.skills.pick_and_place(parcel, weigh_xyz, arm=arm, label="parcel{}_to_weigh".format(index))
            if not result.ok:
                log_warn(self.scene, "失败", "快递 #%02d 称重放置失败: %s", index, result.reason)
                continue
            self.robot.rospy.sleep(0.7)
            regrasp_xyz = (weigh_xyz[0], weigh_xyz[1], weigh_xyz[2] - 0.04)
            weighed = self.make_target_at("parcel", regrasp_xyz, target_bin="drop_box")
            result = self.skills.pick_and_place(weighed, drop_xyz, arm=arm, label="parcel{}_to_box".format(index))
            if not result.ok:
                log_warn(self.scene, "失败", "快递 #%02d 入箱失败: %s", index, result.reason)

        return ActionResult(True, "scene1", "done", duration_sec=time.time() - start)

    def resolve_scene1_yolo_place(self, name: str) -> Optional[Tuple[float, float, float]]:
        detected = self.perception.detect_bins(bin_names=[name])
        detected = [item for item in detected if item.position_base is not None]
        if not detected:
            log_error(self.scene, "目标区", "%s 未被 YOLO 检测到，场景一不再使用规则或几何兜底", name)
            return None
        best = sorted(detected, key=lambda item: item.confidence, reverse=True)[0]
        xyz = best.position_base
        assert xyz is not None
        place = (float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.08)
        log_stage(self.scene, "目标区", "%s 使用 YOLO 检测位置 %s", name, _fmt_xyz(place))
        return place


class Scene2Controller(BaseSceneController):
    PART_TO_BIN = {
        "part_type_a": "sorting_bin_a",
        "part_type_b": "sorting_bin_b",
        "part_type_c": "sorting_bin_c",
    }
    CLASS_ORDER = {"part_type_b": 0, "part_type_a": 1, "part_type_c": 2}

    def run(self) -> ActionResult:
        start = time.time()
        self.wait_for_perception()
        bin_places = {name: self.resolve_place(name) for name in ["sorting_bin_a", "sorting_bin_b", "sorting_bin_c"]}

        targets = self._stable_detect_parts()
        if not targets:
            log_warn(self.scene, "检测", "视觉未得到可用零件，启用公开几何兜底点")
            targets = self._nominal_parts()

        targets.sort(key=self._part_sort_key)
        for index, part in enumerate(targets, start=1):
            bin_name = self.PART_TO_BIN.get(part.class_name)
            if not bin_name:
                log_warn(self.scene, "分拣", "未知类别 %s，跳过", part.class_name)
                continue
            log_stage(
                self.scene,
                "流程",
                "零件 #%02d class=%s -> %s",
                index,
                part.class_name,
                bin_name,
            )
            result = self.skills.pick_and_place(
                part,
                bin_places[bin_name],
                arm=self.skills.choose_arm(part),
                label="{}_to_{}".format(part.class_name, bin_name),
            )
            if not result.ok:
                log_warn(self.scene, "失败", "零件 #%02d 分拣失败: %s", index, result.reason)

        return ActionResult(True, "scene2", "done", duration_sec=time.time() - start)

    def _stable_detect_parts(self) -> List[PerceptionTarget]:
        previous: Optional[List[PerceptionTarget]] = None
        stable_count = 0
        best: List[PerceptionTarget] = []
        for attempt in range(1, SCENE2_DETECT_RETRY + 1):
            targets = self.detect_targets(
                class_names=["part_type_a", "part_type_b", "part_type_c"],
                max_targets=SCENE_TARGET_LIMIT["scene2"],
            )
            if len(targets) > len(best):
                best = list(targets)
            if len(targets) < SCENE2_STABLE_TARGET_COUNT:
                log_warn(
                    self.scene,
                    "检测",
                    "第 %d 次仅检测到 %d/%d 个零件",
                    attempt,
                    len(targets),
                    SCENE2_STABLE_TARGET_COUNT,
                )
                stable_count = 0
                previous = targets
                self.robot.rospy.sleep(0.25)
                continue
            if previous is not None and self._targets_stable(previous, targets):
                stable_count += 1
            else:
                stable_count = 1
            previous = targets
            log_stage(self.scene, "检测", "第 %d 次检测稳定计数 %d/%d", attempt, stable_count, SCENE2_STABLE_FRAMES)
            if stable_count >= SCENE2_STABLE_FRAMES:
                return targets
            self.robot.rospy.sleep(0.25)
        if best:
            log_warn(self.scene, "检测", "未达到连续帧稳定，使用本轮最佳检测结果 %d 个", len(best))
        return best

    def _targets_stable(self, previous: Sequence[PerceptionTarget],
                        current: Sequence[PerceptionTarget]) -> bool:
        if len(previous) != len(current):
            return False
        prev_by_class: Dict[str, List[PerceptionTarget]] = {}
        curr_by_class: Dict[str, List[PerceptionTarget]] = {}
        for target in previous:
            prev_by_class.setdefault(target.class_name, []).append(target)
        for target in current:
            curr_by_class.setdefault(target.class_name, []).append(target)
        for class_name in set(prev_by_class) | set(curr_by_class):
            prev_items = sorted(prev_by_class.get(class_name, []), key=lambda t: t.position_base[1] if t.position_base else 0.0)
            curr_items = sorted(curr_by_class.get(class_name, []), key=lambda t: t.position_base[1] if t.position_base else 0.0)
            if len(prev_items) != len(curr_items):
                return False
            for old, new in zip(prev_items, curr_items):
                if old.position_base is None or new.position_base is None:
                    return False
                if distance3(old.position_base, new.position_base) > SCENE2_STABLE_POS_EPS_M:
                    return False
        return True

    def _nominal_parts(self) -> List[PerceptionTarget]:
        targets = []
        for index, (class_name, xyz) in enumerate(SCENE2_NOMINAL_PARTS_BASE, start=1):
            targets.append(PerceptionTarget(
                class_name=class_name,
                confidence=0.05,
                center_uv=None,
                position_base=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                position_camera=None,
                camera="fallback",
                area_px=0.0,
                yaw_rad=0.0,
                target_bin=self.PART_TO_BIN[class_name],
                timestamp=time.time(),
                debug={"source": "scene2_nominal_fallback", "index": index},
            ))
        return targets

    def _part_sort_key(self, target: PerceptionTarget) -> Tuple[int, float, float]:
        xyz = target.position_base or (0.0, 0.0, 0.0)
        return (self.CLASS_ORDER.get(target.class_name, 99), abs(float(xyz[1])), float(xyz[0]))


class Scene3Controller(BaseSceneController):
    def run(self) -> ActionResult:
        start = time.time()
        if not self.wait_for_perception():
            return ActionResult(False, "scene3", "sensor not ready", duration_sec=time.time() - start)
        outbound_xyz = self.resolve_outbound_place()

        moved_to_rack = False
        completed = 0
        completed_shelves = set()
        max_trays = SCENE_TARGET_LIMIT["scene3"]
        for index in range(1, max_trays + 1):
            if not moved_to_rack:
                self.navigation.drive_for(linear_x=0.16, duration=3.2, label="approach_smt_rack")
                moved_to_rack = True
                self.robot.rospy.sleep(0.5)

            targets = self.detect_reachable_trays(completed_shelves)
            if not targets:
                log_warn(self.scene, "检测", "未检测到更多料盘，结束出库循环")
                break
            targets.sort(key=self._tray_sort_key)
            tray = targets[0]
            shelf = self._tray_shelf(tray)
            arm = self.skills.choose_arm(tray)
            log_stage(
                self.scene,
                "流程",
                "料盘 #%02d shelf=%s 出库，arm=%s base=%s",
                index,
                shelf,
                arm,
                _fmt_xyz(tray.position_base),
            )
            pick_result = self.skills.pick(tray, arm=arm)
            if not pick_result.ok:
                log_warn(self.scene, "失败", "料盘 #%02d 抓取失败: %s", index, pick_result.reason)
                break

            self.navigation.drive_for(linear_x=-0.16, duration=3.4, label="return_to_outbound_box")
            moved_to_rack = False
            self.skills.place(arm, outbound_xyz, label="tray{}_outbound".format(index))
            completed_shelves.add(shelf)
            completed += 1

        if completed == 0:
            return ActionResult(False, "scene3", "no tray moved", duration_sec=time.time() - start)
        return ActionResult(True, "scene3", "moved {} tray(s)".format(completed),
                            duration_sec=time.time() - start)

    def detect_reachable_trays(self, completed_shelves: set) -> List[PerceptionTarget]:
        for attempt in range(1, SCENE3_APPROACH_RETRY + 1):
            raw_targets = self.perception.detect_targets(class_names=["smt_tray"], max_targets=None)
            raw_targets = [
                target for target in raw_targets
                if self._tray_shelf(target) not in completed_shelves
            ]
            log_target_list(self.scene, "检测", raw_targets)

            reachable: List[PerceptionTarget] = []
            rejected: List[Tuple[PerceptionTarget, str]] = []
            far_targets: List[PerceptionTarget] = []
            for target in raw_targets:
                reason = self._target_reject_reason(target)
                if reason is None:
                    reachable.append(target)
                    continue
                rejected.append((target, reason))
                if self._target_is_far_but_approachable(target, reason):
                    far_targets.append(target)

            self._log_rejected_targets(rejected)
            if reachable:
                return reachable
            if not far_targets:
                return []

            far_targets.sort(key=lambda item: float(item.position_base[0]) if item.position_base else 99.0)
            target = far_targets[0]
            assert target.position_base is not None
            x = float(target.position_base[0])
            duration = clamp(
                (x - SCENE3_APPROACH_TARGET_X_M) / SCENE3_APPROACH_SPEED_MPS,
                SCENE3_APPROACH_MIN_DURATION_SEC,
                SCENE3_APPROACH_MAX_DURATION_SEC,
            )
            log_stage(
                self.scene,
                "导航",
                "第 %d 次检测到料盘仍偏远 base=%s，继续靠近 %.1fs 后重试",
                attempt,
                _fmt_xyz(target.position_base),
                duration,
            )
            self.navigation.drive_for(
                linear_x=SCENE3_APPROACH_SPEED_MPS,
                duration=duration,
                label="approach_visible_tray",
            )
            self.robot.rospy.sleep(0.35)
        return []

    def _target_is_far_but_approachable(self, target: PerceptionTarget, reason: str) -> bool:
        if target.position_base is None or "x out of range" not in reason:
            return False
        x = float(target.position_base[0])
        return TARGET_BOUNDS_BASE["scene3"]["x_max"] < x <= SCENE3_APPROACH_MAX_DETECT_X_M

    def _tray_sort_key(self, target: PerceptionTarget) -> Tuple[int, float, float, float]:
        xyz = target.position_base or (99.0, 99.0, 99.0)
        shelf_priority = 0 if self._tray_shelf(target) == "lower" else 1
        return (shelf_priority, abs(float(xyz[1])), float(xyz[0]), -float(target.confidence))

    def _tray_shelf(self, target: PerceptionTarget) -> str:
        xyz = target.position_base or (99.0, 99.0, 99.0)
        return "lower" if float(xyz[2]) < SCENE3_SHELF_SPLIT_Z_M else "upper"

    def resolve_outbound_place(self) -> Tuple[float, float, float]:
        place = PLACE_TARGETS_BASE[self.scene]["outbound_box"]
        log_stage(self.scene, "目标区", "outbound_box 使用预设出库区 %s", _fmt_xyz(place))
        return place


# ---------------------------------------------------------------------------
# 系统入口与装配层
# ---------------------------------------------------------------------------


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


def build_controller(scene: str, robot: RobotBase, perception: RuleBasedPerception,
                     navigation: Navigation, skills: Skills) -> BaseSceneController:
    if scene == "scene1":
        return Scene1Controller(scene, robot, perception, navigation, skills)
    if scene == "scene2":
        return Scene2Controller(scene, robot, perception, navigation, skills)
    if scene == "scene3":
        return Scene3Controller(scene, robot, perception, navigation, skills)
    raise ValueError("unknown scene: {}".format(scene))


def default_yolo_model_for_scene(scene: str) -> str:
    if scene == "scene3":
        return DEFAULT_SCENE3_YOLO_MODEL
    return DEFAULT_SCENE1_YOLO_MODEL


def build_perception(scene: str, scene1_perception: str,
                     yolo_model: Optional[str], yolo_device: str,
                     yolo_conf: float, yolo_imgsz: int) -> RuleBasedPerception:
    if scene == "scene1":
        if scene1_perception == "rule":
            raise ValueError("scene1 rule-based perception has been removed; use YOLO classes parcel/weighing_area/drop_box")
        return YoloScene1Perception(
            scene=scene,
            base_frame=BASE_FRAME,
            model_path=yolo_model or default_yolo_model_for_scene(scene),
            device=yolo_device,
            conf=yolo_conf,
            imgsz=yolo_imgsz,
        )
    if scene == "scene3":
        return YoloScene3Perception(
            scene=scene,
            base_frame=BASE_FRAME,
            model_path=yolo_model or default_yolo_model_for_scene(scene),
            device=yolo_device,
            conf=yolo_conf,
            imgsz=yolo_imgsz,
        )
    return RuleBasedPerception(scene=scene, base_frame=BASE_FRAME)


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True,
              scene1_perception="auto",
              yolo_model=None,
              yolo_device="auto",
              yolo_conf=SCENE1_YOLO_DEFAULT_CONF,
              yolo_imgsz=SCENE1_YOLO_DEFAULT_IMGSZ):
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
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState

    rospy.loginfo("=== %s任务启动 ===", config["title"])

    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)

    rospy.sleep(1.0)
    rospy.loginfo("场景实例已初始化。")

    robot = RobotBase(cmd_vel_pub=cmd_vel_pub, arm_traj_pub=arm_traj_pub, scene=scene)
    perception = build_perception(
        scene=scene,
        scene1_perception=scene1_perception,
        yolo_model=yolo_model,
        yolo_device=yolo_device,
        yolo_conf=yolo_conf,
        yolo_imgsz=yolo_imgsz,
    )
    navigation = Navigation(robot)
    skills = Skills(scene=scene, robot=robot, navigation=navigation)
    controller = build_controller(scene, robot, perception, navigation, skills)

    result = ActionResult(False, "run_scene", "not started")
    try:
        robot.prepare()
        result = controller.run()
        if result.ok:
            rospy.loginfo("[%s][完成] %s duration=%.1fs", scene, result.reason, result.duration_sec)
        else:
            rospy.logwarn("[%s][未完成] stage=%s reason=%s duration=%.1fs",
                          scene, result.stage, result.reason, result.duration_sec)
    finally:
        robot.finish()
        rospy.sleep(0.3)
    return result


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
    parser.add_argument("--scene1-perception", choices=["auto", "yolo"], default="auto",
                        help="场景一识别后端：auto/yolo 均使用 YOLO，不再支持规则识别")
    parser.add_argument("--yolo-model", default=None,
                        help="YOLO .pt 模型路径；未指定时按 scene 选择默认权重")
    parser.add_argument("--yolo-device", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"], default="auto",
                        help="YOLO 推理设备；auto 由 ultralytics 自动选择")
    parser.add_argument("--yolo-conf", type=float, default=SCENE1_YOLO_DEFAULT_CONF,
                        help="YOLO 检测置信度阈值")
    parser.add_argument("--yolo-imgsz", type=int, default=SCENE1_YOLO_DEFAULT_IMGSZ,
                        help="YOLO 推理输入尺寸")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
        scene1_perception=args.scene1_perception,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
    )


if __name__ == "__main__":
    main()
