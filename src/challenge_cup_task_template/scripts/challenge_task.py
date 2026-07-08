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
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 全局配置与常量区
# ---------------------------------------------------------------------------

BASE_FRAME = "base_link"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE3_YOLO_MODEL = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "models", "yolo", "scene3_smt_tray.pt")
)
DEFAULT_SCENE1_YOLO_MODEL = DEFAULT_SCENE3_YOLO_MODEL
DEFAULT_SCENE2_YOLO_MODEL = DEFAULT_SCENE3_YOLO_MODEL
SCENE1_YOLO_CLASS_NAMES = ("parcel", "weighing_area", "drop_box")
SCENE1_YOLO_WORLD_PROMPTS = (
    "parcel",
    "colored rectangular cuboid",
    "different colored rectangular cuboid",
    "long rectangular cuboid",
    "thin rectangular cuboid",
    "small rectangular cuboid",
    "rectangular prism",
    "colored rectangular prism",
    "colored rectangular block",
    "long rectangular block",
    "yellow rectangular cuboid",
    "brown rectangular cuboid",
    "orange rectangular cuboid",
    "white rectangular cuboid",
    "gray rectangular cuboid",
    "taped rectangular cuboid",
    "rectangular box with tape",
    "vivid colored rectangle area",
    "vivid colored square area",
    "bright colored rectangular area",
    "bright colored square area",
    "different colored rectangular area",
    "different colored square area",
    "colored rectangular region",
    "colored square region",
    "colored rectangular patch",
    "colored square patch",
    "flat colored rectangle",
    "flat colored square",
    "green rectangular area",
    "green square area",
    "bright green rectangular area",
    "bright green square area",
    "vivid green rectangular area",
    "vivid green square area",
    "green rectangular target",
    "green square target",
    "green rectangular patch",
    "green square patch",
    "bright green rectangle",
    "bright green square",
    "vivid green mat",
    "bright green mat",
    "green rectangular mat",
    "green square mat",
    "weighing mat",
    "weighing pad",
    "drop box",
    "sorting box",
    "destination box",
    "green sorting box",
    "open green box",
    "open dark green box",
    "green open bin",
    "open top green bin",
    "green storage bin",
    "empty green container",
    "",
)
SCENE1_YOLO_CLASS_ALIASES = {
    "parcel": "parcel",
    "colored_rectangular_cuboid": "parcel",
    "colored rectangular cuboid": "parcel",
    "different_colored_rectangular_cuboid": "parcel",
    "different colored rectangular cuboid": "parcel",
    "long_rectangular_cuboid": "parcel",
    "long rectangular cuboid": "parcel",
    "thin_rectangular_cuboid": "parcel",
    "thin rectangular cuboid": "parcel",
    "small_rectangular_cuboid": "parcel",
    "small rectangular cuboid": "parcel",
    "rectangular_prism": "parcel",
    "rectangular prism": "parcel",
    "colored_rectangular_prism": "parcel",
    "colored rectangular prism": "parcel",
    "colored_rectangular_block": "parcel",
    "colored rectangular block": "parcel",
    "long_rectangular_block": "parcel",
    "long rectangular block": "parcel",
    "yellow_rectangular_cuboid": "parcel",
    "yellow rectangular cuboid": "parcel",
    "brown_rectangular_cuboid": "parcel",
    "brown rectangular cuboid": "parcel",
    "orange_rectangular_cuboid": "parcel",
    "orange rectangular cuboid": "parcel",
    "white_rectangular_cuboid": "parcel",
    "white rectangular cuboid": "parcel",
    "gray_rectangular_cuboid": "parcel",
    "gray rectangular cuboid": "parcel",
    "taped_rectangular_cuboid": "parcel",
    "taped rectangular cuboid": "parcel",
    "rectangular_box_with_tape": "parcel",
    "rectangular box with tape": "parcel",
    "weighing_area": "weighing_area",
    "weighing area": "weighing_area",
    "weigh_area": "weighing_area",
    "vivid_colored_rectangle_area": "weighing_area",
    "vivid colored rectangle area": "weighing_area",
    "vivid_colored_square_area": "weighing_area",
    "vivid colored square area": "weighing_area",
    "bright_colored_rectangular_area": "weighing_area",
    "bright colored rectangular area": "weighing_area",
    "bright_colored_square_area": "weighing_area",
    "bright colored square area": "weighing_area",
    "different_colored_rectangular_area": "weighing_area",
    "different colored rectangular area": "weighing_area",
    "different_colored_square_area": "weighing_area",
    "different colored square area": "weighing_area",
    "colored_rectangular_region": "weighing_area",
    "colored rectangular region": "weighing_area",
    "colored_square_region": "weighing_area",
    "colored square region": "weighing_area",
    "colored_rectangular_patch": "weighing_area",
    "colored rectangular patch": "weighing_area",
    "colored_square_patch": "weighing_area",
    "colored square patch": "weighing_area",
    "flat_colored_rectangle": "weighing_area",
    "flat colored rectangle": "weighing_area",
    "flat_colored_square": "weighing_area",
    "flat colored square": "weighing_area",
    "green_rectangular_area": "weighing_area",
    "green rectangular area": "weighing_area",
    "green_square_area": "weighing_area",
    "green square area": "weighing_area",
    "bright_green_rectangular_area": "weighing_area",
    "bright green rectangular area": "weighing_area",
    "bright_green_square_area": "weighing_area",
    "bright green square area": "weighing_area",
    "vivid_green_rectangular_area": "weighing_area",
    "vivid green rectangular area": "weighing_area",
    "vivid_green_square_area": "weighing_area",
    "vivid green square area": "weighing_area",
    "green_rectangular_target": "weighing_area",
    "green rectangular target": "weighing_area",
    "green_square_target": "weighing_area",
    "green square target": "weighing_area",
    "green_rectangular_patch": "weighing_area",
    "green rectangular patch": "weighing_area",
    "green_square_patch": "weighing_area",
    "green square patch": "weighing_area",
    "bright_green_rectangle": "weighing_area",
    "bright green rectangle": "weighing_area",
    "bright_green_square": "weighing_area",
    "bright green square": "weighing_area",
    "vivid_green_mat": "weighing_area",
    "vivid green mat": "weighing_area",
    "bright_green_mat": "weighing_area",
    "bright green mat": "weighing_area",
    "green_rectangular_mat": "weighing_area",
    "green rectangular mat": "weighing_area",
    "green_square_mat": "weighing_area",
    "green square mat": "weighing_area",
    "weighing_mat": "weighing_area",
    "weighing mat": "weighing_area",
    "weighing_pad": "weighing_area",
    "weighing pad": "weighing_area",
    "drop_box": "drop_box",
    "drop box": "drop_box",
    "sorting_box": "drop_box",
    "sorting box": "drop_box",
    "destination_box": "drop_box",
    "destination box": "drop_box",
    "green_sorting_box": "drop_box",
    "green sorting box": "drop_box",
    "open_green_box": "drop_box",
    "open green box": "drop_box",
    "open_dark_green_box": "drop_box",
    "open dark green box": "drop_box",
    "green_open_bin": "drop_box",
    "green open bin": "drop_box",
    "open_top_green_bin": "drop_box",
    "open top green bin": "drop_box",
    "green_storage_bin": "drop_box",
    "green storage bin": "drop_box",
    "empty_green_container": "drop_box",
    "empty green container": "drop_box",
}
SCENE1_YOLO_TARGET_BINS = {
    "parcel": "drop_box",
    "weighing_area": None,
    "drop_box": None,
}
SCENE2_YOLO_CLASS_NAMES = (
    "part_type_a",
    "part_type_b",
    "part_type_c",
    "sorting_bin_a",
    "sorting_bin_b",
    "sorting_bin_c",
)
SCENE2_YOLO_WORLD_PROMPTS = (
    "pipe fitting",
    "steel pipe fastener",
    "steel pipe fasteners",
    "part type a",
    "pipe clamp",
    "t junction",
    "t-junction",
    "part type b",
    "screwdriver",
    "part type c",
    "sorting bin a",
    "blue sorting bin",
    "sorting bin b",
    "orange sorting bin",
    "sorting bin c",
    "purple sorting bin",
    "",
)
SCENE2_YOLO_CLASS_ALIASES = {
    "pipe_fitting": "part_type_a",
    "pipe fitting": "part_type_a",
    "steel_pipe_fastener": "part_type_a",
    "steel pipe fastener": "part_type_a",
    "steel_pipe_fasteners": "part_type_a",
    "steel pipe fasteners": "part_type_a",
    "part_type_a": "part_type_a",
    "part type a": "part_type_a",
    "pipe_clamp": "part_type_b",
    "pipe clamp": "part_type_b",
    "t_junction": "part_type_b",
    "t junction": "part_type_b",
    "part_type_b": "part_type_b",
    "part type b": "part_type_b",
    "screwdriver": "part_type_c",
    "part_type_c": "part_type_c",
    "part type c": "part_type_c",
    "sorting_bin_a": "sorting_bin_a",
    "sorting bin a": "sorting_bin_a",
    "blue_sorting_bin": "sorting_bin_a",
    "blue sorting bin": "sorting_bin_a",
    "sorting_bin_b": "sorting_bin_b",
    "sorting bin b": "sorting_bin_b",
    "orange_sorting_bin": "sorting_bin_b",
    "orange sorting bin": "sorting_bin_b",
    "sorting_bin_c": "sorting_bin_c",
    "sorting bin c": "sorting_bin_c",
    "purple_sorting_bin": "sorting_bin_c",
    "purple sorting bin": "sorting_bin_c",
}
SCENE2_YOLO_TARGET_BINS = {
    "part_type_a": "sorting_bin_a",
    "part_type_b": "sorting_bin_b",
    "part_type_c": "sorting_bin_c",
    "sorting_bin_a": None,
    "sorting_bin_b": None,
    "sorting_bin_c": None,
}
SCENE1_YOLO_DEFAULT_CONF = 0.35
SCENE1_YOLO_DEFAULT_IMGSZ = 640
SCENE1_YOLO_CENTER_DEPTH_FRACTION = 0.45
SCENE3_YOLO_CLASS_NAME = "smt_tray"
SCENE3_YOLO_TARGET_BIN = "outbound_box"
SCENE3_YOLO_WORLD_PROMPTS = (
    "smt tray",
    "smt component tray",
    "smt material tray",
    "smt reel tray",
    "electronics component tray",
    "tray",
    "electronics tray",
    "component tray",
    "component storage tray",
    "electronic parts tray",
    "plastic tray",
    "plastic parts tray",
    "rectangular plastic tray",
    "black plastic tray",
    "industrial plastic tray",
    "tray on shelf",
    "",
)
SCENE3_YOLO_CLASS_ALIASES = {
    "tray": SCENE3_YOLO_CLASS_NAME,
    "charging_tray": SCENE3_YOLO_CLASS_NAME,
    "smt_tray": SCENE3_YOLO_CLASS_NAME,
    "smt tray": SCENE3_YOLO_CLASS_NAME,
    "smttray": SCENE3_YOLO_CLASS_NAME,
    "smt_component_tray": SCENE3_YOLO_CLASS_NAME,
    "smt component tray": SCENE3_YOLO_CLASS_NAME,
    "smt_material_tray": SCENE3_YOLO_CLASS_NAME,
    "smt material tray": SCENE3_YOLO_CLASS_NAME,
    "smt_reel_tray": SCENE3_YOLO_CLASS_NAME,
    "smt reel tray": SCENE3_YOLO_CLASS_NAME,
    "electronics_tray": SCENE3_YOLO_CLASS_NAME,
    "electronics tray": SCENE3_YOLO_CLASS_NAME,
    "electronics_component_tray": SCENE3_YOLO_CLASS_NAME,
    "electronics component tray": SCENE3_YOLO_CLASS_NAME,
    "component_tray": SCENE3_YOLO_CLASS_NAME,
    "component tray": SCENE3_YOLO_CLASS_NAME,
    "component_storage_tray": SCENE3_YOLO_CLASS_NAME,
    "component storage tray": SCENE3_YOLO_CLASS_NAME,
    "electronic_parts_tray": SCENE3_YOLO_CLASS_NAME,
    "electronic parts tray": SCENE3_YOLO_CLASS_NAME,
    "plastic_tray": SCENE3_YOLO_CLASS_NAME,
    "plastic tray": SCENE3_YOLO_CLASS_NAME,
    "plastic_parts_tray": SCENE3_YOLO_CLASS_NAME,
    "plastic parts tray": SCENE3_YOLO_CLASS_NAME,
    "rectangular_plastic_tray": SCENE3_YOLO_CLASS_NAME,
    "rectangular plastic tray": SCENE3_YOLO_CLASS_NAME,
    "black_plastic_tray": SCENE3_YOLO_CLASS_NAME,
    "black plastic tray": SCENE3_YOLO_CLASS_NAME,
    "industrial_plastic_tray": SCENE3_YOLO_CLASS_NAME,
    "industrial plastic tray": SCENE3_YOLO_CLASS_NAME,
    "tray_on_shelf": SCENE3_YOLO_CLASS_NAME,
    "tray on shelf": SCENE3_YOLO_CLASS_NAME,
}
SCENE3_SHELF_SPLIT_Z_M = 0.95
SCENE3_INITIAL_APPROACH_DURATION_SEC = 0.85
SCENE3_APPROACH_TARGET_X_M = 0.90
SCENE3_APPROACH_MAX_DETECT_X_M = 2.60
SCENE3_APPROACH_SPEED_MPS = 0.16
SCENE3_APPROACH_MIN_DURATION_SEC = 0.20
SCENE3_APPROACH_MAX_DURATION_SEC = 0.60
SCENE3_APPROACH_SETTLE_SEC = 0.10
SCENE3_APPROACH_RETRY = 48
SCENE3_APPROACH_SEARCH_DURATION_SEC = 0.45
SCENE3_APPROACH_MAX_MISSES = 12
SCENE3_WRIST_REFINE_RETRY = 2
SCENE3_WRIST_REFINE_MAX_DELTA_M = 0.45
SCENE3_WRIST_OBSERVE_DZ_M = 0.06
SCENE3_WRIST_OBSERVE_BACKOFF_X_M = 0.14
SCENE3_TRAY_GRASP_OFFSET_BASE = (0.025, 0.0, -0.015)
SCENE3_CLAW_CLOSE_PERCENT = 92.0
# 场景三货架内抓取不从上方垂直抬手：平举/插入高度使用料盘中心高度，
# 抓取点再由 SCENE3_TRAY_GRASP_OFFSET_BASE 做轻微下探。
SCENE3_SHELF_SIDE_CLEARANCE_Y_M = 0.08
SCENE3_SHELF_ENTRY_BACKOFF_X_M = 0.14
SCENE3_SHELF_SIDE_DURATION_SEC = 1.80
SCENE3_SHELF_ENTRY_DURATION_SEC = 1.60
SCENE3_SHELF_INSERT_DURATION_SEC = 2.20
SCENE3_SHELF_RETRACT_DURATION_SEC = 2.20
SCENE3_PRE_PICK_BASE_SETTLE_SEC = 0.80
SCENE3_ARM_MODE_SETTLE_SEC = 2.60
# 场景三不考速度，料盘抓取留出更长稳定时间，避免夹爪/手臂半程动作被重试打断。
SCENE3_CLAW_OPEN_SETTLE_SEC = 0.80
SCENE3_PRE_GRASP_DURATION_SEC = 2.40
SCENE3_GRASP_DURATION_SEC = 1.40
SCENE3_GRASP_DWELL_SEC = 0.60
SCENE3_CLAW_CLOSE_SETTLE_SEC = 1.60
SCENE3_POST_CLOSE_SETTLE_SEC = 0.80
SCENE3_LIFT_DURATION_SEC = 2.00
SCENE3_LIFT_SETTLE_SEC = 0.50
SCENE3_PICK_RETRY = 40
SCENE3_PICK_RETRY_SETTLE_SEC = 1.20
SCENE3_LIFT_RETRY = 3
SCENE3_LIFT_RETRY_BACKOFF_X_M = 0.04

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
SCENE_SAFE_MIN_Z_BASE_M = {
    "scene3": -0.24,
}
SCENE_SAFE_MAX_Z_BASE_M = {
    "scene3": 1.30,
}
REACH_X_MIN_M = 0.12
REACH_X_MAX_M = 0.88
SCENE_REACH_X_MAX_M = {
    "scene3": 0.94,
}
TARGET_WAIT_SEC = 8.0
SKILL_PICK_RETRY = 10
SKILL_PICK_RETRY_SETTLE_SEC = 0.30
SKILL_PRE_GRASP_DURATION_SEC = 1.80
SKILL_GRASP_DURATION_SEC = 1.50
SKILL_POST_GRASP_SETTLE_SEC = 0.35
SKILL_LIFT_DURATION_SEC = 1.50
SKILL_PLACE_PRE_DURATION_SEC = 1.60
SKILL_PLACE_DURATION_SEC = 1.30
SKILL_PLACE_RETREAT_DURATION_SEC = 1.20
SKILL_RELOCALIZE_SETTLE_SEC = 0.20
SCENE1_YOLO_RETRY = 5
SCENE1_YOLO_RETRY_SETTLE_SEC = 0.25
SCENE2_STABLE_TARGET_COUNT = 4
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
        "x_max": 0.90,
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

SCENE2_GRASP_OFFSETS_BASE = {
    # base_link 下的经验偏置。视觉给出的是实例可见区域中心，抓取点需要略微移到
    # 稳定受力位置；数值保持保守，后续按实测 seed 微调。
    "part_type_a": (0.00, 0.00, -0.015),
    "part_type_b": (0.00, 0.00, -0.010),
    "part_type_c": (0.015, 0.00, -0.012),
}

# 场景三出库区为公开指定区域，直接使用 base_link 下的预设点。
# 场景一/二目标区必须由 YOLO-World 检出，不再保留几何兜底点。
PLACE_TARGETS_BASE = {
    "scene3": {
        "outbound_box": (0.58, 0.00, 0.22),
    },
}

SCENE_TARGET_LIMIT = {
    "scene1": 4,
    "scene2": 4,
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


def scene3_shelf_name_from_z(z_base: float) -> str:
    return "lower" if float(z_base) < SCENE3_SHELF_SPLIT_Z_M else "upper"


def arm_side_sign(arm: str) -> float:
    return 1.0 if arm == "left" else -1.0


def reach_x_max_for_scene(scene: str) -> float:
    return SCENE_REACH_X_MAX_M.get(scene, REACH_X_MAX_M)


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
                if self.scene == "scene3" and int(mode) == ARM_MODE_EXTERNAL_CONTROL:
                    log_stage(
                        self.scene,
                        "手臂模式",
                        "等待外部手臂控制稳定 %.1fs 后再发送轨迹",
                        SCENE3_ARM_MODE_SETTLE_SEC,
                    )
                    self.rospy.sleep(SCENE3_ARM_MODE_SETTLE_SEC)
                return True
            except Exception as exc:
                last_exc = exc
                continue
        if warn_only:
            log_warn(self.scene, "手臂模式", "切换模式失败，继续尝试直接发轨迹: %s", last_exc)
            return False
        raise RuntimeError("set_arm_mode failed: {}".format(last_exc))

    def _arm_index_groups(self, arm: str) -> Tuple[range, range, str]:
        if arm == "left":
            return range(0, 7), range(7, 14), "右"
        if arm == "right":
            return range(7, 14), range(0, 7), "左"
        raise ValueError("unknown arm: {}".format(arm))

    def move_arm_deg(self, target_deg: Sequence[float], duration: float = 1.0,
                     active_arm: Optional[str] = None) -> None:
        from sensor_msgs.msg import JointState

        self.set_arm_mode(ARM_MODE_EXTERNAL_CONTROL)
        target = [float(v) for v in target_deg]
        if len(target) != 14:
            raise ValueError("arm target expects 14 joints, got {}".format(len(target)))

        start = self.current_arm_deg()
        if len(start) != 14:
            start = list(self.last_arm_deg)
        if active_arm in ("left", "right"):
            _active, inactive, inactive_label = self._arm_index_groups(active_arm)
            lock = list(self.last_arm_deg)
            if len(lock) != 14:
                lock = list(start)
            sensor_delta = max(abs(float(start[i]) - float(lock[i])) for i in inactive)
            target_delta = max(abs(float(target[i]) - float(lock[i])) for i in inactive)
            for i in inactive:
                start[i] = float(lock[i])
                target[i] = float(lock[i])
            if sensor_delta > 1.0 or target_delta > 1.0:
                log_stage(
                    self.scene,
                    "单臂轨迹",
                    "%s手动作时锁定%s手：sensor_delta=%.1fdeg target_delta=%.1fdeg",
                    "左" if active_arm == "left" else "右",
                    inactive_label,
                    sensor_delta,
                    target_delta,
                )
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

    def hold_arm(self, duration: float, label: str = "hold") -> None:
        from sensor_msgs.msg import JointState

        if duration <= 0.0:
            return
        self.set_arm_mode(ARM_MODE_EXTERNAL_CONTROL)
        target = list(self.last_arm_deg)
        msg = JointState()
        msg.name = list(ARM_JOINT_NAMES)
        msg.velocity = [0.0] * 14
        msg.effort = [0.0] * 14
        msg.position = target
        end = self.rospy.Time.now() + self.rospy.Duration(float(duration))
        rate = self.rospy.Rate(ARM_TRAJ_HZ)
        while not self.rospy.is_shutdown() and self.rospy.Time.now() < end:
            msg.header.stamp = self.rospy.Time.now()
            self.arm_traj_pub.publish(msg)
            rate.sleep()

    def send_single_arm_deg(self, arm: str, active_deg: Sequence[float],
                            duration: float = 1.0) -> None:
        target = list(self.last_arm_deg)
        if len(target) != 14:
            target = list(self.current_arm_deg())
        if len(target) != 14:
            target = list(HOME_ARM_DEG)
        values = [float(v) for v in active_deg]
        if len(values) != 7:
            raise ValueError("single arm target expects 7 joints")
        if arm == "left":
            target[:7] = values
        elif arm == "right":
            target[7:] = values
        else:
            raise ValueError("unknown arm: {}".format(arm))
        self.move_arm_deg(target, duration=duration, active_arm=arm)

    def single_arm_target_from_ik(self, arm: str, q_deg: Sequence[float]) -> List[float]:
        solved = [float(v) for v in q_deg]
        if len(solved) != 14:
            raise ValueError("IK target expects 14 joints, got {}".format(len(solved)))

        target = list(self.last_arm_deg)
        if len(target) != 14:
            target = list(self.current_arm_deg())
        if len(target) != 14:
            target = list(HOME_ARM_DEG)

        if arm == "left":
            active = slice(0, 7)
            inactive = range(7, 14)
        elif arm == "right":
            active = slice(7, 14)
            inactive = range(0, 7)
        else:
            raise ValueError("unknown arm: {}".format(arm))

        inactive_delta = max(abs(solved[i] - target[i]) for i in inactive)
        if inactive_delta > 2.0:
            log_stage(
                self.scene,
                "IK",
                "%s IK 返回中非目标手最大变化 %.1fdeg，已冻结非目标手，仅执行 %s 手",
                arm,
                inactive_delta,
                "左" if arm == "left" else "右",
            )
        target[active] = solved[active]
        return target

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
        z_min = SCENE_SAFE_MIN_Z_BASE_M.get(self.scene, SAFE_MIN_Z_BASE_M)
        z_max = SCENE_SAFE_MAX_Z_BASE_M.get(self.scene, SAFE_MAX_Z_BASE_M)
        x_max = reach_x_max_for_scene(self.scene)
        target = [
            clamp(float(target_xyz_base[0]), -0.05, x_max),
            float(target_xyz_base[1]),
            clamp(float(target_xyz_base[2]), z_min, z_max),
        ]
        quat = quat_down(yaw_rad if arm == "left" else yaw_rad + math.pi)
        try:
            q_deg = self.solve_ik(arm, target, quat_xyzw=quat, constraint_mode=2)
            q_deg = self.single_arm_target_from_ik(arm, q_deg)
            self.move_arm_deg(q_deg, duration=duration, active_arm=arm)
            return True
        except Exception as exc:
            log_warn(self.scene, "IK", "%s 失败，不执行近似关节动作，等待上层重新检测: %s target=%s",
                     label, exc, _fmt_xyz(target))
            return False


# ---------------------------------------------------------------------------
# 感知层：YOLO-World 检测、深度投影和 TF 变换
# ---------------------------------------------------------------------------


class RGBDPerceptionBase:
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
            log_error(self.scene, "感知", "无法导入 cv2/numpy，视觉识别不可用: %s", exc)
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
        raise NotImplementedError

    def detect_bins(self, bin_names: Optional[Sequence[str]] = None) -> List[PerceptionTarget]:
        raise NotImplementedError

    def _default_cameras_for_targets(self) -> List[str]:
        if self.scene == "scene3":
            return ["head", "left_wrist", "right_wrist"]
        return ["head", "left_wrist", "right_wrist"]

    def _mask_from_yolo(self, mask_data: Any, shape_hw: Tuple[int, int]) -> Any:
        mask = (mask_data > 0.5).astype(self.np.uint8) * 255
        if mask.shape[:2] != shape_hw:
            mask = self.cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=self.cv2.INTER_NEAREST)
        return mask

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

    def _find_contours(self, mask: Any) -> List[Any]:
        result = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        contours = result[0] if len(result) == 2 else result[1]
        return list(contours)

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


class YoloWorldPerception(RGBDPerceptionBase):
    def __init__(self, scene: str, base_frame: str,
                 model_path: str,
                 device: str,
                 conf: float,
                 imgsz: int,
                 class_names: Sequence[str],
                 class_aliases: Dict[str, str],
                 target_bins: Dict[str, Optional[str]],
                 world_prompts: Sequence[str],
                 bin_class_names: Sequence[str]):
        super().__init__(scene=scene, base_frame=base_frame)
        self.yolo_model_path = self._normalize_model_ref(model_path)
        self.yolo_device = device
        self.yolo_conf = float(conf)
        self.yolo_imgsz = int(imgsz)
        self.yolo_class_names = tuple(class_names)
        self.yolo_class_aliases = {
            self._normalize_yolo_name(alias): canonical
            for alias, canonical in class_aliases.items()
        }
        self.yolo_target_bins = dict(target_bins)
        self.yolo_world_prompts = tuple(world_prompts)
        self.yolo_bin_class_names = tuple(bin_class_names)
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
        wanted = self._wanted_yolo_classes(bin_names, allowed=self.yolo_bin_class_names)
        return self._detect_yolo_classes(
            class_names=wanted,
            cameras=["head"],
            max_targets=None,
            purpose="目标区",
        )

    def preload_model(self) -> bool:
        return self._ensure_yolo()

    def _detect_yolo_classes(self, class_names: Sequence[str], cameras: Sequence[str],
                             max_targets: Optional[int], purpose: str) -> List[PerceptionTarget]:
        if not class_names:
            return []
        if not self._ensure_yolo():
            reason = self._yolo_unavailable_reason or "unknown"
            self._report_yolo_unavailable("error", "YOLO-World 不可用: %s", reason)
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
        if (not os.path.isfile(self.yolo_model_path)
                and not self._is_ultralytics_world_model_name(self.yolo_model_path)):
            self._yolo_unavailable_reason = (
                "model not found: {}; generate a scene-specific YOLO-World .pt "
                "or pass yolov8s-worldv2.pt as --yolo-model"
            ).format(self.yolo_model_path)
            return False

        load_errors: List[str] = []
        if self._try_load_yolo_world(load_errors):
            return True

        self._yolo_unavailable_reason = "; ".join(load_errors) or "ultralytics/model load failed"
        self._yolo_model = None
        return False

    def _try_load_yolo_world(self, load_errors: List[str]) -> bool:
        try:
            from ultralytics import YOLOWorld

            model = YOLOWorld(self.yolo_model_path)
            prompts_applied = self._apply_world_prompts(model)
            self._yolo_model = model
            self._log_yolo_ready("YOLOWorld", prompts_applied=prompts_applied)
            return True
        except Exception as exc:
            load_errors.append("YOLOWorld load failed: {}".format(exc))
            self._yolo_model = None
            return False

    def _apply_world_prompts(self, model: Any) -> bool:
        if not hasattr(model, "set_classes"):
            return False
        model.set_classes(list(self.yolo_world_prompts))
        return True

    def _is_ultralytics_world_model_name(self, model_path: str) -> bool:
        basename = os.path.basename(model_path)
        return (
            basename == model_path
            and basename.lower().endswith(".pt")
            and "world" in basename.lower()
        )

    def _normalize_model_ref(self, model_path: str) -> str:
        expanded = os.path.expanduser(model_path)
        if os.path.isabs(expanded) or os.path.dirname(expanded):
            return os.path.abspath(expanded)
        return expanded

    def _log_yolo_ready(self, backend: str, prompts_applied: bool) -> None:
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
        source = "yolo_world"
        if mask is not None:
            source = "yolo_world_seg"
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


class YoloScene1Perception(YoloWorldPerception):
    def __init__(self, scene: str, base_frame: str = BASE_FRAME,
                 model_path: str = DEFAULT_SCENE1_YOLO_MODEL,
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
            class_names=SCENE1_YOLO_CLASS_NAMES,
            class_aliases=SCENE1_YOLO_CLASS_ALIASES,
            target_bins=SCENE1_YOLO_TARGET_BINS,
            world_prompts=SCENE1_YOLO_WORLD_PROMPTS,
            bin_class_names=("weighing_area", "drop_box"),
        )


class YoloScene2Perception(YoloWorldPerception):
    def __init__(self, scene: str, base_frame: str = BASE_FRAME,
                 model_path: str = DEFAULT_SCENE2_YOLO_MODEL,
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
            class_names=SCENE2_YOLO_CLASS_NAMES,
            class_aliases=SCENE2_YOLO_CLASS_ALIASES,
            target_bins=SCENE2_YOLO_TARGET_BINS,
            world_prompts=SCENE2_YOLO_WORLD_PROMPTS,
            bin_class_names=("sorting_bin_a", "sorting_bin_b", "sorting_bin_c"),
        )


class YoloScene3Perception(YoloWorldPerception):
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
            class_names=(SCENE3_YOLO_CLASS_NAME,),
            class_aliases=SCENE3_YOLO_CLASS_ALIASES,
            target_bins={SCENE3_YOLO_CLASS_NAME: SCENE3_YOLO_TARGET_BIN},
            world_prompts=SCENE3_YOLO_WORLD_PROMPTS,
            bin_class_names=(),
        )


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
            log_warn(self.scene, "选臂", "class=%s 无 base_link 坐标，默认使用右手", target.class_name)
            return "right"
        y_base = float(target.position_base[1])
        arm = "left" if y_base >= 0.0 else "right"
        log_stage(
            self.scene,
            "选臂",
            "class=%s base_y=%.3fm -> 目标在机器人%s侧，使用%s手",
            target.class_name,
            y_base,
            "左" if arm == "left" else "右",
            "左" if arm == "left" else "右",
        )
        return arm

    def single_arm(self, arm: Optional[str], target: Optional[PerceptionTarget],
                   purpose: str) -> str:
        if arm in ("left", "right"):
            return arm
        if arm == "both":
            chosen = self.choose_arm(target) if target is not None else "right"
            log_warn(
                self.scene,
                "选臂",
                "%s 收到 both，抓放动作不使用双手，改为%s手",
                purpose,
                "左" if chosen == "left" else "右",
            )
            return chosen
        if arm is None:
            return self.choose_arm(target) if target is not None else "right"
        log_warn(self.scene, "选臂", "%s 收到未知手臂 %s，默认使用右手", purpose, arm)
        return "right"

    def _target_debug_summary(self, target: PerceptionTarget) -> str:
        debug = target.debug or {}
        items = [
            "class={}".format(target.class_name),
            "conf={:.2f}".format(float(target.confidence)),
            "cam={}".format(target.camera),
            "uv={}".format(target.center_uv),
            "base={}".format(_fmt_xyz(target.position_base)),
            "cam_xyz={}".format(_fmt_xyz(target.position_camera)),
            "bbox={}".format(target.bbox),
            "area={:.0f}".format(float(target.area_px)),
            "bin={}".format(target.target_bin),
        ]
        for key in ("raw_class", "depth_m", "frame_id", "image_size"):
            if key in debug:
                items.append("{}={}".format(key, debug.get(key)))
        return " ".join(items)

    def _log_pick_failure_detail(self, target: PerceptionTarget, arm: str,
                                 step: str, reason: str,
                                 attempt_index: int = 0, attempt_total: int = 0,
                                 requested: Optional[Sequence[float]] = None,
                                 reachable: Optional[Sequence[float]] = None,
                                 pre: Optional[Sequence[float]] = None,
                                 grasp: Optional[Sequence[float]] = None,
                                 lift: Optional[Sequence[float]] = None,
                                 yaw_rad: float = 0.0) -> None:
        attempt = (
            "#{}/{}".format(attempt_index, attempt_total)
            if attempt_index > 0 and attempt_total > 0
            else "single"
        )
        log_warn(
            self.scene,
            "抓取失败详情",
            (
                "attempt=%s step=%s reason=%s arm=%s yaw=%.1fdeg "
                "requested=%s reachable=%s pre=%s grasp=%s lift=%s %s"
            ),
            attempt,
            step,
            reason,
            arm,
            math.degrees(yaw_rad),
            _fmt_xyz(requested),
            _fmt_xyz(reachable),
            _fmt_xyz(pre),
            _fmt_xyz(grasp),
            _fmt_xyz(lift),
            self._target_debug_summary(target),
        )

    def pick(self, target: PerceptionTarget, arm: Optional[str] = None,
             attempt_index: int = 0, attempt_total: int = 0) -> ActionResult:
        start = time.time()
        arm = self.single_arm(arm, target, "抓取")
        if target.position_base is None:
            reason = "target has no base_link position"
            log_warn(self.scene, "抓取", "%s class=%s", reason, target.class_name)
            self._log_pick_failure_detail(
                target,
                arm,
                "target_position",
                reason,
                attempt_index=attempt_index,
                attempt_total=attempt_total,
            )
            return ActionResult(False, "pick", reason, target, time.time() - start)

        requested_xyz = self._grasp_xyz(target)
        xyz = self._reachable_xyz(requested_xyz)
        yaw = float(target.yaw_rad or 0.0)
        scene3_tray = self.scene == "scene3" and target.class_name == "smt_tray"
        if scene3_tray:
            return self._pick_scene3_tray(target, arm, xyz, yaw, start)

        pre = (xyz[0], xyz[1], xyz[2] + SAFE_PRE_GRASP_DZ_M)
        grasp = (xyz[0], xyz[1], xyz[2])
        lift = (xyz[0], xyz[1], xyz[2] + SAFE_LIFT_DZ_M)

        log_stage(
            self.scene,
            "抓取",
            "class=%s arm=%s pre=%s grasp=%s yaw=%.1fdeg duration(pre/grasp/lift)=%.1f/%.1f/%.1fs settle=%.2fs",
            target.class_name,
            arm,
            _fmt_xyz(pre),
            _fmt_xyz(grasp),
            math.degrees(yaw),
            SKILL_PRE_GRASP_DURATION_SEC,
            SKILL_GRASP_DURATION_SEC,
            SKILL_LIFT_DURATION_SEC,
            SKILL_POST_GRASP_SETTLE_SEC,
        )
        self.navigation.fine_align_to_image_target(target)
        self.robot.open_claw(arm)
        pre_ok = self.robot.move_hand_ik(
            arm,
            pre,
            yaw_rad=yaw,
            duration=SKILL_PRE_GRASP_DURATION_SEC,
            label="pre_grasp",
        )
        if not pre_ok:
            reason = "pre_grasp IK failed"
            self._log_pick_failure_detail(
                target,
                arm,
                "pre_grasp",
                reason,
                attempt_index=attempt_index,
                attempt_total=attempt_total,
                requested=requested_xyz,
                reachable=xyz,
                pre=pre,
                grasp=grasp,
                lift=lift,
                yaw_rad=yaw,
            )
            return ActionResult(False, "pick", reason, target, time.time() - start)

        grasp_ok = self.robot.move_hand_ik(
            arm,
            grasp,
            yaw_rad=yaw,
            duration=SKILL_GRASP_DURATION_SEC,
            label="grasp",
        )
        if not grasp_ok:
            reason = "grasp IK failed"
            self._log_pick_failure_detail(
                target,
                arm,
                "grasp",
                reason,
                attempt_index=attempt_index,
                attempt_total=attempt_total,
                requested=requested_xyz,
                reachable=xyz,
                pre=pre,
                grasp=grasp,
                lift=lift,
                yaw_rad=yaw,
            )
            log_warn(self.scene, "抓取", "%s，不闭爪，等待重试", reason)
            return ActionResult(False, "pick", reason, target, time.time() - start)
        self.robot.close_claw(arm, percent=self._close_percent(target.class_name))
        self.robot.hold_arm(SKILL_POST_GRASP_SETTLE_SEC, label="post_grasp_settle")
        lift_targets = [lift]
        lift_ok = False
        for lift_attempt, lift_target in enumerate(lift_targets, start=1):
            log_stage(
                self.scene,
                "抓取",
                "lift attempt #%d/%d target=%s",
                lift_attempt,
                len(lift_targets),
                _fmt_xyz(lift_target),
            )
            lift_ok = self.robot.move_hand_ik(
                arm,
                lift_target,
                yaw_rad=yaw,
                duration=SKILL_LIFT_DURATION_SEC,
                label="lift#{}".format(lift_attempt),
            )
            if lift_ok:
                break
        if not lift_ok:
            reason = "lift IK failed"
            self._log_pick_failure_detail(
                target,
                arm,
                "lift",
                reason,
                attempt_index=attempt_index,
                attempt_total=attempt_total,
                requested=requested_xyz,
                reachable=xyz,
                pre=pre,
                grasp=grasp,
                lift=lift,
                yaw_rad=yaw,
            )
            return ActionResult(False, "pick", reason, target, time.time() - start)

        return ActionResult(True, "pick", "", target, time.time() - start)

    def _pick_scene3_tray(self, target: PerceptionTarget, arm: str,
                          xyz: Sequence[float], yaw: float, start: float) -> ActionResult:
        tray_center_z = float(target.position_base[2]) if target.position_base is not None else float(xyz[2])
        shelf = scene3_shelf_name_from_z(tray_center_z)
        side_sign = arm_side_sign(arm)
        entry_x = float(xyz[0]) - SCENE3_SHELF_ENTRY_BACKOFF_X_M
        side_y = float(xyz[1]) + side_sign * SCENE3_SHELF_SIDE_CLEARANCE_Y_M
        level_z = tray_center_z

        side_ready = self._reachable_xyz((entry_x, side_y, level_z))
        shelf_entry = self._reachable_xyz((entry_x, xyz[1], level_z))
        insert_pre = self._reachable_xyz((xyz[0], xyz[1], level_z))
        grasp = self._reachable_xyz(xyz)
        retract = self._reachable_xyz((entry_x, xyz[1], xyz[2]))

        log_stage(
            self.scene,
            "抓取",
            (
                "货架路径 shelf=%s arm=%s level_z=%.3fm "
                "side=%s entry=%s insert=%s grasp=%s retract=%s"
            ),
            shelf,
            arm,
            level_z,
            _fmt_xyz(side_ready),
            _fmt_xyz(shelf_entry),
            _fmt_xyz(insert_pre),
            _fmt_xyz(grasp),
            _fmt_xyz(retract),
        )
        log_stage(
            self.scene,
            "抓取",
            "先侧方平举，再水平插入；闭爪后先水平抽出，离开货架后再抬升",
        )

        log_stage(
            self.scene,
            "抓取",
            "抓取前不再触发底盘 fine_align，停车稳定 %.1fs",
            SCENE3_PRE_PICK_BASE_SETTLE_SEC,
        )
        self.robot.stop_base(duration=SCENE3_PRE_PICK_BASE_SETTLE_SEC)
        self.robot.set_claw(arm, CLAW_OPEN_PERCENT, settle_sec=SCENE3_CLAW_OPEN_SETTLE_SEC)

        waypoints = [
            ("side_ready", side_ready, SCENE3_SHELF_SIDE_DURATION_SEC),
            ("shelf_entry", shelf_entry, SCENE3_SHELF_ENTRY_DURATION_SEC),
            ("shelf_insert", insert_pre, SCENE3_SHELF_INSERT_DURATION_SEC),
            ("grasp", grasp, SCENE3_GRASP_DURATION_SEC),
        ]
        for label, waypoint, duration in waypoints:
            ok = self.robot.move_hand_ik(arm, waypoint, yaw_rad=yaw, duration=duration, label=label)
            if not ok:
                reason = "{} IK failed".format(label)
                log_warn(self.scene, "抓取", "%s，不闭爪，重新检测后再试", reason)
                return ActionResult(False, "pick", reason, target, time.time() - start)

        self.robot.hold_arm(SCENE3_GRASP_DWELL_SEC, label="tray_grasp_dwell")
        self.robot.set_claw(
            arm,
            self._close_percent(target.class_name),
            settle_sec=0.0,
        )
        self.robot.hold_arm(SCENE3_CLAW_CLOSE_SETTLE_SEC, label="tray_claw_close")
        self.robot.hold_arm(SCENE3_POST_CLOSE_SETTLE_SEC, label="tray_post_close")

        retract_ok = self.robot.move_hand_ik(
            arm,
            retract,
            yaw_rad=yaw,
            duration=SCENE3_SHELF_RETRACT_DURATION_SEC,
            label="shelf_retract",
        )
        if not retract_ok:
            reason = "shelf retract IK failed"
            log_warn(self.scene, "抓取", "%s，料盘可能仍在货架内", reason)
            return ActionResult(False, "pick", reason, target, time.time() - start)

        lift_targets = [
            (
                retract[0] - SCENE3_LIFT_RETRY_BACKOFF_X_M * index,
                retract[1],
                retract[2] + SAFE_LIFT_DZ_M,
            )
            for index in range(SCENE3_LIFT_RETRY)
        ]
        lift_ok = False
        for lift_attempt, lift_target in enumerate(lift_targets, start=1):
            log_stage(
                self.scene,
                "抓取",
                "离架后抬升 #%d/%d target=%s",
                lift_attempt,
                len(lift_targets),
                _fmt_xyz(lift_target),
            )
            lift_ok = self.robot.move_hand_ik(
                arm,
                lift_target,
                yaw_rad=yaw,
                duration=SCENE3_LIFT_DURATION_SEC,
                label="lift#{}".format(lift_attempt),
            )
            if lift_ok:
                self.robot.hold_arm(SCENE3_LIFT_SETTLE_SEC, label="tray_lift_settle")
                break
        if not lift_ok:
            reason = "lift IK failed"
            log_warn(self.scene, "抓取", "%s，已尝试 %d 次离架后抬升", reason, len(lift_targets))
            return ActionResult(False, "pick", reason, target, time.time() - start)

        return ActionResult(True, "pick", "", target, time.time() - start)

    def pick_with_retry(
        self,
        target: PerceptionTarget,
        arm: Optional[str] = None,
        relocalize: Optional[Callable[[PerceptionTarget], Optional[PerceptionTarget]]] = None,
    ) -> ActionResult:
        start = time.time()
        arm = self.single_arm(arm, target, "抓取重试")
        # 场景三控制器已经有按“重新检测 -> 腕部复检 -> 抓取”的 40 次外层循环，
        # 这里保持单次，避免每轮料盘抓取被嵌套放大到数百次。
        max_attempts = 1 if self.scene == "scene3" else SKILL_PICK_RETRY
        last_result = ActionResult(False, "pick", "not attempted", target)
        for attempt in range(1, max_attempts + 1):
            if max_attempts > 1:
                log_stage(
                    self.scene,
                    "抓取",
                    "尝试 #%d/%d class=%s arm=%s target=%s",
                    attempt,
                    max_attempts,
                    target.class_name,
                    arm,
                    _fmt_xyz(target.position_base),
                )
            result = self.pick(
                target,
                arm=arm,
                attempt_index=attempt,
                attempt_total=max_attempts,
            )
            result.retry_count = attempt - 1
            if result.ok:
                if attempt > 1:
                    log_stage(
                        self.scene,
                        "抓取",
                        "第 %d/%d 次抓取成功 class=%s arm=%s",
                        attempt,
                        max_attempts,
                        target.class_name,
                        arm,
                    )
                return result

            last_result = result
            log_warn(
                self.scene,
                "抓取",
                "尝试 #%d/%d 失败: %s",
                attempt,
                max_attempts,
                result.reason,
            )
            if attempt < max_attempts:
                self.robot.open_claw(arm)
                self.robot.rospy.sleep(SKILL_PICK_RETRY_SETTLE_SEC)
                if relocalize is not None:
                    log_stage(
                        self.scene,
                        "重新定位",
                        "抓取失败后重新 YOLO 定位 class=%s old=%s",
                        target.class_name,
                        _fmt_xyz(target.position_base),
                    )
                    refreshed = relocalize(target)
                    if refreshed is not None and refreshed.position_base is not None:
                        distance = (
                            distance3(target.position_base, refreshed.position_base)
                            if target.position_base is not None
                            else 0.0
                        )
                        log_stage(
                            self.scene,
                            "重新定位",
                            "使用新目标 old=%s new=%s delta=%.3fm conf=%.2f",
                            _fmt_xyz(target.position_base),
                            _fmt_xyz(refreshed.position_base),
                            distance,
                            refreshed.confidence,
                        )
                        target = refreshed
                    else:
                        log_warn(self.scene, "重新定位", "未获得可用新坐标，下一次沿用旧目标")
                    self.robot.rospy.sleep(SKILL_RELOCALIZE_SETTLE_SEC)

        reason = "pick failed after {} attempts; last: {}".format(
            max_attempts,
            last_result.reason,
        )
        return ActionResult(
            False,
            "pick",
            reason,
            target,
            time.time() - start,
            max_attempts - 1,
        )

    def _place_candidates(self, place_xyz_base: Sequence[float], arm: str,
                          label: str) -> List[Tuple[float, float, float]]:
        base = self._reachable_xyz(place_xyz_base)
        side = arm_side_sign(arm)
        if "weigh" in label:
            offsets = (
                (0.00, 0.00, 0.00),
                (0.00, side * 0.04, 0.00),
                (-0.04, side * 0.04, 0.00),
                (0.04, side * 0.04, 0.00),
                (0.00, side * 0.07, 0.00),
            )
        else:
            offsets = (
                (0.00, 0.00, 0.00),
                (0.00, side * 0.08, 0.00),
                (-0.06, side * 0.10, 0.00),
                (-0.10, side * 0.14, 0.00),
                (0.04, side * 0.08, 0.00),
            )

        candidates: List[Tuple[float, float, float]] = []
        seen = set()
        for dx, dy, dz in offsets:
            candidate = self._reachable_xyz((base[0] + dx, base[1] + dy, base[2] + dz))
            key = tuple(round(value, 3) for value in candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    def _log_place_failure_detail(self, arm: str, label: str, step: str, reason: str,
                                  candidate_index: int, candidate_total: int,
                                  place: Sequence[float],
                                  pre: Sequence[float],
                                  retreat: Sequence[float]) -> None:
        log_warn(
            self.scene,
            "放置失败详情",
            (
                "label=%s candidate=#%d/%d step=%s reason=%s arm=%s "
                "place=%s pre=%s retreat=%s"
            ),
            label,
            candidate_index,
            candidate_total,
            step,
            reason,
            arm,
            _fmt_xyz(place),
            _fmt_xyz(pre),
            _fmt_xyz(retreat),
        )

    def place(self, arm: str, place_xyz_base: Sequence[float],
              label: str = "place") -> ActionResult:
        start = time.time()
        arm = self.single_arm(arm, None, "放置")
        candidates = self._place_candidates(place_xyz_base, arm, label)
        last_reason = "no place candidate"
        for index, place in enumerate(candidates, start=1):
            pre = (place[0], place[1], place[2] + SAFE_PLACE_DZ_M)
            retreat = (place[0], place[1], place[2] + SAFE_LIFT_DZ_M)
            log_stage(
                self.scene,
                "放置",
                "arm=%s label=%s candidate=#%d/%d place=%s pre=%s",
                arm,
                label,
                index,
                len(candidates),
                _fmt_xyz(place),
                _fmt_xyz(pre),
            )
            pre_ok = self.robot.move_hand_ik(
                arm,
                pre,
                duration=SKILL_PLACE_PRE_DURATION_SEC,
                label=label + "_pre#{}".format(index),
            )
            if not pre_ok:
                last_reason = "place pre IK failed"
                self._log_place_failure_detail(
                    arm, label, "pre", last_reason, index, len(candidates), place, pre, retreat
                )
                continue

            place_ok = self.robot.move_hand_ik(
                arm,
                place,
                duration=SKILL_PLACE_DURATION_SEC,
                label=label + "#{}".format(index),
            )
            if not place_ok:
                last_reason = "place IK failed"
                self._log_place_failure_detail(
                    arm, label, "place", last_reason, index, len(candidates), place, pre, retreat
                )
                continue

            self.robot.open_claw(arm)
            retreat_ok = self.robot.move_hand_ik(
                arm,
                retreat,
                duration=SKILL_PLACE_RETREAT_DURATION_SEC,
                label=label + "_retreat#{}".format(index),
            )
            if not retreat_ok:
                log_warn(
                    self.scene,
                    "放置",
                    "%s 已松爪，但 retreat IK failed，按已放置继续",
                    label,
                )
            return ActionResult(True, "place", "", None, time.time() - start)

        reason = "{} after {} candidates".format(last_reason, len(candidates))
        log_warn(self.scene, "放置", "label=%s 失败: %s", label, reason)
        return ActionResult(False, "place", reason, None, time.time() - start)

    def pick_and_place(
        self,
        target: PerceptionTarget,
        place_xyz_base: Sequence[float],
        arm: Optional[str] = None,
        label: str = "pick_place",
        relocalize: Optional[Callable[[PerceptionTarget], Optional[PerceptionTarget]]] = None,
    ) -> ActionResult:
        arm = self.single_arm(arm, target, "抓放")
        pick_result = self.pick_with_retry(target, arm=arm, relocalize=relocalize)
        if not pick_result.ok:
            return pick_result
        place_result = self.place(arm, place_xyz_base, label=label)
        return ActionResult(
            place_result.ok,
            "pick_place",
            place_result.reason,
            pick_result.target or target,
            pick_result.duration_sec + place_result.duration_sec,
            pick_result.retry_count,
        )

    def _reachable_xyz(self, xyz: Sequence[float]) -> Tuple[float, float, float]:
        z_min = SCENE_SAFE_MIN_Z_BASE_M.get(self.scene, SAFE_MIN_Z_BASE_M)
        z_max = SCENE_SAFE_MAX_Z_BASE_M.get(self.scene, SAFE_MAX_Z_BASE_M)
        x_max = reach_x_max_for_scene(self.scene)
        return (
            clamp(float(xyz[0]), -0.05, x_max),
            float(xyz[1]),
            clamp(float(xyz[2]), z_min, z_max),
        )

    def _grasp_xyz(self, target: PerceptionTarget) -> Tuple[float, float, float]:
        assert target.position_base is not None
        xyz = list(target.position_base)
        if self.scene == "scene2":
            offset = SCENE2_GRASP_OFFSETS_BASE.get(target.class_name, (0.0, 0.0, 0.0))
            xyz = [float(xyz[i]) + float(offset[i]) for i in range(3)]
        elif self.scene == "scene3" and target.class_name == "smt_tray":
            xyz = [float(xyz[i]) + float(SCENE3_TRAY_GRASP_OFFSET_BASE[i]) for i in range(3)]
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    def _close_percent(self, class_name: str) -> float:
        if class_name == "smt_tray":
            return SCENE3_CLAW_CLOSE_PERCENT
        if class_name == "part_type_c":
            return 82.0
        return CLAW_CLOSE_PERCENT


# ---------------------------------------------------------------------------
# 场景任务控制层
# ---------------------------------------------------------------------------


class BaseSceneController:
    def __init__(self, scene: str, robot: RobotBase, perception: RGBDPerceptionBase,
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

    def preload_yolo_model(self) -> None:
        preload = getattr(self.perception, "preload_model", None)
        if not callable(preload):
            return
        log_stage(self.scene, "YOLO", "预加载 YOLO-World 模型，避免首次检测时阻塞导航")
        if not preload():
            log_warn(self.scene, "YOLO", "模型预加载失败，后续检测会再次尝试并输出原因")

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

    def relocalize_pick_target(self, target: PerceptionTarget) -> Optional[PerceptionTarget]:
        log_stage(
            self.scene,
            "重新定位",
            "按同类别重新检测 class=%s old=%s",
            target.class_name,
            _fmt_xyz(target.position_base),
        )
        candidates = self.detect_targets(class_names=[target.class_name], max_targets=None)
        candidates = [
            item for item in candidates
            if item.class_name == target.class_name and item.position_base is not None
        ]
        if not candidates:
            log_warn(self.scene, "重新定位", "class=%s 没有可用 YOLO-World 坐标", target.class_name)
            return None

        if target.position_base is None:
            best = sorted(candidates, key=lambda item: item.confidence, reverse=True)[0]
            log_stage(
                self.scene,
                "重新定位",
                "旧目标无坐标，使用最高置信度候选 new=%s conf=%.2f",
                _fmt_xyz(best.position_base),
                best.confidence,
            )
            return best

        best = sorted(
            candidates,
            key=lambda item: (
                distance3(target.position_base, item.position_base)
                if item.position_base is not None
                else float("inf"),
                -float(item.confidence),
            ),
        )[0]
        assert best.position_base is not None
        log_stage(
            self.scene,
            "重新定位",
            "候选=%d 最近同类 new=%s delta=%.3fm conf=%.2f",
            len(candidates),
            _fmt_xyz(best.position_base),
            distance3(target.position_base, best.position_base),
            best.confidence,
        )
        return best

    def resolve_place(self, name: str) -> Optional[Tuple[float, float, float]]:
        detected = self.perception.detect_bins(bin_names=[name])
        detected = [item for item in detected if item.position_base is not None]
        if detected:
            best = sorted(detected, key=lambda item: item.confidence, reverse=True)[0]
            xyz = best.position_base
            assert xyz is not None
            place = (float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.08)
            log_stage(self.scene, "目标区", "%s 使用 YOLO-World 检测位置 %s", name, _fmt_xyz(place))
            return place

        log_error(self.scene, "目标区", "%s 未被 YOLO-World 检测到，不使用规则或几何兜底", name)
        return None

    def make_target_at(self, class_name: str, xyz_base: Sequence[float],
                       target_bin: Optional[str] = None) -> PerceptionTarget:
        return PerceptionTarget(
            class_name=class_name,
            confidence=0.15,
            center_uv=None,
            position_base=(float(xyz_base[0]), float(xyz_base[1]), float(xyz_base[2])),
            position_camera=None,
            camera="task_regrasp",
            area_px=0.0,
            target_bin=target_bin,
            timestamp=time.time(),
            debug={"source": "known_place_regrasp"},
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

        targets = self.detect_scene1_targets_with_retry(
            class_names=["parcel"],
            max_targets=SCENE_TARGET_LIMIT["scene1"],
            label="parcel",
        )
        if not targets:
            reason = "no parcel detected"
            log_error(self.scene, "失败", reason)
            return ActionResult(False, "scene1", reason, duration_sec=time.time() - start)

        targets.sort(key=lambda t: (t.position_base[1], t.position_base[0]) if t.position_base else (0.0, 0.0))
        for index, parcel in enumerate(targets, start=1):
            log_stage(self.scene, "流程", "快递 #%02d 抓取 -> 称重 -> 入箱", index)
            arm = self.skills.choose_arm(parcel)
            result = self.skills.pick_and_place(
                parcel,
                weigh_xyz,
                arm=arm,
                label="parcel{}_to_weigh".format(index),
                relocalize=self.relocalize_pick_target,
            )
            if not result.ok:
                log_warn(self.scene, "失败", "快递 #%02d 称重放置失败: %s", index, result.reason)
                continue
            self.robot.rospy.sleep(0.7)
            regrasp_xyz = (weigh_xyz[0], weigh_xyz[1], weigh_xyz[2] - 0.04)
            weighed = self.make_target_at("parcel", regrasp_xyz, target_bin="drop_box")
            refreshed_weighed = self.relocalize_pick_target(weighed)
            if refreshed_weighed is not None:
                log_stage(
                    self.scene,
                    "重新定位",
                    "称重后使用 YOLO-World 更新二次抓取坐标 %s",
                    _fmt_xyz(refreshed_weighed.position_base),
                )
                weighed = refreshed_weighed
            else:
                log_warn(self.scene, "重新定位", "称重后未检测到包裹，沿用称重区估计坐标")
            result = self.skills.pick_and_place(
                weighed,
                drop_xyz,
                arm=arm,
                label="parcel{}_to_box".format(index),
                relocalize=self.relocalize_pick_target,
            )
            if not result.ok:
                log_warn(self.scene, "失败", "快递 #%02d 入箱失败: %s", index, result.reason)

        return ActionResult(True, "scene1", "done", duration_sec=time.time() - start)

    def detect_scene1_targets_with_retry(self, class_names: Sequence[str],
                                         max_targets: Optional[int],
                                         label: str) -> List[PerceptionTarget]:
        best: List[PerceptionTarget] = []
        for attempt in range(1, SCENE1_YOLO_RETRY + 1):
            targets = self.detect_targets(class_names=class_names, max_targets=max_targets)
            if len(targets) > len(best):
                best = list(targets)
            if targets:
                if attempt > 1:
                    log_stage(self.scene, "检测", "%s 第 %d 次 YOLO-World 检测成功", label, attempt)
                return targets
            log_warn(
                self.scene,
                "检测",
                "%s 第 %d/%d 次未检测到，等待后重试",
                label,
                attempt,
                SCENE1_YOLO_RETRY,
            )
            self.robot.rospy.sleep(SCENE1_YOLO_RETRY_SETTLE_SEC)
        return best

    def resolve_scene1_yolo_place(self, name: str) -> Optional[Tuple[float, float, float]]:
        best_seen: Optional[PerceptionTarget] = None
        for attempt in range(1, SCENE1_YOLO_RETRY + 1):
            detected = self.perception.detect_bins(bin_names=[name])
            if detected:
                best_seen = sorted(detected, key=lambda item: item.confidence, reverse=True)[0]
            usable = [item for item in detected if item.position_base is not None]
            if usable:
                best = sorted(usable, key=lambda item: item.confidence, reverse=True)[0]
                xyz = best.position_base
                assert xyz is not None
                place = (float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.08)
                log_stage(
                    self.scene,
                    "目标区",
                    "%s 第 %d 次使用 YOLO-World 检测位置 %s",
                    name,
                    attempt,
                    _fmt_xyz(place),
                )
                return place

            if best_seen is not None:
                log_warn(
                    self.scene,
                    "目标区",
                    "%s 第 %d/%d 次有候选但缺少 base_link 坐标 depth=%s frame=%s，等待后重试",
                    name,
                    attempt,
                    SCENE1_YOLO_RETRY,
                    best_seen.debug.get("depth_m") if best_seen.debug else None,
                    best_seen.debug.get("frame_id") if best_seen.debug else None,
                )
            else:
                log_warn(
                    self.scene,
                    "目标区",
                    "%s 第 %d/%d 次未被 YOLO-World 检测到，等待后重试",
                    name,
                    attempt,
                    SCENE1_YOLO_RETRY,
                )
            self.robot.rospy.sleep(SCENE1_YOLO_RETRY_SETTLE_SEC)

        log_error(self.scene, "目标区", "%s 未被 YOLO-World 检测到，场景一不再使用规则或几何兜底", name)
        return None


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
        targets = self._stable_detect_parts()
        if not targets:
            reason = "no scene2 part detected by YOLO-World"
            log_error(self.scene, "失败", reason)
            return ActionResult(False, "scene2", reason, duration_sec=time.time() - start)

        required_bins = sorted({self.PART_TO_BIN[target.class_name] for target in targets
                                if target.class_name in self.PART_TO_BIN})
        bin_places: Dict[str, Tuple[float, float, float]] = {}
        for name in required_bins:
            place = self.resolve_place(name)
            if place is None:
                reason = "scene2 YOLO-World place target missing: {}".format(name)
                log_error(self.scene, "失败", reason)
                return ActionResult(False, "scene2", reason, duration_sec=time.time() - start)
            bin_places[name] = place

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
                relocalize=self.relocalize_pick_target,
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
        completed_positions: List[Tuple[float, float, float]] = []
        max_trays = SCENE_TARGET_LIMIT["scene3"]
        for index in range(1, max_trays + 1):
            if not moved_to_rack:
                self.navigation.drive_for(
                    linear_x=SCENE3_APPROACH_SPEED_MPS,
                    duration=SCENE3_INITIAL_APPROACH_DURATION_SEC,
                    label="approach_smt_rack",
                )
                moved_to_rack = True
                self.robot.rospy.sleep(SCENE3_APPROACH_SETTLE_SEC)

            tray: Optional[PerceptionTarget] = None
            arm = "right"
            pick_result: Optional[ActionResult] = None
            for pick_attempt in range(1, SCENE3_PICK_RETRY + 1):
                targets = self.detect_reachable_trays(completed_positions)
                if not targets:
                    log_warn(self.scene, "检测", "未检测到更多料盘，结束出库循环")
                    break
                targets.sort(key=self._tray_sort_key)
                tray_candidate = targets[0]
                arm = self.skills.choose_arm(tray_candidate)
                refined = self.refine_tray_with_wrist(tray_candidate, arm)
                if refined is None:
                    log_warn(
                        self.scene,
                        "抓取",
                        "料盘 #%02d 尝试 #%d/%d 腕部复检失败，重新检测",
                        index,
                        pick_attempt,
                        SCENE3_PICK_RETRY,
                    )
                    self.robot.set_arm_mode(ARM_MODE_AUTO_SWING, warn_only=True)
                    self.robot.rospy.sleep(SCENE3_PICK_RETRY_SETTLE_SEC)
                    continue
                tray = refined
                shelf = self._tray_shelf(tray)
                log_stage(
                    self.scene,
                    "流程",
                    "料盘 #%02d 尝试 #%d/%d shelf=%s 出库，arm=%s base=%s",
                    index,
                    pick_attempt,
                    SCENE3_PICK_RETRY,
                    shelf,
                    arm,
                    _fmt_xyz(tray.position_base),
                )
                pick_result = self.skills.pick(tray, arm=arm)
                pick_result.retry_count = pick_attempt - 1
                if pick_result.ok:
                    break
                log_warn(
                    self.scene,
                    "抓取",
                    "料盘 #%02d 尝试 #%d/%d 失败: %s，重新检测后再试",
                    index,
                    pick_attempt,
                    SCENE3_PICK_RETRY,
                    pick_result.reason,
                )
                self.robot.open_claw(arm)
                self.robot.set_arm_mode(ARM_MODE_AUTO_SWING, warn_only=True)
                self.robot.rospy.sleep(SCENE3_PICK_RETRY_SETTLE_SEC)

            if pick_result is None or not pick_result.ok or tray is None:
                reason = pick_result.reason if pick_result is not None else "no target after retry"
                log_warn(self.scene, "失败", "料盘 #%02d 抓取失败: %s", index, reason)
                break

            self.navigation.drive_for(linear_x=-0.16, duration=3.4, label="return_to_outbound_box")
            moved_to_rack = False
            place_result = self.skills.place(arm, outbound_xyz, label="tray{}_outbound".format(index))
            if not place_result.ok:
                log_warn(self.scene, "失败", "料盘 #%02d 放置失败: %s", index, place_result.reason)
                break
            if tray.position_base is not None:
                completed_positions.append(tray.position_base)
            completed += 1

        if completed == 0:
            return ActionResult(False, "scene3", "no tray moved", duration_sec=time.time() - start)
        return ActionResult(True, "scene3", "moved {} tray(s)".format(completed),
                            duration_sec=time.time() - start)

    def refine_tray_with_wrist(self, tray: PerceptionTarget, arm: str) -> Optional[PerceptionTarget]:
        if tray.position_base is None:
            return None
        camera = "{}_wrist".format(arm)
        xyz = tray.position_base
        observe_z = float(xyz[2])
        side_sign = arm_side_sign(arm)
        entry_x = float(xyz[0]) - SCENE3_WRIST_OBSERVE_BACKOFF_X_M
        side_observe_xyz = (
            entry_x,
            float(xyz[1]) + side_sign * SCENE3_SHELF_SIDE_CLEARANCE_Y_M,
            observe_z,
        )
        observe_xyz = (
            entry_x,
            float(xyz[1]),
            observe_z,
        )
        log_stage(
            self.scene,
            "腕部复检",
            "arm=%s camera=%s rough=%s side=%s observe=%s",
            arm,
            camera,
            _fmt_xyz(xyz),
            _fmt_xyz(side_observe_xyz),
            _fmt_xyz(observe_xyz),
        )
        self.robot.open_claw(arm)
        side_ok = self.robot.move_hand_ik(
            arm,
            side_observe_xyz,
            duration=SCENE3_SHELF_SIDE_DURATION_SEC,
            label="wrist_refine_side",
        )
        if not side_ok:
            log_warn(self.scene, "腕部复检", "侧方观察位 IK 失败，跳过腕部复检，沿用头部相机坐标 %s", _fmt_xyz(tray.position_base))
            return tray
        observe_ok = self.robot.move_hand_ik(
            arm,
            observe_xyz,
            duration=SCENE3_SHELF_ENTRY_DURATION_SEC,
            label="wrist_refine_view",
        )
        if not observe_ok:
            log_warn(self.scene, "腕部复检", "观察位 IK 失败，跳过腕部复检，沿用头部相机坐标 %s", _fmt_xyz(tray.position_base))
            return tray
        self.robot.rospy.sleep(0.25)

        for attempt in range(1, SCENE3_WRIST_REFINE_RETRY + 1):
            candidates = self.perception.detect_targets(
                class_names=["smt_tray"],
                cameras=[camera],
                max_targets=None,
            )
            log_target_list(self.scene, "腕部复检", candidates)
            usable: List[PerceptionTarget] = []
            rejected: List[Tuple[PerceptionTarget, str]] = []
            for candidate in candidates:
                reason = self._target_reject_reason(candidate)
                if reason is not None:
                    rejected.append((candidate, reason))
                    continue
                if (candidate.position_base is not None
                        and distance3(candidate.position_base, tray.position_base) <= SCENE3_WRIST_REFINE_MAX_DELTA_M):
                    usable.append(candidate)
                else:
                    rejected.append((candidate, "too far from rough target"))
            self._log_rejected_targets(rejected)
            if usable:
                usable.sort(key=lambda item: (
                    distance3(item.position_base, tray.position_base) if item.position_base is not None else 99.0,
                    -float(item.confidence),
                ))
                refined = usable[0]
                log_stage(
                    self.scene,
                    "腕部复检",
                    "使用腕部相机修正 base %s -> %s",
                    _fmt_xyz(tray.position_base),
                    _fmt_xyz(refined.position_base),
                )
                return refined
            self.robot.rospy.sleep(0.25)

        log_warn(self.scene, "腕部复检", "未获得可用腕部复检结果，沿用头部相机坐标 %s", _fmt_xyz(tray.position_base))
        return tray

    def detect_reachable_trays(self, completed_positions: Sequence[Tuple[float, float, float]]) -> List[PerceptionTarget]:
        miss_count = 0
        attempt = 0
        approach_count = 0
        while not self.robot.rospy.is_shutdown() and attempt < SCENE3_APPROACH_RETRY:
            attempt += 1
            raw_targets = self.perception.detect_targets(
                class_names=["smt_tray"],
                cameras=["head"],
                max_targets=None,
            )
            raw_targets = [
                target for target in raw_targets
                if not self._is_near_completed_position(target, completed_positions)
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
                miss_count += 1
                if miss_count > SCENE3_APPROACH_MAX_MISSES:
                    log_warn(
                        self.scene,
                        "检测",
                        "连续 %d 次没有可靠近候选，结束出库循环",
                        miss_count,
                    )
                    return []
                approach_count += 1
                log_stage(
                    self.scene,
                    "导航",
                    "检测尝试 #%d 未得到可抓/远处料盘候选，接近尝试 #%d 短步靠近 %.2fs 后继续检测",
                    attempt,
                    approach_count,
                    SCENE3_APPROACH_SEARCH_DURATION_SEC,
                )
                self.navigation.drive_for(
                    linear_x=SCENE3_APPROACH_SPEED_MPS,
                    duration=SCENE3_APPROACH_SEARCH_DURATION_SEC,
                    label="approach_search_tray",
                )
                self.robot.rospy.sleep(SCENE3_APPROACH_SETTLE_SEC)
                attempt = 0
                log_stage(self.scene, "导航", "接近尝试 #%d 完成，重置检测尝试次数", approach_count)
                continue

            miss_count = 0
            far_targets.sort(key=lambda item: float(item.position_base[0]) if item.position_base else 99.0)
            target = far_targets[0]
            assert target.position_base is not None
            x = float(target.position_base[0])
            duration = clamp(
                (x - SCENE3_APPROACH_TARGET_X_M) / SCENE3_APPROACH_SPEED_MPS,
                SCENE3_APPROACH_MIN_DURATION_SEC,
                SCENE3_APPROACH_MAX_DURATION_SEC,
            )
            approach_count += 1
            log_stage(
                self.scene,
                "导航",
                "检测尝试 #%d 发现料盘仍偏远 base=%s，接近尝试 #%d 预估靠近 %.2fs 后重试",
                attempt,
                _fmt_xyz(target.position_base),
                approach_count,
                duration,
            )
            self.navigation.drive_for(
                linear_x=SCENE3_APPROACH_SPEED_MPS,
                duration=duration,
                label="approach_visible_tray",
            )
            self.robot.rospy.sleep(SCENE3_APPROACH_SETTLE_SEC)
            attempt = 0
            log_stage(self.scene, "导航", "接近尝试 #%d 完成，重置检测尝试次数", approach_count)
        return []

    def _target_is_far_but_approachable(self, target: PerceptionTarget, reason: str) -> bool:
        if target.position_base is None or "x out of range" not in reason:
            return False
        x = float(target.position_base[0])
        return TARGET_BOUNDS_BASE["scene3"]["x_max"] < x <= SCENE3_APPROACH_MAX_DETECT_X_M

    def _is_near_completed_position(self, target: PerceptionTarget,
                                    completed_positions: Sequence[Tuple[float, float, float]]) -> bool:
        if target.position_base is None:
            return False
        for position in completed_positions:
            if distance3(target.position_base, position) < 0.18:
                log_warn(
                    self.scene,
                    "检测过滤",
                    "跳过已完成位置附近候选 base=%s completed=%s",
                    _fmt_xyz(target.position_base),
                    _fmt_xyz(position),
                )
                return True
        return False

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


def build_controller(scene: str, robot: RobotBase, perception: RGBDPerceptionBase,
                     navigation: Navigation, skills: Skills) -> BaseSceneController:
    if scene == "scene1":
        return Scene1Controller(scene, robot, perception, navigation, skills)
    if scene == "scene2":
        return Scene2Controller(scene, robot, perception, navigation, skills)
    if scene == "scene3":
        return Scene3Controller(scene, robot, perception, navigation, skills)
    raise ValueError("unknown scene: {}".format(scene))


def default_yolo_model_for_scene(scene: str) -> str:
    if scene == "scene1":
        return DEFAULT_SCENE1_YOLO_MODEL
    if scene == "scene2":
        return DEFAULT_SCENE2_YOLO_MODEL
    if scene == "scene3":
        return DEFAULT_SCENE3_YOLO_MODEL
    raise ValueError("unknown scene: {}".format(scene))


def build_perception(scene: str, scene1_perception: str,
                     yolo_model: Optional[str], yolo_device: str,
                     yolo_conf: float, yolo_imgsz: int) -> RGBDPerceptionBase:
    if scene == "scene1":
        if scene1_perception == "rule":
            raise ValueError("scene1 rule-based perception has been removed; use YOLO-World classes parcel/weighing_area/drop_box")
        return YoloScene1Perception(
            scene=scene,
            base_frame=BASE_FRAME,
            model_path=yolo_model or default_yolo_model_for_scene(scene),
            device=yolo_device,
            conf=yolo_conf,
            imgsz=yolo_imgsz,
        )
    if scene == "scene2":
        return YoloScene2Perception(
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
    raise ValueError("unknown scene: {}".format(scene))


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
    preload_yolo = getattr(controller, "preload_yolo_model", None)
    if callable(preload_yolo):
        preload_yolo()

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
                        help="兼容旧参数；auto/yolo 均使用 YOLO-World，不再支持规则识别")
    parser.add_argument("--yolo-model", default=None,
                        help="YOLO-World .pt 模型路径；未指定时按 scene 选择默认权重")
    parser.add_argument("--yolo-device", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"], default="auto",
                        help="YOLO-World 推理设备；auto 由 ultralytics 自动选择")
    parser.add_argument("--yolo-conf", type=float, default=SCENE1_YOLO_DEFAULT_CONF,
                        help="YOLO-World 检测置信度阈值")
    parser.add_argument("--yolo-imgsz", type=int, default=SCENE1_YOLO_DEFAULT_IMGSZ,
                        help="YOLO-World 推理输入尺寸")
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
