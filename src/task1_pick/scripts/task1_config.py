#!/usr/bin/env python3
"""任务一所有可调参数的唯一配置源。"""

import math
import os

# 各包裹抓取前叠加到视觉位置上的修正量。
PICK_OFFSET = {
    "parcel_1": [-0.03, 0.0, 0.002],
    "parcel_2": [-0.03, 0.004, 0.002],
    "parcel_3": [-0.02, -0.0, 0.002],
    "parcel_4": [-0.009, 0.01, 0.0],
}
# base_link 坐标系下的抓取接近与抓后撤离参数。
PICK_RETRACT_X_OFFSET = -0.1
RIGHT_PICK_IK_Z = -0.03
YZ_ALIGN_X_SAFE = 0.184
# 从 x_pre 到 grasp 的下降比例：0=x_pre，1=grasp。
PICK_DESCENT_MID_RATIO = 0.36  # 1.0 / 3.0

# 称重区内右手末端释放点的绝对坐标。
WEIGH_RELEASE_IK = {
    "parcel_1": [0.436, -0.53, 0.144],
    "parcel_2": [0.436, -0.53, 0.154],
    "parcel_3": [0.446, -0.53, 0.164],
    "parcel_4": [0.456, -0.53, 0.164],
}
# 严格重抓后离开称重区时使用的 XY 坐标。
WEIGH_REGRASP_XY = {
    "parcel_1": [0.414, -0.55],
    "parcel_2": [0.414, -0.55],
    "parcel_3": [0.414, -0.55],
    "parcel_4": [0.414, -0.55],
}
# 右臂 r6 严格 IK 的相对位移，求解期间锁定 q7。
WEIGH_REGRASP_R6_OFFSET = [-0.003, 0.006, -0.064]
# 调试开关：False=同姿态 EEF 软 IK、不锁 q7；True=原 r6 严格 IK、锁 q7。
WEIGH_REGRASP_USE_STRICT_R6_IK = True
# 各包裹经过称重区上方时的安全高度。
WEIGH_TRANSIT_Z = {
    "parcel_1": 0.326,
    "parcel_2": 0.326,
    "parcel_3": 0.336,
    "parcel_4": 0.366,
}
# 抓取后的首次抬升高度；包裹 2 只抬升 5 cm，再按原轨迹进入满高度 mid。
LIFT_Z = {
    "parcel_1": 0.5 * WEIGH_TRANSIT_Z["parcel_1"],
    "parcel_2": RIGHT_PICK_IK_Z + PICK_OFFSET["parcel_2"][2] + 0.05,
    "parcel_3": 0.5 * WEIGH_TRANSIT_Z["parcel_3"],
    "parcel_4": 0.6 * WEIGH_TRANSIT_Z["parcel_4"],
}
WEIGH_FINAL_DESCEND_MOVE_TIME = 1.0
REGRASP_LIFT_Z = 0.346
WEIGH_DWELL = 1.0
# 临时调试：按进入称重动作链后的固定仿真时间开右爪，不等待右臂稳定。
WEIGH_DEBUG_FIXED_RELEASE_ENABLED = False
WEIGH_DEBUG_FIXED_RELEASE_TIME = 6.0

# 称重释放与重抓抬升使用的固定组合旋转。
WEIGH_RELEASE_YPR_DEG = ([-10, -100.0, 0.0], [90.0, 0.0, 0.0])
WEIGH_REGRASP_YPR_DEG = ([-10, -90.0, 0.0], [90.0, 0.0, 0.0])

# 右手交接路径和左手接取几何参数。
RIGHT_HANDOFF_IK = [0.283, -0.085, 0.3117]
HANDOFF_TRANSIT_Z = 0.40
HANDOFF_FALLBACK_ZS = [0.37, 0.35]
LEFT_RECEIVE_READY = [0.333, 0.095, 0.2817]
LEFT_RECEIVE_IK_OFFSET = [0.018, 0.06, 0.02]
RIGHT_RETRACT_Y_OFFSET = -0.300
HANDOFF_RIGHT_YPR = ([-0.839, -100.0, 0.0], [90.0, -20.0, 90.0])
HANDOFF_LEFT_YPR = ([-0.839, -90, 80], [-90.0, -0.0, -90.0])

# 左手向收纳箱放置包裹的目标参数。
BOX_DROP_IK = [0.585966, 0.256114, 0.4362]
BOX_DROP_OFFSETS = [
    [0.0, -0.01, 0.0],
    [0.0, 0.02, 0.0],
    [0.04, 0.05, 0.0],
    [0.04, 0.08, 0.0],
]
BOX_DROP_X_FALLBACK = [-0.03, -0.06]
LEFT_BOX_YPR = ([-0.328, -100.935, 0.0], [-90.0, 0.0, 0.369])
BOX_TRANSIT_MOVE_TIME = 2.0
# 放箱抬升后，从实际 FK 连续执行两个后撤下降中间点。
BOX_SAFE_INTERMEDIATE_STEP = [-0.06, 0.0, -0.03]
BOX_SAFE_INTERMEDIATE_COUNT = 2
# 通用指令等待时间与夹爪保持时长。
SETTLE_TIME = 0.05
GRIPPER_HOLD = 0.7
# 手臂稳定判定：关节速度持续低于阈值，时间和超时均使用仿真时间。
ARM_STABLE_DURATION = 0.25
ARM_STABLE_MAX_SPEED_DEG_S = 1.0
ARM_STABLE_TIMEOUT = 10.0

# 软 IK 与硬 IK 的服务模式参数。
IK_MODE_THREE_POINT_MIXED = 0x06
IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_POS_COST_WEIGHT = 0.0
HANDOFF_POS_COST_WEIGHT = 1.0
IK_MAJOR_ITERATIONS_LIMIT = 500
# 任意软 IK 失败时依次插入 1/2、1/4、1/8 临时点。
SOFT_IK_BISECT_ATTEMPTS = 3

# 严格多参考 IK 与轨迹限制。
STRICT_IK_ORIENTATION_TOL = math.radians(2.0)
STRICT_IK_POSITION_TOL = 0.002
STRICT_R6_IK_POSITION_TOL = 0.001
STRICT_IK_MAJOR_ITERATIONS_LIMIT = 100
STRICT_TRAJECTORY_MOVE_TIME = 1.1
STRICT_R6_PATH_INTERMEDIATE_POINTS = 2
STRICT_PATH_MAX_JOINT_STEP_DEG = 25.0

# 初次抓取包裹时右手使用的固定姿态。
RIGHT_PICK_YPR_DEG = ([0.0, -90.0, 0.0], [90.0, 0.0, 0.0])

# 运行话题、超时、轨迹记录及允许的视觉来源。
TRACE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scene1_trace_v2.txt"
)
PARCEL_NAMES = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]
ARM_JOINT_NAMES = ["arm_joint_" + str(i) for i in range(1, 15)]
TOPIC_TIMEOUT = 20.0
ARM_TRAJ_HZ = 100.0
TASK1_VISION_TOPIC = "/task1_vision/objects_json"
# 视觉等待使用 /sensors_data_raw.sensor_time：先告警，达到硬超时才中断。
TASK1_VISION_WARN_TIME = 60.0
TASK1_VISION_TIMEOUT = 180.0
TASK1_VISION_MIN_CONFIDENCE = 0.0
TASK1_VISION_FRAME_ID = "base_link"
TASK1_VISION_MEDIAN_FRAMES = 10
TASK1_VISION_USED_TOPIC = "/task1_vision/used_objects_json"
TASK1_VISION_ALLOWED_POSITION_SOURCES = [
    "depth_roi_top_plane_fit",
]
DEFAULT_VISION_TO_IK_OFFSET = [0.0, 0.0, 0.0]

# 启动时及单手完成任务后使用的关节空间预位轨迹。
PRESET_POINTS_DEG = [
    [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0],
    [20, 90, 0, -55, 0, 0, 0, 20, -90, 0, -55, 0, 0, 0],
    [20, 60, 0, -75, 0, 0, 0, 20, -60, 0, -75, 0, 0, 0],
    [
        29.89,
        30.67,
        29.889,
        -139.1,
        -59.33,
        0,
        0,
        29.89,
        -30.67,
        -29.889,
        -139.1,
        59.33,
        0,
        0,
    ],
    [
        29.89,
        10.67,
        9.889,
        -139.1,
        -59.33,
        0,
        0,
        29.89,
        -10.67,
        -9.889,
        -139.1,
        59.33,
        0,
        0,
    ],
]
PRESET_SEGMENT_TIME = 0.4
PRESET_SETTLE_TIME = 0.5
