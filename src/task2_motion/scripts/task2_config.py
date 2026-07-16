#!/usr/bin/env python3
"""任务二所有可调参数的唯一配置源。"""

import math

# 视觉输入。后续接入 YOLO 时只需保持相同 JSON 字段和 base_link 坐标语义。
VISION_TOPIC = "/task2_vision/objects_json"
VISION_FRAME_ID = "base_link"
VISION_TIMEOUT = 30.0
VISION_MIN_CONFIDENCE = 0.0
VISION_ALLOWED_POSITION_SOURCES = [
    "challenge_secret_layout_ground_truth",
    "yolo_depth_pose",
]

OBJECT_NAMES = [
    "part_type_a_1",
    "part_type_a_2",
    "part_type_b_1",
    "part_type_b_2",
    "part_type_c_1",
    "part_type_c_2",
]
OBJECT_TO_BIN = {
    "part_type_a_1": "sorting_bin_a",
    "part_type_a_2": "sorting_bin_a",
    "part_type_b_1": "sorting_bin_b",
    "part_type_b_2": "sorting_bin_b",
    "part_type_c_1": "sorting_bin_c",
    "part_type_c_2": "sorting_bin_c",
}

# 数采脚本中的默认顺序；可通过下方开关恢复原动态优先级。
SORTING_OBJECT_ORDER = [
    "part_type_a_1",
    "part_type_a_2",
    "part_type_b_1",
    "part_type_b_2",
    "part_type_c_1",
    "part_type_c_2",
]
# False=严格使用上面的默认顺序；True=启用无交接优先、再按 |Y| 排序。
SORTING_USE_DYNAMIC_PRIORITY = False

# base_link 坐标系下的固定抓取偏置：X 正向前、Y 正向左，不随工件 yaw 旋转。
# 数值由原局部偏置按 scene2 默认布局等效转换，默认布局下抓取目标保持不变。
GRASP_OFFSETS_BASE = {
    "part_type_a_1": {
        "left": [0.019725109, 0.005993335, 0.05],
        "right": [0.023731775, -0.055718444, 0.05],
    },
    "part_type_a_2": {
        "left": [0.024611612, 0.004389598, 0.05],
        "right": [0.03110907, -0.030345481, 0.05],
    },
    "part_type_b_1": {
        "left": [0.018787454, -0.006857956, 0.05],
        "right": [0.020181181, -0.012286934, 0.05],
    },
    "part_type_b_2": {
        "left": [0.012518057, 0.021593847, 0.05],
        "right": [0.000924211, 0.014111904, 0.05],
    },
    "part_type_c_1": {
        "left": [0.01564838, 0.044912331, 0.04],
        "right": [-0.014123567, 0.017335653, 0.04],
    },
    "part_type_c_2": {
        "left": [0.012856999, 0.035554725, 0.045],
        "right": [-0.001369873, 0.036029480, 0.05],
    },
}


def pose_degrees(values):
    """将六维 degree 姿态配置转换为内部使用的 radian。"""
    angles = [float(value) for value in values]
    if len(angles) != 6:
        raise ValueError("pose degrees must contain 6 angles")
    return [math.radians(value) for value in angles]


# 姿态参数顺序为 [yaw_bias, pitch, roll, manual_yaw, manual_pitch, manual_roll]，单位 degree。
CLASS_POSES = {
    "part_type_a": {
        "grasp": pose_degrees([0.0, -7.5, 0.0, 90.0, 0.0, 0.0]),
        "lift": pose_degrees([0.0, -90.0, 0.0, 90.0, 0.0, 0.0]),
    },
    "part_type_b": {
        "grasp": pose_degrees([0.0, -10.0, 0.0, 90.0, 0.0, 0.0]),
        "lift": pose_degrees([0.0, -90.0, 0.0, 90.0, 0.0, 0.0]),
    },
    "part_type_c": {
        "grasp": pose_degrees([-45.0, -45.0, 0.0, 90.0, 0.0, 0.0]),
        "lift": pose_degrees([-45.0, -45.0, 0.0, 90.0, 0.0, 0.0]),
    },
}
GRASP_NARROW_EDGE_YAW_OFFSET = math.pi / 2
# A1 的物理长轴是局部 Y；与 base_link Y 平行时使用另一窄边。
A1_BASE_Y_PARALLEL_TOL_DEG = 15.0

# base_link 下的末端经验修正。旧数采的 X/Z 常量是 world 到基座的平移，现已由视觉完成。
HAND_EE_CALIBRATION = {
    "left": [-0.04, -0.02, -0.02],
    "right": [-0.04, 0.02, -0.02],
}

# 左右手交接参考点已由旧 world 坐标转换到 base_link，尚未叠加上面的手侧修正。
HANDOFF_CONFIG = {
    "left_to_right": {
        "place_position": {
            "part_type_a": [0.266, 0.0, -0.038783],
            "part_type_c": [0.316, 0.0, -0.038783],
        },
        "place_pose": {
            "part_type_a": [
                0.0,
                -math.pi / 2,
                0.0,
                -3 * math.pi / 4,
                0.0,
                -math.pi / 2,
            ],
            "part_type_c": [
                0.0,
                -math.pi / 2,
                0.0,
                -2 * math.pi / 3,
                0.0,
                -math.pi / 2,
            ],
        },
        "grasp_position": {
            "part_type_a": [0.316, 0.0, -0.073783],
            "part_type_c": [0.316, 0.0, -0.073783],
        },
        "grasp_pose": {
            "part_type_a": [0.0, -math.pi / 2, 0.0, 2 * math.pi / 3, 0.0, math.pi / 2],
            "part_type_c": [0.0, -math.pi / 2, 0.0, 2 * math.pi / 3, 0.0, math.pi / 2],
        },
    },
    "right_to_left": {
        "place_position": {
            "part_type_a": [0.266, 0.0, -0.038783],
            "part_type_c": [0.306, 0.0, -0.028783],
        },
        "place_pose": {
            "part_type_a": [0.0, -math.pi / 2, 0.0, 3 * math.pi / 4, 0.0, math.pi / 2],
            "part_type_c": [0.0, -math.pi / 2, 0.0, 3 * math.pi / 4, 0.0, math.pi / 2],
        },
        "grasp_position": {
            "part_type_a": [0.316, 0.0, -0.073783],
            "part_type_c": [0.316, 0.0, -0.078783],
        },
        "grasp_pose": {
            "part_type_a": [
                0.0,
                -math.pi / 2,
                0.0,
                -2 * math.pi / 3,
                0.0,
                -math.pi / 2,
            ],
            "part_type_c": [
                0.0,
                -math.pi / 2,
                0.0,
                -2 * math.pi / 3,
                0.0,
                -math.pi / 2,
            ],
        },
    },
}

# 初始、工作和结束关节预位，顺序为左臂 1-7、右臂 1-7，单位 degree。
HOME_JOINTS_DEG = [
    20,
    0,
    0,
    -30,
    0,
    0,
    0,
    20,
    0,
    0,
    -30,
    0,
    0,
    0,
]
SIDE_LIFT_JOINTS_DEG = [
    20,
    70,
    0,
    -35,
    0,
    0,
    0,
    20,
    -70,
    0,
    -35,
    0,
    0,
    0,
]
WORK_POSE_JOINTS_DEG = [
    30,
    10,
    10,
    -120,
    -60,
    0,
    0,
    30,
    -10,
    -10,
    -120,
    60,
    0,
    0,
]

# 启动时使用的连续关节预位路径，与任务一保持一致。
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

# base_link 下的放箱末端位置，按实际可到达的手臂和箱子组合配置。
# 这里是末端绝对 XYZ，不叠加抓取偏置或 HAND_EE_CALIBRATION。
PLACE_POSITIONS_BASE = {
    "right": {
        "sorting_bin_a": [0.535486, -0.443608, 0.244811],
        "sorting_bin_b": [0.535486, -0.013608, 0.244811],
    },
    "left": {
        "sorting_bin_b": [0.535486, -0.013608, 0.244811],
        "sorting_bin_c": [0.545486, 0.416392, 0.364811],
    },
}

# 抓取、交接与放置关键点。
PRE_GRASP_Z_OFFSET = 0.07
LIFT_OFFSETS_BY_CLASS = {
    "part_type_a": [0.0, 0.0, 0.32],
    "part_type_b": [0.0, 0.0, 0.32],
    "part_type_c": [-0.05, 0.0, 0.34],
}
# C 类从 lift 到 release 的连续 q 轨迹所经过的 IK 高度比例。
C_PLACE_IK_Z_FRACTIONS = [0.50]
HANDOFF_APPROACH_Z_OFFSET = 0.20
GRIPPER_CLOSE_TIME = 0.70
RELEASE_DWELL = 0.40

# WBC、IK 与关节插值参数。
ARM_JOINT_NAMES = ["arm_joint_" + str(index) for index in range(1, 15)]
TOPIC_TIMEOUT = 20.0
ARM_TRAJ_HZ = 100.0
# 每个最终关键点使用一次完整移动时长。
ARM_MOVE_TIME = 1.40
JOINT_COMMAND_SEGMENT_TIME = 0.10
MOTION_SETTLE_TIME = 0.05
IK_MODE_THREE_POINT_MIXED = 0x06
IK_POS_COST_WEIGHT = 1.0
IK_LIFT_POS_COST_WEIGHT = 1.0
IK_MAJOR_ITERATIONS_LIMIT = 500
SOFT_IK_BISECT_ATTEMPTS = 3

# 每个关键点完成后只检查作业臂；超时仅告警并继续。
ARM_STABLE_DURATION = 0.25
ARM_STABLE_MAX_SPEED_DEG_S = 1.0
ARM_STABLE_TIMEOUT = 10.0
FK_POSITION_WARN = 0.03
FK_ORIENTATION_WARN_RAD = math.radians(20.0)
