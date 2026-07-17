#!/usr/bin/env python3
"""任务二所有可调参数的唯一配置源。"""

import math

# 独立运行 task2_motion 时默认使用完整真值；统一入口会按模式覆盖该话题。
VISION_TOPIC = "/task2_vision/objects_json"
VISION_TRUTH_TOPIC = "/task2_vision/objects_json"
VISION_MODE0_REFERENCE_TOPIC = "/task2_vision/mode0_reference_json"
VISION_FRAME_ID = "base_link"
VISION_TIMEOUT = 60.0
VISION_MIN_CONFIDENCE = 0.0
VISION_ALLOWED_POSITION_SOURCES = [
    "challenge_secret_layout_ground_truth",
    "qpos_bbox_depth_table_plane_xy_fit",
    "qpos_bbox_cad_depth_render_xy_fit",
    "qpos_bbox_region_depth_gn_xy_fit",
    "qpos_bbox_region_depth_xyz_yaw_fit",
    "yolo_bbox_region_depth_xyz_yaw_fit",
    "handoff_full_yaw_rgbd_cad_fit",
    "yolo_depth_pose",
]

# 数采脚本中的默认顺序；可通过下方开关恢复原动态优先级。
SORTING_OBJECT_ORDER = [
    "part_type_a_1",
    "part_type_b_1",
    "part_type_a_2",
    "part_type_b_2",
    "part_type_c_1",
    "part_type_c_2",
]
# False=严格使用上面的表顺序；True=无交接优先，同组内保持表顺序。
SORTING_USE_DYNAMIC_PRIORITY = True

# base_link 坐标系下的固定抓取偏置：X 正向前、Y 正向左，不随工件 yaw 旋转。
# 数值由原局部偏置按 scene2 默认布局等效转换，默认布局下抓取目标保持不变。
GRASP_OFFSETS_BASE = {
    "part_type_a_1": {
        "left": [0.019725109, 0.005993335, 0.05],
        "right": [0.023731775, -0.048718444, 0.05],
    },
    "part_type_a_2": {
        "left": [0.051611612, 0.010389598, 0.05],
        "right": [0.03110907, -0.030345481, 0.05],
    },
    "part_type_b_1": {
        "left": [0.018787454, -0.006857956, 0.05],
        "right": [0.021181181, -0.012286934, 0.05],
    },
    "part_type_b_2": {
        "left": [0.012518057, 0.026593847, 0.05],
        "right": [0.020924211, 0.003111904, 0.05],
    },
    "part_type_c_1": {
        "left": [0.01564838, 0.044912331, 0.03],
        "right": [0.014123567, 0.017335653, 0.03],
    },
    "part_type_c_2": {
        "left": [0.018856999, 0.035554725, 0.03],
        "right": [0.014369873, 0.036029480, 0.03],
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
        "grasp": pose_degrees([-45.0, -15.0, 0.0, 90.0, 0.0, 0.0]),
        "lift": pose_degrees([-45.0, -90.0, 0.0, 90.0, 0.0, 0.0]),
    },
}
# 各类抓取的两个等价腕部 yaw 候选，单位 degree，两者相差 180°。
# B 类横梁沿物体局部 Y，0°/180° 使夹爪从左右两侧形成 |T| 夹持。
GRASP_YAW_CANDIDATES_DEG_BY_CLASS = {
    "part_type_a": [90.0, -90.0],
    "part_type_b": [90.0, -90.0],
    "part_type_c": [90.0, -90.0],
}
# A/B/C 初次到达 pre-grasp 后，超过该 FK 姿态误差才试探另一抓取 yaw。
PRE_GRASP_ORIENTATION_RETRY_DEG = 30.0

# base_link 下的末端经验修正。旧数采的 X/Z 常量是 world 到基座的平移，现已由视觉完成。
HAND_EE_CALIBRATION = {
    "left": [-0.04, -0.02, -0.02],
    "right": [-0.04, 0.02, -0.02],
}

# 视觉重定位交接：pre/place 都是直接编码的 base_link 绝对末端坐标，不叠加
# HAND_EE_CALIBRATION。place_pose/grasp_pose 由 degree 转为内部 radian。
# [物体 yaw 偏置, 自适应 Y 轴 pitch, 预留 roll（当前无用）,手动 Z 轴 yaw, 手动 Y 轴 pitch, 手动 X 轴 roll]。
HANDOFF_CONFIG = {
    "left_to_right": {
        "pre_place_position": {
            "part_type_a": [0.366, 0.0, 0.218783],
            "part_type_c": [0.366, 0.0, 0.218783],
        },
        "place_position": {
            "part_type_a": [0.366, 0.1, 0.078783],
            "part_type_c": [0.366, 0.1, 0.078783],
        },
        "place_pose": {
            "part_type_a": pose_degrees([0.0, -90.0, 0.0, -90.0, 0.0, -90.0]),
            "part_type_c": pose_degrees([0.0, -90.0, 0.0, -90.0, 0.0, -90.0]),
        },
        "grasp_pose": {
            "part_type_a": pose_degrees([0.0, -90.0, 0.0, 90.0, 0.0, 90.0]),
            "part_type_c": pose_degrees([0.0, -90.0, 0.0, 90.0, 0.0, 90.0]),
        },
    },
    "right_to_left": {
        "pre_place_position": {
            "part_type_a": [0.266, 0.0, 0.278783],
            "part_type_c": [0.306, 0.0, 0.278783],
        },
        "place_position": {
            "part_type_a": [0.266, 0.0, 0.078783],
            "part_type_c": [0.306, 0.0, 0.078783],
        },
        "place_pose": {
            "part_type_a": pose_degrees([0.0, -90.0, 0.0, 135.0, 0.0, 90.0]),
            "part_type_c": pose_degrees([0.0, -90.0, 0.0, 135.0, 0.0, 90.0]),
        },
        "grasp_pose": {
            "part_type_a": pose_degrees([0.0, -90.0, 0.0, -120.0, 0.0, -90.0]),
            "part_type_c": pose_degrees([0.0, -90.0, 0.0, -120.0, 0.0, -90.0]),
        },
    },
}

# 每类交接独立选择一套完整逻辑。source_fk_delay 使用下面的给手 FK 相对偏移
# 和有符号夹爪时序；vision_relocalize 使用上面的绝对放置点，释放后重新视觉抓取。
HANDOFF_MODE_SOURCE_FK_DELAY = "source_fk_delay"
HANDOFF_MODE_VISION_RELOCALIZE = "vision_relocalize"
HANDOFF_MODE_BY_CLASS = {
    "part_type_a": HANDOFF_MODE_SOURCE_FK_DELAY,
    "part_type_c": HANDOFF_MODE_VISION_RELOCALIZE,
}

HANDOFF_RELOCALIZATION_REQUEST_TOPIC = "/task2_depth_xy/handoff_relocalize_request"
HANDOFF_RELOCALIZATION_RESULT_TOPIC = "/task2_depth_xy/handoff_relocalize_result"
HANDOFF_RELOCALIZATION_TIMEOUT = 60.0

# 新交接配置：pre_place_position/place_position/place_pose 为给出手的绝对目标。
# A/C 的接手 pre/grasp 偏移相对给手 place 实测 FK。
# grasp_pose 仍为接手的绝对姿态配置。
# 有符号时序差 = 接手 close 时刻 - 给手 open 时刻，单位仿真时间 s。
# 正数=先 open 给手；0=同时切换；负数=先 close 接手，再 open 给手。
# 调度器使用 /sensors_data_raw.sensor_time 计时，不受仿真实时率影响。
HANDOFF_RECEIVER_CLOSE_DELAY_BY_CLASS = {
    "part_type_a": 0.055,
    "part_type_c": -0.08,
}

HANDOFF_SOURCE_FK_RELATIVE_CONFIG = {
    "left_to_right": {
        "pre_place_position": {
            "part_type_a": [0.346, 0.2, 0.328783],
            "part_type_c": [0.346, 0.2, 0.328783],
        },
        "place_position": {
            "part_type_a": [0.346, 0.0, 0.298783],
            "part_type_c": [0.346, 0.0, 0.298783],
        },
        "place_pose": {
            "part_type_a": pose_degrees([0.0, -100.0, 0.0, -90.0, 0.0, -90.0]),
            "part_type_c": pose_degrees([0.0, -100.0, 0.0, -90.0, 0.0, -90.0]),
        },
        "pre_grasp_position": {
            "part_type_a": [0.0, -0.06, -0.05],
            "part_type_c": [0.0, -0.06, -0.05],
        },
        "grasp_position": {
            "part_type_a": [0.0, 0.0, 0.0],
            "part_type_c": [0.0, 0.0, 0.0],
        },
        "grasp_pose": {
            "part_type_a": pose_degrees([0.0, -90.0, 0.0, 90.0, 0.0, 90.0]),
            "part_type_c": pose_degrees([0.0, -90.0, 0.0, 90.0, 0.0, 90.0]),
        },
    },
    "right_to_left": {
        "pre_place_position": {
            "part_type_a": [0.346, -0.2, 0.348783],
            "part_type_c": [0.346, -0.2, 0.348783],
        },
        "place_position": {
            "part_type_a": [0.346, 0.0, 0.308783],
            "part_type_c": [0.346, 0.0, 0.308783],
        },
        "place_pose": {
            "part_type_a": pose_degrees([0.0, -100.0, 0.0, 90.0, 0.0, 90.0]),
            "part_type_c": pose_degrees([0.0, -100.0, 0.0, 90.0, 0.0, 90.0]),
        },
        "pre_grasp_position": {
            "part_type_a": [0.0, 0.06, -0.05],
            "part_type_c": [0.0, 0.06, -0.05],
        },
        "grasp_position": {
            "part_type_a": [0.0, 0.0, 0.0],
            "part_type_c": [0.0, 0.0, 0.0],
        },
        "grasp_pose": {
            "part_type_a": pose_degrees([0.0, -100.0, 0.0, -90.0, 0.0, -90.0]),
            "part_type_c": pose_degrees([0.0, -100.0, 0.0, -90.0, 0.0, -90.0]),
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
        "sorting_bin_c": [0.580486, 0.416392, 0.364811],
    },
}
# 各类工件按动作语义编号 _1/_2 施加的放箱 Y 偏置，单位 m。
PLACE_Y_OFFSETS_BY_CLASS = {
    "part_type_a": [-0.0, 0.0],
    "part_type_b": [-0.0, 0.0],
    "part_type_c": [-0.02, 0.02],
}

# 抓取、交接与放置关键点。
PRE_GRASP_Z_OFFSET = 0.06
LIFT_OFFSETS_BY_CLASS = {
    "part_type_a": [0.0, 0.0, 0.32],
    "part_type_b": [0.0, 0.0, 0.32],
    "part_type_c": [-0.05, 0.0, 0.34],
}
# C 类闭爪后的第一段低抬升，再进入上面的最终 lift。
C_INITIAL_LIFT_OFFSET = [-0.05, 0.0, 0.10]
# C 类从放箱前当前实测 FK 到 release 的笛卡尔直线等分段数。
# 8 段对应 1/8..7/8 的 7 个中间 IK q 约束和 1 个最终 q。
C_PLACE_IK_SUBDIVISIONS = 8
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
