#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集中存放所有需要对着仿真标定的常量（★TODO）。

调参只需改这一个文件，不必翻逻辑代码。位姿/坐标系/约束模式与官方数采脚本
collect_scene1_handoff_dataset.py 对齐（合法借鉴其调试好的常量），定位改用
头相机视觉，不读真值。
"""

# ---- 坐标系 ----
# IK 目标点用“IK 系绝对坐标”：world + WORLD_TO_IK_OFFSET。机器人初始 z≈0.924，
# 故 IK 系 ≈ base_link，视觉给的 base_link 坐标可近似直接喂 IK。
BASE_FRAME = "base_link"
WORLD_TO_IK_OFFSET = (0.565966, -0.013886, -0.923783)

# /kuavo_arm_traj 关节名（挑战专用，与官方脚本一致；非 zarm_l7_*）。
ARM_JOINT_NAMES = ["arm_joint_%d" % i for i in range(1, 15)]

# ★TODO: 头相机彩色光学坐标系帧名。运行时 `rostopic echo /tf_static` 核对。
DEFAULT_HEAD_CAM_OPTICAL_FRAME = "camera_color_optical_frame"

# ---- 话题 / 服务名 ----
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ARM_TRAJ = "/kuavo_arm_traj"
TOPIC_ARM_TARGET_POSES = "/kuavo_arm_target_poses"
TOPIC_SENSORS = "/sensors_data_raw"
TOPIC_CLAW_STATE = "/leju_claw_state"
TOPIC_HEAD_COLOR = "/cam_h/color/image_raw/compressed"
TOPIC_HEAD_DEPTH = "/cam_h/depth/image_raw/compressedDepth"
TOPIC_HEAD_CAMINFO = "/cam_h/color/camera_info"
TOPIC_LIDAR = "/lidar/points"
TOPIC_HEAD_MOTION = "/robot_head_motion_data"

SRV_CLAW = "/control_robot_leju_claw"
SRV_IK = "/ik/two_arm_hand_pose_cmd_srv"
SRV_FK = "/ik/fk_srv"
SRV_ARM_MODE = "/arm_traj_change_mode"

# ---- 运动参数 ----
ARM_TRAJ_HZ = 100.0      # 关节插值发布频率
ARM_MOVE_TIME = 2.0      # 单步运动时长(s)
ARM_SETTLE_TIME = 0.3    # 运动后稳定等待(s)
ARM_MODE_EXTERNAL = 2    # 手臂外部控制模式

# ---- IK 约束模式（与官方一致）----
IK_MODE_POS_HARD_ORI_SOFT = 0x02
IK_MODE_POS_HARD_ORI_HARD = 0x03     # 最终夹紧：位置+姿态都硬解
IK_MODE_THREE_POINT_MIXED = 0x06     # 普通移动：三点混合
IK_MAJOR_ITERATIONS_LIMIT = 500

# ---- 预设抬手轨迹（度，14 关节）。源自官方 PRESET_POINTS_DEG ----
# 开局必须先平滑走这串路点：把双臂从自然垂手抬到身前对称预设位。
# 作用：①稳定重心避免猛伸手摔倒 ②给后续 IK 一个良好的初始解。
# 段1（前两点）外展张开避障，段2抬到身前对称位；末点也是收尾回归点。
PRESET_POINTS_DEG = [
    [20, 0, 0, -30, 0, 0, 0,  20, 0, 0, -30, 0, 0, 0],            # 起点
    [20, 90, 0, -55, 0, 0, 0,  20, -90, 0, -55, 0, 0, 0],         # 外展峰
    [20, 60, 0, -75, 0, 0, 0,  20, -60, 0, -75, 0, 0, 0],         # 外展峰
    [29.89, 30.67, 29.889, -139.1, -59.33, 0, 0,
     29.89, -30.67, -29.889, -139.1, 59.33, 0, 0],                # 身前预设
    [29.89, 10.67, 9.889, -139.1, -59.33, 0, 0,
     29.89, -10.67, -9.889, -139.1, 59.33, 0, 0],                 # 末端预设/回归点
]
PRESET_SEGMENT_TIME = 1.92    # 相邻路点插值时长(s)
PRESET_SETTLE_TIME = 0.5      # 抵达末端预设后的稳定时间(s)

# ---- 头部姿态（度）。joint_data=[yaw, pitch]，yaw∈[-30,30] pitch∈[-25,25] ----
# 官方 HEAD_TARGET=[0.0, 20.0]：低头看桌面是 pitch=+20（正值）。
HEAD_LOOK_FORWARD = (0.0, 0.0)
HEAD_LOOK_DOWN = (0.0, 20.0)
HEAD_SCAN_YAWS = (0.0, -20.0, 20.0)   # 低头后只转脖子扫桌面，不转身体

# ---- 夹爪行程（0 张开 / 100 闭合）----
CLAW_OPEN = 5.0
CLAW_CLOSE = 95.0

# ---- HSV 颜色表 (下界, 上界)。★TODO: 对着实际画面用取色器精修 ----
HSV_COLORS = {
    "brown":  ((8, 60, 60),    (22, 200, 200)),
    "yellow": ((22, 90, 120),  (35, 255, 255)),
    "orange": ((8, 120, 120),  (20, 255, 255)),
    "blue":   ((100, 80, 120), (130, 255, 255)),
    "gray":   ((0, 0, 120),    (180, 40, 220)),
    "black":  ((0, 0, 0),      (180, 80, 70)),
    "red":    ((0, 120, 80),   (10, 255, 255)),
}

# =====================================================================
# 场景一位姿常量（IK 系绝对坐标，源自官方调试值）
# YPR 为两段叠加(度)，见 hardware.quat_from_ypr。
# =====================================================================

# ---- 右手抓取 ----
RIGHT_PICK_IK_Z = -0.03           # 最终抓取高度基准(IK z)。抓太高/太低先调这里
RIGHT_PICK_TRANSIT_IK_Z = 0.120   # 抓取前横移高度
LIFT_Z_OFFSET = 0.300             # 夹紧后抬起高度
RIGHT_PICK_FIRST_YPR_DEG = (0.0, -90.0, 0.0)
RIGHT_PICK_SECOND_YPR_DEG = (90.0, 0.0, 0.0)

# ---- 称重区释放（A 阶段终点）----
WEIGH_RELEASE_IK = (0.396, -0.574, 0.146217)    # 释放点(已含落点补偿)
WEIGH_TRANSIT_IK_Z = 0.326217                   # 释放前高位
WEIGH_RELEASE_FIRST_YPR_DEG = (0.0, -100.0, 0.0)
WEIGH_RELEASE_SECOND_YPR_DEG = (90.0, 0.0, 0.0)
WEIGH_DWELL = 1.0                               # 放上称重区后停留(s)

# =====================================================================
# B 阶段：二次夹起 → 交给左手 → 左手放进分拣箱（源自官方调试值）
# =====================================================================

# ---- 二次夹起（右手把称重区的快递重新夹起）----
WEIGH_REGRASP_IK = (0.396, -0.574, -0.04)        # 二次夹起点(z=重新夹起高度)
WEIGH_REGRASP_FIRST_YPR_DEG = (0.0, -60.0, 0.0)
WEIGH_REGRASP_SECOND_YPR_DEG = (90.0, 0.0, 0.0)
WEIGH_RELEASE_SETTLE_BEFORE_OPEN = 1.5           # 到释放点后、开爪前停留(s)

# ---- 左手等待预设位 ----
LEFT_PRESET_2_IK = (0.313, 0.239, 0.282)
LEFT_PRESET_2_FIRST_YPR_DEG = (-146.440, 4.966, 0.0)
LEFT_PRESET_2_SECOND_YPR_DEG = (0.0, 0.0, 96.580)

# ---- 右手交接点（先抬到高位避障对齐，再下降到交接点）----
RIGHT_HANDOFF_TO_LEFT_IK = (0.246, -0.044645, 0.3016983)
RIGHT_HANDOFF_TRANSIT_IK_Z = 0.40                # 交接前避障高位
RIGHT_HANDOFF_TO_LEFT_FIRST_YPR_DEG = (-0.839, -100.0, 0.0)
RIGHT_HANDOFF_TO_LEFT_SECOND_YPR_DEG = (90.0, -20.0, 90.0)
RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET = -0.300  # 交接后右手退让 y 偏移

# ---- 左手接收点（先侧方对齐 x/z，再沿 y 接近右手）----
LEFT_HANDOFF_RECEIVE_XZ_READY_IK = (0.266, 0.139, 0.2216983)
LEFT_HANDOFF_RECEIVE_IK = (0.266, 0.04645, 0.2216983)
LEFT_HANDOFF_RECEIVE_FIRST_YPR_DEG = (-0.839, -100.0, 0.0)
LEFT_HANDOFF_RECEIVE_SECOND_YPR_DEG = (-90.0, 0.0, -90.0)

# ---- 左手放进分拣箱 ----
BOX_DROP_IK = (0.605966, 0.226114, 0.556217)     # 放箱基准点(IK 系)
BOX_DROP_FIRST_YPR_DEG = (-0.328, -100.935, 0.0)
BOX_DROP_SECOND_YPR_DEG = (-90.0, 0.0, 0.369)
# 每个快递在基准点上叠加偏移，排成 2x2（按处理顺序 brown/yellow/orange/blue）。
BOX_DROP_OFFSETS = (
    (0.0, -0.01, 0.0),
    (0.0,  0.04, 0.0),
    (0.04, -0.03, 0.0),
    (0.04,  0.03, 0.0),
)
GRIPPER_HOLD_TIME = 1.0                          # 夹爪开合后保持/等待(s)

# ---- 场景一快递颜色序列 ----
SCENE1_PARCEL_COLORS = ("brown", "yellow", "orange", "blue")

# =====================================================================
# 场景二/三（后续实现，先保留颜色与计数）
# =====================================================================
SCENE2_BIN_COLORS = ("gray", "black", "red")    # A→蓝箱 B→橙箱 C→紫箱
SCENE2_PER_CLASS = 2
SCENE3_TRAY_COLOR = "gray"                       # ★TODO: 料盘实际颜色
SCENE3_TRAY_COUNT = 5
