#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scene1 single-file task entry.

Run:
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3

This file keeps the official simulator launcher intact, then runs only scene1
with local perception, IK, arm trajectory, gripper, and task-flow code.
"""

import argparse
import math
import os
import sys
import threading


DEFAULT_SIM_STARTUP_TIMEOUT = 300.0
SIM_TIMEOUT_ENV_VAR = "CHALLENGE_TASK_SIM_TIMEOUT"


# ---------------------------------------------------------------------------
# Scene1 constants
# ---------------------------------------------------------------------------

BASE_FRAME = "base_link"
DEFAULT_HEAD_CAM_OPTICAL_FRAME = "camera_color_optical_frame"

ARM_JOINT_NAMES = ["arm_joint_%d" % i for i in range(1, 15)]

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

ARM_TRAJ_HZ = 100.0
ARM_MOVE_TIME = 2.0
ARM_SETTLE_TIME = 0.3
ARM_MODE_EXTERNAL = 2
ARM_MODE_SERVICE_TIMEOUT = 20.0
ARM_MODE_PRE_SWITCH_STAND_SETTLE = 3.0
ARM_MODE_POST_SWITCH_SETTLE = 1.0

IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_MODE_THREE_POINT_MIXED = 0x06
IK_MAJOR_ITERATIONS_LIMIT = 500

PRESET_POINTS_DEG = [
    [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0],
    [20, 90, 0, -55, 0, 0, 0, 20, -90, 0, -55, 0, 0, 0],
    [20, 60, 0, -75, 0, 0, 0, 20, -60, 0, -75, 0, 0, 0],
    [29.89, 30.67, 29.889, -139.1, -59.33, 0, 0,
     29.89, -30.67, -29.889, -139.1, 59.33, 0, 0],
    [29.89, 10.67, 9.889, -139.1, -59.33, 0, 0,
     29.89, -10.67, -9.889, -139.1, 59.33, 0, 0],
]
PRESET_SEGMENT_TIME = 1.92
PRESET_SETTLE_TIME = 0.5

HEAD_LOOK_FORWARD = (0.0, 0.0)
HEAD_LOOK_DOWN = (0.0, 20.0)
HEAD_SCAN_YAWS = (0.0, -20.0, 20.0)

CLAW_OPEN = 5.0
CLAW_CLOSE = 95.0
CLAW_CLOSE_MIN_STATE = 1
CLAW_OPEN_MIN_STATE = 2

HSV_COLORS = {
    "brown": ((8, 60, 60), (22, 200, 200)),
    "yellow": ((22, 90, 120), (35, 255, 255)),
    "orange": ((8, 120, 120), (20, 255, 255)),
    "blue": ((100, 80, 120), (130, 255, 255)),
}
PARCEL_MIN_AREA = 500
PARCEL_DETECT_MIN_AREA = 500
PARCEL_DETECT_MORPH_KERNEL = 5
PARCEL_DEPTH_WINDOW = 5

RIGHT_PICK_GRASP_Z_OFFSET = 0.100
RIGHT_PICK_GRASP_Z_MIN = 0.045
RIGHT_PICK_GRASP_Z_MAX = 0.110
RIGHT_PICK_HOVER_Z_OFFSET = 0.220
RIGHT_PICK_LIFT_Z_OFFSET = 0.300
RIGHT_PICK_XY_OFFSET = (0.0, 0.0)
RIGHT_PICK_MIN_X = 0.150
RIGHT_PICK_MAX_X = 0.500
RIGHT_PICK_MIN_Y = -0.430
RIGHT_PICK_MAX_Y = -0.180
RIGHT_PICK_FIRST_YPR_DEG = (0.0, -90.0, 0.0)
RIGHT_PICK_SECOND_YPR_DEG = (90.0, 0.0, 0.0)
RIGHT_PICK_TRANSIT_CONSTRAINT_MODE = IK_MODE_THREE_POINT_MIXED
RIGHT_PICK_FINAL_CONSTRAINT_MODE = IK_MODE_THREE_POINT_MIXED
PICK_ALIGN_MOVE_TIME = 1.2
PICK_GRASP_MOVE_TIME = 1.4
PICK_LIFT_MOVE_TIME = 1.2
IK_VALIDATION_TRANSIT_POS_TOL = 0.04
IK_VALIDATION_TRANSIT_ORI_TOL_DEG = None
IK_VALIDATION_FINAL_POS_TOL = 0.04
IK_VALIDATION_FINAL_ORI_TOL_DEG = None

WEIGH_RELEASE_IK = (0.396, -0.574, 0.146217)
WEIGH_TRANSIT_IK_Z = 0.326217
WEIGH_RELEASE_FIRST_YPR_DEG = (0.0, -100.0, 0.0)
WEIGH_RELEASE_SECOND_YPR_DEG = (90.0, 0.0, 0.0)
WEIGH_DWELL = 1.0

WEIGH_REGRASP_IK = (0.396, -0.574, -0.04)
WEIGH_REGRASP_FIRST_YPR_DEG = (0.0, -60.0, 0.0)
WEIGH_REGRASP_SECOND_YPR_DEG = (90.0, 0.0, 0.0)

LEFT_PRESET_2_IK = (0.313, 0.239, 0.282)
LEFT_PRESET_2_FIRST_YPR_DEG = (-146.440, 4.966, 0.0)
LEFT_PRESET_2_SECOND_YPR_DEG = (0.0, 0.0, 96.580)

RIGHT_HANDOFF_TO_LEFT_IK = (0.246, -0.044645, 0.3016983)
RIGHT_HANDOFF_TRANSIT_IK_Z = 0.40
RIGHT_HANDOFF_TO_LEFT_FIRST_YPR_DEG = (-0.839, -100.0, 0.0)
RIGHT_HANDOFF_TO_LEFT_SECOND_YPR_DEG = (90.0, -20.0, 90.0)
RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET = -0.300

LEFT_HANDOFF_RECEIVE_XZ_READY_IK = (0.266, 0.139, 0.2216983)
LEFT_HANDOFF_RECEIVE_IK = (0.266, 0.04645, 0.2216983)
LEFT_HANDOFF_RECEIVE_FIRST_YPR_DEG = (-0.839, -100.0, 0.0)
LEFT_HANDOFF_RECEIVE_SECOND_YPR_DEG = (-90.0, 0.0, -90.0)

BOX_DROP_IK = (0.605966, 0.226114, 0.556217)
BOX_DROP_FIRST_YPR_DEG = (-0.328, -100.935, 0.0)
BOX_DROP_SECOND_YPR_DEG = (-90.0, 0.0, 0.369)
LEFT_BOX_STATION_IK = BOX_DROP_IK
LEFT_BOX_STATION_FIRST_YPR_DEG = BOX_DROP_FIRST_YPR_DEG
LEFT_BOX_STATION_SECOND_YPR_DEG = BOX_DROP_SECOND_YPR_DEG
RIGHT_BOX_HANDOFF_IK = (
    LEFT_BOX_STATION_IK[0] - 0.020,
    LEFT_BOX_STATION_IK[1] - 0.091,
    LEFT_BOX_STATION_IK[2] + 0.080,
)
RIGHT_BOX_HANDOFF_TRANSIT_IK_Z = RIGHT_BOX_HANDOFF_IK[2] + 0.120
BOX_DROP_OFFSETS = (
    (0.0, -0.01, 0.0),
    (0.0, 0.04, 0.0),
    (0.04, -0.03, 0.0),
    (0.04, 0.03, 0.0),
)

GRIPPER_HOLD_TIME = 1.0
SCENE1_PARCEL_COLORS = ("brown", "yellow", "orange", "blue")
SCENE1_MAX_SCAN_RETRIES = 3
SCENE1_PICK_ATTEMPTS = 1


# ---------------------------------------------------------------------------
# Lightweight math helpers
# ---------------------------------------------------------------------------

def _matmul3(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
             for i in range(3)]


def _rot_to_quat_xyzw(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [(m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
                0.25 * s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return [0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[2][1] - m[1][2]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return [(m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s,
                (m[0][2] - m[2][0]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return [(m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s]


def quat_from_ypr(first_ypr_deg, second_ypr_deg=None):
    yaw, pitch, _roll = [math.radians(float(v)) for v in first_ypr_deg]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    matrix = [[cy * cp, -sy, cy * sp],
              [sy * cp, cy, sy * sp],
              [-sp, 0.0, cp]]
    if second_ypr_deg is not None:
        m_yaw, m_pitch, m_roll = [math.radians(float(v)) for v in second_ypr_deg]
        manual = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if abs(m_yaw) > 0.01:
            c, s = math.cos(m_yaw), math.sin(m_yaw)
            manual = _matmul3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], manual)
        if abs(m_pitch) > 0.01:
            c, s = math.cos(m_pitch), math.sin(m_pitch)
            manual = _matmul3([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], manual)
        if abs(m_roll) > 0.01:
            c, s = math.cos(m_roll), math.sin(m_roll)
            manual = _matmul3([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], manual)
        matrix = _matmul3(matrix, manual)
    return _rot_to_quat_xyzw(matrix)


def quat_angle_error_rad(actual_xyzw, desired_xyzw):
    dot = sum(float(actual_xyzw[i]) * float(desired_xyzw[i]) for i in range(4))
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(dot))))


# ---------------------------------------------------------------------------
# Runtime imports and optional dependency flags
# ---------------------------------------------------------------------------

def _import_runtime_modules():
    import rospy
    from geometry_msgs.msg import PointStamped, Twist
    from sensor_msgs.msg import CameraInfo, CompressedImage, JointState, PointCloud2
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import cv2
    except Exception:
        cv2 = None
    try:
        import tf2_ros
        import tf2_geometry_msgs  # noqa: F401
    except Exception:
        tf2_ros = None
    from kuavo_msgs.msg import (
        armTargetPoses,
        ikSolveParam,
        lejuClawState,
        robotHeadMotionData,
        sensorsData,
        twoArmHandPoseCmd,
    )
    from kuavo_msgs.srv import (
        changeArmCtrlMode,
        controlLejuClaw,
        controlLejuClawRequest,
        fkSrv,
        twoArmHandPoseCmdSrv,
    )
    return locals()


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

class Perception(object):
    def __init__(self, rt):
        self.rt = rt
        self.rospy = rt["rospy"]
        self.cv2 = rt["cv2"]
        self.np = rt["np"]
        self._lock = threading.Lock()
        self.sensor_time = None
        self.arm_q = None
        self.claw_state = [0, 0]
        self.color_bgr = None
        self.depth_m = None
        self.cam_K = None
        self.cam_frame = None
        self.cloud = None
        self.rospy.Subscriber(TOPIC_SENSORS, rt["sensorsData"], self._on_sensors, queue_size=1)
        self.rospy.Subscriber(TOPIC_CLAW_STATE, rt["lejuClawState"], self._on_claw, queue_size=1)
        self.rospy.Subscriber(TOPIC_HEAD_COLOR, rt["CompressedImage"], self._on_color, queue_size=1)
        self.rospy.Subscriber(TOPIC_HEAD_DEPTH, rt["CompressedImage"], self._on_depth, queue_size=1)
        self.rospy.Subscriber(TOPIC_HEAD_CAMINFO, rt["CameraInfo"], self._on_caminfo, queue_size=1)
        self.rospy.Subscriber(TOPIC_LIDAR, rt["PointCloud2"], self._on_cloud, queue_size=1)

    def _on_sensors(self, msg):
        with self._lock:
            self.sensor_time = msg.sensor_time.secs + msg.sensor_time.nsecs * 1e-9
            jq = list(msg.joint_data.joint_q)
            if len(jq) >= 26:
                self.arm_q = jq[12:26]

    def _on_claw(self, msg):
        with self._lock:
            self.claw_state = list(msg.state)

    def _on_caminfo(self, msg):
        if self.np is None:
            return
        with self._lock:
            self.cam_K = self.np.array(msg.K, dtype=float).reshape(3, 3)
            self.cam_frame = msg.header.frame_id or self.cam_frame

    def _on_color(self, msg):
        if self.cv2 is None or self.np is None:
            return
        img = self.cv2.imdecode(self.np.frombuffer(msg.data, self.np.uint8), self.cv2.IMREAD_COLOR)
        if img is not None:
            with self._lock:
                self.color_bgr = img

    def _on_depth(self, msg):
        if self.cv2 is None or self.np is None:
            return
        try:
            raw = msg.data[12:] if len(msg.data) > 12 else msg.data
            depth = self.cv2.imdecode(self.np.frombuffer(raw, self.np.uint8), self.cv2.IMREAD_UNCHANGED)
            if depth is None:
                return
            depth_m = depth.astype(self.np.float32)
            if depth.dtype == self.np.uint16:
                depth_m = depth_m / 1000.0
            with self._lock:
                self.depth_m = depth_m
        except Exception as exc:
            self.rospy.logwarn_throttle(10.0, "深度解码失败: %s", exc)

    def _on_cloud(self, msg):
        with self._lock:
            self.cloud = msg

    def snapshot(self):
        with self._lock:
            return dict(
                t=self.sensor_time,
                arm_q=self.arm_q,
                claw=list(self.claw_state),
                color=None if self.color_bgr is None else self.color_bgr.copy(),
                depth=None if self.depth_m is None else self.depth_m.copy(),
                K=None if self.cam_K is None else self.cam_K.copy(),
                cam_frame=self.cam_frame,
                cloud=self.cloud,
            )


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

class Hardware(object):
    def __init__(self, rt, perception):
        self.rt = rt
        self.rospy = rt["rospy"]
        self.perception = perception
        self.cmd_vel_pub = self.rospy.Publisher(TOPIC_CMD_VEL, rt["Twist"], queue_size=10)
        self.arm_traj_pub = self.rospy.Publisher(TOPIC_ARM_TRAJ, rt["JointState"],
                                                 queue_size=10, tcp_nodelay=True)
        self.arm_target_pub = self.rospy.Publisher(TOPIC_ARM_TARGET_POSES, rt["armTargetPoses"],
                                                   queue_size=10)
        self.head_pub = self.rospy.Publisher(TOPIC_HEAD_MOTION, rt["robotHeadMotionData"],
                                             queue_size=10)
        self.tf_buffer = None
        self._tf_listener = None
        if rt["tf2_ros"] is not None:
            self.tf_buffer = rt["tf2_ros"].Buffer()
            self._tf_listener = rt["tf2_ros"].TransformListener(self.tf_buffer)
        self._srv_cache = {}
        self._arm_mode = None
        self._claw_cmd = {"left": CLAW_OPEN, "right": CLAW_OPEN}
        self.locked_left_joints = None

    def _service(self, name, srv_type, timeout=2.0):
        proxy = self._srv_cache.get(name)
        if proxy is not None:
            return proxy
        try:
            self.rospy.wait_for_service(name, timeout=timeout)
            proxy = self.rospy.ServiceProxy(name, srv_type)
            self._srv_cache[name] = proxy
            self.rospy.loginfo("服务已连接: %s", name)
            return proxy
        except Exception as exc:
            self.rospy.logwarn_throttle(10.0, "服务 %s 暂不可用(稍后重试): %s", name, exc)
            return None

    @property
    def claw_srv(self):
        return self._service(SRV_CLAW, self.rt["controlLejuClaw"])

    @property
    def ik_srv(self):
        return self._service(SRV_IK, self.rt["twoArmHandPoseCmdSrv"])

    @property
    def fk_srv(self):
        return self._service(SRV_FK, self.rt["fkSrv"])

    def stop_base(self):
        self.cmd_vel_pub.publish(self.rt["Twist"]())

    def wait_for_stable_stand(self, timeout=90.0):
        end = self.rospy.Time.now() + self.rospy.Duration(timeout)
        rate = self.rospy.Rate(5)
        stable_hits = 0
        while not self.rospy.is_shutdown() and self.rospy.Time.now() < end:
            snap = self.perception.snapshot()
            if snap["arm_q"] is not None and snap["t"] is not None:
                stable_hits += 1
                if stable_hits >= 5:
                    self.rospy.sleep(ARM_MODE_PRE_SWITCH_STAND_SETTLE)
                    return True
            else:
                stable_hits = 0
            rate.sleep()
        self.rospy.logwarn("等待站立/传感器稳定超时")
        return False

    def move_head(self, yaw_deg, pitch_deg, settle=0.8):
        msg = self.rt["robotHeadMotionData"]()
        msg.joint_data = [float(yaw_deg), float(pitch_deg)]
        self.head_pub.publish(msg)
        self.rospy.sleep(settle)
        return True

    def look_down(self):
        return self.move_head(*HEAD_LOOK_DOWN)

    def look_forward(self):
        return self.move_head(*HEAD_LOOK_FORWARD)

    def set_arm_mode(self, mode):
        if self._arm_mode == mode:
            return True
        if not self.wait_for_stable_stand(timeout=ARM_MODE_SERVICE_TIMEOUT):
            return False
        try:
            self.rospy.wait_for_service(SRV_ARM_MODE, timeout=ARM_MODE_SERVICE_TIMEOUT)
            req = self.rt["changeArmCtrlMode"]._request_class()
            req.control_mode = int(mode)
            res = self.rospy.ServiceProxy(SRV_ARM_MODE, self.rt["changeArmCtrlMode"])(req)
            if not getattr(res, "result", False):
                self.rospy.logwarn("切换手臂模式被拒绝: mode=%s message=%s",
                                   mode, getattr(res, "message", ""))
                return False
            self._arm_mode = mode
            self.rospy.loginfo("手臂模式切换成功: mode=%s message=%s",
                               mode, getattr(res, "message", ""))
            self.rospy.sleep(ARM_MODE_POST_SWITCH_SETTLE)
            return True
        except Exception as exc:
            self.rospy.logwarn("切换手臂模式失败: %s", exc)
            return False

    def ensure_external_arm_mode(self, label):
        if self.set_arm_mode(ARM_MODE_EXTERNAL):
            return True
        self.rospy.logerr("%s: 手臂模式切换失败，停止该动作", label)
        return False

    def current_arm_rad(self):
        arm_q = self.perception.snapshot()["arm_q"]
        return list(arm_q) if arm_q else [0.0] * 14

    def call_fk(self, joint_rad):
        if self.fk_srv is None:
            return None
        try:
            res = self.fk_srv(list(joint_rad))
            return res.hand_poses if res.success else None
        except self.rospy.ServiceException as exc:
            self.rospy.logerr("FK 调用失败: %s", exc)
            return None

    def current_eef_pose(self, side):
        fk = self.call_fk(self.current_arm_rad())
        if fk is None:
            return None, None
        pose = fk.left_pose if side == "left" else fk.right_pose
        return list(pose.pos_xyz), list(pose.quat_xyzw)

    def validate_ik_solution(self, side, joint_rad, target_pos, target_quat,
                             max_pos_error=None, max_ori_error_deg=None):
        fk = self.call_fk(joint_rad)
        if fk is None:
            return False
        pose = fk.left_pose if side == "left" else fk.right_pose
        pos_err = math.sqrt(sum((float(pose.pos_xyz[i]) - float(target_pos[i])) ** 2
                                for i in range(3)))
        ori_err_deg = math.degrees(quat_angle_error_rad(pose.quat_xyzw, target_quat))
        pos_ok = max_pos_error is None or pos_err <= float(max_pos_error)
        ori_ok = max_ori_error_deg is None or ori_err_deg <= float(max_ori_error_deg)
        if not (pos_ok and ori_ok):
            self.rospy.logwarn("IK 解校验失败 side=%s pos_err=%.1fmm ori_err=%.1fdeg target=%s",
                               side, pos_err * 1000.0, ori_err_deg,
                               [round(float(v), 4) for v in target_pos])
        return pos_ok and ori_ok

    def _make_ik_param(self, constraint_mode):
        param = self.rt["ikSolveParam"]()
        param.major_optimality_tol = 1e-3
        param.major_feasibility_tol = 1e-3
        param.minor_feasibility_tol = 1e-3
        param.major_iterations_limit = IK_MAJOR_ITERATIONS_LIMIT
        param.oritation_constraint_tol = 1e-3
        param.pos_constraint_tol = 1e-3
        param.pos_cost_weight = 0.0
        param.constraint_mode = constraint_mode
        return param

    def solve_ik(self, side, pos_ik, quat_xyzw, constraint_mode=None):
        if self.ik_srv is None:
            return None
        cur = self.current_arm_rad()
        fk = self.call_fk(cur)
        if fk is None:
            self.rospy.logwarn("solve_ik: FK 失败，无法锁定另一只手")
            return None
        cmd = self.rt["twoArmHandPoseCmd"]()
        cmd.use_custom_ik_param = True
        cmd.joint_angles_as_q0 = True
        cmd.ik_param = self._make_ik_param(constraint_mode or IK_MODE_THREE_POINT_MIXED)
        cmd.hand_poses.left_pose.joint_angles = cur[0:7]
        cmd.hand_poses.right_pose.joint_angles = cur[7:14]
        cmd.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        cmd.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        cmd.hand_poses.left_pose.pos_xyz = list(fk.left_pose.pos_xyz)
        cmd.hand_poses.left_pose.quat_xyzw = list(fk.left_pose.quat_xyzw)
        cmd.hand_poses.right_pose.pos_xyz = list(fk.right_pose.pos_xyz)
        cmd.hand_poses.right_pose.quat_xyzw = list(fk.right_pose.quat_xyzw)
        target = cmd.hand_poses.left_pose if side == "left" else cmd.hand_poses.right_pose
        target.pos_xyz = list(pos_ik)
        target.quat_xyzw = list(quat_xyzw)
        try:
            res = self.ik_srv(cmd)
        except self.rospy.ServiceException as exc:
            self.rospy.logerr("IK 调用失败: %s", exc)
            return None
        if not res.success:
            self.rospy.logwarn("IK 无解: %s", getattr(res, "error_reason", ""))
            return None
        lj = list(res.hand_poses.left_pose.joint_angles)
        rj = list(res.hand_poses.right_pose.joint_angles)
        left_hold = self.locked_left_joints if (
            side == "right" and self.locked_left_joints is not None
        ) else cur[0:7]
        if len(lj) == 7 and len(rj) == 7:
            return (lj + cur[7:14]) if side == "left" else (list(left_hold) + rj)
        if len(res.q_arm) >= 14:
            solved = list(res.q_arm[:14])
            return (solved[0:7] + cur[7:14]) if side == "left" else (list(left_hold) + solved[7:14])
        self.rospy.logwarn("IK 返回未含 14 关节")
        return None

    def _publish_arm_deg_frame(self, deg14):
        js = self.rt["JointState"]()
        js.header.stamp = self.rospy.Time.now()
        js.name = list(ARM_JOINT_NAMES)
        js.position = [float(x) for x in deg14]
        self.arm_traj_pub.publish(js)

    def interp_publish(self, start_deg, target_deg, move_time=None, settle=None):
        if len(start_deg) != 14 or len(target_deg) != 14:
            self.rospy.logerr("interp_publish 需要 14 维，收到 %d/%d",
                              len(start_deg), len(target_deg))
            return
        move_time = ARM_MOVE_TIME if move_time is None else move_time
        settle = ARM_SETTLE_TIME if settle is None else settle
        msg = self.rt["armTargetPoses"]()
        msg.times = [float(move_time)]
        msg.values = [float(v) for v in target_deg]
        self.arm_target_pub.publish(msg)
        steps = max(1, int(round(move_time * ARM_TRAJ_HZ)))
        rate = self.rospy.Rate(ARM_TRAJ_HZ)
        for step in range(steps + 1):
            if self.rospy.is_shutdown():
                break
            a = float(step) / float(steps)
            frame = [start_deg[i] + (target_deg[i] - start_deg[i]) * a for i in range(14)]
            self._publish_arm_deg_frame(frame)
            if step < steps:
                rate.sleep()
        self.rospy.sleep(settle)

    def move_home(self, settle=None):
        if not self.ensure_external_arm_mode("move_home"):
            return False
        prev = [math.degrees(a) for a in self.current_arm_rad()]
        for point in PRESET_POINTS_DEG:
            if self.rospy.is_shutdown():
                break
            self.interp_publish(prev, point, move_time=PRESET_SEGMENT_TIME, settle=0.0)
            prev = point
        self.rospy.sleep(PRESET_SETTLE_TIME if settle is None else settle)
        return True

    def move_eef_to(self, side, pos_ik, first_ypr, second_ypr=None,
                    constraint_mode=None, move_time=None,
                    max_pos_error=None, max_ori_error_deg=None):
        quat = quat_from_ypr(first_ypr, second_ypr)
        sol_rad = self.solve_ik(side, pos_ik, quat, constraint_mode=constraint_mode)
        if sol_rad is None:
            return False
        if not self.validate_ik_solution(side, sol_rad, pos_ik, quat,
                                         max_pos_error=max_pos_error,
                                         max_ori_error_deg=max_ori_error_deg):
            return False
        start_deg = [math.degrees(a) for a in self.current_arm_rad()]
        target_deg = [math.degrees(a) for a in sol_rad]
        self.interp_publish(start_deg, target_deg, move_time=move_time)
        return True

    def lock_current_left_joints(self):
        self.locked_left_joints = list(self.current_arm_rad()[0:7])

    def gripper(self, close, side="both", vel=50.0, effort=1.0):
        if self.claw_srv is None:
            self.rospy.logwarn_throttle(10.0, "夹爪服务缺失，跳过 gripper")
            return False
        pos = CLAW_CLOSE if close else CLAW_OPEN
        if side in ("both", "left"):
            self._claw_cmd["left"] = pos
        if side in ("both", "right"):
            self._claw_cmd["right"] = pos
        left = self._claw_cmd["left"]
        right = self._claw_cmd["right"]
        try:
            req = self.rt["controlLejuClawRequest"]()
            req.data.name = ["left_claw", "right_claw"]
            req.data.position = [left, right]
            req.data.velocity = [vel, vel]
            req.data.effort = [effort, effort]
            return bool(self.claw_srv(req).success)
        except self.rospy.ServiceException as exc:
            self.rospy.logerr("夹爪调用失败: %s", exc)
            return False

    def wait_claw_settled(self, min_state=CLAW_OPEN_MIN_STATE, timeout=3.0, side=None):
        end = self.rospy.Time.now() + self.rospy.Duration(timeout)
        rate = self.rospy.Rate(20)
        while self.rospy.Time.now() < end and not self.rospy.is_shutdown():
            claw_state = self.perception.snapshot()["claw"]
            if side == "left":
                ok = len(claw_state) >= 1 and claw_state[0] >= min_state
            elif side == "right":
                ok = len(claw_state) >= 2 and claw_state[1] >= min_state
            else:
                ok = all(s >= min_state for s in claw_state)
            if ok:
                return True
            rate.sleep()
        return False


# ---------------------------------------------------------------------------
# Vision and scene1 manipulation
# ---------------------------------------------------------------------------

class Vision(object):
    def __init__(self, rt, perception, hardware, head_cam_frame=None):
        self.rt = rt
        self.rospy = rt["rospy"]
        self.cv2 = rt["cv2"]
        self.np = rt["np"]
        self.perception = perception
        self.hardware = hardware
        self.head_cam_frame = head_cam_frame or DEFAULT_HEAD_CAM_OPTICAL_FRAME

    def _color_mask(self, hsv, color_name):
        if self.cv2 is None or self.np is None or color_name not in HSV_COLORS:
            return None
        lo, hi = HSV_COLORS[color_name]
        return self.cv2.inRange(hsv, self.np.array(lo), self.np.array(hi))

    def classify_blob_color(self, hsv, contour, color_names):
        mask_roi = self.np.zeros(hsv.shape[:2], dtype=self.np.uint8)
        self.cv2.drawContours(mask_roi, [contour], -1, 255, thickness=-1)
        best_color = None
        best_score = 0
        for color in color_names:
            color_mask = self._color_mask(hsv, color)
            if color_mask is None:
                continue
            score = int(self.cv2.countNonZero(self.cv2.bitwise_and(color_mask, mask_roi)))
            if score > best_score:
                best_color = color
                best_score = score
        return best_color, best_score

    def pixel_to_base(self, snap, u, v):
        if snap["depth"] is None or snap["K"] is None or self.hardware.tf_buffer is None:
            return None
        depth, k = snap["depth"], snap["K"]
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iv < depth.shape[0] and 0 <= iu < depth.shape[1]):
            return None
        z = float(depth[iv, iu])
        if not (0.05 < z < 5.0):
            return None
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        ps = self.rt["PointStamped"]()
        ps.header.frame_id = snap.get("cam_frame") or self.head_cam_frame
        ps.header.stamp = self.rospy.Time(0)
        ps.point.x = (iu - cx) * z / fx
        ps.point.y = (iv - cy) * z / fy
        ps.point.z = z
        try:
            out = self.hardware.tf_buffer.transform(ps, BASE_FRAME, self.rospy.Duration(1.0))
            return (out.point.x, out.point.y, out.point.z)
        except Exception as exc:
            self.rospy.logwarn_throttle(5.0, "TF 反投影失败: %s", exc)
            return None

    def blob_to_base(self, snap, contour):
        m = self.cv2.moments(contour)
        if m["m00"] == 0:
            return None
        u = int(round(m["m10"] / m["m00"]))
        v = int(round(m["m01"] / m["m00"]))
        depth = snap.get("depth")
        if depth is not None:
            half = int(PARCEL_DEPTH_WINDOW // 2)
            y0, y1 = max(0, v - half), min(depth.shape[0], v + half + 1)
            x0, x1 = max(0, u - half), min(depth.shape[1], u + half + 1)
            patch = depth[y0:y1, x0:x1]
            valid = patch[(patch > 0.05) & (patch < 5.0)]
            if valid.size:
                old = depth[v, u]
                depth[v, u] = float(self.np.median(valid))
                pos = self.pixel_to_base(snap, u, v)
                depth[v, u] = old
                return pos
        return self.pixel_to_base(snap, u, v)

    def detect_parcels(self, color_names=None):
        if self.cv2 is None or self.np is None:
            return []
        snap = self.perception.snapshot()
        if snap["color"] is None:
            return []
        color_names = tuple(color_names or SCENE1_PARCEL_COLORS)
        hsv = self.cv2.cvtColor(snap["color"], self.cv2.COLOR_BGR2HSV)
        union = self.np.zeros(hsv.shape[:2], dtype=self.np.uint8)
        for color in color_names:
            mask = self._color_mask(hsv, color)
            if mask is not None:
                union = self.cv2.bitwise_or(union, mask)
        k = int(PARCEL_DETECT_MORPH_KERNEL)
        kernel = self.np.ones((k, k), self.np.uint8)
        union = self.cv2.morphologyEx(union, self.cv2.MORPH_OPEN, kernel)
        union = self.cv2.morphologyEx(union, self.cv2.MORPH_CLOSE, kernel)
        cnts, _ = self.cv2.findContours(union, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        parcels = []
        for contour in cnts:
            area = float(self.cv2.contourArea(contour))
            if area < float(PARCEL_DETECT_MIN_AREA):
                continue
            color, score = self.classify_blob_color(hsv, contour, color_names)
            if color is None or score < float(PARCEL_MIN_AREA):
                continue
            pos = self.blob_to_base(snap, contour)
            if pos is None:
                continue
            parcels.append({"color": color, "pos": pos, "area": area, "score": score})
        parcels.sort(key=lambda item: (item["pos"][1], item["pos"][0]))
        return parcels


class Manipulation(object):
    def __init__(self, rt, perception, hardware, vision):
        self.rospy = rt["rospy"]
        self.perception = perception
        self.hw = hardware
        self.vision = vision

    @staticmethod
    def _clamp(value, lower, upper):
        return max(float(lower), min(float(upper), float(value)))

    def _pick_targets_for(self, pos_ik):
        x = float(pos_ik[0]) + float(RIGHT_PICK_XY_OFFSET[0])
        y = float(pos_ik[1]) + float(RIGHT_PICK_XY_OFFSET[1])
        if not (RIGHT_PICK_MIN_X <= x <= RIGHT_PICK_MAX_X and
                RIGHT_PICK_MIN_Y <= y <= RIGHT_PICK_MAX_Y):
            return None
        grasp_z = self._clamp(
            float(pos_ik[2]) + float(RIGHT_PICK_GRASP_Z_OFFSET),
            RIGHT_PICK_GRASP_Z_MIN,
            RIGHT_PICK_GRASP_Z_MAX,
        )
        hover_z = grasp_z + float(RIGHT_PICK_HOVER_Z_OFFSET)
        lift_z = grasp_z + float(RIGHT_PICK_LIFT_Z_OFFSET)
        return (x, y, hover_z), (x, y, grasp_z), (x, y, lift_z)

    @staticmethod
    def _claw_state_ok(claw_state, side, min_state):
        if side == "left":
            return len(claw_state) >= 1 and claw_state[0] >= min_state
        if side == "right":
            return len(claw_state) >= 2 and claw_state[1] >= min_state
        return all(s >= min_state for s in claw_state)

    def prepare_left_box_station(self):
        if not self.hw.ensure_external_arm_mode("prepare_left_box_station"):
            return False
        self.hw.gripper(close=False, side="left")
        self.hw.lock_current_left_joints()
        self.rospy.loginfo("左手保持当前安全姿态并锁定关节；先避免开局扫桌/摔倒")
        return True

    def scan_locate_all(self, color_names):
        found = {}
        self.hw.look_down()
        for yaw in HEAD_SCAN_YAWS:
            if self.rospy.is_shutdown():
                break
            self.hw.move_head(yaw, HEAD_LOOK_DOWN[1])
            self.rospy.sleep(0.5)
            for parcel in self.vision.detect_parcels(color_names):
                color = parcel["color"]
                if color in found:
                    continue
                pos = parcel["pos"]
                found[color] = pos
                self.rospy.loginfo("发现包裹 color=%s @ IK(%.3f, %.3f, %.3f) area=%.1f score=%s",
                                   color, pos[0], pos[1], pos[2],
                                   parcel["area"], parcel["score"])
        missing = [color for color in color_names if color not in found]
        if missing:
            self.rospy.logwarn("低头扫描未发现颜色: %s", ", ".join(missing))
        return sorted(found.items(), key=lambda item: (item[1][1], item[1][0]))

    def pick(self, pos_ik, side="right"):
        if not self.hw.ensure_external_arm_mode("pick"):
            return False
        targets = self._pick_targets_for(pos_ik)
        if targets is None:
            self.rospy.logwarn(
                "pick: 目标超出保守抓取工作区，跳过 raw=(%.3f, %.3f, %.3f)",
                pos_ik[0], pos_ik[1], pos_ik[2],
            )
            return False
        hover, grasp, lift = targets
        first, second = RIGHT_PICK_FIRST_YPR_DEG, RIGHT_PICK_SECOND_YPR_DEG
        self.rospy.loginfo(
            "pick: raw=(%.3f, %.3f, %.3f) hover=(%.3f, %.3f, %.3f) grasp=(%.3f, %.3f, %.3f)",
            pos_ik[0], pos_ik[1], pos_ik[2],
            hover[0], hover[1], hover[2],
            grasp[0], grasp[1], grasp[2],
        )
        self.hw.gripper(close=False, side=side)
        self.hw.look_forward()
        if not self.hw.move_eef_to(
                side, hover, first, second,
                constraint_mode=RIGHT_PICK_TRANSIT_CONSTRAINT_MODE,
                move_time=PICK_ALIGN_MOVE_TIME,
                max_pos_error=IK_VALIDATION_TRANSIT_POS_TOL,
                max_ori_error_deg=IK_VALIDATION_TRANSIT_ORI_TOL_DEG):
            self.rospy.logwarn("pick: 横移到上方 IK 无解")
            self.hw.move_home(settle=0.2)
            return False
        actual_hover, _actual_quat = self.hw.current_eef_pose(side)
        if actual_hover is None:
            self.rospy.logwarn("pick: 无法读取右手上方实际位置")
            self.hw.move_home(settle=0.2)
            return False
        vertical_lift = (float(actual_hover[0]), float(actual_hover[1]), float(lift[2]))
        self.rospy.loginfo(
            "pick: 使用实际上方点竖直下降 actual_hover=(%.3f, %.3f, %.3f)",
            actual_hover[0], actual_hover[1], actual_hover[2],
        )
        descended = False
        for z_offset in (0.0, 0.03, 0.06):
            vertical_grasp = (
                float(actual_hover[0]),
                float(actual_hover[1]),
                min(float(grasp[2]) + z_offset, float(RIGHT_PICK_GRASP_Z_MAX)),
            )
            self.rospy.loginfo(
                "pick: 尝试竖直下降 grasp=(%.3f, %.3f, %.3f)",
                vertical_grasp[0], vertical_grasp[1], vertical_grasp[2],
            )
            if self.hw.move_eef_to(
                    side, vertical_grasp, first, second,
                    constraint_mode=RIGHT_PICK_FINAL_CONSTRAINT_MODE,
                    move_time=PICK_GRASP_MOVE_TIME,
                    max_pos_error=IK_VALIDATION_FINAL_POS_TOL,
                    max_ori_error_deg=IK_VALIDATION_FINAL_ORI_TOL_DEG):
                descended = True
                break
        if not descended:
            self.rospy.logwarn("pick: 下降到抓取高度 IK 无解")
            self.hw.move_home(settle=0.2)
            return False
        self.hw.gripper(close=True, side=side)
        self.hw.wait_claw_settled(min_state=CLAW_CLOSE_MIN_STATE, side=side)
        if not self.hw.move_eef_to(
                side, vertical_lift, first, second,
                constraint_mode=RIGHT_PICK_TRANSIT_CONSTRAINT_MODE,
                move_time=PICK_LIFT_MOVE_TIME,
                max_pos_error=IK_VALIDATION_TRANSIT_POS_TOL,
                max_ori_error_deg=IK_VALIDATION_TRANSIT_ORI_TOL_DEG):
            self.rospy.logwarn("pick: 抬起阶段 IK 无解或误差过大")
            self.hw.move_home(settle=0.2)
            return False
        return self._claw_state_ok(self.perception.snapshot()["claw"], side, CLAW_CLOSE_MIN_STATE)

    def place_weigh(self, side="right"):
        if not self.hw.ensure_external_arm_mode("place_weigh"):
            return False
        wx, wy, wz = WEIGH_RELEASE_IK
        first, second = WEIGH_RELEASE_FIRST_YPR_DEG, WEIGH_RELEASE_SECOND_YPR_DEG
        self.rospy.loginfo(
            "place_weigh: 右手移动到桌面称重区 transit=(%.3f, %.3f, %.3f) release=(%.3f, %.3f, %.3f)",
            wx, wy, WEIGH_TRANSIT_IK_Z, wx, wy, wz,
        )
        if not self.hw.move_eef_to(side, (wx, wy, WEIGH_TRANSIT_IK_Z), first, second):
            self.rospy.logwarn("place_weigh: 移到称重上方 IK 无解")
            return False
        if not self.hw.move_eef_to(side, (wx, wy, wz), first, second):
            self.rospy.logwarn("place_weigh: 下降到释放点 IK 无解")
            return False
        self.hw.gripper(close=False, side=side)
        self.hw.wait_claw_settled()
        self.rospy.sleep(WEIGH_DWELL)
        return True

    def regrasp(self, side="right"):
        if not self.hw.ensure_external_arm_mode("regrasp"):
            return False
        rx, ry, rz = WEIGH_REGRASP_IK
        first, second = WEIGH_REGRASP_FIRST_YPR_DEG, WEIGH_REGRASP_SECOND_YPR_DEG
        self.rospy.loginfo(
            "regrasp: 右手从称重区二次夹取 approach=(%.3f, %.3f, %.3f) grasp=(%.3f, %.3f, %.3f)",
            rx, ry, WEIGH_RELEASE_IK[2], rx, ry, rz,
        )
        self.hw.move_eef_to(side, (rx, ry, WEIGH_RELEASE_IK[2]), first, second)
        if not self.hw.move_eef_to(side, (rx, ry, rz), first, second):
            self.rospy.logwarn("regrasp: 下降到二次夹起点 IK 无解")
            return False
        self.hw.gripper(close=True, side=side)
        self.hw.wait_claw_settled(min_state=CLAW_CLOSE_MIN_STATE)
        self.hw.move_eef_to(side, (rx, ry, WEIGH_RELEASE_IK[2]), first, second)
        return self._claw_state_ok(self.perception.snapshot()["claw"], side, CLAW_CLOSE_MIN_STATE)

    def handoff_to_left(self):
        if not self.hw.ensure_external_arm_mode("handoff"):
            return False
        self.hw.gripper(close=False, side="left")
        r_first, r_second = RIGHT_HANDOFF_TO_LEFT_FIRST_YPR_DEG, RIGHT_HANDOFF_TO_LEFT_SECOND_YPR_DEG
        hx, hy, hz = RIGHT_BOX_HANDOFF_IK
        self.rospy.loginfo(
            "handoff: 称重完成后右手送到固定左手附近 transit=(%.3f, %.3f, %.3f) handoff=(%.3f, %.3f, %.3f)",
            hx, hy, RIGHT_BOX_HANDOFF_TRANSIT_IK_Z, hx, hy, hz,
        )
        self.hw.move_eef_to("right", (hx, hy, RIGHT_BOX_HANDOFF_TRANSIT_IK_Z), r_first, r_second)
        if not self.hw.move_eef_to("right", (hx, hy, hz), r_first, r_second):
            self.rospy.logwarn("handoff: 右手到交接点 IK 无解")
            return False
        self.hw.gripper(close=True, side="left")
        self.hw.wait_claw_settled(min_state=CLAW_CLOSE_MIN_STATE, side="left")
        self.hw.gripper(close=False, side="right")
        self.rospy.sleep(GRIPPER_HOLD_TIME)
        self.hw.move_eef_to("right", (hx, hy + RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET, hz),
                            r_first, r_second)
        return True

    def drop_in_box(self, index=0):
        if not self.hw.ensure_external_arm_mode("drop_in_box"):
            return False
        self.hw.gripper(close=False, side="left")
        self.hw.wait_claw_settled(side="left")
        self.rospy.sleep(GRIPPER_HOLD_TIME)
        return True


def run_scene1(rospy, manip):
    rospy.loginfo("[scene1] 快递称重与摆放（完整流程）")
    if not manip.prepare_left_box_station():
        rospy.logerr("[scene1] 左手箱上方固定站位失败，停止任务")
        return
    remaining = set(SCENE1_PARCEL_COLORS)
    completed = []
    scan_round = 0
    while remaining and scan_round < SCENE1_MAX_SCAN_RETRIES and not rospy.is_shutdown():
        scan_round += 1
        parcels = [(color, pos) for color, pos in manip.scan_locate_all(tuple(remaining))
                   if color in remaining]
        if not parcels:
            rospy.logwarn("[scene1] 第 %d 轮未识别到剩余快递: %s",
                          scan_round, ", ".join(sorted(remaining)))
            continue
        progressed = False
        for color, pos in parcels:
            if rospy.is_shutdown() or color not in remaining:
                break
            rospy.loginfo("[scene1] 处理快递 %d: %s", len(completed) + 1, color)
            picked = False
            for attempt in range(SCENE1_PICK_ATTEMPTS):
                if manip.pick(pos, side="right"):
                    picked = True
                    break
                rospy.logwarn("[scene1] %s 第 %d 次抓取失败", color, attempt + 1)
            if not picked:
                rospy.logwarn("[scene1] %s 抓取失败，保留到下一轮重扫", color)
                continue
            rospy.loginfo("[scene1] %s 已抓取，开始移动到桌面称重区", color)
            if not manip.place_weigh(side="right"):
                rospy.logwarn("[scene1] %s 称重放置失败，保留到下一轮重扫", color)
                continue
            rospy.loginfo("[scene1] %s 已放到称重区，开始二次夹取", color)
            if not manip.regrasp(side="right"):
                rospy.logwarn("[scene1] %s 二次夹起失败，保留到下一轮重扫", color)
                continue
            rospy.loginfo("[scene1] %s 称重区二次夹起完成，开始交接给固定左手", color)
            if not manip.handoff_to_left():
                rospy.logwarn("[scene1] %s 交接失败，保留到下一轮重扫", color)
                continue
            if not manip.drop_in_box(index=len(completed)):
                rospy.logwarn("[scene1] %s 放箱失败，保留到下一轮重扫", color)
                continue
            remaining.remove(color)
            completed.append(color)
            progressed = True
            break
        if not progressed:
            rospy.logwarn("[scene1] 第 %d 轮没有完成任何快递，下一轮重扫", scan_round)
    if remaining:
        rospy.logwarn("[scene1] 未完成快递: %s", ", ".join(sorted(remaining)))


def wait_until_ready(rospy, perception, hardware, timeout=90.0):
    end = rospy.Time.now() + rospy.Duration(timeout)
    rate = rospy.Rate(2)
    ik_ok = cam_ok = sensor_ok = False
    while not rospy.is_shutdown() and rospy.Time.now() < end:
        if not ik_ok and hardware.ik_srv is not None and hardware.fk_srv is not None:
            ik_ok = True
            rospy.loginfo("就绪检测: IK/FK 服务已连接")
        snap = perception.snapshot()
        if not sensor_ok and snap["arm_q"] is not None and snap["t"] is not None:
            sensor_ok = True
            rospy.loginfo("就绪检测: /sensors_data_raw 首帧手臂关节已到达")
        if not cam_ok and snap["color"] is not None and snap["K"] is not None:
            cam_ok = True
            rospy.loginfo("就绪检测: 头相机彩色+内参已到达")
        if ik_ok and cam_ok and sensor_ok:
            rospy.sleep(ARM_MODE_PRE_SWITCH_STAND_SETTLE)
            rospy.loginfo("就绪检测: 全部就绪")
            return True
        rate.sleep()
    rospy.logwarn("就绪检测超时(IK=%s, 相机=%s, 传感器=%s)，仍继续尝试执行",
                  ik_ok, cam_ok, sensor_ok)
    return ik_ok and sensor_ok


# ---------------------------------------------------------------------------
# Launcher and CLI
# ---------------------------------------------------------------------------

def _resolve_startup_timeout(timeout):
    if timeout is not None:
        value = timeout
    else:
        raw = os.environ.get(SIM_TIMEOUT_ENV_VAR, "").strip()
        if not raw:
            return DEFAULT_SIM_STARTUP_TIMEOUT
        try:
            value = float(raw)
        except ValueError:
            print("[WARN] challenge_task: 环境变量 {}='{}' 非法，回退到默认 {:.1f}s".format(
                SIM_TIMEOUT_ENV_VAR, raw, DEFAULT_SIM_STARTUP_TIMEOUT))
            return DEFAULT_SIM_STARTUP_TIMEOUT
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SIM_STARTUP_TIMEOUT
    return value if value > 0.0 else DEFAULT_SIM_STARTUP_TIMEOUT


def _load_launcher():
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


def run_scene(scene, seed, node_name=None, timeout=None, time_limit=None, timer_gui=True):
    if scene != "scene1":
        raise ValueError("this single-file entry currently supports only scene1")
    ChallengeSimLauncher = _load_launcher()
    startup_timeout = _resolve_startup_timeout(timeout)
    launcher = ChallengeSimLauncher(scene=scene, seed=seed,
                                    match_time_limit=time_limit, timer_gui=timer_gui)
    print("[INFO] challenge_task: 仿真启动等待上限 {:.1f}s"
          "（可用 --timeout 或 {} 覆盖）".format(startup_timeout, SIM_TIMEOUT_ENV_VAR))
    launcher.start(node_name=node_name or "challenge_task_scene1", timeout=startup_timeout)
    rt = _import_runtime_modules()
    rospy = rt["rospy"]
    rospy.loginfo("=== 场景一：包裹称重与摆放任务启动 ===")
    rospy.sleep(1.0)
    perception = Perception(rt)
    hardware = Hardware(rt, perception)
    vision = Vision(rt, perception, hardware)
    manip = Manipulation(rt, perception, hardware, vision)
    if not wait_until_ready(rospy, perception, hardware, timeout=90.0):
        rospy.logerr("关键链路未就绪，停止任务")
        return
    rospy.loginfo("=== 场景一：包裹称重与摆放任务开始 ===")
    if not hardware.move_home():
        rospy.logerr("开局预设抬手失败，停止任务")
        return
    try:
        run_scene1(rospy, manip)
        rospy.loginfo("=== 场景一：包裹称重与摆放任务流程结束 ===")
    except rospy.ROSInterruptException:
        pass
    finally:
        hardware.stop_base()
    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="挑战杯 scene1 单文件任务入口")
    parser.add_argument("--scene", choices=["scene1"], default="scene1", help="只支持 scene1")
    parser.add_argument("--seed", type=int, default=0, help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None, help="ROS 节点名；默认 challenge_task_scene1")
    parser.add_argument("--timeout", type=float, default=None,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    args = parser.parse_args()
    run_scene(scene=args.scene, seed=args.seed, node_name=args.node_name,
              timeout=args.timeout, time_limit=args.time_limit,
              timer_gui=not args.no_timer_gui)


if __name__ == "__main__":
    main()
