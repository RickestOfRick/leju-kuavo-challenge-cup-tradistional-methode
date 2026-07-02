#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""硬件抽象层：行走、头部、手臂(FK+IK+插值)、夹爪的底层 ROS IO。

手臂控制对齐官方机制：调 IK 前先 FK 锁住不动的手，只解目标手；IK 解出关节角
后用 100Hz 线性插值平滑发布，避免跳变触发保护。所有服务缺失时降级为日志告警。
"""

import math

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

from . import config

try:
    from kuavo_msgs.msg import robotHeadMotionData
    _HAS_HEAD_MSG = True
except Exception:
    _HAS_HEAD_MSG = False

try:
    import tf2_ros
    _HAS_TF = True
except Exception as exc:
    rospy.logwarn("tf2 不可用，坐标变换降级: %s", exc)
    _HAS_TF = False

try:
    from kuavo_msgs.msg import (twoArmHandPoseCmd, ikSolveParam, armTargetPoses)
    from kuavo_msgs.srv import (controlLejuClaw, controlLejuClawRequest,
                                twoArmHandPoseCmdSrv, changeArmCtrlMode, fkSrv)
    _HAS_KUAVO_MSGS = True
except Exception as exc:
    rospy.logwarn("kuavo_msgs 不可用，夹爪/IK 降级为仅日志: %s", exc)
    _HAS_KUAVO_MSGS = False


# ===================== 旋转工具（照搬官方 YPR→四元数约定）=====================
def _matmul3(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _rot_to_quat_xyzw(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def quat_from_ypr(first_ypr_deg, second_ypr_deg=None):
    """两段 YPR(度)叠加成四元数 xyzw，照搬官方 _quat_from_ypr_deg 约定。"""
    yaw, pitch, _roll = [math.radians(float(v)) for v in first_ypr_deg]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    matrix = [
        [cy * cp, -sy, cy * sp],
        [sy * cp, cy, sy * sp],
        [-sp, 0.0, cp],
    ]
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


class Hardware(object):
    """封装底层控制 IO，依赖 Perception 读取当前关节角作 IK 初值/FK 输入。"""

    def __init__(self, perception):
        self.perception = perception
        self.cmd_vel_pub = rospy.Publisher(config.TOPIC_CMD_VEL, Twist, queue_size=10)
        self.arm_traj_pub = rospy.Publisher(config.TOPIC_ARM_TRAJ, JointState,
                                            queue_size=10, tcp_nodelay=True)
        self.arm_target_pub = rospy.Publisher(
            config.TOPIC_ARM_TARGET_POSES, armTargetPoses, queue_size=10) \
            if _HAS_KUAVO_MSGS else None
        self.head_pub = rospy.Publisher(
            config.TOPIC_HEAD_MOTION, robotHeadMotionData, queue_size=10) \
            if _HAS_HEAD_MSG else None

        self.tf_buffer = None
        self._tf_listener = None
        if _HAS_TF:
            self.tf_buffer = tf2_ros.Buffer()
            # 必须持有 listener 引用，否则被 GC 后 TF 缓存停止更新
            self._tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 服务代理惰性连接：仿真启动早期服务可能尚未 ready，首次用到时再连并缓存，
        # 避免开局一次性超时后永久放弃。
        self._srv_cache = {}

    def _service(self, name, srv_type, timeout=2.0):
        """惰性获取服务代理；未就绪返回 None，下次调用会重试。"""
        if not _HAS_KUAVO_MSGS:
            return None
        proxy = self._srv_cache.get(name)
        if proxy is not None:
            return proxy
        try:
            rospy.wait_for_service(name, timeout=timeout)
            proxy = rospy.ServiceProxy(name, srv_type)
            self._srv_cache[name] = proxy
            rospy.loginfo("服务已连接: %s", name)
            return proxy
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "服务 %s 暂不可用(稍后重试): %s", name, exc)
            return None

    @property
    def claw_srv(self):
        return self._service(config.SRV_CLAW, controlLejuClaw)

    @property
    def ik_srv(self):
        return self._service(config.SRV_IK, twoArmHandPoseCmdSrv)

    @property
    def fk_srv(self):
        return self._service(config.SRV_FK, fkSrv)

    # ---------------- 底盘行走 ----------------
    def stop_base(self):
        self.cmd_vel_pub.publish(Twist())

    def drive(self, vx=0.0, vy=0.0, wz=0.0, duration=1.0, rate_hz=20.0):
        """以给定速度持续 duration 秒后停下（开环）。"""
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = vx, vy, wz
        rate = rospy.Rate(rate_hz)
        for _ in range(max(1, int(duration * rate_hz))):
            if rospy.is_shutdown():
                break
            self.cmd_vel_pub.publish(tw)
            rate.sleep()
        self.stop_base()

    # ---------------- 头部 ----------------
    def move_head(self, yaw_deg, pitch_deg, settle=0.8):
        """头部偏航/俯仰(度)。yaw∈[-30,30], pitch∈[-25,25]，正 pitch 为低头。"""
        if self.head_pub is None:
            rospy.logwarn_throttle(10.0, "头部话题缺失，跳过 move_head")
            return False
        msg = robotHeadMotionData()
        msg.joint_data = [float(yaw_deg), float(pitch_deg)]
        self.head_pub.publish(msg)
        rospy.sleep(settle)
        return True

    def look_down(self):
        return self.move_head(*config.HEAD_LOOK_DOWN)

    def look_forward(self):
        return self.move_head(*config.HEAD_LOOK_FORWARD)

    # ---------------- 手臂：模式 / 关节读取 ----------------
    def set_arm_mode(self, mode):
        """切换手臂控制模式：2=external_control（IK 控制前必须）。"""
        if not _HAS_KUAVO_MSGS:
            return False
        try:
            rospy.wait_for_service(config.SRV_ARM_MODE, timeout=1.0)
            rospy.ServiceProxy(config.SRV_ARM_MODE, changeArmCtrlMode)(control_mode=mode)
            return True
        except Exception as exc:
            rospy.logwarn("切换手臂模式失败: %s", exc)
            return False

    def current_arm_rad(self):
        """当前 14 个手臂关节角(rad)，读不到时返回 14 个 0。"""
        arm_q = self.perception.snapshot()["arm_q"]
        return list(arm_q) if arm_q else [0.0] * 14

    # ---------------- 手臂：FK + IK ----------------
    def call_fk(self, joint_rad):
        """正解：给 14 关节角(rad)，返回双手当前位姿 hand_poses 或 None。"""
        if self.fk_srv is None:
            return None
        try:
            res = self.fk_srv(list(joint_rad))
            return res.hand_poses if res.success else None
        except rospy.ServiceException as exc:
            rospy.logerr("FK 调用失败: %s", exc)
            return None

    def _make_ik_param(self, constraint_mode):
        param = ikSolveParam()
        param.major_optimality_tol = 1e-3
        param.major_feasibility_tol = 1e-3
        param.minor_feasibility_tol = 1e-3
        param.major_iterations_limit = config.IK_MAJOR_ITERATIONS_LIMIT
        param.oritation_constraint_tol = 1e-3
        param.pos_constraint_tol = 1e-3
        param.pos_cost_weight = 0.0
        param.constraint_mode = constraint_mode
        return param

    def solve_ik(self, side, pos_ik, quat_xyzw, constraint_mode=None):
        """单臂 IK：先 FK 锁另一只手，只解 side 手。返回 14 维关节角(rad)或 None。

        pos_ik 为 IK 系绝对坐标，quat_xyzw 为目标朝向四元数。
        """
        if self.ik_srv is None:
            return None
        if constraint_mode is None:
            constraint_mode = config.IK_MODE_THREE_POINT_MIXED
        cur = self.current_arm_rad()
        fk = self.call_fk(cur)
        if fk is None:
            rospy.logwarn("solve_ik: FK 失败，无法锁定另一只手")
            return None

        cmd = twoArmHandPoseCmd()
        cmd.use_custom_ik_param = True
        cmd.joint_angles_as_q0 = True
        cmd.ik_param = self._make_ik_param(constraint_mode)
        cmd.hand_poses.left_pose.joint_angles = cur[0:7]
        cmd.hand_poses.right_pose.joint_angles = cur[7:14]
        cmd.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        cmd.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        # 两手先锁在当前 FK 位姿
        cmd.hand_poses.left_pose.pos_xyz = list(fk.left_pose.pos_xyz)
        cmd.hand_poses.left_pose.quat_xyzw = list(fk.left_pose.quat_xyzw)
        cmd.hand_poses.right_pose.pos_xyz = list(fk.right_pose.pos_xyz)
        cmd.hand_poses.right_pose.quat_xyzw = list(fk.right_pose.quat_xyzw)
        # 再覆盖要动的那只手
        target = cmd.hand_poses.left_pose if side == "left" else cmd.hand_poses.right_pose
        target.pos_xyz = list(pos_ik)
        target.quat_xyzw = list(quat_xyzw)

        try:
            res = self.ik_srv(cmd)
        except rospy.ServiceException as exc:
            rospy.logerr("IK 调用失败: %s", exc)
            return None
        if not res.success:
            rospy.logwarn("IK 无解: %s", getattr(res, "error_reason", ""))
            return None
        # 单臂 IK 只采用目标手的新解，另一只手保持当前关节不动。
        # 否则求解器给出的“另一手补偿解”也会被一起发布，容易把上身大幅带偏甚至摔倒。
        lj = list(res.hand_poses.left_pose.joint_angles)
        rj = list(res.hand_poses.right_pose.joint_angles)
        if len(lj) == 7 and len(rj) == 7:
            if side == "left":
                return lj + cur[7:14]
            return cur[0:7] + rj
        if len(res.q_arm) >= 14:
            solved = list(res.q_arm[:14])
            if side == "left":
                return solved[0:7] + cur[7:14]
            return cur[0:7] + solved[7:14]
        rospy.logwarn("IK 返回未含 14 关节")
        return None

    # ---------------- 手臂：100Hz 插值发布 ----------------
    def _publish_arm_deg_frame(self, deg14):
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = list(config.ARM_JOINT_NAMES)
        js.position = [float(x) for x in deg14]
        self.arm_traj_pub.publish(js)

    def interp_publish(self, start_deg, target_deg, move_time=None, settle=None):
        """从 start_deg 到 target_deg 线性插值，100Hz 平滑发布(度)。"""
        if len(start_deg) != 14 or len(target_deg) != 14:
            rospy.logerr("interp_publish 需要 14 维，收到 %d/%d",
                         len(start_deg), len(target_deg))
            return
        move_time = config.ARM_MOVE_TIME if move_time is None else move_time
        settle = config.ARM_SETTLE_TIME if settle is None else settle
        # 同时发一帧目标给 target_poses（与官方一致，双通道）
        if self.arm_target_pub is not None:
            msg = armTargetPoses()
            msg.times = [float(move_time)]
            msg.values = [float(v) for v in target_deg]
            self.arm_target_pub.publish(msg)

        steps = max(1, int(round(move_time * config.ARM_TRAJ_HZ)))
        rate = rospy.Rate(config.ARM_TRAJ_HZ)
        for step in range(steps + 1):
            if rospy.is_shutdown():
                break
            a = float(step) / float(steps)
            frame = [start_deg[i] + (target_deg[i] - start_deg[i]) * a
                     for i in range(14)]
            self._publish_arm_deg_frame(frame)
            if step < steps:
                rate.sleep()
        rospy.sleep(settle)

    def move_home(self, settle=None):
        """开局预设抬手：平滑走 PRESET_POINTS_DEG，把双臂从垂手抬到身前预设位。

        必须在任何抓取动作前调用：稳定重心防摔倒，并给后续 IK 良好初值。
        """
        self.set_arm_mode(config.ARM_MODE_EXTERNAL)
        prev = [math.degrees(a) for a in self.current_arm_rad()]
        for point in config.PRESET_POINTS_DEG:
            if rospy.is_shutdown():
                break
            self.interp_publish(prev, point,
                                move_time=config.PRESET_SEGMENT_TIME, settle=0.0)
            prev = point
        rospy.sleep(config.PRESET_SETTLE_TIME if settle is None else settle)

    def move_eef_to(self, side, pos_ik, first_ypr, second_ypr=None,
                    constraint_mode=None, move_time=None):
        """完整一步：FK 锁另一手 → IK 解目标手 → 100Hz 插值发布。成功返回 True。"""
        quat = quat_from_ypr(first_ypr, second_ypr)
        sol_rad = self.solve_ik(side, pos_ik, quat, constraint_mode=constraint_mode)
        if sol_rad is None:
            return False
        start_deg = [math.degrees(a) for a in self.current_arm_rad()]
        target_deg = [math.degrees(a) for a in sol_rad]
        self.interp_publish(start_deg, target_deg, move_time=move_time)
        return True

    # ---------------- 夹爪 ----------------
    def gripper(self, close, side="both", vel=50.0, effort=1.0):
        """夹爪开合：close=True 闭合，False 张开。"""
        if self.claw_srv is None:
            rospy.logwarn_throttle(10.0, "夹爪服务缺失，跳过 gripper")
            return False
        pos = config.CLAW_CLOSE if close else config.CLAW_OPEN
        left = pos if side in ("both", "left") else config.CLAW_OPEN
        right = pos if side in ("both", "right") else config.CLAW_OPEN
        try:
            req = controlLejuClawRequest()
            req.data.name = ["left_claw", "right_claw"]
            req.data.position = [left, right]
            req.data.velocity = [vel, vel]
            req.data.effort = [effort, effort]
            return bool(self.claw_srv(req).success)
        except rospy.ServiceException as exc:
            rospy.logerr("夹爪调用失败: %s", exc)
            return False

    def wait_claw_settled(self, timeout=3.0):
        """等夹爪到位/抓到(state>=2)，超时返回 False。"""
        end = rospy.Time.now() + rospy.Duration(timeout)
        rate = rospy.Rate(20)
        while rospy.Time.now() < end and not rospy.is_shutdown():
            if all(s >= config.CLAW_OPEN_MIN_STATE for s in self.perception.snapshot()["claw"]):
                return True
            rate.sleep()
        return False
