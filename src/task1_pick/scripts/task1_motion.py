#!/usr/bin/env python3
"""任务一的视觉、机器人控制、IK 与运动原语。

所有可调参数均位于 task1_config.py，本模块只实现动作逻辑。
"""

import datetime
import math
import threading
import time

import rospy
from kuavo_msgs.msg import (
    armTargetPoses,
    ikSolveParam,
    robotHeadMotionData,
    twoArmHandPoseCmd,
)
from kuavo_msgs.srv import (
    changeArmCtrlMode,
    changeArmCtrlModeRequest,
    fkSrv,
    twoArmHandPoseCmdSrv,
)
from sensor_msgs.msg import JointState

import task1_perception as vision
from task1_config import (
    ARM_JOINT_NAMES,
    ARM_STABLE_DURATION,
    ARM_STABLE_MAX_SPEED_DEG_S,
    ARM_STABLE_TIMEOUT,
    ARM_TRAJ_HZ,
    BOX_DROP_IK,
    BOX_DROP_OFFSETS,
    BOX_DROP_X_FALLBACK,
    BOX_SAFE_INTERMEDIATE_COUNT,
    BOX_SAFE_INTERMEDIATE_STEP,
    BOX_TRANSIT_MOVE_TIME,
    GRIPPER_HOLD,
    HANDOFF_FALLBACK_ZS,
    HANDOFF_POS_COST_WEIGHT,
    HANDOFF_TRANSIT_Z,
    IK_MAJOR_ITERATIONS_LIMIT,
    IK_MODE_POS_HARD_ORI_HARD,
    IK_MODE_THREE_POINT_MIXED,
    IK_POS_COST_WEIGHT,
    LEFT_RECEIVE_IK_OFFSET,
    LEFT_RECEIVE_READY,
    LIFT_Z,
    PICK_DESCENT_MID_RATIO,
    PICK_OFFSET,
    PICK_RETRACT_X_OFFSET,
    PRESET_POINTS_DEG,
    PRESET_SEGMENT_TIME,
    PRESET_SETTLE_TIME,
    REGRASP_LIFT_Z,
    RIGHT_HANDOFF_IK,
    RIGHT_PICK_IK_Z,
    RIGHT_RETRACT_Y_OFFSET,
    SETTLE_TIME,
    SOFT_IK_BISECT_ATTEMPTS,
    STRICT_IK_MAJOR_ITERATIONS_LIMIT,
    STRICT_IK_ORIENTATION_TOL,
    STRICT_IK_POSITION_TOL,
    STRICT_PATH_MAX_JOINT_STEP_DEG,
    STRICT_R6_IK_POSITION_TOL,
    STRICT_R6_PATH_INTERMEDIATE_POINTS,
    STRICT_TRAJECTORY_MOVE_TIME,
    TOPIC_TIMEOUT,
    TRACE_FILE,
    WEIGH_DWELL,
    WEIGH_DEBUG_FIXED_RELEASE_ENABLED,
    WEIGH_DEBUG_FIXED_RELEASE_TIME,
    WEIGH_FINAL_DESCEND_MOVE_TIME,
    WEIGH_REGRASP_R6_OFFSET,
    WEIGH_REGRASP_USE_STRICT_R6_IK,
    WEIGH_REGRASP_XY,
    WEIGH_RELEASE_IK,
    WEIGH_TRANSIT_Z,
    YZ_ALIGN_X_SAFE,
)

GREEN = "\033[32m"
RESET = "\033[0m"
_TASK1_MOTION_CONTEXT = threading.local()

# =============================================================================
# 工具
# =============================================================================


def _fmt(v, d=4):
    return " ".join(f"{x:.{d}f}" for x in v)


def _register_task1_motion_context(arm_pub, arm_hold):
    """为当前任务线程保存 task1 自己的运动发布上下文。"""
    _TASK1_MOTION_CONTEXT.arm_pub = arm_pub
    _TASK1_MOTION_CONTEXT.arm_hold = arm_hold


def _default_soft_ik_temporary_executor(side, label):
    arm_pub = getattr(_TASK1_MOTION_CONTEXT, "arm_pub", None)
    arm_hold = getattr(_TASK1_MOTION_CONTEXT, "arm_hold", None)
    if arm_pub is None or arm_hold is None:
        raise RuntimeError(
            "task1 motion context is not initialized before soft IK fallback"
        )
    return _make_soft_ik_temporary_executor(
        arm_pub,
        arm_hold,
        side,
        label,
        move_time=0.5,
        settle=SETTLE_TIME,
    )


def _q_from_ypr(ypr1, ypr2=None):
    ya, pi, _ = [math.radians(v) for v in ypr1]
    cy, sy, cp, sp = math.cos(ya), math.sin(ya), math.cos(pi), math.sin(pi)
    m = [[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0.0, cp]]
    if ypr2:
        my, mp, mr = [math.radians(v) for v in ypr2]
        mm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if abs(my) > 0.01:
            c, s = math.cos(my), math.sin(my)
            mm = _mmul([[c, -s, 0], [s, c, 0], [0, 0, 1]], mm)
        if abs(mp) > 0.01:
            c, s = math.cos(mp), math.sin(mp)
            mm = _mmul([[c, 0, s], [0, 1, 0], [-s, 0, c]], mm)
        if abs(mr) > 0.01:
            c, s = math.cos(mr), math.sin(mr)
            mm = _mmul([[1, 0, 0], [0, c, -s], [0, s, c]], mm)
        m = _mmul(m, mm)
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        w = math.sqrt(t + 1) / 2
        x = (m[2][1] - m[1][2]) / (4 * w)
        y = (m[0][2] - m[2][0]) / (4 * w)
        z = (m[1][0] - m[0][1]) / (4 * w)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        x = math.sqrt(max(0, 1 + m[0][0] - m[1][1] - m[2][2])) / 2
        y = (m[0][1] + m[1][0]) / (4 * x)
        z = (m[0][2] + m[2][0]) / (4 * x)
        w = (m[2][1] - m[1][2]) / (4 * x)
    elif m[1][1] > m[2][2]:
        y = math.sqrt(max(0, 1 + m[1][1] - m[0][0] - m[2][2])) / 2
        x = (m[0][1] + m[1][0]) / (4 * y)
        z = (m[1][2] + m[2][1]) / (4 * y)
        w = (m[0][2] - m[2][0]) / (4 * y)
    else:
        z = math.sqrt(max(0, 1 + m[2][2] - m[0][0] - m[1][1])) / 2
        x = (m[0][2] + m[2][0]) / (4 * z)
        y = (m[1][2] + m[2][1]) / (4 * z)
        w = (m[1][0] - m[0][1]) / (4 * z)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return [x / n, y / n, z / n, w / n] if n > 0 else [0, 0, 0, 1]


def _mmul(a, b):
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]


def _rad2deg(pt):
    return [math.degrees(v) for v in pt]


def _axis_err(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# =============================================================================
# 抓取目标 / 安全对齐
# =============================================================================

# 将视觉输出的 base_link 相对目标点转换成 IK 服务使用的抓取点。
def compute_pick_ik(parcel_name, parcel_position):
    off = PICK_OFFSET.get(parcel_name, PICK_OFFSET["parcel_1"])
    vision_to_ik_offset = vision.get_vision_to_ik_offset()
    ik = [
        vision_to_ik_offset[0] + parcel_position[0],
        vision_to_ik_offset[1] + parcel_position[1],
        vision_to_ik_offset[2] + parcel_position[2],
    ]
    return [ik[0] + off[0], ik[1] + off[1], RIGHT_PICK_IK_Z + off[2]]


def compute_yz_align(pick_pre_ik, cur_r):
    x = (
        min(pick_pre_ik[0], max(cur_r[0], YZ_ALIGN_X_SAFE))
        if cur_r[0] <= pick_pre_ik[0]
        else max(pick_pre_ik[0], min(cur_r[0], YZ_ALIGN_X_SAFE))
    )
    return [x, pick_pre_ik[1], max(cur_r[2], pick_pre_ik[2])]


# =============================================================================
# 夹爪 / 手臂保持器
# =============================================================================


class GripperHold:
    def __init__(self, pub):
        self._pub = pub
        self._lc, self._rc = 0.0, 0.0
        self._lock = threading.Lock()
        self._ev = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._ev.set()
        self._t.join(timeout=1)

    def set_left_open(self):
        with self._lock:
            self._lc = 0.0

    def set_left_close(self):
        with self._lock:
            self._lc = 255.0

    def set_right_open(self):
        with self._lock:
            self._rc = 0.0

    def set_right_close(self):
        with self._lock:
            self._rc = 255.0

    def _run(self):
        r = rospy.Rate(100)
        while not self._ev.is_set() and not rospy.is_shutdown():
            with self._lock:
                lc, rc = self._lc, self._rc
            m = JointState()
            m.header.stamp = rospy.Time.now()
            m.name = ["left_gripper_joint", "right_gripper_joint"]
            m.position = [lc, rc]
            self._pub.publish(m)
            r.sleep()


class ArmHold:
    def __init__(self, pub, deg):
        self._pub = pub
        self._deg = [float(v) for v in deg]
        self._lock = threading.Lock()
        self._ev = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._ev.set()
        self._t.join(timeout=1)

    def set_degrees(self, deg):
        with self._lock:
            self._deg = [float(v) for v in deg]

    def _run(self):
        r = rospy.Rate(100)
        while not self._ev.is_set() and not rospy.is_shutdown():
            with self._lock:
                d = list(self._deg)
            m = JointState()
            m.header.stamp = rospy.Time.now()
            m.name = ARM_JOINT_NAMES
            m.position = d
            self._pub.publish(m)
            r.sleep()


# =============================================================================
# ROS 原语
# =============================================================================


def _wait_conn(pub, timeout=TOPIC_TIMEOUT):
    t0 = time.time()
    while pub.get_num_connections() == 0 and time.time() - t0 < timeout:
        if rospy.is_shutdown():
            return
        rospy.sleep(0.2)
    if pub.get_num_connections() == 0:
        raise RuntimeError(f"topic {pub.name} no sub")


def read_arm_joints(timeout=TOPIC_TIMEOUT):
    from kuavo_msgs.msg import sensorsData

    msg = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=timeout)
    return _arm_values(msg.joint_data.joint_q, "joint_q")


def _arm_values(values, value_name):
    values = list(values)
    if len(values) >= 27:
        return values[13:27]
    if len(values) >= 26:
        return values[12:26]
    raise RuntimeError(f"{value_name} len={len(values)}")


def wait_arm_stable(
    stable_duration=ARM_STABLE_DURATION,
    max_speed_deg_s=ARM_STABLE_MAX_SPEED_DEG_S,
    timeout=ARM_STABLE_TIMEOUT,
    side="both",
):
    from kuavo_msgs.msg import sensorsData

    if side not in ("left", "right", "both"):
        raise ValueError("stable check side must be left, right, or both")
    velocity_slice = {
        "left": slice(0, 7),
        "right": slice(7, 14),
        "both": slice(0, 14),
    }[side]
    stable_since = None
    start_sim_time = rospy.get_time()
    latest_q = None
    latest_speed = float("inf")
    while not rospy.is_shutdown():
        now = rospy.get_time()
        elapsed = max(0.0, now - start_sim_time)
        if elapsed >= float(timeout):
            break
        try:
            msg = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=1.0)
            latest_q = _arm_values(msg.joint_data.joint_q, "joint_q")
            arm_v = _arm_values(msg.joint_data.joint_v, "joint_v")[velocity_slice]
        except Exception:
            continue
        latest_speed = max(abs(math.degrees(value)) for value in arm_v)
        now = rospy.get_time()
        if latest_speed <= float(max_speed_deg_s):
            if stable_since is None:
                stable_since = now
            if now - stable_since >= float(stable_duration):
                rospy.loginfo(
                    "task1: %s arm stable for %.2fs simulation time, max speed=%.3f deg/s",
                    side,
                    stable_duration,
                    latest_speed,
                )
                return latest_q
        else:
            stable_since = None
    if latest_q is None:
        raise RuntimeError(
            "{} arm stability check received no valid joint state within {:.1f}s "
            "simulation time".format(side, timeout)
        )
    rospy.logwarn(
        "task1: %s arm stability check timed out: did not remain below %.3f deg/s "
        "for %.2fs within %.1fs simulation time; last max speed=%.3f deg/s; "
        "continuing with latest joint state",
        side,
        max_speed_deg_s,
        stable_duration,
        timeout,
        latest_speed,
    )
    return latest_q


def call_fk(joints, timeout=TOPIC_TIMEOUT):
    rospy.wait_for_service("/ik/fk_srv", timeout=timeout)
    resp = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(joints)
    if not resp.success:
        raise RuntimeError("/ik/fk_srv failed")
    return resp.hand_poses


def _pose_for_side(fk, side):
    return fk.left_pose if side == "left" else fk.right_pose


def _solve_ik_one_hand_once(
    cur_joints,
    side,
    pos,
    quat,
    constraint_mode=IK_MODE_THREE_POINT_MIXED,
    pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
    major_iterations_limit=IK_MAJOR_ITERATIONS_LIMIT,
):
    fk = call_fk(cur_joints, TOPIC_TIMEOUT)
    req = twoArmHandPoseCmd()
    req.use_custom_ik_param = True
    req.joint_angles_as_q0 = True
    p = ikSolveParam()
    p.major_optimality_tol = 1e-3
    p.major_feasibility_tol = 1e-3
    p.minor_feasibility_tol = 1e-3
    p.major_iterations_limit = major_iterations_limit
    p.oritation_constraint_tol = 1e-3
    p.pos_constraint_tol = 1e-3
    p.pos_cost_weight = pos_cost_weight
    p.constraint_mode = constraint_mode
    req.ik_param = p
    req.hand_poses.left_pose.joint_angles = list(cur_joints[:7])
    req.hand_poses.right_pose.joint_angles = list(cur_joints[7:])
    req.hand_poses.left_pose.elbow_pos_xyz = [0, 0, 0]
    req.hand_poses.right_pose.elbow_pos_xyz = [0, 0, 0]
    target = req.hand_poses.left_pose if side == "left" else req.hand_poses.right_pose
    target.pos_xyz = list(pos)
    target.quat_xyzw = list(quat)
    other = req.hand_poses.right_pose if side == "left" else req.hand_poses.left_pose
    other_pose = fk.right_pose if side == "left" else fk.left_pose
    other.pos_xyz = list(other_pose.pos_xyz)
    other.quat_xyzw = list(other_pose.quat_xyzw)
    rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=TOPIC_TIMEOUT)
    resp = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(
        req
    )
    if not resp.success:
        raise RuntimeError(f"IK failed: {getattr(resp,'error_reason','')}")
    lr = list(resp.hand_poses.left_pose.joint_angles)
    rr = list(resp.hand_poses.right_pose.joint_angles)
    if len(lr) == 7 and len(rr) == 7:
        return lr + rr
    if len(resp.q_arm) >= 14:
        return list(resp.q_arm[:14])
    raise RuntimeError("IK no 14 joints")


def _is_ik_failure(exc):
    return str(exc).startswith("IK ")


def _solve_soft_ik_path(
    cur_joints,
    side,
    pos,
    quat,
    pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
    major_iterations_limit=IK_MAJOR_ITERATIONS_LIMIT,
    fallback_label=None,
    execute_temporary=None,
):
    """返回软 IK 关节路径；直接失败时插入逐级二分的临时末端点。"""
    try:
        return [
            _solve_ik_one_hand_once(
                cur_joints,
                side,
                pos,
                quat,
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=pos_cost_weight,
                major_iterations_limit=major_iterations_limit,
            )
        ]
    except RuntimeError as exc:
        if not _is_ik_failure(exc):
            raise
        last_error = exc

    context = fallback_label or "{} target={}".format(side, _fmt(pos))
    current_pose = _pose_for_side(
        call_fk(cur_joints, TOPIC_TIMEOUT), side
    )
    current_pos = [float(value) for value in current_pose.pos_xyz]
    rospy.logwarn(
        "task1: [SOFT IK FALLBACK] %s direct soft IK failed (%s); "
        "trying temporary Cartesian points from current=%s",
        context,
        last_error,
        _fmt(current_pos),
    )

    if execute_temporary is None:
        execute_temporary = _default_soft_ik_temporary_executor(side, context)

    if execute_temporary is not None:
        recovery_round = 0
        while not rospy.is_shutdown():
            recovery_round += 1
            temporary_solution = None
            temporary_denominator = None
            for attempt in range(1, SOFT_IK_BISECT_ATTEMPTS + 1):
                fraction = 0.5**attempt
                temporary_pos = [
                    current_pos[axis]
                    + fraction * (pos[axis] - current_pos[axis])
                    for axis in range(3)
                ]
                rospy.logwarn(
                    "task1: [SOFT IK FALLBACK] %s recovery round %d, "
                    "temporary attempt %d/%d fraction=1/%d target=%s",
                    context,
                    recovery_round,
                    attempt,
                    SOFT_IK_BISECT_ATTEMPTS,
                    2**attempt,
                    _fmt(temporary_pos),
                )
                try:
                    temporary_solution = _solve_ik_one_hand_once(
                        cur_joints,
                        side,
                        temporary_pos,
                        quat,
                        constraint_mode=IK_MODE_THREE_POINT_MIXED,
                        pos_cost_weight=pos_cost_weight,
                        major_iterations_limit=major_iterations_limit,
                    )
                    temporary_denominator = 2**attempt
                    break
                except RuntimeError as exc:
                    if not _is_ik_failure(exc):
                        raise
                    last_error = exc
                    rospy.logwarn(
                        "task1: [SOFT IK FALLBACK] %s temporary fraction=1/%d "
                        "IK failed: %s",
                        context,
                        2**attempt,
                        exc,
                    )

            if temporary_solution is None:
                raise RuntimeError(
                    "{} soft IK fallback failed: all {} temporary points "
                    "failed in recovery round {}".format(
                        context,
                        SOFT_IK_BISECT_ATTEMPTS,
                        recovery_round,
                    )
                ) from last_error

            rospy.logwarn(
                "task1: [SOFT IK FALLBACK] %s fraction=1/%d IK succeeded; "
                "executing temporary q before retrying the original target",
                context,
                temporary_denominator,
            )
            cur_joints = list(
                execute_temporary(temporary_solution, temporary_denominator)
            )
            current_pose = _pose_for_side(
                call_fk(cur_joints, TOPIC_TIMEOUT), side
            )
            current_pos = [float(value) for value in current_pose.pos_xyz]
            rospy.logwarn(
                "task1: [SOFT IK FALLBACK] %s temporary q executed and "
                "stable at FK=%s; retrying original target=%s from actual q",
                context,
                _fmt(current_pos),
                _fmt(pos),
            )
            try:
                return [
                    _solve_ik_one_hand_once(
                        cur_joints,
                        side,
                        pos,
                        quat,
                        constraint_mode=IK_MODE_THREE_POINT_MIXED,
                        pos_cost_weight=pos_cost_weight,
                        major_iterations_limit=major_iterations_limit,
                    )
                ]
            except RuntimeError as exc:
                if not _is_ik_failure(exc):
                    raise
                last_error = exc
                rospy.logwarn(
                    "task1: [SOFT IK FALLBACK] %s original target retry "
                    "failed after recovery round %d; restarting from actual FK: %s",
                    context,
                    recovery_round,
                    exc,
                )

        raise RuntimeError(
            "{} soft IK fallback stopped during ROS shutdown".format(context)
        ) from last_error

    raise RuntimeError("{} soft IK fallback has no executor".format(context))


def solve_ik_one_hand(
    cur_joints,
    side,
    pos,
    quat,
    constraint_mode=IK_MODE_THREE_POINT_MIXED,
    pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
    major_iterations_limit=IK_MAJOR_ITERATIONS_LIMIT,
    fallback_label=None,
    execute_temporary=None,
):
    if constraint_mode == IK_MODE_THREE_POINT_MIXED:
        return _solve_soft_ik_path(
            cur_joints,
            side,
            pos,
            quat,
            pos_cost_weight=pos_cost_weight,
            major_iterations_limit=major_iterations_limit,
            fallback_label=fallback_label,
            execute_temporary=execute_temporary,
        )[-1]
    return _solve_ik_one_hand_once(
        cur_joints,
        side,
        pos,
        quat,
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
        major_iterations_limit=major_iterations_limit,
    )


def solve_ik_one_hand_strict_multi(
    cur_joints,
    side,
    pos,
    quat,
    use_right_r6_offset=False,
    right_r6_offset=None,
    right_q7_lock_value=None,
    execute_temporary=None,
):
    """优先调用严格多参考 IK，失败时仅将求解器降级为软 IK。"""
    if right_r6_offset is None:
        right_r6_offset = [0.0, 0.0, 0.0]
    if len(right_r6_offset) != 3:
        raise ValueError("right_r6_offset must contain [x, y, z]")
    if use_right_r6_offset and side != "right":
        raise ValueError("right-r6 strict IK is only valid for the right arm")
    if use_right_r6_offset and right_q7_lock_value is None:
        raise ValueError("right-r6 strict IK requires right_q7_lock_value")

    fk = call_fk(cur_joints, TOPIC_TIMEOUT)
    req = twoArmHandPoseCmd()
    req.frame = 2
    req.use_custom_ik_param = True
    req.joint_angles_as_q0 = True

    params = ikSolveParam()
    params.major_optimality_tol = 1e-3
    params.major_feasibility_tol = 1e-3
    params.minor_feasibility_tol = 1e-3
    params.major_iterations_limit = STRICT_IK_MAJOR_ITERATIONS_LIMIT
    params.oritation_constraint_tol = STRICT_IK_ORIENTATION_TOL
    params.pos_constraint_tol = (
        STRICT_R6_IK_POSITION_TOL if use_right_r6_offset else STRICT_IK_POSITION_TOL
    )
    params.pos_cost_weight = 0.0
    params.constraint_mode = IK_MODE_POS_HARD_ORI_HARD
    params.use_right_r6_offset = bool(use_right_r6_offset)
    params.lock_right_q7 = bool(use_right_r6_offset)
    params.right_q7_lock_value = (
        float(right_q7_lock_value) if use_right_r6_offset else 0.0
    )
    params.right_r6_offset = [float(value) for value in right_r6_offset]
    req.ik_param = params

    req.hand_poses.left_pose.pos_xyz = list(fk.left_pose.pos_xyz)
    req.hand_poses.left_pose.quat_xyzw = list(fk.left_pose.quat_xyzw)
    req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    req.hand_poses.left_pose.joint_angles = list(cur_joints[:7])
    req.hand_poses.right_pose.pos_xyz = list(fk.right_pose.pos_xyz)
    req.hand_poses.right_pose.quat_xyzw = list(fk.right_pose.quat_xyzw)
    req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    req.hand_poses.right_pose.joint_angles = list(cur_joints[7:14])

    if not use_right_r6_offset:
        target = (
            req.hand_poses.left_pose if side == "left" else req.hand_poses.right_pose
        )
        target.pos_xyz = list(pos)
        target.quat_xyzw = list(quat)

    service_name = "/ik/two_arm_hand_pose_cmd_srv_muli_refer"
    strict_error = ""
    try:
        rospy.wait_for_service(service_name, timeout=TOPIC_TIMEOUT)
        response = rospy.ServiceProxy(service_name, twoArmHandPoseCmdSrv)(req)
        if response.success and len(response.q_arm) >= 14:
            return list(response.q_arm[:14])
        strict_error = getattr(response, "error_reason", "") or "no 14-joint result"
    except (rospy.ROSException, rospy.ServiceException) as exc:
        strict_error = str(exc) or type(exc).__name__

    if use_right_r6_offset:
        # r6 与末端是固定连接；保持当前姿态时，两者的世界坐标平移量相同。
        soft_pos = [
            float(fk.right_pose.pos_xyz[index]) + float(right_r6_offset[index])
            for index in range(3)
        ]
        soft_quat = list(fk.right_pose.quat_xyzw)
        rospy.logwarn(
            "task1: [STRICT R6 FALLBACK] hard IK failed (%s); this waypoint "
            "uses soft EEF IK 0x06 target=%s, MPC+WBC remains enabled, "
            "q7 is no longer guaranteed to stay locked",
            strict_error,
            _fmt(soft_pos),
        )
        return solve_ik_one_hand(
            cur_joints,
            "right",
            soft_pos,
            soft_quat,
            constraint_mode=IK_MODE_THREE_POINT_MIXED,
            execute_temporary=execute_temporary,
        )

    rospy.logwarn(
        "task1: %s hard IK failed (%s); fallback to soft IK 0x06 for target=%s; "
        "trajectory control remains MPC+WBC",
        side,
        strict_error,
        _fmt(pos),
    )
    return solve_ik_one_hand(
        cur_joints,
        side,
        pos,
        quat,
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        execute_temporary=execute_temporary,
    )


def solve_right_r6_offset_path(
    start_joints, offset_xyz, locked_q7, execute_temporary=None
):
    """优先锁定 q7 求解 r6 路径；严格 IK 失败的路径点允许软 IK 降级。"""
    offset_xyz = [float(value) for value in offset_xyz]
    if len(offset_xyz) != 3:
        raise ValueError("r6 path offset must contain [x, y, z]")
    waypoint_count = STRICT_R6_PATH_INTERMEDIATE_POINTS + 1
    origin_joints = list(start_joints)
    previous = list(start_joints)
    path = []

    for index in range(waypoint_count):
        cumulative_offset = [
            value * (index + 1) / waypoint_count for value in offset_xyz
        ]
        solved = solve_ik_one_hand_strict_multi(
            origin_joints,
            "right",
            None,
            None,
            use_right_r6_offset=True,
            right_r6_offset=cumulative_offset,
            right_q7_lock_value=locked_q7,
            execute_temporary=execute_temporary,
        )
        max_joint_step_deg = max(
            abs(math.degrees(solved[joint] - previous[joint])) for joint in range(14)
        )
        if max_joint_step_deg > STRICT_PATH_MAX_JOINT_STEP_DEG:
            rospy.logwarn(
                "task1: strict r6 waypoint %d/%d branch jump: %.1f deg > %.1f deg; continuing",
                index + 1,
                waypoint_count,
                max_joint_step_deg,
                STRICT_PATH_MAX_JOINT_STEP_DEG,
            )
        rospy.loginfo(
            "task1: strict r6 waypoint %d/%d offset=%s max joint step=%.1f deg",
            index + 1,
            waypoint_count,
            _fmt(cumulative_offset, 3),
            max_joint_step_deg,
        )
        path.append(solved)
        previous = solved
    return path


def solve_right_eef_offset_path_soft(
    start_joints, offset_xyz, execute_temporary=None
):
    """按原累计偏移生成 EEF 软 IK 调试路径，不锁定 q7。"""
    offset_xyz = [float(value) for value in offset_xyz]
    if len(offset_xyz) != 3:
        raise ValueError("EEF path offset must contain [x, y, z]")

    origin_joints = list(start_joints)
    origin_pose = call_fk(origin_joints, TOPIC_TIMEOUT).right_pose
    origin_pos = [float(value) for value in origin_pose.pos_xyz]
    target_quat = [float(value) for value in origin_pose.quat_xyzw]
    waypoint_count = STRICT_R6_PATH_INTERMEDIATE_POINTS + 1
    previous = list(origin_joints)
    path = []

    for index in range(waypoint_count):
        cumulative_offset = [
            value * (index + 1) / waypoint_count for value in offset_xyz
        ]
        target_pos = [
            origin_pos[axis] + cumulative_offset[axis] for axis in range(3)
        ]
        solved = solve_ik_one_hand(
            previous,
            "right",
            target_pos,
            target_quat,
            constraint_mode=IK_MODE_THREE_POINT_MIXED,
            execute_temporary=execute_temporary,
        )
        max_joint_step_deg = max(
            abs(math.degrees(solved[joint] - previous[joint])) for joint in range(14)
        )
        if max_joint_step_deg > STRICT_PATH_MAX_JOINT_STEP_DEG:
            rospy.logwarn(
                "task1: soft EEF waypoint %d/%d branch jump: %.1f deg > %.1f deg; continuing",
                index + 1,
                waypoint_count,
                max_joint_step_deg,
                STRICT_PATH_MAX_JOINT_STEP_DEG,
            )
        rospy.loginfo(
            "task1: soft EEF waypoint %d/%d target=%s offset=%s q7=%.3f deg",
            index + 1,
            waypoint_count,
            _fmt(target_pos),
            _fmt(cumulative_offset, 3),
            math.degrees(solved[13]),
        )
        path.append(solved)
        previous = solved
    return path


def set_wbc_direct_mode(enabled):
    service_name = "/enable_wbc_arm_trajectory_control"
    rospy.wait_for_service(service_name, timeout=TOPIC_TIMEOUT)
    request = changeArmCtrlModeRequest()
    request.control_mode = 1 if enabled else 0
    response = rospy.ServiceProxy(service_name, changeArmCtrlMode)(request)
    if not response.result:
        raise RuntimeError(
            "failed to {} WBC direct mode: {}".format(
                "enable" if enabled else "disable", response.message
            )
        )
    rospy.loginfo(
        "task1: WBC direct mode %s",
        "enabled" if enabled else "disabled for pillar trajectory",
    )


def execute_pillar_strict_trajectory(
    arm_pub,
    arm_hold,
    target_frames_rad,
    side,
    label,
    step_time=STRICT_TRAJECTORY_MOVE_TIME,
):
    """通过标准轨迹通道发布累计的严格 IK 路径点。"""
    if not target_frames_rad:
        raise ValueError("strict trajectory requires at least one frame")
    target_frames_deg = [_rad2deg(frame) for frame in target_frames_rad]
    arm_hold.set_degrees(target_frames_deg[-1])

    message = armTargetPoses()
    message.times = [
        float(step_time) * (index + 1) for index in range(len(target_frames_deg))
    ]
    message.values = [float(value) for frame in target_frames_deg for value in frame]
    message.frame = 2
    arm_pub.publish(message)
    rospy.sleep(message.times[-1])
    rospy.loginfo(
        "task1: [%s] pillar strict trajectory completed (%s, %d frames)",
        label,
        side,
        len(target_frames_deg),
    )
    return target_frames_deg[-1]


def execute_pillar_strict_frame(
    arm_pub,
    arm_hold,
    target_rad,
    side,
    label,
    move_time=STRICT_TRAJECTORY_MOVE_TIME,
):
    return execute_pillar_strict_trajectory(
        arm_pub,
        arm_hold,
        [target_rad],
        side,
        label,
        step_time=move_time,
    )


# =============================================================================
# 持续广播运动的 execute_motion_v2 — FK 闭环修正
# =============================================================================


def _to_deg_14(joints_rad):
    return _rad2deg(joints_rad[:14])


def _interpolate_q_target(
    arm_pub, arm_hold, start_deg, target_deg, move_time
):
    """通过任务一标准 WBC 通道执行一段 14 关节线性插值。"""
    previous = [float(value) for value in start_deg]
    target_deg = [float(value) for value in target_deg]
    segment_count = max(1, int(float(move_time) / 0.1))
    segment_time = float(move_time) / segment_count
    leg_start = list(previous)
    for segment in range(1, segment_count + 1):
        alpha = float(segment) / segment_count
        endpoint = [
            leg_start[index]
            + (target_deg[index] - leg_start[index]) * alpha
            for index in range(14)
        ]
        message = armTargetPoses()
        message.times = [segment_time]
        message.values = [float(value) for value in endpoint]
        arm_pub.publish(message)

        steps = max(1, int(segment_time * 100.0))
        for step in range(steps + 1):
            ratio = float(step) / steps
            command = [
                previous[index]
                + (endpoint[index] - previous[index]) * ratio
                for index in range(14)
            ]
            arm_hold.set_degrees(command)
            rospy.sleep(0.01)
        previous = endpoint
    return target_deg


def _make_soft_ik_temporary_executor(
    arm_pub,
    arm_hold,
    side,
    label,
    move_time=0.5,
    settle=0.05,
    locked_right_deg=None,
):
    """创建 task1 自用的 fallback 临时 q 执行器。"""
    locked_right_deg = (
        None
        if locked_right_deg is None
        else [float(value) for value in locked_right_deg]
    )
    if locked_right_deg is not None and len(locked_right_deg) != 7:
        raise ValueError("locked_right_deg expects 7 values")

    def execute(temporary_solution, denominator):
        actual_q = list(read_arm_joints())
        start_deg = _to_deg_14(actual_q)
        target_deg = _to_deg_14(temporary_solution)
        if locked_right_deg is not None:
            start_deg[7:14] = locked_right_deg
            target_deg[7:14] = locked_right_deg
        _interpolate_q_target(
            arm_pub,
            arm_hold,
            start_deg,
            target_deg,
            move_time,
        )
        rospy.sleep(settle)
        stable_q = list(wait_arm_stable(side=side))
        if locked_right_deg is not None:
            stable_q[7:14] = [
                math.radians(value) for value in locked_right_deg
            ]
        rospy.logwarn(
            "task1: [SOFT IK FALLBACK] [%s] temporary fraction=1/%d "
            "motion completed and %s arm is stable",
            label,
            denominator,
            side,
        )
        return stable_q

    return execute


def _segmented_preset_return(arm_pub, arm_hold, start_deg, target_deg, move_time):
    """从当前关节角分段插值回到预设关节角。"""
    segs = max(1, int(move_time / 0.1))
    seg_t = move_time / segs
    prev = start_deg
    for seg in range(1, segs + 1):
        alpha = seg / segs
        pt = [start_deg[i] + (target_deg[i] - start_deg[i]) * alpha for i in range(14)]
        msg = armTargetPoses()
        msg.times = [float(seg_t)]
        msg.values = [float(v) for v in pt]
        arm_pub.publish(msg)
        steps = max(1, int(seg_t * 100))
        for st in range(steps + 1):
            a2 = st / steps
            cur_pt = [prev[i] + (pt[i] - prev[i]) * a2 for i in range(14)]
            arm_hold.set_degrees(cur_pt)
            rospy.sleep(0.01)
        prev = pt
    deadline = time.time() + 0.0
    while time.time() < deadline and not rospy.is_shutdown():
        arm_hold.set_degrees(target_deg)
        rospy.sleep(0.01)
    rospy.sleep(SETTLE_TIME)


def move_one_step_v2(
    arm_pub,
    arm_hold,
    side,
    target_pos,
    target_quat,
    label,
    constraint_mode=IK_MODE_THREE_POINT_MIXED,
    pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
    move_time=0.5,
    settle=0.05,
    intermediate_positions=None,
):
    """单步运动：顺序求解 IK 路径点，再连续执行关节插值。

    每个 IK 解使用前一个解作为初值，并以 100Hz 更新保持值。仅在软 IK
    fallback 时先实际执行临时二分点，稳定后再以实测关节角重求原目标。
    """
    _register_task1_motion_context(arm_pub, arm_hold)
    cur = read_arm_joints()
    waypoint_positions = [list(pos) for pos in (intermediate_positions or [])]
    waypoint_positions.append(list(target_pos))
    previous_solution = list(cur)
    previous_command_deg = _to_deg_14(cur)
    path_time = float(move_time) / len(waypoint_positions)
    target_deg = None

    def execute_leg(start_deg, end_deg):
        previous = list(start_deg)
        leg_start = list(start_deg)
        segs = max(1, int(path_time / 0.1))
        seg_t = path_time / segs
        for seg in range(1, segs + 1):
            alpha = seg / segs
            point = [
                leg_start[i] + (end_deg[i] - leg_start[i]) * alpha
                for i in range(14)
            ]

            msg = armTargetPoses()
            msg.times = [float(seg_t)]
            msg.values = [float(v) for v in point]
            arm_pub.publish(msg)

            steps = max(1, int(seg_t * 100))
            for step in range(steps + 1):
                ratio = step / steps
                command = [
                    previous[i] + (point[i] - previous[i]) * ratio
                    for i in range(14)
                ]
                arm_hold.set_degrees(command)
                rospy.sleep(0.01)
            previous = point
        return list(end_deg)

    for waypoint_index, waypoint_pos in enumerate(waypoint_positions):
        if constraint_mode == IK_MODE_THREE_POINT_MIXED:
            fallback_context = "[{}] waypoint {}/{}".format(
                label, waypoint_index + 1, len(waypoint_positions)
            )

            def execute_temporary(temporary_solution, denominator):
                nonlocal previous_command_deg, previous_solution
                temporary_deg = _to_deg_14(temporary_solution)
                previous_command_deg = execute_leg(
                    previous_command_deg, temporary_deg
                )
                rospy.sleep(settle)
                previous_solution = list(wait_arm_stable(side=side))
                previous_command_deg = _to_deg_14(previous_solution)
                rospy.logwarn(
                    "task1: [SOFT IK FALLBACK] %s temporary fraction=1/%d "
                    "motion completed; arm is stable",
                    fallback_context,
                    denominator,
                )
                return previous_solution

            solved_path = _solve_soft_ik_path(
                previous_solution,
                side,
                waypoint_pos,
                target_quat,
                pos_cost_weight=pos_cost_weight,
                fallback_label=fallback_context,
                execute_temporary=execute_temporary,
            )
        else:
            solved_path = [
                solve_ik_one_hand(
                    previous_solution,
                    side,
                    waypoint_pos,
                    target_quat,
                    constraint_mode=constraint_mode,
                    pos_cost_weight=pos_cost_weight,
                    fallback_label=label,
                )
            ]
        target_deg = _to_deg_14(solved_path[-1])
        previous_command_deg = execute_leg(previous_command_deg, target_deg)
        previous_solution = list(solved_path[-1])
        if waypoint_index < len(waypoint_positions) - 1:
            rospy.loginfo(
                "task1: [%s] IK intermediate %d/%d=%s",
                label,
                waypoint_index + 1,
                len(waypoint_positions) - 1,
                _fmt(waypoint_pos),
            )

    # 最后持续广播目标值 0.2s (确保停稳)
    deadline = time.time() + 0.0
    while time.time() < deadline and not rospy.is_shutdown():
        arm_hold.set_degrees(target_deg)
        rospy.sleep(0.01)

    rospy.sleep(settle)

    fk = call_fk(read_arm_joints(), TOPIC_TIMEOUT)
    ap = list(_pose_for_side(fk, side).pos_xyz)
    err = _axis_err(ap, target_pos)
    rospy.loginfo("── [%s] %s actual=%s err=%.4f", label, side, _fmt(ap), err)
    if TRACE_FILE:
        try:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            line = f"{ts} | {label:<45s} | T=[{_fmt(target_pos)}] | A=[{_fmt(ap)}] err={err:.4f}\n"
            with open(TRACE_FILE, "a") as f:
                f.write(line)
        except:
            pass

    return target_deg


# =============================================================================
# 所有任务函数（保持既有逻辑，仅替换底层运动实现）
# =============================================================================


def _move_pick_descent_target(
    arm_pub,
    arm_hold,
    parcel_name,
    target_pos,
    target_quat,
    label,
    move_time,
    settle,
):
    try:
        move_one_step_v2(
            arm_pub,
            arm_hold,
            "right",
            target_pos,
            target_quat,
            label,
            constraint_mode=IK_MODE_POS_HARD_ORI_HARD,
            pos_cost_weight=IK_POS_COST_WEIGHT,
            move_time=move_time,
            settle=settle,
        )
    except RuntimeError as exc:
        rospy.logwarn(
            "task1: %s [%s] hard IK 0x03 failed (%s), fallback to soft IK 0x06",
            parcel_name,
            label,
            exc,
        )
        move_one_step_v2(
            arm_pub,
            arm_hold,
            "right",
            target_pos,
            target_quat,
            label,
            constraint_mode=IK_MODE_THREE_POINT_MIXED,
            pos_cost_weight=IK_POS_COST_WEIGHT,
            move_time=move_time,
            settle=settle,
        )


def right_pick(
    arm_pub,
    arm_hold,
    gripper_hold,
    parcel_name,
    parcel_position,
    pick_quat,
    on_grasped=None,
    before_grasp_mid=None,
):
    pick_ik = compute_pick_ik(parcel_name, parcel_position)
    lift_z = float(LIFT_Z.get(parcel_name, LIFT_Z["parcel_1"]))
    pick_pre_ik = [pick_ik[0], pick_ik[1], pick_ik[2] + 0.15]
    descent_mid_ratio = float(PICK_DESCENT_MID_RATIO)
    if not 0.0 < descent_mid_ratio < 1.0:
        raise ValueError("PICK_DESCENT_MID_RATIO must be between 0 and 1")
    # 下降过程保持 x、y 不变，在配置比例处停稳一次。
    pick_descent_mid_ik = [
        pick_ik[0],
        pick_ik[1],
        pick_pre_ik[2]
        + (pick_ik[2] - pick_pre_ik[2]) * descent_mid_ratio,
    ]
    cur = read_arm_joints()
    cur_r = list(call_fk(cur, TOPIC_TIMEOUT).right_pose.pos_xyz)
    yz_ik = compute_yz_align(pick_pre_ik, cur_r)
    gripper_hold.set_right_open()

    rospy.loginfo("task1: %s yz_align %s → %s", parcel_name, _fmt(cur_r), _fmt(yz_ik))
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "right",
        yz_ik,
        pick_quat,
        f"{parcel_name}_step1_yz_align",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=0.5,
        settle=0.01,
    )

    rospy.loginfo("task1: %s x_pre %s", parcel_name, _fmt(pick_pre_ik))
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "right",
        pick_pre_ik,
        pick_quat,
        f"{parcel_name}_step1p5_x_pre",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=2.0,
        settle=0.01,
    )

    if before_grasp_mid is not None:
        before_grasp_mid()

    rospy.loginfo("task1: %s grasp_mid %s", parcel_name, _fmt(pick_descent_mid_ik))
    _move_pick_descent_target(
        arm_pub,
        arm_hold,
        parcel_name,
        pick_descent_mid_ik,
        pick_quat,
        f"{parcel_name}_step1p7_grasp_mid",
        move_time=descent_mid_ratio,
        settle=0.01,
    )
    mid_stable_q = wait_arm_stable(side="right")
    mid_actual_pos = list(
        call_fk(mid_stable_q, TOPIC_TIMEOUT).right_pose.pos_xyz
    )
    mid_dx_mm = (mid_actual_pos[0] - pick_descent_mid_ik[0]) * 1000.0
    mid_dy_mm = (mid_actual_pos[1] - pick_descent_mid_ik[1]) * 1000.0
    mid_xy_error_mm = math.hypot(mid_dx_mm, mid_dy_mm)
    rospy.loginfo(
        GREEN
        + "task1: %s grasp_mid stable FK XY=%.4f %.4f target XY=%.4f %.4f "
        "error dx=%.1f dy=%.1f norm=%.1f mm"
        + RESET,
        parcel_name,
        mid_actual_pos[0],
        mid_actual_pos[1],
        pick_descent_mid_ik[0],
        pick_descent_mid_ik[1],
        mid_dx_mm,
        mid_dy_mm,
        mid_xy_error_mm,
    )

    rospy.loginfo("task1: %s grasp %s", parcel_name, _fmt(pick_ik))
    _move_pick_descent_target(
        arm_pub,
        arm_hold,
        parcel_name,
        pick_ik,
        pick_quat,
        f"{parcel_name}_step2_grasp",
        move_time=1.0 - descent_mid_ratio,
        settle=0.1,
    )

    wait_arm_stable(side="right")
    gripper_hold.set_right_close()
    if on_grasped is not None:
        on_grasped()
    rospy.sleep(GRIPPER_HOLD)

    retract_ik = [pick_ik[0] + PICK_RETRACT_X_OFFSET, pick_ik[1], lift_z]
    rospy.loginfo("task1: %s retract %s", parcel_name, _fmt(retract_ik))
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "right",
        retract_ik,
        pick_quat,
        f"{parcel_name}_step2p5_retract",
        move_time=0.5,
        settle=0.01,
    )
    return retract_ik


def right_weigh_and_regrasp(
    arm_pub,
    arm_hold,
    gripper_hold,
    parcel_name,
    lift_ik,
    release_quat,
    regrasp_quat,
    before_strict=None,
):
    release_ik = list(WEIGH_RELEASE_IK.get(parcel_name, WEIGH_RELEASE_IK["parcel_1"]))
    regrasp_xy = list(
        WEIGH_REGRASP_XY.get(parcel_name, WEIGH_REGRASP_XY["parcel_1"])
    )
    transit_z = float(WEIGH_TRANSIT_Z.get(parcel_name, WEIGH_TRANSIT_Z["parcel_1"]))
    transit_ik = [release_ik[0], release_ik[1], transit_z]
    transit_mid_ik = [
        lift_ik[0],
        0.5 * (lift_ik[1] + transit_ik[1]),
        REGRASP_LIFT_Z,
    ]
    descend_mid_ik = [
        release_ik[0],
        release_ik[1],
        0.5 * (transit_ik[2] + release_ik[2]),
    ]

    release_done = threading.Event()
    release_cancel = threading.Event()
    release_failure = []
    if WEIGH_DEBUG_FIXED_RELEASE_ENABLED:
        chain_start_sim_time = rospy.get_time()
        release_target_sim_time = (
            chain_start_sim_time + float(WEIGH_DEBUG_FIXED_RELEASE_TIME)
        )

        def _release_at_fixed_sim_time():
            try:
                while not rospy.is_shutdown() and not release_cancel.is_set():
                    remaining = release_target_sim_time - rospy.get_time()
                    if remaining <= 0.0:
                        gripper_hold.set_right_open()
                        rospy.logwarn(
                            "task1: %s debug fixed-time release at %.3fs "
                            "after weigh-chain start",
                            parcel_name,
                            max(0.0, rospy.get_time() - chain_start_sim_time),
                        )
                        return
                    rospy.sleep(min(remaining, 0.02))
            except Exception as exc:
                release_failure.append(exc)
            finally:
                release_done.set()

        release_thread = threading.Thread(
            target=_release_at_fixed_sim_time,
            name="task1-fixed-release-{}".format(parcel_name),
            daemon=True,
        )
        release_thread.start()

    try:
        rospy.loginfo("task1: %s weigh_mid %s", parcel_name, _fmt(transit_mid_ik))
        move_one_step_v2(
            arm_pub,
            arm_hold,
            "right",
            transit_mid_ik,
            release_quat,
            f"{parcel_name}_step3p5_weigh_mid",
            move_time=1.5,
            settle=0.01,
        )

        move_one_step_v2(
            arm_pub,
            arm_hold,
            "right",
            transit_ik,
            release_quat,
            f"{parcel_name}_step4_transit",
            move_time=1.5,
            settle=0.01,
        )
        rospy.loginfo("task1: %s descend_mid %s", parcel_name, _fmt(descend_mid_ik))
        move_one_step_v2(
            arm_pub,
            arm_hold,
            "right",
            descend_mid_ik,
            release_quat,
            f"{parcel_name}_step4p5_descend_mid",
            move_time=0.5,
            settle=0.01,
        )
        move_one_step_v2(
            arm_pub,
            arm_hold,
            "right",
            release_ik,
            release_quat,
            f"{parcel_name}_step5_descend",
            move_time=WEIGH_FINAL_DESCEND_MOVE_TIME,
            settle=0.01,
        )
    except Exception:
        release_cancel.set()
        raise

    if WEIGH_DEBUG_FIXED_RELEASE_ENABLED:
        while not release_done.wait(timeout=0.1) and not rospy.is_shutdown():
            pass
        if release_failure:
            raise RuntimeError(
                "fixed-time gripper release failed: {}".format(release_failure[0])
            )

    release_stable_q = wait_arm_stable(side="right")
    release_actual_pose = call_fk(release_stable_q, TOPIC_TIMEOUT).right_pose
    rospy.loginfo(
        "task1: %s release stable %s: right FK=%s right q(deg)=%s",
        parcel_name,
        "after fixed-time open"
        if WEIGH_DEBUG_FIXED_RELEASE_ENABLED
        else "before open",
        _fmt(list(release_actual_pose.pos_xyz)),
        _fmt(_rad2deg(release_stable_q[7:14]), 3),
    )
    if not WEIGH_DEBUG_FIXED_RELEASE_ENABLED:
        gripper_hold.set_right_open()
    rospy.sleep(WEIGH_DWELL)

    if before_strict is not None:
        before_strict()
    current_q = list(release_stable_q)
    if WEIGH_REGRASP_USE_STRICT_R6_IK:
        locked_q7 = float(current_q[13])
        trajectory_label = f"{parcel_name}_step6_strict_r6_regrasp"
        fallback_executor = _make_soft_ik_temporary_executor(
            arm_pub,
            arm_hold,
            "right",
            trajectory_label,
            move_time=STRICT_TRAJECTORY_MOVE_TIME,
            settle=SETTLE_TIME,
        )
        rospy.loginfo(
            "task1: %s strict r6 regrasp offset=%s m, lock q7=%.3f deg",
            parcel_name,
            _fmt(WEIGH_REGRASP_R6_OFFSET, 3),
            math.degrees(locked_q7),
        )
        regrasp_path = solve_right_r6_offset_path(
            current_q,
            WEIGH_REGRASP_R6_OFFSET,
            locked_q7,
            execute_temporary=fallback_executor,
        )
    else:
        trajectory_label = f"{parcel_name}_step6_soft_eef_regrasp"
        fallback_executor = _make_soft_ik_temporary_executor(
            arm_pub,
            arm_hold,
            "right",
            trajectory_label,
            move_time=STRICT_TRAJECTORY_MOVE_TIME,
            settle=SETTLE_TIME,
        )
        rospy.logwarn(
            "task1: %s debug regrasp uses direct soft EEF IK 0x06, "
            "q7 unlocked, offset=%s m",
            parcel_name,
            _fmt(WEIGH_REGRASP_R6_OFFSET, 3),
        )
        regrasp_path = solve_right_eef_offset_path_soft(
            current_q,
            WEIGH_REGRASP_R6_OFFSET,
            execute_temporary=fallback_executor,
        )
    direct_disabled = False
    try:
        set_wbc_direct_mode(False)
        direct_disabled = True
        execute_pillar_strict_trajectory(
            arm_pub,
            arm_hold,
            regrasp_path,
            "right",
            trajectory_label,
        )
        gripper_hold.set_right_close()
        rospy.sleep(GRIPPER_HOLD)
    finally:
        if direct_disabled:
            set_wbc_direct_mode(True)

    regrasp_lift_ik = [regrasp_xy[0], regrasp_xy[1], REGRASP_LIFT_Z]
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "right",
        regrasp_lift_ik,
        regrasp_quat,
        f"{parcel_name}_step8_regrasp_lift",
        move_time=0.99,
        settle=0.01,
    )

    rospy.loginfo("task1: %s 称重完成", parcel_name)


def _solve_left_with_locked_right(
    locked_right_deg,
    left_pos,
    left_quat,
    execute_temporary=None,
):
    locked_right_deg = [float(v) for v in locked_right_deg]
    if len(locked_right_deg) != 7:
        raise ValueError(
            f"locked_right_deg expects 7 values, got {len(locked_right_deg)}"
        )
    cur = read_arm_joints()
    cur_for_ik = list(cur[:14])
    cur_for_ik[7:14] = [math.radians(v) for v in locked_right_deg]
    ik_q = solve_ik_one_hand(
        cur_for_ik,
        "left",
        left_pos,
        left_quat,
        execute_temporary=execute_temporary,
    )
    target_deg = _rad2deg(ik_q)
    target_deg[7:14] = locked_right_deg
    return cur, target_deg


def _move_left_with_locked_right(
    arm_pub, arm_hold, locked_right_deg, left_pos, left_quat, label
):
    """执行既有的左手 IK 和线性 q 插值，并持续锁定右臂。"""
    _register_task1_motion_context(arm_pub, arm_hold)
    fallback_executor = _make_soft_ik_temporary_executor(
        arm_pub,
        arm_hold,
        "left",
        label,
        move_time=0.5,
        settle=SETTLE_TIME,
        locked_right_deg=locked_right_deg,
    )
    cur, target_deg = _solve_left_with_locked_right(
        locked_right_deg,
        left_pos,
        left_quat,
        execute_temporary=fallback_executor,
    )

    start_deg = _rad2deg(cur)
    # 右臂命令在上一步已经到达锁定目标；不能用负载下垂后的实测角作为新插值起点。
    start_deg[7:14] = locked_right_deg
    move_time = 2.0

    msg = armTargetPoses()
    msg.times = [move_time]
    msg.values = [float(v) for v in target_deg]
    arm_pub.publish(msg)

    steps = max(1, int(round(move_time * 100.0)))
    rate = rospy.Rate(100)
    for step in range(steps + 1):
        if rospy.is_shutdown():
            break
        alpha = float(step) / float(steps)
        point = [
            start_deg[i] + (target_deg[i] - start_deg[i]) * alpha for i in range(14)
        ]
        arm_hold.set_degrees(point)
        if step < steps:
            rate.sleep()

    rospy.sleep(SETTLE_TIME)
    fk = call_fk(read_arm_joints(), TOPIC_TIMEOUT)
    ap = list(_pose_for_side(fk, "left").pos_xyz)
    rospy.loginfo("── [%s] left actual=%s", label, _fmt(ap))
    return target_deg


def move_left_to_receive_ready(arm_pub, arm_hold, receive_quat, locked_right_deg=None):
    """使用软 IK 将左臂移动到绝对接取准备位姿。"""
    rospy.loginfo(
        "task1: step11 left_receive_ready absolute=%s", _fmt(LEFT_RECEIVE_READY)
    )
    if locked_right_deg is None:
        target_deg = move_one_step_v2(
            arm_pub,
            arm_hold,
            "left",
            LEFT_RECEIVE_READY,
            receive_quat,
            "step11_left_receive_ready",
            constraint_mode=IK_MODE_THREE_POINT_MIXED,
            pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
            move_time=2.0,
            settle=0.01,
        )
    else:
        target_deg = _move_left_with_locked_right(
            arm_pub,
            arm_hold,
            locked_right_deg,
            LEFT_RECEIVE_READY,
            receive_quat,
            "step11_left_receive_ready",
        )
    stable_q = wait_arm_stable(side="left")
    stable_pos = call_fk(stable_q, TOPIC_TIMEOUT).left_pose.pos_xyz
    rospy.loginfo(
        "task1: left arm ready actual FK=%s; waiting for right handoff FK",
        _fmt(stable_pos),
    )
    return target_deg


def handoff_transfer_right_to_left(
    arm_pub,
    arm_hold,
    gripper_hold,
    parcel_name,
    handoff_quat,
    receive_quat,
    left_ready_prepared=False,
    right_motion_pub=None,
    right_motion_hold=None,
):
    if right_motion_pub is None:
        right_motion_pub = arm_pub
    if right_motion_hold is None:
        right_motion_hold = arm_hold
    gripper_hold.set_left_open()

    for tz in [HANDOFF_TRANSIT_Z] + list(HANDOFF_FALLBACK_ZS):
        try:
            current = read_arm_joints()
            current_right_pose = call_fk(current, TOPIC_TIMEOUT).right_pose
            carry_quat = list(current_right_pose.quat_xyzw)
            raise_ik = [
                float(current_right_pose.pos_xyz[0]),
                float(current_right_pose.pos_xyz[1]),
                float(tz),
            ]
            rospy.loginfo(
                "task1: step9a handoff_raise_before_xy z=%.3f %s",
                tz,
                _fmt(raise_ik),
            )
            move_one_step_v2(
                right_motion_pub,
                right_motion_hold,
                "right",
                raise_ik,
                carry_quat,
                "step9a_handoff_raise_before_xy",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                move_time=1.0,
                settle=0.01,
            )

            xy_align_ik = [RIGHT_HANDOFF_IK[0], RIGHT_HANDOFF_IK[1], float(tz)]
            rospy.loginfo(
                "task1: step9b handoff_xy_align z=%.3f %s",
                tz,
                _fmt(xy_align_ik),
            )
            move_one_step_v2(
                right_motion_pub,
                right_motion_hold,
                "right",
                xy_align_ik,
                handoff_quat,
                "step9b_handoff_xy_align",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                move_time=1.5,
                settle=0.01,
            )
            if tz != HANDOFF_TRANSIT_Z:
                rospy.logwarn("task1: handoff fallback z=%.3f", tz)
            break
        except RuntimeError as exc:
            if tz == HANDOFF_FALLBACK_ZS[-1]:
                raise
            rospy.logwarn("task1: handoff transit z=%.3f IK failed: %s", tz, exc)

    rospy.loginfo("task1: step10a handoff_descend %s", _fmt(RIGHT_HANDOFF_IK))
    right_handoff_target_deg = move_one_step_v2(
        right_motion_pub,
        right_motion_hold,
        "right",
        RIGHT_HANDOFF_IK,
        handoff_quat,
        "step10a_handoff_descend",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=1.5,
        settle=0.01,
    )

    stable_q = wait_arm_stable(side="right")
    right_handoff_actual_pos = list(call_fk(stable_q, TOPIC_TIMEOUT).right_pose.pos_xyz)
    left_receive = [
        right_handoff_actual_pos[i] + LEFT_RECEIVE_IK_OFFSET[i] for i in range(3)
    ]
    rospy.loginfo(
        "task1: stable right handoff FK=%s, left ready absolute=%s, left receive=%s",
        _fmt(right_handoff_actual_pos),
        _fmt(LEFT_RECEIVE_READY),
        _fmt(left_receive),
    )

    # 锁定规划目标而不是负载下垂后的实测关节角, 避免左手接近时固化错误姿态。
    locked_right = list(right_handoff_target_deg[7:14])
    rospy.loginfo("task1: target right handoff joints locked before left receive")

    if left_ready_prepared:
        rospy.loginfo("task1: left receive ready was prepared asynchronously")
    else:
        move_left_to_receive_ready(
            arm_pub,
            arm_hold,
            receive_quat,
            locked_right_deg=locked_right,
        )
    locked_right_rad = [math.radians(value) for value in locked_right]
    left_ready_stable_q = wait_arm_stable(side="left")
    left_ready_stable_q[7:14] = locked_right_rad
    rospy.loginfo("task1: step12 strict left_receive %s", _fmt(left_receive))
    strict_receive_fallback_executor = _make_soft_ik_temporary_executor(
        arm_pub,
        arm_hold,
        "left",
        "step12_left_receive_strict_soft_fallback",
        move_time=STRICT_TRAJECTORY_MOVE_TIME,
        settle=SETTLE_TIME,
        locked_right_deg=locked_right,
    )
    strict_receive_q = solve_ik_one_hand_strict_multi(
        left_ready_stable_q,
        "left",
        left_receive,
        receive_quat,
        execute_temporary=strict_receive_fallback_executor,
    )
    strict_receive_q[7:14] = locked_right_rad

    direct_disabled = False
    try:
        set_wbc_direct_mode(False)
        direct_disabled = True
        execute_pillar_strict_frame(
            arm_pub,
            arm_hold,
            strict_receive_q,
            "left",
            "step12_left_receive_strict",
        )

        gripper_hold.set_left_close()
        rospy.sleep(GRIPPER_HOLD)
        gripper_hold.set_right_open()
        rospy.sleep(GRIPPER_HOLD)
    finally:
        if direct_disabled:
            set_wbc_direct_mode(True)


def recover_right_after_handoff(arm_pub, arm_hold):
    cur = read_arm_joints()
    cur_r = list(call_fk(cur, TOPIC_TIMEOUT).right_pose.pos_xyz)
    retract_pos = [cur_r[0], cur_r[1] + RIGHT_RETRACT_Y_OFFSET, cur_r[2]]
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "right",
        retract_pos,
        list(call_fk(cur, TOPIC_TIMEOUT).right_pose.quat_xyzw),
        "step13_right_retract",
        move_time=1,
        settle=0.01,
    )

    # 右手回到预设关节角 (分段插值, 2s)
    cur2 = read_arm_joints()
    start_deg = list(_rad2deg(cur2))
    target_deg = list(start_deg)
    target_deg[7:14] = PRESET_POINTS_DEG[-1][7:14]
    _segmented_preset_return(arm_pub, arm_hold, start_deg, target_deg, move_time=1.0)


def left_drop_in_box(arm_pub, arm_hold, gripper_hold, parcel_index, box_quat):
    offset = BOX_DROP_OFFSETS[parcel_index % len(BOX_DROP_OFFSETS)]
    box_ik = [
        BOX_DROP_IK[0] + offset[0],
        BOX_DROP_IK[1] + offset[1],
        BOX_DROP_IK[2] + offset[2],
    ]
    box_pre_ik = [box_ik[0], box_ik[1], box_ik[2] + 0.08]

    cur = read_arm_joints()
    cur_l = list(call_fk(cur, TOPIC_TIMEOUT).left_pose.pos_xyz)
    box_raise_ik = [cur_l[0], cur_l[1], box_pre_ik[2]]
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "left",
        box_raise_ik,
        box_quat,
        f"step13p7_box_raise_{parcel_index}",
        move_time=BOX_TRANSIT_MOVE_TIME,
        settle=0.05,
    )
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "left",
        box_pre_ik,
        box_quat,
        f"step13p8_box_pre_{parcel_index}",
        move_time=BOX_TRANSIT_MOVE_TIME,
        settle=0.05,
    )

    for dx in [0.0] + list(BOX_DROP_X_FALLBACK):
        ik_try = [box_ik[0] + dx, box_ik[1], box_ik[2]]
        try:
            move_one_step_v2(
                arm_pub,
                arm_hold,
                "left",
                ik_try,
                box_quat,
                f"step14_box_drop_{parcel_index}",
                move_time=1,
                settle=0.05,
            )
            if dx != 0.0:
                rospy.logwarn("task1: box_drop x fallback dx=%.3f", dx)
            break
        except RuntimeError:
            if dx == BOX_DROP_X_FALLBACK[-1]:
                raise
            rospy.logwarn("task1: box_drop IK failed dx=%.3f", dx)

    gripper_hold.set_left_open()
    rospy.sleep(0.5)

    # x 往回收避免左臂够不到
    lift_ik = [box_ik[0] - 0.10, box_ik[1], box_pre_ik[2] - 0.05]
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "left",
        lift_ik,
        box_quat,
        f"step14p5_box_lift_{parcel_index}",
        move_time=1,
        settle=0.05,
    )
    lift_actual_q = read_arm_joints()
    lift_actual_pos = list(
        call_fk(lift_actual_q, TOPIC_TIMEOUT).left_pose.pos_xyz
    )
    rospy.loginfo("task1: box lift actual FK=%s", _fmt(lift_actual_pos))
    for step_index in range(1, BOX_SAFE_INTERMEDIATE_COUNT + 1):
        safe_mid_ik = [
            lift_actual_pos[axis]
            + step_index * BOX_SAFE_INTERMEDIATE_STEP[axis]
            for axis in range(3)
        ]
        rospy.loginfo(
            "task1: box safe intermediate %d/%d=%s",
            step_index,
            BOX_SAFE_INTERMEDIATE_COUNT,
            _fmt(safe_mid_ik),
        )
        move_one_step_v2(
            arm_pub,
            arm_hold,
            "left",
            safe_mid_ik,
            box_quat,
            f"step14p6_box_safe_mid{step_index}_{parcel_index}",
            move_time=0.5,
            settle=0.05,
        )
    safe_ik = [lift_ik[0] - 0.15, lift_ik[1], lift_ik[2]]
    move_one_step_v2(
        arm_pub,
        arm_hold,
        "left",
        safe_ik,
        box_quat,
        f"step14p7_box_safe_{parcel_index}",
        move_time=0.5,
        settle=0.05,
    )

    # 左手回到预设关节角 (分段插值, 3s)
    cur2 = read_arm_joints()
    start_deg = list(_rad2deg(cur2))
    target_deg = list(start_deg)
    target_deg[:7] = PRESET_POINTS_DEG[-1][:7]
    _segmented_preset_return(arm_pub, arm_hold, start_deg, target_deg, move_time=1.0)
    rospy.loginfo("task1: 放箱完成")


# =============================================================================
# 初始化
# =============================================================================


def set_arm_ext_control():
    rospy.wait_for_service("/arm_traj_change_mode", timeout=TOPIC_TIMEOUT)
    req = changeArmCtrlModeRequest()
    req.control_mode = 2
    resp = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(req)
    if not resp.result:
        raise RuntimeError(f"arm mode change failed: {resp.message}")
    rospy.loginfo("task1: arm mode -> external control")

    rospy.wait_for_service("/enable_wbc_arm_trajectory_control", timeout=TOPIC_TIMEOUT)
    req2 = changeArmCtrlModeRequest()
    req2.control_mode = 1
    resp2 = rospy.ServiceProxy("/enable_wbc_arm_trajectory_control", changeArmCtrlMode)(
        req2
    )
    if not resp2.result:
        rospy.logwarn("task1: WBC direct failed: %s", resp2.message)
    else:
        rospy.loginfo("task1: WBC direct mode enabled")


def init_robot():
    if bool(rospy.get_param("~skip_init_head_motion", False)):
        rospy.loginfo("task1: head pose already set before operator start")
    else:
        hpub = rospy.Publisher(
            "/robot_head_motion_data", robotHeadMotionData, queue_size=10
        )
        _wait_conn(hpub)
        head_yaw = float(rospy.get_param("~head_yaw_deg", 0.0))
        head_pitch = float(rospy.get_param("~head_pitch_deg", 20.0))
        m = robotHeadMotionData()
        m.joint_data = [head_yaw, head_pitch]
        for _ in range(5):
            hpub.publish(m)
            rospy.sleep(0.1)
        rospy.sleep(0.8)
    apub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    _wait_conn(apub)
    gpub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
    _wait_conn(gpub)
    gh = GripperHold(gpub)
    tpub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    _wait_conn(tpub)
    ah = ArmHold(tpub, _rad2deg(read_arm_joints()))
    _register_task1_motion_context(apub, ah)
    set_arm_ext_control()
    return apub, ah, gh


# =============================================================================
# 预设抬手
# =============================================================================


def _cardinal(p0, p1, p2, p3, s, t=0.0):
    c = (1 - t) * 0.5
    s2, s3 = s * s, s * s * s
    return [
        h00 * p1[i]
        + h10 * c * (p2[i] - p0[i])
        + h01 * p2[i]
        + h11 * c * (p3[i] - p1[i])
        for i in range(len(p1))
        for h00, h10, h01, h11 in [
            (2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2)
        ]
    ][: len(p1)]


def move_to_preset(arm_pub, arm_hold):
    pts = [list(p) for p in PRESET_POINTS_DEG]
    pts = [_rad2deg(read_arm_joints())] + pts
    n, total = len(pts), max(
        1, int(round((len(pts) - 1) * PRESET_SEGMENT_TIME * ARM_TRAJ_HZ))
    )
    rate = rospy.Rate(ARM_TRAJ_HZ)
    for step in range(total + 1):
        if rospy.is_shutdown():
            return
        u = (step / total) ** 3 * ((step / total) * ((step / total) * 6 - 15) + 10)
        scaled = u * (n - 1)
        seg = min(int(scaled), n - 2)
        s = scaled - int(scaled)
        p1, p2 = pts[seg], pts[seg + 1]
        p0, p3 = pts[seg - 1] if seg - 1 >= 0 else p1, (
            pts[seg + 2] if seg + 2 < n else p2
        )
        arm_hold.set_degrees(_cardinal(p0, p1, p2, p3, s))
        if step < total:
            rate.sleep()
    arm_hold.set_degrees(pts[-1])
    rospy.sleep(PRESET_SETTLE_TIME)
