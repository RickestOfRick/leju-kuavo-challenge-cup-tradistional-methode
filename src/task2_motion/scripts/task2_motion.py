#!/usr/bin/env python3
"""任务二的 WBC 控制、IK、关节插值、稳定等待和 FK 检查。"""

import math
import threading
import time

import rospy
from kuavo_msgs.msg import armTargetPoses, ikSolveParam, twoArmHandPoseCmd
from kuavo_msgs.srv import (
    changeArmCtrlMode,
    changeArmCtrlModeRequest,
    fkSrv,
    twoArmHandPoseCmdSrv,
)
from sensor_msgs.msg import JointState

from task2_config import (
    ARM_JOINT_NAMES,
    ARM_MOVE_TIME,
    ARM_STABLE_DURATION,
    ARM_STABLE_MAX_SPEED_DEG_S,
    ARM_STABLE_TIMEOUT,
    ARM_TRAJ_HZ,
    FK_ORIENTATION_WARN_RAD,
    FK_POSITION_WARN,
    IK_MAJOR_ITERATIONS_LIMIT,
    IK_MODE_THREE_POINT_MIXED,
    IK_POS_COST_WEIGHT,
    JOINT_COMMAND_SEGMENT_TIME,
    MOTION_SETTLE_TIME,
    PRESET_POINTS_DEG,
    PRESET_SEGMENT_TIME,
    PRESET_SETTLE_TIME,
    SOFT_IK_BISECT_ATTEMPTS,
    TOPIC_TIMEOUT,
)


GREEN = "\033[32m"
RESET = "\033[0m"
ARM_DOF = 14


def _side_slice(side):
    if side == "left":
        return slice(0, 7)
    if side == "right":
        return slice(7, 14)
    if side == "both":
        return slice(0, ARM_DOF)
    raise ValueError("unknown arm {}".format(side))


def fmt(values, digits=4):
    return " ".join("{:.{}f}".format(value, digits) for value in values)


def rad_to_deg(values):
    return [math.degrees(float(value)) for value in values]


def _arm_values(values, value_name):
    values = list(values)
    if len(values) >= 27:
        return values[13:27]
    if len(values) >= 26:
        return values[12:26]
    raise RuntimeError("{} len={}".format(value_name, len(values)))


def _wait_connection(publisher, timeout=TOPIC_TIMEOUT):
    start = time.time()
    while (
        publisher.get_num_connections() == 0
        and time.time() - start < timeout
        and not rospy.is_shutdown()
    ):
        rospy.sleep(0.2)
    if publisher.get_num_connections() == 0:
        raise RuntimeError("topic {} has no subscriber".format(publisher.name))


def _cardinal(p0, p1, p2, p3, s, tension=0.0):
    """计算与任务一预位一致的 Cardinal 曲线插值点。"""
    coefficient = (1.0 - tension) * 0.5
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return [
        h00 * p1[index]
        + h10 * coefficient * (p2[index] - p0[index])
        + h01 * p2[index]
        + h11 * coefficient * (p3[index] - p1[index])
        for index in range(len(p1))
    ]


class ArmHold:
    """持续向 WBC 轨迹通道广播最近一次的 14 关节目标。"""

    def __init__(self, publisher, degrees):
        self._publisher = publisher
        self._degrees = self._validate(degrees)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _validate(degrees):
        values = [float(value) for value in degrees]
        if len(values) != 14:
            raise ValueError("arm command must contain 14 joints")
        return values

    def set_degrees(self, degrees):
        values = self._validate(degrees)
        with self._lock:
            self._degrees = values

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self):
        rate = rospy.Rate(ARM_TRAJ_HZ)
        while not self._stop.is_set() and not rospy.is_shutdown():
            with self._lock:
                values = list(self._degrees)
            message = JointState()
            message.header.stamp = rospy.Time.now()
            message.name = list(ARM_JOINT_NAMES)
            message.position = values
            self._publisher.publish(message)
            rate.sleep()


class GripperHold:
    """持续发布左右夹爪命令，避免单次消息丢失。"""

    def __init__(self, publisher):
        self._publisher = publisher
        self._left = 0.0
        self._right = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def open_all(self):
        with self._lock:
            self._left = 0.0
            self._right = 0.0

    def open(self, side):
        with self._lock:
            if side == "left":
                self._left = 0.0
            elif side == "right":
                self._right = 0.0
            else:
                raise ValueError("unknown arm {}".format(side))

    def close(self, side):
        with self._lock:
            if side == "left":
                self._left = 255.0
            elif side == "right":
                self._right = 255.0
            else:
                raise ValueError("unknown arm {}".format(side))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self):
        rate = rospy.Rate(100.0)
        while not self._stop.is_set() and not rospy.is_shutdown():
            with self._lock:
                left, right = self._left, self._right
            message = JointState()
            message.header.stamp = rospy.Time.now()
            message.name = ["left_gripper_joint", "right_gripper_joint"]
            message.position = [left, right]
            self._publisher.publish(message)
            rate.sleep()


class RobotController:
    """封装任务二执行期间唯一的手臂和夹爪控制通道。"""

    def __init__(self):
        self.arm_target_pub = rospy.Publisher(
            "/kuavo_arm_target_poses", armTargetPoses, queue_size=10
        )
        self.arm_traj_pub = rospy.Publisher(
            "/kuavo_arm_traj", JointState, queue_size=10
        )
        self.gripper_pub = rospy.Publisher(
            "/gripper/command", JointState, queue_size=10
        )
        _wait_connection(self.arm_target_pub)
        _wait_connection(self.arm_traj_pub)
        _wait_connection(self.gripper_pub)

        initial = rad_to_deg(self.read_arm_joints())
        self._command_lock = threading.RLock()
        self._trajectory_degrees = list(initial)
        self._hold_degrees = list(initial)
        self.arm_hold = ArmHold(self.arm_traj_pub, initial)
        self.gripper = GripperHold(self.gripper_pub)
        try:
            self._set_wbc_external_control()
        except Exception:
            self.gripper.stop()
            self.arm_hold.stop()
            raise

    def stop(self):
        self.gripper.stop()
        self.arm_hold.stop()

    def sync_command_to_actual(self):
        """用当前实测 q 初始化后续单臂轨迹的非活动侧保持目标。"""
        joints = self.read_arm_joints()
        degrees = rad_to_deg(joints)
        with self._command_lock:
            self._trajectory_degrees = list(degrees)
            self._hold_degrees = list(degrees)
            self.arm_hold.set_degrees(degrees)
        rospy.loginfo(
            "task2: single-arm command state synchronized to actual q=%s",
            fmt(degrees, 2),
        )
        return joints

    def move_to_preset(self):
        """按任务一的单条 Cardinal 轨迹从当前实测 q 连续进入预位。"""
        points = [rad_to_deg(self.read_arm_joints())]
        points.extend([list(point) for point in PRESET_POINTS_DEG])
        point_count = len(points)
        total_steps = max(
            1,
            int(
                round(
                    (point_count - 1)
                    * float(PRESET_SEGMENT_TIME)
                    * float(ARM_TRAJ_HZ)
                )
            ),
        )
        rate = rospy.Rate(ARM_TRAJ_HZ)
        for step in range(total_steps + 1):
            if rospy.is_shutdown():
                raise RuntimeError("task2: shutdown during initial preset")
            normalized = float(step) / float(total_steps)
            smooth = normalized ** 3 * (
                normalized * (normalized * 6.0 - 15.0) + 10.0
            )
            scaled = smooth * (point_count - 1)
            segment = min(int(scaled), point_count - 2)
            local_s = scaled - int(scaled)
            p1 = points[segment]
            p2 = points[segment + 1]
            p0 = points[segment - 1] if segment > 0 else p1
            p3 = points[segment + 2] if segment + 2 < point_count else p2
            self._set_side_hold(
                "both", _cardinal(p0, p1, p2, p3, local_s)
            )
            if step < total_steps:
                rate.sleep()
        self._set_side_hold("both", points[-1])
        rospy.sleep(PRESET_SETTLE_TIME)
        rospy.loginfo(
            "task2: initial continuous preset completed with %d path points",
            len(PRESET_POINTS_DEG),
        )

    def _publish_side_target(self, side, target_degrees, segment_time):
        joint_slice = _side_slice(side)
        with self._command_lock:
            self._trajectory_degrees[joint_slice] = list(target_degrees)[joint_slice]
            merged = list(self._trajectory_degrees)
        message = armTargetPoses()
        message.times = [float(segment_time)]
        message.values = merged
        self.arm_target_pub.publish(message)

    def _set_side_hold(self, side, target_degrees):
        joint_slice = _side_slice(side)
        with self._command_lock:
            self._hold_degrees[joint_slice] = list(target_degrees)[joint_slice]
            merged = list(self._hold_degrees)
        self.arm_hold.set_degrees(merged)

    def _set_wbc_external_control(self):
        rospy.wait_for_service("/arm_traj_change_mode", timeout=TOPIC_TIMEOUT)
        request = changeArmCtrlModeRequest()
        request.control_mode = 2
        response = rospy.ServiceProxy(
            "/arm_traj_change_mode", changeArmCtrlMode
        )(request)
        if not response.result:
            raise RuntimeError("arm external-control mode failed: {}".format(response.message))

        rospy.wait_for_service(
            "/enable_wbc_arm_trajectory_control", timeout=TOPIC_TIMEOUT
        )
        request = changeArmCtrlModeRequest()
        request.control_mode = 1
        response = rospy.ServiceProxy(
            "/enable_wbc_arm_trajectory_control", changeArmCtrlMode
        )(request)
        if not response.result:
            raise RuntimeError("WBC arm trajectory control failed: {}".format(response.message))
        rospy.loginfo("task2: arm control -> external WBC trajectory mode")

    @staticmethod
    def read_arm_joints(timeout=TOPIC_TIMEOUT):
        from kuavo_msgs.msg import sensorsData

        message = rospy.wait_for_message(
            "/sensors_data_raw", sensorsData, timeout=timeout
        )
        return _arm_values(message.joint_data.joint_q, "joint_q")

    @staticmethod
    def call_fk(joints, timeout=TOPIC_TIMEOUT):
        rospy.wait_for_service("/ik/fk_srv", timeout=timeout)
        response = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(list(joints))
        if not response.success:
            raise RuntimeError("/ik/fk_srv failed")
        return response.hand_poses

    @staticmethod
    def _pose_for_side(hand_poses, side):
        return hand_poses.left_pose if side == "left" else hand_poses.right_pose

    def wait_arm_stable(self, side):
        """等待对应手臂动作完成；超时属于软检查，只告警并返回最新关节角。"""
        from kuavo_msgs.msg import sensorsData

        if side not in ("left", "right", "both"):
            raise ValueError("stable side must be left, right, or both")
        velocity_slice = {
            "left": slice(0, 7),
            "right": slice(7, 14),
            "both": slice(0, 14),
        }[side]
        start_sim = rospy.get_time()
        stable_since = None
        latest_q = None
        latest_speed = float("inf")
        while not rospy.is_shutdown():
            now = rospy.get_time()
            if max(0.0, now - start_sim) >= ARM_STABLE_TIMEOUT:
                break
            try:
                message = rospy.wait_for_message(
                    "/sensors_data_raw", sensorsData, timeout=1.0
                )
                latest_q = _arm_values(message.joint_data.joint_q, "joint_q")
                velocities = _arm_values(
                    message.joint_data.joint_v, "joint_v"
                )[velocity_slice]
            except Exception:
                continue
            latest_speed = max(abs(math.degrees(value)) for value in velocities)
            now = rospy.get_time()
            if latest_speed <= ARM_STABLE_MAX_SPEED_DEG_S:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= ARM_STABLE_DURATION:
                    rospy.loginfo(
                        "task2: %s arm stable for %.2fs, max speed=%.3f deg/s",
                        side,
                        ARM_STABLE_DURATION,
                        latest_speed,
                    )
                    return latest_q
            else:
                stable_since = None

        if latest_q is None:
            raise RuntimeError("task2: {} stability check received no joint state".format(side))
        rospy.logwarn(
            "task2: %s stability check timed out after %.1fs, last speed=%.3f deg/s; continuing",
            side,
            ARM_STABLE_TIMEOUT,
            latest_speed,
        )
        return latest_q

    @staticmethod
    def _quaternion_error(actual, desired):
        actual_norm = math.sqrt(sum(float(value) ** 2 for value in actual))
        desired_norm = math.sqrt(sum(float(value) ** 2 for value in desired))
        if actual_norm <= 1e-12 or desired_norm <= 1e-12:
            return math.pi
        dot = abs(
            sum(float(actual[index]) * float(desired[index]) for index in range(4))
            / (actual_norm * desired_norm)
        )
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    def _solve_ik_once(
        self,
        current,
        side,
        target_position,
        target_quaternion,
        pos_cost_weight,
    ):
        q0 = list(current)
        current_fk = self.call_fk(q0)

        request = twoArmHandPoseCmd()
        request.use_custom_ik_param = True
        request.joint_angles_as_q0 = True
        params = ikSolveParam()
        params.major_optimality_tol = 1e-3
        params.major_feasibility_tol = 1e-3
        params.minor_feasibility_tol = 1e-3
        params.major_iterations_limit = IK_MAJOR_ITERATIONS_LIMIT
        params.oritation_constraint_tol = 1e-3
        params.pos_constraint_tol = 1e-3
        params.pos_cost_weight = float(pos_cost_weight)
        params.constraint_mode = IK_MODE_THREE_POINT_MIXED
        request.ik_param = params

        left = request.hand_poses.left_pose
        right = request.hand_poses.right_pose
        left.joint_angles = list(q0[:7])
        right.joint_angles = list(q0[7:14])
        left.elbow_pos_xyz = [0.0, 0.0, 0.0]
        right.elbow_pos_xyz = [0.0, 0.0, 0.0]
        left.pos_xyz = list(current_fk.left_pose.pos_xyz)
        left.quat_xyzw = list(current_fk.left_pose.quat_xyzw)
        right.pos_xyz = list(current_fk.right_pose.pos_xyz)
        right.quat_xyzw = list(current_fk.right_pose.quat_xyzw)
        active = left if side == "left" else right
        active.pos_xyz = list(target_position)
        active.quat_xyzw = list(target_quaternion)

        rospy.wait_for_service(
            "/ik/two_arm_hand_pose_cmd_srv", timeout=TOPIC_TIMEOUT
        )
        try:
            response = rospy.ServiceProxy(
                "/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv
            )(request)
        except (rospy.ROSException, rospy.ServiceException) as error:
            raise RuntimeError("IK failed: service error {}".format(error))
        if not response.success:
            raise RuntimeError(
                "IK failed: {}".format(getattr(response, "error_reason", ""))
            )
        left_result = list(response.hand_poses.left_pose.joint_angles)
        right_result = list(response.hand_poses.right_pose.joint_angles)
        if len(left_result) == 7 and len(right_result) == 7:
            return left_result + right_result
        if len(response.q_arm) >= 14:
            return list(response.q_arm[:14])
        raise RuntimeError("IK failed: response contains no 14-joint solution")

    def _solve_ik_step(
        self,
        current,
        side,
        target_position,
        target_quaternion,
        pos_cost_weight,
        label,
    ):
        try:
            return (
                self._solve_ik_once(
                    current,
                    side,
                    target_position,
                    target_quaternion,
                    pos_cost_weight,
                ),
                True,
            )
        except RuntimeError as error:
            if not str(error).startswith("IK failed"):
                raise
            last_error = error

        current_pose = self._pose_for_side(self.call_fk(current), side)
        current_position = list(current_pose.pos_xyz)
        rospy.logwarn(
            "task2: [%s] direct soft IK failed (%s); trying Cartesian subdivisions",
            label,
            last_error,
        )
        for attempt in range(1, SOFT_IK_BISECT_ATTEMPTS + 1):
            fraction = 0.5 ** attempt
            temporary_position = [
                current_position[index]
                + fraction * (target_position[index] - current_position[index])
                for index in range(3)
            ]
            rospy.logwarn(
                "task2: [%s] IK fallback %d/%d fraction=1/%d target=%s",
                label,
                attempt,
                SOFT_IK_BISECT_ATTEMPTS,
                2 ** attempt,
                fmt(temporary_position),
            )
            try:
                temporary = self._solve_ik_once(
                    current,
                    side,
                    temporary_position,
                    target_quaternion,
                    pos_cost_weight,
                )
                rospy.logwarn(
                    "task2: [%s] IK fallback fraction=1/%d succeeded; "
                    "execute and stabilize before re-solving the original target",
                    label,
                    2 ** attempt,
                )
                return temporary, False
            except RuntimeError as error:
                last_error = error
        raise RuntimeError(
            "task2: [{}] soft IK failed after {} subdivisions: {}".format(
                label, SOFT_IK_BISECT_ATTEMPTS, last_error
            )
        )

    def _interpolate_leg(self, start_degrees, target_degrees, duration, side):
        start = [float(value) for value in start_degrees]
        target = [float(value) for value in target_degrees]
        segments = max(1, int(duration / JOINT_COMMAND_SEGMENT_TIME))
        segment_time = float(duration) / segments
        previous = list(start)
        for segment in range(1, segments + 1):
            alpha = float(segment) / segments
            endpoint = [
                start[index] + alpha * (target[index] - start[index])
                for index in range(14)
            ]
            self._publish_side_target(side, endpoint, segment_time)

            steps = max(1, int(segment_time * ARM_TRAJ_HZ))
            for step in range(steps + 1):
                ratio = float(step) / steps
                command = [
                    previous[index]
                    + ratio * (endpoint[index] - previous[index])
                    for index in range(14)
                ]
                self._set_side_hold(side, command)
                rospy.sleep(1.0 / ARM_TRAJ_HZ)
            previous = endpoint
        self._set_side_hold(side, target)

    def move_joints(
        self,
        target_degrees,
        label,
        side="both",
        move_time=ARM_MOVE_TIME,
        checked_keypoint=True,
    ):
        start = rad_to_deg(self.read_arm_joints())
        self._interpolate_leg(start, target_degrees, move_time, side)
        if not checked_keypoint:
            rospy.loginfo(
                "task2: [%s] %s joint act completed without stability wait or FK check",
                label,
                side,
            )
            return [math.radians(float(value)) for value in target_degrees]
        rospy.sleep(MOTION_SETTLE_TIME)
        joints = self.wait_arm_stable(side)
        rospy.loginfo(
            GREEN + "── [%s] %s joint keypoint completed q=%s" + RESET,
            label,
            side,
            fmt(rad_to_deg(joints), 2),
        )
        return joints

    def _execute_pose_leg(
        self,
        current,
        side,
        target_position,
        target_quaternion,
        label,
        pos_cost_weight=IK_POS_COST_WEIGHT,
        move_time=ARM_MOVE_TIME,
        wait_for_stability=True,
    ):
        actual = list(current)
        recovery_round = 0
        while not rospy.is_shutdown():
            solution, is_target = self._solve_ik_step(
                actual,
                side,
                target_position,
                target_quaternion,
                pos_cost_weight,
                label,
            )
            previous = rad_to_deg(actual)
            target = rad_to_deg(solution)
            self._interpolate_leg(previous, target, float(move_time), side)
            rospy.sleep(MOTION_SETTLE_TIME)
            if is_target:
                actual = (
                    self.wait_arm_stable(side)
                    if wait_for_stability
                    else self.read_arm_joints()
                )
                return actual
            rospy.logwarn(
                "task2: [%s] temporary fallback q executed; waiting for %s arm "
                "to stabilize before retrying the original target",
                label,
                side,
            )
            actual = self.wait_arm_stable(side)
            recovery_round += 1
            rospy.logwarn(
                "task2: [%s] temporary fallback motion is stable; retrying the "
                "original target from actual q (recovery round %d)",
                label,
                recovery_round,
            )
        raise RuntimeError("task2: shutdown while executing {}".format(label))

    def _log_pose_fk(
        self,
        stable_joints,
        side,
        target_position,
        target_quaternion,
        label,
    ):
        actual_pose = self._pose_for_side(self.call_fk(stable_joints), side)
        actual_position = list(actual_pose.pos_xyz)
        actual_quaternion = list(actual_pose.quat_xyzw)
        position_delta = [
            actual_position[index] - target_position[index]
            for index in range(3)
        ]
        xy_error = math.hypot(position_delta[0], position_delta[1])
        z_error = position_delta[2]
        position_error = math.sqrt(sum(value * value for value in position_delta))
        orientation_error = self._quaternion_error(
            actual_quaternion, target_quaternion
        )
        rospy.loginfo(
            GREEN
            + "── [%s] %s target=%s actual=%s "
            "xy_err=%.3f mm z_err=%+.3f mm xyz_err=%.3f mm ori_err=%.2f deg"
            + RESET,
            label,
            side,
            fmt(target_position),
            fmt(actual_position),
            1000.0 * xy_error,
            1000.0 * z_error,
            1000.0 * position_error,
            math.degrees(orientation_error),
        )
        if (
            position_error > FK_POSITION_WARN
            or orientation_error > FK_ORIENTATION_WARN_RAD
        ):
            rospy.logwarn(
                "task2: [%s] FK tolerance warning "
                "xy=%.4f m z=%+.4f m xyz=%.4f/%.4f m "
                "ori=%.1f/%.1f deg; continuing",
                label,
                xy_error,
                z_error,
                position_error,
                FK_POSITION_WARN,
                math.degrees(orientation_error),
                math.degrees(FK_ORIENTATION_WARN_RAD),
            )
        return stable_joints

    def move_pose(
        self,
        side,
        target_position,
        target_quaternion,
        label,
        pos_cost_weight=IK_POS_COST_WEIGHT,
        move_time=ARM_MOVE_TIME,
        checked_keypoint=True,
    ):
        current = self.read_arm_joints()
        target_position = [float(value) for value in target_position]

        actual_joints = self._execute_pose_leg(
            current,
            side,
            target_position,
            target_quaternion,
            label,
            pos_cost_weight=pos_cost_weight,
            move_time=float(move_time),
            wait_for_stability=checked_keypoint,
        )
        if not checked_keypoint:
            rospy.loginfo(
                "task2: [%s] direct act completed without stability wait or FK check",
                label,
            )
            return actual_joints
        return self._log_pose_fk(
            actual_joints,
            side,
            target_position,
            target_quaternion,
            label,
        )

    def move_pose_via_ik(
        self,
        side,
        intermediate_positions,
        target_position,
        target_quaternion,
        label,
        pos_cost_weight=IK_POS_COST_WEIGHT,
        move_time=ARM_MOVE_TIME,
    ):
        """经多个 IK 中间解连续执行 q 插值，只在最终目标等待稳定和检查 FK。"""
        current = list(self.read_arm_joints())
        intermediate_positions = [
            [float(value) for value in position]
            for position in intermediate_positions
        ]
        if not intermediate_positions or any(
            len(position) != 3 for position in intermediate_positions
        ):
            raise ValueError("IK intermediate positions must contain XYZ points")
        target_position = [float(value) for value in target_position]

        constraint_positions = list(intermediate_positions) + [target_position]
        constraint_labels = [
            "{}_intermediate_{}/{}".format(
                label, index, len(intermediate_positions)
            )
            for index in range(1, len(intermediate_positions) + 1)
        ] + [label]
        leg_time = float(move_time) / len(constraint_positions)

        # 直接 IK 全部成功时保持原有的离线求解和连续 q 插值逻辑。
        solutions = []
        previous_solution = list(current)
        direct_error = None
        for position in constraint_positions:
            try:
                solution = self._solve_ik_once(
                    previous_solution,
                    side,
                    position,
                    target_quaternion,
                    pos_cost_weight,
                )
            except RuntimeError as error:
                if not str(error).startswith("IK failed"):
                    raise
                direct_error = error
                break
            solutions.append(solution)
            previous_solution = solution

        if direct_error is None:
            previous = current
            rospy.loginfo(
                "task2: [%s] continuous q trajectory via %d IK constraints, q legs=%d",
                label,
                len(intermediate_positions),
                len(solutions),
            )
            for solution in solutions:
                self._interpolate_leg(
                    rad_to_deg(previous),
                    rad_to_deg(solution),
                    leg_time,
                    side,
                )
                previous = list(solution)
            rospy.sleep(MOTION_SETTLE_TIME)
            stable_joints = self.wait_arm_stable(side)
        else:
            rospy.logwarn(
                "task2: [%s] continuous IK preplanning failed (%s); entering "
                "execution-aware fallback recovery",
                label,
                direct_error,
            )
            actual = list(current)
            for index, (position, constraint_label) in enumerate(
                zip(constraint_positions, constraint_labels)
            ):
                actual = self._execute_pose_leg(
                    actual,
                    side,
                    position,
                    target_quaternion,
                    constraint_label,
                    pos_cost_weight=pos_cost_weight,
                    move_time=leg_time,
                    wait_for_stability=index == len(constraint_positions) - 1,
                )
            stable_joints = actual
        return self._log_pose_fk(
            stable_joints,
            side,
            target_position,
            target_quaternion,
            label,
        )
