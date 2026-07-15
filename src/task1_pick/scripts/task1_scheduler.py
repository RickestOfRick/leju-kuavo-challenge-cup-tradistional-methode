#!/usr/bin/env python3
"""任务一的双臂独立调度器。"""

import os
import signal
import subprocess
import threading
import time
import traceback

import rospy
from kuavo_msgs.msg import armTargetPoses

import task1_motion as motion
import task1_perception as vision
import task1_config as config

_ARM_DOF = 14
_LEFT_SLICE = slice(0, 7)
_RIGHT_SLICE = slice(7, 14)


def _side_slice(side):
    if side == "left":
        return _LEFT_SLICE
    if side == "right":
        return _RIGHT_SLICE
    if side == "both":
        return slice(0, _ARM_DOF)
    raise ValueError("unknown arm side: {}".format(side))


class _ArmPublisherProxy:
    def __init__(self, mux, side):
        self._mux = mux
        self._side = side

    def publish(self, message):
        self._mux.publish(self._side, message)


class _ArmHoldProxy:
    def __init__(self, mux, side):
        self._mux = mux
        self._side = side

    def set_degrees(self, degrees):
        self._mux.set_hold_degrees(self._side, degrees)


class DualArmCommandMux:
    """合并单臂指令，防止并发线程覆盖另一侧手臂的目标。"""

    def __init__(self, raw_publisher, raw_hold, initial_degrees):
        initial_degrees = [float(value) for value in initial_degrees]
        if len(initial_degrees) != _ARM_DOF:
            raise ValueError("initial arm target must contain 14 joints")
        self._raw_publisher = raw_publisher
        self._raw_hold = raw_hold
        self._trajectory_degrees = list(initial_degrees)
        self._hold_degrees = list(initial_degrees)
        self._lock = threading.RLock()

    def publisher(self, side):
        _side_slice(side)
        return _ArmPublisherProxy(self, side)

    def hold(self, side):
        _side_slice(side)
        return _ArmHoldProxy(self, side)

    def publish(self, side, message):
        values = [float(value) for value in message.values]
        if not values or len(values) % _ARM_DOF != 0:
            raise ValueError(
                "armTargetPoses values must contain one or more 14-joint frames"
            )
        joint_slice = _side_slice(side)
        merged_values = []
        with self._lock:
            for offset in range(0, len(values), _ARM_DOF):
                incoming = values[offset : offset + _ARM_DOF]
                self._trajectory_degrees[joint_slice] = incoming[joint_slice]
                merged_values.extend(self._trajectory_degrees)

            merged = armTargetPoses()
            merged.times = list(message.times)
            merged.values = merged_values
            merged.frame = int(message.frame)
            self._raw_publisher.publish(merged)

    def set_hold_degrees(self, side, degrees):
        degrees = [float(value) for value in degrees]
        if len(degrees) != _ARM_DOF:
            raise ValueError("arm hold target must contain 14 joints")
        joint_slice = _side_slice(side)
        with self._lock:
            self._hold_degrees[joint_slice] = degrees[joint_slice]
            self._raw_hold.set_degrees(self._hold_degrees)


class LeftDropWorker:
    def __init__(
        self, arm_publisher, arm_hold, gripper_hold, box_quat, receive_quat
    ):
        self._arm_publisher = arm_publisher
        self._arm_hold = arm_hold
        self._gripper_hold = gripper_hold
        self._box_quat = list(box_quat)
        self._receive_quat = list(receive_quat)
        self._thread = None
        self._failure = None
        self._lock = threading.Lock()
        self._ready_request_thread = None
        self._ready_request_failure = None
        self._ready_request_lock = threading.Lock()

    def _start_thread(self, target, args, name, log_message, *log_args):
        self.wait()
        with self._lock:
            self._failure = None
            self._thread = threading.Thread(
                target=target,
                args=args,
                name=name,
                daemon=True,
            )
            thread = self._thread
        rospy.loginfo(log_message, *log_args)
        thread.start()

    def start_ready(self):
        self._start_thread(
            self._run_ready,
            (),
            "task1-left-receive-ready",
            "task1: start left receive-ready motion",
        )

    def start(self, parcel_index, parcel_name):
        self._start_thread(
            self._run_drop,
            (parcel_index, parcel_name),
            "task1-left-drop-{}".format(parcel_name),
            "task1: start left drop for %s",
            parcel_name,
        )

    def _prepare_ready(self):
        motion.move_left_to_receive_ready(
            self._arm_publisher,
            self._arm_hold,
            self._receive_quat,
        )
        rospy.loginfo("task1: left arm prepared at absolute receive ready")

    def _run_ready(self):
        try:
            self._prepare_ready()
        except Exception as exc:
            with self._lock:
                self._failure = (exc, traceback.format_exc())

    def request_ready(self):
        """请求左臂在当前动作结束后回到准备位，不阻塞右臂。"""
        with self._ready_request_lock:
            if (
                self._ready_request_thread is not None
                and self._ready_request_thread.is_alive()
            ):
                raise RuntimeError("left receive-ready request is already active")
            self._ready_request_failure = None
            self._ready_request_thread = threading.Thread(
                target=self._run_ready_request,
                name="task1-left-ready-request",
                daemon=True,
            )
            thread = self._ready_request_thread
        rospy.loginfo(
            "task1: right grasp completed; request left receive ready"
        )
        thread.start()

    def _run_ready_request(self):
        try:
            self.wait("left drop and joint-preset return")
            self.start_ready()
            self.wait("left receive-ready completion")
        except Exception as exc:
            with self._ready_request_lock:
                self._ready_request_failure = (exc, traceback.format_exc())

    def wait_ready_request(self, reason=None):
        with self._ready_request_lock:
            thread = self._ready_request_thread
        if thread is None:
            return
        if reason and thread.is_alive():
            rospy.loginfo("task1: wait for left ready before %s", reason)
        while thread.is_alive() and not rospy.is_shutdown():
            thread.join(timeout=0.1)
        with self._ready_request_lock:
            failure = self._ready_request_failure
            self._ready_request_failure = None
            self._ready_request_thread = None
        if failure is not None:
            exc, failure_traceback = failure
            raise RuntimeError(
                "left receive-ready request failed: {}\n{}".format(
                    exc, failure_traceback
                )
            )

    def _run_drop(self, parcel_index, parcel_name):
        try:
            motion.left_drop_in_box(
                self._arm_publisher,
                self._arm_hold,
                self._gripper_hold,
                parcel_index,
                self._box_quat,
            )
            rospy.loginfo(
                "task1: left drop completed and returned to joint preset for %s",
                parcel_name,
            )
        except Exception as exc:
            with self._lock:
                self._failure = (exc, traceback.format_exc())

    def wait(self, reason=None):
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        if reason and thread.is_alive():
            rospy.loginfo("task1: wait for left arm before %s", reason)
        while thread.is_alive() and not rospy.is_shutdown():
            thread.join(timeout=0.1)
        with self._lock:
            failure = self._failure
            self._failure = None
            self._thread = None
        if failure is not None:
            exc, failure_traceback = failure
            raise RuntimeError(
                "left worker failed: {}\n{}".format(exc, failure_traceback)
            )


def _handoff_and_dispatch_left(
    interaction_publisher,
    interaction_hold,
    right_publisher,
    right_hold,
    gripper_hold,
    left_worker,
    parcel_index,
    parcel_name,
    handoff_quat,
    receive_quat,
):
    motion.handoff_transfer_right_to_left(
        interaction_publisher,
        interaction_hold,
        gripper_hold,
        parcel_name,
        handoff_quat,
        receive_quat,
        left_ready_prepared=True,
        right_motion_pub=right_publisher,
        right_motion_hold=right_hold,
    )

    # WBC 直连模式已经恢复，此后左右臂相互独立。
    left_worker.start(parcel_index, parcel_name)
    motion.recover_right_after_handoff(right_publisher, right_hold)
    rospy.loginfo("task1: handoff complete; right recovery complete")


def _start_rosbag(bag_path):
    if not bag_path:
        return None
    command = [
        "rosbag",
        "record",
        "-O",
        bag_path,
        "/gripper/command",
        "/gripper/state",
        "/sensors_data_raw",
        "/joint_cmd",
        "/kuavo_arm_traj",
        "/cam_h/color/image_raw/compressed",
        "/cam_l/color/image_raw/compressed",
        "/cam_r/color/image_raw/compressed",
        "/cam_h/depth/image_raw/compressedDepth",
        "/cam_r/depth/image_rect_raw/compressedDepth",
        "/cam_l/depth/image_rect_raw/compressedDepth",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(1.5)
    return process


def _terminate_process_group(proc, name, wait_sigint=3.0, wait_sigterm=2.0):
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return
    for sig, timeout, label in (
        (signal.SIGINT, wait_sigint, "SIGINT"),
        (signal.SIGTERM, wait_sigterm, "SIGTERM"),
        (signal.SIGKILL, 1.0, "SIGKILL"),
    ):
        if proc.poll() is not None:
            return
        try:
            os.killpg(pgid, sig)
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            rospy.logwarn("task1: %s did not exit after %s", name, label)
        except Exception as exc:
            rospy.logwarn("task1: failed to stop %s with %s: %s", name, label, exc)


def run_scene1_task(seed, bag_path=None, parcel_poses=None):
    rospy.loginfo("=" * 60)
    rospy.loginfo("task1 seed=%d", seed)
    rospy.loginfo("=" * 60)

    if parcel_poses is None:
        parcel_poses = vision.init_from_task1_vision()
    for name, position in parcel_poses.items():
        rospy.loginfo("task1: %s base_link=%s", name, motion._fmt(position))

    raw_arm_hold = None
    gripper_hold = None
    left_worker = None
    bag_process = None

    try:
        raw_arm_publisher, raw_arm_hold, gripper_hold = motion.init_robot()
        gripper_hold.set_right_open()
        motion.move_to_preset(raw_arm_publisher, raw_arm_hold)

        initial_joints = motion.read_arm_joints()
        initial_degrees = motion._rad2deg(initial_joints)
        mux = DualArmCommandMux(raw_arm_publisher, raw_arm_hold, initial_degrees)
        right_publisher, right_hold = mux.publisher("right"), mux.hold("right")
        left_publisher, left_hold = mux.publisher("left"), mux.hold("left")
        both_publisher, both_hold = mux.publisher("both"), mux.hold("both")

        preset_pose = motion.call_fk(initial_joints, config.TOPIC_TIMEOUT).right_pose
        rospy.loginfo(
            "task1: preset right actual=%s", motion._fmt(preset_pose.pos_xyz)
        )
        bag_process = _start_rosbag(bag_path)

        pick_quat = motion._q_from_ypr(*config.RIGHT_PICK_YPR_DEG)
        release_quat = motion._q_from_ypr(*config.WEIGH_RELEASE_YPR_DEG)
        regrasp_quat = motion._q_from_ypr(*config.WEIGH_REGRASP_YPR_DEG)
        handoff_quat = motion._q_from_ypr(*config.HANDOFF_RIGHT_YPR)
        receive_quat = motion._q_from_ypr(*config.HANDOFF_LEFT_YPR)
        box_quat = motion._q_from_ypr(*config.LEFT_BOX_YPR)
        left_worker = LeftDropWorker(
            left_publisher,
            left_hold,
            gripper_hold,
            box_quat,
            receive_quat,
        )
        parcels = [
            (index, name, parcel_poses[name])
            for index, name in enumerate(config.PARCEL_NAMES)
            if name in parcel_poses
        ]
        for order, (parcel_index, parcel_name, parcel_position) in enumerate(parcels):
            rospy.loginfo(
                "--- threaded %s (%d/%d) ---",
                parcel_name,
                order + 1,
                len(parcels),
            )
            lift_ik = motion.right_pick(
                right_publisher,
                right_hold,
                gripper_hold,
                parcel_name,
                parcel_position,
                pick_quat,
                on_grasped=left_worker.request_ready,
                before_grasp_mid=lambda: left_worker.wait(
                    "right pre-to-grasp-mid descent"
                ),
            )

            motion.right_weigh_and_regrasp(
                right_publisher,
                right_hold,
                gripper_hold,
                parcel_name,
                lift_ik,
                release_quat,
                regrasp_quat,
                before_strict=lambda: left_worker.wait_ready_request(
                    "regrasp shared MPC+WBC control"
                ),
            )
            _handoff_and_dispatch_left(
                both_publisher,
                both_hold,
                right_publisher,
                right_hold,
                gripper_hold,
                left_worker,
                parcel_index,
                parcel_name,
                handoff_quat,
                receive_quat,
            )
            rospy.loginfo(
                "task1: %s handed off; right arm starts next pick",
                parcel_name,
            )

        left_worker.wait("task completion")
        rospy.loginfo("task1 all done seed=%d", seed)
        return True
    except Exception:
        rospy.logerr("task1 exception: %s", traceback.format_exc())
        return False
    finally:
        if left_worker is not None:
            try:
                left_worker.wait_ready_request("shutdown")
                left_worker.wait("shutdown")
            except Exception:
                rospy.logerr(
                    "task1 left worker shutdown failure: %s",
                    traceback.format_exc(),
                )
        _terminate_process_group(bag_process, "rosbag")
        if gripper_hold is not None:
            gripper_hold.stop()
        if raw_arm_hold is not None:
            raw_arm_hold.stop()
