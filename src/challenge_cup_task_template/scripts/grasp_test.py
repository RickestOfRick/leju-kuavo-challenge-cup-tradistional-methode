#!/usr/bin/env python3
"""
grasp_test.py — configurable grasp test for Kuavo scene 1.

Modes (set MODE below):
    "calibrate"       print the hand's current XYZ + orientation once a second.
                      Use this to FIND absolute coordinates: move the arm with
                      arm_keyboard_control.py and read the numbers here.
    "measure_gripper" open/close the gripper and report its positions.
    "absolute"        move the hand to TARGET_XYZ with orientation TARGET_RPY,
                      close the gripper, check, lift.
    "relative"        move by small offsets from the current pose.

Run inside the container:
    python3 src/challenge_cup_task_template/scripts/grasp_test.py
"""

import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from kuavo_msgs.msg import (armTargetPoses, lejuClawState, sensorsData,
                            twoArmHandPoseCmd)
from kuavo_msgs.srv import (changeArmCtrlMode, controlLejuClaw,
                            controlLejuClawRequest, fkSrv, twoArmHandPoseCmdSrv)

# ══════════════════════════════════════════════════════════════════
#  CONFIG  —  edit these
# ══════════════════════════════════════════════════════════════════

ARM = "right"                 # "right" or "left"

MODE = "calibrate"            # "calibrate" | "measure_gripper" | "absolute" | "relative"

# ── ABSOLUTE target (used when MODE = "absolute") ──────────────────
# metres, in the SAME frame that FK reports. Find the numbers with MODE="calibrate".
TARGET_XYZ = (0.30, -0.15, 0.90)      # (x, y, z)

# Orientation of the HAND, in degrees:  (roll, pitch, yaw)
#   pitch = how far the gripper tips forward/down   (90 = pointing down)
#   yaw   = rotation about the vertical axis        (choose your approach angle)
#   roll  = roll about the gripper axis             (usually 0)
TARGET_RPY_DEG = (0.0, 90.0, 0.0)

# ── RELATIVE offsets (used when MODE = "relative") ─────────────────
APPROACH_OFFSET = (0.00, 0.00,  0.10)   # up   (10 cm)
GRASP_OFFSET    = (0.00, 0.00, -0.10)   # down (10 cm)
LIFT_OFFSET     = (0.00, 0.00,  0.10)   # up   (10 cm)

# ── motion ─────────────────────────────────────────────────────────
ARM_MODE = 2                  # 2 = external control
MOVE_SECONDS = 4.0            # duration of each arm movement (slower = safer)
SETTLE_AFTER_MOVE = 1.0       # extra wait after each move (s)
PUBLISH_HZ = 100.0            # arm targets must be republished at this rate

# ── gripper ────────────────────────────────────────────────────────
GRIPPER_OPEN_PCT  = 0.0       # 0   = fully open
GRIPPER_CLOSE_PCT = 100.0     # 100 = fully closed
GRIPPER_SETTLE    = 1.5       # wait after each gripper command (s)
HOLD_THRESHOLD    = 90.0      # closed position BELOW this -> something is held

# ── safety ─────────────────────────────────────────────────────────
TILT_LIMIT_DEG = 25.0
STOP_ON_TILT = True
MAX_ABS_XY = 1.2              # refuse an absolute XY beyond this from the base (m)

# ── interfaces ─────────────────────────────────────────────────────
FK_SRV = "/ik/fk_srv"
IK_SRV = "/ik/two_arm_hand_pose_cmd_srv"
CLAW_SRV = "/control_robot_leju_claw"
ARM_MODE_SRV = "/arm_traj_change_mode"
ARM_TRAJ_TOPIC = "/kuavo_arm_target_poses"

ARM_Q_IDX = (13, 27)          # ROBOT_VERSION = 52
CLAW_SIDE = {"left": 0, "right": 1}

# ══════════════════════════════════════════════════════════════════


def log(msg):
    rospy.loginfo(msg)
    print(msg, flush=True)


def quat_to_roll_pitch(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return 0.0, 0.0
    x, y, z, w = x / n, y / n, z / n, w / n
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


class GraspTest(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.joint_q = None
        self.claw_pct = [None, None]
        self.claw_state = [lejuClawState.kUnknown] * 2
        self.roll = self.pitch = 0.0
        self.measurements = []

        rospy.Subscriber("/sensors_data_raw", sensorsData, self._on_sensors,
                         queue_size=1)
        rospy.Subscriber("/leju_claw_state", lejuClawState, self._on_claw,
                         queue_size=1)

        self.arm_pub = rospy.Publisher(ARM_TRAJ_TOPIC, armTargetPoses,
                                       queue_size=10)
        self.stop_pub = rospy.Publisher("/stop_robot", Bool, queue_size=10)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.fk_srv = self.ik_srv = self.claw_srv = None

    # ───────────────────────────────── callbacks
    def _on_sensors(self, msg):
        with self.lock:
            self.joint_q = list(msg.joint_data.joint_q[ARM_Q_IDX[0]:ARM_Q_IDX[1]])
            q = msg.imu_data.quat
            self.roll, self.pitch = quat_to_roll_pitch(q.x, q.y, q.z, q.w)

    def _on_claw(self, msg):
        with self.lock:
            if len(msg.data.position) >= 2:
                self.claw_pct[0] = float(msg.data.position[0])
                self.claw_pct[1] = float(msg.data.position[1])
            if len(getattr(msg, "state", [])) >= 2:
                self.claw_state = [int(msg.state[0]), int(msg.state[1])]

    # ───────────────────────────────── setup
    def wait_for_everything(self, timeout=30.0):
        log("waiting for /sensors_data_raw ...")
        t0 = time.time()
        while self.joint_q is None and not rospy.is_shutdown():
            if time.time() - t0 > timeout:
                raise RuntimeError("no /sensors_data_raw received")
            time.sleep(0.1)
        for name in (IK_SRV, FK_SRV, CLAW_SRV, ARM_MODE_SRV):
            rospy.wait_for_service(name, timeout=timeout)
            log("  service OK: %s" % name)

        self.fk_srv = rospy.ServiceProxy(FK_SRV, fkSrv)
        self.ik_srv = rospy.ServiceProxy(IK_SRV, twoArmHandPoseCmdSrv)
        self.claw_srv = rospy.ServiceProxy(CLAW_SRV, controlLejuClaw)
        rospy.ServiceProxy(ARM_MODE_SRV, changeArmCtrlMode)(ARM_MODE)
        log("  arm control mode -> %d" % ARM_MODE)

    # ───────────────────────────────── safety
    def tilted(self):
        return (abs(math.degrees(self.roll)) > TILT_LIMIT_DEG or
                abs(math.degrees(self.pitch)) > TILT_LIMIT_DEG)

    def safety_stop(self, why):
        rospy.logerr("SAFETY STOP: %s (roll %.1f pitch %.1f)",
                     why, math.degrees(self.roll), math.degrees(self.pitch))
        self.vel_pub.publish(Twist())
        self.stop_pub.publish(Bool(True))
        time.sleep(1.0)

    def check_upright(self):
        if STOP_ON_TILT and self.tilted():
            self.safety_stop("tilt limit")
            return False
        return True

    # ───────────────────────────────── kinematics
    def fk(self):
        with self.lock:
            q = list(self.joint_q)
        res = self.fk_srv(q)
        if not res.success:
            raise RuntimeError("FK failed")
        return res.hand_poses

    def current_pose(self):
        p = self.fk()
        side = "right_pose" if ARM == "right" else "left_pose"
        pose = getattr(p, side)
        return (np.array(pose.pos_xyz, dtype=float),
                list(pose.quat_xyzw))

    def send_targets(self, l_pos, l_quat, r_pos, r_quat):
        with self.lock:
            q = list(self.joint_q)
        cmd = twoArmHandPoseCmd()
        cmd.use_custom_ik_param = False
        cmd.joint_angles_as_q0 = True
        cmd.frame = 0
        cmd.hand_poses.left_pose.joint_angles = q[:7]
        cmd.hand_poses.right_pose.joint_angles = q[-7:]
        cmd.hand_poses.left_pose.elbow_pos_xyz = np.zeros(3)
        cmd.hand_poses.right_pose.elbow_pos_xyz = np.zeros(3)
        cmd.hand_poses.left_pose.pos_xyz = l_pos
        cmd.hand_poses.left_pose.quat_xyzw = l_quat
        cmd.hand_poses.right_pose.pos_xyz = r_pos
        cmd.hand_poses.right_pose.quat_xyzw = r_quat
        res = self.ik_srv(cmd)
        if not res.success:
            raise RuntimeError("IK failed: %s" % res.error_reason)
        qq = (list(res.hand_poses.left_pose.joint_angles) +
              list(res.hand_poses.right_pose.joint_angles))
        msg = armTargetPoses()
        msg.times = [MOVE_SECONDS]
        msg.values = [math.degrees(v) for v in qq]

        # The arm controller needs the target REPEATED, not sent once.
        # The organisers' own scripts publish at 100 Hz, so we do the same.
        rate = rospy.Rate(PUBLISH_HZ)
        t_end = time.time() + MOVE_SECONDS + SETTLE_AFTER_MOVE
        while time.time() < t_end and not rospy.is_shutdown():
            self.arm_pub.publish(msg)
            rate.sleep()
        return self.check_upright()

    def move_by_offset(self, offset, label):
        cur_pos, cur_quat = self.current_pose()
        p = self.fk()
        l_pos = np.array(p.left_pose.pos_xyz, dtype=float)
        r_pos = np.array(p.right_pose.pos_xyz, dtype=float)
        l_quat = list(p.left_pose.quat_xyzw)
        r_quat = list(p.right_pose.quat_xyzw)
        if ARM == "right":
            r_pos = r_pos + np.array(offset, dtype=float)
        else:
            l_pos = l_pos + np.array(offset, dtype=float)
        log("  %-10s -> L %s  R %s" % (label, np.round(l_pos, 3),
                                       np.round(r_pos, 3)))
        return self.send_targets(l_pos, l_quat, r_pos, r_quat)

    def move_to_xyz(self, xyz, rpy_deg, label):
        """go to an ABSOLUTE hand position + orientation"""
        x, y, z = [float(v) for v in xyz]
        if math.sqrt(x * x + y * y) > MAX_ABS_XY:
            raise RuntimeError(
                "refusing: |xy| = %.2f m > MAX_ABS_XY = %.2f m"
                % (math.sqrt(x * x + y * y), MAX_ABS_XY))
        r, pch, yw = [math.radians(float(v)) for v in rpy_deg]
        quat = list(quaternion_from_euler(r, pch, yw))

        fk = self.fk()
        l_pos = np.array(fk.left_pose.pos_xyz, dtype=float)
        r_pos = np.array(fk.right_pose.pos_xyz, dtype=float)
        l_quat = list(fk.left_pose.quat_xyzw)
        r_quat = list(fk.right_pose.quat_xyzw)

        tgt = np.array([x, y, z], dtype=float)
        if ARM == "right":
            r_pos, r_quat = tgt, quat
        else:
            l_pos, l_quat = tgt, quat

        log("  %-10s -> %s   rpy %s deg" % (label, np.round(tgt, 3),
                                            np.round(rpy_deg, 1)))
        return self.send_targets(l_pos, l_quat, r_pos, r_quat)

    # ───────────────────────────────── gripper
    def set_gripper(self, percent):
        req = controlLejuClawRequest()
        req.data.name = ["left_claw", "right_claw"]
        req.data.position = [percent, percent]
        req.data.velocity = []
        req.data.effort = []          # ignored by the simulator
        self.claw_srv(req)
        time.sleep(GRIPPER_SETTLE)

    def read_claw(self):
        with self.lock:
            return self.claw_pct[CLAW_SIDE[ARM]], self.claw_state[CLAW_SIDE[ARM]]

    def record(self, label, value, note=""):
        self.measurements.append((label, value, note))
        v = "%.3f" % value if isinstance(value, float) else str(value)
        log("  MEASURED  %-34s %-9s %s" % (label, v, note))

    def summary(self):
        log("\n──────────────── measured values ────────────────")
        log("%-36s %-10s %s" % ("what", "value", "note"))
        for label, value, note in self.measurements:
            v = "%.3f" % value if isinstance(value, float) else str(value)
            log("%-36s %-10s %s" % (label, v, note))
        log("─────────────────────────────────────────────────")

    # ───────────────────────────────── modes
    def mode_calibrate(self):
        log("\n=== CALIBRATE — printing the hand pose once a second ===")
        log("Move the arm with arm_keyboard_control.py and read the numbers.")
        log("Press Ctrl+C when you have the XYZ you want.\n")
        side = "right" if ARM == "right" else "left"
        while not rospy.is_shutdown():
            pos, quat = self.current_pose()
            r, p, y = euler_from_quaternion(quat)
            log("%s hand  xyz=(%7.3f, %7.3f, %7.3f)   "
                "rpy_deg=(%6.1f, %6.1f, %6.1f)"
                % (side, pos[0], pos[1], pos[2],
                   math.degrees(r), math.degrees(p), math.degrees(y)))
            time.sleep(1.0)

    def mode_measure_gripper(self):
        log("\n[A] measuring the gripper")
        self.set_gripper(GRIPPER_OPEN_PCT)
        pct, st = self.read_claw()
        self.record("gripper OPEN (pct)", pct or -1, "state=%d" % st)
        self.set_gripper(GRIPPER_CLOSE_PCT)
        pct2, st2 = self.read_claw()
        self.record("gripper CLOSED on air (pct)", pct2 or -1, "state=%d" % st2)
        self.summary()

    def mode_absolute(self):
        log("\n[A] gripper baseline")
        self.set_gripper(GRIPPER_OPEN_PCT)
        pct0, _ = self.read_claw()
        self.record("gripper OPEN (pct)", pct0 or -1)
        self.set_gripper(GRIPPER_CLOSE_PCT)
        pct1, _ = self.read_claw()
        self.record("gripper CLOSED on air (pct)", pct1 or -1)
        self.set_gripper(GRIPPER_OPEN_PCT)

        log("\n[B] moving to the absolute target")
        if not self.move_to_xyz(TARGET_XYZ, TARGET_RPY_DEG, "goto"):
            return
        pos, _ = self.current_pose()
        self.record("reached xyz", None,
                    "(%7.3f, %7.3f, %7.3f)" % (pos[0], pos[1], pos[2]))

        log("  closing gripper ...")
        self.set_gripper(GRIPPER_CLOSE_PCT)
        pct2, st2 = self.read_claw()
        self.record("gripper CLOSED at target (pct)", pct2 or -1,
                    "state=%d" % st2)
        holding = pct2 is not None and pct2 < HOLD_THRESHOLD
        log("  ---> %s" % ("HOLDING" if holding else "EMPTY"))

        log("\n[C] lift and verify")
        lift = (pos[0], pos[1], pos[2] + LIFT_OFFSET[2])
        if not self.move_to_xyz(lift, TARGET_RPY_DEG, "lift"):
            return
        pct3, _ = self.read_claw()
        self.record("gripper after LIFT (pct)", pct3 or -1)
        log("  ---> after lifting: %s"
            % ("STILL HOLDING" if (pct3 and pct3 < HOLD_THRESHOLD) else "LOST IT"))
        self.summary()

    def mode_relative(self):
        log("\n[B] relative sequence")
        if not self.move_by_offset(APPROACH_OFFSET, "approach"):
            return
        self.set_gripper(GRIPPER_OPEN_PCT)
        if not self.move_by_offset(GRASP_OFFSET, "descend"):
            return
        self.set_gripper(GRIPPER_CLOSE_PCT)
        pct, st = self.read_claw()
        self.record("gripper CLOSED after descend (pct)", pct or -1,
                    "state=%d" % st)
        log("  ---> %s" % ("HOLDING" if (pct and pct < HOLD_THRESHOLD) else "EMPTY"))
        if not self.move_by_offset(LIFT_OFFSET, "lift"):
            return
        pct2, _ = self.read_claw()
        self.record("gripper after LIFT (pct)", pct2 or -1)
        self.summary()

    def run(self):
        log("=== grasp_test  arm=%s  mode=%s ===" % (ARM, MODE))
        self.wait_for_everything()
        if MODE == "calibrate":
            self.mode_calibrate()
        elif MODE == "measure_gripper":
            self.mode_measure_gripper()
        elif MODE == "absolute":
            self.mode_absolute()
        elif MODE == "relative":
            self.mode_relative()
        else:
            raise RuntimeError("unknown MODE: %s" % MODE)

    def shutdown(self):
        try:
            self.vel_pub.publish(Twist())
            self.stop_pub.publish(Bool(True))
        except Exception:
            pass


def main():
    rospy.init_node("grasp_test", anonymous=True)
    t = GraspTest()
    try:
        t.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logerr("grasp_test failed: %s", exc)
    finally:
        t.shutdown()


if __name__ == "__main__":
    main()
