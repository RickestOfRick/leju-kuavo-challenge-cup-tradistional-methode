#!/usr/bin/env python3
"""
arm_test_control.py — absolute arm position + absolute gripper orientation test tool.

For the Kuavo 5 challenge simulation (ROBOT_VERSION = 52, 14 arm joints).

Start a scene first, then in a SECOND shell inside the same container:

    source /root/kuavo_ws/devel/setup.zsh
    python3 /root/kuavo_ws/src/challenge_cup_task_template/scripts/arm_test_control.py

Then type commands at the `arm>` prompt:

  ---- pose / info ----
    p                               print the current hand pose (xyz + rpy) and gripper
    q                               quit
    h                               help

  ---- ABSOLUTE position (metres, in the frame FK reports) ----
    pos X Y Z                       move the hand to X Y Z, KEEP the current orientation
    pos                             (no args) print BOTH hands' pos + rot + gripper
    x V | y V | z V                 set ONE coordinate absolutely, keep the other two
                                    and the orientation  (e.g. `z 0.45` -> height 0.45 m)
    dx V | dy V | dz V              nudge one coordinate by V (relative)

  ---- ABSOLUTE gripper orientation (degrees; roll/pitch/yaw, same as `p`) ----
    rot R P Y   (or `r`)            set the GRIPPER orientation, KEEP the position
    roll V | pitch V | yaw V        set ONE orientation angle, keep the other two
                                    and the position

  ---- ABSOLUTE position AND orientation together ----
    abs X Y Z R P Y                 set position and orientation in one move

  ---- gripper (only the active arm is moved) ----
    o                               open   (0 %)
    c                               close  (100 %)
    g PCT                           set to PCT % (0 = open, 100 = closed)

  ---- settings / helpers ----
    reset                           return BOTH arms to the neutral zero pose,
                                    open both grippers, restore default settings
    arm right | arm left            choose the active arm (default: right)
    t SEC                           move duration in seconds (default 2.0)
    ik MODE                         IK constraint mode: 2 = pos-hard,
                                    3 = pos+ori-hard, 6 = three-point mixed
                                    (default 6 — most permissive; auto-retries
                                    with 6 if a stricter mode fails)
    save [NAME]                     save the current pose (default name 'home')
    goto NAME                       move back to a saved pose
    list                            list saved poses

The arm is commanded exactly like the official challenge data-collection
scripts do it:
  * /ik/two_arm_hand_pose_cmd_srv  -> IK, with an EXPLICIT constraint mode so
                                      the orientation target is actually honoured
  * /kuavo_arm_target_poses        -> MPC arm target trajectory (degrees)
  * /kuavo_arm_traj (JointState)   -> 100 Hz joint-space stream, interpolated
                                      from the current pose to the target (degrees)
After each move the pose is re-read from FK and the residual position /
orientation error is printed, so a bad IK solve cannot pass silently.

Do NOT click or press keys inside the MuJoCo window (spacebar pauses the sim).
"""

import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf.transformations import (euler_from_quaternion, quaternion_from_euler)

from kuavo_msgs.msg import (armTargetPoses, ikSolveParam, lejuClawState,
                            sensorsData, twoArmHandPoseCmd)
from kuavo_msgs.srv import (changeArmCtrlMode, changeArmCtrlModeRequest,
                            controlLejuClaw, controlLejuClawRequest, fkSrv,
                            twoArmHandPoseCmdSrv)

# ───────────────────────── configuration ─────────────────────────
ARM_MODE = 2                       # 2 = external control
ARM_Q_IDX = (13, 27)               # arm joints in sensorsData.joint_data.joint_q
ARM_NAMES = ["arm_joint_%d" % i for i in range(1, 15)]
CLAW_NAMES = ["left_claw", "right_claw"]
CLAW_SIDE = {"left": 0, "right": 1}

FK_SRV = "/ik/fk_srv"
IK_SRV = "/ik/two_arm_hand_pose_cmd_srv"
CLAW_SRV = "/control_robot_leju_claw"
ARM_MODE_SRV = "/arm_traj_change_mode"
TARGET_POSES_TOPIC = "/kuavo_arm_target_poses"
ARM_TRAJ_TOPIC = "/kuavo_arm_traj"

# IK constraint modes (kuavo_msgs/ikSolveParam.msg + plantIK.h):
#   bit1 (0x02): position    hard(1)/soft(0)
#   bit0 (0x01): orientation hard(1)/soft(0)
#   bit2 (0x04): three-point position mode(1)/traditional pose mode(0)
#     0x00 pos soft + ori soft*    0x01 pos soft + ori hard*
#     0x02 pos hard + ori soft     0x03 pos hard + ori hard
#     0x04 three-point all-soft    0x06 three-point mixed (no hard orientation)
#   (* the node swaps 00<->01 for safety, so external 0 == internal 0x01)
# The official scene1 collector uses 0x06 for general moves and only switches
# to 0x03 for the final precise grasp, because 0x03 is easy to make infeasible.
IK_MODE_POS_HARD_ORI_SOFT = 0x02
IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_MODE_THREE_POINT_MIXED = 0x06
DEFAULT_IK_MODE = IK_MODE_THREE_POINT_MIXED

MOVE_SECONDS = 2.0                 # default duration of a move
SETTLE_SECONDS = 0.4               # extra wait after a move before verifying
STREAM_HZ = 100.0                  # /kuavo_arm_traj publish rate
GRIPPER_SETTLE = 1.0               # wait after a gripper command
TILT_LIMIT_DEG = 25.0              # abort if the base tilts more than this
MAX_ABS_XY = 1.2                   # refuse an absolute XY beyond this from the base
POS_TOL_MM = 15.0                  # warn above this position error
ORI_TOL_DEG = 8.0                  # warn above this orientation error
SERVICE_TIMEOUT = 30.0
POSE_FILE = "/root/.ros/arm_test_poses.json"
# ─────────────────────────────────────────────────────────────────


def log(msg):
    print(msg, flush=True)


def ori_error_deg(q1, q2):
    """Absolute angle between two quaternions (xyzw), in degrees."""
    dot = abs(sum(float(a) * float(b) for a, b in zip(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


class ArmTrajStream(object):
    """Continuously publishes /kuavo_arm_traj at STREAM_HZ, holding the latest
    setpoint in degrees. Moves interpolate the setpoint, the thread does the
    publishing — the same pattern the official collector uses."""

    def __init__(self, pub, degrees, hz=STREAM_HZ):
        self._pub = pub
        self._hz = float(hz)
        self._lock = threading.Lock()
        self._degrees = [float(v) for v in degrees]
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="arm_traj_stream",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set(self, degrees):
        d = [float(v) for v in degrees]
        with self._lock:
            self._degrees = d

    def get(self):
        with self._lock:
            return list(self._degrees)

    def _run(self):
        rate = rospy.Rate(self._hz)
        while not self._stop.is_set() and not rospy.is_shutdown():
            with self._lock:
                degrees = list(self._degrees)
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ARM_NAMES
            msg.position = degrees
            self._pub.publish(msg)
            rate.sleep()


class ArmTestControl(object):
    def __init__(self):
        self.active = "right"
        self.move_time = MOVE_SECONDS
        self.settle = SETTLE_SECONDS
        self.ik_mode = DEFAULT_IK_MODE
        self.lock = threading.Lock()
        self.joint_q = None
        self.roll = self.pitch = 0.0
        self.claw_pct = [None, None]
        self.poses = {}
        self.stream = None
        self._load_poses()

        rospy.Subscriber("/sensors_data_raw", sensorsData, self._on_sensors,
                         queue_size=1)
        rospy.Subscriber("/leju_claw_state", lejuClawState, self._on_claw,
                         queue_size=1)

        self.traj_pub = rospy.Publisher(ARM_TRAJ_TOPIC, JointState, queue_size=10)
        self.target_pub = rospy.Publisher(TARGET_POSES_TOPIC, armTargetPoses,
                                          queue_size=10)
        self.stop_pub = rospy.Publisher("/stop_robot", Bool, queue_size=10)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.fk_srv = self.ik_srv = self.claw_srv = None

    # ───────────────────────────── callbacks
    def _on_sensors(self, msg):
        with self.lock:
            self.joint_q = list(msg.joint_data.joint_q[ARM_Q_IDX[0]:ARM_Q_IDX[1]])
            q = msg.imu_data.quat
            self.roll, self.pitch = self._quat_to_roll_pitch(q.x, q.y, q.z, q.w)

    def _on_claw(self, msg):
        with self.lock:
            if len(msg.data.position) >= 2:
                self.claw_pct[0] = float(msg.data.position[0])
                self.claw_pct[1] = float(msg.data.position[1])

    @staticmethod
    def _quat_to_roll_pitch(x, y, z, w):
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-9:
            return 0.0, 0.0
        x, y, z, w = x / n, y / n, z / n, w / n
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        return roll, pitch

    # ───────────────────────────── setup
    def setup(self):
        log("waiting for /sensors_data_raw ...")
        t0 = time.time()
        while self.joint_q is None and not rospy.is_shutdown():
            if time.time() - t0 > SERVICE_TIMEOUT:
                raise RuntimeError("no /sensors_data_raw — is a scene running?")
            time.sleep(0.1)

        for name in (IK_SRV, FK_SRV, CLAW_SRV, ARM_MODE_SRV):
            rospy.wait_for_service(name, timeout=SERVICE_TIMEOUT)
            log("  service OK: %s" % name)

        self.fk_srv = rospy.ServiceProxy(FK_SRV, fkSrv)
        self.ik_srv = rospy.ServiceProxy(IK_SRV, twoArmHandPoseCmdSrv)
        self.claw_srv = rospy.ServiceProxy(CLAW_SRV, controlLejuClaw)

        mode_req = changeArmCtrlModeRequest()
        mode_req.control_mode = ARM_MODE
        rospy.ServiceProxy(ARM_MODE_SRV, changeArmCtrlMode)(mode_req)
        log("  arm control mode -> %d (external control)" % ARM_MODE)

        # wait for the /kuavo_arm_target_poses subscriber (the arm-trajectory node)
        t0 = time.time()
        while self.target_pub.get_num_connections() == 0 and time.time() - t0 < 10.0:
            if rospy.is_shutdown():
                break
            time.sleep(0.2)
        if self.target_pub.get_num_connections() == 0:
            log("  WARNING: no subscriber on %s — moves may be ignored"
                % TARGET_POSES_TOPIC)

        with self.lock:
            q = list(self.joint_q)
        self.stream = ArmTrajStream(self.traj_pub, [math.degrees(v) for v in q])
        self.stream.start()
        log("  /kuavo_arm_traj stream started at %.0f Hz" % STREAM_HZ)
        log("ready.  active arm = %s   move = %.1f s   IK mode = 0x%02x"
            % (self.active, self.move_time, self.ik_mode))

    # ───────────────────────────── kinematics
    def fk(self, q=None):
        if q is None:
            with self.lock:
                q = list(self.joint_q)
        res = self.fk_srv(q)
        if not res.success:
            raise RuntimeError("FK failed")
        return res.hand_poses

    def current_pose(self):
        p = self.fk()
        pose = getattr(p, self.active + "_pose")
        return np.array(pose.pos_xyz, dtype=float), list(pose.quat_xyzw)

    def _ik_param(self, mode=None):
        p = ikSolveParam()
        p.major_optimality_tol = 1e-3
        p.major_feasibility_tol = 1e-3
        p.minor_feasibility_tol = 1e-3
        p.major_iterations_limit = 500
        p.oritation_constraint_tol = 1e-3
        p.pos_constraint_tol = 1e-3
        p.pos_cost_weight = 1.0
        p.constraint_mode = self.ik_mode if mode is None else mode
        return p

    def _solve_ik(self, q_cur, l_pos, l_quat, r_pos, r_quat, mode):
        """Call the IK service. Returns (q_cmd_14 | None, response)."""
        cmd = twoArmHandPoseCmd()
        cmd.use_custom_ik_param = True
        cmd.joint_angles_as_q0 = True
        cmd.frame = 0
        cmd.ik_param = self._ik_param(mode)
        cmd.hand_poses.left_pose.joint_angles = q_cur[:7]
        cmd.hand_poses.right_pose.joint_angles = q_cur[7:]
        cmd.hand_poses.left_pose.elbow_pos_xyz = np.zeros(3)
        cmd.hand_poses.right_pose.elbow_pos_xyz = np.zeros(3)
        cmd.hand_poses.left_pose.pos_xyz = l_pos
        cmd.hand_poses.left_pose.quat_xyzw = l_quat
        cmd.hand_poses.right_pose.pos_xyz = r_pos
        cmd.hand_poses.right_pose.quat_xyzw = r_quat

        res = self.ik_srv(cmd)
        if not res.success:
            return None, res
        if self.active == "right":
            q_cmd = list(q_cur[:7]) + list(res.hand_poses.right_pose.joint_angles)
        else:
            q_cmd = list(res.hand_poses.left_pose.joint_angles) + list(q_cur[7:])
        return q_cmd, res

    # ───────────────────────────── motion
    def move_to(self, pos=None, quat=None):
        """Move the ACTIVE hand to an absolute position and/or quaternion.
        Whichever of pos/quat is None is taken from the current FK pose."""
        cur_pos, cur_quat = self.current_pose()
        tgt_pos = np.array(pos, dtype=float) if pos is not None else cur_pos.copy()
        tgt_quat = list(quat) if quat is not None else list(cur_quat)

        d_xy = math.hypot(tgt_pos[0], tgt_pos[1])
        if d_xy > MAX_ABS_XY:
            log("  REFUSED: |xy| = %.2f m > MAX_ABS_XY = %.2f m" % (d_xy, MAX_ABS_XY))
            return False

        with self.lock:
            q_cur = list(self.joint_q)
        fk = self.fk(q_cur)
        l_pos = np.array(fk.left_pose.pos_xyz, dtype=float)
        r_pos = np.array(fk.right_pose.pos_xyz, dtype=float)
        l_quat = list(fk.left_pose.quat_xyzw)
        r_quat = list(fk.right_pose.quat_xyzw)

        if self.active == "right":
            r_pos, r_quat = tgt_pos, tgt_quat
        else:
            l_pos, l_quat = tgt_pos, tgt_quat

        q_cmd, res = self._solve_ik(q_cur, l_pos, l_quat, r_pos, r_quat, self.ik_mode)
        if q_cmd is None and self.ik_mode != IK_MODE_THREE_POINT_MIXED:
            log("  IK failed with mode 0x%02x; retrying with three-point mixed 0x06 ..."
                % self.ik_mode)
            q_cmd, res = self._solve_ik(q_cur, l_pos, l_quat, r_pos, r_quat,
                                        IK_MODE_THREE_POINT_MIXED)
        if q_cmd is None:
            log("  IK FAILED — the solver found no solution.")
            log("  This service leaves error_reason empty; the real reason is in the")
            log("  Terminal 1 log (look for 'out of workspace!'). Likely causes:")
            log("    * target too far / too high / too low -> move in smaller steps")
            log("    * orientation unreachable at that position -> try `ik 2` or `ik 6`")
            log("    * solver time_cost %.1f ms" % res.time_cost)
            return False

        log("  ABS %-5s -> pos (%7.3f, %7.3f, %7.3f)   rpy %s"
            % (self.active, tgt_pos[0], tgt_pos[1], tgt_pos[2],
               "(kept)" if quat is None else
               "(%5.1f, %5.1f, %5.1f)" % tuple(math.degrees(v)
                                                for v in euler_from_quaternion(tgt_quat))))

        start_deg = [math.degrees(v) for v in q_cur]
        target_deg = [math.degrees(v) for v in q_cmd]

        # MPC arm target trajectory (degrees, one waypoint at move_time)
        tmsg = armTargetPoses()
        tmsg.times = [float(self.move_time)]
        tmsg.values = [float(v) for v in target_deg]
        self.target_pub.publish(tmsg)

        # joint-space stream, interpolated from start to target
        self._interpolate(start_deg, target_deg, self.move_time)
        rospy.sleep(self.settle or SETTLE_SECONDS)

        if not self._upright():
            return False
        return self._verify(tgt_pos, tgt_quat)

    def move_joints(self, degrees, duration=None):
        """Move BOTH arms to a joint-space preset (14 joint angles, degrees).
        This is how the ACT deploy pipeline reaches its ready pose: joint-space
        playback instead of Cartesian IK, so the two arms land symmetrically."""
        dur = float(duration if duration is not None else self.move_time)
        if len(degrees) != 14:
            log("  move_joints needs 14 values, got %d" % len(degrees))
            return False
        with self.lock:
            q_cur = list(self.joint_q)
        start_deg = [math.degrees(v) for v in q_cur]
        target_deg = [float(v) for v in degrees]
        log("  JOINTS -> [%s]" % ", ".join("%.1f" % v for v in target_deg))
        tmsg = armTargetPoses()
        tmsg.times = [dur]
        tmsg.values = list(target_deg)
        self.target_pub.publish(tmsg)
        self._interpolate(start_deg, target_deg, dur)
        rospy.sleep(self.settle or SETTLE_SECONDS)
        return self._upright()

    def _interpolate(self, start_deg, target_deg, duration):
        steps = max(1, int(round(duration * STREAM_HZ)))
        rate = rospy.Rate(STREAM_HZ)
        for i in range(steps + 1):
            if rospy.is_shutdown():
                break
            a = float(i) / float(steps)
            self.stream.set([start_deg[j] + (target_deg[j] - start_deg[j]) * a
                             for j in range(len(target_deg))])
            if i < steps:
                rate.sleep()

    def _verify(self, tgt_pos, tgt_quat):
        act_pos, act_quat = self.current_pose()
        perr = float(np.linalg.norm(act_pos - tgt_pos)) * 1000.0
        oerr = ori_error_deg(act_quat, tgt_quat)
        log("  reached (%7.3f, %7.3f, %7.3f)   pos err %5.1f mm   ori err %5.1f deg"
            % (act_pos[0], act_pos[1], act_pos[2], perr, oerr))
        if perr > POS_TOL_MM or oerr > ORI_TOL_DEG:
            log("  !! target not reached well (pos tol %.0f mm, ori tol %.0f deg)"
                % (POS_TOL_MM, ORI_TOL_DEG))
            return False
        return True

    def _upright(self):
        r = math.degrees(self.roll)
        p = math.degrees(self.pitch)
        if abs(r) > TILT_LIMIT_DEG or abs(p) > TILT_LIMIT_DEG:
            log("  !! TILT roll %.1f / pitch %.1f — stopping" % (r, p))
            self.vel_pub.publish(Twist())
            self.stop_pub.publish(Bool(True))
            return False
        return True

    # ───────────────────────────── gripper
    def set_gripper(self, pct):
        pct = max(0.0, min(100.0, float(pct)))
        with self.lock:
            left = self.claw_pct[0] if self.claw_pct[0] is not None else 0.0
            right = self.claw_pct[1] if self.claw_pct[1] is not None else 0.0
        if self.active == "right":
            right = pct
        else:
            left = pct
        req = controlLejuClawRequest()
        req.data.name = CLAW_NAMES
        req.data.position = [left, right]
        req.data.velocity = [80.0, 80.0]
        req.data.effort = [1.0, 1.0]
        self.claw_srv(req)
        time.sleep(GRIPPER_SETTLE)
        log("  %s gripper -> %.0f%%" % (self.active, pct))

    # ───────────────────────────── reset
    def reset_all(self):
        """Return BOTH arms to the neutral (all-zero) joint pose, open both
        grippers and restore the default settings."""
        with self.lock:
            q_cur = list(self.joint_q)
        start_deg = [math.degrees(v) for v in q_cur]
        target_deg = [0.0] * len(ARM_NAMES)

        log("  reset: moving both arms to the zero pose ...")
        tmsg = armTargetPoses()
        tmsg.times = [float(self.move_time)]
        tmsg.values = list(target_deg)
        self.target_pub.publish(tmsg)
        self._interpolate(start_deg, target_deg, self.move_time)
        rospy.sleep(self.settle or SETTLE_SECONDS)

        req = controlLejuClawRequest()
        req.data.name = CLAW_NAMES
        req.data.position = [0.0, 0.0]
        req.data.velocity = [80.0, 80.0]
        req.data.effort = [1.0, 1.0]
        self.claw_srv(req)
        time.sleep(GRIPPER_SETTLE)

        self.active = "right"
        self.move_time = MOVE_SECONDS
        self.settle = SETTLE_SECONDS
        self.ik_mode = DEFAULT_IK_MODE
        with self.lock:
            self.claw_pct = [0.0, 0.0]

        self._upright()
        pos, _ = self.current_pose()
        log("  reset done: arms zeroed, both grippers open,"
            " move=%.1fs  ik=0x%02x  active=right" % (self.move_time, self.ik_mode))
        log("             right hand now at (%7.3f, %7.3f, %7.3f)"
            % (pos[0], pos[1], pos[2]))

    # ───────────────────────────── pose bookmarks
    def _load_poses(self):
        try:
            import json
            with open(POSE_FILE) as f:
                self.poses = json.load(f)
        except Exception:
            self.poses = {}

    def _save_poses(self):
        try:
            import json
            with open(POSE_FILE, "w") as f:
                json.dump(self.poses, f, indent=2)
        except Exception as e:
            log("  (could not save poses: %s)" % e)

    def print_pose(self):
        pos, quat = self.current_pose()
        r, p, y = euler_from_quaternion(quat)
        side = CLAW_SIDE[self.active]
        log("  %s hand  xyz = (%7.3f, %7.3f, %7.3f)"
            % (self.active, pos[0], pos[1], pos[2]))
        log("            rpy = (%6.1f, %6.1f, %6.1f) deg   quat = (%.3f, %.3f, %.3f, %.3f)"
            % (math.degrees(r), math.degrees(p), math.degrees(y),
               quat[0], quat[1], quat[2], quat[3]))
        log("  gripper %.1f%%" % (self.claw_pct[side] if self.claw_pct[side] is not None else -1.0))

    def print_all(self):
        """Print both hands' position + orientation, copy-paste friendly."""
        fk = self.fk()
        for name in ("left", "right"):
            pose = getattr(fk, name + "_pose")
            pos = np.array(pose.pos_xyz, dtype=float)
            quat = list(pose.quat_xyzw)
            r, p, y = euler_from_quaternion(quat)
            rd, pd, yd = math.degrees(r), math.degrees(p), math.degrees(y)
            grip = self.claw_pct[CLAW_SIDE[name]]
            log("  %-5s pos=(%7.3f, %7.3f, %7.3f)  rpy=(%6.1f, %6.1f, %6.1f) deg"
                % (name, pos[0], pos[1], pos[2], rd, pd, yd))
            log("        quat=(%.3f, %.3f, %.3f, %.3f)  gripper=%s"
                % (quat[0], quat[1], quat[2], quat[3],
                   "?" if grip is None else "%.1f%%" % grip))
            log("        copy: %s pos=(%.4f, %.4f, %.4f)  rpy=(%.1f, %.1f, %.1f)"
                % (name, pos[0], pos[1], pos[2], rd, pd, yd))

    # ───────────────────────────── REPL
    def repl(self):
        log("")
        log("type a command (h = help, q = quit)")
        while not rospy.is_shutdown():
            try:
                line = input("arm> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            a = parts[1:]

            try:
                if cmd in ("q", "quit", "exit"):
                    break
                elif cmd in ("h", "help", "?"):
                    log(__doc__)
                elif cmd == "p":
                    self.print_pose()
                elif cmd == "pos":
                    if len(a) >= 3:
                        self.move_to(pos=[float(v) for v in a[:3]])
                    else:
                        self.print_all()
                elif cmd in ("x", "y", "z"):
                    pos, _ = self.current_pose()
                    pos[{"x": 0, "y": 1, "z": 2}[cmd]] = float(a[0])
                    self.move_to(pos=pos)
                elif cmd in ("dx", "dy", "dz"):
                    pos, _ = self.current_pose()
                    pos[{"dx": 0, "dy": 1, "dz": 2}[cmd]] += float(a[0])
                    self.move_to(pos=pos)
                elif cmd in ("rot", "r"):
                    r, p_, y = [math.radians(float(v)) for v in a[:3]]
                    self.move_to(quat=list(quaternion_from_euler(r, p_, y)))
                elif cmd in ("roll", "pitch", "yaw"):
                    _, quat = self.current_pose()
                    rpy = list(euler_from_quaternion(quat))
                    rpy[{"roll": 0, "pitch": 1, "yaw": 2}[cmd]] = \
                        math.radians(float(a[0]))
                    self.move_to(quat=list(quaternion_from_euler(*rpy)))
                elif cmd == "abs":
                    r, p_, y = [math.radians(float(v)) for v in a[3:6]]
                    self.move_to(pos=[float(v) for v in a[:3]],
                                 quat=list(quaternion_from_euler(r, p_, y)))
                elif cmd in ("o", "open"):
                    self.set_gripper(0.0)
                elif cmd in ("c", "close"):
                    self.set_gripper(100.0)
                elif cmd == "g":
                    self.set_gripper(float(a[0]))
                elif cmd == "arm":
                    if a and a[0].lower() in ("left", "right"):
                        self.active = a[0].lower()
                        log("  active arm = %s" % self.active)
                    else:
                        log("  usage: arm right | arm left")
                elif cmd == "t":
                    self.move_time = float(a[0])
                    log("  move duration = %.1f s" % self.move_time)
                elif cmd == "ik":
                    self.ik_mode = int(a[0], 0)
                    log("  IK constraint mode = 0x%02x" % self.ik_mode)
                elif cmd == "reset":
                    self.reset_all()
                elif cmd == "save":
                    name = a[0] if a else "home"
                    pos, quat = self.current_pose()
                    self.poses[name] = {"pos": [float(v) for v in pos],
                                        "quat": [float(v) for v in quat]}
                    self._save_poses()
                    log("  saved '%s'" % name)
                elif cmd == "goto":
                    if not a or a[0] not in self.poses:
                        log("  no saved pose '%s' (use 'list')" % (a[0] if a else ""))
                    else:
                        p = self.poses[a[0]]
                        self.move_to(pos=p["pos"], quat=p["quat"])
                elif cmd == "list":
                    log("  saved: %s" % (", ".join(self.poses) or "(none)"))
                else:
                    log("  unknown command: %s   (h = help)" % cmd)
            except (ValueError, IndexError):
                log("  bad arguments.  h = help")
            except Exception as e:
                log("  error: %s" % e)

    def shutdown(self):
        if self.stream is not None:
            self.stream.stop()
        try:
            self.vel_pub.publish(Twist())
            self.stop_pub.publish(Bool(True))
        except Exception:
            pass


def main():
    rospy.init_node("arm_test_control", anonymous=True)
    c = ArmTestControl()
    try:
        c.setup()
        c.repl()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as e:
        rospy.logerr("arm_test_control failed: %s", e)
    finally:
        c.shutdown()
    log("bye")


if __name__ == "__main__":
    main()
