#!/usr/bin/env python3
"""task_sequence.py — scene 1: detect the parcels, then pick one.

    python3 task_sequence.py

At the `task>` prompt:

  ---- detection (perception.py) ----
    head                 tilt the head down
    cloud                capture RGB-D -> point cloud
    ransac               remove floor + table planes
    cluster              cluster + PCA -> objects (x, y, z, yaw)
    weigh                find the green weighing marker
    all                  cloud -> ransac -> cluster
    detect               head -> cloud -> ransac -> cluster -> classify -> weigh (one shot)
    live [SEC]           repeat detection every SEC s (default 1.0)
    pitch V | yaw V      head angles in degrees (pitch>0 looks down)

  ---- arm ----
    gobag [BAG]          replay the go-bag arm trajectory to the ready pose
    home [SEC]           both arms to the ready pose (joint space, default 1.5 s, no bag)
    sethome              capture the CURRENT arm joints as the retract waypoint for `home`
    pick [N]             phase 1: pick parcel N (align above -> descend -> clamp)
    wrist [Y P R]        print or set the 3 wrist joint angles (degrees)
    wstep [DEG] [SIDE]   diagnostic: step each wrist joint, report commanded vs achieved
    joints               print the current 14 arm joint angles (degrees) — to capture a pose
    j INDEX VALUE        set ONE arm joint (0..13) to VALUE degrees (0-6 left, 7-13 right)
    jbias [I V]          per-joint bias (deg): no args print · on|off force · I V set joint I to V
    lift [Z]             move the right hand straight to height Z (default cur z + 0.05)
    topdown              rotate the claw to point straight down (keep the hand position)
    unlock               release the `topdown` orientation lock (IK orientation again)
    freeze [on|off]      freeze the 3 wrist joints so moves only turn the shoulder/elbow
    scale [N]            phase 2: place on the scale, then regrasp
    slift                scale step 1 only: lift straight up to the carry height
    scarry               scale step 2 only: carry above the scale
    sdrop                scale step 3 only: descend to the drop height and open
    handoff              phase 3: hand the parcel to the left hand
    box [N]              phase 4: drop into the sorting box
    task [N]             the whole chain (pick -> scale -> handoff -> box) for parcel N
    runall               the whole chain for EVERY parcel, 0 -> 1 -> 2 -> 3
    67 [N]               bonus: the "6-7" arm swing (joint space), N cycles (default 4)
    set67                capture the CURRENT arm pose as the `67` base (arms out front)

  ---- info / tuning ----
    dets                 list raw detections
    classify             split detections into box + parcels
    parcels              list classified parcels (parcel[0..3])
    frame [X Y Z]        base_link -> organisers' IK-frame offset for pick
    pos X Y Z            move the hand to an absolute position (m)
    pose | p             print the current hand pose
    check                FK vs TF + the true claw axis
    h / q                help / quit

Non-interactive:

    python3 task_sequence.py --auto
"""

import argparse
import json
import math
import os
import time

import rospy

import perception as P

# ── "6-7" bonus move (joint control, no IK) ────────────────────────────────
SIX_SEVEN_SHOULDER_DEG = 45.0  # from READY, bring the SHOULDER this far FORWARD
                               # (SUBTRACTED from joint 0/7; flip the sign to go back)
SIX_SEVEN_JOINT = 3         # joint that SWINGS — 3 = ELBOW (shoulder stays put)
SIX_SEVEN_UP_DEG = 12.0     # SMALL swing (flip the sign if "up" is inverted)
SIX_SEVEN_TIME = 0.30       # seconds per half-swing (fast, bouncy)
SIX_SEVEN_CYCLES = 4        # up-down pairs
SIX_SEVEN_OPPOSITE = True   # True = arms alternate (6-7); False = both move together

DEFAULT_READY_BAG = "/root/kuavo_ws/go.bag"
# ready pose (14 arm joint angles, degrees) — captured from the go-bag replay,
# so we no longer need the bag for `home`
READY_JOINTS = [30.3, 0.0, 9.6, -137.2, -86.8, -2.9, -2.4,
                30.2, 0.0, -9.2, -138.4, 81.4, 8.0, -1.1]
HOME_TIME = 1.5                    # seconds per segment of the home move (was 1.0 — the swing
                                   # from over the box was fast enough to catch a parcel)
# "swing back" retract waypoint (14 joints, deg): shoulder rotated back and the
# elbow straighter than in the ready pose, so the arms clear the table on the way.
# First 7 = left arm, last 7 = right arm.
HOME_RETRACT = [60.0, 10.8, 9.6, -100.0, -86.8, -2.9, -2.4,
                60.0, -11.0, -9.2, -100.0, 81.4, 8.0, -1.1]
DET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.json")
WEIGH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weigh.json")

# ── scene-1 pick constants (ported from the organisers' collector) ─────────
PICK_IK_Z = -0.040                    # absolute grasp height (lower = deeper grip)
# grasp offset split by SIDE (right / left), decided by y vs the row's median y
PICK_OFFSET_RIGHT = [-0.04, -0.02, 0.0]
PICK_OFFSET_LEFT = [-0.03, 0.00, 0.0]
PICK_FIRST_YPR = [0.0, -90, 0.0]      # approach direction (horizontal)
PICK_SECOND_YPR = [90.0, 0.0, 0.0]    # tool roll
PICK_TRANSIT_Z = 0.120                # align height before descending
# optional extra offset on the pick point (m).  The organisers' pick uses the
# detected point as-is, so this stays zero; `frame X Y Z` tunes it live if needed.
FRAME_OFFSET = [0.0, 0.0, 0.0]

# ── scale (weigh) / handoff / box: fixed scene points, IK frame ────────────
WEIGH_TRANSIT_Z = 0.32            # higher carry height from the pick to the scale
WEIGH_RELEASE = [0.396, -0.574, 0.146217]   # x, y = fallback if no detection; [2] = drop height
WEIGH_DROP_Z =  -0.02             # hand height where it OPENS to drop the parcel (tune me)
WEIGH_REGRASP_Z = -0.040              # lower re-grasp (was -0.02)
# shift the scale placement toward the robot (the arm can't reach the centre
# comfortably). [x, y, z] in the base frame: +x away from the body, +y left.
WEIGH_OFFSET = [-0.03, 0.03, 0.0]
# tiny nudge applied to the re-grasp point after weighing (base frame: +x = away from the
# body, +y = LEFT, so "to the right" is -y).
REGRASP_OFFSET = [0.0, -0.01, 0.0]
WEIGH_RELEASE_FIRST_YPR = [-45.0, -90.0, 0.0]   # yaw 45 deg -> parcel not facing front
WEIGH_RELEASE_SECOND_YPR = [90.0, 0.0, 0.0]
WEIGH_REGRASP = [0.396, -0.574, -0.04]
WEIGH_REGRASP_FIRST_YPR = [-45.0, -60.0, 0.0]
WEIGH_REGRASP_SECOND_YPR = [90.0, 0.0, 0.0]
WEIGH_RELEASE_DELAY = 0.3             # pause BEFORE opening the claw on the scale
WEIGH_DWELL = 0.2                     # wait after releasing, before re-grasping
# The claw must point straight down when the parcel is placed on the scale, but the
# arm's small "points-up" tracking error leaves it a few degrees high. So we OVER-ROTATE
# the aim PAST vertical (≈ "curl more"). Positive = curl back toward the robot (-x);
# flip the sign to curl the other way.
WRIST_MAX_FIX = 0.08                  # ignore a wrist-camera fix further than this from the drop
USE_WRIST_LOCATE = False              # re-grasp via the wrist camera (OFF for now)
SCALE_TOPDOWN = False                 # default = ORIGINAL scale placement (no top-down); `td on` to enable
TOPDOWN_TILT_DEG = 8.0
_TOPDOWN_T = math.radians(TOPDOWN_TILT_DEG)
WEIGH_TOPDOWN_DIR = [-math.sin(_TOPDOWN_T), 0.0, -math.cos(_TOPDOWN_T)]

HANDOFF_FWD = 0.0                      # moves BOTH arms; does not change the gap
RIGHT_HANDOFF = [0.246 + HANDOFF_FWD, -0.044645, 0.3016983]
RIGHT_HANDOFF_TRANSIT_Z = 0.40
RIGHT_HANDOFF_FALLBACK_ZS = [0.37, 0.35]
RIGHT_HANDOFF_FIRST_YPR = [-0.839, -100.0, 0.0]
RIGHT_HANDOFF_SECOND_YPR = [90.0, -20.0, 90.0]
LEFT_RECEIVE_READY = [0.266, 0.209, 0.250]
LEFT_RECEIVE = [0.266, 0.05, 0.250]   # restored to the value that worked
# ── handoff poses: identical to collect_scene1_handoff_dataset.py ──
# left is defined RELATIVE to the right handoff, so moving RIGHT_HANDOFF moves both.
# LEFT_RECEIVE_Y: how far the left claw sits toward its own side (+y). Larger = it
# stops further away from the parcel (less deep). This is the "too deep" knob.
LEFT_RECEIVE_X_OFF = -0.005     # half of the previous pull-back (-0.015)
LEFT_RECEIVE_Y_OFF = 0.125      # final grab point (unchanged)
LEFT_RECEIVE_Z_OFF = -0.0850    # 10 mm higher than -0.0950
LEFT_RECEIVE_POS = [RIGHT_HANDOFF[0] + LEFT_RECEIVE_X_OFF,
                    RIGHT_HANDOFF[1] + LEFT_RECEIVE_Y_OFF,
                    RIGHT_HANDOFF[2] + LEFT_RECEIVE_Z_OFF]
LEFT_RECEIVE_READY_POS = [LEFT_RECEIVE_POS[0],
                          LEFT_RECEIVE_POS[1] + 0.0926,
                          LEFT_RECEIVE_POS[2]]

LEFT_RECEIVE_FIRST_YPR = [-0.839, -100.0, 0.0]
LEFT_RECEIVE_SECOND_YPR = [-90.0, -0.0, -90.0]
RIGHT_HANDOFF_RETRACT_Y = -0.300      # right hand retracts along y after release
LEFT_READY_PAUSE = 0.0                # left arm stops at the ready pose before reaching in
HANDOFF_LEFT_SETTLE = 0.3             # stop at the parcel before closing
HANDOFF_TRANSFER_PAUSE = 0.5          # left holds before the right releases
LEFT_APPROACH_Y = 0.100               # stop point further left (+y) before reaching in
LEFT_APPROACH_PAUSE = 0.8             # stop at the approach point before reaching in
HANDOFF_GRIP_EFFORT = 5.0             # left claw force when taking the parcel
RIGHT_GRIP_EFFORT = 4.0               # right claw while carrying (was the default 2.0)
PARCEL_HANG_L = 0.050                 # how far the parcel hangs below the right EEF along the claw axis
HANDOFF_FIX_TOL = 0.004               # only correct the left if it is >4 mm off
HANDOFF_STANDOFF = 0.050              # correct the left this far (along +y) behind the parcel, then approach
HANDOFF_RETRY_DY = 0.012              # if the left grab is EMPTY: push this far (-y) into the parcel, retry once
# Left-hand reach check via the LEFT wrist camera: if the parcel is still farther than
# LEFT_REACH_DIST, step the claw toward it (toward the right hand, -y) and re-measure.
LEFT_REACH_CHECK = False              # left wrist-camera reach check (OFF for now)
LEFT_REACH_DIST = 0.075               # "reached" when the parcel is within this (m)
LEFT_REACH_STEP = 0.010               # how far to step per retry (m)
LEFT_REACH_TRIES = 5                  # max corrections
# the left-hand receive point is defined RELATIVE to the right handoff point, so we
# can send it to where the right hand ACTUALLY is (the right sags under the parcel)
LEFT_RECEIVE_DELTA = [LEFT_RECEIVE[i] - RIGHT_HANDOFF[i] for i in range(3)]
LEFT_RECEIVE_READY_DELTA = [LEFT_RECEIVE_READY[i] - LEFT_RECEIVE[i] for i in range(3)]

BOX_DROP = [0.605966, 0.226114, 0.300]
BOX_DROP_X_FALLBACK = [-0.06, -0.10, -0.14]  # pull x toward the robot if IK fails (box near edge ~0.47)
BOX_SLOT_OFFSETS = [                  # 2x2 layout, by parcel index — disabled: all parcels
    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],  # go to the SAME drop point (set an offset here if
    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],  # you want the old 2x2 spread back)
]
LEFT_BOX_DROP_FIRST_YPR = [0.0, -90.0, 0.0]
LEFT_BOX_DROP_SECOND_YPR = [-90.0, 0.0, 0.0]
LEFT_PRESET_2 = [0.313, 0.239, 0.282]
LEFT_PRESET_2_FIRST_YPR = [-146.440, 4.966, 0.0]
LEFT_PRESET_2_SECOND_YPR = [0.0, 0.0, 96.580]
BOX_LIFT_Z = 0.450                    # lift the left hand to this height BEFORE crossing to the box
BOX_Z_COMP = 0.080                    # command the drop this much HIGHER: the arm sags ~70-80 mm
                                      # under the parcel. 0.300 + comp -> hand ~0.30, claw tip ~0.23,
                                      # i.e. just above the rim (0.219) instead of scraping it.
BOX_RETRACT = [-0.260, 0.0, 0.140]    # pull further back after the box drop
                                      # phase). x = how far back, z = rise during the pull so the hand
                                      # ends ~0.50 m high — home() then swings clear of the parcels
PLACE_DWELL = 0.0                    # wait after releasing in the box (0 = no wait)

# ── timings / IK modes ──
# Motion planning (MoveIt/OMPL) OFF by default: it needs `move_group` running and makes
# the motion non-deterministic (bad for ACT data). Toggle in the REPL with `mp on|off`.
WAIT_STILL_ENABLED = True             # wait for the arm to settle (ws off to disable)
USE_MOTION_PLANNING = False

MOVE_TIME = 1.0
SETTLE_TIME = 0.0                    # settle after each move (0 = no wait)
HANDOFF_MOVE_TIME = 2.2
RIGHT_HANDOFF_MOVE_TIME = 4.5         # the right arm gets there SLOWER (tracks/poses better)
SCALE_TO_HANDOFF_DELAY = 0.0          # the ONE remaining deliberate pause (after re-grasp)
PICK_GRASP_MOVE_TIME = 1.4
PLACE_MOVE_TIME = 2.2
GRIPPER_HOLD = 0.0                   # wait after every open/close (0 = no wait)
# The HANDOFF keeps its original settle times: it is the contact-critical step, so it
# gets its own constants instead of the global (now zeroed) ones.
HANDOFF_GRIPPER_HOLD = 0.0           # after every claw command inside do_handoff
                                     # (original: GRIPPER_HOLD 0.7 + GRIPPER_SETTLE 1.0)
HANDOFF_SETTLE_TIME = 0.0            # after each ABS move inside do_handoff
HANDOFF_MOVE_SETTLE = 0.0            # after each Cartesian (linear) move inside do_handoff
HEAD_SETTLE = 0.0
PICK_SETTLE_TIME = 1.2               # rest on top of the parcel before descending (was 0.0)
IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_MODE_THREE_POINT = 0x06

# parcel / box size thresholds (m), from the scene file
BOX_MIN_LENGTH = 0.30
BOX_MAX_LENGTH = 0.80                 # the sorting box is ~0.5 wide; walls are metres long
BOX_MAX_XY = 1.50                     # the box is within reach; walls/floor are far away
BOX_MIN_SIDE = 0.05                   # a box has depth — rejects flat strips/plane edges
PARCEL_L_MIN, PARCEL_L_MAX = 0.07, 0.20
PARCEL_W_MAX = 0.10
PARCEL_H_MAX = 0.20
MAX_PARCELS = 4
PARCEL_MERGE_DIST = 0.06


def _matmul3(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)]
            for r in range(3)]


def _rot_to_quat(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        w = math.sqrt(tr + 1.0) / 2.0
        x = (m[2][1] - m[1][2]) / (4.0 * w)
        y = (m[0][2] - m[2][0]) / (4.0 * w)
        z = (m[1][0] - m[0][1]) / (4.0 * w)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        x = math.sqrt(max(0.0, 1.0 + m[0][0] - m[1][1] - m[2][2])) / 2.0
        y = (m[0][1] + m[1][0]) / (4.0 * x)
        z = (m[0][2] + m[2][0]) / (4.0 * x)
        w = (m[2][1] - m[1][2]) / (4.0 * x)
    elif m[1][1] > m[2][2]:
        y = math.sqrt(max(0.0, 1.0 + m[1][1] - m[0][0] - m[2][2])) / 2.0
        x = (m[0][1] + m[1][0]) / (4.0 * y)
        z = (m[1][2] + m[2][1]) / (4.0 * y)
        w = (m[0][2] - m[2][0]) / (4.0 * y)
    else:
        z = math.sqrt(max(0.0, 1.0 + m[2][2] - m[0][0] - m[1][1])) / 2.0
        x = (m[0][2] + m[2][0]) / (4.0 * z)
        y = (m[1][2] + m[2][1]) / (4.0 * z)
        w = (m[1][0] - m[0][1]) / (4.0 * z)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return [0.0, 0.0, 0.0, 1.0] if n <= 0 else [x / n, y / n, z / n, w / n]


def _quat_from_ypr(first, second=None):
    """Two-rotation orientation: `first` = approach (yaw, pitch), `second` = tool roll."""
    yaw, pitch, _r = [math.radians(float(v)) for v in first]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    m = [[cy * cp, -sy, cy * sp],
         [sy * cp, cy, sy * sp],
         [-sp, 0.0, cp]]
    if second is not None:
        my, mp, mr = [math.radians(float(v)) for v in second]
        man = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if abs(my) > 0.01:
            c, s = math.cos(my), math.sin(my)
            man = _matmul3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], man)
        if abs(mp) > 0.01:
            c, s = math.cos(mp), math.sin(mp)
            man = _matmul3([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], man)
        if abs(mr) > 0.01:
            c, s = math.cos(mr), math.sin(mr)
            man = _matmul3([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], man)
        m = _matmul3(m, man)
    return _rot_to_quat(m)


def perception_args(a):
    """Namespace the perception.py step functions expect."""
    return argparse.Namespace(
        stride=a.stride, min_dist=a.min_dist, max_dist=a.max_dist,
        iters=a.iters, dist=a.dist, planes=a.planes, tol=a.tol,
        min_cluster=a.min_cluster, timeout=a.timeout, show=a.show,
        green_lo=a.green_lo, green_hi=a.green_hi, live=False,
    )


def log(msg):
    print(msg, flush=True)


def _median(values):
    v = sorted(values)
    return v[len(v) // 2] if v else 0.0


def _with_z(pos, z):
    return [float(pos[0]), float(pos[1]), float(z)]


def _box_drop_ik(idx):
    off = BOX_SLOT_OFFSETS[idx % len(BOX_SLOT_OFFSETS)]
    return [BOX_DROP[0] + off[0], BOX_DROP[1] + off[1], BOX_DROP[2] + off[2]]


class TaskSequence(object):
    def __init__(self, args):
        self.args = args
        self.pa = perception_args(args)
        self.head_pitch = args.head_pitch
        self.head_yaw = args.head_yaw
        self.state = "TILT_HEAD"
        self.arm = None
        self.box = None
        self.parcels = []
        self.frame_offset = list(args.frame)
        self.home_retract = list(HOME_RETRACT) if HOME_RETRACT else None
        self.six_seven_pose = None          # set by `set67`: base pose for the 6-7 move
        self._planners = {}                 # group name -> moveit_planner.ArmPlanner
        self._pd_profile = "free"
        self._pd_mode = "combined"
        self._ee_pub = None
        self._start_ee_stream()
        try:
            self.apply_pd_defaults()
        except Exception as _e:
            log("  arm PD defaults skipped (%s)" % _e)
        self.use_planning = USE_MOTION_PLANNING

    # ── arm (lazy) ──────────────────────────────────────────────────
    def ensure_arm(self):
        if self.arm is None:
            import arm_test_control as AC
            log("initialising arm control (mode 2, 100 Hz stream) ...")
            a = AC.ArmTestControl()
            a.setup()
            self.arm = a
        return self.arm

    def do_set67(self):
        """Capture the CURRENT 14 arm joints as the base pose for the `67` move.

        Put the arms where you want them first (full length to the FRONT), then `set67`,
        then run `67` — the swing happens around THIS pose."""
        ar = self.ensure_arm()
        with ar.lock:
            q = list(ar.joint_q) if ar.joint_q is not None else []
        if not q:
            log("  no joint data yet")
            return
        self.six_seven_pose = [math.degrees(v) for v in q]
        log("  6-7 base pose captured = [%s]"
            % ", ".join("%.1f" % v for v in self.six_seven_pose))

    def do_sethome(self):
        """Capture the CURRENT 14 arm joints as the 'retract' waypoint used by `home`.

        So you can build your own trajectory:  move the arms where you want the
        retract (with `keys` / `pos` / `wrist`), then `sethome`, then `home`."""
        ar = self.ensure_arm()
        with ar.lock:
            q = list(ar.joint_q) if ar.joint_q is not None else []
        if not q:
            log("  no joint data yet")
            return
        self.home_retract = [math.degrees(v) for v in q]
        log("  home retract captured = [%s]"
            % ", ".join("%.1f" % v for v in self.home_retract))
        log("  (now `home` will pass through this pose before the ready pose)")

    def do_jbias(self, arg=None, value=None):
        """Print/set the per-joint bias (deg), or force it on/off.
        `jbias` print · `jbias on|off` force · `jbias 12 10` set joint 12 to +10 deg."""
        ar = self.ensure_arm()
        if arg is None:
            log("  bias_active=%s   joint_bias=[%s]"
                % (ar.bias_active, ", ".join("%.1f" % v for v in ar.joint_bias)))
            return
        if str(arg).lower() in ("on", "off"):
            ar.bias_active = (str(arg).lower() == "on")
            log("  bias_active = %s" % ar.bias_active)
            return
        i = int(arg)
        if not (0 <= i < 14):
            log("  index 0..13  (0-6 left arm, 7-13 right arm)")
            return
        ar.joint_bias[i] = float(value)
        log("  joint_bias[%d] = %.1f   (bias_active=%s)"
            % (i, ar.joint_bias[i], ar.bias_active))

    def dump_pid_debug(self, tag="home", samples=10):
        """Print L3's PID calculation for `samples` controller ticks (then stop)."""
        try:
            from std_msgs.msg import Float64MultiArray
            got = []

            def _cb(msg):
                d = msg.data
                if len(d) >= 2 + 14 * 13 and len(got) < samples:
                    got.append((d[1], d[0], list(d[2 + 2 * 13: 2 + 3 * 13])))

            sub = rospy.Subscriber("/arm_pd/debug", Float64MultiArray, _cb, queue_size=5)
            t0 = time.time()
            while len(got) < samples and time.time() - t0 < 6.0 and not rospy.is_shutdown():
                rospy.sleep(0.05)
            sub.unregister()

            if not got:
                log("  pid debug: no data on /arm_pd/debug")
                return
            log("  --- L3 pid calculation after %s   replace=%d active=%d ---"
                % (tag, got[-1][0], got[-1][1]))
            log("  %3s %7s %7s %7s %5s %5s %5s %7s %8s %8s %8s"
                % ("#", "des", "meas", "err", "kp", "kd", "I", "kp*e", "raw", "clamped", "wbc"))
            for i, (_r, _a, f) in enumerate(got):
                pterm = f[4] * (f[3] / 57.2958)
                log("  %3d %7.1f %7.1f %7.1f %5.1f %5.1f %5.2f %7.3f %8.3f %8.3f %8.3f"
                    % (i, f[0], f[1], f[3], f[4], f[5], f[8], pterm, f[9], f[10], f[11]))
        except Exception as e:
            log("  pid debug unavailable (%s)" % e)

    def do_home(self, duration=None):
        """Move both arms to the ready pose, passing the captured retract waypoint
        first so the arms clear the table."""
        ar = self.ensure_arm()
        dur = HOME_TIME if duration is None else float(duration)
        # always go home with both claws OPEN (a closed empty claw can catch on things)
        self.grip("left", 0.0)
        self.grip("right", 0.0)
        log("  home: both claws opened")
        if self.home_retract is not None:
            log("  home: retract waypoint -> ready pose")
            ar.move_joints(self.home_retract, duration=dur)
        return ar.move_joints(READY_JOINTS, duration=dur)

    def _start_ee_stream(self):
        """Publish the last end-effector error at 10 Hz on /ee_err.

        rqt_plot needs a continuous stream; the error itself only changes once per
        move, so we re-publish the last value."""
        try:
            from std_msgs.msg import Float64MultiArray
            import threading
            self._ee_pub = rospy.Publisher("/ee_err", Float64MultiArray, queue_size=1)

            def _run():
                rate = rospy.Rate(10.0)
                while not rospy.is_shutdown():
                    try:
                        ar = self.arm
                        v = ar.last_ee_err if ar is not None else [0.0, 0.0, 0.0, 0.0]
                        m = Float64MultiArray()
                        m.data = [float(x) for x in v]
                        self._ee_pub.publish(m)
                    except Exception:
                        pass
                    rate.sleep()

            threading.Thread(target=_run, name="ee_err_stream", daemon=True).start()
            log("  /ee_err stream started at 10 Hz (rqt_plot /ee_err/data[0] ...)")
        except Exception as e:
            log("  /ee_err stream unavailable (%s)" % e)

    def ensure_planner(self, group):
        """Lazily create the MoveIt planner for one arm. Requires move_group running:
        roslaunch challenge_cup_task_template moveit_planning.launch"""
        if group in self._planners:
            return self._planners[group]
        import moveit_planner as MP
        p = MP.ArmPlanner(group)
        p.add_scene()
        self._planners[group] = p
        log("  planner(%s): MoveIt ready, scene loaded" % group)
        return p

    # group values (fallback for any joint without a per-joint value)
    ARM_PD_DEFAULTS = {
        "kp": 0.0, "kd": 0.0, "ki": 0.0, "clamp": 4.0,
        "kp_wrist": 20.0, "kd_wrist": 1.0, "ki_wrist": 1.0, "clamp_wrist": 8.0,
    }
    # PER-JOINT defaults: 14 values, order L1..L7, R1..R7
    #   shoulder 10/2/0/3   elbow 14/2/0/4   wrist roll 20/1/1/5
    #   wrist pitch 30/1.5/1/8   wrist yaw 20/1/0/5
    # FREE profile: no payload (default state)
    # ── gain sets ─────────────────────────────────────────────────────────────
    # "combined"  = WBC + PID   (small assist; the WBC does the holding)
    # "pidonly"   = PID only    (per-joint replace; needs integral to carry gravity)
    # order: L1..L7, R1..R7
    #            L1  L2  L3  L4   L5   L6   L7    R1  R2  R3  R4   R5   R6   R7
    ARM_PD_COMBINED_FREE = {
        "kp_joints":    [8, 8, 8, 12, 20, 30, 20,  8, 8, 8, 12, 20, 30, 20],
        "kd_joints":    [2, 2, 2, 3, 1.5, 2, 1.5,  2, 2, 2, 3, 1.5, 2, 1.5],
        "ki_joints":    [0, 0, 0, 0, 1, 1, 0,      0, 0, 0, 0, 1, 1, 0],
        "clamp_joints": [6, 6, 6, 8, 10, 12, 10,   6, 6, 6, 8, 10, 12, 10],
    }
    ARM_PD_COMBINED_LOADED = {
        "kp_joints":    [12, 12, 12, 16, 28, 40, 28,  12, 12, 12, 16, 28, 40, 28],
        "kd_joints":    [3, 3, 3, 4, 2, 2.5, 2,       3, 3, 3, 4, 2, 2.5, 2],
        "ki_joints":    [0, 0, 0, 0, 1.3, 1.3, 0,     0, 0, 0, 0, 1.3, 1.3, 0],
        "clamp_joints": [8, 8, 8, 10, 12, 15, 12,     8, 8, 8, 10, 12, 15, 12],
    }
    # PID only: no WBC torque, so the integral must supply gravity on pitch/roll/elbow.
    # The yaw (L3/R3) carries no gravity -> no integral, or it winds up into the stop.
    ARM_PD_PIDONLY_FREE = {
        "kp_joints":    [40, 40, 40, 40, 30, 50, 30,  40, 40, 40, 40, 30, 50, 30],
        "kd_joints":    [8, 8, 8, 8, 3, 4, 3,         8, 8, 8, 8, 3, 4, 3],
        "ki_joints":    [25, 25, 0, 25, 3, 3, 0,      25, 25, 0, 25, 3, 3, 0],
        "clamp_joints": [30, 30, 15, 30, 12, 15, 12,  30, 30, 15, 30, 12, 15, 12],
    }
    ARM_PD_PIDONLY_LOADED = {
        "kp_joints":    [50, 50, 50, 50, 40, 65, 40,  50, 50, 50, 50, 40, 65, 40],
        "kd_joints":    [10, 10, 10, 10, 4, 5, 4,     10, 10, 10, 10, 4, 5, 4],
        "ki_joints":    [30, 30, 0, 30, 4, 4, 0,      30, 30, 0, 30, 4, 4, 0],
        "clamp_joints": [35, 35, 18, 35, 14, 18, 14,  35, 35, 18, 35, 14, 18, 14],
    }

    def set_pd_profile(self, name):
        """Switch the arm-PD between the free and loaded gain sets."""
        prof = self.ARM_PD_LOADED if name == "loaded" else self.ARM_PD_FREE
        self._pd_profile = name
        for k, v in prof.items():
            rospy.set_param("/arm_pd/" + k, [float(x) for x in v])
        log("  arm PD profile -> %s" % name)

    def _active_set(self):
        mode = getattr(self, "_pd_mode", "combined")
        prof = getattr(self, "_pd_profile", "free")
        table = {
            ("combined", "free"): self.ARM_PD_COMBINED_FREE,
            ("combined", "loaded"): self.ARM_PD_COMBINED_LOADED,
            ("pidonly", "free"): self.ARM_PD_PIDONLY_FREE,
            ("pidonly", "loaded"): self.ARM_PD_PIDONLY_LOADED,
        }
        return table[(mode, prof)], mode

    def set_pd_profile(self, name):
        """Switch between the free and loaded gain sets (same mode)."""
        self._pd_profile = "loaded" if name == "loaded" else "free"
        self.apply_pd_defaults()

    def set_pd_mode(self, mode):
        """'combined' = WBC + PID, 'pidonly' = per-joint replace."""
        self._pd_mode = "pidonly" if mode.startswith("pid") else "combined"
        self.apply_pd_defaults()

    def apply_pd_defaults(self, force=True):
        """Write the active gain set (and the replace flags) to the param server."""
        gains, mode = self._active_set()
        applied = {}
        for k, v in gains.items():
            if force or rospy.get_param("/arm_pd/" + k, None) is None:
                rospy.set_param("/arm_pd/" + k, [float(x) for x in v])
                applied[k] = "14 values"
        if mode == "pidonly":
            rospy.set_param("/arm_pd/replace", False)
            rospy.set_param("/arm_pd/replace_joints", [1.0] * 14)
        else:
            rospy.set_param("/arm_pd/replace", False)
            rospy.set_param("/arm_pd/replace_joints", [0.0] * 14)
        log("  arm PD: %s / %s" % (mode, self._pd_profile))

    @staticmethod
    def _clear_joint_gains():
        """Drop the per-joint override arrays (they would mask pd/pdw with zeros)."""
        for k in ("kp_joints", "kd_joints", "ki_joints", "clamp_joints"):
            try:
                rospy.delete_param("/arm_pd/" + k)
            except Exception:
                pass

    def do_plan_home(self):
        """Move both arms to READY through COLLISION-FREE planned paths (MoveIt/OMPL).

        Falls back to the old joint-space home if move_group is not running."""
        if not self.use_planning:
            log("  planning is OFF (use `mp on`) -> fixed joint-space home")
            return self.do_home()
        ar = self.ensure_arm()
        try:
            self.ensure_planner("left_arm")
        except Exception as e:
            log("  plan home: MoveIt unavailable (%s) -> joint-space home" % e)
            return self.do_home()
        for side, group in (("left", "left_arm"), ("right", "right_arm")):
            p = self.ensure_planner(group)
            ar.active = side
            base = 0 if side == "left" else 7
            with ar.lock:
                cur = [math.degrees(v) for v in ar.joint_q[base:base + 7]]
            target = list(READY_JOINTS[base:base + 7])
            plan = p.plan_joints(target, start_deg=cur)
            if not plan:
                log("  plan home(%s): no plan -> joint-space home" % group)
                return self.do_home()
            log("  plan home(%s): %d waypoints, %.2f s"
                % (group, len(plan), plan[-1][1]))
            ar.follow_joint_path([w[0] for w in plan], [w[1] for w in plan])
        return True

    def do_sixtyseven(self, cycles=None):
        """BONUS: the "6-7" swing — the ELBOWS swing up/down while the shoulders stay put.

        Base pose = whatever `set67` captured, otherwise READY with both SHOULDERS moved
        FORWARD (SIX_SEVEN_SHOULDER_DEG). Then only SIX_SEVEN_JOINT (3 = elbow) swings by
        +-SIX_SEVEN_UP_DEG. Pure JOINT-SPACE control (move_joints plays the 14 angles
        directly — no IK), so it returns exactly to that base pose at the end.
        """
        ar = self.ensure_arm()
        n = int(cycles if cycles is not None else SIX_SEVEN_CYCLES)
        # base pose: whatever `set67` captured, else READY with the arms moved forward
        if self.six_seven_pose is not None:
            base = list(self.six_seven_pose)
            src = "captured pose"
        else:
            # READY, but both SHOULDERS move FORWARD, and only the ELBOW
            # (SIX_SEVEN_JOINT) swings — the shoulder itself then stays put.
            base = list(READY_JOINTS)
            base[0] -= SIX_SEVEN_SHOULDER_DEG                       # left shoulder FORWARD
            base[7] -= SIX_SEVEN_SHOULDER_DEG                       # right shoulder FORWARD
            src = "ready - %.0f deg shoulder (forward)" % SIX_SEVEN_SHOULDER_DEG
        log("  -- 6-7: %d cycles from %s; swinging ONLY joint %d by +-%.0f deg, %.2f s --"
            % (n, src, SIX_SEVEN_JOINT, SIX_SEVEN_UP_DEG, SIX_SEVEN_TIME))
        ar.move_joints(base, duration=HOME_TIME)
        for i in range(n * 2):
            up = (i % 2 == 0)
            d = SIX_SEVEN_UP_DEG if up else -SIX_SEVEN_UP_DEG
            pose = list(base)
            pose[SIX_SEVEN_JOINT] += d                  # left arm
            pose[7 + SIX_SEVEN_JOINT] += (-d if SIX_SEVEN_OPPOSITE else d)   # right arm
            ar.move_joints(pose, duration=SIX_SEVEN_TIME)
        ar.move_joints(base, duration=HOME_TIME)
        log("  -- 6-7 done (back at the base pose; type `home` for the ready pose) --")
        return True

    def do_gobag(self, path=None):
        """Replay the go-bag arm trajectory to reach the ready pose."""
        path = path or self.args.ready_bag or DEFAULT_READY_BAG
        if not path or not os.path.exists(path):
            log("  no go bag at %s" % path)
            return False
        from bag_to_ready import load_arm_traj
        frames = load_arm_traj(path)
        if not frames:
            log("  no arm trajectory found in %s" % path)
            return False
        a = self.ensure_arm()
        t0 = frames[0][0]
        duration = frames[-1][0] - t0
        log("  go bag: %d frames, %.1f s" % (len(frames), duration))
        a.move_joints(frames[0][1])
        hz = getattr(getattr(a, "stream", None), "_hz", 100.0)
        steps = max(1, int(round(duration * hz)))
        j = 0
        rate = rospy.Rate(hz)
        for i in range(steps + 1):
            if rospy.is_shutdown():
                break
            t = t0 + duration * (float(i) / steps)
            while j + 1 < len(frames) and frames[j + 1][0] < t:
                j += 1
            if j + 1 < len(frames):
                ta, da = frames[j]
                tb, db = frames[j + 1]
                span = (tb - ta) or 1e-6
                alpha = min(1.0, max(0.0, (t - ta) / span))
                deg = [da[k] + (db[k] - da[k]) * alpha for k in range(14)]
            else:
                deg = frames[-1][1]
            a.stream.set(deg)
            rate.sleep()
        a.stream.set(frames[-1][1])
        rospy.sleep(a.settle or 0.4)
        a.print_pose()
        return True

    # ── detections ──────────────────────────────────────────────────
    def load_dets(self):
        if not os.path.exists(DET_FILE):
            log("  no detections.json — run `cluster` first")
            return []
        try:
            with open(DET_FILE) as f:
                return json.load(f)
        except Exception as e:
            log("  could not read %s: %s" % (DET_FILE, e))
            return []

    def do_dets(self):
        dets = self.load_dets()
        for i, d in enumerate(dets):
            log("  [%d] x=%.3f y=%.3f z_top=%.3f yaw=%.1f  size=(%.3f x %.3f x %.3f)"
                % (i, d["x"], d["y"], d["z_top"], d["yaw_deg"],
                   d["length"], d["width"], d["height"]))

    def do_classify(self):
        """Split detections into self.box (one) and self.parcels (list), by size."""
        dets = self.load_dets()
        self.box = None
        self.parcels = []
        box_cands = []
        others = 0
        for d in dets:
            L, W, H = d["length"], d["width"], d["height"]
            if L >= BOX_MIN_LENGTH:
                # a BOX is box-sized AND within reach — that rejects the walls/floor,
                # which are metres long and/or metres away, but otherwise look "big"
                if (L <= BOX_MAX_LENGTH
                        and W >= BOX_MIN_SIDE and H >= BOX_MIN_SIDE
                        and math.hypot(d["x"], d["y"]) <= BOX_MAX_XY):
                    box_cands.append(d)
                else:
                    others += 1
            elif PARCEL_L_MIN <= L <= PARCEL_L_MAX and W <= PARCEL_W_MAX and H <= PARCEL_H_MAX:
                self.parcels.append(d)
            else:
                others += 1
        if box_cands:                                 # nearest plausible box wins
            self.box = min(box_cands, key=lambda d: math.hypot(d["x"], d["y"]))
        log("  classified: %d parcel(s), box=%s, %d ignored"
            % (len(self.parcels), "yes" if self.box else "no", others))
        if self.box:
            log("  box: x=%.3f y=%.3f z_top=%.3f  size=(%.2f x %.2f x %.2f)"
                % (self.box["x"], self.box["y"], self.box["z_top"],
                   self.box["length"], self.box["width"], self.box["height"]))

        if len(self.parcels) > 1:                     # merge a split parcel
            ranked = sorted(self.parcels, key=lambda d: -d.get("n", 0))
            merged = []
            for p in ranked:
                if all(math.hypot(p["x"] - m["x"], p["y"] - m["y"]) > PARCEL_MERGE_DIST
                       for m in merged):
                    merged.append(p)
            if len(merged) != len(self.parcels):
                log("  merged %d -> %d (close clusters)"
                    % (len(self.parcels), len(merged)))
            self.parcels = merged

        if len(self.parcels) > MAX_PARCELS:           # scene 1 has exactly 4
            self.parcels = sorted(self.parcels, key=lambda d: -d.get("n", 0))[:MAX_PARCELS]
            log("  kept the %d largest parcels" % MAX_PARCELS)

        # canonical layout order, so the index ALWAYS maps to the same table spot:
        #   by column (x, front -> back) then by side (y, right -> left)
        #   -> 0 = front-right, 1 = front-left, 2 = back-right, 3 = back-left
        # rounded to 0.1 m so small detection jitter can't swap two indices
        self.parcels.sort(key=lambda d: (round(d["x"], 1), round(d["y"], 1)))

        cx = sorted(p["x"] for p in self.parcels)
        cy = sorted(p["y"] for p in self.parcels)
        mx = cx[len(cx) // 2] if cx else 0.0
        my = cy[len(cy) // 2] if cy else 0.0
        for i, p in enumerate(self.parcels):
            col = "front" if p["x"] < mx else "back"
            side = "right" if p["y"] < my else "left"
            log("    parcel[%d] %-9s x=%.3f y=%.3f z_top=%.3f yaw=%.1f  offset=%-5s size=(%.3f x %.3f x %.3f)"
                % (i, "%s-%s" % (col, side), p["x"], p["y"], p["z_top"], p["yaw_deg"],
                   side, p["length"], p["width"], p["height"]))
        return self.parcels

    def ensure_classified(self):
        if not self.parcels and self.box is None:
            self.do_classify()

    def clearance_z(self, margin=0.08):
        """A height safely ABOVE everything we detected (box + parcels), so the hand can
        travel horizontally back to the ready pose without catching on anything.
        Falls back to BOX_LIFT_Z when there is nothing detected."""
        self.ensure_classified()
        tops = []
        if self.box:
            tops.append(float(self.box.get("z_top", 0.0)))
        tops += [float(p.get("z_top", 0.0)) for p in self.parcels]
        return max(tops) + margin if tops else BOX_LIFT_Z

    def do_parcels(self):
        self.ensure_classified()
        for i, p in enumerate(self.parcels):
            log("  parcel[%d] x=%.3f y=%.3f z_top=%.3f yaw=%.1f"
                % (i, p["x"], p["y"], p["z_top"], p["yaw_deg"]))
        if not self.parcels:
            log("  (no parcels — run `cluster` then `classify`)")

    # ── arm actions ─────────────────────────────────────────────────
    def do_pose(self):
        self.ensure_arm().print_pose()

    def do_jset(self, index, value):
        """Set ONE arm joint (0..13) to `value` deg, keeping the others — to explore
        which joint makes a movement. e.g. `j 1 -20` (left shoulder roll)."""
        ar = self.ensure_arm()
        with ar.lock:
            q = list(ar.joint_q) if ar.joint_q is not None else []
        if len(q) != 14:
            log("  no joint data yet")
            return
        i = int(index)
        if not (0 <= i < 14):
            log("  index must be 0..13 (0-6 left arm, 7-13 right arm)")
            return
        deg = [math.degrees(v) for v in q]
        deg[i] = float(value)
        log("  joint %d -> %.1f  (L:1-7, R:8-14)" % (i + 1, float(value)))
        ar.move_joints(deg)

    def do_joints(self):
        """Print the current 14 arm joint angles (degrees), so a pose can be captured."""
        ar = self.ensure_arm()
        with ar.lock:
            q = list(ar.joint_q) if ar.joint_q is not None else []
        if not q:
            log("  no joint data yet")
            return
        deg = [math.degrees(v) for v in q]
        log("  arm joints (deg) = [%s]" % ", ".join("%.1f" % v for v in deg))
        log("  arm joints (rad) = [%s]" % ", ".join("%.4f" % v for v in q))

    def do_posecheck(self):
        """Solve IK for the fixed task poses WITHOUT moving, and report the wrist joint
        angles + how close each is to its mechanical limit. A wrist joint within ~10 deg
        of a limit is where the "weird rotation"/misses come from."""
        ar = self.ensure_arm()
        with ar.lock:
            q_cur = list(ar.joint_q) if ar.joint_q is not None else []
        if len(q_cur) != 14:
            log("  no joint data yet")
            return
        fk = ar.fk(q_cur)
        rp, rq = list(fk.right_pose.pos_xyz), list(fk.right_pose.quat_xyzw)
        lp, lq = list(fk.left_pose.pos_xyz), list(fk.left_pose.quat_xyzw)
        # (min, max) in degrees, from the URDF
        LIM_R = [(-90.0, 90.0), (-40.0, 75.0), (-40.0, 40.0)]
        LIM_L = [(-90.0, 90.0), (-75.0, 40.0), (-40.0, 40.0)]
        lrecv = [RIGHT_HANDOFF[i] + LEFT_RECEIVE_DELTA[i] for i in range(3)]
        poses = [
            ("RIGHT_HANDOFF", "right", list(RIGHT_HANDOFF),
             _quat_from_ypr(RIGHT_HANDOFF_FIRST_YPR, RIGHT_HANDOFF_SECOND_YPR), LIM_R),
            ("LEFT_RECEIVE", "left", lrecv,
             _quat_from_ypr(LEFT_RECEIVE_FIRST_YPR, LEFT_RECEIVE_SECOND_YPR), LIM_L),
            ("BOX_DROP", "left", list(BOX_DROP),
             _quat_from_ypr(LEFT_BOX_DROP_FIRST_YPR, LEFT_BOX_DROP_SECOND_YPR), LIM_L),
            ("LEFT_PRESET_2", "left", list(LEFT_PRESET_2),
             _quat_from_ypr(LEFT_PRESET_2_FIRST_YPR, LEFT_PRESET_2_SECOND_YPR), LIM_L),
        ]
        log("  == posecheck: wrist angles (deg) and margin to limit ==")
        for name, side, pos, quat, lim in poses:
            L_p, L_q = (pos, quat) if side == "left" else (lp, lq)
            R_p, R_q = (pos, quat) if side == "right" else (rp, rq)
            q_cmd, res = ar._solve_ik(q_cur, L_p, L_q, R_p, R_q, IK_MODE_THREE_POINT)
            if q_cmd is None:
                log("  %-14s IK FAILED (time_cost %.1f ms)" % (name, res.time_cost))
                continue
            base = 0 if side == "left" else 7
            w = [math.degrees(q_cmd[base + k]) for k in (4, 5, 6)]
            marg = [min(w[k] - lim[k][0], lim[k][1] - w[k]) for k in range(3)]
            flag = "   <-- NEAR LIMIT" if min(marg) < 10.0 else ""
            log("  %-14s wrist = (%6.1f, %6.1f, %6.1f)  margin = (%5.1f, %5.1f, %5.1f)%s"
                % (name, w[0], w[1], w[2], marg[0], marg[1], marg[2], flag))

    def do_wrist_step(self, delta=5.0, side="right"):
        """DIAGNOSTIC: step each wrist joint by `delta` deg and report commanded vs
        achieved, one joint at a time. Tells us whether the wrist TRACKS or saturates."""
        ar = self.ensure_arm()
        ar.active = side
        with ar.lock:
            q = list(ar.joint_q) if ar.joint_q is not None else []
        if len(q) != 14:
            log("  no joint data yet")
            return
        base = [math.degrees(v) for v in q]
        first = 11 if side == "right" else 4          # 0-based index of wrist joint #1
        log("  == wrist_step(%s) delta=%+.1f deg, base wrist = (%.1f, %.1f, %.1f)"
            % (side, delta, base[first], base[first + 1], base[first + 2]))
        for k in range(3):
            idx = first + k
            tgt = list(base)
            tgt[idx] += delta
            ar.move_joints(tgt, duration=1.5)
            self.wait_still()
            with ar.lock:
                now = [math.degrees(v) for v in ar.joint_q]
            log("    j%-2d cmd %7.1f -> actual %7.1f   err %+6.1f deg"
                % (idx + 1, tgt[idx], now[idx], now[idx] - tgt[idx]))
            ar.move_joints(base, duration=1.5)
            self.wait_still()
            with ar.lock:
                back = [math.degrees(v) for v in ar.joint_q]
            log("         returned to %.1f" % back[idx])
        log("  == wrist_step done ==")

    def do_wrist(self, vals=None):
        """Print the 3 wrist joint angles (yaw, pitch, roll), or set them."""
        ar = self.ensure_arm()
        if vals is None:
            w = ar.current_wrist_deg()
            log("  wrist deg (yaw, pitch, roll) = (%.1f, %.1f, %.1f)" % (w[0], w[1], w[2]))
        else:
            ar.set_wrist(list(vals))


    def do_freeze(self, on=None):
        """Print, or set, the wrist freeze. `freeze` prints; `freeze on|off` sets.
        Frozen = the 3 wrist joints are held at their current angles (moves only turn
        the shoulder/elbow). It does NOT remove the wrist's torque/force."""
        ar = self.ensure_arm()
        if on is None:
            log("  freeze wrist joints = %s" % ar.freeze_wrist)
            return
        ar.freeze_wrist = bool(on)
        log("  freeze wrist joints = %s" % ar.freeze_wrist)

    def do_unlock(self):
        """Release the orientation lock set by `topdown` (back to the IK's orientation)."""
        ar = self.ensure_arm()
        ar.fixed_rpy = None
        log("  orientation lock OFF")

    def do_topdown(self):
        """Rotate the ACTIVE claw to point straight down (top-down), keeping the hand
        where it is. Uses the measured claw axis, so it's exact."""
        ar = self.ensure_arm()
        if not ar.align_claw((0.0, 0.0, -1.0)):
            log("  topdown: cannot align (no TF/claw axis)")
            return
        cur = ar.current_pose()[0]
        ar.ik_mode = IK_MODE_POS_HARD_ORI_HARD
        log("  topdown: claw down at (%.3f, %.3f, %.3f)" % (cur[0], cur[1], cur[2]))
        ok = ar.move_to(pos=[float(cur[0]), float(cur[1]), float(cur[2])])
        if not ok:
            ar.ik_mode = IK_MODE_THREE_POINT
            ar.move_to(pos=[float(cur[0]), float(cur[1]), float(cur[2])])

    def do_lift(self, z=None):
        """Move the RIGHT hand straight up/down to height z (x,y + orientation kept).
        Default: current z + 0.05."""
        ar = self.ensure_arm()
        ar.active = "right"
        ar.wrist_target = False
        ar.ik_mode = IK_MODE_THREE_POINT
        cur = ar.current_pose()[0]
        z = (float(cur[2]) + 0.05) if z is None else float(z)
        log("  lift -> z=%.3f at (%.3f, %.3f)" % (z, cur[0], cur[1]))
        ar.move_to(pos=[float(cur[0]), float(cur[1]), z])

    def do_pick(self, idx=0):
        """Pick parcel idx: align above it at the transit height, descend to the
        grasp height with the hard pose (0x03, fallback 0x06), then clamp."""
        self.ensure_classified()
        if not (0 <= idx < len(self.parcels)):
            log("  parcel index %d out of range" % idx)
            return False
        p = self.parcels[idx]
        px, py = float(p["x"]), float(p["y"])
        pside = "right" if py < _median([q["y"] for q in self.parcels]) else "left"
        off = PICK_OFFSET_RIGHT if pside == "right" else PICK_OFFSET_LEFT
        gx = px + self.frame_offset[0] + off[0]
        gy = py + self.frame_offset[1] + off[1]
        gz = PICK_IK_Z + off[2]
        q = _quat_from_ypr(PICK_FIRST_YPR, PICK_SECOND_YPR)
        log("  pick parcel[%d] -> (%.3f, %.3f, %.3f)  side=%s"
            % (idx, gx, gy, gz, pside))
        self.move_hand("right", [gx, gy, PICK_TRANSIT_Z], q, IK_MODE_THREE_POINT, MOVE_TIME,
                       settle=PICK_SETTLE_TIME)
        # let the claw STOP SWINGING before we descend onto the parcel
        self.wait_still(1.2, 0.3, 1.0)   # fast
        # ONE descent only. The strict 0x03 mode used to fail and trigger a second
        # descent (looks like "it drops twice"), so go straight to the permissive mode.
        self.move_hand("right", [gx, gy, gz], q, IK_MODE_THREE_POINT, PICK_GRASP_MOVE_TIME)
        self.wait_still()          # settle at the grasp pose before closing
        self.grip("right", 100.0, effort=RIGHT_GRIP_EFFORT)
        ar = self.ensure_arm()
        pct = ar.claw_pct[1 if ar.active == "right" else 0]
        log("  pick: gripper = %s" % ("?" if pct is None else "%.1f%%" % pct))
        return True

    # ── primitives ──────────────────────────────────────────────────
    def move_hand(self, side, pos, quat, mode=IK_MODE_THREE_POINT,
                  move_time=MOVE_TIME, settle=SETTLE_TIME):
        """Move ONE hand to an absolute IK-frame pose; the other arm holds."""
        ar = self.ensure_arm()
        ar.active = side
        ar.wrist_target = False
        ar.elbow_pos = None
        ar.ik_mode = mode
        ar.move_time = move_time
        ar.settle = settle
        return ar.move_to(pos=list(pos), quat=list(quat) if quat else None)

    def wait_still(self, tol_deg=1.2, window=0.3, timeout=0.6):
        """Block until the arms STOP MOVING: the biggest joint change over `window`
        seconds is below `tol_deg`. Use before descending/closing so we never act
        while the claw is still swinging after a move."""
        if not WAIT_STILL_ENABLED:
            return True                             # disabled: never block
        ar = self.ensure_arm()
        t0 = time.time()
        hist = []                                   # (t, q) samples in the window
        while not rospy.is_shutdown() and time.time() - t0 < timeout:
            with ar.lock:
                q = list(ar.joint_q) if ar.joint_q is not None else None
            t = time.time()
            if q is not None:
                hist.append((t, q))
                while hist and t - hist[0][0] > window:
                    hist.pop(0)
                if len(hist) >= 2 and t - hist[0][0] >= window * 0.9:
                    span = max(abs(math.degrees(a - b))
                               for a, b in zip(hist[-1][1], hist[0][1]))
                    if span < tol_deg:
                        return True
            rospy.sleep(0.05)
        log("  wait_still: timed out (arm still moving)")
        return False

    def grip(self, side, pct, effort=None):
        """Open/close ONE claw. On close, use the closed-loop grip (hold + grab check);
        the other claw keeps its state. Returns True when the close actually grabbed
        (None when opening)."""
        ar = self.ensure_arm()
        prev = ar.active
        ar.active = side
        got = None
        if pct > 0:
            got = ar.grip_close_loop(effort=effort)
            if got:
                self.set_pd_profile("loaded")      # payload on board
        else:
            ar.set_gripper(pct)
            self.set_pd_profile("free")            # nothing in the claw
        ar.active = prev
        return got

    # ── phase 2: place on the scale, then regrasp ───────────────────
    def weigh_xy(self):
        """Scale placement (x, y): detected marker + WEIGH_OFFSET (toward the body)."""
        try:
            with open(WEIGH_FILE) as f:
                d = json.load(f)
            return ([float(d["x"]) + WEIGH_OFFSET[0],
                     float(d["y"]) + WEIGH_OFFSET[1]], "detected")
        except Exception:
            return ([WEIGH_RELEASE[0] + WEIGH_OFFSET[0],
                     WEIGH_RELEASE[1] + WEIGH_OFFSET[1]], "default")

    def do_slift(self):
        """scale step 1: lift the parcel straight up to the carry height."""
        ar = self.ensure_arm()
        ar.active = "right"
        ar.ik_mode = IK_MODE_THREE_POINT
        q = _quat_from_ypr(PICK_FIRST_YPR, PICK_SECOND_YPR)
        cur = ar.current_pose()[0]
        log("  scale-lift -> z=%.3f  (holding)" % WEIGH_TRANSIT_Z)
        ar.move_linear([cur[0], cur[1], WEIGH_TRANSIT_Z], q, n_points=4)

    def do_scarry(self):
        """scale step 2: carry horizontally to above the scale at the carry height."""
        ar = self.ensure_arm()
        (wx, wy), src = self.weigh_xy()
        q = _quat_from_ypr(PICK_FIRST_YPR, PICK_SECOND_YPR)
        log("  scale-carry -> above scale (%s) (%.3f, %.3f, %.3f)"
            % (src, wx, wy, WEIGH_TRANSIT_Z))
        self.move_hand("right", [wx, wy, WEIGH_TRANSIT_Z], q,
                       IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME, settle=0.0)
        self.wait_still()          # line up over the scale BEFORE descending

    def do_sdrop(self):
        """scale step 3: descend to the drop height, then open (release)."""
        ar = self.ensure_arm()
        (wx, wy), src = self.weigh_xy()
        q_rel = _quat_from_ypr(WEIGH_RELEASE_FIRST_YPR, WEIGH_RELEASE_SECOND_YPR)
        release = [wx, wy, WEIGH_DROP_Z]
        log("  scale-drop -> (%.3f, %.3f, %.3f) then open" % (release[0], release[1], release[2]))
        self.move_hand("right", release, q_rel, IK_MODE_POS_HARD_ORI_HARD,
                       PICK_GRASP_MOVE_TIME)
        self.wait_still()          # claw must stop swinging before we release
        self.grip("right", 0.0)
        rospy.sleep(WEIGH_DWELL)

    def do_scale(self, idx=0):
        """carry to the scale and place it (no drop), then re-grasp.

        SCALE_TOPDOWN = True  -> claw is held TOP-DOWN, so the WRIST doesn't carry the
                                 parcel's weight (load runs along the arm axis) and the
                                 sag is much smaller. Also: wait_still guards, and the
                                 re-grasp uses the ACTUAL release xy.
        SCALE_TOPDOWN = False -> the ORIGINAL behaviour: no top-down, the claw keeps
                                 WHATEVER orientation and just sags; no waits; re-grasp at
                                 the commanded xy. Use this to record the sag difference.
        """
        ar = self.ensure_arm()
        ar.active = "right"
        q_carry = _quat_from_ypr(PICK_FIRST_YPR, PICK_SECOND_YPR)
        (wx, wy), src = self.weigh_xy()
        log("  scale (%s): x=%.3f y=%.3f   carry z=%.3f  place z=%.3f  topdown=%s"
            % (src, wx, wy, WEIGH_TRANSIT_Z, WEIGH_DROP_Z, SCALE_TOPDOWN))

        # 1) lift straight up to the carry height
        cur = ar.current_pose()[0]
        ar.ik_mode = IK_MODE_THREE_POINT
        ar.settle = SETTLE_TIME
        ar.move_linear([cur[0], cur[1], WEIGH_TRANSIT_Z], q_carry, n_points=4)

        # 2) orientation for the carry + drop
        if SCALE_TOPDOWN:
            log("  scale: claw aim = (%.2f, %.2f, %.2f)  (over-tilt %.0f deg)"
                % (WEIGH_TOPDOWN_DIR[0], WEIGH_TOPDOWN_DIR[1], WEIGH_TOPDOWN_DIR[2],
                   TOPDOWN_TILT_DEG))
            if ar.align_claw(WEIGH_TOPDOWN_DIR):
                cur = ar.current_pose()[0]
                ar.ik_mode = IK_MODE_POS_HARD_ORI_HARD
                ar.move_to(pos=[float(cur[0]), float(cur[1]), float(cur[2])])
            carry_q = None            # quat=None -> keep the locked claw-down orientation
            drop_q = None
        else:
            # ORIGINAL: no top-down — hold the nominal release orientation and sag
            ar.fixed_rpy = None       # make sure no old orientation lock is active
            carry_q = _quat_from_ypr(WEIGH_RELEASE_FIRST_YPR, WEIGH_RELEASE_SECOND_YPR)
            drop_q = carry_q
            log("  scale: (no top-down — claw keeps its orientation, expect sag)")

        # 3) carry over the scale
        self.move_hand("right", [wx, wy, WEIGH_TRANSIT_Z], carry_q,
                       IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME, settle=0.0)
        if SCALE_TOPDOWN:
            self.wait_still()          # line up over the scale BEFORE descending

        # 4) place it SLOWLY (lower, don't drop), then open
        self.move_hand("right", [wx, wy, WEIGH_DROP_Z], drop_q,
                       IK_MODE_POS_HARD_ORI_HARD, PLACE_MOVE_TIME)
        if SCALE_TOPDOWN:
            self.wait_still()          # claw must stop swinging before we release
            # the parcel lands where the hand ACTUALLY is -> use that xy for the re-grasp
            rp = ar.current_pose()[0]
            rx, ry = float(rp[0]), float(rp[1])
            log("  scale-regrasp: release xy = (%.3f, %.3f)   (commanded %.3f, %.3f)"
                % (rx, ry, wx, wy))
        else:
            rx, ry = wx, wy            # ORIGINAL: re-grasp at the COMMANDED xy
        if WEIGH_RELEASE_DELAY > 0.0:
            log("  scale: pause %.1f s before releasing" % WEIGH_RELEASE_DELAY)
            rospy.sleep(WEIGH_RELEASE_DELAY)
        self.grip("right", 0.0)
        rospy.sleep(WEIGH_DWELL)

        # 5) re-grasp straight down from the ACTUAL release xy (no guessing).
        #    The wrist-camera version is kept but OFF — set USE_WRIST_LOCATE = True.
        if USE_WRIST_LOCATE:
            loc = None
            try:
                loc = P.wrist_locate(self.pa, "right", hint=(rx, ry))
            except Exception as e:
                log("  scale-regrasp: wrist locate failed (%s)" % e)
            if loc is not None and math.hypot(loc[0] - rx, loc[1] - ry) <= WRIST_MAX_FIX:
                rx, ry = float(loc[0]), float(loc[1])
                log("  scale-regrasp: wrist camera -> claw over (%.3f, %.3f)" % (rx, ry))
            elif loc is not None:
                log("  scale-regrasp: wrist fix (%.3f, %.3f) is %.0f mm from the drop — ignored"
                    % (loc[0], loc[1], 1000.0 * math.hypot(loc[0] - rx, loc[1] - ry)))
        else:
            log("  scale-regrasp: release xy (%.3f, %.3f)  [wrist cam off]"
                % (rx, ry))
        rx += REGRASP_OFFSET[0]
        ry += REGRASP_OFFSET[1]
        log("  scale-regrasp: + offset (%.3f, %.3f, %.3f) -> (%.3f, %.3f)"
            % (REGRASP_OFFSET[0], REGRASP_OFFSET[1], REGRASP_OFFSET[2], rx, ry))
        self.move_hand("right", [rx, ry, WEIGH_DROP_Z], None,
                       IK_MODE_THREE_POINT, MOVE_TIME, settle=0.0)
        if SCALE_TOPDOWN:
            self.wait_still()
        self.move_hand("right", [rx, ry, WEIGH_REGRASP_Z], None,
                       IK_MODE_POS_HARD_ORI_HARD, PICK_GRASP_MOVE_TIME)
        if SCALE_TOPDOWN:
            self.wait_still()      # settle before closing on the parcel
        self.grip("right", 100.0, effort=RIGHT_GRIP_EFFORT)
        rospy.sleep(GRIPPER_HOLD)

        # 6) lift back up, claw down
        cur = ar.current_pose()[0]
        ar.move_linear([cur[0], cur[1], WEIGH_TRANSIT_Z], None, n_points=4)
        # the top-down lock is only for this scale phase — release it
        ar.fixed_rpy = None
        return True

    # ── phase 3: hand off to the left hand ──────────────────────────
    def goto_pose(self, side, pos, quat, tol=0.010, tries=3, move_time=None):
        """Re-issue the same target until the hand is within `tol` (task-level loop).

        The arm is soft: one command lands 2-5 cm short. Re-commanding the SAME absolute
        pose from the measured pose closes that gap without touching the controller."""
        ar = self.ensure_arm()
        ar.active = side
        mt = move_time if move_time is not None else HANDOFF_MOVE_TIME
        for i in range(tries):
            self.move_hand(side, list(pos), quat, IK_MODE_THREE_POINT, mt)
            cur = ar.current_pose()[0]
            d = math.sqrt(sum((cur[j] - pos[j]) ** 2 for j in range(3)))
            log("  goto_pose(%s) pass %d: %.0f mm off" % (side, i + 1, d * 1000.0))
            if d <= tol:
                return True
        return False

    def wait_pose(self, side, pos, tol=0.010, timeout=2.0):
        """Block until the hand is within `tol` of `pos` (or timeout).

        The official collector does this before every grip: commanding a pose is not
        the same as being there, especially with a soft arm."""
        ar = self.ensure_arm()
        ar.active = side
        t0 = time.time()
        while not rospy.is_shutdown() and time.time() - t0 < timeout:
            cur = ar.current_pose()[0]
            d = math.sqrt(sum((cur[i] - pos[i]) ** 2 for i in range(3)))
            if d <= tol:
                return True
            rospy.sleep(0.05)
        cur = ar.current_pose()[0]
        d = math.sqrt(sum((cur[i] - pos[i]) ** 2 for i in range(3)))
        log("  wait_pose(%s): timed out, %.0f mm off" % (side, d * 1000.0))
        return False

    def move_then_orient(self, side, pos, quat, move_time):
        """Translate to `pos` keeping the CURRENT orientation, then rotate in place.

        The IK solves position and orientation together, so a single move makes the
        wrist rotate while the arm travels (looks odd). Splitting gives
        "arm first, then wrist"."""
        ar = self.ensure_arm()
        ar.active = side
        self.move_hand(side, list(pos), None, IK_MODE_THREE_POINT, move_time)
        self.move_hand(side, list(pos), quat, IK_MODE_THREE_POINT, move_time)

    def do_handoff(self):
        """Right -> left handoff, following the official scene-1 collector.

        Fully deterministic: fixed poses, fixed orientations, no measurement and no
        projection. This is the same sequence that produced the ACT training data.
        """
        ar = self.ensure_arm()
        q_right = _quat_from_ypr(RIGHT_HANDOFF_FIRST_YPR, RIGHT_HANDOFF_SECOND_YPR)
        q_left = _quat_from_ypr(LEFT_RECEIVE_FIRST_YPR, LEFT_RECEIVE_SECOND_YPR)

        # open the left claw first so it is ready to receive
        self.grip("left", 0.0)
        ar.fixed_rpy = None
        ar.freeze_wrist = False

        # ── right arm: lift in place, cross, then lower to the handoff pose ──
        ar.active = "right"
        cur = ar.current_pose()[0]
        self.move_hand("right", [cur[0], cur[1], RIGHT_HANDOFF_TRANSIT_Z], q_right,
                       IK_MODE_THREE_POINT, MOVE_TIME)
        self.move_hand("right", [RIGHT_HANDOFF[0], RIGHT_HANDOFF[1],
                                 RIGHT_HANDOFF_TRANSIT_Z], q_right,
                       IK_MODE_THREE_POINT, RIGHT_HANDOFF_MOVE_TIME)
        self.move_then_orient("right", RIGHT_HANDOFF, q_right, HANDOFF_MOVE_TIME)

        # ── left arm: ready → approach (still short of the parcel) → in → grip ──
        ar.active = "left"
        self.move_hand("left", list(LEFT_RECEIVE_READY_POS), q_left,
                       IK_MODE_THREE_POINT, MOVE_TIME)
        if LEFT_READY_PAUSE > 0.0:
            log("  handoff: left waits %.1f s at the ready pose" % LEFT_READY_PAUSE)
            rospy.sleep(LEFT_READY_PAUSE)

        # 1) approach point: same x/z as the receive pose, but pulled BACK along +y
        appr = [LEFT_RECEIVE_POS[0],
                LEFT_RECEIVE_POS[1] + LEFT_APPROACH_Y,
                LEFT_RECEIVE_POS[2]]
        log("  handoff: left approach (%.3f, %.3f, %.3f), %.0f mm short"
            % (appr[0], appr[1], appr[2], LEFT_APPROACH_Y * 1000.0))
        # ONE combined move: position + orientation solved together, so the wrist is
        # already aligned with the parcel when it arrives (no separate re-orient pass)
        self.move_hand("left", appr, q_left, IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME)

        # 2) STOP and let it settle before going in
        if LEFT_APPROACH_PAUSE > 0.0:
            rospy.sleep(LEFT_APPROACH_PAUSE)

        # 3) reach straight in along -y to the parcel
        ar.move_linear(list(LEFT_RECEIVE_POS), q_left, n_points=4, seg_time=0.4)
        self.wait_pose("left", LEFT_RECEIVE_POS, tol=0.012, timeout=2.0)
        rospy.sleep(HANDOFF_LEFT_SETTLE)        # stop, then close
        got = self.grip("left", 100.0, effort=HANDOFF_GRIP_EFFORT)
        log("  handoff: left closure %s — holding %.1f s before the right lets go"
            % ("HOLDING" if got else "EMPTY", HANDOFF_TRANSFER_PAUSE))
        # the left must be firmly on the parcel BEFORE the right opens, otherwise the
        # parcel drops in the gap between the two claws
        rospy.sleep(HANDOFF_TRANSFER_PAUSE)
        self.grip("right", 0.0)

        # re-seat the grip now that the left carries the full weight: the parcel can
        # shift when the right lets go, so close once more at full effort
        rospy.sleep(0.2)
        self.grip("left", 100.0, effort=HANDOFF_GRIP_EFFORT)
        log("  handoff: left re-seated after release")

        # ── right arm retracts straight back ──
        ar.active = "right"
        cur = ar.current_pose()[0]
        self.move_hand("right", [cur[0], cur[1] + RIGHT_HANDOFF_RETRACT_Y, cur[2]],
                       q_right, IK_MODE_THREE_POINT, MOVE_TIME)
        return True

    def do_box(self, idx=0):
        """Drop the parcel (left hand) into the box.

        The box rim sits well above the table and the arm tracks a few cm BELOW its
        target (sag), so we must not cross to the box at working height:
          1) lift straight up (at the current xy) until clear of the rim,
          2) only then move across in xy to above the box,
          3) then lower into the box and release.
        The left arm is at the edge of its reach, so the moves run unconditionally
        (a tolerance miss must not abort); if the hand didn't actually get near the box
        we retry with x pulled toward the robot."""
        q = _quat_from_ypr(LEFT_BOX_DROP_FIRST_YPR, LEFT_BOX_DROP_SECOND_YPR)
        ar = self.ensure_arm()
        base = _box_drop_ik(idx)
        placed = False
        ar.active = "left"
        ar.ik_mode = IK_MODE_THREE_POINT
        for dx in [0.0] + BOX_DROP_X_FALLBACK:
            box = [base[0] + dx, base[1], base[2] + BOX_Z_COMP]
            # Straight Cartesian lines, NOT joint-space moves: a joint interpolation
            # can swing the hand sideways/through the box while it "rises".
            # 1) LIFT straight up at the current xy, clear of the box rim (+ sag margin)
            cur = ar.current_pose()[0]
            ar.move_linear([cur[0], cur[1], BOX_LIFT_Z], q, n_points=5, seg_time=0.5)
            # 2) THEN move across in xy, still at the clearance height
            ar.move_linear([box[0], box[1], BOX_LIFT_Z], q, n_points=5, seg_time=0.5)
            # 3) lower straight down into the box
            ar.move_linear([box[0], box[1], box[2]], q, n_points=5, seg_time=0.5)
            got = ar.current_pose()[0]
            d = math.hypot(got[0] - box[0], got[1] - box[1])
            log("  box try x=%.3f -> reached (%.3f, %.3f, %.3f)  xy off %.0f mm"
                % (box[0], got[0], got[1], got[2], d * 1000.0))
            if d < 0.08:
                placed = True
                break
        if not placed:
            log("  box: left hand never got near the box (out of reach)")
        self.grip("left", 0.0)
        rospy.sleep(PLACE_DWELL)
        # 1) lift straight UP out of the box (clear of the rim), then
        # 2) pull BACK out of the box footprint, then
        # 3) go to the ready pose.
        # The claw fingers hang BELOW the hand frame, so lifting only — or pulling only —
        # drags/knocks them on the box when the arm then swings to the ready pose.
        # After the release, pull straight BACK out of the box in ONE move — no separate
        # lift phase. BOX_RETRACT carries a small +z so the claw fingers (which hang
        # ~7 cm below the hand frame) rise over the rim instead of scraping it.
        cur = ar.current_pose()[0]
        log("  box: retracting back by (%.3f, %.3f, %+.3f)" % tuple(BOX_RETRACT))
        ok = ar.move_linear([cur[0] + BOX_RETRACT[0], cur[1] + BOX_RETRACT[1],
                             cur[2] + BOX_RETRACT[2]], q, n_points=5, seg_time=0.5)
        if not ok:
            # A JOINT-SPACE move needs no IK, so it cannot fail. Doing it here (rather
            # than letting home() do it from over the box) keeps the hand from swinging
            # across the box rim.
            if self.home_retract is not None:
                log("  box: Cartesian retreat blocked -> joint-space retreat waypoint")
                ar.move_joints(self.home_retract, duration=HOME_TIME)
            else:
                cur = ar.current_pose()[0]
                ar.move_to([cur[0] + BOX_RETRACT[0], cur[1] + BOX_RETRACT[1],
                            cur[2] + 0.10], None)
        self.do_home()
        return placed

    # ── whole chain for one parcel ──────────────────────────────────
    def do_task(self, idx=0):
        log("=== task parcel[%d] ===" % idx)
        self.ensure_classified()
        if not (0 <= idx < len(self.parcels)):
            log("  parcel index %d out of range" % idx)
            return False
        self.do_pick(idx)
        self.do_scale(idx)
        # settle after the re-grasp before starting the handoff
        log("  pause %.1f s (settle after re-grasp, before handoff)" % SCALE_TO_HANDOFF_DELAY)
        rospy.sleep(SCALE_TO_HANDOFF_DELAY)
        self.do_handoff()
        self.do_box(idx)
        log("=== task parcel[%d] done ===" % idx)
        return True

    def do_auto(self):
        """One command: detect -> check 4 parcels -> home -> runall."""
        log("=== auto: detect -> home -> runall ===")
        try:
            self.do_detect()
        except Exception as e:
            log("  auto: detect failed (%s)" % e)
            return False
        n = len(self.parcels)
        log("  auto: %d/%d parcel(s) detected" % (n, MAX_PARCELS))
        if n != MAX_PARCELS:
            log("  auto: need %d parcels, got %d -> stopping before home/runall"
                % (MAX_PARCELS, n))
            log("  auto: press Enter after fixing detection, or run `detect` again")
            return False
        self.do_home()
        done = self.do_all()
        log("=== auto done: %d/%d ===" % (done, MAX_PARCELS))
        return done == MAX_PARCELS

    def do_all(self):
        """Run the WHOLE chain for every parcel, one after another (0, 1, 2, 3…).

        The parcels never move while we work, so the first classification stays valid —
        no re-detect is needed between them (and a re-detect would pop a blocking window).
        If nothing is classified yet we run one `detect` first."""
        self.ensure_classified()
        if not self.parcels:
            log("  run all: nothing classified — running detect first")
            self.do_detect()
        n = min(MAX_PARCELS, len(self.parcels))
        log("=== run all: %d parcel(s) ===" % n)
        done = 0
        for idx in range(n):
            log("--- parcel %d/%d ---" % (idx + 1, n))
            ok = False
            try:
                ok = self.do_task(idx)
                if ok:
                    done += 1
            except Exception as e:
                log("  parcel %d failed: %s (continuing)" % (idx, e))
            # do_box already returns the arms to the ready pose, so only home when the
            # parcel bailed out early — that saves a full duplicate home move.
            if not ok:
                try:
                    self.do_home()
                except Exception as e:
                    log("  home after parcel %d failed: %s" % (idx, e))
        log("=== run all done: %d/%d parcel(s) ===" % (done, n))
        return done

    # ── detection steps ─────────────────────────────────────────────
    def do_head(self):
        P.set_head(self.head_yaw, self.head_pitch)

    def do_cloud(self):
        P.step_cloud(self.pa)

    def do_ransac(self):
        P.step_ransac(self.pa)

    def do_cluster(self):
        P.step_cluster(self.pa)

    def do_detect(self):
        """head -> cloud -> ransac -> cluster -> classify -> weigh.

        Only the CLUSTER result is shown (its window blocks until you press a key);
        cloud / ransac / weigh stay silent.
        """
        import io
        import sys
        log("  -- detect: head -> cloud -> ransac -> cluster -> classify -> weigh --")
        self.do_head()
        if HEAD_SETTLE > 0.0:
            log("  detect: head settling %.1f s before capturing" % HEAD_SETTLE)
            rospy.sleep(HEAD_SETTLE)
        self.pa.show = False       # cloud / ransac / weigh: no windows
        self.pa.live = False
        sink = io.StringIO()
        real = sys.stdout
        sys.stdout = sink
        try:
            self.do_cloud()
            self.do_ransac()
        finally:
            sys.stdout = real
        # the cluster overlay is drawn only when args.show is True -> enable it just
        # for this step, so `detect` shows exactly ONE window (the clusters)
        self.pa.show = True
        try:
            self.do_cluster()
        finally:
            self.pa.show = False
        self.do_classify()
        self.do_weigh()

    def do_weigh(self):
        P.step_weigh(self.pa)

    def do_live(self, period):
        self.pa.live = True
        log("live every %.1fs — Ctrl-C to stop" % period)
        try:
            while not rospy.is_shutdown():
                t0 = time.time()
                self.do_cloud()
                self.do_ransac()
                self.do_cluster()
                time.sleep(max(0.0, period - (time.time() - t0)))
        except KeyboardInterrupt:
            log("live stopped")
        finally:
            self.pa.live = False
            P.destroy_windows()

    # ── non-interactive run ─────────────────────────────────────────
    def run_auto(self):
        log("=== task_sequence (auto) ===")
        while not rospy.is_shutdown() and self.state != "DONE":
            handler = getattr(self, "st_" + self.state.lower(), None)
            if handler is None:
                log("no handler for state %s" % self.state)
                break
            nxt = handler()
            log("[state] %s -> %s" % (self.state, nxt))
            self.state = nxt
        log("=== done ===")

    def st_tilt_head(self):
        self.do_head()
        return "ARM_READY"

    def st_arm_ready(self):
        self.do_gobag(self.args.ready_bag)
        return "PERCEIVE_CLOUD"

    def st_perceive_cloud(self):
        P.step_cloud(self.pa)
        return "PERCEIVE_RANSAC"

    def st_perceive_ransac(self):
        P.step_ransac(self.pa)
        return "PERCEIVE_CLUSTER"

    def st_perceive_cluster(self):
        P.step_cluster(self.pa)
        return "DONE"

    # ── interactive ─────────────────────────────────────────────────
    def repl(self):
        log("interactive. type h for help, q to quit.")
        while not rospy.is_shutdown():
            try:
                line = input("task> ").strip()
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
                elif cmd == "head":
                    self.do_head()
                elif cmd == "cloud":
                    self.do_cloud()
                elif cmd == "ransac":
                    self.do_ransac()
                elif cmd == "cluster":
                    self.do_cluster()
                elif cmd == "weigh":
                    self.do_weigh()
                elif cmd == "all":
                    self.do_cloud()
                    self.do_ransac()
                    self.do_cluster()
                elif cmd in ("detect", "scan"):
                    self.do_detect()
                elif cmd == "live":
                    self.do_live(float(a[0]) if a else 1.0)
                elif cmd == "pitch":
                    self.head_pitch = float(a[0])
                    log("head pitch = %.1f" % self.head_pitch)
                elif cmd == "yaw":
                    self.head_yaw = float(a[0])
                    log("head yaw = %.1f" % self.head_yaw)
                elif cmd == "gobag":
                    self.do_gobag(a[0] if a else None)
                elif cmd == "home":
                    #   home          -> planned if `mp on`, else fixed joint-space
                    #   home raw      -> always the fixed joint-space move
                    #   home <secs>   -> fixed move with that segment duration
                    if a and a[0] in ("raw", "joint", "old"):
                        self.do_home()
                    elif a:
                        self.do_home(float(a[0]))
                    elif self.use_planning:
                        self.do_plan_home()
                    else:
                        self.do_home()
                elif cmd in ("planhome", "phome"):
                    self.do_plan_home()
                elif cmd in ("pd", "gains"):
                    # live arm-PD tuning (no rebuild):  pd kp kd [ki] [clamp]
                    #   pd               -> show current values
                    #   pd 0 0           -> disable the arm PD layer
                    if not a:
                        g = {k: rospy.get_param("/arm_pd/" + k, None)
                             for k in ("kp", "kd", "ki", "clamp")}
                        log("  arm PD: %s (set with `pd kp kd [ki] [clamp]`)" % g)
                    elif a[0].lower() in ("reset", "default", "defaults"):
                        self._clear_joint_gains()
                        for k in list(self.ARM_PD_DEFAULTS) + list(self.ARM_PD_JOINTS):
                            try:
                                rospy.delete_param("/arm_pd/" + k)
                            except Exception:
                                pass
                        self.apply_pd_defaults()
                        log("  arm PD reset to defaults")
                    elif a[0].lower() == "off":
                        self._clear_joint_gains()
                        for k in self.ARM_PD_DEFAULTS:
                            rospy.set_param("/arm_pd/" + k, 0.0)
                        log("  arm PD: OFF (all gains 0)")
                    elif len(a) >= 2:
                        self._clear_joint_gains()      # let the group gains apply
                        rospy.set_param("/arm_pd/kp", float(a[0]))
                        rospy.set_param("/arm_pd/kd", float(a[1]))
                        if len(a) > 2:
                            rospy.set_param("/arm_pd/ki", float(a[2]))
                        if len(a) > 3:
                            rospy.set_param("/arm_pd/clamp", float(a[3]))
                        log("  arm PD set: kp=%s kd=%s ki=%s clamp=%s"
                            % (a[0], a[1], a[2] if len(a) > 2 else "-",
                               a[3] if len(a) > 3 else "-"))
                    else:
                        log("  usage: pd kp kd [ki] [clamp]   |   pd")
                elif cmd in ("pdw", "wrist"):
                    # wrist-only gains:  pdw kp kd [ki] [clamp]
                    if not a:
                        g = {k: rospy.get_param("/arm_pd/" + k + "_wrist", None)
                             for k in ("kp", "kd", "ki", "clamp")}
                        log("  arm PD (wrist only): %s" % g)
                    elif len(a) >= 2:
                        self._clear_joint_gains()      # let the wrist gains apply
                        rospy.set_param("/arm_pd/kp_wrist", float(a[0]))
                        rospy.set_param("/arm_pd/kd_wrist", float(a[1]))
                        if len(a) > 2:
                            rospy.set_param("/arm_pd/ki_wrist", float(a[2]))
                        if len(a) > 3:
                            rospy.set_param("/arm_pd/clamp_wrist", float(a[3]))
                        log("  wrist PD set: kp=%s kd=%s" % (a[0], a[1]))
                    else:
                        log("  usage: pdw kp kd [ki] [clamp]   |   pdw")
                elif cmd in ("pdj", "joint"):
                    # per-joint arm gains.  joints 0..13 = left l1..l7, right r1..r7
                    #   pdj                  -> show all 14 kp values
                    #   pdj <idx> <kp> [kd] [ki] [clamp]
                    #   pdj reset            -> clear the per-joint override
                    names = ["L%d" % i for i in range(1, 8)] + ["R%d" % i for i in range(1, 8)]
                    if a and a[0] == "reset":
                        for k in ("kp_joints", "kd_joints", "ki_joints", "clamp_joints"):
                            try:
                                rospy.delete_param("/arm_pd/" + k)
                            except Exception:
                                pass
                        log("  per-joint gains cleared (back to pd / pdw)")
                    elif len(a) >= 2:
                        idx = int(a[0])
                        if not (0 <= idx < 14):
                            log("  index must be 0..13")
                        else:
                            def arr(name, group_name):
                                v = rospy.get_param("/arm_pd/" + name, None)
                                if isinstance(v, list) and len(v) == 14:
                                    return [float(x) for x in v]
                                # inherit the current group gains (pd / pdw) so that
                                # setting ONE joint does not silently zero the others
                                g_all = rospy.get_param("/arm_pd/" + group_name, 0.0) or 0.0
                                g_wr = rospy.get_param("/arm_pd/" + group_name + "_wrist", None)
                                out = []
                                for i in range(14):
                                    wrist = (i % 7) >= 4
                                    out.append(float(g_wr) if (wrist and g_wr is not None)
                                               else float(g_all))
                                return out
                            kp = arr("kp_joints", "kp"); kd = arr("kd_joints", "kd")
                            ki = arr("ki_joints", "ki"); cl = arr("clamp_joints", "clamp")
                            kp[idx] = float(a[1])
                            if len(a) > 2: kd[idx] = float(a[2])
                            if len(a) > 3: ki[idx] = float(a[3])
                            if len(a) > 4: cl[idx] = float(a[4])
                            for name, v in (("kp_joints", kp), ("kd_joints", kd),
                                            ("ki_joints", ki), ("clamp_joints", cl)):
                                rospy.set_param("/arm_pd/" + name, v)
                            log("  %s: kp=%.1f kd=%.1f ki=%.1f clamp=%.1f"
                                % (names[idx], kp[idx], kd[idx], ki[idx], cl[idx]))
                            log("  (note: once kp_joints is set it overrides ALL joints;"
                                " joints left at 0 are disabled)")
                    else:
                        v = rospy.get_param("/arm_pd/kp_joints", None)
                        if isinstance(v, list) and len(v) == 14:
                            for i, n in enumerate(names):
                                log("    [%2d] %-3s kp=%.1f" % (i, n, float(v[i])))
                        else:
                            log("  no per-joint gains set (using pd / pdw)")
                elif cmd in ("ws", "waitstill"):
                    global WAIT_STILL_ENABLED
                    if not a:
                        log("  wait_still: %s" % ("ON" if WAIT_STILL_ENABLED else "OFF"))
                    elif a[0].lower() in ("on", "1", "true"):
                        WAIT_STILL_ENABLED = True
                        log("  wait_still: ON")
                    elif a[0].lower() in ("off", "0", "false"):
                        WAIT_STILL_ENABLED = False
                        log("  wait_still: OFF")
                    else:
                        log("  usage: ws on | ws off | ws")
                elif cmd in ("auto", "go"):
                    self.do_auto()
                elif cmd in ("pdmode", "mode"):
                    if not a:
                        log("  arm PD mode: %s (profile %s)"
                            % (self._pd_mode, self._pd_profile))
                    elif a[0].lower().startswith("pid"):
                        self.set_pd_mode("pidonly")
                        log("  arm PD mode -> pidonly (per-joint replace)")
                    elif a[0].lower().startswith("comb"):
                        self.set_pd_mode("combined")
                        log("  arm PD mode -> combined (WBC + PID)")
                    else:
                        log("  usage: pdmode combined | pdmode pidonly")
                elif cmd in ("pdrep", "replace"):
                    # use ONLY the PID for the arm joints (no WBC torque)
                    if not a:
                        log("  arm PD replace mode: %s"
                            % rospy.get_param("/arm_pd/replace", False))
                    else:
                        val = a[0].lower() in ("on", "1", "true")
                        rospy.set_param("/arm_pd/replace", val)
                        log("  arm PD replace mode: %s" % ("ON (PID only)" if val else "OFF (WBC + PID)"))
                elif cmd in ("td", "topdown"):
                    # toggle the top-down claw orientation on the scale (live)
                    global SCALE_TOPDOWN
                    if not a:
                        log("  scale top-down: %s" % ("ON" if SCALE_TOPDOWN else "OFF"))
                    elif a[0].lower() in ("on", "1", "true"):
                        SCALE_TOPDOWN = True
                        log("  scale top-down: ON")
                    elif a[0].lower() in ("off", "0", "false"):
                        SCALE_TOPDOWN = False
                        log("  scale top-down: OFF (original saggy behaviour)")
                    else:
                        log("  usage: td on | td off | td")
                elif cmd in ("pdinv", "invert"):
                    if not a:
                        log("  wrist-roll inversion: %s"
                            % rospy.get_param("/arm_pd/invert_roll", False))
                    else:
                        val = a[0].lower() in ("on", "1", "true")
                        rospy.set_param("/arm_pd/invert_roll", val)
                        log("  wrist-roll inversion: %s" % ("ON" if val else "OFF"))
                elif cmd in ("mp", "planning"):
                    if not a:
                        log("  motion planning: %s" % ("ON" if self.use_planning else "OFF"))
                    elif a[0].lower() in ("on", "1", "true"):
                        self.use_planning = True
                        log("  motion planning: ON  (home uses MoveIt/OMPL)")
                    elif a[0].lower() in ("off", "0", "false"):
                        self.use_planning = False
                        log("  motion planning: OFF (fixed joint-space moves)")
                    else:
                        log("  usage: mp on | mp off | mp")
                elif cmd in ("sethome", "sh"):
                    self.do_sethome()
                elif cmd in ("dets", "det"):
                    self.do_dets()
                elif cmd == "classify":
                    self.do_classify()
                elif cmd in ("parcels", "parcel"):
                    self.do_parcels()
                elif cmd == "pick":
                    self.do_pick(int(a[0]) if a else 0)
                elif cmd in ("scale", "weigh_place"):
                    self.do_scale(int(a[0]) if a else 0)
                elif cmd == "slift":
                    self.do_slift()
                elif cmd == "scarry":
                    self.do_scarry()
                elif cmd == "sdrop":
                    self.do_sdrop()
                elif cmd == "handoff":
                    self.do_handoff()
                elif cmd == "box":
                    self.do_box(int(a[0]) if a else 0)
                elif cmd in ("task", "run"):
                    self.do_task(int(a[0]) if a else 0)
                elif cmd in ("runall", "alltask", "all4", "run_all"):
                    self.do_all()
                elif cmd in ("67", "sixseven", "sixtyseven", "dance"):
                    self.do_sixtyseven(int(a[0]) if a else None)
                elif cmd in ("set67", "s67"):
                    self.do_set67()
                elif cmd in ("active", "arm"):
                    ar = self.ensure_arm()
                    if a and a[0].lower() in ("l", "left", "r", "right"):
                        ar.active = "left" if a[0].lower().startswith("l") else "right"
                        log("  active arm = %s" % ar.active)
                    else:
                        log("  active arm = %s   (usage: active left|right)" % ar.active)
                elif cmd == "frame":
                    if len(a) >= 3:
                        self.frame_offset = [float(v) for v in a[:3]]
                    log("  frame offset = (%.3f, %.3f, %.3f)"
                        % tuple(self.frame_offset))
                elif cmd == "pos":
                    ar = self.ensure_arm()
                    if len(a) >= 3:
                        ar.move_to(pos=[float(v) for v in a[:3]])
                        ar.print_pose()
                    else:
                        log("  usage: pos X Y Z")
                elif cmd == "wrist":
                    self.do_wrist([float(v) for v in a[:3]] if len(a) >= 3 else None)
                elif cmd == "joints":
                    self.do_joints()
                elif cmd in ("j", "jset"):
                    self.do_jset(int(a[0]), float(a[1]))
                elif cmd == "jbias":
                    if not a:
                        self.do_jbias()
                    elif a[0].lower() in ("on", "off"):
                        self.do_jbias(a[0])
                    else:
                        self.do_jbias(int(a[0]), float(a[1]))
                elif cmd == "lift":
                    self.do_lift(float(a[0]) if a else None)
                elif cmd in ("topdown", "down"):
                    self.do_topdown()
                elif cmd in ("posecheck", "pcheck"):
                    self.do_posecheck()
                elif cmd in ("wrist_step", "wstep", "wtest"):
                    self.do_wrist_step(float(a[0]) if a else 5.0,
                                       a[1].lower() if len(a) > 1 else "right")
                elif cmd in ("unlock", "orient_off"):
                    self.do_unlock()
                elif cmd == "freeze":
                    if a and a[0].lower() in ("on", "off"):
                        self.do_freeze(a[0].lower() == "on")
                    else:
                        self.do_freeze()
                elif cmd in ("pose", "p"):
                    self.do_pose()
                elif cmd == "check":
                    self.ensure_arm().check()
                else:
                    log("unknown command: %s (h = help)" % cmd)
            except (ValueError, IndexError):
                log("bad arguments. h = help")
            except Exception as e:
                log("error: %s" % e)

    def run(self):
        if self.args.live > 0:
            self.do_head()
            self.do_live(self.args.live)
        elif self.args.auto:
            self.run_auto()
        else:
            self.repl()


def main():
    ap = argparse.ArgumentParser(description="scene 1: detection + pick")
    ap.add_argument("--auto", action="store_true",
                    help="run the whole sequence once instead of interactive")
    ap.add_argument("--live", type=float, default=0.0,
                    help="tilt head then repeat detection every N seconds")
    ap.add_argument("--head-pitch", type=float, default=P.HEAD_PITCH)
    ap.add_argument("--head-yaw", type=float, default=P.HEAD_YAW)
    ap.add_argument("--ready-bag", default=None,
                    help="bag whose arm trajectory is replayed (default %s)"
                         % DEFAULT_READY_BAG)
    ap.add_argument("--frame", type=float, nargs=3, default=list(FRAME_OFFSET),
                    metavar=("X", "Y", "Z"),
                    help="base_link -> organisers' IK-frame offset for pick")
    ap.add_argument("--stride", type=int, default=P.DEPTH_STRIDE)
    ap.add_argument("--min-dist", type=float, default=P.DEPTH_MIN)
    ap.add_argument("--max-dist", type=float, default=P.DEPTH_MAX)
    ap.add_argument("--iters", type=int, default=P.RANSAC_ITERS)
    ap.add_argument("--dist", type=float, default=P.RANSAC_DIST)
    ap.add_argument("--planes", type=int, default=P.PLANES)
    ap.add_argument("--tol", type=float, default=P.CLUSTER_TOL)
    ap.add_argument("--min-cluster", type=int, default=P.MIN_CLUSTER)
    ap.add_argument("--green-lo", type=int, nargs=3, default=list(P.GREEN_LO))
    ap.add_argument("--green-hi", type=int, nargs=3, default=list(P.GREEN_HI))
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rospy.init_node("task_sequence", anonymous=True, log_level=rospy.ERROR)
    TaskSequence(args).run()


if __name__ == "__main__":
    main()
