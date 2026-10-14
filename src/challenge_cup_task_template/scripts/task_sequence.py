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

DEFAULT_READY_BAG = "/root/kuavo_ws/go.bag"
# ready pose (14 arm joint angles, degrees) — captured from the go-bag replay,
# so we no longer need the bag for `home`
READY_JOINTS = [30.3, 0.0, 9.6, -137.2, -86.8, -2.9, -2.4,
                30.2, 0.0, -9.2, -138.4, 81.4, 8.0, -1.1]
HOME_TIME = 1.0                    # seconds per segment of the home move (fast)
# "swing back" retract waypoint (14 joints, deg): shoulder rotated back and the
# elbow straighter than in the ready pose, so the arms clear the table on the way.
# First 7 = left arm, last 7 = right arm.
HOME_RETRACT = [60.0, 10.8, 9.6, -100.0, -86.8, -2.9, -2.4,
                60.0, -11.0, -9.2, -100.0, 81.4, 8.0, -1.1]
DET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.json")
WEIGH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weigh.json")

# ── scene-1 pick constants (ported from the organisers' collector) ─────────
PICK_IK_Z = -0.03                     # absolute grasp height
# grasp offset split by SIDE (right / left), decided by y vs the row's median y
PICK_OFFSET_RIGHT = [-0.04, -0.02, 0.0]
PICK_OFFSET_LEFT = [-0.03, 0.00, 0.0]
PICK_FIRST_YPR = [0.0, -100, 0.0]    # approach direction
PICK_SECOND_YPR = [90.0, 0.0, 0.0]    # tool roll
PICK_TRANSIT_Z = 0.120                # align height before descending
# optional extra offset on the pick point (m).  The organisers' pick uses the
# detected point as-is, so this stays zero; `frame X Y Z` tunes it live if needed.
FRAME_OFFSET = [0.0, 0.0, 0.0]

# ── scale (weigh) / handoff / box: fixed scene points, IK frame ────────────
WEIGH_TRANSIT_Z = 0.25            # carry height over the scale
WEIGH_RELEASE = [0.396, -0.574, 0.146217]   # x, y = fallback if no detection; [2] = drop height
WEIGH_DROP_Z =  0.1              # hand height where it OPENS to drop the parcel (tune me)
WEIGH_REGRASP_Z = 0.02               # hand height where it re-grasps after weighing
# shift the scale placement toward the robot (the arm can't reach the centre
# comfortably). [x, y, z] in the base frame: +x away from the body, +y left.
WEIGH_OFFSET = [-0.03, 0.03, 0.0]
# tiny nudge applied to the re-grasp point after weighing (base frame: +x = away from the
# body, +y = LEFT, so "to the right" is -y).
REGRASP_OFFSET = [0.0, -0.01, 0.0]
WEIGH_RELEASE_FIRST_YPR = [0.0, -100.0, 0.0]
WEIGH_RELEASE_SECOND_YPR = [90.0, 0.0, 0.0]
WEIGH_RELEASE_SETTLE = 1.5            # wait before opening
WEIGH_REGRASP = [0.396, -0.574, -0.04]
WEIGH_REGRASP_FIRST_YPR = [0.0, -60.0, 0.0]
WEIGH_REGRASP_SECOND_YPR = [90.0, 0.0, 0.0]
WEIGH_DWELL = 1.0                     # wait after releasing
# The claw must point straight down when the parcel is placed on the scale, but the
# arm's small "points-up" tracking error leaves it a few degrees high. So we OVER-ROTATE
# the aim PAST vertical (≈ "curl more"). Positive = curl back toward the robot (-x);
# flip the sign to curl the other way.
WRIST_MAX_FIX = 0.08                  # ignore a wrist-camera fix further than this from the drop
USE_WRIST_LOCATE = False              # re-grasp via the wrist camera (OFF for now)
TOPDOWN_TILT_DEG = 8.0
_TOPDOWN_T = math.radians(TOPDOWN_TILT_DEG)
WEIGH_TOPDOWN_DIR = [-math.sin(_TOPDOWN_T), 0.0, -math.cos(_TOPDOWN_T)]

RIGHT_HANDOFF = [0.246, -0.044645, 0.3016983]   # 3 cm back OUT, away from the body (-y)
RIGHT_HANDOFF_TRANSIT_Z = 0.40
RIGHT_HANDOFF_FALLBACK_ZS = [0.37, 0.35]
RIGHT_HANDOFF_FIRST_YPR = [-0.839, -100.0, 0.0]
RIGHT_HANDOFF_SECOND_YPR = [90.0, -20.0, 90.0]
LEFT_RECEIVE_READY = [0.266, 0.209, 0.250]
LEFT_RECEIVE = [0.266, 0.05, 0.250]   # 3 cm to the right (-y)
LEFT_RECEIVE_FIRST_YPR = [-0.839, -100.0, 0.0]
LEFT_RECEIVE_SECOND_YPR = [-90.0, -0.0, -90.0]
RIGHT_HANDOFF_RETRACT_Y = -0.300      # right hand retracts along y after release
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
BOX_SLOT_OFFSETS = [                  # 2x2 layout, by parcel index
    [0.0, -0.01, 0.0], [0.0, 0.04, 0.0],
    [0.04, -0.03, 0.0], [0.04, 0.03, 0.0],
]
LEFT_BOX_DROP_FIRST_YPR = [-0.328, -100.935, 0.0]
LEFT_BOX_DROP_SECOND_YPR = [-90.0, 0.0, 0.369]
LEFT_PRESET_2 = [0.313, 0.239, 0.282]
LEFT_PRESET_2_FIRST_YPR = [-146.440, 4.966, 0.0]
LEFT_PRESET_2_SECOND_YPR = [0.0, 0.0, 96.580]
BOX_LIFT_Z = 0.450                    # lift the left hand to this height BEFORE crossing to the box
BOX_RETRACT = [-0.150, 0.0, 0.0]      # after release: pull the left hand BACK before going home
PLACE_DWELL = 0.8

# ── timings / IK modes ──
MOVE_TIME = 1.0
SETTLE_TIME = 0.8
HANDOFF_MOVE_TIME = 2.2
RIGHT_HANDOFF_MOVE_TIME = 4.5         # the right arm gets there SLOWER (tracks/poses better)
SCALE_TO_HANDOFF_DELAY = 4.0          # pause after the re-grasp, before the handoff
PICK_GRASP_MOVE_TIME = 1.4
PLACE_MOVE_TIME = 2.2
GRIPPER_HOLD = 0.7
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

    # ── arm (lazy) ──────────────────────────────────────────────────
    def ensure_arm(self):
        if self.arm is None:
            import arm_test_control as AC
            log("initialising arm control (mode 2, 100 Hz stream) ...")
            a = AC.ArmTestControl()
            a.setup()
            self.arm = a
        return self.arm

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

    def do_home(self, duration=None):
        """Move both arms to the ready pose, passing the captured retract waypoint
        first so the arms clear the table."""
        ar = self.ensure_arm()
        dur = HOME_TIME if duration is None else float(duration)
        if self.home_retract is not None:
            log("  home: retract waypoint -> ready pose")
            ar.move_joints(self.home_retract, duration=dur)
        return ar.move_joints(READY_JOINTS, duration=dur)

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
        self.move_hand("right", [gx, gy, PICK_TRANSIT_Z], q, IK_MODE_THREE_POINT, MOVE_TIME)
        # let the claw STOP SWINGING before we descend onto the parcel
        self.wait_still()
        if not self.move_hand("right", [gx, gy, gz], q, IK_MODE_POS_HARD_ORI_HARD,
                              PICK_GRASP_MOVE_TIME):
            log("  (0x03 failed -> 0x06)")
            self.move_hand("right", [gx, gy, gz], q, IK_MODE_THREE_POINT, PICK_GRASP_MOVE_TIME)
        self.wait_still()          # settle at the grasp pose before closing
        self.grip("right", 100.0)
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

    def wait_still(self, tol_deg=0.6, window=0.3, timeout=5.0):
        """Block until the arms STOP MOVING: the biggest joint change over `window`
        seconds is below `tol_deg`. Use before descending/closing so we never act
        while the claw is still swinging after a move."""
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

    def grip(self, side, pct):
        """Open/close ONE claw. On close, use the closed-loop grip (hold + grab check);
        the other claw keeps its state."""
        ar = self.ensure_arm()
        prev = ar.active
        ar.active = side
        if pct > 0:
            ar.grip_close_loop()
        else:
            ar.set_gripper(pct)
        ar.active = prev

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
        """carry to the scale with the claw TOP-DOWN, place it slowly (no drop),
        then re-grasp straight down from the same pose."""
        ar = self.ensure_arm()
        ar.active = "right"
        q_carry = _quat_from_ypr(PICK_FIRST_YPR, PICK_SECOND_YPR)
        (wx, wy), src = self.weigh_xy()
        log("  scale (%s): x=%.3f y=%.3f   carry z=%.3f  place z=%.3f"
            % (src, wx, wy, WEIGH_TRANSIT_Z, WEIGH_DROP_Z))

        # 1) lift straight up to the carry height
        cur = ar.current_pose()[0]
        ar.ik_mode = IK_MODE_THREE_POINT
        ar.settle = SETTLE_TIME
        ar.move_linear([cur[0], cur[1], WEIGH_TRANSIT_Z], q_carry, n_points=4)

        # 2) turn the claw TOP-DOWN, over-rotated past vertical (sets `fixed_rpy`)
        log("  scale: claw aim = (%.2f, %.2f, %.2f)  (over-tilt %.0f deg)"
            % (WEIGH_TOPDOWN_DIR[0], WEIGH_TOPDOWN_DIR[1], WEIGH_TOPDOWN_DIR[2],
               TOPDOWN_TILT_DEG))
        if ar.align_claw(WEIGH_TOPDOWN_DIR):
            cur = ar.current_pose()[0]
            ar.ik_mode = IK_MODE_POS_HARD_ORI_HARD
            ar.move_to(pos=[float(cur[0]), float(cur[1]), float(cur[2])])
        # from here, quat=None means "keep the claw pointing down"

        # 3) carry over the scale, claw down
        self.move_hand("right", [wx, wy, WEIGH_TRANSIT_Z], None,
                       IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME, settle=0.0)
        self.wait_still()          # line up over the scale BEFORE descending

        # 4) place it SLOWLY (lower, don't drop), then open
        self.move_hand("right", [wx, wy, WEIGH_DROP_Z], None,
                       IK_MODE_POS_HARD_ORI_HARD, PLACE_MOVE_TIME)
        self.wait_still()          # claw must stop swinging before we release
        # the parcel lands where the hand ACTUALLY is (commanded xy != reached xy), so
        # remember that xy as the fallback for the re-grasp below
        rp = ar.current_pose()[0]
        rx, ry = float(rp[0]), float(rp[1])
        log("  scale-regrasp: release xy = (%.3f, %.3f)   (commanded %.3f, %.3f)"
            % (rx, ry, wx, wy))
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
        self.wait_still()
        self.move_hand("right", [rx, ry, WEIGH_REGRASP_Z], None,
                       IK_MODE_POS_HARD_ORI_HARD, PICK_GRASP_MOVE_TIME)
        self.wait_still()          # settle before closing on the parcel
        self.grip("right", 100.0)
        rospy.sleep(GRIPPER_HOLD)

        # 6) lift back up, claw down
        cur = ar.current_pose()[0]
        ar.move_linear([cur[0], cur[1], WEIGH_TRANSIT_Z], None, n_points=4)
        # the top-down lock is only for this scale phase — release it
        ar.fixed_rpy = None
        return True

    # ── phase 3: hand off to the left hand ──────────────────────────
    def do_handoff(self):
        q_recv = _quat_from_ypr(LEFT_RECEIVE_FIRST_YPR, LEFT_RECEIVE_SECOND_YPR)
        ar = self.ensure_arm()
        self.grip("left", 0.0)
        ar.active = "right"
        # Raise + align are TRANSITS — do NOT abort on a tolerance miss. The arm
        # tracks with ~cm error, so move_to often returns False even though it moved.
        # HOLD THE CLAW TOP-DOWN by locking the orientation (not by freezing the wrist):
        # a frozen wrist leaves the attitude UNMANAGED, so the claw tilts as the arm
        # moves. With the lock, the IK keeps the claw aimed down the whole way.
        ar.freeze_wrist = False
        if not ar.align_claw(WEIGH_TOPDOWN_DIR):
            ar.align_claw((0.0, 0.0, -1.0))
        q_h = None                       # quat=None -> moves use the top-down lock
        cur = ar.current_pose()[0]
        self.move_hand("right", [cur[0], cur[1], RIGHT_HANDOFF_TRANSIT_Z], None,
                       IK_MODE_THREE_POINT, MOVE_TIME)
        self.move_hand("right", _with_z(RIGHT_HANDOFF, RIGHT_HANDOFF_TRANSIT_Z), None,
                       IK_MODE_THREE_POINT, RIGHT_HANDOFF_MOVE_TIME)
        # NO final lowering move — stay at the transit height here (the left receive is
        # computed from wherever the right actually is, so it comes to this height)
        # let it STOP before reading — the right drifts/tilts for a moment under the
        # parcel, and we don't want to copy a transient pose or transfer mid-move.
        # (No re-assert here: a fresh IK solve can jump to another branch and swing.)
        self.wait_still()
        # read where the right hand ACTUALLY is (it sags under the parcel), and send
        # the left hand there, instead of assuming the commanded pose
        rp, rq_act = ar.current_pose()
        try:
            from tf.transformations import euler_from_quaternion, quaternion_from_euler
            ra = [math.degrees(v) for v in euler_from_quaternion(rq_act)]
            if q_h is not None:
                q_ref = list(q_h)
            elif ar.fixed_rpy is not None:      # the aim we locked (top-down)
                q_ref = list(quaternion_from_euler(
                    *[math.radians(v) for v in ar.fixed_rpy]))
            else:
                q_ref = None
            if q_ref is not None:
                rw = [math.degrees(v) for v in euler_from_quaternion(q_ref)]
                dot = abs(sum(a * b for a, b in zip(rq_act, q_ref)))
                ang = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))
                log("  handoff: right ORI actual (%.1f, %.1f, %.1f) vs wanted (%.1f, %.1f, %.1f)"
                    "  err %.1f deg" % (ra[0], ra[1], ra[2], rw[0], rw[1], rw[2], ang))
        except Exception as e:
            log("  handoff: (orientation check unavailable: %s)" % e)
        lrecv = [rp[i] + LEFT_RECEIVE_DELTA[i] for i in range(3)]
        lready = [lrecv[i] + LEFT_RECEIVE_READY_DELTA[i] for i in range(3)]
        log("  handoff: right actually at (%.3f, %.3f, %.3f) -> left receive (%.3f, %.3f, %.3f)"
            % (rp[0], rp[1], rp[2], lrecv[0], lrecv[1], lrecv[2]))
        self.move_hand("left", lready, q_recv, IK_MODE_THREE_POINT, MOVE_TIME)
        self.move_hand("left", lrecv, q_recv, IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME)
        self.wait_still()          # both arms stopped before the transfer grips
        # Did it actually reach? Look through the LEFT wrist camera and measure how far
        # the parcel still is; if it's short, step the claw toward the right hand (-y).
        # (OFF for now — set LEFT_REACH_CHECK = True to enable.)
        for attempt in range(LEFT_REACH_TRIES if LEFT_REACH_CHECK else 0):
            try:
                d, npx = P.wrist_distance(self.pa, "left")
            except Exception as e:
                log("  handoff: left reach check failed (%s)" % e)
                break
            if d is None:
                log("  handoff: left reach check: nothing in view (n=%d)" % npx)
                break
            log("  handoff: left claw->parcel %.3f m (n=%d)" % (d, npx))
            if d <= LEFT_REACH_DIST:
                break
            lrecv = [lrecv[0], lrecv[1] - LEFT_REACH_STEP, lrecv[2]]
            log("  handoff: still short -> step in to (%.3f, %.3f, %.3f)"
                % (lrecv[0], lrecv[1], lrecv[2]))
            self.move_hand("left", lrecv, q_recv, IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME)
            self.wait_still()
        self.grip("left", 100.0)
        rospy.sleep(GRIPPER_HOLD)
        self.grip("right", 0.0)
        rospy.sleep(GRIPPER_HOLD)
        cur = ar.current_pose()[0]
        self.move_hand("right", [cur[0], cur[1] + RIGHT_HANDOFF_RETRACT_Y, cur[2]], None,
                       IK_MODE_THREE_POINT, MOVE_TIME)
        ar.fixed_rpy = None              # drop the top-down handoff lock
        return True

    # ── phase 4: drop into the sorting box ──────────────────────────
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
            box = [base[0] + dx, base[1], base[2]]
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
        cur = ar.current_pose()[0]
        clear = max(BOX_LIFT_Z, self.clearance_z())
        log("  box: retreating via clear height z=%.3f" % clear)
        ar.move_linear([cur[0], cur[1], clear], q, n_points=5, seg_time=0.5)
        cur = ar.current_pose()[0]
        log("  box: retracting back by (%.3f, %.3f, %.3f)" % tuple(BOX_RETRACT))
        ar.move_linear([cur[0] + BOX_RETRACT[0], cur[1] + BOX_RETRACT[1],
                        cur[2] + BOX_RETRACT[2]], q, n_points=4, seg_time=0.5)
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
            try:
                if self.do_task(idx):
                    done += 1
            except Exception as e:
                log("  parcel %d failed: %s (continuing)" % (idx, e))
            # make sure both arms are back at the ready pose before the next parcel
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
        """head -> cloud -> ransac -> cluster -> classify -> weigh, printing only the
        cluster/parcel result (cloud + ransac are silenced)."""
        import io
        import sys
        log("  -- detect: head -> cloud -> ransac -> cluster -> classify -> weigh --")
        self.do_head()
        sink = io.StringIO()
        real = sys.stdout
        sys.stdout = sink
        try:
            self.do_cloud()
            self.do_ransac()
        finally:
            sys.stdout = real
        # draw the cluster overlay automatically (blocking window: it shows, you press
        # a key in it to continue)
        self.pa.show = True
        self.pa.live = False
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
                    self.do_home(float(a[0]) if a else None)
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
