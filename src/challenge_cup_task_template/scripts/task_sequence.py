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
    lift [Z]             move the right hand straight to height Z (default cur z + 0.05)
    keys                 raw single-key mode to nudge hand/wrist (W/S z, A/D y, R/F x, I/K wrist-pitch, ...)
    scale [N]            phase 2: place on the scale, then regrasp
    handoff              phase 3: hand the parcel to the left hand
    box [N]              phase 4: drop into the sorting box
    task [N]             the whole chain (pick -> scale -> handoff -> box)

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
PICK_OFFSET_RIGHT = [-0.03, 0.0, 0.0]
PICK_OFFSET_LEFT = [-0.03, -0.02, 0.0]
PICK_FIRST_YPR = [0.0, -100, 0.0]    # approach direction
PICK_SECOND_YPR = [90.0, 0.0, 0.0]    # tool roll
PICK_TRANSIT_Z = 0.120                # align height before descending
# optional extra offset on the pick point (m).  The organisers' pick uses the
# detected point as-is, so this stays zero; `frame X Y Z` tunes it live if needed.
FRAME_OFFSET = [0.0, 0.0, 0.0]

# ── scale (weigh) / handoff / box: fixed scene points, IK frame ────────────
WEIGH_TRANSIT_Z = 0.25            # carry height over the scale
WEIGH_RELEASE = [0.396, -0.574, 0.146217]   # x, y = fallback if no detection; [2] = drop height
WEIGH_DROP_Z =  0.146217              # hand height where it OPENS to drop the parcel (tune me)
WEIGH_REGRASP_Z = -0.04               # hand height where it re-grasps after weighing
WEIGH_RELEASE_FIRST_YPR = [0.0, -100.0, 0.0]
WEIGH_RELEASE_SECOND_YPR = [90.0, 0.0, 0.0]
WEIGH_RELEASE_SETTLE = 1.5            # wait before opening
WEIGH_REGRASP = [0.396, -0.574, -0.04]
WEIGH_REGRASP_FIRST_YPR = [0.0, -60.0, 0.0]
WEIGH_REGRASP_SECOND_YPR = [90.0, 0.0, 0.0]
WEIGH_DWELL = 1.0                     # wait after releasing

RIGHT_HANDOFF = [0.246, -0.044645, 0.3016983]
RIGHT_HANDOFF_TRANSIT_Z = 0.40
RIGHT_HANDOFF_FALLBACK_ZS = [0.37, 0.35]
RIGHT_HANDOFF_FIRST_YPR = [-0.839, -100.0, 0.0]
RIGHT_HANDOFF_SECOND_YPR = [90.0, -20.0, 90.0]
LEFT_RECEIVE_READY = [0.266, 0.139, 0.2216983]
LEFT_RECEIVE = [0.266, 0.04645, 0.2216983]
LEFT_RECEIVE_FIRST_YPR = [-0.839, -100.0, 0.0]
LEFT_RECEIVE_SECOND_YPR = [-90.0, -0.0, -90.0]
RIGHT_HANDOFF_RETRACT_Y = -0.300      # right hand retracts along y after release

BOX_DROP = [0.605966, 0.226114, 0.556217]
BOX_DROP_X_FALLBACK = [-0.03, -0.06]  # pull x toward the robot if IK fails
BOX_SLOT_OFFSETS = [                  # 2x2 layout, by parcel index
    [0.0, -0.01, 0.0], [0.0, 0.04, 0.0],
    [0.04, -0.03, 0.0], [0.04, 0.03, 0.0],
]
LEFT_BOX_DROP_FIRST_YPR = [-0.328, -100.935, 0.0]
LEFT_BOX_DROP_SECOND_YPR = [-90.0, 0.0, 0.369]
LEFT_PRESET_2 = [0.313, 0.239, 0.282]
LEFT_PRESET_2_FIRST_YPR = [-146.440, 4.966, 0.0]
LEFT_PRESET_2_SECOND_YPR = [0.0, 0.0, 96.580]
PLACE_APPROACH_Z = 0.060
PLACE_DWELL = 0.8

# ── timings / IK modes ──
MOVE_TIME = 1.0
SETTLE_TIME = 0.8
HANDOFF_MOVE_TIME = 2.2
PICK_GRASP_MOVE_TIME = 1.4
PLACE_MOVE_TIME = 2.2
GRIPPER_HOLD = 0.7
IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_MODE_THREE_POINT = 0x06

# parcel / box size thresholds (m), from the scene file
BOX_MIN_LENGTH = 0.30
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
        others = 0
        for d in dets:
            L, W, H = d["length"], d["width"], d["height"]
            if L >= BOX_MIN_LENGTH:
                if self.box is None:
                    self.box = d
                else:
                    others += 1
            elif PARCEL_L_MIN <= L <= PARCEL_L_MAX and W <= PARCEL_W_MAX and H <= PARCEL_H_MAX:
                self.parcels.append(d)
            else:
                others += 1
        log("  classified: %d parcel(s), box=%s, %d ignored"
            % (len(self.parcels), "yes" if self.box else "no", others))

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

    def do_keys(self, step_pos=0.01, step_wrist=2.0):
        """Raw single-key mode to nudge the hand / wrist. Esc (or Ctrl-C) to exit.
        W/S z,  A/D y,  R/F x,  I/K wrist pitch,  J/L wrist yaw,  U/O wrist roll."""
        import sys
        import termios
        import tty
        ar = self.ensure_arm()
        log("  keys:  W/S z   A/D y   R/F x   I/K wrist-pitch   J/L wrist-yaw   U/O wrist-roll   (Esc to exit)")
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not rospy.is_shutdown():
                ch = sys.stdin.read(1)
                if ch in ("\x1b", "\x03"):
                    break
                k = ch.lower()
                if k in "wsadrf":
                    cur = ar.current_pose()[0]
                    dz = step_pos if k == "w" else (-step_pos if k == "s" else 0.0)
                    dy = step_pos if k == "a" else (-step_pos if k == "d" else 0.0)
                    dx = step_pos if k == "r" else (-step_pos if k == "f" else 0.0)
                    ar.wrist_target = False
                    ar.ik_mode = IK_MODE_THREE_POINT
                    ar.move_to(pos=[cur[0] + dx, cur[1] + dy, cur[2] + dz])
                elif k in "ikjluo":
                    w = ar.current_wrist_deg()
                    dw = [0.0, 0.0, 0.0]
                    if k == "i":
                        dw[1] += step_wrist
                    elif k == "k":
                        dw[1] -= step_wrist
                    elif k == "j":
                        dw[0] += step_wrist
                    elif k == "l":
                        dw[0] -= step_wrist
                    elif k == "u":
                        dw[2] += step_wrist
                    elif k == "o":
                        dw[2] -= step_wrist
                    ar.set_wrist([w[0] + dw[0], w[1] + dw[1], w[2] + dw[2]])
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        log("  keys mode off")

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
        ar = self.ensure_arm()
        ar.wrist_target = False            # these are eef targets
        ar.elbow_pos = None
        side = 1 if ar.active == "right" else 0
        log("  pick parcel[%d] -> (%.3f, %.3f, %.3f)  side=%s"
            % (idx, gx, gy, gz, pside))
        ar.ik_mode = 6
        ar.move_to(pos=[gx, gy, PICK_TRANSIT_Z], quat=q)
        ar.ik_mode = 3
        ok = ar.move_to(pos=[gx, gy, gz], quat=q)
        if not ok:
            log("  (0x03 failed -> 0x06)")
            ar.ik_mode = 6
            ar.move_to(pos=[gx, gy, gz], quat=q)
        ar.set_gripper(100.0)
        pct = ar.claw_pct[side]
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
        """Scale centre (x, y) from the detected green marker, else the fixed default."""
        try:
            with open(WEIGH_FILE) as f:
                d = json.load(f)
            return [float(d["x"]), float(d["y"])], "detected"
        except Exception:
            return [WEIGH_RELEASE[0], WEIGH_RELEASE[1]], "default"

    def do_scale(self, idx=0):
        q_carry = _quat_from_ypr(PICK_FIRST_YPR, PICK_SECOND_YPR)
        q_rel = _quat_from_ypr(WEIGH_RELEASE_FIRST_YPR, WEIGH_RELEASE_SECOND_YPR)
        (wx, wy), src = self.weigh_xy()
        release = [wx, wy, WEIGH_DROP_Z]
        regrasp = [wx, wy, WEIGH_REGRASP_Z]
        log("  scale target (%s): x=%.3f y=%.3f" % (src, wx, wy))
        ar = self.ensure_arm()
        ar.active = "right"
        cur = ar.current_pose()[0]
        ar.ik_mode = IK_MODE_THREE_POINT
        ar.move_linear([cur[0], cur[1], WEIGH_TRANSIT_Z], q_carry, n_points=4)
        self.move_hand("right", _with_z(release, WEIGH_TRANSIT_Z), q_carry,
                       IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME, settle=0.0)
        self.move_hand("right", release, q_rel, IK_MODE_POS_HARD_ORI_HARD,
                       PICK_GRASP_MOVE_TIME)
        rospy.sleep(WEIGH_RELEASE_SETTLE)
        self.grip("right", 0.0)
        rospy.sleep(WEIGH_DWELL)
        q_re = _quat_from_ypr(WEIGH_REGRASP_FIRST_YPR, WEIGH_REGRASP_SECOND_YPR)
        pre = _with_z(regrasp, release[2])
        self.move_hand("right", pre, q_re, IK_MODE_THREE_POINT, MOVE_TIME, settle=0.0)
        self.move_hand("right", regrasp, q_re, IK_MODE_POS_HARD_ORI_HARD,
                       PICK_GRASP_MOVE_TIME)
        self.grip("right", 100.0)
        rospy.sleep(GRIPPER_HOLD)
        cur = ar.current_pose()[0]
        ar.move_linear([cur[0], cur[1], release[2]], q_re, n_points=3)
        return True

    # ── phase 3: hand off to the left hand ──────────────────────────
    def do_handoff(self):
        q_h = _quat_from_ypr(RIGHT_HANDOFF_FIRST_YPR, RIGHT_HANDOFF_SECOND_YPR)
        q_recv = _quat_from_ypr(LEFT_RECEIVE_FIRST_YPR, LEFT_RECEIVE_SECOND_YPR)
        ar = self.ensure_arm()
        self.grip("left", 0.0)
        ar.active = "right"
        # Raise + align are TRANSITS — do NOT abort on a tolerance miss. The arm
        # tracks with ~cm error, so move_to often returns False even though it moved.
        cur = ar.current_pose()[0]
        self.move_hand("right", [cur[0], cur[1], RIGHT_HANDOFF_TRANSIT_Z], None,
                       IK_MODE_THREE_POINT, MOVE_TIME)
        self.move_hand("right", _with_z(RIGHT_HANDOFF, RIGHT_HANDOFF_TRANSIT_Z), q_h,
                       IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME)
        self.move_hand("right", RIGHT_HANDOFF, q_h, IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME)
        self.move_hand("left", LEFT_RECEIVE_READY, q_recv, IK_MODE_THREE_POINT, MOVE_TIME)
        self.move_hand("left", LEFT_RECEIVE, q_recv, IK_MODE_THREE_POINT, HANDOFF_MOVE_TIME)
        self.grip("left", 100.0)
        rospy.sleep(GRIPPER_HOLD)
        self.grip("right", 0.0)
        rospy.sleep(GRIPPER_HOLD)
        cur = ar.current_pose()[0]
        self.move_hand("right", [cur[0], cur[1] + RIGHT_HANDOFF_RETRACT_Y, cur[2]], None,
                       IK_MODE_THREE_POINT, MOVE_TIME)
        return True

    # ── phase 4: drop into the sorting box ──────────────────────────
    def do_box(self, idx=0):
        q = _quat_from_ypr(LEFT_BOX_DROP_FIRST_YPR, LEFT_BOX_DROP_SECOND_YPR)
        ar = self.ensure_arm()
        base = _box_drop_ik(idx)
        ok = False
        for dx in [0.0] + BOX_DROP_X_FALLBACK:
            box = [base[0] + dx, base[1], base[2]]
            cur = ar.current_pose()[0]
            raise_pt = [cur[0], cur[1], box[2] + PLACE_APPROACH_Z]
            pre = [box[0], box[1], box[2] + PLACE_APPROACH_Z]
            if (self.move_hand("left", raise_pt, q, IK_MODE_THREE_POINT, MOVE_TIME)
                    and self.move_hand("left", pre, q, IK_MODE_THREE_POINT, PLACE_MOVE_TIME)
                    and self.move_hand("left", box, q, IK_MODE_THREE_POINT, PLACE_MOVE_TIME)):
                if dx != 0.0:
                    log("  (box drop x fallback %.3f)" % dx)
                ok = True
                break
        if not ok:
            log("  box drop failed (IK)")
            return False
        self.grip("left", 0.0)
        rospy.sleep(PLACE_DWELL)
        q2 = _quat_from_ypr(LEFT_PRESET_2_FIRST_YPR, LEFT_PRESET_2_SECOND_YPR)
        self.move_hand("left", LEFT_PRESET_2, q2, IK_MODE_THREE_POINT, PLACE_MOVE_TIME)
        return True

    # ── whole chain for one parcel ──────────────────────────────────
    def do_task(self, idx=0):
        log("=== task parcel[%d] ===" % idx)
        self.ensure_classified()
        if not (0 <= idx < len(self.parcels)):
            log("  parcel index %d out of range" % idx)
            return False
        self.do_pick(idx)
        self.do_scale(idx)
        self.do_handoff()
        self.do_box(idx)
        log("=== task parcel[%d] done ===" % idx)
        return True

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
                elif cmd == "handoff":
                    self.do_handoff()
                elif cmd == "box":
                    self.do_box(int(a[0]) if a else 0)
                elif cmd in ("task", "run"):
                    self.do_task(int(a[0]) if a else 0)
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
                elif cmd == "lift":
                    self.do_lift(float(a[0]) if a else None)
                elif cmd in ("keys", "k"):
                    self.do_keys()
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
