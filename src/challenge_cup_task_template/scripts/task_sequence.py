#!/usr/bin/env python3
"""task_sequence.py — Task 1: camera detection + head, and go-bag arm trajectory.

Interactive (default):

    python3 task_sequence.py

At the `task>` prompt:

    head                 tilt the head down
    cloud                capture RGB-D -> point cloud        (detection step 1)
    ransac               remove floor + table planes         (detection step 2)
    cluster              cluster + PCA -> object (x,y,z,yaw) (detection step 3)
    weigh                find the green weighing marker      (detection step 4)
    all                  cloud -> ransac -> cluster
    live [SEC]           repeat detection every SEC s (default 1.0)
    show [on|off]        toggle the overlay window
    pitch V | yaw V      head angles in degrees (pitch>0 looks down)

    gobag [BAG]          replay a bag's arm trajectory to the ready pose
    dets                 list detected objects (from detections.json)
    classify             split detections into box + parcels (by size)
    parcels              list classified parcels
    hover [N] [HEIGHT]   move right hand HEIGHT m above parcel N (default 0/0.10)
    pos X Y Z            move the hand to an absolute position (m), keep orientation
    freeze [on|off]      freeze the wrist JOINTS (angles locked) — not a target
    target [eef|wrist]   what hover aims at: the WRIST (default) or the eef point
    offset [X Y Z]       add a fixed offset (m) to every hover target
    back [V]             pull the target back toward the robot (x -= V, default 0.05)
    face [x|-x|y|-y|z|-z|down]  lock the claw to point along an axis
    orient [R P Y|off]   lock the claw orientation (degrees), or unlock
    ik MODE              IK constraint mode: 6 = mixed (default), 3 = pos+ori hard,
                         2 = pos hard (use 3 so `face/orient` is actually enforced)
    pose | p             print the current hand pose (xyz + rpy)
    check                FK vs TF pose + the TRUE claw axis (wrist->eef from TF)
    down                 lock the orientation so the SIM claw points straight DOWN
    topdown [on|off]     on = claw-down + ik 3 for hover (default OFF); off = revert
    align X Y Z          lock the orientation so the claw points along (X,Y,Z)
    wrist [Y P R]        set the 3 wrist joints (hand yaw/pitch/roll, deg); no
                         args prints them. Shoulder + elbow stay put.
    wristdown            after a wrist hover, rotate the claw down in place
    look [on|off]        hover faces the hovered parcel horizontally (default ON)
    elbow [X Y Z|off]    preferred elbow position (base frame) to avoid weird poses
    h / q                help / quit

Non-interactive:

    python3 task_sequence.py --auto
    python3 task_sequence.py --auto --ready-bag /root/kuavo_ws/go.bag

`--auto` runs: TILT_HEAD -> ARM_READY (go-bag replay) -> CLOUD -> RANSAC
-> CLUSTER -> DONE.
"""

import argparse
import json
import math
import os
import time

import rospy

import perception as P

DEFAULT_READY_BAG = "/root/kuavo_ws/go.bag"
DET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.json")

# Size thresholds (metres) for scene 1, from the scene file:
#   parcel : 0.045 x 0.12 x 0.10   (footprint ~0.12 x 0.045, height ~0.10)
#   box    : ~0.45 x 0.35 x 0.30   (footprint ~0.5 x 0.43,  height ~0.29)
BOX_MIN_LENGTH = 0.30
PARCEL_L_MIN, PARCEL_L_MAX = 0.07, 0.20
PARCEL_W_MAX = 0.10
PARCEL_H_MAX = 0.20
MAX_PARCELS = 4          # scene 1 has exactly 4 parcels
PARCEL_MERGE_DIST = 0.06  # merge parcel detections closer than this (same object)


def perception_args(a):
    """Build the Namespace the perception.py step functions expect."""
    return argparse.Namespace(
        stride=a.stride, min_dist=a.min_dist, max_dist=a.max_dist,
        iters=a.iters, dist=a.dist, planes=a.planes, tol=a.tol,
        min_cluster=a.min_cluster, timeout=a.timeout, show=a.show,
        green_lo=a.green_lo, green_hi=a.green_hi, live=False,
    )


def log(msg):
    print(msg, flush=True)


class TaskSequence(object):
    def __init__(self, args):
        self.args = args
        self.pa = perception_args(args)
        self.head_pitch = args.head_pitch
        self.head_yaw = args.head_yaw
        self.state = "TILT_HEAD"
        self.arm = None
        self.box = None        # single box detection (dict)
        self.parcels = []      # list of parcel detections (dicts)
        self.target_offset = list(args.offset)   # add to every hover target (m)
        self.armed = False                       # claw-down + ik3 set up yet?
        self.use_topdown = False                 # opt-in; off = keep orientation
        self.elbow = list(args.elbow) if args.elbow else None   # optional elbow preference
        self.look_enabled = not args.no_look         # hover always faces the parcel
        self.wrist_target = True                     # hover targets the WRIST by default

    # ── arm (lazy) ──────────────────────────────────────────────────
    def ensure_arm(self):
        if self.arm is None:
            import arm_test_control as AC
            log("initialising arm control (mode 2, 100 Hz stream) ...")
            a = AC.ArmTestControl()
            a.setup()
            self.arm = a
        return self.arm

    def do_gobag(self, path=None):
        """Replay a bag's whole arm trajectory (ACT `go` style) to reach ready."""
        path = path or self.args.ready_bag or DEFAULT_READY_BAG
        if not path or not os.path.exists(path):
            log("  no go bag at %s — pass a path: gobag /path/to/go.bag" % path)
            return False
        from bag_to_ready import load_arm_traj
        frames = load_arm_traj(path)
        if not frames:
            log("  no arm trajectory found in %s" % path)
            return False
        a = self.ensure_arm()
        t0 = frames[0][0]
        duration = frames[-1][0] - t0
        log("  go bag: %d frames, %.1f s — moving to the first frame, then replaying"
            % (len(frames), duration))

        # 1) ease from the current joints to the bag's first frame
        a.move_joints(frames[0][1])

        # 2) follow the recorded trajectory in real time, resampled at the stream rate
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

    # ── detections / arm positioning ────────────────────────────────
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
        if not dets:
            return
        for i, d in enumerate(dets):
            log("  [%d] x=%.3f y=%.3f z_top=%.3f yaw=%.1f  size=(%.3f x %.3f x %.3f)"
                % (i, d["x"], d["y"], d["z_top"], d["yaw_deg"],
                   d["length"], d["width"], d["height"]))
        log("  use: hover <parcel-index> [height]   (run `classify` first)")

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

        # dedup: merge parcels whose centres are close (one parcel split in two)
        if len(self.parcels) > 1:
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

        # cap: keep the largest MAX_PARCELS (scene 1 has exactly 4)
        if len(self.parcels) > MAX_PARCELS:
            self.parcels = sorted(self.parcels, key=lambda d: -d.get("n", 0))[:MAX_PARCELS]
            log("  kept the %d largest parcels" % MAX_PARCELS)

        # stable, spatial order so indices mean the same thing every run
        self.parcels.sort(key=lambda d: (round(d["x"], 2), round(d["y"], 2)))

        if self.box:
            log("    box      x=%.3f y=%.3f z_top=%.3f  size=(%.3f x %.3f x %.3f)"
                % (self.box["x"], self.box["y"], self.box["z_top"],
                   self.box["length"], self.box["width"], self.box["height"]))
        for i, p in enumerate(self.parcels):
            log("    parcel[%d] x=%.3f y=%.3f z_top=%.3f yaw=%.1f  size=(%.3f x %.3f x %.3f)"
                % (i, p["x"], p["y"], p["z_top"], p["yaw_deg"],
                   p["length"], p["width"], p["height"]))
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

    def do_pose(self):
        a = self.ensure_arm()
        a.print_pose()

    def do_wristdown(self):
        """After a `hover` (with `target wrist`), rotate the claw to point down
        while keeping the wrist where it is."""
        ar = self.ensure_arm()
        if not ar.wrist_target:
            log("  enable 'target wrist' first (so the wrist position is what's held)")
            return False
        res = ar._tf_claw_axis()
        if res is None:
            log("  cannot read the claw axis (no TF)")
            return False
        _, wrist_pos, _ = res
        ar.align_claw((0.0, 0.0, -1.0))
        ar.ik_mode = 3
        log("  rotating claw down, keeping wrist at (%.3f, %.3f, %.3f)"
            % (wrist_pos[0], wrist_pos[1], wrist_pos[2]))
        ok = ar.move_to(pos=list(wrist_pos))
        ar.print_pose()
        return ok

    def apply_look(self, ar, pos, idx):
        """Align the claw horizontally toward parcel `idx`. Returns the target
        direction (or None)."""
        if not (0 <= idx < len(self.parcels)):
            return None
        p = self.parcels[idx]
        dx = p["x"] - pos[0]
        dy = p["y"] - pos[1]
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return None
        d = [dx / n, dy / n, 0.0]           # horizontal direction toward the parcel
        ar.align_claw(d)
        ar.ik_mode = 3
        log("  look -> claw horizontal toward parcel[%d], dir=(%.2f, %.2f, 0.00)"
            % (idx, d[0], d[1]))
        return d

    def look_correct(self, ar, pos, d, tries=2, tol_deg=4.0):
        """After a move, measure the real claw axis (TF) and re-align if it
        drifted off the horizontal target direction."""
        for k in range(tries):
            res = ar._tf_claw_axis()
            if res is None:
                return
            ax = res[0]
            dot = max(-1.0, min(1.0, ax[0] * d[0] + ax[1] * d[1] + ax[2] * d[2]))
            ang = math.degrees(math.acos(dot))
            if ang <= tol_deg:
                return
            log("  look correct %d/%d: claw off by %.1f deg -> re-align" % (k + 1, tries, ang))
            ar.align_claw(d)
            ar.ik_mode = 3
            ar.move_to(pos=pos)

    def do_hover(self, idx=0, height=0.10, side="right"):
        """Move the hand to a point `height` metres ABOVE parcel `idx`."""
        self.ensure_classified()
        if idx < 0 or idx >= len(self.parcels):
            log("  parcel index %d out of range (0..%d)" % (idx, len(self.parcels) - 1))
            return False
        p = self.parcels[idx]
        ox, oy, oz = self.target_offset
        pos = [p["x"] + ox, p["y"] + oy, p["z_top"] + height + oz]
        a = self.ensure_arm()
        a.active = side
        a.wrist_target = self.wrist_target
        if self.elbow:
            e = list(self.elbow)
            if side == "left":
                e[1] = -e[1]         # mirror y for the left arm
            a.elbow_pos = e
        if self.use_topdown and not self.armed:
            log("  (arming: claw-down orientation + ik 3)")
            a.align_claw((0.0, 0.0, -1.0))
            a.ik_mode = 3
            self.armed = True
        d = self.apply_look(a, pos, idx) if self.look_enabled else None
        log("  hover %s above parcel[%d] (%.3f, %.3f, %.3f)  +%.2f m  offset=(%.3f, %.3f, %.3f)  target=%s"
            % (side, idx, pos[0], pos[1], pos[2], height, ox, oy, oz,
               "wrist" if a.wrist_target else "eef"))
        ok = a.move_to(pos=pos)
        if d is not None:
            self.look_correct(a, pos, d)
        a.print_pose()
        return ok

    # ── detection steps ─────────────────────────────────────────────
    def do_head(self):        P.set_head(self.head_yaw, self.head_pitch)

    def do_cloud(self):
        P.step_cloud(self.pa)

    def do_ransac(self):
        P.step_ransac(self.pa)

    def do_cluster(self):
        P.step_cluster(self.pa)

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

    # TILT_HEAD -> ARM_READY -> PERCEIVE_CLOUD -> PERCEIVE_RANSAC
    #           -> PERCEIVE_CLUSTER -> DONE
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
                elif cmd == "live":
                    self.do_live(float(a[0]) if a else 1.0)
                elif cmd == "show":
                    if a and a[0].lower() in ("on", "off"):
                        self.pa.show = (a[0].lower() == "on")
                    else:
                        self.pa.show = not self.pa.show
                    log("show = %s" % self.pa.show)
                elif cmd == "pitch":
                    self.head_pitch = float(a[0])
                    log("head pitch = %.1f" % self.head_pitch)
                elif cmd == "yaw":
                    self.head_yaw = float(a[0])
                    log("head yaw = %.1f" % self.head_yaw)
                elif cmd == "gobag":
                    self.do_gobag(a[0] if a else None)
                elif cmd in ("dets", "det"):
                    self.do_dets()
                elif cmd == "classify":
                    self.do_classify()
                elif cmd in ("parcels", "parcel"):
                    self.do_parcels()
                elif cmd == "hover":
                    idx = int(a[0]) if a else 0
                    h = float(a[1]) if len(a) > 1 else 0.10
                    self.do_hover(idx, h)
                elif cmd == "pos":
                    ar = self.ensure_arm()
                    if len(a) >= 3:
                        ar.move_to(pos=[float(v) for v in a[:3]])
                        ar.print_pose()
                    else:
                        log("  usage: pos X Y Z")
                elif cmd in ("pose", "p"):
                    self.do_pose()
                elif cmd == "check":
                    self.ensure_arm().check()
                elif cmd == "down":
                    self.ensure_arm().align_claw((0.0, 0.0, -1.0))
                elif cmd == "topdown":
                    ar = self.ensure_arm()
                    if a and a[0].lower() in ("on", "off"):
                        self.use_topdown = (a[0].lower() == "on")
                    else:
                        self.use_topdown = True
                    if self.use_topdown:
                        ar.align_claw((0.0, 0.0, -1.0))
                        ar.ik_mode = 3
                        self.armed = True
                        log("  topdown ON  (claw down + ik 3)")
                    else:
                        import arm_test_control as AC
                        ar.fixed_rpy = None
                        ar.ik_mode = AC.DEFAULT_IK_MODE
                        self.armed = False
                        log("  topdown OFF (keep current orientation)")
                elif cmd == "align":
                    ar = self.ensure_arm()
                    if len(a) >= 3:
                        ar.align_claw([float(v) for v in a[:3]])
                    else:
                        log("  usage: align X Y Z")
                elif cmd == "wrist":
                    ar = self.ensure_arm()
                    if len(a) >= 3:
                        ar.set_wrist([float(v) for v in a[:3]])
                    else:
                        w = ar.current_wrist_deg()
                        log("  wrist deg (yaw, pitch, roll) = (%.1f, %.1f, %.1f)"
                            % (w[0], w[1], w[2]))
                elif cmd == "elbow":
                    ar = self.ensure_arm()
                    if a and a[0].lower() == "off":
                        self.elbow = None
                        ar.elbow_pos = None
                        log("  elbow preference OFF")
                    elif len(a) >= 3:
                        self.elbow = [float(v) for v in a[:3]]
                        ar.elbow_pos = list(self.elbow)
                        log("  elbow preference = (%.3f, %.3f, %.3f)"
                            % tuple(self.elbow))
                    else:
                        log("  elbow preference = %s" % self.elbow)
                elif cmd == "wristdown":
                    self.do_wristdown()
                elif cmd == "look":
                    if a and a[0].lower() in ("on", "off"):
                        self.look_enabled = (a[0].lower() == "on")
                    else:
                        self.look_enabled = not self.look_enabled
                    log("  look = %s (hover will face the hovered parcel horizontally)"
                        % self.look_enabled)
                elif cmd == "freeze":
                    ar = self.ensure_arm()
                    if a and a[0].lower() in ("on", "off"):
                        ar.freeze_wrist = (a[0].lower() == "on")
                    else:
                        ar.freeze_wrist = not ar.freeze_wrist
                    log("  freeze wrist joints = %s" % ar.freeze_wrist)
                elif cmd == "target":
                    ar = self.ensure_arm()
                    if a and a[0].lower() in ("eef", "wrist"):
                        self.wrist_target = (a[0].lower() == "wrist")
                    else:
                        self.wrist_target = not self.wrist_target
                    ar.wrist_target = self.wrist_target
                    log("  IK target = %s" % ("wrist" if self.wrist_target else "eef"))
                elif cmd == "offset":
                    if len(a) >= 3:
                        self.target_offset = [float(v) for v in a[:3]]
                    log("  target offset = (%.3f, %.3f, %.3f)"
                        % tuple(self.target_offset))
                elif cmd == "back":
                    v = float(a[0]) if a else 0.05
                    self.target_offset[0] -= v
                    log("  target offset = (%.3f, %.3f, %.3f)"
                        % tuple(self.target_offset))
                elif cmd == "face":
                    ar = self.ensure_arm()
                    which = a[0].lower() if a else "x"
                    ar.fixed_rpy = {
                        "x": [0.0, 90.0, 0.0], "+x": [0.0, 90.0, 0.0],
                        "-x": [0.0, -90.0, 0.0],
                        "y": [0.0, 90.0, 90.0], "+y": [0.0, 90.0, 90.0],
                        "-y": [0.0, 90.0, -90.0],
                        "z": [0.0, 0.0, 0.0], "+z": [0.0, 0.0, 0.0],
                        "-z": [0.0, 180.0, 0.0], "down": [0.0, 180.0, 0.0],
                        "up": [0.0, 0.0, 0.0],
                    }.get(which)
                    log("  claw faces %s -> orientation %s" % (which, ar.fixed_rpy))
                elif cmd == "orient":
                    ar = self.ensure_arm()
                    if a and a[0].lower() == "off":
                        ar.fixed_rpy = None
                        log("  orientation lock OFF")
                    else:
                        ar.fixed_rpy = [float(v) for v in a[:3]]
                        log("  orientation locked to (%.1f, %.1f, %.1f) deg"
                            % tuple(ar.fixed_rpy))
                elif cmd == "ik":
                    ar = self.ensure_arm()
                    ar.ik_mode = int(a[0], 0)
                    log("  IK constraint mode = 0x%02x" % ar.ik_mode)
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
    ap = argparse.ArgumentParser(description="Task 1: detection + go-bag arm trajectory")
    ap.add_argument("--auto", action="store_true",
                    help="run the whole sequence once instead of interactive")
    ap.add_argument("--live", type=float, default=0.0,
                    help="tilt head then repeat detection every N seconds")
    ap.add_argument("--head-pitch", type=float, default=P.HEAD_PITCH,
                    help="head pitch in degrees (positive = look down)")
    ap.add_argument("--head-yaw", type=float, default=P.HEAD_YAW)
    ap.add_argument("--ready-bag", default=None,
                    help="bag whose arm trajectory is replayed (default %s)"
                         % DEFAULT_READY_BAG)
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
    ap.add_argument("--offset", type=float, nargs=3, default=[-0.3, 0.0, 0.0],
                    metavar=("X", "Y", "Z"),
                    help="offset added to every hover target (m)")
    ap.add_argument("--no-look", action="store_true",
                    help="disable the automatic horizontal claw aim in hover")
    ap.add_argument("--elbow", type=float, nargs=3, default=[0.15, -0.25, 0.10],
                    metavar=("X", "Y", "Z"),
                    help="preferred elbow position for hover (base frame, m); "
                         "use `elbow off` / `--elbow` to change")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rospy.init_node("task_sequence", anonymous=True)
    TaskSequence(args).run()


if __name__ == "__main__":
    main()
