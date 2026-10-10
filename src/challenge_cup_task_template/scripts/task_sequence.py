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
    h / q                help / quit

Non-interactive:

    python3 task_sequence.py --auto
    python3 task_sequence.py --auto --ready-bag /root/kuavo_ws/go.bag

`--auto` runs: TILT_HEAD -> ARM_READY (go-bag replay) -> CLOUD -> RANSAC
-> CLUSTER -> DONE.
"""

import argparse
import os
import time

import rospy

import perception as P

DEFAULT_READY_BAG = "/root/kuavo_ws/go.bag"


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

    # ── detection steps ─────────────────────────────────────────────
    def do_head(self):
        P.set_head(self.head_yaw, self.head_pitch)

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
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rospy.init_node("task_sequence", anonymous=True)
    TaskSequence(args).run()


if __name__ == "__main__":
    main()
