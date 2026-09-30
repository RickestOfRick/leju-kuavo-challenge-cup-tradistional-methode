#!/usr/bin/env python3
"""task_sequence.py — Task 1 sequence / state machine.

Default is INTERACTIVE so you can test one step at a time:

    python3 task_sequence.py

At the `task>` prompt type:

    head                 tilt the head (uses the current pitch/yaw)
    cloud                capture RGB-D -> point cloud  (step 1)
    ransac               fit + remove the table plane  (step 2)
    cluster              cluster + PCA per object      (step 3)
    weigh                find the green weighing marker (colour)
    all                  run cloud -> ransac -> cluster
    live [SEC]           repeat cloud->ransac->cluster every SEC s (default 1.0)
    show [on|off]        toggle the overlay window for the next step
    pitch V | yaw V      head angles in degrees (pitch>0 looks down)
    h                    help
    q                    quit

The overlay works for `cloud` (valid-depth pixels) and `cluster` (object
contours), same as `perception.py --show`.

Non-interactive (runs TILT_HEAD -> ... -> DONE once):

    python3 task_sequence.py --auto
    python3 task_sequence.py --auto --show
"""

import argparse
import time

import rospy

import perception as P


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

    # ── steps ───────────────────────────────────────────────────────
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
                elif cmd == "live":
                    self.do_live(float(a[0]) if a else 1.0)
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
    ap = argparse.ArgumentParser(description="Task 1 sequence / state machine")
    ap.add_argument("--auto", action="store_true",
                    help="run the whole sequence once instead of interactive")
    ap.add_argument("--live", type=float, default=0.0,
                    help="tilt head then repeat perception every N seconds")
    ap.add_argument("--head-pitch", type=float, default=P.HEAD_PITCH,
                    help="head pitch in degrees (positive = look down)")
    ap.add_argument("--head-yaw", type=float, default=P.HEAD_YAW)
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
