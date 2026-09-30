#!/usr/bin/env python3
"""
gripper_test.py — standalone gripper close / grip-detection test.

IDEA
----
Command the claws to close all the way (target closure = 100 %, i.e. gap = 0).
Then watch the MEASURED closure:
  * if the MAXIMUM closure reaches ~100 % -> jaws met, nothing between them
    -> EMPTY / missed
  * if it stalls below -> something is between the jaws -> HOLDING
This is exactly the "did I actually grab it?" check.

Run inside the container, with a scene already running:

    source /root/kuavo_ws/devel/setup.zsh
    python3 /root/kuavo_ws/gripper_test.py --arm right

Commands at the `grip>` prompt:

    c               run one test: close -> measure -> HOLDING / EMPTY
                    (press `o` first if you want to start from open)
    o               open the active gripper (0 %)
    k               close the active gripper (100 %)
    p               print the current measured state, once
    w [SEC]         watch live for SEC seconds (default 5; 0 = until Ctrl-C)
    a right|left    switch the active gripper
    v VEL           set closing velocity (0-100, default 80)
    e EFFORT        set effort in A (default 1.0)
    t PCT           set the EMPTY/HOLDING threshold in % (default 95)
    h               help
    q               quit

The active gripper moves; the other one is left where it is.

What you read:
    closure%   measured closing, 0 = fully open, 100 = fully closed
    gap%       100 - closure%  ("how far from closed"; ~0 means jaws met)
    driver     raw /gripper/state joint angle in rad (0.0 open .. 0.8 closed)
    state      kMoving / kReached from /leju_claw_state
"""

import argparse
import threading
import time

import rospy
from kuavo_msgs.msg import lejuClawState
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest
from sensor_msgs.msg import JointState

# ───────────────────────────── config ─────────────────────────────
CLAW_SRV = "/control_robot_leju_claw"
CLAW_NAMES = ["left_claw", "right_claw"]
SIDE_INDEX = {"left": 0, "right": 1}
GRIPPER_JOINT = {"left": "left_gripper_joint", "right": "right_gripper_joint"}

DRIVER_MAX_RAD = 0.8          # sim driver joint range: 0 open .. 0.8 closed
DEFAULT_VELOCITY = 80.0       # 0-100
DEFAULT_EFFORT = 1.0          # A
DEFAULT_EMPTY_PCT = 95.0      # max closure >= this -> jaws met -> EMPTY
TEST_TIMEOUT = 4.0            # seconds to watch the closing
POLL_HZ = 50.0
SERVICE_TIMEOUT = 30.0

STATE_NAME = {lejuClawState.kError: "kError", lejuClawState.kUnknown: "kUnknown",
              lejuClawState.kMoving: "kMoving", lejuClawState.kReached: "kReached",
              lejuClawState.kGrabbed: "kGrabbed"}
# ──────────────────────────────────────────────────────────────────


def log(msg):
    print(msg, flush=True)


class GripperTest(object):
    def __init__(self, arm="right"):
        self.arm = arm if arm in SIDE_INDEX else "right"
        self.velocity = DEFAULT_VELOCITY
        self.effort = DEFAULT_EFFORT
        self.empty_pct = DEFAULT_EMPTY_PCT
        self.lock = threading.Lock()
        self.pct = [None, None]     # measured percent per side
        self.rad = [None, None]     # measured driver rad per side
        self.state = [lejuClawState.kUnknown] * 2
        self.claw_srv = None

        rospy.Subscriber("/leju_claw_state", lejuClawState, self._on_claw, queue_size=1)
        rospy.Subscriber("/gripper/state", JointState, self._on_gripper, queue_size=1)

    # ── callbacks
    def _on_claw(self, msg):
        with self.lock:
            for i in range(min(2, len(msg.data.position))):
                self.pct[i] = float(msg.data.position[i])
            if len(getattr(msg, "state", [])) >= 2:
                self.state = [int(msg.state[0]), int(msg.state[1])]

    def _on_gripper(self, msg):
        with self.lock:
            for name, pos in zip(msg.name, msg.position):
                for side, joint in GRIPPER_JOINT.items():
                    if name == joint:
                        self.rad[SIDE_INDEX[side]] = float(pos)

    # ── state
    def read(self):
        i = SIDE_INDEX[self.arm]
        with self.lock:
            return self.pct[i], self.rad[i], self.state[i]

    def print_state(self):
        pct, rad, st = self.read()
        p = "?" if pct is None else "%5.1f%%" % pct
        r = "?" if rad is None else "%6.4f rad" % rad
        log("  %-5s closure=%s  gap=%s  driver=%s  state=%s"
            % (self.arm, p,
               "?" if pct is None else "%5.1f%%" % (100.0 - pct),
               r, STATE_NAME.get(st, str(st))))

    # ── command
    def command(self, pct):
        """Set the ACTIVE claw target; leave the other claw at its last value."""
        with self.lock:
            left = self.pct[0] if self.pct[0] is not None else 0.0
            right = self.pct[1] if self.pct[1] is not None else 0.0
        pos = [left, right]
        pos[SIDE_INDEX[self.arm]] = max(0.0, min(100.0, float(pct)))

        req = controlLejuClawRequest()
        req.data.name = CLAW_NAMES
        req.data.position = pos
        req.data.velocity = [self.velocity, self.velocity]
        req.data.effort = [self.effort, self.effort]
        try:
            res = self.claw_srv(req)
        except rospy.ServiceException as e:
            log("  claw service failed: %s" % e)
            return False
        if not res.success:
            log("  claw service rejected: %s" % res.message)
            return False
        return True

    def open(self):
        return self.command(0.0)

    def close(self):
        return self.command(100.0)

    # ── waits
    def setup(self):
        rospy.wait_for_service(CLAW_SRV, timeout=SERVICE_TIMEOUT)
        self.claw_srv = rospy.ServiceProxy(CLAW_SRV, controlLejuClaw)
        log("waiting for /leju_claw_state and /gripper/state ...")
        t0 = time.time()
        while (self.read()[0] is None or self.read()[1] is None) \
                and not rospy.is_shutdown():
            if time.time() - t0 > SERVICE_TIMEOUT:
                log("  WARNING: no gripper state yet — is a scene running?")
                break
            time.sleep(0.1)
        log("ready.  active=%s  velocity=%.0f  effort=%.1f  empty_threshold=%.0f%%"
            % (self.arm, self.velocity, self.effort, self.empty_pct))
        self.print_state()

    # ── the test
    def close_test(self):
        start_pct, _, _ = self.read()
        self.empty_pct = float(self.empty_pct)
        log("")
        log("[1/2] closing %s gripper (target closure = 100%%, gap = 0) ..." % self.arm)
        log("      started at %s"
            % ("?" if start_pct is None else "%.1f%%" % start_pct))
        self.close()
        t0 = time.time()
        last_resend = t0
        peak = 0.0
        peak_rad = None
        peak_state = lejuClawState.kUnknown
        # Run for the whole window; only stop early on a genuine "reached".
        # A mid-stroke pause must NOT be mistaken for a blockage, so we classify
        # on the MAXIMUM closure achieved, not on the first plateau.
        while not rospy.is_shutdown() and time.time() - t0 < TEST_TIMEOUT:
            pct, rad, st = self.read()
            if pct is not None and pct > peak:
                peak, peak_rad, peak_state = pct, rad, st
            if pct is not None:
                # "keep trying to close" while it is not fully closed
                if time.time() - last_resend > 0.5:
                    self.close()
                    last_resend = time.time()
                if st == lejuClawState.kReached and pct >= self.empty_pct:
                    break
            time.sleep(1.0 / POLL_HZ)

        if peak <= 0.0:
            log("[2/2] no measurement — gripper state unavailable")
            return
        gap = 100.0 - peak
        log("[2/2] result after %.1fs" % (time.time() - t0))
        log("      start=%s  max closure=%.1f%%  gap=%.1f%%  driver=%.4f rad  state=%s"
            % ("?" if start_pct is None else "%.1f%%" % start_pct,
               peak, gap,
               -1.0 if peak_rad is None else peak_rad,
               STATE_NAME.get(peak_state, str(peak_state))))
        if peak >= self.empty_pct:
            log("      ===> EMPTY  (jaws met: closure %.1f%% >= %.0f%%)"
                % (peak, self.empty_pct))
            log("           nothing held, or the object was dropped / missed.")
        else:
            log("      ===> HOLDING  (stalled at closure %.1f%% < %.0f%%, gap %.1f%%)"
                % (peak, self.empty_pct, gap))
            log("           something is between the jaws.")
        log("")

    def watch(self, seconds):
        log("  watching %s (%.0fs) — Ctrl-C to stop ..." % (self.arm, seconds))
        t0 = time.time()
        try:
            while not rospy.is_shutdown() and (seconds <= 0 or time.time() - t0 < seconds):
                pct, rad, st = self.read()
                if pct is None:
                    log("  (no state yet)")
                else:
                    verdict = "HOLDING" if pct < self.empty_pct else "EMPTY"
                    log("  closure=%5.1f%%  gap=%5.1f%%  driver=%6.4f rad  %-9s  %s"
                        % (pct, 100.0 - pct, -1.0 if rad is None else rad,
                           STATE_NAME.get(st, str(st)), verdict))
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass

    # ── REPL
    def repl(self):
        log("")
        log("type a command (h = help, q = quit)")
        while not rospy.is_shutdown():
            try:
                line = input("grip> ").strip()
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
                    self.print_state()
                elif cmd in ("c", "test"):
                    self.close_test()
                elif cmd in ("o", "open"):
                    self.open()
                    log("  opened %s" % self.arm)
                elif cmd in ("k", "close"):
                    self.close()
                    log("  closed %s" % self.arm)
                elif cmd == "w":
                    self.watch(float(a[0]) if a else 5.0)
                elif cmd == "a":
                    if a and a[0].lower() in SIDE_INDEX:
                        self.arm = a[0].lower()
                        log("  active gripper = %s" % self.arm)
                    else:
                        log("  usage: a right | a left")
                elif cmd == "v":
                    self.velocity = max(0.0, min(100.0, float(a[0])))
                    log("  velocity = %.0f" % self.velocity)
                elif cmd == "e":
                    self.effort = float(a[0])
                    log("  effort = %.1f A" % self.effort)
                elif cmd == "t":
                    self.empty_pct = float(a[0])
                    log("  EMPTY/HOLDING threshold: closure >= %.0f%% is EMPTY"
                        % self.empty_pct)
                else:
                    log("  unknown command: %s   (h = help)" % cmd)
            except (ValueError, IndexError):
                log("  bad arguments.  h = help")
            except Exception as e:
                log("  error: %s" % e)


def main():
    parser = argparse.ArgumentParser(description="gripper close / grip-detection test")
    parser.add_argument("--arm", choices=sorted(SIDE_INDEX), default="right")
    args = parser.parse_args()

    rospy.init_node("gripper_test", anonymous=True)
    t = GripperTest(arm=args.arm)
    try:
        t.setup()
        t.repl()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as e:
        rospy.logerr("gripper_test failed: %s", e)
    log("bye")


if __name__ == "__main__":
    main()
