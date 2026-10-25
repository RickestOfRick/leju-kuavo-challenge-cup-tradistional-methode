#!/usr/bin/env python3
"""Rotate L3 only, and print the full PID calculation while it happens.

Everything else keeps its normal gains; only L3 is given kp/kd/clamp here (in
ADDITIVE mode, so the WBC still holds the rest of the arm).

    python3 l3_rotate.py            rotate L3 by +20 deg
    python3 l3_rotate.py -20        rotate by -20 deg
    python3 l3_rotate.py 20 8       step 20 deg, kp 8

Prints one row per /arm_pd/debug sample (5 Hz):
    des meas err kp kd ki I kp*e tau_raw tau_full wbc tau_cmd
"""

import math
import sys
import time

import rospy
from std_msgs.msg import Float64MultiArray

import arm_test_control as AC

ARM_OFF = 13
IDX = 2
NF = 13

STEP = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
KP = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
KD = 5.0
CLAMP = 8.0

dbg = {"rows": []}


def align(a, b):
    if abs(a - b) <= 180.0:
        return a, b
    return (a - 360.0, b) if a > b else (a, b - 360.0)


def on_debug(msg):
    d = msg.data
    if len(d) >= 2 + 14 * NF:
        dbg["rows"].append(list(d[2 + IDX * NF: 2 + (IDX + 1) * NF]))
        dbg["rows"] = dbg["rows"][-40:]


def main():
    rospy.init_node("l3_rotate", anonymous=True)
    rospy.Subscriber("/arm_pd/debug", Float64MultiArray, on_debug, queue_size=5)

    # only L3 gets a PD term; additive so nothing else is disturbed
    z = [0.0] * 14
    kp = list(z); kp[IDX] = KP
    kd = list(z); kd[IDX] = KD
    cl = list(z); cl[IDX] = CLAMP
    # PID ONLY for L3; the other 13 joints keep WBC + their normal gains
    rospy.set_param("/arm_pd/replace", False)
    repl = [0.0] * 14
    repl[IDX] = 1.0
    rospy.set_param("/arm_pd/replace_joints", repl)
    rospy.set_param("/arm_pd/kp_joints", kp)
    rospy.set_param("/arm_pd/kd_joints", kd)
    rospy.set_param("/arm_pd/ki_joints", z)
    rospy.set_param("/arm_pd/clamp_joints", cl)
    rospy.sleep(0.8)

    ar = AC.ArmTestControl()
    ar.setup()
    rospy.sleep(0.8)

    with ar.lock:
        base = [math.degrees(v) for v in ar.joint_q]
    target = list(base)
    target[IDX] += STEP
    print("L3 now %.1f -> target %.1f   (kp %.1f kd %.1f clamp %.1f, additive)"
          % (base[IDX], target[IDX], KP, KD, CLAMP), flush=True)

    dbg["rows"] = []
    t0 = time.time()
    rate = rospy.Rate(50)
    while time.time() - t0 < 4.0 and not rospy.is_shutdown():
        ar.stream.set(target)
        rate.sleep()

    # measure where it ended
    with ar.lock:
        after = [math.degrees(v) for v in ar.joint_q]
    print("\nL3 ended %.1f  (moved %+.1f, target was %+.1f)"
          % (after[IDX], after[IDX] - base[IDX], STEP))

    print("\n%3s %7s %7s %7s %5s %5s %5s %7s %8s %8s %8s %8s"
          % ("#", "des", "meas", "err", "kp", "kd", "I", "kp*e", "tau_raw",
             "tau_out", "wbc", "tau_cmd"))
    for i, f in enumerate(dbg["rows"]):
        pterm = f[4] * (f[3] / 57.2958)
        print("%3d %7.1f %7.1f %7.1f %5.1f %5.1f %5.2f %7.3f %8.3f %8.3f %8.3f %8.3f"
              % (i, f[0], f[1], f[3], f[4], f[5], f[8], pterm, f[9], f[10], f[11], f[12]))

    for name in ("kp_joints", "kd_joints", "ki_joints", "clamp_joints"):
        rospy.set_param("/arm_pd/" + name, [0.0] * 14)
    rospy.set_param("/arm_pd/replace_joints", [0.0] * 14)


if __name__ == "__main__":
    main()
