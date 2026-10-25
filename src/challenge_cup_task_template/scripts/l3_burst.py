#!/usr/bin/env python3
"""Print the L3 PID calculation for ~10 control samples, then exit.

Reads /arm_pd/debug (published at 5 Hz by the controller) and prints, for L3 only:

    des  meas  err  kp  kd  ki  qd  I  tau_raw  tau_clamped  wbc  tau_cmd

with tau_raw computed as kp*err - kd*qd + I, so you can check each term against
the total.

    python3 l3_burst.py        10 samples
    python3 l3_burst.py 20     20 samples
"""

import sys

import rospy
from std_msgs.msg import Float64MultiArray

NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
NF = 13
IDX = 2          # L3


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rospy.init_node("l3_burst", anonymous=True)

    got = []

    def cb(msg):
        d = msg.data
        if len(d) >= 2 + 14 * NF and len(got) < n:
            got.append((d[1], d[0], list(d[2 + IDX * NF: 2 + (IDX + 1) * NF])))

    rospy.Subscriber("/arm_pd/debug", Float64MultiArray, cb, queue_size=5)
    print("collecting %d samples of L3 ..." % n)
    while len(got) < n and not rospy.is_shutdown():
        rospy.sleep(0.05)

    if not got:
        print("no data")
        return

    print("\nreplace=%d  active=%d\n" % (got[-1][0], got[-1][1]))
    print("%3s %7s %7s %7s %5s %5s %5s %8s %7s %9s %9s %8s %9s"
          % ("#", "des", "meas", "err", "kp", "kd", "ki", "qd", "I",
             "kp*e", "tau_raw", "clamped", "wbc"))
    for i, (rep, act, f) in enumerate(got):
        kp, kd, ki = f[4], f[5], f[6]
        err, qd, I = f[3], f[2], f[8]
        pterm = kp * (err / 57.2958)
        print("%3d %7.1f %7.1f %7.1f %5.1f %5.1f %5.2f %8.1f %7.3f %9.3f %9.3f %8.3f %9.3f"
              % (i, f[0], f[1], err, kp, kd, ki, qd, I,
                 pterm, f[9], f[10], f[11]))


if __name__ == "__main__":
    main()
