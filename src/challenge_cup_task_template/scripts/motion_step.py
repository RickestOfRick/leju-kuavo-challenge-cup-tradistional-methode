#!/usr/bin/env python3
"""LIVE smoothness monitor: joint change per control tick (10 ms).

    commanded : /kuavo_arm_traj            (degrees, L1..L7, R1..R7, 100 Hz)
    publishes : /motion_step  Float64MultiArray
                  data[0..13] = |delta| of each joint since the previous tick
                  data[14]    = max of the 14  (what you watch)

One tick is 10 ms, so data[14] is the "deg per 10 ms" of the smoothness plot.
Keep it under 5 and no single command step can be a jump.

    bash motion_step.sh            # graph of the max
    bash motion_step.sh 3          # graph of one joint (0..13)
    python3 motion_step.py         # text: max + the joint that moved most

Needs the simulator running and something publishing /kuavo_arm_traj
(task_sequence.py, or run_act_policy.py).
"""

import sys
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "R1", "R2", "R3", "R4", "R5", "R6", "R7"]

QUIET = "--plot" in sys.argv
last = None
latest = None
pub = None


def on_traj(msg):
    global last, latest
    pos = np.asarray(msg.position[:14], dtype=float)
    if pos.size < 14:
        return
    if last is not None:
        d = np.abs(pos - last)
        latest = d
        pub.publish(Float64MultiArray(data=list(d) + [float(d.max())]))
    last = pos


def main():
    global pub
    rospy.init_node("motion_step", anonymous=True)
    pub = rospy.Publisher("/motion_step", Float64MultiArray, queue_size=10)
    rospy.Subscriber("/kuavo_arm_traj", JointState, on_traj, queue_size=1)
    rate = rospy.Rate(20)

    if not QUIET:
        print("%-8s %-8s %s" % ("max", "joint", "per-joint deg/10ms"))
        print("(limit: 5.0)")

    t_next = time.time() + 1.0
    while not rospy.is_shutdown():
        if latest is not None and time.time() >= t_next:
            t_next += 1.0
            if not QUIET:
                j = int(np.argmax(latest))
                print("%-8.2f %-8s %s" % (
                    latest.max(), NAMES[j],
                    " ".join("%5.2f" % v for v in latest)))
                sys.stdout.flush()
        rate.sleep()


if __name__ == "__main__":
    main()
