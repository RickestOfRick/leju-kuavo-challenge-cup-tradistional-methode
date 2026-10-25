#!/usr/bin/env python3
"""Live error graph for ALL 14 arm joints (commanded vs measured).

    commanded : /kuavo_arm_traj            (degrees, order L1..L7, R1..R7)
    measured  : /sensors_data_raw.joint_q  (radians,  arms start at index 13)

Publishes /all_err  (14 values, degrees) and re-publishes at 10 Hz so rqt_plot
shows continuous curves.

    bash all_plot.sh          # graph of all 14
    python3 all_err.py        # text table
"""

import math
import sys

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from kuavo_msgs.msg import sensorsData

ARM_OFF = 13
NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "R1", "R2", "R3", "R4", "R5", "R6", "R7"]

state = {"des": None, "act": None}


def on_traj(msg):
    if len(msg.position) >= 14:
        state["des"] = [float(v) for v in msg.position[:14]]


def on_sensor(msg):
    q = msg.joint_data.joint_q
    if len(q) >= ARM_OFF + 14:
        state["act"] = [math.degrees(q[ARM_OFF + i]) for i in range(14)]


def main():
    rospy.init_node("all_err", anonymous=True)
    rospy.Subscriber("/kuavo_arm_traj", JointState, on_traj, queue_size=1)
    rospy.Subscriber("/sensors_data_raw", sensorsData, on_sensor, queue_size=1)
    pub = rospy.Publisher("/all_err", Float64MultiArray, queue_size=1)

    rate = rospy.Rate(10.0)
    last_print = 0.0
    while not rospy.is_shutdown():
        if state["des"] is not None and state["act"] is not None:
            errs = [d - a for d, a in zip(state["des"], state["act"])]
            m = Float64MultiArray()
            m.data = errs
            pub.publish(m)

            now = rospy.get_time()
            if "--plot" not in sys.argv and now - last_print > 0.5:
                last_print = now
                line = "  ".join("%s %+5.1f" % (NAMES[i], errs[i]) for i in range(14))
                print(line + "   | max %+5.1f" % max(errs, key=abs), flush=True)
        rate.sleep()


if __name__ == "__main__":
    main()
