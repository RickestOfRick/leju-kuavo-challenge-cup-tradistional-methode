#!/usr/bin/env python3
"""Compare the controller's measured arm angles (jointPosWBC_) with the raw sensor.

If the two columns disagree, the PID is computing its error against the WRONG value.

    python3 cmp_meas.py
"""

import math

import rospy
from std_msgs.msg import Float64MultiArray

from kuavo_msgs.msg import sensorsData

NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
NF = 13


def main():
    rospy.init_node("cmp_meas", anonymous=True)
    d = rospy.wait_for_message("/arm_pd/debug", Float64MultiArray, timeout=5).data
    s = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=5)
    q = [math.degrees(s.joint_data.joint_q[13 + i]) for i in range(14)]

    print("%-4s %10s %10s %8s" % ("jnt", "controller", "sensor", "diff"))
    for i in range(14):
        f = d[2 + i * NF: 2 + (i + 1) * NF]
        print("%-4s %10.2f %10.2f %8.2f" % (NAMES[i], f[1], q[i], f[1] - q[i]))

    print("\nif a column is a shifted copy of the other, the index mapping is off:")
    print("  controller:", " ".join("%.0f" % d[2 + i * NF + 1] for i in range(14)))
    print("  sensor    :", " ".join("%.0f" % v for v in q))


if __name__ == "__main__":
    main()
