#!/usr/bin/env python3
"""Detect sign-inverted arm joints.

For each of the 14 arm joints: command +STEP degrees on /kuavo_arm_traj, wait,
read the measured change from /sensors_data_raw, then command back.

    same sign   -> no inversion needed
    opposite    -> this joint's sensor/command convention is inverted
                   (needs pdinv / negative gains)

Run with the sim idle at the ready pose and the PD OFF first:
    pd off
    python3 joint_sign_test.py
"""

import math
import time

import rospy
from sensor_msgs.msg import JointState

from kuavo_msgs.msg import sensorsData, armTargetPoses

ARM_OFFSET = 13                 # arms start at index 13 in joint_q
NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
STEP = 8.0                      # degrees
SETTLE = 1.5                    # seconds to let the arm move


def stream(pub, msg, seconds, hz=100.0):
    """Publish at 100 Hz for `seconds` — a single message does not move the arm."""
    rate = rospy.Rate(hz)
    n = int(seconds * hz)
    for _ in range(n):
        if rospy.is_shutdown():
            return
        pub.publish(msg)
        rate.sleep()


def read_arm():
    msg = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=5.0)
    q = msg.joint_data.joint_q
    return [math.degrees(q[ARM_OFFSET + i]) for i in range(14)]


def main():
    rospy.init_node("joint_sign_test", anonymous=True)
    pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    tpub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    rospy.sleep(1.0)

    base = read_arm()
    print("base:", " ".join("%6.1f" % v for v in base), flush=True)
    print("%-4s %8s %8s  %s" % ("jnt", "cmd", "meas", "verdict"), flush=True)

    inverted = []
    for i in range(14):
        cmd = list(base)
        cmd[i] += STEP
        msg = JointState()
        msg.name = ["arm_joint_%d" % (k + 1) for k in range(14)]
        msg.position = cmd
        tp = armTargetPoses()
        tp.times = [SETTLE]
        tp.values = list(cmd)
        tpub.publish(tp)                  # the MPC target, same as move_joints()
        stream(pub, msg, SETTLE)          # 100 Hz, like the task does
        moved = read_arm()

        # back to base
        msg2 = JointState()
        msg2.name = msg.name
        msg2.position = list(base)
        tp2 = armTargetPoses()
        tp2.times = [SETTLE]
        tp2.values = list(base)
        tpub.publish(tp2)
        stream(pub, msg2, SETTLE)

        d_cmd = STEP
        d_meas = moved[i] - base[i]
        ok = d_meas >= 0.0
        verdict = "same" if ok else "INVERTED"
        if not ok:
            inverted.append(NAMES[i])
        print("%-4s %+8.1f %+8.1f  %s" % (NAMES[i], d_cmd, d_meas, verdict), flush=True)

    print("\ninverted joints:", ", ".join(inverted) if inverted else "none", flush=True)


if __name__ == "__main__":
    main()
