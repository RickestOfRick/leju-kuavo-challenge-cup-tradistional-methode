#!/usr/bin/env python3
"""Sign map for all 14 arm joints.

For each joint: give it a small P+D term, set its reference 10 deg away from where
it currently is, wait, and report which way it moved.

    moved toward the target  -> '+'  (sign correct)
    moved away               -> '-'  (sign inverted for this joint)
    barely moved             -> '.'

Everything is additive (WBC keeps holding the other joints), so the arm stays put.

Run with the sim up, task_sequence NOT running:
    python3 sign_map.py
"""

import math
import time

import rospy

from kuavo_msgs.msg import sensorsData
import arm_test_control as AC

ARM_OFF = 13
NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
STEP = 10.0          # deg


def read():
    d = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=5.0)
    return [math.degrees(d.joint_data.joint_q[ARM_OFF + i]) for i in range(14)]


def set_gains(idx):
    z = [0.0] * 14
    kp = list(z); kp[idx] = 6.0
    kd = list(z); kd[idx] = 2.0
    cl = list(z); cl[idx] = 3.0
    rospy.set_param("/arm_pd/replace", False)      # additive: WBC holds the rest
    rospy.set_param("/arm_pd/kp_joints", kp)
    rospy.set_param("/arm_pd/kd_joints", kd)
    rospy.set_param("/arm_pd/ki_joints", z)
    rospy.set_param("/arm_pd/clamp_joints", cl)


def clear():
    for k in ("kp_joints", "kd_joints", "ki_joints", "clamp_joints"):
        rospy.set_param("/arm_pd/" + k, [0.0] * 14)


def main():
    rospy.init_node("sign_map", anonymous=True)
    ar = AC.ArmTestControl()
    ar.setup()
    rospy.sleep(1.0)

    base = read()
    print("base:", " ".join("%.0f" % v for v in base), flush=True)
    result = []
    for idx in range(14):
        set_gains(idx)
        rospy.sleep(0.6)
        target = list(base)
        target[idx] = base[idx] + STEP
        t0 = time.time()
        rate = rospy.Rate(50)
        while time.time() - t0 < 2.0 and not rospy.is_shutdown():
            ar.stream.set(target)
            rate.sleep()
        now = read()
        d = now[idx] - base[idx]
        mark = "+" if d > 2.5 else ("-" if d < -2.5 else ".")
        result.append((NAMES[idx], d, mark))
        print("  %-3s target %+5.1f  moved %+6.1f   %s" % (NAMES[idx], STEP, d, mark), flush=True)
        # back to base
        t0 = time.time()
        while time.time() - t0 < 1.2 and not rospy.is_shutdown():
            ar.stream.set(base)
            rate.sleep()

    clear()
    print("\n= sign map =")
    print("  " + "  ".join("%s%s" % (n, m) for n, _, m in result))
    print("  '+' correct, '-' inverted, '.' no response")


if __name__ == "__main__":
    main()
