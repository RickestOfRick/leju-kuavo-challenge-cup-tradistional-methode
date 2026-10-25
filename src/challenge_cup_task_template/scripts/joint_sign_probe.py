#!/usr/bin/env python3
"""Sign test using the task's OWN arm controller (arm_test_control), so it uses the
same topics, names and interpolation path that the running task uses.

For each of the 14 arm joints: move +STEP degrees, wait, read the measured change,
move back. Prints a table with a verdict per joint.

    python3 joint_sign_probe.py [step_deg]
"""

import math
import sys
import time

import rospy

import arm_test_control as AC

NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
STEP = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
MOVE_T = 1.5
SETTLE = 1.0


def read_deg(ar):
    with ar.lock:
        q = list(ar.joint_q)
    return [math.degrees(v) for v in q]


def main():
    rospy.init_node("joint_sign_probe", anonymous=True)
    ar = AC.ArmTestControl()
    rospy.sleep(1.0)

    base = read_deg(ar)
    print("base:", " ".join("%6.1f" % v for v in base), flush=True)
    print("%-4s %8s %8s  %s" % ("jnt", "cmd", "meas", "verdict"), flush=True)

    inverted = []
    for i in range(14):
        cmd = list(base)
        cmd[i] += STEP
        ar.move_joints(cmd, duration=MOVE_T)
        rospy.sleep(SETTLE)
        moved = read_deg(ar)

        ar.move_joints(list(base), duration=MOVE_T)
        rospy.sleep(SETTLE)

        d = moved[i] - base[i]
        ok = d >= 0.0
        if not ok:
            inverted.append(NAMES[i])
        print("%-4s %+8.1f %+8.1f  %s" % (NAMES[i], STEP, d, "same" if ok else "INVERTED"),
              flush=True)

    print("\ninverted joints:", ", ".join(inverted) if inverted else "none", flush=True)


if __name__ == "__main__":
    main()
