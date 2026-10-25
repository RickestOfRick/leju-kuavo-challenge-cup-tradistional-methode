#!/usr/bin/env python3
"""Isolate the torque path on ONE joint.

Everything else is disabled; only L3 (index 2) gets a P-term, and the reference
is set +20 deg away from the current L3. Then we watch:

  * L3 moves TOWARD the target   -> sign correct, torque path correct
  * L3 moves AWAY from the target-> sign inverted for this joint
  * nothing moves                -> torque for index 2 is not reaching L3
  * a DIFFERENT joint moves      -> index mismatch

Run with the sim up (task_sequence NOT running):
    python3 torque_test.py
"""

import math
import time

import rospy
from sensor_msgs.msg import JointState

from kuavo_msgs.msg import sensorsData
import arm_test_control as AC

ARM_OFF = 13
IDX = 2            # L3
STEP = 20.0        # deg away from current


def read_l3():
    d = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=5.0)
    return [math.degrees(d.joint_data.joint_q[ARM_OFF + i]) for i in range(14)]


def main():
    rospy.init_node("torque_test", anonymous=True)

    # 1) everything off, only L3 gets a P term
    z = [0.0] * 14
    kp = list(z); kp[IDX] = 20.0
    kd = list(z); kd[IDX] = 5.0        # damped, so it cannot overshoot
    ki = list(z)
    cl = list(z); cl[IDX] = 3.0
    # ADDITIVE: the WBC still holds every other joint (otherwise the arm collapses)
    rospy.set_param("/arm_pd/replace", False)
    rospy.set_param("/arm_pd/kp_joints", kp)
    rospy.set_param("/arm_pd/kd_joints", kd)
    rospy.set_param("/arm_pd/ki_joints", ki)
    rospy.set_param("/arm_pd/clamp_joints", cl)
    rospy.sleep(1.0)

    ar = AC.ArmTestControl()
    ar.setup()
    rospy.sleep(1.0)

    base = read_l3()
    target = list(base)
    target[IDX] = base[IDX] + STEP
    print("L3 now = %+.1f, reference will be %+.1f, only L3 has kp=20 kd=5 clamp=3 (additive)"
          % (base[IDX], target[IDX]), flush=True)

    # 2) stream the target for 4 s (this also feeds arm_joint_trajectory_.pos)
    t0 = time.time()
    rate = rospy.Rate(50)
    while time.time() - t0 < 4.0 and not rospy.is_shutdown():
        ar.stream.set(target)
        rate.sleep()

    after = read_l3()
    d = after[IDX] - base[IDX]
    print("\nL3 moved %+.1f deg  (target was %+.1f away)" % (d, STEP))
    if d > 5.0:
        print("=> TOWARD the target: sign and torque path are CORRECT")
    elif d < -5.0:
        print("=> AWAY from the target: SIGN INVERTED for this joint")
    else:
        print("=> no motion: the torque for index 2 is NOT reaching L3")

    moved = ["%d:%+.1f" % (i, after[i] - base[i])
             for i in range(14) if abs(after[i] - base[i]) > 1.0 and i != IDX]
    if moved:
        print("   other joints that moved:", ", ".join(moved))

    # 3) leave everything off
    for name in ("kp_joints", "kd_joints", "ki_joints", "clamp_joints"):
        rospy.set_param("/arm_pd/" + name, [0.0] * 14)


if __name__ == "__main__":
    main()
