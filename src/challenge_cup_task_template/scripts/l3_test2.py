#!/usr/bin/env python3
"""L3 only: read it (both sources), rotate it, read it again — with a live trace.

The direction is chosen automatically so the target stays inside the joint range:
  L3 > 50  -> go negative      L3 < -50 -> go positive      otherwise -> +20

    python3 l3_test2.py
"""

import math
import time

import rospy
from std_msgs.msg import Float64MultiArray

from kuavo_msgs.msg import sensorsData, armTargetPoses
import arm_test_control as AC

IDX = 2
NF = 13
STEP = 20.0

dbg = {"v": None}
sensor = {"q": None}


def on_dbg(msg):
    d = msg.data
    if len(d) >= 2 + 14 * NF:
        dbg["v"] = list(d[2 + IDX * NF: 2 + (IDX + 1) * NF])


def on_sensor(msg):
    sensor["q"] = math.degrees(msg.joint_data.joint_q[13 + IDX])


def both():
    return (dbg["v"][1] if dbg["v"] else float("nan"),
            sensor["q"] if sensor["q"] is not None else float("nan"))


def main():
    rospy.init_node("l3_test2", anonymous=True)
    rospy.Subscriber("/arm_pd/debug", Float64MultiArray, on_dbg, queue_size=5)
    rospy.Subscriber("/sensors_data_raw", sensorsData, on_sensor, queue_size=5)

    # L3 only, additive (everything else keeps its normal gains)
    z = [0.0] * 14
    kp = list(z); kp[IDX] = 20.0
    kd = list(z); kd[IDX] = 5.0
    cl = list(z); cl[IDX] = 8.0
    rospy.set_param("/arm_pd/replace", False)
    rospy.set_param("/arm_pd/replace_joints", [0.0] * 14)
    rospy.set_param("/arm_pd/kp_joints", kp)
    rospy.set_param("/arm_pd/kd_joints", kd)
    rospy.set_param("/arm_pd/ki_joints", z)
    rospy.set_param("/arm_pd/clamp_joints", cl)
    rospy.sleep(1.0)

    tpub = rospy.Publisher("/humanoid_mpc_target_arm", armTargetPoses, queue_size=5)
    ar = AC.ArmTestControl()
    ar.setup()
    rospy.sleep(1.0)

    with ar.lock:
        base_arr = [math.degrees(v) for v in ar.joint_q]
    base = base_arr[IDX]

    step = STEP if -50.0 <= base <= 50.0 else (-STEP if base > 0 else STEP)
    target_val = base + step
    target = list(base_arr)
    target[IDX] = target_val

    c0, s0 = both()
    print("start: controller %.2f  sensor %.2f   step %+0.1f -> target %.1f"
          % (c0, s0, step, target_val), flush=True)

    tp = armTargetPoses()
    tp.times = [0.20]
    tp.values = list(target)
    tpub.publish(tp)          # feed the MPC target as well (silences the warning)

    print("\n%5s %12s %12s" % ("t[s]", "controller", "sensor"))
    t0 = time.time()
    rate = rospy.Rate(20)
    next_print = 0.0
    while time.time() - t0 < 5.0 and not rospy.is_shutdown():
        ar.stream.set(target)
        t = time.time() - t0
        if t >= next_print:
            c, s = both()
            print("%5.2f %12.2f %12.2f" % (t, c, s), flush=True)
            next_print += 0.25
        rate.sleep()

    c1, s1 = both()
    print("\nend:   controller %.2f  sensor %.2f   (moved %+.2f / %+.2f, target was %+.1f)"
          % (c1, s1, c1 - c0, s1 - s0, step))

    for name in ("kp_joints", "kd_joints", "ki_joints", "clamp_joints"):
        rospy.set_param("/arm_pd/" + name, [0.0] * 14)


if __name__ == "__main__":
    main()
