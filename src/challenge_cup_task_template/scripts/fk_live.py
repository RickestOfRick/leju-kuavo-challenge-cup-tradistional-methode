#!/usr/bin/env python3
"""LIVE end-effector error (FK of commanded vs FK of measured), 10 Hz.

    commanded : /kuavo_arm_traj            degrees, L1..L7, R1..R7
    measured  : /sensors_data_raw.joint_q   radians, arms start at index 13
    publishes : /fk_err      Float64MultiArray
                  data[0] left hand position error   [mm]
                  data[1] right hand position error  [mm]
                  data[2] left hand rotation error   [deg]
                  data[3] right hand rotation error  [deg]
                  data[4] max of the two position errors [mm]
              /fk_err_xyz  Float64MultiArray  RIGHT hand only, signed
                  data[0..2] dx, dy, dz = commanded - measured  [mm]
                  data[3]    norm of the same vector [mm]
              /ee_track    Float64MultiArray  RIGHT hand position [mm]
                  data[0..2] desired  x, y, z = FK(commanded joints)
                  data[3..5] measured x, y, z = FK(measured joints)

Needs the simulator / arm-control node (FK service /ik/fk_srv). Commands nothing.

    bash fk_live.sh          # graph: both hands in mm
    bash fk_live.sh all      # + rotation errors
    python3 fk_live.py       # text table instead of a graph
"""

import math
import sys

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from kuavo_msgs.msg import sensorsData
from kuavo_msgs.srv import fkSrv
from fk_units import calibrate

FK_SRV = "/ik/fk_srv"
ARM_OFF = 13
QUIET = "--plot" in sys.argv
JUST_CALIBRATE = "--calibrate" in sys.argv
OUT_SCALE = 1.0                       # --scale=N multiplies what is published
for _a in sys.argv[1:]:
    if _a.startswith("--scale="):
        try:
            OUT_SCALE = float(_a.split("=", 1)[1])
        except ValueError:
            pass

cmd = None
meas = None
fk = None
pub = None
SCALE = 1.0          # radians -> whatever unit the FK service wants (set at startup)


def on_traj(msg):
    global cmd
    cmd = np.radians(np.asarray(msg.position[:14], dtype=float))


def on_state(msg):
    global meas
    q = msg.joint_data.joint_q
    if len(q) >= 27:
        meas = np.asarray(q[ARM_OFF:ARM_OFF + 14], dtype=float)   # already radians


def quat_deg(a, b):
    d = min(1.0, abs(float(np.dot(a, b))))
    return math.degrees(2.0 * math.acos(d))


def hands(q):
    res = fk(list(map(float, np.asarray(q) * SCALE)))
    if not res.success:
        return None
    out = {}
    for side in ("left", "right"):
        p = getattr(res.hand_poses, side + "_pose")
        out[side] = (np.array(p.pos_xyz, float), np.array(p.quat_xyzw, float))
    return out


def main():
    global fk, pub
    rospy.init_node("fk_live", anonymous=True, log_level=rospy.ERROR)
    rospy.wait_for_service(FK_SRV, timeout=10.0)
    fk = rospy.ServiceProxy(FK_SRV, fkSrv)
    global SCALE
    SCALE = calibrate(fk, log=print)
    if JUST_CALIBRATE:
        print("scale %.3f -> %s" % (SCALE, "degrees" if SCALE > 2 else "radians"))
        return
    pub = rospy.Publisher("/fk_err", Float64MultiArray, queue_size=10)
    pub_xyz = rospy.Publisher("/fk_err_xyz", Float64MultiArray, queue_size=10)
    pub_track = rospy.Publisher("/ee_track", Float64MultiArray, queue_size=10)
    rospy.Subscriber("/kuavo_arm_traj", JointState, on_traj, queue_size=1)
    rospy.Subscriber("/sensors_data_raw", sensorsData, on_state, queue_size=1)

    if OUT_SCALE != 1.0:
        print("published /ee_track x%.0f  (axis is NOT plain mm)" % OUT_SCALE)
    if not QUIET:
        print("FK scale: %.3f (%s)" % (SCALE, "degrees" if SCALE > 2 else "radians"))
        print("%-8s %-8s %-8s %s" % ("L[mm]", "R[mm]", "max[mm]", "rotation L/R [deg]"))
        print("(a few mm is normal; tens of mm only during fast moves)")

    rate = rospy.Rate(10)
    tick = 0
    while not rospy.is_shutdown():
        if cmd is None or meas is None:
            rate.sleep()
            continue
        try:
            a = hands(cmd)
            b = hands(meas)
        except Exception:
            rate.sleep()
            continue
        if a is None or b is None:
            rate.sleep()
            continue
        mm = {s: 1000.0 * float(np.linalg.norm(a[s][0] - b[s][0])) for s in a}
        dg = {s: quat_deg(a[s][1], b[s][1]) for s in a}
        pub.publish(Float64MultiArray(data=[mm["left"], mm["right"],
                                            dg["left"], dg["right"],
                                            max(mm["left"], mm["right"])]))
        pc, pm = a["right"][0], b["right"][0]              # desired / measured [m]
        pub_track.publish(Float64MultiArray(data=[float(v) * 1000.0 * OUT_SCALE
                                                  for v in (pc[0], pc[1], pc[2],
                                                            pm[0], pm[1], pm[2])]))
        dp = (pc - pm) * 1000.0                           # commanded - measured
        pub_xyz.publish(Float64MultiArray(data=[float(dp[0]), float(dp[1]),
                                                float(dp[2]),
                                                float(np.linalg.norm(dp))]))
        tick += 1
        if not QUIET and tick % 10 == 0:
            print("%-8.1f %-8.1f %-8.1f %.1f / %.1f"
                  % (mm["left"], mm["right"], max(mm["left"], mm["right"]),
                     dg["left"], dg["right"]))
        rate.sleep()


if __name__ == "__main__":
    main()
