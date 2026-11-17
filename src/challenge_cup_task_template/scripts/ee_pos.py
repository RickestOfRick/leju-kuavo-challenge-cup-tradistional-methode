#!/usr/bin/env python3
"""Live RIGHT-hand position: measured and desired, in millimetres.

    measured = FK(/sensors_data_raw joints)   -> /ee_pos     [x, y, z] mm
    desired  = FK(/kuavo_arm_traj  joints)    -> /ee_target  [x, y, z] mm

    python3 ee_pos.py            # text: desired vs measured once per second
    bash ee_pos.sh               # graph: 3 desired + 3 measured curves
    bash ee_pos.sh x             # one axis (y, z also work)

Needs the simulator / arm-control node running (FK service /ik/fk_srv).
"""

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from kuavo_msgs.msg import sensorsData
from kuavo_msgs.srv import fkSrv

from fk_units import calibrate

FK_SRV = "/ik/fk_srv"
ARM_OFF = 13

q_meas = None
q_cmd = None


def on_state(msg):
    global q_meas
    q = msg.joint_data.joint_q
    if len(q) >= 27:
        q_meas = np.asarray(q[ARM_OFF:ARM_OFF + 14], dtype=float)


def on_traj(msg):
    global q_cmd
    q_cmd = np.radians(np.asarray(msg.position[:14], dtype=float))


def main():
    rospy.init_node("ee_pos", anonymous=True, log_level=rospy.ERROR)
    rospy.wait_for_service(FK_SRV, timeout=10.0)
    fk = rospy.ServiceProxy(FK_SRV, fkSrv)
    scale = calibrate(fk, log=print)

    pub_m = rospy.Publisher("/ee_pos", Float64MultiArray, queue_size=10)
    pub_t = rospy.Publisher("/ee_target", Float64MultiArray, queue_size=10)
    rospy.Subscriber("/sensors_data_raw", sensorsData, on_state, queue_size=1)
    rospy.Subscriber("/kuavo_arm_traj", JointState, on_traj, queue_size=1)

    def pos(q):
        r = fk([float(v) for v in np.asarray(q) * scale])
        if not r.success:
            return None
        return np.array(r.hand_poses.right_pose.pos_xyz, float) * 1000.0

    rate = rospy.Rate(10)
    print("%-22s %-22s %s" % ("desired [mm]", "measured [mm]", "error [mm]"))
    tick = 0
    while not rospy.is_shutdown():
        if q_meas is not None:
            try:
                pm = pos(q_meas)
                pt = pos(q_cmd) if q_cmd is not None else None
                if pm is not None:
                    pub_m.publish(Float64MultiArray(
                        data=[float(pm[0]), float(pm[1]), float(pm[2])]))
                if pt is not None:
                    pub_t.publish(Float64MultiArray(
                        data=[float(pt[0]), float(pt[1]), float(pt[2])]))
                tick += 1
                if tick % 10 == 0:
                    if pt is not None:
                        e = float(np.linalg.norm(pt - pm))
                        print("%7.1f %7.1f %7.1f    %7.1f %7.1f %7.1f    %5.1f"
                              % (pt[0], pt[1], pt[2], pm[0], pm[1], pm[2], e))
                    else:
                        print("%-22s %7.1f %7.1f %7.1f"
                              % ("(no command yet)", pm[0], pm[1], pm[2]))
            except Exception:
                pass
        rate.sleep()


if __name__ == "__main__":
    main()
