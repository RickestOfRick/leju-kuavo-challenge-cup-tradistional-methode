#!/usr/bin/env python3
"""Move the RIGHT hand straight up, stop, down, stop, up again — in Cartesian Z.

A plumbing + motion-profile demo: the hand keeps its orientation and its x/y, only
z changes, with a hold at each waypoint.

    python3 ee_updown_test.py                 # +10 cm, hold, -20 cm, hold, back
    python3 ee_updown_test.py --dz 0.06 --hold 2.0
    python3 ee_updown_test.py --dry-run       # solve IK and print, publish nothing

Needs the simulator + arm-control node (services /ik/fk_srv and the IK service).
Uses the same IK request the task pipeline uses, so the motion is what the robot
would really do. Publishes /kuavo_arm_traj at 100 Hz (degrees), right arm moved,
left arm held at its current joints.
"""

import argparse
import math
import sys

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from kuavo_msgs.msg import sensorsData, ikSolveParam, twoArmHandPoseCmd
from kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv

FK_SRV = "/ik/fk_srv"
IK_SRV = "/ik/two_arm_hand_pose_cmd_srv"
TRAJ_TOPIC = "/kuavo_arm_traj"
ARM_NAMES = ["arm_joint_%d" % i for i in range(1, 15)]
ARM_OFF = 13
IK_MODE = 0x06                 # three-point mixed, the pipeline default
PUB_HZ = 100.0


def ik_param():
    p = ikSolveParam()
    p.major_optimality_tol = 1e-3
    p.major_feasibility_tol = 1e-3
    p.minor_feasibility_tol = 1e-3
    p.major_iterations_limit = 500
    p.oritation_constraint_tol = 1e-3
    p.pos_constraint_tol = 1e-3
    p.pos_cost_weight = 1.0
    p.constraint_mode = IK_MODE
    return p


def solve_ik(ik, q_cur, l_pos, l_quat, r_pos, r_quat):
    cmd = twoArmHandPoseCmd()
    cmd.use_custom_ik_param = True
    cmd.joint_angles_as_q0 = True
    cmd.frame = 0                                   # keep the current frame
    cmd.ik_param = ik_param()
    cmd.hand_poses.left_pose.joint_angles = [float(v) for v in q_cur[:7]]
    cmd.hand_poses.right_pose.joint_angles = [float(v) for v in q_cur[7:]]
    cmd.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    cmd.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    cmd.hand_poses.left_pose.pos_xyz = [float(v) for v in l_pos]
    cmd.hand_poses.left_pose.quat_xyzw = [float(v) for v in l_quat]
    cmd.hand_poses.right_pose.pos_xyz = [float(v) for v in r_pos]
    cmd.hand_poses.right_pose.quat_xyzw = [float(v) for v in r_quat]
    res = ik(cmd)
    if not res.success:
        return None
    return np.array(res.hand_poses.right_pose.joint_angles, float)     # radians


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dz", type=float, default=0.10, help="travel each way [m]")
    ap.add_argument("--move-time", type=float, default=1.5, help="s per segment")
    ap.add_argument("--hold", type=float, default=1.5, help="s pause at each stop")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="start even if another node already publishes /kuavo_arm_traj")
    args = ap.parse_args()

    rospy.init_node("ee_updown_test", anonymous=True, log_level=rospy.ERROR)

    # another publisher on the same topic makes the arm receive two interleaved
    # target streams -> the position oscillates like a triangle wave
    try:
        import rosgraph
        state = rosgraph.Master("/ee_updown_probe").getSystemState()
        pubs = sorted({n for t, nodes in state[0] if t == TRAJ_TOPIC for n in nodes})
    except Exception:
        pubs = []
    pubs = [n for n in pubs if "ee_updown_test" not in n]
    if pubs:
        print("WARNING: %s is already published by:" % TRAJ_TOPIC)
        for n in pubs:
            print("   ", n)
        print("   two target streams interleave -> the arm oscillates.")
        print("   close that node (or pass --force to run anyway).")
        if not args.force:
            return 1
    else:
        print("publishers on %s: none besides this node (good)" % TRAJ_TOPIC)
    rospy.wait_for_service(FK_SRV, timeout=10.0)
    rospy.wait_for_service(IK_SRV, timeout=10.0)
    fk = rospy.ServiceProxy(FK_SRV, fkSrv)
    ik = rospy.ServiceProxy(IK_SRV, twoArmHandPoseCmdSrv)
    pub = rospy.Publisher(TRAJ_TOPIC, JointState, queue_size=10)

    q = [None]

    def on_state(msg):
        j = msg.joint_data.joint_q
        if len(j) >= 27 and q[0] is None:
            q[0] = np.asarray(j[ARM_OFF:ARM_OFF + 14], float)

    rospy.Subscriber("/sensors_data_raw", sensorsData, on_state, queue_size=1)
    t0 = rospy.get_time()
    while q[0] is None and not rospy.is_shutdown():
        if rospy.get_time() - t0 > 15.0:
            print("no /sensors_data_raw — is the simulator running?")
            return 1
        rospy.sleep(0.1)

    q_cur = np.array(q[0])
    f = fk([float(v) for v in q_cur])
    if not f.success:
        print("FK failed")
        return 1
    l_pos = np.array(f.hand_poses.left_pose.pos_xyz, float)
    l_quat = np.array(f.hand_poses.left_pose.quat_xyzw, float)
    r_pos = np.array(f.hand_poses.right_pose.pos_xyz, float)
    r_quat = np.array(f.hand_poses.right_pose.quat_xyzw, float)
    print("start: right hand x=%.0f y=%.0f z=%.0f mm"
          % (r_pos[0] * 1000, r_pos[1] * 1000, r_pos[2] * 1000))

    # up +dz, hold, down to -dz, hold, back to start
    waypoints = [("up", r_pos + [0, 0, args.dz]),
                 ("down", r_pos + [0, 0, -args.dz]),
                 ("back", r_pos.copy())]

    def measure_z():
        m = fk([float(v) for v in q[0]]).hand_poses.right_pose.pos_xyz
        return m[2] * 1000.0

    target_deg = [math.degrees(v) for v in q_cur]
    for name, tgt in waypoints:
        q_sol = solve_ik(ik, q[0], l_pos, l_quat, tgt, r_quat)
        if q_sol is None:
            print("IK failed for %s — aborting" % name)
            return 1
        # the IK response gives the RIGHT arm's 7 joints only; the left keeps its pose
        goal_deg = [math.degrees(v) for v in np.concatenate([q_cur[:7], q_sol])]
        print("%-5s -> z %+.0f mm  (deg J11 %.1f -> %.1f)"
              % (name, tgt[2] * 1000 - r_pos[2] * 1000,
                 target_deg[10], goal_deg[10]))
        if args.dry_run:
            continue
        start = list(target_deg)
        n = int(args.move_time * PUB_HZ)
        for i in range(n + 1):
            a = i / float(n)
            a = 3 * a * a - 2 * a * a * a            # smooth accel/decel
            pos = [s + (g - s) * a for s, g in zip(start, goal_deg)]
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ARM_NAMES
            msg.position = [float(v) for v in pos]
            pub.publish(msg)
            rospy.sleep(1.0 / PUB_HZ)
        target_deg = goal_deg
        for _ in range(int(args.hold * PUB_HZ)):     # hold at the waypoint
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ARM_NAMES
            msg.position = [float(v) for v in target_deg]
            pub.publish(msg)
            rospy.sleep(1.0 / PUB_HZ)
        print("        measured z now %.0f mm" % measure_z())

    print("done — holding the last waypoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
