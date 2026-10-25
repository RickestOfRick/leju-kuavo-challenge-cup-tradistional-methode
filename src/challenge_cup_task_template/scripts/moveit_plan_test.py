#!/usr/bin/env python3
"""Smoke test: can MoveIt plan for the Kuavo arms with our generated config?

Plan only — nothing is executed. Run inside the container after:
    roslaunch challenge_cup_task_template moveit_planning.launch
    rosrun challenge_cup_task_template moveit_plan_test.py
"""

import sys

import moveit_commander
import rospy


def show(group_name, target):
    g = moveit_commander.MoveGroupCommander(group_name)
    g.set_planning_time(5.0)
    g.set_num_planning_attempts(3)
    g.set_named_target(target)
    out = g.plan()
    # MoveIt 1.x returns (RobotTrajectory, fraction), (success, RobotTrajectory, ...)
    # or just a bool depending on the version -> dig the trajectory out of any shape
    plan = None
    if isinstance(out, tuple):
        for item in out:
            if hasattr(item, "joint_trajectory"):
                plan = item
                break
    elif hasattr(out, "joint_trajectory"):
        plan = out
    if plan is None:
        plan = g.get_plan()
    pts = plan.joint_trajectory.points
    print("  %-10s -> %-8s  success=%s  points=%d  duration=%.2fs"
          % (group_name, target, bool(pts), len(pts),
             plan.joint_trajectory.points[-1].time_from_start.to_sec() if pts else 0.0))
    if pts:
        print("             joints: %s" % ", ".join(plan.joint_trajectory.joint_names))
        for i in (0, len(pts) // 2, len(pts) - 1):
            q = pts[i].positions
            print("             [%2d] %s" % (i, " ".join("%7.2f" % (v * 57.2958) for v in q)))
    return bool(pts)


if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_plan_test", anonymous=True)

    ok = True
    for name in ("left_arm", "right_arm"):
        try:
            ok = show(name, "ready") and ok
        except Exception as e:
            print("  %s FAILED: %s" % (name, e))
            ok = False

    # pose goal: above the sorting box, TCP pointing down (180 deg about x)
    try:
        from geometry_msgs.msg import Pose
        g = moveit_commander.MoveGroupCommander("left_arm")
        g.set_planning_time(5.0)
        g.set_num_planning_attempts(3)
        # no simulator here -> give MoveIt an explicit start state (the ready pose)
        from moveit_msgs.msg import RobotState
        from sensor_msgs.msg import JointState
        js = JointState()
        js.name = ["zarm_l%d_joint" % i for i in range(1, 8)]
        js.position = [v * 3.141592653589793 / 180.0 for v in
                       (30.3, 0.0, 9.6, -137.2, -86.8, -2.9, -2.4)]
        rs = RobotState()
        rs.joint_state = js
        g.set_start_state(rs)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = 0.606, 0.226, 0.450
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = 1.0, 0.0, 0.0, 0.0
        g.set_pose_target(pose)
        out = g.plan()
        plan = None
        for item in (out if isinstance(out, tuple) else [out]):
            if hasattr(item, "joint_trajectory"):
                plan = item
        pts = plan.joint_trajectory.points if plan is not None else []
        print("  left_arm   -> pose above box (0.606,0.226,0.45)  success=%s  points=%d"
              % (bool(pts), len(pts)))
        if pts:
            print("             final %s" % " ".join("%7.2f" % (v * 57.2958) for v in pts[-1].positions))
        ok = bool(pts) and ok
    except Exception as e:
        print("  pose goal FAILED: %s" % e)
        ok = False

    print("RESULT:", "OK" if ok else "FAILED")
    moveit_commander.roscpp_shutdown()
