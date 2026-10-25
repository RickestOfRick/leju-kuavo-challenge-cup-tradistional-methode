#!/usr/bin/env python3
"""Plan-only MoveIt client for the Kuavo arms.

MoveIt is used as a PLANNER only: it returns joint waypoints, which task_sequence
streams through its existing /kuavo_arm_traj publisher. No controllers, no execution
through MoveIt (so the sag/creep compensation stays entirely in our own code).

    p = ArmPlanner("left_arm")
    p.add_scene()                       # table + hollow box + scale (collision_model)
    wps = p.plan_joints([...7 deg...])  # -> [(joints_deg, t_sec), ...]
    wps = p.plan_pose((0.606, 0.226, 0.36))   # claw-down orientation, their convention

Run standalone after `roslaunch challenge_cup_task_template moveit_planning.launch`:
    python3 moveit_planner.py
"""

import ctypes as _ctypes
import math

# This workspace ships an OLD libruckig.so that shadows ROS's newer one, which breaks
# moveit_commander with "undefined symbol ... PositionStep2::get_profile". Preloading the
# correct library as RTLD_GLOBAL makes the later dlopen reuse it, so no LD_LIBRARY_PATH
# juggling is needed when running from the sim shell.
try:
    _ctypes.CDLL("/opt/ros/noetic/lib/x86_64-linux-gnu/libruckig.so",
                 mode=_ctypes.RTLD_GLOBAL)
except OSError:
    pass

import moveit_commander
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

import collision_model as cm

JOINTS = {
    "left_arm": ["zarm_l%d_joint" % i for i in range(1, 8)],
    "right_arm": ["zarm_r%d_joint" % i for i in range(1, 8)],
}
READY_DEG = {
    "left_arm": [30.3, 0.0, 9.6, -137.2, -86.8, -2.9, -2.4],
    "right_arm": [30.2, 0.0, -9.2, -138.4, 81.4, 8.0, -1.1],
}
# their own two-rotation convention: approach direction + tool roll
PICK_FIRST_YPR = [0.0, -100.0, 0.0]
PICK_SECOND_YPR = [90.0, 0.0, 0.0]
BOX_FIRST_YPR = [-0.328, -100.935, 0.0]
BOX_SECOND_YPR = [-90.0, 0.0, 0.369]


def quat_from_ypr(first, second=None):
    """Same maths as task_sequence._quat_from_ypr (kept local so this file has no ROS
    import of the task module)."""
    def matmul3(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)]

    yaw, pitch, _ = [math.radians(float(v)) for v in first]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    m = [[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0.0, cp]]
    if second is not None:
        my, mp, mr = [math.radians(float(v)) for v in second]
        man = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if abs(my) > 0.01:
            c, s = math.cos(my), math.sin(my)
            man = matmul3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], man)
        if abs(mp) > 0.01:
            c, s = math.cos(mp), math.sin(mp)
            man = matmul3([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], man)
        if abs(mr) > 0.01:
            c, s = math.cos(mr), math.sin(mr)
            man = matmul3([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], man)
        m = matmul3(m, man)
    # rotation matrix -> quaternion (xyzw)
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x = 0.25 * s, (m[2][1] - m[1][2]) / s
        y, z = (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w, x = (m[2][1] - m[1][2]) / s, 0.25 * s
        y, z = (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w, x = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s
        y, z = 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w, x = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s
        y, z = (m[1][2] + m[2][1]) / s, 0.25 * s
    return (x, y, z, w)


class ArmPlanner(object):
    def __init__(self, group="left_arm", planning_time=3.0, attempts=5):
        self.name = group
        self.group = moveit_commander.MoveGroupCommander(group)
        self.group.set_planning_time(planning_time)
        self.group.set_num_planning_attempts(attempts)
        self.group.set_max_velocity_scaling_factor(1.0)
        self.group.set_max_acceleration_scaling_factor(1.0)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.joints = JOINTS[group]
        self.ready = READY_DEG[group]

    # ── scene ──────────────────────────────────────────────────────────
    def add_box(self, name, center, size):
        # PlanningSceneInterface wants a PoseStamped (it copies pose.header)
        ps = PoseStamped()
        ps.header.frame_id = self.group.get_planning_frame()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = \
            [float(v) for v in center]
        ps.pose.orientation.w = 1.0
        self.scene.add_box(name, ps, size=[float(v) for v in size])

    def add_scene(self, model=None):
        """Add table + hollow sorting box + scale from the shared collision model."""
        m = model or cm.SceneModel(cm.FB_BOX_XY, 0.219, cm.FB_SCALE_XY)
        for o in m.obstacles(include_parcels=False):
            self.add_box(o.name, (o.cx, o.cy, o.cz), (o.sx, o.sy, o.sz))
        return m

    def add_parcels(self, model):
        for o in model.parcels:
            self.add_box(o.name, (o.cx, o.cy, o.cz), (o.sx, o.sy, o.sz))

    def attach_parcel(self, link, center_xyz, size, name="held_parcel"):
        """Make a held parcel part of the hand for collision checking.

        `center_xyz` is in the LINK frame (the TCP sits ~0.17 m below the l7 frame, so
        the parcel centre is roughly (0, 0, -0.22) once the claw has closed).
        """
        ps = PoseStamped()
        ps.header.frame_id = link
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = \
            [float(v) for v in center_xyz]
        ps.pose.orientation.w = 1.0
        self.scene.attach_box(link, name, ps, size=[float(v) for v in size])

    def detach_parcel(self, link, name="held_parcel"):
        self.scene.remove_attached_object(link, name)

    # ── planning ───────────────────────────────────────────────────────
    def _start_state(self, start_deg):
        if start_deg is None:
            return
        js = JointState()
        js.name = list(self.joints)
        js.position = [math.radians(v) for v in start_deg]
        rs = RobotState()
        rs.joint_state = js
        self.group.set_start_state(rs)

    def _trajectory(self, plan):
        pts = plan.joint_trajectory.points
        out = []
        for p in pts:
            out.append(([math.degrees(v) for v in p.positions],
                        p.time_from_start.to_sec()))
        return out

    def plan_joints(self, target_deg, start_deg=None):
        """Plan to a 7-joint target (degrees). Returns [(joints_deg, t), ...] or None."""
        self._start_state(start_deg)
        self.group.set_joint_value_target([math.radians(v) for v in target_deg])
        out = self.group.plan()
        plan = None
        for item in (out if isinstance(out, tuple) else [out]):
            if hasattr(item, "joint_trajectory"):
                plan = item
        if plan is None or not plan.joint_trajectory.points:
            return None
        return self._trajectory(plan)

    def plan_pose(self, xyz, ypr_first=BOX_FIRST_YPR, ypr_second=BOX_SECOND_YPR,
                  start_deg=None):
        """Plan to a Cartesian pose using the handoff/pick orientation convention."""
        self._start_state(start_deg)
        q = quat_from_ypr(ypr_first, ypr_second)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in xyz]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        self.group.set_pose_target(pose)
        out = self.group.plan()
        plan = None
        for item in (out if isinstance(out, tuple) else [out]):
            if hasattr(item, "joint_trajectory"):
                plan = item
        if plan is None or not plan.joint_trajectory.points:
            return None
        return self._trajectory(plan)

    def clear_scene(self):
        self.scene.remove_world_object()


# ────────────────────────────────────────────────────────────────────── demo
if __name__ == "__main__":
    import sys
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("arm_planner_demo", anonymous=True)

    p = ArmPlanner("left_arm")
    model = p.add_scene()
    rospy.sleep(1.5)
    print("scene objects:", sorted(p.scene.get_known_object_names()))

    # 1) plain joint goal (always works, no IK involved)
    wps = p.plan_joints(READY_DEG["left_arm"], start_deg=[0.0] * 7)
    print("joint goal ready : %s" % ("FAILED" if not wps else
                                     "%d pts, %.2fs" % (len(wps), wps[-1][1])))

    # 2) pose goal above the box, claw pointing down (their orientation convention)
    wps = p.plan_pose((0.606, 0.226, 0.36))
    print("pose above box   : %s" % ("FAILED" if not wps else
                                     "%d pts, %.2fs, final=%s" % (
                                         len(wps), wps[-1][1],
                                         " ".join("%.1f" % v for v in wps[-1][0]))))

    moveit_commander.roscpp_shutdown()
