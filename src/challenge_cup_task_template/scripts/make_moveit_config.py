#!/usr/bin/env python3
"""Generate a MoveIt planning model for the Kuavo arms.

The vendor URDF (challenge_cup_simulator/models/biped_s52/urdf/biped_s52.urdf) has NO
<collision> elements on the arm links, so instead of trimming heavy meshes we synthesise
simple primitives (cylinders + a claw box + a torso box). That is exactly what makes the
collision checking fast enough for planning.

Outputs (all inside this package, the vendor files are never touched):
    config/moveit/planning_robot.urdf
    config/moveit/biped_s52_plan.srdf
    config/moveit/moveit_planning.launch

Run:  python3 make_moveit_config.py
"""

import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC_URDF = os.path.join(PKG, "..", "challenge_cup_simulator", "models",
                        "biped_s52", "urdf", "biped_s52.urdf")
OUT_DIR = os.path.join(PKG, "config", "moveit")

# ── primitive sizes (metres) — tune these after watching a plan ---------------
R_UPPER = 0.050        # zarm_*3_link  (upper arm)
R_FORE = 0.045         # zarm_*4_link  (forearm)
R_WRIST = 0.038        # zarm_*5/6_link
R_HAND = 0.035         # zarm_*7_link  (wrist -> TCP)
CLAW_BOX = (0.060, 0.110, 0.090)   # x, y, z  — an OPEN two-finger claw
CLAW_CENTER = (-0.170 - 0.045)     # z offset from the l7 frame (below the TCP)
TORSO_BOX = (0.240, 0.360, 0.500)
TORSO_CENTER = (0.0, 0.0, 0.120)
TORSO_LINK = "waist_yaw_link"

# ready pose (LEFT then RIGHT), same as task_sequence.READY_JOINTS
READY_LEFT = [30.3, 0.0, 9.6, -137.2, -86.8, -2.9, -2.4]
READY_RIGHT = [30.2, 0.0, -9.2, -138.4, 81.4, 8.0, -1.1]


def cylinder(name, origin_xyz, radius, length, axis="z"):
    """A cylinder from `origin_xyz` (the child joint position) back to this link frame."""
    x, y, z = origin_xyz
    # the link's own joint is at (0,0,0); the next joint sits at origin_xyz, so the
    # cylinder is centred halfway and aligned with that offset
    cx, cy, cz = x * 0.5, y * 0.5, z * 0.5
    rpy = "0 0 0"
    if abs(z) >= abs(x) and abs(z) >= abs(y):
        rpy = "0 0 0"                      # cylinder default axis is +z
        ln = abs(z)
    elif abs(x) >= abs(y):
        rpy = "0 1.5707963 0"              # rotate +z -> +x
        ln = abs(x)
    else:
        rpy = "1.5707963 0 0"              # rotate +z -> +y
        ln = abs(y)
    return (
        '    <collision name="%s_collision">\n'
        '      <origin xyz="%.6f %.6f %.6f" rpy="%s"/>\n'
        '      <geometry>\n'
        '        <cylinder radius="%.4f" length="%.4f"/>\n'
        '      </geometry>\n'
        '    </collision>\n' % (name, cx, cy, cz, rpy, radius, ln))


def box(name, center, size):
    return (
        '    <collision name="%s_collision">\n'
        '      <origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>\n'
        '      <geometry>\n'
        '        <box size="%.4f %.4f %.4f"/>\n'
        '      </geometry>\n'
        '    </collision>\n' % (name, center[0], center[1], center[2],
                               size[0], size[1], size[2]))


def arm_joint_offsets(root, side):
    """{link_name: xyz of the joint that starts at that link} for one arm."""
    pref = "zarm_%s" % side
    off = {}
    for j in root.findall("joint"):
        par = j.find("parent").get("link")
        if not par.startswith(pref):
            continue
        o = j.find("origin")
        xyz = [0.0, 0.0, 0.0]
        if o is not None and o.get("xyz"):
            xyz = [float(v) for v in o.get("xyz").split()]
        off[par] = xyz
    return off


def build_urdf():
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}

    added = 0
    for side in ("l", "r"):
        off = arm_joint_offsets(root, side)
        radii = {"3": R_UPPER, "4": R_FORE, "5": R_WRIST, "6": R_WRIST, "7": R_HAND}
        for link_name, xyz in off.items():
            num = link_name.replace("zarm_%s" % side, "").replace("_link", "")
            if num not in radii:
                continue
            ln = max(abs(xyz[0]), abs(xyz[1]), abs(xyz[2]))
            if ln < 0.005:
                continue                    # co-located joints (shoulder) -> nothing to draw
            link = links.get(link_name)
            if link is None or link.find("collision") is not None:
                continue
            xml = cylinder(link_name, xyz, radii[num], ln)
            link.append(ET.fromstring(xml.strip()))
            added += 1

        # claw: a single box below the TCP, wider in y (the fingers travel in y)
        l7 = links.get("zarm_%s7_link" % side)
        if l7 is not None:
            l7.append(ET.fromstring(
                box("zarm_%s7_claw" % side, (0.0, 0.0, -0.215), CLAW_BOX).strip()))
            added += 1

    # torso so the arms cannot plan through the body
    torso = links.get(TORSO_LINK)
    if torso is not None and torso.find("collision") is None:
        torso.append(ET.fromstring(
            box(TORSO_LINK, TORSO_CENTER, TORSO_BOX).strip()))
        added += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "planning_robot.urdf")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out, added


def build_srdf(urdf_path):
    # MoveIt rejects a semantic description whose robot name differs from the URDF's
    name = ET.parse(urdf_path).getroot().get("name")
    # the URDF root must be the virtual-joint child, otherwise the tree is disconnected
    root = ET.parse(urdf_path).getroot()
    links = {l.get("name") for l in root.findall("link")}
    kids = {j.find("child").get("link") for j in root.findall("joint")}
    urdf_root = sorted(links - kids)[0]
    parts = ['<?xml version="1.0"?>',
             '<robot name="%s">' % name,
             '  <virtual_joint name="virtual_fixed" type="fixed"'
             ' parent_frame="odom" child_link="%s"/>' % urdf_root,
             '  <!-- arms only: the legs are not planned for -->',
             '  <group name="left_arm">',
             '    <chain base_link="%s" tip_link="zarm_l7_end_effector"/>' % TORSO_LINK,
             '  </group>',
             '  <group name="right_arm">',
             '    <chain base_link="%s" tip_link="zarm_r7_end_effector"/>' % TORSO_LINK,
             '  </group>',
             '  <group_state name="ready" group="left_arm">']
    for i, v in enumerate(READY_LEFT, 1):
        parts.append('    <joint name="zarm_l%d_joint" value="%.4f"/>'
                     % (i, v * 3.141592653589793 / 180.0))
    parts += ['  </group_state>',
              '  <group_state name="ready" group="right_arm">']
    for i, v in enumerate(READY_RIGHT, 1):
        parts.append('    <joint name="zarm_r%d_joint" value="%.4f"/>'
                     % (i, v * 3.141592653589793 / 180.0))
    parts += ['  </group_state>',
              '  <!-- without these a POSE goal is rejected ("only joint-space goals are',
              '       supported") because the group has no declared end effector -->',
              '  <end_effector name="left_ee" parent_link="zarm_l7_end_effector" group="left_arm"/>',
              '  <end_effector name="right_ee" parent_link="zarm_r7_end_effector" group="right_arm"/>',
              '  <!-- adjacent links are welded together; without these MoveIt reports',
              '       self-collisions for every chain neighbour -->']

    # adjacent pairs along both chains
    for side in ("l", "r"):
        prev = TORSO_LINK
        for i in range(1, 8):
            cur = "zarm_%s%d_link" % (side, i)
            parts.append('  <disable_collisions link1="%s" link2="%s" reason="Adjacent"/>'
                         % (prev, cur))
            prev = cur
        parts.append('  <disable_collisions link1="%s" link2="zarm_%s7_end_effector"'
                     ' reason="Adjacent"/>' % (prev, side))
        parts.append('  <disable_collisions link1="%s" link2="zarm_%s7_end_effector"'
                     ' reason="Never"/>' % (TORSO_LINK, side))

    parts.append("</robot>")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "biped_s52_plan.srdf")
    open(out, "w").write("\n".join(parts) + "\n")
    return out


def build_yaml():
    kin = """left_arm:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.05
right_arm:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.05
"""
    ompl = """planner_configs:
  RRTConnect:
    type: geometric::RRTConnect
    range: 0.0
left_arm:
  default_planner_config: RRTConnect
  planner_configs: [RRTConnect]
  longest_valid_segment_fraction: 0.005
right_arm:
  default_planner_config: RRTConnect
  planner_configs: [RRTConnect]
  longest_valid_segment_fraction: 0.005
"""
    os.makedirs(OUT_DIR, exist_ok=True)
    k = os.path.join(OUT_DIR, "kinematics.yaml")
    o = os.path.join(OUT_DIR, "ompl_planning.yaml")
    open(k, "w").write(kin)
    open(o, "w").write(ompl)
    return k, o


def build_launch(urdf_path, srdf_path):
    txt = """<launch>
  <!-- Plan-only MoveIt for the Kuavo arms. No controllers: we read the planned joint
       trajectory and stream it ourselves over /kuavo_arm_traj. -->
  <!-- package-relative: the workspace path differs inside the container -->
  <arg name="robot_description"
       default="$(find challenge_cup_task_template)/config/moveit/planning_robot.urdf"/>
  <arg name="srdf"
       default="$(find challenge_cup_task_template)/config/moveit/biped_s52_plan.srdf"/>
  <arg name="rviz" default="false"/>

  <param name="robot_description" textfile="$(arg robot_description)"/>
  <param name="robot_description_semantic" textfile="$(arg srdf)"/>
  <rosparam command="load"
            file="$(find challenge_cup_task_template)/config/moveit/kinematics.yaml"/>
  <!-- standalone only: in the real run /joint_states comes from the simulator -->
  <node pkg="joint_state_publisher" type="joint_state_publisher" name="joint_state_publisher">
    <param name="rate" value="30"/>
  </node>

  <node pkg="moveit_ros_move_group" type="move_group" name="move_group" output="screen">
    <!-- these MUST be private (~): with them global move_group logs
         "Loading planning pipeline ''" and silently falls back to CHOMP -->
    <param name="planning_plugin" value="ompl_interface/OMPLPlanner"/>
    <param name="request_adapters"
           value="default_planner_request_adapters/AddTimeParameterization
                  default_planner_request_adapters/FixWorkspaceBounds
                  default_planner_request_adapters/FixStartStateBounds
                  default_planner_request_adapters/FixStartStateCollision
                  default_planner_request_adapters/FixStartStatePathConstraints"/>
    <param name="start_state_max_bounds_error" value="0.1"/>
    <rosparam command="load"
              file="$(find challenge_cup_task_template)/config/moveit/ompl_planning.yaml"/>
    <!-- we never execute through MoveIt; this keeps it from aborting on startup -->
    <param name="moveit_controller_manager"
           value="moveit_fake_controller_manager/MoveItFakeControllerManager"/>
    <param name="allow_trajectory_execution" value="false"/>
    <param name="fake_execution" value="true"/>
    <param name="publish_planning_scene" value="true"/>
    <param name="publish_geometry_updates" value="true"/>
    <param name="publish_state_updates" value="true"/>
    <param name="publish_transforms_updates" value="true"/>
  </node>

  <!-- plan-only: no ros_control, no joint_trajectory_controller -->
</launch>
"""
    out = os.path.join(OUT_DIR, "moveit_planning.launch")
    open(out, "w").write(txt)
    return out


if __name__ == "__main__":
    u, added = build_urdf()
    s = build_srdf(u)
    build_yaml()
    l = build_launch(u, s)
    print("collision primitives added : %d" % added)
    print("urdf   ->", u)
    print("srdf   ->", s)
    print("launch ->", l)
