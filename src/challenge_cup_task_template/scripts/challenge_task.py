#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import math
import os
import sys


ARM_JOINT_NAMES = [
    "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
    "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
    "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
    "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
]

HOME_ARM_DEG = [20, 20, 0, -35, 0, 0, 0, 20, -20, 0, -35, 0, 0, 0]

# 这些坐标来自公开场景配置，后续可由相机/点云识别结果覆盖。
SCENE_TASKS = {
    "scene1": {
        "weigh": [-0.17, -0.56, 0.90],
        "drop": [0.10, 0.29, 1.05],
        "objects": [
            {"name": "parcel_1", "pick": [-0.26, -0.31, 0.90], "arm": "right"},
            {"name": "parcel_2", "pick": [-0.26, -0.09, 0.90], "arm": "right"},
            {"name": "parcel_3", "pick": [-0.11, -0.31, 0.90], "arm": "right"},
            {"name": "parcel_4", "pick": [-0.11, -0.09, 0.90], "arm": "right"},
        ],
    },
    "scene2": {
        "bins": {
            "part_type_a": [-0.04, -0.43, 0.96],
            "part_type_b": [-0.04, 0.00, 0.96],
        },
        "objects": [
            {"name": "part_type_a_1", "pick": [-0.28, -0.35, 0.87], "arm": "right", "type": "part_type_a"},
            {"name": "part_type_a_2", "pick": [-0.28, -0.25, 0.87], "arm": "right", "type": "part_type_a"},
            {"name": "part_type_b_1", "pick": [-0.28, -0.07, 0.89], "arm": "right", "type": "part_type_b"},
            {"name": "part_type_b_2", "pick": [-0.28, 0.03, 0.89], "arm": "left", "type": "part_type_b"},
        ],
    },
    "scene3": {
        "outbound": [-2.39, 0.00, 1.05],
        "objects": [
            {"name": "smt_tray_lower", "pick": [0.983, 0.15, 0.78], "arm": "left"},
            {"name": "smt_tray_upper", "pick": [0.983, 0.25, 1.18], "arm": "left"},
        ],
    },
}


SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
    },
}


def _quat_down(yaw=0.0):
    # 末端朝下的抓取姿态，按 yaw 做桌面内旋转。
    half_yaw = yaw * 0.5
    half_pitch = -math.pi * 0.25
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    return [sy * sp, cy * sp, sy * cp, cy * cp]


def _load_launcher():
    # 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
    # 从那里导入，确保完整性校验无法被绕过。
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


class ChallengeTaskController:
    def __init__(self, cmd_vel_pub, arm_traj_pub):
        import rospy

        self.rospy = rospy
        self.cmd_vel_pub = cmd_vel_pub
        self.arm_traj_pub = arm_traj_pub
        self.current_arm_rad = [0.0] * 14
        rospy.Subscriber("/sensors_data_raw", self._sensors_type(), self._sensor_cb, queue_size=1)

    def _sensors_type(self):
        from kuavo_msgs.msg import sensorsData
        return sensorsData

    def _sensor_cb(self, msg):
        joint_q = list(getattr(getattr(msg, "joint_data", None), "joint_q", []))
        if len(joint_q) >= 26:
            self.current_arm_rad = joint_q[12:26]

    def prepare(self):
        self._set_arm_mode(2)
        self.open_claw("both")
        self.move_arm_deg(HOME_ARM_DEG, duration=2.0)

    def finish(self):
        self.move_arm_deg(HOME_ARM_DEG, duration=1.5)
        self.stop_base(1.0)

    def _set_arm_mode(self, mode):
        try:
            from kuavo_msgs.srv import changeArmCtrlMode
            self.rospy.wait_for_service("/humanoid_change_arm_ctrl_mode", timeout=3.0)
            self.rospy.ServiceProxy("/humanoid_change_arm_ctrl_mode", changeArmCtrlMode)(mode)
        except Exception as exc:
            self.rospy.logwarn("切换手臂外部控制模式失败，继续尝试直接发手臂轨迹: %s", exc)

    def move_arm_deg(self, positions, duration=1.0):
        from sensor_msgs.msg import JointState

        msg = JointState()
        msg.name = list(ARM_JOINT_NAMES)
        msg.position = [float(v) for v in positions]
        msg.velocity = [0.0] * 14
        msg.effort = [0.0] * 14
        end = self.rospy.Time.now() + self.rospy.Duration(float(duration))
        rate = self.rospy.Rate(30)
        while not self.rospy.is_shutdown() and self.rospy.Time.now() < end:
            msg.header.stamp = self.rospy.Time.now()
            self.arm_traj_pub.publish(msg)
            rate.sleep()

    def stop_base(self, duration=0.4):
        from geometry_msgs.msg import Twist

        msg = Twist()
        end = self.rospy.Time.now() + self.rospy.Duration(float(duration))
        rate = self.rospy.Rate(20)
        while not self.rospy.is_shutdown() and self.rospy.Time.now() < end:
            self.cmd_vel_pub.publish(msg)
            rate.sleep()

    def drive_for(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, duration=1.0):
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)
        end = self.rospy.Time.now() + self.rospy.Duration(float(duration))
        rate = self.rospy.Rate(20)
        while not self.rospy.is_shutdown() and self.rospy.Time.now() < end:
            self.cmd_vel_pub.publish(msg)
            rate.sleep()
        self.stop_base(0.5)

    def open_claw(self, arm):
        self._set_claw(arm, 0)
        self.rospy.sleep(0.8)

    def close_claw(self, arm):
        self._set_claw(arm, 78)
        self.rospy.sleep(1.0)

    def _set_claw(self, arm, position):
        try:
            from kuavo_msgs.msg import endEffectorData
            from kuavo_msgs.srv import controlLejuClaw

            names = ["left_claw", "right_claw"]
            if arm == "left":
                positions = [position, 0]
            elif arm == "right":
                positions = [0, position]
            else:
                positions = [position, position]
            data = endEffectorData()
            data.name = names
            data.position = [float(v) for v in positions]
            data.velocity = [60.0, 60.0]
            data.effort = [1.5, 1.5]
            self.rospy.wait_for_service("/control_robot_leju_claw", timeout=3.0)
            result = self.rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)(data)
            if hasattr(result, "success") and not result.success:
                self.rospy.logwarn("夹爪命令被拒绝: %s", getattr(result, "message", ""))
        except Exception as exc:
            self.rospy.logwarn("夹爪控制失败: %s", exc)

    def pick_and_place(self, item, place_xyz, waypoint_xyz=None):
        name = item["name"]
        arm = item.get("arm") or ("left" if item["pick"][1] >= 0.0 else "right")
        pick_xyz = item["pick"]
        self.rospy.loginfo("执行 %s: 抓取 %s -> 放置 %s", name, pick_xyz, place_xyz)
        self.open_claw(arm)
        self._move_hand(arm, self._above(pick_xyz, 0.14), label=name + "_pre_pick")
        self._move_hand(arm, pick_xyz, label=name + "_pick")
        self.close_claw(arm)
        self._move_hand(arm, self._above(pick_xyz, 0.16), label=name + "_lift")
        if waypoint_xyz is not None:
            self._move_hand(arm, waypoint_xyz, label=name + "_waypoint")
        self._move_hand(arm, self._above(place_xyz, 0.12), label=name + "_pre_place")
        self._move_hand(arm, place_xyz, label=name + "_place")
        self.open_claw(arm)
        self._move_hand(arm, self._above(place_xyz, 0.16), label=name + "_retreat")

    def _above(self, xyz, dz):
        return [float(xyz[0]), float(xyz[1]), float(xyz[2]) + float(dz)]

    def _move_hand(self, arm, target_xyz, label):
        try:
            target_deg = self._solve_ik(arm, target_xyz)
            self.move_arm_deg(target_deg, duration=1.6)
        except Exception as exc:
            self.rospy.logwarn("%s IK 失败，使用预置关节兜底: %s", label, exc)
            self.move_arm_deg(self._fallback_pose(arm, target_xyz), duration=1.2)

    def _solve_ik(self, arm, target_xyz):
        from kuavo_msgs.msg import ikSolveParam, twoArmHandPoseCmd
        from kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv

        current = list(self.current_arm_rad)
        self.rospy.wait_for_service("/ik/fk_srv", timeout=2.0)
        fk = self.rospy.ServiceProxy("/ik/fk_srv", fkSrv)(current).hand_poses
        req = twoArmHandPoseCmd()
        req.use_custom_ik_param = True
        req.joint_angles_as_q0 = True
        req.ik_param = ikSolveParam()
        req.ik_param.major_optimality_tol = 1e-3
        req.ik_param.major_feasibility_tol = 1e-3
        req.ik_param.minor_feasibility_tol = 1e-3
        req.ik_param.major_iterations_limit = 80
        req.ik_param.oritation_constraint_tol = 1e-2
        req.ik_param.pos_constraint_tol = 1e-3
        req.ik_param.pos_cost_weight = 1.0
        req.ik_param.constraint_mode = 0
        req.hand_poses.left_pose.joint_angles = current[:7]
        req.hand_poses.right_pose.joint_angles = current[7:]
        req.hand_poses.left_pose.pos_xyz = list(fk.left_pose.pos_xyz)
        req.hand_poses.right_pose.pos_xyz = list(fk.right_pose.pos_xyz)
        req.hand_poses.left_pose.quat_xyzw = list(fk.left_pose.quat_xyzw)
        req.hand_poses.right_pose.quat_xyzw = list(fk.right_pose.quat_xyzw)

        quat = _quat_down(0.0 if arm == "left" else math.pi)
        if arm == "left":
            req.hand_poses.left_pose.pos_xyz = list(target_xyz)
            req.hand_poses.left_pose.quat_xyzw = quat
        else:
            req.hand_poses.right_pose.pos_xyz = list(target_xyz)
            req.hand_poses.right_pose.quat_xyzw = quat

        self.rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=2.0)
        res = self.rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(req)
        if not getattr(res, "success", False):
            raise RuntimeError(getattr(res, "error_reason", "IK service returned false"))
        q_arm = list(res.q_arm) if hasattr(res, "q_arm") else []
        if len(q_arm) < 14:
            q_arm = list(res.hand_poses.left_pose.joint_angles) + list(res.hand_poses.right_pose.joint_angles)
        if len(q_arm) < 14:
            raise RuntimeError("IK response did not contain 14 arm joints")
        return [math.degrees(float(v)) for v in q_arm[:14]]

    def _fallback_pose(self, arm, target_xyz):
        pose = list(HOME_ARM_DEG)
        y = float(target_xyz[1])
        z = float(target_xyz[2])
        reach = max(-45.0, min(35.0, (0.95 - z) * 140.0))
        side = max(-35.0, min(35.0, y * 70.0))
        if arm == "left":
            pose[:7] = [30 + reach, 20 - side, 0, -65, 0, -20, 0]
        else:
            pose[7:] = [30 + reach, -20 - side, 0, -65, 0, -20, 0]
        return pose


def execute_scene_task(scene, controller):
    if scene == "scene1":
        cfg = SCENE_TASKS["scene1"]
        for item in cfg["objects"]:
            controller.pick_and_place(item, cfg["weigh"])
            controller.rospy.sleep(1.0)
            weighed = dict(item)
            weighed["pick"] = list(cfg["weigh"])
            controller.pick_and_place(weighed, cfg["drop"])
        return

    if scene == "scene2":
        cfg = SCENE_TASKS["scene2"]
        for item in cfg["objects"]:
            controller.pick_and_place(item, cfg["bins"][item["type"]])
        return

    if scene == "scene3":
        cfg = SCENE_TASKS["scene3"]
        for item in cfg["objects"]:
            arm = item["arm"]
            pick_xyz = item["pick"]
            controller.drive_for(linear_x=0.18, duration=4.5)
            controller.open_claw(arm)
            controller._move_hand(arm, controller._above(pick_xyz, 0.12), label=item["name"] + "_pre_pick")
            controller._move_hand(arm, pick_xyz, label=item["name"] + "_pick")
            controller.close_claw(arm)
            controller._move_hand(arm, controller._above(pick_xyz, 0.18), label=item["name"] + "_lift")
            controller.drive_for(linear_x=-0.18, duration=7.0)
            controller._move_hand(arm, controller._above(cfg["outbound"], 0.12), label=item["name"] + "_pre_place")
            controller._move_hand(arm, cfg["outbound"], label=item["name"] + "_place")
            controller.open_claw(arm)
            controller._move_hand(arm, controller._above(cfg["outbound"], 0.18), label=item["name"] + "_retreat")
            controller.drive_for(linear_x=0.18, duration=2.5)
        return


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState

    rospy.loginfo("=== %s任务启动 ===", config["title"])

    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)

    rospy.sleep(1.0)
    rospy.loginfo("场景实例已初始化。")

    controller = ChallengeTaskController(cmd_vel_pub, arm_traj_pub)
    controller.prepare()
    execute_scene_task(scene, controller)
    controller.finish()

    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认按 scene 自动设置")
    parser.add_argument("--timeout", type=int, default=120,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
    )


if __name__ == "__main__":
    main()
