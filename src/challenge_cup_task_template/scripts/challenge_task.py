#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import importlib
import os
import signal
import subprocess
import sys

GREEN = "\033[32m"
RESET = "\033[0m"

# 动作开始确认: 1=需要人工按 Enter, 0=固定初始位姿后等待 1s 自动开始。
NEED_OPERATOR_ENTER = 0

# 任务一视觉模式: 0=qpos 投影假框+深度拟合, 1=YOLO RGB 框+深度拟合。
TASK1_VISION_MODE = 1

TASK1_PACKAGE = "task1_pick"
TASK1_MODULE = "task1_node"
TASK2_PACKAGE = "task2_motion"
TASK2_MODULE = "task2_node"

SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
        "head_camera": {
            "enabled": True,
            "yaw_deg": 0.0,
            "pitch_deg": 20.0,
            "settle_time": 0.8,
        },
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
        "head_camera": {
            "enabled": True,
            "yaw_deg": 0.0,
            "pitch_deg": 20.0,
            "settle_time": 0.4,
        },
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
        "head_camera": {
            "enabled": True,
            "yaw_deg": 0.0,
            "pitch_deg": 10.0,
            "settle_time": 0.4,
        },
    },
}


# 按进程组逐级停止子进程, 避免 Ctrl+C 后 roslaunch 子节点残留。
def _terminate_process_group(proc, name, wait_sigint=3.0, wait_sigterm=2.0):
    if proc is None or proc.poll() is not None:
        return
    import rospy

    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return

    for sig, timeout, label in (
        (signal.SIGINT, wait_sigint, "SIGINT"),
        (signal.SIGTERM, wait_sigterm, "SIGTERM"),
        (signal.SIGKILL, 1.0, "SIGKILL"),
    ):
        if proc.poll() is not None:
            return
        try:
            os.killpg(pgid, sig)
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            rospy.logwarn("challenge_task: %s did not exit after %s", name, label)
        except Exception as exc:
            rospy.logwarn(
                "challenge_task: failed to stop %s with %s: %s", name, label, exc
            )


def _load_launcher():
    # 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
    # 从那里导入，确保完整性校验无法被绕过。
    try:
        import rospkg

        sim_utils = os.path.join(
            rospkg.RosPack().get_path("challenge_cup_simulator"), "utils"
        )
    except Exception:
        sim_utils = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "challenge_cup_simulator",
            "utils",
        )
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher

    return ChallengeSimLauncher


def _load_task1_node():
    try:
        import rospkg

        scripts_dir = os.path.join(rospkg.RosPack().get_path(TASK1_PACKAGE), "scripts")
    except Exception:
        scripts_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            TASK1_PACKAGE,
            "scripts",
        )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module(TASK1_MODULE)


def _load_task2_node():
    try:
        import rospkg

        scripts_dir = os.path.join(rospkg.RosPack().get_path(TASK2_PACKAGE), "scripts")
    except Exception:
        scripts_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            TASK2_PACKAGE,
            "scripts",
        )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module(TASK2_MODULE)


# 管理随主任务启动和退出的辅助进程。
class ManagedProcess:
    def __init__(self, name, cmd):
        self.name = name
        self.cmd = list(cmd)
        self.proc = None

    # 启动需要随主流程退出而回收的辅助进程。
    def start(self):
        import rospy

        rospy.loginfo("challenge_task: starting %s", self.name)
        self.proc = subprocess.Popen(self.cmd, preexec_fn=os.setsid)
        rospy.on_shutdown(self.stop)
        return self

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        _terminate_process_group(self.proc, self.name)
        self.proc = None


# 按场景启动任务依赖的辅助 ROS 进程。
def _start_scene_services(scene, seed=0):
    if scene == "scene1":
        return [
            ManagedProcess(
                "task1_vision",
                [
                    "roslaunch",
                    "task1_vision",
                    "task1_vision_scene1.launch",
                    "start_rqt:=false",
                    "vision_mode:={}".format(int(TASK1_VISION_MODE)),
                ],
            ).start()
        ]
    if scene == "scene2":
        return [
            ManagedProcess(
                "task2_vision",
                [
                    "roslaunch",
                    "task2_vision",
                    "task2_vision_scene2.launch",
                    "seed:={}".format(int(seed)),
                ],
            ).start()
        ]
    return []


# 按相反顺序停止由主脚本拉起的辅助进程。
def _stop_scene_services(services):
    for service in reversed(services):
        service.stop()


# 根据场景配置设置头部相机姿态。
def _apply_head_camera(head_camera):
    if not head_camera.get("enabled", False):
        return
    import rospy

    try:
        from task1_vision.head_control import apply_head_camera_pose_from_params

        apply_head_camera_pose_from_params(
            enabled=True,
            yaw_deg=head_camera.get("yaw_deg", 0.0),
            pitch_deg=head_camera.get("pitch_deg", 8.0),
            settle_time=head_camera.get("settle_time", 1.0),
        )
        rospy.set_param("~head_yaw_deg", float(head_camera.get("yaw_deg", 0.0)))
        rospy.set_param("~head_pitch_deg", float(head_camera.get("pitch_deg", 8.0)))
        rospy.set_param("~skip_init_head_motion", True)
    except Exception as exc:
        rospy.logwarn("task1_vision head camera pose setup failed: %s", exc)


# 动作开始前等待人工确认。
def _wait_for_operator_start():
    import rospy

    if int(NEED_OPERATOR_ENTER) == 0:
        rospy.loginfo(GREEN + "准备完成,1s后自动开始动作" + RESET)
        rospy.sleep(1.0)
        return

    rospy.loginfo(GREEN + "准备完成,按enter开始动作" + RESET)
    try:
        input()
    except EOFError:
        raise RuntimeError("stdin unavailable; cannot confirm action start")


# 配置动作模块只接受深度 RANSAC 顶面拟合结果。
def _configure_task1_vision_source():
    import rospy

    allowed_sources = ["depth_roi_top_plane_fit"]
    rospy.set_param("~task1_vision_allowed_position_sources", allowed_sources)
    rospy.loginfo(
        "challenge_task: task1 vision allowed position sources=%s",
        allowed_sources,
    )


def _run_scene1_task1(seed):
    import rospy

    task1_node = _load_task1_node()
    rospy.loginfo("challenge_task: task1 entry=%s/%s", TASK1_PACKAGE, TASK1_MODULE)
    _configure_task1_vision_source()
    parcel_poses = task1_node.init_from_task1_vision()
    rospy.loginfo(
        GREEN + "task1: fixed initial parcel poses from task1_vision: %s" + RESET,
        ", ".join(sorted(parcel_poses)),
    )
    _wait_for_operator_start()

    ok = task1_node.run_scene1_task(
        seed,
        bag_path=None,
        parcel_poses=parcel_poses,
    )
    if not ok:
        raise RuntimeError("task1 action failed")


def _run_scene2_task2(seed):
    import rospy

    task2_node = _load_task2_node()
    rospy.loginfo("challenge_task: task2 entry=%s/%s", TASK2_PACKAGE, TASK2_MODULE)
    object_poses = task2_node.init_from_task2_vision()
    rospy.loginfo(
        GREEN + "task2: fixed initial object poses from task2_vision: %s" + RESET,
        ", ".join(sorted(object_poses)),
    )
    _wait_for_operator_start()
    if not task2_node.run_scene2_task(seed, object_poses=object_poses):
        raise RuntimeError("task2 action failed")


def run_scene(
    scene, seed, node_name=None, timeout=120, time_limit=None, timer_gui=True
):
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

    rospy.loginfo("=== %s任务启动 ===", config["title"])

    services = _start_scene_services(scene, seed)
    try:
        _apply_head_camera(config.get("head_camera", {}))
        rospy.sleep(1.0)
        rospy.loginfo("场景实例已初始化。")

        if scene == "scene1":
            _run_scene1_task1(seed)
            rospy.loginfo("task1 动作流程结束。")
            return

        if scene == "scene2":
            _run_scene2_task2(seed)
            rospy.loginfo("task2 动作流程结束。")
            return

        rospy.spin()
    finally:
        _stop_scene_services(services)


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_CONFIGS),
        default="scene1",
        help="要启动的比赛场景",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="场景种子；正式评测 seed 由组委会指定"
    )
    parser.add_argument(
        "--node-name", default=None, help="ROS 节点名；默认按 scene 自动设置"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="等待仿真就绪的超时时间，单位秒"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时",
    )
    parser.add_argument(
        "--no-timer-gui",
        action="store_true",
        help="不弹出计时器窗口，仅保留后台计时日志",
    )
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
