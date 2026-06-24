#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import os
import sys


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

    rospy.loginfo("=== %s任务启动 ===", config["title"])
    rospy.sleep(1.0)
    rospy.loginfo("场景实例已初始化。")

    # ====================================================================
    # 选手任务实现区（本区为允许选手修改的部分）
    # --------------------------------------------------------------------
    # 任务逻辑已拆分到本包内的 Python 子包 challenge_task_lib，按职责分文件：
    #   challenge_task_lib/config.py        所有需标定的常量(★TODO 集中地)
    #   challenge_task_lib/perception.py    合法传感器订阅与缓存
    #   challenge_task_lib/hardware.py      行走/手臂/夹爪/IK 底层 IO
    #   challenge_task_lib/vision.py        颜色定位 + 深度反投影到 base_link
    #   challenge_task_lib/manipulation.py  抓取/放置/搜索动作宏
    #   challenge_task_lib/scenes.py        三场景流水线
    #   challenge_task_lib/runner.py        顶层组装入口 execute_task()
    #
    # 调参时主要改 config.py；扩展逻辑改对应模块。本入口只负责导入并调用。
    # ====================================================================

    # 兼容直接 rosrun（未编译 catkin python 包）的情况：把 ../src 加入路径
    _lib_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    if os.path.isdir(_lib_src) and _lib_src not in sys.path:
        sys.path.insert(0, _lib_src)

    from challenge_task_lib.runner import execute_task

    execute_task(scene=scene, title=config["title"])


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
