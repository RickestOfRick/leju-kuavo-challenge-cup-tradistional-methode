#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挑战杯任务薄入口。

入口文件只负责解析命令并分发到对应场景实现，方便多人分别维护
scripts/challenge_task_scene*.py。
"""

import argparse
import importlib
import importlib.util
import os
import sys


SCENE_CONFIGS = {
    "scene1": {
        "module": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    # 后续队友可以按同样约定新增：
    # "scene2": {"module": "challenge_task_scene2", "title": "场景二：分拣归档"},
    # "scene3": {"module": "challenge_task_scene3", "title": "场景三：SMT 料盘出库"},
}


def run_scene(scene, seed, node_name=None, timeout=None,
              time_limit=None, timer_gui=True):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    scene_module = _import_scene_module(config["module"])
    return scene_module.run_scene(
        scene=scene,
        seed=seed,
        node_name=node_name,
        timeout=timeout,
        time_limit=time_limit,
        timer_gui=timer_gui,
    )


def _import_scene_module(module_name):
    """Load sibling scene scripts from source, not catkin's devel wrappers."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, module_name + ".py")
    if not os.path.exists(script_path):
        return importlib.import_module(module_name)

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scene module from {}".format(script_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="挑战杯任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认由对应 scene 实现决定")
    parser.add_argument("--timeout", type=float, default=None,
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
