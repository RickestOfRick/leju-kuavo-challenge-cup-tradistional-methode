#!/usr/bin/env python3
"""任务二运动流程的唯一 ROS 节点入口。"""

import argparse
import os
import sys

import rospy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from task2_scheduler import run_scene2_task
import task2_perception


def init_from_task2_vision():
    return task2_perception.wait_for_task2_objects()


def main():
    parser = argparse.ArgumentParser(description="Run Challenge Cup Task 2")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("task2_motion")
    raise SystemExit(0 if run_scene2_task(arguments.seed) else 1)


if __name__ == "__main__":
    main()
