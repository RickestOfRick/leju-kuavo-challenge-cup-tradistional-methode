#!/usr/bin/env python3
"""任务一流程的唯一 ROS 节点入口。"""

import argparse
import os
import sys

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from task1_scheduler import run_scene1_task
from task1_perception import init_from_task1_vision


def main():
    parser = argparse.ArgumentParser(description="Run Challenge Cup Task 1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("task1")
    if not run_scene1_task(args.seed):
        raise SystemExit(1)
    rospy.loginfo("task1: action complete; keeping ROS node alive until shutdown")
    rospy.spin()


if __name__ == "__main__":
    main()
