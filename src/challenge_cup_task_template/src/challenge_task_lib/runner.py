#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""顶层任务编排：组装感知/硬件/视觉/操作层，按场景分发执行。

入口脚本 scripts/challenge_task.py 在受保护启动流程跑完、ROS 节点就绪后，
调用本模块的 execute_task()。这样可编辑区只剩“导入 + 调用”，逻辑全在包内。
"""

import rospy

from . import config, scenes
from .perception import Perception
from .hardware import Hardware
from .vision import Vision
from .manipulation import Manipulation


def execute_task(scene, title=None, head_cam_frame=None):
    """构建各层对象并执行指定场景任务。

    参数:
        scene:          "scene1" | "scene2" | "scene3"
        title:          日志用场景标题(可选)
        head_cam_frame: 头相机光学坐标系帧名(可选，缺省读 config)
    """
    title = title or scene
    if scene not in scenes.SCENE_DISPATCH:
        rospy.logerr("未知场景: %s", scene)
        return

    perception = Perception()
    hardware = Hardware(perception)
    vision = Vision(perception, hardware, head_cam_frame=head_cam_frame)
    manip = Manipulation(perception, hardware, vision)

    # 等关键服务/数据就绪：仿真启动早期 IK 节点仍在编译 cppad，过早开始会全程失败。
    _wait_until_ready(perception, hardware, timeout=90.0)
    rospy.loginfo("=== %s任务开始 ===", title)

    # 开局预设抬手：把双臂从垂手抬到身前预设位，稳定重心防摔倒并给 IK 良好初值。
    hardware.move_home()

    try:
        scenes.SCENE_DISPATCH[scene](manip)
        rospy.loginfo("=== %s任务流程结束 ===", title)
    except rospy.ROSInterruptException:
        pass
    finally:
        hardware.stop_base()

    rospy.spin()


def _wait_until_ready(perception, hardware, timeout=90.0):
    """轮询等待 IK 服务可用 + 首帧相机数据到达，最多等 timeout 秒。"""
    end = rospy.Time.now() + rospy.Duration(timeout)
    rate = rospy.Rate(2)
    ik_ok = cam_ok = False
    while not rospy.is_shutdown() and rospy.Time.now() < end:
        if not ik_ok and hardware.ik_srv is not None and hardware.fk_srv is not None:
            ik_ok = True
            rospy.loginfo("就绪检测: IK/FK 服务已连接")
        snap = perception.snapshot()
        if not cam_ok and snap["color"] is not None and snap["K"] is not None:
            cam_ok = True
            rospy.loginfo("就绪检测: 头相机彩色+内参已到达")
        if ik_ok and cam_ok:
            rospy.loginfo("就绪检测: 全部就绪")
            return
        rate.sleep()
    rospy.logwarn("就绪检测超时(IK=%s, 相机=%s)，仍继续尝试执行", ik_ok, cam_ok)
