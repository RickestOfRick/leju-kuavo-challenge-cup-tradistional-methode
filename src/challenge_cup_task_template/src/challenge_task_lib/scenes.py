#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三场景任务流水线。每个场景一个函数，逻辑清晰、便于单独调试。

位姿/颜色等可标定量集中在 config.py，本文件只描述流程。
当前场景一实现到 A 阶段（右手抓取 + 放称重区）；B 阶段（二次夹起→交接→放箱）
待 A 跑通后接入。场景二/三为基础骨架。
"""

import rospy

from . import config


def run_scene1(manip):
    """场景一完整 5 步：低头扫描 → 右手抓取 → 放称重区 → 二次夹起 →
    交给左手 → 左手放进分拣箱。机器人初始就在桌前，无需走动。
    """
    rospy.loginfo("[scene1] 快递称重与摆放（完整流程）")
    for index, color in enumerate(config.SCENE1_PARCEL_COLORS):
        if rospy.is_shutdown():
            break
        rospy.loginfo("[scene1] 处理快递 %d: %s", index + 1, color)

        # 1) 视觉定位
        pos = manip.scan_locate(color)
        if pos is None:
            continue
        # 2) 右手抓取
        if not manip.pick(pos, side="right"):
            rospy.logwarn("[scene1] %s 抓取失败，跳过", color)
            continue
        # 3) 放到称重区
        if not manip.place_weigh(side="right"):
            rospy.logwarn("[scene1] %s 称重放置失败，跳过", color)
            continue
        # 4) 二次夹起
        if not manip.regrasp(side="right"):
            rospy.logwarn("[scene1] %s 二次夹起失败，跳过", color)
            continue
        # 5) 交给左手 → 左手放进分拣箱
        if not manip.handoff_to_left():
            rospy.logwarn("[scene1] %s 交接失败，跳过", color)
            continue
        manip.drop_in_box(index=index)


def run_scene2(manip):
    """场景二（骨架）：零件分拣归档，颜色即类别。待 scene1 跑通后完善放置。"""
    rospy.loginfo("[scene2] 零件分拣归档（骨架）")
    for color in config.SCENE2_BIN_COLORS:
        for _ in range(config.SCENE2_PER_CLASS):
            if rospy.is_shutdown():
                break
            pos = manip.scan_locate(color)
            if pos is None:
                break
            manip.pick(pos, side="right")
            # ★TODO: 放入对应收纳箱（待标定箱体落点）


def run_scene3(manip):
    """场景三（骨架）：SMT 料盘出库。待 scene1 跑通后完善取放。"""
    rospy.loginfo("[scene3] SMT 料盘出库（骨架）")
    for i in range(config.SCENE3_TRAY_COUNT):
        if rospy.is_shutdown():
            break
        rospy.loginfo("[scene3] 取第 %d/%d 个料盘", i + 1, config.SCENE3_TRAY_COUNT)
        pos = manip.scan_locate(config.SCENE3_TRAY_COLOR)
        if pos is None:
            continue
        manip.pick(pos, side="right")
        # ★TODO: 放到出库区（待标定出库落点 + 高低层处理）


SCENE_DISPATCH = {
    "scene1": run_scene1,
    "scene2": run_scene2,
    "scene3": run_scene3,
}
