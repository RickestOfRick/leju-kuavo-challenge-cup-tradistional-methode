#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""操作层：把感知、视觉、硬件组合成抓取/称重等高层动作宏（场景一）。

抓取走官方分步：横移到上方 → 下降到抓取高度 → 0x03 硬解夹紧 → 抬起。
坐标均为 IK 系绝对坐标（≈base_link），视觉直接给出。
"""

import rospy

from . import config


class Manipulation(object):
    """高层抓放动作。"""

    def __init__(self, perception, hardware, vision):
        self.perception = perception
        self.hw = hardware
        self.vision = vision

    @staticmethod
    def _pick_offset_for(pos_ik):
        """按目标 y 所在分区给抓取点做小偏移，避免夹在长边端头或压深不足。"""
        if pos_ik is None:
            return (0.0, 0.0, 0.0)
        if pos_ik[1] > config.RIGHT_PICK_NEAR_FAR_Y_THRESHOLD:
            return config.RIGHT_PICK_OFFSET_NEAR_ROW
        return config.RIGHT_PICK_OFFSET_FAR_ROW

    @staticmethod
    def _claw_state_ok(claw_state, side, min_state):
        if side == "left":
            return len(claw_state) >= 1 and claw_state[0] >= min_state
        if side == "right":
            return len(claw_state) >= 2 and claw_state[1] >= min_state
        return all(s >= min_state for s in claw_state)

    # ---------------- 搜索：低头扫描定位 ----------------
    def scan_locate(self, color_name):
        """低头保持低头姿态，转脖子扫桌面找指定颜色物体。

        返回 IK 系 (x,y,z) 或 None。不走动、不转身（机器人初始就在桌前）。
        """
        self.hw.look_down()
        for yaw in config.HEAD_SCAN_YAWS:
            if rospy.is_shutdown():
                return None
            self.hw.move_head(yaw, config.HEAD_LOOK_DOWN[1])  # 保持低头，仅改偏航
            rospy.sleep(0.5)                                   # 等图像稳定
            pos = self.vision.locate_object(color_name)
            if pos is not None:
                rospy.loginfo("发现 %s @ IK(%.3f, %.3f, %.3f)",
                              color_name, pos[0], pos[1], pos[2])
                return pos
        rospy.logwarn("低头扫描未发现 %s", color_name)
        return None

    # ---------------- 右手抓取 ----------------
    def pick(self, pos_ik, side="right"):
        """对 IK 系坐标执行抓取：横移到上方 → 下降 → 硬解夹紧 → 抬起。"""
        if pos_ik is None:
            rospy.logwarn("pick: 目标位置为空")
            return False
        ox, oy, oz = self._pick_offset_for(pos_ik)
        x, y, _z = pos_ik[0] + ox, pos_ik[1] + oy, pos_ik[2] + oz
        first = config.RIGHT_PICK_FIRST_YPR_DEG
        second = config.RIGHT_PICK_SECOND_YPR_DEG

        self.hw.set_arm_mode(config.ARM_MODE_EXTERNAL)
        self.hw.gripper(close=False)
        self.hw.look_forward()   # 抓取前头回正，避免遮挡

        # 1) 横移到抓取上方（用固定横移高度，xy 取视觉值）
        if not self.hw.move_eef_to(side, (x, y, config.RIGHT_PICK_TRANSIT_IK_Z),
                                   first, second,
                                   constraint_mode=config.RIGHT_PICK_TRANSIT_CONSTRAINT_MODE):
            rospy.logwarn("pick: 横移到上方 IK 无解")
            return False
        # 2) 下降到抓取高度（z 用标定基准，不用视觉 z，避免深度噪声扎太深）
        if not self.hw.move_eef_to(side, (x, y, config.RIGHT_PICK_IK_Z),
                                   first, second,
                                   constraint_mode=config.RIGHT_PICK_FINAL_CONSTRAINT_MODE):
            rospy.logwarn("pick: 下降到抓取高度 IK 无解")
            return False
        # 3) 硬解夹紧
        self.hw.gripper(close=True)
        self.hw.wait_claw_settled()
        # 4) 抬起
        self.hw.move_eef_to(side, (x, y, config.RIGHT_PICK_IK_Z + config.LIFT_Z_OFFSET),
                            first, second,
                            constraint_mode=config.RIGHT_PICK_TRANSIT_CONSTRAINT_MODE)
        return self._claw_state_ok(
            self.perception.snapshot()["claw"], side, config.CLAW_CLOSE_MIN_STATE
        )

    # ---------------- 放到称重区（A 阶段终点）----------------
    def place_weigh(self, side="right"):
        """把已抓起的快递放到称重区：移到称重点上方 → 下降释放 → 开爪停留。"""
        self.hw.set_arm_mode(config.ARM_MODE_EXTERNAL)
        wx, wy, wz = config.WEIGH_RELEASE_IK
        first = config.WEIGH_RELEASE_FIRST_YPR_DEG
        second = config.WEIGH_RELEASE_SECOND_YPR_DEG

        # 1) 移到称重点上方（高位）
        if not self.hw.move_eef_to(side, (wx, wy, config.WEIGH_TRANSIT_IK_Z),
                                   first, second):
            rospy.logwarn("place_weigh: 移到称重上方 IK 无解")
            return False
        # 2) 下降到释放点
        if not self.hw.move_eef_to(side, (wx, wy, wz), first, second):
            rospy.logwarn("place_weigh: 下降到释放点 IK 无解")
            return False
        # 3) 释放并停留（让称重判定生效）
        self.hw.gripper(close=False, side=side)
        self.hw.wait_claw_settled()
        rospy.sleep(config.WEIGH_DWELL)
        return True

    # ---------------- B 阶段：二次夹起 ----------------
    def regrasp(self, side="right"):
        """把称重区的快递重新夹起：高位对齐 → 下降到二次夹起点 → 闭爪 → 抬起。"""
        self.hw.set_arm_mode(config.ARM_MODE_EXTERNAL)
        rx, ry, rz = config.WEIGH_REGRASP_IK
        first = config.WEIGH_REGRASP_FIRST_YPR_DEG
        second = config.WEIGH_REGRASP_SECOND_YPR_DEG
        # 高位对齐（用称重释放高位的 z）
        self.hw.move_eef_to(side, (rx, ry, config.WEIGH_RELEASE_IK[2]), first, second)
        # 下降到二次夹起点
        if not self.hw.move_eef_to(side, (rx, ry, rz), first, second):
            rospy.logwarn("regrasp: 下降到二次夹起点 IK 无解")
            return False
        self.hw.gripper(close=True, side=side)
        self.hw.wait_claw_settled()
        # 抬回高位
        self.hw.move_eef_to(side, (rx, ry, config.WEIGH_RELEASE_IK[2]), first, second)
        return self._claw_state_ok(
            self.perception.snapshot()["claw"], side, config.CLAW_CLOSE_MIN_STATE
        )

    # ---------------- B 阶段：右手交给左手 ----------------
    def handoff_to_left(self):
        """右手把快递交给左手：右手抬高位对齐→到交接点；左手侧方对齐→沿 y 接近；
        左手闭爪、右手开爪、右手退让。"""
        self.hw.set_arm_mode(config.ARM_MODE_EXTERNAL)
        self.hw.gripper(close=False, side="left")   # 左手先张开准备接

        r_first = config.RIGHT_HANDOFF_TO_LEFT_FIRST_YPR_DEG
        r_second = config.RIGHT_HANDOFF_TO_LEFT_SECOND_YPR_DEG
        hx, hy, hz = config.RIGHT_HANDOFF_TO_LEFT_IK
        # 右手先抬到避障高位（x/y 取交接点，z 取高位）对齐姿态
        self.hw.move_eef_to("right", (hx, hy, config.RIGHT_HANDOFF_TRANSIT_IK_Z),
                            r_first, r_second)
        # 右手下降到交接点
        if not self.hw.move_eef_to("right", (hx, hy, hz), r_first, r_second):
            rospy.logwarn("handoff: 右手到交接点 IK 无解")
            return False

        l_first = config.LEFT_HANDOFF_RECEIVE_FIRST_YPR_DEG
        l_second = config.LEFT_HANDOFF_RECEIVE_SECOND_YPR_DEG
        # 左手先侧方对齐 x/z，再沿 y 接近右手
        self.hw.move_eef_to("left", config.LEFT_HANDOFF_RECEIVE_XZ_READY_IK,
                            l_first, l_second)
        if not self.hw.move_eef_to("left", config.LEFT_HANDOFF_RECEIVE_IK,
                                   l_first, l_second):
            rospy.logwarn("handoff: 左手接近接收点 IK 无解")
            return False
        # 左手夹住、右手松开
        self.hw.gripper(close=True, side="left")
        self.hw.wait_claw_settled()
        self.hw.gripper(close=False, side="right")
        rospy.sleep(config.GRIPPER_HOLD_TIME)
        # 右手按 y 偏移退让，让出空间
        self.hw.move_eef_to("right",
                            (hx, hy + config.RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET, hz),
                            r_first, r_second)
        return True

    # ---------------- B 阶段：左手放进分拣箱 ----------------
    def drop_in_box(self, index=0):
        """左手把快递放进分拣箱：移到箱上方 → 下降 → 开爪。index 决定 2x2 落点。"""
        self.hw.set_arm_mode(config.ARM_MODE_EXTERNAL)
        bx, by, bz = config.BOX_DROP_IK
        off = config.BOX_DROP_OFFSETS[index % len(config.BOX_DROP_OFFSETS)]
        tx, ty, tz = bx + off[0], by + off[1], bz + off[2]
        first = config.BOX_DROP_FIRST_YPR_DEG
        second = config.BOX_DROP_SECOND_YPR_DEG

        # 移到箱上方（抬高 z 余量）再下降到落点
        if not self.hw.move_eef_to("left", (tx, ty, tz + 0.10), first, second):
            rospy.logwarn("drop_in_box: 移到箱上方 IK 无解")
            return False
        self.hw.move_eef_to("left", (tx, ty, tz), first, second)
        self.hw.gripper(close=False, side="left")
        self.hw.wait_claw_settled()
        rospy.sleep(config.GRIPPER_HOLD_TIME)
        return True
