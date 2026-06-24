#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""感知层：订阅合法传感器话题，缓存并解码头相机彩色/深度（线程安全）。

只使用机器人自身视角的合法接口，绝不订阅 /mujoco/qpos、/ground_truth/state
等真值接口。所有可选依赖（cv2 等）缺失时降级为空数据，不阻断启动。
"""

import threading

import rospy

from . import config

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    import cv2
    _HAS_CV2 = True
except Exception as exc:
    rospy.logwarn("OpenCV 不可用，视觉降级: %s", exc)
    _HAS_CV2 = False

from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2

try:
    from kuavo_msgs.msg import sensorsData, lejuClawState
    _HAS_KUAVO_MSGS = True
except Exception as exc:
    rospy.logwarn("kuavo_msgs 不可用，传感器/夹爪状态降级: %s", exc)
    _HAS_KUAVO_MSGS = False


class Perception(object):
    """集中缓存合法传感器数据，供决策层只读访问。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.sensor_time = None        # 仿真时间(float, s)，与计时器一致
        self.arm_q = None              # 14 个手臂关节角(rad)
        self.claw_state = [0, 0]       # 2=到位 3=抓到
        self.color_bgr = None          # 头相机彩色(BGR ndarray)
        self.depth_m = None            # 头相机深度(米, ndarray)
        self.cam_K = None              # 3x3 内参
        self.cam_frame = None          # 相机光学坐标系帧名(从 CameraInfo header 自适应读取)
        self.cloud = None              # 最新点云

        if _HAS_KUAVO_MSGS:
            rospy.Subscriber(config.TOPIC_SENSORS, sensorsData,
                             self._on_sensors, queue_size=1)
            rospy.Subscriber(config.TOPIC_CLAW_STATE, lejuClawState,
                             self._on_claw, queue_size=1)
        rospy.Subscriber(config.TOPIC_HEAD_COLOR, CompressedImage,
                         self._on_color, queue_size=1)
        rospy.Subscriber(config.TOPIC_HEAD_DEPTH, CompressedImage,
                         self._on_depth, queue_size=1)
        rospy.Subscriber(config.TOPIC_HEAD_CAMINFO, CameraInfo,
                         self._on_caminfo, queue_size=1)
        rospy.Subscriber(config.TOPIC_LIDAR, PointCloud2,
                         self._on_cloud, queue_size=1)

    # ---- 回调 ----
    def _on_sensors(self, msg):
        with self._lock:
            self.sensor_time = msg.sensor_time.secs + msg.sensor_time.nsecs * 1e-9
            jq = list(msg.joint_data.joint_q)
            if len(jq) >= 26:
                self.arm_q = jq[12:26]   # 12~25 为双臂

    def _on_claw(self, msg):
        with self._lock:
            self.claw_state = list(msg.state)

    def _on_caminfo(self, msg):
        if not _HAS_NUMPY:
            return
        with self._lock:
            self.cam_K = np.array(msg.K, dtype=float).reshape(3, 3)
            if msg.header.frame_id:
                self.cam_frame = msg.header.frame_id

    def _on_color(self, msg):
        if not (_HAS_CV2 and _HAS_NUMPY):
            return
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            with self._lock:
                self.color_bgr = img

    def _on_depth(self, msg):
        # compressedDepth：前 12 字节是 ROS 自定义头，其后为 PNG(16UC1, 毫米)
        if not (_HAS_CV2 and _HAS_NUMPY):
            return
        try:
            raw = msg.data[12:] if len(msg.data) > 12 else msg.data
            depth = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if depth is None:
                return
            depth_m = depth.astype(np.float32)
            if depth.dtype == np.uint16:
                depth_m = depth_m / 1000.0
            with self._lock:
                self.depth_m = depth_m
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "深度解码失败: %s", exc)

    def _on_cloud(self, msg):
        with self._lock:
            self.cloud = msg

    # ---- 只读快照 ----
    def snapshot(self):
        with self._lock:
            return dict(
                t=self.sensor_time,
                arm_q=self.arm_q,
                claw=list(self.claw_state),
                color=None if self.color_bgr is None else self.color_bgr.copy(),
                depth=None if self.depth_m is None else self.depth_m.copy(),
                K=None if self.cam_K is None else self.cam_K.copy(),
                cam_frame=self.cam_frame,
                cloud=self.cloud,
            )
