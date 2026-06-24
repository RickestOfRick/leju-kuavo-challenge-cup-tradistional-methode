#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉层：头相机彩色分割定位物体，结合深度+内参反投影到 base_link 系。

比赛物体无 AprilTag、按颜色区分，因此用 HSV 颜色分割找像素，再用深度图
反投影出 3D 点，最后经 TF 转到机器人基座系。全程只用合法相机/TF 接口。
"""

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
except Exception:
    _HAS_CV2 = False

try:
    import tf2_ros  # noqa: F401  (异常类型用)
    import tf2_geometry_msgs  # noqa: F401  (注册 PointStamped 的 transform)
    from geometry_msgs.msg import PointStamped
    _HAS_TF = True
except Exception:
    _HAS_TF = False


class Vision(object):
    """基于头相机的颜色定位 + 深度反投影。"""

    def __init__(self, perception, hardware, head_cam_frame=None):
        self.perception = perception
        self.hardware = hardware
        self.head_cam_frame = head_cam_frame or config.DEFAULT_HEAD_CAM_OPTICAL_FRAME

    def detect_color_blob(self, snap, color_name, min_area=150):
        """找指定颜色最大连通块，返回 (u, v, area) 或 None。"""
        if not (_HAS_CV2 and _HAS_NUMPY):
            return None
        if snap["color"] is None or color_name not in config.HSV_COLORS:
            return None
        hsv = cv2.cvtColor(snap["color"], cv2.COLOR_BGR2HSV)
        lo, hi = config.HSV_COLORS[color_name]
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return None
        m = cv2.moments(c)
        if m["m00"] == 0:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"], cv2.contourArea(c))

    def pixel_to_base(self, snap, u, v):
        """像素(u,v)+深度+内参 → 相机系 3D → TF 转 base_link，返回 (x,y,z) 或 None。"""
        if snap["depth"] is None or snap["K"] is None or not _HAS_TF:
            return None
        depth, k = snap["depth"], snap["K"]
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iv < depth.shape[0] and 0 <= iu < depth.shape[1]):
            return None
        z = float(depth[iv, iu])
        if not (0.05 < z < 5.0):     # 无效深度
            return None
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        x_cam = (iu - cx) * z / fx
        y_cam = (iv - cy) * z / fy
        # 相机帧名优先用 CameraInfo header 自适应读到的，回退到 config 默认值
        cam_frame = snap.get("cam_frame") or self.head_cam_frame
        try:
            ps = PointStamped()
            ps.header.frame_id = cam_frame
            ps.header.stamp = rospy.Time(0)
            ps.point.x, ps.point.y, ps.point.z = x_cam, y_cam, z
            out = self.hardware.tf_buffer.transform(
                ps, config.BASE_FRAME, rospy.Duration(1.0))
            return (out.point.x, out.point.y, out.point.z)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "TF 反投影失败(cam_frame=%s): %s", cam_frame, exc)
            return None

    def locate_object(self, color_name):
        """颜色定位 + 反投影，返回物体在 base_link 的 (x,y,z) 或 None。"""
        snap = self.perception.snapshot()
        blob = self.detect_color_blob(snap, color_name)
        if blob is None:
            return None
        return self.pixel_to_base(snap, blob[0], blob[1])
