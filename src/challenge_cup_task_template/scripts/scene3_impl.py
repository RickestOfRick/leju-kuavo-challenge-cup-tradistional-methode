#!/usr/bin/env python3
"""场景三：SMT 料盘出库 — YOLO检测 + IK抓取。"""
import math
import os
from datetime import datetime

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from kuavo_msgs.msg import lejuClawCommand, twoArmHandPoseCmd
from kuavo_msgs.srv import (changeArmCtrlMode, changeArmCtrlModeRequest,
                             twoArmHandPoseCmdSrv)
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState, PointCloud2
from sensor_msgs.point_cloud2 import read_points
from tf.transformations import quaternion_from_euler


def deg(r):
    return r * 180.0 / math.pi


def now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ══════════════════════════════════════════════════════════════
class TaskLogger:
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "..", ".scratch", "scene3", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "scene3_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._file = open(self._path, "w")
    def _log(self, level, msg):
        line = "[%s] [%s] %s" % (now_str(), level, msg)
        rospy.loginfo(line)
        self._file.write(line + "\n")
        self._file.flush()
    def state_enter(self, name):
        self._log("STATE", "进入: %s" % name)
    def state_exit(self, name, ok):
        self._log("STATE", "退出: %s (%s)" % (name, "成功" if ok else "失败"))
    def sensor(self, l, d):
        self._log("SENSOR", "%s = %s" % (l, d))
    def decision(self, m):
        self._log("DECISION", m)
    def error(self, m):
        self._log("ERROR", m)
    def summary(self, c, t):
        self._log("SUMMARY", "完成 %d/%d" % (c, t))
    def close(self):
        self._file.close()


# ══════════════════════════════════════════════════════════════
class SensorBuffer:
    Z_MIN, Z_MAX, X_SELF = 0.3, 2.0, 0.3
    def __init__(self):
        self.lidar_points = None
        self.head_rgb = None
        self.head_depth = None
        self.head_camera_info = None
        self.wrist_r_rgb = None
        self._bridge = CvBridge()
    def _on_lidar(self, msg):
        pts = read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.lidar_points = np.array(list(pts), dtype=np.float32)
    def _on_head_rgb(self, msg):
        self.head_rgb = msg
    def _on_head_depth(self, msg):
        self.head_depth = msg
    def _on_head_cam_info(self, msg):
        self.head_camera_info = msg
    def _on_wrist_r_rgb(self, msg):
        self.wrist_r_rgb = msg
    def head_rgb_cv(self):
        if self.head_rgb is None:
            return None
        try:
            return self._bridge.compressed_imgmsg_to_cv2(self.head_rgb)
        except Exception:
            return None
    def head_depth_cv(self):
        if self.head_depth is None:
            return None
        try:
            import cv2
            raw = self.head_depth.data
            p = raw.find(b'\x89PNG')
            if p < 0:
                return None
            arr = np.frombuffer(raw[p:], dtype=np.uint8)
            d = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if d is None:
                c = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if c is not None and len(c.shape) == 3:
                    d = c[:, :, 2].astype(np.uint16) * 256 + c[:, :, 1].astype(np.uint16)
            return d
        except Exception:
            return None
    def _filter_env(self, pts, x_min=None, y_range=None):
        if pts is None or len(pts) == 0:
            return np.array([]).reshape(0, 3)
        m = (pts[:, 2] > self.Z_MIN) & (pts[:, 2] < self.Z_MAX)
        if x_min is not None:
            m &= pts[:, 0] > x_min
        if y_range is not None:
            m &= np.abs(pts[:, 1]) < y_range
        return pts[m]
    def lidar_detect_rack(self):
        if self.lidar_points is None or len(self.lidar_points) == 0:
            return None
        env = self._filter_env(self.lidar_points, x_min=self.X_SELF, y_range=1.5)
        if len(env) < 100:
            return None
        x_vals = env[:, 0]
        bins = np.arange(0.3, 4.1, 0.1)
        hist, edges = np.histogram(x_vals, bins=bins)
        if hist.max() < 20:
            return None
        thr = hist.max() * 0.3
        for i in range(len(hist)):
            if hist[i] >= thr:
                pi = i
                break
        else:
            pi = np.argmax(hist)
        rx = (edges[pi] + edges[pi + 1]) / 2.0
        rp = env[(x_vals >= edges[pi]) & (x_vals < edges[pi + 1])]
        y_vals = rp[:, 1]
        yh, ye = np.histogram(y_vals, bins=15, range=(-1.0, 1.0))
        top2 = np.argsort(yh)[-2:]
        py = sorted([(ye[i] + ye[i + 1]) / 2.0 for i in top2])
        yc = (py[0] + py[-1]) / 2.0 if len(py) >= 2 else float(np.median(y_vals))
        return {"x_dist": float(rx), "y_center": yc, "confidence": float(hist[pi])}


# ══════════════════════════════════════════════════════════════
class Scene3Controller:
    # ── 物理常量 ──
    TORSO_Z = 0.90          # torso 原点离地高度 (m)
    CAM_X = 0.12            # 头部相机在 torso 坐标系的前向偏移 (m)
    RACK_STANDOFF = 0.35    # 导航到货架前的停靠距离 (m)

    # ── 手臂运动参数 ──
    SAFE_Z = 0.38            # 安全抬升高度（torso 坐标系），确保远离货架
    PRE_GRASP_DZ = 0.10      # 预抓取高度：目标上方距离 (m)，手腕对准保持此高度
    DESCEND_DURATION = 6.0   # 对准后下降用时 (s)
    MOVE_DURATION = 3.0      # 一般手臂移动用时 (s)

    # ── 手腕摄像头 servo 参数（夹取定位的唯一依据）──
    WRIST_TARGET_CX_FRAC = 0.50  # 目标：托盘在图像中的水平位置（0~1，0.5=居中）
    WRIST_TARGET_CY_FRAC = 0.50  # 目标：托盘在图像中的垂直位置（0~1，0.5=居中）
    WRIST_PIXEL_TO_METER = 0.0003  # 像素→米 粗略转换（10cm 距离）
    WRIST_SERVO_GAIN = 0.6      # servo 增益（0~1，越小越保守）
    WRIST_ALIGN_THRESHOLD_PX = 12  # 对准阈值 (px)，小于此值认为已对准
    WRIST_MAX_SERVO_ITER = 5    # 最大 servo 迭代次数
    WRIST_SEARCH_RADIUS = 0.04  # 搜索半径 (m)，找不到托盘时小幅搜索

    def __init__(self, cmd_vel_pub, arm_traj_pub):
        self.cmd_vel_pub = cmd_vel_pub
        self.arm_traj_pub = arm_traj_pub
        self.log = TaskLogger()
        self.sensors = SensorBuffer()
        self._state = None
        self._tray_order = [
            {"idx": i, "name": "smt_tray_%d" % i, "approx_z": z}
            for i, z in [(1, 0.75), (2, 0.75), (3, 1.15), (4, 1.15), (5, 1.15)]
        ]
        self._current_tray_idx = 0
        self._grasped_count = 0

        self._claw_pub = rospy.Publisher("/leju_claw_command", lejuClawCommand, queue_size=10)

    # ── init ──
    def _init_services(self):
        rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=10.0)
        self._ik = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)
        rospy.wait_for_service("/humanoid_change_arm_ctrl_mode", timeout=10.0)
        ctrl = rospy.ServiceProxy("/humanoid_change_arm_ctrl_mode", changeArmCtrlMode)
        ctrl(changeArmCtrlModeRequest(control_mode=2))
        rospy.loginfo("IK 服务就绪，臂模式 external_control")

    def _init_subs(self):
        rospy.Subscriber("/lidar/points", PointCloud2, self.sensors._on_lidar, queue_size=1)
        rospy.Subscriber("/cam_h/color/image_raw/compressed", CompressedImage, self.sensors._on_head_rgb, queue_size=1)
        rospy.Subscriber("/cam_h/depth/image_raw/compressedDepth", CompressedImage, self.sensors._on_head_depth, queue_size=1)
        rospy.Subscriber("/cam_h/color/camera_info", CameraInfo, self.sensors._on_head_cam_info, queue_size=1)
        rospy.Subscriber("/cam_r/color/image_raw/compressed", CompressedImage, self.sensors._on_wrist_r_rgb, queue_size=1)

    # ── walk ──
    def _stop(self):
        self.cmd_vel_pub.publish(Twist())
    def _walk(self, x=0.0, y=0.0, az=0.0):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = x, y, az
        self.cmd_vel_pub.publish(t)

    # ── arm ──
    def _move_arm(self, pos_torso, rpy=None, duration=2.5):
        """pos_torso: [x=前, y=左, z=上] in torso frame. duration: 轨迹发布时长(秒)"""
        if rpy is None:
            rpy = [-math.pi / 2, 0.0, math.pi / 2]
        q = list(quaternion_from_euler(*rpy))
        req = twoArmHandPoseCmd()
        req.hand_poses.right_pose.pos_xyz = list(map(float, pos_torso))
        req.hand_poses.right_pose.quat_xyzw = q
        req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.left_pose.pos_xyz = [0.0, 0.3, -0.5]
        req.hand_poses.left_pose.quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        try:
            resp = self._ik(req)
        except Exception as e:
            self.log.error("IK服务失败: %s" % e)
            return False
        if not resp.success:
            self.log.error("IK求解失败: %s" % resp.error_reason)
            return False
        js = JointState()
        js.name = [
            "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
            "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
            "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
            "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
        ]
        js.position = [deg(x) for x in resp.q_arm]
        iterations = max(40, int(duration / 0.02))
        for _ in range(iterations):
            js.header.stamp = rospy.Time.now()
            self.arm_traj_pub.publish(js)
            rospy.sleep(0.02)
        return True

    def _retract_arm(self):
        js = JointState()
        js.name = [
            "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
            "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
            "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
            "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
        ]
        js.position = [10, 0, 0, -20, 0, 0, 0] * 2
        for _ in range(15):
            js.header.stamp = rospy.Time.now()
            self.arm_traj_pub.publish(js)
            rospy.sleep(0.02)
        rospy.sleep(1.0)

    def _claw(self, position_pct, velocity=40):
        """position_pct: 0=开, 100=闭. velocity: 夹爪速度(默认40, 抓取时用更低速度)"""
        msg = lejuClawCommand()
        msg.data.name = ["left_claw", "right_claw"]
        msg.data.position = [float(position_pct), float(position_pct)]
        msg.data.velocity = [int(velocity), int(velocity)]
        msg.data.effort = [1.0, 1.0]
        self._claw_pub.publish(msg)
        rospy.sleep(1.0)

    def _claw_state_str(self):
        return "no_sub"  # 简化版不订阅夹爪状态

    # ── state machine ──
    def _go(self, ns):
        if self._state:
            self.log.state_exit(self._state, True)
        self._state = ns
        self.log.state_enter(ns)

    def run(self):
        self._init_services()
        self._init_subs()
        rospy.sleep(2.0)

        if self.sensors.lidar_points is None:
            self.log.error("LiDAR 缺失")
        if self.sensors.wrist_r_rgb is None:
            self.log.error("手腕摄像头缺失")
        self.log.decision("传感器检查完成")

        self._go("INIT")
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self._tick()
            self._show_cameras()
            rate.sleep()
        self._stop()
        self.log.close()

    def _show_cameras(self):
        """每5帧更新一次相机画面，降低开销。"""
        cnt = getattr(self, '_cam_cnt', 0) + 1
        self._cam_cnt = cnt
        if cnt % 4 != 0:
            return
        import cv2
        h_img = self.sensors.head_rgb_cv()
        w_img = None
        if self.sensors.wrist_r_rgb is not None:
            try:
                w_img = self.sensors._bridge.compressed_imgmsg_to_cv2(self.sensors.wrist_r_rgb)
            except Exception:
                pass
        th = 280
        if h_img is not None:
            h_img = cv2.resize(h_img, (int(th * h_img.shape[1] / h_img.shape[0]), th))
        else:
            h_img = np.zeros((th, th, 3), dtype=np.uint8)
        if w_img is not None:
            w_img = cv2.resize(w_img, (int(th * w_img.shape[1] / w_img.shape[0]), th))
        else:
            w_img = np.zeros((th, th, 3), dtype=np.uint8)
        cv2.putText(h_img, "Head", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(w_img, "Wrist", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Cameras", np.hstack([h_img, w_img]))
        cv2.waitKey(1)

    def _tick(self):
        s = self._state or "INIT"
        dispatch = {
            "INIT": self._tick_init,
            "TURN_TO_RACK": self._tick_turn,
            "NAV_TO_RACK": self._tick_nav,
            "WRIST_APPROACH": self._tick_wrist_approach,
            "WRIST_ALIGN": self._tick_wrist_align,
            "GRASP": self._tick_grasp,
            "NEXT_TRAY": self._tick_next,
            "DONE": self._tick_done,
        }
        dispatch.get(s, lambda: None)()

    # ── INIT ──
    def _tick_init(self):
        self._retract_arm()
        self._claw(30)  # 半开
        self._stop()
        rospy.sleep(1.0)
        self.log.decision("初始化完成")
        self._go("TURN_TO_RACK")

    # ── TURN ──
    def _tick_turn(self):
        self.log.decision("转身面朝货架")
        elapsed = 0.0
        while elapsed < 6.0 and not rospy.is_shutdown():
            rack = self.sensors.lidar_detect_rack()
            if rack is not None and rack["confidence"] > 20:
                self.log.sensor("turn", "x=%.2f" % rack["x_dist"])
                break
            self._walk(az=0.5)
            rospy.sleep(0.2)
            elapsed += 0.2
        self._stop()
        rospy.sleep(0.5)
        self._go("NAV_TO_RACK")

    # ── NAV ──
    def _tick_nav(self):
        target = self.RACK_STANDOFF
        rack = self.sensors.lidar_detect_rack()
        if rack is None:
            self._walk(x=0.08)
            return
        dist, y_off = rack["x_dist"], rack["y_center"]
        prev = getattr(self, '_nav_ema', None)
        a = 0.4
        self._nav_ema = dist if prev is None else a * dist + (1 - a) * prev
        smooth = self._nav_ema
        cnt = getattr(self, '_nav_tick', 0) + 1
        self._nav_tick = cnt
        if cnt % 10 == 0:
            self.log.sensor("nav", "d=%.2f (target %.2f)" % (smooth, target))
        if smooth > target + 0.15:
            self._walk(x=max(0.06, min(0.15, (smooth - target) * 0.3)))
        elif smooth < target - 0.12:
            self._walk(x=-0.06)
        else:
            if abs(y_off) > 0.12:
                self._walk(y=0.04 if y_off > 0 else -0.04)
                rospy.sleep(0.5)
                self._stop()
                return
            self._stop()
            self._nav_ema = None
            self._nav_tick = 0
            self.log.decision("到达货架前 d=%.2f" % smooth)
            self._go("WRIST_APPROACH")

    # ── WRIST_APPROACH (用 LiDAR + 货架高度 粗定位，手臂移到预抓取高度) ──
    def _tick_wrist_approach(self):
        """手臂抬升→前伸→预降，全部用 LiDAR + 预定义高度。手腕摄像头接管后续精定位。"""
        tray = self._tray_order[self._current_tray_idx]
        tray_z_world = tray["approx_z"]  # 货架层绝对高度

        # 从 LiDAR 获取货架距离
        rack = self.sensors.lidar_detect_rack()
        if rack is not None:
            rack_dist = rack["x_dist"]
            rack_y = rack["y_center"]
        else:
            rack_dist = 0.7
            rack_y = 0.0
            self.log.sensor("approach", "LiDAR 无数据，用默认距离 %.2f" % rack_dist)

        # 粗定位：手臂往前伸到货架位置，高度由货架层决定
        torso_x = self.CAM_X + rack_dist + 0.02   # 货架前方 2cm
        torso_y = -rack_y                           # 货架中心（左右）
        torso_z = tray_z_world - self.TORSO_Z + self.PRE_GRASP_DZ  # 预抓取高度

        self.log.sensor("approach", "rack_dist=%.2f y=%.2f → arm=(%.3f, %.3f, %.3f)" % (
            rack_dist, rack_y, torso_x, torso_y, torso_z))

        # 保存给 wrist_align 用
        self._approach_x = torso_x
        self._approach_y = torso_y
        self._approach_z_grasp = tray_z_world - self.TORSO_Z  # 最终抓取高度（去掉预抓取偏移）

        step = getattr(self, '_app_step', 0)
        if step == 0:
            self.log.decision("1/3 抬升至安全高度 z=%.2f" % self.SAFE_Z)
            if self._move_arm([0.25, -0.15, self.SAFE_Z], duration=self.MOVE_DURATION):
                rospy.sleep(2.0)
                self._app_step = 1
            else:
                self._go("NEXT_TRAY")
                self._app_step = 0
        elif step == 1:
            self.log.decision("2/3 前伸至货架前上方 (%.3f, %.3f, %.3f)" % (torso_x, torso_y, self.SAFE_Z))
            if self._move_arm([torso_x, torso_y, self.SAFE_Z], duration=self.MOVE_DURATION):
                rospy.sleep(2.0)
                self._app_step = 2
            else:
                self._go("NEXT_TRAY")
                self._app_step = 0
        else:
            pre_grasp = [torso_x, torso_y, torso_z]
            self.log.decision("3/3 预降至抓取上方 %.0fcm → 交给手腕摄像头" % (self.PRE_GRASP_DZ * 100))
            if self._move_arm(pre_grasp, duration=self.MOVE_DURATION):
                rospy.sleep(3.0)
                self._app_step = 0
                self._go("WRIST_ALIGN")
            else:
                self._go("NEXT_TRAY")
                self._app_step = 0

    # ── 手腕摄像头 ──
    def _wrist_detect(self):
        """手腕摄像头 YOLO 检测，返回 (bbox_cx, bbox_cy, conf, img_w, img_h) 或 None"""
        import cv2
        if self.sensors.wrist_r_rgb is None:
            return None
        try:
            img = self.sensors._bridge.compressed_imgmsg_to_cv2(self.sensors.wrist_r_rgb)
        except Exception:
            return None
        if img is None or img.size == 0:
            return None

        if not hasattr(self, '_wrist_yolo'):
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tray_model.pt")
            self._wrist_yolo = YOLO(model_path)
            self.log.decision("手腕 YOLO 模型已加载")

        results = self._wrist_yolo(img, conf=0.25, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        best = max(boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        conf = float(best.conf[0])
        h, w = img.shape[:2]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy, conf, w, h)

    # ── WRIST_ALIGN (手腕摄像头 visual servoing，夹取定位的唯一依据) ──
    def _tick_wrist_align(self):
        """
        完全基于手腕摄像头的 visual servoing:
        1. 检测托盘在图像中的位置
        2. 计算与目标位置的像素误差
        3. 像素误差 → 手臂 XY 调整量
        4. 迭代直到托盘在目标位置（对准）
        5. 对准后缓慢下降到抓取高度
        """
        servo_iter = getattr(self, '_servo_iter', 0)

        if servo_iter == 0:
            # 初始化：从 approach 阶段获取粗定位
            self._servo_x = self._approach_x
            self._servo_y = self._approach_y
            self._servo_z_grasp = self._approach_z_grasp
            self._servo_search_phase = 0
            self.log.decision("手腕 servo 开始: 初始 (%.3f, %.3f), 抓取高度 z=%.3f" % (
                self._servo_x, self._servo_y, self._servo_z_grasp))

        # ── 手腕摄像头检测 ──
        wrist = self._wrist_detect()

        if wrist is None:
            # 没检测到托盘 → 小范围搜索
            search_phase = getattr(self, '_servo_search_phase', 0)
            search_offsets = [
                (0, 0), (self.WRIST_SEARCH_RADIUS, 0), (-self.WRIST_SEARCH_RADIUS, 0),
                (0, self.WRIST_SEARCH_RADIUS), (0, -self.WRIST_SEARCH_RADIUS),
                (self.WRIST_SEARCH_RADIUS, self.WRIST_SEARCH_RADIUS),
                (-self.WRIST_SEARCH_RADIUS, -self.WRIST_SEARCH_RADIUS),
            ]
            if search_phase < len(search_offsets):
                dx, dy = search_offsets[search_phase]
                self._servo_search_phase = search_phase + 1
                sx = self._servo_x + dx
                sy = self._servo_y + dy
                pre_z = self._servo_z_grasp + self.PRE_GRASP_DZ
                self.log.decision("手腕搜索 %d: 尝试 (%.3f, %.3f)" % (search_phase, sx, sy))
                self._move_arm([sx, sy, pre_z], duration=1.5)
                rospy.sleep(1.5)
            else:
                self.log.error("手腕搜索失败，跳过此料盘")
                self._servo_iter = 0
                self._go("NEXT_TRAY")
            return

        cx, cy, conf, iw, ih = wrist
        self._servo_search_phase = 0  # 找到了，重置搜索

        # ── 计算像素误差 ──
        target_cx = iw * self.WRIST_TARGET_CX_FRAC
        target_cy = ih * self.WRIST_TARGET_CY_FRAC
        err_px = cx - target_cx   # 正=托盘偏右
        err_py = cy - target_cy   # 正=托盘偏下

        self.log.sensor("wrist", "tray@(%d,%d) target@(%d,%d) err=(%+.0f,%+.0f)px conf=%.2f" % (
            int(cx), int(cy), int(target_cx), int(target_cy), err_px, err_py, conf))

        # ── 像素误差 → 手臂调整量 (visual servoing) ──
        # 图像 x (右) → torso y (左), 所以 torso_y 调整 = -err_px
        # 图像 y (下) → torso x (前), 所以 torso_x 调整 = +err_py
        adj_y = -err_px * self.WRIST_PIXEL_TO_METER * self.WRIST_SERVO_GAIN
        adj_x = +err_py * self.WRIST_PIXEL_TO_METER * self.WRIST_SERVO_GAIN

        self._servo_iter += 1

        # ── 判断是否对准 ──
        aligned = abs(err_px) < self.WRIST_ALIGN_THRESHOLD_PX and abs(err_py) < self.WRIST_ALIGN_THRESHOLD_PX

        if aligned or self._servo_iter > self.WRIST_MAX_SERVO_ITER:
            reason = "已对准" if aligned else "达到最大迭代次数"
            self.log.decision("手腕 servo 完成 (%s, %d 次迭代)，慢速下降 %.0fcm" % (
                reason, self._servo_iter, self.PRE_GRASP_DZ * 100))

            # ── 对准后，极慢下降到抓取高度 ──
            self._move_arm([self._servo_x, self._servo_y, self._servo_z_grasp],
                           duration=self.DESCEND_DURATION)
            rospy.sleep(3.0)

            self._servo_iter = 0
            self._go("GRASP")
        else:
            # ── 微调位置（保持预抓取高度，不下降！）──
            self._servo_x += adj_x
            self._servo_y += adj_y
            pre_z = self._servo_z_grasp + self.PRE_GRASP_DZ
            self.log.decision("手腕 servo %d/%d: err=(%+.0f,%+.0f)px → adj=(%+.4f,%+.4f)m → (%.3f, %.3f)" % (
                self._servo_iter, self.WRIST_MAX_SERVO_ITER,
                err_px, err_py, adj_x, adj_y, self._servo_x, self._servo_y))
            self._move_arm([self._servo_x, self._servo_y, pre_z], duration=1.5)
            rospy.sleep(1.5)

    # ── GRASP ──
    def _tick_grasp(self):
        tray_name = self._tray_order[self._current_tray_idx]["name"]

        # 1) 极慢速闭合夹爪
        self.log.decision("闭合夹爪抓取 %s" % tray_name)
        self._claw(90, velocity=15)
        rospy.sleep(3.5)

        # 2) 先竖直抬升到安全高度（绝对不能横向移动，防止撞料盘）
        self.log.decision("竖直抬升到安全高度")
        self._move_arm([0.25, -0.15, self.SAFE_Z], duration=self.MOVE_DURATION)
        rospy.sleep(2.5)

        # 3) 收回手臂
        self._retract_arm()
        rospy.sleep(2.0)

        self._grasped_count += 1
        self.log.decision("已抓取 %s (%d/%d)" % (tray_name, self._grasped_count, len(self._tray_order)))
        self._go("NEXT_TRAY")

    # ── NEXT ──
    def _tick_next(self):
        self._current_tray_idx += 1
        if self._current_tray_idx >= len(self._tray_order):
            self._go("DONE")
        else:
            self._go("WRIST_APPROACH")

    # ── DONE ──
    def _tick_done(self):
        self._stop()
        self._retract_arm()
        self._claw(30)
        self.log.summary(self._grasped_count, len(self._tray_order))
        self.log.close()
        rospy.signal_shutdown("完成")
