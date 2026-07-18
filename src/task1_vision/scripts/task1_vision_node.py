#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
import os
import struct
import time
import xml.etree.ElementTree as ET

import rospy
from geometry_msgs.msg import Point, Quaternion, Vector3
from kuavo_msgs.msg import sensorsData
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Float64MultiArray, String
import tf2_ros
from vision_msgs.msg import (
    BoundingBox2D,
    BoundingBox3D,
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

try:
    import mujoco as mj

    _MUJOCO_AVAILABLE = True
    _MUJOCO_IMPORT_ERROR = None
except Exception as exc:
    mj = None
    _MUJOCO_AVAILABLE = False
    _MUJOCO_IMPORT_ERROR = exc

try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge

    _CV_AVAILABLE = True
    _CV_IMPORT_ERROR = None
except Exception as exc:
    cv2 = None
    np = None
    CvBridge = None
    _CV_AVAILABLE = False
    _CV_IMPORT_ERROR = exc


def _parse_vec(value, length, default=None):
    if value is None:
        return list(default or [0.0] * length)
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = list(value)
    result = [float(v) for v in parts[:length]]
    while len(result) < length:
        result.append(0.0)
    return result


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _quat_from_yaw(yaw):
    half = 0.5 * yaw
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _quat_from_axis_angle(axis, angle):
    norm = math.sqrt(sum(v * v for v in axis))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    half = 0.5 * angle
    scale = math.sin(half) / norm
    return [axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half)]


def _quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_inverse(q):
    x, y, z, w = q
    norm2 = x * x + y * y + z * z + w * w
    if norm2 <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [-x / norm2, -y / norm2, -z / norm2, w / norm2]


def _yaw_from_quat_xyzw(q):
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_wxyz_to_xyzw(q):
    qw, qx, qy, qz = q
    return [qx, qy, qz, qw]


def _rotate_vector_by_quat_xyzw(vec, quat):
    x, y, z = vec
    qx, qy, qz, qw = quat
    # 展开计算四元数 q * v * q^-1，其中向量四元数 v=(x,y,z,0)。
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    return [
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    ]


def _compose_pose(parent_pos, parent_quat, local_pos, local_quat):
    rotated = _rotate_vector_by_quat_xyzw(local_pos, parent_quat)
    return (
        [parent_pos[i] + rotated[i] for i in range(3)],
        _quat_multiply(parent_quat, local_quat),
    )


def _transform_point(transform, point):
    quat = [
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ]
    rotated = _rotate_vector_by_quat_xyzw(point, quat)
    return [
        rotated[0] + transform.translation.x,
        rotated[1] + transform.translation.y,
        rotated[2] + transform.translation.z,
    ]


def _distance(a, b):
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return None
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _assign_parcel_names_by_relative_xy(items):
    """按 base_link 中 x 向上、y 向左的相对关系给四个拟合结果命名。"""
    if len(items) != 4:
        return []

    by_x = sorted(
        items,
        key=lambda item: (
            float(item["fit"]["position"][0]),
            float(item["fit"]["position"][1]),
        ),
    )
    lower = sorted(by_x[:2], key=lambda item: float(item["fit"]["position"][1]))
    upper = sorted(by_x[2:], key=lambda item: float(item["fit"]["position"][1]))
    return [
        ("parcel_1", lower[0]),
        ("parcel_2", lower[1]),
        ("parcel_3", upper[0]),
        ("parcel_4", upper[1]),
    ]


def _quat_to_rotation_matrix(quat):
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _project_camera_point(camera_info, point):
    x, y, z = point
    if z <= 0.03:
        return None
    k = camera_info.K
    if not k or len(k) < 6:
        return None
    u = k[0] * x / z + k[2]
    v = k[4] * y / z + k[5]
    return [u, v]


def _make_point(values):
    return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _make_quaternion(values):
    return Quaternion(
        x=float(values[0]),
        y=float(values[1]),
        z=float(values[2]),
        w=float(values[3]),
    )


def _make_vector3(values):
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _decode_raw_depth_image(msg, bridge):
    if not _CV_AVAILABLE or bridge is None:
        return None, "cv_bridge/cv2 unavailable: {}".format(_CV_IMPORT_ERROR)
    try:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except Exception as exc:
        return None, "raw depth decode failed: {}".format(exc)
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if msg.encoding in ("16UC1", "mono16"):
        depth = arr.astype(np.float32) * 0.001
    else:
        depth = arr.astype(np.float32)
    return depth, None


# 解码普通 RGB 图像, 供深度 ROI 的颜色前景筛选使用。
def _decode_raw_color_image(msg, bridge):
    if not _CV_AVAILABLE or bridge is None:
        return None, "cv_bridge/cv2 unavailable: {}".format(_CV_IMPORT_ERROR)
    try:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception as exc:
        return None, "raw color decode failed: {}".format(exc)
    return image, None


# 解码压缩 RGB 图像, 输出 OpenCV BGR uint8 图像。
def _decode_compressed_color_image(msg):
    if not _CV_AVAILABLE:
        return None, "cv2/numpy unavailable: {}".format(_CV_IMPORT_ERROR)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size == 0:
        return None, "empty compressed color message"
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None, "compressed color decode failed"
    return image, None


def _decode_compressed_depth_image(msg):
    if not _CV_AVAILABLE:
        return None, "cv2/numpy unavailable: {}".format(_CV_IMPORT_ERROR)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size == 0:
        return None, "empty compressed depth message"

    fmt = msg.format or ""
    offsets = [0, 12, 16]
    if "compressedDepth" in fmt:
        offsets = [12, 16, 0]

    image = None
    for offset in offsets:
        if raw.size <= offset:
            continue
        decoded = cv2.imdecode(raw[offset:], cv2.IMREAD_UNCHANGED)
        if decoded is not None:
            image = decoded
            break
    if image is None:
        return None, "compressed depth PNG decode failed"

    if image.ndim == 3:
        image = image[:, :, 0]

    if "32FC1" in fmt and image.dtype == np.uint16:
        depth_quant_a = None
        depth_quant_b = None
        if len(msg.data) >= 12:
            try:
                depth_quant_a, depth_quant_b = struct.unpack("<ff", msg.data[4:12])
            except Exception:
                depth_quant_a, depth_quant_b = None, None
        if (
            depth_quant_a is not None
            and depth_quant_b is not None
            and math.isfinite(depth_quant_a)
            and math.isfinite(depth_quant_b)
            and depth_quant_a > 0.0
        ):
            raw_depth = image.astype(np.float32)
            depth = np.zeros(raw_depth.shape, dtype=np.float32)
            valid = raw_depth > 0.0
            denom = raw_depth[valid] - float(depth_quant_b)
            denom_valid = np.abs(denom) > 1e-6
            values = np.zeros_like(denom, dtype=np.float32)
            values[denom_valid] = float(depth_quant_a) / denom[denom_valid]
            depth[valid] = values
            return depth, None

    if image.dtype == np.uint16:
        scale = 0.001
        if image.size:
            max_value = float(np.nanmax(image))
            if max_value <= 20.0:
                scale = 1.0
        return image.astype(np.float32) * scale, None

    depth = image.astype(np.float32)
    return depth, None


class MujocoJointIndex:
    """按照 MuJoCo XML 关节顺序建立 qpos 索引。"""

    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.qpos_by_joint = {}
        self.nq = 0

    def load(self):
        self.qpos_by_joint = {}
        self.nq = 0
        self._visit_xml(self.xml_path, os.path.dirname(self.xml_path))
        rospy.loginfo("task1_vision: indexed %d qpos values from %s", self.nq, self.xml_path)

    def _visit_xml(self, xml_path, base_dir):
        if not xml_path or not os.path.exists(xml_path):
            rospy.logwarn("task1_vision: MuJoCo XML not found: %s", xml_path)
            return
        root = ET.parse(xml_path).getroot()
        current_dir = os.path.dirname(xml_path)
        for child in list(root):
            if child.tag == "include":
                include_file = child.get("file")
                if include_file:
                    include_path = include_file
                    if not os.path.isabs(include_path):
                        include_path = os.path.normpath(os.path.join(current_dir, include_path))
                    self._visit_xml(include_path, os.path.dirname(include_path))
            elif child.tag == "worldbody":
                self._visit_worldbody(child)

    def _visit_worldbody(self, worldbody):
        for elem in worldbody.iter():
            if elem.tag == "freejoint":
                self._register_joint(elem.get("name"), 7)
            elif elem.tag == "joint":
                joint_type = elem.get("type", "hinge")
                if joint_type == "free":
                    width = 7
                elif joint_type == "ball":
                    width = 4
                else:
                    width = 1
                self._register_joint(elem.get("name"), width)

    def _register_joint(self, name, width):
        if name:
            self.qpos_by_joint[name] = self.nq
        self.nq += width


class Task1VisionNode:
    def __init__(self):
        self.scene = rospy.get_param("~scene", "scene1")
        self.vision_mode = int(rospy.get_param("~vision_mode", 0))
        if self.vision_mode not in (0, 1):
            raise RuntimeError("task1_vision vision_mode only supports 0 or 1")
        self.detection_mode = (
            "qpos_bbox_depth_top_plane_fit"
            if self.vision_mode == 0
            else "yolo_bbox_depth_top_plane_fit"
        )
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.truth_frame_id = rospy.get_param("~truth_frame_id", self.frame_id)
        self.output_frame_id = rospy.get_param("~output_frame_id", self.frame_id)
        self.base_frame_id = rospy.get_param("~base_frame_id", "base_link")
        self.qpos_topic = rospy.get_param("~qpos_topic", "/mujoco/qpos")
        self.objects = rospy.get_param("~objects", [])
        self.require_camera_tf = bool(rospy.get_param("~require_camera_tf", True))
        self.head_image_topic = rospy.get_param("~head_image_topic", "/cam_h/color/image_raw/compressed")
        self.head_image_type = rospy.get_param("~head_image_type", "auto")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/cam_h/color/camera_info")
        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/cam_h/depth/image_raw/compressedDepth")
        self.depth_image_type = rospy.get_param("~depth_image_type", "auto")
        self.startup_warn_time = float(rospy.get_param("~startup_warn_time", 60.0))
        self.startup_wait_timeout = float(
            rospy.get_param("~startup_wait_timeout", 180.0)
        )
        self.sensor_time_topic = str(
            rospy.get_param("~sensor_time_topic", "/sensors_data_raw")
        )
        self.latest_sensor_time = None
        self.camera_info = None
        self.latest_depth = None
        self.latest_depth_stamp = None
        self.latest_depth_frame_id = None
        self.latest_color = None
        self.latest_color_stamp = None
        self.latest_color_frame_id = None
        self.latest_color_sequence = 0
        self.depth_bridge = CvBridge() if _CV_AVAILABLE else None
        self._projection_warned = set()

        self.yolo_model = None
        self.yolo_model_path = None
        self._yolo_cached_sequence = -1
        self._yolo_cached_candidates = []
        self._deployment_frozen = False
        self._deployment_frozen_detections = []
        self._deployment_frozen_bbox_candidates = []
        self._deployment_last_bbox_candidates = []
        self.used_objects_sub = None
        self._plane_fit_last_log = {}

        self.latest_qpos = None

        self.model_xml_path = self._resolve_model_xml()
        self.joint_index = MujocoJointIndex(self.model_xml_path)
        self.joint_index.load()
        self.mj_model = None
        self.mj_data = None
        self.mj_model_path = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_2d = rospy.Publisher("~detections2d", Detection2DArray, queue_size=1)
        self.pub_3d = rospy.Publisher("~detections3d", Detection3DArray, queue_size=1)
        self.pub_markers = rospy.Publisher("~markers", MarkerArray, queue_size=1)
        self.pub_json = rospy.Publisher("~objects_json", String, queue_size=1)

        self.qpos_sub = rospy.Subscriber(
            self.qpos_topic,
            Float64MultiArray,
            self._qpos_callback,
            queue_size=1,
        )
        self.sensor_time_sub = rospy.Subscriber(
            self.sensor_time_topic,
            sensorsData,
            self._sensor_time_callback,
            queue_size=1,
        )
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self._camera_info_callback,
            queue_size=1,
        )
        if not _CV_AVAILABLE:
            raise RuntimeError(
                "task1_vision requires cv_bridge/cv2 for depth top-plane fitting: {}".format(
                    _CV_IMPORT_ERROR
                )
            )
        if self._depth_topic_is_compressed():
            self.depth_sub = rospy.Subscriber(
                self.depth_image_topic,
                CompressedImage,
                self._compressed_depth_callback,
                queue_size=1,
            )
        else:
            self.depth_sub = rospy.Subscriber(
                self.depth_image_topic,
                Image,
                self._raw_depth_callback,
                queue_size=1,
            )
        if self.vision_mode == 1 or bool(self._param("depth_color_filter_enabled", True)):
            if self._color_topic_is_compressed():
                self.color_sub = rospy.Subscriber(
                    self.head_image_topic,
                    CompressedImage,
                    self._compressed_color_callback,
                    queue_size=1,
                )
            else:
                self.color_sub = rospy.Subscriber(
                    self.head_image_topic,
                    Image,
                    self._raw_color_callback,
                    queue_size=1,
                )

        if self.vision_mode == 1:
            self._load_yolo_model()
            if bool(self._param("yolo_freeze_after_used_objects", True)):
                used_topic = str(
                    self._param(
                        "used_objects_topic", "/task1_vision/used_objects_json"
                    )
                )
                self.used_objects_sub = rospy.Subscriber(
                    used_topic,
                    String,
                    self._used_objects_callback,
                    queue_size=1,
                )
                rospy.loginfo(
                    "task1_vision: YOLO will stop after task coordinates arrive on %s",
                    used_topic,
                )
        rospy.loginfo("task1_vision: detection mode=%s", self.detection_mode)

    def _resolve_model_xml(self):
        configured = rospy.get_param("~model_xml", "")
        if configured:
            return configured
        try:
            import rospkg

            sim_path = rospkg.RosPack().get_path("challenge_cup_simulator")
            active = os.path.join(
                sim_path,
                "models",
                "biped_s52",
                "xml",
                "_scene_{}_active.xml".format(self.scene),
            )
            if os.path.exists(active):
                return active
            return os.path.join(sim_path, "models", "biped_s52", "xml", "{}.xml".format(self.scene))
        except Exception:
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.normpath(
                os.path.join(
                    here,
                    "..",
                    "..",
                    "challenge_cup_simulator",
                    "models",
                    "biped_s52",
                    "xml",
                    "{}.xml".format(self.scene),
                )
            )

    def _qpos_callback(self, msg):
        self.latest_qpos = list(msg.data)

    def _sensor_time_callback(self, msg):
        seconds = float(msg.sensor_time.to_sec())
        if math.isfinite(seconds):
            self.latest_sensor_time = seconds

    def _camera_info_callback(self, msg):
        self.camera_info = msg

    def _depth_topic_is_compressed(self):
        topic_type = str(self.depth_image_type).lower()
        if topic_type in ("compressed", "compresseddepth", "compressed_depth"):
            return True
        if topic_type in ("raw", "image"):
            return False
        return self.depth_image_topic.endswith("/compressed") or self.depth_image_topic.endswith("/compressedDepth")

    def _color_topic_is_compressed(self):
        topic_type = str(self.head_image_type).lower()
        if topic_type in ("compressed", "jpg", "jpeg", "png"):
            return True
        if topic_type in ("raw", "image"):
            return False
        return self.head_image_topic.endswith("/compressed")

    def _raw_depth_callback(self, msg):
        depth, error = _decode_raw_depth_image(msg, self.depth_bridge)
        if error:
            self._warn_projection_once("depth_raw_decode", "task1_vision: %s", error)
            return
        self.latest_depth = depth
        self.latest_depth_stamp = msg.header.stamp
        self.latest_depth_frame_id = msg.header.frame_id or self._camera_frame_id()

    def _compressed_depth_callback(self, msg):
        depth, error = _decode_compressed_depth_image(msg)
        if error:
            self._warn_projection_once("depth_compressed_decode", "task1_vision: %s", error)
            return
        self.latest_depth = depth
        self.latest_depth_stamp = msg.header.stamp
        self.latest_depth_frame_id = msg.header.frame_id or self._camera_frame_id()

    def _raw_color_callback(self, msg):
        image, error = _decode_raw_color_image(msg, self.depth_bridge)
        if error:
            self._warn_projection_once("color_raw_decode", "task1_vision: %s", error)
            return
        self.latest_color = image
        self.latest_color_stamp = msg.header.stamp
        self.latest_color_frame_id = msg.header.frame_id or self._camera_frame_id()
        self.latest_color_sequence += 1

    def _compressed_color_callback(self, msg):
        image, error = _decode_compressed_color_image(msg)
        if error:
            self._warn_projection_once("color_compressed_decode", "task1_vision: %s", error)
            return
        self.latest_color = image
        self.latest_color_stamp = msg.header.stamp
        self.latest_color_frame_id = msg.header.frame_id or self._camera_frame_id()
        self.latest_color_sequence += 1

    def _param(self, name, default):
        return rospy.get_param("~" + name, default)

    def _used_objects_callback(self, msg):
        if self.vision_mode != 1 or self._deployment_frozen:
            return
        try:
            payload = json.loads(msg.data)
            objects = payload.get("objects", [])
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0, "task1_vision: invalid used objects payload: %s", exc
            )
            return

        required_names = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]
        objects_by_name = {
            str(item.get("name", "")): item
            for item in objects
            if isinstance(item, dict)
        }
        if not all(name in objects_by_name for name in required_names):
            return

        self._deployment_frozen_detections = [
            copy.deepcopy(objects_by_name[name]) for name in required_names
        ]
        self._deployment_frozen_bbox_candidates = copy.deepcopy(
            self._deployment_last_bbox_candidates
        )
        self._deployment_frozen = True
        rospy.loginfo(
            "task1_vision: task coordinates locked; stop YOLO inference and reuse fixed results"
        )

    def _release_yolo_model(self):
        if self.yolo_model is None:
            return
        self.yolo_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        rospy.loginfo("task1_vision: YOLO model released after coordinate lock")

    def _resolve_yolo_model_path(self):
        configured = str(self._param("yolo_model_path", "models/parcel_best.pt")).strip()
        if not configured:
            configured = "models/parcel_best.pt"
        configured = os.path.expandvars(os.path.expanduser(configured))
        if configured.startswith("package://"):
            package_and_path = configured[len("package://") :].split("/", 1)
            if len(package_and_path) != 2:
                raise RuntimeError("invalid yolo_model_path: {}".format(configured))
            try:
                import rospkg

                package_root = rospkg.RosPack().get_path(package_and_path[0])
            except Exception as exc:
                raise RuntimeError(
                    "cannot resolve yolo model package path {}: {}".format(configured, exc)
                )
            return os.path.abspath(os.path.join(package_root, package_and_path[1]))
        if os.path.isabs(configured):
            return os.path.abspath(configured)
        try:
            import rospkg

            package_root = rospkg.RosPack().get_path("task1_vision")
        except Exception:
            package_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.abspath(os.path.join(package_root, configured))

    def _load_yolo_model(self):
        self.yolo_model_path = self._resolve_yolo_model_path()
        if not os.path.isfile(self.yolo_model_path):
            raise RuntimeError(
                "task1_vision mode 1 YOLO weights not found: {}".format(
                    self.yolo_model_path
                )
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "task1_vision mode 1 requires ultralytics in the ROS Python environment"
            ) from exc
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
        except Exception as exc:
            raise RuntimeError(
                "failed to load task1 YOLO weights {}: {}".format(
                    self.yolo_model_path, exc
                )
            )
        rospy.loginfo("task1_vision: loaded YOLO weights %s", self.yolo_model_path)

    def _infer_yolo_candidates(self):
        if self.latest_color is None or self.yolo_model is None:
            return []
        sequence = self.latest_color_sequence
        if sequence == self._yolo_cached_sequence:
            return [dict(item) for item in self._yolo_cached_candidates]

        image = self.latest_color
        predict_kwargs = {
            "source": image,
            "conf": float(self._param("yolo_confidence_threshold", 0.25)),
            "iou": float(self._param("yolo_iou_threshold", 0.70)),
            "imgsz": int(self._param("yolo_image_size", 960)),
            "classes": [int(self._param("yolo_class_id", 0))],
            "max_det": 4,
            "verbose": False,
        }
        device = self._param("yolo_device", "")
        if str(device).strip():
            predict_kwargs["device"] = device
        if bool(self._param("yolo_half", False)):
            predict_kwargs["half"] = True

        try:
            results = self.yolo_model.predict(**predict_kwargs)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "task1_vision: YOLO inference failed: %s", exc)
            return []

        candidates = []
        result = results[0] if results else None
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy()
            color_h, color_w = image.shape[:2]
            info_w = float(self.camera_info.width if self.camera_info is not None else color_w)
            info_h = float(self.camera_info.height if self.camera_info is not None else color_h)
            scale_x = info_w / float(max(1, color_w))
            scale_y = info_h / float(max(1, color_h))
            for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
                x1, y1, x2, y2 = [float(value) for value in coords[:4]]
                width = max(2.0, (x2 - x1) * scale_x)
                height = max(2.0, (y2 - y1) * scale_y)
                candidates.append(
                    {
                        "class_id": int(class_id),
                        "class_name": "parcel",
                        "confidence": float(confidence),
                        "bbox_center": [
                            (x1 + x2) * 0.5 * scale_x,
                            (y1 + y2) * 0.5 * scale_y,
                        ],
                        "bbox_size": [width, height],
                        "bbox_source": "yolo_rgb",
                    }
                )

        candidates.sort(key=lambda item: float(item["confidence"]), reverse=True)
        candidates = candidates[:4]
        self._yolo_cached_sequence = sequence
        self._yolo_cached_candidates = [dict(item) for item in candidates]
        return candidates

    def _deployment_truth_reference(self, spec, camera_position):
        empty = {
            "truth_position": [],
            "truth_position_source": "",
            "truth_camera_position": [],
            "truth_camera_reference_error": None,
            "reference_camera_position": [],
            "camera_reference_error": None,
        }
        truth = self._truth_pose(spec)
        if truth is None:
            return empty
        truth_position, truth_yaw = truth
        measurement = self._vision_measurement_from_truth(
            truth_position, _quat_from_yaw(truth_yaw)
        )
        if measurement is None:
            return empty
        output_position, _, truth_camera_position, source, truth_camera_reference = measurement
        camera_reference = self._mujoco_camera_reference(
            truth_position, camera_position if len(camera_position) >= 3 else None
        )
        return {
            "truth_position": output_position,
            "truth_position_source": source,
            "truth_camera_position": truth_camera_position or [],
            "truth_camera_reference_error": truth_camera_reference.get(
                "camera_reference_error"
            ),
            "reference_camera_position": camera_reference.get(
                "reference_camera_position", []
            ),
            "camera_reference_error": camera_reference.get("camera_reference_error"),
        }

    def _make_deployment_detection(self, name, item):
        specs_by_name = {
            str(spec.get("name", "")): spec for spec in self.objects
        }
        spec = specs_by_name.get(name)
        if spec is None:
            raise RuntimeError("missing task1_vision object config for {}".format(name))

        fit = item["fit"]
        size = _parse_vec(spec.get("size"), 3, [0.05, 0.05, 0.05])
        position = fit["position"]
        camera_position = fit.get("camera_position") or []
        truth = self._deployment_truth_reference(spec, camera_position)
        truth_position = truth["truth_position"]
        truth_error = _distance(position, truth_position)
        yaw = float(fit.get("yaw", 0.0))
        item["name"] = name
        item["fit_succeeded"] = True
        return {
            "name": name,
            "class_id": int(spec.get("class_id", item.get("class_id", 0))),
            "class_name": str(spec.get("class_name", "parcel")),
            "confidence": _clip(float(item.get("confidence", 0.0)), 0.0, 1.0),
            "position": position,
            "camera_position": camera_position,
            "position_source": "depth_roi_top_plane_fit",
            "truth_position": truth_position,
            "truth_position_source": truth["truth_position_source"],
            "truth_position_error": truth_error,
            "truth_camera_position": truth["truth_camera_position"],
            "truth_camera_reference_error": truth["truth_camera_reference_error"],
            "depth_fit_position": position,
            "depth_fit_truth_error": truth_error,
            "reference_camera_position": truth["reference_camera_position"],
            "camera_reference_error": truth["camera_reference_error"],
            "fit_method": fit.get("method", ""),
            "fit_points": int(fit.get("points", 0)),
            "fit_cluster_points": int(fit.get("cluster_points", 0)),
            "fit_error": fit.get("error"),
            "fit_extent": fit.get("fit_extent", []),
            "fit_outside_rmse": fit.get("fit_outside_rmse"),
            "fit_plane_normal": fit.get("plane_normal", []),
            "fit_plane_tilt_deg": fit.get("plane_tilt_deg"),
            "fit_plane_inliers": fit.get("plane_inliers"),
            "fit_plane_inlier_ratio": fit.get("plane_inlier_ratio"),
            "fit_plane_rmse": fit.get("plane_rmse"),
            "fit_plane_top_z": fit.get("plane_top_z"),
            "size": size,
            "yaw": yaw,
            "yaw_valid": True,
            "orientation": _quat_from_yaw(yaw),
            "bbox_center": list(item["bbox_center"]),
            "bbox_size": list(item["bbox_size"]),
            "bbox_projected": False,
            "bbox_source": "yolo_rgb",
            "label_assignment_source": "relative_xy",
            "color_rgba": _parse_vec(
                spec.get("color_rgba"), 4, [0.1, 0.8, 0.2, 0.5]
            ),
        }

    def _detect_deployment_objects(self):
        candidates = self._infer_yolo_candidates()
        if len(candidates) != 4:
            rospy.logwarn_throttle(
                1.0,
                "task1_vision: deployment mode requires exactly 4 YOLO boxes, got %d",
                len(candidates),
            )
            return [], candidates

        generic_spec = dict(self.objects[0]) if self.objects else {
            "class_name": "parcel",
            "size": [0.045, 0.120, 0.100],
        }
        size = _parse_vec(generic_spec.get("size"), 3, [0.045, 0.120, 0.100])
        fitted = []
        for index, candidate in enumerate(candidates):
            fit_spec = dict(generic_spec)
            fit_spec["name"] = "yolo_candidate_{}".format(index + 1)
            fit = self._fit_position_from_depth_bbox(
                candidate["bbox_center"], candidate["bbox_size"], size, fit_spec
            )
            candidate["fit_succeeded"] = fit is not None
            if fit is not None:
                candidate["fit"] = fit
                fitted.append(candidate)

        if len(fitted) != 4:
            rospy.logwarn_throttle(
                1.0,
                "task1_vision: only %d/4 YOLO boxes completed depth top-plane fitting",
                len(fitted),
            )
            return [], candidates

        assigned = _assign_parcel_names_by_relative_xy(fitted)
        detections = [
            self._make_deployment_detection(name, item) for name, item in assigned
        ]
        detections.sort(key=lambda item: item["name"])
        return detections, candidates

    def _detect_object(self, spec):
        truth = self._truth_pose(spec)
        if truth is None:
            return None

        true_size = _parse_vec(spec.get("size"), 3, [0.05, 0.05, 0.05])
        truth_position, truth_yaw = truth
        orientation = _quat_from_yaw(truth_yaw)

        truth_measurement = self._vision_measurement_from_truth(
            truth_position, orientation
        )
        if truth_measurement is None:
            self._warn_projection_once(
                "measurement_" + str(spec.get("name", "")),
                "task1_vision: no output-frame truth reference for %s; detection skipped",
                str(spec.get("name", "")),
            )
            return None
        truth_output_position, truth_output_orientation, truth_camera_position, truth_source, truth_camera_reference = truth_measurement

        projected_bbox = self._project_bbox(
            truth_output_position, truth_output_orientation, true_size
        )
        if projected_bbox is None:
            self._warn_projection_once(
                "bbox_projection_" + str(spec.get("name", "")),
                "task1_vision: qpos bbox projection failed for %s; detection skipped",
                str(spec.get("name", "")),
            )
            return None
        bbox_center, bbox_size = projected_bbox

        fit = self._fit_position_from_depth_bbox(
            bbox_center, bbox_size, true_size, spec
        )
        if fit is None:
            self._warn_projection_once(
                "depth_fit_failed_" + str(spec.get("name", "")),
                "task1_vision: depth top-plane fit failed for %s; detection skipped",
                str(spec.get("name", "")),
            )
            return None

        output_position = fit["position"]
        camera_position = fit.get("camera_position") or None
        reported_yaw = float(fit.get("yaw", truth_yaw))
        output_orientation = _quat_from_yaw(reported_yaw)
        reference = self._mujoco_camera_reference(truth_position, camera_position)
        truth_error = _distance(output_position, truth_output_position)

        return {
            "name": str(spec.get("name", "")),
            "class_id": int(spec.get("class_id", 0)),
            "class_name": str(spec.get("class_name", spec.get("name", "object"))),
            "confidence": _clip(float(self._param("confidence", 1.0)), 0.0, 1.0),
            "position": output_position,
            "camera_position": camera_position if camera_position is not None else [],
            "position_source": "depth_roi_top_plane_fit",
            "truth_position": truth_output_position,
            "truth_position_source": truth_source,
            "truth_position_error": truth_error,
            "truth_camera_position": truth_camera_position if truth_camera_position is not None else [],
            "truth_camera_reference_error": truth_camera_reference.get("camera_reference_error"),
            "depth_fit_position": output_position,
            "depth_fit_truth_error": truth_error,
            "reference_camera_position": reference.get("reference_camera_position", []),
            "camera_reference_error": reference.get("camera_reference_error"),
            "fit_method": fit.get("method", ""),
            "fit_points": int(fit.get("points", 0)),
            "fit_cluster_points": int(fit.get("cluster_points", 0)),
            "fit_error": fit.get("error"),
            "fit_extent": fit.get("fit_extent", []),
            "fit_outside_rmse": fit.get("fit_outside_rmse"),
            "fit_plane_normal": fit.get("plane_normal", []),
            "fit_plane_tilt_deg": fit.get("plane_tilt_deg"),
            "fit_plane_inliers": fit.get("plane_inliers"),
            "fit_plane_inlier_ratio": fit.get("plane_inlier_ratio"),
            "fit_plane_rmse": fit.get("plane_rmse"),
            "fit_plane_top_z": fit.get("plane_top_z"),
            "size": true_size,
            "yaw": reported_yaw,
            "yaw_valid": True,
            "orientation": output_orientation,
            "bbox_center": bbox_center,
            "bbox_size": bbox_size,
            "bbox_projected": True,
            "bbox_source": "qpos_projected_3d_corners",
            "color_rgba": _parse_vec(spec.get("color_rgba"), 4, [0.1, 0.8, 0.2, 0.5]),
        }

    def _camera_frame_id(self):
        if self.camera_info is not None and self.camera_info.header.frame_id:
            return self.camera_info.header.frame_id
        return str(self._param("camera_frame_id", "Head Camera View"))

    def _lookup_transform(self, target_frame, source_frame, key):
        if not target_frame or not source_frame:
            return None
        if target_frame == source_frame:
            return "identity"
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(float(self._param("tf_timeout", 0.02))),
            ).transform
        except Exception as exc:
            self._warn_projection_once(key, "task1_vision: TF %s <- %s unavailable: %s", target_frame, source_frame, exc)
            return None

    def _apply_transform_point(self, transform, point):
        if transform == "identity":
            return list(point)
        return _transform_point(transform, point)

    def _transform_points_numpy(self, transform, points):
        if transform == "identity":
            return points.copy()
        quat = [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
        rot = _quat_to_rotation_matrix(quat)
        trans = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            dtype=np.float64,
        )
        return points.dot(rot.T) + trans

    # 从二维框对应的深度 ROI 反投影点云，并用 RANSAC 顶面恢复物体中心。
    def _fit_position_from_depth_bbox(self, bbox_center, bbox_size, size, spec):
        if not _CV_AVAILABLE:
            self._warn_projection_once("depth_fit_cv_unavailable", "task1_vision: depth fit unavailable: %s", _CV_IMPORT_ERROR)
            return None
        if self.camera_info is None:
            self._warn_projection_once("depth_fit_no_camera_info", "task1_vision: depth fit waiting for %s", self.camera_info_topic)
            return None
        if self.latest_depth is None:
            self._warn_projection_once("depth_fit_no_depth", "task1_vision: depth fit waiting for %s", self.depth_image_topic)
            return None

        points_camera = self._depth_roi_to_camera_points(bbox_center, bbox_size, spec)
        if points_camera is None:
            return None

        depth_frame = self.latest_depth_frame_id or self._camera_frame_id()
        camera_to_output = self._lookup_transform(self.output_frame_id, depth_frame, "depth_camera_to_output")
        if camera_to_output is None:
            if self.require_camera_tf:
                self._warn_projection_once(
                    "depth_camera_to_output_required",
                    "task1_vision: required TF %s <- %s unavailable; depth fit skipped",
                    self.output_frame_id,
                    depth_frame,
                )
                return None
            return None

        points_output = self._transform_points_numpy(camera_to_output, points_camera)
        fit = self._fit_top_plane_ransac(points_output, size, spec)
        if fit is None:
            return None

        output_to_camera = self._lookup_transform(depth_frame, self.output_frame_id, "depth_output_to_camera")
        if output_to_camera is not None:
            fit["camera_position"] = self._apply_transform_point(output_to_camera, fit["position"])
        else:
            fit["camera_position"] = []
        fit["points"] = int(points_camera.shape[0])
        return fit

    # 将深度图 ROI 中的有效像素按 CameraInfo 反投影成相机坐标系点云。
    def _depth_roi_to_camera_points(self, bbox_center, bbox_size, spec):
        depth = self.latest_depth
        if depth is None or depth.ndim != 2:
            self._warn_projection_once("depth_fit_bad_depth", "task1_vision: depth image is empty or not single-channel")
            return None

        img_h, img_w = depth.shape[:2]
        info_w = float(self.camera_info.width or img_w)
        info_h = float(self.camera_info.height or img_h)
        if img_w <= 1 or img_h <= 1 or info_w <= 1.0 or info_h <= 1.0:
            self._warn_projection_once("depth_fit_bad_size", "task1_vision: invalid depth/camera_info size")
            return None

        scale_x = float(img_w) / info_w
        scale_y = float(img_h) / info_h
        expand = float(self._param("depth_roi_expand", 1.05))
        cx = float(bbox_center[0]) * scale_x
        cy = float(bbox_center[1]) * scale_y
        bw = max(2.0, float(bbox_size[0]) * scale_x * expand)
        bh = max(2.0, float(bbox_size[1]) * scale_y * expand)
        u0 = max(0, int(math.floor(cx - bw * 0.5)))
        u1 = min(img_w, int(math.ceil(cx + bw * 0.5)))
        v0 = max(0, int(math.floor(cy - bh * 0.5)))
        v1 = min(img_h, int(math.ceil(cy + bh * 0.5)))
        if u1 <= u0 + 1 or v1 <= v0 + 1:
            self._warn_projection_once("depth_fit_empty_roi", "task1_vision: empty depth ROI for bbox %.1f %.1f %.1f %.1f", cx, cy, bw, bh)
            return None

        stride = max(1, int(self._param("depth_roi_stride", 1)))
        roi = depth[v0:v1:stride, u0:u1:stride]
        min_depth = float(self._param("depth_min_m", 0.05))
        max_depth = float(self._param("depth_max_m", 4.0))
        valid = np.isfinite(roi) & (roi > min_depth) & (roi < max_depth)
        min_points = int(self._param("depth_min_points", 40))
        if int(np.count_nonzero(valid)) < min_points:
            self._warn_projection_once(
                "depth_fit_few_points",
                "task1_vision: depth ROI has too few valid points: %d",
                int(np.count_nonzero(valid)),
            )
            return None

        valid = self._apply_color_filter_to_roi(valid, bbox_center, bbox_size, roi.shape, spec)
        if valid is None:
            return None

        if bool(self._param("depth_foreground_filter", True)):
            valid_depth = roi[valid]
            percentile = float(self._param("depth_front_percentile", 8.0))
            front_depth = float(np.percentile(valid_depth, percentile))
            margin = float(self._param("depth_front_margin_m", 0.10))
            valid = valid & (roi <= front_depth + margin)

        if int(np.count_nonzero(valid)) < min_points:
            self._warn_projection_once(
                "depth_fit_few_foreground_points",
                "task1_vision: foreground depth ROI has too few valid points: %d",
                int(np.count_nonzero(valid)),
            )
            return None

        vv, uu = np.mgrid[v0:v1:stride, u0:u1:stride]
        uu = uu[valid].astype(np.float32)
        vv = vv[valid].astype(np.float32)
        zz = roi[valid].astype(np.float32)

        k = self.camera_info.K
        fx = float(k[0]) * scale_x
        fy = float(k[4]) * scale_y
        px = float(k[2]) * scale_x
        py = float(k[5]) * scale_y
        if abs(fx) < 1e-6 or abs(fy) < 1e-6:
            self._warn_projection_once("depth_fit_bad_intrinsics", "task1_vision: invalid camera intrinsics fx/fy")
            return None

        xx = (uu - px) * zz / fx
        yy = (vv - py) * zz / fy
        return np.stack([xx, yy, zz], axis=1).astype(np.float64)

    # 用 RGB 颜色 mask 保留目标物体像素, 替代单纯按深度远近做背景过滤。
    def _apply_color_filter_to_roi(self, valid, bbox_center, bbox_size, roi_shape, spec):
        if not bool(self._param("depth_color_filter_enabled", True)):
            return valid
        color = self.latest_color
        if color is None or color.ndim != 3:
            self._warn_projection_once("depth_color_no_image", "task1_vision: color filter waiting for %s", self.head_image_topic)
            return None

        img_h, img_w = color.shape[:2]
        info_w = float(self.camera_info.width or img_w)
        info_h = float(self.camera_info.height or img_h)
        if img_w <= 1 or img_h <= 1 or info_w <= 1.0 or info_h <= 1.0:
            self._warn_projection_once("depth_color_bad_size", "task1_vision: invalid color/camera_info size")
            return None

        scale_x = float(img_w) / info_w
        scale_y = float(img_h) / info_h
        expand = float(self._param("depth_roi_expand", 1.05))
        cx = float(bbox_center[0]) * scale_x
        cy = float(bbox_center[1]) * scale_y
        bw = max(2.0, float(bbox_size[0]) * scale_x * expand)
        bh = max(2.0, float(bbox_size[1]) * scale_y * expand)
        u0 = max(0, int(math.floor(cx - bw * 0.5)))
        u1 = min(img_w, int(math.ceil(cx + bw * 0.5)))
        v0 = max(0, int(math.floor(cy - bh * 0.5)))
        v1 = min(img_h, int(math.ceil(cy + bh * 0.5)))
        if u1 <= u0 + 1 or v1 <= v0 + 1:
            self._warn_projection_once("depth_color_empty_roi", "task1_vision: empty color ROI for bbox %.1f %.1f %.1f %.1f", cx, cy, bw, bh)
            return None

        color_roi = color[v0:v1, u0:u1]
        target_hw = (int(roi_shape[0]), int(roi_shape[1]))
        if color_roi.shape[:2] != target_hw:
            color_roi = cv2.resize(color_roi, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)

        color_mask = self._object_color_mask(color_roi, spec)
        combined = valid & color_mask
        min_points = int(self._param("depth_color_min_points", self._param("depth_min_points", 40)))
        count = int(np.count_nonzero(combined))
        if count < min_points:
            self._warn_projection_once(
                "depth_color_few_points_" + str(spec.get("name", "")),
                "task1_vision: color ROI for %s has too few valid target-color points: %d",
                spec.get("name", "<unnamed>"),
                count,
            )
            return None
        return combined

    # 根据仿真配置中的物体颜色和桌面颜色生成 ROI 颜色 mask。
    def _object_color_mask(self, color_roi_bgr, spec):
        rgb = color_roi_bgr[:, :, ::-1].astype(np.float32) / 255.0
        target = np.array(_parse_vec(spec.get("color_rgba"), 4, [0.88, 0.84, 0.73, 1.0])[:3], dtype=np.float32)
        rgb_sum = np.maximum(np.sum(rgb, axis=2, keepdims=True), 1e-6)
        rgb_chroma = rgb / rgb_sum
        target_chroma = target / max(float(np.sum(target)), 1e-6)

        chroma_threshold = float(self._param("depth_color_target_chroma_threshold", 0.24))
        rgb_threshold = float(self._param("depth_color_target_rgb_threshold", 0.45))
        chroma_dist = np.linalg.norm(rgb_chroma - target_chroma.reshape(1, 1, 3), axis=2)
        rgb_dist = np.linalg.norm(rgb - target.reshape(1, 1, 3), axis=2)
        mask = (chroma_dist <= chroma_threshold) | (rgb_dist <= rgb_threshold)

        class_name = str(spec.get("class_name", ""))
        if class_name == "parcel" and bool(self._param("depth_color_reject_table_for_parcels", True)):
            table = np.array(_parse_vec(self._param("depth_table_color_rgba", [0.10, 0.30, 0.10, 1.0]), 4, [0.10, 0.30, 0.10, 1.0])[:3], dtype=np.float32)
            table_chroma = table / max(float(np.sum(table)), 1e-6)
            table_dist = np.linalg.norm(rgb_chroma - table_chroma.reshape(1, 1, 3), axis=2)
            min_value = float(self._param("depth_color_min_value", 0.05))
            non_table = (table_dist >= float(self._param("depth_color_table_chroma_threshold", 0.18))) & (np.max(rgb, axis=2) >= min_value)
            mask = mask | non_table

        if bool(self._param("depth_color_morphology", True)) and mask.size:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask_u8 = (mask.astype(np.uint8) * 255)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            mask = mask_u8 > 0
        return mask

    # 在 ROI 点云中选择空间连续的主簇，避免背景点和相邻物体污染顶面拟合。
    def _select_main_cluster(self, points):
        if not bool(self._param("depth_cluster_enabled", True)):
            return points
        min_points = int(self._param("depth_cluster_min_points", 30))
        if points.shape[0] < min_points:
            return None

        max_points = int(self._param("depth_cluster_max_points", 7000))
        if points.shape[0] > max_points:
            step = int(math.ceil(float(points.shape[0]) / float(max_points)))
            points = points[::step]

        voxel = max(0.002, float(self._param("depth_cluster_tolerance_m", 0.015)))
        keys = np.floor(points / voxel).astype(np.int64)
        buckets = {}
        for index, key in enumerate(keys):
            buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(index)

        neighbors = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        visited = set()
        best_indices = []
        for start in buckets:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            cluster_voxels = []
            while stack:
                current = stack.pop()
                cluster_voxels.append(current)
                for dx, dy, dz in neighbors:
                    nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if nxt in buckets and nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)

            indices = []
            for voxel_key in cluster_voxels:
                indices.extend(buckets[voxel_key])
            if len(indices) > len(best_indices):
                best_indices = indices

        if len(best_indices) < min_points:
            self._warn_projection_once(
                "depth_cluster_too_small",
                "task1_vision: main depth cluster too small: %d",
                len(best_indices),
            )
            return None
        return points[np.array(best_indices, dtype=np.int64)]

    # 使用已知长宽搜索 yaw，并从可见 XY 点估计物体水平中心。
    def _fit_top_plane_xy_center(self, xy, size, spec):
        if xy.shape[0] < 3:
            return None
        size = np.array(
            [max(0.001, float(v)) for v in size[:3]], dtype=np.float64
        )
        yaw_candidates = self._top_plane_yaw_candidates(xy, spec)
        low_pct = float(self._param("top_plane_xy_low_percentile", 5.0))
        high_pct = float(self._param("top_plane_xy_high_percentile", 95.0))
        half_xy = size[:2] * 0.5

        best = None
        for yaw in yaw_candidates:
            c = math.cos(yaw)
            s = math.sin(yaw)
            local_x = c * xy[:, 0] + s * xy[:, 1]
            local_y = -s * xy[:, 0] + c * xy[:, 1]
            local = np.stack([local_x, local_y], axis=1)
            lo = np.percentile(local, low_pct, axis=0)
            hi = np.percentile(local, high_pct, axis=0)
            center_local = (lo + hi) * 0.5
            extent = np.maximum(hi - lo, 1e-6)
            center_x = c * center_local[0] - s * center_local[1]
            center_y = s * center_local[0] + c * center_local[1]

            rel_x = c * (xy[:, 0] - center_x) + s * (xy[:, 1] - center_y)
            rel_y = -s * (xy[:, 0] - center_x) + c * (xy[:, 1] - center_y)
            overflow = np.maximum(np.stack([np.abs(rel_x), np.abs(rel_y)], axis=1) - half_xy, 0.0)
            outside_rmse = math.sqrt(float(np.mean(np.sum(overflow * overflow, axis=1))))
            extent_over = np.maximum(extent - size[:2], 0.0) / size[:2]
            extent_under = np.maximum(size[:2] * 0.45 - extent, 0.0) / size[:2]
            score = outside_rmse + 0.03 * float(np.sum(extent_over * extent_over + extent_under * extent_under))

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "yaw": yaw,
                    "center_xy": [float(center_x), float(center_y)],
                    "extent": [float(extent[0]), float(extent[1])],
                    "outside_rmse": outside_rmse,
                }
        return best

    # 在 base_link 点云中拟合近似水平的最高平面，并按已知高度仅沿 Z 轴恢复中心。
    def _fit_top_plane_ransac(self, points_output, size, spec):
        cluster = self._select_main_cluster(points_output)
        if cluster is None:
            return None

        min_points = max(
            3, int(self._param("top_plane_ransac_min_inliers", 30))
        )
        if cluster.shape[0] < min_points:
            return None

        iterations = max(
            1, int(self._param("top_plane_ransac_iterations", 250))
        )
        distance_threshold = max(
            1e-5,
            float(self._param("top_plane_ransac_distance_threshold_m", 0.004)),
        )
        max_tilt_deg = max(
            0.0, float(self._param("top_plane_ransac_max_tilt_deg", 12.0))
        )
        min_ratio = max(
            0.0,
            min(
                1.0,
                float(self._param("top_plane_ransac_min_inlier_ratio", 0.08)),
            ),
        )
        max_sample_points = max(
            min_points,
            int(self._param("top_plane_ransac_max_sample_points", 5000)),
        )
        sample_points = cluster
        if sample_points.shape[0] > max_sample_points:
            step = int(
                math.ceil(float(sample_points.shape[0]) / float(max_sample_points))
            )
            sample_points = sample_points[::step]

        random_seed = int(self._param("top_plane_ransac_seed", 42))
        rng = np.random.RandomState(random_seed)
        best_plane = None
        for _ in range(iterations):
            sample_indices = rng.choice(sample_points.shape[0], 3, replace=False)
            p0, p1, p2 = sample_points[sample_indices]
            normal = np.cross(p1 - p0, p2 - p0)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-9:
                continue
            normal = normal / normal_norm
            if normal[2] < 0.0:
                normal = -normal
            tilt_deg = math.degrees(
                math.acos(float(np.clip(normal[2], -1.0, 1.0)))
            )
            if tilt_deg > max_tilt_deg:
                continue

            plane_offset = -float(np.dot(normal, p0))
            distances = np.abs(cluster.dot(normal) + plane_offset)
            inlier_mask = distances <= distance_threshold
            inlier_count = int(np.count_nonzero(inlier_mask))
            inlier_ratio = float(inlier_count) / float(cluster.shape[0])
            if inlier_count < min_points or inlier_ratio < min_ratio:
                continue

            inliers = cluster[inlier_mask]
            plane_top_z = float(np.median(inliers[:, 2]))
            plane_rmse = math.sqrt(
                float(np.mean((inliers.dot(normal) + plane_offset) ** 2))
            )
            candidate = {
                "normal": normal,
                "offset": plane_offset,
                "tilt_deg": tilt_deg,
                "inlier_mask": inlier_mask,
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio,
                "plane_top_z": float(plane_top_z),
                "plane_rmse": plane_rmse,
            }
            if (
                best_plane is None
                or candidate["plane_top_z"] > best_plane["plane_top_z"] + 1e-5
                or (
                    abs(candidate["plane_top_z"] - best_plane["plane_top_z"])
                    <= 1e-5
                    and candidate["inlier_count"] > best_plane["inlier_count"]
                )
            ):
                best_plane = candidate

        if best_plane is None:
            rospy.logwarn_throttle(
                1.0,
                "task1_vision: no RANSAC top plane aligned with base Z "
                "(max tilt %.1f deg)",
                max_tilt_deg,
            )
            return None

        initial_inliers = cluster[best_plane["inlier_mask"]]
        centroid = np.mean(initial_inliers, axis=0)
        try:
            _, _, vh = np.linalg.svd(initial_inliers - centroid)
        except np.linalg.LinAlgError:
            return None
        normal = vh[-1]
        if normal[2] < 0.0:
            normal = -normal
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-9:
            return None
        normal = normal / normal_norm
        tilt_deg = math.degrees(
            math.acos(float(np.clip(normal[2], -1.0, 1.0)))
        )
        if tilt_deg > max_tilt_deg:
            return None

        plane_offset = -float(np.dot(normal, centroid))
        distances = np.abs(cluster.dot(normal) + plane_offset)
        inlier_mask = distances <= distance_threshold
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = float(inlier_count) / float(cluster.shape[0])
        if inlier_count < min_points or inlier_ratio < min_ratio:
            return None
        inliers = cluster[inlier_mask]
        plane_rmse = math.sqrt(
            float(np.mean((inliers.dot(normal) + plane_offset) ** 2))
        )
        best_plane.update(
            {
                "normal": normal,
                "offset": plane_offset,
                "tilt_deg": tilt_deg,
                "inlier_mask": inlier_mask,
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio,
                "plane_rmse": plane_rmse,
            }
        )

        plane_points = cluster[best_plane["inlier_mask"]]
        size_array = np.array(
            [max(0.001, float(v)) for v in size[:3]], dtype=np.float64
        )
        xy_fit = self._fit_top_plane_xy_center(
            plane_points[:, :2], size_array, spec
        )
        if xy_fit is None:
            return None

        center_x, center_y = xy_fit["center_xy"]
        normal = best_plane["normal"]
        plane_top_z = -(
            normal[0] * center_x
            + normal[1] * center_y
            + best_plane["offset"]
        ) / normal[2]
        center_z = float(plane_top_z - size_array[2] * 0.5)

        name = str(spec.get("name", "<unnamed>"))
        now = time.time()
        if now - self._plane_fit_last_log.get(name, 0.0) >= 1.0:
            rospy.loginfo(
                "task1_vision: %s RANSAC top normal=%.4f %.4f %.4f "
                "tilt=%.2f deg inliers=%d/%d rmse=%.2f mm center=%s",
                name,
                normal[0],
                normal[1],
                normal[2],
                best_plane["tilt_deg"],
                best_plane["inlier_count"],
                cluster.shape[0],
                best_plane["plane_rmse"] * 1000.0,
                " ".join(
                    "{:.4f}".format(value)
                    for value in (center_x, center_y, center_z)
                ),
            )
            self._plane_fit_last_log[name] = now

        return {
            "method": "roi_depth_voxel_cluster_ransac_top_plane_known_size",
            "position": [float(center_x), float(center_y), center_z],
            "yaw": float(xy_fit["yaw"]),
            "cluster_points": int(cluster.shape[0]),
            "error": float(xy_fit["score"] + best_plane["plane_rmse"]),
            "fit_extent": xy_fit["extent"],
            "fit_outside_rmse": xy_fit["outside_rmse"],
            "plane_normal": [float(value) for value in normal],
            "plane_tilt_deg": float(best_plane["tilt_deg"]),
            "plane_inliers": int(best_plane["inlier_count"]),
            "plane_inlier_ratio": float(best_plane["inlier_ratio"]),
            "plane_rmse": float(best_plane["plane_rmse"]),
            "plane_top_z": float(plane_top_z),
        }

    def _top_plane_yaw_candidates(self, xy, spec):
        candidates = []
        configured = spec.get(
            "yaw_candidates_deg",
            self._param("top_plane_yaw_candidates_deg", []),
        )
        for value in configured or []:
            candidates.append(math.radians(float(value)))

        if xy.shape[0] >= 3:
            centered = xy - np.mean(xy, axis=0)
            cov = np.cov(centered.T)
            try:
                vals, vecs = np.linalg.eigh(cov)
                axis = vecs[:, int(np.argmax(vals))]
                pca_yaw = math.atan2(float(axis[1]), float(axis[0]))
                candidates.extend([pca_yaw, pca_yaw + math.pi * 0.5])
            except Exception:
                pass

        step_deg = float(self._param("top_plane_yaw_search_step_deg", 15.0))
        if step_deg > 0.0:
            steps = max(1, int(round(180.0 / step_deg)))
            for i in range(steps):
                candidates.append(math.radians(i * 180.0 / steps))

        normalized = []
        for yaw in candidates:
            value = math.fmod(float(yaw), math.pi)
            if value < 0.0:
                value += math.pi
            if all(abs(math.atan2(math.sin(value - old), math.cos(value - old))) > math.radians(1.0) for old in normalized):
                normalized.append(value)
        return normalized or [0.0]

    def _vision_measurement_from_truth(self, truth_position, output_orientation):
        """构造 qpos 二维框投影和误差计算使用的真值参考。"""
        camera_frame = self._camera_frame_id()
        output_frame = self.output_frame_id

        qpos_base_measurement = self._output_from_qpos_base(truth_position, output_orientation)
        if qpos_base_measurement is None:
            self._warn_projection_once(
                "vision_output_base_unavailable",
                "task1_vision: qpos-derived %s output unavailable; detection cannot produce base-relative position",
                self.base_frame_id,
            )
            return None

        output_position, output_quat = qpos_base_measurement
        camera_position = None
        source = "qpos_base"

        base_to_camera = self._lookup_transform(camera_frame, output_frame, "base_to_camera")
        if base_to_camera is not None:
            camera_position = self._apply_transform_point(base_to_camera, output_position)
            camera_to_output = self._lookup_transform(output_frame, camera_frame, "camera_to_output")
            if camera_to_output is None and self.require_camera_tf:
                self._warn_projection_once(
                    "vision_camera_to_output_missing",
                    "task1_vision: required TF %s <- %s unavailable; detection skipped",
                    output_frame,
                    camera_frame,
                )
                return None
            if camera_to_output is not None:
                source = "qpos_base_tf_camera_checked"
            else:
                self._warn_projection_once(
                    "vision_camera_to_output_missing",
                    "task1_vision: TF %s <- %s unavailable; output keeps qpos-derived %s position",
                    output_frame,
                    camera_frame,
                    self.base_frame_id,
                )
                source = "qpos_base_tf_camera_no_output_tf"
        else:
            if self.require_camera_tf:
                self._warn_projection_once(
                    "vision_base_to_camera_missing",
                    "task1_vision: required TF %s <- %s unavailable; detection skipped",
                    camera_frame,
                    output_frame,
                )
                return None
            self._warn_projection_once(
                "vision_base_to_camera_missing",
                "task1_vision: TF %s <- %s unavailable; camera_position omitted, output keeps qpos-derived %s position",
                camera_frame,
                output_frame,
                self.base_frame_id,
            )
            source = "qpos_base_no_camera_tf"

        reference = self._mujoco_camera_reference(truth_position, camera_position)
        return output_position, output_quat, camera_position, source, reference

    def _project_bbox(self, position, orientation, size):
        return self._project_bbox_from_tf(position, orientation, size)

    def _project_bbox_from_tf(self, position, orientation, size, source_frame=None):
        if self.camera_info is None:
            return None
        target_frame = self.camera_info.header.frame_id
        if not target_frame:
            return None
        source_frame = source_frame or self.output_frame_id
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.01),
            ).transform
        except Exception as exc:
            self._warn_projection_once("tf", "task1_vision: TF projection unavailable: %s", exc)
            return None

        half = [max(0.001, s) * 0.5 for s in size]
        pixels = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local = [sx * half[0], sy * half[1], sz * half[2]]
                    source_delta = _rotate_vector_by_quat_xyzw(local, orientation)
                    source_point = [position[i] + source_delta[i] for i in range(3)]
                    camera_point = _transform_point(transform, source_point)
                    pixel = _project_camera_point(self.camera_info, camera_point)
                    if pixel is not None:
                        pixels.append(pixel)
        if not pixels:
            return None

        min_x = min(p[0] for p in pixels)
        max_x = max(p[0] for p in pixels)
        min_y = min(p[1] for p in pixels)
        max_y = max(p[1] for p in pixels)
        width = float(self.camera_info.width or 1280)
        height = float(self.camera_info.height or 720)
        if max_x < 0.0 or min_x >= width or max_y < 0.0 or min_y >= height:
            return None

        min_x = _clip(min_x, 0.0, width - 1.0)
        max_x = _clip(max_x, 0.0, width - 1.0)
        min_y = _clip(min_y, 0.0, height - 1.0)
        max_y = _clip(max_y, 0.0, height - 1.0)
        return [
            [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5],
            [max(2.0, max_x - min_x), max(2.0, max_y - min_y)],
        ]

    # 加载与当前仿真一致的 MuJoCo 模型, 用 mj_forward 得到 body 真值位姿。
    def _mujoco_model_and_data(self):
        if not _MUJOCO_AVAILABLE:
            self._warn_projection_once(
                "mujoco_import",
                "task1_vision: mujoco unavailable, fallback to raw qpos truth: %s",
                _MUJOCO_IMPORT_ERROR,
            )
            return None
        model_path = rospy.get_param("legged_robot_scene_param", self.model_xml_path)
        if not model_path:
            return None
        model_path = os.path.abspath(str(model_path))
        if not os.path.isfile(model_path):
            self._warn_projection_once(
                "mujoco_model_missing",
                "task1_vision: MuJoCo model XML not found, fallback to raw qpos truth: %s",
                model_path,
            )
            return None
        if self.mj_model is None or self.mj_model_path != model_path:
            try:
                self.mj_model = mj.MjModel.from_xml_path(model_path)
                self.mj_data = mj.MjData(self.mj_model)
                self.mj_model_path = model_path
                rospy.loginfo("task1_vision: using MuJoCo body truth from %s", model_path)
            except Exception as exc:
                self._warn_projection_once(
                    "mujoco_model_load",
                    "task1_vision: failed to load MuJoCo model, fallback to raw qpos truth: %s",
                    exc,
                )
                return None
        return self.mj_model, self.mj_data

    # 用当前 /mujoco/qpos 前向计算指定 body 的世界位姿。
    def _body_pose_from_mujoco(self, body_name):
        if self.latest_qpos is None:
            return None
        model_data = self._mujoco_model_and_data()
        if model_data is None:
            return None
        model, data = model_data
        if len(self.latest_qpos) < model.nq:
            self._warn_projection_once(
                "mujoco_qpos_len",
                "task1_vision: qpos len %d < MuJoCo model nq %d, fallback to raw qpos truth",
                len(self.latest_qpos),
                model.nq,
            )
            return None
        try:
            data.qpos[: model.nq] = self.latest_qpos[: model.nq]
            mj.mj_forward(model, data)
            body = data.body(str(body_name))
            position = [float(v) for v in body.xpos]
            quat_xyzw = _quat_wxyz_to_xyzw([float(v) for v in body.xquat])
            return position, quat_xyzw
        except Exception as exc:
            self._warn_projection_once(
                "mujoco_body_" + str(body_name),
                "task1_vision: MuJoCo body %s unavailable, fallback to raw qpos truth: %s",
                str(body_name),
                exc,
            )
            return None

    # 从 /mujoco/qpos 读取机器人 base 在仿真世界中的位姿。
    def _base_pose_from_qpos(self):
        body_pose = self._body_pose_from_mujoco(self.base_frame_id)
        if body_pose is not None:
            return body_pose

        q = self.latest_qpos
        if q is None or len(q) < 7:
            return None

        base_addr = int(self._param("robot_base_qpos_addr", 0))
        if base_addr < 0 or base_addr + 6 >= len(q):
            self._warn_projection_once(
                "base_qpos",
                "task1_vision: robot base qpos addr %d outside qpos length %d",
                base_addr,
                len(q),
            )
            return None

        base_pos = [q[base_addr], q[base_addr + 1], q[base_addr + 2]]
        quat_base = _quat_wxyz_to_xyzw(q[base_addr + 3 : base_addr + 7])
        return base_pos, quat_base

    # 在没有 world TF 时, 用 qpos 把真值世界坐标转换成 base_link 相对坐标。
    def _output_from_qpos_base(self, truth_position, output_orientation):
        if self.output_frame_id != self.base_frame_id:
            return None
        base_pose = self._base_pose_from_qpos()
        if base_pose is None:
            return None

        base_pos, base_quat = base_pose
        inv_base = _quat_inverse(base_quat)
        delta = [truth_position[i] - base_pos[i] for i in range(3)]
        output_position = _rotate_vector_by_quat_xyzw(delta, inv_base)
        output_quat = _quat_multiply(inv_base, output_orientation)
        return output_position, output_quat

    # 只把 MuJoCo world->camera 真值作为参考对比, 不参与 task1_vision 输出。
    def _mujoco_camera_reference(self, truth_position, camera_position):
        result = {
            "reference_camera_position": [],
            "camera_reference_error": None,
        }
        camera_pose = self._head_camera_pose_from_qpos()
        if camera_pose is None:
            return result

        reference = self._world_point_to_ros_camera_point(truth_position, camera_pose)
        result["reference_camera_position"] = reference
        if camera_position is not None:
            result["camera_reference_error"] = math.sqrt(
                sum((camera_position[i] - reference[i]) ** 2 for i in range(3))
            )
        return result

    def _head_camera_pose_from_qpos(self):
        base_pose = self._base_pose_from_qpos()
        if base_pose is None:
            return None

        base_pos, quat_base = base_pose

        waist = self._joint_qpos_value("waist_yaw_joint", 0.0)
        head_yaw = self._joint_qpos_value("zhead_1_joint", 0.0)
        head_pitch = self._joint_qpos_value("zhead_2_joint", 0.0)

        pos, quat = base_pos, quat_base
        pos, quat = _compose_pose(pos, quat, [0.0, 0.0, 0.1114], _quat_from_axis_angle([0.0, 0.0, 1.0], waist))
        pos, quat = _compose_pose(pos, quat, [-0.003, 0.0, 0.49535], _quat_from_axis_angle([0.0, 0.0, 1.0], head_yaw))
        pos, quat = _compose_pose(pos, quat, [0.0, 0.0, 0.0], _quat_from_axis_angle([0.0, 1.0, 0.0], head_pitch))

        camera_pos = _parse_vec(self._param("mujoco_head_camera_pos", [0.121, 0.0, 0.0345]), 3)
        camera_quat = _quat_wxyz_to_xyzw(
            _parse_vec(self._param("mujoco_head_camera_quat_wxyz", [0.612372, 0.353553, -0.353553, -0.612372]), 4)
        )
        pos, quat = _compose_pose(pos, quat, camera_pos, camera_quat)
        return pos, quat

    def _joint_qpos_value(self, joint_name, default):
        addr = self.joint_index.qpos_by_joint.get(joint_name)
        if addr is None or addr >= len(self.latest_qpos):
            self._warn_projection_once(
                "joint_" + joint_name,
                "task1_vision: joint %s not found in qpos map; using %.3f",
                joint_name,
                default,
            )
            return default
        return self.latest_qpos[addr]

    def _world_point_to_ros_camera_point(self, world_point, camera_pose):
        camera_pos, camera_quat = camera_pose
        delta = [world_point[i] - camera_pos[i] for i in range(3)]
        camera_local = _rotate_vector_by_quat_xyzw(delta, _quat_inverse(camera_quat))
        return [camera_local[0], -camera_local[1], -camera_local[2]]

    def _warn_projection_once(self, key, fmt, *args):
        if key in self._projection_warned:
            return
        self._projection_warned.add(key)
        rospy.logwarn(fmt, *args)

    def _startup_missing_requirements(self):
        missing = []
        if self.latest_sensor_time is None:
            missing.append(self.sensor_time_topic + ".sensor_time")
        if self.vision_mode == 0 and self.latest_qpos is None:
            missing.append(self.qpos_topic)
        if self.camera_info is None:
            missing.append(self.camera_info_topic)
        if self.latest_depth is None:
            missing.append(self.depth_image_topic)
        if (
            self.vision_mode == 1
            or bool(self._param("depth_color_filter_enabled", True))
        ) and self.latest_color is None:
            missing.append(self.head_image_topic)

        camera_frame = self._camera_frame_id()
        if self.require_camera_tf and self.camera_info is not None:
            try:
                self.tf_buffer.lookup_transform(
                    camera_frame,
                    self.output_frame_id,
                    rospy.Time(0),
                    rospy.Duration(float(self._param("tf_timeout", 0.02))),
                )
            except Exception:
                missing.append("TF {} <- {}".format(camera_frame, self.output_frame_id))
            try:
                self.tf_buffer.lookup_transform(
                    self.output_frame_id,
                    self.latest_depth_frame_id or camera_frame,
                    rospy.Time(0),
                    rospy.Duration(float(self._param("tf_timeout", 0.02))),
                )
            except Exception:
                missing.append("TF {} <- {}".format(self.output_frame_id, self.latest_depth_frame_id or camera_frame))
        return missing

    # 启动阶段等待当前模式需要的图像、qpos 和关键 TF 齐全。
    def _wait_until_ready(self):
        if self.startup_wait_timeout <= 0.0:
            return True
        rate = rospy.Rate(5.0)
        last_report = 0.0
        start_sim_time = None
        previous_sim_time = None
        warned = False
        while not rospy.is_shutdown():
            missing = self._startup_missing_requirements()
            if not missing:
                rospy.loginfo("task1_vision: startup inputs ready; begin publishing detections")
                return True

            now_wall = time.monotonic()
            now_sim = self.latest_sensor_time
            if now_sim is not None:
                if previous_sim_time is not None and now_sim + 1e-12 < previous_sim_time:
                    rospy.logwarn(
                        "task1_vision: sensor_time moved backwards; restarting startup timeout"
                    )
                    start_sim_time = now_sim
                    warned = False
                elif start_sim_time is None:
                    start_sim_time = now_sim
                previous_sim_time = now_sim
                elapsed = now_sim - start_sim_time
            else:
                elapsed = 0.0

            if (
                not warned
                and self.startup_warn_time > 0.0
                and elapsed + 1e-12 >= self.startup_warn_time
                and elapsed + 1e-12 < self.startup_wait_timeout
            ):
                rospy.logwarn(
                    "task1_vision: startup inputs still missing after %.1fs simulation "
                    "time; continuing until %.1fs: %s",
                    self.startup_warn_time,
                    self.startup_wait_timeout,
                    ", ".join(missing),
                )
                warned = True
            if elapsed + 1e-12 >= self.startup_wait_timeout:
                rospy.logwarn(
                    "task1_vision: startup wait timed out after %.1fs simulation "
                    "time; still missing: %s",
                    self.startup_wait_timeout,
                    ", ".join(missing),
                )
                return False
            if now_wall - last_report >= 2.0:
                rospy.loginfo("task1_vision: waiting startup inputs: %s", ", ".join(missing))
                last_report = now_wall
            rate.sleep()
        return False

    def _truth_pose(self, spec):
        name = str(spec.get("name", ""))
        qpos_joint = str(spec.get("qpos_joint", name))
        position = None
        quat_xyzw = None

        if self.latest_qpos is None:
            return None
        body_pose = self._body_pose_from_mujoco(qpos_joint)
        if body_pose is not None:
            position, quat_xyzw = body_pose
        qpos_addr = self.joint_index.qpos_by_joint.get(qpos_joint)
        if position is None and qpos_addr is not None and qpos_addr + 6 < len(self.latest_qpos):
            q = self.latest_qpos
            position = [q[qpos_addr], q[qpos_addr + 1], q[qpos_addr + 2]]
            quat_xyzw = _quat_wxyz_to_xyzw(q[qpos_addr + 3 : qpos_addr + 7])

        if position is None or quat_xyzw is None:
            self._warn_projection_once(
                "missing_pose_" + name,
                "task1_vision: no pose available for %s; detection skipped",
                name,
            )
            return None

        offset = _parse_vec(spec.get("center_offset"), 3, [0.0, 0.0, 0.0])
        rotated_offset = _rotate_vector_by_quat_xyzw(offset, quat_xyzw)
        center = [position[i] + rotated_offset[i] for i in range(3)]
        return center, _yaw_from_quat_xyzw(quat_xyzw)

    def spin(self):
        if not self._wait_until_ready():
            rospy.signal_shutdown("task1_vision startup inputs unavailable")
            return
        rate = rospy.Rate(float(rospy.get_param("~publish_rate", 10.0)))
        while not rospy.is_shutdown():
            self.publish_once()
            rate.sleep()

    def publish_once(self):
        stamp = rospy.Time.now()
        bbox_candidates = []
        if self.vision_mode == 0:
            detections = []
            for spec in self.objects:
                detection = self._detect_object(spec)
                if detection is not None:
                    detections.append(detection)
        elif self._deployment_frozen:
            self._release_yolo_model()
            detections = copy.deepcopy(self._deployment_frozen_detections)
            bbox_candidates = copy.deepcopy(
                self._deployment_frozen_bbox_candidates
            )
        else:
            detections, bbox_candidates = self._detect_deployment_objects()
            if len(detections) == 4:
                self._deployment_last_bbox_candidates = copy.deepcopy(
                    bbox_candidates
                )

        if bool(self._param("publish_detection2d", True)):
            self.pub_2d.publish(self._make_detection2d_array(stamp, detections))
        if bool(self._param("publish_detection3d", True)):
            self.pub_3d.publish(self._make_detection3d_array(stamp, detections))
        if bool(self._param("publish_markers", True)):
            self.pub_markers.publish(self._make_marker_array(stamp, detections))
        if bool(self._param("publish_json", True)):
            self.pub_json.publish(
                String(
                    data=json.dumps(
                        {
                            "stamp": stamp.to_sec(),
                            "frame_id": self.output_frame_id,
                            "truth_frame_id": self.truth_frame_id,
                            "base_frame_id": self.base_frame_id,
                            "camera_frame_id": self._camera_frame_id(),
                            "color_stamp": (
                                self.latest_color_stamp.to_sec()
                                if self.latest_color_stamp is not None
                                else None
                            ),
                            "image_width": (
                                int(self.camera_info.width)
                                if self.camera_info is not None
                                else 0
                            ),
                            "image_height": (
                                int(self.camera_info.height)
                                if self.camera_info is not None
                                else 0
                            ),
                            "vision_mode": self.vision_mode,
                            "detection_mode": self.detection_mode,
                            "bbox_candidates": bbox_candidates,
                            "objects": detections,
                        }
                    )
                )
            )

    def _make_hypothesis(self, det):
        hyp = ObjectHypothesisWithPose()
        if hasattr(hyp, "id"):
            hyp.id = det["class_id"]
            hyp.score = det["confidence"]
        elif hasattr(hyp, "hypothesis"):
            hyp.hypothesis.id = det["class_id"]
            hyp.hypothesis.score = det["confidence"]
        hyp.pose.pose.position = _make_point(det["position"])
        hyp.pose.pose.orientation = _make_quaternion(det["orientation"])
        return hyp

    def _make_detection2d_array(self, stamp, detections):
        msg = Detection2DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_info.header.frame_id if self.camera_info is not None else self.frame_id
        for det in detections:
            item = Detection2D()
            item.header = msg.header
            if hasattr(item, "id"):
                item.id = det["name"]
            item.bbox = BoundingBox2D()
            item.bbox.center.x = det["bbox_center"][0]
            item.bbox.center.y = det["bbox_center"][1]
            item.bbox.center.theta = det["yaw"] if det["yaw_valid"] else 0.0
            item.bbox.size_x = det["bbox_size"][0]
            item.bbox.size_y = det["bbox_size"][1]
            item.results.append(self._make_hypothesis(det))
            msg.detections.append(item)
        return msg

    def _make_detection3d_array(self, stamp, detections):
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.output_frame_id
        for det in detections:
            item = Detection3D()
            item.header = msg.header
            if hasattr(item, "id"):
                item.id = det["name"]
            item.bbox = BoundingBox3D()
            item.bbox.center.position = _make_point(det["position"])
            item.bbox.center.orientation = _make_quaternion(det["orientation"])
            item.bbox.size = _make_vector3(det["size"])
            item.results.append(self._make_hypothesis(det))
            msg.detections.append(item)
        return msg

    def _make_marker_array(self, stamp, detections):
        markers = MarkerArray()
        for index, det in enumerate(detections):
            cube = Marker()
            cube.header.stamp = stamp
            cube.header.frame_id = self.output_frame_id
            cube.ns = "task1_vision_objects"
            cube.id = index * 2
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position = _make_point(det["position"])
            cube.pose.orientation = _make_quaternion(det["orientation"])
            cube.scale = _make_vector3(det["size"])
            color = det["color_rgba"]
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = color
            cube.lifetime = rospy.Duration(0.3)
            markers.markers.append(cube)

            text = Marker()
            text.header = cube.header
            text.ns = "task1_vision_labels"
            text.id = index * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = _make_point(
                [det["position"][0], det["position"][1], det["position"][2] + det["size"][2] * 0.7 + 0.03]
            )
            text.pose.orientation.w = 1.0
            text.scale.z = 0.035
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.95
            yaw_text = "{:.1f}deg".format(math.degrees(det["yaw"])) if det["yaw_valid"] else "yaw?"
            text.text = "{} {} {:.2f} {}".format(det["name"], det["class_name"], det["confidence"], yaw_text)
            text.lifetime = rospy.Duration(0.3)
            markers.markers.append(text)
        return markers


if __name__ == "__main__":
    rospy.init_node("task1_vision")
    Task1VisionNode().spin()
