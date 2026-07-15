#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""从场景三 qpos 投影五个 SMT 料盘的二维真值框。"""

import json
import math
import os
import time
import xml.etree.ElementTree as ET

import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float64MultiArray, String
import tf2_ros
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


def _parse_vec(value, length, default=None):
    if value is None:
        return list(default or [0.0] * length)
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    else:
        values = list(value)
    result = [float(item) for item in values[:length]]
    while len(result) < length:
        result.append(0.0)
    return result


def _clip(value, low, high):
    return max(low, min(high, value))


def _quat_wxyz_to_xyzw(quat):
    return [quat[1], quat[2], quat[3], quat[0]]


def _quat_normalize(quat):
    norm = math.sqrt(sum(float(value) ** 2 for value in quat))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(value) / norm for value in quat]


def _quat_inverse(quat):
    x, y, z, w = _quat_normalize(quat)
    return [-x, -y, -z, w]


def _quat_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _rotate_vector(vector, quat):
    x, y, z = vector
    qx, qy, qz, qw = _quat_normalize(quat)
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    return [
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    ]


def _yaw_from_quat(quat):
    x, y, z, w = _quat_normalize(quat)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _transform_point(transform, point):
    rotation = [
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ]
    rotated = _rotate_vector(point, rotation)
    return [
        rotated[0] + transform.translation.x,
        rotated[1] + transform.translation.y,
        rotated[2] + transform.translation.z,
    ]


def _project_camera_point(camera_info, point):
    x, y, z = point
    if z <= 0.03:
        return None
    matrix = camera_info.K
    if len(matrix) < 6 or matrix[0] <= 0.0 or matrix[4] <= 0.0:
        return None
    return [matrix[0] * x / z + matrix[2], matrix[4] * y / z + matrix[5]]


class MujocoJointIndex:
    """按照 MuJoCo XML 展开后的关节顺序建立 qpos 地址。"""

    def __init__(self, xml_path):
        self.xml_path = os.path.abspath(xml_path)
        self.qpos_by_joint = {}
        self.nq = 0

    def load(self):
        self.qpos_by_joint = {}
        self.nq = 0
        self._visit_xml(self.xml_path)
        rospy.loginfo(
            "task3_vision: indexed %d qpos values from %s",
            self.nq,
            self.xml_path,
        )

    def _visit_xml(self, xml_path):
        if not os.path.isfile(xml_path):
            raise RuntimeError("MuJoCo XML not found: {}".format(xml_path))
        root = ET.parse(xml_path).getroot()
        current_dir = os.path.dirname(xml_path)
        for child in list(root):
            if child.tag == "include":
                include_file = child.get("file", "")
                if include_file:
                    if not os.path.isabs(include_file):
                        include_file = os.path.normpath(
                            os.path.join(current_dir, include_file)
                        )
                    self._visit_xml(include_file)
            elif child.tag == "worldbody":
                self._visit_worldbody(child)

    def _visit_worldbody(self, worldbody):
        for element in worldbody.iter():
            if element.tag == "freejoint":
                self._register(element.get("name"), 7)
            elif element.tag == "joint":
                joint_type = element.get("type", "hinge")
                width = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
                self._register(element.get("name"), width)

    def _register(self, name, width):
        if name:
            self.qpos_by_joint[str(name)] = self.nq
        self.nq += int(width)


class Task3VisionNode:
    def __init__(self):
        self.scene = str(rospy.get_param("~scene", "scene3"))
        self.frame_id = str(rospy.get_param("~frame_id", "base_link"))
        self.base_frame_id = str(
            rospy.get_param("~base_frame_id", self.frame_id)
        )
        self.qpos_topic = str(rospy.get_param("~qpos_topic", "/mujoco/qpos"))
        self.camera_info_topic = str(
            rospy.get_param("~camera_info_topic", "/cam_h/color/camera_info")
        )
        self.head_image_topic = str(
            rospy.get_param(
                "~head_image_topic", "/cam_h/color/image_raw/compressed"
            )
        )
        self.startup_wait_timeout = float(
            rospy.get_param("~startup_wait_timeout", 60.0)
        )
        self.objects = list(rospy.get_param("~objects", []))
        if not self.objects:
            raise RuntimeError("task3_vision requires a non-empty objects list")

        self.model_xml_path = self._resolve_model_xml()
        self.joint_index = MujocoJointIndex(self.model_xml_path)
        self.joint_index.load()
        self._validate_object_joints()

        self.latest_qpos = None
        self.camera_info = None
        self.latest_color_stamp = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._warned = set()

        self.pub_2d = rospy.Publisher(
            "~detections2d", Detection2DArray, queue_size=1
        )
        self.pub_json = rospy.Publisher("~objects_json", String, queue_size=1)
        self.qpos_sub = rospy.Subscriber(
            self.qpos_topic,
            Float64MultiArray,
            self._qpos_callback,
            queue_size=1,
        )
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self._camera_info_callback,
            queue_size=1,
        )
        self.color_sub = rospy.Subscriber(
            self.head_image_topic,
            CompressedImage,
            self._color_callback,
            queue_size=1,
        )

    def _resolve_model_xml(self):
        configured = str(rospy.get_param("~model_xml", "")).strip()
        if configured:
            return os.path.abspath(configured)
        runtime_path = str(rospy.get_param("legged_robot_scene_param", "")).strip()
        if runtime_path and os.path.isfile(runtime_path):
            return os.path.abspath(runtime_path)
        try:
            import rospkg

            simulator_path = rospkg.RosPack().get_path("challenge_cup_simulator")
        except Exception:
            simulator_path = os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "..",
                    "challenge_cup_simulator",
                )
            )
        xml_dir = os.path.join(
            simulator_path, "models", "biped_s52", "xml"
        )
        active = os.path.join(xml_dir, "_scene_{}_active.xml".format(self.scene))
        if os.path.isfile(active):
            return active
        return os.path.join(xml_dir, "{}.xml".format(self.scene))

    def _validate_object_joints(self):
        missing = []
        for spec in self.objects:
            name = str(spec.get("name", ""))
            joint = str(spec.get("qpos_joint", name))
            if joint not in self.joint_index.qpos_by_joint:
                missing.append(joint)
        if missing:
            raise RuntimeError(
                "qpos joints missing from scene XML: {}".format(", ".join(missing))
            )

    def _qpos_callback(self, msg):
        self.latest_qpos = list(msg.data)

    def _camera_info_callback(self, msg):
        self.camera_info = msg

    def _color_callback(self, msg):
        self.latest_color_stamp = msg.header.stamp

    def _camera_frame_id(self):
        if self.camera_info is not None and self.camera_info.header.frame_id:
            return str(self.camera_info.header.frame_id)
        return str(rospy.get_param("~camera_frame_id", "Head Camera View"))

    def _warn_once(self, key, message, *args):
        if key in self._warned:
            return
        self._warned.add(key)
        rospy.logwarn(message, *args)

    def _free_joint_pose(self, joint_name):
        if self.latest_qpos is None:
            return None
        address = self.joint_index.qpos_by_joint.get(str(joint_name))
        if address is None or address + 6 >= len(self.latest_qpos):
            return None
        values = self.latest_qpos
        position = [float(values[address + index]) for index in range(3)]
        orientation = _quat_wxyz_to_xyzw(values[address + 3 : address + 7])
        return position, _quat_normalize(orientation)

    def _base_pose(self):
        if self.latest_qpos is None:
            return None
        address = int(rospy.get_param("~robot_base_qpos_addr", 0))
        if address < 0 or address + 6 >= len(self.latest_qpos):
            return None
        values = self.latest_qpos
        return (
            [float(values[address + index]) for index in range(3)],
            _quat_normalize(
                _quat_wxyz_to_xyzw(values[address + 3 : address + 7])
            ),
        )

    def _object_pose(self, spec):
        name = str(spec.get("name", ""))
        joint_name = str(spec.get("qpos_joint", name))
        object_pose = self._free_joint_pose(joint_name)
        base_pose = self._base_pose()
        if object_pose is None or base_pose is None:
            return None

        world_position, world_orientation = object_pose
        center_offset = _parse_vec(
            spec.get("center_offset"), 3, [0.0, 0.0, 0.0]
        )
        rotated_offset = _rotate_vector(center_offset, world_orientation)
        world_center = [
            world_position[index] + rotated_offset[index] for index in range(3)
        ]

        base_position, base_orientation = base_pose
        world_to_base = _quat_inverse(base_orientation)
        relative_world = [
            world_center[index] - base_position[index] for index in range(3)
        ]
        base_center = _rotate_vector(relative_world, world_to_base)
        base_object_orientation = _quat_normalize(
            _quat_multiply(world_to_base, world_orientation)
        )
        return (
            base_center,
            base_object_orientation,
            world_center,
            world_orientation,
        )

    def _camera_transform(self):
        camera_frame = self._camera_frame_id()
        try:
            return self.tf_buffer.lookup_transform(
                camera_frame,
                self.base_frame_id,
                rospy.Time(0),
                rospy.Duration(float(rospy.get_param("~tf_timeout", 0.02))),
            ).transform
        except Exception as exc:
            self._warn_once(
                "camera_tf",
                "task3_vision: TF %s <- %s unavailable: %s",
                camera_frame,
                self.base_frame_id,
                exc,
            )
            return None

    def _project_bbox(self, position, orientation, size, camera_transform):
        if self.camera_info is None or camera_transform is None:
            return None
        half_size = [max(0.001, float(value)) * 0.5 for value in size]
        pixels = []
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                for sign_z in (-1.0, 1.0):
                    local_corner = [
                        sign_x * half_size[0],
                        sign_y * half_size[1],
                        sign_z * half_size[2],
                    ]
                    base_delta = _rotate_vector(local_corner, orientation)
                    base_corner = [
                        position[index] + base_delta[index] for index in range(3)
                    ]
                    pixel = _project_camera_point(
                        self.camera_info,
                        _transform_point(camera_transform, base_corner),
                    )
                    if pixel is not None:
                        pixels.append(pixel)
        if not pixels:
            return None

        minimum_x = min(pixel[0] for pixel in pixels)
        maximum_x = max(pixel[0] for pixel in pixels)
        minimum_y = min(pixel[1] for pixel in pixels)
        maximum_y = max(pixel[1] for pixel in pixels)
        image_width = float(self.camera_info.width)
        image_height = float(self.camera_info.height)
        if image_width <= 1.0 or image_height <= 1.0:
            return None
        if (
            maximum_x < 0.0
            or minimum_x >= image_width
            or maximum_y < 0.0
            or minimum_y >= image_height
        ):
            return None

        minimum_x = _clip(minimum_x, 0.0, image_width - 1.0)
        maximum_x = _clip(maximum_x, 0.0, image_width - 1.0)
        minimum_y = _clip(minimum_y, 0.0, image_height - 1.0)
        maximum_y = _clip(maximum_y, 0.0, image_height - 1.0)
        width = maximum_x - minimum_x
        height = maximum_y - minimum_y
        if width <= 1.0 or height <= 1.0:
            return None
        return [
            [(minimum_x + maximum_x) * 0.5, (minimum_y + maximum_y) * 0.5],
            [width, height],
        ]

    def _detect_objects(self):
        camera_transform = self._camera_transform()
        if camera_transform is None:
            return []
        detections = []
        for spec in self.objects:
            pose = self._object_pose(spec)
            name = str(spec.get("name", ""))
            if pose is None:
                self._warn_once(
                    "pose_" + name,
                    "task3_vision: qpos pose unavailable for %s",
                    name,
                )
                continue
            position, orientation, world_position, world_orientation = pose
            size = _parse_vec(spec.get("size"), 3, [0.18, 0.18, 0.032])
            bbox = self._project_bbox(
                position, orientation, size, camera_transform
            )
            if bbox is None:
                continue
            camera_position = _transform_point(camera_transform, position)
            detections.append(
                {
                    "name": name,
                    "qpos_joint": str(spec.get("qpos_joint", name)),
                    "class_id": int(spec.get("class_id", 0)),
                    "class_name": str(spec.get("class_name", "tray")),
                    "level": str(spec.get("level", "")),
                    "confidence": _clip(
                        float(rospy.get_param("~confidence", 1.0)), 0.0, 1.0
                    ),
                    "position": position,
                    "position_source": "qpos_base_truth",
                    "truth_position": position,
                    "truth_position_source": "qpos_base_truth",
                    "camera_position": camera_position,
                    "truth_camera_position": camera_position,
                    "orientation": orientation,
                    "world_position": world_position,
                    "world_orientation": world_orientation,
                    "size": size,
                    "yaw": _yaw_from_quat(orientation),
                    "yaw_valid": True,
                    "bbox_center": bbox[0],
                    "bbox_size": bbox[1],
                    "bbox_projected": True,
                    "bbox_source": "qpos_projected_3d_corners",
                }
            )
        detections.sort(key=lambda item: item["name"])
        return detections

    def _make_hypothesis(self, detection):
        hypothesis = ObjectHypothesisWithPose()
        if hasattr(hypothesis, "id"):
            hypothesis.id = detection["class_id"]
            hypothesis.score = detection["confidence"]
        elif hasattr(hypothesis, "hypothesis"):
            hypothesis.hypothesis.id = detection["class_id"]
            hypothesis.hypothesis.score = detection["confidence"]
        hypothesis.pose.pose.position = Point(
            x=detection["position"][0],
            y=detection["position"][1],
            z=detection["position"][2],
        )
        hypothesis.pose.pose.orientation.x = detection["orientation"][0]
        hypothesis.pose.pose.orientation.y = detection["orientation"][1]
        hypothesis.pose.pose.orientation.z = detection["orientation"][2]
        hypothesis.pose.pose.orientation.w = detection["orientation"][3]
        return hypothesis

    def _make_detection2d_array(self, stamp, detections):
        message = Detection2DArray()
        message.header.stamp = stamp
        message.header.frame_id = self._camera_frame_id()
        for detection in detections:
            item = Detection2D()
            item.header = message.header
            if hasattr(item, "id"):
                item.id = detection["name"]
            item.bbox = BoundingBox2D()
            item.bbox.center.x = detection["bbox_center"][0]
            item.bbox.center.y = detection["bbox_center"][1]
            item.bbox.center.theta = 0.0
            item.bbox.size_x = detection["bbox_size"][0]
            item.bbox.size_y = detection["bbox_size"][1]
            item.results.append(self._make_hypothesis(detection))
            message.detections.append(item)
        return message

    def _startup_missing(self):
        missing = []
        if self.latest_qpos is None:
            missing.append(self.qpos_topic)
        elif len(self.latest_qpos) < self.joint_index.nq:
            missing.append(
                "{} length {}<{}".format(
                    self.qpos_topic, len(self.latest_qpos), self.joint_index.nq
                )
            )
        if self.camera_info is None:
            missing.append(self.camera_info_topic)
        if self.latest_color_stamp is None:
            missing.append(self.head_image_topic)
        if self.camera_info is not None:
            try:
                available = self.tf_buffer.can_transform(
                    self._camera_frame_id(),
                    self.base_frame_id,
                    rospy.Time(0),
                    rospy.Duration(0.01),
                )
            except Exception:
                available = False
            if not available:
                missing.append(
                    "TF {}<-{}".format(
                        self._camera_frame_id(), self.base_frame_id
                    )
                )
        return missing

    def _wait_until_ready(self):
        deadline = time.monotonic() + self.startup_wait_timeout
        last_report = 0.0
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            missing = self._startup_missing()
            if not missing:
                rospy.loginfo(
                    "task3_vision: startup ready; publishing five tray qpos boxes"
                )
                return True
            now = time.monotonic()
            if now >= deadline:
                rospy.logerr(
                    "task3_vision: startup timed out; missing: %s",
                    ", ".join(missing),
                )
                return False
            if now - last_report >= 2.0:
                rospy.loginfo(
                    "task3_vision: waiting for %s", ", ".join(missing)
                )
                last_report = now
            rate.sleep()
        return False

    def publish_once(self):
        stamp = rospy.Time.now()
        detections = self._detect_objects()
        if bool(rospy.get_param("~publish_detection2d", True)):
            self.pub_2d.publish(self._make_detection2d_array(stamp, detections))
        if bool(rospy.get_param("~publish_json", True)):
            payload = {
                "stamp": stamp.to_sec(),
                "color_stamp": (
                    self.latest_color_stamp.to_sec()
                    if self.latest_color_stamp is not None
                    else None
                ),
                "scene": self.scene,
                "frame_id": self.frame_id,
                "truth_frame_id": self.frame_id,
                "base_frame_id": self.base_frame_id,
                "camera_frame_id": self._camera_frame_id(),
                "image_width": int(self.camera_info.width),
                "image_height": int(self.camera_info.height),
                "detection_mode": "qpos_projected_3d_corners",
                "objects": detections,
            }
            self.pub_json.publish(
                String(data=json.dumps(payload, ensure_ascii=False))
            )

    def spin(self):
        if not self._wait_until_ready():
            rospy.signal_shutdown("task3_vision startup inputs unavailable")
            return
        rate = rospy.Rate(float(rospy.get_param("~publish_rate", 10.0)))
        while not rospy.is_shutdown():
            self.publish_once()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("task3_vision")
    Task3VisionNode().spin()
