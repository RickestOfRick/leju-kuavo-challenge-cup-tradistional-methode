#!/usr/bin/env python3
"""发布场景二工件真值位姿和由实时 qpos 投影得到的类别检测框。"""

import copy
import json
import os

import rospkg
import rospy
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float64MultiArray, String
import tf2_ros
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

try:
    import mujoco as mj
except ImportError as exc:
    mj = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None

from task2_vision.layout import (
    INSTANCE_NAMING_RELATIVE_POSITION,
    OBJECT_NAMES,
    POSITION_SOURCE,
    build_published_objects,
    load_scene2_layout,
    normalize_instance_naming_method,
    update_published_objects_base_frame,
)
from task2_vision.mesh_geometry import load_body_visual_vertices


FRAME_ID = "base_link"
CLASS_IDS = {
    "part_type_a": 0,
    "part_type_b": 1,
    "part_type_c": 2,
}
BBOX_SOURCE = "mujoco_qpos_projected_visual_mesh"
COLLISION_BBOX_SOURCE = "mujoco_qpos_projected_collision_geometry"


def _quat_inverse_wxyz(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    norm_squared = w * w + x * x + y * y + z * z
    if norm_squared <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [
        w / norm_squared,
        -x / norm_squared,
        -y / norm_squared,
        -z / norm_squared,
    ]


def _rotate_vector_wxyz(quaternion, vector):
    w, x, y, z = [float(value) for value in quaternion]
    vx, vy, vz = [float(value) for value in vector]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _transform_point(transform, point):
    quaternion = [
        float(transform.rotation.w),
        float(transform.rotation.x),
        float(transform.rotation.y),
        float(transform.rotation.z),
    ]
    rotated = _rotate_vector_wxyz(quaternion, point)
    return [
        rotated[0] + float(transform.translation.x),
        rotated[1] + float(transform.translation.y),
        rotated[2] + float(transform.translation.z),
    ]


def _project_camera_point(camera_info, point):
    x, y, z = [float(value) for value in point]
    if z <= 0.03 or len(camera_info.K) < 6:
        return None
    return [
        float(camera_info.K[0]) * x / z + float(camera_info.K[2]),
        float(camera_info.K[4]) * y / z + float(camera_info.K[5]),
    ]


class Task2VisionNode:
    """Publish live base-frame truth and head-camera boxes from the same qpos."""

    def __init__(self):
        self.seed = int(rospy.get_param("~seed", 0))
        self.vision_mode = int(rospy.get_param("~vision_mode", 0))
        if self.vision_mode not in (0, 1):
            raise RuntimeError("task2_vision vision_mode only supports 0 or 1")
        self.requested_instance_naming_method = normalize_instance_naming_method(
            rospy.get_param("~instance_naming_method", "relative_position")
        )
        # Mode 1 must not use truth yaw for naming. Relative-position names are
        # temporary ROI identities; task2_depth_xy remaps them from fitted yaw.
        self.instance_naming_method = (
            self.requested_instance_naming_method
            if self.vision_mode == 0
            else INSTANCE_NAMING_RELATIVE_POSITION
        )
        self.publish_hz = max(0.1, float(rospy.get_param("~publish_hz", 2.0)))
        self.qpos_topic = str(rospy.get_param("~qpos_topic", "/mujoco/qpos"))
        self.base_qpos_addr = int(rospy.get_param("~robot_base_qpos_addr", 0))
        self.qpos_timeout = float(rospy.get_param("~qpos_timeout", 20.0))
        self.camera_info_topic = str(
            rospy.get_param("~camera_info_topic", "/cam_h/color/camera_info")
        )
        self.camera_frame_id = str(
            rospy.get_param("~camera_frame_id", "Head Camera View")
        )
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.02))
        self.bbox_padding_pixels = max(
            0.0, float(rospy.get_param("~bbox_padding_pixels", 2.0))
        )
        self.latest_qpos = None
        self.camera_info = None
        self._warned = set()

        simulator_path = rospkg.RosPack().get_path("challenge_cup_simulator")
        layout = load_scene2_layout(simulator_path, self.seed)
        initial_qpos = self._read_initial_qpos()
        base_position, base_quat_wxyz = self._base_pose_from_raw_qpos(initial_qpos)
        self.initial_world_base_position = list(base_position)
        self.initial_world_base_quat_wxyz = list(base_quat_wxyz)
        self.world_base_position = list(base_position)
        self.world_base_quat_wxyz = list(base_quat_wxyz)
        self.truth_is_dynamic = False
        self.objects = build_published_objects(
            layout,
            base_position,
            base_quat_wxyz,
            self.instance_naming_method,
        )

        self.model, self.data, self.model_path = self._load_mujoco_model()
        self.visual_vertices_by_body = self._load_visual_mesh_vertices(self.model_path)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
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
        self.json_pub = rospy.Publisher("~objects_json", String, queue_size=1, latch=True)
        self.poses_pub = rospy.Publisher("~poses", PoseArray, queue_size=1, latch=True)
        self.detections2d_pub = rospy.Publisher(
            "~detections2d", Detection2DArray, queue_size=1
        )

        rospy.loginfo(
            "task2_vision: mode=%d loaded %d scene2 objects from seed=%d "
            "ROI naming=%s requested naming=%s",
            self.vision_mode,
            len(self.objects),
            self.seed,
            self.instance_naming_method,
            self.requested_instance_naming_method,
        )
        for item in self.objects:
            rospy.loginfo(
                "task2_vision: motion=%s ground_truth=%s position=%s yaw=%.3f deg",
                item["name"],
                item["ground_truth_name"],
                [round(value, 6) for value in item["position"]],
                item["yaw_deg"],
            )

    def _warn_once(self, key, message, *args):
        if key in self._warned:
            return
        self._warned.add(key)
        rospy.logwarn(message, *args)

    def _read_initial_qpos(self):
        message = rospy.wait_for_message(
            self.qpos_topic, Float64MultiArray, timeout=self.qpos_timeout
        )
        values = list(message.data)
        self.latest_qpos = values
        return values

    def _base_pose_from_raw_qpos(self, values):
        address = self.base_qpos_addr
        if address < 0 or address + 6 >= len(values):
            raise RuntimeError(
                "robot base qpos address {} outside qpos length {}".format(
                    address, len(values)
                )
            )
        return (
            [float(values[address + index]) for index in range(3)],
            [float(values[address + 3 + index]) for index in range(4)],
        )

    def _load_mujoco_model(self):
        if mj is None or not hasattr(mj, "MjModel"):
            rospy.logwarn(
                "task2_vision: qpos category boxes disabled because the official "
                "MuJoCo Python binding is unavailable: %s",
                _MUJOCO_IMPORT_ERROR or "mujoco.MjModel is missing",
            )
            return None, None, None
        model_path = str(rospy.get_param("legged_robot_scene_param", ""))
        if not model_path:
            rospy.logwarn(
                "task2_vision: qpos category boxes disabled because ROS param "
                "legged_robot_scene_param is missing"
            )
            return None, None, None
        model_path = os.path.abspath(model_path)
        if not os.path.isfile(model_path):
            rospy.logwarn(
                "task2_vision: qpos category boxes disabled because MuJoCo scene "
                "XML does not exist: %s",
                model_path,
            )
            return None, None, None
        try:
            model = mj.MjModel.from_xml_path(model_path)
            data = mj.MjData(model)
        except Exception as exc:
            rospy.logwarn(
                "task2_vision: qpos category boxes disabled because MuJoCo model "
                "loading failed: %s",
                exc,
            )
            return None, None, None
        rospy.loginfo("task2_vision: qpos bbox model=%s", model_path)
        return model, data, model_path

    def _qpos_callback(self, message):
        self.latest_qpos = list(message.data)

    def _camera_info_callback(self, message):
        self.camera_info = message

    def _forward_latest_qpos(self):
        if self.model is None or self.data is None:
            return False
        qpos = self.latest_qpos
        if qpos is None:
            return False
        if len(qpos) < self.model.nq:
            self._warn_once(
                "qpos_length",
                "task2_vision: qpos length %d is smaller than model nq %d",
                len(qpos),
                self.model.nq,
            )
            return False
        self.data.qpos[: self.model.nq] = qpos[: self.model.nq]
        mj.mj_forward(self.model, self.data)
        return True

    def _load_visual_mesh_vertices(self, model_path):
        if not model_path:
            return {}
        try:
            required_bodies = {
                str(item["ground_truth_name"]) for item in self.objects
            }
            by_body = load_body_visual_vertices(model_path, required_bodies)
            for body_name, body_vertices in sorted(by_body.items()):
                rospy.loginfo(
                    "task2_vision: visual bbox mesh %s vertices=%d",
                    body_name,
                    len(body_vertices),
                )
            missing = sorted(required_bodies - set(by_body))
            if missing:
                rospy.logwarn(
                    "task2_vision: visual mesh unavailable for %s; collision bbox "
                    "fallback will be used",
                    ", ".join(missing),
                )
            return by_body
        except Exception as exc:
            rospy.logwarn(
                "task2_vision: visual mesh loading failed; collision bbox fallback "
                "will be used: %s",
                exc,
            )
            return {}

    def _geom_half_extents(self, geom_id):
        geom_type = int(self.model.geom_type[geom_id])
        size = [float(value) for value in self.model.geom_size[geom_id]]
        if geom_type == int(mj.mjtGeom.mjGEOM_BOX):
            return size[:3]
        if geom_type == int(mj.mjtGeom.mjGEOM_SPHERE):
            return [size[0], size[0], size[0]]
        if geom_type in (
            int(mj.mjtGeom.mjGEOM_CAPSULE),
            int(mj.mjtGeom.mjGEOM_CYLINDER),
        ):
            axial = size[1]
            if geom_type == int(mj.mjtGeom.mjGEOM_CAPSULE):
                axial += size[0]
            return [size[0], size[0], axial]
        if geom_type == int(mj.mjtGeom.mjGEOM_ELLIPSOID):
            return size[:3]
        return None

    def _body_collision_corners_world(self, body_name):
        try:
            body_id = int(self.model.body(str(body_name)).id)
        except KeyError:
            self._warn_once(
                "body_{}".format(body_name),
                "task2_vision: MuJoCo body not found: %s",
                body_name,
            )
            return []

        first_geom = int(self.model.body_geomadr[body_id])
        geom_count = int(self.model.body_geomnum[body_id])
        corners = []
        for geom_id in range(first_geom, first_geom + geom_count):
            geom_name = str(
                mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
            )
            if "collision" not in geom_name:
                continue
            half = self._geom_half_extents(geom_id)
            if half is None:
                self._warn_once(
                    "geom_type_{}".format(geom_name),
                    "task2_vision: unsupported collision geom type for %s",
                    geom_name,
                )
                continue
            center = self.data.geom_xpos[geom_id]
            rotation = [
                float(value)
                for value in self.data.geom_xmat[geom_id].reshape(-1)
            ]
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        local = [sx * half[0], sy * half[1], sz * half[2]]
                        corners.append(
                            [
                                float(center[row])
                                + sum(
                                    rotation[row * 3 + column] * local[column]
                                    for column in range(3)
                                )
                                for row in range(3)
                            ]
                        )
        return corners

    def _body_visual_vertices_world(self, body_name):
        local_vertices = self.visual_vertices_by_body.get(str(body_name), [])
        if not local_vertices:
            return []
        try:
            body = self.data.body(str(body_name))
        except KeyError:
            return []
        center = [float(value) for value in body.xpos]
        rotation = [float(value) for value in body.xmat.reshape(-1)]
        return [
            [
                center[row]
                + sum(
                    rotation[row * 3 + column] * vertex[column]
                    for column in range(3)
                )
                for row in range(3)
            ]
            for vertex in local_vertices
        ]

    def _camera_transform(self):
        if self.camera_info is None:
            return None
        camera_frame = self.camera_info.header.frame_id or self.camera_frame_id
        try:
            return self.tf_buffer.lookup_transform(
                camera_frame,
                FRAME_ID,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout),
            ).transform
        except Exception as exc:
            self._warn_once(
                "camera_tf",
                "task2_vision: TF %s <- %s unavailable: %s",
                camera_frame,
                FRAME_ID,
                exc,
            )
            return None

    def _project_object_bbox(self, body_name, world_to_base_quat, base_position, transform):
        pixels = []
        world_points = self._body_visual_vertices_world(body_name)
        bbox_source = BBOX_SOURCE
        if not world_points:
            world_points = self._body_collision_corners_world(body_name)
            bbox_source = COLLISION_BBOX_SOURCE
        for world_point in world_points:
            relative_world = [
                world_point[index] - base_position[index] for index in range(3)
            ]
            base_point = _rotate_vector_wxyz(world_to_base_quat, relative_world)
            camera_point = _transform_point(transform, base_point)
            pixel = _project_camera_point(self.camera_info, camera_point)
            if pixel is not None:
                pixels.append(pixel)
        if not pixels:
            return None

        padding = self.bbox_padding_pixels
        min_x = min(point[0] for point in pixels) - padding
        max_x = max(point[0] for point in pixels) + padding
        min_y = min(point[1] for point in pixels) - padding
        max_y = max(point[1] for point in pixels) + padding
        width = float(self.camera_info.width or 1280)
        height = float(self.camera_info.height or 720)
        if max_x < 0.0 or min_x >= width or max_y < 0.0 or min_y >= height:
            return None
        min_x = max(0.0, min(width - 1.0, min_x))
        max_x = max(0.0, min(width - 1.0, max_x))
        min_y = max(0.0, min(height - 1.0, min_y))
        max_y = max(0.0, min(height - 1.0, max_y))
        return {
            "center": [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5],
            "size": [max(2.0, max_x - min_x), max(2.0, max_y - min_y)],
            "source": bbox_source,
        }

    def _objects_with_qpos_bboxes(self):
        objects = copy.deepcopy(self.objects)
        for item in objects:
            item["class_id"] = CLASS_IDS[item["class_name"]]
            item["confidence"] = 1.0
            item["bbox_projected"] = False
            item["bbox_source"] = BBOX_SOURCE

        if not self._forward_latest_qpos():
            return objects

        base_body = self.data.body(FRAME_ID)
        base_position = [float(value) for value in base_body.xpos]
        base_quat_wxyz = [float(value) for value in base_body.xquat]
        world_to_base_quat = _quat_inverse_wxyz(
            base_quat_wxyz
        )
        world_poses = {}
        for item in objects:
            body = self.data.body(str(item["ground_truth_name"]))
            world_poses[str(item["ground_truth_name"])] = (
                [float(value) for value in body.xpos],
                [float(value) for value in body.xquat],
            )
        objects = update_published_objects_base_frame(
            objects,
            base_position,
            base_quat_wxyz,
            world_poses,
        )
        self.world_base_position = list(base_position)
        self.world_base_quat_wxyz = list(base_quat_wxyz)
        self.truth_is_dynamic = True

        if self.camera_info is None:
            return objects
        transform = self._camera_transform()
        if transform is None:
            return objects
        for item in objects:
            bbox = self._project_object_bbox(
                item["ground_truth_name"],
                world_to_base_quat,
                base_position,
                transform,
            )
            if bbox is None:
                continue
            item["bbox_center"] = bbox["center"]
            item["bbox_size"] = bbox["size"]
            item["bbox_source"] = bbox["source"]
            item["bbox_projected"] = True
        return objects

    def _make_json(self, stamp, objects):
        payload = {
            "stamp": stamp.to_sec(),
            "frame_id": FRAME_ID,
            "scene": "scene2",
            "seed": self.seed,
            "vision_mode": self.vision_mode,
            "vision_mode_name": (
                "full_truth" if self.vision_mode == 0 else "rgbd_cad_pose"
            ),
            "frozen": True,
            "locked": self.vision_mode == 0,
            "ready_for_motion": self.vision_mode == 0,
            "bbox_frozen": False,
            "position_source": POSITION_SOURCE,
            "detection_mode": "qpos_category_bbox",
            "bbox_source": BBOX_SOURCE,
            "camera_frame_id": (
                self.camera_info.header.frame_id
                if self.camera_info is not None
                else self.camera_frame_id
            ),
            "image_width": int(self.camera_info.width) if self.camera_info else 0,
            "image_height": int(self.camera_info.height) if self.camera_info else 0,
            "world_base_position": self.world_base_position,
            "world_base_quat_wxyz": self.world_base_quat_wxyz,
            "initial_world_base_position": self.initial_world_base_position,
            "initial_world_base_quat_wxyz": self.initial_world_base_quat_wxyz,
            "base_translation_drift": [
                float(
                    self.world_base_position[index]
                    - self.initial_world_base_position[index]
                )
                for index in range(3)
            ],
            "truth_reference": (
                "current_mujoco_qpos_base_link"
                if self.truth_is_dynamic
                else "startup_qpos_base_link"
            ),
            "instance_naming_method": self.instance_naming_method,
            "requested_instance_naming_method": (
                self.requested_instance_naming_method
            ),
            "motion_to_ground_truth": {
                item["name"]: item["ground_truth_name"]
                for item in self.objects
            },
            "motion_object_order": list(OBJECT_NAMES),
            "objects": objects,
        }
        return String(data=json.dumps(payload, ensure_ascii=True, sort_keys=True))

    def _make_detection2d_array(self, stamp, objects):
        message = Detection2DArray()
        message.header.stamp = stamp
        message.header.frame_id = (
            self.camera_info.header.frame_id
            if self.camera_info is not None
            else self.camera_frame_id
        )
        for obj in objects:
            if not obj.get("bbox_projected", False):
                continue
            detection = Detection2D()
            detection.header = message.header
            if hasattr(detection, "id"):
                detection.id = obj["name"]
            detection.bbox = BoundingBox2D()
            detection.bbox.center.x = float(obj["bbox_center"][0])
            detection.bbox.center.y = float(obj["bbox_center"][1])
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = float(obj["bbox_size"][0])
            detection.bbox.size_y = float(obj["bbox_size"][1])

            hypothesis = ObjectHypothesisWithPose()
            if hasattr(hypothesis, "id"):
                hypothesis.id = int(obj["class_id"])
                hypothesis.score = float(obj["confidence"])
            else:
                hypothesis.hypothesis.id = int(obj["class_id"])
                hypothesis.hypothesis.score = float(obj["confidence"])
            detection.results.append(hypothesis)
            message.detections.append(detection)
        return message

    def _make_pose_array(self, stamp, objects):
        message = PoseArray()
        message.header.stamp = stamp
        message.header.frame_id = FRAME_ID
        for item in objects:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = item["position"]
            orientation = item["orientation"]
            pose.orientation.x = orientation[0]
            pose.orientation.y = orientation[1]
            pose.orientation.z = orientation[2]
            pose.orientation.w = orientation[3]
            message.poses.append(pose)
        return message

    def spin(self):
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            objects = self._objects_with_qpos_bboxes()
            self.json_pub.publish(self._make_json(stamp, objects))
            self.poses_pub.publish(self._make_pose_array(stamp, objects))
            self.detections2d_pub.publish(
                self._make_detection2d_array(stamp, objects)
            )
            rate.sleep()


def main():
    rospy.init_node("task2_vision")
    try:
        Task2VisionNode().spin()
    except rospy.ROSInterruptException:
        return
    except Exception as exc:
        if rospy.is_shutdown():
            return
        rospy.logfatal("task2_vision: startup failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
