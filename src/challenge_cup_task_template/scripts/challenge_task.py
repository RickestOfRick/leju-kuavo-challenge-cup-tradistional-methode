#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import math
import os
import sys
import threading
import time

import numpy as np

# ── Import IK/grasp module from scene2 pipeline ──────────────────────
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_GRASP_IK_DIR = os.path.join(
    _FILE_DIR, "..", "..", "challenge_cup_simulator", "test", "collect_scene2_dataset"
)
if os.path.isdir(_GRASP_IK_DIR):
    sys.path.insert(0, _GRASP_IK_DIR)

try:
    from scene2_part_grasp_ik import (
        GraspRuntime,
        execute_part_grasp,
        get_object_arm,
        get_object_lift_quat_xyzw,
        measure_hand_pose,
        move_arm_ik_once,
        rad_to_deg,
    )
    # grasp_offset: 零件中心 → 夹爪抓取点偏移（2-5cm）
    from scene2_part_grasp_ik import (
        _grasp_offset_local_for_side as _grasp_offset_local,
        _object_local_offset_to_world as _local_to_world,
        get_object_yaw_z_rad,
    )
    _HAS_GRASP_IK = True

    def _apply_grasp_offset_to_estimated(part_world, object_name, active_arm):
        """将夹爪抓取点偏移（grasp_offset）施加到估计的零件位置上。"""
        object_y = part_world[1]  # 用估计 Y 选 offset 区间
        offset_local = _grasp_offset_local(object_name, active_arm)
        try:
            offset_world = _local_to_world(object_name, offset_local)
        except Exception:
            offset_world = list(offset_local)  # 回退：直接加局部偏移
        return [
            part_world[0] + offset_world[0],
            part_world[1] + offset_world[1],
            part_world[2] + offset_world[2],
        ]
except ImportError:
    _HAS_GRASP_IK = False

# ── YOLO ONNX model path (OpenCV DNN, no ultralytics dependency) ──────
_YOLO_ONNX_PATH = os.path.join(_FILE_DIR, "..", "weights", "yolov8n_augmented.onnx")


# ══════════════════════════════════════════════════════════════════════
# Scene config
# ══════════════════════════════════════════════════════════════════════

SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
    },
}


# ══════════════════════════════════════════════════════════════════════
# Scene2 constants (mirrored from scene2_data_collection_pipeline.py)
# ══════════════════════════════════════════════════════════════════════

ARM_JOINT_NAMES = ["arm_joint_" + str(i) for i in range(1, 15)]

SIDE_LIFT_JOINTS_DEG = [
    20, 70, 0, -35, 0, 0, 0,
    20, -70, 0, -35, 0, 0, 0,
]

WORK_POSE_JOINTS_DEG = [
    30, 10, 10, -120, -60, 0, 0,
    30, -10, -10, -120, 60, 0, 0,
]

HOME_JOINTS_DEG = [
    20, 0, 0, -30, 0, 0, 0,
    20, 0, 0, -30, 0, 0, 0,
]

PLACE_ACTIVE_ARM_JOINTS_DEG = {
    "right": {
        "sorting_bin_a": [-30, -10, -10, -80, 70, 0, 0],
        "sorting_bin_b": [-30, 0, 30, -80, 70, 0, 0],
    },
    "left": {
        "sorting_bin_b": [-30, 0, -30, -80, -70, 0, 0],
        "sorting_bin_c": [-30, 10, 10, -80, -70, 0, 0],
    },
}

WORLD_TO_EE_OFFSET_X = 0.566
WORLD_TO_EE_OFFSET_Y_RIGHT = -0.014
WORLD_TO_EE_OFFSET_Y_LEFT = 0.014
WORLD_TO_EE_OFFSET_Z = -0.923783
PRE_GRASP_Z_OFFSET = 0.1
GRASP_POSITION_TOLERANCE = 0.06  # 从 0.10 降到 0.06，过滤假成功
ORIENTATION_TOLERANCE_RAD = math.radians(50.0)  # 从 60° 降到 50°

ARM_MODE_EXTERNAL_CONTROL = 2
ARM_MODE_AUTO_SWING = 1
ARM_MODE_SERVICE = "/arm_traj_change_mode"
ARM_TARGET_POSES_TOPIC = "/kuavo_arm_target_poses"
ARM_TRAJ_TOPIC = "/kuavo_arm_traj"
ARM_TRAJ_HZ = 100.0
ARM_MOVE_TIME = 1.4
ARM_SETTLE_TIME = 0.15
GRIPPER_CLOSE_TIME = 0.6
GRIPPER_COMMAND_HZ = 100.0
RIGHT_GRIPPER_OPEN = 0.0
LEFT_GRIPPER_OPEN = 0.0
RIGHT_GRIPPER_CLOSE = 255.0
LEFT_GRIPPER_CLOSE = 255.0

TOPIC_TIMEOUT = 20.0
IK_MODE_THREE_POINT_MIXED = 0x06
LIFT_Z_OFFSET = 0.30
PLACE_DWELL = 0.4
FAST_GRASP_SETTLE_HOLD = 0.8

HEAD_TARGET = [0.0, 20.0]
HEAD_SETTLE_TIME = 0.4

# Fixed sort order and bin mapping (YOLO class_id → action)
SORT_ORDER = [
    (1, "right", "sorting_bin_b"),   # part_b_1: B middle → middle bin
    (1, "right", "sorting_bin_b"),   # part_b_2
    (0, "right", "sorting_bin_a"),   # part_a_1: A right → right bin
    (0, "right", "sorting_bin_a"),   # part_a_2
    (2, "left", "sorting_bin_c"),    # part_c_1: C left → left bin
    (2, "left", "sorting_bin_c"),    # part_c_2
]

CLASS_TO_PART = {0: "part_type_a", 1: "part_type_b", 2: "part_type_c"}




# ══════════════════════════════════════════════════════════════════════
# Infrastructure helpers (replicate pipeline internals for portability)
# ══════════════════════════════════════════════════════════════════════

def _wait_for_connection(pub, timeout):
    import rospy
    start = time.time()
    while pub.get_num_connections() == 0 and time.time() - start < timeout and not rospy.is_shutdown():
        rospy.sleep(0.2)
    if pub.get_num_connections() == 0:
        raise RuntimeError(f"topic {pub.name} has no subscriber")


def _set_arm_mode(mode, timeout):
    import rospy
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
    rospy.wait_for_service(ARM_MODE_SERVICE, timeout=timeout)
    proxy = rospy.ServiceProxy(ARM_MODE_SERVICE, changeArmCtrlMode)
    req = changeArmCtrlModeRequest()
    req.control_mode = mode
    resp = proxy(req)
    if not resp.result:
        raise RuntimeError(f"{ARM_MODE_SERVICE} rejected mode {mode}: {resp.message}")
    rospy.loginfo("scene2: arm mode -> %s: %s", mode, resp.message)


def _publish_head_target(timeout):
    import rospy
    from kuavo_msgs.msg import robotHeadMotionData
    pub = rospy.Publisher("/robot_head_motion_data", robotHeadMotionData, queue_size=10)
    _wait_for_connection(pub, timeout)
    msg = robotHeadMotionData()
    msg.joint_data = list(HEAD_TARGET)
    for _ in range(5):
        if rospy.is_shutdown():
            break
        pub.publish(msg)
        rospy.sleep(0.1)
    rospy.sleep(HEAD_SETTLE_TIME)


def _read_current_arm_joints(timeout):
    import rospy
    from kuavo_msgs.msg import sensorsData
    msg = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=timeout)
    joint_q = list(msg.joint_data.joint_q)
    if len(joint_q) >= 27:
        return joint_q[13:27]
    if len(joint_q) >= 26:
        return joint_q[12:26]
    raise RuntimeError(f"/sensors_data_raw joint_q has {len(joint_q)} values")


class GripperCommandHold:
    def __init__(self, pub, hz=GRIPPER_COMMAND_HZ):
        self._pub = pub
        self._hz = float(hz)
        self._left_cmd = float(LEFT_GRIPPER_OPEN)
        self._right_cmd = float(RIGHT_GRIPPER_OPEN)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gripper_cmd_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_open(self):
        self._set_command(LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_OPEN)

    def set_right_closed(self):
        self._set_command(LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_CLOSE)

    def set_left_closed(self):
        self._set_command(LEFT_GRIPPER_CLOSE, RIGHT_GRIPPER_OPEN)

    def _set_command(self, left_cmd, right_cmd):
        with self._lock:
            self._left_cmd = float(left_cmd)
            self._right_cmd = float(right_cmd)

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState
        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                lc, rc = self._left_cmd, self._right_cmd
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ["left_gripper_joint", "right_gripper_joint"]
            msg.position = [lc, rc]
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break


class ArmTrajHold:
    def __init__(self, pub, degrees_list, hz=ARM_TRAJ_HZ):
        self._pub = pub
        self._hz = float(hz)
        self._degrees = self._validate_degrees(degrees_list)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="arm_traj_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_degrees(self, degrees_list):
        with self._lock:
            self._degrees = self._validate_degrees(degrees_list)

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState
        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                degrees = list(self._degrees)
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ARM_JOINT_NAMES
            msg.position = degrees
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break

    @staticmethod
    def _validate_degrees(degrees_list):
        degrees = [float(v) for v in degrees_list]
        if len(degrees) != len(ARM_JOINT_NAMES):
            raise ValueError(
                f"arm traj command has {len(degrees)} joints, expected {len(ARM_JOINT_NAMES)}"
            )
        return degrees


def _start_gripper_hold(timeout):
    import rospy
    from sensor_msgs.msg import JointState
    pub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
    _wait_for_connection(pub, timeout)
    hold = GripperCommandHold(pub)
    hold.start()
    return hold


def _start_arm_traj_hold(timeout):
    import rospy
    from sensor_msgs.msg import JointState
    pub = rospy.Publisher(ARM_TRAJ_TOPIC, JointState, queue_size=10)
    _wait_for_connection(pub, timeout)
    initial_degrees = rad_to_deg(_read_current_arm_joints(timeout))
    hold = ArmTrajHold(pub, initial_degrees)
    hold.start()
    return hold


def _publish_gripper_open(gripper_hold):
    gripper_hold.set_open()


def _publish_arm_gripper_close(gripper_hold, arm):
    if arm == "left":
        gripper_hold.set_left_closed()
    elif arm == "right":
        gripper_hold.set_right_closed()
    else:
        raise ValueError(f"unknown arm: {arm}")


def _publish_arm_traj_interpolation(arm_hold, start_degrees, target_degrees, duration):
    import rospy
    start = [float(v) for v in start_degrees]
    target = [float(v) for v in target_degrees]
    if len(start) != len(target):
        raise ValueError(f"arm traj length mismatch: {len(start)} != {len(target)}")
    steps = max(1, int(round(float(duration) * ARM_TRAJ_HZ)))
    rate = rospy.Rate(ARM_TRAJ_HZ)
    for step in range(steps + 1):
        if rospy.is_shutdown():
            break
        alpha = float(step) / float(steps)
        point = [start[i] + (target[i] - start[i]) * alpha for i in range(len(target))]
        arm_hold.set_degrees(point)
        if step < steps:
            rate.sleep()


def _execute_arm_motion(arm_hold, start_degrees, target_degrees, move_time, settle):
    import rospy
    _publish_arm_traj_interpolation(arm_hold, start_degrees, target_degrees, move_time)
    rospy.sleep(settle)


def _locked_motion_start_degrees(active_arm=None, locked_other_arm_joints=None):
    start = list(_read_current_arm_joints(TOPIC_TIMEOUT))
    if locked_other_arm_joints is not None:
        locked = [float(v) for v in locked_other_arm_joints]
        if active_arm == "left":
            start[7:14] = locked
        elif active_arm == "right":
            start[:7] = locked
        else:
            raise ValueError(f"unknown arm: {active_arm}")
    return rad_to_deg(start)


def _move_arm_to(arm_hold, degrees_list, move_time=ARM_MOVE_TIME, settle=ARM_SETTLE_TIME,
                 active_arm=None, locked_other_arm_joints=None):
    start = _locked_motion_start_degrees(active_arm, locked_other_arm_joints)
    _execute_arm_motion(arm_hold, start, degrees_list, move_time, settle)


def _move_to_work_pose_joints(arm_hold, active_arm=None, locked_other_arm_joints=None):
    import rospy
    rospy.loginfo("scene2: return to work pose (joint angles)")
    _move_arm_to(arm_hold, WORK_POSE_JOINTS_DEG, move_time=ARM_MOVE_TIME,
                 settle=FAST_GRASP_SETTLE_HOLD, active_arm=active_arm,
                 locked_other_arm_joints=locked_other_arm_joints)


def _enter_work_pose(arm_hold):
    import rospy
    rospy.loginfo("scene2: side lift -> work pose")
    _move_arm_to(arm_hold, SIDE_LIFT_JOINTS_DEG, move_time=ARM_MOVE_TIME,
                 settle=ARM_SETTLE_TIME)
    _move_to_work_pose_joints(arm_hold)


def _fixed_work_pose_other_arm_joints(active_arm):
    if active_arm == "left":
        other_deg = WORK_POSE_JOINTS_DEG[7:14]
    elif active_arm == "right":
        other_deg = WORK_POSE_JOINTS_DEG[:7]
    else:
        raise ValueError(f"unknown arm: {active_arm}")
    return [math.radians(float(v)) for v in other_deg]


def _place_active_arm_joints(active_arm, bin_name):
    try:
        joints = PLACE_ACTIVE_ARM_JOINTS_DEG[active_arm][bin_name]
    except KeyError:
        raise RuntimeError(f"no place pose for {active_arm} → {bin_name}")
    if len(joints) != 7:
        raise ValueError(f"place pose for {active_arm} → {bin_name} must have 7 joints")
    return [float(v) for v in joints]


def _compose_single_arm_place_joints(active_arm, active_joints_deg, locked_other_arm_joints):
    other_deg = rad_to_deg(locked_other_arm_joints)
    if active_arm == "left":
        return list(active_joints_deg) + other_deg
    if active_arm == "right":
        return other_deg + list(active_joints_deg)
    raise ValueError(f"unknown arm: {active_arm}")


# ══════════════════════════════════════════════════════════════════════
# Vision module (ONNX + OpenCV DNN — zero external Python deps)
# ══════════════════════════════════════════════════════════════════════

_ONNX_INPUT_SIZE = 640
_ONNX_CONF_THRESH = 0.25
_ONNX_IOU_THRESH = 0.45
_CLASS_NAMES = ["part_a", "part_b", "part_c"]


def _greedy_nms(boxes_xyxy, scores, iou_thresh):
    """Standard greedy NMS.  Returns list of kept indices."""
    if len(boxes_xyxy) == 0:
        return []
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thresh]
    return keep


class ONNXDetector:
    """YOLOv8n ONNX detector — OpenCV DNN backend, onnxruntime fallback."""

    def __init__(self, onnx_path):
        import cv2
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        self._onnx_path = onnx_path
        self._backend = None
        self._net = None      # cv2.dnn.Net
        self._session = None  # onnxruntime.InferenceSession

        # Try OpenCV DNN first (fast, zero extra deps)
        try:
            self._net = cv2.dnn.readNetFromONNX(onnx_path)
            try:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                pass
            self._backend = "opencv"
            return
        except Exception as e:
            import sys
            print(f"[scene2] OpenCV DNN failed: {e}", file=sys.stderr)

        # Fallback: try onnxruntime
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(onnx_path, providers=[
                "CUDAExecutionProvider", "CPUExecutionProvider"
            ])
            self._backend = "onnxruntime"
            return
        except ImportError:
            pass

        raise RuntimeError(
            "No ONNX backend available. "
            "OpenCV DNN failed (try OpenCV >= 4.5 or install 'onnxruntime').\n"
            "In Docker: pip install onnxruntime"
        )

    def detect(self, bgr_image):
        import cv2
        h0, w0 = bgr_image.shape[:2]
        r = min(_ONNX_INPUT_SIZE / h0, _ONNX_INPUT_SIZE / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        dw = (_ONNX_INPUT_SIZE - nw) // 2
        dh = (_ONNX_INPUT_SIZE - nh) // 2

        lb = cv2.resize(bgr_image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        lb = cv2.copyMakeBorder(
            lb, dh, _ONNX_INPUT_SIZE - nh - dh, dw, _ONNX_INPUT_SIZE - nw - dw,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        if self._backend == "opencv":
            blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (_ONNX_INPUT_SIZE, _ONNX_INPUT_SIZE),
                                         swapRB=True, crop=False)
            self._net.setInput(blob)
            output = self._net.forward()
        elif self._backend == "onnxruntime":
            blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (_ONNX_INPUT_SIZE, _ONNX_INPUT_SIZE),
                                         swapRB=True, crop=False)
            ort_inputs = {self._session.get_inputs()[0].name: blob}
            output = self._session.run(None, ort_inputs)[0]
        else:
            raise RuntimeError("detector not initialized")

        # Both backends produce (1, 7, 8400); normalize to (8400, 7)
        preds = output[0].T  # (8400, 7): [cx, cy, w, h, cls0, cls1, cls2]

        scores = preds[:, 4:].max(axis=1)
        class_ids = preds[:, 4:].argmax(axis=1)
        mask = scores > _ONNX_CONF_THRESH
        if mask.sum() == 0:
            return []

        boxes_raw = preds[mask, :4]
        scores_f = scores[mask]
        class_ids_f = class_ids[mask]

        boxes_scaled = np.zeros((len(boxes_raw), 4), dtype=np.float32)
        boxes_scaled[:, 0] = (boxes_raw[:, 0] - boxes_raw[:, 2] / 2 - dw) / r
        boxes_scaled[:, 1] = (boxes_raw[:, 1] - boxes_raw[:, 3] / 2 - dh) / r
        boxes_scaled[:, 2] = (boxes_raw[:, 0] + boxes_raw[:, 2] / 2 - dw) / r
        boxes_scaled[:, 3] = (boxes_raw[:, 1] + boxes_raw[:, 3] / 2 - dh) / r
        boxes_scaled[:, 0] = np.clip(boxes_scaled[:, 0], 0, w0)
        boxes_scaled[:, 1] = np.clip(boxes_scaled[:, 1], 0, h0)
        boxes_scaled[:, 2] = np.clip(boxes_scaled[:, 2], 0, w0)
        boxes_scaled[:, 3] = np.clip(boxes_scaled[:, 3], 0, h0)

        kept = []
        for cls_id in range(3):
            c_mask = class_ids_f == cls_id
            if c_mask.sum() == 0:
                continue
            c_boxes = boxes_scaled[c_mask]
            c_scores = scores_f[c_mask]
            indices = _greedy_nms(c_boxes, c_scores, _ONNX_IOU_THRESH)
            for i in indices:
                x1, y1, x2, y2 = c_boxes[i]
                kept.append((int(cls_id), float((x1 + x2) / 2), float((y1 + y2) / 2),
                             float(c_scores[i])))
        return kept


def _get_camera_intrinsics(timeout=10.0):
    import rospy
    from sensor_msgs.msg import CameraInfo
    msg = rospy.wait_for_message("/cam_h/color/camera_info", CameraInfo, timeout=timeout)
    return np.array(msg.K).reshape(3, 3)


def _get_head_rgb(timeout=10.0):
    import rospy, cv2
    from sensor_msgs.msg import CompressedImage
    msg = rospy.wait_for_message(
        "/cam_h/color/image_raw/compressed", CompressedImage, timeout=timeout
    )
    bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    return bgr, msg.header


def _get_head_depth(timeout=10.0):
    import rospy, cv2
    from sensor_msgs.msg import CompressedImage
    msg = rospy.wait_for_message(
        "/cam_h/depth/image_raw/compressedDepth", CompressedImage, timeout=timeout
    )
    buf = np.frombuffer(msg.data, np.uint8)

    # Strip leading garbage: find PNG signature \x89PNG\r\n\x1a\n
    png_sig = bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    sig_pos = bytes(buf[:64].tobytes()).find(png_sig)
    if sig_pos > 0:
        rospy.logdebug("depth: stripping %d leading bytes before PNG sig", sig_pos)
        buf = buf[sig_pos:]

    # Method 1: cv_bridge
    try:
        from cv_bridge import CvBridge
        # Re-pack message with cleaned buffer
        clean_msg = CompressedImage()
        clean_msg.header = msg.header
        clean_msg.format = msg.format
        clean_msg.data = buf.tobytes()
        depth = CvBridge().compressed_imgmsg_to_cv2(clean_msg, desired_encoding="passthrough")
        if depth is not None and depth.size > 0:
            return _depth_to_meters(depth)
    except Exception as e:
        rospy.logdebug("depth cv_bridge: %s", e)

    # Method 2: OpenCV
    for flag in (cv2.IMREAD_UNCHANGED, cv2.IMREAD_ANYDEPTH):
        depth = cv2.imdecode(buf, flag)
        if depth is not None and depth.size > 0:
            return _depth_to_meters(depth)

    # Method 3: PIL
    try:
        from PIL import Image
        import io
        depth = np.array(Image.open(io.BytesIO(buf.tobytes())))
        if depth is not None and depth.size > 0:
            return _depth_to_meters(depth)
    except Exception:
        pass

    raise RuntimeError(
        f"failed to decode depth image ({len(buf)} bytes, format='{getattr(msg, 'format', '?')}')"
    )


def _depth_to_meters(depth):
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * 0.001
    if depth.dtype == np.float32:
        return depth
    if depth.ndim == 3:
        return depth[:, :, 0].astype(np.float32) * 0.001
    return depth.astype(np.float32) * 0.001


def _pixel_to_world(cx, cy, depth_img, K, tf_buffer, timeout=5.0):
    import rospy, tf2_ros

    px, py = int(round(cx)), int(round(cy))
    H, W = depth_img.shape[:2]

    # ── 5×5 neighborhood depth sampling (median for robustness) ──────
    r = 2  # radius → 5×5 window
    y0, y1 = max(0, py - r), min(H, py + r + 1)
    x0, x1 = max(0, px - r), min(W, px + r + 1)
    patch = depth_img[y0:y1, x0:x1]
    valid = patch[(patch > 0.0) & (patch <= 3.0)]
    if len(valid) == 0:
        raise RuntimeError(f"no valid depth in {r*2+1}×{r*2+1} window at ({px},{py})")
    Z_center = float(depth_img[min(py, H-1), min(px, W-1)])
    Z_median = float(np.median(valid))
    Z_min = float(valid.min())
    Z_max = float(valid.max())
    Z_range = Z_max - Z_min
    Z = Z_median  # use median as canonical depth

    rospy.loginfo(
        "scene2: depth @ (%d,%d) window=%d×%d valid=%d/%d "
        "center=%.4f median=%.4f min=%.4f max=%.4f range=%.4f",
        px, py, y1-y0, x1-x0, len(valid), patch.size,
        Z_center, Z_median, Z_min, Z_max, Z_range)
    if Z_range > 0.10:
        rospy.logwarn("scene2: depth variance >10cm in 5×5 window — edge or noise?")
    if Z <= 0.0 or Z > 3.0:
        raise RuntimeError(f"invalid median depth at ({px},{py}): Z={Z:.3f}")
    if Z < 0.1:
        rospy.logwarn("scene2: suspiciously low depth Z=%.3f — edge or noise?", Z)

    # ── Camera-frame 3D ──────────────────────────────────────────────
    X_cam = (cx - K[0, 2]) * Z / K[0, 0]
    Y_cam = (cy - K[1, 2]) * Z / K[1, 1]
    P_cam = np.array([X_cam, Y_cam, Z])
    rospy.loginfo("scene2: pixel=(%.0f,%.0f) Z_canon=%.4f P_cam=[%.4f,%.4f,%.4f] "
                  "K_fx=%.1f K_cx=%.1f K_fy=%.1f K_cy=%.1f",
                  cx, cy, Z, X_cam, Y_cam, Z, K[0,0], K[0,2], K[1,1], K[1,2])

    # ── Camera → base_link TF ────────────────────────────────────────
    # The first 3 names are deliberately wrong (cam_h_* vs camera_* in URDF)
    # so they always fall through to "Head Camera View" — the static TF frame
    # calibrated to match the MuJoCo camera's actual render pose.
    # The URDF frames (camera_color_optical_frame etc.) point HORIZONTALLY
    # forward, but the MuJoCo camera looks DOWN at the table.  Using the URDF
    # optical frame would rotate the depth Z axis by ~90°, producing wrong 3D.
    _CAM_FRAMES = [
        "cam_h_color_optical_frame",
        "cam_h_color_frame",
        "cam_h_link",
        "Head Camera View",
    ]
    from scipy.spatial.transform import Rotation

    for cam_frame in _CAM_FRAMES:
        try:
            tf_s = tf_buffer.lookup_transform("base_link", cam_frame, rospy.Time(0))
        except Exception:
            continue

        t = tf_s.transform.translation
        r = tf_s.transform.rotation
        R_mat = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = [t.x, t.y, t.z]

        P_base = (T @ np.append(P_cam, 1.0))[:3]
        rospy.loginfo("scene2: TF base_link→%s  T_xyz=[%.4f,%.4f,%.4f] "
                      "P_cam=[%.4f,%.4f,%.4f] → P_base=[%.4f,%.4f,%.4f]",
                      cam_frame, t.x, t.y, t.z,
                      P_cam[0], P_cam[1], P_cam[2],
                      P_base[0], P_base[1], P_base[2])

        # Dump TF tree only on first call (per process)
        if not hasattr(_pixel_to_world, "_tf_dumped"):
            _pixel_to_world._tf_dumped = True
            try:
                all_frames = tf_buffer.all_frames_as_string()
                rospy.loginfo("scene2: TF tree (first lookup):\n%s", all_frames)
            except Exception:
                pass
        return list(P_base)

    raise RuntimeError(f"no camera TF found for base_link→camera. Tried: {_CAM_FRAMES}")


def _draw_detections(bgr_image, detections):
    """Draw bounding boxes and labels on a copy of the image."""
    import cv2
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # G B R for part_a/b/c
    vis = bgr_image.copy()
    for cls_id, cx, cy, conf in detections:
        color = colors[cls_id % 3]
        label = f"{_CLASS_NAMES[cls_id]} {conf:.2f}"
        radius = 15
        cv2.circle(vis, (int(cx), int(cy)), radius, color, 2)
        cv2.putText(vis, label, (int(cx) + radius + 2, int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


class _VizWorker:
    """Continuous detection visualization thread.

    Captures head camera frames, runs YOLO detection, draws results,
    and publishes to /scene2/yolo_detections at ~10 Hz.
    First frame is saved to disk only; all subsequent frames are streamed.
    """

    def __init__(self, detector, viz_pub, viz_dir):
        self._detector = detector
        self._viz_pub = viz_pub
        self._viz_dir = viz_dir
        self._stop_event = threading.Event()
        self._thread = None
        self._frame_count = 0
        self._first_published = False
        self._error_count = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="viz_worker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self):
        import rospy, cv2
        from cv_bridge import CvBridge
        bridge = CvBridge()
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            try:
                img, header = _get_head_rgb(timeout=5.0)
                all_dets = self._detector.detect(img)
                vis = _draw_detections(img, all_dets)

                self._frame_count += 1

                # Always save first frame to disk
                if self._frame_count == 1:
                    fname = os.path.join(self._viz_dir, "detect_first.jpg")
                    cv2.imwrite(fname, vis)
                    rospy.loginfo("scene2 viz: first frame saved to %s", fname)

                # Save every 5th frame to disk
                if self._frame_count % 5 == 0:
                    fname = os.path.join(self._viz_dir, f"detect_{self._frame_count:04d}.jpg")
                    cv2.imwrite(fname, vis)

                # Skip publishing the first frame
                if not self._first_published:
                    self._first_published = True
                    continue

                msg = bridge.cv2_to_imgmsg(vis, encoding="bgr8")
                msg.header = header
                self._viz_pub.publish(msg)

            except Exception as e:
                self._error_count += 1
                if self._error_count <= 1 or self._error_count % 50 == 0:
                    rospy.logwarn("scene2 viz: %s", e)

            rospy.sleep(0.1)


# ══════════════════════════════════════════════════════════════════════
# Scene2 main pipeline
# ══════════════════════════════════════════════════════════════════════

def _run_scene2_sorting():
    import rospy, cv2

    if not _HAS_GRASP_IK:
        rospy.logerr("scene2: cannot import scene2_part_grasp_ik — abort")
        return False

    # ── 1. Init YOLO ONNX detector ────────────────────────────────────
    onnx_path = os.path.abspath(_YOLO_ONNX_PATH)
    rospy.loginfo("scene2: loading ONNX detector from %s", onnx_path)
    detector = ONNXDetector(onnx_path)

    # ── 2. Init visualization ─────────────────────────────────────────
    from sensor_msgs.msg import Image as RosImage
    viz_pub = rospy.Publisher("/scene2/yolo_detections", RosImage, queue_size=5, latch=True)
    viz_dir = os.path.join(_FILE_DIR, "..", "viz_output")
    os.makedirs(viz_dir, exist_ok=True)
    rospy.loginfo("scene2: viz publishing to /scene2/yolo_detections (subscribers=%d)",
                  viz_pub.get_num_connections())

    # ── 3. Init TF + camera ──────────────────────────────────────────
    import tf2_ros
    K = _get_camera_intrinsics(timeout=TOPIC_TIMEOUT)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)

    # ── 4. Arm / gripper init ────────────────────────────────────────
    _set_arm_mode(ARM_MODE_EXTERNAL_CONTROL, timeout=TOPIC_TIMEOUT)
    _publish_head_target(TOPIC_TIMEOUT)
    gripper_hold = _start_gripper_hold(TOPIC_TIMEOUT)
    arm_hold = _start_arm_traj_hold(TOPIC_TIMEOUT)

    # ── 5. GraspRuntime ──────────────────────────────────────────────
    grasp_runtime = GraspRuntime(
        world_to_ee_offset_x=WORLD_TO_EE_OFFSET_X,
        world_to_ee_offset_y_left=WORLD_TO_EE_OFFSET_Y_LEFT,
        world_to_ee_offset_y_right=WORLD_TO_EE_OFFSET_Y_RIGHT,
        world_to_ee_offset_z=WORLD_TO_EE_OFFSET_Z,
        pre_grasp_z_offset=PRE_GRASP_Z_OFFSET,
        grasp_position_tolerance=GRASP_POSITION_TOLERANCE,
        orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
        gripper_close_time=GRIPPER_CLOSE_TIME,
        timeout=TOPIC_TIMEOUT,
        move_time=ARM_MOVE_TIME,
        settle_time=ARM_SETTLE_TIME,
        ik_mode_pos_hard_ori_hard=IK_MODE_THREE_POINT_MIXED,
        read_current_arm_joints_cb=lambda: _read_current_arm_joints(TOPIC_TIMEOUT),
        execute_arm_motion_cb=lambda sd, td, mt, st: _execute_arm_motion(arm_hold, sd, td, mt, st),
        publish_arm_gripper_close_cb=lambda a: _publish_arm_gripper_close(gripper_hold, a),
        sleep_cb=rospy.sleep,
        loginfo_cb=rospy.loginfo,
        logwarn_cb=rospy.logwarn,
    )

    # ── 6. Enter work pose ───────────────────────────────────────────
    _publish_gripper_open(gripper_hold)
    _enter_work_pose(arm_hold)

    # ── 7. Sorting loop ──────────────────────────────────────────────
    viz_worker = _VizWorker(detector, viz_pub, viz_dir)
    viz_worker.start()
    rospy.loginfo("scene2: viz worker started")

    picked_class_count = {0: 0, 1: 0, 2: 0}

    for step_index, (target_cls, target_arm, target_bin) in enumerate(SORT_ORDER):
        rospy.loginfo("scene2: step %d/6 — picking class %d (%s) arm=%s bin=%s",
                      step_index + 1, target_cls, CLASS_TO_PART[target_cls],
                      target_arm, target_bin)

        # 7a. Vision detection with retries — capture RGB+depth from same frame
        detections = []
        target_dets = []
        matched_depth_img = None
        for retry in range(15):
            if rospy.is_shutdown():
                return False
            img, _ = _get_head_rgb(timeout=5.0)
            depth_img = _get_head_depth(timeout=2.0)  # same simulation frame
            all_dets = detector.detect(img)

            target_dets = [d for d in all_dets if d[0] == target_cls]
            if target_dets:
                detections = target_dets
                matched_depth_img = depth_img
                break
            rospy.logwarn("scene2: no class-%d detections (retry %d/15) raw=%d",
                          target_cls, retry + 1, len(all_dets))
            rospy.sleep(0.3)

        if not detections:
            rospy.logerr("scene2: failed to detect class %d after retries — skipping", target_cls)
            continue

        # 7b. Pick highest-confidence detection
        best = max(detections, key=lambda d: d[3])
        cls_id, cx, cy, conf = best
        rospy.loginfo("scene2: detected %s at (%.0f, %.0f) conf=%.2f",
                      _CLASS_NAMES[cls_id], cx, cy, conf)

        # 7c. 2D → 3D directly in base_link frame via camera TF
        try:
            part_in_base = _pixel_to_world(cx, cy, matched_depth_img, K, tf_buffer)
        except RuntimeError as e:
            rospy.logerr("scene2: pixel_to_world failed: %s", e)
            continue
        rospy.loginfo("scene2: part_in_base = [%.4f, %.4f, %.4f]", *part_in_base)

        # Convert part_in_base (base_link) → part_world (world frame needed by pipeline).
        try:
            world_to_base = tf_buffer.lookup_transform("world", "base_link", rospy.Time(0))
            base_world = [
                world_to_base.transform.translation.x,
                world_to_base.transform.translation.y,
                world_to_base.transform.translation.z,
            ]
        except Exception:
            base_world = [-0.65, 0.0, 0.9263857]
        part_world = [
            float(part_in_base[0]) + base_world[0],
            float(part_in_base[1]) + base_world[1],
            float(part_in_base[2]) + base_world[2],
        ]
        rospy.loginfo("scene2: base_world=[%.4f,%.4f,%.4f] part_world=[%.4f,%.4f,%.4f]",
                      *base_world, *part_world)
        # ── 3-axis correction (calibrated via run_debug_2d3d.py, 3 seeds) ──
        # dX=-0.063: estimate is 6.3cm behind GT → push forward  +0.063
        # dY=-0.003: negligible → no correction
        # dZ=+0.042: estimate is 4.2cm above GT → push down     -0.042
        part_world[0] += 0.063
        part_world[2] -= 0.042
        rospy.loginfo(
            "[3D_CORR] cls=%d pixel=(%.0f,%.0f) "
            "part_world=[%.4f,%.4f,%.4f]",
            cls_id, cx, cy, part_world[0], part_world[1], part_world[2])

        # 7d. Object name
        picked_class_count[target_cls] += 1
        object_name = f"{CLASS_TO_PART[target_cls]}_{picked_class_count[target_cls]}"

        # ── Apply grasp_offset (零件中心 → 夹爪抓取点，2-5cm) ──────
        part_world = _apply_grasp_offset_to_estimated(part_world, object_name, target_arm)
        rospy.loginfo(
            "[GRASP_OFFSET] %s arm=%s grasp_target=[%.4f,%.4f,%.4f]",
            object_name, target_arm, part_world[0], part_world[1], part_world[2])

        # 7e. Grasp via original pipeline (handles IK targets, pre-grasp, descend, close).
        locked_other = _fixed_work_pose_other_arm_joints(target_arm)
        _publish_gripper_open(gripper_hold)

        grasp_quat = get_object_lift_quat_xyzw(object_name, active_arm=target_arm)
        success = execute_part_grasp(
            runtime=grasp_runtime, object_name=object_name,
            world_xyz=part_world, active_arm=target_arm,
            locked_other_arm_joints=locked_other,
            grasp_quat_xyzw=grasp_quat,
        )
        rospy.loginfo("scene2: %s grasp %s", object_name, "OK" if success else "TOLERANCE MISS")

        # 7f. Lift from current EE position (not from computed target).
        ee_pos, _ = measure_hand_pose(grasp_runtime, target_arm)
        lift_target = [ee_pos[0], ee_pos[1], ee_pos[2] + LIFT_Z_OFFSET]
        lift_err, _, _, _, _ = move_arm_ik_once(
            runtime=grasp_runtime, active_arm=target_arm,
            active_pos=lift_target, locked_other_arm_joints=locked_other,
            active_quat=grasp_quat, label=f"{object_name}_lift",
            constraint_mode=IK_MODE_THREE_POINT_MIXED, pos_cost_weight=2.0,
            move_time=ARM_MOVE_TIME, settle_time=FAST_GRASP_SETTLE_HOLD,
        )
        rospy.loginfo("scene2: %s lift xyz_err=%.4f m", object_name, lift_err)

        # 7g. Place (joint-space)
        place_joints = _place_active_arm_joints(target_arm, target_bin)
        target_joints = _compose_single_arm_place_joints(target_arm, place_joints, locked_other)
        rospy.loginfo("scene2: %s %s-hand place → %s joints=%s",
                      object_name, target_arm, target_bin,
                      [round(v, 2) for v in place_joints])
        _move_arm_to(arm_hold, target_joints, move_time=ARM_MOVE_TIME,
                     settle=FAST_GRASP_SETTLE_HOLD, active_arm=target_arm,
                     locked_other_arm_joints=locked_other)
        _publish_gripper_open(gripper_hold)
        rospy.sleep(PLACE_DWELL)

        # 7h. Return to work pose
        _move_to_work_pose_joints(arm_hold, active_arm=target_arm,
                                  locked_other_arm_joints=locked_other)

    # ── 8. Retract to home ──────────────────────────────────────────
    viz_worker.stop()
    rospy.loginfo("scene2: viz worker stopped")
    rospy.loginfo("scene2: all 6 parts sorted, retracting to home")
    _move_to_work_pose_joints(arm_hold)
    _move_arm_to(arm_hold, SIDE_LIFT_JOINTS_DEG)
    _move_arm_to(arm_hold, HOME_JOINTS_DEG)

    # ── 9. Cleanup ──────────────────────────────────────────────────
    _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
    gripper_hold.stop()
    arm_hold.stop()
    rospy.loginfo("scene2: sorting complete")
    return True


# ══════════════════════════════════════════════════════════════════════
# Launcher & entry point
# ══════════════════════════════════════════════════════════════════════

def _load_launcher():
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState

    rospy.loginfo("=== %s任务启动 ===", config["title"])

    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)

    rospy.sleep(1.0)
    rospy.loginfo("场景实例已初始化。")

    # ── Scene routing ────────────────────────────────────────────────
    if scene == "scene1":
        rospy.loginfo("scene1: 包裹称重与摆放 — not yet implemented")
        rospy.spin()

    elif scene == "scene2":
        try:
            success = _run_scene2_sorting()
            if not success:
                rospy.logerr("scene2: vision sorting pipeline failed")
        except Exception as exc:
            rospy.logerr("scene2: unhandled exception: %s", exc)
            import traceback
            traceback.print_exc()

    elif scene == "scene3":
        rospy.loginfo("scene3: SMT 料盘出库 — not yet implemented")
        rospy.spin()

    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认按 scene 自动设置")
    parser.add_argument("--timeout", type=int, default=120,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
    )


if __name__ == "__main__":
    main()
