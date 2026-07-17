#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect task1/task2 RGB, qpos-projected YOLO boxes, and metadata by seed."""

import collections
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.abspath(__file__)
GREEN = "\033[32m"
RESET = "\033[0m"

# =============================================================================
# 全局采集参数
# =============================================================================

# task1=4 个 parcel 单类别；task2=6 个部件、A/B/C 三类别。
DATA_TASK = "task2"

# 两个任务共用同一 seed 范围和相机视角采样逻辑。
SEED_START = 1
SEED_END = 1000

# 每个 seed 采三张图；pitch 在范围内均匀随机且由 seed 决定。
PITCH_VIEWS_PER_SEED = 3
PITCH_MIN_DEG = 8.0
PITCH_MAX_DEG = 24.0
HEAD_YAW_DEG = 0.0
HEAD_SETTLE_TIME = 0.8
HEAD_STABLE_DURATION = 0.2
HEAD_STABLE_SPEED_DEG_S = 1.0
HEAD_STABLE_TIMEOUT = 5.0

TASK_CONFIGS = {
    "task1": {
        "scene": "scene1",
        "dataset_directory": "scene1_parcel_yolo",
        "objects_topic": "/task1_vision/objects_json",
        "vision_package": "task1_vision",
        "vision_launch": "task1_vision_scene1.launch",
        "object_names": ("parcel_1", "parcel_2", "parcel_3", "parcel_4"),
        "class_ids": {"parcel": 0},
    },
    "task2": {
        "scene": "scene2",
        "dataset_directory": "scene2_parts_yolo",
        "objects_topic": "/task2_vision/objects_json",
        "vision_package": "task2_vision",
        "vision_launch": "task2_vision_scene2.launch",
        "object_names": (
            "part_type_a_1",
            "part_type_a_2",
            "part_type_b_1",
            "part_type_b_2",
            "part_type_c_1",
            "part_type_c_2",
        ),
        "class_ids": {
            "part_type_a": 0,
            "part_type_b": 1,
            "part_type_c": 2,
        },
    },
}
if DATA_TASK not in TASK_CONFIGS:
    raise ValueError("DATA_TASK must be task1 or task2")
TASK_CONFIG = TASK_CONFIGS[DATA_TASK]
SCENE_NAME = str(TASK_CONFIG["scene"])
OBJECT_NAMES = tuple(TASK_CONFIG["object_names"])
CLASS_ID_BY_NAME = dict(TASK_CONFIG["class_ids"])
DATASET_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "..",
        "data",
        TASK_CONFIG["dataset_directory"],
    )
)

# seed 能被该值整除时写入 val，其余写入 train。
VALIDATION_SEED_MODULO = 10
MAX_ATTEMPTS_PER_SEED = 2
WORKER_TIMEOUT_SEC = 240.0
# 数据采集不需要比赛倒计时进程；场景完整性校验和随机初始化仍保留。
DISABLE_MATCH_TIMER = True
CAPTURE_TIMEOUT_SEC = 20.0
CAPTURE_ATTEMPTS_PER_VIEW = 3
IMAGE_SYNC_TOLERANCE_SEC = 0.08

OBJECTS_TOPIC = str(TASK_CONFIG["objects_topic"])
IMAGE_TOPIC = "/cam_h/color/image_raw/compressed"
WORKER_RESULT_PREFIX = "__CHALLENGE_DATA_RESULT__ "
WORKER_SEED_ENV = "CHALLENGE_DATA_WORKER_SEED"
WORKER_ROOT_ENV = "CHALLENGE_DATA_WORKER_ROOT"

# 每轮采集只允许存在一套以下进程。既用于启动前清除上次异常退出的
# 残留，也用于当前 seed 完成后的兜底回收。
RESIDUAL_PROCESS_MARKERS = (
    SCRIPT_PATH,
    "task1_vision_scene1.launch",
    "task2_vision_scene2.launch",
    "load_kuavo_mujoco_challenge.launch",
    "challenge_secret.run_match_timer",
    "challenge_secret.run_cheat_monitor",
    "__name:=task1_vision",
    "__name:=task2_vision",
    "__name:=nodelet_manager",
    "__name:=humanoid_sqp_mpc",
    "__name:=arms_ik_node",
)
RESIDUAL_ROS_NODE_PREFIXES = (
    "/task1_vision",
    "/task2_vision",
    "/challenge_data_collector",
    "/nodelet_",
    "/ekf_se",
    "/arms_ik_node",
    "/kuavo_monitor_node",
    "/system_info_publisher",
    "/sim_leju_claw_interface",
    "/gripper_tf_publisher",
    "/sim_camera_info_publisher",
    "/camera_base_to_head_camera_view",
    "/left_wrist_camera_link_to_left_wrist_camera_view",
    "/right_wrist_camera_link_to_right_wrist_camera_view",
    "/lidar_mid360",
    "/humanoid_",
    "/collect_launch_info",
)

_STOP_REQUESTED = threading.Event()


def _handle_stop_signal(signum, _frame):
    print("[WARN] data: received signal {}; stopping".format(signum), flush=True)
    _STOP_REQUESTED.set()


def _install_signal_handlers():
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)


def _process_rows():
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pgid=,stat=,args="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 4:
            continue
        try:
            pid, ppid, pgid = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        rows.append(
            (pid, ppid, pgid, parts[3], parts[4] if len(parts) > 4 else "")
        )
    return rows


def _descendant_process_groups(root_pid):
    rows = _process_rows()
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _pgid, _stat, _args in rows:
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    groups = {
        pgid
        for pid, _ppid, pgid, _stat, _args in rows
        if pid in descendants and pgid > 1 and pgid != os.getpgrp()
    }
    try:
        root_group = os.getpgid(root_pid)
        if root_group > 1 and root_group != os.getpgrp():
            groups.add(root_group)
    except Exception:
        pass
    return groups


def _residual_process_groups():
    groups = set()
    current_pid = os.getpid()
    current_group = os.getpgrp()
    for pid, _ppid, pgid, stat, args in _process_rows():
        if (
            pid == current_pid
            or pgid <= 1
            or pgid == current_group
            or stat.startswith("Z")
        ):
            continue
        if any(marker in args for marker in RESIDUAL_PROCESS_MARKERS):
            groups.add(pgid)
    return groups


def _live_groups(groups):
    requested = set(groups)
    return {
        pgid
        for _pid, _ppid, pgid, stat, _args in _process_rows()
        if pgid in requested and not stat.startswith("Z")
    }


def _signal_groups(groups, sig):
    for pgid in sorted(set(groups)):
        if pgid <= 1 or pgid == os.getpgrp():
            continue
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except Exception as exc:
            print(
                "[WARN] data: failed to signal process group {}: {}".format(
                    pgid, exc
                ),
                flush=True,
            )


def _wait_groups(groups, timeout):
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if not _live_groups(groups):
            return True
        time.sleep(0.2)
    return not _live_groups(groups)


def _terminate_groups(groups, term_wait=5.0):
    groups = {
        pgid for pgid in groups if pgid > 1 and pgid != os.getpgrp()
    }
    if not groups:
        return
    _signal_groups(groups, signal.SIGTERM)
    if _wait_groups(groups, term_wait):
        return
    remaining = _live_groups(groups)
    _signal_groups(remaining, signal.SIGKILL)
    if not _wait_groups(remaining, 2.0):
        print(
            "[WARN] data: process groups still alive after SIGKILL: {}".format(
                sorted(_live_groups(remaining))
            ),
            flush=True,
        )


def _terminate_process_group(
    proc, name="process", wait_sigint=20.0, wait_sigterm=5.0
):
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return
    for sig, timeout, label in (
        (signal.SIGINT, wait_sigint, "SIGINT"),
        (signal.SIGTERM, wait_sigterm, "SIGTERM"),
        (signal.SIGKILL, 2.0, "SIGKILL"),
    ):
        if proc.poll() is not None:
            return
        try:
            os.killpg(pgid, sig)
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            print(
                "[WARN] data: {} did not exit after {}".format(name, label),
                flush=True,
            )
        except Exception as exc:
            print(
                "[WARN] data: failed to stop {} with {}: {}".format(
                    name, label, exc
                ),
                flush=True,
            )


def _cleanup_ros_graph():
    try:
        listed = subprocess.run(
            ["rosnode", "list"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        nodes = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        targets = [
            node
            for node in nodes
            if any(node.startswith(prefix) for prefix in RESIDUAL_ROS_NODE_PREFIXES)
        ]
        if targets:
            subprocess.run(
                ["rosnode", "kill"] + targets,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=12,
                check=False,
            )
    except Exception:
        pass


def _cleanup_residuals(reason):
    _cleanup_ros_graph()
    groups = _residual_process_groups()
    if groups:
        print(
            "[INFO] data: cleanup {} residual process groups {}".format(
                reason, sorted(groups)
            ),
            flush=True,
        )
        _terminate_groups(groups)
    _cleanup_ros_graph()


def _pitch_commands(seed):
    commands = []
    for view_index in range(PITCH_VIEWS_PER_SEED):
        rng = random.Random((int(seed) + 1) * 1000003 + view_index * 9176)
        commands.append(rng.uniform(PITCH_MIN_DEG, PITCH_MAX_DEG))
    return commands


def _split_for_seed(seed):
    modulo = int(VALIDATION_SEED_MODULO)
    if modulo > 0 and int(seed) % modulo == 0:
        return "val"
    return "train"


def _relative(root, path):
    return os.path.relpath(path, root).replace(os.sep, "/")


def _atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp.{}".format(os.getpid())
    with open(temporary, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path, text):
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_json(path, payload):
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _append_json_line(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _prepare_dataset(root):
    for split in ("train", "val"):
        for kind in ("images", "labels", "metadata"):
            os.makedirs(os.path.join(root, kind, split), exist_ok=True)
    os.makedirs(os.path.join(root, "seeds"), exist_ok=True)
    ordered_classes = sorted(CLASS_ID_BY_NAME.items(), key=lambda item: item[1])
    expected_ids = list(range(len(ordered_classes)))
    if [class_id for _, class_id in ordered_classes] != expected_ids:
        raise ValueError("YOLO class ids must be contiguous and start at zero")
    dataset_yaml = "\n".join(
        [
            "path: {}".format(root),
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        + ["  {}: {}".format(class_id, name) for name, class_id in ordered_classes]
        + [""]
    )
    _atomic_write_text(os.path.join(root, "data.yaml"), dataset_yaml)
    _atomic_write_text(
        os.path.join(root, "classes.txt"),
        "".join("{}\n".format(name) for name, _ in ordered_classes),
    )


def _seed_marker_path(root, seed):
    return os.path.join(root, "seeds", "seed_{:05d}.json".format(int(seed)))


def _load_seed_marker(root, seed):
    path = _seed_marker_path(root, seed)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            marker = json.load(fh)
    except Exception:
        return None
    if int(marker.get("seed", -1)) != int(seed):
        return None
    marker_task = marker.get("task")
    if marker_task is not None and str(marker_task) != DATA_TASK:
        return None
    for sample in marker.get("samples", []):
        for key in ("image", "label", "metadata"):
            relative_path = sample.get(key)
            if not relative_path or not os.path.isfile(
                os.path.join(root, relative_path)
            ):
                return None
    if len(marker.get("samples", [])) != PITCH_VIEWS_PER_SEED:
        return None
    return marker


def _find_first_incomplete_seed(root, seed_start, seed_end):
    for seed in range(int(seed_start), int(seed_end) + 1):
        if _load_seed_marker(root, seed) is None:
            return seed
    return int(seed_end) + 1


def _rebuild_manifest(root, seed_start, seed_end_exclusive):
    samples = []
    for seed in range(int(seed_start), int(seed_end_exclusive)):
        marker = _load_seed_marker(root, seed)
        if marker is None:
            raise RuntimeError(
                "cannot rebuild manifest: seed {} in the complete prefix is invalid".format(
                    seed
                )
            )
        samples.extend(marker["samples"])
    text = "".join(
        json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
        for sample in samples
    )
    _atomic_write_text(os.path.join(root, "manifest.jsonl"), text)
    return len(samples)


def _image_extension(image_msg):
    image_format = str(getattr(image_msg, "format", "")).lower()
    if "jpeg" in image_format or "jpg" in image_format:
        return ".jpg"
    if "png" in image_format:
        return ".png"
    raise RuntimeError("unsupported compressed RGB format: {}".format(image_format))


def _float_vector(value, length=3):
    try:
        result = [float(item) for item in value[:length]]
    except Exception:
        return None
    if len(result) != length or not all(math.isfinite(item) for item in result):
        return None
    return result


def _object_base_position(obj):
    for key in ("truth_position", "position"):
        position = _float_vector(obj.get(key, []))
        if position is not None:
            return position
    return None


def _dataset_objects(payload):
    by_name = {}
    for obj in payload.get("objects", []):
        name = str(obj.get("name", ""))
        class_name = str(obj.get("class_name", ""))
        if name not in OBJECT_NAMES or class_name not in CLASS_ID_BY_NAME:
            continue
        if not bool(obj.get("bbox_projected", False)):
            continue
        truth_position = _object_base_position(obj)
        bbox_center = _float_vector(obj.get("bbox_center", []), length=2)
        bbox_size = _float_vector(obj.get("bbox_size", []), length=2)
        if truth_position is None or bbox_center is None or bbox_size is None:
            continue
        if bbox_size[0] <= 1.0 or bbox_size[1] <= 1.0:
            continue
        by_name[name] = obj
    if any(name not in by_name for name in OBJECT_NAMES):
        return None
    return [by_name[name] for name in OBJECT_NAMES]


class SynchronizedCapture:
    def __init__(self):
        import rospy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        self._condition = threading.Condition()
        self._images = collections.deque(maxlen=90)
        self._payloads = collections.deque(maxlen=90)
        self._image_sub = rospy.Subscriber(
            IMAGE_TOPIC, CompressedImage, self._image_callback, queue_size=10
        )
        self._objects_sub = rospy.Subscriber(
            OBJECTS_TOPIC, String, self._objects_callback, queue_size=10
        )

    def close(self):
        image_sub, self._image_sub = self._image_sub, None
        objects_sub, self._objects_sub = self._objects_sub, None
        for subscriber in (image_sub, objects_sub):
            if subscriber is None:
                continue
            try:
                subscriber.unregister()
            except Exception:
                pass
        with self._condition:
            self._images.clear()
            self._payloads.clear()
            self._condition.notify_all()

    def _image_callback(self, msg):
        stamp = float(msg.header.stamp.to_sec())
        with self._condition:
            self._images.append((stamp, msg))
            self._condition.notify_all()

    def _objects_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        with self._condition:
            self._payloads.append(payload)
            self._condition.notify_all()

    def capture_after(self, not_before, timeout):
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while time.monotonic() < deadline:
                match = self._find_match(float(not_before))
                if match is not None:
                    return match
                self._condition.wait(timeout=0.1)
        raise RuntimeError(
            "no synchronized projected boxes and RGB image after {:.3f}".format(
                not_before
            )
        )

    def _find_match(self, not_before):
        for payload in reversed(self._payloads):
            try:
                detection_stamp = float(payload.get("stamp"))
            except (TypeError, ValueError):
                continue
            try:
                color_stamp = float(payload.get("color_stamp", detection_stamp))
            except (TypeError, ValueError):
                color_stamp = detection_stamp
            if detection_stamp < not_before or color_stamp < not_before:
                continue
            if _dataset_objects(payload) is None:
                continue
            if not self._images:
                continue
            image_stamp, image_msg = min(
                self._images, key=lambda item: abs(item[0] - color_stamp)
            )
            if abs(image_stamp - color_stamp) <= IMAGE_SYNC_TOLERANCE_SEC:
                return payload, image_msg
        return None


def _normalized_box(obj, image_width, image_height):
    center = _float_vector(obj.get("bbox_center", []), length=2)
    size = _float_vector(obj.get("bbox_size", []), length=2)
    if center is None or size is None:
        raise RuntimeError("invalid bbox for {}".format(obj.get("name", "object")))
    x1 = max(0.0, center[0] - 0.5 * size[0])
    y1 = max(0.0, center[1] - 0.5 * size[1])
    x2 = min(float(image_width), center[0] + 0.5 * size[0])
    y2 = min(float(image_height), center[1] + 0.5 * size[1])
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        raise RuntimeError("projected bbox is outside the RGB image")
    normalized = [
        ((x1 + x2) * 0.5) / float(image_width),
        ((y1 + y2) * 0.5) / float(image_height),
        (x2 - x1) / float(image_width),
        (y2 - y1) / float(image_height),
    ]
    return [max(0.0, min(1.0, value)) for value in normalized]


def _save_sample(
    root, seed, view_index, pitch_command, actual_head, payload, image_msg
):
    split = _split_for_seed(seed)
    image_width = int(payload.get("image_width", 0))
    image_height = int(payload.get("image_height", 0))
    if image_width <= 0 or image_height <= 0:
        raise RuntimeError(
            "{} vision payload has invalid image dimensions".format(DATA_TASK)
        )
    objects = _dataset_objects(payload)
    if objects is None:
        raise RuntimeError(
            "capture does not contain all {} qpos-projected {} objects".format(
                len(OBJECT_NAMES), DATA_TASK
            )
        )

    stem = "seed_{:05d}_view_{}".format(int(seed), int(view_index))
    extension = _image_extension(image_msg)
    image_path = os.path.join(root, "images", split, stem + extension)
    label_path = os.path.join(root, "labels", split, stem + ".txt")
    metadata_path = os.path.join(root, "metadata", split, stem + ".json")

    yolo_lines = []
    object_metadata = []
    for label_index, obj in enumerate(objects):
        class_name = str(obj.get("class_name", ""))
        class_id = int(CLASS_ID_BY_NAME[class_name])
        yolo_box = _normalized_box(obj, image_width, image_height)
        yolo_lines.append(
            "{} {}\n".format(
                class_id, " ".join("{:.8f}".format(value) for value in yolo_box)
            )
        )
        object_metadata.append(
            {
                "label_index": label_index,
                "source_name": str(obj.get("name", "")),
                "ground_truth_name": str(
                    obj.get("ground_truth_name", obj.get("name", ""))
                ),
                "class_id": class_id,
                "class_name": class_name,
                "bbox_xywh_pixels": [
                    float(value)
                    for value in obj.get("bbox_center", [])[:2]
                    + obj.get("bbox_size", [])[:2]
                ],
                "bbox_yolo_normalized": yolo_box,
                "position_source": str(
                    obj.get(
                        "truth_position_source", obj.get("position_source", "")
                    )
                ),
            }
        )

    metadata = {
        "task": DATA_TASK,
        "scene": SCENE_NAME,
        "seed": int(seed),
        "view_index": int(view_index),
        "split": split,
        "pitch_range_deg": [float(PITCH_MIN_DEG), float(PITCH_MAX_DEG)],
        "pitch_command_deg": float(pitch_command),
        "head_actual_yaw_deg": actual_head[0] if actual_head else None,
        "head_actual_pitch_deg": actual_head[1] if actual_head else None,
        "image_stamp": float(image_msg.header.stamp.to_sec()),
        "detection_stamp": float(payload.get("stamp")),
        "image_width": image_width,
        "image_height": image_height,
        "frame_id": str(payload.get("frame_id", "")),
        "camera_frame_id": str(payload.get("camera_frame_id", "")),
        "bbox_source": str(payload.get("bbox_source", "qpos_projected")),
        "image": _relative(root, image_path),
        "label": _relative(root, label_path),
        "metadata": _relative(root, metadata_path),
        "objects": object_metadata,
    }

    _atomic_write_bytes(image_path, bytes(image_msg.data))
    _atomic_write_text(label_path, "".join(yolo_lines))
    _atomic_write_json(metadata_path, metadata)
    return metadata


def _wait_head_stable():
    import rospy
    from kuavo_msgs.msg import sensorsData

    stable_since = None
    start_time = rospy.get_time()
    last_speed = float("inf")
    while not rospy.is_shutdown():
        if rospy.get_time() - start_time >= HEAD_STABLE_TIMEOUT:
            break
        try:
            msg = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=1.0)
            joint_q = list(msg.joint_data.joint_q)
            joint_v = list(msg.joint_data.joint_v)
            if len(joint_q) < 2 or len(joint_v) < 2:
                continue
        except Exception:
            continue
        last_speed = max(
            abs(math.degrees(float(joint_v[-2]))),
            abs(math.degrees(float(joint_v[-1]))),
        )
        now = rospy.get_time()
        if last_speed <= HEAD_STABLE_SPEED_DEG_S:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= HEAD_STABLE_DURATION:
                return [
                    math.degrees(float(joint_q[-2])),
                    math.degrees(float(joint_q[-1])),
                ]
        else:
            stable_since = None
    raise RuntimeError(
        "head did not remain below {:.3f} deg/s for {:.2f}s within {:.1f}s; "
        "last max speed={:.3f} deg/s".format(
            HEAD_STABLE_SPEED_DEG_S,
            HEAD_STABLE_DURATION,
            HEAD_STABLE_TIMEOUT,
            last_speed,
        )
    )


def _start_data_vision(challenge_task, seed):
    command = [
        "roslaunch",
        str(TASK_CONFIG["vision_package"]),
        str(TASK_CONFIG["vision_launch"]),
        "start_rqt:=false",
        "vision_mode:=0",
    ]
    if DATA_TASK == "task2":
        command.extend(
            [
                "seed:={}".format(int(seed)),
                "start_depth_xy:=false",
                "instance_naming_method:=relative_position",
            ]
        )
    return [
        challenge_task.ManagedProcess(
            "{}_data_vision".format(DATA_TASK), command
        ).start()
    ]


def _collect_seed(seed, root):
    import challenge_task
    import rospy

    if DATA_TASK == "task2":
        from task2_vision.head_control import publish_head_camera_pose
    else:
        from task1_vision.head_control import publish_head_camera_pose

    launcher_class = challenge_task._load_launcher()
    launcher = launcher_class(
        scene=SCENE_NAME,
        seed=int(seed),
        match_time_limit=0.0,
        timer_gui=False,
    )
    services = []
    capture = None
    samples = []
    try:
        launcher.start(
            node_name="challenge_data_collector_{}_seed_{}".format(
                DATA_TASK, int(seed)
            ),
            timeout=120,
        )
        services = _start_data_vision(challenge_task, seed)
        capture = SynchronizedCapture()

        commands = _pitch_commands(seed)
        for view_index, pitch in enumerate(commands):
            last_error = None
            for attempt in range(1, CAPTURE_ATTEMPTS_PER_VIEW + 1):
                publish_head_camera_pose(
                    yaw_deg=HEAD_YAW_DEG,
                    pitch_deg=pitch,
                    settle_time=HEAD_SETTLE_TIME,
                    wait_for_subscriber=True,
                    strict=True,
                )
                actual_head = _wait_head_stable()
                not_before = rospy.Time.now().to_sec()
                try:
                    payload, image_msg = capture.capture_after(
                        not_before, CAPTURE_TIMEOUT_SEC
                    )
                    sample = _save_sample(
                        root,
                        seed,
                        view_index,
                        pitch,
                        actual_head,
                        payload,
                        image_msg,
                    )
                    samples.append(sample)
                    rospy.loginfo(
                        "data: seed=%d view=%d pitch=%.3f saved %s",
                        seed,
                        view_index,
                        pitch,
                        sample["image"],
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    rospy.logwarn(
                        "data: seed=%d view=%d capture attempt %d/%d failed: %s",
                        seed,
                        view_index,
                        attempt,
                        CAPTURE_ATTEMPTS_PER_VIEW,
                        exc,
                    )
            else:
                raise RuntimeError(
                    "seed {} view {} capture failed: {}".format(
                        seed, view_index, last_error
                    )
                )

        marker = {
            "task": DATA_TASK,
            "scene": SCENE_NAME,
            "seed": int(seed),
            "pitch_commands_deg": commands,
            "samples": samples,
        }
        _atomic_write_json(_seed_marker_path(root, seed), marker)
        return marker
    finally:
        if capture is not None:
            try:
                capture.close()
            except Exception as exc:
                print("[WARN] data: failed to close capture: {}".format(exc), flush=True)
        try:
            challenge_task._stop_scene_services(services)
        except Exception as exc:
            print(
                "[WARN] data: failed to stop scene services: {}".format(exc),
                flush=True,
            )
        try:
            launcher.stop()
        except Exception as exc:
            print("[WARN] data: failed to stop simulator: {}".format(exc), flush=True)


def _worker_main(seed, root):
    _install_signal_handlers()
    _prepare_dataset(root)
    try:
        marker = _collect_seed(seed, root)
        result = {"ok": True, "seed": int(seed), "samples": len(marker["samples"])}
        print(WORKER_RESULT_PREFIX + json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        result = {"ok": False, "seed": int(seed), "error": str(exc)}
        print(WORKER_RESULT_PREFIX + json.dumps(result), flush=True)
        return 1


def _run_seed_worker(seed, root):
    _cleanup_residuals("before data seed {}".format(seed))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env[WORKER_SEED_ENV] = str(int(seed))
    env[WORKER_ROOT_ENV] = os.path.abspath(root)
    if DISABLE_MATCH_TIMER:
        env["CHALLENGE_TIMER_DISABLE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", SCRIPT_PATH],
        # Keep stdin open without sending data.  The humanoid controller polls
        # stdin and treats /dev/null's permanent EOF as a keyboard command.
        stdin=subprocess.PIPE,
        env=env,
        preexec_fn=os.setsid,
    )
    tracked_groups = set()
    try:
        tracked_groups.add(os.getpgid(proc.pid))
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + WORKER_TIMEOUT_SEC
    next_group_scan = 0.0
    try:
        while proc.poll() is None:
            # Simulator, timer and the selected vision node create their own process
            # group. Remember them while the worker is still their parent so
            # they can also be reaped after the worker itself exits.
            now = time.monotonic()
            if now >= next_group_scan:
                tracked_groups.update(_descendant_process_groups(proc.pid))
                next_group_scan = now + 1.0
            if _STOP_REQUESTED.is_set():
                _terminate_process_group(
                    proc, "challenge_data_seed_{}".format(seed), 20.0, 5.0
                )
                return False, "interrupted"
            if now >= deadline:
                _terminate_process_group(
                    proc, "challenge_data_seed_{}".format(seed), 10.0, 5.0
                )
                return False, "worker timeout"
            time.sleep(0.25)
        return proc.returncode == 0, "exit={}".format(proc.returncode)
    finally:
        if proc.poll() is None:
            tracked_groups.update(_descendant_process_groups(proc.pid))
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
        if proc.poll() is None:
            _terminate_process_group(
                proc, "challenge_data_seed_{}".format(seed), 5.0, 3.0
            )
        live_tracked_groups = _live_groups(tracked_groups)
        if live_tracked_groups:
            print(
                "[INFO] data: seed {} cleanup tracked process groups {}".format(
                    seed, sorted(live_tracked_groups)
                ),
                flush=True,
            )
        _terminate_groups(tracked_groups)
        _cleanup_residuals("after data seed {}".format(seed))


def _batch_main():
    _install_signal_handlers()
    root = os.path.abspath(DATASET_ROOT)
    _prepare_dataset(root)
    resume_seed = _find_first_incomplete_seed(root, SEED_START, SEED_END)
    existing_samples = _rebuild_manifest(root, SEED_START, resume_seed)
    if resume_seed <= int(SEED_END):
        if resume_seed > int(SEED_START):
            resume_text = (
                "[INFO] data: 连续完整 seed={}..{}；第一个不完整 seed={}；"
                "从 seed {} 开始采集"
            ).format(SEED_START, resume_seed - 1, resume_seed, resume_seed)
        else:
            resume_text = (
                "[INFO] data: seed {} 是第一个不完整 seed；从 seed {} 开始采集"
            ).format(resume_seed, resume_seed)
    else:
        resume_text = ("[INFO] data: seed {}..{} 已全部完整，无需继续采集").format(
            SEED_START, SEED_END
        )
    print(GREEN + resume_text + RESET, flush=True)
    print(
        "[INFO] data: task={} scene={} dataset={} existing_samples={} seeds={}..{} views_per_seed={} pitch_range=[{:.1f}, {:.1f}] deg".format(
            DATA_TASK,
            SCENE_NAME,
            root,
            existing_samples,
            SEED_START,
            SEED_END,
            PITCH_VIEWS_PER_SEED,
            PITCH_MIN_DEG,
            PITCH_MAX_DEG,
        ),
        flush=True,
    )
    if resume_seed > int(SEED_END):
        return 0
    _cleanup_residuals("data batch startup")
    manifest_path = os.path.join(root, "manifest.jsonl")
    failures_path = os.path.join(root, "failures.jsonl")
    interrupted = False
    failure_count = 0
    try:
        for seed in range(int(resume_seed), int(SEED_END) + 1):
            if _STOP_REQUESTED.is_set():
                interrupted = True
                break

            success = False
            detail = "not started"
            for attempt in range(1, int(MAX_ATTEMPTS_PER_SEED) + 1):
                if _STOP_REQUESTED.is_set():
                    interrupted = True
                    break
                print(
                    "[INFO] data: seed {} attempt {}/{}".format(
                        seed, attempt, MAX_ATTEMPTS_PER_SEED
                    ),
                    flush=True,
                )
                success, detail = _run_seed_worker(seed, root)
                if success:
                    marker = _load_seed_marker(root, seed)
                    if marker is None:
                        success = False
                        detail = (
                            "worker exited successfully without complete seed marker"
                        )
                    else:
                        for sample in marker["samples"]:
                            _append_json_line(manifest_path, sample)
                        break
            if interrupted:
                break
            if not success:
                failure_count += 1
                failure = {"seed": int(seed), "error": detail}
                _append_json_line(failures_path, failure)
                print("[ERROR] data: seed {} failed: {}".format(seed, detail))
            else:
                print(
                    "[INFO] data: seed {} complete, {} images written".format(
                        seed, len(marker["samples"])
                    ),
                    flush=True,
                )
    finally:
        _cleanup_residuals("data batch shutdown")
    if interrupted:
        return 130
    return 1 if failure_count else 0


def main():
    worker_seed = os.environ.get(WORKER_SEED_ENV)
    if worker_seed is not None:
        worker_root = os.path.abspath(os.environ.get(WORKER_ROOT_ENV, DATASET_ROOT))
        return _worker_main(int(worker_seed), worker_root)
    return _batch_main()


if __name__ == "__main__":
    raise SystemExit(main())
