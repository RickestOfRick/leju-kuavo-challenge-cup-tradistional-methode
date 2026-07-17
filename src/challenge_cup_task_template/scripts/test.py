#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

# ========================= 用户配置 =========================

# 外层按列表顺序执行任务，前一个任务的全部 seed 结束后才进入下一个任务。
TASK_QUEUE = [
    {"task": "task2", "scene": "scene2", "seed_start": 1, "seed_end": 200},
    {"task": "task1", "scene": "scene1", "seed_start": 1, "seed_end": 100},
]

TIME_LIMIT_SEC = 360.0
LAUNCH_WAIT_SEC = 180.0
POST_SUCCESS_WAIT_SEC = 20.0
POLL_INTERVAL_SEC = 0.25
WORKER_WATCHDOG_MARGIN_SEC = 90.0
GEOMETRY_EPS = 1e-6

QPOS_TOPIC = "/mujoco/qpos"
SCENE_SUCCESS_RULES = {
    "scene1": {
        "object_to_box": {
            "parcel_1": "sorting_box_0p4_0p3_0p3",
            "parcel_2": "sorting_box_0p4_0p3_0p3",
            "parcel_3": "sorting_box_0p4_0p3_0p3",
            "parcel_4": "sorting_box_0p4_0p3_0p3",
        },
        "box_floor_geoms": {
            "sorting_box_0p4_0p3_0p3": "sorting_box_0p4_0p3_0p3_floor",
        },
        "relative_z_range": [0.0, 0.25],
    },
    "scene2": {
        "object_to_box": {
            "part_type_a_1": "sorting_bin_a",
            "part_type_a_2": "sorting_bin_a",
            "part_type_b_1": "sorting_bin_b",
            "part_type_b_2": "sorting_bin_b",
            "part_type_c_1": "sorting_bin_c",
            "part_type_c_2": "sorting_bin_c",
        },
        "box_floor_geoms": {
            "sorting_bin_a": "sorting_bin_a_floor",
            "sorting_bin_b": "sorting_bin_b_floor",
            "sorting_bin_c": "sorting_bin_c_floor",
        },
        "relative_z_range": [0.0, 0.30],
    },
}

# ======================= 用户配置结束 =======================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(SCRIPT_DIR, "result.txt")


def _load_config_order(package_name, attribute, fallback):
    config_path = os.path.abspath(
        os.path.join(
            SCRIPT_DIR,
            "..",
            "..",
            package_name,
            "scripts",
            "{}_config.py".format(package_name.replace("_motion", "")),
        )
    )
    if package_name == "task1_pick":
        config_path = os.path.join(os.path.dirname(config_path), "task1_config.py")
    elif package_name == "task2_motion":
        config_path = os.path.join(os.path.dirname(config_path), "task2_config.py")
    try:
        spec = importlib.util.spec_from_file_location(
            "challenge_test_{}_config".format(package_name), config_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = [str(value) for value in getattr(module, attribute)]
        if values:
            return values
    except Exception as exc:
        print(
            "[WARN] test: cannot load {} from {}: {}".format(
                attribute, config_path, exc
            ),
            flush=True,
        )
    return list(fallback)


TASK1_VISION_ERROR_ORDER = _load_config_order(
    "task1_pick",
    "PARCEL_NAMES",
    ["parcel_1", "parcel_2", "parcel_3", "parcel_4"],
)
TASK2_VISION_ERROR_ORDER = _load_config_order(
    "task2_motion",
    "GRASP_OFFSETS_BASE",
    [
        "part_type_a_1",
        "part_type_a_2",
        "part_type_b_1",
        "part_type_b_2",
        "part_type_c_1",
        "part_type_c_2",
    ],
)


def _result_object_order(task_name):
    if str(task_name) == "task1":
        return list(TASK1_VISION_ERROR_ORDER)
    if str(task_name) == "task2":
        return list(TASK2_VISION_ERROR_ORDER)
    return []


def _unknown_placement_detail(task_name):
    return "[{}]".format(",".join("n/a" for _ in _result_object_order(task_name)))


# 仅供 test.py 创建的内部 worker 使用；用户不需要修改。
WORKER_TASK = None
WORKER_SCENE = None
WORKER_SEED = None

WORKER_RESULT_PREFIX = "__CHALLENGE_TEST_RESULT__ "
_STOP_REQUESTED = threading.Event()
_FORCE_STOP_REQUESTED = threading.Event()
_VISION_ERROR_COLLECTOR = None

# 只匹配本测试会启动的任务、仿真、计时器和视觉进程。
RESIDUAL_PROCESS_MARKERS = (
    "challenge_task.run_scene(",
    "challenge_task.py --scene",
    "task1_vision_scene1.launch",
    "task2_vision_scene2.launch",
    "load_kuavo_mujoco_challenge.launch",
    "task1_node.py",
    "task2_node.py",
    "roslaunch task1_pick",
    "roslaunch task2_motion",
    "challenge_secret.run_match_timer",
    "challenge_secret.run_cheat_monitor",
)

# 主脚本导入 task1_node；直接 rosrun/roslaunch 时节点名为 task1。
RESIDUAL_ROS_NODE_PREFIXES = (
    "/challenge_task",
    "/task1_vision",
    "/task1",
    "/task2_vision",
    "/task2",
    "/challenge_batch_test",
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


def _handle_stop_signal(signum, _frame):
    if not _STOP_REQUESTED.is_set():
        print(
            "[WARN] test: received signal {}; starting fast cleanup, "
            "press Ctrl+C again to force stop".format(signum),
            flush=True,
        )
        _STOP_REQUESTED.set()
        return
    if not _FORCE_STOP_REQUESTED.is_set():
        print(
            "[WARN] test: second interrupt; force-killing remaining child groups",
            flush=True,
        )
        _FORCE_STOP_REQUESTED.set()


def _install_signal_handlers():
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)


def _is_keyboard_command_noise(text):
    return "[keyboard command]:" in str(text)


class ChildLog:
    def __init__(self, proc):
        self.proc = proc
        self.timer_ready_wall = None
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    # 转发主脚本输出，并捕获官方计时器真正启动的时刻。
    def _read(self):
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            text = line.rstrip()
            if _is_keyboard_command_noise(text):
                continue
            if text:
                print(text, flush=True)
            timer_ready = "challenge timer armed" in text or "比赛计时器已启动" in text
            if timer_ready and self.timer_ready_wall is None:
                self.timer_ready_wall = time.monotonic()

    def close(self):
        if self.proc.stdout is not None:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
        self._thread.join(timeout=1.0)


class WorkerLog:
    def __init__(self, proc):
        self.proc = proc
        self.result = None
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            text = line.rstrip()
            if _is_keyboard_command_noise(text):
                continue
            if text.startswith(WORKER_RESULT_PREFIX):
                try:
                    payload = json.loads(text[len(WORKER_RESULT_PREFIX) :])
                    self.result = (
                        float(payload["value"]),
                        str(payload.get("vision_error", "[]")),
                        str(payload["detail"]),
                    )
                except Exception as exc:
                    print("[WARN] test: invalid worker result: {}".format(exc))
                continue
            if text:
                print(text, flush=True)

    def join(self, timeout=2.0):
        self._thread.join(timeout=timeout)


class VisionErrorCollector:
    def __init__(self, task_name):
        import rospy
        from std_msgs.msg import String

        self.task_name = str(task_name)
        self._lock = threading.Lock()
        self._payloads = {}
        self._subscribers = []
        if self.task_name == "task2":
            self.order = list(TASK2_VISION_ERROR_ORDER)
            topics = (
                ("used", "/task2_depth_xy/objects_json"),
                ("raw", "/task2_vision/objects_json"),
            )
        elif self.task_name == "task1":
            self.order = list(TASK1_VISION_ERROR_ORDER)
            topics = (
                ("used", "/task1_vision/used_objects_json"),
                ("raw", "/task1_vision/objects_json"),
            )
        else:
            self.order = []
            topics = ()
        for source, topic in topics:
            self._subscribers.append(
                rospy.Subscriber(
                    topic,
                    String,
                    self._callback,
                    callback_args=source,
                    queue_size=1,
                )
            )

    def _callback(self, message, source):
        try:
            payload = json.loads(message.data)
        except Exception:
            return
        with self._lock:
            self._payloads[str(source)] = payload

    @staticmethod
    def _xy_error(item, full_truth=False):
        position = item.get("position", [])
        truth_position = item.get("truth_position", [])
        if len(position) >= 2 and len(truth_position) >= 2:
            try:
                dx = float(position[0]) - float(truth_position[0])
                dy = float(position[1]) - float(truth_position[1])
                return math.hypot(dx, dy)
            except (TypeError, ValueError):
                return None
        if full_truth and len(position) >= 2:
            return 0.0
        return None

    @staticmethod
    def _task2_pose_error(item, full_truth=False):
        position = item.get("position", [])
        if len(position) < 3:
            position = item.get("fit_position", [])
        truth_position = item.get("truth_position", [])
        if full_truth and len(position) >= 3:
            return (0.0, 0.0, 0.0, 0.0)
        if len(position) < 3 or len(truth_position) < 3:
            return None
        try:
            yaw = float(item.get("fit_yaw", item.get("yaw")))
            truth_yaw = float(item["truth_yaw"])
            yaw_error = (yaw - truth_yaw + math.pi) % (2.0 * math.pi) - math.pi
            return (
                float(position[0]) - float(truth_position[0]),
                float(position[1]) - float(truth_position[1]),
                float(position[2]) - float(truth_position[2]),
                yaw_error,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def errors(self):
        with self._lock:
            used_payload = dict(self._payloads.get("used", {}))
            raw_payload = dict(self._payloads.get("raw", {}))
        payload = used_payload if used_payload.get("objects") else raw_payload
        objects = {
            str(item.get("name", "")): item for item in payload.get("objects", [])
        }
        full_truth = (
            self.task_name == "task2" and int(payload.get("vision_mode", -1)) == 0
        )
        if self.task_name == "task2":
            return [
                self._task2_pose_error(objects.get(name, {}), full_truth=full_truth)
                for name in self.order
            ]
        return [self._xy_error(objects.get(name, {})) for name in self.order]

    def formatted(self):
        values = self.errors()
        if self.task_name == "task2":
            return "[{}]".format(
                ",".join(
                    (
                        "n/a"
                        if value is None
                        else "[{:+.1f},{:+.1f},{:+.1f},{:+.1f}]".format(
                            1000.0 * value[0],
                            1000.0 * value[1],
                            1000.0 * value[2],
                            math.degrees(value[3]),
                        )
                    )
                    for value in values
                )
            )
        return "[{}]".format(
            ",".join(
                "n/a" if value is None else "{:.1f}".format(1000.0 * value)
                for value in values
            )
        )

    def object_body_names(self):
        with self._lock:
            used_payload = dict(self._payloads.get("used", {}))
            raw_payload = dict(self._payloads.get("raw", {}))
        payload = used_payload if used_payload.get("objects") else raw_payload
        return {
            str(item.get("name", "")): str(
                item.get("ground_truth_name") or item.get("name", "")
            )
            for item in payload.get("objects", [])
            if item.get("name")
        }

    def close(self):
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass


def _latest_vision_error_text(task_name):
    collector = _VISION_ERROR_COLLECTOR
    if collector is not None and collector.task_name == str(task_name):
        return collector.formatted()
    order = (
        TASK2_VISION_ERROR_ORDER
        if str(task_name) == "task2"
        else TASK1_VISION_ERROR_ORDER if str(task_name) == "task1" else []
    )
    return "[{}]".format(",".join("n/a" for _ in order))


def _latest_object_body_names(task_name):
    collector = _VISION_ERROR_COLLECTOR
    if collector is not None and collector.task_name == str(task_name):
        return collector.object_body_names()
    return {}


def _process_rows():
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pgid=,args="],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid, ppid, pgid = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        rows.append((pid, ppid, pgid, parts[3] if len(parts) > 3 else ""))
    return rows


def _descendant_process_groups(root_pid):
    rows = _process_rows()
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _pgid, _args in rows:
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    groups = {
        pgid
        for pid, _ppid, pgid, _args in rows
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
    for pid, _ppid, pgid, args in _process_rows():
        if pid == current_pid or pgid <= 1 or pgid == current_group:
            continue
        if any(marker in args for marker in RESIDUAL_PROCESS_MARKERS):
            groups.add(pgid)
    return groups


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


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
                "[WARN] test: failed to signal process group {}: {}".format(pgid, exc)
            )


def _wait_groups(groups, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_group_exists(pgid) for pgid in groups):
            return True
        if _FORCE_STOP_REQUESTED.is_set():
            return False
        time.sleep(0.2)
    return not any(_group_exists(pgid) for pgid in groups)


def _terminate_groups(groups, term_wait=5.0):
    groups = {pgid for pgid in groups if pgid > 1 and pgid != os.getpgrp()}
    if not groups:
        return
    if _FORCE_STOP_REQUESTED.is_set():
        _signal_groups(groups, signal.SIGKILL)
        _wait_groups(groups, 0.5)
        return
    if _STOP_REQUESTED.is_set():
        term_wait = min(float(term_wait), 1.5)
    _signal_groups(groups, signal.SIGTERM)
    if _wait_groups(groups, term_wait):
        return
    _signal_groups(groups, signal.SIGKILL)
    _wait_groups(groups, 2.0)


def _terminate_process_group(proc, name="process", wait_sigint=25.0, wait_sigterm=8.0):
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
    if _FORCE_STOP_REQUESTED.is_set():
        stop_sequence = ((signal.SIGKILL, 0.5, "SIGKILL"),)
    elif _STOP_REQUESTED.is_set():
        stop_sequence = (
            (signal.SIGTERM, min(float(wait_sigterm), 2.0), "SIGTERM"),
            (signal.SIGKILL, 1.0, "SIGKILL"),
        )
    else:
        stop_sequence = (
            (signal.SIGINT, wait_sigint, "SIGINT"),
            (signal.SIGTERM, wait_sigterm, "SIGTERM"),
            (signal.SIGKILL, 2.0, "SIGKILL"),
        )
    for sig, timeout, label in stop_sequence:
        if proc.poll() is not None:
            return
        try:
            os.killpg(pgid, sig)
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            print("[WARN] test: {} did not exit after {}".format(name, label))
        except Exception as exc:
            print("[WARN] test: failed to stop {} with {}: {}".format(name, label, exc))


def _cleanup_ros_graph():
    # 中断时由进程组清理负责回收，避免 rosnode 命令累计等待几十秒。
    if _STOP_REQUESTED.is_set():
        return
    try:
        listed = subprocess.run(
            ["rosnode", "list"],
            text=True,
            encoding="utf-8",
            errors="replace",
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
        subprocess.run(
            ["rosnode", "cleanup"],
            input="y\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=6,
            check=False,
        )
    except Exception:
        pass


def _cleanup_residuals(reason):
    _cleanup_ros_graph()
    groups = _residual_process_groups()
    if groups:
        print(
            "[INFO] test: cleanup {} residual process groups {}".format(
                reason, sorted(groups)
            )
        )
        _terminate_groups(groups)
    _cleanup_ros_graph()


def _cleanup_seed_processes(proc, task_name, scene, seed, tracked_groups):
    if proc is not None and proc.poll() is None:
        tracked_groups.update(_descendant_process_groups(proc.pid))
    run_label = "{}_{}_seed_{}".format(task_name, scene, seed)
    _terminate_process_group(proc, "challenge_task_{}".format(run_label))
    if proc is not None and proc.stdin is not None:
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
    _cleanup_ros_graph()
    _terminate_groups(tracked_groups)
    _cleanup_residuals("after {}".format(run_label))


def _start_challenge_task(task_name, scene, seed):
    code = f"""
import sys
sys.path.insert(0, {SCRIPT_DIR!r})
import challenge_task
challenge_task.NEED_OPERATOR_ENTER = 0
task_name = {str(task_name)!r}
scene = {str(scene)!r}
seed = {int(seed)!r}
challenge_task.run_scene(
    scene=scene,
    seed=seed,
    node_name="challenge_task_{{}}_{{}}_seed_{{}}".format(task_name, scene, seed),
    timeout=120,
    time_limit={float(TIME_LIMIT_SEC)!r},
    timer_gui=False,
)
"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            code,
        ],
        # 保持 stdin 打开但不发送数据，避免底层键盘轮询把 EOF 当成命令。
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        preexec_fn=os.setsid,
    )


def _wait_for_ros_master(timeout, proc=None):
    import rosgraph

    master = rosgraph.Master("/challenge_batch_test_probe")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not _STOP_REQUESTED.is_set():
        if proc is not None and proc.poll() is not None:
            return False
        try:
            master.getPid()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _ensure_rospy(proc):
    if not _wait_for_ros_master(LAUNCH_WAIT_SEC, proc=proc):
        return False
    import rospy

    rospy.init_node("challenge_batch_test", anonymous=True, disable_signals=True)
    return True


def _read_sim_time(timeout=0.5):
    import rospy
    from kuavo_msgs.msg import sensorsData

    message = rospy.wait_for_message(
        "/sensors_data_raw", sensorsData, timeout=float(timeout)
    )
    return float(message.sensor_time.to_sec())


def _joint_qpos_size(element):
    if element.tag == "freejoint":
        return 7
    if element.tag != "joint":
        return 0
    joint_type = element.get("type", "hinge")
    if joint_type == "free":
        return 7
    if joint_type == "ball":
        return 4
    return 1


def _iter_xml_children_with_includes(path):
    path = os.path.abspath(path)
    root = ET.parse(path).getroot()
    base_dir = os.path.dirname(path)
    for child in list(root):
        if child.tag == "include":
            include_path = child.get("file")
            if include_path:
                if not os.path.isabs(include_path):
                    include_path = os.path.join(base_dir, include_path)
                yield from _iter_xml_children_with_includes(include_path)
            continue
        yield child


def _walk_body_description(body, qpos_addr, qpos_map, geoms):
    for child in list(body):
        size = _joint_qpos_size(child)
        if not size:
            continue
        name = child.get("name")
        if name:
            qpos_map[str(name)] = qpos_addr
        qpos_addr += size
    for child in list(body):
        if child.tag == "geom" and child.get("name"):
            geoms[str(child.get("name"))] = child
    for child in list(body):
        if child.tag == "body":
            qpos_addr = _walk_body_description(child, qpos_addr, qpos_map, geoms)
    return qpos_addr


def _load_xml_qpos_description(xml_path):
    qpos_addr = 0
    qpos_map = {}
    geoms = {}
    for child in _iter_xml_children_with_includes(xml_path):
        if child.tag != "worldbody":
            continue
        for body in list(child):
            if body.tag == "body":
                qpos_addr = _walk_body_description(body, qpos_addr, qpos_map, geoms)
    return qpos_addr, qpos_map, geoms


def _xml_vector(element, attribute, count, default):
    raw = str(element.get(attribute, "")).strip()
    values = [float(value) for value in raw.split()] if raw else list(default)
    if len(values) < count:
        raise ValueError(
            "{} {} requires {} values, got {}".format(
                element.tag, attribute, count, len(values)
            )
        )
    return values[:count]


def _rotation_matrix_wxyz(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= GEOMETRY_EPS:
        raise ValueError("freejoint quaternion has zero norm")
    w, x, y, z = [value / norm for value in (w, x, y, z)]
    return [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]


class QposBoxChecker:
    def __init__(self, task_name, scene):
        if scene not in SCENE_SUCCESS_RULES:
            raise ValueError("no success rule for {}".format(scene))
        self.task_name = str(task_name)
        self.scene = str(scene)
        self.rule = SCENE_SUCCESS_RULES[self.scene]
        self.object_order = _result_object_order(self.task_name)
        if not self.object_order:
            raise ValueError("no result object order for {}".format(task_name))
        self.model_path = None
        self.expected_nq = 0
        self.qpos_map = {}
        self.floor_geoms = {}
        self.qpos = []

    def _load_description(self):
        import rospy

        model_path = os.path.abspath(str(rospy.get_param("legged_robot_scene_param")))
        if self.qpos_map and self.model_path == model_path:
            return
        if not os.path.isfile(model_path):
            raise RuntimeError("MuJoCo scene XML not found: {}".format(model_path))
        expected_nq, qpos_map, geoms = _load_xml_qpos_description(model_path)
        required_joints = set(self.object_order)
        required_joints.update(self.rule["object_to_box"].values())
        missing_joints = sorted(required_joints - set(qpos_map))
        if missing_joints:
            raise RuntimeError(
                "scene XML missing freejoints: {}".format(", ".join(missing_joints))
            )
        floor_geoms = {}
        for floor_name in self.rule["box_floor_geoms"].values():
            if floor_name not in geoms:
                raise RuntimeError(
                    "scene XML missing floor geom: {}".format(floor_name)
                )
            floor = geoms[floor_name]
            floor_geoms[floor_name] = {
                "size": _xml_vector(floor, "size", 3, [0.0, 0.0, 0.0]),
                "pos": _xml_vector(floor, "pos", 3, [0.0, 0.0, 0.0]),
            }
        self.model_path = model_path
        self.expected_nq = expected_nq
        self.qpos_map = qpos_map
        self.floor_geoms = floor_geoms

    def _read_latest_qpos(self, timeout=0.5):
        import rospy
        from std_msgs.msg import Float64MultiArray

        self._load_description()
        message = rospy.wait_for_message(
            QPOS_TOPIC, Float64MultiArray, timeout=float(timeout)
        )
        qpos = list(message.data)
        if len(qpos) < self.expected_nq:
            raise RuntimeError(
                "qpos len {} < XML nq {}".format(len(qpos), self.expected_nq)
            )
        self.qpos = qpos

    def _freejoint_pose(self, name):
        address = self.qpos_map[str(name)]
        if address + 6 >= len(self.qpos):
            raise RuntimeError(
                "qpos address {} for {} outside length {}".format(
                    address, name, len(self.qpos)
                )
            )
        return (
            [float(value) for value in self.qpos[address : address + 3]],
            [float(value) for value in self.qpos[address + 3 : address + 7]],
        )

    def _object_in_box(self, object_name, box_name, object_body_names):
        body_name = object_body_names.get(object_name, object_name)
        object_pos, _object_quaternion = self._freejoint_pose(body_name)
        box_pos, box_quaternion = self._freejoint_pose(box_name)
        box_matrix = _rotation_matrix_wxyz(box_quaternion)
        delta = [object_pos[index] - box_pos[index] for index in range(3)]
        local_x = (
            box_matrix[0] * delta[0]
            + box_matrix[3] * delta[1]
            + box_matrix[6] * delta[2]
        )
        local_y = (
            box_matrix[1] * delta[0]
            + box_matrix[4] * delta[1]
            + box_matrix[7] * delta[2]
        )
        local_z = (
            box_matrix[2] * delta[0]
            + box_matrix[5] * delta[1]
            + box_matrix[8] * delta[2]
        )

        floor_geom_name = self.rule["box_floor_geoms"][box_name]
        floor_geom = self.floor_geoms[floor_geom_name]
        floor_size = floor_geom["size"]
        floor_pos = floor_geom["pos"]
        relative_x = local_x - floor_pos[0]
        relative_y = local_y - floor_pos[1]
        box_bottom_local_z = floor_pos[2] - floor_size[2]
        relative_z = local_z - box_bottom_local_z
        z_min, z_max = [float(value) for value in self.rule["relative_z_range"]]
        return (
            abs(relative_x) <= floor_size[0] + GEOMETRY_EPS
            and abs(relative_y) <= floor_size[1] + GEOMETRY_EPS
            and z_min - GEOMETRY_EPS <= relative_z <= z_max + GEOMETRY_EPS
        )

    def all_objects_in_expected_boxes(self, object_body_names=None):
        self._read_latest_qpos(timeout=0.5)
        object_body_names = object_body_names or {}
        object_to_box = self.rule["object_to_box"]
        placed = [
            self._object_in_box(
                object_name, object_to_box[object_name], object_body_names
            )
            for object_name in self.object_order
        ]
        detail = "[{}]".format(",".join("1" if value else "0" for value in placed))
        return all(placed), detail


def _run_one_seed(task_name, scene, seed, result_callback=None):
    global _VISION_ERROR_COLLECTOR

    if _VISION_ERROR_COLLECTOR is not None:
        _VISION_ERROR_COLLECTOR.close()
    _VISION_ERROR_COLLECTOR = None
    run_label = "{} {} seed {}".format(task_name, scene, seed)
    print("[INFO] test: start {}".format(run_label), flush=True)
    _cleanup_residuals("before {}".format(run_label))
    proc = None
    log = None
    tracked_groups = set()
    start_wall = time.monotonic()
    timer_start_sim = None
    last_sim_time = None
    last_sim_sample_wall = None
    checker = QposBoxChecker(task_name, scene)
    success_time = None
    latest_placement_detail = _unknown_placement_detail(task_name)
    result_emitted = False
    last_checker_error = None
    last_checker_warning_wall = 0.0

    def finished(value, detail):
        nonlocal result_emitted
        if result_callback is not None and not result_emitted:
            result_callback(value, detail)
            result_emitted = True
        return value, detail

    def failed(reason):
        print("[WARN] test: {} failed: {}".format(run_label, reason), flush=True)
        return finished(-1.0, latest_placement_detail)

    try:
        proc = _start_challenge_task(task_name, scene, seed)
        tracked_groups.update(_descendant_process_groups(proc.pid))
        log = ChildLog(proc)
        if not _ensure_rospy(proc):
            if proc.poll() is not None:
                return failed("challenge task exited before ROS master became ready")
            return failed("ros master unavailable")
        _VISION_ERROR_COLLECTOR = VisionErrorCollector(task_name)

        while not _STOP_REQUESTED.is_set():
            now_wall = time.monotonic()
            timer_ready_wall = log.timer_ready_wall
            if timer_ready_wall is None and now_wall - start_wall > LAUNCH_WAIT_SEC:
                return failed("launch timeout before timer")

            elapsed_sim = None
            if timer_ready_wall is not None:
                try:
                    last_sim_time = _read_sim_time(timeout=0.5)
                    last_sim_sample_wall = now_wall
                except Exception:
                    if timer_start_sim is None and now_wall - timer_ready_wall > 5.0:
                        return failed("simulation time unavailable after timer start")
                    if (
                        timer_start_sim is not None
                        and last_sim_sample_wall is not None
                        and now_wall - last_sim_sample_wall > 5.0
                    ):
                        return failed("simulation time stopped updating")
                if last_sim_time is not None:
                    if timer_start_sim is None:
                        timer_start_sim = last_sim_time
                        print(
                            "[INFO] test: simulation timer start={:.3f}s".format(
                                timer_start_sim
                            ),
                            flush=True,
                        )
                    elapsed_sim = max(0.0, last_sim_time - timer_start_sim)
                    if elapsed_sim > TIME_LIMIT_SEC:
                        return failed("simulation timeout")

            if proc.poll() is not None:
                try:
                    ok, latest_placement_detail = checker.all_objects_in_expected_boxes(
                        _latest_object_body_names(task_name)
                    )
                    if ok and elapsed_sim is not None:
                        return finished(elapsed_sim, latest_placement_detail)
                    return failed("process exited before success")
                except Exception as exc:
                    return failed(
                        "process exited before qpos success check: {}".format(exc)
                    )

            try:
                ok, latest_placement_detail = checker.all_objects_in_expected_boxes(
                    _latest_object_body_names(task_name)
                )
                last_checker_error = None
            except Exception as exc:
                error_text = str(exc)
                if (
                    error_text != last_checker_error
                    or now_wall - last_checker_warning_wall >= 5.0
                ):
                    print(
                        "[WARN] test: {} QPOS success check unavailable: {}".format(
                            run_label, error_text
                        ),
                        flush=True,
                    )
                    last_checker_error = error_text
                    last_checker_warning_wall = now_wall
                time.sleep(POLL_INTERVAL_SEC)
                continue
            if ok:
                if elapsed_sim is None:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                success_time = elapsed_sim
                finished(success_time, latest_placement_detail)
                break
            time.sleep(POLL_INTERVAL_SEC)

        if _STOP_REQUESTED.is_set():
            return failed("interrupted")

        wait_deadline = time.monotonic() + POST_SUCCESS_WAIT_SEC
        while proc.poll() is None and time.monotonic() < wait_deadline:
            if _STOP_REQUESTED.wait(0.25):
                break
        return finished(success_time, latest_placement_detail)
    finally:
        if _VISION_ERROR_COLLECTOR is not None:
            _VISION_ERROR_COLLECTOR.close()
        _cleanup_seed_processes(proc, task_name, scene, seed, tracked_groups)
        if log is not None:
            log.close()


def _worker_main(task_name, scene, seed):
    _install_signal_handlers()
    emitted = [False]

    def emit_result(value, detail):
        vision_error = _latest_vision_error_text(task_name)
        payload = json.dumps(
            {
                "task": task_name,
                "scene": scene,
                "seed": seed,
                "value": value,
                "vision_error": vision_error,
                "detail": detail,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        print(WORKER_RESULT_PREFIX + payload, flush=True)
        emitted[0] = True

    value, detail = _run_one_seed(task_name, scene, seed, result_callback=emit_result)
    if not emitted[0]:
        emit_result(value, detail)
    return 0 if value >= 0.0 else 1


def _run_seed_worker(task_name, scene, seed, result_callback=None):
    code = f"""
import sys
sys.path.insert(0, {SCRIPT_DIR!r})
import test as challenge_test
challenge_test.WORKER_TASK = {str(task_name)!r}
challenge_test.WORKER_SCENE = {str(scene)!r}
challenge_test.WORKER_SEED = {int(seed)!r}
raise SystemExit(challenge_test.main())
"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            code,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        preexec_fn=os.setsid,
    )
    log = WorkerLog(proc)
    deadline = time.monotonic() + (
        LAUNCH_WAIT_SEC
        + TIME_LIMIT_SEC
        + POST_SUCCESS_WAIT_SEC
        + WORKER_WATCHDOG_MARGIN_SEC
    )
    tracked_groups = _descendant_process_groups(proc.pid)
    result_reported = False
    success_cleanup_deadline = None

    def report_result():
        nonlocal result_reported
        if log.result is None or result_reported:
            return False
        if result_callback is not None:
            result_callback(*log.result)
        result_reported = True
        return True

    try:
        while proc.poll() is None:
            if report_result():
                tracked_groups.update(_descendant_process_groups(proc.pid))
                if log.result[0] < 0.0:
                    break
                success_cleanup_deadline = (
                    time.monotonic() + POST_SUCCESS_WAIT_SEC + 2.0
                )
            if (
                success_cleanup_deadline is not None
                and time.monotonic() >= success_cleanup_deadline
            ):
                print(
                    "[WARN] test: {} {} seed {} cleanup grace expired; "
                    "forcing worker cleanup".format(task_name, scene, seed),
                    flush=True,
                )
                tracked_groups.update(_descendant_process_groups(proc.pid))
                break
            if _STOP_REQUESTED.is_set():
                tracked_groups.update(_descendant_process_groups(proc.pid))
                _terminate_process_group(
                    proc,
                    "seed_worker_{}_{}_{}".format(task_name, scene, seed),
                    40.0,
                    8.0,
                )
                break
            if time.monotonic() >= deadline:
                print(
                    "[ERROR] test: {} {} seed {} worker watchdog timeout".format(
                        task_name, scene, seed
                    )
                )
                tracked_groups.update(_descendant_process_groups(proc.pid))
                _terminate_process_group(
                    proc,
                    "seed_worker_{}_{}_{}".format(task_name, scene, seed),
                    10.0,
                    5.0,
                )
                break
            time.sleep(0.25)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _terminate_process_group(
                proc,
                "seed_worker_{}_{}_{}".format(task_name, scene, seed),
                2.0,
                2.0,
            )
        log.join()
        if log.result is not None:
            report_result()
            return log.result
        return (
            -1.0,
            _latest_vision_error_text(task_name),
            _unknown_placement_detail(task_name),
        )
    finally:
        if proc.poll() is None:
            tracked_groups.update(_descendant_process_groups(proc.pid))
        _terminate_process_group(
            proc,
            "seed_worker_{}_{}_{}".format(task_name, scene, seed),
            5.0,
            3.0,
        )
        _terminate_groups(tracked_groups)
        _cleanup_residuals("parent after {} {} seed {}".format(task_name, scene, seed))


def _append_result(fh, task_name, scene, seed, value, vision_error, detail):
    detail = str(detail).replace("\r", " ").replace("\n", " | ")
    vision_error = str(vision_error).replace("\r", " ").replace("\n", " ")
    if value >= 0:
        line = "{} {} {} {:.3f} {} {}\n".format(
            task_name, scene, seed, value, vision_error, detail
        )
    else:
        line = "{} {} {} -1 {} {}\n".format(
            task_name, scene, seed, vision_error, detail
        )
    fh.write(line)
    fh.flush()
    os.fsync(fh.fileno())


def _normalized_task_queue():
    normalized = []
    for index, item in enumerate(TASK_QUEUE, start=1):
        if not isinstance(item, dict):
            raise ValueError("TASK_QUEUE item {} must be a dict".format(index))
        task_name = str(item.get("task", "")).strip()
        scene = str(item.get("scene", "")).strip()
        seed_start = int(item.get("seed_start", 1))
        seed_end = int(item.get("seed_end", seed_start))
        if not task_name:
            raise ValueError("TASK_QUEUE item {} has empty task".format(index))
        if scene not in SCENE_SUCCESS_RULES:
            raise ValueError(
                "TASK_QUEUE item {} has unsupported scene {}".format(index, scene)
            )
        if seed_start > seed_end:
            raise ValueError(
                "TASK_QUEUE item {} seed_start {} > seed_end {}".format(
                    index, seed_start, seed_end
                )
            )
        normalized.append(
            {
                "task": task_name,
                "scene": scene,
                "seed_start": seed_start,
                "seed_end": seed_end,
            }
        )
    if not normalized:
        raise ValueError("TASK_QUEUE must contain at least one task")
    return normalized


def _preflight():
    missing = [
        name for name in ("roslaunch", "rosnode", "ps") if not shutil.which(name)
    ]
    if missing:
        raise RuntimeError("missing required commands: {}".format(", ".join(missing)))


def _batch_main():
    _install_signal_handlers()
    _preflight()
    task_queue = _normalized_task_queue()
    _cleanup_residuals("batch startup")
    interrupted = False
    try:
        with open(RESULT_FILE, "w", encoding="utf-8") as fh:
            fh.write("# task scene seed sim_time_sec vision_error detail\n")
            fh.flush()
            for task_index, task_config in enumerate(task_queue, start=1):
                task_name = task_config["task"]
                scene = task_config["scene"]
                seed_start = task_config["seed_start"]
                seed_end = task_config["seed_end"]
                print(
                    "[INFO] test: queue task {}/{} {} {} seeds {}-{}".format(
                        task_index,
                        len(task_queue),
                        task_name,
                        scene,
                        seed_start,
                        seed_end,
                    ),
                    flush=True,
                )
                for seed in range(seed_start, seed_end + 1):
                    if _STOP_REQUESTED.is_set():
                        interrupted = True
                        break
                    result_recorded = [False]

                    def record_result(value, vision_error, detail):
                        if result_recorded[0]:
                            return
                        _append_result(
                            fh,
                            task_name,
                            scene,
                            seed,
                            value,
                            vision_error,
                            detail,
                        )
                        result_recorded[0] = True
                        print(
                            "[INFO] test: recorded {} {} seed {} result {:.3f} "
                            "vision_error={}".format(
                                task_name,
                                scene,
                                seed,
                                value,
                                vision_error,
                            ),
                            flush=True,
                        )

                    value, vision_error, detail = _run_seed_worker(
                        task_name,
                        scene,
                        seed,
                        result_callback=record_result,
                    )
                    if not result_recorded[0]:
                        record_result(value, vision_error, detail)
                    if _STOP_REQUESTED.is_set():
                        interrupted = True
                        break
                if interrupted:
                    break
                print(
                    "[INFO] test: completed queue task {}/{} {} {}".format(
                        task_index, len(task_queue), task_name, scene
                    ),
                    flush=True,
                )
    finally:
        _cleanup_residuals("batch shutdown")
    if _STOP_REQUESTED.is_set():
        print("[INFO] test: cleanup complete", flush=True)
    return 130 if interrupted or _STOP_REQUESTED.is_set() else 0


def main():
    if WORKER_SEED is not None:
        if not WORKER_TASK or not WORKER_SCENE:
            raise RuntimeError("internal worker requires WORKER_TASK and WORKER_SCENE")
        return _worker_main(WORKER_TASK, WORKER_SCENE, int(WORKER_SEED))
    return _batch_main()


if __name__ == "__main__":
    raise SystemExit(main())
