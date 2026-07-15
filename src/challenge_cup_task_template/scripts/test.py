#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

mj = None
_MUJOCO_IMPORT_ERROR = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.abspath(__file__)
RESULT_FILE = os.path.join(SCRIPT_DIR, "result.txt")

# 0: 原 task1 动作批量测试；1: 运行 data.py 的包裹数据采集批处理。
TEST_MODE = 0

SEED_START = 1
SEED_END = 100
SCENE = "scene1"

TIME_LIMIT_SEC = 360.0
LAUNCH_WAIT_SEC = 180.0
POST_SUCCESS_WAIT_SEC = 20.0
POLL_INTERVAL_SEC = 0.25
WORKER_WATCHDOG_MARGIN_SEC = 90.0

QPOS_TOPIC = "/mujoco/qpos"
PARCEL_NAMES = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]
BOX_BODY = "sorting_box_0p4_0p3_0p3"
BOX_FLOOR_GEOM = "sorting_box_0p4_0p3_0p3_floor"
BOX_INNER_SIZE_XY = [0.40, 0.30]
BOX_RELATIVE_Z_RANGE = [0.0, 0.25]

WORKER_RESULT_PREFIX = "__TASK1_TEST_RESULT__ "
_STOP_REQUESTED = threading.Event()
_FORCE_STOP_REQUESTED = threading.Event()

# 只匹配本测试会启动的任务、仿真、计时器和视觉进程。
RESIDUAL_PROCESS_MARKERS = (
    "challenge_task.run_scene(",
    "challenge_task.py --scene",
    "challenge_cup_task_template/scripts/data.py",
    "task1_vision_scene1.launch",
    "load_kuavo_mujoco_challenge.launch",
    "task1_node.py",
    "roslaunch task1_pick",
    "challenge_secret.run_match_timer",
    "challenge_secret.run_cheat_monitor",
)

# 主脚本导入 task1_node；直接 rosrun/roslaunch 时节点名为 task1。
RESIDUAL_ROS_NODE_PREFIXES = (
    "/challenge_task",
    "/task1_vision",
    "/task1",
    "/task1_batch_test",
    "/parcel_data_collector",
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


def _load_mujoco():
    global mj, _MUJOCO_IMPORT_ERROR
    if mj is not None:
        return mj
    try:
        import mujoco as mujoco_module
    except Exception as exc:
        _MUJOCO_IMPORT_ERROR = exc
        return None
    mj = mujoco_module
    _MUJOCO_IMPORT_ERROR = None
    return mj


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
            timer_ready = (
                "challenge timer armed" in text or "比赛计时器已启动" in text
            )
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
                    self.result = (float(payload["value"]), str(payload["detail"]))
                except Exception as exc:
                    print("[WARN] test: invalid worker result: {}".format(exc))
                continue
            if text:
                print(text, flush=True)

    def join(self, timeout=2.0):
        self._thread.join(timeout=timeout)


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
                "[WARN] test: failed to signal process group {}: {}".format(
                    pgid, exc
                )
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


def _terminate_process_group(
    proc, name="process", wait_sigint=25.0, wait_sigterm=8.0
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


def _cleanup_seed_processes(proc, seed, tracked_groups):
    if proc is not None and proc.poll() is None:
        tracked_groups.update(_descendant_process_groups(proc.pid))
    _terminate_process_group(proc, "challenge_task_seed_{}".format(seed))
    if proc is not None and proc.stdin is not None:
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
    _cleanup_ros_graph()
    _terminate_groups(tracked_groups)
    _cleanup_residuals("after seed {}".format(seed))


def _start_challenge_task(seed):
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import challenge_task
challenge_task.NEED_OPERATOR_ENTER = 0
seed = int(sys.argv[2])
challenge_task.run_scene(
    scene=sys.argv[3],
    seed=seed,
    node_name="challenge_task_scene1_seed_{}".format(seed),
    timeout=120,
    time_limit=float(sys.argv[4]),
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
            SCRIPT_DIR,
            str(seed),
            SCENE,
            str(TIME_LIMIT_SEC),
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

    master = rosgraph.Master("/task1_batch_test_probe")
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

    rospy.init_node("task1_batch_test", anonymous=True, disable_signals=True)
    return True


def _read_sim_time(timeout=0.5):
    import rospy
    from kuavo_msgs.msg import sensorsData

    message = rospy.wait_for_message(
        "/sensors_data_raw", sensorsData, timeout=float(timeout)
    )
    return float(message.sensor_time.to_sec())


class QposBoxChecker:
    def __init__(self):
        self.model_path = None
        self.model = None
        self.data = None

    def _load_model(self):
        import rospy

        model_path = os.path.abspath(str(rospy.get_param("legged_robot_scene_param")))
        if self.model is not None and self.model_path == model_path:
            return
        mujoco_module = _load_mujoco()
        if mujoco_module is None:
            raise RuntimeError("mujoco import failed: {}".format(_MUJOCO_IMPORT_ERROR))
        if not os.path.isfile(model_path):
            raise RuntimeError("MuJoCo scene XML not found: {}".format(model_path))
        self.model = mujoco_module.MjModel.from_xml_path(model_path)
        self.data = mujoco_module.MjData(self.model)
        self.model_path = model_path

    def _forward_latest_qpos(self, timeout=0.5):
        import rospy
        from std_msgs.msg import Float64MultiArray

        self._load_model()
        msg = rospy.wait_for_message(QPOS_TOPIC, Float64MultiArray, timeout=timeout)
        qpos = list(msg.data)
        if len(qpos) < self.model.nq:
            raise RuntimeError(
                "qpos len {} < model nq {}".format(len(qpos), self.model.nq)
            )
        self.data.qpos[: self.model.nq] = qpos[: self.model.nq]
        _load_mujoco().mj_forward(self.model, self.data)

    def _parcel_in_box(self, parcel_name):
        parcel_pos = [float(value) for value in self.data.body(parcel_name).xpos]
        box = self.data.body(BOX_BODY)
        box_pos = [float(value) for value in box.xpos]
        box_mat = [float(value) for value in box.xmat]

        delta = [parcel_pos[index] - box_pos[index] for index in range(3)]
        local_x = box_mat[0] * delta[0] + box_mat[3] * delta[1] + box_mat[6] * delta[2]
        local_y = box_mat[1] * delta[0] + box_mat[4] * delta[1] + box_mat[7] * delta[2]
        local_z = box_mat[2] * delta[0] + box_mat[5] * delta[1] + box_mat[8] * delta[2]

        floor_size = self.model.geom(BOX_FLOOR_GEOM).size
        floor_pos = self.model.geom(BOX_FLOOR_GEOM).pos
        box_bottom_local_z = float(floor_pos[2]) - float(floor_size[2])
        relative_z = local_z - box_bottom_local_z
        z_min, z_max = [float(value) for value in BOX_RELATIVE_Z_RANGE]
        half_x = BOX_INNER_SIZE_XY[0] * 0.5
        half_y = BOX_INNER_SIZE_XY[1] * 0.5
        inside = (
            abs(local_x) <= half_x
            and abs(local_y) <= half_y
            and z_min <= relative_z <= z_max
        )
        detail = "{} local_xy=[{:.3f},{:.3f}] relative_z={:.3f} range=[{:.3f},{:.3f}]".format(
            parcel_name, local_x, local_y, relative_z, z_min, z_max
        )
        return inside, detail

    def all_parcels_in_box(self):
        self._forward_latest_qpos(timeout=0.5)
        details = []
        for name in PARCEL_NAMES:
            inside, detail = self._parcel_in_box(name)
            details.append(detail)
            if not inside:
                return False, "; ".join(details)
        return True, "; ".join(details)


def _run_one_seed(seed):
    print("[INFO] test: start seed {}".format(seed), flush=True)
    _cleanup_residuals("before seed {}".format(seed))
    proc = None
    log = None
    tracked_groups = set()
    start_wall = time.monotonic()
    timer_start_sim = None
    last_sim_time = None
    last_sim_sample_wall = None
    checker = QposBoxChecker()
    success_time = None
    success_detail = ""

    try:
        proc = _start_challenge_task(seed)
        tracked_groups.update(_descendant_process_groups(proc.pid))
        log = ChildLog(proc)
        if not _ensure_rospy(proc):
            if proc.poll() is not None:
                return -1.0, "challenge task exited before ROS master became ready"
            return -1.0, "ros master unavailable"

        while not _STOP_REQUESTED.is_set():
            now_wall = time.monotonic()
            timer_ready_wall = log.timer_ready_wall
            if timer_ready_wall is None and now_wall - start_wall > LAUNCH_WAIT_SEC:
                return -1.0, "launch timeout before timer"

            elapsed_sim = None
            if timer_ready_wall is not None:
                try:
                    last_sim_time = _read_sim_time(timeout=0.5)
                    last_sim_sample_wall = now_wall
                except Exception:
                    if timer_start_sim is None and now_wall - timer_ready_wall > 5.0:
                        return -1.0, "simulation time unavailable after timer start"
                    if (
                        timer_start_sim is not None
                        and last_sim_sample_wall is not None
                        and now_wall - last_sim_sample_wall > 5.0
                    ):
                        return -1.0, "simulation time stopped updating"
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
                        return -1.0, "simulation timeout"

            if proc.poll() is not None:
                try:
                    ok, detail = checker.all_parcels_in_box()
                    if ok and elapsed_sim is not None:
                        return elapsed_sim, detail
                    return -1.0, "process exited before success: {}".format(detail)
                except Exception as exc:
                    return -1.0, (
                        "process exited before qpos success check: {}".format(exc)
                    )

            try:
                ok, detail = checker.all_parcels_in_box()
            except Exception:
                time.sleep(POLL_INTERVAL_SEC)
                continue
            if ok:
                if elapsed_sim is None:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                success_time = elapsed_sim
                success_detail = detail
                break
            time.sleep(POLL_INTERVAL_SEC)

        if _STOP_REQUESTED.is_set():
            return -1.0, "interrupted"

        wait_deadline = time.monotonic() + POST_SUCCESS_WAIT_SEC
        while proc.poll() is None and time.monotonic() < wait_deadline:
            if _STOP_REQUESTED.wait(0.25):
                break
        return success_time, success_detail
    finally:
        _cleanup_seed_processes(proc, seed, tracked_groups)
        if log is not None:
            log.close()


def _worker_main(seed):
    _install_signal_handlers()
    value, detail = _run_one_seed(seed)
    payload = json.dumps(
        {"value": value, "detail": detail}, ensure_ascii=False, separators=(",", ":")
    )
    print(WORKER_RESULT_PREFIX + payload, flush=True)
    return 0 if value >= 0.0 else 1


def _run_seed_worker(seed):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", SCRIPT_PATH, "--worker-seed", str(seed)],
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
    try:
        while proc.poll() is None:
            if _STOP_REQUESTED.is_set():
                tracked_groups.update(_descendant_process_groups(proc.pid))
                _terminate_process_group(proc, "seed_worker_{}".format(seed), 40.0, 8.0)
                break
            if time.monotonic() >= deadline:
                print("[ERROR] test: seed {} worker watchdog timeout".format(seed))
                tracked_groups.update(_descendant_process_groups(proc.pid))
                _terminate_process_group(proc, "seed_worker_{}".format(seed), 10.0, 5.0)
                break
            time.sleep(0.25)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc, "seed_worker_{}".format(seed), 2.0, 2.0)
        log.join()
        if log.result is not None:
            return log.result
        return -1.0, "worker exited without result (exit={})".format(proc.returncode)
    finally:
        if proc.poll() is None:
            tracked_groups.update(_descendant_process_groups(proc.pid))
        _terminate_process_group(proc, "seed_worker_{}".format(seed), 5.0, 3.0)
        _terminate_groups(tracked_groups)
        _cleanup_residuals("parent after seed {}".format(seed))


def _append_result(fh, seed, value, detail):
    detail = str(detail).replace("\r", " ").replace("\n", " | ")
    if value >= 0:
        line = "{} {:.3f} {}\n".format(seed, value, detail)
    else:
        line = "{} -1 {}\n".format(seed, detail)
    fh.write(line)
    fh.flush()
    os.fsync(fh.fileno())


def _preflight():
    missing = [name for name in ("roslaunch", "rosnode", "ps") if not shutil.which(name)]
    if missing:
        raise RuntimeError("missing required commands: {}".format(", ".join(missing)))
    if _load_mujoco() is None:
        raise RuntimeError("mujoco import failed: {}".format(_MUJOCO_IMPORT_ERROR))


def _batch_main():
    _install_signal_handlers()
    _preflight()
    _cleanup_residuals("batch startup")
    interrupted = False
    try:
        with open(RESULT_FILE, "w", encoding="utf-8") as fh:
            fh.write("# seed sim_time_sec detail\n")
            fh.flush()
            for seed in range(SEED_START, SEED_END + 1):
                if _STOP_REQUESTED.is_set():
                    interrupted = True
                    break
                value, detail = _run_seed_worker(seed)
                _append_result(fh, seed, value, detail)
                print(
                    "[INFO] test: seed {} result {:.3f}".format(seed, value),
                    flush=True,
                )
                if _STOP_REQUESTED.is_set():
                    interrupted = True
                    break
    finally:
        _cleanup_residuals("batch shutdown")
    if _STOP_REQUESTED.is_set():
        print("[INFO] test: cleanup complete", flush=True)
    return 130 if interrupted or _STOP_REQUESTED.is_set() else 0


def main():
    parser = argparse.ArgumentParser(description="Batch test challenge scene1 task1")
    parser.add_argument("--worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_seed is not None:
        return _worker_main(args.worker_seed)
    if int(TEST_MODE) == 1:
        _install_signal_handlers()
        return subprocess.call(
            [sys.executable, os.path.join(SCRIPT_DIR, "data.py")]
        )
    return _batch_main()


if __name__ == "__main__":
    raise SystemExit(main())
