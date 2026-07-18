#!/usr/bin/env python3
"""读取并固定 task2_vision 发布的六个工件位姿。"""

import json
import math
import threading
import time

import rospy
from kuavo_msgs.msg import sensorsData
from std_msgs.msg import String

from task2_config import (
    GRASP_OFFSETS_BASE,
    HANDOFF_RELOCALIZATION_REQUEST_TOPIC,
    HANDOFF_RELOCALIZATION_RESULT_TOPIC,
    HANDOFF_RELOCALIZATION_TIMEOUT,
    HANDOFF_RELOCALIZATION_WARN_TIME,
    TOPIC_TIMEOUT,
    VISION_ALLOWED_POSITION_SOURCES,
    VISION_FIT_COORDINATE_REFERENCE,
    VISION_FRAME_ID,
    VISION_MIN_CONFIDENCE,
    VISION_TIMEOUT,
    VISION_TOPIC,
    VISION_TRUTH_TOPIC,
    VISION_WARN_TIME,
)


class _SensorTimeClock:
    """Track MuJoCo simulation time published with the raw sensor stream."""

    def __init__(self, topic="/sensors_data_raw"):
        self.topic = str(topic)
        self._condition = threading.Condition()
        self._seconds = None
        self._subscriber = rospy.Subscriber(
            self.topic, sensorsData, self._callback, queue_size=1
        )

    def _callback(self, message):
        seconds = float(message.sensor_time.to_sec())
        if not math.isfinite(seconds):
            return
        with self._condition:
            self._seconds = seconds
            self._condition.notify_all()

    def wait_until_ready(self, wall_timeout=TOPIC_TIMEOUT):
        wall_deadline = time.monotonic() + float(wall_timeout)
        with self._condition:
            while self._seconds is None and not rospy.is_shutdown():
                remaining = wall_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "timed out waiting for {} sensor_time".format(self.topic)
                    )
                self._condition.wait(timeout=min(0.1, remaining))
            if rospy.is_shutdown():
                raise RuntimeError("shutdown while waiting for simulation time")
            return float(self._seconds)

    def seconds(self):
        with self._condition:
            if self._seconds is None:
                raise RuntimeError("{} sensor_time is unavailable".format(self.topic))
            return float(self._seconds)


class _SimulationDeadline:
    """Expire after simulation seconds; wall time only detects a stopped clock."""

    def __init__(
        self,
        duration,
        warning_time=0.0,
        label="vision",
        stall_timeout=TOPIC_TIMEOUT,
    ):
        self.duration = float(duration)
        self.warning_time = max(0.0, float(warning_time))
        self.label = str(label)
        if not math.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("simulation timeout must be finite and positive")
        self.clock = _sensor_time_clock()
        self.start = self.clock.wait_until_ready()
        self.previous = self.start
        self.last_progress_wall = time.monotonic()
        self.stall_timeout = float(stall_timeout)
        self.warned = False

    def expired(self):
        now = self.clock.seconds()
        if now + 1e-12 < self.previous:
            raise RuntimeError("simulation time moved backwards")
        if now > self.previous + 1e-12:
            self.last_progress_wall = time.monotonic()
        elif time.monotonic() - self.last_progress_wall >= self.stall_timeout:
            raise RuntimeError(
                "{} sensor_time stopped advancing".format(self.clock.topic)
            )
        self.previous = now
        elapsed = now - self.start
        if (
            not self.warned
            and self.warning_time > 0.0
            and elapsed + 1e-12 >= self.warning_time
            and elapsed + 1e-12 < self.duration
        ):
            rospy.logwarn(
                "%s still waiting after %.1fs simulation time; continuing until %.1fs",
                self.label,
                self.warning_time,
                self.duration,
            )
            self.warned = True
        return elapsed + 1e-12 >= self.duration


_SENSOR_TIME_CLOCK = None
_SENSOR_TIME_CLOCK_LOCK = threading.Lock()


def _sensor_time_clock():
    global _SENSOR_TIME_CLOCK
    with _SENSOR_TIME_CLOCK_LOCK:
        if _SENSOR_TIME_CLOCK is None:
            _SENSOR_TIME_CLOCK = _SensorTimeClock()
        return _SENSOR_TIME_CLOCK


def _wait_for_message_before_deadline(topic, message_type, deadline, label):
    while not rospy.is_shutdown():
        if deadline.expired():
            raise RuntimeError(
                "{} timed out after {:.1f}s simulation time".format(
                    label, deadline.duration
                )
            )
        try:
            message = rospy.wait_for_message(topic, message_type, timeout=0.5)
        except Exception:
            continue
        if deadline.expired():
            raise RuntimeError(
                "{} timed out after {:.1f}s simulation time".format(
                    label, deadline.duration
                )
            )
        return message
    raise RuntimeError("shutdown while waiting for {}".format(label))


def _yaw_quaternion(yaw):
    half = 0.5 * float(yaw)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _parse_object(item):
    name = str(item.get("name", ""))
    if name not in GRASP_OFFSETS_BASE:
        return None
    class_name = str(item.get("class_name", ""))
    if not class_name:
        class_name = name.rsplit("_", 1)[0]
    if class_name not in ("part_type_a", "part_type_b", "part_type_c"):
        return None
    if float(item.get("confidence", 1.0)) < VISION_MIN_CONFIDENCE:
        return None
    source = str(item.get("position_source", ""))
    if VISION_ALLOWED_POSITION_SOURCES and source not in VISION_ALLOWED_POSITION_SOURCES:
        return None

    position = [float(value) for value in item.get("position", [])[:3]]
    if len(position) != 3:
        return None
    yaw = float(item.get("yaw", 0.0))
    orientation = [float(value) for value in item.get("orientation", [])[:4]]
    if len(orientation) != 4:
        orientation = _yaw_quaternion(yaw)
    if not all(math.isfinite(value) for value in position + orientation + [yaw]):
        return None
    ground_truth_name = str(item.get("ground_truth_name", name))

    return {
        "name": name,
        "motion_name": name,
        "ground_truth_name": ground_truth_name,
        "class_name": class_name,
        "position": position,
        "orientation": orientation,
        "yaw": yaw,
        "confidence": float(item.get("confidence", 1.0)),
        "position_source": source,
    }


def wait_for_task2_objects():
    """等待一帧带完整实例名的六工件输入并固定。"""
    motion_names = list(GRASP_OFFSETS_BASE)
    topic = rospy.get_param("~task2_vision_topic", VISION_TOPIC)
    expected_mode = int(rospy.get_param("~task2_vision_mode", -1))
    expected_naming_method = str(
        rospy.get_param("~task2_instance_naming_method", "")
    )
    expected_frame = rospy.get_param("~task2_vision_frame_id", VISION_FRAME_ID)
    timeout = float(rospy.get_param("~task2_vision_timeout", VISION_TIMEOUT))
    warning_time = float(
        rospy.get_param("~task2_vision_warn_time", VISION_WARN_TIME)
    )
    deadline = _SimulationDeadline(
        timeout,
        warning_time=warning_time,
        label="task2 initial vision",
    )
    last_missing = list(motion_names)

    rospy.loginfo("task2: waiting for complete object poses from %s", topic)
    while not rospy.is_shutdown() and not deadline.expired():
        try:
            message = rospy.wait_for_message(topic, String, timeout=1.0)
            payload = json.loads(message.data)
        except Exception:
            continue
        if deadline.expired():
            break
        if str(payload.get("frame_id", "")) != expected_frame:
            rospy.logwarn_throttle(
                2.0,
                "task2: ignore vision frame=%s, expected=%s",
                payload.get("frame_id"),
                expected_frame,
            )
            continue
        try:
            payload_mode = int(payload.get("vision_mode", -1))
        except (TypeError, ValueError):
            rospy.logwarn_throttle(
                2.0,
                "task2: ignore invalid vision mode=%s",
                payload.get("vision_mode"),
            )
            continue
        if expected_mode >= 0 and payload_mode != expected_mode:
            rospy.logwarn_throttle(
                2.0,
                "task2: ignore vision mode=%s, expected=%d",
                payload.get("vision_mode"),
                expected_mode,
            )
            continue
        if expected_mode in (1, 2):
            coordinate_reference = str(
                payload.get("fit_coordinate_reference", "")
            )
            if coordinate_reference != VISION_FIT_COORDINATE_REFERENCE:
                rospy.logwarn_throttle(
                    2.0,
                    "task2: ignore fit coordinate reference=%s, expected=%s",
                    coordinate_reference,
                    VISION_FIT_COORDINATE_REFERENCE,
                )
                continue
        payload_naming_method = str(payload.get("instance_naming_method", ""))
        if (
            expected_naming_method
            and payload_naming_method != expected_naming_method
        ):
            rospy.logwarn_throttle(
                2.0,
                "task2: ignore instance naming=%s, expected=%s",
                payload_naming_method,
                expected_naming_method,
            )
            continue

        parsed = {}
        for item in payload.get("objects", []):
            observation = _parse_object(item)
            if observation is not None:
                parsed[observation["name"]] = observation
        last_missing = [name for name in motion_names if name not in parsed]
        if not last_missing:
            rospy.loginfo("task2: fixed %d base_link object poses", len(parsed))
            for name in motion_names:
                observation = parsed[name]
                rospy.loginfo(
                    "task2: vision naming motion=%s ground_truth=%s",
                    name,
                    observation["ground_truth_name"],
                )
            return parsed

    raise RuntimeError(
        "task2 vision timed out after {:.1f}s simulation time; missing {}".format(
            timeout, ", ".join(last_missing)
        )
    )


def _mode0_relocalized_object(source_observation, timeout, warning_time):
    """Read the corresponding live truth object when the whole task uses mode 0."""
    truth_name = str(
        source_observation.get("ground_truth_name", source_observation["name"])
    )
    motion_name = str(source_observation["name"])
    topic = str(rospy.get_param("~task2_truth_topic", VISION_TRUTH_TOPIC))
    deadline = _SimulationDeadline(
        timeout,
        warning_time=warning_time,
        label="task2 mode-0 handoff relocalization",
    )
    while not rospy.is_shutdown() and not deadline.expired():
        try:
            message = rospy.wait_for_message(topic, String, timeout=1.0)
            payload = json.loads(message.data)
        except Exception:
            continue
        if deadline.expired():
            break
        for item in payload.get("objects", []):
            if str(item.get("ground_truth_name", item.get("name", ""))) != truth_name:
                continue
            renamed = dict(item)
            renamed["name"] = motion_name
            renamed["ground_truth_name"] = truth_name
            observation = _parse_object(renamed)
            if observation is not None:
                return observation
    raise RuntimeError(
        "task2 mode-0 handoff relocalization timed out after {:.1f}s simulation "
        "time for {}".format(timeout, motion_name)
    )


def relocalize_handoff_object(source_observation, expected_position):
    """Request a fresh full-angle RGB-D/CAD fit for one released object."""
    timeout = float(
        rospy.get_param(
            "~task2_handoff_relocalization_timeout",
            HANDOFF_RELOCALIZATION_TIMEOUT,
        )
    )
    warning_time = float(
        rospy.get_param(
            "~task2_handoff_relocalization_warn_time",
            HANDOFF_RELOCALIZATION_WARN_TIME,
        )
    )
    expected_mode = int(rospy.get_param("~task2_vision_mode", -1))
    if expected_mode == 0:
        return _mode0_relocalized_object(
            source_observation, timeout, warning_time
        )
    if expected_mode not in (1, 2):
        raise RuntimeError(
            "task2 handoff relocalization requires vision mode 0, 1, or 2"
        )

    expected = [float(value) for value in expected_position]
    if len(expected) != 3 or not all(math.isfinite(value) for value in expected):
        raise ValueError("handoff relocalization expected position must contain xyz")
    request_topic = str(
        rospy.get_param(
            "~task2_handoff_relocalization_request_topic",
            HANDOFF_RELOCALIZATION_REQUEST_TOPIC,
        )
    )
    result_topic = str(
        rospy.get_param(
            "~task2_handoff_relocalization_result_topic",
            HANDOFF_RELOCALIZATION_RESULT_TOPIC,
        )
    )
    request_id = "{}-{}".format(
        source_observation["name"], int(time.time() * 1000000000.0)
    )
    request = {
        "request_id": request_id,
        "name": str(source_observation["name"]),
        "ground_truth_name": str(
            source_observation.get(
                "ground_truth_name", source_observation["name"]
            )
        ),
        "class_name": str(source_observation["class_name"]),
        "expected_position": expected,
        "vision_mode": expected_mode,
    }
    publisher = rospy.Publisher(request_topic, String, queue_size=1, latch=True)
    connection_deadline = time.monotonic() + min(3.0, timeout)
    while (
        not rospy.is_shutdown()
        and publisher.get_num_connections() == 0
        and time.monotonic() < connection_deadline
    ):
        rospy.sleep(0.05)
    if publisher.get_num_connections() == 0:
        raise RuntimeError(
            "task2 handoff relocalization has no subscriber on {}".format(
                request_topic
            )
        )
    publisher.publish(
        String(data=json.dumps(request, ensure_ascii=True, sort_keys=True))
    )
    rospy.loginfo(
        "task2: requested full-angle handoff relocalization id=%s object=%s "
        "expected=%s",
        request_id,
        source_observation["name"],
        [round(value, 4) for value in expected],
    )

    deadline = _SimulationDeadline(
        timeout,
        warning_time=warning_time,
        label="task2 handoff relocalization for {}".format(
            source_observation["name"]
        ),
    )
    while not rospy.is_shutdown() and not deadline.expired():
        try:
            message = rospy.wait_for_message(result_topic, String, timeout=1.0)
            payload = json.loads(message.data)
        except Exception:
            continue
        if deadline.expired():
            break
        if str(payload.get("request_id", "")) != request_id:
            continue
        if not bool(payload.get("success", False)):
            raise RuntimeError(
                "task2 handoff relocalization failed for {}: {}".format(
                    source_observation["name"],
                    payload.get("reason", "unknown"),
                )
            )
        observation = _parse_object(payload.get("object", {}))
        if observation is None:
            raise RuntimeError("task2 handoff relocalization returned invalid object")
        rospy.loginfo(
            "task2: handoff relocalized %s base_link=%.4f %.4f %.4f "
            "yaw=%.2f deg",
            observation["name"],
            observation["position"][0],
            observation["position"][1],
            observation["position"][2],
            math.degrees(observation["yaw"]),
        )
        return observation
    raise RuntimeError(
        "task2 handoff relocalization timed out after {:.1f}s simulation time "
        "for {}".format(
            timeout, source_observation["name"]
        )
    )
