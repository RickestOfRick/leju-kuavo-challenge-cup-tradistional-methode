#!/usr/bin/env python3
"""读取并固定 task2_vision 发布的六个工件位姿。"""

import json
import math
import time

import rospy
from std_msgs.msg import String

from task2_config import (
    OBJECT_NAMES,
    VISION_ALLOWED_POSITION_SOURCES,
    VISION_FRAME_ID,
    VISION_MIN_CONFIDENCE,
    VISION_TIMEOUT,
    VISION_TOPIC,
)


def _yaw_quaternion(yaw):
    half = 0.5 * float(yaw)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _parse_object(item):
    name = str(item.get("name", ""))
    if name not in OBJECT_NAMES:
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
    topic = rospy.get_param("~task2_vision_topic", VISION_TOPIC)
    expected_frame = rospy.get_param("~task2_vision_frame_id", VISION_FRAME_ID)
    timeout = float(rospy.get_param("~task2_vision_timeout", VISION_TIMEOUT))
    deadline = time.time() + timeout
    last_missing = list(OBJECT_NAMES)

    rospy.loginfo("task2: waiting for complete object poses from %s", topic)
    while not rospy.is_shutdown() and time.time() < deadline:
        try:
            message = rospy.wait_for_message(topic, String, timeout=1.0)
            payload = json.loads(message.data)
        except Exception:
            continue
        if str(payload.get("frame_id", "")) != expected_frame:
            rospy.logwarn_throttle(
                2.0,
                "task2: ignore vision frame=%s, expected=%s",
                payload.get("frame_id"),
                expected_frame,
            )
            continue

        parsed = {}
        for item in payload.get("objects", []):
            observation = _parse_object(item)
            if observation is not None:
                parsed[observation["name"]] = observation
        last_missing = [name for name in OBJECT_NAMES if name not in parsed]
        if not last_missing:
            rospy.loginfo("task2: fixed %d base_link object poses", len(parsed))
            for name in OBJECT_NAMES:
                observation = parsed[name]
                rospy.loginfo(
                    "task2: vision naming motion=%s ground_truth=%s",
                    name,
                    observation["ground_truth_name"],
                )
            return parsed

    raise RuntimeError(
        "task2 vision timed out after {:.1f}s; missing {}".format(
            timeout, ", ".join(last_missing)
        )
    )
