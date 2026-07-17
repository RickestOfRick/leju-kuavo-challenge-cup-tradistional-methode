#!/usr/bin/env python3
"""读取并固定 task2_vision 发布的六个工件位姿。"""

import json
import math
import time

import rospy
from std_msgs.msg import String

from task2_config import (
    GRASP_OFFSETS_BASE,
    VISION_ALLOWED_POSITION_SOURCES,
    VISION_FRAME_ID,
    VISION_MODE0_REFERENCE_TOPIC,
    VISION_MIN_CONFIDENCE,
    VISION_TIMEOUT,
    VISION_TOPIC,
    VISION_TRUTH_TOPIC,
)


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
    deadline = time.time() + timeout
    last_missing = list(motion_names)
    reference_publisher = None

    if expected_mode == 1:
        truth_topic = str(
            rospy.get_param("~task2_truth_topic", VISION_TRUTH_TOPIC)
        )
        reference_topic = str(
            rospy.get_param(
                "~task2_mode0_reference_topic",
                VISION_MODE0_REFERENCE_TOPIC,
            )
        )
        rospy.loginfo(
            "task2: mode 1 first reading mode-0 truth reference from %s",
            truth_topic,
        )
        reference_message = rospy.wait_for_message(
            truth_topic, String, timeout=timeout
        )
        reference_payload = json.loads(reference_message.data)
        if str(reference_payload.get("frame_id", "")) != expected_frame:
            raise RuntimeError(
                "task2 mode-0 reference frame={} expected={}".format(
                    reference_payload.get("frame_id"), expected_frame
                )
            )
        reference_names = {
            str(item.get("name", ""))
            for item in reference_payload.get("objects", [])
            if len(item.get("position", [])) >= 3
        }
        reference_missing = [
            name for name in motion_names if name not in reference_names
        ]
        if reference_missing:
            raise RuntimeError(
                "task2 mode-0 reference missing {}".format(
                    ", ".join(reference_missing)
                )
            )
        reference_payload["reference_role"] = "mode0_equivalent_motion_input"
        reference_by_name = {
            str(item.get("name", "")): item
            for item in reference_payload.get("objects", [])
        }
        for name in motion_names:
            item = reference_by_name[name]
            rospy.loginfo(
                "task2: same-run mode-0 reference %s base_link=%.4f %.4f %.4f "
                "yaw=%.2f deg",
                name,
                float(item["position"][0]),
                float(item["position"][1]),
                float(item["position"][2]),
                math.degrees(float(item.get("yaw", 0.0))),
            )
        reference_message = String(
            data=json.dumps(reference_payload, ensure_ascii=True, sort_keys=True)
        )
        reference_publisher = rospy.Publisher(
            reference_topic, String, queue_size=1, latch=True
        )
        connection_deadline = time.time() + min(5.0, timeout)
        while (
            not rospy.is_shutdown()
            and reference_publisher.get_num_connections() == 0
            and time.time() < connection_deadline
        ):
            rospy.sleep(0.05)
        reference_publisher.publish(reference_message)
        rospy.loginfo(
            "task2: fixed same-run mode-0 reference for %d objects on %s",
            len(reference_names),
            reference_topic,
        )

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
        "task2 vision timed out after {:.1f}s; missing {}".format(
            timeout, ", ".join(last_missing)
        )
    )
