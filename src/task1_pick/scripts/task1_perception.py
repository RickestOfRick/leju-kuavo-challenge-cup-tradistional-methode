#!/usr/bin/env python3
"""任务一的 task1_vision 采样与包裹位姿固定。"""

import json
import math
import statistics
import time

import rospy
from std_msgs.msg import String

from task1_config import (
    DEFAULT_VISION_TO_IK_OFFSET,
    TASK1_VISION_ALLOWED_POSITION_SOURCES,
    TASK1_VISION_FRAME_ID,
    TASK1_VISION_MEDIAN_FRAMES,
    TASK1_VISION_MIN_CONFIDENCE,
    TASK1_VISION_TIMEOUT,
    TASK1_VISION_TOPIC,
    TASK1_VISION_USED_TOPIC,
    PARCEL_NAMES,
)

_vision_to_ik_offset = [0.0, 0.0, 0.0]
_warned_generic_mapping = False
_warned_position_sources = set()
_used_vision_pub = None


def _fmt(values, digits=4):
    return " ".join("{:.{}f}".format(value, digits) for value in values)


def _param_vec(name, default, length):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    result = [float(part) for part in value]
    if len(result) != length:
        raise RuntimeError("{} expects {} values, got {}".format(name, length, result))
    return result


def _param_str_list(name, default):
    value = rospy.get_param(name, default)
    if value is None:
        return []
    if isinstance(value, str):
        return [
            part.strip() for part in value.replace(",", " ").split() if part.strip()
        ]
    return [str(part) for part in value]


def get_vision_to_ik_offset():
    return list(_vision_to_ik_offset)


def _position_source_allowed(obj, allowed_sources):
    if not allowed_sources:
        return True
    source = str(obj.get("position_source", ""))
    if source in allowed_sources:
        return True
    name = str(obj.get("name", ""))
    key = (name, source)
    if key not in _warned_position_sources:
        rospy.logwarn(
            "task1: skip %s because position_source=%s is not allowed; allowed=%s",
            name or "<unnamed>",
            source or "<missing>",
            allowed_sources,
        )
        _warned_position_sources.add(key)
    return False


def _parse_objects(payload, required_names, min_confidence, allowed_sources):
    global _warned_generic_mapping
    named = {}
    generic = []
    for obj in payload.get("objects", []):
        try:
            confidence = float(obj.get("confidence", 1.0))
            if confidence < min_confidence:
                continue
            if not _position_source_allowed(obj, allowed_sources):
                continue
            position = [float(value) for value in obj.get("position", [])[:3]]
            if len(position) != 3:
                continue
        except Exception:
            continue

        name = str(obj.get("name", ""))
        class_name = str(obj.get("class_name", ""))
        if name in required_names:
            named[name] = position
        elif class_name == "parcel" or name.startswith("parcel"):
            generic.append((name, position))

    result = {name: named[name] for name in required_names if name in named}
    missing = [name for name in required_names if name not in result]
    if missing and generic:
        if not _warned_generic_mapping:
            rospy.logwarn(
                "task1: task1_vision object names incomplete; fallback to spatial "
                "mapping for missing parcels: %s",
                missing,
            )
            _warned_generic_mapping = True
        generic.sort(key=lambda item: (item[1][1], item[1][0], item[0]))
        for name, (_, position) in zip(missing, generic):
            result[name] = position
    return result


def _median_position(samples, name):
    positions = [result[name] for _, result in samples if name in result]
    if not positions:
        return None
    return [
        float(statistics.median(position[axis] for position in positions))
        for axis in range(3)
    ]


def _named_object(payload, name):
    for obj in payload.get("objects", []):
        if str(obj.get("name", "")) == name:
            return obj
    return None


def _build_used_payload(samples, required_names, frame_id):
    selected_objects = []
    for name in required_names:
        position = _median_position(samples, name)
        if position is None:
            continue

        source_objects = []
        for payload, result in samples:
            if name not in result:
                continue
            obj = _named_object(payload, name)
            if obj is not None:
                source_objects.append(obj)

        selected = dict(source_objects[-1]) if source_objects else {"name": name}
        truth_positions = []
        for obj in source_objects:
            try:
                truth_position = [
                    float(value) for value in obj.get("truth_position", [])[:3]
                ]
            except Exception:
                continue
            if len(truth_position) == 3:
                truth_positions.append(truth_position)

        selected["position"] = position
        selected["used_for_task"] = True
        selected["median_sample_count"] = sum(
            1 for _, result in samples if name in result
        )
        if truth_positions:
            truth_position = [
                float(statistics.median(values)) for values in zip(*truth_positions)
            ]
            selected["truth_position"] = truth_position
            selected["truth_position_error"] = math.sqrt(
                sum((position[i] - truth_position[i]) ** 2 for i in range(3))
            )
        selected_objects.append(selected)

    return {
        "stamp": rospy.Time.now().to_sec(),
        "frame_id": frame_id,
        "selection_method": "coordinate_median",
        "selection_sample_count": len(samples),
        "objects": selected_objects,
    }


def _publish_used_payload(payload):
    global _used_vision_pub
    topic = rospy.get_param("~task1_vision_used_topic", TASK1_VISION_USED_TOPIC)
    if _used_vision_pub is None or _used_vision_pub.resolved_name != topic:
        _used_vision_pub = rospy.Publisher(topic, String, queue_size=1, latch=True)
    _used_vision_pub.publish(String(data=json.dumps(payload)))


def init_from_task1_vision():
    global _vision_to_ik_offset
    topic = rospy.get_param("~task1_vision_topic", TASK1_VISION_TOPIC)
    timeout = float(rospy.get_param("~task1_vision_timeout", TASK1_VISION_TIMEOUT))
    min_confidence = float(
        rospy.get_param("~task1_vision_min_confidence", TASK1_VISION_MIN_CONFIDENCE)
    )
    expected_frame = rospy.get_param("~task1_vision_frame_id", TASK1_VISION_FRAME_ID)
    allowed_sources = _param_str_list(
        "~task1_vision_allowed_position_sources", TASK1_VISION_ALLOWED_POSITION_SOURCES
    )
    require_all = bool(rospy.get_param("~require_all_parcels", True))
    required_names = list(rospy.get_param("~parcel_names", PARCEL_NAMES))
    median_frames = max(
        1,
        int(rospy.get_param("~task1_vision_median_frames", TASK1_VISION_MEDIAN_FRAMES)),
    )
    _vision_to_ik_offset = _param_vec(
        "~vision_to_ik_offset", DEFAULT_VISION_TO_IK_OFFSET, 3
    )

    deadline = time.time() + timeout
    last_result = {}
    samples = []
    rospy.loginfo(
        "task1: waiting for %d valid task1_vision frames from %s",
        median_frames,
        topic,
    )
    while not rospy.is_shutdown() and time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        try:
            msg = rospy.wait_for_message(topic, String, timeout=min(1.0, remaining))
            payload = json.loads(msg.data)
        except Exception:
            continue
        frame_id = str(payload.get("frame_id", expected_frame))
        if frame_id != expected_frame:
            raise RuntimeError(
                "task1_vision frame mismatch: expected {}, got {}".format(
                    expected_frame, frame_id
                )
            )
        result = _parse_objects(
            payload, required_names, min_confidence, allowed_sources
        )
        if result:
            last_result = result
        if result and (not require_all or all(name in result for name in required_names)):
            samples.append((payload, result))
            if len(samples) < median_frames:
                continue
            used_payload = _build_used_payload(samples, required_names, expected_frame)
            selected = {
                obj["name"]: list(obj["position"])
                for obj in used_payload["objects"]
            }
            _publish_used_payload(used_payload)
            rospy.loginfo(
                "task1: fixed parcel poses from coordinate median of %d frames",
                len(samples),
            )
            for obj in used_payload["objects"]:
                error = obj.get("truth_position_error")
                error_text = (
                    "n/a" if error is None else "{:.1f} mm".format(error * 1000.0)
                )
                rospy.loginfo(
                    "task1: vision median %s=%s truth error=%s",
                    obj["name"],
                    _fmt(obj["position"]),
                    error_text,
                )
            rospy.loginfo(
                "task1: vision_to_ik offset=%s", _fmt(_vision_to_ik_offset)
            )
            return selected

    if samples and not require_all:
        used_payload = _build_used_payload(samples, required_names, expected_frame)
        selected = {
            obj["name"]: list(obj["position"])
            for obj in used_payload["objects"]
        }
        _publish_used_payload(used_payload)
        rospy.logwarn(
            "task1: task1_vision timeout; using median of %d partial frames: %s",
            len(samples),
            sorted(selected),
        )
        return selected
    raise RuntimeError(
        "task1_vision did not provide {} valid frames with required parcels on {} "
        "within {:.1f}s; collected {}, last got {}".format(
            median_frames, topic, timeout, len(samples), sorted(last_result)
        )
    )
