#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""任务二独立使用的头部相机姿态控制工具。"""

import math

import rospy


HEAD_TOPIC = "/robot_head_motion_data"
SENSORS_TOPIC = "/sensors_data_raw"
YAW_LIMIT_DEG = (-30.0, 30.0)
PITCH_LIMIT_DEG = (-25.0, 25.0)
GREEN = "\033[32m"
RESET = "\033[0m"


def _clamp(value, limits):
    return max(limits[0], min(limits[1], float(value)))


def _wait_for_connection(pub, timeout):
    deadline = rospy.Time.now() + rospy.Duration(float(timeout))
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if pub.get_num_connections() > 0:
            return True
        if rospy.Time.now() >= deadline:
            return False
        rate.sleep()
    return False


def _resolve_index(index, length):
    index = int(index)
    if index < 0:
        index = length + index
    if index < 0 or index >= length:
        raise IndexError("index {} outside joint_q length {}".format(index, length))
    return index


def read_actual_head_pose(
    sensors_topic=SENSORS_TOPIC,
    yaw_index=-2,
    pitch_index=-1,
    timeout=1.0,
):
    """从传感器消息读取实际头部 yaw 和 pitch，返回角度制。"""
    from kuavo_msgs.msg import sensorsData

    msg = rospy.wait_for_message(sensors_topic, sensorsData, timeout=float(timeout))
    joint_q = list(msg.joint_data.joint_q)
    yaw_i = _resolve_index(yaw_index, len(joint_q))
    pitch_i = _resolve_index(pitch_index, len(joint_q))
    return math.degrees(float(joint_q[yaw_i])), math.degrees(float(joint_q[pitch_i]))


def publish_head_camera_pose(
    yaw_deg=0.0,
    pitch_deg=8.0,
    topic=HEAD_TOPIC,
    repeats=5,
    interval=0.1,
    settle_time=0.8,
    wait_timeout=3.0,
    wait_for_subscriber=True,
    strict=False,
    sensors_topic=SENSORS_TOPIC,
    actual_yaw_index=-2,
    actual_pitch_index=-1,
    actual_wait_timeout=1.0,
):
    """发布头部相机 yaw、pitch 指令，并读取稳定后的实际角度。"""
    from kuavo_msgs.msg import robotHeadMotionData

    yaw = _clamp(yaw_deg, YAW_LIMIT_DEG)
    pitch = _clamp(pitch_deg, PITCH_LIMIT_DEG)

    pub = rospy.Publisher(topic, robotHeadMotionData, queue_size=10)
    connected = True
    if wait_for_subscriber:
        connected = _wait_for_connection(pub, wait_timeout)
        if not connected:
            msg = "task2_vision head_control: no subscriber on {} within {:.1f}s".format(
                topic, wait_timeout
            )
            if strict:
                raise RuntimeError(msg)
            rospy.logwarn(msg)

    command = robotHeadMotionData()
    command.joint_data = [yaw, pitch]
    for _ in range(max(1, int(repeats))):
        if rospy.is_shutdown():
            break
        pub.publish(command)
        rospy.sleep(float(interval))

    if settle_time > 0.0:
        rospy.sleep(float(settle_time))

    actual = None
    try:
        actual = read_actual_head_pose(
            sensors_topic=sensors_topic,
            yaw_index=actual_yaw_index,
            pitch_index=actual_pitch_index,
            timeout=actual_wait_timeout,
        )
    except Exception as exc:
        rospy.logwarn("task2_vision head_control: cannot read actual head pose: %s", exc)

    if actual is None:
        rospy.loginfo(
            GREEN
            + "task2_vision head_control: command yaw=%.1f deg pitch=%.1f deg; actual unavailable"
            + RESET,
            yaw,
            pitch,
        )
    else:
        rospy.loginfo(
            GREEN
            + "task2_vision head_control: command yaw=%.1f deg pitch=%.1f deg; actual yaw=%.1f deg pitch=%.1f deg"
            + RESET,
            yaw,
            pitch,
            actual[0],
            actual[1],
        )
    return connected


def apply_head_camera_pose_from_params(namespace="/task2_vision/head_control", **overrides):
    """读取任务二命名空间下的 ROS 参数并发布头部相机姿态。"""
    enabled = bool(overrides.get("enabled", rospy.get_param(namespace + "/enabled", True)))
    if not enabled:
        rospy.loginfo("task2_vision head_control: disabled")
        return False

    params = {
        "yaw_deg": rospy.get_param(namespace + "/yaw_deg", 0.0),
        "pitch_deg": rospy.get_param(namespace + "/pitch_deg", 8.0),
        "topic": rospy.get_param(namespace + "/topic", HEAD_TOPIC),
        "repeats": rospy.get_param(namespace + "/repeats", 5),
        "interval": rospy.get_param(namespace + "/interval", 0.1),
        "settle_time": rospy.get_param(namespace + "/settle_time", 0.8),
        "wait_timeout": rospy.get_param(namespace + "/wait_timeout", 3.0),
        "wait_for_subscriber": rospy.get_param(namespace + "/wait_for_subscriber", True),
        "strict": rospy.get_param(namespace + "/strict", False),
        "sensors_topic": rospy.get_param(namespace + "/sensors_topic", SENSORS_TOPIC),
        "actual_yaw_index": rospy.get_param(namespace + "/actual_yaw_index", -2),
        "actual_pitch_index": rospy.get_param(namespace + "/actual_pitch_index", -1),
        "actual_wait_timeout": rospy.get_param(namespace + "/actual_wait_timeout", 1.0),
    }
    params.update(
        {
            key: value
            for key, value in overrides.items()
            if value is not None and key != "enabled"
        }
    )
    return publish_head_camera_pose(**params)
