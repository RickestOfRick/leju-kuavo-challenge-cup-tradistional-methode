#!/usr/bin/env python3
"""发布场景二由 challenge_secret 生成的工件中心与 yaw 真值。"""

import json

import rospkg
import rospy
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float64MultiArray, String

from task2_vision.layout import (
    POSITION_SOURCE,
    build_published_objects,
    load_scene2_layout,
)


FRAME_ID = "base_link"


class Task2VisionNode:
    """读取一次固定 seed 布局，并通过两个 ROS 话题重复发布。"""

    def __init__(self):
        self.seed = int(rospy.get_param("~seed", 0))
        self.publish_hz = max(0.1, float(rospy.get_param("~publish_hz", 2.0)))
        self.qpos_topic = str(rospy.get_param("~qpos_topic", "/mujoco/qpos"))
        self.base_qpos_addr = int(rospy.get_param("~robot_base_qpos_addr", 0))
        self.qpos_timeout = float(rospy.get_param("~qpos_timeout", 20.0))
        simulator_path = rospkg.RosPack().get_path("challenge_cup_simulator")
        layout = load_scene2_layout(simulator_path, self.seed)
        base_position, base_quat_wxyz = self._read_world_base_pose()
        self.world_base_position = base_position
        self.world_base_quat_wxyz = base_quat_wxyz
        self.objects = build_published_objects(
            layout, base_position, base_quat_wxyz
        )

        self.json_pub = rospy.Publisher("~objects_json", String, queue_size=1, latch=True)
        self.poses_pub = rospy.Publisher("~poses", PoseArray, queue_size=1, latch=True)

        rospy.loginfo(
            "task2_vision: loaded %d scene2 objects from seed=%d",
            len(self.objects),
            self.seed,
        )
        for item in self.objects:
            rospy.loginfo(
                "task2_vision: %s position=%s yaw=%.3f deg",
                item["name"],
                [round(value, 6) for value in item["position"]],
                item["yaw_deg"],
            )

    def _read_world_base_pose(self):
        message = rospy.wait_for_message(
            self.qpos_topic, Float64MultiArray, timeout=self.qpos_timeout
        )
        values = list(message.data)
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

    def _make_json(self, stamp):
        payload = {
            "stamp": stamp.to_sec(),
            "frame_id": FRAME_ID,
            "scene": "scene2",
            "seed": self.seed,
            "frozen": True,
            "position_source": POSITION_SOURCE,
            "world_base_position": self.world_base_position,
            "world_base_quat_wxyz": self.world_base_quat_wxyz,
            "objects": self.objects,
        }
        return String(data=json.dumps(payload, ensure_ascii=True, sort_keys=True))

    def _make_pose_array(self, stamp):
        message = PoseArray()
        message.header.stamp = stamp
        message.header.frame_id = FRAME_ID
        for item in self.objects:
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
            self.json_pub.publish(self._make_json(stamp))
            self.poses_pub.publish(self._make_pose_array(stamp))
            rate.sleep()


def main():
    rospy.init_node("task2_vision")
    try:
        Task2VisionNode().spin()
    except Exception as exc:
        rospy.logfatal("task2_vision: startup failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
