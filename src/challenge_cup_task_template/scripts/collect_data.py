#!/usr/bin/env python3
"""
拍头部 RGB 图用于标注。用不同 seed 启动仿真，每次运行拍 N 张。

每次运行：仿真启动 → 拍 N 张 → 保存到 .scratch/scene3/training_data/images/
"""
import os
import time
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


class Collector:
    def __init__(self, save_dir):
        self.bridge = CvBridge()
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        self.head_rgb = None
        self.count = 0

    def _on_rgb(self, msg):
        self.head_rgb = msg

    def snap(self, seed):
        rospy.sleep(0.3)
        if self.head_rgb is None:
            rospy.logwarn("无 RGB")
            return
        rgb = self.bridge.compressed_imgmsg_to_cv2(self.head_rgb, "bgr8")
        name = "s%d_%04d.jpg" % (seed, self.count)
        cv2.imwrite(os.path.join(self.save_dir, "images", name), rgb,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.count += 1


def main():
    rospy.init_node("collect_images")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "..", ".scratch", "scene3", "training_data")
    c = Collector(save_dir)
    rospy.Subscriber("/cam_h/color/image_raw/compressed", CompressedImage, c._on_rgb, queue_size=1)
    rospy.sleep(1.5)

    seed = rospy.get_param("~seed", 0)
    n = rospy.get_param("~n", 30)
    rospy.loginfo("Seed=%d, 拍 %d 张, dir=%s", seed, n, save_dir)
    for _ in range(n):
        c.snap(seed)
        time.sleep(0.3)
    rospy.loginfo("完成: %d 张", c.count)


if __name__ == "__main__":
    main()
