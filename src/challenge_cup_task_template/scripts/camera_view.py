#!/usr/bin/env python3
"""同时显示头部和腕部相机，左右拼接。"""
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


class DualView:
    def __init__(self):
        self.bridge = CvBridge()
        self.head = None
        self.wrist = None

    def on_head(self, msg):
        try:
            self.head = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def on_wrist(self, msg):
        try:
            self.wrist = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def run(self):
        rospy.Subscriber("/cam_h/color/image_raw/compressed", CompressedImage, self.on_head, queue_size=1)
        rospy.Subscriber("/cam_r/color/image_raw/compressed", CompressedImage, self.on_wrist, queue_size=1)

        while not rospy.is_shutdown():
            h = self.head
            w = self.wrist

            if h is None and w is None:
                rospy.sleep(0.1)
                continue

            # 统一高度
            th = 360
            if h is not None:
                h = cv2.resize(h, (int(th * h.shape[1] / h.shape[0]), th))
            else:
                h = np.zeros((th, th, 3), dtype=np.uint8)
                cv2.putText(h, "No Head", (10, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if w is not None:
                w = cv2.resize(w, (int(th * w.shape[1] / w.shape[0]), th))
            else:
                w = np.zeros((th, th, 3), dtype=np.uint8)
                cv2.putText(w, "No Wrist", (10, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            canvas = np.hstack([h, w])
            cv2.putText(canvas, "Head", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(canvas, "Wrist", (h.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Cameras", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rospy.sleep(0.05)


if __name__ == "__main__":
    rospy.init_node("camera_view")
    DualView().run()
