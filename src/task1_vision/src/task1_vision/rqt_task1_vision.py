#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math

import rospy
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from rqt_gui_py.plugin import Plugin
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

try:
    import cv2
    from cv_bridge import CvBridge

    _CV_AVAILABLE = True
    _CV_IMPORT_ERROR = None
except Exception as exc:
    cv2 = None
    CvBridge = None
    _CV_AVAILABLE = False
    _CV_IMPORT_ERROR = exc


CAMERA_MODES = {
    "head": {
        "title": "Task1 Vision - Head Camera",
        "topic_param": "/task1_vision/head_image_topic",
        "topic_default": "/cam_h/color/image_raw/compressed",
        "draw_detections": True,
        "show_detection_ui": True,
    },
    "right": {
        "title": "Task1 Vision - Right Arm Camera",
        "topic_param": "/task1_vision/right_arm_image_topic",
        "topic_default": "/cam_r/color/image_raw/compressed",
        "draw_detections": False,
        "show_detection_ui": False,
    },
    "left": {
        "title": "Task1 Vision - Left Arm Camera",
        "topic_param": "/task1_vision/left_arm_image_topic",
        "topic_default": "/cam_l/color/image_raw/compressed",
        "draw_detections": False,
        "show_detection_ui": False,
    },
}


# Parse plugin arguments: -h/-r/-l select head/right-arm/left-arm RGB images.
def _parse_camera_mode(argv):
    mode = "head"
    for arg in argv:
        if arg in ("-h", "--head"):
            mode = "head"
        elif arg in ("-r", "--right"):
            mode = "right"
        elif arg in ("-l", "--left"):
            mode = "left"
    return mode


class Task1VisionPlugin(Plugin):
    def __init__(self, context):
        super(Task1VisionPlugin, self).__init__(context)
        self.setObjectName("Task1VisionPlugin")
        self._camera_mode = _parse_camera_mode(context.argv())
        self._camera_config = CAMERA_MODES[self._camera_mode]
        self._objects = []
        self._bbox_candidates = []
        self._used_objects = []
        self._used_objects_by_name = {}
        self._detection_mode = ""
        self._latest_image = None
        self._image_error = ""
        self._widget = QWidget()
        self._widget.setWindowTitle(self._camera_config["title"])

        layout = QVBoxLayout(self._widget)

        self._status = QLabel("Waiting for image")
        layout.addWidget(self._status)

        self._image_topic = rospy.get_param(
            self._camera_config["topic_param"],
            self._camera_config["topic_default"],
        )
        self._image_label = QLabel("Waiting for {}".format(self._image_topic))
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(640, 360)
        self._image_label.setStyleSheet("background-color: #111; color: #ddd;")
        layout.addWidget(self._image_label)

        self._table = None
        self._sub = None
        if self._camera_config["show_detection_ui"]:
            self._status.setText("Waiting for /task1_vision/objects_json")
            self._table = QTableWidget(0, 9)
            self._table.setHorizontalHeaderLabels(["name", "class", "conf", "x", "y", "z", "yaw", "valid", "source"])
            self._table.verticalHeader().setVisible(False)
            self._table.setAlternatingRowColors(True)
            layout.addWidget(self._table)

        context.add_widget(self._widget)

        if self._camera_config["show_detection_ui"]:
            self._sub = rospy.Subscriber("/task1_vision/objects_json", String, self._objects_callback, queue_size=1)
            self._used_sub = rospy.Subscriber(
                "/task1_vision/used_objects_json",
                String,
                self._used_objects_callback,
                queue_size=1,
            )
        self._image_sub = None
        if _CV_AVAILABLE:
            self._bridge = CvBridge()
            self._image_sub = rospy.Subscriber(self._image_topic, CompressedImage, self._image_callback, queue_size=1)
            self._image_label.setText("Waiting for {}".format(self._image_topic))
        else:
            self._image_label.setText("cv_bridge/cv2 unavailable: {}".format(_CV_IMPORT_ERROR))

        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._refresh_view)
        self._timer.start(200)

    def _objects_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            self._objects = payload.get("objects", [])
            self._bbox_candidates = payload.get("bbox_candidates", [])
            self._detection_mode = str(payload.get("detection_mode", ""))
            if self._used_objects:
                return
            if self._detection_mode == "yolo_bbox_depth_top_plane_fit":
                fitted = sum(
                    1
                    for candidate in self._bbox_candidates
                    if candidate.get("fit_succeeded", False)
                )
                self._status.setText(
                    "YOLO RGB + RANSAC top plane  boxes: {}  fitted: {}  objects: {}".format(
                        len(self._bbox_candidates), fitted, len(self._objects)
                    )
                )
        except Exception as exc:
            self._status.setText("bad json: {}".format(exc))

    def _used_objects_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            objects = payload.get("objects", [])
            if not objects:
                return
            self._used_objects = objects
            self._used_objects_by_name = {
                str(obj.get("name", "")): obj for obj in objects
            }
            self._status.setText(
                "task input locked: coordinate median of {} frames".format(
                    int(payload.get("selection_sample_count", 0))
                )
            )
        except Exception as exc:
            self._status.setText("bad used-vision json: {}".format(exc))

    def _used_measurement(self, obj):
        name = str(obj.get("name", ""))
        return self._used_objects_by_name.get(name, obj)

    def _image_callback(self, msg):
        try:
            self._latest_image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._image_error = ""
            if not self._camera_config["show_detection_ui"]:
                self._status.setText("{}  {}".format(self._camera_mode, self._image_topic))
        except Exception as exc:
            self._image_error = str(exc)

    def _refresh_view(self):
        self._refresh_image()
        self._refresh_table()

    def _refresh_image(self):
        if not _CV_AVAILABLE:
            return
        if self._latest_image is None:
            if self._image_error:
                self._image_label.setText("image error: {}".format(self._image_error))
            return

        frame = self._latest_image.copy()
        if self._camera_config["draw_detections"]:
            if not self._objects and self._bbox_candidates:
                self._draw_bbox_candidates(frame)
            self._draw_objects(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        label_size = self._image_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(pixmap)

    def _draw_objects(self, frame):
        height, width = frame.shape[:2]
        for obj in self._objects:
            center = obj.get("bbox_center")
            size = obj.get("bbox_size")
            if not center or not size or len(center) < 2 or len(size) < 2:
                continue
            cx, cy = float(center[0]), float(center[1])
            bw, bh = max(2.0, float(size[0])), max(2.0, float(size[1]))
            x1 = int(max(0, min(width - 1, cx - bw * 0.5)))
            y1 = int(max(0, min(height - 1, cy - bh * 0.5)))
            x2 = int(max(0, min(width - 1, cx + bw * 0.5)))
            y2 = int(max(0, min(height - 1, cy + bh * 0.5)))
            color = self._color_for_object(obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
            label = str(obj.get("name") or obj.get("class_name") or "")
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_bbox_candidates(self, frame):
        height, width = frame.shape[:2]
        for candidate in self._bbox_candidates:
            center = candidate.get("bbox_center")
            size = candidate.get("bbox_size")
            if not center or not size or len(center) < 2 or len(size) < 2:
                continue
            cx, cy = float(center[0]), float(center[1])
            bw, bh = max(2.0, float(size[0])), max(2.0, float(size[1]))
            x1 = int(max(0, min(width - 1, cx - bw * 0.5)))
            y1 = int(max(0, min(height - 1, cy - bh * 0.5)))
            x2 = int(max(0, min(width - 1, cx + bw * 0.5)))
            y2 = int(max(0, min(height - 1, cy + bh * 0.5)))
            fitted = bool(candidate.get("fit_succeeded", False))
            color = (40, 210, 255) if fitted else (60, 60, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = "parcel {:.2f} {}".format(
                float(candidate.get("confidence", 0.0)),
                "fit" if fitted else "no depth fit",
            )
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _color_for_object(self, obj):
        class_name = obj.get("class_name", "")
        if class_name == "parcel":
            return (40, 210, 255)
        return (240, 240, 240)

    def _refresh_table(self):
        if self._table is None:
            return
        objects = self._used_objects if self._used_objects else self._objects
        self._table.setRowCount(len(objects))
        for row, obj in enumerate(objects):
            position = obj.get("position", [0.0, 0.0, 0.0])
            yaw = obj.get("yaw", 0.0)
            values = [
                obj.get("name", ""),
                obj.get("class_name", ""),
                "{:.2f}".format(float(obj.get("confidence", 0.0))),
                "{:.3f}".format(float(position[0])),
                "{:.3f}".format(float(position[1])),
                "{:.3f}".format(float(position[2])),
                "{:.1f}".format(math.degrees(float(yaw))),
                "yes" if obj.get("yaw_valid", False) else "no",
                obj.get("position_source", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()

    def shutdown_plugin(self):
        if hasattr(self, "_timer"):
            self._timer.stop()
        if getattr(self, "_sub", None) is not None:
            self._sub.unregister()
        if getattr(self, "_used_sub", None) is not None:
            self._used_sub.unregister()
        if getattr(self, "_image_sub", None) is not None:
            self._image_sub.unregister()
