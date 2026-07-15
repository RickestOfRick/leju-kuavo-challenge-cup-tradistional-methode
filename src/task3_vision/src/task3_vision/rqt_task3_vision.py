#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""任务三 qpos 假框 RQT 检查界面。"""

import json

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
    import numpy as np

    _CV_AVAILABLE = True
    _CV_IMPORT_ERROR = None
except Exception as exc:
    cv2 = None
    np = None
    _CV_AVAILABLE = False
    _CV_IMPORT_ERROR = exc


class Task3VisionPlugin(Plugin):
    def __init__(self, context):
        super(Task3VisionPlugin, self).__init__(context)
        self.setObjectName("Task3VisionPlugin")
        self._objects = []
        self._latest_image = None
        self._image_error = ""

        self._widget = QWidget()
        self._widget.setWindowTitle("任务三视觉 - 料盘假框")
        layout = QVBoxLayout(self._widget)

        self._status = QLabel("等待 /task3_vision/objects_json")
        layout.addWidget(self._status)

        self._image_label = QLabel("等待头部相机图像")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(640, 360)
        self._image_label.setStyleSheet("background-color: #111; color: #ddd;")
        layout.addWidget(self._image_label)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["名称", "层级", "类别", "x", "y", "z", "框宽", "框高"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)
        context.add_widget(self._widget)

        self._image_topic = rospy.get_param(
            "/task3_vision/head_image_topic",
            "/cam_h/color/image_raw/compressed",
        )
        self._objects_sub = rospy.Subscriber(
            "/task3_vision/objects_json",
            String,
            self._objects_callback,
            queue_size=1,
        )
        self._image_sub = rospy.Subscriber(
            self._image_topic,
            CompressedImage,
            self._image_callback,
            queue_size=1,
        )

        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(200)

    def _objects_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            self._objects = list(payload.get("objects", []))
            lower_count = sum(
                1 for item in self._objects if item.get("level") == "lower"
            )
            upper_count = sum(
                1 for item in self._objects if item.get("level") == "upper"
            )
            self._status.setText(
                "qpos 投影框：{}，下层：{}，上层：{}".format(
                    len(self._objects), lower_count, upper_count
                )
            )
        except Exception as exc:
            self._status.setText("结果解析失败：{}".format(exc))

    def _image_callback(self, msg):
        if not _CV_AVAILABLE:
            self._image_error = "OpenCV/NumPy 不可用：{}".format(
                _CV_IMPORT_ERROR
            )
            return
        try:
            encoded = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise RuntimeError("压缩图像解码为空")
            self._latest_image = image
            self._image_error = ""
        except Exception as exc:
            self._image_error = str(exc)

    @staticmethod
    def _object_color(obj):
        if obj.get("level") == "upper":
            return (60, 220, 255)
        return (80, 220, 80)

    def _draw_objects(self, image):
        height, width = image.shape[:2]
        for obj in self._objects:
            center = obj.get("bbox_center", [])
            size = obj.get("bbox_size", [])
            if len(center) < 2 or len(size) < 2:
                continue
            center_x, center_y = float(center[0]), float(center[1])
            box_width, box_height = float(size[0]), float(size[1])
            x1 = int(max(0, min(width - 1, center_x - box_width * 0.5)))
            y1 = int(max(0, min(height - 1, center_y - box_height * 0.5)))
            x2 = int(max(0, min(width - 1, center_x + box_width * 0.5)))
            y2 = int(max(0, min(height - 1, center_y + box_height * 0.5)))
            color = self._object_color(obj)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(image, (int(center_x), int(center_y)), 3, color, -1)
            label = "{} {}".format(
                obj.get("name", "tray"), obj.get("level", "")
            )
            cv2.putText(
                image,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    def _refresh_image(self):
        if self._latest_image is None:
            if self._image_error:
                self._image_label.setText(self._image_error)
            return
        frame = self._latest_image.copy()
        self._draw_objects(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(pixmap)

    def _refresh_table(self):
        self._table.setRowCount(len(self._objects))
        for row, obj in enumerate(self._objects):
            position = list(obj.get("position", [0.0, 0.0, 0.0]))
            size = list(obj.get("bbox_size", [0.0, 0.0]))
            values = [
                str(obj.get("name", "")),
                str(obj.get("level", "")),
                str(obj.get("class_name", "")),
                "{:.3f}".format(float(position[0])),
                "{:.3f}".format(float(position[1])),
                "{:.3f}".format(float(position[2])),
                "{:.1f}".format(float(size[0])),
                "{:.1f}".format(float(size[1])),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, column, item)
        self._table.resizeColumnsToContents()

    def _refresh(self):
        self._refresh_image()
        self._refresh_table()

    def shutdown_plugin(self):
        self._timer.stop()
        self._objects_sub.unregister()
        self._image_sub.unregister()
