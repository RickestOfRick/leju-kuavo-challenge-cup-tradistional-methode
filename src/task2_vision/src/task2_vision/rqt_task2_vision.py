#!/usr/bin/env python3

import json

import cv2
import numpy as np
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

from task2_vision.layout import OBJECT_NAMES


CLASS_COLORS = {
    "part_type_a": (58, 201, 92),
    "part_type_b": (230, 155, 54),
    "part_type_c": (72, 82, 235),
}
CLASS_LABELS = {
    "part_type_a": "A",
    "part_type_b": "B",
    "part_type_c": "C",
}


def _number(item, key, digits=3, scale=1.0, missing="n/a"):
    value = item.get(key)
    if value is None:
        return missing
    try:
        value = float(value) * float(scale)
    except (TypeError, ValueError):
        return missing
    if not np.isfinite(value):
        return missing
    return ("{:.%df}" % int(digits)).format(value)


class Task2VisionPlugin(Plugin):
    def __init__(self, context):
        super(Task2VisionPlugin, self).__init__(context)
        self.setObjectName("Task2VisionPlugin")
        self._motion_object_order = list(OBJECT_NAMES)
        self._objects = []
        self._fit_locked = False
        self._valid_frame_count = 0
        self._required_valid_frames = 5
        self._latest_image = None
        self._image_error = ""
        self._handoff_fixed_roi = [
            float(value)
            for value in rospy.get_param(
                "/task2_depth_xy/handoff_relocalization_fixed_roi",
                [0.4166666667, 0.7425, 0.1666666667, 0.205],
            )
        ]
        if len(self._handoff_fixed_roi) != 4:
            self._handoff_fixed_roi = [
                0.4166666667,
                0.7425,
                0.1666666667,
                0.205,
            ]
        self._image_topic = rospy.get_param(
            "/task2_depth_xy/color_image_topic",
            "/cam_h/color/image_raw/compressed",
        )

        self._widget = QWidget()
        self._widget.setWindowTitle("Task 2 Vision - RGB-D/CAD Pose")
        layout = QVBoxLayout(self._widget)
        self._status = QLabel("Waiting for /task2_depth_xy/objects_json")
        layout.addWidget(self._status)

        self._image_label = QLabel("Waiting for {}".format(self._image_topic))
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(640, 360)
        self._image_label.setStyleSheet("background-color: #111; color: #ddd;")
        layout.addWidget(self._image_label)

        headers = [
            "motion",
            "class",
            "x",
            "y",
            "z",
            "yaw deg",
            "points",
            "depth mm",
            "CAD cov",
            "obs cov",
            "IoU",
            "GN iter",
            "depth pts",
            "score",
            "yaw data",
            "prior pen",
            "yaw prior",
            "assign",
            "direct",
            "swap",
            "margin",
            "edge deg",
            "frames",
            "status",
        ]
        self._table = QTableWidget(0, len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)
        context.add_widget(self._widget)

        self._fit_sub = rospy.Subscriber(
            "/task2_depth_xy/objects_json",
            String,
            self._fit_callback,
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
        self._timer.start(100)

    @staticmethod
    def _motion_name(item):
        return str(item.get("motion_name") or item.get("name", ""))

    def _ordered_objects(self, objects):
        by_name = {}
        extras = []
        for source in objects:
            item = dict(source)
            name = self._motion_name(item)
            if not name or name in by_name:
                continue
            item["name"] = name
            item["motion_name"] = name
            by_name[name] = item
            if name not in self._motion_object_order:
                extras.append(name)
        return [
            by_name[name]
            for name in self._motion_object_order + extras
            if name in by_name
        ]

    def _fit_callback(self, message):
        try:
            payload = json.loads(message.data)
            payload_order = [
                str(name) for name in payload.get("motion_object_order", [])
            ]
            if payload_order and len(set(payload_order)) == len(payload_order):
                self._motion_object_order = payload_order
            self._objects = self._ordered_objects(payload.get("objects", []))
            self._fit_locked = bool(payload.get("locked", False))
            self._valid_frame_count = int(payload.get("selection_sample_count", 0))
            self._required_valid_frames = int(
                payload.get("required_valid_frames", 5)
            )
            projected = sum(
                1 for item in self._objects if item.get("bbox_projected", False)
            )
            self._update_status(projected, payload.get("camera_frame_id", ""))
        except Exception as exc:
            self._status.setText("Invalid pose JSON: {}".format(exc))

    def _update_status(self, projected, camera_frame):
        state = "locked" if self._fit_locked else "waiting"
        fixed_roi = any(
            item.get("bbox_source") == "handoff_fixed_roi"
            for item in self._objects
        )
        roi_label = "handoff ROI" if fixed_roi else "object ROIs"
        self._status.setText(
            "YOLO+RGB-D/CAD | {}: {}/{} | valid frames: {}/{} {} | {}".format(
                roi_label,
                projected,
                len(self._objects),
                self._valid_frame_count,
                self._required_valid_frames,
                state,
                camera_frame or "camera unavailable",
            )
        )

    def _image_callback(self, message):
        try:
            encoded = np.frombuffer(message.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("compressed image decode failed")
            self._latest_image = image
            self._image_error = ""
        except Exception as exc:
            self._image_error = str(exc)

    def _draw_boxes(self, frame):
        height, width = frame.shape[:2]
        for item in self._objects:
            if not item.get("bbox_projected", False):
                continue
            center = item.get("bbox_center", [])
            size = item.get("bbox_size", [])
            if len(center) < 2 or len(size) < 2:
                continue
            cx, cy = float(center[0]), float(center[1])
            box_width, box_height = float(size[0]), float(size[1])
            x1 = int(max(0, min(width - 1, cx - 0.5 * box_width)))
            y1 = int(max(0, min(height - 1, cy - 0.5 * box_height)))
            x2 = int(max(0, min(width - 1, cx + 0.5 * box_width)))
            y2 = int(max(0, min(height - 1, cy + 0.5 * box_height)))
            class_name = str(item.get("class_name", ""))
            color = CLASS_COLORS.get(class_name, (230, 230, 230))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            fit_pixel = item.get("fit_pixel_center", [])
            if len(fit_pixel) >= 2:
                cv2.drawMarker(
                    frame,
                    (int(round(float(fit_pixel[0]))), int(round(float(fit_pixel[1])))),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    12,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                CLASS_LABELS.get(class_name, "?"),
                (x1 + 4, min(y2 - 4, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_handoff_fixed_roi(self, frame):
        center_x, center_y, roi_width, roi_height = self._handoff_fixed_roi
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(round((center_x - 0.5 * roi_width) * width))))
        y1 = max(0, min(height - 1, int(round((center_y - 0.5 * roi_height) * height))))
        x2 = max(0, min(width - 1, int(round((center_x + 0.5 * roi_width) * width))))
        y2 = max(0, min(height - 1, int(round((center_y + 0.5 * roi_height) * height))))
        color = (255, 220, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            "HANDOFF ROI",
            (x1 + 4, min(y2 - 4, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def _refresh(self):
        self._refresh_image()
        self._refresh_table()

    def _refresh_image(self):
        if self._latest_image is None:
            if self._image_error:
                self._image_label.setText("Image error: {}".format(self._image_error))
            return
        frame = self._latest_image.copy()
        self._draw_boxes(frame)
        self._draw_handoff_fixed_roi(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(pixmap)

    def _refresh_table(self):
        self._table.setRowCount(len(self._objects))
        for row, item in enumerate(self._objects):
            position = list(item.get("position", []))
            if len(position) < 3:
                position = list(item.get("fit_position", []))
            yaw = item.get("fit_yaw", item.get("yaw"))
            frames = "{}/{}".format(
                item.get("fit_sample_count", self._valid_frame_count),
                item.get("fit_required_samples", self._required_valid_frames),
            )
            rejection = str(item.get("fit_alignment_rejection_reason", ""))
            status = str(item.get("fit_status", ""))
            if not status:
                status = rejection or ("locked" if self._fit_locked else "waiting")
            values = [
                self._motion_name(item),
                str(item.get("class_name", "")),
                "{:.3f}".format(float(position[0])) if len(position) >= 3 else "n/a",
                "{:.3f}".format(float(position[1])) if len(position) >= 3 else "n/a",
                "{:.3f}".format(float(position[2])) if len(position) >= 3 else "n/a",
                "{:.1f}".format(np.degrees(float(yaw)))
                if yaw is not None and np.isfinite(float(yaw))
                else "n/a",
                str(item.get("fit_points", "n/a")),
                _number(item, "fit_render_depth_error_m", 2, 1000.0),
                _number(item, "fit_render_coverage"),
                _number(item, "fit_render_observed_coverage"),
                _number(item, "fit_render_iou"),
                str(item.get("fit_alignment_iterations", "n/a")),
                str(item.get("fit_alignment_depth_inliers", "n/a")),
                _number(item, "fit_render_score"),
                _number(item, "fit_yaw_data_score"),
                _number(item, "fit_yaw_prior_penalty"),
                str(item.get("fit_yaw_prior_selected_name", "n/a") or "n/a"),
                str(item.get("fit_yaw_joint_assignment_mode", "n/a") or "n/a"),
                _number(item, "fit_yaw_joint_direct_cost"),
                _number(item, "fit_yaw_joint_swapped_cost"),
                _number(item, "fit_yaw_joint_assignment_margin"),
                _number(item, "fit_yaw_window_edge_distance_deg", 1),
                frames,
                status,
            ]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, column, cell)
        self._table.resizeColumnsToContents()

    def shutdown_plugin(self):
        self._timer.stop()
        self._image_sub.unregister()
        self._fit_sub.unregister()
