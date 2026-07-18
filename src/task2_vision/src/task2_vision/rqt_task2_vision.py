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


class Task2VisionPlugin(Plugin):
    def __init__(self, context):
        super(Task2VisionPlugin, self).__init__(context)
        self.setObjectName("Task2VisionPlugin")
        self._vision_mode = None
        self._motion_object_order = list(OBJECT_NAMES)
        self._objects = []
        self._fit_objects = {}
        self._fit_motion_name_by_ground_truth = {}
        self._fit_locked = False
        self._valid_frame_count = 0
        self._required_valid_frames = 5
        self._base_drift = [0.0, 0.0, 0.0]
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
            "/task2_vision/head_image_topic",
            "/cam_h/color/image_raw/compressed",
        )

        self._widget = QWidget()
        self._widget.setWindowTitle("Task 2 Vision - RGB-D/CAD Pose")
        layout = QVBoxLayout(self._widget)

        self._status = QLabel("Waiting for /task2_vision/objects_json")
        layout.addWidget(self._status)

        self._image_label = QLabel("Waiting for {}".format(self._image_topic))
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(640, 360)
        self._image_label.setStyleSheet("background-color: #111; color: #ddd;")
        layout.addWidget(self._image_label)

        self._table = QTableWidget(0, 32)
        self._table.setHorizontalHeaderLabels(
            [
                "motion",
                "class",
                "truth x",
                "truth y",
                "truth z",
                "truth yaw deg",
                "motion x",
                "motion y",
                "motion z",
                "motion yaw deg",
                "dx mm",
                "dy mm",
                "dz mm",
                "dyaw deg",
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
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)
        context.add_widget(self._widget)

        self._objects_sub = rospy.Subscriber(
            "/task2_vision/objects_json",
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
        self._fit_sub = rospy.Subscriber(
            "/task2_depth_xy/objects_json",
            String,
            self._fit_callback,
            queue_size=1,
        )
        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

    @staticmethod
    def _motion_name(item):
        return str(item.get("motion_name") or item.get("name", ""))

    def _ordered_motion_objects(self, objects):
        by_name = {}
        extras = []
        for source in objects:
            item = dict(source)
            motion_name = self._motion_name(item)
            item["name"] = motion_name
            item["motion_name"] = motion_name
            if motion_name in by_name:
                continue
            by_name[motion_name] = item
            if motion_name not in self._motion_object_order:
                extras.append(motion_name)
        return [
            by_name[name]
            for name in self._motion_object_order + extras
            if name in by_name
        ]

    def _apply_fit_naming_to_truth_objects(self, objects):
        remapped = []
        for source in objects:
            item = dict(source)
            ground_truth_name = str(item.get("ground_truth_name", ""))
            motion_name = self._fit_motion_name_by_ground_truth.get(
                ground_truth_name
            )
            if motion_name:
                item["name"] = motion_name
                item["motion_name"] = motion_name
            remapped.append(item)
        return remapped

    def _objects_callback(self, message):
        try:
            payload = json.loads(message.data)
            new_mode = int(payload.get("vision_mode", 0))
            if self._vision_mode is not None and new_mode != self._vision_mode:
                self._fit_objects = {}
                self._fit_motion_name_by_ground_truth = {}
                self._fit_locked = False
                self._valid_frame_count = 0
            self._vision_mode = new_mode
            payload_order = [
                str(name) for name in payload.get("motion_object_order", [])
            ]
            if payload_order and len(set(payload_order)) == len(payload_order):
                self._motion_object_order = payload_order
            truth_objects = self._apply_fit_naming_to_truth_objects(
                payload.get("objects", [])
            )
            if new_mode != 2 or not self._fit_objects:
                self._objects = self._ordered_motion_objects(truth_objects)
            drift = list(payload.get("base_translation_drift", []))
            if len(drift) >= 3:
                self._base_drift = [float(value) for value in drift[:3]]
            if self._vision_mode == 0:
                self._fit_objects = {}
                self._fit_motion_name_by_ground_truth = {}
                self._fit_locked = True
                self._valid_frame_count = 0
            projected = sum(
                1 for obj in self._objects if obj.get("bbox_projected", False)
            )
            self._update_status(projected, payload.get("camera_frame_id", ""))
        except Exception as exc:
            self._status.setText("Invalid objects JSON: {}".format(exc))

    def _fit_callback(self, message):
        try:
            payload = json.loads(message.data)
            fit_mode = int(payload.get("vision_mode", 1))
            if fit_mode not in (1, 2):
                return
            fit_items = [dict(item) for item in payload.get("objects", [])]
            self._fit_objects = {
                self._motion_name(item): item
                for item in fit_items
            }
            if fit_mode == 2:
                self._fit_motion_name_by_ground_truth = {}
                self._objects = self._ordered_motion_objects(fit_items)
            elif payload.get("instance_naming_method") == "yaw_prior":
                self._fit_motion_name_by_ground_truth = {
                    str(item.get("ground_truth_name", "")): self._motion_name(item)
                    for item in fit_items
                }
                self._objects = self._ordered_motion_objects(
                    self._apply_fit_naming_to_truth_objects(self._objects)
                )
            elif payload.get("instance_naming_method") == "relative_position":
                self._fit_motion_name_by_ground_truth = {}
            self._fit_locked = bool(payload.get("locked", False))
            self._valid_frame_count = int(payload.get("selection_sample_count", 0))
            self._required_valid_frames = int(
                payload.get("required_valid_frames", 5)
            )
            projected = sum(
                1 for obj in self._objects if obj.get("bbox_projected", False)
            )
            self._update_status(projected, payload.get("camera_frame_id", ""))
        except Exception as exc:
            self._status.setText("Invalid depth-fit JSON: {}".format(exc))

    def _update_status(self, projected, camera_frame):
        drift_text = "Base drift: {:+.1f}/{:+.1f}/{:+.1f} mm".format(
            1000.0 * self._base_drift[0],
            1000.0 * self._base_drift[1],
            1000.0 * self._base_drift[2],
        )
        if self._vision_mode == 0:
            self._status.setText(
                "Mode 0 full truth | Qpos boxes: {}/{} | motion ready | {} | {}".format(
                    projected,
                    len(self._objects),
                    camera_frame or "camera unavailable",
                    drift_text,
                )
            )
            return
        state = "locked" if self._fit_locked else "waiting"
        mode_label = "YOLO+RGB-D" if self._vision_mode == 2 else "Qpos+RGB-D"
        fixed_handoff_roi = any(
            obj.get("bbox_source") == "handoff_fixed_roi" for obj in self._objects
        )
        box_label = (
            "ROIs"
            if fixed_handoff_roi
            else ("YOLO boxes" if self._vision_mode == 2 else "Qpos boxes")
        )
        self._status.setText(
            "Mode {} {} pose | {}: {}/{} | Valid frames: "
            "{}/{} {} | {} | {}".format(
                self._vision_mode,
                mode_label,
                box_label,
                projected,
                len(self._objects),
                self._valid_frame_count,
                self._required_valid_frames,
                state,
                camera_frame or "camera unavailable",
                drift_text,
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
        for obj in self._objects:
            if not obj.get("bbox_projected", False):
                continue
            if obj.get("bbox_source") == "handoff_fixed_roi":
                continue
            center = obj.get("bbox_center", [])
            size = obj.get("bbox_size", [])
            if len(center) < 2 or len(size) < 2:
                continue
            cx, cy = float(center[0]), float(center[1])
            box_width, box_height = float(size[0]), float(size[1])
            x1 = int(max(0, min(width - 1, cx - 0.5 * box_width)))
            y1 = int(max(0, min(height - 1, cy - 0.5 * box_height)))
            x2 = int(max(0, min(width - 1, cx + 0.5 * box_width)))
            y2 = int(max(0, min(height - 1, cy + 0.5 * box_height)))
            class_name = str(obj.get("class_name", "unknown"))
            color = CLASS_COLORS.get(class_name, (230, 230, 230))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(round(cx)), int(round(cy))), 3, color, -1)
            fit = (
                self._fit_objects.get(self._motion_name(obj))
                if self._vision_mode in (1, 2)
                else None
            )
            if fit is not None:
                fit_pixel = fit.get("fit_pixel_center", [])
                if len(fit_pixel) >= 2:
                    fit_x = int(round(float(fit_pixel[0])))
                    fit_y = int(round(float(fit_pixel[1])))
                    cv2.drawMarker(
                        frame,
                        (fit_x, fit_y),
                        (255, 255, 255),
                        cv2.MARKER_CROSS,
                        12,
                        2,
                        cv2.LINE_AA,
                    )
            label = CLASS_LABELS.get(class_name, "?")
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_top = y1
            label_bottom = min(y2, y1 + label_height + baseline + 6)
            cv2.rectangle(
                frame,
                (x1, label_top),
                (min(x2, x1 + label_width + 8), label_bottom),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 4, min(y2 - baseline, y1 + label_height + 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

    def _draw_handoff_fixed_roi(self, frame):
        if self._vision_mode != 2:
            return
        center_x, center_y, roi_width, roi_height = self._handoff_fixed_roi
        height, width = frame.shape[:2]
        x1 = int(round((center_x - 0.5 * roi_width) * width))
        y1 = int(round((center_y - 0.5 * roi_height) * height))
        x2 = int(round((center_x + 0.5 * roi_width) * width))
        y2 = int(round((center_y + 0.5 * roi_height) * height))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
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
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(pixmap)

    def _refresh_table(self):
        self._table.setRowCount(len(self._objects))
        for row, obj in enumerate(self._objects):
            live_truth_position = list(
                obj.get("position", [0.0, 0.0, 0.0])
            )
            if self._vision_mode == 0:
                fit = {}
                reference_position = live_truth_position
                motion_position = live_truth_position
                reference_yaw = float(obj.get("yaw", 0.0))
                motion_yaw = reference_yaw
                points = "n/a"
                render_depth = "n/a"
                render_coverage = "n/a"
                render_observed_coverage = "n/a"
                render_iou = "n/a"
                alignment_iterations = "n/a"
                alignment_depth_inliers = "n/a"
                render_score = "n/a"
                yaw_data_score = "n/a"
                yaw_prior_penalty = "n/a"
                yaw_prior = "n/a"
                yaw_assignment = "n/a"
                yaw_direct_cost = "n/a"
                yaw_swapped_cost = "n/a"
                yaw_assignment_margin = "n/a"
                yaw_window_edge = "n/a"
                frames = "truth"
                fit_status = "full_truth"
            else:
                fit = self._fit_objects.get(self._motion_name(obj), {})
                reference_position = list(
                    fit.get("truth_position", live_truth_position)
                )
                reference_yaw = float(
                    fit.get("truth_yaw", obj.get("yaw", 0.0))
                )
                motion_position = list(fit.get("position", []))
                if len(motion_position) < 2:
                    motion_position = list(fit.get("fit_position", []))
                motion_yaw = fit.get("fit_yaw")
                points = str(fit.get("fit_points", "n/a"))
                render_depth = (
                    "{:.2f}".format(
                        1000.0 * float(fit["fit_render_depth_error_m"])
                    )
                    if "fit_render_depth_error_m" in fit
                    else "n/a"
                )
                render_coverage = (
                    "{:.3f}".format(float(fit["fit_render_coverage"]))
                    if "fit_render_coverage" in fit
                    else "n/a"
                )
                render_observed_coverage = (
                    "{:.3f}".format(
                        float(fit["fit_render_observed_coverage"])
                    )
                    if "fit_render_observed_coverage" in fit
                    else "n/a"
                )
                render_iou = (
                    "{:.3f}".format(float(fit["fit_render_iou"]))
                    if "fit_render_iou" in fit
                    else "n/a"
                )
                alignment_iterations = str(
                    fit.get("fit_alignment_iterations", "n/a")
                )
                alignment_depth_inliers = str(
                    fit.get("fit_alignment_depth_inliers", "n/a")
                )
                render_score = (
                    "{:.3f}".format(float(fit["fit_render_score"]))
                    if "fit_render_score" in fit
                    else "n/a"
                )
                yaw_data_score = (
                    "{:.3f}".format(float(fit["fit_yaw_data_score"]))
                    if fit.get("fit_yaw_data_score") is not None
                    else "n/a"
                )
                yaw_prior_penalty = (
                    "{:.3f}".format(float(fit["fit_yaw_prior_penalty"]))
                    if fit.get("fit_yaw_prior_penalty") is not None
                    else "n/a"
                )
                yaw_prior = str(
                    fit.get("fit_yaw_prior_selected_name", "n/a") or "n/a"
                )
                yaw_assignment = str(
                    fit.get("fit_yaw_joint_assignment_mode", "n/a") or "n/a"
                )
                yaw_direct_cost = (
                    "{:.3f}".format(float(fit["fit_yaw_joint_direct_cost"]))
                    if fit.get("fit_yaw_joint_direct_cost") is not None
                    else "n/a"
                )
                yaw_swapped_cost = (
                    "{:.3f}".format(float(fit["fit_yaw_joint_swapped_cost"]))
                    if fit.get("fit_yaw_joint_swapped_cost") is not None
                    else "n/a"
                )
                yaw_assignment_margin = (
                    "{:.3f}".format(
                        float(fit["fit_yaw_joint_assignment_margin"])
                    )
                    if fit.get("fit_yaw_joint_assignment_margin") is not None
                    else "n/a"
                )
                yaw_window_edge = (
                    "{:.1f}".format(
                        float(fit["fit_yaw_window_edge_distance_deg"])
                    )
                    if fit.get("fit_yaw_window_edge_distance_deg") is not None
                    else "n/a"
                )
                frames = "{}/{}".format(
                    fit.get("fit_sample_count", self._valid_frame_count),
                    fit.get("fit_required_samples", self._required_valid_frames),
                )
                rejection_reason = str(
                    fit.get("fit_alignment_rejection_reason", "")
                )
                explicit_status = str(fit.get("fit_status", ""))
                if explicit_status:
                    fit_status = explicit_status
                elif rejection_reason:
                    fit_status = rejection_reason
                else:
                    fit_status = "locked" if self._fit_locked else "waiting"
            has_position_fit = (
                len(reference_position) >= 3 and len(motion_position) >= 3
            )
            if has_position_fit:
                position_error = [
                    float(motion_position[index])
                    - float(reference_position[index])
                    for index in range(3)
                ]
            else:
                position_error = []
            has_yaw_fit = motion_yaw is not None
            if has_yaw_fit:
                yaw_error = (
                    float(motion_yaw) - reference_yaw + np.pi
                ) % (2.0 * np.pi) - np.pi
            else:
                yaw_error = 0.0
            values = [
                self._motion_name(obj),
                str(obj.get("class_name", "")),
                "{:.3f}".format(float(reference_position[0])),
                "{:.3f}".format(float(reference_position[1])),
                "{:.3f}".format(float(reference_position[2]))
                if len(reference_position) >= 3
                else "n/a",
                "{:.1f}".format(np.degrees(reference_yaw)),
                "{:.3f}".format(float(motion_position[0]))
                if has_position_fit
                else "n/a",
                "{:.3f}".format(float(motion_position[1]))
                if has_position_fit
                else "n/a",
                "{:.3f}".format(float(motion_position[2]))
                if has_position_fit
                else "n/a",
                "{:.1f}".format(np.degrees(float(motion_yaw)))
                if has_yaw_fit
                else "n/a",
                "{:+.1f}".format(1000.0 * float(position_error[0]))
                if has_position_fit
                else "n/a",
                "{:+.1f}".format(1000.0 * float(position_error[1]))
                if has_position_fit
                else "n/a",
                "{:+.1f}".format(1000.0 * float(position_error[2]))
                if has_position_fit
                else "n/a",
                "{:+.1f}".format(np.degrees(yaw_error))
                if has_yaw_fit
                else "n/a",
                points,
                render_depth,
                render_coverage,
                render_observed_coverage,
                render_iou,
                alignment_iterations,
                alignment_depth_inliers,
                render_score,
                yaw_data_score,
                yaw_prior_penalty,
                yaw_prior,
                yaw_assignment,
                yaw_direct_cost,
                yaw_swapped_cost,
                yaw_assignment_margin,
                yaw_window_edge,
                frames,
                fit_status,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, column, item)
        self._table.resizeColumnsToContents()

    def shutdown_plugin(self):
        self._timer.stop()
        self._objects_sub.unregister()
        self._image_sub.unregister()
        self._fit_sub.unregister()
