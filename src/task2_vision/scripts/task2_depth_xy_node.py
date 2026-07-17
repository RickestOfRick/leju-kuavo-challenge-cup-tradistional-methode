#!/usr/bin/env python3
"""Fit task 2 object XYZ and yaw from qpos ROIs and RGB-D."""

import copy
import json
import math
import os
import struct
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String
import tf2_ros

from task2_vision.cad_depth_alignment import CadDepthAligner
from task2_vision.color_segmentation import green_table_mask
from task2_vision.layout import (
    CLASS_NAMES,
    INSTANCE_NAMING_RELATIVE_POSITION,
    INSTANCE_NAMING_YAW_PRIOR,
    OBJECT_NAMES,
    assign_motion_names,
    joint_yaw_prior_assignment,
    joint_yaw_window_assignment,
    normalize_instance_naming_method,
    yaw_from_quat_wxyz,
    yaw_prior_candidates,
    yaw_prior_regularized_score,
    yaw_prior_window_candidates,
)
from task2_vision.mesh_geometry import (
    load_body_visual_triangles,
    load_body_visual_triangles_by_geom,
    load_body_visual_vertices,
)
from task2_vision.pose_estimation import (
    angle_difference,
    circular_median,
    orientation_xyzw_from_yaw,
    quaternion_xyzw_from_rotation,
    rotation_from_yaw,
    wrap_angle,
    yaw_from_rotation,
    yaw_orientation_xyzw,
)
from task2_vision.reference_frame import (
    transform_point_between_base_payloads,
    transform_rotation_between_base_payloads,
)
from task2_vision.rest_pose import body_origin_z_from_table


FRAME_ID = "base_link"
POSE_SOURCE = "qpos_bbox_region_depth_xyz_yaw_fit"
Z_POSITION_SOURCE = "table_plane_class_rest_pose"


def _decode_compressed_depth(message):
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError("empty compressed depth message")
    offsets = [12, 16, 0] if "compressedDepth" in message.format else [0, 12, 16]
    image = None
    for offset in offsets:
        if raw.size <= offset:
            continue
        image = cv2.imdecode(raw[offset:], cv2.IMREAD_UNCHANGED)
        if image is not None:
            break
    if image is None:
        raise ValueError("compressed depth PNG decode failed")
    if image.ndim == 3:
        image = image[:, :, 0]

    if "32FC1" in message.format and image.dtype == np.uint16:
        if len(message.data) < 12:
            raise ValueError("32FC1 compressedDepth header is incomplete")
        depth_quant_a, depth_quant_b = struct.unpack("<ff", message.data[4:12])
        raw_depth = image.astype(np.float32)
        depth = np.zeros(raw_depth.shape, dtype=np.float32)
        valid = raw_depth > 0.0
        denominator = raw_depth[valid] - float(depth_quant_b)
        denominator_valid = np.abs(denominator) > 1e-6
        values = np.zeros_like(denominator, dtype=np.float32)
        values[denominator_valid] = float(depth_quant_a) / denominator[
            denominator_valid
        ]
        depth[valid] = values
        return depth
    if image.dtype == np.uint16:
        return image.astype(np.float32) * 0.001
    return image.astype(np.float32)


def _decode_compressed_color(message):
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError("empty compressed color message")
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("compressed color decode failed")
    return image


def _quaternion_xyzw_to_matrix(quaternion):
    x, y, z, w = [float(value) for value in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_to_matrix(transform):
    rotation = _quaternion_xyzw_to_matrix(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    )
    translation = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=np.float64,
    )
    return rotation, translation


def _scaled_intrinsics(camera_info, image_shape):
    height, width = image_shape[:2]
    info_width = float(camera_info.width or width)
    info_height = float(camera_info.height or height)
    scale_x = float(width) / info_width
    scale_y = float(height) / info_height
    return (
        float(camera_info.K[0]) * scale_x,
        float(camera_info.K[4]) * scale_y,
        float(camera_info.K[2]) * scale_x,
        float(camera_info.K[5]) * scale_y,
        scale_x,
        scale_y,
    )


def _pixels_to_base(depth_values, pixels_u, pixels_v, intrinsics, rotation, translation):
    fx, fy, cx, cy = intrinsics[:4]
    camera_points = np.column_stack(
        [
            (pixels_u - cx) * depth_values / fx,
            (pixels_v - cy) * depth_values / fy,
            depth_values,
        ]
    )
    return camera_points.dot(rotation.T) + translation


def _fit_horizontal_plane_ransac(
    points,
    iterations,
    distance_threshold,
    max_tilt_deg,
    histogram_bin_size,
    random_seed,
):
    if points.shape[0] < 100:
        return None

    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    bin_count = max(1, int(math.ceil((z_max - z_min) / histogram_bin_size)))
    histogram, edges = np.histogram(points[:, 2], bins=bin_count)
    dominant_index = int(np.argmax(histogram))
    dominant_z = 0.5 * (edges[dominant_index] + edges[dominant_index + 1])
    candidates = points[np.abs(points[:, 2] - dominant_z) <= 0.035]
    if candidates.shape[0] < 100:
        candidates = points

    max_points = 12000
    if candidates.shape[0] > max_points:
        step = int(math.ceil(float(candidates.shape[0]) / float(max_points)))
        candidates = candidates[::step]

    rng = np.random.RandomState(int(random_seed))
    max_slope = math.tan(math.radians(float(max_tilt_deg)))
    best = None
    design_all = np.column_stack(
        [candidates[:, 0], candidates[:, 1], np.ones(candidates.shape[0])]
    )
    for _ in range(max(1, int(iterations))):
        indices = rng.choice(candidates.shape[0], 3, replace=False)
        design = design_all[indices]
        if abs(float(np.linalg.det(design))) <= 1e-10:
            continue
        coefficients = np.linalg.solve(design, candidates[indices, 2])
        if math.hypot(float(coefficients[0]), float(coefficients[1])) > max_slope:
            continue
        residuals = np.abs(candidates[:, 2] - design_all.dot(coefficients))
        inlier_mask = residuals <= distance_threshold
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < 100:
            continue
        rmse = math.sqrt(float(np.mean(residuals[inlier_mask] ** 2)))
        if best is None or (inlier_count, -rmse) > (best[0], -best[1]):
            best = (inlier_count, rmse, inlier_mask)
    if best is None:
        return None

    inlier_points = candidates[best[2]]
    design = np.column_stack(
        [
            inlier_points[:, 0],
            inlier_points[:, 1],
            np.ones(inlier_points.shape[0]),
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, inlier_points[:, 2], rcond=None)
    residuals = inlier_points[:, 2] - design.dot(coefficients)
    return {
        "coefficients": [float(value) for value in coefficients],
        "inliers": int(inlier_points.shape[0]),
        "rmse": math.sqrt(float(np.mean(residuals * residuals))),
        "height_at_origin": float(coefficients[2]),
    }


class Task2DepthXYNode:
    def __init__(self):
        self.vision_mode = int(rospy.get_param("~vision_mode", 1))
        if self.vision_mode != 1:
            raise RuntimeError("task2_depth_xy only runs in task2 vision mode 1")
        self.instance_naming_method = normalize_instance_naming_method(
            rospy.get_param("~instance_naming_method", "relative_position")
        )
        self.truth_topic = str(
            rospy.get_param("~truth_objects_topic", "/task2_vision/objects_json")
        )
        self.mode0_reference_topic = str(
            rospy.get_param(
                "~mode0_reference_topic",
                "/task2_vision/mode0_reference_json",
            )
        )
        self.depth_topic = str(
            rospy.get_param(
                "~depth_image_topic", "/cam_h/depth/image_raw/compressedDepth"
            )
        )
        self.camera_info_topic = str(
            rospy.get_param("~camera_info_topic", "/cam_h/color/camera_info")
        )
        self.color_topic = str(
            rospy.get_param(
                "~color_image_topic", "/cam_h/color/image_raw/compressed"
            )
        )
        self.publish_hz = max(
            0.1, float(rospy.get_param("~depth_xy_publish_hz", 2.0))
        )
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.02))
        self.depth_min = float(rospy.get_param("~depth_min_m", 0.15))
        self.depth_max = float(rospy.get_param("~depth_max_m", 3.0))
        self.green_filter_enabled = bool(
            rospy.get_param("~green_table_filter_enabled", True)
        )
        self.green_hsv_lower = [
            int(value)
            for value in rospy.get_param("~green_table_hsv_lower", [35, 40, 10])
        ]
        self.green_hsv_upper = [
            int(value)
            for value in rospy.get_param("~green_table_hsv_upper", [90, 255, 255])
        ]
        if len(self.green_hsv_lower) != 3 or len(self.green_hsv_upper) != 3:
            raise RuntimeError("green table HSV limits must each contain three values")
        self.green_dilate_pixels = max(
            0, int(rospy.get_param("~green_table_dilate_pixels", 1))
        )
        self.green_chroma_min = float(
            rospy.get_param("~green_table_chroma_min", 0.38)
        )
        self.green_chroma_dominance_min = float(
            rospy.get_param("~green_table_chroma_dominance_min", 0.04)
        )
        self.green_shadow_min_intensity_sum = float(
            rospy.get_param("~green_table_shadow_min_intensity_sum", 3.0)
        )
        self.color_depth_max_skew = max(
            0.0, float(rospy.get_param("~color_depth_max_skew_sec", 0.20))
        )
        self.table_stride = max(1, int(rospy.get_param("~table_fit_stride", 4)))
        self.table_x_range = [
            float(value)
            for value in rospy.get_param("~table_fit_x_range", [0.15, 0.52])
        ]
        self.table_y_range = [
            float(value)
            for value in rospy.get_param("~table_fit_y_range", [-0.50, 0.50])
        ]
        self.table_z_range = [
            float(value)
            for value in rospy.get_param("~table_fit_z_range", [-0.25, 0.05])
        ]
        self.table_ransac_iterations = max(
            1, int(rospy.get_param("~table_ransac_iterations", 240))
        )
        self.table_distance_threshold = float(
            rospy.get_param("~table_ransac_distance_threshold_m", 0.0035)
        )
        self.table_max_tilt_deg = float(
            rospy.get_param("~table_ransac_max_tilt_deg", 12.0)
        )
        self.table_histogram_bin_size = float(
            rospy.get_param("~table_histogram_bin_size_m", 0.004)
        )
        self.roi_scale = max(1.0, float(rospy.get_param("~depth_roi_scale", 1.03)))
        self.table_roi_exclusion_scale = max(
            1.0, float(rospy.get_param("~table_roi_exclusion_scale", 1.12))
        )
        self.object_min_height = float(
            rospy.get_param("~object_min_height_above_table_m", 0.0025)
        )
        self.object_max_height = float(
            rospy.get_param("~object_max_height_above_table_m", 0.15)
        )
        self.component_min_pixels = max(
            1, int(rospy.get_param("~object_component_min_pixels", 5))
        )
        self.fit_min_points = max(
            3, int(rospy.get_param("~object_fit_min_points", 25))
        )
        self.bound_percentiles = [
            float(value)
            for value in rospy.get_param("~object_bound_percentiles", [1.0, 99.0])
        ]
        self.cad_surface_samples = max(
            1000, int(rospy.get_param("~cad_gn_surface_samples", 5000))
        )
        self.cad_maximum_iterations = max(
            1, int(rospy.get_param("~cad_gn_maximum_iterations", 6))
        )
        self.cad_region_sigma_px = float(
            rospy.get_param("~cad_gn_region_sigma_px", 2.0)
        )
        self.cad_depth_sigma = float(
            rospy.get_param("~cad_gn_depth_sigma_m", 0.003)
        )
        self.cad_region_weight = float(
            rospy.get_param("~cad_gn_region_weight", 1.0)
        )
        self.cad_depth_weight = float(
            rospy.get_param("~cad_gn_depth_weight", 1.0)
        )
        self.cad_robust_delta = float(
            rospy.get_param("~cad_gn_robust_delta", 2.5)
        )
        self.cad_maximum_depth_residual = float(
            rospy.get_param("~cad_gn_maximum_depth_residual_m", 0.04)
        )
        self.cad_maximum_correspondence_pixels = float(
            rospy.get_param("~cad_gn_maximum_correspondence_pixels", 4.0)
        )
        self.cad_maximum_step = float(
            rospy.get_param("~cad_gn_maximum_step_m", 0.02)
        )
        self.cad_maximum_translation = float(
            rospy.get_param("~cad_gn_maximum_translation_m", 0.08)
        )
        self.cad_convergence_translation = float(
            rospy.get_param("~cad_gn_convergence_translation_m", 0.0002)
        )
        self.cad_damping = float(rospy.get_param("~cad_gn_damping", 0.001))
        self.cad_rendered_mask_dilate_pixels = max(
            0, int(rospy.get_param("~cad_gn_rendered_mask_dilate_pixels", 1))
        )
        self.cad_maximum_region_points = max(
            20, int(rospy.get_param("~cad_gn_maximum_region_points", 600))
        )
        self.cad_maximum_depth_points = max(
            20, int(rospy.get_param("~cad_gn_maximum_depth_points", 1600))
        )
        self.cad_minimum_overlap_ratio = float(
            rospy.get_param("~cad_gn_minimum_bidirectional_coverage", 0.45)
        )
        self.cad_minimum_iou = float(
            rospy.get_param("~cad_gn_minimum_iou", 0.30)
        )
        self.cad_minimum_overlap_by_class = {
            str(key): float(value)
            for key, value in rospy.get_param(
                "~cad_gn_minimum_bidirectional_coverage_by_class",
                {
                    "part_type_a": 0.25,
                    "part_type_b": 0.30,
                    "part_type_c": 0.45,
                },
            ).items()
        }
        self.cad_minimum_iou_by_class = {
            str(key): float(value)
            for key, value in rospy.get_param(
                "~cad_gn_minimum_iou_by_class",
                {
                    "part_type_a": 0.15,
                    "part_type_b": 0.18,
                    "part_type_c": 0.30,
                },
            ).items()
        }
        self.yaw_refine_top_k = max(
            1, int(rospy.get_param("~cad_yaw_refine_top_k", 1))
        )
        self.yaw_refinement_steps = [
            math.radians(abs(float(value)))
            for value in rospy.get_param(
                "~cad_yaw_refinement_steps_deg", [3.0, 1.0]
            )
            if abs(float(value)) > 0.0
        ]
        self.yaw_prior_window = math.radians(
            abs(float(rospy.get_param("~cad_yaw_prior_window_deg", 20.0)))
        )
        self.yaw_prior_window_step = math.radians(
            abs(float(rospy.get_param("~cad_yaw_prior_window_step_deg", 10.0)))
        )
        if self.yaw_prior_window <= 0.0 or self.yaw_prior_window_step <= 0.0:
            raise RuntimeError(
                "cad_yaw_prior_window_deg and step must be positive"
            )
        self.yaw_prior_regularization_weight = max(
            0.0,
            float(rospy.get_param("~cad_yaw_prior_regularization_weight", 0.1)),
        )
        yaw_prior_sigma_deg = rospy.get_param(
            "~cad_yaw_prior_sigma_deg_by_class",
            {
                "part_type_a": 12.0,
                "part_type_b": 8.0,
                "part_type_c": 12.0,
            },
        )
        self.yaw_prior_sigma_by_class = {
            class_name: math.radians(
                max(1e-3, abs(float(yaw_prior_sigma_deg.get(class_name, 12.0))))
            )
            for class_name in CLASS_NAMES
        }
        self.yaw_handle_score_weight = max(
            0.0, float(rospy.get_param("~cad_yaw_handle_score_weight", 1.0))
        )
        self.yaw_surface_samples = max(
            500, int(rospy.get_param("~cad_yaw_surface_samples", 1200))
        )
        self.yaw_xy_iterations = max(
            1, int(rospy.get_param("~cad_yaw_xy_iterations", 2))
        )
        self.c_handle_refinement_enabled = bool(
            rospy.get_param("~c_handle_refinement_enabled", True)
        )
        self.c_handle_hue_low_max = int(
            rospy.get_param("~c_handle_red_hue_low_max", 12)
        )
        self.c_handle_hue_high_min = int(
            rospy.get_param("~c_handle_red_hue_high_min", 170)
        )
        self.c_handle_saturation_min = int(
            rospy.get_param("~c_handle_red_saturation_min", 70)
        )
        self.c_handle_value_min = int(
            rospy.get_param("~c_handle_red_value_min", 18)
        )
        self.c_handle_minimum_pixels = max(
            5, int(rospy.get_param("~c_handle_minimum_pixels", 20))
        )
        self.c_handle_minimum_overlap = float(
            rospy.get_param("~c_handle_minimum_bidirectional_coverage", 0.25)
        )
        self.c_handle_minimum_iou = float(
            rospy.get_param("~c_handle_minimum_iou", 0.15)
        )
        self.c_handle_surface_samples = max(
            500, int(rospy.get_param("~c_handle_surface_samples", 2000))
        )
        self.c_handle_maximum_iterations = max(
            1, int(rospy.get_param("~c_handle_maximum_iterations", 6))
        )
        self.median_frames = max(
            1, int(rospy.get_param("~pose_median_frames", 5))
        )
        self.random_seed = int(rospy.get_param("~depth_fit_random_seed", 42))

        model_path = os.path.abspath(
            str(rospy.get_param("legged_robot_scene_param", ""))
        )
        if not model_path or not os.path.isfile(model_path):
            raise RuntimeError("MuJoCo scene XML is unavailable for CAD pose fitting")
        self.model_vertices = load_body_visual_vertices(model_path, OBJECT_NAMES)
        self.model_triangles = load_body_visual_triangles(model_path, OBJECT_NAMES)
        self.model_triangles_by_geom = load_body_visual_triangles_by_geom(
            model_path, OBJECT_NAMES
        )
        missing_models = [name for name in OBJECT_NAMES if name not in self.model_vertices]
        missing_models.extend(
            name for name in OBJECT_NAMES if name not in self.model_triangles
        )
        missing_models = sorted(set(missing_models))
        if missing_models:
            raise RuntimeError(
                "visual mesh vertices missing for {}".format(", ".join(missing_models))
            )
        self.cad_aligner = CadDepthAligner(
            self.model_triangles,
            sample_count=self.cad_surface_samples,
            random_seed=self.random_seed,
            maximum_iterations=self.cad_maximum_iterations,
            region_sigma_px=self.cad_region_sigma_px,
            depth_sigma=self.cad_depth_sigma,
            region_weight=self.cad_region_weight,
            depth_weight=self.cad_depth_weight,
            robust_delta=self.cad_robust_delta,
            maximum_depth_residual=self.cad_maximum_depth_residual,
            maximum_correspondence_pixels=self.cad_maximum_correspondence_pixels,
            maximum_step=self.cad_maximum_step,
            maximum_translation=self.cad_maximum_translation,
            convergence_translation=self.cad_convergence_translation,
            damping=self.cad_damping,
            rendered_mask_dilate_pixels=self.cad_rendered_mask_dilate_pixels,
            maximum_region_points=self.cad_maximum_region_points,
            maximum_depth_points=self.cad_maximum_depth_points,
            minimum_rendered_pixels=self.fit_min_points,
            minimum_overlap_ratio=self.cad_minimum_overlap_ratio,
            minimum_iou=self.cad_minimum_iou,
        )
        self.cad_yaw_aligner = CadDepthAligner(
            self.model_triangles,
            sample_count=self.yaw_surface_samples,
            random_seed=self.random_seed + 3571,
            maximum_iterations=self.yaw_xy_iterations,
            region_sigma_px=self.cad_region_sigma_px,
            depth_sigma=self.cad_depth_sigma,
            region_weight=self.cad_region_weight,
            depth_weight=self.cad_depth_weight,
            robust_delta=self.cad_robust_delta,
            maximum_depth_residual=self.cad_maximum_depth_residual,
            maximum_correspondence_pixels=self.cad_maximum_correspondence_pixels,
            maximum_step=self.cad_maximum_step,
            maximum_translation=self.cad_maximum_translation,
            convergence_translation=self.cad_convergence_translation,
            damping=self.cad_damping,
            rendered_mask_dilate_pixels=self.cad_rendered_mask_dilate_pixels,
            maximum_region_points=min(self.cad_maximum_region_points, 350),
            maximum_depth_points=min(self.cad_maximum_depth_points, 700),
            minimum_rendered_pixels=self.fit_min_points,
            minimum_overlap_ratio=self.cad_minimum_overlap_ratio,
            minimum_iou=self.cad_minimum_iou,
        )
        handle_triangles = {}
        for name in OBJECT_NAMES:
            if not name.startswith("part_type_c"):
                continue
            matching = [
                triangles
                for geom_name, triangles in self.model_triangles_by_geom.get(
                    name, {}
                ).items()
                if "handle" in geom_name
            ]
            if matching:
                handle_triangles[name] = [
                    triangle for triangles in matching for triangle in triangles
                ]
        self.c_handle_aligner = None
        if self.c_handle_refinement_enabled:
            missing_handles = [
                name
                for name in OBJECT_NAMES
                if name.startswith("part_type_c") and name not in handle_triangles
            ]
            if missing_handles:
                raise RuntimeError(
                    "C handle CAD missing for {}".format(", ".join(missing_handles))
                )
            self.c_handle_aligner = CadDepthAligner(
                handle_triangles,
                sample_count=self.c_handle_surface_samples,
                random_seed=self.random_seed + 7919,
                maximum_iterations=self.c_handle_maximum_iterations,
                region_sigma_px=1.5,
                depth_sigma=self.cad_depth_sigma,
                region_weight=2.0,
                depth_weight=0.5,
                robust_delta=self.cad_robust_delta,
                maximum_depth_residual=self.cad_maximum_depth_residual,
                maximum_correspondence_pixels=3.0,
                maximum_step=0.01,
                maximum_translation=0.04,
                convergence_translation=0.0001,
                damping=self.cad_damping,
                rendered_mask_dilate_pixels=1,
                maximum_region_points=500,
                maximum_depth_points=1000,
                minimum_rendered_pixels=self.c_handle_minimum_pixels,
                minimum_overlap_ratio=self.c_handle_minimum_overlap,
                minimum_iou=self.c_handle_minimum_iou,
            )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.latest_truth = None
        self.latest_depth = None
        self.latest_depth_frame = ""
        self.latest_depth_stamp = None
        self.latest_color = None
        self.latest_color_stamp = None
        self.depth_sequence = 0
        self.processed_depth_sequence = -1
        self.camera_info = None
        self.current_seed = None
        self.mode0_reference = None
        self.valid_samples = []
        self.locked_payload = None

        self.truth_sub = rospy.Subscriber(
            self.truth_topic, String, self._truth_callback, queue_size=1
        )
        self.mode0_reference_sub = rospy.Subscriber(
            self.mode0_reference_topic,
            String,
            self._mode0_reference_callback,
            queue_size=1,
        )
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, CompressedImage, self._depth_callback, queue_size=1
        )
        self.color_sub = rospy.Subscriber(
            self.color_topic, CompressedImage, self._color_callback, queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._camera_info_callback, queue_size=1
        )
        self.objects_pub = rospy.Publisher("~objects_json", String, queue_size=1, latch=True)
        rospy.loginfo(
            "task2_depth_xy: truth=%s mode0_reference=%s depth=%s color=%s "
            "output=%s CAD samples=%d "
            "region+depth GN iterations=%d green_hsv=%s..%s instance_naming=%s",
            self.truth_topic,
            self.mode0_reference_topic,
            self.depth_topic,
            self.color_topic,
            rospy.resolve_name("~objects_json"),
            self.cad_surface_samples,
            self.cad_maximum_iterations,
            self.green_hsv_lower,
            self.green_hsv_upper,
            self.instance_naming_method,
        )

    def _truth_callback(self, message):
        try:
            payload = json.loads(message.data)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "task2_depth_xy: invalid truth JSON: %s", exc)
            return
        seed = payload.get("seed")
        if self.current_seed is not None and seed != self.current_seed:
            self.valid_samples = []
            self.locked_payload = None
            self.mode0_reference = None
        self.current_seed = seed
        self.latest_truth = payload

    def _mode0_reference_callback(self, message):
        try:
            payload = json.loads(message.data)
            objects = {
                str(item.get("name", "")): item
                for item in payload.get("objects", [])
            }
            missing = [name for name in OBJECT_NAMES if name not in objects]
            if missing:
                raise ValueError("missing {}".format(", ".join(missing)))
            if str(payload.get("frame_id", "")) != FRAME_ID:
                raise ValueError(
                    "frame={} expected={}".format(payload.get("frame_id"), FRAME_ID)
                )
            transform_point_between_base_payloads(
                [0.0, 0.0, 0.0], payload, payload
            )
        except Exception as exc:
            rospy.logwarn(
                "task2_depth_xy: invalid same-run mode-0 reference: %s", exc
            )
            return
        self.mode0_reference = payload
        self.valid_samples = []
        self.locked_payload = None
        self.processed_depth_sequence = -1
        rospy.loginfo(
            "task2_depth_xy: accepted same-run mode-0 reference stamp=%s seed=%s; "
            "restarting %d-frame fit",
            payload.get("stamp"),
            payload.get("seed"),
            self.median_frames,
        )

    def _depth_callback(self, message):
        if self.locked_payload is not None:
            return
        try:
            self.latest_depth = _decode_compressed_depth(message)
            self.latest_depth_frame = str(message.header.frame_id)
            self.latest_depth_stamp = message.header.stamp
            self.depth_sequence += 1
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "task2_depth_xy: depth decode failed: %s", exc)

    def _color_callback(self, message):
        if self.locked_payload is not None:
            return
        try:
            self.latest_color = _decode_compressed_color(message)
            self.latest_color_stamp = message.header.stamp
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "task2_depth_xy: color decode failed: %s", exc)

    def _camera_info_callback(self, message):
        self.camera_info = message

    def _camera_to_base_transform(self):
        source_frame = self.latest_depth_frame
        if not source_frame and self.camera_info is not None:
            source_frame = str(self.camera_info.header.frame_id)
        if not source_frame:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                FRAME_ID,
                source_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout),
            ).transform
            return _transform_to_matrix(transform)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "task2_depth_xy: TF %s <- %s unavailable: %s",
                FRAME_ID,
                source_frame,
                exc,
            )
            return None

    def _truth_objects(self, truth_payload=None):
        payload = self.latest_truth if truth_payload is None else truth_payload
        if payload is None:
            return None
        objects = {}
        for item in payload.get("objects", []):
            name = str(item.get("name", ""))
            if name in OBJECT_NAMES and item.get("bbox_projected", False):
                objects[name] = item
        if any(name not in objects for name in OBJECT_NAMES):
            return None
        return objects

    @staticmethod
    def _reference_objects(reference_payload):
        if reference_payload is None:
            return None
        objects = {
            str(item.get("name", "")): item
            for item in reference_payload.get("objects", [])
            if len(item.get("position", [])) >= 3
        }
        if any(name not in objects for name in OBJECT_NAMES):
            return None
        return objects

    @staticmethod
    def _fit_in_mode0_reference(
        fit, _current_detection_item, current_payload, reference_payload
    ):
        if fit is None or "xy" not in fit:
            return fit
        result = copy.deepcopy(fit)
        current_xy = [float(value) for value in result["xy"]]
        current_z = float(result["z"])
        reference_point = transform_point_between_base_payloads(
            [current_xy[0], current_xy[1], current_z],
            current_payload,
            reference_payload,
        )
        result["current_xy"] = current_xy
        result["current_z"] = current_z
        result["xy"] = [float(reference_point[0]), float(reference_point[1])]
        result["z"] = float(reference_point[2])
        current_rotation = _quaternion_xyzw_to_matrix(result["orientation"])
        reference_rotation = transform_rotation_between_base_payloads(
            current_rotation, current_payload, reference_payload
        )
        result["current_yaw"] = float(result["yaw"])
        result["current_orientation"] = list(result["orientation"])
        result["yaw"] = yaw_from_rotation(reference_rotation)
        result["yaw_deg"] = math.degrees(result["yaw"])
        result["orientation"] = quaternion_xyzw_from_rotation(
            reference_rotation
        )
        result["yaw_orientation"] = yaw_orientation_xyzw(result["yaw"])
        if len(result.get("initial_xy", [])) >= 2:
            current_initial = [float(value) for value in result["initial_xy"]]
            current_initial_z = float(result.get("initial_z", current_z))
            reference_initial = transform_point_between_base_payloads(
                [current_initial[0], current_initial[1], current_initial_z],
                current_payload,
                reference_payload,
            )
            result["current_initial_xy"] = current_initial
            result["current_initial_z"] = current_initial_z
            result["initial_xy"] = [
                float(reference_initial[0]),
                float(reference_initial[1]),
            ]
            result["initial_z"] = float(reference_initial[2])
            result["render_offset_xy"] = [
                result["xy"][index] - result["initial_xy"][index]
                for index in range(2)
            ]
        return result

    def _fit_table_plane(self, depth, truth_objects, intrinsics, transform):
        height, width = depth.shape[:2]
        stride = self.table_stride
        pixels_v, pixels_u = np.mgrid[0:height:stride, 0:width:stride]
        sampled_depth = depth[0:height:stride, 0:width:stride]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= self.depth_min)
            & (sampled_depth <= self.depth_max)
        )

        image_width = float(self.latest_truth.get("image_width") or width)
        image_height = float(self.latest_truth.get("image_height") or height)
        scale_x = float(width) / image_width
        scale_y = float(height) / image_height
        for item in truth_objects.values():
            center = item["bbox_center"]
            size = item["bbox_size"]
            half_width = 0.5 * float(size[0]) * scale_x * self.table_roi_exclusion_scale
            half_height = 0.5 * float(size[1]) * scale_y * self.table_roi_exclusion_scale
            center_x = float(center[0]) * scale_x
            center_y = float(center[1]) * scale_y
            valid &= ~(
                (np.abs(pixels_u - center_x) <= half_width)
                & (np.abs(pixels_v - center_y) <= half_height)
            )

        depths = sampled_depth[valid].astype(np.float64)
        pixels_u = pixels_u[valid].astype(np.float64)
        pixels_v = pixels_v[valid].astype(np.float64)
        if depths.size < 100:
            return None
        points = _pixels_to_base(
            depths, pixels_u, pixels_v, intrinsics, transform[0], transform[1]
        )
        workspace = (
            (points[:, 0] >= self.table_x_range[0])
            & (points[:, 0] <= self.table_x_range[1])
            & (points[:, 1] >= self.table_y_range[0])
            & (points[:, 1] <= self.table_y_range[1])
            & (points[:, 2] >= self.table_z_range[0])
            & (points[:, 2] <= self.table_z_range[1])
        )
        return _fit_horizontal_plane_ransac(
            points[workspace],
            self.table_ransac_iterations,
            self.table_distance_threshold,
            self.table_max_tilt_deg,
            self.table_histogram_bin_size,
            self.random_seed + self.depth_sequence,
        )

    def _roi_bounds(self, item, depth_shape):
        height, width = depth_shape[:2]
        image_width = float(self.latest_truth.get("image_width") or width)
        image_height = float(self.latest_truth.get("image_height") or height)
        scale_x = float(width) / image_width
        scale_y = float(height) / image_height
        center = item["bbox_center"]
        size = item["bbox_size"]
        center_x = float(center[0]) * scale_x
        center_y = float(center[1]) * scale_y
        half_width = 0.5 * float(size[0]) * scale_x * self.roi_scale
        half_height = 0.5 * float(size[1]) * scale_y * self.roi_scale
        return (
            max(0, int(math.floor(center_x - half_width))),
            max(0, int(math.floor(center_y - half_height))),
            min(width, int(math.ceil(center_x + half_width + 1.0))),
            min(height, int(math.ceil(center_y + half_height + 1.0))),
        )

    def _cad_xy_offset_and_extent(self, model_name, rotation):
        vertices = np.asarray(self.model_vertices[model_name], dtype=np.float64)
        oriented = vertices.dot(rotation.T)
        minimum = np.min(oriented[:, :2], axis=0)
        maximum = np.max(oriented[:, :2], axis=0)
        return 0.5 * (minimum + maximum), maximum - minimum

    def _alignment_selection_score(
        self, context, alignment, rotation, use_handle=True
    ):
        score = float(alignment["score"])
        handle_iou = 0.0
        handle_score = 0.0
        if (
            context["class_name"] == "part_type_c"
            and use_handle
            and self.c_handle_aligner is not None
            and context["handle_mask"] is not None
        ):
            body_z = body_origin_z_from_table(
                context["class_name"], alignment["xy"], context["plane"]
            )
            handle_alignment = self.c_handle_aligner.evaluate_xy(
                context["model_name"],
                alignment["xy"],
                body_z,
                rotation,
                context["intrinsics"],
                context["transform"],
                context["roi"],
                context["plane"],
                context["height_range"],
                context["handle_depth"],
                context["handle_mask"],
                minimum_overlap_ratio=0.0,
                minimum_iou=0.0,
            )
            if handle_alignment is None:
                handle_score = 50.0
            else:
                handle_score = float(handle_alignment["score"])
                handle_iou = float(handle_alignment["iou"])
            score += self.yaw_handle_score_weight * handle_score
        return score, handle_score, handle_iou

    def _evaluate_yaw_candidate(
        self, context, yaw, cad_center_xy, use_handle=True
    ):
        yaw = wrap_angle(yaw)
        rotation = rotation_from_yaw(context["class_name"], yaw)
        cad_offset, cad_extent = self._cad_xy_offset_and_extent(
            context["model_name"], rotation
        )
        initial_xy = np.asarray(cad_center_xy, dtype=np.float64) - cad_offset
        body_z = body_origin_z_from_table(
            context["class_name"], initial_xy, context["plane"]
        )
        alignment = self.cad_yaw_aligner.evaluate_xy(
            context["model_name"],
            initial_xy,
            body_z,
            rotation,
            context["intrinsics"],
            context["transform"],
            context["roi"],
            context["plane"],
            context["height_range"],
            context["observed_depth"],
            context["observed_mask"],
            minimum_overlap_ratio=context["minimum_overlap"],
            minimum_iou=context["minimum_iou"],
        )
        if alignment is None:
            return None
        score, handle_score, handle_iou = self._alignment_selection_score(
            context, alignment, rotation, use_handle=use_handle
        )
        return {
            "yaw": yaw,
            "rotation": rotation,
            "cad_offset": cad_offset,
            "cad_extent": cad_extent,
            "alignment": alignment,
            "selection_score": score,
            "handle_score": handle_score,
            "handle_iou": handle_iou,
        }

    def _apply_yaw_prior_objective(
        self, context, candidate, active_window_priors
    ):
        if candidate is None or not active_window_priors:
            return candidate
        prior_name, prior_yaw = min(
            active_window_priors,
            key=lambda value: abs(
                angle_difference(float(candidate["yaw"]), value[1])
            ),
        )
        terms = yaw_prior_regularized_score(
            candidate["selection_score"],
            candidate["yaw"],
            prior_yaw,
            self.yaw_prior_sigma_by_class[context["class_name"]],
            self.yaw_prior_regularization_weight,
        )
        result = dict(candidate)
        result["yaw_prior_selected_name"] = prior_name
        result["yaw_data_score"] = terms["data_score"]
        result["yaw_prior_penalty"] = terms["prior_penalty"]
        result["yaw_prior_deviation_deg"] = math.degrees(
            terms["deviation_rad"]
        )
        result["selection_score"] = terms["objective_score"]
        return result

    def _refine_yaw_candidate(
        self, context, candidate, final=False, use_handle=False
    ):
        yaw = float(candidate["yaw"])
        rotation = rotation_from_yaw(context["class_name"], yaw)
        initial_xy = candidate["alignment"]["xy"]
        body_z = body_origin_z_from_table(
            context["class_name"], initial_xy, context["plane"]
        )
        aligner = self.cad_aligner if final else self.cad_yaw_aligner
        alignment = aligner.refine_xy(
            context["model_name"],
            initial_xy,
            body_z,
            rotation,
            context["intrinsics"],
            context["transform"],
            context["roi"],
            context["plane"],
            context["height_range"],
            context["observed_depth"],
            context["observed_mask"],
            minimum_overlap_ratio=context["minimum_overlap"],
            minimum_iou=context["minimum_iou"],
        )
        if alignment is None:
            return None
        score, handle_score, handle_iou = self._alignment_selection_score(
            context, alignment, rotation, use_handle=use_handle
        )
        result = dict(candidate)
        result.update(
            {
                "rotation": rotation,
                "alignment": alignment,
                "selection_score": score,
                "handle_score": handle_score,
                "handle_iou": handle_iou,
            }
        )
        return result

    def _fit_xy_yaw_without_truth(
        self,
        context,
        observed_points,
        observed_center,
        forced_yaw_prior_name=None,
    ):
        prior_candidates = []
        active_window_priors = []
        if self.instance_naming_method == INSTANCE_NAMING_YAW_PRIOR:
            anisotropy = 0.0
            yaw_search_mode = "class_yaw_prior_match"
            prior_candidates = [
                (
                    motion_name,
                    wrap_angle(
                        prior_yaw + context["yaw_prior_frame_offset"]
                    ),
                )
                for motion_name, prior_yaw in yaw_prior_candidates(
                    context["class_name"]
                )
            ]
            if forced_yaw_prior_name is not None:
                prior_candidates = [
                    candidate
                    for candidate in prior_candidates
                    if candidate[0] == forced_yaw_prior_name
                ]
                if not prior_candidates:
                    raise ValueError(
                        "invalid {} yaw prior {}".format(
                            context["class_name"], forced_yaw_prior_name
                        )
                    )
                yaw_search_mode = "class_yaw_prior_joint_forced"
            yaw_candidates = [yaw for _, yaw in prior_candidates]
        else:
            anisotropy = 0.0
            yaw_search_mode = "class_yaw_prior_window_{:.0f}".format(
                math.degrees(self.yaw_prior_window)
            )
            window_candidates = yaw_prior_window_candidates(
                context["class_name"],
                context["yaw_prior_frame_offset"],
                self.yaw_prior_window,
                self.yaw_prior_window_step,
            )
            if forced_yaw_prior_name is not None:
                window_candidates = [
                    candidate
                    for candidate in window_candidates
                    if candidate[0] == forced_yaw_prior_name
                ]
                if not window_candidates:
                    raise ValueError(
                        "invalid {} yaw-prior window {}".format(
                            context["class_name"], forced_yaw_prior_name
                        )
                    )
                yaw_search_mode += "_joint_candidate"
            active_prior_names = sorted(
                {prior_name for prior_name, _ in window_candidates}
            )
            active_window_priors = [
                (
                    prior_name,
                    wrap_angle(prior_yaw + context["yaw_prior_frame_offset"]),
                )
                for prior_name, prior_yaw in yaw_prior_candidates(
                    context["class_name"]
                )
                if prior_name in active_prior_names
            ]
            yaw_candidates = [yaw for _, yaw in window_candidates]
            if not yaw_candidates:
                raise RuntimeError(
                    "{} produced no yaw-prior window candidates".format(
                        context["class_name"]
                    )
                )
        unique_candidates = []
        use_handle_for_candidates = True
        for yaw in yaw_candidates:
            if all(abs(angle_difference(yaw, other)) > 1e-6 for other in unique_candidates):
                unique_candidates.append(yaw)

        evaluations = []
        for yaw in unique_candidates:
            candidate = self._evaluate_yaw_candidate(
                context,
                yaw,
                observed_center,
                use_handle=use_handle_for_candidates,
            )
            candidate = self._apply_yaw_prior_objective(
                context, candidate, active_window_priors
            )
            if candidate is not None:
                evaluations.append(candidate)
        if not evaluations:
            return None
        prior_candidate_scores = {}
        if prior_candidates:
            for motion_name, prior_yaw in prior_candidates:
                matching = [
                    value
                    for value in evaluations
                    if abs(angle_difference(value["yaw"], prior_yaw)) <= 1e-6
                ]
                prior_candidate_scores[motion_name] = (
                    float(matching[0]["selection_score"])
                    if matching
                    else None
                )
        evaluations.sort(key=lambda value: value["selection_score"])
        coarse_best_yaw = float(evaluations[0]["yaw"])

        refined = []
        for candidate in evaluations[: self.yaw_refine_top_k]:
            result = self._refine_yaw_candidate(context, candidate)
            result = self._apply_yaw_prior_objective(
                context, result, active_window_priors
            )
            if result is not None:
                refined.append(result)
        if not refined:
            return None
        best = min(refined, key=lambda value: value["selection_score"])
        yaw_evaluations = len(evaluations) + len(refined)

        if self.instance_naming_method != INSTANCE_NAMING_YAW_PRIOR:
            for step in self.yaw_refinement_steps:
                cad_center = (
                    np.asarray(best["alignment"]["xy"], dtype=np.float64)
                    + np.asarray(best["cad_offset"], dtype=np.float64)
                )
                local = []
                for direction in (-1.0, 1.0):
                    candidate_yaw = best["yaw"] + direction * step
                    inside_prior_window = any(
                        abs(
                            angle_difference(
                                candidate_yaw,
                                prior_yaw,
                            )
                        )
                        <= self.yaw_prior_window + 1e-9
                        for _, prior_yaw in active_window_priors
                    )
                    if not inside_prior_window:
                        continue
                    candidate = self._evaluate_yaw_candidate(
                        context,
                        candidate_yaw,
                        cad_center,
                        use_handle=False,
                    )
                    candidate = self._apply_yaw_prior_objective(
                        context, candidate, active_window_priors
                    )
                    if candidate is not None:
                        local.append(candidate)
                yaw_evaluations += len(local)
                if local:
                    local_best = min(
                        local, key=lambda value: value["selection_score"]
                    )
                    if local_best["selection_score"] < best["selection_score"]:
                        best = local_best

        final = self._refine_yaw_candidate(context, best, final=True)
        final = self._apply_yaw_prior_objective(
            context, final, active_window_priors
        )
        yaw_evaluations += 1
        if final is None:
            return None
        final["yaw_initial"] = coarse_best_yaw
        final["yaw_pca_anisotropy"] = float(anisotropy)
        final["yaw_evaluations"] = int(yaw_evaluations)
        final["yaw_search_mode"] = yaw_search_mode
        if prior_candidates:
            final["yaw_prior_candidate_scores"] = prior_candidate_scores
            final["yaw_prior_selected_name"] = min(
                prior_candidates,
                key=lambda value: abs(
                    angle_difference(final["yaw"], value[1])
                ),
            )[0]
        elif active_window_priors:
            selected_prior_name, selected_prior_yaw = min(
                active_window_priors,
                key=lambda value: abs(angle_difference(final["yaw"], value[1])),
            )
            final["yaw_prior_selected_name"] = selected_prior_name
            final["yaw_prior_candidate_scores"] = {
                selected_prior_name: float(final["selection_score"])
            }
            edge_distance = max(
                0.0,
                self.yaw_prior_window
                - abs(angle_difference(final["yaw"], selected_prior_yaw)),
            )
            final["yaw_window_edge_distance_deg"] = math.degrees(edge_distance)
        return final

    def _assign_output_motion_names(self, output_objects):
        if self.instance_naming_method == INSTANCE_NAMING_RELATIVE_POSITION:
            for item in output_objects:
                item["instance_naming_method"] = INSTANCE_NAMING_RELATIVE_POSITION
            return output_objects, INSTANCE_NAMING_RELATIVE_POSITION

        if not all("fit_yaw" in item for item in output_objects):
            for item in output_objects:
                item["instance_naming_method"] = (
                    "relative_position_pending_yaw_prior"
                )
            return output_objects, "relative_position_pending_yaw_prior"

        return (
            assign_motion_names(output_objects, INSTANCE_NAMING_YAW_PRIOR),
            INSTANCE_NAMING_YAW_PRIOR,
        )

    def _c_handle_mask(self, color_roi, foreground_mask):
        if color_roi is None or self.c_handle_aligner is None:
            return None
        hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        red = (
            ((hue <= self.c_handle_hue_low_max) | (hue >= self.c_handle_hue_high_min))
            & (saturation >= self.c_handle_saturation_min)
            & (value >= self.c_handle_value_min)
            & np.asarray(foreground_mask, dtype=bool)
        )
        image = red.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            image, connectivity=8
        )
        if count <= 1:
            return None
        candidates = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= self.c_handle_minimum_pixels
        ]
        if not candidates:
            return None
        largest = max(candidates, key=lambda label: stats[label, cv2.CC_STAT_AREA])
        return labels == largest

    def _fit_object(
        self,
        item,
        depth,
        color_bgr,
        intrinsics,
        transform,
        plane,
        table_green_pixels,
        yaw_prior_frame_offset,
        forced_yaw_prior_name=None,
    ):
        x1, y1, x2, y2 = self._roi_bounds(item, depth.shape)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return {"accepted": False, "rejection_reason": "invalid_roi"}
        roi = depth[y1:y2, x1:x2]
        local_v, local_u = np.mgrid[0 : roi.shape[0], 0 : roi.shape[1]]
        valid = (
            np.isfinite(roi)
            & (roi >= self.depth_min)
            & (roi <= self.depth_max)
        )
        if int(np.count_nonzero(valid)) < self.fit_min_points:
            return {
                "accepted": False,
                "rejection_reason": "insufficient_roi_depth",
            }

        depths = roi[valid].astype(np.float64)
        pixels_u = (local_u[valid] + x1).astype(np.float64)
        pixels_v = (local_v[valid] + y1).astype(np.float64)
        points = _pixels_to_base(
            depths, pixels_u, pixels_v, intrinsics, transform[0], transform[1]
        )
        coefficients = plane["coefficients"]
        table_z = (
            coefficients[0] * points[:, 0]
            + coefficients[1] * points[:, 1]
            + coefficients[2]
        )
        point_height = points[:, 2] - table_z
        foreground = (
            (point_height >= self.object_min_height)
            & (point_height <= self.object_max_height)
        )
        green_roi = table_green_pixels[y1:y2, x1:x2]
        green_valid = green_roi[local_v[valid], local_u[valid]]
        foreground &= ~green_valid
        binary = np.zeros(roi.shape, dtype=np.uint8)
        binary[local_v[valid][foreground], local_u[valid][foreground]] = 1
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        keep_labels = [
            label
            for label in range(1, component_count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= self.component_min_pixels
        ]
        if not keep_labels:
            return {
                "accepted": False,
                "rejection_reason": "no_foreground_component",
            }
        keep_mask = np.isin(labels, keep_labels)
        selected = foreground & keep_mask[local_v[valid], local_u[valid]]
        selected_points = points[selected]
        if selected_points.shape[0] < self.fit_min_points:
            return {
                "accepted": False,
                "rejection_reason": "insufficient_foreground_points",
            }

        low_percentile, high_percentile = self.bound_percentiles
        observed_min = np.percentile(selected_points[:, :2], low_percentile, axis=0)
        observed_max = np.percentile(selected_points[:, :2], high_percentile, axis=0)
        observed_center = 0.5 * (observed_min + observed_max)
        observed_extent = observed_max - observed_min
        class_name = str(item.get("class_name", ""))
        observed_depth = np.full(roi.shape, np.nan, dtype=np.float64)
        selected_local_u = local_u[valid][selected]
        selected_local_v = local_v[valid][selected]
        observed_depth[selected_local_v, selected_local_u] = depths[selected]
        model_name = str(item.get("ground_truth_name", item["name"]))
        minimum_overlap = self.cad_minimum_overlap_by_class.get(
            class_name, self.cad_minimum_overlap_ratio
        )
        minimum_iou = self.cad_minimum_iou_by_class.get(
            class_name, self.cad_minimum_iou
        )
        color_roi = color_bgr[y1:y2, x1:x2] if color_bgr is not None else None
        handle_mask = (
            self._c_handle_mask(color_roi, keep_mask)
            if class_name == "part_type_c"
            else None
        )
        handle_depth = np.full(roi.shape, np.nan, dtype=np.float64)
        if handle_mask is not None:
            handle_depth[handle_mask] = roi[handle_mask]
        context = {
            "class_name": class_name,
            "model_name": model_name,
            "intrinsics": intrinsics,
            "transform": transform,
            "roi": [x1, y1, x2, y2],
            "plane": plane,
            "height_range": [self.object_min_height, self.object_max_height],
            "observed_depth": observed_depth,
            "observed_mask": keep_mask,
            "minimum_overlap": minimum_overlap,
            "minimum_iou": minimum_iou,
            "handle_mask": handle_mask,
            "handle_depth": handle_depth,
            "yaw_prior_frame_offset": float(yaw_prior_frame_offset),
        }
        pose_fit = self._fit_xy_yaw_without_truth(
            context,
            selected_points,
            observed_center,
            forced_yaw_prior_name=forced_yaw_prior_name,
        )
        if pose_fit is None:
            return {
                "accepted": False,
                "rejection_reason": "truth_independent_xy_yaw_fit_failed",
            }
        alignment = pose_fit["alignment"]
        fitted_yaw = float(pose_fit["yaw"])
        body_rotation = pose_fit["rotation"]
        cad_extent = np.asarray(pose_fit["cad_extent"], dtype=np.float64)
        extent_ratio = observed_extent / np.maximum(cad_extent, 1e-6)
        initial_body_z = body_origin_z_from_table(
            class_name, alignment["initial_xy"], plane
        )
        handle_used = False
        handle_pixels = 0
        handle_iou = float(pose_fit["handle_iou"])
        handle_iterations = 0
        if (
            class_name == "part_type_c"
            and self.c_handle_aligner is not None
            and handle_mask is not None
        ):
            aligned_body_z = body_origin_z_from_table(
                class_name, alignment["xy"], plane
            )
            handle_pixels = int(np.count_nonzero(handle_mask))
            handle_alignment = self.c_handle_aligner.refine_xy(
                model_name,
                alignment["xy"],
                aligned_body_z,
                body_rotation,
                intrinsics,
                transform,
                [x1, y1, x2, y2],
                plane,
                [self.object_min_height, self.object_max_height],
                handle_depth,
                handle_mask,
                minimum_overlap_ratio=self.c_handle_minimum_overlap,
                minimum_iou=self.c_handle_minimum_iou,
            )
            if handle_alignment is not None:
                handle_iou = float(handle_alignment["iou"])
                handle_iterations = int(handle_alignment["iterations"])
            if handle_alignment is not None and handle_alignment["accepted"]:
                handle_body_z = body_origin_z_from_table(
                    class_name, handle_alignment["xy"], plane
                )
                verified = self.cad_aligner.evaluate_xy(
                    model_name,
                    handle_alignment["xy"],
                    handle_body_z,
                    body_rotation,
                    intrinsics,
                    transform,
                    [x1, y1, x2, y2],
                    plane,
                    [self.object_min_height, self.object_max_height],
                    observed_depth,
                    keep_mask,
                    minimum_overlap_ratio=minimum_overlap,
                    minimum_iou=minimum_iou,
                )
                if verified is not None and verified["accepted"]:
                    for key in (
                        "xy",
                        "score",
                        "depth_error_m",
                        "coverage",
                        "observed_coverage",
                        "iou",
                        "cad_to_observed_px",
                        "observed_to_cad_px",
                        "rendered_pixels",
                        "accepted",
                        "rejection_reason",
                        "minimum_overlap_ratio",
                        "minimum_iou",
                    ):
                        alignment[key] = verified[key]
                    alignment["iterations"] += handle_alignment["iterations"]
                    alignment["evaluations"] += handle_alignment["evaluations"]
                    alignment["converged"] = handle_alignment["converged"]
                    handle_used = True
        fitted_xy = np.asarray(alignment["xy"], dtype=np.float64)
        fitted_body_z = body_origin_z_from_table(class_name, fitted_xy, plane)

        fitted_base = np.array(
            [fitted_xy[0], fitted_xy[1], fitted_body_z],
            dtype=np.float64,
        )
        camera_point = (fitted_base - transform[1]).dot(transform[0])
        if camera_point[2] > 0.03:
            fitted_pixel = [
                float(intrinsics[0] * camera_point[0] / camera_point[2] + intrinsics[2]),
                float(intrinsics[1] * camera_point[1] / camera_point[2] + intrinsics[3]),
            ]
        else:
            fitted_pixel = []

        selected_pixels_u = pixels_u[selected]
        selected_pixels_v = pixels_v[selected]
        return {
            "accepted": bool(alignment["accepted"]),
            "rejection_reason": str(alignment["rejection_reason"]),
            "minimum_overlap_ratio": float(alignment["minimum_overlap_ratio"]),
            "minimum_iou": float(alignment["minimum_iou"]),
            "xy": [float(fitted_xy[0]), float(fitted_xy[1])],
            "z": float(fitted_body_z),
            "yaw": fitted_yaw,
            "yaw_deg": math.degrees(fitted_yaw),
            "orientation": orientation_xyzw_from_yaw(class_name, fitted_yaw),
            "yaw_orientation": yaw_orientation_xyzw(fitted_yaw),
            "initial_z": float(initial_body_z),
            "fit_pixel_center": fitted_pixel,
            "foreground_pixel_center": [
                float(np.median(selected_pixels_u)),
                float(np.median(selected_pixels_v)),
            ],
            "points": int(selected_points.shape[0]),
            "observed_extent": [float(value) for value in observed_extent],
            "cad_extent": [float(value) for value in cad_extent],
            "extent_ratio": [float(value) for value in extent_ratio],
            "height_range": [
                float(np.min(point_height[selected])),
                float(np.max(point_height[selected])),
            ],
            "roi": [x1, y1, x2, y2],
            "initial_xy": alignment["initial_xy"],
            "render_offset_xy": alignment["offset_xy"],
            "render_score": alignment["score"],
            "render_depth_error_m": alignment["depth_error_m"],
            "render_coverage": alignment["coverage"],
            "render_observed_coverage": alignment["observed_coverage"],
            "render_iou": alignment["iou"],
            "render_cad_to_observed_px": alignment["cad_to_observed_px"],
            "render_observed_to_cad_px": alignment["observed_to_cad_px"],
            "rendered_pixels": alignment["rendered_pixels"],
            "render_evaluations": alignment["evaluations"],
            "yaw_initial": float(pose_fit["yaw_initial"]),
            "yaw_pca_anisotropy": float(pose_fit["yaw_pca_anisotropy"]),
            "yaw_evaluations": int(pose_fit["yaw_evaluations"]),
            "yaw_search_mode": str(pose_fit["yaw_search_mode"]),
            "yaw_prior_candidate_scores": dict(
                pose_fit.get("yaw_prior_candidate_scores", {})
            ),
            "yaw_prior_selected_name": str(
                pose_fit.get("yaw_prior_selected_name", "")
            ),
            "yaw_window_edge_distance_deg": float(
                pose_fit.get("yaw_window_edge_distance_deg", 0.0)
            ),
            "yaw_data_score": float(
                pose_fit.get("yaw_data_score", pose_fit["selection_score"])
            ),
            "yaw_prior_penalty": float(pose_fit.get("yaw_prior_penalty", 0.0)),
            "yaw_prior_deviation_deg": float(
                pose_fit.get("yaw_prior_deviation_deg", 0.0)
            ),
            "yaw_selection_score": float(pose_fit["selection_score"]),
            "yaw_handle_score": float(pose_fit["handle_score"]),
            "alignment_iterations": alignment["iterations"],
            "alignment_converged": alignment["converged"],
            "alignment_region_points": alignment["region_points"],
            "alignment_depth_inliers": alignment["depth_inliers"],
            "alignment_region_cost": alignment["region_cost"],
            "alignment_depth_cost": alignment["depth_cost"],
            "c_handle_used": handle_used,
            "c_handle_pixels": handle_pixels,
            "c_handle_iou": handle_iou,
            "c_handle_iterations": handle_iterations,
            "green_rejected_pixels": int(np.count_nonzero(green_valid)),
        }

    def _aggregate_fit(self, name, current_fit=None):
        fits = [sample[name] for sample in self.valid_samples]
        if not fits and current_fit is not None:
            fits = [current_fit]
        if not fits:
            return None

        result = copy.deepcopy(fits[-1])
        vector_keys = (
            "xy",
            "current_xy",
            "fit_pixel_center",
            "foreground_pixel_center",
            "observed_extent",
            "cad_extent",
            "extent_ratio",
            "height_range",
            "initial_xy",
            "current_initial_xy",
            "render_offset_xy",
        )
        for key in vector_keys:
            values = [fit.get(key, []) for fit in fits]
            lengths = {len(value) for value in values}
            if len(lengths) == 1 and next(iter(lengths), 0) > 0:
                result[key] = [
                    float(value)
                    for value in np.median(
                        np.asarray(values, dtype=np.float64), axis=0
                    )
                ]
        result["points"] = int(
            round(float(np.median([fit["points"] for fit in fits])))
        )
        for key in (
            "z",
            "current_z",
            "initial_z",
            "current_initial_z",
            "render_score",
            "render_depth_error_m",
            "render_coverage",
            "render_observed_coverage",
            "render_iou",
            "render_cad_to_observed_px",
            "render_observed_to_cad_px",
            "rendered_pixels",
            "alignment_iterations",
            "alignment_region_points",
            "alignment_depth_inliers",
            "alignment_region_cost",
            "alignment_depth_cost",
            "yaw_pca_anisotropy",
            "yaw_evaluations",
            "yaw_selection_score",
            "yaw_handle_score",
            "c_handle_pixels",
            "c_handle_iou",
            "c_handle_iterations",
            "green_rejected_pixels",
        ):
            result[key] = float(np.median([fit[key] for fit in fits]))
        for key in (
            "yaw_window_edge_distance_deg",
            "yaw_joint_selected_cost",
            "yaw_joint_direct_cost",
            "yaw_joint_swapped_cost",
            "yaw_joint_assignment_margin",
            "yaw_data_score",
            "yaw_prior_penalty",
            "yaw_prior_deviation_deg",
        ):
            values = [
                float(fit[key])
                for fit in fits
                if fit.get(key) is not None
                and math.isfinite(float(fit[key]))
            ]
            if values:
                result[key] = float(np.median(values))
            else:
                result.pop(key, None)
        for key in (
            "yaw_prior_selected_name",
            "yaw_joint_assignment_mode",
        ):
            values = [str(fit.get(key, "")) for fit in fits if fit.get(key)]
            if values:
                result[key] = max(
                    sorted(set(values)), key=lambda value: values.count(value)
                )
        result["alignment_converged"] = all(
            bool(fit["alignment_converged"]) for fit in fits
        )
        result["c_handle_used"] = all(bool(fit["c_handle_used"]) for fit in fits)
        class_name = str(name).rsplit("_", 1)[0]
        result["yaw"] = circular_median([fit["yaw"] for fit in fits])
        result["yaw_deg"] = math.degrees(result["yaw"])
        result["orientation"] = orientation_xyzw_from_yaw(
            class_name, result["yaw"]
        )
        result["yaw_orientation"] = yaw_orientation_xyzw(result["yaw"])
        result["yaw_initial"] = circular_median(
            [fit["yaw_initial"] for fit in fits]
        )
        return result

    def _make_output_object(
        self, truth_item, reference_item, fit, locked, fit_status
    ):
        output = copy.deepcopy(truth_item)
        output["current_truth_position"] = [
            float(value) for value in truth_item["position"]
        ]
        output["current_truth_yaw"] = float(truth_item.get("yaw", 0.0))
        output["truth_position"] = [
            float(value) for value in reference_item["position"]
        ]
        output["truth_yaw"] = float(reference_item.get("yaw", 0.0))
        output["truth_yaw_deg"] = math.degrees(output["truth_yaw"])
        output["truth_orientation"] = copy.deepcopy(
            reference_item.get("orientation", [])
        )
        output["position"] = []
        output["orientation"] = []
        output["truth_reference"] = "same_run_mode0_equivalent_input"
        output["fit_coordinate_reference"] = "same_run_mode0_equivalent_input"
        output["position_source"] = POSE_SOURCE
        output["xy_position_source"] = POSE_SOURCE
        output["z_position_source"] = Z_POSITION_SOURCE
        output["yaw_source"] = POSE_SOURCE
        output["orientation_source"] = POSE_SOURCE
        output["fit_method"] = "pca_cad_region_depth_xy_yaw_fit"
        output["fit_valid"] = False
        output["confidence"] = 0.0
        output["fit_required_samples"] = self.median_frames
        output["fit_sample_count"] = len(self.valid_samples)
        output["fit_status"] = "locked" if locked else "waiting"
        output["fit_diagnostic_status"] = fit_status
        if fit is None:
            output["position"] = []
            return output
        output["fit_alignment_accepted"] = bool(fit.get("accepted", False))
        output["fit_alignment_rejection_reason"] = str(
            fit.get("rejection_reason", "")
        )
        if "xy" not in fit:
            output["position"] = []
            return output

        median_xy = np.asarray(fit["xy"], dtype=np.float64)
        accepted_xy = np.asarray(
            [sample[truth_item["name"]]["xy"] for sample in self.valid_samples],
            dtype=np.float64,
        )
        deviations = (
            np.linalg.norm(accepted_xy - median_xy.reshape(1, 2), axis=1)
            if accepted_xy.size
            else np.asarray([0.0], dtype=np.float64)
        )
        fit_position = [
            float(median_xy[0]),
            float(median_xy[1]),
            float(fit["z"]),
        ]
        fit_yaw = float(fit["yaw"])
        position_error = [
            fit_position[index] - float(reference_item["position"][index])
            for index in range(3)
        ]
        yaw_error = angle_difference(fit_yaw, output["truth_yaw"])
        output["fit_position"] = fit_position
        output["fit_yaw"] = fit_yaw
        output["fit_yaw_deg"] = math.degrees(fit_yaw)
        output["fit_orientation"] = list(fit["orientation"])
        output["position"] = list(fit_position) if locked else []
        output["yaw"] = fit_yaw
        output["yaw_deg"] = math.degrees(fit_yaw)
        output["orientation"] = list(fit["orientation"])
        output["yaw_orientation"] = list(fit["yaw_orientation"])
        output["fit_valid"] = locked
        output["confidence"] = 1.0 if locked else 0.0
        output["fit_xy_mad_m"] = float(np.median(deviations))
        output["fit_z"] = float(fit["z"])
        output["x_error_m"] = float(position_error[0])
        output["y_error_m"] = float(position_error[1])
        output["z_error_m"] = float(position_error[2])
        output["yaw_error_rad"] = float(yaw_error)
        output["yaw_error_deg"] = math.degrees(yaw_error)
        output["fit_pixel_center"] = fit.get("fit_pixel_center", [])
        output["foreground_pixel_center"] = fit.get(
            "foreground_pixel_center", []
        )
        output["fit_points"] = int(fit.get("points", 0))
        if "render_score" not in fit:
            return output
        output["fit_observed_extent"] = fit["observed_extent"]
        output["fit_cad_extent"] = fit["cad_extent"]
        output["fit_extent_ratio"] = fit["extent_ratio"]
        output["fit_height_range"] = fit["height_range"]
        output["fit_roi"] = fit["roi"]
        output["fit_initial_xy"] = fit["initial_xy"]
        output["fit_current_xy"] = fit["current_xy"]
        output["fit_current_initial_xy"] = fit["current_initial_xy"]
        output["fit_render_offset_xy"] = fit["render_offset_xy"]
        output["fit_render_score"] = float(fit["render_score"])
        output["fit_render_depth_error_m"] = float(fit["render_depth_error_m"])
        output["fit_render_coverage"] = float(fit["render_coverage"])
        output["fit_render_observed_coverage"] = float(
            fit["render_observed_coverage"]
        )
        output["fit_render_iou"] = float(fit["render_iou"])
        output["fit_render_cad_to_observed_px"] = float(
            fit["render_cad_to_observed_px"]
        )
        output["fit_render_observed_to_cad_px"] = float(
            fit["render_observed_to_cad_px"]
        )
        output["fit_rendered_pixels"] = int(round(fit["rendered_pixels"]))
        output["fit_render_evaluations"] = int(fit["render_evaluations"])
        output["fit_yaw_initial_deg"] = math.degrees(fit["yaw_initial"])
        output["fit_yaw_pca_anisotropy"] = float(
            fit["yaw_pca_anisotropy"]
        )
        output["fit_yaw_evaluations"] = int(round(fit["yaw_evaluations"]))
        output["fit_yaw_search_mode"] = str(
            fit.get("yaw_search_mode", "unknown")
        )
        output["fit_yaw_prior_selected_name"] = str(
            fit.get("yaw_prior_selected_name", "")
        )
        output["fit_yaw_prior_candidate_scores"] = dict(
            fit.get("yaw_prior_candidate_scores", {})
        )
        output["fit_yaw_window_edge_distance_deg"] = fit.get(
            "yaw_window_edge_distance_deg"
        )
        output["fit_yaw_data_score"] = fit.get("yaw_data_score")
        output["fit_yaw_prior_penalty"] = fit.get("yaw_prior_penalty")
        output["fit_yaw_prior_deviation_deg"] = fit.get(
            "yaw_prior_deviation_deg"
        )
        output["fit_yaw_joint_assignment_mode"] = str(
            fit.get("yaw_joint_assignment_mode", "")
        )
        output["fit_yaw_joint_selected_cost"] = fit.get(
            "yaw_joint_selected_cost"
        )
        output["fit_yaw_joint_direct_cost"] = fit.get(
            "yaw_joint_direct_cost"
        )
        output["fit_yaw_joint_swapped_cost"] = fit.get(
            "yaw_joint_swapped_cost"
        )
        output["fit_yaw_joint_assignment_margin"] = fit.get(
            "yaw_joint_assignment_margin"
        )
        output["fit_yaw_selection_score"] = float(
            fit["yaw_selection_score"]
        )
        output["fit_yaw_handle_score"] = float(fit["yaw_handle_score"])
        output["fit_alignment_iterations"] = int(
            round(fit["alignment_iterations"])
        )
        output["fit_alignment_converged"] = bool(
            fit["alignment_converged"]
        )
        output["fit_alignment_region_points"] = int(
            round(fit["alignment_region_points"])
        )
        output["fit_alignment_depth_inliers"] = int(
            round(fit["alignment_depth_inliers"])
        )
        output["fit_alignment_region_cost"] = float(
            fit["alignment_region_cost"]
        )
        output["fit_alignment_depth_cost"] = float(
            fit["alignment_depth_cost"]
        )
        output["fit_c_handle_used"] = bool(fit["c_handle_used"])
        output["fit_c_handle_pixels"] = int(round(fit["c_handle_pixels"]))
        output["fit_c_handle_iou"] = float(fit["c_handle_iou"])
        output["fit_c_handle_iterations"] = int(
            round(fit["c_handle_iterations"])
        )
        output["fit_green_rejected_pixels"] = int(
            round(fit["green_rejected_pixels"])
        )
        return output

    def process_once(self):
        if self.locked_payload is not None:
            return
        if (
            self.latest_depth is None
            or (self.green_filter_enabled and self.latest_color is None)
            or self.camera_info is None
            or self.latest_truth is None
            or self.mode0_reference is None
            or self.depth_sequence == self.processed_depth_sequence
        ):
            return
        truth_payload = copy.deepcopy(self.latest_truth)
        reference_payload = copy.deepcopy(self.mode0_reference)
        if truth_payload.get("seed") != reference_payload.get("seed"):
            rospy.logwarn_throttle(
                2.0,
                "task2_depth_xy: waiting for same-run mode-0 reference seed=%s, "
                "current seed=%s",
                reference_payload.get("seed"),
                truth_payload.get("seed"),
            )
            return
        truth_objects = self._truth_objects(truth_payload)
        reference_objects = self._reference_objects(reference_payload)
        if truth_objects is None or reference_objects is None:
            return
        transform = self._camera_to_base_transform()
        if transform is None:
            return
        depth = self.latest_depth
        depth_stamp = self.latest_depth_stamp
        depth_sequence = self.depth_sequence
        color = self.latest_color
        color_stamp = self.latest_color_stamp
        if self.green_filter_enabled:
            if (
                depth_stamp is not None
                and color_stamp is not None
                and self.color_depth_max_skew > 0.0
            ):
                skew = abs((depth_stamp - color_stamp).to_sec())
                if skew > self.color_depth_max_skew:
                    rospy.logwarn_throttle(
                        2.0,
                        "task2_depth_xy: waiting for synchronized RGB-D, skew=%.3f s",
                        skew,
                    )
                    return
            table_green_pixels = green_table_mask(
                color,
                depth.shape,
                self.green_hsv_lower,
                self.green_hsv_upper,
                self.green_dilate_pixels,
                self.green_chroma_min,
                self.green_chroma_dominance_min,
                self.green_shadow_min_intensity_sum,
            )
        else:
            table_green_pixels = np.zeros(depth.shape, dtype=bool)
        color_bgr = color
        if color_bgr is not None and color_bgr.shape[:2] != depth.shape[:2]:
            color_bgr = cv2.resize(
                color_bgr,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        intrinsics = _scaled_intrinsics(self.camera_info, depth.shape)
        plane = self._fit_table_plane(depth, truth_objects, intrinsics, transform)
        self.processed_depth_sequence = depth_sequence
        if plane is None:
            rospy.logwarn_throttle(2.0, "task2_depth_xy: table plane fit failed")
            return

        yaw_prior_frame_offset = 0.0
        try:
            yaw_prior_frame_offset = -yaw_from_quat_wxyz(
                truth_payload.get("world_base_quat_wxyz", [])
            )
        except (TypeError, ValueError) as exc:
            rospy.logwarn_throttle(
                2.0,
                "task2_depth_xy: base yaw unavailable for prior search: %s",
                exc,
            )
            return

        def fit_object(name, forced_yaw_prior_name=None):
            return self._fit_object(
                truth_objects[name],
                depth,
                color_bgr,
                intrinsics,
                transform,
                plane,
                table_green_pixels,
                yaw_prior_frame_offset,
                forced_yaw_prior_name=forced_yaw_prior_name,
            )

        raw_fits = {}
        frame_joint_assignments = {}
        if self.instance_naming_method == INSTANCE_NAMING_RELATIVE_POSITION:
            window_fits = {}
            for name in OBJECT_NAMES:
                class_name = str(truth_objects[name].get("class_name", ""))
                window_fits[name] = {
                    prior_name: fit_object(
                        name, forced_yaw_prior_name=prior_name
                    )
                    for prior_name, _ in yaw_prior_candidates(class_name)
                }

            for class_name in CLASS_NAMES:
                class_objects = [
                    name
                    for name in OBJECT_NAMES
                    if truth_objects[name].get("class_name") == class_name
                ]
                decision = joint_yaw_window_assignment(
                    class_name, class_objects, window_fits
                )
                if decision is None:
                    frame_joint_assignments[class_name] = {
                        "selected_mode": "unavailable",
                        "assignment": {},
                        "direct_cost": None,
                        "swapped_cost": None,
                        "margin": None,
                    }
                    for name in class_objects:
                        raw_fits[name] = {
                            "accepted": False,
                            "rejection_reason": (
                                "joint_yaw_window_assignment_failed"
                            ),
                        }
                    continue

                frame_joint_assignments[class_name] = copy.deepcopy(decision)
                for name, prior_name in decision["assignment"].items():
                    selected_fit = copy.deepcopy(window_fits[name][prior_name])
                    selected_fit["yaw_prior_selected_name"] = prior_name
                    selected_fit["yaw_joint_assignment_mode"] = decision[
                        "selected_mode"
                    ]
                    selected_fit["yaw_joint_selected_cost"] = decision[
                        "selected_cost"
                    ]
                    selected_fit["yaw_joint_direct_cost"] = decision[
                        "direct_cost"
                    ]
                    selected_fit["yaw_joint_swapped_cost"] = decision[
                        "swapped_cost"
                    ]
                    selected_fit["yaw_joint_assignment_margin"] = decision[
                        "margin"
                    ]
                    raw_fits[name] = selected_fit
        else:
            for name in OBJECT_NAMES:
                raw_fits[name] = fit_object(name)

            for class_name in CLASS_NAMES:
                class_objects = [
                    name
                    for name in OBJECT_NAMES
                    if truth_objects[name].get("class_name") == class_name
                ]
                assignment = joint_yaw_prior_assignment(
                    class_name, class_objects, raw_fits
                )
                if assignment is None:
                    for name in class_objects:
                        raw_fits[name] = {
                            "accepted": False,
                            "rejection_reason": (
                                "joint_yaw_prior_assignment_failed"
                            ),
                        }
                    continue
                for name, forced_prior_name in assignment.items():
                    fit = raw_fits.get(name)
                    selected_name = (
                        fit.get("yaw_prior_selected_name", "")
                        if fit is not None
                        else ""
                    )
                    if selected_name == forced_prior_name:
                        continue
                    rospy.logwarn_throttle(
                        2.0,
                        "task2_depth_xy: joint yaw-prior assignment overrides "
                        "%s from %s to %s",
                        name,
                        selected_name or "none",
                        forced_prior_name,
                    )
                    forced_fit = fit_object(
                        name, forced_yaw_prior_name=forced_prior_name
                    )
                    if (
                        forced_fit is None
                        or forced_fit.get("yaw_prior_selected_name")
                        != forced_prior_name
                    ):
                        raw_fits[name] = {
                            "accepted": False,
                            "rejection_reason": (
                                "joint_yaw_prior_assignment_failed"
                            ),
                        }
                    else:
                        raw_fits[name] = forced_fit

        current_fits = {}
        for name in OBJECT_NAMES:
            truth_item = truth_objects[name]
            current_fits[name] = self._fit_in_mode0_reference(
                raw_fits[name],
                truth_item,
                truth_payload,
                reference_payload,
            )

        complete = all(
            current_fits[name] is not None
            and bool(current_fits[name].get("accepted", False))
            for name in OBJECT_NAMES
        )
        accepted = False
        if complete:
            self.valid_samples.append(copy.deepcopy(current_fits))
            accepted = True

        locked = len(self.valid_samples) >= self.median_frames
        if locked:
            fit_status = "locked"
        elif accepted:
            fit_status = "collecting"
        else:
            fit_status = "incomplete_frame"

        output_objects = []
        for name in OBJECT_NAMES:
            # RQT previews the current depth frame until the final 5-frame
            # coordinate median is locked for motion.
            displayed_fit = (
                self._aggregate_fit(name) if locked else current_fits.get(name)
            )
            output_objects.append(
                self._make_output_object(
                    truth_objects[name],
                    reference_objects[name],
                    displayed_fit,
                    locked,
                    fit_status,
                )
            )
        output_objects, active_naming_method = self._assign_output_motion_names(
            output_objects
        )
        payload = {
            "stamp": rospy.Time.now().to_sec(),
            "depth_stamp": (
                depth_stamp.to_sec() if depth_stamp is not None else None
            ),
            "color_stamp": (
                color_stamp.to_sec() if color_stamp is not None else None
            ),
            "frame_id": FRAME_ID,
            "camera_frame_id": self.latest_depth_frame,
            "scene": "scene2",
            "seed": truth_payload.get("seed"),
            "vision_mode": 1,
            "vision_mode_name": "rgbd_cad_pose",
            "frozen": locked,
            "locked": locked,
            "position_source": POSE_SOURCE,
            "z_position_source": Z_POSITION_SOURCE,
            "yaw_source": POSE_SOURCE,
            "ready_for_motion": locked,
            "selection_method": (
                "xyz_median_circular_yaw_and_{}_instance_matching".format(
                    self.instance_naming_method
                )
            ),
            "instance_naming_method": active_naming_method,
            "requested_instance_naming_method": self.instance_naming_method,
            "yaw_joint_assignments": frame_joint_assignments,
            "selection_sample_count": len(self.valid_samples),
            "required_valid_frames": self.median_frames,
            "motion_to_ground_truth": {
                item["name"]: item.get("ground_truth_name", item["name"])
                for item in output_objects
            },
            "motion_object_order": list(OBJECT_NAMES),
            "mode0_reference_stamp": reference_payload.get("stamp"),
            "mode0_reference_world_base_position": reference_payload.get(
                "world_base_position", []
            ),
            "mode0_reference_world_base_quat_wxyz": reference_payload.get(
                "world_base_quat_wxyz", []
            ),
            "fit_world_base_position": truth_payload.get(
                "world_base_position", []
            ),
            "fit_world_base_quat_wxyz": truth_payload.get(
                "world_base_quat_wxyz", []
            ),
            "truth_reference": "same_run_mode0_equivalent_input",
            "fit_coordinate_reference": "same_run_mode0_equivalent_input",
            "fit_method": "pca_cad_region_depth_xy_yaw_fit",
            "green_table_filter": self.green_filter_enabled,
            "green_table_filter_method": (
                "opencv_hsv_chromaticity_shadow_hard_rejection"
            ),
            "table_plane": plane,
            "objects": output_objects,
        }
        self.objects_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=True, sort_keys=True))
        )
        if locked:
            self.locked_payload = copy.deepcopy(payload)
            rospy.loginfo(
                "task2_depth_xy: locked coordinate median from %d complete valid frames",
                len(self.valid_samples),
            )
            for item in output_objects:
                rospy.loginfo(
                    "task2_depth_xy: locked motion=%s ground_truth=%s "
                    "fit=%s yaw=%.2f deg "
                    "truth=%s yaw=%.2f deg error_xyz=%+.2f/%+.2f/%+.2f mm "
                    "yaw_error=%+.2f deg "
                    "render_depth=%.2f mm coverage=%.3f/%.3f iou=%.3f "
                    "GN=%d depth_inliers=%d score=%.3f",
                    item["name"],
                    item.get("ground_truth_name", ""),
                    [round(value, 6) for value in item["fit_position"]],
                    float(item["fit_yaw_deg"]),
                    [round(value, 6) for value in item["truth_position"][:3]],
                    float(item["truth_yaw_deg"]),
                    1000.0 * float(item["x_error_m"]),
                    1000.0 * float(item["y_error_m"]),
                    1000.0 * float(item["z_error_m"]),
                    float(item["yaw_error_deg"]),
                    1000.0 * float(item["fit_render_depth_error_m"]),
                    float(item["fit_render_coverage"]),
                    float(item["fit_render_observed_coverage"]),
                    float(item["fit_render_iou"]),
                    int(item["fit_alignment_iterations"]),
                    int(item["fit_alignment_depth_inliers"]),
                    float(item["fit_render_score"]),
                )
        x_error_values = [
            1000.0 * abs(float(item["x_error_m"]))
            for item in output_objects
            if len(item.get("fit_position", [])) >= 3
        ]
        y_error_values = [
            1000.0 * abs(float(item["y_error_m"]))
            for item in output_objects
            if len(item.get("fit_position", [])) >= 3
        ]
        z_error_values = [
            1000.0 * abs(float(item["z_error_m"]))
            for item in output_objects
            if len(item.get("fit_position", [])) >= 3
        ]
        yaw_error_values = [
            abs(float(item["yaw_error_deg"]))
            for item in output_objects
            if len(item.get("fit_position", [])) >= 3
        ]
        error_text = (
                "abs_x/y/z={:.2f}/{:.2f}/{:.2f} mm yaw={:.2f} deg".format(
                float(np.mean(x_error_values)),
                float(np.mean(y_error_values)),
                float(np.mean(z_error_values)),
                float(np.mean(yaw_error_values)),
            )
            if x_error_values and y_error_values and z_error_values and yaw_error_values
            else "warming up"
        )
        rospy.loginfo_throttle(
            2.0,
            "task2_depth_xy: valid_frames=%d/%d locked=%s mean_error=%s plane_rmse=%.2f mm",
            len(self.valid_samples),
            self.median_frames,
            locked,
            error_text,
            1000.0 * float(plane["rmse"]),
        )

    def spin(self):
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            started = time.monotonic()
            try:
                self.process_once()
            except Exception as exc:
                rospy.logerr_throttle(2.0, "task2_depth_xy: processing failed: %s", exc)
            elapsed = time.monotonic() - started
            if elapsed > 1.0 / self.publish_hz:
                rospy.logwarn_throttle(
                    2.0, "task2_depth_xy: processing took %.3f s", elapsed
                )
            rate.sleep()


def main():
    rospy.init_node("task2_depth_xy")
    try:
        Task2DepthXYNode().spin()
    except rospy.ROSInterruptException:
        return
    except Exception as exc:
        if rospy.is_shutdown():
            return
        rospy.logfatal("task2_depth_xy: startup failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
