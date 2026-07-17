"""Constrained region and depth Gauss-Newton alignment for task 2 CAD models."""

import cv2
import numpy as np


def sample_triangle_surfaces(
    triangles, sample_count, random_seed, return_normals=False
):
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape Nx3x3")
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    crosses = np.cross(edges_a, edges_b)
    cross_norms = np.linalg.norm(crosses, axis=1)
    usable = cross_norms > 1e-12
    triangles = triangles[usable]
    crosses = crosses[usable]
    cross_norms = cross_norms[usable]
    if triangles.shape[0] == 0:
        raise ValueError("CAD mesh contains no non-degenerate triangles")

    areas = 0.5 * cross_norms
    face_normals = crosses / cross_norms[:, None]
    count = max(int(sample_count), triangles.shape[0])
    rng = np.random.RandomState(int(random_seed))
    selected = rng.choice(
        triangles.shape[0], count, replace=True, p=areas / np.sum(areas)
    )
    chosen = triangles[selected]
    root_u = np.sqrt(rng.random_sample(count))
    v = rng.random_sample(count)
    samples = (
        (1.0 - root_u)[:, None] * chosen[:, 0]
        + (root_u * (1.0 - v))[:, None] * chosen[:, 1]
        + (root_u * v)[:, None] * chosen[:, 2]
    )
    points = np.vstack([samples, triangles.reshape(-1, 3)])
    normals = np.vstack(
        [face_normals[selected], np.repeat(face_normals, 3, axis=0)]
    )
    if return_normals:
        return points, normals
    return points


def _trimmed_mean(values, keep_fraction, clip_max=None):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    if clip_max is not None:
        values = np.minimum(values, float(clip_max))
    keep_count = max(1, int(round(values.size * float(keep_fraction))))
    if keep_count < values.size:
        values = np.partition(values, keep_count - 1)[:keep_count]
    return float(np.mean(values))


def _huber_weights(residuals, delta):
    residuals = np.asarray(residuals, dtype=np.float64)
    absolute = np.abs(residuals)
    weights = np.ones_like(absolute)
    outside = absolute > float(delta)
    weights[outside] = float(delta) / np.maximum(absolute[outside], 1e-12)
    return weights


def _bilinear_sample(image, pixels_u, pixels_v):
    image = np.asarray(image, dtype=np.float64)
    pixels_u = np.asarray(pixels_u, dtype=np.float64)
    pixels_v = np.asarray(pixels_v, dtype=np.float64)
    height, width = image.shape[:2]
    valid = (
        np.isfinite(pixels_u)
        & np.isfinite(pixels_v)
        & (pixels_u >= 0.0)
        & (pixels_v >= 0.0)
        & (pixels_u <= width - 1.001)
        & (pixels_v <= height - 1.001)
    )
    result = np.full(pixels_u.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return result, valid
    u = pixels_u[valid]
    v = pixels_v[valid]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)
    du = u - u0
    dv = v - v0
    result[valid] = (
        (1.0 - du) * (1.0 - dv) * image[v0, u0]
        + du * (1.0 - dv) * image[v0, u1]
        + (1.0 - du) * dv * image[v1, u0]
        + du * dv * image[v1, u1]
    )
    return result, valid


def _signed_distance_field(mask):
    mask = np.asarray(mask, dtype=bool)
    outside = cv2.distanceTransform(
        (~mask).astype(np.uint8), cv2.DIST_L2, 5
    ).astype(np.float64)
    inside = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, 5
    ).astype(np.float64)
    signed = outside - inside
    gradient_v, gradient_u = np.gradient(signed)
    return signed, gradient_u, gradient_v


def _nearest_pixel_lookup(mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    distances, labels = cv2.distanceTransformWithLabels(
        (~mask).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    pixels_v, pixels_u = np.nonzero(mask)
    pixel_labels = labels[pixels_v, pixels_u]
    label_to_pixel = np.full(int(np.max(labels)) + 1, -1, dtype=np.int32)
    label_to_pixel[pixel_labels] = np.arange(pixels_u.size, dtype=np.int32)
    return {
        "distances": distances.astype(np.float64),
        "labels": labels,
        "label_to_pixel": label_to_pixel,
        "pixels_u": pixels_u,
        "pixels_v": pixels_v,
    }


class CadDepthAligner:
    def __init__(
        self,
        triangles_by_name,
        sample_count=8000,
        random_seed=42,
        maximum_iterations=8,
        region_sigma_px=2.0,
        depth_sigma=0.003,
        region_weight=1.0,
        depth_weight=1.0,
        robust_delta=2.5,
        maximum_depth_residual=0.04,
        maximum_correspondence_pixels=4.0,
        maximum_step=0.02,
        maximum_translation=0.08,
        convergence_translation=0.0002,
        damping=0.001,
        rendered_mask_dilate_pixels=1,
        maximum_region_points=600,
        maximum_depth_points=1600,
        minimum_rendered_pixels=20,
        minimum_overlap_ratio=0.45,
        minimum_iou=0.30,
    ):
        self.surface_points = {}
        self.surface_normals = {}
        for index, (name, triangles) in enumerate(sorted(triangles_by_name.items())):
            points, normals = sample_triangle_surfaces(
                triangles,
                sample_count,
                random_seed + index * 1009,
                return_normals=True,
            )
            self.surface_points[str(name)] = points
            self.surface_normals[str(name)] = normals
        self.maximum_iterations = max(1, int(maximum_iterations))
        self.region_sigma_px = max(1e-6, float(region_sigma_px))
        self.depth_sigma = max(1e-6, float(depth_sigma))
        self.region_weight = max(0.0, float(region_weight))
        self.depth_weight = max(0.0, float(depth_weight))
        self.robust_delta = max(0.1, float(robust_delta))
        self.maximum_depth_residual = max(
            self.depth_sigma, float(maximum_depth_residual)
        )
        self.maximum_correspondence_pixels = max(
            0.0, float(maximum_correspondence_pixels)
        )
        self.maximum_step = max(1e-6, float(maximum_step))
        self.maximum_translation = max(self.maximum_step, float(maximum_translation))
        self.convergence_translation = max(
            1e-8, float(convergence_translation)
        )
        self.damping = max(0.0, float(damping))
        self.rendered_mask_dilate_pixels = max(
            0, int(rendered_mask_dilate_pixels)
        )
        self.maximum_region_points = max(20, int(maximum_region_points))
        self.maximum_depth_points = max(20, int(maximum_depth_points))
        self.minimum_rendered_pixels = max(1, int(minimum_rendered_pixels))
        self.minimum_overlap_ratio = float(minimum_overlap_ratio)
        self.minimum_iou = float(minimum_iou)

    @staticmethod
    def _render_depth(
        body_points,
        orientation,
        position,
        intrinsics,
        camera_to_base,
        roi,
        table_plane,
        height_range,
    ):
        points_base = np.asarray(body_points, dtype=np.float64).dot(
            np.asarray(orientation, dtype=np.float64).T
        ) + np.asarray(position, dtype=np.float64)
        coefficients = table_plane["coefficients"]
        table_z = (
            coefficients[0] * points_base[:, 0]
            + coefficients[1] * points_base[:, 1]
            + coefficients[2]
        )
        heights = points_base[:, 2] - table_z
        points_base = points_base[
            (heights >= height_range[0]) & (heights <= height_range[1])
        ]
        if points_base.shape[0] == 0:
            return None
        camera_rotation, camera_translation = camera_to_base
        points_camera = (points_base - camera_translation).dot(camera_rotation)
        points_camera = points_camera[points_camera[:, 2] > 0.03]
        if points_camera.shape[0] == 0:
            return None
        fx, fy, cx, cy = intrinsics[:4]
        pixels_u = np.rint(
            fx * points_camera[:, 0] / points_camera[:, 2] + cx
        ).astype(np.int32)
        pixels_v = np.rint(
            fy * points_camera[:, 1] / points_camera[:, 2] + cy
        ).astype(np.int32)
        x1, y1, x2, y2 = [int(value) for value in roi]
        inside = (
            (pixels_u >= x1)
            & (pixels_u < x2)
            & (pixels_v >= y1)
            & (pixels_v < y2)
        )
        width = x2 - x1
        height = y2 - y1
        rendered = np.full(height * width, np.inf, dtype=np.float64)
        if np.any(inside):
            flat = (pixels_v[inside] - y1) * width + pixels_u[inside] - x1
            np.minimum.at(rendered, flat, points_camera[inside, 2])
        return rendered.reshape(height, width)

    def _project_visible(
        self,
        name,
        xy,
        body_z,
        orientation,
        intrinsics,
        camera_to_base,
        roi,
        table_plane,
        height_range,
    ):
        body_points = self.surface_points[str(name)]
        body_normals = self.surface_normals[str(name)]
        body_rotation = np.asarray(orientation, dtype=np.float64)
        points_base = body_points.dot(body_rotation.T) + np.array(
            [xy[0], xy[1], body_z], dtype=np.float64
        )
        normals_base = body_normals.dot(body_rotation.T)
        coefficients = table_plane["coefficients"]
        table_z = (
            coefficients[0] * points_base[:, 0]
            + coefficients[1] * points_base[:, 1]
            + coefficients[2]
        )
        heights = points_base[:, 2] - table_z
        keep = (heights >= height_range[0]) & (heights <= height_range[1])
        points_base = points_base[keep]
        normals_base = normals_base[keep]
        if points_base.shape[0] == 0:
            return None

        camera_rotation, camera_translation = camera_to_base
        points_camera = (points_base - camera_translation).dot(camera_rotation)
        normals_camera = normals_base.dot(camera_rotation)
        front = points_camera[:, 2] > 0.03
        points_camera = points_camera[front]
        normals_camera = normals_camera[front]
        if points_camera.shape[0] == 0:
            return None

        fx, fy, cx, cy = intrinsics[:4]
        pixels_u_global = fx * points_camera[:, 0] / points_camera[:, 2] + cx
        pixels_v_global = fy * points_camera[:, 1] / points_camera[:, 2] + cy
        x1, y1, x2, y2 = [int(value) for value in roi]
        pixels_u = pixels_u_global - x1
        pixels_v = pixels_v_global - y1
        pixels_u_int = np.rint(pixels_u).astype(np.int32)
        pixels_v_int = np.rint(pixels_v).astype(np.int32)
        width = x2 - x1
        height = y2 - y1
        inside = (
            (pixels_u_int >= 0)
            & (pixels_u_int < width)
            & (pixels_v_int >= 0)
            & (pixels_v_int < height)
        )
        points_camera = points_camera[inside]
        normals_camera = normals_camera[inside]
        pixels_u = pixels_u[inside]
        pixels_v = pixels_v[inside]
        pixels_u_int = pixels_u_int[inside]
        pixels_v_int = pixels_v_int[inside]
        if points_camera.shape[0] == 0:
            return None

        flat = pixels_v_int * width + pixels_u_int
        order = np.lexsort((points_camera[:, 2], flat))
        sorted_flat = flat[order]
        first = np.empty(sorted_flat.size, dtype=bool)
        first[0] = True
        first[1:] = sorted_flat[1:] != sorted_flat[:-1]
        visible = order[first]
        points_camera = points_camera[visible]
        normals_camera = normals_camera[visible]
        pixels_u = pixels_u[visible]
        pixels_v = pixels_v[visible]
        pixels_u_int = pixels_u_int[visible]
        pixels_v_int = pixels_v_int[visible]

        visible_mask = np.zeros((height, width), dtype=bool)
        visible_mask[pixels_v_int, pixels_u_int] = True
        rendered_depth = np.full((height, width), np.nan, dtype=np.float64)
        rendered_depth[pixels_v_int, pixels_u_int] = points_camera[:, 2]
        rendered_mask = self._filled_render_mask(visible_mask)
        return {
            "points_camera": points_camera,
            "normals_camera": normals_camera,
            "pixels_u": pixels_u,
            "pixels_v": pixels_v,
            "pixels_u_int": pixels_u_int,
            "pixels_v_int": pixels_v_int,
            "visible_mask": visible_mask,
            "rendered_mask": rendered_mask,
            "rendered_depth": rendered_depth,
        }

    def _filled_render_mask(self, visible_mask):
        image = np.asarray(visible_mask, dtype=np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if self.rendered_mask_dilate_pixels > 0:
            radius = self.rendered_mask_dilate_pixels
            size = 2 * radius + 1
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (size, size)
            )
            image = cv2.dilate(image, dilate_kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        filled = np.zeros_like(image)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled > 0

    @staticmethod
    def _projection_jacobian(points_camera, intrinsics, camera_rotation):
        points_camera = np.asarray(points_camera, dtype=np.float64)
        fx, fy = float(intrinsics[0]), float(intrinsics[1])
        basis = np.asarray(camera_rotation, dtype=np.float64)[:2, :]
        x = points_camera[:, 0]
        y = points_camera[:, 1]
        z = points_camera[:, 2]
        z_squared = np.maximum(z * z, 1e-12)
        jacobian_u = np.column_stack(
            [
                fx * (basis[index, 0] * z - x * basis[index, 2]) / z_squared
                for index in range(2)
            ]
        )
        jacobian_v = np.column_stack(
            [
                fy * (basis[index, 1] * z - y * basis[index, 2]) / z_squared
                for index in range(2)
            ]
        )
        return jacobian_u, jacobian_v

    def _contour_model_indices(self, rendering):
        contours, _ = cv2.findContours(
            rendering["rendered_mask"].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            return np.empty(0, dtype=np.int32)
        contour_pixels = np.concatenate(
            [contour.reshape(-1, 2) for contour in contours], axis=0
        )
        nearest = _nearest_pixel_lookup(rendering["visible_mask"])
        if nearest is None:
            return np.empty(0, dtype=np.int32)
        labels = nearest["labels"][
            contour_pixels[:, 1], contour_pixels[:, 0]
        ]
        mapped = nearest["label_to_pixel"][labels]
        distances = nearest["distances"][
            contour_pixels[:, 1], contour_pixels[:, 0]
        ]
        mapped = mapped[(mapped >= 0) & (distances <= 3.0)]
        indices = np.unique(mapped)
        if indices.size > self.maximum_region_points:
            select = np.linspace(
                0, indices.size - 1, self.maximum_region_points
            ).astype(np.int32)
            indices = indices[select]
        return indices

    @staticmethod
    def _accumulate_normal_equations(residuals, jacobian, weight, robust_delta):
        if residuals.size == 0 or weight <= 0.0:
            return np.zeros((2, 2)), np.zeros(2), 0.0
        robust = _huber_weights(residuals, robust_delta)
        scale = float(weight) / float(max(1, residuals.size))
        weighted_jacobian = robust[:, None] * jacobian
        hessian = scale * jacobian.T.dot(weighted_jacobian)
        gradient = scale * jacobian.T.dot(robust * residuals)
        cost = float(np.sum(robust * residuals * residuals) / np.sum(robust))
        return hessian, gradient, cost

    def _region_terms(
        self,
        rendering,
        signed_distance,
        gradient_u,
        gradient_v,
        intrinsics,
        camera_rotation,
    ):
        indices = self._contour_model_indices(rendering)
        if indices.size == 0:
            return np.empty(0), np.empty((0, 2))
        pixels_u = rendering["pixels_u"][indices]
        pixels_v = rendering["pixels_v"][indices]
        residuals, valid = _bilinear_sample(signed_distance, pixels_u, pixels_v)
        derivative_u, valid_u = _bilinear_sample(gradient_u, pixels_u, pixels_v)
        derivative_v, valid_v = _bilinear_sample(gradient_v, pixels_u, pixels_v)
        valid &= valid_u & valid_v & np.isfinite(residuals)
        points_camera = rendering["points_camera"][indices][valid]
        residuals = residuals[valid] / self.region_sigma_px
        derivative_u = derivative_u[valid]
        derivative_v = derivative_v[valid]
        jacobian_u, jacobian_v = self._projection_jacobian(
            points_camera, intrinsics, camera_rotation
        )
        jacobian = (
            derivative_u[:, None] * jacobian_u
            + derivative_v[:, None] * jacobian_v
        ) / self.region_sigma_px
        finite = np.isfinite(residuals) & np.all(np.isfinite(jacobian), axis=1)
        return residuals[finite], jacobian[finite]

    def _depth_terms(
        self,
        rendering,
        observed_depth,
        observed_mask,
        intrinsics,
        camera_rotation,
        roi,
    ):
        valid_observed = np.asarray(observed_mask, dtype=bool) & np.isfinite(
            observed_depth
        )
        nearest = _nearest_pixel_lookup(valid_observed)
        if nearest is None:
            return np.empty(0), np.empty((0, 2)), 0
        model_u = rendering["pixels_u_int"]
        model_v = rendering["pixels_v_int"]
        labels = nearest["labels"][model_v, model_u]
        mapped = nearest["label_to_pixel"][labels]
        distances = nearest["distances"][model_v, model_u]
        valid = (mapped >= 0) & (
            distances <= self.maximum_correspondence_pixels
        )
        model_indices = np.nonzero(valid)[0]
        if model_indices.size > self.maximum_depth_points:
            select = np.linspace(
                0, model_indices.size - 1, self.maximum_depth_points
            ).astype(np.int32)
            model_indices = model_indices[select]
        if model_indices.size == 0:
            return np.empty(0), np.empty((0, 2)), 0
        observed_indices = mapped[model_indices]
        observed_u = nearest["pixels_u"][observed_indices]
        observed_v = nearest["pixels_v"][observed_indices]
        observed_z = observed_depth[observed_v, observed_u]
        x1, y1 = float(roi[0]), float(roi[1])
        fx, fy, cx, cy = [float(value) for value in intrinsics[:4]]
        observed_points = np.column_stack(
            [
                (observed_u + x1 - cx) * observed_z / fx,
                (observed_v + y1 - cy) * observed_z / fy,
                observed_z,
            ]
        )
        model_points = rendering["points_camera"][model_indices]
        model_normals = rendering["normals_camera"][model_indices]
        residuals_m = np.einsum(
            "ij,ij->i", model_normals, model_points - observed_points
        )
        basis = np.asarray(camera_rotation, dtype=np.float64)[:2, :]
        jacobian_m = model_normals.dot(basis.T)
        valid = (
            np.isfinite(residuals_m)
            & (np.abs(residuals_m) <= self.maximum_depth_residual)
            & np.all(np.isfinite(jacobian_m), axis=1)
        )
        residuals = residuals_m[valid] / self.depth_sigma
        jacobian = jacobian_m[valid] / self.depth_sigma
        return residuals, jacobian, int(np.count_nonzero(valid))

    def _centroid_pre_alignment(
        self, xy, rendering, observed_mask, intrinsics, camera_rotation
    ):
        rendered_pixels = np.argwhere(rendering["rendered_mask"])
        observed_pixels = np.argwhere(observed_mask)
        if rendered_pixels.shape[0] < 3 or observed_pixels.shape[0] < 3:
            return np.asarray(xy, dtype=np.float64)
        pixel_error = np.array(
            [
                np.mean(observed_pixels[:, 1]) - np.mean(rendered_pixels[:, 1]),
                np.mean(observed_pixels[:, 0]) - np.mean(rendered_pixels[:, 0]),
            ],
            dtype=np.float64,
        )
        center_camera = np.mean(rendering["points_camera"], axis=0, keepdims=True)
        jacobian_u, jacobian_v = self._projection_jacobian(
            center_camera, intrinsics, camera_rotation
        )
        pixel_jacobian = np.vstack([jacobian_u[0], jacobian_v[0]])
        if np.linalg.matrix_rank(pixel_jacobian) < 2:
            return np.asarray(xy, dtype=np.float64)
        delta = np.linalg.lstsq(pixel_jacobian, pixel_error, rcond=None)[0]
        norm = float(np.linalg.norm(delta))
        if norm > self.maximum_step:
            delta *= self.maximum_step / norm
        return np.asarray(xy, dtype=np.float64) + delta

    def _metrics(self, rendering, observed_depth, observed_mask):
        rendered_mask = rendering["rendered_mask"]
        observed_mask = np.asarray(observed_mask, dtype=bool)
        intersection = rendered_mask & observed_mask
        union = rendered_mask | observed_mask
        rendered_count = int(np.count_nonzero(rendered_mask))
        observed_count = int(np.count_nonzero(observed_mask))
        intersection_count = int(np.count_nonzero(intersection))
        coverage = float(intersection_count) / float(max(1, rendered_count))
        observed_coverage = float(intersection_count) / float(max(1, observed_count))
        iou = float(intersection_count) / float(max(1, np.count_nonzero(union)))

        rendered_boundary = rendered_mask & ~cv2.erode(
            rendered_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        observed_boundary = observed_mask & ~cv2.erode(
            observed_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        observed_distance = cv2.distanceTransform(
            (~observed_boundary).astype(np.uint8), cv2.DIST_L2, 5
        )
        rendered_distance = cv2.distanceTransform(
            (~rendered_boundary).astype(np.uint8), cv2.DIST_L2, 5
        )
        cad_to_observed = _trimmed_mean(
            observed_distance[rendered_boundary], 0.98, clip_max=20.0
        )
        observed_to_cad = _trimmed_mean(
            rendered_distance[observed_boundary], 0.98, clip_max=20.0
        )

        model_u = rendering["pixels_u_int"]
        model_v = rendering["pixels_v_int"]
        same_pixel = np.isfinite(observed_depth[model_v, model_u])
        depth_error = _trimmed_mean(
            np.abs(
                rendering["points_camera"][same_pixel, 2]
                - observed_depth[model_v[same_pixel], model_u[same_pixel]]
            ),
            0.90,
            clip_max=0.04,
        )
        if not np.isfinite(depth_error):
            depth_error = 0.04
        score = (
            self.depth_weight * depth_error / self.depth_sigma
            + self.region_weight * 0.5 * (cad_to_observed + observed_to_cad)
            + 4.0 * (2.0 - coverage - observed_coverage)
        )
        return {
            "score": float(score),
            "depth_error_m": float(depth_error),
            "coverage": coverage,
            "observed_coverage": observed_coverage,
            "iou": iou,
            "cad_to_observed_px": float(cad_to_observed),
            "observed_to_cad_px": float(observed_to_cad),
            "rendered_pixels": rendered_count,
        }

    def _apply_quality_thresholds(
        self, result, minimum_overlap_ratio=None, minimum_iou=None
    ):
        overlap_threshold = (
            self.minimum_overlap_ratio
            if minimum_overlap_ratio is None
            else float(minimum_overlap_ratio)
        )
        iou_threshold = self.minimum_iou if minimum_iou is None else float(minimum_iou)
        minimum_coverage = min(result["coverage"], result["observed_coverage"])
        accepted = True
        rejection_reason = ""
        if minimum_coverage < overlap_threshold:
            accepted = False
            rejection_reason = "bidirectional_coverage_below_threshold"
        elif result["iou"] < iou_threshold:
            accepted = False
            rejection_reason = "iou_below_threshold"
        result.update(
            {
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "minimum_overlap_ratio": overlap_threshold,
                "minimum_iou": iou_threshold,
            }
        )
        return result

    def evaluate_xy(
        self,
        name,
        xy,
        body_z,
        orientation,
        intrinsics,
        camera_to_base,
        roi,
        table_plane,
        height_range,
        observed_depth,
        observed_mask,
        minimum_overlap_ratio=None,
        minimum_iou=None,
    ):
        rendering = self._project_visible(
            name,
            xy,
            body_z,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            table_plane,
            height_range,
        )
        if rendering is None:
            return None
        metrics = self._metrics(rendering, observed_depth, observed_mask)
        if metrics["rendered_pixels"] < self.minimum_rendered_pixels:
            return None
        metrics["xy"] = [float(xy[0]), float(xy[1])]
        return self._apply_quality_thresholds(
            metrics, minimum_overlap_ratio, minimum_iou
        )

    def refine_xy(
        self,
        name,
        initial_xy,
        body_z,
        orientation,
        intrinsics,
        camera_to_base,
        roi,
        table_plane,
        height_range,
        observed_depth,
        observed_mask,
        minimum_overlap_ratio=None,
        minimum_iou=None,
    ):
        initial_xy = np.asarray(initial_xy, dtype=np.float64)
        observed_depth = np.asarray(observed_depth, dtype=np.float64)
        observed_mask = np.asarray(observed_mask, dtype=bool)
        if int(np.count_nonzero(observed_mask)) < self.minimum_rendered_pixels:
            return None
        signed_distance, gradient_u, gradient_v = _signed_distance_field(
            observed_mask
        )
        camera_rotation = np.asarray(camera_to_base[0], dtype=np.float64)
        xy = initial_xy.copy()
        rendering = self._project_visible(
            name,
            xy,
            body_z,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            table_plane,
            height_range,
        )
        if rendering is None:
            return None
        xy = self._centroid_pre_alignment(
            xy, rendering, observed_mask, intrinsics, camera_rotation
        )

        converged = False
        iteration_count = 0
        last_region_count = 0
        last_depth_count = 0
        last_region_cost = 0.0
        last_depth_cost = 0.0
        for iteration in range(self.maximum_iterations):
            rendering = self._project_visible(
                name,
                xy,
                body_z,
                orientation,
                intrinsics,
                camera_to_base,
                roi,
                table_plane,
                height_range,
            )
            if rendering is None:
                return None
            region_residuals, region_jacobian = self._region_terms(
                rendering,
                signed_distance,
                gradient_u,
                gradient_v,
                intrinsics,
                camera_rotation,
            )
            depth_residuals, depth_jacobian, depth_count = self._depth_terms(
                rendering,
                observed_depth,
                observed_mask,
                intrinsics,
                camera_rotation,
                roi,
            )
            if region_residuals.size + depth_residuals.size < 3:
                return None
            region_hessian, region_gradient, region_cost = (
                self._accumulate_normal_equations(
                    region_residuals,
                    region_jacobian,
                    self.region_weight,
                    self.robust_delta,
                )
            )
            depth_hessian, depth_gradient, depth_cost = (
                self._accumulate_normal_equations(
                    depth_residuals,
                    depth_jacobian,
                    self.depth_weight,
                    self.robust_delta,
                )
            )
            hessian = region_hessian + depth_hessian
            gradient = region_gradient + depth_gradient
            if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gradient)):
                return None
            diagonal = np.maximum(np.diag(hessian), 1.0)
            damped_hessian = hessian + self.damping * np.diag(diagonal)
            if np.linalg.matrix_rank(damped_hessian) < 2:
                return None
            delta = -np.linalg.solve(damped_hessian, gradient)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > self.maximum_step:
                delta *= self.maximum_step / delta_norm
                delta_norm = self.maximum_step
            candidate = xy + delta
            total_offset = candidate - initial_xy
            total_norm = float(np.linalg.norm(total_offset))
            if total_norm > self.maximum_translation:
                candidate = initial_xy + total_offset * (
                    self.maximum_translation / total_norm
                )
            xy = candidate
            iteration_count = iteration + 1
            last_region_count = int(region_residuals.size)
            last_depth_count = int(depth_count)
            last_region_cost = float(region_cost)
            last_depth_cost = float(depth_cost)
            if delta_norm <= self.convergence_translation:
                converged = True
                break

        result = self.evaluate_xy(
            name,
            xy,
            body_z,
            orientation,
            intrinsics,
            camera_to_base,
            roi,
            table_plane,
            height_range,
            observed_depth,
            observed_mask,
            minimum_overlap_ratio,
            minimum_iou,
        )
        if result is None:
            return None
        result.update(
            {
                "xy": [float(xy[0]), float(xy[1])],
                "initial_xy": [float(initial_xy[0]), float(initial_xy[1])],
                "offset_xy": [
                    float(xy[0] - initial_xy[0]),
                    float(xy[1] - initial_xy[1]),
                ],
                "evaluations": int(iteration_count + 2),
                "iterations": int(iteration_count),
                "converged": bool(converged),
                "region_points": int(last_region_count),
                "depth_inliers": int(last_depth_count),
                "region_cost": last_region_cost,
                "depth_cost": last_depth_cost,
            }
        )
        return result
