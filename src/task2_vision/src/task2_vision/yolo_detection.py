"""Task 2 YOLO inference and conversion to class/bbox-only detections."""

import copy

import numpy as np

from task2_vision.layout import OBJECT_NAMES


CLASS_NAME_BY_ID = {
    0: "part_type_a",
    1: "part_type_b",
    2: "part_type_c",
}
EXPECTED_CLASS_COUNTS = {name: 2 for name in CLASS_NAME_BY_ID.values()}


def normalize_model_names(names):
    if isinstance(names, dict):
        normalized = {int(key): str(value) for key, value in names.items()}
    else:
        normalized = {index: str(value) for index, value in enumerate(names)}
    expected = dict(CLASS_NAME_BY_ID)
    if normalized != expected:
        raise ValueError(
            "task2 YOLO classes must be {}, got {}".format(expected, normalized)
        )
    return normalized


def build_task2_detections(
    xyxy,
    confidences,
    class_ids,
    source_shape,
    output_shape,
):
    """Select two boxes per class and convert them to the fitter input schema."""
    source_height, source_width = [int(value) for value in source_shape[:2]]
    output_height, output_width = [int(value) for value in output_shape[:2]]
    if min(source_height, source_width, output_height, output_width) <= 0:
        raise ValueError("YOLO source/output image dimensions must be positive")

    coordinates = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(confidences, dtype=np.float64).reshape(-1)
    labels = np.asarray(class_ids, dtype=np.float64).reshape(-1)
    if not (len(coordinates) == len(scores) == len(labels)):
        raise ValueError("YOLO box, confidence, and class arrays must have equal length")

    scale_x = float(output_width) / float(source_width)
    scale_y = float(output_height) / float(source_height)
    grouped = {class_name: [] for class_name in CLASS_NAME_BY_ID.values()}
    for box, confidence, raw_class_id in zip(coordinates, scores, labels):
        if not np.isfinite(box).all() or not np.isfinite(confidence):
            continue
        class_id = int(raw_class_id)
        if abs(float(raw_class_id) - class_id) > 1e-6:
            continue
        class_name = CLASS_NAME_BY_ID.get(class_id)
        if class_name is None:
            continue
        x1 = max(0.0, min(float(source_width - 1), float(box[0])))
        y1 = max(0.0, min(float(source_height - 1), float(box[1])))
        x2 = max(0.0, min(float(source_width - 1), float(box[2])))
        y2 = max(0.0, min(float(source_height - 1), float(box[3])))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue
        grouped[class_name].append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox_center": [
                    0.5 * (x1 + x2) * scale_x,
                    0.5 * (y1 + y2) * scale_y,
                ],
                "bbox_size": [
                    (x2 - x1) * scale_x,
                    (y2 - y1) * scale_y,
                ],
                "bbox_projected": True,
                "bbox_source": "yolo_rgb",
                "detection_source": "yolo_rgb",
            }
        )

    detections = []
    for class_name in CLASS_NAME_BY_ID.values():
        candidates = sorted(
            grouped[class_name],
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )[: EXPECTED_CLASS_COUNTS[class_name]]
        candidates.sort(
            key=lambda item: (
                float(item["bbox_center"][0]),
                float(item["bbox_center"][1]),
            )
        )
        for index, source in enumerate(candidates, start=1):
            item = copy.deepcopy(source)
            temporary_name = "{}_{}".format(class_name, index)
            item["name"] = temporary_name
            item["motion_name"] = None
            item["detection_name"] = temporary_name
            detections.append(item)
    return detections


def task2_detection_counts(detections):
    counts = {name: 0 for name in CLASS_NAME_BY_ID.values()}
    for item in detections:
        class_name = str(item.get("class_name", ""))
        if class_name in counts:
            counts[class_name] += 1
    return counts


def has_complete_task2_detections(detections):
    return task2_detection_counts(detections) == EXPECTED_CLASS_COUNTS


def remap_fitted_detections_by_assignment(
    detection_objects, fits, source_to_motion_name, naming_method
):
    """Apply per-class naming decisions without coupling unrelated classes."""
    expected_names = set(OBJECT_NAMES)
    if set(detection_objects) != expected_names or set(fits) != expected_names:
        raise ValueError("task2 detection and fit maps must contain all object keys")

    assignment = {
        source_name: str(source_to_motion_name.get(source_name, source_name))
        for source_name in OBJECT_NAMES
    }
    if set(assignment.values()) != expected_names:
        raise ValueError("task2 motion-name assignment must be one-to-one")

    remapped_detections = {}
    remapped_fits = {}
    for source_name in OBJECT_NAMES:
        motion_name = assignment[source_name]
        detection = copy.deepcopy(detection_objects[source_name])
        class_name = str(detection.get("class_name", ""))
        if not motion_name.startswith(class_name + "_"):
            raise ValueError(
                "task2 assignment crosses classes: {} -> {}".format(
                    source_name, motion_name
                )
            )
        detection["name"] = motion_name
        detection["motion_name"] = motion_name
        detection["detection_name"] = source_name
        detection["instance_naming_source_name"] = source_name
        detection["instance_naming_method"] = str(naming_method)
        remapped_detections[motion_name] = detection
        fit = copy.deepcopy(fits[source_name])
        fit["motion_name"] = motion_name
        fit["instance_naming_source_name"] = source_name
        remapped_fits[motion_name] = fit
    return remapped_detections, remapped_fits


class Task2YoloDetector:
    def __init__(
        self,
        model_path,
        confidence=0.25,
        iou=0.70,
        image_size=1280,
        device="",
        half=False,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "task2 vision requires ultralytics in the ROS Python environment"
            ) from exc
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        normalize_model_names(getattr(self.model, "names", {}))
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.device = str(device).strip()
        self.half = bool(half)

    def infer(self, image_bgr, output_shape):
        source = np.asarray(image_bgr)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("YOLO input must be a BGR image")
        kwargs = {
            "source": source,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "classes": sorted(CLASS_NAME_BY_ID),
            "max_det": 12,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.half:
            kwargs["half"] = True
        results = self.model.predict(**kwargs)
        result = results[0] if results else None
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        return build_task2_detections(
            boxes.xyxy.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
            boxes.cls.detach().cpu().numpy(),
            source.shape[:2],
            output_shape,
        )

    def release(self):
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
