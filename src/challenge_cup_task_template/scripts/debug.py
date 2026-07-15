#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reverse YOLO labels onto collected RGB images for visual inspection."""

import json
import os
import sys

try:
    import cv2
    import numpy as np
except Exception as exc:
    cv2 = None
    np = None
    _CV_IMPORT_ERROR = exc
else:
    _CV_IMPORT_ERROR = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 全局检查参数
# =============================================================================

# Inclusive ranges. Multiple disjoint ranges are allowed.
SEED_RANGES = ((160, 165),)
VIEW_INDICES = (0, 1, 2)
DATASET_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "scene1_parcel_yolo")
)
OUTPUT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "debug_output", "scene1_parcel_yolo")
)
CONTACT_VIEW_SIZE = (640, 360)
JPEG_QUALITY = 95

BOX_COLORS_BGR = (
    (40, 230, 255),
    (60, 220, 80),
    (255, 170, 50),
    (220, 80, 220),
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _find_sample(dataset_root, seed, view_index):
    stem = "seed_{:05d}_view_{}".format(int(seed), int(view_index))
    for split in ("train", "val"):
        image_dir = os.path.join(dataset_root, "images", split)
        for extension in IMAGE_EXTENSIONS:
            image_path = os.path.join(image_dir, stem + extension)
            if not os.path.isfile(image_path):
                continue
            label_path = os.path.join(dataset_root, "labels", split, stem + ".txt")
            metadata_path = os.path.join(
                dataset_root, "metadata", split, stem + ".json"
            )
            return {
                "seed": int(seed),
                "view_index": int(view_index),
                "split": split,
                "stem": stem,
                "image": image_path,
                "label": label_path,
                "metadata": metadata_path,
            }
    return None


def _parse_yolo_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    "{}:{} expects 5 fields, got {}".format(
                        path, line_number, len(fields)
                    )
                )
            try:
                class_id = int(fields[0])
                cx, cy, width, height = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(
                    "{}:{} contains a non-numeric YOLO label".format(path, line_number)
                ) from exc
            values = (cx, cy, width, height)
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(
                    "{}:{} normalized values outside [0, 1]: {}".format(
                        path, line_number, values
                    )
                )
            if width <= 0.0 or height <= 0.0:
                raise ValueError(
                    "{}:{} has non-positive box size".format(path, line_number)
                )
            labels.append(
                {
                    "class_id": class_id,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                    "line_number": line_number,
                }
            )
    return labels


def _to_pixel_box(label, image_width, image_height):
    cx = label["cx"] * image_width
    cy = label["cy"] * image_height
    width = label["width"] * image_width
    height = label["height"] * image_height
    x1 = int(round(max(0.0, cx - width * 0.5)))
    y1 = int(round(max(0.0, cy - height * 0.5)))
    x2 = int(round(min(float(image_width - 1), cx + width * 0.5)))
    y2 = int(round(min(float(image_height - 1), cy + height * 0.5)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("normalized label produced an empty pixel box")
    return x1, y1, x2, y2


def _metadata_objects_by_label_index(metadata):
    result = {}
    if not isinstance(metadata, dict):
        return result
    for obj in metadata.get("objects", []):
        try:
            result[int(obj.get("label_index"))] = obj
        except (TypeError, ValueError):
            continue
    return result


def _sample_header(sample, metadata, label_count):
    text = "seed {:05d}  view {}  {}  {} boxes".format(
        sample["seed"], sample["view_index"], sample["split"], label_count
    )
    if isinstance(metadata, dict):
        command = metadata.get("pitch_command_deg")
        actual = metadata.get("head_actual_pitch_deg")
        try:
            text += "  pitch cmd={:.2f} actual={:.2f} deg".format(
                float(command), float(actual)
            )
        except (TypeError, ValueError):
            pass
    return text


def _draw_text_with_background(image, text, origin, color, scale=0.55):
    x, y = origin
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x1 = max(0, int(x))
    y1 = max(0, int(y - text_height - baseline - 5))
    x2 = min(image.shape[1] - 1, x1 + text_width + 8)
    y2 = min(image.shape[0] - 1, int(y + 2))
    cv2.rectangle(image, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.putText(
        image,
        text,
        (x1 + 4, max(text_height + 2, int(y - baseline - 1))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _annotate_sample(sample):
    if not os.path.isfile(sample["label"]):
        raise FileNotFoundError("label is missing: {}".format(sample["label"]))
    image = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cannot decode image: {}".format(sample["image"]))
    labels = _parse_yolo_labels(sample["label"])
    metadata = _read_json(sample["metadata"])
    metadata_by_index = _metadata_objects_by_label_index(metadata)
    image_height, image_width = image.shape[:2]

    for label_index, label in enumerate(labels):
        x1, y1, x2, y2 = _to_pixel_box(label, image_width, image_height)
        color = BOX_COLORS_BGR[label_index % len(BOX_COLORS_BGR)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        source = metadata_by_index.get(label_index, {}).get("source_name", "")
        object_text = "#{}".format(label_index + 1)
        if source:
            object_text += " p{}".format(str(source).rsplit("_", 1)[-1])
        _draw_text_with_background(
            image, object_text, (x1 + 2, y1 + 21), color, scale=0.48
        )

    header = _sample_header(sample, metadata, len(labels))
    _draw_text_with_background(image, header, (12, 30), (255, 255, 255), scale=0.62)
    return image, labels


def _write_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    extension = os.path.splitext(path)[1].lower()
    params = []
    if extension in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)]
    if not cv2.imwrite(path, image, params):
        raise RuntimeError("failed to write image: {}".format(path))


def _make_contact_sheet(images):
    if not images:
        return None
    images_by_view = dict(images)
    tiles = []
    target_width, target_height = CONTACT_VIEW_SIZE
    for view_index in VIEW_INDICES:
        image = images_by_view.get(view_index)
        if image is None:
            tile = np.full((target_height, target_width, 3), 40, dtype=np.uint8)
            cv2.putText(
                tile,
                "view {} missing".format(view_index),
                (30, target_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
        else:
            tile = cv2.resize(
                image, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
        cv2.putText(
            tile,
            "view {}".format(view_index),
            (12, target_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    return cv2.hconcat(tiles)


def _expand_seed_ranges(seed_ranges):
    seeds = []
    seen = set()
    for start, end in seed_ranges:
        if int(start) < 0 or int(end) < int(start):
            raise ValueError("invalid seed range: {}:{}".format(start, end))
        for seed in range(int(start), int(end) + 1):
            if seed not in seen:
                seeds.append(seed)
                seen.add(seed)
    return seeds


def generate_annotations(dataset_root, output_dir, seed_ranges):
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable: {}".format(_CV_IMPORT_ERROR))
    if not os.path.isdir(dataset_root):
        raise RuntimeError("dataset root does not exist: {}".format(dataset_root))
    os.makedirs(output_dir, exist_ok=True)
    seeds = _expand_seed_ranges(seed_ranges)
    if not seeds:
        raise RuntimeError("no seeds selected")

    summary = {
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "seed_ranges": [[int(start), int(end)] for start, end in seed_ranges],
        "seeds": seeds,
        "images_written": 0,
        "boxes_drawn": 0,
        "missing_samples": [],
        "errors": [],
    }

    for seed in seeds:
        seed_images = []
        for view_index in VIEW_INDICES:
            sample = _find_sample(dataset_root, seed, view_index)
            if sample is None:
                summary["missing_samples"].append(
                    "seed_{:05d}_view_{}".format(seed, view_index)
                )
                continue
            try:
                annotated, labels = _annotate_sample(sample)
                output_path = os.path.join(output_dir, sample["stem"] + ".jpg")
                _write_image(output_path, annotated)
                seed_images.append((view_index, annotated))
                summary["images_written"] += 1
                summary["boxes_drawn"] += len(labels)
                print(
                    "[OK] {}: {} boxes -> {}".format(
                        sample["stem"], len(labels), output_path
                    )
                )
            except Exception as exc:
                message = "{}: {}".format(sample["stem"], exc)
                summary["errors"].append(message)
                print("[ERROR] {}".format(message), file=sys.stderr)

        contact_sheet = _make_contact_sheet(seed_images)
        if contact_sheet is not None:
            contact_path = os.path.join(
                output_dir, "seed_{:05d}_contact.jpg".format(seed)
            )
            _write_image(contact_path, contact_sheet)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "[DONE] images={} boxes={} missing={} errors={} output={}".format(
            summary["images_written"],
            summary["boxes_drawn"],
            len(summary["missing_samples"]),
            len(summary["errors"]),
            output_dir,
        )
    )
    if summary["missing_samples"]:
        print(
            "[WARN] samples not collected yet: {}".format(
                ", ".join(summary["missing_samples"])
            )
        )
    return 1 if summary["errors"] else 0


def main():
    return generate_annotations(DATASET_ROOT, OUTPUT_DIR, SEED_RANGES)


if __name__ == "__main__":
    raise SystemExit(main())
