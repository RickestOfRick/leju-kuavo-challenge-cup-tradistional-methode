#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将任务二 YOLO 标签还原到最后若干个 seed 的采集图像上。"""

import collections
import math
import os
import re
import sys

import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATASET_ROOT = os.path.join(REPOSITORY_ROOT, "data", "scene2_parts_yolo")
OUTPUT_ROOT = os.path.join(REPOSITORY_ROOT, "yolo_debug")
LAST_SEED_COUNT = 10
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
BOX_THICKNESS = 2
FONT_SCALE = 0.55
FONT_THICKNESS = 1

# OpenCV 使用 BGR；避开与场景桌面接近的绿色。
CLASS_COLORS = (
    (40, 40, 255),
    (255, 120, 20),
    (20, 190, 255),
)

SAMPLE_PATTERN = re.compile(r"^seed_(\d+)_view_(\d+)$")


def _load_class_names(dataset_root):
    path = os.path.join(dataset_root, "classes.txt")
    with open(path, "r", encoding="utf-8") as fh:
        names = [line.strip() for line in fh if line.strip()]
    if not names:
        raise RuntimeError("no class names in {}".format(path))
    return names


def _find_image(dataset_root, split, stem):
    image_directory = os.path.join(dataset_root, "images", split)
    matches = [
        os.path.join(image_directory, stem + extension)
        for extension in IMAGE_EXTENSIONS
        if os.path.isfile(os.path.join(image_directory, stem + extension))
    ]
    if len(matches) > 1:
        raise RuntimeError("multiple images found for {}/{}".format(split, stem))
    return matches[0] if matches else None


def _discover_samples(dataset_root):
    samples_by_seed = collections.defaultdict(list)
    seen_stems = set()
    for split in ("train", "val"):
        label_directory = os.path.join(dataset_root, "labels", split)
        if not os.path.isdir(label_directory):
            continue
        for filename in sorted(os.listdir(label_directory)):
            stem, extension = os.path.splitext(filename)
            if extension.lower() != ".txt":
                continue
            match = SAMPLE_PATTERN.match(stem)
            if match is None:
                continue
            sample_key = (split, stem)
            if sample_key in seen_stems:
                raise RuntimeError("duplicate sample: {}/{}".format(split, stem))
            seen_stems.add(sample_key)

            image_path = _find_image(dataset_root, split, stem)
            if image_path is None:
                print(
                    "[WARN] skip label without image: {}/{}".format(split, filename),
                    file=sys.stderr,
                )
                continue
            seed = int(match.group(1))
            samples_by_seed[seed].append(
                {
                    "seed": seed,
                    "view": int(match.group(2)),
                    "split": split,
                    "stem": stem,
                    "image": image_path,
                    "label": os.path.join(label_directory, filename),
                }
            )

    for samples in samples_by_seed.values():
        samples.sort(key=lambda item: (item["view"], item["split"]))
    return samples_by_seed


def _load_yolo_labels(path, class_count):
    labels = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError(
                    "{}:{} expected 5 fields, got {}".format(
                        path, line_number, len(fields)
                    )
                )
            try:
                class_id = int(fields[0])
                center_x, center_y, box_width, box_height = (
                    float(value) for value in fields[1:]
                )
            except ValueError as exc:
                raise RuntimeError(
                    "{}:{} invalid YOLO label: {}".format(path, line_number, exc)
                )
            values = (center_x, center_y, box_width, box_height)
            if class_id < 0 or class_id >= class_count:
                raise RuntimeError(
                    "{}:{} class id {} outside [0, {})".format(
                        path, line_number, class_id, class_count
                    )
                )
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("{}:{} contains non-finite values".format(path, line_number))
            if box_width <= 0.0 or box_height <= 0.0:
                raise RuntimeError("{}:{} has non-positive box size".format(path, line_number))
            labels.append((class_id, center_x, center_y, box_width, box_height))
    return labels


def _pixel_box(label, image_width, image_height):
    _, center_x, center_y, box_width, box_height = label
    x1 = int(round((center_x - 0.5 * box_width) * image_width))
    y1 = int(round((center_y - 0.5 * box_height) * image_height))
    x2 = int(round((center_x + 0.5 * box_width) * image_width))
    y2 = int(round((center_y + 0.5 * box_height) * image_height))
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("YOLO box is outside the image after clipping")
    return x1, y1, x2, y2


def _draw_label(image, box, text, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
    )
    label_height = text_height + baseline + 6
    label_top = y1 - label_height if y1 >= label_height else y1
    label_bottom = min(image.shape[0] - 1, label_top + label_height)
    label_right = min(image.shape[1] - 1, x1 + text_width + 8)
    cv2.rectangle(
        image,
        (x1, label_top),
        (label_right, label_bottom),
        color,
        thickness=-1,
    )
    text_y = min(label_bottom - baseline - 3, image.shape[0] - baseline - 1)
    cv2.putText(
        image,
        text,
        (x1 + 4, max(text_height, text_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _render_sample(sample, class_names, output_directory):
    image = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cannot decode image: {}".format(sample["image"]))
    labels = _load_yolo_labels(sample["label"], len(class_names))
    image_height, image_width = image.shape[:2]
    for label in labels:
        class_id = label[0]
        box = _pixel_box(label, image_width, image_height)
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        _draw_label(image, box, class_names[class_id], color)

    header = "seed={}  view={}  split={}  boxes={}".format(
        sample["seed"], sample["view"], sample["split"], len(labels)
    )
    cv2.putText(
        image,
        header,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        header,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_path = os.path.join(output_directory, sample["stem"] + ".jpg")
    if not cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError("cannot write image: {}".format(output_path))
    return output_path, len(labels)


def main():
    if LAST_SEED_COUNT <= 0:
        raise RuntimeError("LAST_SEED_COUNT must be positive")
    if not os.path.isdir(DATASET_ROOT):
        raise RuntimeError("task2 YOLO dataset does not exist: {}".format(DATASET_ROOT))

    class_names = _load_class_names(DATASET_ROOT)
    samples_by_seed = _discover_samples(DATASET_ROOT)
    available_seeds = sorted(samples_by_seed)
    if not available_seeds:
        raise RuntimeError("no paired task2 YOLO images and labels found")
    selected_seeds = available_seeds[-LAST_SEED_COUNT:]
    output_directory = os.path.join(
        OUTPUT_ROOT,
        "seed_{:05d}_to_{:05d}".format(selected_seeds[0], selected_seeds[-1]),
    )
    os.makedirs(output_directory, exist_ok=True)

    image_count = 0
    box_count = 0
    for seed in selected_seeds:
        for sample in samples_by_seed[seed]:
            output_path, sample_box_count = _render_sample(
                sample, class_names, output_directory
            )
            image_count += 1
            box_count += sample_box_count
            print(
                "[OK] seed={} view={} boxes={} -> {}".format(
                    seed, sample["view"], sample_box_count, output_path
                )
            )

    print(
        "[DONE] seeds={} images={} boxes={} output={}".format(
            selected_seeds, image_count, box_count, output_directory
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        sys.exit(1)
