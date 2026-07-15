#!/usr/bin/env python3
"""
Data augmentation script for manual_dataset YOLO-format labels.

Reads manual_dataset/images/ + manual_dataset/labels/,
applies 12 bbox-safe augmentation pipelines per labeled image,
outputs augmented pairs to manual_dataset_augmented/.

Result: 60 original + 720 augmented = 780 labeled frames (13x).
"""

import os
import sys
import shutil
import glob
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import albumentations as A

# ============== CONFIGURATION ==============

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_IMAGES_DIR = PROJECT_ROOT / "manual_dataset" / "images"
SRC_LABELS_DIR = PROJECT_ROOT / "manual_dataset" / "labels"
OUT_DIR = PROJECT_ROOT / "manual_dataset_augmented"
OUT_IMAGES_DIR = OUT_DIR / "images"
OUT_LABELS_DIR = OUT_DIR / "labels"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
RANDOM_SEED = 42

# ============== AUGMENTATION PIPELINES ==============

def build_pipelines() -> List[Tuple[str, A.Compose]]:
    """Build 12 diverse bbox-safe augmentation pipelines."""

    # Shared bbox params: YOLO format, normalized coords
    bbox_params = A.BboxParams(
        format="yolo",
        min_visibility=0.3,       # drop boxes <30% visible after crop/rotate
        min_area=0.0,
        label_fields=["class_labels"],
    )

    pipelines = []

    # 1. HSV jitter — simulates lighting changes
    pipelines.append(("hsv_jitter", A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=1.0
        ),
    ], bbox_params=bbox_params)))

    # 2. Brightness + Contrast — exposure variation
    pipelines.append(("brightness_contrast", A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=1.0
        ),
    ], bbox_params=bbox_params)))

    # 3. RGB channel shift — color temperature shifts
    pipelines.append(("rgb_shift", A.Compose([
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
    ], bbox_params=bbox_params)))

    # 4. Gaussian blur — defocus / depth-of-field
    pipelines.append(("gaussian_blur", A.Compose([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    ], bbox_params=bbox_params)))

    # 5. Motion blur — camera shake
    pipelines.append(("motion_blur", A.Compose([
        A.MotionBlur(blur_limit=(5, 9), p=1.0),
    ], bbox_params=bbox_params)))

    # 6. Gaussian noise — sensor noise
    pipelines.append(("gauss_noise", A.Compose([
        A.GaussNoise(std_range=(0.01, 0.05), per_channel=True, p=1.0),
    ], bbox_params=bbox_params)))

    # 7. Horizontal flip — mirror the scene
    pipelines.append(("hflip", A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=bbox_params)))

    # 8. Rotate ±5° + slight scale — mild viewpoint shift
    pipelines.append(("rotate_mild", A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.0, scale_limit=0.05, rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0,
        ),
    ], bbox_params=bbox_params)))

    # 9. Rotate ±10° + scale ±10% — stronger viewpoint shift
    pipelines.append(("rotate_moderate", A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.0, scale_limit=0.10, rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0,
        ),
    ], bbox_params=bbox_params)))

    # 10. Slight perspective — viewpoint change
    pipelines.append(("perspective", A.Compose([
        A.Perspective(scale=(0.02, 0.06), p=1.0),
    ], bbox_params=bbox_params)))

    # 11. Combo mild: HSV + BrightnessContrast + slight blur
    pipelines.append(("combo_mild", A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.8
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.8
        ),
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
    ], bbox_params=bbox_params)))

    # 12. Combo heavy: HSV + blur + noise + rotate
    pipelines.append(("combo_heavy", A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=20, p=0.9
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.6),
        A.GaussNoise(std_range=(0.01, 0.04), per_channel=True, p=0.6),
        A.ShiftScaleRotate(
            shift_limit=0.0, scale_limit=0.05, rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5,
        ),
    ], bbox_params=bbox_params)))

    return pipelines


# ============== DATA LOADING ==============

def read_yolo_label(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """
    Read a YOLO-format label file.
    Returns (bboxes, class_labels) where bboxes = [[cx, cy, w, h], ...]
    """
    bboxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            bboxes.append([cx, cy, w, h])
            class_labels.append(cls)
    return bboxes, class_labels


def write_yolo_label(label_path: Path, bboxes: List[List[float]], class_labels: List[int]):
    """Write YOLO-format label file."""
    with open(label_path, "w") as f:
        for bbox, cls in zip(bboxes, class_labels):
            f.write(f"{int(cls)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def get_labeled_pairs() -> List[Tuple[Path, Path]]:
    """Find (image_path, label_path) pairs that have labels."""
    pairs = []
    label_files = list(SRC_LABELS_DIR.glob("*.txt"))

    for lbl_path in label_files:
        stem = lbl_path.stem  # e.g. "cam_h_s3390_1783683964132323009"
        # Find matching image
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = SRC_IMAGES_DIR / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"  ⚠️  No image found for label: {lbl_path.name}, skipping")
            continue
        pairs.append((img_path, lbl_path))

    return pairs


# ============== MAIN ==============

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Validate source
    if not SRC_IMAGES_DIR.exists():
        print(f"❌ Source images directory not found: {SRC_IMAGES_DIR}")
        sys.exit(1)
    if not SRC_LABELS_DIR.exists():
        print(f"❌ Source labels directory not found: {SRC_LABELS_DIR}")
        sys.exit(1)

    # Find labeled pairs
    pairs = get_labeled_pairs()
    print(f"📦 Found {len(pairs)} labeled image-label pairs")

    if len(pairs) == 0:
        print("❌ No labeled pairs found!")
        sys.exit(1)

    # Build pipelines
    pipelines = build_pipelines()
    print(f"🔧 Built {len(pipelines)} augmentation pipelines")
    for name, _ in pipelines:
        print(f"   • {name}")

    # Setup output
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Stats
    total_generated = 0
    total_skipped_boxes = 0

    # Process each labeled pair
    for idx, (img_path, lbl_path) in enumerate(pairs):
        stem = lbl_path.stem
        print(f"\n[{idx+1}/{len(pairs)}] {stem}")

        # Read original image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Cannot read image: {img_path}, skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Read labels
        orig_bboxes, orig_classes = read_yolo_label(lbl_path)
        if len(orig_bboxes) == 0:
            print(f"  ⚠️  Empty labels, skipping")
            continue

        # 0. Copy original
        orig_out_img = OUT_IMAGES_DIR / f"{stem}.jpg"
        orig_out_lbl = OUT_LABELS_DIR / f"{stem}.txt"
        if not orig_out_img.exists():
            cv2.imwrite(str(orig_out_img), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            write_yolo_label(orig_out_lbl, orig_bboxes, orig_classes)

        # Apply each pipeline
        for pipe_name, pipeline in pipelines:
            try:
                transformed = pipeline(
                    image=image,
                    bboxes=orig_bboxes.copy(),
                    class_labels=orig_classes.copy(),
                )
            except Exception as e:
                print(f"  ⚠️  Pipeline '{pipe_name}' failed: {e}, skipping")
                continue

            aug_image = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_classes = transformed["class_labels"]

            # Skip if all boxes were dropped (e.g., crop removed everything)
            if len(aug_bboxes) == 0:
                total_skipped_boxes += len(orig_bboxes)
                continue

            # Write augmented pair
            out_stem = f"{stem}_aug_{pipe_name}"
            out_img = OUT_IMAGES_DIR / f"{out_stem}.jpg"
            out_lbl = OUT_LABELS_DIR / f"{out_stem}.txt"

            cv2.imwrite(str(out_img), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            write_yolo_label(out_lbl, aug_bboxes, aug_classes)
            total_generated += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Augmentation complete!")
    print(f"   Original labeled pairs: {len(pairs)}")
    print(f"   Augmented pairs generated: {total_generated}")
    print(f"   Total output (original + augmented): {len(pairs) + total_generated}")
    print(f"   Boxes skipped (lost to crop/occlusion): {total_skipped_boxes}")
    print(f"   Output directory: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
