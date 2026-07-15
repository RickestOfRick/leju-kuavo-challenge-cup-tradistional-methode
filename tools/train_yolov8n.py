#!/usr/bin/env python3
"""
Train YOLOv8n on augmented manual_dataset with 700/80 train/val split.

Split is done at the original-seed level: all 13 variants (1 original + 12
augmentations) of a given seed image stay together in either train or val,
preventing data leakage from augmented variants crossing the split boundary.

Usage:
    python tools/train_yolov8n.py

Requires: conda env torch2.8.0 with ultralytics, torch, torchvision.
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import List, Set

import yaml

# ============== CONFIG ==============

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "manual_dataset_augmented"
SRC_IMAGES = SRC_DIR / "images"
SRC_LABELS = SRC_DIR / "labels"

DATASET_DIR = PROJECT_ROOT / "augmented_yolo_dataset"
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"
VAL_IMAGES = DATASET_DIR / "val" / "images"
VAL_LABELS = DATASET_DIR / "val" / "labels"

DATASET_YAML = DATASET_DIR / "dataset.yaml"

SEED = 42
TRAIN_SEEDS = 54   # 54 * 13 = 702 ≈ 700
VAL_SEEDS = 6       # 6 * 13 = 78 ≈ 80
# (total: 60 seeds, 780 image-label pairs)

CLASS_NAMES = {
    0: "part_a",
    1: "part_b",
    2: "part_c",
}

TRAIN_CONFIG = {
    "model": "yolov8n.pt",
    "epochs": 200,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 8,
    "patience": 30,         # early stopping
    "save_period": 10,      # save checkpoint every 10 epochs
    "project": str(PROJECT_ROOT / "runs" / "train"),
    "name": "augmented_yolov8n",
}


# ============== DATASET SPLIT ==============

def extract_original_stems() -> List[str]:
    """Extract unique original image stems by stripping _aug_* suffixes."""
    stems: Set[str] = set()
    for f in SRC_IMAGES.glob("*.jpg"):
        name = f.stem
        if "_aug_" in name:
            original = name.split("_aug_")[0]
        else:
            original = name
        stems.add(original)
    return sorted(stems)


def setup_dataset():
    """Create train/val directory structure with seed-level split."""
    # Clean existing
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)

    # Seed-level split
    all_stems = extract_original_stems()
    random.seed(SEED)
    random.shuffle(all_stems)

    train_stems = set(all_stems[:TRAIN_SEEDS])
    val_stems = set(all_stems[TRAIN_SEEDS:])

    print(f"Total original seeds: {len(all_stems)}")
    print(f"Train seeds: {len(train_stems)} → {len(train_stems) * 13} images")
    print(f"Val seeds:   {len(val_stems)} → {len(val_stems) * 13} images")

    train_count = 0
    val_count = 0

    for img_path in sorted(SRC_IMAGES.glob("*.jpg")):
        name = img_path.stem
        base = name.split("_aug_")[0] if "_aug_" in name else name

        lbl_path = SRC_LABELS / f"{name}.txt"
        if not lbl_path.exists():
            print(f"  ⚠ Missing label: {lbl_path}, skipping")
            continue

        if base in train_stems:
            dst_img = TRAIN_IMAGES / img_path.name
            dst_lbl = TRAIN_LABELS / lbl_path.name
            train_count += 1
        else:
            dst_img = VAL_IMAGES / img_path.name
            dst_lbl = VAL_LABELS / lbl_path.name
            val_count += 1

        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)

    print(f"Copied: {train_count} train + {val_count} val = {train_count + val_count} total")

    # Write dataset.yaml
    dataset_yaml = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": CLASS_NAMES,
    }
    with open(DATASET_YAML, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"Dataset YAML: {DATASET_YAML}")


# ============== TRAINING ==============

def train():
    """Train YOLOv8n using ultralytics."""
    from ultralytics import YOLO

    # Load pretrained model
    model = YOLO(TRAIN_CONFIG["model"])

    results = model.train(
        data=str(DATASET_YAML),
        epochs=TRAIN_CONFIG["epochs"],
        imgsz=TRAIN_CONFIG["imgsz"],
        batch=TRAIN_CONFIG["batch"],
        device=TRAIN_CONFIG["device"],
        workers=TRAIN_CONFIG["workers"],
        patience=TRAIN_CONFIG["patience"],
        save_period=TRAIN_CONFIG["save_period"],
        project=TRAIN_CONFIG["project"],
        name=TRAIN_CONFIG["name"],
        # Built-in YOLO augmentations during training
        augment=True,
        hsv_h=0.015,    # hue augmentation
        hsv_s=0.7,      # saturation
        hsv_v=0.4,      # value
        degrees=0.0,    # rotation (off — we already augmented)
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,     # mosaic augmentation
        mixup=0.1,      # mixup augmentation
    )

    # Save final model
    final_path = PROJECT_ROOT / "weights" / "yolov8n_augmented.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    print(f"\nModel saved to: {final_path}")

    return results


# ============== MAIN ==============

def main():
    print("=" * 60)
    print("YOLOv8n Training on Augmented Dataset")
    print("=" * 60)

    # Step 1: Setup dataset split
    print("\n📦 Setting up train/val split (seed-level, no leakage)...")
    setup_dataset()

    # Step 2: Train
    print(f"\n🚀 Starting training: {TRAIN_CONFIG['model']}, {TRAIN_CONFIG['epochs']} epochs")
    results = train()

    print("\n✅ Training complete!")
    print(f"Results saved to: {TRAIN_CONFIG['project']}/{TRAIN_CONFIG['name']}")


if __name__ == "__main__":
    main()
