#!/usr/bin/env python3
"""
scene2_label_visualizer.py — YOLO 标注数据集可视化审核工具

功能:
  - 浏览已标注图片，叠加 bbox 和类别标签
  - 键盘导航: ← → 翻页, q 退出, s 统计
  - 颜色编码: A类=蓝, B类=橙, C类=紫
  - --html 导出 HTML 图库

用法:
  # 交互浏览
  python3 tools/scene2_label_visualizer.py --dataset datasets/scene2_yolo/

  # 导出 HTML 图库
  python3 tools/scene2_label_visualizer.py --dataset datasets/scene2_yolo/ --html gallery.html --max-samples 50

  # 指定图片和标注分目录
  python3 tools/scene2_label_visualizer.py --img-dir path/to/images --lbl-dir path/to/labels
"""

import argparse
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
CLASS_NAMES = ["part_a", "part_b", "part_c"]
CLASS_COLORS = {
    0: (255, 140, 50),   # A: 蓝色 (BGR)
    1: (50, 140, 255),   # B: 橙色
    2: (200, 100, 255),  # C: 紫色
}
WIN_NAME = "scene2 label visualizer | ← → navigate | q quit | s stats"


# ===================================================================
# 标注读取
# ===================================================================

def read_yolo_label(label_path, img_w, img_h):
    """读取 YOLO 格式标注文件，返回 [(cls_id, cx, cy, w, h, x1, y1, x2, y2), ...]"""
    detections = []
    if not os.path.exists(label_path):
        return detections
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            detections.append((cls_id, cx, cy, w, h, x1, y1, x2, y2))
    return detections


def find_image_label_pairs(img_dir, lbl_dir):
    """扫描图片和标注文件，返回匹配的配对列表"""
    pairs = []
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in img_extensions:
            continue
        label_path = os.path.join(lbl_dir, stem + ".txt")
        img_path = os.path.join(img_dir, fname)
        pairs.append((img_path, label_path))
    return pairs


# ===================================================================
# 可视化
# ===================================================================

def draw_detections(img, detections):
    """在图像上绘制所有检测框和标签"""
    out = img.copy()
    for cls_id, cx, cy, w, h, x1, y1, x2, y2 in detections:
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"

        # bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # 标签背景
        label = f"{name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 左上角信息
    info = f"{len(detections)} objects"
    cv2.putText(out, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return out


def show_stats(pairs):
    """统计数据集信息"""
    total_imgs = len(pairs)
    cls_counts = Counter()
    total_objects = 0
    empty_imgs = 0

    for _, lbl_path in pairs:
        dets = read_yolo_label(lbl_path, 1280, 720)
        if not dets:
            empty_imgs += 1
        for det in dets:
            cls_counts[det[0]] += 1
            total_objects += 1

    print("\n" + "=" * 50)
    print("  Dataset Statistics")
    print("=" * 50)
    print(f"  Total images:     {total_imgs}")
    print(f"  Empty (no label): {empty_imgs}")
    print(f"  Total objects:    {total_objects}")
    print(f"  Avg objects/img:  {total_objects / max(total_imgs, 1):.1f}")
    print("  ---")
    for cls_id in sorted(cls_counts.keys()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        print(f"  {name}: {cls_counts[cls_id]}")
    print("=" * 50 + "\n")


def generate_html_gallery(pairs, output_path, max_samples=100):
    """生成 HTML 图库文件"""
    if len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)

    rows = []
    for img_path, lbl_path in pairs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = read_yolo_label(lbl_path, w, h)
        annotated = draw_detections(img, dets)

        # 编码为 base64
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = buf.tobytes().hex()

        fname = os.path.basename(img_path)
        n_objs = len(dets)
        cls_summary = ", ".join(
            f"{CLASS_NAMES[d[0]]}" for d in dets
        ) if dets else "none"

        rows.append(f"""
        <div class="card">
          <img src="data:image/jpeg;base64,{b64}" loading="lazy" />
          <div class="info">
            <strong>{fname}</strong><br/>
            objects: {n_objs} ({cls_summary})
          </div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>scene2 label gallery</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
  h1 {{ text-align: center; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 16px; }}
  .card {{ background: #16213e; border-radius: 8px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; }}
  .card .info {{ padding: 8px 12px; font-size: 13px; color: #aaa; }}
</style></head><body>
<h1>scene2 YOLO Label Gallery ({len(rows)} samples)</h1>
<div class="grid">{''.join(rows)}</div>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[visualizer] HTML gallery saved: {output_path} ({len(rows)} samples)")


def interactive_browse(pairs):
    """交互式浏览标注图片"""
    if not pairs:
        print("[visualizer] no image/label pairs found")
        return

    idx = 0
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    while True:
        img_path, lbl_path = pairs[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"[visualizer] WARN: cannot read {img_path}")
            idx = (idx + 1) % len(pairs)
            continue

        h, w = img.shape[:2]
        dets = read_yolo_label(lbl_path, w, h)
        annotated = draw_detections(img, dets)

        # 状态栏
        fname = os.path.basename(img_path)
        status = f"[{idx+1}/{len(pairs)}] {fname} | {len(dets)} objects"
        annotated = cv2.copyMakeBorder(annotated, 0, 28, 0, 0,
                                       cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(annotated, status, (10, annotated.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WIN_NAME, annotated)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:  # q or ESC
            break
        elif key == ord("s"):
            show_stats(pairs)
        elif key in (81, 83, 2, 97):  # left arrow / 'a'
            idx = (idx - 1) % len(pairs)
        elif key in (82, 84, 3, 100):  # right arrow / 'd'
            idx = (idx + 1) % len(pairs)
        elif key == ord("n") or key == ord(" "):
            idx = (idx + 1) % len(pairs)
        elif key == ord("p"):
            idx = (idx - 1) % len(pairs)

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="scene2 YOLO label visualizer — review auto-generated annotations"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", default=None,
                     help="YOLO dataset root dir (must contain images/ and labels/)")
    src.add_argument("--img-dir", default=None, help="Image directory")
    parser.add_argument("--lbl-dir", default=None, help="Label directory (default: img-dir/../labels)")
    parser.add_argument("--html", default=None, help="Export HTML gallery to this path")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max samples in HTML gallery")
    parser.add_argument("--stats-only", action="store_true", help="Print stats and exit")
    args = parser.parse_args()

    # 解析路径
    if args.dataset:
        img_dir = os.path.join(args.dataset, "images")
        lbl_dir = os.path.join(args.dataset, "labels")
    else:
        img_dir = args.img_dir
        lbl_dir = args.lbl_dir or os.path.join(os.path.dirname(img_dir), "labels")

    if not os.path.isdir(img_dir):
        print(f"[visualizer] ERROR: image dir not found: {img_dir}")
        sys.exit(1)
    if not os.path.isdir(lbl_dir):
        print(f"[visualizer] ERROR: label dir not found: {lbl_dir}")
        sys.exit(1)

    pairs = find_image_label_pairs(img_dir, lbl_dir)
    print(f"[visualizer] found {len(pairs)} image/label pairs")

    if len(pairs) == 0:
        print("[visualizer] no pairs found — check that images/*.jpg and labels/*.txt exist")
        sys.exit(1)

    # 统计
    show_stats(pairs)

    if args.stats_only:
        return

    # HTML 导出
    if args.html:
        generate_html_gallery(pairs, args.html, args.max_samples)

    # 交互浏览
    interactive_browse(pairs)


if __name__ == "__main__":
    main()
