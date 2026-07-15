#!/usr/bin/env python3
"""
manual_labeler.py — OpenCV 交互式标注工具 → YOLO 格式

操作:
  鼠标拖拽  — 画框
  0/1/2     — 切换类别 (part_a / part_b / part_c)
  D         — 删除上一个框
  C         — 清除所有框
  S         — 保存标注 (自动保存到 labels/ 目录)
  N / →     — 下一张
  P / ←     — 上一张
  Q / Esc   — 退出
  R         — 重置当前图片标注 (重新加载)

用法:
  python3 tools/manual_labeler.py my_images/                    # 图片目录
  python3 tools/manual_labeler.py my_images/ --classes part_a,part_b,part_c
"""

import argparse, json, os, sys
import cv2, numpy as np

CLASS_NAMES = ["part_a", "part_b", "part_c"]
CLASS_COLORS = [(255, 140, 50), (50, 140, 255), (200, 100, 255)]  # BGR

class Labeler:
    def __init__(self, img_dir, lbl_dir, class_names):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.class_names = class_names
        os.makedirs(lbl_dir, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.files = sorted(
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not self.files:
            print(f"No images found in {img_dir}")
            sys.exit(1)
        print(f"Found {len(self.files)} images")

        self.idx = 0
        self.img = None
        self.annotated = None
        self.boxes = []       # [(cls, x1, y1, x2, y2), ...]
        self.drawing = False
        self.pt1 = (0, 0)
        self.pt2 = (0, 0)
        self.cur_class = 0
        self.modified = False
        self.win = "manual_labeler"

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 1280, 720)
        cv2.setMouseCallback(self.win, self._mouse)
        self._load()

    def _label_path(self, fname):
        stem = os.path.splitext(fname)[0]
        return os.path.join(self.lbl_dir, stem + ".txt")

    def _load(self):
        fname = self.files[self.idx]
        path = os.path.join(self.img_dir, fname)
        self.img = cv2.imread(path)
        if self.img is None:
            print(f"WARN: cannot read {path}")
            return
        self.h, self.w = self.img.shape[:2]

        # Load existing labels if present
        lp = self._label_path(fname)
        self.boxes = []
        if os.path.exists(lp):
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = int((cx - bw/2) * self.w)
                        y1 = int((cy - bh/2) * self.h)
                        x2 = int((cx + bw/2) * self.w)
                        y2 = int((cy + bh/2) * self.h)
                        self.boxes.append((cls, x1, y1, x2, y2))
        self.modified = False
        self._redraw()

    def _save(self):
        fname = self.files[self.idx]
        lp = self._label_path(fname)
        with open(lp, "w") as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                cx = ((x1 + x2) / 2) / self.w
                cy = ((y1 + y2) / 2) / self.h
                bw = (x2 - x1) / self.w
                bh = (y2 - y1) / self.h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        self.modified = False
        print(f"Saved: {lp} ({len(self.boxes)} boxes)")

    def _redraw(self):
        self.annotated = self.img.copy()
        for cls, x1, y1, x2, y2 in self.boxes:
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]
            cv2.rectangle(self.annotated, (x1, y1), (x2, y2), color, 2)
            name = self.class_names[cls]
            cv2.putText(self.annotated, name, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Status bar
        fname = self.files[self.idx]
        cls_name = self.class_names[self.cur_class]
        mod = " [MOD]" if self.modified else ""
        status = f"[{self.idx+1}/{len(self.files)}] {fname} | class={cls_name}({self.cur_class}) | boxes={len(self.boxes)}{mod}"
        self.annotated = cv2.copyMakeBorder(self.annotated, 0, 30, 0, 0,
                                            cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.putText(self.annotated, status, (8, self.h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # Help hint
        hint = "0-2:class  drag:bbox  S:save  N/P:nav  D:del  C:clear  Q:quit"
        cv2.putText(self.annotated, hint, (8, self.h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    def _mouse(self, event, x, y, flags, param):
        if y >= self.h:  # status bar
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1 = (x, y)
            self.pt2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt2 = (x, y)
            tmp = self.annotated.copy()
            cv2.rectangle(tmp[:self.h], self.pt1, self.pt2, (0, 255, 0), 1)
            cv2.imshow(self.win, tmp)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.pt1[0], x), min(self.pt1[1], y)
            x2, y2 = max(self.pt1[0], x), max(self.pt1[1], y)
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.boxes.append((self.cur_class, x1, y1, x2, y2))
                self.modified = True
            self._redraw()

    def run(self):
        while True:
            cv2.imshow(self.win, self.annotated)
            key = cv2.waitKey(0) & 0xFF

            if key in (27, ord("q")):  # ESC / Q
                if self.modified:
                    self._save()
                break
            elif key in (ord("n"), 83, 3):  # N / right
                if self.modified:
                    self._save()
                self.idx = min(self.idx + 1, len(self.files) - 1)
                self._load()
            elif key in (ord("p"), 81, 2):  # P / left
                if self.modified:
                    self._save()
                self.idx = max(self.idx - 1, 0)
                self._load()
            elif key == ord("s"):
                self._save()
                self._redraw()
            elif key == ord("d"):  # Delete last
                if self.boxes:
                    self.boxes.pop()
                    self.modified = True
                    self._redraw()
            elif key == ord("c"):  # Clear all
                self.boxes.clear()
                self.modified = True
                self._redraw()
            elif key == ord("r"):  # Reload
                self._load()
            elif key in (ord("0"), ord("1"), ord("2")):
                self.cur_class = key - ord("0")
                self._redraw()

        cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser(description="Manual YOLO labeler")
    p.add_argument("img_dir", help="Directory of images to label")
    p.add_argument("--lbl-dir", default=None, help="Label output dir (default: img_dir/../labels)")
    p.add_argument("--classes", default="part_a,part_b,part_c",
                   help="Comma-separated class names")
    args = p.parse_args()

    classes = [s.strip() for s in args.classes.split(",")]
    lbl_dir = args.lbl_dir or os.path.join(os.path.dirname(args.img_dir.rstrip("/")), "labels")

    Labeler(args.img_dir, lbl_dir, classes).run()

if __name__ == "__main__":
    main()
