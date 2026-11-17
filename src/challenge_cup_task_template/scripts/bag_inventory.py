#!/usr/bin/env python3
"""Inventory of recorded bags: counts, seeds, sizes, durations -> table + chart.

    python3 bag_inventory.py                       # scans bags/act and bags/pick0
    python3 bag_inventory.py --dir /path/to/bags
    python3 bag_inventory.py --dir DIR --png figs/inventory.png

Prints one line per seed and a summary; the PNG is a per-seed bar chart you can
drop on a slide as evidence of what is actually on disk.
"""

import argparse
import glob
import os
import re

from PIL import Image, ImageDraw, ImageFont

CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJKB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
DARK = (31, 42, 55)
GREY = (120, 128, 138)
GREEN = (27, 94, 32)
LIGHT = (232, 245, 233)


def f(size, bold=False):
    try:
        return ImageFont.truetype(CJKB if bold else CJK, size)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--duration-hz", type=float, default=None,
                    help="optional: frames = duration*hz (needs rosbag)")
    args = ap.parse_args()

    dirs = args.dir or ["/root/kuavo_ws/bags/act", "/root/kuavo_ws/bags/pick0"]
    per_seed = {}
    total_bytes = 0
    total_bags = 0
    bad = []

    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.bag"))):
            name = os.path.basename(path)
            m = re.search(r"seed(\d+)_(?:p|pick|full)(\d+)", name)
            if not m:
                continue
            seed = int(m.group(1))
            size = os.path.getsize(path)
            ok = "_ok1_" in name
            per_seed.setdefault(seed, {"n": 0, "bytes": 0, "bad": 0})
            per_seed[seed]["n"] += 1
            per_seed[seed]["bytes"] += size
            if not ok:
                per_seed[seed]["bad"] += 1
                bad.append(name)
            total_bytes += size
            total_bags += 1

    if not per_seed:
        print("no bags found in:", ", ".join(dirs))
        return 1

    seeds = sorted(per_seed)
    print("%-6s %6s %10s %8s" % ("seed", "bags", "size", "failed"))
    for s in seeds:
        v = per_seed[s]
        print("%-6d %6d %9.2f GB %8d" % (s, v["n"], v["bytes"] / 1e9, v["bad"]))
    print("-" * 34)
    print("%-6s %6d %9.2f GB %8d" % ("all", total_bags, total_bytes / 1e9, len(bad)))
    print("\nseeds %d–%d, %d bags, %.2f GB on disk" % (seeds[0], seeds[-1], total_bags,
                                                       total_bytes / 1e9))
    if bad:
        print("failed bags:", ", ".join(bad[:5]), "..." if len(bad) > 5 else "")

    if args.png:
        W, H = 1400, 620
        img = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(img)
        fb, fs = f(30, True), f(18)
        dr.text((48, 24), "已录制数据清单", fill=DARK, font=fb)
        dr.text((50, 66), "%d 个种子 · %d 条 bag · %.1f GB（%s）"
                % (len(seeds), total_bags, total_bytes / 1e9, "bag 目录扫描"), fill=GREY, font=fs)
        L, R, T, B = 90, 40, 120, 90
        x0, x1, y0, y1 = L, W - R, T, H - B
        vmax = max(v["n"] for v in per_seed.values()) * 1.2
        bw = (x1 - x0) / max(len(seeds), 1)
        for i, s in enumerate(seeds):
            h = (y1 - y0) * per_seed[s]["n"] / vmax
            dr.rectangle([x0 + i * bw + bw * 0.15, y1 - h,
                          x0 + i * bw + bw * 0.85, y1], fill=GREEN)
            dr.text((x0 + i * bw + bw * 0.25, y1 - h - 26), str(per_seed[s]["n"]),
                    fill=DARK, font=fs)
            if i % max(1, len(seeds) // 20) == 0:
                dr.text((x0 + i * bw, y1 + 10), str(s), fill=DARK, font=fs)
        dr.line([(x0, y1), (x1, y1)], fill=DARK)
        dr.line([(x0, y0), (x0, y1)], fill=DARK)
        dr.text((x0, y1 + 40), "种子编号", fill=DARK, font=fs)
        dr.text((x0, y0 - 30), "条数", fill=DARK, font=fs)
        os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
        img.save(args.png)
        print("chart ->", args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
