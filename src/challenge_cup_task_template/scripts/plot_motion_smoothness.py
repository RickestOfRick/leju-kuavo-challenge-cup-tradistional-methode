#!/usr/bin/env python3
"""Arm smoothness: max joint change per control tick (10 ms), all bags on one chart.

Reads /kuavo_arm_traj (100 Hz commanded stream, 14 joints), resamples to a uniform
100 Hz grid so publish jitter cannot fake a jump, then plots max|d(deg)| per tick
against time for every bag, with the limit line and a stats box. Ticks above the
limit are printed with bag / time / joint so they can be tracked down.

    python3 plot_motion_smoothness.py                  # all bags in bags/act
    python3 plot_motion_smoothness.py --limit 5 --bag FILE

PNG: bags/act/plots/motion_smoothness.png
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = "/root/kuavo_ws/bags/act/plots/motion_smoothness.png"
W, H = 1500, 780
PAD_L, PAD_R, PAD_T, PAD_B = 80, 40, 70, 65
JOINT = ["L%d" % i for i in range(1, 8)] + ["R%d" % i for i in range(1, 8)]


def load(bag_path, dt=0.01):
    """-> (t, D) with t seconds since the first message and D (n,14) |delta| per tick."""
    import rosbag
    b = rosbag.Bag(bag_path)
    ts, q = [], []
    for topic, msg, t in b.read_messages(topics=["/kuavo_arm_traj"]):
        ts.append(t.to_sec())
        q.append(list(msg.position[:14]))
    b.close()
    if len(q) < 10:
        return None
    ts = np.asarray(ts, np.float64)
    q = np.asarray(q, np.float64)
    grid = np.arange(ts[0], ts[-1], dt)                 # uniform 100 Hz
    r = np.stack([np.interp(grid, ts, q[:, j]) for j in range(14)], axis=1)
    return grid - grid[0], np.abs(np.diff(r, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=None)
    ap.add_argument("--limit", type=float, default=5.0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    bags = [args.bag] if args.bag else sorted(glob.glob("/root/kuavo_ws/bags/act/*.bag"))
    curves, allv, outliers = [], [], []
    for b in bags:
        name = os.path.basename(b).replace(".bag", "")
        try:
            r = load(b)
        except Exception as e:
            print("skip %-28s (%s)" % (name, e))
            continue
        if r is None:
            print("skip %-28s (no /kuavo_arm_traj)" % name)
            continue
        t, D = r
        curves.append((name, t, D.max(axis=1)))
        allv.append(D.max(axis=1))
        for i in np.where(D.max(axis=1) > args.limit)[0]:
            j = int(np.argmax(D[i]))
            outliers.append((name, float(t[i]), JOINT[j], float(D[i, j])))

    if not allv:
        print("no usable bags")
        return 1
    allv = np.concatenate(allv)
    tmax = max(1.0, max(c[1].max() for c in curves))
    ymax = max(args.limit * 1.5, allv.max() * 1.15)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        f = fb = fs = ImageFont.load_default()

    x0, x1, y0, y1 = PAD_L, W - PAD_R, PAD_T, H - PAD_B
    d.text((PAD_L, 16), "arm smoothness — max joint change per 10 ms tick (%d bags)"
           % len(curves), fill="black", font=fb)
    d.text((PAD_L, 46), "commanded /kuavo_arm_traj, resampled to a uniform 100 Hz grid",
           fill=(100, 100, 100), font=fs)

    for i in range(6):
        yv = ymax * i / 5.0
        y = y1 - (y1 - y0) * i / 5.0
        d.line([(x0, y), (x1, y)], fill=(232, 232, 232))
        d.text((20, y - 8), "%4.1f" % yv, fill=(80, 80, 80), font=fs)
    for i in range(7):
        xv = tmax * i / 6.0
        x = x0 + (x1 - x0) * i / 6.0
        d.text((x - 14, y1 + 8), "%.0f s" % xv, fill=(80, 80, 80), font=fs)
    d.line([(x0, y1), (x1, y1)], fill="black")
    d.line([(x0, y0), (x0, y1)], fill="black")
    d.text((16, y0 - 26), "deg / 10 ms", fill="black", font=f)

    for _, t, v in curves:
        d.line([(x0 + (x1 - x0) * ti / tmax, y1 - (y1 - y0) * min(vi, ymax) / ymax)
                for ti, vi in zip(t, v)], fill=(70, 120, 200), width=1)

    yl = y1 - (y1 - y0) * args.limit / ymax
    for x in range(x0, x1, 14):
        d.line([(x, yl), (min(x + 7, x1), yl)], fill=(210, 40, 40), width=3)
    d.text((x1 - 110, yl - 22), "%.0f° limit" % args.limit, fill=(210, 40, 40), font=f)

    over = int((allv > args.limit).sum())
    rows = [("ticks", "%d" % len(allv)),
            ("median", "%.2f°" % np.median(allv)),
            ("p99", "%.2f°" % np.percentile(allv, 99)),
            ("p99.9", "%.2f°" % np.percentile(allv, 99.9)),
            ("max", "%.2f°" % allv.max()),
            ("> %.0f°" % args.limit, "%d (%.4f%%)" % (over, 100.0 * over / len(allv)))]
    yy = y0 + 6
    for k, v in rows:
        d.text((x1 - 250, yy), "%-7s %s" % (k, v), fill="black", font=f)
        yy += 23

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img.save(args.out)

    print("%d bags, %d ticks: median %.2f  p99 %.2f  p99.9 %.2f  max %.2f deg/10ms"
          % (len(curves), len(allv), np.median(allv), np.percentile(allv, 99),
             np.percentile(allv, 99.9), allv.max()))
    if outliers:
        print("\nticks over %.0f°:" % args.limit)
        for name, t, j, v in outliers:
            print("  %-28s t=%5.1fs  %-3s  %.2f°" % (name, t, j, v))
    else:
        print("\nno tick over %.0f°" % args.limit)
    print("\nplot -> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
