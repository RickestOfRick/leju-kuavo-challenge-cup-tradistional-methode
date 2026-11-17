#!/usr/bin/env python3
"""Draw the arm tracking error for a whole bag: |commanded - measured| per joint.

Matplotlib in this container is broken by a system/package conflict, so the chart
is drawn with PIL — no dependencies beyond numpy/PIL.

    python3 plot_arm_error.py /root/kuavo_ws/bags/act/seed0_p0_ok1_ik0.bag
    python3 plot_arm_error.py --all          # one PNG per bag + a summary bar chart

PNGs land next to the bags in bags/act/plots/.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/root/kuavo_ws/bags/act/plots"
W, H = 1400, 700
PAD_L, PAD_R, PAD_T, PAD_B = 70, 220, 60, 60
COLORS = [(220, 60, 60), (240, 140, 40), (230, 200, 50), (120, 200, 70),
          (60, 200, 190), (60, 160, 230), (90, 110, 220), (160, 90, 220),
          (220, 90, 180), (200, 120, 90), (150, 150, 90), (90, 160, 120),
          (120, 120, 200), (200, 100, 140)]


def resample(ts, values, grid):
    ts = np.asarray(ts, float)
    values = np.asarray(values, float)
    return np.stack([np.interp(grid, ts, values[:, j]) for j in range(values.shape[1])], 1)


def load_errors(path):
    import rosbag
    bag = rosbag.Bag(path)
    act, st = [], []
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()
        if topic == "/act_action":
            act.append((ts, list(msg.data[:14])))
        elif topic == "/sensors_data_raw":
            q = msg.joint_data.joint_q
            if len(q) >= 27:
                st.append((ts, [math.degrees(q[13 + i]) for i in range(14)]))
    dur = bag.get_end_time() - bag.get_start_time()
    bag.close()
    if not act or not st:
        raise RuntimeError("missing /act_action or /sensors_data_raw")
    t0 = act[0][0]
    grid = np.arange(0.0, act[-1][0] - t0, 0.1)
    A = resample([a[0] - t0 for a in act], [a[1] for a in act], grid)
    S = resample([s[0] - t0 for s in st], [s[1] for s in st], grid)
    return grid, np.abs(A - S), A, S, dur


def draw(grid, err, title, subtitle, out_path):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        f = fb = fs = ImageFont.load_default()

    x0, x1 = PAD_L, W - PAD_R
    y0, y1 = PAD_T, H - PAD_B
    tmax = max(1.0, float(grid[-1]))
    ymax = max(2.0, float(np.percentile(err, 99.5)) * 1.15)

    d.text((PAD_L, 18), title, fill="black", font=fb)
    d.text((PAD_L, 42), subtitle, fill=(90, 90, 90), font=fs)

    # axes + grid
    for i in range(6):
        yv = ymax * i / 5.0
        y = y1 - (y1 - y0) * i / 5.0
        d.line([(x0, y), (x1, y)], fill=(230, 230, 230))
        d.text((10, y - 8), "%5.1f" % yv, fill=(80, 80, 80), font=fs)
    d.line([(x0, y1), (x1, y1)], fill="black")
    d.line([(x0, y0), (x0, y1)], fill="black")
    for i in range(6):
        xv = tmax * i / 5.0
        x = x0 + (x1 - x0) * i / 5.0
        d.text((x - 12, y1 + 6), "%.0fs" % xv, fill=(80, 80, 80), font=fs)
    d.text(((x0 + x1) / 2 - 40, y1 + 30), "time", fill="black", font=f)
    d.text((14, y0 - 34), "|error| [deg]", fill="black", font=f)

    def px(t, v):
        return (x0 + (x1 - x0) * t / tmax, y1 - (y1 - y0) * min(v, ymax) / ymax)

    for j in range(14):
        pts = [px(grid[i], err[i, j]) for i in range(len(grid))]
        d.line(pts, fill=COLORS[j], width=1)
    mean = err.mean(1)
    d.line([px(grid[i], mean[i]) for i in range(len(grid))], fill=(0, 0, 0), width=3)

    # legend
    lx = x1 + 15
    d.text((lx, y0), "mean", fill="black", font=f)
    for j in range(14):
        yy = y0 + 22 + j * 20
        d.line([(lx, yy + 8), (lx + 22, yy + 8)], fill=COLORS[j], width=3)
        d.text((lx + 28, yy), "j%02d" % (j + 1), fill=(60, 60, 60), font=fs)
    yy = y0 + 22 + 14 * 20 + 10
    d.text((lx, yy), "median %.2f" % np.median(err), fill="black", font=f)
    d.text((lx, yy + 20), "p95    %.2f" % np.percentile(err, 95), fill="black", font=f)
    d.text((lx, yy + 40), "max    %.2f" % err.max(), fill="black", font=f)

    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", nargs="?", help="bag file")
    ap.add_argument("--all", action="store_true", help="every bag in /root/kuavo_ws/bags/act")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    bags = sorted(glob.glob("/root/kuavo_ws/bags/act/*.bag")) if args.all else [args.bag]
    if not bags or bags[0] is None:
        print("give a bag path or --all")
        return 1

    summary = []
    for b in bags:
        name = os.path.basename(b).replace(".bag", "")
        try:
            grid, err, _, _, dur = load_errors(b)
        except Exception as e:
            print("%-30s FAILED %s" % (name, e))
            continue
        out = os.path.join(OUT_DIR, "arm_error_%s.png" % name)
        draw(grid, err, "arm tracking error  %s" % name,
             "|commanded - measured| per joint, %.1f s, 10 Hz" % dur, out)
        summary.append((name, float(np.median(err)), float(np.percentile(err, 95)), float(err.max())))
        print("%-30s median %5.2f  p95 %5.2f  max %6.2f  -> %s"
              % (name, summary[-1][1], summary[-1][2], summary[-1][3], out))

    if summary:
        med = np.array([s[1] for s in summary])
        print("\n%d bags: median of medians %.2f deg, worst p95 %.2f"
              % (len(summary), med.mean(), max(s[2] for s in summary)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
