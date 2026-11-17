#!/usr/bin/env python3
"""End-effector accuracy in millimetres, from the bags, via the robot's own FK.

For every sample: run FK on the COMMANDED 14 joint angles and on the MEASURED 14
joint angles, then compare the two hand poses:

    position error  = |p_cmd - p_meas|            [mm]
    orientation err = angle between the two quats  [deg]

That is the number people quote for arm accuracy ("the hand follows the command
within N mm"), as opposed to joint error in degrees.

Requires the simulator / humanoid arm control node to be running, because FK is a
ROS service (/ik/fk_srv). Does NOT move anything.

    python3 fk_error.py                       # all bags, 10 Hz, table + CSV
    python3 fk_error.py --bag FILE --hz 20
    python3 fk_error.py --plot                # + PNG bar chart of RMS per bag

Output: bags/act/plots/fk_error.csv  and  fk_error.png
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import rospy
from kuavo_msgs.srv import fkSrv
from fk_units import calibrate

FK_SRV = "/ik/fk_srv"
OUT_DIR = "/root/kuavo_ws/bags/act/plots"


def resample(ts, values, grid):
    ts = np.asarray(ts, float)
    values = np.asarray(values, float)
    return np.stack([np.interp(grid, ts, values[:, j]) for j in range(values.shape[1])], 1)


def quat_angle_deg(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return math.degrees(2.0 * math.acos(d))


def read_bag(path):
    import rosbag
    bag = rosbag.Bag(path)
    ac_t, ac_q, st_t, st_q = [], [], [], []
    for topic, msg, t in bag.read_messages(topics=["/kuavo_arm_traj", "/sensors_data_raw"]):
        ts = t.to_sec()
        if topic == "/kuavo_arm_traj":
            ac_t.append(ts)
            # FK takes radians (arm_test_control passes raw joint_q)
            ac_q.append([math.radians(float(v)) for v in msg.position[:14]])
        else:
            q = msg.joint_data.joint_q
            if len(q) >= 27:
                st_t.append(ts)
                st_q.append([float(q[13 + i]) for i in range(14)])   # radians
    bag.close()
    return np.array(ac_t), np.array(ac_q), np.array(st_t), np.array(st_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=None)
    ap.add_argument("--bags", default="/root/kuavo_ws/bags/act/*.bag")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--csv", default=os.path.join(OUT_DIR, "fk_error.csv"))
    args = ap.parse_args()

    rospy.init_node("fk_error", anonymous=True, log_level=rospy.ERROR)
    rospy.wait_for_service(FK_SRV, timeout=10.0)
    fk = rospy.ServiceProxy(FK_SRV, fkSrv)
    scale = calibrate(fk, log=print)
    print("FK service: %s  (%.0f Hz sampling, unit scale %.3f)"
          % (FK_SRV, args.hz, scale))

    bags = [args.bag] if args.bag else sorted(glob.glob(args.bags))
    rows = []
    curves = []
    for path in bags:
        name = os.path.basename(path).replace(".bag", "")
        try:
            ac_t, ac_q, st_t, st_q = read_bag(path)
        except Exception as e:
            print("skip %-28s (%s)" % (name, e))
            continue
        if len(ac_t) < 10 or len(st_t) < 10:
            print("skip %-28s (no arm data)" % name)
            continue
        grid = np.arange(max(ac_t[0], st_t[0]), min(ac_t[-1], st_t[-1]), 1.0 / args.hz)
        C = resample(ac_t, ac_q, grid)
        S = resample(st_t, st_q, grid)

        perr = {"left": [], "right": []}
        oerr = {"left": [], "right": []}
        for c, s in zip(C, S):
            try:
                a = fk(list(map(float, np.asarray(c) * scale)))
                b = fk(list(map(float, np.asarray(s) * scale)))
            except Exception:
                continue
            if not (a.success and b.success):
                continue
            for side in ("left", "right"):
                pa = getattr(a.hand_poses, side + "_pose")
                pb = getattr(b.hand_poses, side + "_pose")
                perr[side].append(1000.0 * float(np.linalg.norm(
                    np.array(pa.pos_xyz) - np.array(pb.pos_xyz))))
                oerr[side].append(quat_angle_deg(np.array(pa.quat_xyzw),
                                                 np.array(pb.quat_xyzw)))
        if not perr["right"]:
            print("skip %-28s (FK failed)" % name)
            continue
        allp = np.concatenate([np.array(perr["left"]), np.array(perr["right"])])
        rp = np.array(perr["right"])
        rows.append((name, len(allp), np.sqrt((allp ** 2).mean()), np.percentile(allp, 95),
                     allp.max(), np.median(rp), np.mean(oerr["right"])))
        curves.append((name, grid - grid[0],
                       np.maximum(np.array(perr["left"]), np.array(perr["right"])),
                       np.maximum(np.array(oerr["left"]), np.array(oerr["right"]))))
        print("%-28s n=%4d  RMS %5.1f mm  p95 %5.1f  max %6.1f  | right median %4.1f mm  "
              "rot %4.1f deg" % rows[-1])

    if not rows:
        print("no data — is the simulator running?")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.csv, "w") as f:
        f.write("bag,n,rms_mm,p95_mm,max_mm,right_median_mm,rot_deg\n")
        for r in rows:
            f.write("%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n" % r)
    rms = np.array([r[2] for r in rows])
    print("\n%d bags: RMS %.1f mm (mean of bags), worst max %.1f mm"
          % (len(rows), rms.mean(), max(r[4] for r in rows)))
    print("csv -> %s" % args.csv)

    if args.plot:
        plot(rows, os.path.join(OUT_DIR, "fk_error_bars.png"))
        plot_timeseries(curves, os.path.join(OUT_DIR, "fk_error_timeseries.png"))
    return 0


def plot(rows, out_path):
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1400, 620
    PAD_L, PAD_R, PAD_T, PAD_B = 80, 40, 60, 150
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = fb = ImageFont.load_default()
    x0, x1, y0, y1 = PAD_L, W - PAD_R, PAD_T, H - PAD_B
    ymax = max(10.0, max(r[2] for r in rows) * 1.25)
    d.text((PAD_L, 16), "end-effector accuracy per bag (FK, commanded vs measured)",
           fill="black", font=fb)
    for i in range(6):
        yv = ymax * i / 5.0
        y = y1 - (y1 - y0) * i / 5.0
        d.line([(x0, y), (x1, y)], fill=(232, 232, 232))
        d.text((20, y - 7), "%5.1f" % yv, fill=(80, 80, 80), font=f)
    d.text((16, y0 - 24), "RMS [mm]", fill="black", font=f)
    n = len(rows)
    bw = (x1 - x0) / max(n, 1)
    for i, r in enumerate(rows):
        h = (y1 - y0) * min(r[2], ymax) / ymax
        x = x0 + i * bw + bw * 0.15
        d.rectangle([x, y1 - h, x + bw * 0.7, y1], fill=(70, 120, 200))
        d.text((x + 2, y1 + 4), r[0].split("_")[0][-1] + r[0].split("_p")[1][:1],
               fill=(60, 60, 60), font=f)
    d.line([(x0, y1), (x1, y1)], fill="black")
    d.line([(x0, y0), (x0, y1)], fill="black")
    img.save(out_path)
    print("plot -> %s" % out_path)


def plot_timeseries(curves, out_path):
    """Two stacked panels: position error (mm) and rotation error (deg) over time."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1500, 780
    L, R, T = 80, 40, 64
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = fb = ImageFont.load_default()

    x0, x1 = L, W - R
    tmax = max(1.0, max(c[1].max() for c in curves))
    panels = [(T, 400, "hand position error [mm]", 0),
              (470, 720, "hand rotation error [deg]", 1)]
    d.text((L, 16), "end-effector tracking error — %d bags" % len(curves),
           fill="black", font=fb)
    d.text((L, 42), "FK(commanded) vs FK(measured), worst of the two hands, 10 Hz",
           fill=(100, 100, 100), font=f)

    for (y0, y1, label, idx) in panels:
        vals = np.concatenate([c[2 + idx] for c in curves])
        ymax = max(np.percentile(vals, 99.5) * 1.3, 1.0)
        for i in range(5):
            yv = ymax * i / 4.0
            y = y1 - (y1 - y0) * i / 4.0
            d.line([(x0, y), (x1, y)], fill=(232, 232, 232))
            d.text((20 if idx == 0 else x1 + 8, y - 7), "%5.1f" % yv,
                   fill=(80, 80, 80), font=f)
        d.line([(x0, y1), (x1, y1)], fill="black")
        d.text((x0 + 6, y0 - 22), label, fill="black", font=f)
        for _, t, pos, rot in curves:
            v = pos if idx == 0 else rot
            d.line([(x0 + (x1 - x0) * ti / tmax, y1 - (y1 - y0) * min(vi, ymax) / ymax)
                    for ti, vi in zip(t, v)], fill=(70, 120, 200), width=1)
        d.text((x1 - 230, y0 + 4), "median %5.1f" % np.median(vals), fill="black", font=f)
        d.text((x1 - 230, y0 + 24), "p95    %5.1f" % np.percentile(vals, 95), fill="black", font=f)
        d.text((x1 - 230, y0 + 44), "max    %5.1f" % vals.max(), fill="black", font=f)

    for i in range(7):
        xv = tmax * i / 6.0
        x = x0 + (x1 - x0) * i / 6.0
        d.text((x - 14, 726), "%.0f s" % xv, fill=(80, 80, 80), font=f)
    img.save(out_path)
    print("plot -> %s" % out_path)


if __name__ == "__main__":
    sys.exit(main())
