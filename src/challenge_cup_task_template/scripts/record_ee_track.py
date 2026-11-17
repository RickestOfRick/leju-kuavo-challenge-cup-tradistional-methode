#!/usr/bin/env python3
"""Record the right hand for N seconds and save desired-vs-measured plots.

One process: subscribes to the arm topics, runs FK on the commanded and on the
measured joints, then draws x/y/z as PNG files (PIL, no rqt_plot, no display).

    python3 record_ee_track.py --seconds 30
    python3 record_ee_track.py --seconds 30 --out /tmp/ee

While it runs, run `pick 0` (or `home`, `scale 0`) in task_sequence so the arm
moves. Output:
    <out>_xyz.png     three panels: x, y, z  desired vs measured  [mm]
    <out>_err.png     |desired - measured| in mm over time
    <out>.csv         raw data
Prints the FK unit it detected, the travel of the hand and the error statistics.
"""

import argparse
import sys
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from kuavo_msgs.msg import sensorsData
from kuavo_msgs.srv import fkSrv
from fk_units import calibrate

FK_SRV = "/ik/fk_srv"
ARM_OFF = 13


class Rec(object):
    def __init__(self):
        self.cmd = None
        self.meas = None
        self.t_cmd = 0.0
        self.t_meas = 0.0
        rospy.Subscriber("/kuavo_arm_traj", JointState, self._on_cmd, queue_size=1)
        rospy.Subscriber("/sensors_data_raw", sensorsData, self._on_meas, queue_size=1)

    def _on_cmd(self, msg):
        self.cmd = np.radians(np.asarray(msg.position[:14], float))
        self.t_cmd = time.time()

    def _on_meas(self, msg):
        q = msg.joint_data.joint_q
        if len(q) >= 27:
            self.meas = np.asarray(q[ARM_OFF:ARM_OFF + 14], float)
            self.t_meas = time.time()


def right_pos(fk, q, scale):
    r = fk([float(v) for v in np.asarray(q) * scale])
    if not r.success:
        return None
    return np.array(r.hand_poses.right_pose.pos_xyz, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="/tmp/ee_track")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    rospy.init_node("record_ee_track", anonymous=True, log_level=rospy.ERROR)
    rospy.wait_for_service(FK_SRV, timeout=10.0)
    fk = rospy.ServiceProxy(FK_SRV, fkSrv)
    scale = calibrate(fk, log=print)
    print("FK scale %.3f (%s)" % (scale, "degrees" if scale > 2 else "radians"))
    print("recording %.0f s — move the arm now (pick 0 / home / scale 0)" % args.seconds)

    rec = Rec()
    t0 = None
    rows = []
    rate = rospy.Rate(args.hz)
    while not rospy.is_shutdown():
        if rec.cmd is None or rec.meas is None:
            rate.sleep()
            continue
        if t0 is None:
            t0 = time.time()
        try:
            pc = right_pos(fk, rec.cmd, scale)
            pm = right_pos(fk, rec.meas, scale)
        except Exception as e:
            rate.sleep()
            continue
        if pc is not None and pm is not None:
            rows.append((time.time() - t0, *pc, *pm))
        if time.time() - t0 > args.seconds:
            break
        rate.sleep()

    if len(rows) < 5:
        print("not enough samples (%d) — was the arm being commanded?" % len(rows))
        return 1
    A = np.array(rows)                       # t, cx cy cz, mx my mz
    err = np.linalg.norm(A[:, 1:4] - A[:, 4:7], axis=1) * 1000.0

    print("\nsamples %d over %.1f s" % (len(A), A[-1, 0]))
    for i, ax in enumerate("xyz"):
        print("  %s: desired %.0f..%.0f mm   measured %.0f..%.0f mm"
              % (ax, A[:, 1 + i].min() * 1000, A[:, 1 + i].max() * 1000,
                 A[:, 4 + i].min() * 1000, A[:, 4 + i].max() * 1000))
    print("  error: median %.1f mm  p95 %.1f  max %.1f"
          % (np.median(err), np.percentile(err, 95), err.max()))

    np.savetxt(args.out + ".csv", A, delimiter=",", header="t,cx,cy,cz,mx,my,mz")

    # ---- plots
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        f = fb = ImageFont.load_default()

    def panel_axes(d, w, h, tmax, lo, hi, title):
        L, R, T, B = 80, 30, 40, 50
        d.text((L, 10), title, fill="black", font=fb)
        for i in range(5):
            yv = lo + (hi - lo) * i / 4.0
            y = h - B - (h - B - T) * i / 4.0
            d.line([(L, y), (w - R, y)], fill=(230, 230, 230))
            d.text((16, y - 7), "%.0f" % yv, fill=(80, 80, 80), font=f)
        for i in range(5):
            xv = tmax * i / 4.0
            x = L + (w - R - L) * i / 4.0
            d.text((x - 12, h - B + 6), "%.0fs" % xv, fill=(80, 80, 80), font=f)
        d.line([(L, h - B), (w - R, h - B)], fill="black")
        d.line([(L, T), (L, h - B)], fill="black")
        return L, R, T, B

    # xyz panel
    W, H = 1200, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    tmax = float(A[-1, 0])
    ph = H // 3
    for i, ax in enumerate("xyz"):
        vals = np.concatenate([A[:, 1 + i], A[:, 4 + i]]) * 1000.0
        lo, hi = vals.min() - 5, vals.max() + 5
        sub = img.crop((0, i * ph, W, (i + 1) * ph))
        dd = ImageDraw.Draw(sub)
        L, R, T, B = panel_axes(dd, W, ph, tmax, lo, hi,
                                "hand %s [mm]   red = desired, blue = measured" % ax)
        for col, idx in (("red", 1 + i), ("blue", 4 + i)):
            pts = [(L + (W - R - L) * A[k, 0] / tmax,
                    (ph - B) - (ph - B - T) * (A[k, idx] * 1000.0 - lo) / (hi - lo))
                   for k in range(len(A))]
            dd.line(pts, fill=col, width=2)
        img.paste(sub, (0, i * ph))

    p1 = args.out + "_xyz.png"
    img.save(p1)

    img2 = Image.new("RGB", (1200, 420), "white")
    d2 = ImageDraw.Draw(img2)
    L, R, T, B = panel_axes(d2, 1200, 420, tmax, 0, max(err.max() * 1.15, 1.0),
                            "|desired - measured| [mm]")
    d2.line([(L + (1200 - R - L) * A[k, 0] / tmax,
              (420 - B) - (420 - B - T) * err[k] / max(err.max() * 1.15, 1.0))
             for k in range(len(A))], fill="black", width=2)
    p2 = args.out + "_err.png"
    img2.save(p2)
    print("\nplot -> %s\nplot -> %s\ncsv  -> %s" % (p1, p2, args.out + ".csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
