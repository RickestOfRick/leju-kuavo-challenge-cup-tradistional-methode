#!/usr/bin/env python3
"""Convert the ACT bags into per-episode NPZ files for behaviour cloning.

One bag = one parcel = one episode. Everything is resampled to a fixed FPS grid
(10 Hz by default) so the three cameras, the measured joint state and the
commanded action line up frame by frame.

Per episode (episode_000.npz) — numpy only, no h5py in the container:

    qpos      (T, 16)  float32  measured: 14 arm joints [deg] + 2 claws [%]
    action    (T, 16)  float32  commanded: 14 joints [deg] + 2 claws [0/1]
    cam_h_data / cam_h_off     JPEG bytes of every frame + offsets (same for cam_l/cam_r)
    fps, seed, parcel, bag

Frames are stored as JPEG blobs on purpose: 32 episodes of raw 640x480x3 would be
~31 GB, as JPEG they are ~2 GB, and decoding costs ~2 ms per camera.

    python3 export_act_dataset.py                      # all bags -> /root/kuavo_ws/bags/act_dataset
    python3 export_act_dataset.py --bags DIR --out DIR
    python3 export_act_dataset.py --limit 1 --out /tmp/ds_test
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import rosbag

CAMS = ["cam_h", "cam_l", "cam_r"]


def resample(ts, values, grid):
    ts = np.asarray(ts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    return np.stack([np.interp(grid, ts, values[:, j]) for j in range(values.shape[1])],
                    axis=1)


def parse_name(path):
    m = re.search(r"seed(\d+)_(?:p|pick)(\d+)", os.path.basename(path))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def export_one(path, out_path, fps, width, height, quality, log=print):
    import cv2

    bag = rosbag.Bag(path)
    act_t, act_v = [], []          # commanded action (16)
    st_t, st_v = [], []            # measured joints (14) [deg]
    claw_t, claw_v = [], []        # measured claws (2) [%]
    cam_t = {c: [] for c in CAMS}
    cam_b = {c: [] for c in CAMS}
    topic_of = {}

    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()
        if topic == "/act_action":
            act_t.append(ts)
            act_v.append(list(msg.data[:16]))
        elif topic == "/sensors_data_raw":
            q = msg.joint_data.joint_q
            if len(q) >= 27:
                st_t.append(ts)
                st_v.append([math.degrees(q[13 + i]) for i in range(14)])
        elif topic == "/leju_claw_state":
            try:
                p = list(msg.data.position)
                if len(p) >= 2:
                    claw_t.append(ts)
                    claw_v.append([float(p[0]), float(p[1])])
            except Exception:
                pass
        elif topic.endswith("color/image_raw/compressed"):
            for c in CAMS:
                if "/%s/" % c in topic:
                    cam_t[c].append(ts)
                    cam_b[c].append(msg.data)

    dur = bag.get_end_time() - bag.get_start_time()
    bag.close()

    if not act_t or not st_t:
        raise RuntimeError("missing /act_action or /sensors_data_raw")
    for c in CAMS:
        if len(cam_t[c]) < 10:
            raise RuntimeError("camera %s has only %d frames" % (c, len(cam_t[c])))

    t0 = act_t[0]
    t1 = act_t[-1]
    n = int((t1 - t0) * fps) + 1
    grid = t0 + np.arange(n) / float(fps)

    # action: nearest command (it is already 50 Hz and piecewise constant)
    A = np.array(act_v, dtype=np.float32)
    idx = np.clip(np.searchsorted(np.array(act_t), grid), 0, len(act_t) - 1)
    action = np.empty((n, 16), dtype=np.float32)
    action[:, :14] = A[idx, :14]
    action[:, 14:] = (A[idx, 14:] > 0).astype(np.float32)      # 0/255 -> 0/1

    # state: measured joints + measured claws
    qpos = np.empty((n, 16), dtype=np.float32)
    qpos[:, :14] = resample(st_t, st_v, grid)
    if claw_t:
        qpos[:, 14:] = resample(claw_t, claw_v, grid)
    else:
        qpos[:, 14:] = 0.0

    # images: nearest frame, decoded -> resized -> re-encoded as JPEG
    payload = {"qpos": qpos, "action": action}
    for c in CAMS:
        ct = np.array(cam_t[c])
        ci = np.clip(np.searchsorted(ct, grid), 0, len(ct) - 1)
        chunks = []
        off = np.zeros(n + 1, dtype=np.int64)
        last = -1
        enc = b""
        for i, j in enumerate(ci):
            if j != last:
                im = cv2.imdecode(np.frombuffer(cam_b[c][j], np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    im = np.zeros((height, width, 3), np.uint8)
                if im.shape[1] != width or im.shape[0] != height:
                    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                enc = buf.tobytes()
                last = j
            chunks.append(enc)
            off[i + 1] = off[i] + len(enc)
        payload["%s_data" % c] = np.frombuffer(b"".join(chunks), np.uint8)
        payload["%s_off" % c] = off

    np.savez(out_path, fps=np.float32(fps), image_h=height, image_w=width,
             duration=np.float32(dur), **payload)

    log("  %-26s %4.1f s -> %3d frames, %.0f MB"
        % (os.path.basename(path), dur, n, os.path.getsize(out_path) / 1e6))
    return n


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="bags -> ACT episode HDF5")
    ap.add_argument("--bags", default="/root/kuavo_ws/bags/act")
    ap.add_argument("--out", default="/root/kuavo_ws/bags/act_dataset")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--limit", type=int, default=0, help="only the first N bags (test)")
    args = ap.parse_args()

    bags = sorted(glob.glob(os.path.join(args.bags, "*.bag")))
    bags = [b for b in bags if parse_name(b)[0] is not None]
    if args.limit:
        bags = bags[:args.limit]
    if not bags:
        print("no bags found in %s" % args.bags)
        return 1

    os.makedirs(args.out, exist_ok=True)
    eps = []
    for i, b in enumerate(bags):
        seed, parcel = parse_name(b)
        out_path = os.path.join(args.out, "episode_%03d.npz" % i)
        print("[%2d/%2d] %s" % (i + 1, len(bags), os.path.basename(b)))
        try:
            n = export_one(b, out_path, args.fps, args.width, args.height, args.quality)
        except Exception as e:
            print("  FAILED: %s" % e)
            continue
        eps.append({"episode": i, "file": os.path.basename(out_path), "frames": n,
                    "seed": seed, "parcel": parcel, "bag": os.path.basename(b)})

    meta = {"fps": args.fps, "width": args.width, "height": args.height,
            "cameras": CAMS, "state_dim": 16, "action_dim": 16, "episodes": eps}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    good = [e for e in eps if e["frames"] > 10]   # 5 s bags give ~49 frames
    print("\n%d/%d episodes exported, %d frames total, %.1f GB"
          % (len(good), len(bags), sum(e["frames"] for e in good),
             sum(os.path.getsize(os.path.join(args.out, e["file"])) for e in good) / 1e9))
    print("train seeds:", sorted({e["seed"] for e in good}))
    print("meta:", os.path.join(args.out, "meta.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
