#!/usr/bin/env python3
"""Cut every episode down to ONE phase (default: the pick, first 7 s).

The full bag is 47 s with four phases; a 32-demo policy cannot hold that together.
A single phase is short, repeated 32 times and single-armed — the regime where
behaviour cloning actually works.

    python3 make_phase_dataset.py --end 7.0                 # ready -> grasp -> lift
    python3 make_phase_dataset.py --start 7 --end 20 --out .../act_dataset_scale

Writes a new npz per episode + meta.json, same structure as the source, so
train.py works unchanged.
"""

import argparse
import glob
import json
import os

import numpy as np


def keep_mask(action, thresh):
    """Drop frames that repeat the previous one (fixed pauses the policy cannot see).

    A pause and "finished moving" give identical observations, so a cloned policy
    simply holds forever. Collapsing the pause to a single frame removes the
    ambiguity. Claw events are always kept: their action value changes by 1.0.
    """
    if thresh <= 0:
        return np.ones(len(action), bool)
    keep = np.ones(len(action), bool)
    for i in range(1, len(action)):
        keep[i] = np.abs(action[i] - action[i - 1]).max() > thresh
    keep[0] = True
    return keep


def slice_episode(src, dst, start_frame, end_frame, thresh=0.0):
    d = np.load(src)
    n = len(d["qpos"])
    a = max(0, min(start_frame, n))
    b = max(a + 5, min(end_frame, n))
    keep = keep_mask(d["action"][a:b], thresh)
    idx = np.arange(a, b)[keep]
    out = {
        "qpos": d["qpos"][idx],
        "action": d["action"][idx],
        "fps": d["fps"],
        "image_h": d["image_h"],
        "image_w": d["image_w"],
        "duration": np.float32((b - a) / float(d["fps"])),
    }
    for c in ("cam_h", "cam_l", "cam_r"):
        off = d["%s_off" % c]
        data = d["%s_data" % c]
        chunks = [data[off[i]:off[i + 1]] for i in idx]
        sizes = np.array([len(x) for x in chunks], np.int64)
        out["%s_data" % c] = (np.concatenate(chunks) if chunks
                              else np.zeros(0, np.uint8))
        out["%s_off" % c] = np.concatenate([[0], np.cumsum(sizes)])
    np.savez(dst, **out)
    return len(idx), b - a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/kuavo_ws/bags/act_dataset")
    ap.add_argument("--out", default="/root/kuavo_ws/bags/act_dataset_pick")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=7.0, help="seconds")
    ap.add_argument("--drop-static", type=float, default=0.0,
                    help="drop frames that repeat the previous one (deg per step, "
                         "e.g. 0.3); removes the fixed pauses")
    args = ap.parse_args()

    with open(os.path.join(args.src, "meta.json")) as f:
        meta = json.load(f)
    fps = meta["fps"]
    a, b = int(round(args.start * fps)), int(round(args.end * fps))
    os.makedirs(args.out, exist_ok=True)

    eps = []
    for e in meta["episodes"]:
        dst = os.path.join(args.out, e["file"])
        n, n_before = slice_episode(os.path.join(args.src, e["file"]), dst, a, b,
                                    args.drop_static)
        eps.append(dict(e, frames=n))
        print("  %s  %d -> %d frames%s" % (e["file"], n_before, n,
              "  (-%d static)" % (n_before - n) if n_before != n else ""))

    out_meta = dict(meta, episodes=eps, phase="%.1f-%.1f s" % (a / fps, b / fps))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(out_meta, f, indent=2)
    print("\n%d episodes, %d frames total -> %s"
          % (len(eps), sum(e["frames"] for e in eps), args.out))


if __name__ == "__main__":
    main()
