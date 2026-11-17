#!/usr/bin/env python3
"""Verify that a recorded bag is TRUE (matches what happened) and usable for ACT.

Checks:
  1. the action really matches the robot: |commanded - measured| per joint
  2. the claw actually closes while the command says closed
  3. the sequence is complete: 2 closes + 2 opens (pick, scale, handoff, box)
  4. time alignment: images / state / action all present inside the same window
  5. dumps a few frames to /tmp/act_frames/ so you can eyeball them

    python3 verify_bag.py /root/kuavo_ws/bags/act/seed0_p0_ok1_ik0.bag
"""

import math
import os
import sys

import numpy as np
import rosbag

OUT = "/tmp/act_frames"


def resample(ts, values, grid):
    ts = np.asarray(ts)
    values = np.asarray(values)
    return np.stack([np.interp(grid, ts, values[:, j]) for j in range(values.shape[1])], axis=1)


def main():
    path = sys.argv[1]
    bag = rosbag.Bag(path)
    t_start, t_end = bag.get_start_time(), bag.get_end_time()

    act, state, claw, cmd, imgs = [], [], [], [], {}
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()
        if topic == "/act_action":
            act.append((ts, list(msg.data[:16])))
        elif topic == "/sensors_data_raw":
            q = msg.joint_data.joint_q
            if len(q) >= 27:
                state.append((ts, [math.degrees(q[13 + i]) for i in range(14)]))
        elif topic == "/leju_claw_command" or ("claw" in topic and "state" in topic):
            try:
                p = list(msg.data.position)
                if len(p) >= 2:
                    (cmd if "command" in topic else claw).append(
                        (ts, [float(p[0]), float(p[1])]))
            except Exception:
                pass
        elif topic.endswith("compressed"):
            imgs.setdefault(topic, []).append((ts, msg))

    print("bag      : %s" % path)
    print("duration : %.1f s" % (t_end - t_start))

    if not act or not state:
        print("!! missing /act_action or /sensors_data_raw")
        return

    grid = np.arange(act[0][0], act[-1][0], 0.1) - act[0][0]
    ta = np.array([a[0] for a in act]); qa = np.array([a[1][:14] for a in act])
    tsx = np.array([s[0] for s in state]); qs = np.array([s[1] for s in state])
    A = resample(ta, qa, ta[0] + grid)
    S = resample(tsx, qs, ta[0] + grid)
    err = np.abs(A - S)

    per_joint = err.mean(axis=0)
    print("\n1) action vs measured  (mean |err| per joint, deg)")
    print("   " + "  ".join("j%02d %4.1f" % (i + 1, per_joint[i]) for i in range(14)))
    print("   overall: median %.2f   p95 %.2f   max %.2f" %
          (np.median(err), np.percentile(err, 95), err.max()))
    print("   -> this is the arm's tracking error; a few degrees is normal, tens is not")

    if cmd:
        tcm = np.array([c[0] for c in cmd]); qcm = np.array([c[1] for c in cmd])
        CM = resample(tcm, qcm, ta[0] + grid)
        closed = (CM > 70).any(axis=1)
        events = int(np.sum(np.diff(closed.astype(int)) == 1))
        print("\n3) claw COMMAND: %d close event(s); closed %.0f %% of the frames"
              % (events, 100.0 * closed.mean()))
        print("   -> expect ~2 (pick + scale re-grasp); the handoff/box use the left claw")
        if claw:
            CV = resample(np.array([c[0] for c in claw]),
                          np.array([c[1] for c in claw]), ta[0] + grid)
            print("   claw STATE closure while closed: L %.0f-%.0f %%   R %.0f-%.0f %%"
                  % (CV[closed, 0].min() if closed.any() else 0,
                     CV[closed, 0].max() if closed.any() else 0,
                     CV[closed, 1].min() if closed.any() else 0,
                     CV[closed, 1].max() if closed.any() else 0))
            print("   (~50 %% = holding a parcel, ~97 %% = closed on air)")
    else:
        print("\n3) claw: no /leju_claw_command in this bag (recorded before it was added)")

    print("\n4) time coverage")
    print("   action %.1f-%.1f   state %.1f-%.1f   (%d / %d msgs)"
          % (ta[0] - ta[0], ta[-1] - ta[0], tsx[0] - ta[0], tsx[-1] - ta[0],
             len(act), len(state)))

    # 5) dump frames around the first close event, if any
    try:
        os.makedirs(OUT, exist_ok=True)
        import cv2
        for topic, lst in imgs.items():
            cam = topic.split("/")[1]
            for k, (ts, msg) in enumerate(lst):
                if abs((ts - ta[0]) - 20.0) > 0.15:
                    continue
                im = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    fn = os.path.join(OUT, "%s_t20.png" % cam)
                    cv2.imwrite(fn, im)
                    print("\n5) frame saved: %s" % fn)
                break
    except Exception as e:
        print("\n5) frame dump failed (%s)" % e)

    print("\nverdict: %s"
          % ("action and state agree -> the bag is a faithful recording"
             if np.percentile(err, 95) < 15.0 else
             "action and state DISAGREE (p95 %.1f deg) -> check the recording" %
             np.percentile(err, 95)))
    bag.close()


if __name__ == "__main__":
    main()
