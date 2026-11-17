#!/usr/bin/env python3
"""Check whether an ACT bag is usable.

Reads the bag and reports the things that matter for behaviour cloning:

  * rates of every topic
  * /act_action: resampled to 10 Hz, max joint jump per step, % static frames
  * claw values: do they actually change (open -> closed)?
  * images: count per camera + mean brightness (catches black frames)

    python3 check_bag.py /root/kuavo_ws/bags/act/seed0_p0_ok1_ik0.bag
"""

import sys

import numpy as np
import rosbag

JUMP_LIMIT_DEG = 8.0      # per 100 ms step
STATIC_DEG = 0.2          # below this the frame counts as "not moving"


def main():
    path = sys.argv[1]
    bag = rosbag.Bag(path)

    counts = {}
    act = []          # (t, [16])
    claw = []         # (t, [2])
    imgs = {}         # topic -> list of (t, brightness)

    for topic, msg, t in bag.read_messages():
        counts[topic] = counts.get(topic, 0) + 1
        ts = t.to_sec()
        if topic == "/act_action":
            act.append((ts, list(msg.data[:16])))
        elif "claw" in topic and hasattr(msg, "data"):
            try:
                pos = list(msg.data.position)
                if len(pos) >= 2:
                    claw.append((ts, [float(pos[0]), float(pos[1])]))
            except Exception:
                pass
        elif topic.endswith("compressed"):
            if len(imgs.get(topic, [])) < 25:
                try:
                    import cv2
                    arr = np.frombuffer(msg.data, np.uint8)
                    im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    imgs.setdefault(topic, []).append((ts, float(im.mean()) if im is not None else -1))
                except Exception:
                    pass

    dur = bag.get_end_time() - bag.get_start_time()
    print("bag      : %s" % path)
    print("duration : %.1f s" % dur)
    print("\n%-38s %7s %8s" % ("topic", "msgs", "Hz"))
    for t, c in sorted(counts.items()):
        print("%-38s %7d %8.1f" % (t, c, c / max(dur, 1e-6)))

    ok = True
    print("\n--- action stream ---")
    if len(act) < 10:
        print("  MISSING /act_action -> bag unusable")
        ok = False
    else:
        t0 = act[0][0]
        ts = np.array([a[0] for a in act]) - t0
        q = np.array([a[1][:14] for a in act])
        clawv = np.array([a[1][14:16] for a in act])
        # resample to 10 Hz
        grid = np.arange(0, ts[-1], 0.1)
        qi = np.stack([np.interp(grid, ts, q[:, j]) for j in range(14)], axis=1)
        d = np.abs(np.diff(qi, axis=0))
        jumps = np.percentile(d.max(axis=1), [50, 95, 100])
        static = float((d.max(axis=1) < STATIC_DEG).mean()) * 100.0
        print("  frames @10Hz      : %d" % len(grid))
        print("  joint step deg    : median %.2f   p95 %.2f   max %.2f" %
              (jumps[0], jumps[1], jumps[2]))
        print("  static frames     : %.0f %%" % static)
        print("  claw range        : L %.0f..%.0f   R %.0f..%.0f" %
              (clawv[:, 0].min(), clawv[:, 0].max(),
               clawv[:, 1].min(), clawv[:, 1].max()))
        # Distinguish a genuine DISCONTINUITY (isolated spike) from fast motion
        # (several large steps in a row). Only the spike is a problem for ACT.
        dm = d.max(axis=1)                       # max over joints, per step
        spikes = []
        for i in range(1, len(dm) - 1):
            if dm[i] > JUMP_LIMIT_DEG and dm[i - 1] < 0.4 * dm[i] and dm[i + 1] < 0.4 * dm[i]:
                spikes.append((grid[i], int(np.argmax(d[i])) + 1, dm[i]))
        print("  isolated spikes   : %d" % len(spikes))
        for t, j, v in spikes[:5]:
            print("     SPIKE t=%6.2f s  j%02d  %5.1f deg" % (t, j, v))
        peaks = np.argsort(dm)[-3:][::-1]
        print("  fastest motion    : " + ", ".join(
            "t=%.1f j%02d %.0f deg/0.1s" % (grid[i], int(np.argmax(d[i])) + 1, dm[i]) for i in peaks))
        if spikes:
            print("  !! %d isolated jump(s): the command changes direction/step instantly"
                  % len(spikes))
            ok = False
        if static > 45.0:
            print("  !! %.0f %% of frames are static -> timing too slow / long waits" % static)
            ok = False
        if clawv[:, 0].max() == clawv[:, 0].min() and clawv[:, 1].max() == clawv[:, 1].min():
            print("  !! the claw value never changes -> no grip in the data?")
            ok = False

    print("\n--- cameras ---")
    for t, vals in sorted(imgs.items()):
        b = [v for _, v in vals if v >= 0]
        if b:
            print("  %-38s frames=%d  brightness=%.0f" % (t, len(vals), float(np.mean(b))))
            if float(np.mean(b)) < 5:
                print("    !! almost black")
                ok = False
        else:
            print("  %-38s (could not decode)" % t)

    print("\nVERDICT: %s" % ("LOOKS USABLE" if ok else "PROBLEMS FOUND (see !! above)"))
    bag.close()


if __name__ == "__main__":
    main()
