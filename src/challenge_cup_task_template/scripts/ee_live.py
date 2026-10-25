#!/usr/bin/env python3
"""LIVE window of the end-effector XYZ error.

Reads /tmp/ee_err.csv (written by arm_test_control._verify after every ABS move)
and redraws twice a second. No ROS topics involved, so it cannot miss messages.

    python3 ee_live.py          # opens the window
    python3 ee_live.py --reset  # clear the log and start fresh
"""

import csv
import os
import sys

PATH = "/tmp/ee_err.csv"
MAX_POINTS = 60


def read_rows():
    if not os.path.exists(PATH):
        return []
    out = []
    try:
        with open(PATH) as f:
            for r in csv.DictReader(f):
                try:
                    out.append((float(r["dx"]), float(r["dy"]),
                                float(r["dz"]), float(r["norm"])))
                except (KeyError, ValueError):
                    continue
    except OSError:
        pass
    return out[-MAX_POINTS:]


def main():
    if "--reset" in sys.argv and os.path.exists(PATH):
        os.remove(PATH)
        print("log cleared")

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.canvas.set_window_title("end-effector error (target - actual)")

    while True:
        rows = read_rows()
        ax1.clear()
        ax2.clear()
        if rows:
            x = list(range(len(rows)))
            for i, name in enumerate(("dx", "dy", "dz")):
                ax1.plot(x, [r[i] for r in rows], marker="o", ms=3, label=name)
            ax1.axhline(0, color="k", lw=0.8)
            ax1.set_ylabel("target - actual [mm]")
            ax1.set_title("end-effector error  (%d moves)" % len(rows))
            ax1.grid(alpha=0.3)
            ax1.legend(ncol=3, loc="upper right")
            ax2.plot(x, [r[3] for r in rows], color="crimson", marker="o", ms=3)
            ax2.set_ylabel("|error| [mm]")
            ax2.set_xlabel("move index")
            ax2.grid(alpha=0.3)
        else:
            ax1.set_title("waiting for /tmp/ee_err.csv (run a move in the task)")
        plt.pause(0.5)


if __name__ == "__main__":
    main()
