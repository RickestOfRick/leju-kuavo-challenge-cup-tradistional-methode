#!/usr/bin/env python3
"""Plot the end-effector XYZ error logged by arm_test_control._verify().

Each finished ABS move appends one row to /tmp/ee_err.csv:
    t, dx_mm, dy_mm, dz_mm, |e|_mm      (target - actual)

Usage (inside the container, after a few moves):
    python3 ee_plot_csv.py                # opens a window
    python3 ee_plot_csv.py --save         # writes /tmp/ee_err.png instead
    python3 ee_plot_csv.py --reset        # delete the log first
"""

import csv
import os
import sys

PATH = "/tmp/ee_err.csv"


def main():
    if "--reset" in sys.argv and os.path.exists(PATH):
        os.remove(PATH)
        print("log cleared:", PATH)
        return

    if not os.path.exists(PATH):
        print("no log yet (%s) — run some moves in the task first" % PATH)
        return

    rows = []
    with open(PATH) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["dx"]), float(r["dy"]), float(r["dz"]), float(r["norm"])))
            except (KeyError, ValueError):
                continue
    if not rows:
        print("log is empty")
        return

    import matplotlib
    matplotlib.use("Agg" if "--save" in sys.argv else "TkAgg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = range(len(rows))
    for i, name in enumerate(("dx", "dy", "dz")):
        ax1.plot(list(x), [r[i] for r in rows], marker="o", ms=3, label=name)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("target - actual [mm]")
    ax1.set_title("end-effector error per move (%d moves)" % len(rows))
    ax1.grid(alpha=0.3)
    ax1.legend(ncol=3)

    ax2.plot(list(x), [r[3] for r in rows], color="crimson", marker="o", ms=3)
    ax2.set_ylabel("|error| [mm]")
    ax2.set_xlabel("move index")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if "--save" in sys.argv:
        out = "/tmp/ee_err.png"
        plt.savefig(out, dpi=130)
        print("saved", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
