#!/usr/bin/env python3
"""Live arm-joint error monitor: commanded vs measured, for the wrist joints.

  /kuavo_arm_traj   JointState, position[] in DEGREES, order = L1..L7, R1..R7
  /sensors_data_raw  kuavo_msgs/sensorsData, joint_q in RADIANS,
                     order = legs(12), waist(1), arms(14)  -> arms start at 13

Prints the 6 wrist joints (L5,L6,L7,R5,R6,R7) at 5 Hz, plus the max.

Run inside the container with the sim up:
    python3 wrist_err.py            # wrists only
    python3 wrist_err.py all        # all 14 arm joints
"""

import collections
import math
import sys

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

ARM_OFFSET = 13          # arms start at index 13 in joint_q (12 legs + 1 waist)
NAMES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
WRIST = [4, 5, 6, 11, 12, 13]        # indices inside the 14 arm joints

state = {"des": None, "act": None}


def on_traj(msg):
    if len(msg.position) >= 14:
        state["des"] = [float(v) for v in msg.position[:14]]


def on_sensors(msg):
    q = msg.joint_data.joint_q
    if len(q) >= ARM_OFFSET + 14:
        state["act"] = [math.degrees(q[ARM_OFFSET + i]) for i in range(14)]


def run_plot(show, window=300):
    """Live matplotlib plot of the selected joints' errors (degrees)."""
    import matplotlib
    matplotlib.use("TkAgg" if "TkAgg" else "Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    hist = {i: collections.deque(maxlen=window) for i in show}
    fig, ax = plt.subplots(figsize=(10, 5))
    lines = {}
    for i in show:
        ln, = ax.plot([], [], label=NAMES[i])
        lines[i] = ln
    ax.set_xlabel("samples (5 Hz)")
    ax.set_ylabel("commanded - measured  [deg]")
    ax.set_title("wrist joint error  (0 = holding the commanded pose)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.axhline(0.0, color="k", lw=0.8)

    def update(_):
        if state["des"] is not None and state["act"] is not None:
            for i in show:
                hist[i].append(state["des"][i] - state["act"][i])
                lines[i].set_data(range(len(hist[i])), list(hist[i]))
            ax.set_xlim(0, window)
            vals = [v for i in show for v in hist[i]]
            if vals:
                m = max(10.0, max(abs(v) for v in vals) * 1.2)
                ax.set_ylim(-m, m)
        return list(lines.values())

    ani = FuncAnimation(fig, update, interval=200, blit=False)
    plt.show()
    return ani


def main():
    rospy.init_node("wrist_err", anonymous=True)
    rospy.Subscriber("/kuavo_arm_traj", JointState, on_traj, queue_size=1)
    from kuavo_msgs.msg import sensorsData
    rospy.Subscriber("/sensors_data_raw", sensorsData, on_sensors, queue_size=1)

    pub = rospy.Publisher("/wrist_err", Float64MultiArray, queue_size=1)
    show = WRIST if len(sys.argv) < 2 or sys.argv[1] != "all" else list(range(14))
    if "--plot" in sys.argv:
        run_plot(show)
        return
    rate = rospy.Rate(5.0)
    print("waiting for /kuavo_arm_traj + /sensors_data_raw ...")
    while not rospy.is_shutdown():
        if state["des"] is not None and state["act"] is not None:
            errs = [state["des"][i] - state["act"][i] for i in show]
            m = Float64MultiArray()
            m.data = errs                      # same order as `show`
            pub.publish(m)
            line = "  ".join("%s %+6.1f" % (NAMES[i], e) for i, e in zip(show, errs))
            worst = max(errs, key=abs)
            print("%s   | max %+6.1f deg" % (line, worst), flush=True)
        rate.sleep()


if __name__ == "__main__":
    main()
