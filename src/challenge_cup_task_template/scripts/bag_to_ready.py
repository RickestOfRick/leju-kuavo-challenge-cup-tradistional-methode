#!/usr/bin/env python3
"""bag_to_ready.py — read a rosbag and print the arm joint pose (14 joints).

    python3 bag_to_ready.py /path/to/go.bag
    python3 bag_to_ready.py /path/to/go.bag --frame first
    python3 bag_to_ready.py /path/to/go.bag --out ready.json

Reads /kuavo_arm_traj (sensor_msgs/JointState, values in DEGREES) if present,
otherwise /sensors_data_raw (kuavo_msgs/sensorsData, radians in joint_q[13:27]).
Prints the chosen frame as a 14-value list you can paste into READY_JOINTS.
"""

import argparse
import json
import sys


def extract_deg(topic, msg):
    """Return 14 arm joint angles in DEGREES, or None."""
    if topic == "/kuavo_arm_traj":
        pos = list(getattr(msg, "position", []))
        if len(pos) >= 14:
            return [float(v) for v in pos[:14]]          # already degrees
        return None
    if topic == "/sensors_data_raw":
        q = list(msg.joint_data.joint_q)
        if len(q) >= 27:
            import math
            return [math.degrees(v) for v in q[13:27]]   # radians -> degrees
        if len(q) >= 26:
            import math
            return [math.degrees(v) for v in q[12:26]]
    return None


def load_arm_traj(path, topic=None):
    """Load a bag's arm joint trajectory as a list of (t_seconds, [14 degrees])."""
    import rosbag

    topics = [topic] if topic else ["/kuavo_arm_traj", "/sensors_data_raw"]
    frames = []
    with rosbag.Bag(path) as bag:
        for topic_name, msg, t in bag.read_messages(topics=topics):
            deg = extract_deg(topic_name, msg)
            if deg is not None:
                frames.append((t.to_sec(), deg))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--frame", choices=["last", "first"], default="last")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        import rosbag
    except Exception as e:
        print("rosbag not available (run inside the container): %s" % e)
        sys.exit(1)

    want = ["/kuavo_arm_traj", "/sensors_data_raw"]
    first, last, counts = {}, {}, {}

    with rosbag.Bag(args.bag) as bag:
        info = bag.get_type_and_topic_info().topics
        print("topics in bag:")
        for t in sorted(info):
            print("  %-38s %-40s %d msgs" % (t, info[t].msg_type,
                                             info[t].message_count))
        for topic, msg, _ in bag.read_messages(topics=want):
            deg = extract_deg(topic, msg)
            if deg is None:
                continue
            counts[topic] = counts.get(topic, 0) + 1
            first.setdefault(topic, deg)
            last[topic] = deg

    if not last:
        print("\nno arm joint data found in %s" % want)
        return

    print("\nframes read:")
    for t in last:
        print("  %-38s %d msg(s)" % (t, counts[t]))

    print()
    for topic in last:
        for label, data in (("first", first.get(topic)), ("last", last.get(topic))):
            if data is None:
                continue
            print("%s %s  (degrees):" % (topic, label))
            print("  [%s]" % ", ".join("%.2f" % v for v in data))

    chosen_topic = "/kuavo_arm_traj" if "/kuavo_arm_traj" in last else list(last)[0]
    chosen = (first if args.frame == "first" else last)[chosen_topic]
    print("\nchosen %s frame of %s:" % (args.frame, chosen_topic))
    print("READY_JOINTS = [%s]" % ", ".join("%.2f" % v for v in chosen))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"topic": chosen_topic, "frame": args.frame,
                       "joints_deg": chosen}, f, indent=2)
        print("saved %s" % args.out)


if __name__ == "__main__":
    main()
