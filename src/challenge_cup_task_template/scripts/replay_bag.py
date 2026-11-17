#!/usr/bin/env python3
"""Replay a recorded ACT bag on the robot (arm + claw).

Unlike `rosbag play`, this feeds the recorded joint targets into the normal
100 Hz stream and republished claw commands, so it works with the standard arm
interface (no /sensors_data_raw conflicts).

    python3 replay_bag.py /root/kuavo_ws/bags/act/seed0_p0_ok1_ik0.bag
    python3 replay_bag.py <bag> --speed 2.0
    python3 replay_bag.py <bag> 1      # same as --speed 1

Start the sim first and leave the arm at the ready pose.
"""

import sys
import time

import rosbag
import rospy

import arm_test_control as AC
from kuavo_msgs.msg import lejuClawCommand, armTargetPoses


def main():
    path = sys.argv[1]
    speed = 1.0
    if "--speed" in sys.argv:
        speed = float(sys.argv[sys.argv.index("--speed") + 1])
    elif len(sys.argv) > 2:
        try:
            speed = float(sys.argv[2])
        except ValueError:
            pass

    rospy.init_node("replay_bag", anonymous=True)

    if "--stiff" in sys.argv:
        # firmer arm while watching: shoulders stay 0 (the MPC owns them), elbow and
        # wrist get a real P+D so the replay follows the recording closely
        rospy.set_param("/arm_pd/replace", False)
        rospy.set_param("/arm_pd/replace_joints", [0.0] * 14)
        # EXACTLY the "combined / free" set from task_sequence, so the replay matches
        # the arm that produced the bag
        rospy.set_param("/arm_pd/kp_joints",
                        [0, 0, 0, 12, 20, 30, 20, 0, 0, 0, 12, 20, 30, 20])
        rospy.set_param("/arm_pd/kd_joints",
                        [0, 0, 0, 3, 1.5, 2, 1.5, 0, 0, 0, 3, 1.5, 2, 1.5])
        rospy.set_param("/arm_pd/ki_joints",
                        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0])
        rospy.set_param("/arm_pd/clamp_joints",
                        [4, 4, 4, 8, 10, 12, 10, 4, 4, 4, 8, 10, 12, 10])
        print("arm PD = task_sequence 'combined / free' (same as the recording)")
    ar = AC.ArmTestControl()
    ar.setup()
    rospy.sleep(1.0)
    # stop the controller's own 50 Hz claw hold, otherwise it fights the replayed
    # commands on the same topic (the claw then twitches instead of closing)
    try:
        if ar.claw_hold is not None:
            ar.claw_hold.stop()
            print("claw hold thread stopped (replay owns the claw)")
    except Exception as e:
        print("could not stop the claw hold thread: %s" % e)
    claw_pub = rospy.Publisher("/leju_claw_command", lejuClawCommand, queue_size=10)
    # the MPC needs its own target too, otherwise it keeps its old plan and fights
    # the streamed positions (the arm then moves weakly)
    mpc_pub = rospy.Publisher("/humanoid_mpc_target_arm", armTargetPoses, queue_size=10)
    rospy.sleep(0.3)

    items = []
    bag = rosbag.Bag(path)
    for topic, msg, t in bag.read_messages(
            topics=["/kuavo_arm_traj", "/leju_claw_command", "/gripper/command"]):
        items.append((t.to_sec(), topic, msg))
    bag.close()
    if not items:
        print("no /kuavo_arm_traj in that bag")
        return

    print("replaying %d messages, %.1f s at speed %.1fx"
          % (len(items), items[-1][0] - items[0][0], speed))

    # --- go to the bag's START POSE first (the example assumes its own ready pose)
    first = next((m for _t, tp, m in items if tp == "/kuavo_arm_traj"), None)
    if first is not None:
        lead = 3.0
        if "--lead-in" in sys.argv:
            lead = float(sys.argv[sys.argv.index("--lead-in") + 1])
        cur = list(ar.stream.get())
        tgt = list(first.position)
        n = max(1, int(lead * AC.STREAM_HZ))
        print("moving to the bag's start pose over %.1f s ..." % lead)
        for k in range(n + 1):
            if rospy.is_shutdown():
                return
            a = AC.ArmTestControl._ramp(float(k) / n)
            q = [cur[j] + (tgt[j] - cur[j]) * a for j in range(14)]
            ar.stream.set(q)
            tp = armTargetPoses()
            tp.times = [0.01]
            tp.values = q
            mpc_pub.publish(tp)
            rospy.sleep(1.0 / AC.STREAM_HZ)
        # hold at the start pose so the arm settles before the demonstration begins
        for _ in range(int(AC.STREAM_HZ * 0.5)):
            ar.stream.set(tgt)
            tp = armTargetPoses()
            tp.times = [0.01]
            tp.values = tgt
            mpc_pub.publish(tp)
            rospy.sleep(1.0 / AC.STREAM_HZ)
        print("at the bag start pose: [%s]" % ", ".join("%.1f" % v for v in tgt[:7]))

    t0 = items[0][0]
    wall0 = time.time()
    mpc_tick = 0
    for ts, topic, msg in items:
        target = (ts - t0) / speed
        while time.time() - wall0 < target:
            time.sleep(0.005)
            if rospy.is_shutdown():
                return
        if topic == "/kuavo_arm_traj":
            ar.stream.set(list(msg.position))
            # feed the MPC target at ~10 Hz, not 100 Hz: a new 50 ms trajectory every
            # 10 ms makes the optimiser keep restarting and the arm crawls
            mpc_tick += 1
            if mpc_tick % 10 == 0:
                tp = armTargetPoses()
                tp.times = [0.15]
                tp.values = list(msg.position)
                mpc_pub.publish(tp)
        elif topic == "/gripper/command":
            c = lejuClawCommand()
            c.data.name = ["left_claw", "right_claw"]
            p = list(msg.position)
            c.data.position = [float(p[0]), float(p[1])] if len(p) >= 2 else [0.0, 0.0]
            c.data.velocity = [80.0, 80.0]
            c.data.effort = [2.0, 2.0]
            claw_pub.publish(c)
        else:
            claw_pub.publish(msg)
    print("replay finished")


if __name__ == "__main__":
    main()
