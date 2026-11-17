#!/usr/bin/env python3
"""Closed-loop rollout of a trained ACT policy in the scene-1 simulator.

The policy replaces `task_sequence.py`: it reads the same three cameras and the
measured joints, and publishes arm targets + claw commands on the same topics the
data was collected with (/kuavo_arm_traj, /leju_claw_command).

Start the simulator first (challenge_task.py --scene scene1 --seed N), put the arm
at the ready pose, then:

    python3 run_act_policy.py --ckpt /root/act/runs/act_2h/policy_best.ckpt --duration 40

Ctrl-C stops it; the arm keeps the last commanded pose.
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState, CompressedImage
from kuavo_msgs.msg import sensorsData, lejuClawCommand, lejuClawState

# torch / ACT live outside ROS
for p in ("/root/act_libs", "/root/act", "/root/act/detr"):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2                                              # noqa: E402
import torch                                            # noqa: E402
from policy import ACTPolicy                            # noqa: E402

ARM_NAMES = ["arm_joint_%d" % i for i in range(1, 15)]
CLAW_NAMES = ["left_claw", "right_claw"]
CAMS = ["cam_h", "cam_l", "cam_r"]
CAM_TOPICS = {c: "/%s/color/image_raw/compressed" % c for c in CAMS}
TRAJ_TOPIC = "/kuavo_arm_traj"
CLAW_TOPIC = "/leju_claw_command"

IMG_H, IMG_W = 480, 640
PUB_HZ = 100.0
CLAW_EFFORT = 4.0
MAX_JUMP = 3.0                      # deg per 0.1 s (the bags move ~1.8 deg/100 ms)


class ArmPublisher(object):
    """90->100 Hz publisher that ramps to each new target instead of stepping."""

    def __init__(self, topic, hz=PUB_HZ):
        self.pub = rospy.Publisher(topic, JointState, queue_size=10)
        self.lock = threading.Lock()
        self.cur = [0.0] * 14
        self.start = [0.0] * 14
        self.goal = [0.0] * 14
        self.t0 = time.time()
        self.ready = False
        self.stop = threading.Event()
        threading.Thread(target=self._run, args=(hz,), daemon=True).start()

    def set(self, target):
        with self.lock:
            self.start = list(self.cur)
            self.goal = [float(v) for v in target]
            self.t0 = time.time()
            self.cur = list(self.start)

    def _run(self, hz):
        rate = rospy.Rate(hz)
        while not self.stop.is_set() and not rospy.is_shutdown():
            with self.lock:
                a = min(1.0, (time.time() - self.t0) / 0.10)
                self.cur = [s + (g - s) * a for s, g in zip(self.start, self.goal)]
                pos = list(self.cur)
                ready = self.ready
            if ready:
                msg = JointState()
                msg.header.stamp = rospy.Time.now()
                msg.name = ARM_NAMES
                msg.position = pos
                self.pub.publish(msg)
            rate.sleep()


class ClawPublisher(object):
    def __init__(self, topic, hz=50.0):
        self.pub = rospy.Publisher(topic, lejuClawCommand, queue_size=10)
        self.pos = [0.0, 0.0]
        self.stop = threading.Event()
        threading.Thread(target=self._run, args=(hz,), daemon=True).start()

    def set(self, left, right):
        self.pos = [float(left), float(right)]

    def _run(self, hz):
        rate = rospy.Rate(hz)
        while not self.stop.is_set() and not rospy.is_shutdown():
            msg = lejuClawCommand()
            msg.header.stamp = rospy.Time.now()
            msg.data.name = list(CLAW_NAMES)
            msg.data.position = list(self.pos)
            msg.data.velocity = [80.0, 80.0]
            msg.data.effort = [CLAW_EFFORT, CLAW_EFFORT]
            self.pub.publish(msg)
            rate.sleep()


class State(object):
    def __init__(self):
        self.q = None           # 14 joints [deg]
        self.claw = None        # 2 claws [%]
        self.img = {c: None for c in CAMS}
        self.t_img = {c: 0.0 for c in CAMS}
        self.lock = threading.Lock()

        rospy.Subscriber("/sensors_data_raw", sensorsData, self._on_state, queue_size=1)
        rospy.Subscriber("/leju_claw_state", lejuClawState, self._on_claw, queue_size=1)
        for c in CAMS:
            rospy.Subscriber(CAM_TOPICS[c], CompressedImage,
                             lambda m, c=c: self._on_img(m, c), queue_size=1)

    def _on_state(self, msg):
        q = msg.joint_data.joint_q
        if len(q) >= 27:
            with self.lock:
                self.q = [np.degrees(q[13 + i]) for i in range(14)]

    def _on_claw(self, msg):
        p = list(msg.data.position)
        if len(p) >= 2:
            with self.lock:
                self.claw = [float(p[0]), float(p[1])]

    def _on_img(self, msg, cam):
        buf = np.frombuffer(msg.data, np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            return
        if im.shape[1] != IMG_W or im.shape[0] != IMG_H:
            im = cv2.resize(im, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        with self.lock:
            self.img[cam] = im[:, :, ::-1]          # BGR -> RGB (as in training)
            self.t_img[cam] = time.time()


# ── arm PD/PID profile ───────────────────────────────────────────────────────
# Same per-joint numbers task_sequence.py writes at startup (kp / ki / kd / clamp).
# The controller also has an "inert" guard that wipes the integrator whenever the
# SCALAR /arm_pd/{kp,kd,ki} are all zero, so those are always set to a non-zero
# sentinel here — otherwise ki_joints has no effect at all.
PID_KP = [8.0, 8.0, 8.0, 12.0, 20.0, 30.0, 20.0] * 2
PID_KI = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0] * 2
PID_KD = [2.0, 2.0, 2.0, 3.0, 1.5, 2.0, 1.5] * 2
PID_CLAMP = [6.0, 6.0, 6.0, 8.0, 10.0, 12.0, 10.0] * 2


def ensure_pid():
    """Guarantee that the joint-level PID is actually active during the rollout."""
    existing = {}
    for name, default in (("kp_joints", PID_KP), ("ki_joints", PID_KI),
                          ("kd_joints", PID_KD), ("clamp_joints", PID_CLAMP)):
        cur = rospy.get_param("/arm_pd/" + name, None)
        if cur is None or len(cur) != 14:
            cur = default
            rospy.set_param("/arm_pd/" + name, [float(v) for v in cur])
        existing[name] = cur
    # non-zero scalars: defeats the controller's "patch is inert" reset
    for name, val in (("kp", 1.0), ("kd", 1.0), ("ki", 1.0), ("clamp", 3.0)):
        rospy.set_param("/arm_pd/" + name, val)
    rospy.set_param("/arm_pd/replace", False)
    rospy.set_param("/arm_pd/replace_joints", [0.0] * 14)
    return existing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--fps", type=float, default=None,
                    help="control rate; default = the rate the checkpoint was trained at")
    ap.add_argument("--agg", action="store_true", default=True)
    ap.add_argument("--no-agg", dest="agg", action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print targets, publish nothing")
    ap.add_argument("--freeze-left", action="store_true", default=True,
                    help="keep the left arm at its start pose (default on)")
    ap.add_argument("--move-left", dest="freeze_left", action="store_false")
    ap.add_argument("--max-deg-step", type=float, default=MAX_JUMP,
                    help="max joint change per control step (deg)")
    ap.add_argument("--claw-latch", action="store_true", default=True,
                    help="once the right claw grips, keep it closed (default on)")
    ap.add_argument("--no-claw-latch", dest="claw_latch", action="store_false")
    ap.add_argument("--no-pid", action="store_true",
                    help="do not touch /arm_pd (use whatever gains are on the server)")
    ap.add_argument("--execute-chunk", action="store_true",
                    help="predict once, execute the whole chunk, then re-plan "
                         "(carries the arm through the demonstrated pauses)")
    ap.add_argument("--test-sweep", action="store_true",
                    help="ignore the policy: sweep the right shoulder +-15 deg (plumbing test)")
    ap.add_argument("--save-frame", default=None,
                    help="save the first observation (3 cams + state) to this directory")
    args = ap.parse_args()

    rospy.init_node("act_policy_runner", anonymous=True)

    ck = torch.load(args.ckpt, map_location="cuda")
    stats, cfg = ck["stats"], ck["policy_config"]
    ck_fps = float(ck.get("fps", 10.0))
    if args.fps is None:
        args.fps = ck_fps
    elif abs(args.fps - ck_fps) > 1e-6:
        print("WARNING: --fps %.1f differs from the trained rate %.1f -> the motion will "
              "play %.2fx %s" % (args.fps, ck_fps, ck_fps / args.fps,
                                 "faster" if args.fps > ck_fps else "slower"), flush=True)
    policy = ACTPolicy(cfg)
    policy.load_state_dict(ck["model"])
    policy.cuda()
    policy.eval()
    if not args.no_pid:
        prof = ensure_pid()
        print("arm PID: enabled  ki=%s  (elbow/wrist only), clamp max %.0f"
              % (" ".join("%.0f" % v for v in prof["ki_joints"]),
                 max(prof["clamp_joints"])), flush=True)
    else:
        print("arm PID: left untouched (--no-pid)", flush=True)

    print("loaded %s  (chunk %d, %d cameras)"
          % (args.ckpt, cfg["num_queries"], len(cfg["camera_names"])), flush=True)

    state = State()
    arm = ArmPublisher(TRAJ_TOPIC)
    claw = ClawPublisher(CLAW_TOPIC)

    # the first commanded pose is the current one, so the ramp starts smoothly
    t_start = time.time()
    while not rospy.is_shutdown():
        if state.q is not None and all(state.img[c] is not None for c in CAMS):
            break
        if time.time() - t_start > 20.0:
            print("no sensor data — is the simulator running?", flush=True)
            return 1
        time.sleep(0.2)
    arm.set(state.q)
    arm.ready = not args.dry_run

    lo = stats["qpos_mean"][:14] - 10.0 * stats["qpos_std"][:14]
    hi = stats["qpos_mean"][:14] + 10.0 * stats["qpos_std"][:14]
    left_hold = np.array(state.q[:7]) if args.freeze_left else None
    prev = np.array(state.q, dtype=np.float64)      # commands ramp from the real pose
    claw_pos = [0.0, 0.0]
    claw_latched = False
    cached_chunk = None          # --execute-chunk: the plan being executed
    qm, qs = stats["qpos_mean"], stats["qpos_std"]
    am, ast = stats["action_mean"], stats["action_std"]
    chunk = cfg["num_queries"]

    horizon = int(args.duration * args.fps) + chunk
    all_actions = np.zeros((horizon + chunk, 16), np.float32)

    period = 1.0 / args.fps
    next_t = time.time()
    step = 0
    clipped = 0
    mode = ("execute-chunk (plan once, run it out)" if args.execute_chunk
            else ("temporal aggregation" if args.agg else "re-plan every step"))
    print("running %.0f s at %.0f Hz  mode: %s" % (args.duration, args.fps, mode), flush=True)

    try:
        while not rospy.is_shutdown() and step < int(args.duration * args.fps):
            with state.lock:
                q = state.q
                cl = state.claw if state.claw is not None else [0.0, 0.0]
                imgs = [state.img[c] for c in CAMS]
                fresh = max(state.t_img.values())
            if q is None or any(i is None for i in imgs) or time.time() - fresh > 1.0:
                rospy.sleep(0.05)
                continue

            qpos = np.array(list(q) + list(cl), np.float32)
            img = np.stack(imgs, 0).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            qn = (qpos - qm) / qs

            if args.execute_chunk:
                # one plan, executed to the end before the next query. The policy is
                # only asked every `chunk` steps, so a demonstrated pause in the
                # middle of the plan cannot trap the arm in "keep holding".
                if cached_chunk is None:
                    with torch.inference_mode():
                        a = policy(torch.from_numpy(qn).float().cuda().unsqueeze(0),
                                   torch.from_numpy(img).float().cuda().unsqueeze(0))
                    cached_chunk = a.squeeze(0).cpu().numpy() * ast + am
                act = np.array(cached_chunk[step % chunk], dtype=np.float64)
            else:
                with torch.inference_mode():
                    a = policy(torch.from_numpy(qn).float().cuda().unsqueeze(0),
                               torch.from_numpy(img).float().cuda().unsqueeze(0))
                a = a.squeeze(0).cpu().numpy() * ast + am

                if args.agg:
                    all_actions[step:step + chunk] = a
                    n = step + 1
                    wts = np.exp(-0.01 * np.arange(n)[::-1])      # newest wins
                    wts /= wts.sum()
                    act = (all_actions[:n] * wts[:, None]).sum(0)
                else:
                    act = a[step % chunk]

            if args.test_sweep:
                args.freeze_left = True
                left_hold = np.array(state.q[:7])
                ph = 2.0 * np.pi * step / 20.0        # 2 s period
                raw = np.array(state.q, dtype=np.float64)
                raw[10] += 15.0 * np.sin(ph)          # right shoulder pitch
                raw[7] += 8.0 * np.sin(ph)            # right shoulder yaw
            raw = np.array(act[:14], dtype=np.float64) if not args.test_sweep else raw
            if args.execute_chunk and (step + 1) % chunk == 0:
                cached_chunk = None          # plan finished -> re-plan next step

            if left_hold is not None:
                raw[:7] = left_hold
            # ramp from the last command — never jump, not even on the first step
            joints = prev + np.clip(raw - prev, -args.max_deg_step, args.max_deg_step)
            joints = np.clip(joints, lo, hi)
            if np.any(np.abs(joints - raw) > 0.5):
                clipped += 1
            # deadband: ignore sub-0.3 deg jitter so the command does not chatter
            joints[np.abs(joints - prev) < 0.3] = prev[np.abs(joints - prev) < 0.3]
            prev = joints
            if not args.dry_run:
                arm.set(joints)
                # hysteresis: a claw changes only when the policy is clearly on the
                # other side, otherwise an under-trained model toggles it every step
                for i, idx in enumerate((14, 15)):
                    if args.claw_latch and i == 1 and claw_latched:
                        continue                 # already gripping: never reopen
                    a = float(act[idx])
                    if a > 0.6:
                        claw_pos[i] = 255.0
                    elif a < 0.4:
                        claw_pos[i] = 0.0
                    if i == 1 and claw_pos[1] == 255.0 and args.claw_latch:
                        claw_latched = True
                claw.set(claw_pos[0], claw_pos[1])

            if args.save_frame and step == 0:
                import json
                os.makedirs(args.save_frame, exist_ok=True)
                for c, im in zip(CAMS, imgs):
                    cv2.imwrite(os.path.join(args.save_frame, "live_%s.png" % c),
                                im[:, :, ::-1])
                with open(os.path.join(args.save_frame, "live_state.json"), "w") as f:
                    json.dump({"qpos": list(map(float, qpos)), "t": time.time()}, f)
                print("  saved first frame to %s" % args.save_frame, flush=True)

            if step % 10 == 0:
                cur = np.array(state.q) if state.q is not None else np.zeros(14)
                print("t=%5.1fs  want R %s | meas R %s"
                      % (step / args.fps,
                         " ".join("%6.1f" % v for v in raw[7:]),
                         " ".join("%6.1f" % v for v in cur[7:])), flush=True)
            step += 1
            next_t += period
            rospy.sleep(max(0.0, next_t - time.time()))
    except KeyboardInterrupt:
        pass
    finally:
        arm.ready = False
        print("stopped after %d steps (%d clipped values)" % (step, clipped), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
