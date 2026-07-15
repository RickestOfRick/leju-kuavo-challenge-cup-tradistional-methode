#!/usr/bin/env python3
"""extract_bag_frames.py — 从 rosbag 提取头部相机帧到图片目录

用法:
  python3 tools/extract_bag_frames.py bags/scene2/scene2_seed_3390_xxx.bag -o my_images/
  python3 tools/extract_bag_frames.py bags/scene2/ -o my_images/ --every 30  # 每30帧取1帧
"""
import argparse, os, sys
import cv2, numpy as np

def main():
    p = argparse.ArgumentParser(description="Extract head camera frames from rosbag")
    p.add_argument("input", help="rosbag file or directory of .bag files")
    p.add_argument("-o", "--output", default="./frames", help="Output directory")
    p.add_argument("--every", type=int, default=10, help="Extract every Nth frame")
    p.add_argument("--max", type=int, default=None, help="Max frames to extract")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find bag files
    if os.path.isdir(args.input):
        bags = sorted(f for f in os.listdir(args.input) if f.endswith(".bag") and not f.endswith(".active"))
        bags = [os.path.join(args.input, f) for f in bags]
    else:
        bags = [args.input]

    # Lazy import (ROS env required)
    import rosbag
    from cv_bridge import CvBridge
    bridge = CvBridge()

    total = 0
    for bag_path in bags:
        print(f"Opening {os.path.basename(bag_path)} ...")
        bag = rosbag.Bag(bag_path, "r")
        frame_idx = 0
        for topic, msg, t in bag.read_messages(topics=["/cam_h/color/image_raw/compressed"]):
            if frame_idx % args.every != 0:
                frame_idx += 1; continue
            if args.max is not None and total >= args.max:
                break
            try:
                img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                print(f"  skip frame {frame_idx}: {e}")
                frame_idx += 1; continue
            ts_ns = t.to_nsec()
            seed_tag = ""
            for part in bag_path.split("_"):
                if part.isdigit() and len(part) == 4:
                    seed_tag = f"_s{part}"
                    break
            fname = f"cam_h{seed_tag}_{ts_ns}.jpg"
            cv2.imwrite(os.path.join(args.output, fname), img)
            total += 1
            frame_idx += 1
        bag.close()
        if args.max is not None and total >= args.max:
            break

    print(f"Done: {total} frames saved to {args.output}/")

if __name__ == "__main__":
    main()
