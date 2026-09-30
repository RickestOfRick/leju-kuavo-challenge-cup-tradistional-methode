#!/usr/bin/env python3
"""cam_view.py — view a simulator camera topic (color JPEG or compressed depth).

    python3 /root/kuavo_ws/cam_view.py                       # head color
    python3 /root/kuavo_ws/cam_view.py --topic /cam_r/color/image_raw/compressed
    python3 /root/kuavo_ws/cam_view.py --topic /cam_h/depth/image_raw/compressedDepth

For depth, values are printed (min/median/max in metres) so the encoding can be
verified. Press q in the window to quit.
"""

import argparse
import sys

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

try:
    import cv2
except Exception as e:  # pragma: no cover
    print("opencv (cv2) not available: %s" % e)
    sys.exit(1)


def decode_depth(msg):
    """Decode a compressedDepth CompressedImage. Returns (image, header_bytes)."""
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    for off in (12, 8, 4, 0):
        try:
            img = cv2.imdecode(raw[off:], cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None
        if img is not None and img.ndim == 2 and img.size > 0:
            return img, off
    return None, -1


def to_metres(img):
    """Best-effort depth -> metres."""
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 1000.0
    return img.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/cam_h/color/image_raw/compressed")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale")
    args = ap.parse_args()

    is_depth = "depth" in args.topic
    rospy.init_node("cam_view", anonymous=True)
    shown = {"n": 0}

    def cb(msg):
        if is_depth:
            img, off = decode_depth(msg)
            if img is None:
                print("decode failed (topic=%s format=%s bytes=%d)"
                      % (msg.header.frame_id, msg.format, len(msg.data)), flush=True)
                return
            d = to_metres(img)
            valid = d[np.isfinite(d) & (d > 0)]
            if valid.size:
                print("depth %s frame=%r off=%d dtype=%s min=%.3f median=%.3f max=%.3f m"
                      % (img.shape, msg.header.frame_id, off, img.dtype,
                         float(valid.min()), float(np.median(valid)), float(valid.max())),
                      flush=True)
            vis = np.clip(d, 0.2, 2.0)
            vis = ((vis - 0.2) / 1.8 * 255.0).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        else:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            vis = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if vis is None:
                print("decode failed (format=%s bytes=%d)" % (msg.format, len(msg.data)),
                      flush=True)
                return
            if shown["n"] == 0:
                print("color %s frame=%r format=%s" % (vis.shape, msg.header.frame_id, msg.format),
                      flush=True)

        shown["n"] += 1
        if args.scale != 1.0:
            vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale)
        cv2.imshow("cam_view: " + args.topic, vis)
        cv2.waitKey(1)

    rospy.Subscriber(args.topic, CompressedImage, cb, queue_size=1)
    print("waiting for %s (depth=%s) ..." % (args.topic, is_depth), flush=True)
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
