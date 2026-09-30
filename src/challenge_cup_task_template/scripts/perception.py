#!/usr/bin/env python3
"""perception.py — step-by-step classical perception for the Kuavo sim.

Test ONE step at a time:

    python3 /root/kuavo_ws/perception.py --step cloud     # RGB-D -> point cloud
    python3 /root/kuavo_ws/perception.py --step ransac    # fit + drop table plane
    python3 /root/kuavo_ws/perception.py --step cluster   # cluster + PCA per object

`cloud` captures the head camera once, deprojects depth into a 3D point cloud and
saves it. It also tries to transform it to base_link (prints whether TF works).
`ransac` loads the saved cloud, fits the dominant plane (the table) with RANSAC,
removes it and saves the remaining object points.
`cluster` clusters those points and prints (x, y, z, yaw) per object.

Saved files live next to this script: last_cloud.npy, last_points_frame.txt,
objects.npy.
"""

import argparse
import os
import sys

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage

try:
    import cv2
except Exception as e:  # pragma: no cover
    print("opencv (cv2) not available: %s" % e)
    sys.exit(1)

# ───────────────────────────── config ─────────────────────────────
COLOR_TOPIC = "/cam_h/color/image_raw/compressed"
DEPTH_TOPIC = "/cam_h/depth/image_raw/compressedDepth"
INFO_TOPIC = "/cam_h/color/camera_info"
TARGET_FRAME = "base_link"

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD_FILE = os.path.join(HERE, "last_cloud.npy")
FRAME_FILE = os.path.join(HERE, "last_points_frame.txt")
OBJECTS_FILE = os.path.join(HERE, "objects.npy")

DEPTH_STRIDE = 4          # sample every Nth pixel
RANSAC_ITERS = 500
RANSAC_DIST = 0.008       # inlier distance to plane (m)
CLUSTER_TOL = 0.02        # cluster distance (m)
MIN_CLUSTER = 30          # min points for a cluster
# ──────────────────────────────────────────────────────────────────


def log(msg):
    print(msg, flush=True)


def decode_depth(msg):
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    for off in (12, 8, 4, 0):
        try:
            img = cv2.imdecode(raw[off:], cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None
        if img is not None and img.ndim == 2 and img.size > 0:
            return img
    return None


def to_metres(img):
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 1000.0
    return img.astype(np.float32)


# ───────────────────────────── TF ─────────────────────────────
class Tf(object):
    def __init__(self):
        self.buffer = None
        self.listener = None
        try:
            import tf2_ros
            self.buffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.buffer)
        except Exception as e:
            log("  (tf2 not available: %s)" % e)

    def lookup(self, target, source, timeout=2.0):
        """Returns 4x4 matrix target<-source, or None."""
        if self.buffer is None:
            return None
        try:
            tr = self.buffer.lookup_transform(target, source, rospy.Time(0),
                                              rospy.Duration(timeout))
        except Exception as e:
            log("  TF %s <- %s failed: %s" % (target, source, e))
            return None
        t = tr.transform.translation
        q = tr.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        n = np.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / n, y / n, z / n, w / n
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = [t.x, t.y, t.z]
        return M


def transform_points(M, pts):
    return (M[:3, :3] @ pts.T).T + M[:3, 3]


# ───────────────────────────── steps ─────────────────────────────
def step_cloud(args):
    info = rospy.wait_for_message(INFO_TOPIC, CameraInfo, timeout=args.timeout)
    depth = rospy.wait_for_message(DEPTH_TOPIC, CompressedImage, timeout=args.timeout)
    img = decode_depth(depth)
    if img is None:
        log("FAILED to decode %s (format=%s)" % (DEPTH_TOPIC, depth.format))
        return
    d = to_metres(img)
    K = list(info.K)
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]

    log("depth image : %s dtype=%s  frame_id=%r" % (img.shape, img.dtype, depth.header.frame_id))
    log("camera_info : fx=%.2f fy=%.2f cx=%.2f cy=%.2f  frame_id=%r"
        % (fx, fy, cx, cy, info.header.frame_id))

    h, w = d.shape
    us = np.arange(0, w, args.stride)
    vs = np.arange(0, h, args.stride)
    z = d[np.ix_(vs, us)]
    uu, vv = np.meshgrid(us, vs)
    mask = np.isfinite(z) & (z > 0.0)
    zz = z[mask]
    pts = np.stack([(uu[mask] - cx) * zz / fx,
                    (vv[mask] - cy) * zz / fy,
                    zz], axis=1)
    log("points      : %d valid (camera optical frame)" % len(pts))
    if len(pts) == 0:
        log("no valid depth pixels — is the scene running?")
        return
    log("  camera-frame bounds  x[%.2f %.2f] y[%.2f %.2f] z[%.2f %.2f]"
        % (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(),
           pts[:, 2].min(), pts[:, 2].max()))

    frame = depth.header.frame_id or info.header.frame_id
    tf = Tf()
    rospy.sleep(0.3)
    M = tf.lookup(TARGET_FRAME, frame) if frame else None
    if M is not None:
        pts = transform_points(M, pts)
        log("  -> transformed to %s" % TARGET_FRAME)
        log("  base-frame bounds    x[%.2f %.2f] y[%.2f %.2f] z[%.2f %.2f]"
            % (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(),
               pts[:, 2].min(), pts[:, 2].max()))
        frame = TARGET_FRAME
    else:
        log("  !! no TF to %s — keeping the camera frame" % TARGET_FRAME)

    np.save(CLOUD_FILE, pts)
    with open(FRAME_FILE, "w") as f:
        f.write(frame)
    log("saved %s (%d points, frame=%s)" % (CLOUD_FILE, len(pts), frame))


def ransac_plane(pts, iters, dist, seed=0):
    n = len(pts)
    if n < 3:
        return None, None
    rng = np.random.default_rng(seed)
    best_in = None
    best = None
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -float(nrm.dot(p0))
        inl = np.abs(pts @ nrm + d) < dist
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (nrm, d)
    return best, best_in


def step_ransac(args):
    if not os.path.exists(CLOUD_FILE):
        log("no %s — run --step cloud first" % CLOUD_FILE)
        return
    pts = np.load(CLOUD_FILE)
    frame = open(FRAME_FILE).read().strip() if os.path.exists(FRAME_FILE) else "?"
    log("loaded %d points (frame=%s)" % (len(pts), frame))

    (nrm, d), inl = ransac_plane(pts, args.iters, args.dist)
    if nrm is None:
        log("RANSAC failed (too few points)")
        return
    # orient the normal upward (positive z) for readability
    if nrm[2] < 0:
        nrm, d = -nrm, -d
    inl_n = int(inl.sum())
    log("plane       : normal=(%.3f, %.3f, %.3f)  d=%.4f" % (nrm[0], nrm[1], nrm[2], d))
    log("  tilt from vertical: %.1f deg" % np.degrees(np.arccos(min(1.0, nrm[2]))))
    log("  inliers    : %d / %d (%.1f%%)" % (inl_n, len(pts), 100.0 * inl_n / len(pts)))
    if inl_n:
        on = pts[inl]
        c = on.mean(axis=0)
        log("  plane center (%.3f, %.3f, %.3f)" % (c[0], c[1], c[2]))

    objs = pts[~inl]
    log("after removing the plane: %d points" % len(objs))
    if len(objs):
        log("  obj bounds  x[%.2f %.2f] y[%.2f %.2f] z[%.2f %.2f]"
            % (objs[:, 0].min(), objs[:, 0].max(), objs[:, 1].min(), objs[:, 1].max(),
               objs[:, 2].min(), objs[:, 2].max()))
    np.save(OBJECTS_FILE, objs)
    log("saved %s" % OBJECTS_FILE)


def step_cluster(args):
    if not os.path.exists(OBJECTS_FILE):
        log("no %s — run --step ransac first" % OBJECTS_FILE)
        return
    pts = np.load(OBJECTS_FILE)
    log("loaded %d object points" % len(pts))

    labels = _cluster(pts, args.tol)
    log("clusters    : %d" % (labels.max() + 1 if len(labels) else 0))
    for k in range(labels.max() + 1 if len(labels) else 0):
        c = pts[labels == k]
        if len(c) < args.min_cluster:
            continue
        center = c.mean(axis=0)
        yaw = _pca_yaw(c)
        log("  #%d  n=%4d  center=(%.3f, %.3f, %.3f)  yaw=%.1f deg  top_z=%.3f"
            % (k, len(c), center[0], center[1], center[2],
               np.degrees(yaw), c[:, 2].max()))


def _cluster(pts, tol):
    """Voxel-grid BFS clustering (no external deps)."""
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=int)
    keys = np.floor(pts / tol).astype(np.int64)
    buckets = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)

    labels = -np.ones(n, dtype=int)
    nxt = 0
    neigh = [(dx, dy, dz) for dx in (-1, 0, 1)
             for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for seed in range(n):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = nxt
        while stack:
            i = stack.pop()
            kx, ky, kz = keys[i]
            for dx, dy, dz in neigh:
                b = buckets.get((kx + dx, ky + dy, kz + dz))
                if not b:
                    continue
                for j in b:
                    if labels[j] < 0:
                        labels[j] = nxt
                        stack.append(j)
        nxt += 1
    return labels


def _pca_yaw(pts):
    xy = pts[:, :2]
    if len(xy) < 3:
        return 0.0
    xy = xy - xy.mean(axis=0)
    cov = np.cov(xy.T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, np.argmax(w)]
    return np.arctan2(axis[1], axis[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["cloud", "ransac", "cluster"], required=True)
    ap.add_argument("--stride", type=int, default=DEPTH_STRIDE)
    ap.add_argument("--iters", type=int, default=RANSAC_ITERS)
    ap.add_argument("--dist", type=float, default=RANSAC_DIST)
    ap.add_argument("--tol", type=float, default=CLUSTER_TOL)
    ap.add_argument("--min-cluster", type=int, default=MIN_CLUSTER)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    rospy.init_node("perception_" + args.step, anonymous=True)
    if args.step == "cloud":
        step_cloud(args)
    elif args.step == "ransac":
        step_ransac(args)
    else:
        step_cluster(args)


if __name__ == "__main__":
    main()
