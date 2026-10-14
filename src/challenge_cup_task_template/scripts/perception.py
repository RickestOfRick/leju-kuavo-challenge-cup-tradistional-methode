#!/usr/bin/env python3
"""perception.py — step-by-step classical perception for the Kuavo sim.

Test ONE step at a time:

    python3 perception.py --step cloud     # RGB-D -> point cloud
    python3 perception.py --step ransac    # fit + drop table plane
    python3 perception.py --step cluster   # cluster + PCA per object
    # add --show to any step for an overlay window drawn on the head camera

`cloud` captures the head camera once, deprojects depth into a 3D point cloud and
saves it. It also tries to transform it to base_link (prints whether TF works).
`ransac` loads the saved cloud, fits the dominant plane (the table) with RANSAC,
removes it and saves the remaining object points.
`cluster` clusters those points and prints (x, y, z, yaw) per object.
`--show` (optional, off by default) opens a window: valid-depth pixels for
`cloud`, removed-plane (red) vs kept (green) points for `ransac`, object
contours for `cluster`, the green marker for `weigh`. Without it everything is
headless.

Saved files live next to this script: last_cloud.npy, last_points_frame.txt,
objects.npy, last_uv.npy, last_K.npy, last_M.npy, last_kept.npy.
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
HEAD_TOPIC = "/robot_head_motion_data"
TARGET_FRAME = "base_link"
HEAD_PITCH = 10.0         # degrees, positive = look down
HEAD_YAW = 0.0

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD_FILE = os.path.join(HERE, "last_cloud.npy")
FRAME_FILE = os.path.join(HERE, "last_points_frame.txt")
OBJECTS_FILE = os.path.join(HERE, "objects.npy")
DET_FILE = os.path.join(HERE, "detections.json")
WEIGH_FILE = os.path.join(HERE, "weigh.json")
UV_FILE = os.path.join(HERE, "last_uv.npy")
K_FILE = os.path.join(HERE, "last_K.npy")
M_FILE = os.path.join(HERE, "last_M.npy")
KEPT_FILE = os.path.join(HERE, "last_kept.npy")
WRIST_DEBUG_FILE = os.path.join(HERE, "wrist_last.png")   # debug: what the wrist cam saw

DEPTH_STRIDE = 4          # sample every Nth pixel
DEPTH_MIN = 0.15          # ignore closer than this (m)
DEPTH_MAX = 3.0           # ignore farther than this (m) -> drops no-return pixels
RANSAC_ITERS = 500
RANSAC_DIST = 0.008       # inlier distance to plane (m)
PLANES = 2                # remove this many dominant planes (floor + table)
CLUSTER_TOL = 0.02        # cluster distance (m)
MIN_CLUSTER = 30          # min points for a cluster
GREEN_LO = (62, 180, 110)  # HSV lower bound for the bright green weigh marker
GREEN_HI = (78, 255, 255)
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


def decode_color(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _show_window(name, img, live):
    if os.environ.get("PERCEPTION_HEADLESS"):
        return                  # non-interactive runs: don't open/block on a window
    cv2.imshow(name, img)
    if live:
        cv2.waitKey(1)          # non-blocking, for the live loop
    else:
        log("  press any key in the window ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def destroy_windows():
    cv2.destroyAllWindows()


def project(pts_cam, K):
    """Camera-frame points -> pixel (u, v) and a validity mask."""
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    z = pts_cam[:, 2]
    ok = z > 1e-6
    uv = np.full((len(pts_cam), 2), -1.0)
    uv[ok, 0] = fx * pts_cam[ok, 0] / z[ok] + cx
    uv[ok, 1] = fy * pts_cam[ok, 1] / z[ok] + cy
    return uv, ok


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


# ───────────────────── wrist-camera parcel locate ─────────────────────
WRIST_TOPICS = {
    "right": ("/cam_r/color/image_raw/compressed",
              "/cam_r/depth/image_rect_raw/compressedDepth",
              "/cam_r/color/camera_info"),
    "left": ("/cam_l/color/image_raw/compressed",
             "/cam_l/depth/image_rect_raw/compressedDepth",
             "/cam_l/color/camera_info"),
}


WRIST_CLAW_DIST = 0.06    # anything closer than this to the wrist cam is the CLAW, not a parcel


def wrist_locate(args, cam="right", near_min=WRIST_CLAW_DIST, near_max=0.60,
                 near_band=0.0, cluster_tol=0.02, min_pts=15, max_extent=0.16,
                 hint=None, timeout=None):
    """Look through a WRIST camera and return the (x, y, z_top) of the parcel below the
    hand, in TARGET_FRAME (base_link) — or None if nothing usable is seen.

    WHOLE image (no border cropping), and two cuts:
      1) DISTANCE cut for the claw: drop everything nearer than `near_min` (the claw is
         the closest thing to the camera);
      2) NEAREST-SURFACE cut: keep only points within `near_band` of the nearest remaining
         surface — that isolates the parcel's TOP FACE (clustering alone cannot separate
         the parcel from the scale/table it is resting on, they touch);
    then cluster what's left and pick the parcel = the cluster nearest `hint` (the xy where
    it was dropped), or the largest cluster when no hint is given.
    """
    ct, dt, it = WRIST_TOPICS[cam]
    t = timeout if timeout is not None else args.timeout
    info = rospy.wait_for_message(it, CameraInfo, timeout=t)
    depth = rospy.wait_for_message(dt, CompressedImage, timeout=t)
    img = decode_depth(depth)
    if img is None:
        log("  wrist_locate[%s]: could not decode %s" % (cam, dt))
        return None
    try:
        color = decode_color(rospy.wait_for_message(ct, CompressedImage, timeout=t))
    except Exception:
        color = None

    def show(note, marks=(), chosen=None):
        """ALWAYS show the window (blocking) — even when nothing was found — so the
        detection can be inspected. `marks` = [(mu_arr, mv_arr, colour), ...]."""
        try:
            vis = color.copy() if color is not None else np.zeros((240, 320, 3), np.uint8)
            for mu_, mv_, col_ in marks:
                for u_, v_ in zip(mu_, mv_):
                    cv2.circle(vis, (int(u_), int(v_)), 2, col_, -1)
            if chosen is not None:
                cv2.drawMarker(vis, (int(np.median(chosen[0])), int(np.median(chosen[1]))),
                               (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(vis, note, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imwrite(WRIST_DEBUG_FILE, vis)
            log("  wrist_locate[%s]: %s — press any key in the window ..." % (cam, note))
            _show_window("wrist parcel [%s]" % cam, vis, getattr(args, "live", False))
        except Exception as e:
            log("  wrist_locate[%s]: window failed (%s)" % (cam, e))

    d = to_metres(img)
    K = list(info.K)
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    h, w = d.shape
    step = max(1, args.stride)
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    z = d[np.ix_(vs, us)]
    uu, vv = np.meshgrid(us, vs)
    mask = np.isfinite(z) & (z >= near_min) & (z <= near_max)
    if mask.sum() < min_pts:
        log("  wrist_locate[%s]: only %d px in %.2f-%.2f m (claw too close? lower near_min)"
            % (cam, int(mask.sum()), near_min, near_max))
        show("only %d px in %.2f-%.2f m" % (int(mask.sum()), near_min, near_max))
        return None
    zz = z[mask]
    mu = uu[mask]
    mv = vv[mask]
    if near_band and near_band > 0:
        d_near = float(np.percentile(zz, 5))          # nearest non-claw surface = parcel top
        keep = zz <= d_near + near_band
        log("  wrist_locate[%s]: nearest %.3f m, keeping %.3f-%.3f m (%d of %d px)"
            % (cam, d_near, d_near, d_near + near_band, int(keep.sum()), len(zz)))
        if keep.sum() >= min_pts:
            zz, mu, mv = zz[keep], mu[keep], mv[keep]
    pts_cam = np.stack([(mu - cx) * zz / fx, (mv - cy) * zz / fy, zz], axis=1)
    frame = depth.header.frame_id or info.header.frame_id
    tf = Tf()
    rospy.sleep(0.3)
    M = tf.lookup(TARGET_FRAME, frame) if frame else None
    if M is None:
        log("  wrist_locate[%s]: no TF %s <- %s" % (cam, TARGET_FRAME, frame))
        show("no TF %s <- %s" % (TARGET_FRAME, frame))
        return None
    pts = transform_points(M, pts_cam)

    # cluster everything that isn't the claw, then pick the parcel cluster.
    # A PARCEL is small: reject anything whose bounding box is bigger than `max_extent`
    # (that is the table / the scale). Then take the survivor nearest `hint`.
    labels = _cluster(pts, cluster_tol)
    cands = []
    rej = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) < min_pts:
            continue
        P_ = pts[idx]
        ext = float(np.linalg.norm(P_.max(axis=0) - P_.min(axis=0)))
        cx_ = float(np.median(P_[:, 0]))
        cy_ = float(np.median(P_[:, 1]))
        zt_ = float(np.percentile(P_[:, 2], 90))
        if max_extent and ext > max_extent:
            log("  wrist_locate[%s] reject n=%-5d ext=%.3f base(%.3f, %.3f)  (too big: table/scale)"
                % (cam, len(idx), ext, cx_, cy_))
            rej.append((idx, ext, cx_, cy_))
            continue
        cands.append((idx, cx_, cy_, zt_, ext))
    # the debug view: rejected clusters orange, candidates green, chosen gets a cross
    marks = [(mu[c[0]], mv[c[0]], (0, 165, 255)) for c in rej]
    marks += [(mu[c[0]], mv[c[0]], (0, 255, 0)) for c in cands]
    if not cands:
        log("  wrist_locate[%s]: %d px but no parcel-sized cluster >= %d pts"
            % (cam, int(mask.sum()), min_pts))
        show("no parcel-sized cluster (rej=%d)" % len(rej))
        return None
    if hint is not None:
        cands.sort(key=lambda c: float(np.hypot(c[1] - hint[0], c[2] - hint[1])))
    else:
        cands.sort(key=lambda c: c[4])                  # smallest bounding box first
    idx, x, y, ztop, ext = cands[0]
    for c in cands[:4]:
        d_hint = (float(np.hypot(c[1] - hint[0], c[2] - hint[1])) if hint is not None else -1.0)
        log("  wrist_locate[%s] cand: n=%-5d ext=%.3f base(%.3f, %.3f) top z=%.3f  hint d=%.3f"
            % (cam, len(c[0]), c[4], c[1], c[2], c[3], d_hint))
    log("  wrist_locate[%s]: picked n=%d ext=%.3f -> base (%.3f, %.3f)  top z=%.3f  (claw cut %.2f)"
        % (cam, len(idx), ext, x, y, ztop, near_min))
    show("n=%d base(%.3f,%.3f) z=%.3f  green=cands orange=rejected"
         % (len(idx), x, y, ztop),
         marks=marks, chosen=(mu[idx], mv[idx]))
    return x, y, ztop


def wrist_distance(args, cam="left", near_min=0.02, near_max=0.60, timeout=None):
    """How CLOSE is the hand to whatever is in front of its wrist camera (the parcel)?

    Returns (nearest_depth_m, n_valid_px) — the 5th-percentile depth of the valid middle
    pixels, i.e. the nearest surface. Use it to tell whether the claw has actually reached
    the parcel. (None, 0) when nothing valid is seen.
    """
    ct, dt, it = WRIST_TOPICS[cam]
    t = timeout if timeout is not None else args.timeout
    info = rospy.wait_for_message(it, CameraInfo, timeout=t)
    depth = rospy.wait_for_message(dt, CompressedImage, timeout=t)
    img = decode_depth(depth)
    if img is None:
        return None, 0
    d = to_metres(img)
    h, w = d.shape
    step = max(1, args.stride)
    us = np.arange(int(w * 0.2), int(w * 0.8), step)
    vs = np.arange(int(h * 0.2), int(h * 0.8), step)
    z = d[np.ix_(vs, us)]
    mask = np.isfinite(z) & (z >= near_min) & (z <= near_max)
    n = int(mask.sum())
    if n < 5:
        return None, n
    return float(np.percentile(z[mask], 5)), n


def set_head(yaw_deg, pitch_deg, settle=0.8):
    """Tilt the head. joint_data = [yaw, pitch] in degrees (pitch>0 looks down)."""
    from kuavo_msgs.msg import robotHeadMotionData
    pub = rospy.Publisher(HEAD_TOPIC, robotHeadMotionData, queue_size=10)
    t0 = rospy.Time.now()
    while pub.get_num_connections() == 0 and (rospy.Time.now() - t0).to_sec() < 5.0:
        rospy.sleep(0.1)
    msg = robotHeadMotionData()
    msg.joint_data = [float(yaw_deg), float(pitch_deg)]
    for _ in range(5):
        pub.publish(msg)
        rospy.sleep(0.1)
    rospy.sleep(settle)
    log("head -> yaw=%.1f pitch=%.1f deg" % (yaw_deg, pitch_deg))


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
    mask = np.isfinite(z) & (z >= args.min_dist) & (z <= args.max_dist)
    zz = z[mask]
    pts = np.stack([(uu[mask] - cx) * zz / fx,
                    (vv[mask] - cy) * zz / fy,
                    zz], axis=1)
    log("points      : %d valid (depth %.2f-%.2f m, camera optical frame)"
        % (len(pts), args.min_dist, args.max_dist))
    if len(pts) == 0:
        log("no valid depth pixels — is the scene running?")
        return
    log("  camera-frame bounds  x[%.2f %.2f] y[%.2f %.2f] z[%.2f %.2f]"
        % (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(),
           pts[:, 2].min(), pts[:, 2].max()))

    uv, _ = project(pts, np.array(K))     # pixel coords, before any transform

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
        q = np.percentile(pts[:, 2], [5, 25, 50, 75, 95])
        log("  base z percentiles   5%%=%.2f 25%%=%.2f 50%%=%.2f 75%%=%.2f 95%%=%.2f"
            % tuple(q))
        frame = TARGET_FRAME
    else:
        log("  !! no TF to %s — keeping the camera frame" % TARGET_FRAME)

    np.save(CLOUD_FILE, pts)
    np.save(UV_FILE, uv)
    np.save(K_FILE, np.array(K, dtype=float))
    np.save(M_FILE, M if M is not None else np.eye(4))
    with open(FRAME_FILE, "w") as f:
        f.write(frame)
    log("saved %s (%d points, frame=%s)" % (CLOUD_FILE, len(pts), frame))

    if args.show:
        color = decode_color(rospy.wait_for_message(COLOR_TOPIC, CompressedImage,
                                                    timeout=args.timeout))
        for u, v in uv[::3]:
            if u >= 0:
                cv2.circle(color, (int(u), int(v)), 1, (0, 255, 0), -1)
        cv2.putText(color, "valid depth: %d pts" % len(pts), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        _show_window("perception: cloud", color, getattr(args, "live", False))


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

    remaining = pts
    kept = np.arange(len(pts))
    n_removed = 0
    heights = []
    for i in range(args.planes):
        (nrm, d), inl = ransac_plane(remaining, args.iters, args.dist)
        if nrm is None:
            log("plane #%d: RANSAC failed (too few points)" % i)
            break
        if nrm[2] < 0:                       # orient upward for readability
            nrm, d = -nrm, -d
        inl_n = int(inl.sum())
        tilt = np.degrees(np.arccos(min(1.0, abs(nrm[2]))))
        log("plane #%d    : normal=(%.3f, %.3f, %.3f)  height z=%.3f  tilt=%.1f deg"
            % (i, nrm[0], nrm[1], nrm[2], -d, tilt))
        log("  inliers    : %d / %d (%.1f%%)"
            % (inl_n, len(remaining), 100.0 * inl_n / len(remaining)))
        if abs(nrm[2]) < 0.9:
            log("  not horizontal — stopping (won't remove it)")
            break
        heights.append(-d)
        keep = ~inl
        kept = kept[keep]
        remaining = remaining[keep]
        n_removed += inl_n
        if len(remaining) < 3:
            break

    if len(heights) >= 2:
        log("plane gap   : %.3f m" % abs(heights[0] - heights[1]))

    objs = remaining
    np.save(KEPT_FILE, kept)
    log("removed %d plane point(s); %d object points remain" % (n_removed, len(objs)))
    if len(objs):
        log("  obj bounds  x[%.2f %.2f] y[%.2f %.2f] z[%.2f %.2f]"
            % (objs[:, 0].min(), objs[:, 0].max(), objs[:, 1].min(), objs[:, 1].max(),
               objs[:, 2].min(), objs[:, 2].max()))
    np.save(OBJECTS_FILE, objs)
    log("saved %s" % OBJECTS_FILE)

    if args.show and os.path.exists(UV_FILE):
        uv_all = np.load(UV_FILE)
        color = decode_color(rospy.wait_for_message(COLOR_TOPIC, CompressedImage,
                                                    timeout=args.timeout))
        kept_mask = np.zeros(len(uv_all), dtype=bool)
        kept_mask[kept] = True
        for u, v in uv_all[~kept_mask][::4]:
            if u >= 0:
                cv2.circle(color, (int(u), int(v)), 1, (0, 0, 255), -1)   # removed (plane)
        for u, v in uv_all[kept_mask][::4]:
            if u >= 0:
                cv2.circle(color, (int(u), int(v)), 1, (0, 255, 0), -1)   # kept (objects)
        cv2.putText(color, "RANSAC: red=removed plane  green=objects", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        _show_window("perception: ransac", color, getattr(args, "live", False))


def step_cluster(args):
    if not os.path.exists(OBJECTS_FILE):
        log("no %s — run --step ransac first" % OBJECTS_FILE)
        return
    pts = np.load(OBJECTS_FILE)
    log("loaded %d object points" % len(pts))

    labels = _cluster(pts, args.tol)
    n_clusters = labels.max() + 1 if len(labels) else 0
    log("clusters    : %d" % n_clusters)

    detections = []
    for k in range(n_clusters):
        c = pts[labels == k]
        if len(c) < args.min_cluster:
            continue
        center_xy, yaw, ext = _pca_pose(c)
        top_z = float(c[:, 2].max())
        bot_z = float(c[:, 2].min())
        det = {
            "label": int(k),
            "n": int(len(c)),
            "x": float(center_xy[0]), "y": float(center_xy[1]),
            "z_top": top_z, "z_bot": bot_z, "z": float(c[:, 2].mean()),
            "yaw_deg": float(np.degrees(yaw)),
            "length": float(ext[0]), "width": float(ext[1]),
            "height": float(top_z - bot_z),
        }
        detections.append((det, c))
        log("  #%d  n=%4d  center=(%.3f, %.3f, %.3f)  yaw=%.1f deg  "
            "size=(%.3f x %.3f x %.3f)  top_z=%.3f"
            % (k, det["n"], det["x"], det["y"], det["z"], det["yaw_deg"],
               det["length"], det["width"], det["height"], det["z_top"]))

    if detections:
        import json
        with open(DET_FILE, "w") as f:
            json.dump([d for d, _ in detections], f, indent=2)
        log("saved %s (%d detection(s))" % (DET_FILE, len(detections)))

    if args.show:
        _show_contours(args, pts, labels, detections)


def step_weigh(args):
    """Find the green weighing marker by colour + depth, return its base-frame pose.

    Several surfaces are green (floor, table top, sorting box inner), so we
    deproject EVERY green blob and keep the ones landing on the table
    (base_link y < 0), then take the largest of those.  All candidates are logged."""
    info = rospy.wait_for_message(INFO_TOPIC, CameraInfo, timeout=args.timeout)
    color = rospy.wait_for_message(COLOR_TOPIC, CompressedImage, timeout=args.timeout)
    depth = rospy.wait_for_message(DEPTH_TOPIC, CompressedImage, timeout=args.timeout)
    img = decode_color(color)
    if img is None:
        log("FAILED to decode color image")
        return
    K = list(info.K)
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(args.green_lo, np.uint8),
                       np.array(args.green_hi, np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        log("no green region found — adjust --green-lo/--green-hi")
        return
    inside = hsv[mask > 0]
    show = getattr(args, "show", False)
    if show and inside.size:
        log("mask HSV   : H=%.0f S=%.0f V=%.0f (median)"
            % tuple(np.median(inside, axis=0)))

    d = to_metres(decode_depth(depth))
    frame = color.header.frame_id or info.header.frame_id
    tf = Tf()
    rospy.sleep(0.3)
    M = tf.lookup(TARGET_FRAME, frame) if frame else None
    if M is None:
        log("  !! no TF to %s (keeping camera frame)" % TARGET_FRAME)

    def deproject(cnt):
        area = cv2.contourArea(cnt)
        x0, y0, w0, h0 = cv2.boundingRect(cnt)
        mm = cv2.moments(cnt)
        if mm["m00"] <= 0:
            return None
        cu = mm["m10"] / mm["m00"]
        cv_ = mm["m01"] / mm["m00"]
        y0i, y1i = max(0, y0), min(d.shape[0], y0 + h0)
        x0i, x1i = max(0, x0), min(d.shape[1], x0 + w0)
        region = d[y0i:y1i, x0i:x1i]
        valid = region[np.isfinite(region) & (region > 0)]
        if valid.size == 0:
            return None
        Z = float(np.median(valid))
        p_cam = np.array([(cu - cx) * Z / fx, (cv_ - cy) * Z / fy, Z])
        p_base = transform_points(M, p_cam.reshape(1, 3))[0] if M is not None else p_cam
        return {"cnt": cnt, "area": area, "bbox": (x0, y0, w0, h0),
                "centroid": (cu, cv_), "Z": Z, "pos": p_base,
                "width": (x1i - x0i) * Z / fx, "height": (y1i - y0i) * Z / fy}

    cands = [c for c in (deproject(c) for c in cnts) if c is not None]
    if not cands:
        log("no green candidate with valid depth")
        return
    if show:
        for c in cands:
            p = c["pos"]
            log("  candidate area=%5d px  Z=%.3f  base=(%.3f, %.3f, %.3f)  size≈%.3f x %.3f"
                % (c["area"], c["Z"], p[0], p[1], p[2], c["width"], c["height"]))

    on_table = [c for c in cands if c["pos"][1] < 0.0]   # marker is on the table; box is not
    if not on_table:
        log("  (no green blob on the table side — using the largest overall)")
    best = max(on_table or cands, key=lambda c: c["area"])

    x0, y0, w0, h0 = best["bbox"]
    cu, cv_ = best["centroid"]
    p_base = best["pos"]
    if show:
        log("green marker: area=%d px  bbox=(%d,%d %dx%d)  centroid=(%.0f, %.0f)"
            % (best["area"], x0, y0, w0, h0, cu, cv_))
        log("  depth Z=%.3f m  size≈%.3f x %.3f m"
            % (best["Z"], best["width"], best["height"]))
    log("weigh: %s = (%.3f, %.3f, %.3f)"
        % (TARGET_FRAME if M is not None else frame, p_base[0], p_base[1], p_base[2]))

    import json
    with open(WEIGH_FILE, "w") as f:
        json.dump({"x": float(p_base[0]), "y": float(p_base[1]),
                   "z": float(p_base[2]), "width": float(best["width"]),
                   "height": float(best["height"]), "area": int(best["area"]),
                   "frame": frame}, f, indent=2)
    if show:
        log("saved %s" % WEIGH_FILE)

    if args.show:
        vis = img.copy()
        cv2.drawContours(vis, [best["cnt"]], -1, (0, 255, 0), 2)
        cv2.circle(vis, (int(cu), int(cv_)), 4, (0, 0, 255), -1)
        cv2.putText(vis, "weigh (%.2f, %.2f, %.2f)" % tuple(p_base), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        _show_window("perception: weigh", vis, getattr(args, "live", False))


def _show_contours(args, pts, labels, detections):
    if not (os.path.exists(UV_FILE) and os.path.exists(K_FILE) and os.path.exists(M_FILE)):
        log("  (cannot overlay: cached uv/K/M missing — re-run --step cloud)")
        return
    uv_all = np.load(UV_FILE)
    K = np.load(K_FILE)
    M = np.load(M_FILE)
    if os.path.exists(KEPT_FILE):
        kept = np.load(KEPT_FILE)
        uv_obj = uv_all[kept]
    elif len(uv_all) == len(pts):
        uv_obj = uv_all
    else:
        log("  (cannot overlay: uv count mismatch — re-run --step cloud)")
        return

    color = decode_color(rospy.wait_for_message(COLOR_TOPIC, CompressedImage,
                                                timeout=args.timeout))
    Minv = np.linalg.inv(M)
    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 0, 255),
              (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 255, 255)]
    for i, (det, c) in enumerate(detections):
        sel = labels == det["label"]
        p_cam = (Minv[:3, :3] @ pts[sel].T).T + Minv[:3, 3]
        uv, ok = project(p_cam, K)
        uv = uv[ok]
        if len(uv) < 3:
            continue
        uv_i = uv.astype(np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(uv_i)
        col = colors[i % len(colors)]
        cv2.drawContours(color, [hull], -1, col, 2)
        cu, cv_ = hull.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(color, "#%d %.2f,%.2f" % (det["label"], det["x"], det["y"]),
                    (cu, max(20, cv_)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    if not detections:
        cv2.putText(color, "no clusters", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    _show_window("perception: objects", color, getattr(args, "live", False))


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


def _pca_pose(pts):
    """PCA of a cluster's XY footprint -> (center_xy, yaw_rad, extents).

    extents = (along-major-axis, along-minor-axis)."""
    xy = pts[:, :2]
    center = xy.mean(axis=0)
    d = xy - center
    if len(xy) < 3:
        return center, 0.0, np.zeros(2)
    cov = np.cov(d.T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    yaw = float(np.arctan2(axis[1], axis[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    p = d @ np.array([[c, s], [-s, c]]).T
    ext = p.max(axis=0) - p.min(axis=0)
    return center, yaw, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["cloud", "ransac", "cluster", "weigh"],
                    required=True)
    ap.add_argument("--stride", type=int, default=DEPTH_STRIDE)
    ap.add_argument("--min-dist", type=float, default=DEPTH_MIN)
    ap.add_argument("--max-dist", type=float, default=DEPTH_MAX)
    ap.add_argument("--iters", type=int, default=RANSAC_ITERS)
    ap.add_argument("--dist", type=float, default=RANSAC_DIST)
    ap.add_argument("--planes", type=int, default=PLANES)
    ap.add_argument("--tol", type=float, default=CLUSTER_TOL)
    ap.add_argument("--min-cluster", type=int, default=MIN_CLUSTER)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--green-lo", type=int, nargs=3, default=list(GREEN_LO),
                    metavar=("H", "S", "V"))
    ap.add_argument("--green-hi", type=int, nargs=3, default=list(GREEN_HI),
                    metavar=("H", "S", "V"))
    ap.add_argument("--show", action="store_true",
                    help="open a window with an overlay (contours on the camera)")
    args = ap.parse_args()

    rospy.init_node("perception_" + args.step, anonymous=True)
    if args.step == "cloud":
        step_cloud(args)
    elif args.step == "ransac":
        step_ransac(args)
    elif args.step == "weigh":
        step_weigh(args)
    else:
        step_cluster(args)


if __name__ == "__main__":
    main()
