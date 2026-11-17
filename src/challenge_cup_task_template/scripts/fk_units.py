#!/usr/bin/env python3
"""Work out what unit the FK service wants for the arm joints (degrees or rad).

The service is meant to take radians (arm_test_control passes raw joint_q), but a
wrong convention silently shrinks every computed error by ~57x, which looks like
"the arm is accurate to 1 mm". So we probe it once at startup:

    FK(q)                    -> pose0
    FK(q with joint 13 +10)  -> pose_deg     (10 degrees, if that is the unit)
    FK(q with joint 13 +0.1745) -> pose_rad  (10 degrees expressed in radians)

Whichever probe moves the hand by centimetres tells us the unit. Returns the
factor to multiply our internal radians by before calling the service.
"""

import numpy as np

DEG_PER_RAD = 180.0 / np.pi


def _pos(fk, q):
    try:
        r = fk([float(v) for v in q])
    except Exception:
        return None
    if not r.success:
        return None
    return np.array(r.hand_poses.right_pose.pos_xyz, dtype=float)


def _ready_pose_rad():
    """The ready pose (a non-singular configuration) in our internal unit (rad)."""
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ready.json")
    try:
        with open(path) as f:
            return [float(np.radians(v)) for v in json.load(f)["joints_deg"]]
    except Exception:
        return [0.0] * 14


def calibrate(fk, log=print, joint=12, base_rad=None):
    """-> scale factor (57.2958 for degrees, 1.0 for radians)."""
    q0 = list(base_rad) if base_rad is not None else _ready_pose_rad()
    p0 = _pos(fk, q0)
    if p0 is None:
        log("  FK units: probe failed (service error) — assuming radians")
        return 1.0

    def probe(delta):
        q = list(q0)
        q[joint] += delta
        p = _pos(fk, q)
        if p is None:
            return 0.0
        v = float(np.linalg.norm(p - p0)) * 1000.0
        return v if np.isfinite(v) else 0.0

    log("  FK probe: base (deg) %s  -> pose (%.3f, %.3f, %.3f) m"
        % (" ".join("%.1f" % np.degrees(v) for v in q0), p0[0], p0[1], p0[2]))

    d_deg = probe(10.0)              # 10 assumed degrees
    d_rad = probe(10.0 / DEG_PER_RAD)  # 10 degrees expressed in radians

    # The probe moves a WRIST joint by 10 degrees, which shifts the hand by roughly
    # 15-40 mm. Reading it as 10 radians (573 deg) produces an implausible 100+ mm,
    # and reading 0.1745 as degrees gives a fraction of a millimetre. So a tight
    # band around the wrist lever decides it unambiguously.
    def sane(v):
        return 8.0 < v < 60.0

    if sane(d_rad) and not sane(d_deg):
        log("  FK units: RADIANS (probe +0.1745 rad -> %.1f mm, +10 probe %.1f mm)"
            % (d_rad, d_deg))
        return 1.0
    if sane(d_deg) and not sane(d_rad):
        log("  FK units: DEGREES (probe +10 -> %.1f mm, radians probe %.1f mm)"
            % (d_deg, d_rad))
        return DEG_PER_RAD
    if d_deg < 0.5 and d_rad < 0.5:
        log("  FK units: the service did NOT move for either probe (%.2f / %.2f mm)"
            " -> it is probably ignoring req.q; mm numbers are meaningless"
            % (d_deg, d_rad))
        return 1.0
    log("  FK units: inconclusive (%.1f mm vs %.1f mm) — assuming radians; "
        "check /ik/fk_srv if the numbers look too small" % (d_deg, d_rad))
    return 1.0
