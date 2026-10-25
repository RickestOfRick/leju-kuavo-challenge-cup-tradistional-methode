#!/usr/bin/env python3
"""Axis-aligned collision model for scene 1 (table / sorting box / scale / parcels).

No ROS, no numpy, no simulator: plain AABBs in the SAME frame as the arm FK output
(the frame `move_linear()` / `current_pose()` use), so a planner built on top of this
can hand its waypoints straight to the existing motion primitives.

WHERE THE NUMBERS COME FROM
---------------------------
Dimensions  -> challenge_cup_simulator/config/scenes/scene1.yaml (MuJoCo "size" is
               HALF-extents there, so they are doubled below).
Positions   -> the detection the task already produces:
                 parcels : perception -> self.parcels[i]  (x, y, z_top, length, width, height)
                 box     : perception -> self.box          (x, y, z_top, length, width, height)
                 scale   : perception -> do_scale()'s (x, y)
               Fallbacks below are the values seen in the task logs, used only when
               nothing was detected.

VERIFY THESE (marked [?]) — they are my assumptions, not measured:
  [?] box wall thickness / floor: from scene1.yaml (0.024 / 0.016) -> believed correct
  [?] table slab extents:         derived from top_size 0.40 x 0.70 (half) -> 0.80 x 1.40
  [?] table top height:           taken as the LOWEST parcel bottom (self-consistent,
                                  no absolute frame assumption)
  [?] scale footprint:            the 0.2 x 0.2 m weighing marker in scene1.yaml
  [?] parcel orientation:         scene1.yaml yaws the parcels 90 deg, so their real
                                  footprint is 0.12 along x and 0.045 along y; this model
                                  uses (length=0.045, width=0.12) as-is -> swap if wrong
"""

import math

# --------------------------------------------------------------------------- scene1.yaml
# parc
PARCEL_FULL = (0.045, 0.120, 0.100)      # size "0.0225 0.060 0.050" (half) -> full
# sorting_box_0p4_0p3_0p3
BOX_INNER_FULL = (0.400, 0.300, 0.300)   # inner_size "0.40 0.30 0.30" (already full)
BOX_WALL = 0.024                         # wall_thickness
BOX_FLOOR = 0.016                        # floor_thickness
BOX_OUTER_FULL = (BOX_INNER_FULL[0] + 2.0 * BOX_WALL,
                  BOX_INNER_FULL[1] + 2.0 * BOX_WALL,
                  BOX_FLOOR + BOX_INNER_FULL[2])
# challenge_table
TABLE_FULL = (0.800, 1.400, 0.030)       # top_size "0.40 0.70 0.015" (half) -> full
# weighing_area_0p2m_square
SCALE_FULL = (0.200, 0.200, 0.010)       # size "0.10 0.10 0.002" (half) -> full

# --------------------------------------------------------------------------- fallbacks
# used only when detection is empty (values from the task logs)
# BOX_DROP (0.606, 0.226) is the DROP POINT, not the box centre: task_sequence documents
# the box NEAR EDGE at x ~= 0.47 ("pull x toward the robot ... box near edge ~0.47").
# box centre = near edge + half outer width = 0.470 + 0.224 = 0.694
FB_BOX_XY = (0.694, 0.226)               # box CENTRE (drop point is offset toward the robot)
FB_SCALE_XY = (0.386, -0.533)            # "scale (detected): x=0.386 y=-0.533"
FB_PARCELS_XY = [(0.293, -0.101), (0.293, -0.326), (0.435, -0.098), (0.435, -0.326)]


class AABB(object):
    """Axis-aligned box: centre + full size, all in metres, arm base frame."""

    __slots__ = ("name", "cx", "cy", "cz", "sx", "sy", "sz")

    def __init__(self, name, center, size):
        self.name = name
        self.cx, self.cy, self.cz = [float(v) for v in center]
        self.sx, self.sy, self.sz = [float(v) for v in size]

    # -- bounds ------------------------------------------------------------
    @property
    def x0(self):
        return self.cx - self.sx * 0.5

    @property
    def x1(self):
        return self.cx + self.sx * 0.5

    @property
    def y0(self):
        return self.cy - self.sy * 0.5

    @property
    def y1(self):
        return self.cy + self.sy * 0.5

    @property
    def z0(self):
        return self.cz - self.sz * 0.5

    @property
    def z1(self):
        return self.cz + self.sz * 0.5

    def inflated(self, r, name=None):
        """Grow the box by r on every side (Minkowski sum with a sphere of radius r)."""
        return AABB(name or self.name, (self.cx, self.cy, self.cz),
                    (self.sx + 2.0 * r, self.sy + 2.0 * r, self.sz + 2.0 * r))

    # -- queries -----------------------------------------------------------
    def contains(self, p, slop=0.0):
        return (self.x0 - slop <= p[0] <= self.x1 + slop and
                self.y0 - slop <= p[1] <= self.y1 + slop and
                self.z0 - slop <= p[2] <= self.z1 + slop)

    def distance_to(self, p):
        """Euclidean distance from point p to the box surface (0 if inside)."""
        dx = max(self.x0 - p[0], 0.0, p[0] - self.x1)
        dy = max(self.y0 - p[1], 0.0, p[1] - self.y1)
        dz = max(self.z0 - p[2], 0.0, p[2] - self.z1)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def overlaps(self, other):
        return (self.x0 < other.x1 and other.x0 < self.x1 and
                self.y0 < other.y1 and other.y0 < self.y1 and
                self.z0 < other.z1 and other.z0 < self.z1)

    def segment_hits(self, a, b, step=0.01):
        """True if the straight segment a->b passes through this box.

        Sampled, not exact — 1 cm steps are far below the size of anything here.
        """
        n = max(1, int(math.ceil(dist(a, b) / max(step, 1e-3))))
        for i in range(n + 1):
            t = float(i) / float(n)
            if self.contains((a[0] + (b[0] - a[0]) * t,
                              a[1] + (b[1] - a[1]) * t,
                              a[2] + (b[2] - a[2]) * t)):
                return True
        return False

    def __repr__(self):
        return ("%-16s c=(%6.3f,%6.3f,%6.3f) size=(%.3f,%.3f,%.3f)"
                % (self.name, self.cx, self.cy, self.cz, self.sx, self.sy, self.sz))


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# --------------------------------------------------------------------------- model
class SceneModel(object):
    """Static scene: table + hollow sorting box + scale. Parcels are added separately."""

    def __init__(self, box_xy, box_top_z, scale_xy, table_top_z=None):
        # The box sits ON the table, so the table height follows from the rim we detect:
        #   table_top = rim_z - box_total_height      (no absolute frame assumption)
        if table_top_z is None:
            table_top_z = float(box_top_z) - BOX_OUTER_FULL[2]
        self.table_top_z = float(table_top_z)
        self.box_xy = tuple(box_xy)
        self.box_top_z = float(box_top_z)
        self.scale_xy = tuple(scale_xy)
        self.parcels = []                 # list of AABB, filled by set_parcels()
        self.box_parts = []               # 5 AABBs: floor + 4 walls (hollow!)
        self.scale_parts = []
        self.table = None
        self._build()

    # -- construction ------------------------------------------------------
    def _build(self):
        tz = self.table_top_z
        # table: a slab whose TOP is at tz (the hand must never go below it)
        self.table = AABB("table", (0.50, -0.06, tz - TABLE_FULL[2] * 0.5), TABLE_FULL)

        # sorting box: floor + 4 thin walls, so the inside stays free for the drop.
        # bottom sits on the table, rim at box_top_z
        bx, by = self.box_xy
        ox, oy, oz = BOX_OUTER_FULL
        inside_height = oz - BOX_FLOOR
        floor_c = (bx, by, tz + BOX_FLOOR * 0.5)
        wall_cz = tz + BOX_FLOOR + inside_height * 0.5
        wall_hz = inside_height
        hx = ox * 0.5
        hy = oy * 0.5
        self.box_parts = [
            AABB("box_floor", floor_c, (ox, oy, BOX_FLOOR)),
            AABB("box_wall_x0", (bx - hx + BOX_WALL * 0.5, by, wall_cz),
                 (BOX_WALL, oy, wall_hz)),
            AABB("box_wall_x1", (bx + hx - BOX_WALL * 0.5, by, wall_cz),
                 (BOX_WALL, oy, wall_hz)),
            AABB("box_wall_y0", (bx, by - hy + BOX_WALL * 0.5, wall_cz),
                 (ox, BOX_WALL, wall_hz)),
            AABB("box_wall_y1", (bx, by + hy - BOX_WALL * 0.5, wall_cz),
                 (ox, BOX_WALL, wall_hz)),
        ]

        # scale: platform slab on the table
        sx, sy = self.scale_xy
        self.scale_parts = [
            AABB("scale", (sx, sy, tz + SCALE_FULL[2] * 0.5), SCALE_FULL),
        ]

    def set_parcels(self, detections, skip_index=None):
        """detections: list of dicts with x, y, z_top, length, width, height.

        `skip_index` leaves that parcel OUT of the model — the one being handled must
        not block its own grasp pose (this is the cheap alternative to an allowed-
        collision matrix).
        """
        self.parcels = []
        for i, d in enumerate(detections):
            if skip_index is not None and i == skip_index:
                continue
            h = float(d.get("height") or PARCEL_FULL[2])
            self.parcels.append(AABB(
                "parcel_%d" % i,
                (float(d["x"]), float(d["y"]), float(d["z_top"]) - h * 0.5),
                (float(d.get("length") or PARCEL_FULL[0]),
                 float(d.get("width") or PARCEL_FULL[1]), h)))

    # -- queries -----------------------------------------------------------
    def obstacles(self, include_parcels=True):
        out = [self.table] + self.box_parts + self.scale_parts
        if include_parcels:
            out += self.parcels
        return out

    def collides(self, point, radius=0.0, include_parcels=True):
        """Is the hand (a sphere of `radius`) in collision at `point`?"""
        for o in self.obstacles(include_parcels):
            if o.inflated(radius, o.name).contains(point):
                return o.name
        return None

    def segment_collides(self, a, b, radius=0.0, include_parcels=True):
        """Which obstacle (if any) the straight hand path a->b would hit."""
        for o in self.obstacles(include_parcels):
            if o.inflated(radius, o.name).segment_hits(a, b):
                return o.name
        return None

    def clearance_height(self, margin=0.08):
        """A z the hand can travel at, above everything detected."""
        tops = [o.z1 for o in self.obstacles(True)]
        return max(tops) + margin

    # -- debug -------------------------------------------------------------
    def dump(self):
        print("SceneModel  frame = arm base   (table top z = %.3f)" % self.table_top_z)
        print("  box   xy=(%.3f, %.3f)  rim z=%.3f  outer=%.3f x %.3f x %.3f"
              % (self.box_xy[0], self.box_xy[1], self.box_top_z,
                 BOX_OUTER_FULL[0], BOX_OUTER_FULL[1], BOX_OUTER_FULL[2]))
        print("  scale xy=(%.3f, %.3f)  size=%.2f x %.2f"
              % (self.scale_xy[0], self.scale_xy[1], SCALE_FULL[0], SCALE_FULL[1]))
        for o in self.obstacles(True):
            print("   ", o)


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    # same numbers the task uses when nothing is detected
    m = SceneModel(FB_BOX_XY, box_top_z=0.219,        # rim height seen in the logs
                   scale_xy=FB_SCALE_XY)               # table top derived from it
    m.set_parcels([{"x": x, "y": y, "z_top": m.table_top_z + PARCEL_FULL[2],
                    "length": PARCEL_FULL[0], "width": PARCEL_FULL[1],
                    "height": PARCEL_FULL[2]} for x, y in FB_PARCELS_XY],
                  skip_index=0)
    m.dump()

    print("\nclearance height for travel : %.3f m" % m.clearance_height())
    drop = (0.606, 0.226, 0.300)
    print("box drop point %s -> inside box: %s" % (drop, m.collides(drop, 0.0, False)))
    print("hand at table level -> %s" % (m.collides((0.60, 0.22, -0.04), 0.0, False),))
    home = (0.30, 0.06, 0.45)
    print("home path from above box -> %s"
          % (m.segment_collides((0.606, 0.226, 0.45), home, 0.0, True),))
