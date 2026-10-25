#!/usr/bin/env python3
"""A* path planner over the scene-1 collision model.

Plans a TOP-DOWN path for the hand at a fixed travel height z, on a 2 cm grid, using
the AABBs from collision_model.py. The output is a list of Cartesian waypoints, i.e.
exactly what `move_linear()` already accepts:

    for p in plan_to((0.606, 0.226, 0.45), (0.30, 0.06)):
        ar.move_linear(p, q, n_points=4, seg_time=0.4)

2.5-D: the grid is 2-D, but occupancy is evaluated at the travel height, so an obstacle
only blocks the plane it actually reaches into (e.g. the box walls stop at the rim).

No ROS, no numpy. Run directly for a demo + SVG.
"""

import heapq
import math

from collision_model import SceneModel, FB_BOX_XY, FB_SCALE_XY, FB_PARCELS_XY, PARCEL_FULL

CELL = 0.02                 # grid resolution (m)
HAND_R = 0.045              # hand + claw clearance radius (m)  [tune]
CLEAR_SOFT = 0.10           # within this distance of an obstacle, add cost (m)
CLEAR_W = 2.5               # weight of that penalty


class Grid(object):
    def __init__(self, model, z, hand_r=HAND_R,
                 x0=0.15, x1=0.95, y0=-0.65, y1=0.55, include_parcels=True):
        self.model = model
        self.z = float(z)
        self.r = float(hand_r)
        self.x0, self.y0 = float(x0), float(y0)
        self.nx = int(math.ceil((x1 - x0) / CELL)) + 1
        self.ny = int(math.ceil((y1 - y0) / CELL)) + 1
        self.blocked = [[False] * self.ny for _ in range(self.nx)]
        self.clear = [[0.0] * self.ny for _ in range(self.nx)]
        obs = model.obstacles(include_parcels)
        for i in range(self.nx):
            x = self._cx(i)
            for j in range(self.ny):
                y = self._cy(j)
                p = (x, y, self.z)
                hit = False
                best = 1e9
                for o in obs:
                    d = o.distance_to(p) - self.r
                    if d <= 0.0:
                        hit = True
                        break
                    if d < best:
                        best = d
                self.blocked[i][j] = hit
                self.clear[i][j] = 0.0 if hit else best

    # -- coordinates -------------------------------------------------------
    def _cx(self, i):
        return self.x0 + i * CELL

    def _cy(self, j):
        return self.y0 + j * CELL

    def idx(self, p):
        """World point -> nearest cell index (clamped)."""
        i = int(round((p[0] - self.x0) / CELL))
        j = int(round((p[1] - self.y0) / CELL))
        return (min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1))

    def point(self, i, j):
        return (self._cx(i), self._cy(j), self.z)

    def nearest_free(self, cell, max_r=8):
        if not self.blocked[cell[0]][cell[1]]:
            return cell
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = cell[0] + di, cell[1] + dj
                    if 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[i][j]:
                        return (i, j)
        return None


NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1)]


def astar(grid, start, goal, max_expand=200000):
    """8-connected A*. Returns a list of cells, or None."""
    s = grid.nearest_free(grid.idx(start))
    g = grid.nearest_free(grid.idx(goal))
    if s is None or g is None:
        return None

    def h(c):
        return math.hypot(c[0] - g[0], c[1] - g[1])

    openq = [(h(s), 0.0, s)]
    came = {}
    cost = {s: 0.0}
    seen = set()
    while openq:
        _, gcost, cur = heapq.heappop(openq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if len(seen) > max_expand:
            return None
        for di, dj in NEIGHBORS:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < grid.nx and 0 <= nj < grid.ny):
                continue
            if grid.blocked[ni][nj]:
                continue
            if di and dj:                       # no corner cutting
                if grid.blocked[cur[0] + di][cur[1]] or grid.blocked[cur[0]][cur[1] + dj]:
                    continue
            step = math.sqrt(2.0) if (di and dj) else 1.0
            pen = 1.0 + CLEAR_W * max(0.0, CLEAR_SOFT - grid.clear[ni][nj]) / CLEAR_SOFT
            nc = gcost + step * CELL * pen
            if nc < cost.get((ni, nj), 1e18):
                cost[(ni, nj)] = nc
                came[(ni, nj)] = cur
                heapq.heappush(openq, (nc + h((ni, nj)) * CELL, nc, (ni, nj)))
    return None


def shortcut(grid, cells):
    """Drop waypoints while the straight segment stays collision-free."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1:
            a, b = grid.point(*cells[i]), grid.point(*cells[j])
            if grid.model.segment_collides(
                    (a[0], a[1], grid.z), (b[0], b[1], grid.z), grid.r, True) is None:
                break
            j -= 1
        out.append(cells[j])
        i = j
    return out


def plan(start, goal, model, z=None, hand_r=HAND_R, include_parcels=True):
    """Plan from start (x,y[,z]) to goal (x,y[,z]) at a constant travel height.

    `z` defaults to max(start_z, goal_z) so a lift is implicit.
    Returns a list of (x, y, z) waypoints including both ends, or None.
    """
    z = float(z if z is not None else max(start[2] if len(start) > 2 else 0.0,
                                          goal[2] if len(goal) > 2 else 0.0))
    grid = Grid(model, z, hand_r=hand_r, include_parcels=include_parcels)
    cells = astar(grid, start, goal)
    if cells is None:
        return None
    cells = shortcut(grid, cells)
    pts = [grid.point(*c) for c in cells]
    pts[0] = (start[0], start[1], z)
    pts[-1] = (goal[0], goal[1], z)
    return pts


# --------------------------------------------------------------------------- svg
def write_svg(path, plan_pts, model, z, name="plan.svg", scale=900.0):
    """Top-down picture: obstacles at the travel plane, the grid, and the path."""
    xs = [p[0] for p in plan_pts]
    ys = [p[1] for p in plan_pts]
    pad = 0.06
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    x0, x1 = min(x0, 0.20), max(x1, 0.85)
    y0, y1 = min(y0, -0.62), max(y1, 0.50)
    W, H = (x1 - x0) * scale, (y1 - y0) * scale

    def px(x):
        return (x - x0) * scale

    def py(y):
        return H - (y - y0) * scale          # y up

    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" '
           'viewBox="0 0 %.0f %.0f">' % (W, H, W, H),
           '<rect width="100%" height="100%" fill="#ffffff"/>']

    # obstacles that reach the travel plane
    for o in model.obstacles(True):
        if not (o.z0 <= z + 0.02 and o.z1 >= z - 0.02):
            continue
        col = "#c9d6e8"
        if o.name.startswith("parcel"):
            col = "#e8d6b8"
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" '
                   'stroke="#8899aa" stroke-width="1"/>'
                   % (px(o.x0), py(o.y1), o.sx * scale, o.sy * scale, col))

    # path
    d = " ".join(("%s%.1f,%.1f" % ("M" if i == 0 else "L", px(p[0]), py(p[1])))
                 for i, p in enumerate(plan_pts))
    out.append('<path d="%s" fill="none" stroke="#d33" stroke-width="3"/>' % d)
    for i, p in enumerate(plan_pts):
        r = 7 if i in (0, len(plan_pts) - 1) else 3
        out.append('<circle cx="%.1f" cy="%.1f" r="%d" fill="#d33"/>' % (px(p[0]), py(p[1]), r))
    out.append('<text x="12" y="24" font-family="sans-serif" font-size="16" fill="#333">'
               'A* path at z=%.2f m &nbsp; (%d waypoints)</text>' % (z, len(plan_pts)))
    out.append('</svg>')
    open(name, "w").write("\n".join(out))
    return name


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    m = SceneModel(FB_BOX_XY, 0.219, FB_SCALE_XY)
    parcels = [{"x": x, "y": y, "z_top": m.table_top_z + PARCEL_FULL[2],
                "length": PARCEL_FULL[0], "width": PARCEL_FULL[1],
                "height": PARCEL_FULL[2]} for x, y in FB_PARCELS_XY]
    m.set_parcels(parcels)

    start = (0.606, 0.226, 0.45)      # above the sorting box
    goal = (0.300, 0.060, 0.45)       # ready pose area
    pts = plan(start, goal, m, z=0.45)
    print("plan above box -> ready : %s" % ("FAILED" if pts is None else "%d waypoints" % len(pts)))
    if pts:
        for p in pts:
            print("   (%6.3f, %6.3f, %6.3f)" % p)

    # low path: below the rim the box walls block, so the planner must route around it
    low = plan((0.70, -0.20, 0.15), (0.70, 0.50, 0.15), m, z=0.15)
    print("\nplan around the box at z=0.18 : %s"
          % ("FAILED" if low is None else "%d waypoints" % len(low)))
    if low:
        for p in low:
            print("   (%6.3f, %6.3f, %6.3f)" % p)

    if low:
        f = write_svg(low, low, m, 0.15, name="/tmp/opencode/plan_around_box.svg")
    else:
        f = write_svg(pts or [(start[0], start[1], 0.45), (goal[0], goal[1], 0.45)],
                      pts or [(start[0], start[1], 0.45), (goal[0], goal[1], 0.45)],
                      m, 0.45, name="/tmp/opencode/plan_box_to_ready.svg")
    print("\nsvg ->", f)
