"""Conservative grid planning on the estimated occupancy map.

A robot-centre cell is traversable only if every cell within the circumscribed
footprint radius (+ margin) is FREE. UNKNOWN is never treated as free. Goals are
rejected with an explicit reason when unsupported.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..mapping.occupancy import FREE, OCCUPIED


@dataclass
class PlannerConfig:
    footprint_radius: float = 0.262
    margin: float = 0.03
    clearance_weight: float = 0.6
    start_snap: float = 0.12  # the robot may start slightly inside the clearance band
    goal_snap: float = 0.12   # on replans only: nearest certified cell to a goal that lost clearance
    avoid_weight: float = 12.0


class GoalRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Planner:
    def __init__(self, config: PlannerConfig | None = None):
        self.cfg = config or PlannerConfig()
        self._cache_rev = None
        self._clear = None
        self._trav = None

    def prepare(self, grid):
        if self._cache_rev == (id(grid), grid.revision):
            return
        free = (grid.classes() == FREE).astype(np.uint8)
        # distance (cells) from each free cell to the nearest non-free cell
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * grid.cfg.resolution
        self._clear = dist
        self._trav = dist >= self.cfg.footprint_radius + self.cfg.margin
        # distance to the nearest cell with net positive obstacle evidence (at least one
        # observation more than it has free support; never-seen cells are ignored)
        not_occ = ~((grid.classes() == OCCUPIED) | (grid.logodds > 0.3))
        not_occ = not_occ.astype(np.uint8)
        self._occ_clear = cv2.distanceTransform(not_occ, cv2.DIST_L2, 5) * grid.cfg.resolution
        self._cache_rev = (id(grid), grid.revision)

    def traversable_xy(self, grid, xy):
        self.prepare(grid)
        ix, iy = grid.to_cell(np.asarray(xy, float))
        ok = grid.inside(ix, iy)
        out = np.zeros(len(ix), bool)
        out[ok] = self._trav[iy[ok], ix[ok]]
        return out

    def clearance_xy(self, grid, xy):
        self.prepare(grid)
        ix, iy = grid.to_cell(np.asarray(xy, float))
        ok = grid.inside(ix, iy)
        out = np.zeros(len(ix))
        out[ok] = self._clear[iy[ok], ix[ok]]
        return out

    def obstacle_clearance_xy(self, grid, xy):
        """Distance (m) to the nearest cell with obstacle evidence: is turning in place safe here?"""
        self.prepare(grid)
        ix, iy = grid.to_cell(np.asarray(xy, float))
        ok = grid.inside(ix, iy)
        out = np.zeros(len(ix))
        out[ok] = self._occ_clear[iy[ok], ix[ok]]
        return out

    def _snap_start(self, grid, start):
        ix, iy = grid.to_cell(np.asarray(start, float))
        ix, iy = int(ix[0]), int(iy[0])
        if grid.inside(np.array([ix]), np.array([iy]))[0] and self._trav[iy, ix]:
            return ix, iy
        r = int(math.ceil(self.cfg.start_snap / grid.cfg.resolution))
        best = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = ix + dx, iy + dy
                if 0 <= x < grid.n and 0 <= y < grid.n and self._trav[y, x]:
                    d = dx * dx + dy * dy
                    if best is None or d < best[0]:
                        best = (d, x, y)
        if best is None:
            raise GoalRejected("start_not_in_certified_free_space")
        return best[1], best[2]

    def check_goal(self, grid, goal):
        self.prepare(grid)
        ix, iy = grid.to_cell(np.asarray(goal, float))
        if not grid.inside(ix, iy)[0]:
            raise GoalRejected("goal_outside_map")
        cls = grid.classes()[iy[0], ix[0]]
        if cls != FREE:
            raise GoalRejected("goal_in_unknown_space" if cls == 0 else "goal_occupied")
        if not self._trav[iy[0], ix[0]]:
            raise GoalRejected("goal_lacks_footprint_clearance")
        return int(ix[0]), int(iy[0])

    def snap_goal(self, grid, goal):
        """Nearest certified-traversable cell within ``goal_snap`` of ``goal``.

        Used only when an already-accepted goal loses certification while driving
        (new evidence near it). New goals are still checked strictly."""
        self.prepare(grid)
        ix, iy = grid.to_cell(np.asarray(goal, float))
        ix, iy = int(ix[0]), int(iy[0])
        r = int(math.ceil(self.cfg.goal_snap / grid.cfg.resolution))
        best = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = ix + dx, iy + dy
                if 0 <= x < grid.n and 0 <= y < grid.n and self._trav[y, x]:
                    d = math.hypot(dx, dy) * grid.cfg.resolution
                    if d <= self.cfg.goal_snap and (best is None or d < best[0]):
                        best = (d, x, y)
        if best is None:
            raise GoalRejected("goal_lost_certification_no_snap")
        return grid.to_xy(np.array([best[1]]), np.array([best[2]]))[0]

    def plan(self, grid, start, goal, *, avoid=()):
        """A* over traversable cells. ``avoid``: [(xy, radius)] soft keep-out regions
        (learned aversions) that add cost but never make unknown space free."""
        gx, gy = self.check_goal(grid, goal)
        sx, sy = self._snap_start(grid, start)
        n = grid.n
        trav, clear = self._trav, self._clear
        res = grid.cfg.resolution
        penalty = np.zeros((n, n), np.float32)
        if avoid:
            iy, ix = np.mgrid[0:n, 0:n]
            centres = grid.to_xy(ix.ravel(), iy.ravel()).reshape(n, n, 2)
            for xy, radius in avoid:
                d = np.linalg.norm(centres - np.asarray(xy)[None, None], axis=2)
                penalty += (self.cfg.avoid_weight * np.clip(1 - d / radius, 0, 1)).astype(np.float32)
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        open_heap = [(0.0, 0.0, sx, sy)]
        came = {}
        cost = {(sx, sy): 0.0}
        found = False
        while open_heap:
            _, g, x, y = heapq.heappop(open_heap)
            if (x, y) == (gx, gy):
                found = True
                break
            if g > cost.get((x, y), math.inf) + 1e-9:
                continue
            for dx, dy, step in moves:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n) or not trav[ny, nx]:
                    continue
                c = step * res * (1 + self.cfg.clearance_weight * 0.1 / max(clear[ny, nx], 0.05)) + penalty[ny, nx] * res
                ng = g + c
                if ng < cost.get((nx, ny), math.inf):
                    cost[(nx, ny)] = ng
                    came[(nx, ny)] = (x, y)
                    h = math.hypot(gx - nx, gy - ny) * res
                    heapq.heappush(open_heap, (ng + h, ng, nx, ny))
        if not found:
            raise GoalRejected("goal_unreachable_through_certified_space")
        cells = [(gx, gy)]
        while cells[-1] != (sx, sy):
            cells.append(came[cells[-1]])
        cells.reverse()
        xy = grid.to_xy(np.array([c[0] for c in cells]), np.array([c[1] for c in cells]))
        xy[-1] = goal
        return self._shortcut(grid, xy)

    @staticmethod
    def segment_samples(grid, a, b):
        """Points checked along a straight segment (both ends included, about half-cell
        spacing: the planner's original shortcut sampling, so paths are unchanged).
        The planner's shortcut test and the navigator's per-step corridor test use
        these SAME points: with different samplers a one-cell sliver at the edge of
        the clearance band could pass one test and fail the other, and the robot then
        held forever on a path the planner kept re-issuing. (A denser quarter-cell
        spacing changed path shapes and was measured worse: docs/NAVIGATION_RESULTS.md.)"""
        a, b = np.asarray(a, float), np.asarray(b, float)
        steps = max(2, int(np.linalg.norm(b - a) / (grid.cfg.resolution / 2)))
        return np.linspace(a, b, steps)

    def segment_clear(self, grid, a, b):
        return bool(self.traversable_xy(grid, self.segment_samples(grid, a, b)).all())

    def _shortcut(self, grid, xy):
        if len(xy) <= 2:
            return xy
        out = [xy[0]]
        i = 0
        while i < len(xy) - 1:
            j = len(xy) - 1
            while j > i + 1 and not self.segment_clear(grid, xy[i], xy[j]):
                j -= 1
            out.append(xy[j])
            i = j
        return np.array(out)

    def path_valid(self, grid, path, from_index=0):
        path = np.asarray(path)
        for a, b in zip(path[from_index:-1], path[from_index + 1:]):
            if not self.segment_clear(grid, a, b):
                return False
        return True
