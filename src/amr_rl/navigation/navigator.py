"""Goal navigation: plan on certified free space, follow with pure pursuit.

Produces body-frame (v, w) requests only; the supervisor decides whether they
reach the wheels. Arrival is a navigation result, never an interaction outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..perception.camera_model import wrap
from .planner import GoalRejected, Planner


@dataclass
class FollowerConfig:
    v_max: float = 0.22
    w_max: float = 1.0
    lookahead: float = 0.25
    rotate_threshold: float = 0.7
    goal_tolerance: float = 0.07
    heading_tolerance: float = 0.08
    replan_period: float = 1.5
    max_position_sigma: float = 0.06
    max_replans: int = 6
    corridor: float = 0.6          # path length ahead checked every step
    corridor_skip: float = 0.15    # the robot may sit inside the clearance band
    turn_margin: float = 0.02      # occupied evidence inside radius+margin: do not turn in place
    backoff_speed: float = 0.06
    max_backoff: float = 0.4


class Navigator:
    def __init__(self, planner: Planner | None = None, config: FollowerConfig | None = None):
        self.planner = planner or Planner()
        self.cfg = config or FollowerConfig()
        self.guard = None  # optional NearFieldGuard (speed gate on freshly verified floor)
        self.reset()

    def reset(self):
        self.goal = None
        self.goal_heading = None
        self.path = np.zeros((0, 2))
        self.index = 1
        self.status = "idle"
        self.reason = ""
        self.last_plan = -math.inf
        self.replans = 0
        self.rejections = 0
        self.avoid = ()
        self.backed_off = 0.0
        self._last_xy = None
        self.holding = False
        self.original_goal = None
        self.goal_snaps = 0

    def set_goal(self, grid, pose, goal, *, heading=None, now=0.0, avoid=()):
        self.reset_goal_state()
        self.avoid = tuple(avoid)
        try:
            self.path = self.planner.plan(grid, pose[:2], np.asarray(goal, float), avoid=self.avoid)
        except GoalRejected as error:
            self.status, self.reason = "rejected", error.reason
            self.rejections += 1
            return False
        self.goal, self.goal_heading = np.asarray(goal, float), heading
        self.original_goal = self.goal.copy()
        self.index, self.status, self.reason = 1, "following", ""
        self.last_plan = now
        if self.guard is not None:
            self.guard.reset()
        return True

    def reset_goal_state(self):
        self.goal = None
        self.goal_heading = None
        self.path = np.zeros((0, 2))
        self.index = 1
        self.status = "idle"
        self.reason = ""
        self.replans = 0
        self.backed_off = 0.0
        self._last_xy = None
        self.holding = False
        self.original_goal = None
        self.goal_snaps = 0

    def cancel(self, reason="cancelled"):
        self.reset_goal_state()
        self.status, self.reason = "cancelled", reason

    def step(self, grid, pose, sigma, now):
        """Return (v, w). Status becomes arrived / blocked / localization_uncertain."""
        if self.status not in ("following", "aligning"):
            return 0.0, 0.0
        if pose is None or sigma is None or sigma > self.cfg.max_position_sigma:
            self.status, self.reason = "localization_uncertain", "position uncertainty above bound"
            return 0.0, 0.0
        x, y, th = pose
        to_goal = self.goal - np.array([x, y])
        if self.status == "following" and np.linalg.norm(to_goal) <= self.cfg.goal_tolerance:
            self.status = "aligning" if self.goal_heading is not None else "arrived"
        if self.status == "aligning":
            err = wrap(self.goal_heading - th)
            if abs(err) <= self.cfg.heading_tolerance:
                self.status = "arrived"
                return 0.0, 0.0
            return self._turn_or_back_off(grid, pose, float(np.clip(1.5 * err, -0.6, 0.6)))
        if self.status == "arrived":
            return 0.0, 0.0
        # Every step: the next stretch of path must still be certified; if not, stop
        # at once (hold). Replanning happens at most once per replan period, so a
        # transient change in evidence does not burn the replan budget.
        corridor_ok = self._corridor_clear(grid, pose)
        if now - self.last_plan >= self.cfg.replan_period:
            self.last_plan = now
            if not corridor_ok or not self.planner.path_valid(grid, self.path, from_index=max(self.index - 1, 0)):
                self.replans += 1
                if self.replans > self.cfg.max_replans:
                    self.status, self.reason = "blocked", "replan budget exhausted"
                    return 0.0, 0.0
                try:
                    self.path = self._replan(grid, pose)
                    self.index = 1
                    if self.guard is not None:
                        self.guard.reset()
                except GoalRejected as error:
                    self.status, self.reason = "blocked", error.reason
                    return 0.0, 0.0
                corridor_ok = self._corridor_clear(grid, pose)
        if not corridor_ok:
            self.holding = True
            return 0.0, 0.0
        self.holding = False
        # Advance the carrot along the path.
        while self.index < len(self.path) - 1 and np.linalg.norm(self.path[self.index] - pose[:2]) < self.cfg.lookahead:
            self.index += 1
        target = self.path[min(self.index, len(self.path) - 1)]
        d = target - np.array([x, y])
        heading_err = wrap(math.atan2(d[1], d[0]) - th)
        dist_goal = float(np.linalg.norm(to_goal))
        if abs(heading_err) > self.cfg.rotate_threshold:
            return self._turn_or_back_off(grid, pose, float(np.clip(1.8 * heading_err, -self.cfg.w_max,
                                                                      self.cfg.w_max)))
        v = self.cfg.v_max * max(0.25, math.cos(heading_err)) * min(1.0, dist_goal / 0.35 + 0.15)
        w = float(np.clip(2.2 * heading_err, -self.cfg.w_max, self.cfg.w_max))
        if self.guard is not None:
            lo, hi = self.guard.cfg.band
            ahead = self._path_points(grid, pose, lo, min(hi, max(dist_goal - 0.05, lo)))
            if len(ahead):
                n = np.array([-math.sin(th), math.cos(th)])
                hw = self.guard.cfg.band_halfwidth
                band = np.vstack([ahead + k * hw * n[None] for k in (-1.0, -0.5, 0.0, 0.5, 1.0)])
            else:
                band = np.zeros((0, 2))
            self.guard.set_band(band)
            if len(band):
                limit, hold, timed_out = self.guard.speed_limit(now)
                if timed_out:
                    self.status, self.reason = "blocked", "path_ahead_not_verified"
                    self.holding = True
                    return 0.0, 0.0
                if hold:
                    self.holding = True
                    return 0.0, 0.0
                if limit is not None:
                    v = min(v, limit)
        return float(v), w

    def _replan(self, grid, pose):
        """Replan to the goal; if the goal itself lost certification while driving,
        move it to the nearest certified cell within the planner's goal_snap
        (bounded; reported in ``goal_snaps``)."""
        try:
            return self.planner.plan(grid, pose[:2], self.goal, avoid=self.avoid)
        except GoalRejected as error:
            if error.reason not in ("goal_lacks_footprint_clearance", "goal_in_unknown_space", "goal_occupied"):
                raise
            snapped = self.planner.snap_goal(grid, self.original_goal)
            path = self.planner.plan(grid, pose[:2], snapped, avoid=self.avoid)
            self.goal = np.asarray(snapped, float)
            self.goal_snaps += 1
            return path

    def _path_points(self, grid, pose, lo, hi):
        """Points sampled along the planned polyline (not the robot->carrot chord: the
        robot may legitimately sit inside the clearance band after a start snap) whose
        distance from the robot lies in [lo, hi]."""
        here = np.asarray(pose[:2], float)
        verts = self.path[max(self.index - 1, 0):]
        pts = []
        for a, b in zip(verts[:-1], verts[1:]):
            n = max(1, int(np.linalg.norm(b - a) / (grid.cfg.resolution / 2)))
            seg = a + (b - a) * (np.arange(1, n + 1)[:, None] / n)
            pts.append(seg)
            if np.linalg.norm(b - here) > hi + 0.3:
                break
        if not pts:
            return np.zeros((0, 2))
        pts = np.vstack(pts)
        d = np.linalg.norm(pts - here[None], axis=1)
        return pts[(d >= lo) & (d <= hi)]

    def _corridor_clear(self, grid, pose):
        """Is the certified path still certified for the next ``corridor`` metres?"""
        pts = self._path_points(grid, pose, self.cfg.corridor_skip, self.cfg.corridor)
        return len(pts) == 0 or bool(self.planner.traversable_xy(grid, pts).all())

    def _turn_or_back_off(self, grid, pose, w):
        """Turn in place only if no known obstacle lies inside the turning circle;
        otherwise reverse straight (the way the robot came) by at most max_backoff."""
        radius = self.planner.cfg.footprint_radius + self.cfg.turn_margin
        here = float(self.planner.obstacle_clearance_xy(grid, [pose[:2]])[0])
        if here >= radius:
            self._last_xy = None
            return 0.0, w
        if self._last_xy is not None:
            self.backed_off += float(np.linalg.norm(np.asarray(pose[:2]) - self._last_xy))
        self._last_xy = np.asarray(pose[:2], float)
        behind = np.asarray(pose[:2]) - 0.1 * np.array([math.cos(pose[2]), math.sin(pose[2])])
        if self.backed_off >= self.cfg.max_backoff or \
                float(self.planner.obstacle_clearance_xy(grid, [behind])[0]) <= here:
            self.status, self.reason = "blocked", "no_certified_room_to_turn"
            return 0.0, 0.0
        return -self.cfg.backoff_speed, 0.0

    def snapshot(self):
        return {
            "goal": None if self.goal is None else [float(v) for v in self.goal],
            "path": [[float(a), float(b)] for a, b in self.path],
            "status": self.status,
            "reason": self.reason,
            "rejected_goals": self.rejections,
            "goal_snaps": self.goal_snaps,
        }
