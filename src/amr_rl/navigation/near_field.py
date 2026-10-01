"""Near-field guard: stop for objects that appeared on the route since it was mapped.

Round 2: a box put down on the robot's route was touched in 4 of 5 tests, and
scripts/dev/turn_then_box_probe.py reproduces it (home_a: 4/4 scenarios end in
contact). The persistent map needs about three keyframes (one per 10 cm of travel)
of obstacle evidence to overturn saturated free evidence, and an in-place turn gives
no parallax at all, so the box is reached before the map changes.

This guard uses a single-frame cue that needs no motion: obstacle points from the
monocular depth detector (perception/near_depth.py), metric because the floor fixes
the model's scale every frame. Every ``period`` seconds while a path is being
followed (including turning toward it):

* obstacle points within ``footprint_radius + margin`` of the path ahead are found;
  the along-path distance to the first such path point is the free run;
* free run <= ``stop_run``: stop (one frame suffices; a false alarm costs one period);
* points seen in >= 2 frames within ``confirm_window`` (same 5 cm cell, 3x3) are
  written into the grid as occupied, so the ordinary corridor check holds the robot
  and the planner routes around or reports blocked;
* free run <= ``slow_run``: creep at ``creep_speed``;
* stopped by the guard for ``hold_timeout``: the goal ends blocked ("obstacle_ahead").

Rejected camera-only variants based on the plane-parallax floor test (asserting
probe obstacle bases: 83 cells on open floor; requiring freshly verified floor:
4/5 open-floor trips failed) are documented in docs/results/near-field-guard.md.

Pixels in, decisions out: only the robot's own images, calibration and pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GuardConfig:
    period: float = 0.3          # s between detector frames while following a path
    margin: float = 0.0          # m added to the footprint radius for "on the path" (the planner keeps 3 cm more)
    look: float = 1.0            # m of path ahead considered
    min_points: int = 6          # obstacle points needed (robust to speckle)
    stop_run: float = 0.20       # m of free path left: stop
    slow_run: float = 0.50       # m: creep
    creep_speed: float = 0.07    # m/s
    confirm_window: float = 1.5  # s
    assert_level: float = 1.5    # log-odds written into confirmed obstacle cells
    assert_range: float = 1.0    # m; only points this close are written
    hold_timeout: float = 4.0    # s
    observe_only: bool = False   # evaluation: detect and record, change nothing


class DepthGuard:
    def __init__(self, detector, grid, footprint_radius, config: GuardConfig | None = None):
        self.detector = detector
        self.grid = grid
        self.radius = float(footprint_radius)
        self.cfg = config or GuardConfig()
        self.path_ahead = np.zeros((0, 2))
        self.last_frame = -np.inf
        self.stats = {"frames": 0, "stops": 0, "asserted_cells": 0, "assert_events": 0, "creep_steps": 0,
                      "hold_timeouts": 0}
        self.asserted_log = []
        self.recent = []
        self.free_run = None
        self.free_run_t = -np.inf
        self.hold_since = None

    def reset(self):
        """New goal: forget the hold timer (the latest detection stays valid until stale)."""
        self.hold_since = None

    def set_path_ahead(self, pts):
        self.path_ahead = np.asarray(pts, float).reshape(-1, 2)

    # ------------------------------------------------------------ detection
    def maybe_detect(self, now, pose, rgb):
        if now - self.last_frame < self.cfg.period or pose is None or rgb is None or self.detector is None:
            return None
        self.last_frame = now
        out = self.detector.detect(rgb)
        if out is None:
            return None
        self.stats["frames"] += 1
        pts = out["points"]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        world = np.column_stack([pose[0] + c * pts[:, 0] - s * pts[:, 1],
                                 pose[1] + s * pts[:, 0] + c * pts[:, 1]]) if len(pts) else np.zeros((0, 2))
        here = np.asarray(pose[:2], float)
        self.free_run = self._free_run(world, here)
        self.free_run_t = now
        if not self.cfg.observe_only:
            self._confirm_and_assert(now, world, here)
        return {"points": len(pts), "free_run": self.free_run}

    def _free_run(self, world, here):
        """Along-path distance to the first path point where the footprint would
        overlap obstacle points (None: nothing on the path within ``look``)."""
        path = self.path_ahead
        if len(path) == 0 or len(world) < self.cfg.min_points:
            return None
        seg = np.r_[np.linalg.norm(path[0] - here), np.linalg.norm(np.diff(path, axis=0), axis=1)]
        along = np.cumsum(seg)
        keep = along <= self.cfg.look
        path, along = path[keep], along[keep]
        if len(path) == 0:
            return None
        d = np.linalg.norm(world[:, None, :] - path[None, :, :], axis=2)  # points x path
        hits = (d <= self.radius + self.cfg.margin).sum(axis=0)
        first = np.flatnonzero(hits >= self.cfg.min_points)
        return float(along[first[0]]) if len(first) else None

    def _confirm_and_assert(self, now, world, here):
        cells = set()
        if len(world):
            near = np.linalg.norm(world - here[None], axis=1) <= self.cfg.assert_range
            ix, iy = self.grid.to_cell(world[near])
            ok = self.grid.inside(ix, iy)
            if ok.any():
                uniq, counts = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0, return_counts=True)
                cells = {tuple(c) for c, n in zip(uniq.tolist(), counts) if n >= 2}
        self.recent = [(t, c) for t, c in self.recent if now - t <= self.cfg.confirm_window]
        confirmed = set()
        for _, older in self.recent:
            grown = {(x + dx, y + dy) for x, y in older for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            confirmed |= cells & grown
        self.recent.append((now, cells))
        if not confirmed:
            return
        xs = np.array([c[0] for c in confirmed])
        ys = np.array([c[1] for c in confirmed])
        before = self.grid.logodds[ys, xs].copy()
        self.grid.logodds[ys, xs] = np.maximum(before, self.cfg.assert_level)
        changed = int((before < self.cfg.assert_level).sum())
        if changed:
            self.grid.revision += 1
            self.stats["asserted_cells"] += changed
            self.stats["assert_events"] += 1
            self.asserted_log.append((float(now), self.grid.to_xy(xs, ys).tolist()))

    # ------------------------------------------------------------ decisions
    def speed_limit(self, now):
        """(v_limit or None, hold, timed_out)."""
        if self.cfg.observe_only or self.free_run is None or now - self.free_run_t > 2 * self.cfg.period:
            self.hold_since = None
            return None, False, False
        if self.free_run <= self.cfg.stop_run:
            if self.hold_since is None:
                self.hold_since = now
                self.stats["stops"] += 1
            if now - self.hold_since >= self.cfg.hold_timeout:
                self.stats["hold_timeouts"] += 1
                return 0.0, True, True
            return 0.0, True, False
        self.hold_since = None
        if self.free_run <= self.cfg.slow_run:
            self.stats["creep_steps"] += 1
            return self.cfg.creep_speed, False, False
        return None, False, False
