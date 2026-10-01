"""Near-field guard (EXPERIMENTAL, OFF by default): do not drive into space that was
not freshly seen to be floor.

Round 2 found that a box put down on the robot's route was touched in 4 of 5 tests,
and scripts/dev/turn_then_box_probe.py reproduces it (home_a: 4/4 scenarios end in
contact). The persistent map needs about three keyframes (one per 10 cm of travel)
of obstacle evidence to overturn saturated free evidence; after an in-place turn
there is no parallax at all, so a box ~0.3 m ahead of the bumper is reached before
the map changes.

While the robot follows a path, the guard runs the map's own plane-parallax floor
test between the CURRENT frame and a recent keyframe with enough baseline, every
``period`` seconds, without writing anything to the map, and asks what fraction of
the path band just ahead was just verified as floor: full speed above
``fast_above``, creep below it, stop after ``hold_probes`` probes below
``hold_below``, and end the goal as blocked after ``hold_timeout``.

Measured on the reproducible scenarios (docs/results/near-field-guard.md), neither
variant is good enough to enable:

* band verification (this default): 4 of 5 ordinary trips on open floor ended
  "path_ahead_not_verified" (open-floor bands verify at 0.45-1.0), and with the box
  ahead the band still verified at 0.45-0.6, so one of two box scenarios still
  ended in contact;
* ``assert_obstacles`` (two agreeing probes write obstacle cells into the grid):
  83 cells asserted on open floor in one run, making later goals unreachable.

The plane-parallax test is not discriminative enough at 0.3-0.5 m for a per-frame
decision. A dedicated near-field cue (for example a monocular depth or floor
segmentation model behind the same camera-only contract) is the next thing to try.

Pixels in, decisions out: the guard uses only the robot's own images and pose estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..perception.vslam import Keyframe


@dataclass
class GuardConfig:
    period: float = 0.3           # s between probes while following a path
    min_baseline: float = 0.04    # m between the probe frame and its reference keyframe
    max_ref_age: float = 6.0      # s; older keyframes are not used as references
    verify_window: float = 1.0    # s; floor verified this recently counts
    band: tuple = (0.30, 0.50)    # m from the robot along the path
    band_halfwidth: float = 0.10  # m either side of the path
    fast_above: float = 0.75
    hold_below: float = 0.50
    hold_probes: int = 2
    hold_timeout: float = 2.0     # s stopped by the guard before the goal is reported blocked
    creep_speed: float = 0.07     # m/s
    assert_obstacles: bool = False  # rejected variant (see module docstring)
    assert_level: float = 1.5
    near: float = 1.0
    confirm_window: float = 1.5
    observe_only: bool = False    # evaluation: record probes, change nothing
    debug: bool = False           # evaluation: keep every probe's evidence


class NearFieldGuard:
    def __init__(self, mapper, grid, config: GuardConfig | None = None):
        self.mapper = mapper
        self.grid = grid
        self.cfg = config or GuardConfig()
        self.free_t = np.full(grid.logodds.shape, -np.inf, np.float32)
        self.last_probe = -np.inf
        self.band_xy = np.zeros((0, 2))
        self.recent = []
        self.stats = {"probes": 0, "skipped": 0, "creep_steps": 0, "holds": 0, "hold_timeouts": 0,
                      "asserted_cells": 0, "assert_events": 0}
        self.asserted_log = []
        self.debug_log = []
        self.reset()

    def reset(self):
        """New goal or new path: forget the verdict about the old band."""
        self.recent = []
        self.last_fraction = None
        self.last_eval = -np.inf
        self.low_streak = 0
        self.hold_since = None

    # ------------------------------------------------------------ band
    def set_band(self, pts):
        self.band_xy = np.asarray(pts, float).reshape(-1, 2)

    def band_fraction(self, now):
        if len(self.band_xy) == 0:
            return None
        ix, iy = self.grid.to_cell(self.band_xy)
        ok = self.grid.inside(ix, iy)
        cells = np.unique(np.stack([ix[ok], iy[ok]], 1), axis=0)
        if len(cells) == 0:
            return None
        fresh = (now - self.free_t[cells[:, 1], cells[:, 0]]) <= self.cfg.verify_window
        return float(fresh.mean())

    # ------------------------------------------------------------ probing
    def maybe_probe(self, now, pose, keyframes, frame):
        """``frame``: (gray, pts, desc) of the current image, as tracked."""
        if now - self.last_probe < self.cfg.period or pose is None or not keyframes or frame is None:
            return None
        gray, pts, desc = frame
        if gray is None:
            return None
        self.last_probe = now
        here = np.asarray(pose[:2], float)
        ref = None
        for kf in reversed(keyframes[-8:]):
            if kf.gray is None or now - kf.timestamp > self.cfg.max_ref_age:
                continue
            if np.linalg.norm(np.asarray(kf.pose[:2]) - here) >= self.cfg.min_baseline:
                ref = kf
                break
        if ref is None:
            self.stats["skipped"] += 1
            return None
        cur = Keyframe(-1, now, np.asarray(pose, float).copy(), pts, desc, np.full(len(pts), -1), gray)
        r = self.mapper.update(ref, cur, defer_bases=True, commit_free=False)
        if not r.get("updated"):
            self.stats["skipped"] += 1
            return None
        self.stats["probes"] += 1
        free_xy = r.get("free_xy")
        if free_xy is not None and len(free_xy):
            ix, iy = self.grid.to_cell(free_xy)
            ok = self.grid.inside(ix, iy)
            self.free_t[iy[ok], ix[ok]] = now
        frac = self.band_fraction(now)
        if frac is not None:
            self.last_fraction, self.last_eval = frac, now
            self.low_streak = self.low_streak + 1 if frac < self.cfg.hold_below else 0
        if self.cfg.debug:
            b = r.get("base_xy")
            fc = [] if free_xy is None or not len(free_xy) else \
                np.unique(np.stack(self.grid.to_cell(free_xy), 1), axis=0).tolist()
            self.debug_log.append({"t": float(now), "ref": int(ref.id), "pose": [float(v) for v in pose],
                                   "band_fraction": frac, "base_xy": [] if b is None else np.asarray(b).tolist(),
                                   "free_cells": fc})
        if self.cfg.assert_obstacles and not self.cfg.observe_only:
            self._assert_bases(now, here, r.get("base_xy"))
        return {"band_fraction": frac}

    def _assert_bases(self, now, here, base):
        cells = set()
        if base is not None and len(base):
            near = np.linalg.norm(base - here[None], axis=1) <= self.cfg.near
            ix, iy = self.grid.to_cell(base[near])
            ok = self.grid.inside(ix, iy)
            cells = set(zip(ix[ok].tolist(), iy[ok].tolist()))
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
            self.asserted_log = self.asserted_log[-5000:]

    # ------------------------------------------------------------ decisions
    def speed_limit(self, now):
        """(v_limit or None, hold, timed_out) for the current band."""
        if self.cfg.observe_only:
            return None, False, False
        if self.low_streak >= self.cfg.hold_probes:
            if self.hold_since is None:
                self.hold_since = now
                self.stats["holds"] += 1
            timed_out = now - self.hold_since >= self.cfg.hold_timeout
            if timed_out:
                self.stats["hold_timeouts"] += 1
            return 0.0, True, timed_out
        self.hold_since = None
        fresh = self.last_fraction is not None and now - self.last_eval <= self.cfg.verify_window
        if fresh and self.last_fraction >= self.cfg.fast_above:
            return None, False, False
        self.stats["creep_steps"] += 1
        return self.cfg.creep_speed, False, False
