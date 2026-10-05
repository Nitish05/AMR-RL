"""View check for in-place turns: does the camera face a near surface?

ENGINEERED geometry and thresholds (nothing here is learned).

The single forward camera sits ``cam_forward`` (0.125 m) ahead of the rotation
axis. Turning in place while it faces a surface closer than about 0.5 m makes the
visual heading estimate over-rotate (measured live, round 9: +2.6 deg/rad bias;
4.6 % of turning time but 35 % of the 1 s windows with > 3 deg heading error; the
two worst runs jumped 16-43 deg in a survey sweeping a shelf 0.34-0.45 m away).
Perception-aware turning (behaviour/activities.py, navigation/navigator.py, behind
``view_aware_turns``, off by default) uses these helpers to keep optional sweeps
away from near surfaces and to back off before necessary turns.

Evidence is only the robot's own: its occupancy grid (estimated from onboard RGB)
and, optionally, recent obstacle points from the near-field depth guard
(``DepthGuard.view_points``), which cover the start when the grid is still empty.
The grid lags and misses near geometry (recall vs true geometry: 68 % at 0.5 m,
88 % at 0.6 m), hence the 0.6 m threshold. UNKNOWN cells are reported separately
and are NOT counted as near: at the start almost everything is unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..mapping.occupancy import FREE, OCCUPIED, UNKNOWN


@dataclass
class ViewCheckConfig:
    cam_forward: float = 0.125     # m, optical centre ahead of the rotation axis
    near: float = 0.6              # m, "facing a near surface" threshold (grid recall 88 % at 0.6 m)
    max_range: float = 1.5         # m, rays stop here
    step: float = math.radians(5)  # heading sampling of a swept sector
    occupied_level: float = 0.3    # log-odds: obstacle evidence as in Planner.obstacle_clearance_xy
    point_halfwidth: float = 0.05  # m, extra points within this lateral distance of a ray count
    min_points: int = 3            # a ray is stopped by the k-th nearest extra point (speckle)
    # Navigator (necessary turns)
    min_turn: float = 0.35         # rad; smaller turns are not checked (bias ~ angle turned)
    end_margin: float = 0.2        # rad; the final heading's neighbourhood is not checked (facing a target is the point)
    backoff_steps: tuple = (0.1, 0.2, 0.3)  # m, candidate reverse distances (within FollowerConfig.max_backoff)
    long_way_window: float = 0.35  # rad; turn the long way only when |error| > pi - this
    # Activities (optional sweeps)
    panorama_near_fraction: float = 0.3  # skip a 360 deg look when more of it faces near
    min_sweep: float = math.radians(10)  # a sweep side shortened below this is dropped
    survey_lookahead: float = 0.35       # rad checked ahead of the camera during a survey


@dataclass
class FacingResult:
    distance: np.ndarray  # m from the camera to the first obstacle (grid or point); inf if none in range
    unknown: np.ndarray   # m from the camera to the first UNKNOWN cell before that obstacle; inf if none
    source: list          # per heading: "grid", "points" or None


def camera_xy(pose, heading, cam_forward):
    return np.asarray(pose[:2], float) + cam_forward * np.array([math.cos(heading), math.sin(heading)])


def facing_distances(grid, pose, headings, cam_forward=0.125, max_range=1.5, *, points=None,
                     occupied_level=0.3, point_halfwidth=0.05, min_points=3) -> FacingResult:
    """For each candidate heading of a robot turning in place at ``pose[:2]``: cast a
    ray from the CAMERA position (base + cam_forward * (cos h, sin h)) along h on the
    robot's own occupancy grid and return the distance to the first cell with obstacle
    evidence (OCCUPIED class incl. keep-out, or log-odds > ``occupied_level``) and,
    separately, the distance to the first UNKNOWN cell before it.

    ``points``: optional extra obstacle points (world xy, e.g. the depth guard's
    recent near-field points). A ray is stopped by them at the ``min_points``-th
    nearest point lying within ``point_halfwidth`` of the ray. The result is the
    nearer of the two sources."""
    headings = np.atleast_1d(np.asarray(headings, float))
    res = grid.cfg.resolution
    steps = np.arange(res / 2, max_range + 1e-9, res / 2)
    classes = grid.classes()
    occ = (classes == OCCUPIED) | (grid.logodds > occupied_level)
    unknown_cls = classes == UNKNOWN
    n_h = len(headings)
    dist = np.full(n_h, np.inf)
    unk = np.full(n_h, np.inf)
    source = [None] * n_h
    u = np.stack([np.cos(headings), np.sin(headings)], 1)                      # (h, 2)
    cams = np.asarray(pose[:2], float)[None] + cam_forward * u                  # (h, 2)
    samples = cams[:, None, :] + u[:, None, :] * steps[None, :, None]           # (h, s, 2)
    ix, iy = grid.to_cell(samples.reshape(-1, 2))
    ok = grid.inside(ix, iy)
    hit = np.zeros(len(ix), bool)
    unseen = np.ones(len(ix), bool)  # outside the map counts as unknown
    hit[ok] = occ[iy[ok], ix[ok]]
    unseen[ok] = unknown_cls[iy[ok], ix[ok]] & ~hit[ok]
    hit = hit.reshape(n_h, -1)
    unseen = unseen.reshape(n_h, -1)
    for k in range(n_h):
        first = np.flatnonzero(hit[k])
        if len(first):
            dist[k] = float(steps[first[0]]) - res / 4  # middle of the last sampling interval
            source[k] = "grid"
        lim = first[0] if len(first) else len(steps)
        first_unk = np.flatnonzero(unseen[k, :lim])
        if len(first_unk):
            unk[k] = float(steps[first_unk[0]])
    if points is not None:
        pts = np.asarray(points, float).reshape(-1, 2)
        if len(pts) >= max(min_points, 1):
            rel = pts[None, :, :] - cams[:, None, :]                            # (h, p, 2)
            along = np.einsum("hpk,hk->hp", rel, u)
            across = np.abs(rel[..., 0] * u[:, None, 1] - rel[..., 1] * u[:, None, 0])
            on_ray = (along > 0) & (along <= max_range) & (across <= point_halfwidth)
            for k in range(n_h):
                d = np.sort(along[k][on_ray[k]])
                if len(d) >= min_points and d[min_points - 1] < dist[k]:
                    dist[k] = float(d[min_points - 1])
                    source[k] = "points"
    return FacingResult(dist, unk, source)


def sector_headings(start_heading, end_heading, step=math.radians(5)):
    """Headings swept turning from ``start_heading`` to ``end_heading`` (signed and
    unwrapped: start + 2 pi is a full turn left), the start excluded, the end included."""
    span = float(end_heading - start_heading)
    if abs(span) < 1e-9:
        return np.zeros(0)
    n = max(1, int(math.ceil(abs(span) / step - 1e-9)))
    return start_heading + span * np.arange(1, n + 1) / n


def _facing(grid, pose, headings, cfg, points):
    return facing_distances(grid, pose, headings, cfg.cam_forward, cfg.max_range, points=points,
                            occupied_level=cfg.occupied_level, point_halfwidth=cfg.point_halfwidth,
                            min_points=cfg.min_points)


def sector_near_fraction(grid, pose, start_heading, end_heading, step=math.radians(5), *,
                         cfg: ViewCheckConfig | None = None, points=None) -> float:
    """Fraction of the headings swept from ``start_heading`` to ``end_heading`` at which
    the camera faces an obstacle closer than ``cfg.near`` (0 = the sweep stays clear)."""
    cfg = cfg or ViewCheckConfig()
    hs = sector_headings(start_heading, end_heading, step)
    if len(hs) == 0:
        return 0.0
    return float(np.mean(_facing(grid, pose, hs, cfg, points).distance < cfg.near))


# Name used in the round-9 plan: "sector_clear" -> fraction of the sector facing < near.
sector_clear = sector_near_fraction


def clear_extent(grid, pose, direction, max_angle, *, cfg: ViewCheckConfig | None = None, points=None) -> float:
    """Largest angle (<= ``max_angle``) the robot can turn from ``pose[2]`` in
    ``direction`` (+1 left, -1 right) with the camera never facing an obstacle closer
    than ``cfg.near`` (the current heading itself is not checked)."""
    cfg = cfg or ViewCheckConfig()
    hs = sector_headings(pose[2], pose[2] + direction * max_angle, cfg.step)
    if len(hs) == 0:
        return float(max_angle)
    near = _facing(grid, pose, hs, cfg, points).distance < cfg.near
    if not near.any():
        return float(max_angle)
    first = int(np.flatnonzero(near)[0])
    return 0.0 if first == 0 else float(abs(hs[first - 1] - pose[2]))


def faces_near(grid, pose, heading, *, cfg: ViewCheckConfig | None = None, points=None) -> bool:
    cfg = cfg or ViewCheckConfig()
    return bool(_facing(grid, pose, [heading], cfg, points).distance[0] < cfg.near)


def plan_sweep(grid, pose, sweep, *, cfg: ViewCheckConfig | None = None, points=None):
    """Perception-aware version of an explore look-around ``[a, -2a, a]`` (left, right,
    back to centre): each side is shortened to stop before the first heading facing
    near, and dropped if shorter than ``cfg.min_sweep``. Returns the new list of
    signed turns (may start to the right, or be empty)."""
    cfg = cfg or ViewCheckConfig()
    side = abs(float(sweep[0]))
    left = clear_extent(grid, pose, +1, side, cfg=cfg, points=points)
    right = clear_extent(grid, pose, -1, side, cfg=cfg, points=points)
    left = left if left >= cfg.min_sweep else 0.0
    right = right if right >= cfg.min_sweep else 0.0
    if left > 0 and right > 0:
        return [left, -(left + right), right]
    if left > 0:
        return [left, -left]
    if right > 0:
        return [-right, right]
    return []


def plan_turn(grid, planner, pose, err, *, cfg: ViewCheckConfig | None = None, points=None, budget=0.4,
              clearance_radius=None):
    """Decide how to make a NECESSARY in-place turn by ``err`` (rad) from ``pose``.

    Returns ``(mode, value)``:
    * ``("turn", why)``: turn the short way; ``why`` is None (turn too small to
      check), "clear" (the short-way sector, up to ``end_margin`` before the target
      heading, keeps the camera off near surfaces) or "unavoidable" (no better option);
    * ``("long", direction)``: |err| is within ``long_way_window`` of pi and the long
      way is clear while the short way is not;
    * ``("back_off", distance)``: reverse straight by ``distance`` (<= ``budget``) first;
      chosen as the smallest candidate after which the short-way sector is clear, else
      the candidate that most increases the nearest facing distance (by >= 0.1 m).
      A candidate needs obstacle clearance >= ``clearance_radius`` along the reverse
      segment, and every cell the rear sweeps through (centre line back to the
      footprint radius behind the end) must be FREE: never reverse into unknown space.
    """
    cfg = cfg or ViewCheckConfig()
    if abs(err) < cfg.min_turn:
        return "turn", None
    th = float(pose[2])
    sign = 1.0 if err > 0 else -1.0
    span = abs(err) - cfg.end_margin
    if span <= 0:
        return "turn", None

    def sector(p, direction, length):
        hs = sector_headings(th, th + direction * length, cfg.step)
        d = _facing(grid, p, hs, cfg, points).distance
        return float(np.mean(d < cfg.near)), float(d.min()) if len(d) else math.inf

    short_frac, short_min = sector(pose, sign, span)
    if short_frac == 0.0:
        return "turn", "clear"
    if abs(err) > math.pi - cfg.long_way_window:
        long_frac, _ = sector(pose, -sign, 2 * math.pi - abs(err) - cfg.end_margin)
        if long_frac == 0.0:
            return "long", -sign
    if planner is None:
        return "turn", "unavoidable"
    radius = clearance_radius if clearance_radius is not None else planner.cfg.footprint_radius
    back = -np.array([math.cos(th), math.sin(th)])
    best = None
    for b in cfg.backoff_steps:
        if b > budget + 1e-9:
            break
        here = np.asarray(pose[:2], float)[None]
        seg = here + back[None] * np.linspace(0.02, b, max(2, int(b / 0.025)))[:, None]
        if float(planner.obstacle_clearance_xy(grid, seg).min()) < radius:
            break  # reversing farther would only get closer to something behind
        rear_len = b + planner.cfg.footprint_radius
        rear = here + back[None] * np.linspace(0.0, rear_len, max(2, int(rear_len / 0.025)))[:, None]
        if not (grid.classify_xy(rear) == FREE).all():
            break
        p = np.array([seg[-1, 0], seg[-1, 1], th])
        frac, dmin = sector(p, sign, span)
        if frac == 0.0:
            return "back_off", float(b)
        if dmin >= short_min + 0.1 and (best is None or dmin > best[1]):
            best = (float(b), dmin)
    if best is not None:
        return "back_off", best[0]
    return "turn", "unavoidable"
