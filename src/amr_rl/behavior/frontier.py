"""Frontier candidates for map exploration (estimated map only)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..mapping.occupancy import FREE, UNKNOWN


def frontier_goals(grid, planner, pose, *, exhausted=(), min_cells=6, standoff=0.45, max_goals=6,
                   min_clearance=0.45, keep_out=(), tile=0.6, visited=(),
                   revisit_radius=0.4):
    """Return [(goal_xy, heading, gain_cells, key)] sorted by gain/cost.

    A frontier is FREE space adjacent to UNKNOWN. The goal is a traversable cell
    near the frontier from which the camera can look at it; ``heading`` faces the
    frontier tile. ``exhausted`` holds tile keys already tried twice. ``visited``
    holds positions where sweeps already happened: goals within ``revisit_radius``
    of them are penalised, because another sweep from the same spot adds no
    translational parallax (monocular evidence needs baseline).
    """
    planner.prepare(grid)
    cls = grid.classes()
    free = (cls == FREE).astype(np.uint8)
    unknown = (cls == UNKNOWN).astype(np.uint8)
    near_unknown = cv2.dilate(unknown, np.ones((3, 3), np.uint8))
    frontier = (free & near_unknown).astype(np.uint8)
    _, labels = cv2.connectedComponents(frontier, connectivity=8)
    # Prefer look-around poses well away from surfaces: rotating in place facing a
    # nearby wall is a degenerate view for monocular tracking.
    trav = planner._trav & (planner._clear >= min_clearance)
    if not trav.any():
        trav = planner._trav
    # Only goals reachable from the robot: its connected traversable component.
    n_comp, comp = cv2.connectedComponents(planner._trav.astype(np.uint8), connectivity=8)
    rx, ry = grid.to_cell(np.asarray(pose[:2], float))
    r = int(np.ceil(0.15 / grid.cfg.resolution))
    window = comp[max(0, ry[0] - r):ry[0] + r + 1, max(0, rx[0] - r):rx[0] + r + 1]
    here = window[window > 0]
    if len(here):
        mine = np.bincount(here).argmax()
        trav = trav & (comp == mine)
    ty, tx = np.nonzero(trav)
    if len(tx) == 0:
        return []
    trav_xy = grid.to_xy(tx, ty)
    for centre, radius in keep_out:  # e.g. learned aversions: do not explore next to them
        far = np.linalg.norm(trav_xy - np.asarray(centre)[None], axis=1) > radius + 0.2
        trav_xy = trav_xy[far]
    if len(trav_xy) == 0:
        return []
    # Split frontier components into spatial tiles: a frontier that rings the whole
    # known area has its centroid inside free space, which is not a useful target.
    fy, fx = np.nonzero(frontier)
    if len(fx) == 0:
        return []
    fxy = grid.to_xy(fx, fy)
    tiles = np.floor(fxy / tile).astype(int)
    groups = {}
    for i, (comp_id, tx_, ty_) in enumerate(zip(labels[fy, fx], tiles[:, 0], tiles[:, 1])):
        groups.setdefault((int(comp_id), int(tx_), int(ty_)), []).append(i)
    revisit_penalty = np.zeros(len(trav_xy))
    if len(visited):
        dv = np.min(np.linalg.norm(trav_xy[:, None] - np.asarray(visited)[None], axis=2), axis=1)
        revisit_penalty = 1.5 * np.clip(revisit_radius - dv, 0.0, None)
    out = []
    for (_, tx_, ty_), members in groups.items():
        if len(members) < min_cells:
            continue
        pts = fxy[members]
        target = pts[np.argmin(np.linalg.norm(pts - pts.mean(0)[None], axis=1))]  # a frontier cell
        key = (tx_, ty_)
        if key in exhausted:
            continue
        d = np.linalg.norm(trav_xy - target[None], axis=1)
        ok = d >= standoff * 0.6
        if not ok.any():
            continue
        score = d + revisit_penalty
        idx = np.flatnonzero(ok)[np.argmin(score[ok])]
        goal = trav_xy[idx]
        if np.linalg.norm(goal - target) > 1.5:
            continue
        heading = math.atan2(target[1] - goal[1], target[0] - goal[0])
        cost = float(np.linalg.norm(goal - np.asarray(pose[:2]))) + 0.3
        out.append((goal, heading, len(members), key, cost))
    # Prefer large frontiers; distance is a mild penalty so trips reach new ground.
    out.sort(key=lambda item: -(item[2] - 12.0 * item[4]))
    return [(g, h, gain, key) for g, h, gain, key, _ in out[:max_goals]]
